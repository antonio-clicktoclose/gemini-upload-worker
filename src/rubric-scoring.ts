// Phase A: Pure rubric-scoring helpers shared by:
//   - supabase/functions/score-sales-call (Deno edge function)
//   - External Node worker (EasyPanel) consuming scoring_jobs pgmq queue
//
// MUST stay framework-free (no Deno.* / no Node-only APIs) so the same file
// can be vendored into the worker repo. Keep functions pure & deterministic
// so the parity vitest fixture can assert byte-for-byte equality.

// Bump whenever scoring math changes. Both runtimes log this on boot;
// ops verifies edge function + worker report the same value before
// flipping any rubric to advanced_scoring_enabled = true.
export const RUBRIC_SCORING_VERSION = "phaseA.1";

export type QualificationColor = "GREEN" | "YELLOW" | "RED";

export interface RubricStep {
  key: string;
  label?: string;
  max_points: number;
  // Optional per-phase qualification modifier — multiplies adjusted score
  // when the AI labels the prospect with this color. Missing = 1.0.
  qualification_modifier?: Partial<Record<QualificationColor, number>>;
  criteria?: Array<string | RubricCriterion>;
  [k: string]: unknown;
}

export interface RubricCriterion {
  text?: string;
  hard_fail?: boolean;
  hard_fail_id?: string;
}

export interface RubricConfig {
  advanced_scoring_enabled?: boolean;
  hard_fail_cap?: number; // default 49
  auto_bench_threshold?: number; // default 70
  steps: RubricStep[];
  max_total_score?: number;
}

export interface StepScoreInput {
  step_key: string;
  score: number;
  max_score: number;
}

export interface ScoreBreakdownPhase {
  step_key: string;
  raw_score: number;
  adjusted_score: number;
  modifier: number;
  max_score: number;
}

export interface ScoreBreakdown {
  raw_total: number;
  adjusted_total: number;
  max_total: number;
  raw_percentage: number;
  adjusted_percentage: number;
  qualification_color: QualificationColor;
  hard_fail_triggered: boolean;
  hard_fail_reason: string | null;
  hard_fails_triggered: string[];
  cap_applied: number | null;
  phases: ScoreBreakdownPhase[];
}

const DEFAULT_HARD_FAIL_CAP = 49;

function clampPct(n: number): number {
  if (!Number.isFinite(n)) return 0;
  if (n < 0) return 0;
  if (n > 100) return 100;
  return n;
}

function modifierFor(step: RubricStep, color: QualificationColor): number {
  const mod = step.qualification_modifier?.[color];
  return typeof mod === "number" && Number.isFinite(mod) && mod > 0 ? mod : 1;
}

/**
 * Apply per-phase qualification modifiers to AI step scores.
 * Returns adjusted phase rows (raw + adjusted side-by-side).
 */
export function applyPhaseModifiers(
  steps: RubricStep[],
  stepScores: StepScoreInput[],
  qualificationColor: QualificationColor,
): ScoreBreakdownPhase[] {
  const stepByKey = new Map(steps.map((s) => [s.key, s]));
  return stepScores.map((s) => {
    const step = stepByKey.get(s.step_key);
    const modifier = step ? modifierFor(step, qualificationColor) : 1;
    const raw = Number(s.score) || 0;
    const max = Number(s.max_score) || step?.max_points || 0;
    const adjusted = Math.min(max, raw * modifier);
    return {
      step_key: s.step_key,
      raw_score: raw,
      adjusted_score: adjusted,
      modifier,
      max_score: max,
    };
  });
}

/**
 * Apply hard-fail cap. If any of the listed hard_fail ids are matched against
 * a criterion that has hard_fail=true, cap the score at rubric.hard_fail_cap.
 */
export function applyHardFailCap(
  rubric: RubricConfig,
  rawPct: number,
  hardFailsTriggered: string[],
): { cappedPct: number; triggered: boolean; reason: string | null; cap: number } {
  const cap = rubric.hard_fail_cap ?? DEFAULT_HARD_FAIL_CAP;
  if (!hardFailsTriggered || hardFailsTriggered.length === 0) {
    return { cappedPct: rawPct, triggered: false, reason: null, cap };
  }

  // Validate that each id corresponds to a criterion flagged hard_fail=true
  // anywhere in the rubric. Unknown ids are ignored (defensive).
  const validIds = new Set<string>();
  for (const step of rubric.steps || []) {
    for (const c of step.criteria || []) {
      if (typeof c === "object" && c?.hard_fail && c.hard_fail_id) {
        validIds.add(c.hard_fail_id);
      }
    }
  }
  const matched = hardFailsTriggered.filter((id) => validIds.has(id));
  if (matched.length === 0) {
    return { cappedPct: rawPct, triggered: false, reason: null, cap };
  }
  if (rawPct <= cap) {
    return { cappedPct: rawPct, triggered: true, reason: `Hard fail: ${matched.join(", ")}`, cap };
  }
  return { cappedPct: cap, triggered: true, reason: `Hard fail: ${matched.join(", ")}`, cap };
}

/**
 * Build the canonical score_breakdown JSON written to call_scores.
 * Always safe to call; if advanced_scoring_enabled is false the adjusted
 * values equal the raw values and no cap is applied.
 */
export function buildScoreBreakdown(
  rubric: RubricConfig,
  stepScores: StepScoreInput[],
  qualificationColor: QualificationColor = "GREEN",
  hardFailsTriggered: string[] = [],
): ScoreBreakdown {
  const enabled = rubric.advanced_scoring_enabled === true;
  const color: QualificationColor = enabled ? qualificationColor : "GREEN";
  const fails = enabled ? hardFailsTriggered : [];

  const phases = applyPhaseModifiers(rubric.steps || [], stepScores, color);
  const rawTotal = phases.reduce((acc, p) => acc + p.raw_score, 0);
  const adjustedTotal = phases.reduce((acc, p) => acc + p.adjusted_score, 0);
  const maxTotal = phases.reduce((acc, p) => acc + p.max_score, 0)
    || rubric.max_total_score
    || 0;

  const rawPct = maxTotal > 0 ? (rawTotal / maxTotal) * 100 : 0;
  const adjustedPctUncapped = maxTotal > 0 ? (adjustedTotal / maxTotal) * 100 : 0;
  const cap = applyHardFailCap(rubric, adjustedPctUncapped, fails);

  return {
    raw_total: rawTotal,
    adjusted_total: adjustedTotal,
    max_total: maxTotal,
    raw_percentage: clampPct(Math.round(rawPct * 100) / 100),
    adjusted_percentage: clampPct(Math.round(cap.cappedPct * 100) / 100),
    qualification_color: color,
    hard_fail_triggered: cap.triggered,
    hard_fail_reason: cap.reason,
    hard_fails_triggered: fails,
    cap_applied: cap.triggered ? cap.cap : null,
    phases,
  };
}

/**
 * Resolve the final overall_score (in points) the engine should write.
 * - advanced_scoring_enabled=false → returns rawTotal (legacy behavior, unchanged)
 * - advanced_scoring_enabled=true  → returns the adjusted+capped equivalent in points
 */
export function resolveFinalScore(
  rubric: RubricConfig,
  breakdown: ScoreBreakdown,
): { overall_score: number; max_possible_score: number; score_percentage: number } {
  const enabled = rubric.advanced_scoring_enabled === true;
  const max = breakdown.max_total;
  if (!enabled) {
    const pct = max > 0 ? Math.round((breakdown.raw_total / max) * 100) : 0;
    return {
      overall_score: breakdown.raw_total,
      max_possible_score: max,
      score_percentage: pct,
    };
  }
  // Convert adjusted_percentage back to a points value so existing
  // consumers that read overall_score continue to work.
  const points = max > 0 ? (breakdown.adjusted_percentage / 100) * max : 0;
  return {
    overall_score: Math.round(points * 100) / 100,
    max_possible_score: max,
    score_percentage: Math.round(breakdown.adjusted_percentage),
  };
}
