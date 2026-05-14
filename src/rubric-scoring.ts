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
export const RUBRIC_SCORING_VERSION = "phaseA.4";

export type QualificationColor = "GREEN" | "YELLOW" | "RED";

// A gate step's outcome type controls how the engine derives the outcome
// value used to look up gate_branches. Two types are supported:
//   - "pass_fail":            outcome is "pass" when AI score == max, else "fail"
//   - "qualification_color":  outcome is the call's qualification_color (the
//                             step's score is computed normally; the gate
//                             reads the call-level color regardless)
export type GateOutcomeType = "pass_fail" | "qualification_color";

export interface RubricStep {
  key: string;
  label?: string;
  max_points: number;
  // Optional per-phase qualification modifier — multiplies adjusted score
  // when the AI labels the prospect with this color. Missing = 1.0.
  qualification_modifier?: Partial<Record<QualificationColor, number>>;
  // -- Gate steps -------------------------------------------------------
  // When is_gate=true, this step's outcome decides which other steps are
  // marked N/A (full credit, no AI drag, "N/A" pill on the scorecard).
  // gate_branches maps each possible outcome value to a list of step keys.
  // Empty/missing branch = no skips for that outcome.
  is_gate?: boolean;
  gate_outcome_type?: GateOutcomeType; // default "pass_fail"
  gate_branches?: Record<string, string[]>;
  // -- Legacy (no-op; kept to avoid breaking read of old rubrics) -------
  applies_to_colors?: QualificationColor[];
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
  // Present only when the step was skipped by an upstream gate.
  // When true: adjusted_score == max_score (full credit, no drag).
  skipped?: boolean;
  skip_reason?: string;
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
 * Determine a gate step's outcome value for this call.
 * - qualification_color gates: returns the call's color
 * - pass_fail gates: "pass" if score == max, else "fail"
 */
function resolveGateOutcome(
  step: RubricStep,
  scoreInput: StepScoreInput | undefined,
  qualificationColor: QualificationColor,
): string {
  const type: GateOutcomeType = step.gate_outcome_type === "qualification_color"
    ? "qualification_color"
    : "pass_fail";
  if (type === "qualification_color") return qualificationColor;
  const score = Number(scoreInput?.score) || 0;
  const max = Number(scoreInput?.max_score) || step.max_points || 0;
  if (max <= 0) return "pass";
  return score >= max ? "pass" : "fail";
}

/**
 * Walk gate steps in order and return a Map of skipped step keys → reason.
 * First gate to skip a given step wins for skip_reason attribution.
 * Defensive: a gate step can never skip itself or a step before it.
 */
export function computeSkippedStepIds(
  steps: RubricStep[],
  scoresByKey: Map<string, StepScoreInput>,
  qualificationColor: QualificationColor,
): Map<string, string> {
  const skipped = new Map<string, string>();
  const stepIndex = new Map<string, number>();
  steps.forEach((s, i) => stepIndex.set(s.key, i));

  steps.forEach((step, gateIdx) => {
    if (!step.is_gate || !step.gate_branches) return;
    const outcome = resolveGateOutcome(step, scoresByKey.get(step.key), qualificationColor);
    const targets = step.gate_branches[outcome];
    if (!Array.isArray(targets) || targets.length === 0) return;
    const reason = `${step.label || step.key} → ${outcome}`;
    for (const targetKey of targets) {
      if (typeof targetKey !== "string" || !targetKey) continue;
      if (targetKey === step.key) continue; // gates can't skip themselves
      const targetIdx = stepIndex.get(targetKey);
      if (targetIdx === undefined || targetIdx <= gateIdx) continue; // forward only
      if (!skipped.has(targetKey)) skipped.set(targetKey, reason);
    }
  });

  return skipped;
}

/**
 * Apply per-phase qualification modifiers to AI step scores.
 * Returns adjusted phase rows (raw + adjusted side-by-side).
 * Steps marked skipped by an upstream gate are returned with full credit.
 */
export function applyPhaseModifiers(
  steps: RubricStep[],
  stepScores: StepScoreInput[],
  qualificationColor: QualificationColor,
): ScoreBreakdownPhase[] {
  const stepByKey = new Map(steps.map((s) => [s.key, s]));
  const scoresByKey = new Map(stepScores.map((s) => [s.step_key, s]));
  const skippedMap = computeSkippedStepIds(steps, scoresByKey, qualificationColor);

  return stepScores.map((s) => {
    const step = stepByKey.get(s.step_key);
    const max = Number(s.max_score) || step?.max_points || 0;
    const skipReason = skippedMap.get(s.step_key);
    if (skipReason) {
      return {
        step_key: s.step_key,
        raw_score: max,
        adjusted_score: max,
        modifier: 1,
        max_score: max,
        skipped: true,
        skip_reason: skipReason,
      };
    }
    const modifier = step ? modifierFor(step, qualificationColor) : 1;
    const raw = Number(s.score) || 0;
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
 * Extract every hard-fail id known to the rubric. Used by applyHardFailCap to
 * validate AI-returned ids without requiring structured criteria.
 */
export function extractHardFailIdsFromRubric(rubric: RubricConfig): Set<string> {
  const ids = new Set<string>();
  const tagRe = /\[HARD-?FAIL\s+id\s*=\s*([a-zA-Z0-9_-]+)\s*\]/gi;
  const bareRe = /\bhf_[a-z0-9_]+/gi;
  const scan = (s: unknown) => {
    if (typeof s !== "string" || !s) return;
    let m: RegExpExecArray | null;
    tagRe.lastIndex = 0;
    while ((m = tagRe.exec(s))) ids.add(m[1]);
    bareRe.lastIndex = 0;
    while ((m = bareRe.exec(s))) ids.add(m[0]);
  };
  for (const step of rubric.steps || []) {
    scan(step.label);
    scan((step as any).description);
    for (const c of step.criteria || []) {
      if (typeof c === "string") {
        scan(c);
      } else if (c && typeof c === "object") {
        if (c.hard_fail && c.hard_fail_id) ids.add(c.hard_fail_id);
        scan(c.text);
      }
    }
  }
  return ids;
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

  const validIds = extractHardFailIdsFromRubric(rubric);
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
  const points = max > 0 ? (breakdown.adjusted_percentage / 100) * max : 0;
  return {
    overall_score: Math.round(points * 100) / 100,
    max_possible_score: max,
    score_percentage: Math.round(breakdown.adjusted_percentage),
  };
}
