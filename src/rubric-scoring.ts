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
//
// phaseB.2 — fix: applyPhaseModifiers now iterates rubric.steps (authoritative)
// instead of AI stepScores (unreliable). The AI's per-step max_score is ignored;
// rubric.max_points is used exclusively. This eliminates the denominator drift
// (62–128 instead of 111) caused by AI key mismatches and gate-skip miscounting.
export const RUBRIC_SCORING_VERSION = "phaseB.2";

/**
 * Worker version-mismatch guard. The edge function stamps this version onto
 * every job payload it enqueues to `scoring_jobs`. The external worker calls
 * this on dequeue — if the worker's vendored copy of rubric-scoring.ts is on
 * a different version than what produced the payload, scoring math will
 * differ. We log loudly so ops sees it without trawling boot logs.
 *
 * Returns true if versions match, false otherwise. Worker should still
 * proceed (don't fail the job — operator decides), but the [VERSION MISMATCH]
 * line will trip alerts.
 */
export function assertScoringVersionOrWarn(payloadVersion: string | null | undefined): boolean {
  if (!payloadVersion) {
    console.warn(
      `[VERSION MISMATCH] payload missing rubric_scoring_version; worker is ${RUBRIC_SCORING_VERSION}`,
    );
    return false;
  }
  if (payloadVersion !== RUBRIC_SCORING_VERSION) {
    console.error(
      `[VERSION MISMATCH] payload=${payloadVersion} worker=${RUBRIC_SCORING_VERSION} — redeploy worker or edge function to align`,
    );
    return false;
  }
  return true;
}

/**
 * Fail-closed version guard for consumers (workers).
 *
 * Throws on mismatch — the caller doesn't process the job; pgmq redelivers
 * after the visibility timeout. Makes silent drift impossible (the
 * worn-out lesson from v8.9, where the worker stayed on phaseB.1 while the
 * edge fn moved to phaseB.2 for weeks, silently mis-scoring every job).
 *
 * Use this in the worker's processJob entry point. Producers (the edge fn)
 * don't call this — they stamp `rubric_scoring_version` into the payload.
 */
export function assertScoringVersionOrThrow(payloadVersion: string | null | undefined): void {
  if (!payloadVersion) {
    throw new Error(
      `[VERSION MISMATCH] payload missing rubric_scoring_version; worker is ${RUBRIC_SCORING_VERSION}. Job will redeliver until producer is aligned.`,
    );
  }
  if (payloadVersion !== RUBRIC_SCORING_VERSION) {
    throw new Error(
      `[VERSION MISMATCH] payload=${payloadVersion} worker=${RUBRIC_SCORING_VERSION}. Redeploy worker or producer to align. Job will redeliver until then.`,
    );
  }
}

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

export type HardFailMode = "cap" | "deduct" | "flag_only";

export interface HardFailModeConfig {
  mode: HardFailMode;
  /** Only honored when mode === "deduct". Percentage points to subtract. */
  points?: number;
}

export interface RubricConfig {
  advanced_scoring_enabled?: boolean;
  hard_fail_cap?: number; // default 49
  auto_bench_threshold?: number; // default 70
  /** Per-id mode override. Missing entries default to { mode: "cap" }. */
  hard_fail_modes?: Record<string, HardFailModeConfig>;
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
  /** Cap value applied (when any cap-mode hard-fail fired), else null. */
  cap_applied: number | null;
  /** Total percentage points subtracted by deduct-mode hard-fails, else null. */
  deduction_applied: number | null;
  /** Map of triggered id → resolved mode. Empty when none triggered. */
  hard_fail_modes: Record<string, HardFailMode>;
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
  // Use the rubric's declared max_points for gate pass/fail, not the AI's max_score.
  const max = step.max_points || 0;
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
 * Returns one ScoreBreakdownPhase per RUBRIC step (not per AI step).
 *
 * KEY DESIGN: This function iterates rubric.steps (authoritative), not stepScores
 * (AI output). The AI frequently returns invented step_key names or extra phantom
 * steps that don't match the rubric. By anchoring iteration to the rubric:
 *
 *   1. Every rubric step always appears in the breakdown — no phantom steps,
 *      no missing steps.
 *   2. Gate skips fire against rubric positions (correct indices), not against
 *      whatever the AI happened to return.
 *   3. The authoritative max_points from the rubric is always used — the AI's
 *      per-step max_score is ignored entirely, preventing denominator drift.
 *
 * If the AI returned a score for a step_key not in the rubric, that score is
 * silently discarded (the unknown key had no meaningful rubric backing anyway).
 *
 * Steps marked skipped by an upstream gate are returned with full credit
 * (adjusted_score = max_score), so they don't drag the percentage down.
 */
export function applyPhaseModifiers(
  steps: RubricStep[],
  stepScores: StepScoreInput[],
  qualificationColor: QualificationColor,
): ScoreBreakdownPhase[] {
  // Build lookup by rubric step key → AI-reported score.
  // NOTE: we only use the AI's `score` (numerator), never its `max_score`.
  const scoresByKey = new Map(stepScores.map((s) => [s.step_key, s]));
  const skippedMap = computeSkippedStepIds(steps, scoresByKey, qualificationColor);

  // Iterate the rubric's authoritative step list. This guarantees:
  //   - One entry per rubric step (no phantom AI steps inflate the denominator)
  //   - Gate skip logic fires against correct rubric-defined positions
  //   - max_score in each phase row = rubric max_points (never AI-drifted value)
  return steps.map((step) => {
    const max = Number(step.max_points) || 0;
    const aiScore = scoresByKey.get(step.key);
    const raw = Number(aiScore?.score) || 0;
    const skipReason = skippedMap.get(step.key);

    if (skipReason) {
      // Skipped steps get full credit — they were conditionally N/A for this call,
      // so they should not drag the score. Use the rubric's declared max as the credit.
      return {
        step_key: step.key,
        raw_score: max,
        adjusted_score: max,
        modifier: 1,
        max_score: max,
        skipped: true,
        skip_reason: skipReason,
      };
    }

    const modifier = modifierFor(step, qualificationColor);
    const adjusted = Math.min(max, raw * modifier);
    return {
      step_key: step.key,
      raw_score: raw,
      adjusted_score: adjusted,
      modifier,
      max_score: max,
    };
  });
}

/**
 * Extract every hard-fail id known to the rubric.
 *
 * STRICT: only ids declared by structured criteria
 * (`{ hard_fail: true, hard_fail_id: "..." }`) count. Bare `hf_*` mentions
 * and `[HARD-FAIL id=...]` tags in step descriptions or string criteria are
 * ignored — they're prompt scaffolding for the AI, not contracts the engine
 * should treat as authoritative. Removing every structured criterion from a
 * rubric must mean "no hard fails."
 */
export function extractHardFailIdsFromRubric(rubric: RubricConfig): Set<string> {
  const ids = new Set<string>();
  for (const step of rubric.steps || []) {
    for (const c of step.criteria || []) {
      if (c && typeof c === "object" && c.hard_fail === true && c.hard_fail_id) {
        ids.add(c.hard_fail_id);
      }
    }
  }
  return ids;
}

interface HardFailResolution {
  adjustedPct: number;
  triggered: boolean;
  reason: string | null;
  cap_applied: number | null;
  deduction_applied: number | null;
  modes_triggered: Record<string, HardFailMode>;
}

/**
 * Resolve hard-fails against the rubric and apply per-id modes.
 * Order: deduct → cap (if any cap-mode fired) → floor at 0.
 * `flag_only` ids are recorded but do not touch the score.
 */
export function applyHardFails(
  rubric: RubricConfig,
  adjustedPct: number,
  hardFailsTriggered: string[],
): HardFailResolution {
  const cap = rubric.hard_fail_cap ?? DEFAULT_HARD_FAIL_CAP;
  const empty: HardFailResolution = {
    adjustedPct,
    triggered: false,
    reason: null,
    cap_applied: null,
    deduction_applied: null,
    modes_triggered: {},
  };
  if (!hardFailsTriggered || hardFailsTriggered.length === 0) return empty;

  const validIds = extractHardFailIdsFromRubric(rubric);
  const matched = hardFailsTriggered.filter((id) => validIds.has(id));
  if (matched.length === 0) return empty;

  const modes = rubric.hard_fail_modes || {};
  const modesTriggered: Record<string, HardFailMode> = {};
  let totalDeduct = 0;
  let anyCap = false;

  for (const id of matched) {
    const cfg = modes[id];
    const mode: HardFailMode = cfg?.mode === "deduct" || cfg?.mode === "flag_only"
      ? cfg.mode
      : "cap";
    modesTriggered[id] = mode;
    if (mode === "cap") {
      anyCap = true;
    } else if (mode === "deduct") {
      const pts = Number(cfg?.points);
      if (Number.isFinite(pts) && pts > 0) totalDeduct += pts;
    }
  }

  let next = adjustedPct;
  const deductionApplied = totalDeduct > 0 ? totalDeduct : null;
  if (deductionApplied !== null) next = next - deductionApplied;
  let capApplied: number | null = null;
  if (anyCap && next > cap) {
    next = cap;
    capApplied = cap;
  } else if (anyCap) {
    capApplied = cap;
  }
  if (next < 0) next = 0;

  const reasonParts = matched.map((id) => {
    const m = modesTriggered[id];
    if (m === "deduct") {
      const pts = Number(modes[id]?.points);
      return `${id} (-${Number.isFinite(pts) && pts > 0 ? pts : 0}pt)`;
    }
    if (m === "flag_only") return `${id} (flag)`;
    return `${id} (cap)`;
  });

  return {
    adjustedPct: next,
    triggered: true,
    reason: `Hard fail: ${reasonParts.join(", ")}`,
    cap_applied: capApplied,
    deduction_applied: deductionApplied,
    modes_triggered: modesTriggered,
  };
}

/**
 * @deprecated Back-compat wrapper around applyHardFails. New callers should
 * use applyHardFails directly to access deduction_applied / modes_triggered.
 */
export function applyHardFailCap(
  rubric: RubricConfig,
  rawPct: number,
  hardFailsTriggered: string[],
): { cappedPct: number; triggered: boolean; reason: string | null; cap: number } {
  const cap = rubric.hard_fail_cap ?? DEFAULT_HARD_FAIL_CAP;
  const r = applyHardFails(rubric, rawPct, hardFailsTriggered);
  return { cappedPct: r.adjustedPct, triggered: r.triggered, reason: r.reason, cap };
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

  // applyPhaseModifiers now iterates rubric.steps (not stepScores), so phases
  // always has exactly one entry per rubric step with authoritative max_points.
  const phases = applyPhaseModifiers(rubric.steps || [], stepScores, color);
  const rawTotal = phases.reduce((acc, p) => acc + p.raw_score, 0);
  const adjustedTotal = phases.reduce((acc, p) => acc + p.adjusted_score, 0);

  // Denominator: rubric.max_total_score is authoritative. Declared step sum is
  // the verified fallback (phases now uses rubric max_points so aiPhaseMax ==
  // declaredStepMax when phases are complete). AI phase sum is last resort only.
  const rubricMax = Number(rubric.max_total_score) || 0;
  const declaredStepMax = (rubric.steps || []).reduce(
    (acc, s) => acc + (Number(s.max_points) || 0),
    0,
  );
  const aiPhaseMax = phases.reduce((acc, p) => acc + p.max_score, 0);
  const maxTotal = rubricMax || declaredStepMax || aiPhaseMax || 0;

  const rawPct = maxTotal > 0 ? (rawTotal / maxTotal) * 100 : 0;
  const adjustedPctUncapped = maxTotal > 0 ? (adjustedTotal / maxTotal) * 100 : 0;
  const resolved = applyHardFails(rubric, adjustedPctUncapped, fails);

  return {
    raw_total: rawTotal,
    adjusted_total: adjustedTotal,
    max_total: maxTotal,
    raw_percentage: clampPct(Math.round(rawPct * 100) / 100),
    adjusted_percentage: clampPct(Math.round(resolved.adjustedPct * 100) / 100),
    qualification_color: color,
    hard_fail_triggered: resolved.triggered,
    hard_fail_reason: resolved.reason,
    hard_fails_triggered: fails,
    cap_applied: resolved.cap_applied,
    deduction_applied: resolved.deduction_applied,
    hard_fail_modes: resolved.modes_triggered,
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
    // The AI can return per-step scores summing above the rubric's fixed max
    // (it over-allocates max_score per step). Clamp so overall_score never
    // exceeds max_possible_score and the percentage stays within 0–100.
    const overall = Math.min(breakdown.raw_total, max);
    const pct = max > 0 ? clampPct(Math.round((overall / max) * 100)) : 0;
    return {
      overall_score: overall,
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
