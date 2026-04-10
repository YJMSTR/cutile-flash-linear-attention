# Plan: Paddle GDN/KDA Backward Performance Investigation

## Goal Description

Investigate the Paddle performance gap relative to Torch for `chunk_gdn` and `chunk_kda`
in `fwdbwd` mode across the full benchmark shape set, identify the dominant root cause with
reproducible evidence, and validate the conclusion with the smallest viable code change if a
clear fix candidate emerges.

This plan is about root-cause attribution first. A performance improvement is desirable, but the
primary success condition is a defensible conclusion backed by rerunnable benchmark and
instrumentation evidence. If a minimal validation fix is implemented, it must preserve correctness
and be measured against the same full-shape benchmark set.

## Acceptance Criteria

- AC-1: Reproduce and store a current full-shape benchmark baseline for Paddle and Torch on
  `chunk_gdn` and `chunk_kda` in `fwdbwd` mode using the repository's benchmark flow.
  - Positive Tests (expected to PASS):
    - Running the benchmark flow produces structured per-shape results for both frameworks.
    - The reproduced output includes all agreed full-shape configurations and can be compared
      shape by shape against the existing `benchmark_outputs/framework_compare` artifact.
  - Negative Tests (expected to FAIL):
    - A run that only covers smoke shapes is not accepted as satisfying this criterion.
    - A run that only reports aggregate averages without per-shape detail is not accepted.

- AC-2: Localize the dominant slowdown for each target op to one or more concrete layers:
  benchmark harness overhead, Paddle autograd wrapping, op orchestration/recompute, Triton kernel
  execution, or autotune/config selection.
  - Positive Tests (expected to PASS):
    - The investigation produces measurable evidence that attributes a meaningful portion of the
      gap for `chunk_gdn`.
    - The investigation produces measurable evidence that attributes a meaningful portion of the
      gap for `chunk_kda`.
  - Negative Tests (expected to FAIL):
    - A conclusion that only lists code differences without timing or profiling evidence does not
      satisfy this criterion.
    - A conclusion that assumes `gdn` and `kda` share the same cause without evidence does not
      satisfy this criterion.

- AC-3: Create a Torch-vs-Paddle divergence map for the active backward path and explicitly mark
  each major difference as irrelevant, unproven, or evidence-supported.
  - Positive Tests (expected to PASS):
    - The final analysis covers the key backward-path differences that were inspected.
    - The final analysis states whether the dominant bottleneck is shared across both ops or not.
  - Negative Tests (expected to FAIL):
    - A report that skips Paddle `PyLayer` effects, recomputation behavior, or kernel-wrapper
      differences without justification does not satisfy this criterion.

- AC-4: Preserve correctness while investigating and while validating any minimal fix.
  - Positive Tests (expected to PASS):
    - Existing targeted correctness tests for Paddle `gdn` and `kda` still pass after any code
      changes used to validate the hypothesis.
    - Any diagnostic-only code paths do not change production behavior unless intentionally enabled.
  - Negative Tests (expected to FAIL):
    - A performance change that breaks existing Paddle correctness tests does not satisfy this
      criterion.

- AC-5: If a minimal fix is applied, rerun the full-shape benchmark and report before/after
  results per shape, including unresolved regressions.
  - Positive Tests (expected to PASS):
    - The rerun shows the actual effect of the minimal fix shape by shape.
    - The final write-up distinguishes resolved shapes, unchanged shapes, and remaining slow shapes.
  - Negative Tests (expected to FAIL):
    - Claiming improvement without rerunning the full-shape benchmark does not satisfy this
      criterion.
    - Reporting only a best-case shape and hiding neutral or regressed shapes does not satisfy this
      criterion.

## Path Boundaries

### Upper Bound (Maximum Scope)

The implementation may include:

- benchmark or profiling helpers narrowly scoped to diagnosing this regression
- instrumentation in benchmark runners or Paddle op wrappers to attribute time by layer
- a small production fix in Paddle GDN/KDA code or related benchmark wrapper code
- updated benchmark outputs and a concise investigation summary

The implementation should not expand into a broad Paddle performance cleanup outside the
`chunk_gdn` / `chunk_kda` backward investigation.

### Lower Bound (Minimum Scope)

The minimum acceptable outcome is:

- a rerunnable full-shape benchmark baseline
- reproducible evidence narrowing the slowdown to a concrete layer or kernel family
- a final conclusion that states the most likely dominant root cause for `gdn` and `kda`

This lower bound is acceptable even if no production fix is landed, provided the uncertainty is
materially reduced and the remaining blocker is clearly identified.

### Allowed Choices

- Can use:
  - existing benchmark runners and outputs as the starting point
  - temporary diagnostic instrumentation
  - targeted profilers or timers
  - minimal code changes that directly validate a root-cause hypothesis
- Cannot use:
  - unrelated cleanup refactors
  - smoke-only validation as the final benchmark evidence
  - speculative multi-fix bundles that make attribution impossible
  - success claims without full-shape reruns after code changes

## Dependencies and Sequence

### Milestone 1: Establish a Stable Baseline

1. Re-run the full-shape Torch and Paddle benchmark flow for `chunk_gdn` and `chunk_kda` in
   `fwdbwd` mode.
2. Save the structured outputs in a location that can be compared round to round.
3. Confirm whether the reproduced gap matches the direction and rough shape sensitivity from the
   existing artifact.

### Milestone 2: Attribute the Gap by Layer

1. Add minimal timing/profiling hooks to isolate benchmark-harness, `PyLayer`, recompute, and
   Triton-kernel portions of the Paddle path.
2. Collect evidence for at least one representative "small/overhead-sensitive" shape and one
   "large/kernel-sensitive" shape for each op.
3. Decide which layer dominates the observed gap for `gdn` and `kda`.

### Milestone 3: Build the Divergence Map

1. Compare Torch and Paddle active backward paths for both ops.
2. Record major differences and classify them as irrelevant, unproven, or evidence-supported.
3. Identify whether the main contributor is shared between `gdn` and `kda` or differs by op.

### Milestone 4: Validate the Leading Hypothesis

1. Apply the smallest code change that directly tests the leading root-cause hypothesis.
2. Run targeted correctness tests covering the touched Paddle paths.
3. Re-run the full-shape benchmark and compare per-shape before/after results.

### Milestone 5: Summarize Resolved and Unresolved Findings

1. Write a concise conclusion describing the dominant root cause for each op.
2. State which regressions were reduced, which remain, and what evidence supports each claim.
3. Ensure RLCR round summaries capture the hypothesis/evidence/result chain across iterations.

## Feasibility Hints

- Start with shapes where the current artifact shows the largest slowdowns, because these are most
  likely to reveal the dominant issue quickly.
- Use the recurrent-path parity as a control signal: if recurrent is near parity while chunk
  `fwdbwd` is not, prioritize chunk backward orchestration and wrapper differences first.
- The Paddle benchmark currently routes backward through `paddle.autograd.PyLayer`, while Torch
  routes through `torch.autograd.Function`; wrapper overhead and tensor-save/restore behavior are
  reasonable early hypotheses, but they still require measurement.
- For KDA, distinguish between PyLayer-level overhead and Triton-kernel/config behavior, because
  the Paddle kernels include compatibility wrappers that may or may not matter.
- Do not overfit conclusions to one shape. Use at least one small and one large representative
  shape before claiming a root cause is general.

## Quantitative Guidance

The existing `paddle_over_torch` ratios are baseline observations, not draft-stage hard pass/fail
targets. The hard requirement is root-cause attribution with reproducible evidence.

If a minimal fix is implemented:

- report the per-shape before/after ratios
- describe the trend across the full shape set
- explicitly list any shapes that remain materially behind Torch

## Implementation Notes

- Do not put plan labels such as `AC-1` or `Milestone 2` into production code.
- Keep diagnostic additions easy to remove once the investigation is complete.
- Prefer shape-by-shape tables or JSON evidence over prose-only summaries.
- Each RLCR round should test one concrete hypothesis or eliminate one concrete suspect.
