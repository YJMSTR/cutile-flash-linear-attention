# Paddle GDN/KDA Backward Performance Investigation Draft

## Context

The repository already contains cross-framework benchmark artifacts under `benchmark_outputs/`.
Those results show that Paddle still lags Torch on chunked GDN and KDA training-path execution,
while recurrent forward paths are close to parity.

This draft is intended to be the direct input document for `humanize gen plan`.
The downstream implementation and analysis flow should use RLCR and may include a minimal code
change once the root cause is supported by evidence.

## Problem Statement

Current benchmark evidence indicates a Paddle vs Torch performance gap on the backward-inclusive
chunked GDN and KDA paths:

- Source of truth: `benchmark_outputs/framework_compare/comparison_report.json`
- Focus ops: `chunk_gdn`, `chunk_kda`
- Focus mode: `fwdbwd`
- Environment in artifact: H800, Torch `2.9.1+cu129`, Paddle `3.4.0.dev20260407`

Representative full-shape results from the existing artifact:

- `chunk_gdn` `fwdbwd`
  - `B1 T8192 H96 D128`: `1.111647x`
  - `B2 T16384 H16 D128`: `1.317242x`
  - `B4 T2048 H16 D128`: `1.717176x`
  - `B4 T4096 H64 D128`: `1.149247x`
  - `B8 T1024 H8 D64`: `1.717988x`
  - `B8 T2048 H32 D256`: `0.972177x`
- `chunk_kda` `fwdbwd`
  - `B1 T8192 H96 D128`: `1.100925x`
  - `B2 T16384 H16 D128`: `1.184902x`
  - `B4 T2048 H16 D128`: `1.108127x`
  - `B4 T4096 H64 D128`: `1.101542x`
  - `B8 T1024 H8 D64`: `1.733878x`
  - `B8 T2048 H32 D256`: `1.312199x`

Additional smoke evidence from `benchmark_outputs/framework_compare_recurrent_smoke/comparison_report.json`
shows the same direction on `fwdbwd`:

- `chunk_gdn`: `1.604905x`
- `chunk_kda`: `1.637901x`
- `recurrent_gdn`: `1.003215x` on forward-only smoke
- `recurrent_kda`: `1.006699x` on forward-only smoke

The problem to solve is not "Paddle is globally slower everywhere." The working hypothesis is that
the main regression is concentrated in the chunk backward path and/or its Paddle-side execution
wrapping, not in the recurrent kernels or the whole benchmark framework.

## Goal

Identify the dominant root cause or causes behind the Paddle performance gap for `chunk_gdn` and
`chunk_kda` in `fwdbwd` mode across the full benchmark shape set, and validate the conclusion with
reproducible evidence.

If the root cause is sufficiently isolated and the remediation is small, apply the smallest
possible change to validate the hypothesis and measure the before/after effect.

## In Scope

- Full-shape benchmark reproduction for `chunk_gdn` and `chunk_kda` in `fwdbwd` mode
- Comparison of Paddle and Torch code paths for the same ops
- Evidence-driven attribution across these layers:
  - benchmark harness overhead
  - Paddle `PyLayer` / autograd wrapper overhead
  - forward recomputation and op-level orchestration
  - Triton kernel execution and autotune/config selection
- Instrumentation or diagnostic scripts needed to isolate the bottleneck
- A minimal validation fix if and only if the root cause is supported by evidence

## Out of Scope

- Broad refactors unrelated to the measured regression
- Optimizing all Paddle ops or the entire benchmark suite
- Context parallel feature parity work
- Recurrent path optimization unless evidence shows it contributes directly
- Large API redesigns or behavior changes to match Torch in unrelated areas

## Constraints

- Use the existing benchmark artifacts and runner layout as the initial source of truth
- Preserve correctness of Paddle implementations and existing public entry points
- Keep any validation fix minimal and reversible
- Do not rely on guess-and-check changes without first collecting root-cause evidence
- Treat full-shape results as the acceptance baseline, not smoke-only results

## Investigation Strategy

### Phase 1: Reproduce and Freeze the Baseline

Re-run the full-shape cross-framework benchmark for:

- `chunk_gdn`
- `chunk_kda`
- mode `fwdbwd`

Capture per-shape timing and ratio data so later rounds can compare against a stable baseline.
The output must make it obvious whether the regression behaves like:

- a roughly fixed host/framework overhead
- a shape-sensitive kernel slowdown
- a mixed issue that differs between `gdn` and `kda`

### Phase 2: Attribute Time by Layer

Split one Paddle `fwdbwd` execution into measurable segments where practical:

- benchmark harness setup and gradient clearing
- `PyLayer.forward`
- `PyLayer.backward`
- l2norm preprocessing/postprocessing
- forward recomputation inside backward
- op-specific backward helper calls
- Triton kernel launches and any obvious compatibility wrappers

The first objective is not perfect profiling resolution. The objective is to answer which layer
owns most of the gap and therefore deserves focused analysis.

### Phase 3: Build a Torch vs Paddle Divergence Map

For both `gdn` and `kda`, compare Torch and Paddle along the active backward path and classify
differences into:

- confirmed irrelevant
- plausible but unproven
- evidence-supported contributors

Priority checkpoints include:

- `torch.autograd.Function` vs `paddle.autograd.PyLayer`
- dummy `final_state` handling when `output_final_state=False`
- `save_for_backward` / `saved_tensor` behavior and tensor filtering logic
- repeated casts or gradient cleanup logic in the benchmark harness
- forward recomputation behavior in backward
- Triton wrapper differences such as `enable_compat_on_triton_kernel`
- autotune config selection and shared-memory gating

The output of this phase should explicitly separate shared root causes from op-specific ones.

### Phase 4: Validate the Leading Hypothesis

Once one dominant hypothesis is supported by evidence, implement the smallest possible validation:

- remove or bypass a measurable overhead
- narrow a compatibility path
- adjust an unnecessary wrapper step
- align a Paddle-side execution path with the Torch path
- or change a kernel/config selection only when profiling shows it is the main issue

Then rerun the full-shape benchmark and compare:

- per-shape ratios
- aggregate direction of change
- correctness and existing tests

## Expected Deliverables

- A reproducible benchmark baseline for `chunk_gdn` and `chunk_kda` `fwdbwd`
- Evidence artifacts that localize the slowdown to one or more layers
- A written conclusion stating:
  - the primary root cause for `gdn`
  - the primary root cause for `kda`
  - whether they share the same dominant bottleneck
- If applicable, a minimal code change validating the hypothesis
- Before/after benchmark data for the minimal validation change

## Acceptance Criteria

- AC-1: The full benchmark shape set can be rerun for `chunk_gdn` and `chunk_kda` in `fwdbwd`
  mode, and the reproduced results are stored in a structured form suitable for comparison.
- AC-2: The investigation identifies at least one evidence-supported dominant contributor for each
  target op, rather than listing only speculative differences.
- AC-3: The final analysis distinguishes whether the observed gap is mainly caused by benchmark
  harness overhead, Paddle autograd wrapping, op-level recomputation/orchestration, Triton kernel
  behavior, or a combination of these.
- AC-4: The investigation produces reusable evidence artifacts such as benchmark outputs,
  layer-attribution measurements, or profiler-derived summaries that another engineer can rerun.
- AC-5: If a minimal fix is applied, existing correctness checks continue to pass and the
  full-shape benchmark is rerun to show the actual effect size shape by shape.
- AC-6: The final write-up explicitly states which gaps remain unresolved if the minimal fix does
  not close all regressions.

## Quantitative Metric Guidance

The current benchmark ratios are baseline observations, not hard pass/fail product targets.
For this draft, the hard requirement is root-cause attribution with reproducible evidence.
If a minimal fix is attempted, quantitative improvement must be reported per shape against the
baseline, but no fixed ratio threshold is required at the draft stage.

## Path Boundaries

### Upper Bound

An acceptable upper-bound outcome includes:

- one or more small diagnostic utilities or benchmark flags
- one minimal production code change that validates the diagnosed bottleneck
- updated benchmark evidence and a concise final analysis

### Lower Bound

An acceptable lower-bound outcome includes:

- no production fix landed
- but a defensible root-cause conclusion supported by reproducible evidence and narrowed to a
  concrete layer or kernel family

### Explicit Non-Goals

- No unrelated cleanup-only refactors
- No broad migration of Paddle operators to new abstractions
- No performance claims without rerunning the full-shape benchmark set

## Dependencies and Sequencing

1. Confirm the benchmark baseline on the current branch.
2. Add the minimum instrumentation needed to attribute time by layer.
3. Compare Torch and Paddle backward paths using the instrumented evidence.
4. Select one leading hypothesis and validate it with the smallest possible change.
5. Rerun correctness checks and the full-shape benchmark set.
6. Summarize resolved and unresolved findings.

## RLCR Round Expectations

Each RLCR round summary should record:

- the current hypothesis
- the evidence gathered in that round
- whether the hypothesis was supported or rejected
- the next step and why it is now the highest-value next action

The loop should not advance by stacking speculative fixes. Each round must narrow uncertainty or
validate a concrete hypothesis.

## Implementation Notes

- Keep benchmark additions narrowly scoped to diagnosis of this regression.
- Prefer shape-by-shape evidence over aggregate averages when drawing conclusions.
- Treat `chunk_gdn` and `chunk_kda` as related but not automatically identical problems.
- Do not claim success based only on smoke benchmarks when the agreed scope is the full shape set.
