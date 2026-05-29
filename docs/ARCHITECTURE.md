# Architecture

This document describes the design of `bayesian-native-ssm` and pins
down the open engineering decisions that are explicitly Stage-1 work.
A reader should walk away knowing (a) what the skeleton already
guarantees, (b) where the Stage-1 work hooks in, and (c) which research
questions remain genuinely open.

## 1. Stack split: library code vs Stage-1 work

The package is intentionally lop-sided. The loss side is implemented
in full; the model and training sides are interfaces only.

| Subsystem                       | Status in skeleton           | Stage-1 work                                                |
| ------------------------------- | ---------------------------- | ----------------------------------------------------------- |
| `losses/trust_posterior.py`     | Implemented + tested         | Hyperparameter exposure per AEGIS dimension                 |
| `losses/buhlmann.py`            | Implemented + tested         | Online estimator for `k = σ²/τ²`                             |
| `losses/likelihoods.py`         | Three classes implemented    | Optional fourth class (categorical/Dirichlet)               |
| `models/trust_state.py`         | Reference holder + update    | Selective-scan-integrated update                            |
| `models/mamba_wrapper.py`       | Interface stub               | Full integration with `mamba-ssm` kernel                    |
| `training/config.py`            | Pydantic schema              | Trainer that consumes the schema                            |
| `eval/trust_diagnostics.py`     | ECE + worst-group implemented| Sub-population partitioner that consumes trainer logs       |
| `eval/benchmarks.py`            | Interface stubs              | HELM/BBQ/MMLU wrappers + RLHF-baseline harness              |

The skeleton choice is deliberate: the loss-side primitives are the
parts of the theory that are most easily mis-implemented under time
pressure, so they get pinned down first and tested. The backbone
integration is mechanical engineering work that depends on a CUDA host
and is therefore Stage-1.

## 2. Trust-State augmentation

The state carried by the backbone is augmented from `h_t ∈ ℝ^d` to

```
h̃_t = (h_t, τ_t) ∈ ℝ^(d + d_τ)
```

where `τ_t ∈ ℝ^d_τ` is the propagated trust-posterior slot. The
skeleton's reference update is a simple exponential decay,

```
τ_t = decay · τ_{t-1} + Δτ_t
```

and exists in `AugmentedTrustState.update_tau` so notebooks and unit
tests can exercise the shape contract on CPU. The Stage-1 production
update replaces this with a selective gate driven by `(x_t, τ_t)`; see
§4 for the open research question.

`d_τ` is a hyperparameter. Sensible starting range for ≤1B-parameter
models is `d_τ ∈ [8, 64]`. Empirical study of optimal `d_τ` is
Stage-1 work and surfaces directly on the `BackboneSpec.trust_dim`
field of the config schema.

## 3. Mamba integration plan

**Variant.** Mamba-2 is the Stage-1 primary backbone, with classic
Mamba available as a fallback for hyperparameter ablations. RWKV is
the Stage-1 ablation comparator and lives behind the same
`MambaTrustWrapper` interface — naming holds because the interface is
about "selective-scan-with-trust", not about which specific kernel
implements it.

**Modification points in the upstream kernel.** Three candidate sites
exist for hooking trust-state propagation into the selective scan:

1. **Post-layer hook (cheapest).** Run the upstream scan unmodified;
   compute `Δτ_t` from the post-layer hidden state via a learned head;
   update `τ_t` outside the scan. Loss: no influence of `τ_t` on the
   backbone's selective dynamics.

2. **Pre-scan input conditioning (medium).** Concatenate `τ_t` to the
   token embedding before the scan. Cheap in code; gives `τ_t` a route
   into the selective gating without modifying the kernel. Loss: `τ_t`
   only enters at the layer boundary, not inside the scan.

3. **In-scan gating (full).** Make the selective matrices
   `A(x_t, τ_t), B(x_t, τ_t), C(x_t, τ_t)` τ-conditional. Requires a
   forked scan kernel. Highest expressivity, highest engineering cost.

Stage-1 starts with (1), proves the loss-side machinery converges,
then promotes to (2). Decision on (3) is deferred to Stage-1 mid-point
based on whether (2) hits the trust-diagnostic targets.

## 4. Open research question — selective gating on τ_t

Whether the selective parameters should be τ-conditional is the single
unresolved theoretical question in the backbone design. Arguments for:
trust-updates are input-dependent and should therefore enter the
selective mechanism in the same way as token content does. Arguments
against: τ-conditional gating couples backbone capacity to trust
signal, which complicates the comparison against a frozen-backbone
baseline (cannot disentangle "trust loss helps" from "extra capacity
helps").

Stage-1 resolves this by training (a) frozen-selective + post-hoc-τ
and (b) τ-conditional-selective under matched parameter budgets, and
comparing on the trust-diagnostic + capability-baseline pair. The
config schema already separates `BackboneSpec.trust_dim` from the
selective parameters so both runs can share infrastructure.

## 5. Evaluation pipeline

The eval surface has two levels:

**External benchmarks** (`eval/benchmarks.py`). Wrappers around HELM-
Safety, BBQ, and MMLU. The wrappers are stubs in the skeleton. The
RLHF-comparison method is signature-pinned to refuse ad-hoc self-
trained RLHF baselines — Stage-1 baseline is Pythia-160M + Tülu-3 (or
TinyLlama-1.1B-Chat), both published with documented RLHF pipelines.
The skeleton enforces this by rejecting unknown baseline names in
`run_rlhf_baseline_comparison`.

**Internal trust diagnostics** (`eval/trust_diagnostics.py`). Two
metrics are implemented: expected calibration error (ECE) and worst-
group accuracy. Both are real because the loss tests and the LAB_DEMO
notebooks call them end-to-end. Worst-group accuracy is the diagnostic
that surfaces the Goodhart-style sub-population drift the Bühlmann
weighting is supposed to suppress — it is the empirical hook for the
theoretical claim in `LAB_THEORY_mathematik.md` §5.

## 6. Pending engineering decisions (carried to LAB_BUILD)

These are explicit deferred decisions, not unknowns:

- Online estimator for the Bühlmann crossover `k`. Currently a
  config-supplied constant; Stage-1 work is a running estimator from
  process/structural variances during training.
- Whether `λ_j` weights are static or learned. Skeleton treats them as
  static config; Stage-2 may relax this with a Dirichlet prior.
- Multi-modal trust signal sources (M7 in `LAB_BUILD_roadmap`). The
  skeleton's `trust_signals` dict accepts arbitrary keys, so the
  interface does not need to change when new signal sources land.
- CUDA kernel fork-vs-upstream-extension for option (3) in §3.

Everything else is implementation, not design.
