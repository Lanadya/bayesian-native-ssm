# Architecture

This document describes the design of `bayesian-native-ssm` and pins
down the open engineering decisions that are explicitly Stage-1 work.
A reader should walk away knowing (a) what the skeleton already
guarantees, (b) where the Stage-1 work hooks in, and (c) which research
questions remain genuinely open.

The companion SPRIND submission *Credence-State Foundation Models*
(May 2026) describes the architectural proposal in full; this
document is the engineering-side complement and uses the same v5
terminology. Code symbol names retain their legacy labels (`τ_t`,
`TrustStateConfig`, `trust_posterior_loss`, `AEGIS-dimension`) to keep
the existing test suite intact; the mapping to v5 terminology lives in
the repository README.

## 1. Stack split: library code vs Stage-1 work

The package is intentionally lop-sided. The loss side is implemented
in full; the model and training sides are interfaces only.

| Subsystem                       | Status in skeleton           | Stage-1 work                                                |
| ------------------------------- | ---------------------------- | ----------------------------------------------------------- |
| `losses/trust_posterior.py`     | Implemented + tested         | Per-reliability-dimension hyperparameter exposure           |
| `losses/buhlmann.py`            | Implemented + tested         | Online estimator for `k = σ²/τ²`                             |
| `losses/likelihoods.py`         | Three classes implemented    | Optional fourth class (categorical/Dirichlet/Beta for bounded or ordinal reliability variables) |
| `models/trust_state.py`         | Reference holder + update    | Selective-scan-integrated update                            |
| `models/mamba_wrapper.py`       | Interface stub               | Full integration with `mamba-ssm` kernel                    |
| `training/config.py`            | Pydantic schema              | Trainer that consumes the schema                            |
| `eval/trust_diagnostics.py`     | ECE + worst-group implemented| Custom Credence-State eval suite consuming trainer logs     |
| `eval/benchmarks.py`            | Interface stubs              | MMLU-subset / HumanEval capability-floor wrappers + published-RLHF-baseline harness |

The skeleton choice is deliberate: the loss-side primitives are the
parts of the theory that are most easily mis-implemented under time
pressure, so they get pinned down first and tested. The backbone
integration is mechanical engineering work that depends on a CUDA host
and is therefore Stage-1.

## 2. Credence-State augmentation

The recurrent state carried by the backbone is augmented from
`h_t ∈ ℝ^d` to

```
h̃_t = (h_t, c_t) ∈ ℝ^(d + d_c)
```

where `c_t ∈ ℝ^{d_c}` is the propagated Credence-State slot. `c_t`
contains the posterior parameters over the reliability variables
`z_t = (u_t, r_t, s_t, k_t)`; initial implementation
`c_t = (μ_t, log σ_t)` under a diagonal Gaussian, with mean-only vs.
variance-aware vs. sampled conditioning as Stage-1 ablations.

> Code symbol mapping: in the source, `c_t` is represented by the
> tensor named `τ_t` (legacy naming), `d_c` is configured via
> `TrustStateConfig.trust_dim`, and the holder class is
> `AugmentedTrustState`.

The skeleton's reference update is a simple exponential decay,

```
c_t = decay · c_{t-1} + Δc_t
```

and exists in `AugmentedTrustState.update_tau` so notebooks and unit
tests can exercise the shape contract on CPU. The Stage-1 production
update replaces this with a selective gate driven by `(x_t, c_t)`; see
§4 for the open research question.

`d_c` is a hyperparameter. Sensible starting range for ≤1B-parameter
models is `d_c ∈ [8, 64]`. Empirical study of optimal `d_c` is
Stage-1 work and surfaces directly on the `BackboneSpec.trust_dim`
field of the config schema.

## 3. Mamba integration plan

**Variant.** Mamba-2 is the Stage-1 primary backbone, with classic
Mamba available as a fallback for hyperparameter ablations. RWKV is
the Stage-1 ablation comparator and lives behind the same
`MambaTrustWrapper` interface — naming holds because the interface is
about "selective-scan-with-Credence-State", not about which specific
kernel implements it.

**Modification points in the upstream kernel.** Three candidate sites
exist for hooking Credence-State propagation into the selective scan:

1. **Post-layer hook (cheapest).** Run the upstream scan unmodified;
   compute `Δc_t` from the post-layer hidden state via a learned head;
   update `c_t` outside the scan. Loss: no influence of `c_t` on the
   backbone's selective dynamics.

2. **Pre-scan input conditioning (medium).** Concatenate `c_t` to the
   token embedding before the scan. Cheap in code; gives `c_t` a route
   into the selective gating without modifying the kernel. Loss: `c_t`
   only enters at the layer boundary, not inside the scan.

3. **In-scan selective-parameter conditioning (full).** Make the
   input-dependent selective parameters (gates, discretization
   parameters, projections) functions of `(x_t, c_t)` rather than
   `x_t` alone. Requires a forked scan kernel. Highest expressivity,
   highest engineering cost.

Stage-1 starts with (1), proves the loss-side machinery converges,
then promotes to (2). Decision on (3) is deferred to Stage-1 mid-point
based on whether (2) hits the Credence-State diagnostic targets.

## 4. Open research question — selective-parameter conditioning on c_t

Whether the selective parameters should be `(x_t, c_t)`-conditional
rather than `x_t`-only is the single unresolved theoretical question
in the backbone design. Arguments for: Credence-State updates are
input-dependent and should therefore enter the selective mechanism in
the same way as token content does. Arguments against: making the
selective parameters `c_t`-conditional couples backbone capacity to
the reliability signal, which complicates the comparison against a
frozen-backbone baseline (cannot disentangle "Credence-State loss
helps" from "extra capacity helps").

Stage-1 resolves this by training (a) frozen-selective + post-hoc
Credence-State and (b) `(x_t, c_t)`-conditional selective parameters
under matched parameter budgets, and comparing on the Credence-State
diagnostic + capability-baseline pair. The config schema already
separates `BackboneSpec.trust_dim` (i.e. `d_c`) from the selective
parameters so both runs can share infrastructure.

## 5. Evaluation pipeline

The eval surface has two levels:

**External benchmarks** (`eval/benchmarks.py`). Wrappers around
HELM-style holistic evaluation (hygiene only, per the v5 submission),
BBQ, and MMLU-subset capability-floor checks. The wrappers are stubs
in the skeleton. The RLHF-comparison method is signature-pinned to
refuse ad-hoc self-trained RLHF baselines — Stage-1 baseline is
Pythia-160M + Tülu-3 (or TinyLlama-1.1B-Chat), both published with
documented RLHF pipelines. The skeleton enforces this by rejecting
unknown baseline names in `run_rlhf_baseline_comparison`.

**Internal Credence-State diagnostics** (`eval/trust_diagnostics.py`).
Two metrics are implemented: expected calibration error (ECE) and
worst-group accuracy. Both are real because the loss tests and the
toy-demonstration notebook call them end-to-end. Worst-group accuracy
is the diagnostic that surfaces the Goodhart-style sub-population
drift the Bühlmann-Straub credibility weighting is intended to
suppress.

The v5 submission's four primary metrics (Field 12) — proxy-gold gap,
Credence calibration, risk-adjusted action quality, causal state use
— are the Stage-1 expansion of this eval surface.

## 6. Pending engineering decisions

These are explicit deferred decisions, not unknowns:

- Online estimator for the Bühlmann crossover `k`. Currently a
  config-supplied constant; Stage-1 work is a running estimator from
  process/structural variances during training.
- Whether `λ_j` weights are static or learned. Skeleton treats them as
  static config; Stage-2 may relax this with a Dirichlet prior.
- CUDA kernel fork-vs-upstream-extension for option (3) in §3.
- Reliability-event label pipeline (FEVER, MultiClaim,
  NaturalQuestions / SQuAD-Sufficient, FEVER + Wikipedia
  edit-histories) — the skeleton's `trust_signals` dict accepts
  arbitrary keys, so the interface does not need to change when new
  signal sources land.

Everything else is implementation, not design.
