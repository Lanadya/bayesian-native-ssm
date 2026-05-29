# bayesian-native-ssm

Reference skeleton accompanying the SPRIND Next Frontier AI Challenge
Stage-1 application **Credence-State Foundation Models** (ARQON GmbH
in formation, founder/PI Nina Klee, May 2026).

The repository pins down the loss-side primitives that the
architectural proposal requires, plus an honest stub layer for the
Mamba/RWKV backbone integration that is itself Stage-1 implementation
work.

This is **not** a working model. It is the engineering substrate
underneath the architectural proposal — the place where the
Bühlmann-Straub credibility weighting (exposure form), the
five-component loss decomposition, and three concrete likelihood
classes live as runnable, type-hinted, tested code that a Stage-1 ML
engineer can start extending in under 30 minutes.

## Architecture (one paragraph)

The model maintains an amortized posterior-structured approximation
`q_φ(z_t | x_≤t, o_≤t, a_<t)` over reliability variables
`z_t = (u_t, r_t, s_t, k_t)` — epistemic uncertainty, source/context
reliability, evidential sufficiency, conflict. The Credence-State `c_t`
contains the posterior parameters needed for control (initial
implementation `c_t = (μ_t, log σ_t)` under a diagonal Gaussian, with
mean-only vs. variance-aware vs. sampled conditioning as Stage-1
ablations). The generative state update is causally conditioned by
this state: `h_t = F_θ(h_{t-1}, x_t, c_{t-1})`. Output/action decisions
follow `G_θ(h_t, c_t)` with action-conditioned risk `ρ_t(a)`. See the
SPRIND submission Field 4 for the full architectural specification.

## Loss decomposition

The five-component loss the application proposes is

```
L(θ, φ) = L_LM + λ₁·L_credence + λ₂·L_update + λ₃·L_action + λ₄·L_calibration
```

with `L_update` carrying a Bühlmann-Straub credibility-weighted
shrinkage prior in exposure form

```
Z = w / (w + k)
```

— noisy local evidence is weighted against population-level reliability
estimates by exposure and variance. Stage 1 tests whether this improves
update stability and calibration; we do not claim Bayesian optimality
from the theory alone.

## Naming map (legacy → submission)

The code uses pre-submission internal names. The mapping to the
terminology used in the SPRIND submission is:

| Code symbol / module                | Submission terminology                          |
| ----------------------------------- | ----------------------------------------------- |
| `trust_posterior` (module)          | Credence-State loss assembly                    |
| `L_data + L_trust + L_prior`        | Renamed; v5 uses five-component decomposition above |
| AEGIS trust tensor `(c, j)`         | Credence-State `c_t` over reliability variables `z_t = (u_t, r_t, s_t, k_t)` |
| AEGIS dimension weights `λ_j`       | Loss weights `λ₁..λ₄` in five-component form    |
| `buhlmann.py` exposure-form `Z`     | Update prior in `L_update` (unchanged)          |
| `trust_state.py` augmented state    | `(h_t, c_t)` recurrent state pair                |

Function and module names are deliberately not renamed to keep the
existing test suite intact. A Stage-1 milestone is migrating the
internal naming to the submission terminology in lock-step with the
data-pipeline build-out.

## Stack overview

```
src/bayesian_native_ssm/
├── losses/
│   ├── trust_posterior.py   # Credence-State loss assembly
│   ├── buhlmann.py          # exposure-form Z, weighted-loss combinator
│   └── likelihoods.py       # Gauss / Wasserstein-2 / Huber
├── models/
│   ├── mamba_wrapper.py     # Stage-1 stub around mamba-ssm
│   └── trust_state.py       # augmented state (h_t, c_t)
├── training/
│   └── config.py            # Pydantic schema for YAML experiment configs
└── eval/
    ├── benchmarks.py        # MMLU/HumanEval capability-floor stubs
    └── trust_diagnostics.py # ECE, worst-group accuracy
tests/
├── test_losses.py           # likelihood correctness + dispatch
└── test_buhlmann.py         # exposure-vs-counting form, long-tail pulling
docs/ARCHITECTURE.md         # design doc for the Stage-1 integration
```

## Quick start

```bash
uv pip install -e ".[dev]"
pytest tests/
```

The base install brings CPU PyTorch, NumPy, Pydantic, and PyYAML — that
is enough to run the loss tests. The Mamba integration is gated behind
the `[mamba]` extra because `mamba-ssm` needs a CUDA toolchain and not
every contributor has a GPU locally:

```bash
uv pip install -e ".[mamba]"   # GPU required
```

## Roadmap

| Status                     | Item                                                                  |
| -------------------------- | --------------------------------------------------------------------- |
| Implemented (skeleton)     | Credence-State loss assembly, Bühlmann-Straub exposure-form Z, three likelihoods |
| Implemented (skeleton)     | Pydantic config schema, calibration ECE, worst-group accuracy diagnostic |
| In development (Stage 1)   | Mamba-ssm integration in `models/mamba_wrapper.py`; selective-parameter conditioning on `(x_t, c_t)` |
| In development (Stage 1)   | Five-component loss training loop, RLHF baseline harness, custom Credence-State eval suite |
| In development (Stage 1)   | Reliability-event label pipeline (FEVER, MultiClaim, NaturalQuestions/SQuAD-Sufficient, FEVER + Wikipedia edit-histories) |
| In development (Stage 1)   | Seven-variant ablation matrix + five intervention tests (incl. path-patching) |
| Planned (Stage 2)          | 1B–3B-class Credence-State model; reproducible MLOps/eval platform; public-administration or legal pilot |
| Planned (Stage 3)          | 3B–10B-class model family; European lab operations; regulated-sector deployment path |

## Scope of this skeleton

This repository ships only what can be ship-grade today: the loss-side
math, the Pydantic config, the diagnostic metrics, and the public
interfaces of the Stage-1 work. Wherever the Stage-1 integration is
not yet done — the Mamba forward pass, the training loop, the
benchmark wrappers — the corresponding method raises
`NotImplementedError` with a pointer to the design doc. No fake
training code, no placeholder gradients, no silent stubs.

## License

Apache 2.0. See `LICENSE`.

## Citation

This implementation accompanies the SPRIND NFAI 2026 Stage-1
application from ARQON GmbH (in formation). Citation details will be
added once the companion technical report has an arXiv identifier.

## Contact

Nina Klee — nina.klee@arqon.group
