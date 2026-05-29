# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Credence-State training-loss assembly.

Implements a three-component pre-Stage-1 decomposition of the
negative-log-posterior::

    L(θ) = L_data(θ) + L_trust(θ) + L_prior(θ)
         = -log P(D | θ) - log P(T | θ) - log P(θ)

This is the loss form the skeleton ships with. The companion SPRIND
submission "Credence-State Foundation Models" describes the five-
component v5 form

    L = L_LM + λ_1 L_credence + λ_2 L_update + λ_3 L_action + λ_4 L_calibration

with the migration ``L_data → L_LM`` and ``L_trust → L_credence + L_update``
(L_credence carries the per-reliability-dimension likelihood mass,
L_update carries the Bühlmann-Straub credibility shrinkage). Stage-1
expands this skeleton into the five-component form and adds
``L_action`` and ``L_calibration``.

Three components in the skeleton:

- **L_data**  — token cross-entropy on next-token prediction
                (``-Σ_t log P_θ(y_t | y_<t, x)``).

- **L_trust** — dimension-factorised reliability log-likelihood
                ``Σ_j λ_j L_j(θ)`` over the reliability-variable index
                ``j`` (in v5: a component of the Credence-State
                ``c_t = (u_t, r_t, s_t, k_t)``), with an optional
                Bühlmann-Straub credibility weighting across sub-
                population contexts.

- **L_prior** — Gaussian weight-decay prior, attached by the caller;
                we expose a slot here so the bookkeeping is honest, but
                no model parameters are owned at this layer.

The likelihood class is a caller-required choice (``"gauss"`` |
``"wasserstein"`` | ``"huber"``) — see ``likelihoods.py``. There is no
default that silently collapses to MSE.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from bayesian_native_ssm.losses.buhlmann import buhlmann_weighted_loss, buhlmann_z
from bayesian_native_ssm.losses.likelihoods import (
    gauss_likelihood,
    huber_likelihood,
    wasserstein_likelihood,
)

_LIKELIHOOD_CLASSES = {"gauss", "wasserstein", "huber"}


def _per_dimension_trust_loss(
    likelihood_class: str,
    signal: Tensor,
    target: Tensor,
) -> Tensor:
    """Dispatch to the chosen likelihood class for one reliability dimension.

    Hyperparameters of each likelihood (``sigma``, ``epsilon``, ``delta``)
    are pinned to neutral defaults here; the Stage-1 trainer will surface
    them per-dimension through the config schema (``training/config.py``).
    """
    if likelihood_class == "gauss":
        return gauss_likelihood(signal, target, sigma=1.0)
    if likelihood_class == "wasserstein":
        return wasserstein_likelihood(signal, target)
    if likelihood_class == "huber":
        return huber_likelihood(signal, target, delta=1.0)
    raise ValueError(
        f"Unknown likelihood_class {likelihood_class!r}. "
        f"Expected one of {_LIKELIHOOD_CLASSES}."
    )


def trust_posterior_loss(
    model_output: Tensor,
    data_targets: Tensor,
    trust_signals: dict[str, Tensor],
    trust_targets: dict[str, Tensor],
    context_assignments: Optional[Tensor] = None,
    likelihood_class: str = "gauss",
    buhlmann_k: float = 1.0,
    lambdas: Optional[dict[str, float]] = None,
    prior_term: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Assemble L_total = L_data + L_trust + L_prior.

    Args:
        model_output: Logits tensor of shape ``(B, T, V)`` for the data
            likelihood (next-token prediction).
        data_targets: Long tensor of shape ``(B, T)`` with target token
            ids. Indices below zero are treated as padding and ignored
            (standard ``ignore_index = -100`` convention).
        trust_signals: Mapping from reliability-dimension name ``j`` to
            the model-derived signal tensor ``s_j(θ, x)`` for that
            dimension. In the v5 frame, ``j`` indexes a component of the
            Credence-State ``c_t = (u_t, r_t, s_t, k_t)`` —
            uncertainty / source-reliability / evidential-sufficiency /
            conflict.
        trust_targets: Mapping from the same dimension names ``j`` to the
            externally-supplied reliability-target tensor (same shape as
            the corresponding signal).
        context_assignments: Optional long tensor of shape ``(B,)``
            assigning each batch element to a sub-population context
            ``c ∈ {0, ..., C-1}``. When provided, the per-dimension
            reliability loss is Bühlmann-Straub-weighted across contexts.
            When ``None``, all samples are treated as one global context
            and ``L_trust`` is the unweighted sum.
        likelihood_class: One of ``{"gauss", "wasserstein", "huber"}``.
            Caller-required choice — no silent default that would
            collapse the per-dimension likelihood to MSE.
        buhlmann_k: Crossover constant ``k = σ²/τ²`` for the credibility
            factor. Ignored if ``context_assignments is None``.
        lambdas: Mapping from reliability-dimension name to its weight
            ``λ_j`` in the per-dimension aggregation. Defaults to uniform
            ``λ_j = 1.0`` per provided dimension. Unknown keys raise.
        prior_term: Optional scalar tensor with a precomputed prior
            contribution ``L_prior``. The trainer owns the parameters,
            so the prior is supplied from outside; defaults to zero.

    Returns:
        Dict with scalar tensors keyed ``"L_data"``, ``"L_trust"``,
        ``"L_prior"``, ``"L_total"``. Suitable for direct ``.backward()``
        on ``L_total`` while logging the components.

    Raises:
        ValueError: on key mismatch between signals/targets/lambdas, or
            on an unknown likelihood class.
    """
    # ---- L_data ----------------------------------------------------------
    if model_output.dim() != 3:
        raise ValueError(
            f"model_output must be (B, T, V); got shape {tuple(model_output.shape)}."
        )
    b, t, v = model_output.shape
    l_data = F.cross_entropy(
        model_output.reshape(b * t, v),
        data_targets.reshape(b * t),
        ignore_index=-100,
        reduction="sum",
    )

    # ---- L_trust ---------------------------------------------------------
    if set(trust_signals) != set(trust_targets):
        raise ValueError(
            "trust_signals and trust_targets must have identical keys; "
            f"got {sorted(trust_signals)} vs {sorted(trust_targets)}."
        )
    if lambdas is None:
        lambdas = {j: 1.0 for j in trust_signals}
    else:
        unknown = set(lambdas) - set(trust_signals)
        if unknown:
            raise ValueError(f"lambdas has unknown keys: {sorted(unknown)}.")

    l_trust = torch.zeros((), dtype=model_output.dtype, device=model_output.device)
    for j, signal in trust_signals.items():
        target = trust_targets[j]
        lam = float(lambdas.get(j, 1.0))

        if context_assignments is None:
            l_j = _per_dimension_trust_loss(likelihood_class, signal, target)
        else:
            # Bühlmann-Straub credibility weighting across sub-population
            # contexts. Local loss is computed per context; the global
            # loss is computed once over the full batch.
            global_loss_j = _per_dimension_trust_loss(likelihood_class, signal, target)
            n_contexts = int(context_assignments.max().item()) + 1

            local_losses = torch.zeros(n_contexts, dtype=signal.dtype, device=signal.device)
            exposure = torch.zeros(n_contexts, dtype=signal.dtype, device=signal.device)
            for c in range(n_contexts):
                mask = (context_assignments == c)
                w_c = mask.sum().to(signal.dtype)
                exposure[c] = w_c
                if w_c > 0:
                    local_losses[c] = _per_dimension_trust_loss(
                        likelihood_class, signal[mask], target[mask]
                    )

            z = buhlmann_z(exposure, k=buhlmann_k)
            per_ctx = buhlmann_weighted_loss(local_losses, global_loss_j, z)
            # Aggregate contexts with exposure-proportional priors π_c.
            pi = exposure / exposure.sum().clamp_min(1e-12)
            l_j = (pi * per_ctx).sum()

        l_trust = l_trust + lam * l_j

    # ---- L_prior ---------------------------------------------------------
    if prior_term is None:
        l_prior = torch.zeros((), dtype=model_output.dtype, device=model_output.device)
    else:
        l_prior = prior_term.to(dtype=model_output.dtype, device=model_output.device)

    l_total = l_data + l_trust + l_prior
    return {
        "L_data": l_data,
        "L_trust": l_trust,
        "L_prior": l_prior,
        "L_total": l_total,
    }
