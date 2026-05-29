# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Concrete likelihood classes for the per-reliability-dimension
observation model P(target_j | signal_j(θ)).

Three classes, callable on a per-reliability-dimension basis:

- ``gauss_likelihood``       — negative log-likelihood of a Gaussian.
                                Baseline. Operationally equivalent to MSE;
                                a Gauss-only choice would collapse the
                                structural difference to MSE-RLHF, so the
                                Wasserstein and Huber alternatives below
                                exist to widen the likelihood class
                                genuinely.

- ``wasserstein_likelihood`` — squared W_2 distance between two
                                distributions, Sinkhorn-regularised. Use when
                                the reliability signal is a distribution
                                (e.g. a response distribution over multiple
                                sampled rollouts) rather than a point
                                estimate. Avoids the mode-collapse failure
                                mode of Gauss-NLL when signal and target are
                                both distributions.

- ``huber_likelihood``       — Huber M-estimator. Adversarial-resistant:
                                heavy-tailed contamination (a single drift
                                rollout) has bounded influence, unlike Gauss
                                where one outlier dominates the gradient.

A fourth class (e.g. categorical / Dirichlet / Beta for bounded or
ordinal reliability variables) slots in here without touching callers.
The architectural claim is not tied to a Gaussian assumption — it is
tied to causal state-conditioning; bounded / ordinal likelihood families
for sources, sufficiency and conflict are an ablated Stage-1 design
choice.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Gauss
# ---------------------------------------------------------------------------

def gauss_likelihood(signal: Tensor, target: Tensor, sigma: float | Tensor) -> Tensor:
    """Negative log-likelihood of a Gaussian observation model.

    Models ``T_j ~ N(s_j(θ), σ²)`` and returns ``-log p(T_j | s_j(θ))``,
    summed across all elements.

    Args:
        signal: Tensor of model-derived signal values ``s_j(θ, x)``.
        target: Tensor of trust-target values ``τ_j`` (same shape as ``signal``).
        sigma: Observation standard deviation. Scalar ``float`` or a Tensor
            broadcastable to ``signal``. Must be strictly positive.

    Returns:
        Scalar tensor with the summed negative log-likelihood::

            sum [ 0.5 * ((signal - target) / sigma)**2
                  + log(sigma) + 0.5 * log(2 * pi) ]

        Cross-checked against ``scipy.stats.norm.logpdf`` in the tests.

    Raises:
        ValueError: if ``sigma`` is non-positive.
    """
    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive (got {sigma}).")
        sigma_t = torch.tensor(float(sigma), dtype=signal.dtype, device=signal.device)
    else:
        if torch.any(sigma <= 0):
            raise ValueError("sigma must be strictly positive everywhere.")
        sigma_t = sigma.to(dtype=signal.dtype, device=signal.device)

    residual = (signal - target) / sigma_t
    log_norm = torch.log(sigma_t) + 0.5 * math.log(2.0 * math.pi)
    return (0.5 * residual.pow(2) + log_norm).sum()


# ---------------------------------------------------------------------------
# Wasserstein-2 (Sinkhorn-regularised)
# ---------------------------------------------------------------------------

def wasserstein_likelihood(
    signal_distribution: Tensor,
    target_distribution: Tensor,
    epsilon: float = 0.05,
    n_iter: int = 50,
) -> Tensor:
    """Sinkhorn-regularised squared Wasserstein-2 cost between two empirical
    distributions over the same support.

    Both inputs are interpreted as probability *weights* (non-negative,
    summing to 1 along the last axis) over a shared 1-D support, which is
    normalised internally to positions in ``[0, 1]``. This support
    normalisation keeps the default ``epsilon = 0.05`` numerically well-
    conditioned regardless of ``N`` — without it, large ``N`` causes the
    Sinkhorn kernel ``exp(-cost / ε)`` to underflow.

    The Sinkhorn algorithm (Cuturi 2013) is used as an entropic
    regularisation of optimal transport — strictly convex, differentiable,
    and O(N²) per iteration. This is a 1-D specialisation; Stage-1 work
    accepts arbitrary support tensors.

    Args:
        signal_distribution: Tensor of shape ``(..., N)`` with probability
            weights for the model-derived distribution.
        target_distribution: Tensor of shape ``(..., N)`` with probability
            weights for the trust-target distribution.
        epsilon: Sinkhorn entropic-regularisation strength on the
            normalised support ``[0, 1]``. Smaller ``ε`` → sharper
            transport plan but slower convergence.
        n_iter: Number of Sinkhorn matrix-scaling iterations.

    Returns:
        Scalar tensor with the summed regularised W_2² cost across all
        leading batch dimensions.
    """
    a = signal_distribution.clamp_min(1e-12)
    b = target_distribution.clamp_min(1e-12)
    a = a / a.sum(dim=-1, keepdim=True)
    b = b / b.sum(dim=-1, keepdim=True)

    n = a.shape[-1]
    # Normalise the 1-D support to [0, 1] so the default epsilon
    # holds regardless of N (otherwise exp(-cost/eps) underflows).
    positions = torch.linspace(0.0, 1.0, n, dtype=a.dtype, device=a.device)
    cost = (positions.unsqueeze(0) - positions.unsqueeze(1)).pow(2)  # (N, N)
    kernel = torch.exp(-cost / epsilon)  # (N, N)

    u = torch.ones_like(a)
    for _ in range(n_iter):
        v = b / (u @ kernel + 1e-12)
        u = a / (v @ kernel.T + 1e-12)

    transport = u.unsqueeze(-1) * kernel * v.unsqueeze(-2)
    return (transport * cost).sum()


# ---------------------------------------------------------------------------
# Huber
# ---------------------------------------------------------------------------

def huber_likelihood(signal: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    """Huber M-estimator loss — adversarial-resistant likelihood surrogate.

    Quadratic for residuals within ``delta``, linear outside — bounded-
    influence-function form. The corresponding observation model is the
    Huber pseudo-likelihood; not a normalised density, but a proper convex
    surrogate that is consistent under heavy-tailed contamination
    (Huber 1964).

    Args:
        signal: Tensor of model-derived signal values ``s_j(θ, x)``.
        target: Tensor of trust-target values ``τ_j``.
        delta: Crossover threshold between quadratic and linear regimes.
            Must be positive. Smaller ``delta`` → more robust to outliers,
            less sensitive in the bulk.

    Returns:
        Scalar tensor with the summed Huber loss.

    Raises:
        ValueError: if ``delta`` is non-positive.
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive (got {delta}).")
    residual = (signal - target).abs()
    quadratic = torch.minimum(residual, torch.tensor(delta, dtype=residual.dtype, device=residual.device))
    linear = residual - quadratic
    return (0.5 * quadratic.pow(2) + delta * linear).sum()
