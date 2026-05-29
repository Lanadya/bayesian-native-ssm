# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Bühlmann-Straub credibility weighting.

Reference: Bühlmann & Straub, "Glaubwürdigkeit für Schadensätze",
Mitteilungen der Vereinigung Schweizerischer Versicherungsmathematiker
70 (1970), 111-133.

This module implements the **exposure form**

    Z_c = w_c / (w_c + k)

with exposure weights ``w_c`` per sub-population. It is **not** the
counting form ``Z = N / (N + k)`` from Bühlmann 1967, which is a
narrower special case (uniform exposure). The distinction is
load-bearing for sub-population credibility weighting in ML training
and is enforced by ``tests/test_buhlmann.py::test_buhlmann_z_correct_form``.

In the Credence-State Foundation Models architecture (SPRIND submission
May 2026) this weighting enters ``L_update`` as a shrinkage-style update
prior. The variance-optimality result holds in the original linear-class
setting; transferring it to over-parametrised neural networks is
treated as a heuristic, not as a theorem.
"""

from __future__ import annotations

import torch
from torch import Tensor


def buhlmann_z(exposure_weights: Tensor, k: float) -> Tensor:
    """Bühlmann-Straub credibility factor in exposure form.

    Computes ``Z_c = w_c / (w_c + k)`` for each sub-population ``c``.

    Args:
        exposure_weights: Tensor of shape ``(C,)`` with non-negative
            exposure weights ``w_c`` per sub-population. May be sample
            counts, importance masses, or any confidence-aggregated
            non-negative quantity.
        k: Crossover constant ``k = E[process-variance] / Var[structural-mean]``
            ``= σ² / τ²``. Must be positive. In actuarial practice ``k`` is
            calibrated from historical data; in the NN setting it must be
            estimated from running process/structural variances during
            training (Stage-1 research question).

    Returns:
        Tensor of shape ``(C,)`` with credibility factors ``Z_c ∈ [0, 1)``.
        Small ``w_c`` (rare sub-populations) pull ``Z_c → 0``, dragging
        the local estimate toward the global mean — automatic variance
        regularisation against overfitting to rare classes.

    Raises:
        ValueError: if ``k <= 0`` or any ``exposure_weights < 0``.
    """
    if k <= 0:
        raise ValueError(f"k must be positive (got {k}); k = σ²/τ² is a variance ratio.")
    if torch.any(exposure_weights < 0):
        raise ValueError("exposure_weights must be non-negative.")
    return exposure_weights / (exposure_weights + k)


def buhlmann_weighted_loss(
    local_losses: Tensor,
    global_loss: Tensor,
    z: Tensor,
) -> Tensor:
    """Credibility-weighted convex combination of local and global loss.

    Computes ``L_c = Z_c · L̂_c + (1 - Z_c) · L̂_global`` per sub-population.

    Args:
        local_losses: Tensor of shape ``(C,)`` (or broadcast-compatible) with
            sub-population-specific sample-loss estimates ``L̂_c(θ)``.
        global_loss: Scalar (or broadcast-compatible) tensor with the
            population-global loss ``L̂_global(θ)``.
        z: Tensor of shape ``(C,)`` of credibility factors from
            :func:`buhlmann_z`.

    Returns:
        Tensor of shape ``(C,)`` with weighted per-context losses ``L_c``.
        To aggregate to a scalar training loss across contexts, weight
        by relative context priors ``π_c`` and sum::

            L_pop = (pi * buhlmann_weighted_loss(L_c, L_global, Z)).sum()
    """
    return z * local_losses + (1.0 - z) * global_loss
