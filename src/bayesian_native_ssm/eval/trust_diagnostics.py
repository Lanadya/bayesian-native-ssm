# Copyright 2026 Posterior Labs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Internal trust diagnostics: calibration ECE and worst-group accuracy.

These are *real* implementations — the eval-pipeline glue and the
LAB_DEMO notebooks both consume them. The benchmark wrappers in
``benchmarks.py`` are stubs; the metrics here are not.
"""

from __future__ import annotations

import torch
from torch import Tensor


def expected_calibration_error(
    confidences: Tensor,
    correct: Tensor,
    n_bins: int = 15,
) -> Tensor:
    """Expected Calibration Error (Guo et al. 2017).

    Args:
        confidences: Tensor of shape ``(N,)`` with model confidence on
            the predicted class (max-softmax), in ``[0, 1]``.
        correct: Boolean / 0-1 tensor of shape ``(N,)`` indicating whether
            the predicted class was correct.
        n_bins: Number of equal-width confidence bins.

    Returns:
        Scalar tensor with ECE — lower is better. Reference value:
        a uniformly-random binary classifier has ECE ≈ 0; a softmaxed
        but over-confident classifier has ECE >> 0.
    """
    if confidences.shape != correct.shape:
        raise ValueError(
            f"shape mismatch: confidences {tuple(confidences.shape)} vs "
            f"correct {tuple(correct.shape)}."
        )
    correct = correct.to(dtype=confidences.dtype)
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidences.device)
    ece = torch.zeros((), dtype=confidences.dtype, device=confidences.device)
    n = confidences.numel()
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        bin_count = mask.sum()
        if bin_count > 0:
            acc = correct[mask].mean()
            conf = confidences[mask].mean()
            ece = ece + (bin_count.to(confidences.dtype) / n) * (acc - conf).abs()
    return ece


def worst_group_accuracy(
    correct: Tensor,
    group_assignments: Tensor,
) -> Tensor:
    """Worst-group accuracy — minimum per-group accuracy across sub-populations.

    Directly relevant to LAB_REDTECH Killshot 3 mitigation: this is the
    diagnostic that surfaces sub-population drift the Bühlmann weighting
    is supposed to suppress.

    Args:
        correct: 0/1 tensor of shape ``(N,)``.
        group_assignments: Long tensor of shape ``(N,)`` with group ids
            ``∈ {0, ..., G-1}``.

    Returns:
        Scalar tensor with the minimum mean accuracy over groups that
        actually have at least one sample.
    """
    if correct.shape != group_assignments.shape:
        raise ValueError(
            f"shape mismatch: correct {tuple(correct.shape)} vs "
            f"group_assignments {tuple(group_assignments.shape)}."
        )
    correct = correct.to(torch.float32)
    n_groups = int(group_assignments.max().item()) + 1
    per_group = []
    for g in range(n_groups):
        mask = group_assignments == g
        if mask.any():
            per_group.append(correct[mask].mean())
    if not per_group:
        raise ValueError("No groups with samples found.")
    return torch.stack(per_group).min()
