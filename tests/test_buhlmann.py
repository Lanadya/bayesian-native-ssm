# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the Bühlmann-Straub credibility weighting module.

Two of the tests are load-bearing:

- ``test_buhlmann_z_correct_form`` pins the **exposure form**
  ``Z = w / (w + k)`` (Straub 1970) and demonstrates explicit
  divergence from the counting form ``Z = N / (N + k)`` (Bühlmann 1967).
  The distinction matters: the counting form is a narrower special
  case under uniform exposure.

- ``test_buhlmann_z_long_tail_pulling`` shows the rare-subgroup
  regularisation behaviour that motivates the choice over a flat mean.
"""

from __future__ import annotations

import pytest
import torch

from bayesian_native_ssm.losses.buhlmann import buhlmann_weighted_loss, buhlmann_z


def test_buhlmann_z_correct_form():
    """Z must equal w/(w+k), exposure form (Straub 1970) — NOT N/(N+k)."""
    # Exposure weights that intentionally differ from the sample counts.
    # If a buggy implementation used counts (here all equal to 1) the two
    # would coincide. We make exposure ≠ count so the divergence is visible.
    w = torch.tensor([0.5, 2.0, 10.0])
    k = 1.0

    z = buhlmann_z(w, k=k)

    expected_exposure_form = w / (w + k)
    torch.testing.assert_close(z, expected_exposure_form, rtol=0.0, atol=0.0)

    # Show the counting form gives a different answer, so the test
    # genuinely discriminates between the two.
    n_samples_per_class = torch.tensor([1.0, 1.0, 1.0])
    counting_form = n_samples_per_class / (n_samples_per_class + k)
    assert not torch.allclose(z, counting_form), (
        "Implementation appears to use counting form Z = N/(N+k); "
        "must be exposure form Z = w/(w+k) per Bühlmann-Straub 1970."
    )


def test_buhlmann_z_long_tail_pulling():
    """Rare subgroups (small w_c) pull toward the global loss."""
    w = torch.tensor([0.01, 100.0])  # one very rare, one very common
    k = 1.0
    z = buhlmann_z(w, k=k)

    # Rare class: Z very small → local loss is essentially ignored.
    assert z[0] < 0.05, f"rare-class Z should be ≪ 1, got {z[0].item()}"
    # Common class: Z ≈ 1 → local loss is trusted.
    assert z[1] > 0.95, f"common-class Z should be ≈ 1, got {z[1].item()}"

    # And the weighted-loss combiner pulls the rare class toward the global mean.
    local_losses = torch.tensor([5.0, 5.0])  # both equally bad locally
    global_loss = torch.tensor(1.0)  # but globally things look fine
    weighted = buhlmann_weighted_loss(local_losses, global_loss, z)

    # Rare class: pulled hard toward global (~1.0).
    assert weighted[0] < 1.5, (
        f"rare-class loss should be pulled toward global ~1.0; got {weighted[0].item()}"
    )
    # Common class: stays near its local value (~5.0).
    assert weighted[1] > 4.5, (
        f"common-class loss should retain local autonomy ~5.0; got {weighted[1].item()}"
    )


def test_buhlmann_z_rejects_invalid_inputs():
    """Negative exposure or non-positive k must raise."""
    with pytest.raises(ValueError, match="positive"):
        buhlmann_z(torch.tensor([1.0]), k=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        buhlmann_z(torch.tensor([-1.0, 2.0]), k=1.0)


def test_buhlmann_weighted_loss_convex_combination():
    """L_c = Z_c · L̂_c + (1 - Z_c) · L̂_global is a convex combination."""
    z = torch.tensor([0.0, 0.5, 1.0])
    local_losses = torch.tensor([10.0, 10.0, 10.0])
    global_loss = torch.tensor(2.0)
    weighted = buhlmann_weighted_loss(local_losses, global_loss, z)
    torch.testing.assert_close(
        weighted,
        torch.tensor([2.0, 6.0, 10.0]),
        rtol=0.0,
        atol=1e-6,
    )
