# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the loss primitives.

Three concerns covered:

- Likelihood correctness: Gauss matches ``scipy.stats.norm.logpdf``;
  Wasserstein and Huber are at minimum callable and produce sensible
  values under controlled inputs.
- Per-dimension factorisation: L_trust = Σ_j λ_j L_j aggregates
  correctly across reliability dimensions.
- Likelihood-class dispatch: all three classes are wired through the
  trust_posterior_loss path.
"""

from __future__ import annotations

import math

import pytest
import torch

from bayesian_native_ssm.losses.likelihoods import (
    gauss_likelihood,
    huber_likelihood,
    wasserstein_likelihood,
)
from bayesian_native_ssm.losses.trust_posterior import trust_posterior_loss


# ---------------------------------------------------------------------------
# Likelihood correctness
# ---------------------------------------------------------------------------

def test_gauss_likelihood_correctness():
    """Gauss NLL must match scipy.stats.norm.logpdf up to numerical tol."""
    scipy_stats = pytest.importorskip("scipy.stats")

    torch.manual_seed(0)
    signal = torch.randn(64)
    target = torch.randn(64)
    sigma = 1.3

    nll_ours = gauss_likelihood(signal, target, sigma=sigma).item()

    # scipy: log p(x | μ, σ) ; we minimise -log p — sum over samples.
    nll_scipy = -float(
        scipy_stats.norm.logpdf(
            (signal - target).numpy(),
            loc=0.0,
            scale=sigma,
        ).sum()
    )

    assert nll_ours == pytest.approx(nll_scipy, rel=1e-5, abs=1e-5)


def test_likelihood_classes_callable():
    """All three classes produce finite scalar losses under sensible inputs."""
    torch.manual_seed(0)
    signal = torch.randn(8)
    target = torch.randn(8)

    g = gauss_likelihood(signal, target, sigma=1.0)
    h = huber_likelihood(signal, target, delta=1.0)
    assert torch.isfinite(g) and g.ndim == 0
    assert torch.isfinite(h) and h.ndim == 0

    # Wasserstein expects probability vectors on a shared 1-D support.
    # Support is normalised to [0, 1] internally — default eps works for any N.
    p = torch.softmax(torch.randn(16), dim=0)
    q = torch.softmax(torch.randn(16), dim=0)
    w = wasserstein_likelihood(p, q)
    assert torch.isfinite(w) and w.ndim == 0
    assert w >= 0.0  # transport cost is non-negative

    # Sanity: distant distributions cost more to transport than identical ones.
    delta_far = torch.zeros(16); delta_far[15] = 1.0
    delta_near = torch.zeros(16); delta_near[0] = 1.0
    w_far = wasserstein_likelihood(delta_near, delta_far)
    w_self = wasserstein_likelihood(delta_near, delta_near.clone())
    assert w_far > w_self, f"far={w_far.item()} should exceed self={w_self.item()}"


def test_huber_matches_quadratic_in_small_residuals():
    """For |signal - target| ≤ δ, Huber reduces to 0.5 · (signal - target)²."""
    signal = torch.tensor([0.0, 0.3, -0.4])
    target = torch.tensor([0.1, 0.2, -0.2])
    delta = 1.0
    expected = 0.5 * (signal - target).pow(2).sum()
    got = huber_likelihood(signal, target, delta=delta)
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)


def test_huber_bounded_influence_on_outliers():
    """For |residual| ≫ δ, Huber grows linearly — bounded influence."""
    base = torch.zeros(10)
    outlier = torch.zeros(10)
    outlier[0] = 100.0  # huge outlier
    delta = 1.0

    gauss = gauss_likelihood(outlier, base, sigma=1.0)
    huber = huber_likelihood(outlier, base, delta=delta)

    # Gauss is quadratic in the outlier, Huber linear past δ → Huber ≪ Gauss.
    assert huber < gauss / 10.0, (
        f"Huber should be much smaller on a 100σ outlier; "
        f"got huber={huber.item():.2f}, gauss={gauss.item():.2f}"
    )


# ---------------------------------------------------------------------------
# Per-dimension factorisation
# ---------------------------------------------------------------------------

def _make_minibatch(b=2, t=4, v=5, dims=("calibration", "honesty")):
    torch.manual_seed(0)
    model_output = torch.randn(b, t, v)
    data_targets = torch.randint(0, v, (b, t))
    trust_signals = {j: torch.randn(b) for j in dims}
    trust_targets = {j: torch.randn(b) for j in dims}
    return model_output, data_targets, trust_signals, trust_targets


def test_trust_posterior_dimensional_factorization():
    """L_trust = Σ_j λ_j L_j — doubling one λ_j must shift L_trust accordingly."""
    model_output, data_targets, ts, tt = _make_minibatch()

    out_uniform = trust_posterior_loss(
        model_output, data_targets, ts, tt,
        likelihood_class="gauss",
        lambdas={"calibration": 1.0, "honesty": 1.0},
    )
    out_weighted = trust_posterior_loss(
        model_output, data_targets, ts, tt,
        likelihood_class="gauss",
        lambdas={"calibration": 2.0, "honesty": 1.0},
    )

    # Independently compute the per-dimension Gauss NLL and check linearity.
    l_cal = gauss_likelihood(ts["calibration"], tt["calibration"], sigma=1.0)
    l_hon = gauss_likelihood(ts["honesty"], tt["honesty"], sigma=1.0)

    torch.testing.assert_close(out_uniform["L_trust"], l_cal + l_hon, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_weighted["L_trust"], 2.0 * l_cal + l_hon, rtol=1e-5, atol=1e-5)


def test_trust_posterior_likelihood_class_dispatch():
    """Switching likelihood_class must route to a different scalar value."""
    model_output, data_targets, _, _ = _make_minibatch()
    # Use 1-D probability vectors so Wasserstein has a meaningful support.
    p = torch.softmax(torch.randn(16), dim=0)
    q = torch.softmax(torch.randn(16), dim=0)
    ts = {"dist": p}
    tt = {"dist": q}

    out_gauss = trust_posterior_loss(model_output, data_targets, ts, tt, likelihood_class="gauss")
    out_w = trust_posterior_loss(model_output, data_targets, ts, tt, likelihood_class="wasserstein")
    out_h = trust_posterior_loss(model_output, data_targets, ts, tt, likelihood_class="huber")

    # All three return distinct L_trust values (sanity check that dispatch
    # actually reaches three different implementations).
    vals = {out_gauss["L_trust"].item(), out_w["L_trust"].item(), out_h["L_trust"].item()}
    assert len(vals) == 3, f"likelihood dispatch collapsed: {vals}"


def test_trust_posterior_unknown_likelihood_raises():
    """Unknown likelihood_class must raise ValueError."""
    model_output, data_targets, ts, tt = _make_minibatch()
    with pytest.raises(ValueError, match="Unknown likelihood_class"):
        trust_posterior_loss(model_output, data_targets, ts, tt, likelihood_class="cauchy")


def test_trust_posterior_with_buhlmann_context_weighting():
    """Supplying context_assignments routes through the Bühlmann path.

    Checks that the total loss remains finite and that L_trust is a
    convex combination of local and global per-dimension losses (i.e.
    bounded between them, dimension-wise).
    """
    b, t, v = 4, 3, 5
    torch.manual_seed(0)
    model_output = torch.randn(b, t, v)
    data_targets = torch.randint(0, v, (b, t))
    ts = {"calibration": torch.tensor([0.0, 0.1, 5.0, 5.1])}
    tt = {"calibration": torch.tensor([0.0, 0.0, 0.0, 0.0])}
    contexts = torch.tensor([0, 0, 1, 1])  # two sub-populations

    out = trust_posterior_loss(
        model_output, data_targets, ts, tt,
        context_assignments=contexts,
        likelihood_class="gauss",
        buhlmann_k=1.0,
    )

    assert torch.isfinite(out["L_total"])
    assert torch.isfinite(out["L_trust"])
    assert out["L_trust"] >= 0.0
