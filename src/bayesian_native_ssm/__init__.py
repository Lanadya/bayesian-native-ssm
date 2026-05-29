# Copyright 2026 Posterior Labs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Bayesian-Native State-Space Model — reference skeleton.

Public API surface is intentionally narrow at this stage. The package
exposes the loss primitives that LAB_THEORY pins down (Trust-Posterior
decomposition, Bühlmann-Straub credibility weighting, three likelihood
classes) and an honest stub layer for the Mamba backbone integration
that is Stage-1 implementation work.
"""

from __future__ import annotations

__version__ = "0.0.1"

from bayesian_native_ssm.losses.buhlmann import (
    buhlmann_weighted_loss,
    buhlmann_z,
)
from bayesian_native_ssm.losses.likelihoods import (
    gauss_likelihood,
    huber_likelihood,
    wasserstein_likelihood,
)
from bayesian_native_ssm.losses.trust_posterior import (
    trust_posterior_loss,
)

__all__ = [
    "__version__",
    # losses
    "trust_posterior_loss",
    "buhlmann_z",
    "buhlmann_weighted_loss",
    "gauss_likelihood",
    "wasserstein_likelihood",
    "huber_likelihood",
]
