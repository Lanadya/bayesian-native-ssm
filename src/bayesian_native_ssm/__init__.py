# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Credence-State Foundation Models — reference skeleton.

Public API surface is intentionally narrow at this stage. The package
exposes the loss-side primitives that the architectural proposal pins
down (Credence-State loss assembly, Bühlmann-Straub credibility weighting
in exposure form, three concrete likelihood classes) and an honest stub
layer for the Mamba backbone integration that is Stage-1 implementation
work.

The companion SPRIND submission "Credence-State Foundation Models"
(May 2026) describes the five-component loss decomposition

    L = L_LM + λ_1 L_credence + λ_2 L_update + λ_3 L_action + λ_4 L_calibration

that Stage-1 builds out. This skeleton implements an earlier
three-component pre-Stage-1 form (L_data + L_trust + L_prior); the
naming map in the repository README documents the migration. Code
symbol names retain the legacy 'trust' / 'tau' / 'AEGIS-dimension'
labels to keep the existing test suite intact.
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
