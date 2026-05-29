# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Augmented hidden state h̃_t = (h_t, τ_t).

In the Credence-State Foundation Models architecture (SPRIND submission
May 2026) the recurrent state is augmented from the base backbone
hidden state ``h_t ∈ ℝ^d`` by a Credence-State slot ``c_t ∈ ℝ^{d_c}``
carrying posterior parameters over the reliability variables
``z_t = (u_t, r_t, s_t, k_t)`` — epistemic uncertainty, source/context
reliability, evidential sufficiency, conflict.

Code symbol mapping (legacy → v5 submission terminology):

- ``τ_t``                ↔  ``c_t``  (the Credence-State slot)
- ``trust_dim``          ↔  ``d_c`` (size of the Credence-State slot)
- ``trust_update``       ↔  the per-step update to ``c_t`` from new
                              evidence observations

Code names are not renamed to keep the existing test suite intact. The
v5 submission's architectural specification (Field 4) and the migration
plan are documented in the repository README.

The full integration with the Mamba selective-scan kernel is Stage-1
work; this module captures the shape contract and a small reference
update step so callers can prototype against the interface without a
GPU. See ``docs/ARCHITECTURE.md`` for the selective-gating design
question (whether the selective parameters should be conditioned on
``(x_t, c_t)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TrustStateConfig:
    """Configuration for the Credence-State slot in the augmented hidden state.

    Code name retained as ``TrustStateConfig`` for test-suite stability;
    in v5 terminology this is the Credence-State slot configuration.

    Attributes:
        hidden_dim: Size ``d`` of the base Mamba hidden state ``h_t``.
        trust_dim: Size ``d_c`` (legacy: ``d_τ``) of the Credence-State
            slot ``c_t``. Stage-1 research question: optimal value.
            Sensible starting range ``d_c ∈ [8, 64]`` for sub-1B-parameter
            models.
        decay: Per-step exponential decay applied to ``c_t`` before the
            additive update. ``decay = 1.0`` means no forgetting.
    """

    hidden_dim: int
    trust_dim: int
    decay: float = 0.95


class AugmentedTrustState:
    """Reference-only carrier for h̃_t = (h_t, τ_t).

    Code name retained as ``AugmentedTrustState`` for test-suite
    stability; in v5 terminology this carries the augmented hidden state
    ``(h_t, c_t)`` with the Credence-State slot.

    This is *not* an ``nn.Module``. It is a small holder that documents
    the shape contract and provides a CPU-side exponential update used in
    unit tests and notebooks. The production update lives inside the
    selective-scan kernel of the Mamba wrapper and is Stage-1 work.
    """

    def __init__(self, config: TrustStateConfig):
        self.config = config

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        """Allocate zero-initialised ``(h_t, τ_t)`` for a batch.

        In v5 terminology: zero-initialised ``(h_t, c_t)``.
        """
        h = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        tau = torch.zeros(batch_size, self.config.trust_dim, device=device)
        return h, tau

    def update_tau(self, tau_prev: Tensor, trust_update: Tensor) -> Tensor:
        """Reference exponential-decay update step for ``τ_t`` (v5: ``c_t``).

        Computes ``τ_t = decay * τ_{t-1} + trust_update``. The
        Stage-1 production form may replace this with a selective gate
        that depends on the input token; this reference form is the
        simplest stable starting point and is what the unit tests use.
        """
        if tau_prev.shape != trust_update.shape:
            raise ValueError(
                f"tau_prev and trust_update must agree in shape; "
                f"got {tuple(tau_prev.shape)} vs {tuple(trust_update.shape)}."
            )
        return self.config.decay * tau_prev + trust_update
