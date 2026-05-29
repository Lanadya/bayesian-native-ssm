# Copyright 2026 Posterior Labs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Augmented trust state h̃_t = (h_t, τ_t).

The full integration with the Mamba selective-scan kernel is Stage-1
work; this module captures the shape contract and a small reference
update step so callers can prototype against the interface without a
GPU.

See ``docs/ARCHITECTURE.md`` for the selective-gating-on-τ_t design
question (whether ``A(x_t, τ_t)``, ``B(x_t, τ_t)`` should be τ-conditional).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TrustStateConfig:
    """Configuration for the trust-state slot in the augmented hidden state.

    Attributes:
        hidden_dim: Size ``d`` of the base Mamba hidden state ``h_t``.
        trust_dim: Size ``d_τ`` of the trust-state slot ``τ_t``. Stage-1
            research question: optimal value. Sensible starting range
            ``d_τ ∈ [8, 64]`` for sub-1B-parameter models.
        decay: Per-step exponential decay applied to ``τ_t`` before the
            additive update. ``decay = 1.0`` means no forgetting.
    """

    hidden_dim: int
    trust_dim: int
    decay: float = 0.95


class AugmentedTrustState:
    """Reference-only carrier for h̃_t = (h_t, τ_t).

    This is *not* an ``nn.Module``. It is a small holder that documents
    the shape contract and provides a CPU-side exponential update used in
    unit tests and notebooks. The production update lives inside the
    selective-scan kernel of the Mamba wrapper and is Stage-1 work.
    """

    def __init__(self, config: TrustStateConfig):
        self.config = config

    def init_state(self, batch_size: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        """Allocate zero-initialised ``(h_t, τ_t)`` for a batch."""
        h = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        tau = torch.zeros(batch_size, self.config.trust_dim, device=device)
        return h, tau

    def update_tau(self, tau_prev: Tensor, trust_update: Tensor) -> Tensor:
        """Reference exponential-decay update step for ``τ_t``.

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
