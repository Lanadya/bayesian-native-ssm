# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Wrapper around the ``mamba-ssm`` selective-scan backbone.

This is an honest stub. The body of ``forward`` raises
``NotImplementedError`` because the integration with the upstream
``mamba-ssm`` package (state-spaces/mamba) is Stage-1 implementation
work. What this file *does* fix is the public interface — anyone
building against the package can rely on the signatures below not
shifting under their feet between v0.0.x releases of the skeleton.

See ``docs/ARCHITECTURE.md`` for the integration plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor, nn

from bayesian_native_ssm.models.trust_state import TrustStateConfig


@dataclass
class MambaBackboneConfig:
    """Stub configuration for the upstream Mamba backbone.

    The concrete fields will mirror ``mamba_ssm.models.config_mamba.MambaConfig``
    once the integration is done. Kept loose here to avoid pinning the
    skeleton to a specific upstream API revision.
    """

    d_model: int = 768
    n_layer: int = 24
    vocab_size: int = 50_257
    ssm_cfg: dict[str, Any] = field(default_factory=dict)
    variant: str = "mamba2"  # "mamba" | "mamba2" — Stage-1 primary: mamba2


class MambaTrustWrapper(nn.Module):
    """Wrap a Mamba backbone and propagate h̃_t = (h_t, τ_t).

    Code name retained as ``MambaTrustWrapper`` for interface stability;
    in v5 terminology this propagates ``(h_t, c_t)`` with the
    Credence-State slot ``c_t``.

    The wrapper exposes two extension points that the upstream Mamba does
    not natively have:

    1. A Credence-State slot of dimension ``d_c`` (code: ``d_τ``) that
       travels alongside ``h_t`` through the selective scan (see
       :class:`AugmentedTrustState`).
    2. A reliability-signals head per reliability dimension (code:
       ``trust_signals`` / ``trust_dimension_names``) that reads from
       the last layer's hidden state and produces the per-dimension
       signal consumed by ``trust_posterior_loss``.

    Stage-1 implementation tasks:

    - Pick the modification point in the selective-scan kernel
      (input-dependent selective parameters conditioned on ``(x_t, c_t)``)
      — open research question, see ``ARCHITECTURE.md``.
    - Decide whether ``c_t`` is updated inside the scan or by a
      post-layer hook — depends on whether mamba-ssm's scan kernel can
      be extended without forking the kernel.
    - Write the corresponding CUDA / Triton paths if the answer is
      "inside the scan".
    """

    def __init__(
        self,
        backbone_config: MambaBackboneConfig,
        trust_state_config: TrustStateConfig,
        trust_dimension_names: list[str],
    ):
        super().__init__()
        self.backbone_config = backbone_config
        self.trust_state_config = trust_state_config
        self.trust_dimension_names = list(trust_dimension_names)

        # Sanity-check the dimensions match. The upstream Mamba enforces
        # d_model itself; here we just guard the Credence-State slot.
        if trust_state_config.hidden_dim != backbone_config.d_model:
            raise ValueError(
                f"TrustStateConfig.hidden_dim ({trust_state_config.hidden_dim}) "
                f"must match MambaBackboneConfig.d_model ({backbone_config.d_model})."
            )

        # No real submodules instantiated in the skeleton — those land in
        # Stage-1. A no-op parameter keeps the module non-empty so that
        # ``.parameters()`` and ``.to(device)`` behave normally for code
        # that introspects the wrapper before Stage-1 implementation lands.
        self._placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        input_ids: Tensor,
        trust_signals: Optional[dict[str, Tensor]] = None,
    ) -> dict[str, Tensor]:
        """Forward pass — Stage-1 stub.

        Args:
            input_ids: Long tensor of shape ``(B, T)``.
            trust_signals: Optional external reliability signals injected
                at step ``t`` (e.g. from an oracle during teacher-forcing
                of reliability-event trajectories).

        Returns:
            Stage-1 contract: dict with keys ``"logits"`` of shape
            ``(B, T, V)``, ``"trust_signals"`` (mapping reliability-
            dimension name to ``(B, T)`` tensor of signal values), and
            ``"tau_trajectory"`` of shape ``(B, T, d_c)`` (the
            Credence-State trajectory).

        Raises:
            NotImplementedError: always. Integration with ``mamba-ssm`` is
                Stage-1 implementation work.
        """
        raise NotImplementedError(
            "MambaTrustWrapper.forward is Stage-1 implementation work: "
            "integration with the mamba-ssm selective-scan kernel is "
            "pending. See docs/ARCHITECTURE.md for the design and the "
            "open research question on selective-parameter conditioning "
            "on the Credence-State slot."
        )
