# Copyright 2026 Posterior Labs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Pydantic schema for the YAML training configuration.

Single Pydantic file, no abstract-class hierarchies — Anker 4 in the
LAB_REPO brief ("Lesbar in <30 Minuten"). The schema captures *what* a
Stage-1 trainer must know, not *how* it must be built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

LikelihoodClass = Literal["gauss", "wasserstein", "huber"]


class LikelihoodSpec(BaseModel):
    """Per-AEGIS-dimension likelihood choice and its hyperparameters."""

    name: str = Field(..., description="AEGIS dimension name, e.g. 'calibration'.")
    likelihood_class: LikelihoodClass = "gauss"
    lambda_weight: float = Field(1.0, gt=0.0, description="Dimension weight λ_j.")
    sigma: float | None = Field(None, gt=0.0, description="Gauss σ; required if class='gauss'.")
    epsilon: float | None = Field(None, gt=0.0, description="Sinkhorn ε; class='wasserstein'.")
    delta: float | None = Field(None, gt=0.0, description="Huber δ; class='huber'.")


class BuhlmannSpec(BaseModel):
    """Bühlmann-Straub credibility weighting (exposure form)."""

    enabled: bool = False
    k: float = Field(1.0, gt=0.0, description="k = σ²/τ²; exposure-form crossover.")
    context_field: str | None = Field(
        None,
        description="Batch field name that carries the sub-population context id.",
    )


class BackboneSpec(BaseModel):
    """Mamba-class backbone configuration (Stage-1 integration target)."""

    variant: Literal["mamba", "mamba2", "rwkv"] = "mamba2"
    d_model: int = Field(768, gt=0)
    n_layer: int = Field(24, gt=0)
    vocab_size: int = Field(50_257, gt=0)
    trust_dim: int = Field(16, gt=0, description="Size d_τ of the trust slot.")


class TrainingSpec(BaseModel):
    """Optimiser, schedule, and run-level controls."""

    optimizer: Literal["adamw", "lion"] = "adamw"
    lr: float = Field(3e-4, gt=0.0)
    weight_decay: float = Field(0.1, ge=0.0)
    warmup_steps: int = Field(2_000, ge=0)
    max_steps: int = Field(50_000, ge=1)
    batch_size: int = Field(64, gt=0)
    grad_accum_steps: int = Field(1, gt=0)
    seed: int = 0


class ExperimentConfig(BaseModel):
    """Top-level YAML schema for a Stage-1 run."""

    experiment_name: str
    backbone: BackboneSpec
    training: TrainingSpec
    likelihoods: list[LikelihoodSpec]
    buhlmann: BuhlmannSpec = BuhlmannSpec()

    @field_validator("likelihoods")
    @classmethod
    def _unique_dimension_names(cls, v: list[LikelihoodSpec]) -> list[LikelihoodSpec]:
        names = [spec.name for spec in v]
        if len(names) != len(set(names)):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate AEGIS dimension names: {sorted(duplicates)}.")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load and validate an ``ExperimentConfig`` from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
