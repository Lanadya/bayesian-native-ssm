# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""External-benchmark wrapper stubs.

Stage-1 will hook these up to the upstream evaluators. The skeleton
fixes the call signature so the eval-pipeline glue can be written
against a stable surface.

For the published-RLHF-baseline comparison the wrapper
``run_rlhf_baseline_comparison`` is signature-pinned to accept only
peer-reviewed or publicly-released RLHF models (e.g. Pythia-160M +
Tülu-3, TinyLlama-1.1B-Chat). Ad-hoc self-trained RLHF baselines are
deliberately rejected to forestall the "your RLHF baseline was tuned
badly" review objection. Capability-floor wrappers (MMLU-subset,
HumanEval) follow the same pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class BenchmarkResult:
    """Uniform return shape across benchmark wrappers."""

    name: str
    score: float
    per_subset: dict[str, float]
    notes: str = ""


class _ModelLike(Protocol):
    """Minimal protocol the evaluators expect from a model wrapper."""

    def generate(self, prompts: list[str], **kwargs: Any) -> list[str]: ...


def run_helm_safety(model: _ModelLike, subsets: list[str] | None = None) -> BenchmarkResult:
    """HELM-style holistic-evaluation wrapper — Stage-1 stub.

    Used as hygiene only, not as a decisive evaluation in the v5
    submission (Field 12).
    """
    raise NotImplementedError(
        "HELM-style wrapper is Stage-1 work. Pin a HELM commit hash and "
        "implement via the `helm` package's CLI runner."
    )


def run_bbq(model: _ModelLike, subsets: list[str] | None = None) -> BenchmarkResult:
    """BBQ (Bias-Benchmark-QA) evaluation wrapper — Stage-1 stub."""
    raise NotImplementedError(
        "BBQ wrapper is Stage-1 work. See https://github.com/nyu-mll/BBQ."
    )


def run_mmlu(model: _ModelLike, subsets: list[str] | None = None) -> BenchmarkResult:
    """MMLU-subset capability-floor wrapper — Stage-1 stub.

    In the v5 submission (Field 12), MMLU and HumanEval serve as 200M-
    class capability-floor checks, not as decisive evaluations.
    """
    raise NotImplementedError(
        "MMLU wrapper is Stage-1 work. Use the `lm-evaluation-harness` "
        "MMLU task at a pinned commit for reproducibility."
    )


def run_rlhf_baseline_comparison(
    bayesian_model: _ModelLike,
    rlhf_baseline_name: str,
) -> dict[str, BenchmarkResult]:
    """Side-by-side comparison against a *published* RLHF baseline.

    The ``rlhf_baseline_name`` argument MUST refer to a peer-reviewed or
    publicly-released model with documented RLHF training (e.g.
    ``"pythia-160m-tulu-3"``, ``"tinyllama-1.1b-chat-v1.0"``). The
    Stage-1 implementation rejects ad-hoc self-trained RLHF baselines
    here to forestall the "your RLHF baseline was tuned badly" review
    objection.
    """
    raise NotImplementedError(
        "RLHF baseline comparison is Stage-1 work. The Stage-1 baseline list "
        "is fixed to published RLHF models (Pythia-160M + Tülu-3, "
        "TinyLlama-1.1B-Chat) — no self-trained RLHF baselines."
    )
