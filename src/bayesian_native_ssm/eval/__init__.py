# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Evaluation-side stubs.

Two thin modules:

- ``benchmarks``        — capability-floor wrapper stubs (MMLU,
                           HumanEval) plus the published-RLHF-baseline
                           comparison harness.
- ``trust_diagnostics`` — calibration ECE and worst-group accuracy.
                          ``expected_calibration_error`` and
                          ``worst_group_accuracy`` are real implementations
                          because the loss tests rely on them being
                          callable end-to-end; benchmark wrappers are
                          honest stubs.

Stage-1 expands these with the custom Credence-State eval suite
described in the SPRIND submission (Field 12 primary metrics: proxy-gold
gap, Credence calibration, risk-adjusted action quality, causal state
use).
"""
