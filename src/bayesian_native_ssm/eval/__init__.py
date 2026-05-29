# Copyright 2026 Posterior Labs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Evaluation-side stubs.

Two thin modules:

- ``benchmarks``        — HELM-Safety / BBQ / MMLU wrapper stubs.
- ``trust_diagnostics`` — calibration ECE and worst-group accuracy.
                          ``ece`` and ``worst_group_accuracy`` are real
                          implementations because the loss tests rely on
                          them being callable end-to-end; benchmark
                          wrappers are honest stubs.
"""
