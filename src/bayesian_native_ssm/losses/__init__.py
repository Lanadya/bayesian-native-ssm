# Copyright 2026 ARQON GmbH (in formation)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Loss primitives for the Credence-State training pipeline.

Three concerns, three modules:

- ``trust_posterior``  — assembles a three-component pre-Stage-1 loss
                          L_total = L_data + L_trust + L_prior, with
                          ``L_data`` mapping to ``L_LM`` and ``L_trust``
                          splitting into ``L_credence`` + ``L_update`` in
                          the five-component v5 decomposition.
- ``buhlmann``         — Bühlmann-Straub credibility weighting in
                          exposure form (Straub 1970), the update prior
                          carried into ``L_update`` in the v5 form.
- ``likelihoods``      — three concrete classes for the per-dimension
                          observation model: Gauss, Wasserstein-2, Huber.

The likelihood-class choice is an explicit caller decision (no silent
default that collapses the structural difference to MSE-RLHF).
"""
