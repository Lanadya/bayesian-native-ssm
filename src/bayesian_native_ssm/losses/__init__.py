# Copyright 2026 Posterior Labs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Loss primitives for Bayesian-Native training.

Three concerns, three modules:

- ``trust_posterior``  — assembles L_total = L_data + L_trust + L_prior
- ``buhlmann``         — Bühlmann-Straub credibility weighting (exposure form)
- ``likelihoods``      — three concrete classes for P(T_j | s_j(θ)):
                          Gauss, Wasserstein-2, Huber

The likelihood classes exist precisely to answer LAB_REDTECH Killshot 1:
"Trust-Likelihood is nowhere defined". Picking a class is now an explicit
caller decision, not a hand-wave.
"""
