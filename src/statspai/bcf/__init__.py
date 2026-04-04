"""
Bayesian Causal Forest (BCF) for heterogeneous treatment effects.

Decomposes the outcome into a prognostic function mu(X) and a
treatment effect function tau(X), each modeled by BART:

    Y_i = mu(X_i) + tau(X_i) * D_i + epsilon_i

This separation allows regularization-induced confounding (RIC) to be
mitigated, producing better CATE estimates than standard BART.

References
----------
Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020).
Bayesian Regression Tree Models for Causal Inference: Regularization,
Confounding, and Heterogeneous Effects.
Bayesian Analysis, 15(3), 965-1056.
"""

from .bcf import bcf, BayesianCausalForest

__all__ = ['bcf', 'BayesianCausalForest']
