"""
Bayesian econometrics by MCMC (``statspai.mcmc``).

NumPy-only samplers for the standard regression models, convergence
diagnostics for any chain, and marginal likelihoods. The PyMC-backed
estimators for causal designs live in :mod:`statspai.bayes`.
"""

from __future__ import annotations

from .bma import BMAResult, bma
from .bootstrap import bayes_bootstrap
from .compare import BayesFactorResult, bayes_factor, savage_dickey
from .diagnostics import (
    MCMCDiagnostic,
    gelman_rubin,
    geweke_diag,
    heidel_diag,
    hpd_interval,
    mcmc_ess,
    mcmc_summary,
    raftery_diag,
)
from .iv import bayes_ivreg
from .mixed import BayesMixedResult, bayes_mixed
from .regress import BayesRegressResult, bayes_regress
from .shrinkage import bayes_shrink
from .sur import bayes_sur

__all__ = [
    "MCMCDiagnostic",
    "BayesRegressResult",
    "BayesFactorResult",
    "BayesMixedResult",
    "BMAResult",
    "bayes_bootstrap",
    "bayes_factor",
    "bayes_ivreg",
    "bayes_mixed",
    "bayes_regress",
    "bayes_shrink",
    "bayes_sur",
    "bma",
    "savage_dickey",
    "gelman_rubin",
    "geweke_diag",
    "heidel_diag",
    "hpd_interval",
    "mcmc_ess",
    "mcmc_summary",
    "raftery_diag",
]
