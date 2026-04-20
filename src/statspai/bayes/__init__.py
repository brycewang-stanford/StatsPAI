"""
Bayesian causal inference (``statspai.bayes``).

PyMC-backed Bayesian estimators for the canonical causal designs.
PyMC and ArviZ are **optional** dependencies — importing this
sub-package never imports them. Each estimator resolves PyMC at call
time and raises :class:`ImportError` with the install recipe if the
extras are missing.

Install with:

    pip install "statspai[bayes]"

Available estimators:

- :func:`bayes_did` — 2×2 and panel difference-in-differences with
  optional hierarchical random effects on unit / time.
- :func:`bayes_rd` — sharp regression discontinuity with local
  polynomial + Normal prior on the jump.
"""
from __future__ import annotations

from ._base import (
    BayesianCausalResult,
    BayesianHTEIVResult,
    BayesianMTEResult,
)
from .did import bayes_did
from .rd import bayes_rd
from .iv import bayes_iv
from .fuzzy_rd import bayes_fuzzy_rd
from .hte_iv import bayes_hte_iv
from .mte import bayes_mte

__all__ = [
    'bayes_did',
    'bayes_rd',
    'bayes_iv',
    'bayes_fuzzy_rd',
    'bayes_hte_iv',
    'bayes_mte',
    'BayesianCausalResult',
    'BayesianHTEIVResult',
    'BayesianMTEResult',
]
