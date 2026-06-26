"""Rigorous (data-driven) Lasso — a faithful port of R's ``hdm`` package.

Public surface
--------------
- :func:`rlasso` — rigorous (post-)Lasso with a data-driven, theory-
  justified penalty (``hdm::rlasso``).
- :func:`rlasso_effect` / :func:`rlasso_effects` — treatment-effect
  inference after Lasso-selecting controls (``hdm::rlassoEffect(s)``).
- :func:`rlasso_iv` — instrumental-variables estimation with Lasso
  selection of instruments and/or controls (``hdm::rlassoIV``).
- :class:`RlassoRegressor` / :class:`RlassoClassifier` — scikit-learn
  adapters so the rigorous Lasso can serve as a Double-ML nuisance
  learner (``sp.dml(..., ml_g='rlasso')``).

Every estimator is validated to agree numerically with ``hdm`` (see
``tests/reference_parity/test_rlasso_parity.py``): the core ``rlasso``
matches to machine precision; the IV/effect estimators to ~1e-6.

References
----------
Chernozhukov, V., Hansen, C. and Spindler, M. (2016). "hdm:
    High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
    [@chernozhukov2016hdm]
"""

from __future__ import annotations

from ._core import RLassoFit, rlasso
from ._logit import RLassoLogitFit, rlassologit
from .effect import RLassoEffectResult, rlasso_effect, rlasso_effects
from .iv import RLassoIVResult, rlasso_iv
from .learner import RlassoClassifier, RlassologitClassifier, RlassoRegressor
from .logit_effect import (
    RLassoLogitEffectResult,
    rlassologit_effect,
    rlassologit_effects,
)

__all__ = [
    "rlasso",
    "RLassoFit",
    "rlasso_effect",
    "rlasso_effects",
    "RLassoEffectResult",
    "rlasso_iv",
    "RLassoIVResult",
    "rlassologit",
    "rlassologit_effect",
    "rlassologit_effects",
    "RLassoLogitEffectResult",
    "RLassoLogitFit",
    "RlassologitClassifier",
    "RlassoRegressor",
    "RlassoClassifier",
]
