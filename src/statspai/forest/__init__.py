"""
Forest-based causal inference estimators for StatsPAI.

Hosts ``CausalForest`` (grf-style honest causal forests) and its
companions:

- :func:`causal_forest` / :class:`CausalForest` — heterogeneous
  treatment-effect estimation via honest random forests
  (Wager-Athey 2018; Athey-Tibshirani-Wager 2019).
- :func:`iv_forest` — instrumental-variable causal forests
  (Athey-Tibshirani-Wager 2019).
- :func:`multi_arm_forest` — multi-arm extension.
- :func:`calibration_test` / :func:`test_calibration` /
  :func:`rate` / :func:`honest_variance` — post-fit honesty &
  calibration diagnostics.

This package was previously named ``statspai.causal`` — the old
name is kept as a deprecation shim for one minor version cycle.
Use ``from statspai.forest import ...`` going forward.
"""

from .causal_forest import CausalForest, causal_forest
from .forest_inference import (
    calibration_test, test_calibration, rate, honest_variance,
)
from .multi_arm_forest import multi_arm_forest, MultiArmForestResult
from .iv_forest import iv_forest, IVForestResult

__all__ = [
    "CausalForest",
    "causal_forest",
    "calibration_test",
    "test_calibration",
    "rate",
    "honest_variance",
    "multi_arm_forest", "MultiArmForestResult",
    "iv_forest", "IVForestResult",
]
