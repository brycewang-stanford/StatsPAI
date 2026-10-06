"""
Forest-based causal inference estimators for StatsPAI.

Hosts ``CausalForest`` (grf-style honest causal forests) and its
companions:

- :func:`causal_forest` / :class:`CausalForest` — heterogeneous
  treatment-effect estimation via honest random forests
  (Wager-Athey 2018; Athey-Tibshirani-Wager 2019).
- :func:`iv_forest` / :func:`instrumental_forest` — instrumental forests
  for conditional LATEs (Athey-Tibshirani-Wager 2019).
- :func:`multi_arm_forest` / :func:`lm_forest` — multi-arm causal forests
  and conditionally linear models (Nie-Wager 2021 with GRF weights).
- :func:`regression_forest`, :func:`multi_regression_forest`,
  :func:`probability_forest`, :func:`quantile_forest`,
  :func:`survival_forest` — the prediction forests of the GRF family.
- :func:`variable_importance`, :func:`best_linear_projection`,
  :func:`get_scores` — grf's post-estimation for any of them.
- :func:`forest_group_effects` / :func:`forest_support` /
  :func:`cate_pretrend_test` -- group effects with valid SEs (imputation
  scores for ``fe=`` forests), support of counterfactual predictions, and
  pre-trends by predicted-effect group.
- :func:`calibration_test` / :func:`test_calibration` /
  :func:`rate` / :func:`honest_variance` — post-fit honesty &
  calibration diagnostics.

This package was previously named ``statspai.causal`` — the old
name is kept as a deprecation shim for one minor version cycle.
Use ``from statspai.forest import ...`` going forward.
"""

from .causal_forest import CausalForest, causal_forest
from .forest_heterogeneity import (
    cate_pretrend_test,
    forest_group_effects,
    forest_policy_tree,
    forest_support,
    rate_split,
)
from .forest_inference import (
    average_treatment_effect,
    calibrate_cate,
    calibration_test,
    forest_diagnostics,
    honest_variance,
    rate,
    test_calibration,
)
from .forest_tools import best_linear_projection, get_scores, variable_importance
from .iv_forest import IVForestResult, instrumental_forest, iv_forest
from .lm_forest import LMForestResult, lm_forest
from .multi_arm_forest import MultiArmForestResult, multi_arm_forest
from .regression_forests import (
    PredictionForest,
    multi_regression_forest,
    probability_forest,
    quantile_forest,
    regression_forest,
)
from .survival_forest import SurvivalForestResult, survival_forest
from .tuning import tune_causal_forest

__all__ = [
    "tune_causal_forest",
    "CausalForest",
    "causal_forest",
    "calibrate_cate",
    "calibration_test",
    "test_calibration",
    "rate",
    "honest_variance",
    "average_treatment_effect",
    "forest_diagnostics",
    "forest_group_effects",
    "forest_support",
    "cate_pretrend_test",
    "rate_split",
    "forest_policy_tree",
    "multi_arm_forest",
    "MultiArmForestResult",
    "iv_forest",
    "instrumental_forest",
    "IVForestResult",
    "lm_forest",
    "LMForestResult",
    "regression_forest",
    "multi_regression_forest",
    "probability_forest",
    "quantile_forest",
    "PredictionForest",
    "survival_forest",
    "SurvivalForestResult",
    "variable_importance",
    "best_linear_projection",
    "get_scores",
]
