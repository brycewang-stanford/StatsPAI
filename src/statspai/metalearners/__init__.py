"""
Meta-Learners for heterogeneous treatment effect estimation.

Provides S/T/X/R/DR-Learner implementations that decompose CATE
estimation into standard supervised-learning sub-problems. All learners
accept any scikit-learn compatible estimator.

References
----------
Kunzel et al. (2019). Metalearners for estimating heterogeneous treatment
effects using machine learning. PNAS, 116(10), 4156-4165. [@kunzel2019metalearners]

Nie & Wager (2021). Quasi-oracle estimation of heterogeneous treatment
effects. Biometrika, 108(2), 299-319. [@nie2021quasi]

Kennedy (2023). Towards optimal doubly robust estimation of heterogeneous
causal effects. Electronic Journal of Statistics, 17(2), 3008-3049. [@kennedy2023towards]
"""

from .auto_cate import AutoCATEResult, auto_cate
from .auto_cate_tuned import auto_cate_tuned
from .cate_curves import CATEGainCurveResult, cate_gain_curve

# v1.13 backbone-agnostic CATE evaluation (Yadlowsky 2025 RATE)
from .cate_eval import CATEEvalResult, cate_eval
from .cluster_cate import ClusterCATEResult, cluster_cate
from .diagnostics import (
    blp_test,
    cate_by_group,
    cate_group_plot,
    cate_plot,
    cate_summary,
    compare_metalearners,
    gate_test,
    predict_cate,
)

# v0.10 meta-learner frontier
from .focal import FunctionalCATEResult, focal_cate
from .metalearners import DRLearner, RLearner, SLearner, TLearner, XLearner, metalearner

__all__ = [
    "metalearner",
    "SLearner",
    "TLearner",
    "XLearner",
    "RLearner",
    "DRLearner",
    "cate_summary",
    "cate_by_group",
    "cate_plot",
    "cate_group_plot",
    "predict_cate",
    "compare_metalearners",
    "gate_test",
    "blp_test",
    "auto_cate",
    "AutoCATEResult",
    "auto_cate_tuned",
    "focal_cate",
    "FunctionalCATEResult",
    "cluster_cate",
    "ClusterCATEResult",
    "cate_eval",
    "CATEEvalResult",
    "cate_gain_curve",
    "CATEGainCurveResult",
]
