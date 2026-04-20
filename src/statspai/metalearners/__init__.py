"""
Meta-Learners for heterogeneous treatment effect estimation.

Provides S/T/X/R/DR-Learner implementations that decompose CATE
estimation into standard supervised-learning sub-problems. All learners
accept any scikit-learn compatible estimator.

References
----------
Kunzel et al. (2019). Metalearners for estimating heterogeneous treatment
effects using machine learning. PNAS, 116(10), 4156-4165.

Nie & Wager (2021). Quasi-oracle estimation of heterogeneous treatment
effects. Biometrika, 108(2), 299-319.

Kennedy (2023). Towards optimal doubly robust estimation of heterogeneous
causal effects. Electronic Journal of Statistics, 17(2), 3008-3049.
"""

from .metalearners import (
    metalearner,
    SLearner,
    TLearner,
    XLearner,
    RLearner,
    DRLearner,
)
from .diagnostics import (
    cate_summary,
    cate_by_group,
    cate_plot,
    cate_group_plot,
    predict_cate,
    compare_metalearners,
    gate_test,
    blp_test,
)
from .auto_cate import auto_cate, AutoCATEResult
from .auto_cate_tuned import auto_cate_tuned

__all__ = [
    'metalearner',
    'SLearner',
    'TLearner',
    'XLearner',
    'RLearner',
    'DRLearner',
    'cate_summary',
    'cate_by_group',
    'cate_plot',
    'cate_group_plot',
    'predict_cate',
    'compare_metalearners',
    'gate_test',
    'blp_test',
    'auto_cate',
    'AutoCATEResult',
    'auto_cate_tuned',
]
