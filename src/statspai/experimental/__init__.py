"""
Experimental design and analysis tools.

Provides randomization, balance checking, attrition analysis,
pre-analysis plan generation, group sequential designs, always-valid
(mixture SPRT) inference and CUPED variance reduction for RCTs and A/B tests.
"""

from .adaptive import (
    AdaptiveInferenceResult,
    BanditExperimentResult,
    adaptive_inference,
    bandit_allocate,
    bandit_experiment,
    contextual_bandit,
)
from .attrition import AttritionResult, attrition_bounds, attrition_test
from .cuped import cuped
from .design import BalanceResult, RandomizationResult, balance_check, randomize
from .group_sequential import GroupSequentialResult, group_sequential
from .msprt import MSPRTResult, msprt, msprt_lambda
from .optimal import OptimalDesignResult, optimal_design
from .switchback import switchback, switchback_design

__all__ = [
    "randomize",
    "RandomizationResult",
    "balance_check",
    "BalanceResult",
    "attrition_test",
    "attrition_bounds",
    "AttritionResult",
    "optimal_design",
    "OptimalDesignResult",
    "switchback",
    "switchback_design",
    "bandit_allocate",
    "bandit_experiment",
    "contextual_bandit",
    "adaptive_inference",
    "BanditExperimentResult",
    "AdaptiveInferenceResult",
    "group_sequential",
    "GroupSequentialResult",
    "msprt",
    "msprt_lambda",
    "MSPRTResult",
    "cuped",
]
