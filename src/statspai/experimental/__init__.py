"""
Experimental design and analysis tools.

Provides randomization, balance checking, attrition analysis,
and pre-analysis plan generation for RCTs.
"""

from .attrition import AttritionResult, attrition_bounds, attrition_test
from .design import BalanceResult, RandomizationResult, balance_check, randomize
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
]
