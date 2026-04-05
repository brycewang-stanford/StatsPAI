"""
Experimental design and analysis tools.

Provides randomization, balance checking, attrition analysis,
and pre-analysis plan generation for RCTs.
"""

from .design import randomize, RandomizationResult, balance_check, BalanceResult
from .attrition import attrition_test, attrition_bounds, AttritionResult
from .optimal import optimal_design, OptimalDesignResult

__all__ = [
    "randomize", "RandomizationResult", "balance_check", "BalanceResult",
    "attrition_test", "attrition_bounds", "AttritionResult",
    "optimal_design", "OptimalDesignResult",
]
