"""
Variable selection tools.

- ``stepwise``: Stepwise regression with AIC/BIC/p-value criteria
- ``lasso_select``: LASSO-based variable selection with coordinate descent
"""

from .stepwise import stepwise, lasso_select, SelectionResult

__all__ = [
    "stepwise", "lasso_select", "SelectionResult",
]
