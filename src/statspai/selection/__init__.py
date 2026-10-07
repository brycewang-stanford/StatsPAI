"""
Variable selection tools.

- ``stepwise``: Stepwise regression with AIC/BIC/p-value criteria
- ``lasso_select``: LASSO-based variable selection with coordinate descent
- ``best_subset``: exact best-subset selection by branch and bound
- ``glmnet``: lasso / ridge / elastic net with the conventions of R
  ``glmnet`` (penalty path, penalty factors, ``lambda.min`` / ``lambda.1se``)
- ``shrinkage``: ridge / lasso / principal-components prediction, tuned by
  cross-validation
"""

from .best_subset import best_subset
from .elastic_net import GlmnetResult, glmnet
from .shrinkage import ShrinkageResult, shrinkage
from .stepwise import SelectionResult, lasso_select, stepwise

__all__ = [
    "stepwise",
    "lasso_select",
    "best_subset",
    "SelectionResult",
    "shrinkage",
    "glmnet",
    "GlmnetResult",
    "ShrinkageResult",
]
