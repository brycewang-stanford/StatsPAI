"""
Regression module initialization
"""

from .ols import regress, OLSRegression, OLSEstimator
from .iv import ivreg, IVRegression, IVEstimator
from .heckman import heckman
from .quantile import qreg, sqreg

__all__ = [
    "regress",
    "OLSRegression",
    "OLSEstimator",
    "ivreg",
    "IVRegression",
    "IVEstimator",
    "heckman",
    "qreg",
    "sqreg",
]
