"""
Regression module initialization
"""

from .ols import regress, OLSRegression, OLSEstimator
from .iv import iv, ivreg, IVRegression, IVEstimator
from .heckman import heckman
from .quantile import qreg, sqreg
from .tobit import tobit
from .logit_probit import logit, probit, cloglog
from .glm import glm, GLMRegression, GLMEstimator
from .zeroinflated import zip_model, zinb, hurdle
from .count import poisson, nbreg, ppmlhdfe
from .multinomial import mlogit, ologit, oprobit, clogit

__all__ = [
    "regress",
    "OLSRegression",
    "OLSEstimator",
    "iv",
    "ivreg",
    "IVRegression",
    "IVEstimator",
    "heckman",
    "qreg",
    "sqreg",
    "tobit",
    "logit",
    "probit",
    "cloglog",
    "glm",
    "GLMRegression",
    "GLMEstimator",
    "zip_model",
    "zinb",
    "hurdle",
    "poisson",
    "nbreg",
    "ppmlhdfe",
    "mlogit",
    "ologit",
    "oprobit",
    "clogit",
]
