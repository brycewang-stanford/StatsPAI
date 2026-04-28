"""
Production function estimation — proxy-variable estimators.

Public surface
--------------
* :func:`prod_fn`              — unified ``method=`` dispatcher
* :func:`olley_pakes`          — OP (1996), investment proxy
* :func:`levinsohn_petrin`     — LP (2003), intermediate-input proxy
* :func:`ackerberg_caves_frazer` — ACF (2015), corrected identification
* :func:`wooldridge_prod`      — Wooldridge (2009), one-step joint GMM
* :func:`markup`               — De Loecker & Warzynski (2012) firm-time markup
* :class:`ProductionResult`    — unified result container
"""

from ._result import ProductionResult
from ._dispatcher import prod_fn
from .op_lp_acf import (
    ackerberg_caves_frazer,
    levinsohn_petrin,
    olley_pakes,
)
from .wooldridge import wooldridge_prod
from .markup import markup

# Convenience aliases that match common Stata / R names.
acf = ackerberg_caves_frazer
opreg = olley_pakes
levpet = levinsohn_petrin

__all__ = [
    "prod_fn",
    "olley_pakes",
    "opreg",
    "levinsohn_petrin",
    "levpet",
    "ackerberg_caves_frazer",
    "acf",
    "wooldridge_prod",
    "markup",
    "ProductionResult",
]
