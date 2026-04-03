"""
High-dimensional fixed effects estimation via pyfixest.

This module provides thin wrappers around pyfixest's estimation functions,
converting results into StatsPAI's ``EconometricResults`` for seamless
integration with ``outreg2`` and the rest of the ecosystem.

Requires: ``pip install pyfixest``

Examples
--------
>>> from statspai.fixest import feols, fepois
>>>
>>> # Two-way fixed effects with clustered SEs
>>> result = feols("wage ~ experience | firm + year",
...               data=df, vcov={"CRV1": "firm"})
>>> print(result.summary())
>>>
>>> # Poisson regression
>>> result = fepois("patents ~ rd_spending | firm", data=df)
>>>
>>> # Works with outreg2
>>> from statspai import outreg2
>>> outreg2(result, filename="table.xlsx")
"""

from .wrapper import feols, fepois, feglm, etable

__all__ = [
    "feols",
    "fepois",
    "feglm",
    "etable",
]
