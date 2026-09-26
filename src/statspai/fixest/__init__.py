"""
High-dimensional fixed effects estimation via pyfixest.

This module provides thin wrappers around pyfixest's estimation functions,
converting results into StatsPAI's ``EconometricResults`` for seamless
integration with ``outreg2`` and the rest of the ecosystem.

Requires: ``pip install pyfixest``

Examples
--------
>>> import os
>>> import tempfile
>>> import numpy as np
>>> import pandas as pd
>>> from statspai.fixest import feols, fepois
>>> rng = np.random.default_rng(0)
>>> n = 300
>>> df = pd.DataFrame({'firm': rng.integers(0, 20, n),
...                    'year': rng.integers(2000, 2006, n),
...                    'experience': rng.normal(10, 3, n),
...                    'rd_spending': rng.normal(5, 1, n)})
>>> df['wage'] = 1 + 0.5 * df['experience'] + rng.normal(size=n)
>>> df['patents'] = rng.poisson(np.exp(0.2 * df['rd_spending'] - 0.5))
>>>
>>> # Two-way fixed effects with clustered SEs
>>> # (pyfixest's first call JIT-compiles for ~20 s, hence the skips)
>>> result = feols("wage ~ experience | firm + year",  # doctest: +SKIP
...               data=df, vcov={"CRV1": "firm"})
>>> report = result.summary()  # doctest: +SKIP
>>>
>>> # Poisson regression
>>> result = fepois("patents ~ rd_spending | firm", data=df)  # doctest: +SKIP
>>>
>>> # Works with outreg2
>>> from statspai import outreg2
>>> path = os.path.join(tempfile.mkdtemp(), "table.xlsx")
>>> outreg2(result, filename=path)  # doctest: +SKIP, +ELLIPSIS
Regression results exported to: ...table.xlsx
"""

from .wrapper import etable, feglm, feols, fepois

__all__ = [
    "feols",
    "fepois",
    "feglm",
    "etable",
]
