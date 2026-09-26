"""
Parametric g-formula via Iterative Conditional Expectation (ICE).

Sequential g-computation for longitudinal data with time-varying
treatments and time-varying confounding -- the estimator pioneered
by Robins (1986) and made tractable in its ICE form by Bang &
Robins (2005).

Public API
----------
>>> import numpy as np
>>> import pandas as pd
>>> import statspai as sp
>>> rng = np.random.default_rng(0)
>>> n = 300
>>> l0 = rng.normal(size=n)
>>> a0 = rng.binomial(1, 0.5, size=n)
>>> l1 = 0.5 * l0 + 0.3 * a0 + rng.normal(size=n)
>>> a1 = rng.binomial(1, 0.5, size=n)
>>> l2 = 0.5 * l1 + 0.3 * a1 + rng.normal(size=n)
>>> a2 = rng.binomial(1, 0.5, size=n)
>>> y = (1 + 0.8 * a0 + 1.2 * a1 + 0.5 * a2
...      + 0.5 * l0 + 0.4 * l1 + 0.3 * l2 + rng.normal(size=n))
>>> df = pd.DataFrame({"id": range(n), "t": 0, "L0": l0, "A0": a0, "L1": l1,
...                    "A1": a1, "L2": l2, "A2": a2, "Y": y})
>>> result = sp.gformula.ice(
...     data=df,
...     id_col="id", time_col="t",
...     treatment_cols=["A0", "A1", "A2"],
...     confounder_cols=[["L0"], ["L1"], ["L2"]],
...     outcome_col="Y",
...     treatment_strategy=[1, 1, 1],  # always-treat
... )
>>> type(result).__name__, result.strategy
('ICEResult', [1, 1, 1])
"""

from .ice import ICEResult, gformula_ice, ice
from .mc import MCGFormulaResult, gformula_mc

__all__ = [
    "ice",
    "gformula_ice",
    "ICEResult",
    "gformula_mc",
    "MCGFormulaResult",
]
