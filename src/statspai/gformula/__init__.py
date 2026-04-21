"""
Parametric g-formula via Iterative Conditional Expectation (ICE).

Sequential g-computation for longitudinal data with time-varying
treatments and time-varying confounding -- the estimator pioneered
by Robins (1986) and made tractable in its ICE form by Bang &
Robins (2005).

Public API
----------
>>> import statspai as sp
>>> result = sp.gformula.ice(
...     data=df,
...     id_col="id", time_col="t",
...     treatment_cols=["A0", "A1", "A2"],
...     confounder_cols=["L0", "L1", "L2"],
...     outcome_col="Y",
...     treatment_strategy=[1, 1, 1],  # always-treat
... )
>>> result.summary()
"""

from .ice import ice, gformula_ice, ICEResult
from .mc import gformula_mc, MCGFormulaResult

__all__ = [
    "ice", "gformula_ice", "ICEResult",
    "gformula_mc", "MCGFormulaResult",
]
