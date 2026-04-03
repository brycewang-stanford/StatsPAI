"""
StatsPAI: The AI-powered Statistics & Econometrics Toolkit for Python

Unified API for causal inference and econometrics:

>>> import statspai as sp
>>>
>>> # OLS regression
>>> result = sp.regress("y ~ x1 + x2", data=df)
>>>
>>> # Difference-in-Differences
>>> result = sp.did(df, y='wage', treat='treated', time='post')
>>>
>>> # Staggered DID (Callaway & Sant'Anna)
>>> result = sp.did(df, y='wage', treat='first_treat',
...                time='year', id='worker_id')
>>>
>>> # Causal Forest
>>> cf = sp.causal_forest("y ~ treatment | x1 + x2", data=df)
>>>
>>> # Publication-quality export
>>> sp.outreg2(result, filename="results.xlsx")
"""

__version__ = "0.1.0"
__author__ = "Bryce Wang"
__email__ = "bryce@copaper.ai"

from .core.results import EconometricResults, CausalResult
from .regression.ols import regress
from .regression.iv import ivreg, IVRegression
from .causal.causal_forest import CausalForest, causal_forest
from .did import did, did_2x2, callaway_santanna
from .rd import rdrobust, rdplot
from .output.outreg2 import OutReg2, outreg2

__all__ = [
    # Core
    "EconometricResults",
    "CausalResult",
    # Regression
    "regress",
    "ivreg",
    "IVRegression",
    # DID
    "did",
    "did_2x2",
    "callaway_santanna",
    # RD
    "rdrobust",
    "rdplot",
    # Causal Forest
    "CausalForest",
    "causal_forest",
    # Output
    "OutReg2",
    "outreg2",
]
