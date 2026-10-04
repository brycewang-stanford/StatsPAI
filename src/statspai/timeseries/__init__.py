"""
Time series methods for causal inference contexts.

Provides VAR (vector autoregression), structural break tests,
Granger causality, unit-root tests, and cointegration analysis.
"""

from .ardl import ARDLResult, ardl
from .arima import ARIMAResult, arima
from .bvar import BVARResult, bvar
from .cointegration import CointegrationResult, engle_granger, johansen
from .corrgram import corrgram
from .garch import GARCHResult, garch
from .its import ITSResult, its
from .local_projections import LocalProjectionsResult, local_projections
from .structural_break import StructuralBreakResult, cusum_test, structural_break
from .svar import SVARResult, svar
from .unit_root import UnitRootResult, unitroot
from .var import VARResult, granger_causality, irf, var
from .var_diagnostics import varsoc
from .vecm import VECResult, vec

__all__ = [
    "var",
    "VARResult",
    "svar",
    "SVARResult",
    "granger_causality",
    "irf",
    "structural_break",
    "StructuralBreakResult",
    "cusum_test",
    "engle_granger",
    "johansen",
    "CointegrationResult",
    "local_projections",
    "LocalProjectionsResult",
    "garch",
    "GARCHResult",
    "arima",
    "ARIMAResult",
    "bvar",
    "BVARResult",
    "its",
    "ITSResult",
    "unitroot",
    "UnitRootResult",
    "ardl",
    "ARDLResult",
    "corrgram",
    "varsoc",
    "vec",
    "VECResult",
]
