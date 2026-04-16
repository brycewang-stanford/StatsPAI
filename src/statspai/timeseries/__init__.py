"""
Time series methods for causal inference contexts.

Provides VAR (vector autoregression), structural break tests,
Granger causality, and cointegration analysis.
"""

from .var import var, VARResult, granger_causality, irf
from .structural_break import structural_break, StructuralBreakResult, cusum_test
from .cointegration import engle_granger, johansen, CointegrationResult
from .local_projections import local_projections, LocalProjectionsResult

__all__ = [
    "var", "VARResult", "granger_causality", "irf",
    "structural_break", "StructuralBreakResult", "cusum_test",
    "engle_granger", "johansen", "CointegrationResult",
    "local_projections", "LocalProjectionsResult",
]
