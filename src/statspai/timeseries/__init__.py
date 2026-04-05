"""
Time series methods for causal inference contexts.

Provides VAR (vector autoregression), structural break tests,
Granger causality, and cointegration analysis.
"""

from .var import var, VARResult, granger_causality, irf
from .structural_break import structural_break, StructuralBreakResult, cusum_test

__all__ = [
    "var", "VARResult", "granger_causality", "irf",
    "structural_break", "StructuralBreakResult", "cusum_test",
]
