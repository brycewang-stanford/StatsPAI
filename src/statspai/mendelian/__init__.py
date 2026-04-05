"""
Mendelian Randomization methods.

Uses genetic variants as instrumental variables to estimate causal effects
in epidemiological/health economics studies.
"""

from .mr import mendelian_randomization, MRResult, mr_egger, mr_ivw, mr_median, mr_plot

__all__ = [
    "mendelian_randomization", "MRResult",
    "mr_egger", "mr_ivw", "mr_median", "mr_plot",
]
