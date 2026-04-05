"""
Regression Discontinuity (RD) module for StatsPAI.

Provides:
- Sharp, Fuzzy, and Kink RD estimation (CCT 2014)
- Donut-hole RD for handling manipulation near the cutoff
- MSE-optimal bandwidth selection
- RD plots with binned scatter and polynomial fits
- Bandwidth sensitivity analysis
- Covariate balance tests at the cutoff
- Placebo cutoff tests
"""

from .rdrobust import rdrobust, rdplot, rdplotdensity
from .diagnostics import rdbwsensitivity, rdbalance, rdplacebo, rdsummary

__all__ = [
    'rdrobust',
    'rdplot',
    'rdplotdensity',
    'rdbwsensitivity',
    'rdbalance',
    'rdplacebo',
    'rdsummary',
]
