"""
Regression Discontinuity (RD) module for StatsPAI.

Provides:
- Sharp and Fuzzy RD estimation with robust bias-corrected inference (CCT 2014)
- MSE-optimal bandwidth selection
- RD plots with binned scatter and polynomial fits

Planned:
- McCrary (2008) density manipulation test
- Cattaneo-Jansson-Ma (2020) manipulation test
- Kink RD
"""

from .rdrobust import rdrobust, rdplot

__all__ = [
    'rdrobust',
    'rdplot',
]
