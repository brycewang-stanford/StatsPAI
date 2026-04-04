"""
Diagnostics and sensitivity analysis for StatsPAI.

Provides:
- Oster (2019) coefficient stability bounds
- McCrary (2008) density discontinuity test for RD manipulation
"""

from .sensitivity import oster_bounds, mccrary_test

__all__ = [
    'oster_bounds',
    'mccrary_test',
]
