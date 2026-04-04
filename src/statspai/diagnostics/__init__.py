"""
Diagnostics and sensitivity analysis for StatsPAI.

Provides:
- Oster (2019) coefficient stability bounds
- McCrary (2008) density discontinuity test for RD manipulation
"""

from .sensitivity import oster_bounds, mccrary_test
from .tests import diagnose, het_test, reset_test, vif
from .sensemakr import sensemakr
from .rddensity import rddensity
from .hausman import hausman_test
from .weak_iv import anderson_rubin_test

__all__ = [
    'oster_bounds',
    'mccrary_test',
    'diagnose',
    'het_test',
    'reset_test',
    'vif',
    'sensemakr',
    'rddensity',
    'hausman_test',
    'anderson_rubin_test',
]
