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
from .weak_iv import (
    anderson_rubin_test, effective_f_test, tF_critical_value,
    weakrobust, WeakRobustResult,
)
from .evalue import evalue, evalue_from_result
from .battery import diagnose_result
from .estat import estat
from .late_test import kitagawa_test, KitagawaResult
from .rosenbaum import rosenbaum_bounds, rosenbaum_gamma, RosenbaumResult

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
    'effective_f_test',
    'tF_critical_value',
    'weakrobust',
    'WeakRobustResult',
    'evalue',
    'evalue_from_result',
    'diagnose_result',
    'estat',
    'kitagawa_test',
    'KitagawaResult',
    'rosenbaum_bounds',
    'rosenbaum_gamma',
    'RosenbaumResult',
]
