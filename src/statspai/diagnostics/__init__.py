"""
Diagnostics and sensitivity analysis for StatsPAI.

Provides:
- Oster (2019) coefficient stability bounds
- McCrary (2008) density discontinuity test for RD manipulation
"""

from .battery import diagnose_result
from .cmtest import cmtest
from .confounder_bias import confounder_adjust, confounder_tip
from .estat import estat
from .evalue import bias_factor, evalue, evalue_from_result, evalue_rd
from .hausman import hausman, hausman_test
from .late_test import KitagawaResult, kitagawa_test
from .rddensity import rddensity
from .rosenbaum import RosenbaumResult, rosenbaum_bounds, rosenbaum_gamma
from .sensemakr import sensemakr
from .sensitivity import mccrary_test, oster_bounds
from .tests import diagnose, het_test, reset_test, vif
from .vuong import vuong
from .weak_iv import (
    WeakRobustResult,
    anderson_rubin_test,
    effective_f_test,
    tF_critical_value,
    weakrobust,
)

__all__ = [
    "oster_bounds",
    "mccrary_test",
    "diagnose",
    "het_test",
    "reset_test",
    "cmtest",
    "vuong",
    "vif",
    "sensemakr",
    "rddensity",
    "hausman",
    "hausman_test",
    "anderson_rubin_test",
    "effective_f_test",
    "tF_critical_value",
    "weakrobust",
    "WeakRobustResult",
    "evalue",
    "evalue_from_result",
    "evalue_rd",
    "bias_factor",
    "confounder_adjust",
    "confounder_tip",
    "diagnose_result",
    "estat",
    "kitagawa_test",
    "KitagawaResult",
    "rosenbaum_bounds",
    "rosenbaum_gamma",
    "RosenbaumResult",
]
