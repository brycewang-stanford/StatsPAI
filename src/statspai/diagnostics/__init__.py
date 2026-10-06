"""
Diagnostics and sensitivity analysis for StatsPAI.

Provides:
- Oster (2019) coefficient stability bounds
- McCrary (2008) density discontinuity test for RD manipulation
"""

from .battery import diagnose_result
from .binned import binned_residuals, binned_residuals_plot
from .causal_gap import causal_gap
from .cmtest import cmtest
from .confounder_bias import confounder_adjust, confounder_tip
from .estat import estat
from .evalue import bias_factor, evalue, evalue_from_result, evalue_rd
from .hausman import hausman, hausman_test
from .influence import influence_measures, logit_gof, logit_influence
from .late_test import KitagawaResult, kitagawa_test
from .rddensity import rddensity
from .rosenbaum import RosenbaumResult, rosenbaum_bounds, rosenbaum_gamma
from .rosenbaum_strata import (
    EvidenceFactorsResult,
    SensitivityTestResult,
    amplify,
    evidence_factors,
    noether_test,
    rosenbaum_stratified,
    truncated_product,
)
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
from .weighted_rank import (
    WeightedRankPowerResult,
    WeightedRankResult,
    weighted_rank,
    weighted_rank_power,
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
    "binned_residuals",
    "binned_residuals_plot",
    "influence_measures",
    "logit_gof",
    "logit_influence",
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
    "causal_gap",
    "confounder_adjust",
    "confounder_tip",
    "diagnose_result",
    "estat",
    "kitagawa_test",
    "KitagawaResult",
    "rosenbaum_bounds",
    "rosenbaum_gamma",
    "RosenbaumResult",
    "weighted_rank",
    "weighted_rank_power",
    "WeightedRankResult",
    "WeightedRankPowerResult",
    "rosenbaum_stratified",
    "noether_test",
    "evidence_factors",
    "truncated_product",
    "amplify",
    "SensitivityTestResult",
    "EvidenceFactorsResult",
]
