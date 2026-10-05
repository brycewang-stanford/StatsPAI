"""
Robustness analysis tools.

- ``spec_curve``: Specification Curve Analysis (Simonsohn et al. 2020)
- ``robustness_report``: Automated battery of robustness checks
- ``subgroup_analysis``: Subgroup heterogeneity analysis with forest plot
- ``refute``: rerun an estimator on altered data whose answer is known
"""

from .refute import RefutationResult, refute
from .robustness_report import RobustnessResult, robustness_report
from .sensitivity_frontier import (
    FrontierSensitivityResult,
    calibrate_confounding_strength,
    copula_sensitivity,
    survival_sensitivity,
)
from .spec_curve import SpecCurveResult, spec_curve
from .subgroup import SubgroupResult, subgroup_analysis
from .unified_sensitivity import SensitivityDashboard, unified_sensitivity

__all__ = [
    "spec_curve",
    "SpecCurveResult",
    "robustness_report",
    "RobustnessResult",
    "subgroup_analysis",
    "SubgroupResult",
    "refute",
    "RefutationResult",
    "SensitivityDashboard",
    "unified_sensitivity",
    "copula_sensitivity",
    "survival_sensitivity",
    "calibrate_confounding_strength",
    "FrontierSensitivityResult",
]
