"""
Robustness analysis tools.

- ``spec_curve``: Specification Curve Analysis (Simonsohn et al. 2020)
- ``robustness_report``: Automated battery of robustness checks
- ``subgroup_analysis``: Subgroup heterogeneity analysis with forest plot
"""

from .spec_curve import spec_curve, SpecCurveResult
from .robustness_report import robustness_report, RobustnessResult
from .subgroup import subgroup_analysis, SubgroupResult
from .unified_sensitivity import (
    SensitivityDashboard,
    unified_sensitivity,
)

__all__ = [
    "spec_curve", "SpecCurveResult",
    "robustness_report", "RobustnessResult",
    "subgroup_analysis", "SubgroupResult",
    "SensitivityDashboard", "unified_sensitivity",
]
