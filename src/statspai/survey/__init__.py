"""
Survey design and weighted estimation — StatsPAI's answer to R's ``survey``
package and Stata's ``svy:`` prefix.

Supports stratified, clustered, and weighted survey designs with
design-corrected standard errors for means, totals, and regression.

>>> import statspai as sp
>>> design = sp.svydesign(data=df, weights='pw', strata='stratum',
...                       cluster='psu')
>>> design.mean('income')
>>> design.total('income')
>>> design.glm('income ~ education + age')
"""

from .design import SurveyDesign, svydesign
from .estimators import svymean, svytotal, svyglm
from .calibration import rake, linear_calibration, CalibrationResult

__all__ = [
    "SurveyDesign",
    "svydesign",
    "svymean",
    "svytotal",
    "svyglm",
    "rake",
    "linear_calibration",
    "CalibrationResult",
]
