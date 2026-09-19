"""
Survey design and weighted estimation — StatsPAI's answer to R's ``survey``
package and Stata's ``svy:`` prefix.

Supports stratified, clustered, and weighted survey designs with
design-corrected standard errors for means, totals, and regression.

>>> import statspai as sp
>>> design = sp.svydesign(data=df, weights='pw', strata='stratum',
...                       cluster='psu')  # doctest: +SKIP
>>> design.mean('income')  # doctest: +SKIP
>>> design.total('income')  # doctest: +SKIP
>>> design.glm('income ~ education + age')  # doctest: +SKIP
"""

from .calibration import CalibrationResult, linear_calibration, rake
from .design import SurveyDesign, svydesign
from .estimators import svyglm, svymean, svytotal

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
