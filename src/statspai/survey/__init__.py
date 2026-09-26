"""
Survey design and weighted estimation — StatsPAI's answer to R's ``survey``
package and Stata's ``svy:`` prefix.

Supports stratified, clustered, and weighted survey designs with
design-corrected standard errors for means, totals, and regression.

>>> import numpy as np
>>> import pandas as pd
>>> import statspai as sp
>>> rng = np.random.default_rng(0)
>>> n = 200
>>> df = pd.DataFrame({'stratum': rng.integers(0, 4, n),
...                    'education': rng.integers(8, 18, n),
...                    'age': rng.integers(20, 65, n),
...                    'pw': rng.uniform(1, 3, n)})
>>> df['psu'] = df['stratum'] * 10 + rng.integers(0, 5, n)
>>> df['income'] = (10 + 2 * df['education'] + 0.3 * df['age']
...                 + rng.normal(0, 5, n))
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
