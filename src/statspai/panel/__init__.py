"""
Panel regression module for StatsPAI.

Wraps linearmodels to provide Stata-style panel regression with:
- Fixed Effects (within estimator)
- Random Effects (GLS)
- Between estimator
- First Differences
- Clustered / robust standard errors

References
----------
Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data.
Arellano, M. (1987). Computing Robust Standard Errors for Within-groups Estimators.
"""

from .panel_reg import panel, PanelRegression

__all__ = ['panel', 'PanelRegression']
