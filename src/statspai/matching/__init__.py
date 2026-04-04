"""
Matching module for StatsPAI.

Provides:
- Propensity Score Matching (PSM) with logit propensity scores
- Mahalanobis distance matching
- Coarsened Exact Matching (CEM)
- Balance diagnostics (standardized mean differences)

References
----------
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
Abadie, A. and Imbens, G. (2006). Econometrica, 74(1), 235-267.
Iacus, S.M., King, G., and Porro, G. (2012). Political Analysis, 20(1), 1-24.
"""

from .match import match, MatchEstimator

__all__ = ['match', 'MatchEstimator']
