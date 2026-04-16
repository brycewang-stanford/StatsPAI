"""
Matching module for StatsPAI.

Unified interface for matching estimators:

- Nearest-neighbor matching (propensity score, Mahalanobis, Euclidean)
- Exact matching
- Coarsened Exact Matching (CEM)
- Propensity score stratification / subclassification
- Abadie-Imbens (2011) bias correction
- Entropy balancing (Hainmueller 2012)

References
----------
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
Abadie, A. and Imbens, G.W. (2006). Econometrica, 74(1), 235-267.
Abadie, A. and Imbens, G.W. (2011). JBES, 29(1), 1-11.
Iacus, S.M., King, G., and Porro, G. (2012). Political Analysis, 20(1), 1-24.
Hainmueller, J. (2012). Political Analysis, 20(1), 25-46.
Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press.
"""

from .match import match, MatchEstimator, balanceplot, psplot
from .ebalance import ebalance
from .ps_diagnostics import (
    propensity_score, overlap_plot, trimming, love_plot,
    ps_balance, PSBalanceResult,
)
from .optimal import (
    optimal_match, cardinality_match,
    OptimalMatchResult, CardinalityMatchResult,
)

__all__ = [
    'match', 'MatchEstimator', 'ebalance', 'balanceplot', 'psplot',
    'propensity_score', 'overlap_plot', 'trimming', 'love_plot',
    'ps_balance', 'PSBalanceResult',
    'optimal_match', 'cardinality_match',
    'OptimalMatchResult', 'CardinalityMatchResult',
]
