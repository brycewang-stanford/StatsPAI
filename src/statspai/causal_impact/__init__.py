"""
Causal Impact module for StatsPAI.

Estimates the causal effect of an intervention on a time series by
constructing a synthetic counterfactual from control series using
a structural time-series model.

Equivalent to Google's R CausalImpact package.

References
----------
Brodersen, K.H., Gallusser, F., Koehler, J., Remy, N., and Scott, S.L. (2015).
"Inferring Causal Impact Using Bayesian Structural Time-Series Models."
*Annals of Applied Statistics*, 9(1), 247-274.
"""

from .impact import causal_impact, CausalImpactEstimator, impactplot

__all__ = ['causal_impact', 'CausalImpactEstimator', 'impactplot']
