"""
Causal Impact module for StatsPAI.

Estimates the causal effect of an intervention on a time series by
constructing a synthetic counterfactual from control series using
a structural time-series model (regression + AR(1) state, fit by plug-in
estimates and the Kalman filter).

Same estimand and interface idea as Google's R CausalImpact package, but a
different, frequentist model: R CausalImpact fits a Bayesian local-level +
spike-and-slab regression by MCMC, so the numbers differ.

References
----------
Brodersen, K.H., Gallusser, F., Koehler, J., Remy, N., and Scott, S.L. (2015).
"Inferring Causal Impact Using Bayesian Structural Time-Series Models."
*Annals of Applied Statistics*, 9(1), 247-274. [@brodersen2015inferring]
"""

from .impact import CausalImpactEstimator, causal_impact, impactplot

__all__ = ["causal_impact", "CausalImpactEstimator", "impactplot"]
