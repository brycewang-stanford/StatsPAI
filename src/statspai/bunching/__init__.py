"""
Bunching Estimator for kink/notch analysis.

Estimates behavioural responses to policy thresholds (tax kinks,
regulatory notches) by comparing the observed distribution of a
running variable to a counterfactual polynomial distribution.

References
----------
Kleven, H. J. & Waseem, M. (2013).
Using Notches to Uncover Optimization Frictions and Structural Elasticities.
QJE, 128(2), 669-723.

Chetty, R., Friedman, J. N., Olsen, T., & Pistaferri, L. (2011).
Adjustment Costs, Firm Responses, and Micro vs. Macro Labor Supply
Elasticities. QJE, 126(2), 749-804.
"""

from .bunching import bunching, BunchingEstimator

__all__ = ['bunching', 'BunchingEstimator']
