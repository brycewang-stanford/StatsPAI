"""
Marginal Structural Models (MSM) for time-varying treatments.

Implements Robins' inverse-probability-of-treatment-weighted (IPTW)
estimator for longitudinal data with time-varying confounding.
"""

from .msm import msm, MarginalStructuralModel, stabilized_weights

__all__ = ['msm', 'MarginalStructuralModel', 'stabilized_weights']
