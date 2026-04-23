"""
Continuous Treatment Effects: Dose-Response Estimation.

Estimates the dose-response function E[Y(t)] for continuous treatments
using the generalized propensity score (GPS) approach and kernel-based
methods.

References
----------
Hirano, K. & Imbens, G. W. (2004).
The Propensity Score with Continuous Treatments.
Applied Bayesian Modeling and Causal Inference from Incomplete-Data
Perspectives, 226164, 73-84. [@hirano2004propensity]

Kennedy, E. H., Ma, Z., McHugh, M. D., & Small, D. S. (2017).
Non-parametric methods for doubly robust estimation of continuous
treatment effects. JRSS-B, 79(4), 1229-1245. [@kennedy2017parametric]
"""

from .gps import dose_response, DoseResponse
from .vcnet import vcnet, scigan, VCNetResult

__all__ = [
    'dose_response', 'DoseResponse',
    'vcnet', 'scigan', 'VCNetResult',
]
