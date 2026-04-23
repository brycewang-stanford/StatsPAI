"""
Bunching Estimator for kink/notch analysis.

Estimates behavioural responses to policy thresholds (tax kinks,
regulatory notches) by comparing the observed distribution of a
running variable to a counterfactual polynomial distribution.

References
----------
Kleven, H. J. & Waseem, M. (2013).
Using Notches to Uncover Optimization Frictions and Structural Elasticities.
QJE, 128(2), 669-723. [@kleven2013using]

Chetty, R., Friedman, J. N., Olsen, T., & Pistaferri, L. (2011).
Adjustment Costs, Firm Responses, and Micro vs. Macro Labor Supply
Elasticities. QJE, 126(2), 749-804. [@chetty2011adjustment]
"""

from .bunching import bunching, BunchingEstimator
from .notch import notch, NotchResult
# v0.10 high-order + unified-with-RDD/RKD
from .general import general_bunching, GeneralBunchingResult
from .kink_unified import kink_unified, KinkUnifiedResult

__all__ = [
    'bunching', 'BunchingEstimator', 'notch', 'NotchResult',
    'general_bunching', 'GeneralBunchingResult',
    'kink_unified', 'KinkUnifiedResult',
]
