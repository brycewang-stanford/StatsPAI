"""
Conformal Causal Inference: Distribution-free prediction intervals for ITE.

Provides prediction intervals for individual treatment effects (ITE)
without distributional assumptions, using conformal inference.

References
----------
Lei, L. & Candes, E. J. (2021).
Conformal Inference of Counterfactuals and Individual Treatment Effects.
JRSS-B, 83(5), 911-938.

Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021).
An Exact and Robust Conformal Inference Method for Counterfactual and
Synthetic Controls. JASA, 116(536), 1849-1864.
"""

from .conformal_ite import conformal_cate, ConformalCATE
from .counterfactual import (
    weighted_conformal_prediction,
    conformal_counterfactual,
    conformal_ite_interval,
    ConformalCounterfactualResult,
    ConformalITEResult,
)

# v0.10 conformal frontier: density / multidp / debiased / fair
from .conformal_density import conformal_density_ite, ConformalDensityResult
from .conformal_multidp import conformal_ite_multidp, MultiDPConformalResult
from .conformal_debiased import conformal_debiased_ml, DebiasedConformalResult
from .conformal_fair import conformal_fair_ite, FairConformalResult

# v1.0 conformal frontier: continuous-treatment + interference
from .extended import (
    conformal_continuous, conformal_interference,
    ContinuousConformalResult, InterferenceConformalResult,
)

__all__ = [
    'conformal_cate', 'ConformalCATE',
    'weighted_conformal_prediction',
    'conformal_counterfactual', 'ConformalCounterfactualResult',
    'conformal_ite_interval', 'ConformalITEResult',
    'conformal_density_ite', 'ConformalDensityResult',
    'conformal_ite_multidp', 'MultiDPConformalResult',
    'conformal_debiased_ml', 'DebiasedConformalResult',
    'conformal_fair_ite', 'FairConformalResult',
    'conformal_continuous', 'conformal_interference',
    'ContinuousConformalResult', 'InterferenceConformalResult',
]
