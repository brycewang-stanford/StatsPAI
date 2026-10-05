"""
Conformal Causal Inference: Distribution-free prediction intervals for ITE.

Provides prediction intervals for individual treatment effects (ITE)
without distributional assumptions, using conformal inference.

References
----------
Lei, L. & Candes, E. J. (2021).
Conformal Inference of Counterfactuals and Individual Treatment Effects.
JRSS-B, 83(5), 911-938. [@lei2021conformal]

Chernozhukov, V., Wuthrich, K., & Zhu, Y. (2021).
An Exact and Robust Conformal Inference Method for Counterfactual and
Synthetic Controls. JASA, 116(536), 1849-1864. [@chernozhukov2021exact]
"""

from .conformal_debiased import DebiasedConformalResult, conformal_debiased_ml

# v0.10 conformal frontier: density / multidp / debiased / fair
from .conformal_density import ConformalDensityResult, conformal_density_ite
from .conformal_fair import FairConformalResult, conformal_fair_ite
from .conformal_ite import ConformalCATE, conformal_cate
from .conformal_multidp import MultiDPConformalResult, conformal_ite_multidp
from .counterfactual import (
    ConformalCounterfactualResult,
    ConformalITEResult,
    conformal_counterfactual,
    conformal_ite_interval,
    weighted_conformal_prediction,
)

# v1.5 unified dispatcher
from .dispatcher import available_kinds as conformal_available_kinds
from .dispatcher import conformal

# v1.0 conformal frontier: continuous-treatment + interference
from .extended import (
    ContinuousConformalResult,
    InterferenceConformalResult,
    conformal_continuous,
    conformal_interference,
)
from .regression import conformal_regression

__all__ = [
    "conformal_cate",
    "ConformalCATE",
    "weighted_conformal_prediction",
    "conformal_regression",
    "conformal_counterfactual",
    "ConformalCounterfactualResult",
    "conformal_ite_interval",
    "ConformalITEResult",
    "conformal_density_ite",
    "ConformalDensityResult",
    "conformal_ite_multidp",
    "MultiDPConformalResult",
    "conformal_debiased_ml",
    "DebiasedConformalResult",
    "conformal_fair_ite",
    "FairConformalResult",
    "conformal_continuous",
    "conformal_interference",
    "ContinuousConformalResult",
    "InterferenceConformalResult",
    # v1.5 dispatcher
    "conformal",
    "conformal_available_kinds",
]
