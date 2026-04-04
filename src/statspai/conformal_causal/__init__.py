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

__all__ = ['conformal_cate', 'ConformalCATE']
