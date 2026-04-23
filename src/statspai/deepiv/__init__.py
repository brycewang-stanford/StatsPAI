"""
DeepIV: Deep Learning Instrumental Variables (Hartford et al. 2017).

Uses a two-stage neural network approach:
  Stage 1: Mixture Density Network estimates P(T | Z, X)
  Stage 2: Response network minimises counterfactual loss using
           Monte-Carlo samples from the learned treatment distribution.

References
----------
Hartford, J., Lewis, G., Leyton-Brown, K., & Taddy, M. (2017).
"Deep IV: A Flexible Approach for Counterfactual Prediction."
*Proceedings of the 34th International Conference on Machine Learning (ICML)*. [@hartford2017deep]
"""

from .deep_iv import deepiv, DeepIV

__all__ = ['deepiv', 'DeepIV']
