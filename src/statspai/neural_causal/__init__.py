"""
Neural Causal Inference Models.

Deep learning approaches to treatment effect estimation that leverage
representation learning to handle high-dimensional confounders and
complex outcome surfaces.

Models
------
- **TARNet** : Treatment-Agnostic Representation Network (Shalit et al. 2017)
- **CFRNet** : Counterfactual Regression with IPM regularisation (Shalit et al. 2017)
- **DragonNet** : Targeted regularisation for causal estimation (Shi et al. 2019)

All models require PyTorch:
    pip install statspai[neural]  # or: pip install torch

References
----------
Shalit, U., Johansson, F. D., & Sontag, D. (2017).
Estimating individual treatment effect: generalization bounds and algorithms.
Proceedings of the 34th International Conference on Machine Learning (ICML).

Shi, C., Blei, D. M., & Veitch, V. (2019).
Adapting neural networks for the estimation of treatment effects.
Advances in Neural Information Processing Systems (NeurIPS), 32.
"""

from .models import (
    tarnet,
    cfrnet,
    dragonnet,
    TARNet,
    CFRNet,
    DragonNet,
)
from .gnn_causal import gnn_causal, GNNCausalResult

__all__ = [
    'tarnet',
    'cfrnet',
    'dragonnet',
    'TARNet',
    'CFRNet',
    'DragonNet',
    'gnn_causal', 'GNNCausalResult',
]
