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
- **CEVAE** : Causal effect variational autoencoder for latent confounding

All models require PyTorch:
    pip install statspai[neural]  # or: pip install torch

References
----------
Shalit, U., Johansson, F. D., & Sontag, D. (2017).
Estimating individual treatment effect: generalization bounds and algorithms.
Proceedings of the 34th International Conference on Machine Learning
(ICML). [@shalit2017estimating]

Shi, C., Blei, D. M., & Veitch, V. (2019).
Adapting neural networks for the estimation of treatment effects.
Advances in Neural Information Processing Systems (NeurIPS), 32. [@shi2019adapting]
"""

from typing import Any

__all__ = [
    "tarnet",
    "cfrnet",
    "dragonnet",
    "TARNet",
    "CFRNet",
    "DragonNet",
    "cevae",
    "CEVAE",
    "CEVAEResult",
    "gnn_causal",
    "GNNCausalResult",
    "neural_effects_frame",
    "neural_summary_frame",
    "neural_training_frame",
    "neural_causal_to_markdown",
    "neural_causal_to_html",
    "neural_causal_to_excel",
    "neural_causal_plot",
]

_LAZY_ATTRS = {
    "tarnet": ("models", "tarnet"),
    "cfrnet": ("models", "cfrnet"),
    "dragonnet": ("models", "dragonnet"),
    "TARNet": ("models", "TARNet"),
    "CFRNet": ("models", "CFRNet"),
    "DragonNet": ("models", "DragonNet"),
    "cevae": ("cevae", "cevae"),
    "CEVAE": ("cevae", "CEVAE"),
    "CEVAEResult": ("cevae", "CEVAEResult"),
    "gnn_causal": ("gnn_causal", "gnn_causal"),
    "GNNCausalResult": ("gnn_causal", "GNNCausalResult"),
    "neural_effects_frame": ("exports", "neural_effects_frame"),
    "neural_summary_frame": ("exports", "neural_summary_frame"),
    "neural_training_frame": ("exports", "neural_training_frame"),
    "neural_causal_to_markdown": ("exports", "neural_causal_to_markdown"),
    "neural_causal_to_html": ("exports", "neural_causal_to_html"),
    "neural_causal_to_excel": ("exports", "neural_causal_to_excel"),
    "neural_causal_plot": ("plots", "neural_causal_plot"),
}


def __getattr__(name: str) -> Any:
    """Resolve optional PyTorch/sklearn-backed neural exports lazily."""
    if name in _LAZY_ATTRS:
        import importlib

        module_name, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{module_name}", package=__name__)
        obj = getattr(mod, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'statspai.neural_causal' has no attribute {name!r}")
