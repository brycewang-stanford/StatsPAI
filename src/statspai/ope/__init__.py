"""
Off-Policy Evaluation (``sp.ope``): estimate the value of a target
policy from data collected under a different behaviour policy. Covers
contextual bandits and off-policy reinforcement learning evaluation.

Implemented: DM, IPS, SNIPS, DR, Switch-DR, sharp OPE under
unobserved confounding (Hess, Frauen, Melnychuk & Feuerriegel 2025,
arXiv:2502.13022), causal-policy forest (Kato 2025, arXiv:2512.22846).
"""

from typing import Any

from .estimators import (
    direct_method,
    ips,
    snips,
    doubly_robust,
    switch_dr,
    evaluate,
    OPEResult,
)

__all__ = [
    "direct_method",
    "ips",
    "snips",
    "doubly_robust",
    "switch_dr",
    "evaluate",
    "OPEResult",
    "sharp_ope_unobserved",
    "causal_policy_forest",
    "SharpOPEResult",
    "CausalPolicyForestResult",
]

_SHARP_EXPORTS = {
    "sharp_ope_unobserved",
    "causal_policy_forest",
    "SharpOPEResult",
    "CausalPolicyForestResult",
}


def __getattr__(name: str) -> Any:
    """Load sklearn-backed sharp OPE extensions only on first use."""
    if name in _SHARP_EXPORTS:
        from . import sharp_confounding as _sharp

        obj = getattr(_sharp, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module 'statspai.ope' has no attribute {name!r}")
