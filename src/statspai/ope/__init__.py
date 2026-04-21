"""
Off-Policy Evaluation (``sp.ope``): estimate the value of a target
policy from data collected under a different behaviour policy. Covers
contextual bandits and off-policy reinforcement learning evaluation.

Implemented: DM, IPS, SNIPS, DR, Switch-DR, sharp OPE under
unobserved confounding (Kallus-Mao-Uehara 2025), causal-policy forest
(arXiv:2512.22846).
"""

from .estimators import (
    direct_method,
    ips,
    snips,
    doubly_robust,
    switch_dr,
    evaluate,
    OPEResult,
)
from .sharp_confounding import (
    sharp_ope_unobserved, causal_policy_forest,
    SharpOPEResult, CausalPolicyForestResult,
)

__all__ = [
    "direct_method",
    "ips",
    "snips",
    "doubly_robust",
    "switch_dr",
    "evaluate",
    "OPEResult",
    "sharp_ope_unobserved", "causal_policy_forest",
    "SharpOPEResult", "CausalPolicyForestResult",
]
