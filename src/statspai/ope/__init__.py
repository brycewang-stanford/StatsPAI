"""
Off-Policy Evaluation (``sp.ope``): estimate the value of a target
policy from data collected under a different behaviour policy. Covers
contextual bandits and off-policy reinforcement learning evaluation.

Implemented: DM, IPS, SNIPS, DR, Switch-DR.
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

__all__ = [
    "direct_method",
    "ips",
    "snips",
    "doubly_robust",
    "switch_dr",
    "evaluate",
    "OPEResult",
]
