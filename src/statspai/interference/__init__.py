"""
Interference and Spillover Effects.

Estimates direct and spillover treatment effects when SUTVA
(Stable Unit Treatment Value Assumption) is violated, i.e.,
one unit's treatment affects another unit's outcome.

References
----------
Hudgens, M. G. & Halloran, M. E. (2008).
Toward Causal Inference with Interference.
JASA, 103(482), 832-842.

Aronow, P. M. & Samii, C. (2017).
Estimating Average Causal Effects Under General Interference.
Annals of Applied Statistics, 11(4), 1912-1947.
"""

from .spillover import spillover, SpilloverEstimator
from .network_exposure import network_exposure, NetworkExposureResult
from .peer_effects import peer_effects, PeerEffectsResult
from .orthogonal import (
    network_hte, inward_outward_spillover,
    NetworkHTEResult, InwardOutwardResult,
)

__all__ = [
    'spillover', 'SpilloverEstimator',
    'network_exposure', 'NetworkExposureResult',
    'peer_effects', 'PeerEffectsResult',
    'network_hte', 'inward_outward_spillover',
    'NetworkHTEResult', 'InwardOutwardResult',
]
