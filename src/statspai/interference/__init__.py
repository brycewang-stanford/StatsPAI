"""
Interference and Spillover Effects.

Estimates direct and spillover treatment effects when SUTVA
(Stable Unit Treatment Value Assumption) is violated, i.e.,
one unit's treatment affects another unit's outcome.

References
----------
Hudgens, M. G. & Halloran, M. E. (2008).
Toward Causal Inference with Interference.
JASA, 103(482), 832-842. [@hudgens2008toward]

Aronow, P. M. & Samii, C. (2017).
Estimating Average Causal Effects Under General Interference.
Annals of Applied Statistics, 11(4), 1912-1947. [@aronow2017estimating]
"""

from .cluster_cross import CrossClusterRCTResult, cluster_cross_interference

# v0.10 Cluster RCT × interference suite
from .cluster_matched_pair import MatchedPairResult, cluster_matched_pair
from .cluster_staggered import StaggeredClusterRCTResult, cluster_staggered_rollout

# v1.5 unified dispatcher
from .dispatcher import available_designs as interference_available_designs
from .dispatcher import interference
from .dnc_gnn_did import DNCGNNDiDResult, dnc_gnn_did
from .network_exposure import NetworkExposureResult, network_exposure
from .orthogonal import (
    InwardOutwardResult,
    NetworkHTEResult,
    inward_outward_spillover,
    network_hte,
)
from .peer_effects import PeerEffectsResult, peer_effects
from .randomization_test import InterferenceTestResult, interference_test
from .spillover import SpilloverEstimator, spillover

__all__ = [
    "spillover",
    "SpilloverEstimator",
    "network_exposure",
    "NetworkExposureResult",
    "interference_test",
    "InterferenceTestResult",
    "peer_effects",
    "PeerEffectsResult",
    "network_hte",
    "inward_outward_spillover",
    "NetworkHTEResult",
    "InwardOutwardResult",
    "cluster_matched_pair",
    "MatchedPairResult",
    "cluster_cross_interference",
    "CrossClusterRCTResult",
    "cluster_staggered_rollout",
    "StaggeredClusterRCTResult",
    "dnc_gnn_did",
    "DNCGNNDiDResult",
    # v1.5 dispatcher
    "interference",
    "interference_available_designs",
]
