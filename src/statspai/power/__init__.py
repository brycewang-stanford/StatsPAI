"""
Power and sample size calculations for causal inference designs.

Supports RCT, DID, RD, IV, cluster RCT, and OLS — with power curves,
minimum detectable effect (MDE), and sample-size solving.
"""

from .power import (
    power,
    PowerResult,
    power_rct,
    power_did,
    power_rd,
    power_iv,
    power_cluster_rct,
    power_ols,
    mde,
)

__all__ = [
    "power",
    "PowerResult",
    "power_rct",
    "power_did",
    "power_rd",
    "power_iv",
    "power_cluster_rct",
    "power_ols",
    "mde",
]
