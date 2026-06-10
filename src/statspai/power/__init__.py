"""
Power and sample size calculations for causal inference and epidemiological
designs.

Supports RCT, DID, RD, IV, cluster RCT, and OLS — plus epidemiological study
designs (two-proportion, log-rank/survival, case-control) — with power
curves, minimum detectable effect (MDE), and sample-size solving.
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
from .study_designs import (
    power_two_proportions,
    power_logrank,
    power_case_control,
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
    "power_two_proportions",
    "power_logrank",
    "power_case_control",
]
