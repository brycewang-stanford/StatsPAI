"""
Utility functions for StatsPAI.

Provides Stata-style data manipulation tools:
- Variable labels (label_var, get_label)
- Pairwise correlation with stars (pwcorr)
- Winsorization (winsor)
"""

from .data_tools import pwcorr, winsor
from .dgp import (
    dgp_bartik,
    dgp_bunching,
    dgp_cluster_rct,
    dgp_did,
    dgp_iv,
    dgp_observational,
    dgp_panel,
    dgp_rct,
    dgp_rd,
    dgp_rd_2d,
    dgp_rd_hte,
    dgp_rd_kink,
    dgp_rd_multi,
    dgp_rdit,
    dgp_synth,
)
from .egen import (
    outlier_indicator,
    rank,
    rowcount,
    rowmax,
    rowmean,
    rowmin,
    rowsd,
    rowtotal,
)
from .io import read_data, write_data
from .iv_helpers import scalar_iv_projection
from .labels import describe, get_label, get_labels, label_var, label_vars

__all__ = [
    "label_var",
    "label_vars",
    "get_label",
    "get_labels",
    "describe",
    "pwcorr",
    "winsor",
    "rowmean",
    "rowtotal",
    "rowmax",
    "rowmin",
    "rowsd",
    "rowcount",
    "rank",
    "outlier_indicator",
    "read_data",
    "write_data",
    "scalar_iv_projection",
    # Data Generating Processes
    "dgp_did",
    "dgp_rd",
    "dgp_rd_kink",
    "dgp_rd_multi",
    "dgp_rd_hte",
    "dgp_rd_2d",
    "dgp_rdit",
    "dgp_iv",
    "dgp_rct",
    "dgp_panel",
    "dgp_observational",
    "dgp_cluster_rct",
    "dgp_bunching",
    "dgp_synth",
    "dgp_bartik",
]
