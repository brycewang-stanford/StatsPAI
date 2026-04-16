"""
Utility functions for StatsPAI.

Provides Stata-style data manipulation tools:
- Variable labels (label_var, get_label)
- Pairwise correlation with stars (pwcorr)
- Winsorization (winsor)
"""

from .labels import label_var, label_vars, get_label, get_labels, describe
from .data_tools import pwcorr, winsor
from .egen import (
    rowmean, rowtotal, rowmax, rowmin, rowsd, rowcount, rank,
    outlier_indicator,
)
from .io import read_data
from .dgp import (
    dgp_did,
    dgp_rd,
    dgp_rd_kink,
    dgp_rd_multi,
    dgp_rd_hte,
    dgp_rd_2d,
    dgp_rdit,
    dgp_iv,
    dgp_rct,
    dgp_panel,
    dgp_observational,
    dgp_cluster_rct,
    dgp_bunching,
    dgp_synth,
    dgp_bartik,
)

__all__ = [
    'label_var',
    'label_vars',
    'get_label',
    'get_labels',
    'describe',
    'pwcorr',
    'winsor',
    'rowmean',
    'rowtotal',
    'rowmax',
    'rowmin',
    'rowsd',
    'rowcount',
    'rank',
    'outlier_indicator',
    'read_data',
    # Data Generating Processes
    'dgp_did',
    'dgp_rd',
    'dgp_rd_kink',
    'dgp_rd_multi',
    'dgp_rd_hte',
    'dgp_rd_2d',
    'dgp_rdit',
    'dgp_iv',
    'dgp_rct',
    'dgp_panel',
    'dgp_observational',
    'dgp_cluster_rct',
    'dgp_bunching',
    'dgp_synth',
    'dgp_bartik',
]
