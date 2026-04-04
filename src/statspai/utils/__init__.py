"""
Utility functions for StatsPAI.

Provides Stata-style data manipulation tools:
- Variable labels (label_var, get_label)
- Pairwise correlation with stars (pwcorr)
- Winsorization (winsor)
"""

from .labels import label_var, label_vars, get_label, get_labels, describe
from .data_tools import pwcorr, winsor
from .io import read_data

__all__ = [
    'label_var',
    'label_vars',
    'get_label',
    'get_labels',
    'describe',
    'pwcorr',
    'winsor',
    'read_data',
]
