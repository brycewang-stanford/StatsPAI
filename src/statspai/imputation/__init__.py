"""
Missing data handling and multiple imputation.

Provides MICE (Multiple Imputation by Chained Equations),
EM imputation, and analysis tools for multiply-imputed data.
"""

from .mice import mice, MICEResult, mi_estimate

__all__ = ["mice", "MICEResult", "mi_estimate"]
