"""
Missing data handling and multiple imputation.

Provides MICE (Multiple Imputation by Chained Equations),
EM imputation, and analysis tools for multiply-imputed data.
"""

from ._mi_test import mi_test
from .mice import MICEResult, mi_estimate, mice

__all__ = ["mice", "MICEResult", "mi_estimate", "mi_test"]
