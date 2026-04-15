"""Shared plumbing for spatial regression models."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import formulaic
except ImportError as e:                     # pragma: no cover
    raise ImportError(
        "formulaic is required for spatial regression formulas. "
        "It is already a core StatsPAI dependency — "
        "please reinstall with `pip install statspai`."
    ) from e


def build_design_matrix(
    formula: str, data: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Parse `y ~ x1 + x2` into (y, X, column_names).

    Column names come from the formulaic model matrix header and may include
    "Intercept" (when the intercept is present).
    """
    y_mat, X_mat = formulaic.Formula(formula).get_model_matrix(data)
    y = np.asarray(y_mat).ravel().astype(float)
    X = np.asarray(X_mat, dtype=float)
    names = list(X_mat.columns)
    return y, X, names
