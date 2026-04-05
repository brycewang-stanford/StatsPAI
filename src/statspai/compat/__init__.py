"""
Ecosystem compatibility layer.

Provides sklearn-compatible wrappers so that StatsPAI estimators can
participate in ``sklearn.pipeline.Pipeline``, cross-validation,
``GridSearchCV``, etc.

>>> from statspai.compat import SklearnOLS, SklearnDID, SklearnDML
>>> from sklearn.model_selection import cross_val_score
>>> scores = cross_val_score(SklearnOLS(robust='hc1'), X, y, cv=5)
"""

from .sklearn import (
    SklearnOLS,
    SklearnIV,
    SklearnDML,
    SklearnCausalForest,
)

__all__ = [
    "SklearnOLS",
    "SklearnIV",
    "SklearnDML",
    "SklearnCausalForest",
]
