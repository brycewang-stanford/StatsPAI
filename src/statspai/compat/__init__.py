"""
Ecosystem compatibility layer.

Provides sklearn-compatible wrappers so that StatsPAI estimators can
participate in ``sklearn.pipeline.Pipeline``, cross-validation,
``GridSearchCV``, etc.

>>> import numpy as np
>>> from statspai.compat import SklearnOLS, SklearnIV, SklearnDML
>>> from sklearn.model_selection import cross_val_score
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(200, 2))
>>> y = 1.0 + X @ np.array([0.5, -0.3]) + rng.normal(size=200)
>>> scores = cross_val_score(SklearnOLS(robust='hc1'), X, y, cv=5)
>>> scores.shape
(5,)
"""

from .sklearn import SklearnCausalForest, SklearnDML, SklearnIV, SklearnOLS

__all__ = [
    "SklearnOLS",
    "SklearnIV",
    "SklearnDML",
    "SklearnCausalForest",
]
