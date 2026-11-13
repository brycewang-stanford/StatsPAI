"""Learners shared by the meta-learner reference script and its test.

The reference (``econml``) and StatsPAI run in different environments with
different scikit-learn versions, and scikit-learn's logistic regression
stops at an optimiser tolerance that is not the same across versions. The
propensity model here is the maximum-likelihood logistic fit by Newton's
method, converged to rounding, so both sides hold the same propensities.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NewtonLogit(ClassifierMixin, BaseEstimator):
    """Unpenalised logistic regression with an intercept."""

    def fit(self, X, y, sample_weight=None):
        A = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        beta = np.zeros(A.shape[1])
        for _ in range(100):
            p = 1 / (1 + np.exp(-A @ beta))
            step = np.linalg.solve(
                A.T @ (A * (w * p * (1 - p))[:, None]), A.T @ (w * (y - p))
            )
            beta = beta + step
            if np.max(np.abs(step)) < 1e-13:
                break
        self.coef_ = beta
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        A = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        p = 1 / (1 + np.exp(-A @ self.coef_))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
