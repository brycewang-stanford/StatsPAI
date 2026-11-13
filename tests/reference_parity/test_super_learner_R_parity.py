"""``sp.super_learner`` against R ``SuperLearner``.

Reference: SuperLearner 2.0.40 (GPL-3, run as a black box) with
``method = "method.CC_LS"``, the convex-combination least-squares
meta-learner, which is the problem ``sp.super_learner`` solves (weights on
the simplex minimising the cross-validated squared error). R's default
``method.NNLS`` rescales a non-negative least-squares fit and is a
different estimator; it is not compared.

Same data (``_fixtures/super_learner_data.csv``), same validation folds
(the CSV stores the folds ``sp.super_learner`` draws at ``n_folds=5,
random_state=42`` and R receives them as ``validRows``) and the same
deterministic learners on both sides: the mean, least squares on all
covariates, on ``x1`` alone and on the full quadratic expansion; for the
binary outcome the mean and two unpenalised logistic regressions. Three
fits: regression, regression with observation weights, classification
(``_fixtures/_generate_super_learner_R.R`` -> ``super_learner_R.json``).

Tolerances: cross-validated predictions 1e-9 absolute (observed 3e-14 for
least squares); cross-validated risks 1e-9 relative; weights 1e-6
absolute (quadprog on one side, SLSQP on the other); ensemble predictions
1e-6 absolute. With observation weights R reports the risk as
``mean(w * loss)`` and StatsPAI as ``sum(w * loss) / sum(w)``; the test
multiplies by the mean weight before comparing.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures

import statspai as sp
from statspai.tmle.super_learner import SuperLearner

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "super_learner_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "super_learner_data.csv")
X = DATA[["x1", "x2", "x3", "x4"]].to_numpy()
W = 0.5 + (np.arange(1, len(DATA) + 1) % 4) * 0.5


class _Mean(RegressorMixin, BaseEstimator):
    """R ``SL.mean``: the (weighted) mean of the outcome."""

    def fit(self, X, y, sample_weight=None):
        self.mean_ = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class _MeanClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0.0, 1.0])
        self.mean_ = float(np.average(y, weights=sample_weight))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.mean_)
        return np.column_stack([1 - p, p])


class _Linear(RegressorMixin, BaseEstimator):
    """Least squares on the first ``n_cols`` columns, expanded to ``degree``."""

    def __init__(self, n_cols=4, degree=1):
        self.n_cols = n_cols
        self.degree = degree

    def _design(self, X):
        Z = np.asarray(X)[:, : self.n_cols]
        if self.degree > 1:
            Z = PolynomialFeatures(self.degree, include_bias=False).fit_transform(Z)
        return Z

    def fit(self, X, y, sample_weight=None):
        self.model_ = LinearRegression().fit(
            self._design(X), y, sample_weight=sample_weight
        )
        return self

    def predict(self, X):
        return self.model_.predict(self._design(X))


class _Logit(ClassifierMixin, BaseEstimator):
    """Unpenalised logistic regression on the first ``n_cols`` columns.

    Newton's method to convergence, so the fit is the maximum-likelihood
    estimate to rounding, as R's ``glm`` at ``epsilon = 1e-14`` is.
    """

    def __init__(self, n_cols=4):
        self.n_cols = n_cols

    def _design(self, X):
        Z = np.asarray(X)[:, : self.n_cols]
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array([0.0, 1.0])
        A = self._design(X)
        w = np.ones(len(y)) if sample_weight is None else sample_weight
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
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self._design(X) @ self.coef_))
        return np.column_stack([1 - p, p])


def _reg_library():
    return [_Mean(), _Linear(4), _Linear(1), _Linear(4, degree=2)]


def _check(sl, ref, risk_scale=1.0):
    np.testing.assert_allclose(sl.cv_preds_, np.asarray(ref["Z"]), rtol=0, atol=1e-9)
    np.testing.assert_allclose(sl.cv_risks_ * risk_scale, ref["cv_risk"], rtol=1e-9)
    np.testing.assert_allclose(sl.weights_, ref["weights"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(sl.predict(X[:25]), ref["pred"], rtol=0, atol=1e-6)


def test_fixture_folds_are_the_ones_super_learner_draws():
    assert REF["versions"]["SuperLearner"] == "2.0.40"
    fold = np.zeros(len(DATA), dtype=int)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(X)):
        fold[te] = k + 1
    assert np.array_equal(fold, DATA["fold_reg"].to_numpy())
    fold = np.zeros(len(DATA), dtype=int)
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for k, (_, te) in enumerate(splitter.split(X, DATA["yb"].to_numpy())):
        fold[te] = k + 1
    assert np.array_equal(fold, DATA["fold_clf"].to_numpy())


def test_regression_ensemble_is_method_cc_ls():
    sl = sp.super_learner(X, DATA["y"].to_numpy(), library=_reg_library(), n_folds=5)
    _check(sl, REF["regression"])
    # the reference puts the weight on the quadratic model and a little on x1
    assert np.argmax(REF["regression"]["weights"]) == 3
    assert 0.01 < REF["regression"]["weights"][2] < 0.1


def test_weighted_regression_ensemble_is_method_cc_ls_with_obs_weights():
    sl = SuperLearner(library=_reg_library(), n_folds=5).fit(
        X, DATA["y"].to_numpy(), sample_weight=W
    )
    # Convention: R reports the risk as mean(w * loss); StatsPAI as the
    # weighted mean sum(w * loss) / sum(w). They differ by the mean weight
    # (1.25 here), the same factor for every learner, so the weights and the
    # ranking of the learners are unaffected.
    assert W.mean() == 1.25
    _check(sl, REF["regression_weighted"], risk_scale=W.mean())


def test_classification_ensemble_is_method_cc_ls():
    sl = sp.super_learner(
        X,
        DATA["yb"].to_numpy(),
        library=[_MeanClassifier(), _Logit(4), _Logit(2)],
        n_folds=5,
        task="classification",
    )
    _check(sl, REF["classification"])
