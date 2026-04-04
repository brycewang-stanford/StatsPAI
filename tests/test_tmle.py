"""
Tests for TMLE and Super Learner.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from statspai.tmle import tmle, TMLE, super_learner, SuperLearner
from statspai.core.results import CausalResult


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def constant_effect_data():
    """
    Constant ATE = 3.0.
    Y = 3.0*D + sin(X1) + X2^2 + eps
    P(D=1|X) = logistic(0.5*X1 + X2)
    """
    rng = np.random.default_rng(42)
    n = 2000

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    eps = rng.normal(0, 0.5, n)

    logit = 0.5 * X1 + X2
    prob = 1 / (1 + np.exp(-logit))
    D = rng.binomial(1, prob, n).astype(float)

    Y = 3.0 * D + np.sin(X1) + X2 ** 2 + eps

    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


@pytest.fixture
def binary_outcome_data():
    """
    Binary outcome with known ATE.
    P(Y=1|D,X) = logistic(1.5*D + X1)
    """
    rng = np.random.default_rng(42)
    n = 2000

    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)

    logit_d = 0.3 * X1
    D = rng.binomial(1, 1 / (1 + np.exp(-logit_d)), n).astype(float)

    logit_y = 1.5 * D + X1
    Y = rng.binomial(1, 1 / (1 + np.exp(-logit_y)), n).astype(float)

    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


@pytest.fixture
def small_data():
    rng = np.random.default_rng(99)
    n = 300
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 2.0 * D + X1 + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


# ======================================================================
# Super Learner Tests
# ======================================================================

class TestSuperLearner:
    """Tests for Super Learner ensemble."""

    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.3, 200)

        sl = super_learner(X, y, n_folds=3)
        preds = sl.predict(X)
        assert len(preds) == 200

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.3, 200)

        sl = super_learner(X, y, n_folds=3)
        assert abs(sl.weights_.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.3, 200)

        sl = super_learner(X, y, n_folds=3)
        assert np.all(sl.weights_ >= 0)

    def test_custom_library(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.3, 200)

        lib = [LinearRegression(), RandomForestRegressor(n_estimators=10,
                                                          random_state=42)]
        sl = super_learner(X, y, library=lib, n_folds=3)
        assert len(sl.weights_) == 2

    def test_classification(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        y = (X[:, 0] + rng.normal(0, 0.5, 200) > 0).astype(float)

        sl = super_learner(X, y, task='classification', n_folds=3)
        probs = sl.predict_proba(X)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_summary(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (200, 3))
        y = X[:, 0] + rng.normal(0, 0.3, 200)

        sl = super_learner(X, y, n_folds=3)
        s = sl.summary()
        assert 'Weight' in s
        assert 'CV Risk' in s


# ======================================================================
# TMLE Tests
# ======================================================================

class TestTMLE:
    """Tests for TMLE estimator."""

    def test_returns_causal_result(self, small_data):
        result = tmle(small_data, y='y', treat='d',
                      covariates=['x1', 'x2'], n_folds=3)
        assert isinstance(result, CausalResult)
        assert result.method == 'TMLE (van der Laan & Rose 2011)'

    def test_has_required_fields(self, small_data):
        result = tmle(small_data, y='y', treat='d',
                      covariates=['x1', 'x2'], n_folds=3)
        assert result.estimate is not None
        assert result.se > 0
        assert 0 <= result.pvalue <= 1
        assert len(result.ci) == 2
        assert result.ci[0] < result.ci[1]

    def test_eif_standard_error(self, small_data):
        """SE should come from efficient influence function."""
        result = tmle(small_data, y='y', treat='d',
                      covariates=['x1', 'x2'], n_folds=3)
        assert result.model_info['se_method'] == 'efficient_influence_function'

    def test_constant_effect_recovery(self, constant_effect_data):
        """ATE should be close to 3.0."""
        result = tmle(constant_effect_data, y='y', treat='d',
                      covariates=['x1', 'x2'], n_folds=5)
        assert abs(result.estimate - 3.0) < 1.5, \
            f"TMLE ATE={result.estimate:.2f}, expected ~3.0"

    def test_binary_outcome(self, binary_outcome_data):
        """Should work with binary outcomes."""
        result = tmle(binary_outcome_data, y='y', treat='d',
                      covariates=['x1', 'x2'], n_folds=3)
        assert isinstance(result, CausalResult)
        assert result.model_info['outcome_type'] == 'binary'

    def test_att_estimand(self, small_data):
        """Should support ATT estimation."""
        result = tmle(small_data, y='y', treat='d',
                      covariates=['x1', 'x2'],
                      estimand='ATT', n_folds=3)
        assert result.estimand == 'ATT'

    def test_custom_libraries(self, small_data):
        """Should accept custom learner libraries."""
        outcome_lib = [LinearRegression(),
                       RandomForestRegressor(n_estimators=20, random_state=42)]
        prop_lib = [LogisticRegression(max_iter=1000)]

        result = tmle(small_data, y='y', treat='d',
                      covariates=['x1', 'x2'],
                      outcome_library=outcome_lib,
                      propensity_library=prop_lib,
                      n_folds=3)
        assert isinstance(result, CausalResult)

    def test_model_info_has_sl_weights(self, small_data):
        result = tmle(small_data, y='y', treat='d',
                      covariates=['x1', 'x2'], n_folds=3)
        assert 'sl_outcome_weights' in result.model_info
        assert 'sl_propensity_weights' in result.model_info

    def test_summary_renders(self, small_data):
        result = tmle(small_data, y='y', treat='d',
                      covariates=['x1', 'x2'], n_folds=3)
        summary = result.summary()
        assert 'TMLE' in summary

    def test_citation(self, small_data):
        result = tmle(small_data, y='y', treat='d',
                      covariates=['x1', 'x2'], n_folds=3)
        bib = result.cite()
        assert 'vanderlaan' in bib

    def test_missing_column_raises(self, small_data):
        with pytest.raises(ValueError, match="Columns not found"):
            tmle(small_data, y='y', treat='d',
                 covariates=['x1', 'nonexistent'], n_folds=3)

    def test_non_binary_treatment_raises(self):
        df = pd.DataFrame({
            'y': [1, 2, 3, 4], 'd': [0, 1, 2, 0],
            'x1': [1, 2, 3, 4]
        })
        with pytest.raises(ValueError, match="binary"):
            tmle(df, y='y', treat='d', covariates=['x1'], n_folds=2)

    def test_nan_handling(self):
        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame({
            'y': rng.normal(0, 1, n),
            'd': rng.binomial(1, 0.5, n).astype(float),
            'x1': rng.normal(0, 1, n),
        })
        df.loc[0, 'x1'] = np.nan
        result = tmle(df, y='y', treat='d', covariates=['x1'], n_folds=3)
        assert result.n_obs == n - 1


# ======================================================================
# Import Tests
# ======================================================================

class TestImports:
    def test_import_from_statspai(self):
        import statspai as sp
        assert hasattr(sp, 'tmle')
        assert hasattr(sp, 'TMLE')
        assert hasattr(sp, 'super_learner')
        assert hasattr(sp, 'SuperLearner')
