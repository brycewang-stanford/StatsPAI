"""
Tests for Policy Learning (PolicyTree).
"""

import pytest
import numpy as np
import pandas as pd

from statspai.policy_learning import policy_tree, PolicyTree, policy_value


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def positive_effect_data():
    """Everyone benefits from treatment (ATE = 2.0)."""
    rng = np.random.default_rng(42)
    n = 1000
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    Y = 2.0 * D + X1 + rng.normal(0, 0.5, n)
    return pd.DataFrame({'y': Y, 'd': D, 'x1': X1, 'x2': X2})


@pytest.fixture
def heterogeneous_policy_data():
    """
    Only units with X1 > 0 benefit from treatment.
    CATE(X) = 3*I(X1 > 0) - 1
    """
    rng = np.random.default_rng(42)
    n = 2000
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = rng.binomial(1, 0.5, n).astype(float)
    tau = 3.0 * (X1 > 0).astype(float) - 1.0
    Y = tau * D + X2 + rng.normal(0, 0.3, n)
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
# PolicyTree Tests
# ======================================================================

class TestPolicyTree:

    def test_returns_dict(self, small_data):
        result = policy_tree(small_data, y='y', treat='d',
                             covariates=['x1', 'x2'],
                             max_depth=1, n_folds=3)
        assert isinstance(result, dict)
        assert 'policy' in result
        assert 'value_policy' in result
        assert 'rules' in result
        assert 'fraction_treated' in result

    def test_binary_policy(self, small_data):
        result = policy_tree(small_data, y='y', treat='d',
                             covariates=['x1', 'x2'],
                             max_depth=1, n_folds=3)
        policy = result['policy']
        assert set(np.unique(policy)) <= {0, 1}

    def test_positive_effect_treats_all(self, positive_effect_data):
        """When everyone benefits, policy should treat most."""
        result = policy_tree(positive_effect_data, y='y', treat='d',
                             covariates=['x1', 'x2'],
                             max_depth=2, n_folds=3)
        assert result['fraction_treated'] > 0.5

    def test_heterogeneous_policy(self, heterogeneous_policy_data):
        """Policy should roughly split on X1 > 0."""
        result = policy_tree(heterogeneous_policy_data, y='y', treat='d',
                             covariates=['x1', 'x2'],
                             max_depth=2, n_folds=5)
        # Some fraction should not be treated
        assert result['fraction_treated'] < 0.95
        assert result['fraction_treated'] > 0.1

    def test_rules_readable(self, small_data):
        result = policy_tree(small_data, y='y', treat='d',
                             covariates=['x1', 'x2'],
                             max_depth=2, n_folds=3)
        rules = result['rules']
        assert 'TREAT' in rules or "DON'T TREAT" in rules

    def test_predict_new_data(self, small_data):
        result = policy_tree(small_data, y='y', treat='d',
                             covariates=['x1', 'x2'],
                             max_depth=1, n_folds=3)
        tree = result['tree']
        X_new = np.random.randn(10, 2)
        policy = tree.predict(X_new)
        assert len(policy) == 10
        assert set(np.unique(policy)) <= {0, 1}

    def test_depth_1(self, small_data):
        result = policy_tree(small_data, y='y', treat='d',
                             covariates=['x1', 'x2'],
                             max_depth=1, n_folds=3)
        assert isinstance(result, dict)

    def test_depth_3(self, small_data):
        result = policy_tree(small_data, y='y', treat='d',
                             covariates=['x1', 'x2'],
                             max_depth=3, n_folds=3,
                             min_leaf_size=10)
        assert isinstance(result, dict)

    def test_policy_covariates(self, small_data):
        """Policy covariates can differ from nuisance covariates."""
        result = policy_tree(small_data, y='y', treat='d',
                             covariates=['x1', 'x2'],
                             policy_covariates=['x1'],
                             max_depth=1, n_folds=3)
        assert isinstance(result, dict)

    def test_missing_column_raises(self, small_data):
        with pytest.raises(ValueError, match="Columns not found"):
            policy_tree(small_data, y='y', treat='d',
                        covariates=['x1', 'nonexistent'],
                        max_depth=1, n_folds=3)

    def test_unfitted_predict_raises(self):
        df = pd.DataFrame({
            'y': [1, 2], 'd': [0, 1],
            'x1': [1, 2]
        })
        est = PolicyTree(data=df, y='y', treat='d', covariates=['x1'])
        with pytest.raises(ValueError, match="fitted"):
            est.predict(np.array([[1]]))


# ======================================================================
# policy_value Tests
# ======================================================================

class TestPolicyValue:

    def test_treat_all_positive(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        policy = np.array([1, 1, 1, 1])
        val = policy_value(scores, policy)
        assert abs(val - 2.5) < 1e-10

    def test_treat_none(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        policy = np.array([0, 0, 0, 0])
        val = policy_value(scores, policy)
        assert val == 0.0

    def test_selective(self):
        scores = np.array([1.0, -1.0, 2.0, -2.0])
        policy = np.array([1, 0, 1, 0])
        val = policy_value(scores, policy)
        # (1*1 + (-1)*0 + 2*1 + (-2)*0) / 4 = 3/4 = 0.75
        assert abs(val - 0.75) < 1e-10


# ======================================================================
# Import Tests
# ======================================================================

class TestImports:
    def test_import_from_statspai(self):
        import statspai as sp
        assert hasattr(sp, 'policy_tree')
        assert hasattr(sp, 'PolicyTree')
        assert hasattr(sp, 'policy_value')
