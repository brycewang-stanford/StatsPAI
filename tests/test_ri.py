"""
Tests for Randomization Inference.
"""

import pytest
import numpy as np
import pandas as pd

from statspai.inference.randomization import ri_test


@pytest.fixture
def rct_data():
    """Simulated RCT with true ATE = 2.0."""
    rng = np.random.default_rng(42)
    n = 200
    D = rng.binomial(1, 0.5, n)
    Y = 1 + 2.0 * D + rng.normal(0, 1, n)
    return pd.DataFrame({'y': Y, 'd': D,
                          'cluster': np.repeat(range(20), 10)})


@pytest.fixture
def null_data():
    """Data with zero treatment effect."""
    rng = np.random.default_rng(42)
    n = 200
    D = rng.binomial(1, 0.5, n)
    Y = rng.normal(0, 1, n)  # no effect
    return pd.DataFrame({'y': Y, 'd': D})


class TestRITest:
    def test_basic_run(self, rct_data):
        result = ri_test(rct_data, y='y', treat='d', n_perms=299, seed=42)
        assert 'p_value' in result
        assert 'observed' in result
        assert result['n_perms'] == 299

    def test_significant_effect(self, rct_data):
        """True ATE=2 should be significant."""
        result = ri_test(rct_data, y='y', treat='d', n_perms=999, seed=42)
        assert result['p_value'] < 0.05

    def test_null_not_significant(self):
        """Zero effect should NOT be significant (most of the time)."""
        rng = np.random.default_rng(123)
        n = 200
        D = rng.binomial(1, 0.5, n)
        Y = rng.normal(0, 1, n)
        df = pd.DataFrame({'y': Y, 'd': D})
        result = ri_test(df, y='y', treat='d', n_perms=999, seed=123)
        assert result['p_value'] > 0.01

    def test_t_stat(self, rct_data):
        result = ri_test(rct_data, y='y', treat='d', stat='t',
                         n_perms=299, seed=42)
        assert result['p_value'] < 0.05

    def test_ks_stat(self, rct_data):
        result = ri_test(rct_data, y='y', treat='d', stat='ks',
                         n_perms=299, seed=42)
        assert 'p_value' in result

    def test_custom_stat(self, rct_data):
        """Custom test statistic function."""
        result = ri_test(rct_data, y='y', treat='d',
                         stat=lambda y, d: np.median(y[d == 1]) - np.median(y[d == 0]),
                         n_perms=299, seed=42)
        assert result['p_value'] < 0.05

    def test_cluster_permutation(self, rct_data):
        """Cluster-level permutation."""
        result = ri_test(rct_data, y='y', treat='d',
                         cluster='cluster', n_perms=299, seed=42)
        assert 'p_value' in result

    def test_perm_distribution(self, rct_data):
        result = ri_test(rct_data, y='y', treat='d', n_perms=99, seed=42)
        assert len(result['perm_distribution']) == 99

    def test_one_sided(self, rct_data):
        result = ri_test(rct_data, y='y', treat='d', n_perms=499, seed=42)
        assert 'p_one_sided' in result
        # One-sided should be <= two-sided
        assert result['p_one_sided'] <= result['p_value'] + 1e-10


class TestIntegration:
    def test_import(self):
        import statspai as sp
        assert hasattr(sp, 'ri_test')


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
