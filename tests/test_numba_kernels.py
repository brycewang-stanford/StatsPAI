"""Tests for Numba-accelerated computational kernels."""

import numpy as np
import pytest


def _make_ols_data(n=200, k=4, seed=42):
    rng = np.random.RandomState(seed)
    X = np.column_stack([np.ones(n), rng.randn(n, k - 1)])
    beta_true = rng.randn(k)
    beta_true[:4] = [1.0, 2.0, -0.5, 0.3][:k]
    y = X @ beta_true + rng.randn(n) * 0.5
    return X, y, beta_true


class TestOLSFit:
    def test_ols_fit_recovers_params(self):
        from statspai.core._numba_kernels import ols_fit
        X, y, beta_true = _make_ols_data()
        params, fitted, residuals = ols_fit(X, y)
        np.testing.assert_allclose(params, beta_true, atol=0.3)
        np.testing.assert_allclose(fitted + residuals, y, atol=1e-10)

    def test_ols_fit_shapes(self):
        from statspai.core._numba_kernels import ols_fit
        X, y, _ = _make_ols_data(n=100, k=3)
        params, fitted, residuals = ols_fit(X, y)
        assert params.shape == (3,)
        assert fitted.shape == (100,)
        assert residuals.shape == (100,)


class TestSandwichHC:
    def test_hc1_positive_diagonal(self):
        from statspai.core._numba_kernels import ols_fit, sandwich_hc
        X, y, _ = _make_ols_data()
        params, _, residuals = ols_fit(X, y)
        XtX_inv = np.linalg.inv(X.T @ X)
        vcov = sandwich_hc(X, residuals, XtX_inv, "hc1")
        assert np.all(np.diag(vcov) > 0)

    def test_hc_types(self):
        from statspai.core._numba_kernels import ols_fit, sandwich_hc
        X, y, _ = _make_ols_data()
        _, _, residuals = ols_fit(X, y)
        XtX_inv = np.linalg.inv(X.T @ X)
        for hc in ["hc0", "hc1", "hc2", "hc3"]:
            vcov = sandwich_hc(X, residuals, XtX_inv, hc)
            assert vcov.shape == (4, 4)
            assert np.allclose(vcov, vcov.T)  # symmetric


class TestClusterMeat:
    def test_cluster_meat_basic(self):
        from statspai.core._numba_kernels import cluster_meat
        rng = np.random.RandomState(0)
        n, k = 200, 3
        X = np.column_stack([np.ones(n), rng.randn(n, k - 1)])
        residuals = rng.randn(n)
        cluster_ids = np.repeat(np.arange(20), 10)
        meat = cluster_meat(X, residuals, cluster_ids)
        assert meat.shape == (k, k)
        assert np.allclose(meat, meat.T)

    def test_single_obs_clusters(self):
        from statspai.core._numba_kernels import cluster_meat
        rng = np.random.RandomState(1)
        n, k = 50, 2
        X = np.column_stack([np.ones(n), rng.randn(n, k - 1)])
        residuals = rng.randn(n)
        cluster_ids = np.arange(n)  # every obs is its own cluster
        meat = cluster_meat(X, residuals, cluster_ids)
        assert meat.shape == (k, k)


class TestHACMeat:
    def test_hac_meat_symmetric(self):
        from statspai.core._numba_kernels import hac_meat
        rng = np.random.RandomState(2)
        n, k = 300, 3
        X = np.column_stack([np.ones(n), rng.randn(n, k - 1)])
        residuals = rng.randn(n)
        meat = hac_meat(X, residuals)
        assert meat.shape == (k, k)
        assert np.allclose(meat, meat.T)


class TestOLSRegressionIntegration:
    """Test that the OLS regress() function still works after kernel integration."""

    def test_regress_basic(self):
        import pandas as pd
        from statspai import regress
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "y": rng.randn(n),
            "x1": rng.randn(n),
            "x2": rng.randn(n),
        })
        result = regress("y ~ x1 + x2", data=df)
        assert hasattr(result, "params")
        assert len(result.params) == 3  # const + x1 + x2

    def test_regress_hc1(self):
        import pandas as pd
        from statspai import regress
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "y": rng.randn(n),
            "x1": rng.randn(n),
        })
        result = regress("y ~ x1", data=df, robust="hc1")
        assert np.all(result.std_errors > 0)

    def test_regress_cluster(self):
        import pandas as pd
        from statspai import regress
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "y": rng.randn(n),
            "x1": rng.randn(n),
            "cluster": np.repeat(np.arange(20), 10),
        })
        result = regress("y ~ x1", data=df, cluster="cluster")
        assert np.all(result.std_errors > 0)
