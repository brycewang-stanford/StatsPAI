"""
Tests for Round 3 features:
- Spatial econometrics (SAR, SEM, SDM)
- General bootstrap inference
- Registry updates
"""

import numpy as np
import pandas as pd
import pytest

# ====================================================================== #
#  Spatial Econometrics
# ====================================================================== #


def _make_spatial_data(n=50, seed=42):
    """Generate data with spatial dependence."""
    rng = np.random.RandomState(seed)

    # Simple contiguity matrix (ring topology)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = 1
        W[i, (i - 1) % n] = 1

    x1 = rng.randn(n)
    x2 = rng.randn(n)
    X = np.column_stack([np.ones(n), x1, x2])

    # Generate spatially correlated data: Y = 0.4*WY + 1 + 2*x1 - 0.5*x2 + ε
    rho_true = 0.4
    beta_true = np.array([1.0, 2.0, -0.5])
    A = np.eye(n) - rho_true * (W / W.sum(axis=1, keepdims=True))
    eps = rng.randn(n) * 0.5
    y = np.linalg.solve(A, X @ beta_true + eps)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return W, df


class TestSAR:
    def test_sar_basic(self):
        from statspai import sar

        W, df = _make_spatial_data()
        result = sar(W, data=df, formula="y ~ x1 + x2")
        assert "rho" in result.params.index
        assert result.params["rho"] != 0
        assert np.all(result.std_errors > 0)

    def test_sar_rho_range(self):
        from statspai import sar

        W, df = _make_spatial_data()
        result = sar(W, data=df, formula="y ~ x1 + x2")
        rho = result.params["rho"]
        assert -1 < rho < 1

    def test_sar_summary(self):
        from statspai import sar

        W, df = _make_spatial_data()
        result = sar(W, data=df, formula="y ~ x1 + x2")
        summary = result.summary()
        assert "rho" in summary
        assert "SAR" in summary or "Spatial" in summary


class TestSEM:
    def test_sem_basic(self):
        from statspai import sem

        W, df = _make_spatial_data()
        result = sem(W, data=df, formula="y ~ x1 + x2")
        assert "lambda" in result.params.index
        assert np.all(result.std_errors > 0)

    def test_sem_lambda_range(self):
        from statspai import sem

        W, df = _make_spatial_data()
        result = sem(W, data=df, formula="y ~ x1 + x2")
        lam = result.params["lambda"]
        assert -1 < lam < 1


class TestSDM:
    def test_sdm_basic(self):
        from statspai import sdm

        W, df = _make_spatial_data()
        result = sdm(W, data=df, formula="y ~ x1 + x2")
        assert "rho" in result.params.index
        assert "W_x1" in result.params.index
        assert "W_x2" in result.params.index

    def test_sdm_effects(self):
        from statspai import sdm

        W, df = _make_spatial_data()
        result = sdm(W, data=df, formula="y ~ x1 + x2")
        assert "Direct effects" in result.diagnostics
        assert "Indirect effects" in result.diagnostics
        assert "Total effects" in result.diagnostics

    def test_sdm_w_shape_mismatch(self):
        from statspai.spatial import SpatialModel

        W = np.eye(10)
        df = pd.DataFrame({"y": np.ones(5), "x": np.ones(5)})
        with pytest.raises(ValueError, match="must be"):
            SpatialModel(W, df, "y ~ x", model_type="sar")


# ====================================================================== #
#  Bootstrap
# ====================================================================== #


class TestBootstrap:
    def test_bootstrap_mean(self):
        from statspai import bootstrap

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"y": rng.randn(200) + 5})
        result = bootstrap(df, lambda d: d["y"].mean(), n_boot=200, seed=42)
        assert abs(result.estimate - 5.0) < 0.5
        assert result.se > 0
        assert result.ci_lower < result.estimate < result.ci_upper

    def test_bootstrap_cluster(self):
        from statspai import bootstrap

        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame(
            {
                "y": rng.randn(n) + 3,
                "cluster": np.repeat(np.arange(20), 10),
            }
        )
        result = bootstrap(
            df, lambda d: d["y"].mean(), n_boot=100, cluster="cluster", seed=42
        )
        assert result.se > 0

    def test_bootstrap_normal_ci(self):
        from statspai import bootstrap

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"y": rng.randn(100)})
        result = bootstrap(
            df, lambda d: d["y"].mean(), n_boot=100, ci_method="normal", seed=42
        )
        assert result.ci_method == "normal"
        assert result.ci_lower < result.ci_upper

    def test_bootstrap_bca_ci(self):
        from statspai import bootstrap

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"y": rng.randn(50) + 2})
        result = bootstrap(
            df, lambda d: d["y"].mean(), n_boot=100, ci_method="bca", seed=42
        )
        assert result.ci_method == "bca"
        assert result.ci_lower < result.ci_upper

    def test_bca_cluster_jackknife_deletes_clusters(self):
        """BCa under cluster= must leave-one-cluster-out (matching the
        cluster bootstrap), not delete rows."""
        from statspai import bootstrap

        rng = np.random.RandomState(0)
        n = 120
        df = pd.DataFrame({"y": rng.randn(n) + 2, "cl": rng.randint(0, 12, n)})
        res = bootstrap(
            df,
            lambda d: float(d["y"].mean()),
            n_boot=300,
            ci_method="bca",
            cluster="cl",
            seed=1,
        )
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_bca_warns_and_subsamples_large_jackknife(self):
        """>200 units: the acceleration jackknife is capped with a loud
        warning rather than silently using only the first 200 rows."""
        from statspai import bootstrap

        rng = np.random.RandomState(1)
        df = pd.DataFrame({"y": rng.randn(500)})
        with pytest.warns(RuntimeWarning, match="capped for cost"):
            bootstrap(
                df,
                lambda d: float(d["y"].mean()),
                n_boot=200,
                ci_method="bca",
                seed=2,
            )

    def test_bca_warns_on_failed_jackknife_points(self):
        """Failed jackknife replicates are dropped (not imputed with
        theta_hat, which biases the acceleration toward 0) and reported."""
        from statspai import bootstrap

        rng = np.random.RandomState(2)
        df = pd.DataFrame({"y": rng.randn(50)})
        calls = {"n": 0}

        def flaky(d):
            calls["n"] += 1
            if len(d) == 49 and calls["n"] % 5 == 0:
                raise ValueError("boom")
            return float(d["y"].mean())

        with pytest.warns(RuntimeWarning, match="jackknife replicate"):
            res = bootstrap(df, flaky, n_boot=200, ci_method="bca", seed=3)
        assert np.isfinite(res.ci_lower) and np.isfinite(res.ci_upper)

    def test_bootstrap_summary(self):
        from statspai import bootstrap

        rng = np.random.RandomState(42)
        df = pd.DataFrame({"y": rng.randn(100)})
        result = bootstrap(df, lambda d: d["y"].mean(), n_boot=50, seed=42)
        text = result.summary()
        assert "Bootstrap" in text
        assert "Estimate" in text


# ====================================================================== #
#  Registry updates
# ====================================================================== #


class TestRegistryRound3:
    def test_new_functions_registered(self):
        from statspai import list_functions

        funcs = list_functions()
        for name in [
            "ipw",
            "dag",
            "event_study",
            "augsynth",
            "sar",
            "sem",
            "sdm",
            "bootstrap",
            "diagnose_result",
        ]:
            assert name in funcs, f"{name} not found in registry"

    def test_spatial_category(self):
        from statspai import list_functions

        spatial = list_functions(category="spatial")
        assert "sar" in spatial
        assert "sem" in spatial
        assert "sdm" in spatial

    def test_search_spatial(self):
        from statspai import search_functions

        results = search_functions("spatial")
        names = [r["name"] for r in results]
        assert any("sar" in n or "sem" in n or "sdm" in n for n in names)

    def test_schema_count_increased(self):
        from statspai import all_schemas

        schemas = all_schemas()
        # Should have 20+ from round 1 plus ~10 new ones
        assert len(schemas) >= 25
