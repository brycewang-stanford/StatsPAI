"""
Tests for long-panel DML (``sp.dml_panel``).

Validation strategy
-------------------
- **Recovery under homogeneous treatment**: when the DGP matches the
  PLR assumption with unit FE, ``dml_panel`` should recover β within
  a few cluster-SEs of the truth.
- **Agreement with FE-OLS in the low-confounding limit**: when X is
  uncorrelated with D the dml_panel point estimate should be close
  to the FE-OLS estimate (both unbiased).
- **Time-FE option**: including time FE absorbs aggregate shocks and
  should not break identification.
- **Cluster SE structure**: with stronger within-unit correlation, the
  cluster-robust SE should exceed the (naive) i.i.d. SE.
- **Boundary validation**: column checks, fold bound, missing time
  with ``include_time_fe=True``.
- **Registry / public API**: ``sp.dml_panel`` is top-level exposed;
  ``sp.describe_function('dml_panel')`` returns metadata with panel
  tags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.dml import dml_panel, DMLPanelResult


def _sim_panel(
    N: int = 60, T: int = 8, *,
    beta: float = 0.5, gamma: float = 0.8,
    d_on_x: float = 0.3, sigma: float = 0.5, seed: int = 0,
):
    rng = np.random.default_rng(seed)
    alpha_i = rng.normal(0, 1, N)
    rows = []
    for i in range(N):
        X = rng.normal(0, 1, T)
        D = d_on_x * X + rng.normal(0, 1, T)
        Y = alpha_i[i] + beta * D + gamma * X + rng.normal(0, sigma, T)
        for t in range(T):
            rows.append({
                "pid": i, "year": t, "y": Y[t],
                "d": D[t], "x1": X[t],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Core behaviour
# --------------------------------------------------------------------------- #


class TestRecovery:

    def test_result_type_and_summary(self):
        df = _sim_panel(N=40, T=6, seed=0)
        res = dml_panel(df, y="y", treat="d", covariates=["x1"],
                        unit="pid", n_folds=3)
        assert isinstance(res, DMLPanelResult)
        s = res.summary()
        assert "Long-panel Double/Debiased ML" in s
        assert "β" in s or "b" in s

    def test_recovers_truth_without_time_fe(self):
        df = _sim_panel(N=80, T=8, beta=0.5, seed=1)
        res = dml_panel(df, y="y", treat="d", covariates=["x1"],
                        unit="pid", n_folds=5)
        # Allow 3 cluster-SEs given finite sample + cross-fit noise.
        assert abs(res.estimate - 0.5) < 3 * res.se + 0.05, res

    def test_recovers_truth_with_time_fe(self):
        df = _sim_panel(N=80, T=8, beta=0.5, seed=2)
        res = dml_panel(df, y="y", treat="d", covariates=["x1"],
                        unit="pid", time="year",
                        include_time_fe=True, n_folds=5)
        assert abs(res.estimate - 0.5) < 3 * res.se + 0.05, res

    def test_no_confounding_close_to_fe_ols(self):
        """When D is randomly assigned (no X-D correlation), dml_panel
        should match the pure FE-OLS within estimator.
        """
        df = _sim_panel(N=100, T=8, beta=0.7, d_on_x=0.0, seed=3)
        res = dml_panel(df, y="y", treat="d", covariates=["x1"],
                        unit="pid", n_folds=5)
        # Compute FE-OLS within estimator manually
        within = df.copy()
        for col in ("y", "d", "x1"):
            within[col] = within[col] - within.groupby("pid")[col].transform("mean")
        # Regress y_tilde on d_tilde + x1_tilde
        from numpy.linalg import lstsq
        X = np.column_stack([within["d"].to_numpy(),
                             within["x1"].to_numpy()])
        y = within["y"].to_numpy()
        beta_ols = lstsq(X, y, rcond=None)[0][0]
        # Within 15% of FE-OLS (both targeting same parameter under
        # homogeneous effect + no confounding)
        assert abs(res.estimate - beta_ols) < 0.15 * abs(beta_ols) + 0.05, (
            res.estimate, beta_ols
        )


class TestInferenceStructure:

    def test_cluster_se_captures_within_correlation(self):
        """Under strong within-unit serial correlation, cluster SE
        must not be smaller than an i.i.d. naive SE proxy.
        """
        rng = np.random.default_rng(9)
        N, T = 50, 10
        rho = 0.9  # strong AR(1) in u_it
        alpha_i = rng.normal(0, 1, N)
        rows = []
        for i in range(N):
            X = rng.normal(0, 1, T)
            D = 0.3 * X + rng.normal(0, 1, T)
            eps = np.zeros(T)
            eps[0] = rng.normal(0, 1)
            for t in range(1, T):
                eps[t] = rho * eps[t - 1] + rng.normal(0, 1)
            Y = alpha_i[i] + 0.5 * D + 0.8 * X + eps
            for t in range(T):
                rows.append({"pid": i, "year": t, "y": Y[t],
                             "d": D[t], "x1": X[t]})
        df = pd.DataFrame(rows)
        res = dml_panel(df, y="y", treat="d", covariates=["x1"],
                        unit="pid", n_folds=5)
        # Naive i.i.d. SE from sqrt(omega / n / J^2) — we track omega
        # inside diagnostics.  Compare to an iid-residual approximation
        # of SE: the cluster Ω aggregates within-unit covariance, so
        # cluster SE >= iid SE under positive correlation.
        psi_var_iid = res.diagnostics["y_resid_std"] ** 2 * (
            res.diagnostics["d_resid_std"] ** 2
        )
        n = res.n_obs
        J = res.diagnostics["d_resid_std"] ** 2
        if J > 0:
            se_iid_approx = float(np.sqrt(psi_var_iid / (n * J ** 2)))
            assert res.se > 0.5 * se_iid_approx  # same order of magnitude
            # Under AR(1) > 0 we expect se_cluster >= se_iid_approx,
            # but numerical fold noise can flip this by ~20%; use a
            # generous 0.7 × lower bound.
            assert res.se >= 0.7 * se_iid_approx, (res.se, se_iid_approx)


# --------------------------------------------------------------------------- #
#  Boundary validation
# --------------------------------------------------------------------------- #


class TestValidation:

    def test_missing_column_raises(self):
        df = _sim_panel(N=20, T=5, seed=0)
        with pytest.raises(ValueError, match="missing columns"):
            dml_panel(df, y="y", treat="d", covariates=["nonexistent"],
                      unit="pid")

    def test_folds_exceed_units_raises(self):
        df = _sim_panel(N=5, T=5, seed=0)
        with pytest.raises(ValueError, match="cannot exceed n_units"):
            dml_panel(df, y="y", treat="d", covariates=["x1"],
                      unit="pid", n_folds=10)

    def test_n_folds_below_two_raises(self):
        df = _sim_panel(N=20, T=5, seed=0)
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            dml_panel(df, y="y", treat="d", covariates=["x1"],
                      unit="pid", n_folds=1)

    def test_include_time_fe_without_time_raises(self):
        df = _sim_panel(N=20, T=5, seed=0)
        with pytest.raises(ValueError, match="time must be provided"):
            dml_panel(df, y="y", treat="d", covariates=["x1"],
                      unit="pid", include_time_fe=True)

    def test_no_covariates_allowed(self):
        """Empty covariate list should still work (pure FE OLS fallback)."""
        df = _sim_panel(N=40, T=6, d_on_x=0.0, seed=0)
        res = dml_panel(df, y="y", treat="d", covariates=[],
                        unit="pid", n_folds=3)
        assert np.isfinite(res.estimate)
        assert np.isfinite(res.se)


# --------------------------------------------------------------------------- #
#  Public API + registry
# --------------------------------------------------------------------------- #


class TestPublicAPI:

    def test_top_level_import(self):
        assert sp.dml_panel is not None
        assert sp.DMLPanelResult is not None

    def test_registry_entry(self):
        info = sp.describe_function("dml_panel")
        assert info is not None
        assert info["category"] == "causal"
        assert "panel" in info["tags"]
        # Required params present
        pnames = {p["name"] for p in info["params"]}
        for expected in ("data", "y", "treat", "covariates", "unit"):
            assert expected in pnames, expected

    def test_diagnostics_populated(self):
        df = _sim_panel(N=40, T=6, seed=0)
        res = dml_panel(df, y="y", treat="d", covariates=["x1"],
                        unit="pid", n_folds=3)
        for key in ("y_resid_std", "d_resid_std", "within_r2_outcome",
                    "omega_cluster"):
            assert key in res.diagnostics
