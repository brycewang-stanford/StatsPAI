"""Tests for ``sp.fast.feols`` — native OLS HDFE estimator."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# ---------------------------------------------------------------------------
# Synthetic OLS DGP (panel with within-cluster correlation)
# ---------------------------------------------------------------------------


def _ols_panel(n=4_000, n_units=100, n_periods=20, seed=0, beta=(0.30, -0.20)):
    rng = np.random.default_rng(seed)
    i = np.repeat(np.arange(n_units), n_periods)[:n]
    t = np.tile(np.arange(n_periods), n_units)[:n]
    n = i.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = rng.normal(0, 0.5, size=n_units)[i]  # unit FE
    g = rng.normal(0, 0.3, size=n_periods)[t]  # time FE
    cl = rng.normal(0, 0.4, size=n_units)[i]  # cluster shock
    eps = cl + rng.normal(size=n)
    y = beta[0] * x1 + beta[1] * x2 + a + g + eps
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "fe1": i.astype(np.int32),
            "fe2": t.astype(np.int32),
        }
    )


# ---------------------------------------------------------------------------
# Coefficient parity vs pyfixest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_feols_coef_matches_pyfixest_iid(seed):
    pf = pytest.importorskip("pyfixest")
    df = _ols_panel(seed=seed)
    fit = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")
    pf_fit = pf.feols(
        fml="y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid", fixef_rm="singleton"
    )
    for k in ("x1", "x2"):
        diff = abs(float(fit.coef()[k]) - float(pf_fit.coef()[k]))
        assert diff < 1e-10, f"coef[{k}] diff {diff:.3e}"


def test_feols_se_iid_close_to_pyfixest():
    """Order-of-magnitude SE parity vs pyfixest IID — 1% tolerance.

    pyfixest mirrors fixest's default ``ssc(fixef.K="nested")`` which
    for non-nested 2-FE counts ``Σ G_k - K + 1`` (the true matrix rank,
    accounting for the shared-intercept collinearity). StatsPAI uses
    ``Σ (G_k - 1)`` consistently across fast/* — off by 1 DOF for K=2,
    which translates to a ~3e-4 SE drift on this panel. That's well
    inside 1% (factor of 30×) but we don't promise machine epsilon.
    """
    pf = pytest.importorskip("pyfixest")
    df = _ols_panel(seed=4)
    fit = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")
    pf_fit = pf.feols(
        fml="y ~ x1 + x2 | fe1 + fe2", data=df, vcov="iid", fixef_rm="singleton"
    )
    for k in ("x1", "x2"):
        rel = abs(float(fit.se()[k]) - float(pf_fit.se()[k])) / max(
            abs(float(pf_fit.se()[k])), 1e-15
        )
        assert rel < 0.01, f"SE[{k}] rel {rel:.3e}"


def test_feols_se_hc1_close_to_pyfixest():
    """Same convention drift as iid; tolerance 1%."""
    pf = pytest.importorskip("pyfixest")
    df = _ols_panel(seed=5)
    fit = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov="hc1")
    pf_fit = pf.feols(
        fml="y ~ x1 + x2 | fe1 + fe2", data=df, vcov="hetero", fixef_rm="singleton"
    )
    for k in ("x1", "x2"):
        rel = abs(float(fit.se()[k]) - float(pf_fit.se()[k])) / max(
            abs(float(pf_fit.se()[k])), 1e-15
        )
        assert rel < 0.01, f"hc1 SE[{k}] rel {rel:.3e}"


def test_feols_se_cr1_close_to_pyfixest():
    """Cluster-SE parity is loosest because pyfixest uses
    ``ssc(fixef.K="nested", cluster.adj=TRUE)`` which doesn't charge
    FEs nested within the cluster (here, ``fe1`` is nested in
    cluster=fe1). StatsPAI charges all FEs (``fixef.K="full"`` mode);
    the resulting SE is uniformly larger by sqrt((n-k-fe_dof_full) /
    (n-k-fe_dof_nested)) — typically ~3% on a unit-clustered panel.
    Tolerance is 5% — accept the convention drift, reject anything
    larger as a likely sandwich-formula bug.
    """
    pf = pytest.importorskip("pyfixest")
    df = _ols_panel(seed=6)
    fit = sp.fast.feols(
        "y ~ x1 + x2 | fe1 + fe2",
        df,
        vcov="cr1",
        cluster="fe1",
    )
    pf_fit = pf.feols(
        fml="y ~ x1 + x2 | fe1 + fe2",
        data=df,
        vcov={"CRV1": "fe1"},
        fixef_rm="singleton",
    )
    for k in ("x1", "x2"):
        rel = abs(float(fit.se()[k]) - float(pf_fit.se()[k])) / max(
            abs(float(pf_fit.se()[k])), 1e-15
        )
        assert rel < 0.05, f"cr1 SE[{k}] rel {rel:.3e}"


# ---------------------------------------------------------------------------
# Pure OLS path (no FEs) — intercept auto-added
# ---------------------------------------------------------------------------


def test_feols_no_fe_matches_numpy_lstsq():
    df = _ols_panel(seed=7, n_units=50, n_periods=10)
    fit = sp.fast.feols("y ~ x1 + x2", df, vcov="iid")
    # NumPy ground truth
    X = np.column_stack([np.ones(len(df)), df["x1"], df["x2"]])
    y = df["y"].to_numpy()
    beta_np = np.linalg.solve(X.T @ X, X.T @ y)
    np.testing.assert_allclose(
        fit.coef_vec,
        beta_np,
        atol=1e-12,
        rtol=1e-12,
    )
    # SE from sigma² (X'X)^-1 / df
    n, p = X.shape
    resid = y - X @ beta_np
    sigma2 = float(resid @ resid / (n - p))
    V_np = sigma2 * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(
        np.diag(fit.vcov_matrix),
        np.diag(V_np),
        rtol=1e-12,
    )


def test_feols_no_fe_intercept_naming():
    df = _ols_panel(seed=8, n_units=20, n_periods=5)
    fit = sp.fast.feols("y ~ x1", df)
    assert list(fit.coef_names) == ["(Intercept)", "x1"]


# ---------------------------------------------------------------------------
# Weighted feols (WLS)
# ---------------------------------------------------------------------------


def test_feols_unweighted_equals_uniform_weights():
    df = _ols_panel(seed=9)
    df["w_one"] = 1.0
    fit_a = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df)
    fit_b = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, weights="w_one")
    for k in ("x1", "x2"):
        assert abs(float(fit_a.coef()[k]) - float(fit_b.coef()[k])) < 1e-12
        assert abs(float(fit_a.se()[k]) - float(fit_b.se()[k])) < 1e-12


def test_feols_weighted_matches_pyfixest():
    pf = pytest.importorskip("pyfixest")
    rng = np.random.default_rng(10)
    df = _ols_panel(seed=10)
    df["w"] = 0.5 + rng.random(len(df))
    fit = sp.fast.feols(
        "y ~ x1 + x2 | fe1 + fe2",
        df,
        weights="w",
        vcov="iid",
    )
    pf_fit = pf.feols(
        fml="y ~ x1 + x2 | fe1 + fe2",
        data=df,
        weights="w",
        vcov="iid",
        fixef_rm="singleton",
    )
    for k in ("x1", "x2"):
        diff = abs(float(fit.coef()[k]) - float(pf_fit.coef()[k]))
        assert diff < 1e-8, f"weighted coef[{k}] diff {diff:.3e}"


# ---------------------------------------------------------------------------
# Validation / edge cases
# ---------------------------------------------------------------------------


def test_feols_unknown_vcov_rejected():
    df = _ols_panel(seed=12)
    with pytest.raises(ValueError, match="vcov"):
        sp.fast.feols("y ~ x1 | fe1", df, vcov="bogus")


def test_feols_cluster_validation():
    df = _ols_panel(seed=13)
    with pytest.raises(ValueError, match="cluster"):
        sp.fast.feols("y ~ x1 | fe1", df, vcov="cr1")
    with pytest.raises(ValueError, match="vcov='iid'"):
        sp.fast.feols("y ~ x1 | fe1", df, vcov="iid", cluster="fe1")


def test_feols_cluster_nan_rejected():
    df = _ols_panel(seed=14)
    df["cl"] = df["fe1"].astype(float)
    df.loc[df.index[3], "cl"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        sp.fast.feols("y ~ x1 | fe1", df, vcov="cr1", cluster="cl")


def test_feols_missing_column_raises():
    df = _ols_panel(seed=15)
    with pytest.raises(MethodIncompatibility, match="missing"):
        sp.fast.feols("y ~ x_missing | fe1", df)


def test_feols_y_nonfinite_raises():
    df = _ols_panel(seed=16)
    df.loc[df.index[2], "y"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sp.fast.feols("y ~ x1 | fe1", df)


def test_feols_x_nonfinite_raises():
    df = _ols_panel(seed=17)
    df.loc[df.index[3], "x1"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        sp.fast.feols("y ~ x1 | fe1", df)


def test_feols_empty_data_raises():
    df = _ols_panel(seed=171).iloc[0:0].copy()
    with pytest.raises(ValueError, match="at least one row"):
        sp.fast.feols("y ~ x1 | fe1", df)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"fe_maxiter": 0}, "fe_maxiter"),
        ({"fe_tol": -1.0}, "fe_tol"),
        ({"fe_tol": np.nan}, "fe_tol"),
    ],
)
def test_feols_invalid_demean_controls_raise(kwargs, match):
    df = _ols_panel(seed=172)
    with pytest.raises(ValueError, match=match):
        sp.fast.feols("y ~ x1 | fe1", df, **kwargs)


def test_feols_all_zero_weights_raise():
    df = _ols_panel(seed=173)
    df["w"] = 0.0
    with pytest.raises(ValueError, match="no positive mass"):
        sp.fast.feols("y ~ x1 | fe1", df, weights="w")


def test_feols_kept_sample_all_zero_weights_raise():
    df = _ols_panel(seed=174, n_units=10, n_periods=5)
    df["w"] = 0.0
    extra = df.iloc[[0]].copy()
    extra["fe1"] = 999
    extra["w"] = 1.0
    df_aug = pd.concat([df, extra], ignore_index=True)

    with pytest.raises(ValueError, match="no positive mass"):
        sp.fast.feols("y ~ x1 | fe1 + fe2", df_aug, weights="w")


def test_feols_singleton_drop():
    """Singletons should be removed by default; n_kept reflects this."""
    df = _ols_panel(seed=18, n_units=10, n_periods=5)
    # Append a singleton row whose unit appears only once
    extra = df.iloc[[0]].copy()
    extra["fe1"] = 999
    df_aug = pd.concat([df, extra], ignore_index=True)
    fit = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df_aug)
    assert fit.n_dropped_singletons >= 1
    assert fit.n_kept == fit.n_obs - fit.n_dropped_singletons


# ---------------------------------------------------------------------------
# Result-object surface
# ---------------------------------------------------------------------------


def test_feols_result_accessors_shape():
    df = _ols_panel(seed=20)
    fit = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df)
    assert isinstance(fit.coef(), pd.Series)
    assert isinstance(fit.se(), pd.Series)
    assert isinstance(fit.vcov(), pd.DataFrame)
    assert isinstance(fit.tidy(), pd.DataFrame)
    assert "feols" in fit.summary()
    assert fit.coef_vec.size == 2  # x1, x2
    assert fit.vcov_matrix.shape == (2, 2)
    assert 0.0 <= fit.r_squared_within <= 1.0


def test_feols_result_protocol_json_safe():
    df = _ols_panel(seed=25, n=800, n_units=40, n_periods=20)
    fit = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov="hc1")

    full = fit.to_dict()
    agent = fit.to_agent_summary(max_terms=1)

    assert full["kind"] == "fast_feols_result"
    assert full["model"] == "ols_hdfe"
    assert len(full["coefficients"]) == 2
    assert full["vcov"]["terms"] == ["x1", "x2"]
    assert agent["kind"] == "fast_feols_agent_summary"
    assert len(agent["coefficients"]) == 1
    assert agent["truncated_terms"] == 1
    json.dumps(full)
    json.dumps(agent)


def test_feols_df_resid_accounting():
    """df_resid = n_kept - p - sum(G_k - 1) — pin the convention."""
    df = _ols_panel(seed=21, n_units=30, n_periods=10)
    fit = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df)
    expected = fit.n_kept - fit.coef_vec.size - sum(c - 1 for c in fit.fe_cardinality)
    assert fit.df_resid == expected


def test_feols_fixest_ssc_uses_full_fe_rank_for_iid():
    df = _ols_panel(seed=22, n_units=30, n_periods=10)
    fit_default = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")
    fit_fixest = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid", ssc="fixest")

    expected_default_df = (
        fit_default.n_kept
        - fit_default.coef_vec.size
        - sum(c - 1 for c in fit_default.fe_cardinality)
    )
    expected_fixest_df = (
        fit_fixest.n_kept
        - fit_fixest.coef_vec.size
        - (sum(fit_fixest.fe_cardinality) - 1)
    )

    assert fit_default.df_resid == expected_default_df
    assert fit_fixest.df_resid == expected_fixest_df
    np.testing.assert_allclose(fit_default.coef_vec, fit_fixest.coef_vec)
    ratio = float(fit_fixest.se()["x1"] / fit_default.se()["x1"])
    assert ratio == pytest.approx(
        np.sqrt(expected_default_df / expected_fixest_df),
        rel=1e-12,
    )


def test_feols_fixest_ssc_excludes_cluster_nested_fe_from_cr1():
    df = _ols_panel(seed=23, n_units=30, n_periods=10)
    fit_default = sp.fast.feols(
        "y ~ x1 + x2 | fe1 + fe2", df, vcov="cr1", cluster="fe1"
    )
    fit_fixest = sp.fast.feols(
        "y ~ x1 + x2 | fe1 + fe2",
        df,
        vcov="cr1",
        cluster="fe1",
        ssc="fixest",
    )

    n = fit_default.n_kept
    p = fit_default.coef_vec.size
    g = df.loc[: n - 1, "fe1"].nunique()
    default_fe_dof = sum(c - 1 for c in fit_default.fe_cardinality)
    # fe1 is nested in cluster=fe1, so only the non-nested time FE enters
    # fixest/reghdfe's CR1 small-sample denominator.
    fixest_cluster_fe_dof = fit_fixest.fe_cardinality[1]
    default_factor = (g / (g - 1)) * ((n - 1) / (n - p - default_fe_dof))
    fixest_factor = (g / (g - 1)) * ((n - 1) / (n - p - fixest_cluster_fe_dof))

    np.testing.assert_allclose(fit_default.coef_vec, fit_fixest.coef_vec)
    ratio = float(fit_fixest.se()["x1"] / fit_default.se()["x1"])
    assert ratio == pytest.approx(np.sqrt(fixest_factor / default_factor), rel=1e-12)


def test_feols_unknown_ssc_rejected():
    df = _ols_panel(seed=24)
    with pytest.raises(ValueError, match="ssc"):
        sp.fast.feols("y ~ x1 | fe1", df, ssc="bogus")


# ---------------------------------------------------------------------------
# Cross-engine: R fixest parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not on PATH")
def test_feols_coef_matches_r_fixest(tmp_path):
    df = _ols_panel(seed=30)
    csv_path = tmp_path / "panel.csv"
    df.to_csv(csv_path, index=False)
    fit = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")

    r_script = (
        "suppressMessages({library(data.table); library(fixest); library(jsonlite)})\n"
        f"d <- fread('{csv_path}')\n"
        "f <- feols(y ~ x1 + x2 | fe1 + fe2, data=d, ssc=ssc(fixef.K='full'))\n"
        "out <- list(coefs = as.list(coef(f)), se = as.list(se(f)))\n"
        "cat(toJSON(out, auto_unbox=TRUE, digits=14))\n"
    )
    proc = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(f"Rscript failed: {proc.stderr[:200]}")
    r_out = json.loads(proc.stdout.strip().splitlines()[-1])
    for k in ("x1", "x2"):
        diff = abs(float(fit.coef()[k]) - float(r_out["coefs"][k]))
        assert diff < 1e-10, f"vs fixest::feols coef[{k}] diff {diff:.3e}"


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not on PATH")
def test_feols_se_iid_close_to_r_fixest(tmp_path):
    """SE iid parity vs fixest::feols within 1% (1-DOF off-by-true-rank
    convention same as the rest of fast/*)."""
    df = _ols_panel(seed=31)
    csv_path = tmp_path / "panel.csv"
    df.to_csv(csv_path, index=False)
    fit = sp.fast.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")

    r_script = (
        "suppressMessages({library(data.table); library(fixest); library(jsonlite)})\n"
        f"d <- fread('{csv_path}')\n"
        "f <- feols(y ~ x1 + x2 | fe1 + fe2, data=d, ssc=ssc(fixef.K='full'))\n"
        "out <- list(se = as.list(se(f)))\n"
        "cat(toJSON(out, auto_unbox=TRUE, digits=14))\n"
    )
    proc = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(f"Rscript failed: {proc.stderr[:200]}")
    r_out = json.loads(proc.stdout.strip().splitlines()[-1])
    for k in ("x1", "x2"):
        sp_se = float(fit.se()[k])
        r_se = float(r_out["se"][k])
        rel = abs(sp_se - r_se) / max(abs(r_se), 1e-15)
        assert rel < 0.01, f"SE drift at {k}: sp={sp_se:.6e} fixest={r_se:.6e}"


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not on PATH")
def test_feols_se_cluster_close_to_r_fixest(tmp_path):
    """Cluster CR1 parity vs fixest::feols(... cluster=~fe1) within 2%."""
    df = _ols_panel(seed=32)
    csv_path = tmp_path / "panel.csv"
    df.to_csv(csv_path, index=False)
    fit = sp.fast.feols(
        "y ~ x1 + x2 | fe1 + fe2",
        df,
        vcov="cr1",
        cluster="fe1",
    )

    r_script = (
        "suppressMessages({library(data.table); library(fixest); library(jsonlite)})\n"
        f"d <- fread('{csv_path}')\n"
        "f <- feols(y ~ x1 + x2 | fe1 + fe2, data=d, cluster=~fe1,\n"
        "           ssc=ssc(fixef.K='full'))\n"
        "out <- list(se = as.list(se(f)))\n"
        "cat(toJSON(out, auto_unbox=TRUE, digits=14))\n"
    )
    proc = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        pytest.skip(f"Rscript failed: {proc.stderr[:200]}")
    r_out = json.loads(proc.stdout.strip().splitlines()[-1])
    for k in ("x1", "x2"):
        sp_se = float(fit.se()[k])
        r_se = float(r_out["se"][k])
        rel = abs(sp_se - r_se) / max(abs(r_se), 1e-15)
        assert rel < 0.02, f"cluster SE drift at {k}: sp={sp_se:.6e} fixest={r_se:.6e}"
