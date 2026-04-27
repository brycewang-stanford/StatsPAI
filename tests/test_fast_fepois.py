"""Tests for ``sp.fast.fepois`` — native Poisson HDFE estimator.

Phase 2 acceptance gate. Ground truth here is fixest (the most
battle-tested HDFE implementation) and ``ppmlhdfe`` (Stata's reference);
since neither is callable in-process, we use ``pyfixest.fepois`` as a
proxy and add a separate Rscript-based fixest comparison if R is found.
"""
from __future__ import annotations

import shutil
import subprocess
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Synthetic Poisson DGP
# ---------------------------------------------------------------------------

def _poisson_panel(n=5_000, n_units=200, n_periods=25, seed=0,
                    beta=(0.30, -0.20)):
    rng = np.random.default_rng(seed)
    i = np.repeat(np.arange(n_units), n_periods)[:n]
    t = np.tile(np.arange(n_periods), n_units)[:n]
    n = i.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = rng.normal(0, 0.5, size=n_units)[i]
    g = rng.normal(0, 0.3, size=n_periods)[t]
    eta = 0.5 + beta[0] * x1 + beta[1] * x2 + a + g
    eta = np.clip(eta, -10, 10)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.int64)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2,
                         "fe1": i.astype(np.int32),
                         "fe2": t.astype(np.int32)})


# ---------------------------------------------------------------------------
# Coefficient + SE parity against pyfixest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 7])
def test_coef_matches_pyfixest_iid(seed):
    """Coefficients should match pyfixest to ~1e-12 (we solve the same
    optimisation problem with the same convergence tolerance)."""
    pf = pytest.importorskip("pyfixest")
    df = _poisson_panel(seed=seed)

    fit = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")
    pf_fit = pf.fepois(fml="y ~ x1 + x2 | fe1 + fe2", data=df,
                      vcov="iid", fixef_rm="singleton")

    for k in ("x1", "x2"):
        diff = abs(float(fit.coef()[k]) - float(pf_fit.coef()[k]))
        assert diff < 1e-10, f"coef[{k}] diff {diff:.3e} > 1e-10"


def test_se_matches_pyfixest_iid_with_ssc():
    """SEs should match pyfixest after applying the standard
    ``ssc(adj=TRUE)`` correction."""
    pf = pytest.importorskip("pyfixest")
    df = _poisson_panel(seed=2)

    fit = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")
    pf_fit = pf.fepois(fml="y ~ x1 + x2 | fe1 + fe2", data=df,
                      vcov="iid", fixef_rm="singleton")
    for k in ("x1", "x2"):
        diff = abs(float(fit.se()[k]) - float(pf_fit.se()[k]))
        assert diff < 1e-7, f"SE[{k}] diff {diff:.3e} > 1e-7"


def test_iterations_and_convergence():
    df = _poisson_panel(seed=3)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", df)
    assert fit.converged
    assert 1 <= fit.iterations <= 50


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_separation_rows_dropped():
    """Rows in all-zero FE clusters must be removed by the pre-pass."""
    rng = np.random.default_rng(11)
    n = 500
    fe1 = rng.integers(0, 10, size=n)
    # Force fe1=0 to be all-zero outcomes (perfect separation)
    fe2 = rng.integers(0, 5, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = 0.3 * x1 - 0.2 * x2 + 0.5
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.int64)
    y[fe1 == 0] = 0  # introduce separation
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "fe1": fe1, "fe2": fe2})

    fit = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", df,
                          drop_separation=True)
    # All rows in fe1==0 have y==0, so they are detected by the separation
    # pre-pass. (Pre-fix, the singleton/separation counts were lumped
    # under ``n_dropped_singletons``; the split is now correct.)
    assert fit.n_dropped_separation >= int((fe1 == 0).sum())
    assert fit.converged


def test_no_fe_means_intercept_only_poisson():
    """``y ~ x1 + x2`` (no FE) should still run; reduces to plain Poisson GLM."""
    rng = np.random.default_rng(13)
    n = 1000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    mu = np.exp(0.5 + 0.2 * x1 - 0.1 * x2)
    y = rng.poisson(mu).astype(np.int64)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    # no '|' — no FE
    fit = sp.fast.fepois("y ~ x1 + x2", df)
    assert fit.converged
    # With no FE, beta tells us the slope but not intercept (we don't
    # estimate an intercept here — we leave that absorbed by FE).
    assert abs(fit.coef()["x1"] - 0.2) < 0.05
    assert abs(fit.coef()["x2"] - (-0.1)) < 0.05


def test_negative_y_rejected():
    df = pd.DataFrame({
        "y": [1, -1, 2, 3], "x1": [0.0, 1.0, 2.0, 3.0],
        "fe1": [0, 1, 0, 1],
    })
    with pytest.raises(ValueError, match="non-negative"):
        sp.fast.fepois("y ~ x1 | fe1", df)


def test_missing_column_rejected():
    df = pd.DataFrame({"y": [1, 2], "fe1": [0, 1]})
    with pytest.raises(KeyError, match="missing"):
        sp.fast.fepois("y ~ nope | fe1", df)


def test_unknown_vcov_rejected():
    df = _poisson_panel(seed=14, n=200, n_units=20, n_periods=10)
    with pytest.raises(ValueError, match="vcov"):
        sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", df, vcov="bogus")


# ---------------------------------------------------------------------------
# Result object surface
# ---------------------------------------------------------------------------

def test_result_object_api():
    df = _poisson_panel(seed=5)
    fit = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", df)
    # Coef accessor
    coefs = fit.coef()
    assert isinstance(coefs, pd.Series)
    assert list(coefs.index) == ["x1", "x2"]
    # SE accessor
    se = fit.se()
    assert isinstance(se, pd.Series)
    assert (se > 0).all()
    # Tidy table
    tidy = fit.tidy()
    assert set(tidy.columns) == {"Estimate", "Std. Error", "z value", "Pr(>|z|)"}
    # Summary string non-empty
    s = fit.summary()
    assert "fepois" in s
    assert "Poisson" in s
    # vcov matrix shape
    V = fit.vcov_matrix
    assert V.shape == (2, 2)
    assert np.allclose(V, V.T)


# ---------------------------------------------------------------------------
# Cross-engine verification with R fixest (parity test)
# ---------------------------------------------------------------------------

def test_weights_unweighted_matches_default():
    """Passing all-1 weights must reproduce the unweighted fit bit-for-bit."""
    df = _poisson_panel(seed=20)
    df["w_one"] = 1.0
    fit_default = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")
    fit_w = sp.fast.fepois(
        "y ~ x1 + x2 | fe1 + fe2", df, vcov="iid", weights="w_one",
    )
    for k in ("x1", "x2"):
        assert abs(float(fit_default.coef()[k]) - float(fit_w.coef()[k])) < 1e-12
        assert abs(float(fit_default.se()[k]) - float(fit_w.se()[k])) < 1e-12


def test_weights_match_pyfixest_with_obs_weights():
    """Weighted MLE must match pyfixest's weighted fepois to ~1e-10."""
    pf = pytest.importorskip("pyfixest")
    rng = np.random.default_rng(21)
    df = _poisson_panel(seed=21, n=4000, n_units=80, n_periods=20)
    # Heterogeneous obs weights (uniform ish, all positive)
    df["w"] = 0.5 + rng.random(len(df))
    fit = sp.fast.fepois(
        "y ~ x1 + x2 | fe1 + fe2", df, vcov="iid", weights="w",
    )
    pf_fit = pf.fepois(
        fml="y ~ x1 + x2 | fe1 + fe2", data=df,
        vcov="iid", fixef_rm="singleton", weights="w",
    )
    for k in ("x1", "x2"):
        diff = abs(float(fit.coef()[k]) - float(pf_fit.coef()[k]))
        assert diff < 1e-8, f"weighted coef[{k}] diff {diff:.3e}"


def test_weights_negative_rejected():
    df = _poisson_panel(seed=22)
    df["w"] = 1.0
    df.loc[df.index[3], "w"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        sp.fast.fepois("y ~ x1 | fe1 + fe2", df, weights="w")


def test_weights_nonfinite_rejected():
    df = _poisson_panel(seed=23)
    df["w"] = 1.0
    df.loc[df.index[5], "w"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sp.fast.fepois("y ~ x1 | fe1 + fe2", df, weights="w")


def test_weights_missing_column_rejected():
    df = _poisson_panel(seed=24)
    with pytest.raises(KeyError):
        sp.fast.fepois("y ~ x1 | fe1 + fe2", df, weights="missing_col")


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not on PATH")
def test_coefs_match_r_fixest(tmp_path):
    """Round-trip the synthetic data through R fixest::fepois and compare."""
    df = _poisson_panel(seed=4)
    csv_path = tmp_path / "panel.csv"
    df.to_csv(csv_path, index=False)

    fit = sp.fast.fepois("y ~ x1 + x2 | fe1 + fe2", df, vcov="iid")

    r_script = (
        "suppressMessages({library(data.table); library(fixest); library(jsonlite)})\n"
        f"d <- fread('{csv_path}')\n"
        "f <- fepois(y ~ x1 + x2 | fe1 + fe2, data=d, glm.tol=1e-10, glm.iter=50)\n"
        "out <- list(coefs = as.list(coef(f)), se = as.list(se(f)))\n"
        "cat(toJSON(out, auto_unbox=TRUE, digits=12))\n"
    )
    proc = subprocess.run(
        ["Rscript", "-e", r_script], capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        pytest.skip(f"Rscript failed: {proc.stderr[:200]}")
    r_out = json.loads(proc.stdout.strip().splitlines()[-1])

    # Coefficients should agree to better than 1e-6 (fixest's IRLS tol)
    for k in ("x1", "x2"):
        diff_coef = abs(float(fit.coef()[k]) - float(r_out["coefs"][k]))
        assert diff_coef < 1e-6, f"vs fixest::fepois coef[{k}] diff {diff_coef:.3e}"
