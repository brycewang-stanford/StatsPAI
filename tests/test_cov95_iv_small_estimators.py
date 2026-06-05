"""Coverage tests for smaller statspai.iv estimators: post_lasso, bayesian_iv,
ivdml, many_weak, kernel_iv, continuous_late — internal branches and summaries.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp

_pl = importlib.import_module("statspai.iv.post_lasso")
_bi = importlib.import_module("statspai.iv.bayesian_iv")
_mw = importlib.import_module("statspai.iv.many_weak")
_cl = importlib.import_module("statspai.iv.continuous_late")
_ki = importlib.import_module("statspai.iv.kernel_iv")


def _many_iv_df(n=400, seed=0, p=8):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    v = rng.normal(size=n)
    coefs = np.zeros(p)
    coefs[:3] = [0.7, 0.5, 0.4]
    d = Z @ coefs + v
    u = 0.5 * v + rng.normal(0, 0.5, size=n)
    y = 1.0 + 2.0 * d + u
    cols = {f"z{i}": Z[:, i] for i in range(p)}
    return pd.DataFrame({"y": y, "d": d, "x": rng.normal(size=n), **cols})


# ─── post_lasso ──────────────────────────────────────────────────────────

def test_post_lasso_helpers():
    assert _pl._as_matrix(np.array([1.0, 2.0])).shape == (2, 1)
    arr = _pl._grab(np.array([1.0, 2.0]), None)
    np.testing.assert_array_equal(arr, [1.0, 2.0])
    M = np.arange(6.0).reshape(3, 2)
    np.testing.assert_array_equal(_pl._residualize(M, None), M)
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert _pl._names(df, "z", 2) == ["a", "b"]
    assert _pl._names(["q", "r"], "z", 2) == ["q", "r"]
    assert _pl._names(np.zeros((3, 2)), "z", 2) == ["z0", "z1"]


def test_post_lasso_summary_and_robust():
    df = _many_iv_df(seed=1)
    zs = [f"z{i}" for i in range(8)]
    r = sp.iv.bch_post_lasso_iv(y="y", endog="d", instruments=zs, exog="x",
                                data=df, robust=True)
    s = r.summary()
    assert "Post-Lasso IV" in s
    assert np.isfinite(r.beta.iloc[0])


def test_post_lasso_nonrobust_se():
    df = _many_iv_df(seed=2)
    zs = [f"z{i}" for i in range(8)]
    r = sp.iv.bch_post_lasso_iv(y="y", endog="d", instruments=zs, data=df,
                                robust=False)
    assert np.isfinite(r.std_errors.iloc[0])


def test_post_lasso_ensure_min_instruments():
    # weak instruments forcing the ensure_min_instruments top-up path
    rng = np.random.default_rng(3)
    n, p = 300, 8
    Z = rng.normal(size=(n, p))
    d = 0.05 * Z[:, 0] + rng.normal(size=n)  # very weak
    y = 1.0 + 2.0 * d + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, **{f"z{i}": Z[:, i] for i in range(p)}})
    zs = [f"z{i}" for i in range(p)]
    r = sp.iv.bch_post_lasso_iv(y="y", endog="d", instruments=zs, data=df,
                                ensure_min_instruments=2)
    assert r.n_selected >= 1


def test_bch_lambda():
    lam = _pl.bch_lambda(100, 10, alpha=0.05, c=1.1)
    assert lam > 0


# ─── bayesian_iv ─────────────────────────────────────────────────────────

def test_bayesian_iv_summary_and_frame():
    rng = np.random.default_rng(10)
    n = 400
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    d = 0.8 * z + v
    y = 1.0 + 2.0 * d + 0.5 * v + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "z": z, "x": rng.normal(size=n)})
    r = sp.iv.bayesian_iv(y="y", endog="d", instruments="z", exog="x", data=df,
                          n_draws=800, n_warmup=400, random_state=0)
    s = r.summary()
    assert "Bayesian IV" in s
    assert "HPD" in s
    fr = r.to_frame()
    assert "beta" in fr.columns
    assert np.isfinite(r.posterior_mean)


def test_bayesian_iv_helpers():
    arr = _bi._grab(np.array([1.0, 2.0]), None)
    np.testing.assert_array_equal(arr, [1.0, 2.0])
    M = np.arange(6.0).reshape(3, 2)
    np.testing.assert_array_equal(_bi._residualize(M, None), M)
    # hpd where n_in >= n
    lo, hi = _bi._hpd(np.array([1.0, 2.0, 3.0]), level=0.999999)
    assert lo == 1.0 and hi == 3.0
    # ess with small n and zero-variance
    assert _bi._ess(np.ones(5)) == 5.0
    assert _bi._ess(np.ones(100)) == 100.0  # var < 1e-12 -> n


# ─── many_weak ───────────────────────────────────────────────────────────

def test_many_weak_jive_with_exog():
    df = _many_iv_df(seed=20)
    zs = [f"z{i}" for i in range(8)]
    r = _mw.jive(data=df, y="y", endog="d", instruments=zs, exog=["x"])
    assert r.estimator == "JIVE"
    assert np.isfinite(r.estimate)


def test_many_weak_ar_default_grid_with_exog():
    df = _many_iv_df(seed=21)
    zs = [f"z{i}" for i in range(8)]
    r = _mw.many_weak_ar(data=df, y="y", endog="d", instruments=zs, exog=["x"])
    assert np.isfinite(r.estimate)


def test_many_weak_ar_custom_grid():
    df = _many_iv_df(seed=22)
    zs = [f"z{i}" for i in range(8)]
    grid = np.linspace(0.0, 4.0, 81)
    r = _mw.many_weak_ar(data=df, y="y", endog="d", instruments=zs,
                         beta_grid=grid)
    assert np.isfinite(r.estimate)


# ─── continuous_late ─────────────────────────────────────────────────────

def test_continuous_late_summary():
    rng = np.random.default_rng(30)
    n = 600
    z = rng.uniform(-1, 1, size=n)
    d = (z + 0.3 * rng.normal(size=n) > 0).astype(float)
    y = 1.5 * d + 0.4 * z + rng.normal(size=n) * 0.4
    df = pd.DataFrame({"y": y, "treat": d, "z": z})
    r = sp.iv.continuous_iv_late(df, y="y", treat="treat", instrument="z",
                                 n_quantiles=4, n_boot=30, seed=0)
    s = r.summary()
    assert "Continuous-Instrument LATE" in s
    assert r.n_obs == n


def test_continuous_late_wald_per_bin_degenerate():
    # constant instrument -> qcut collapses -> <2 bins fallback (line 82)
    rng = np.random.default_rng(31)
    n = 200
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "treat": (rng.uniform(size=n) > 0.5).astype(float),
        "z": np.ones(n),  # constant instrument
    })
    r = sp.iv.continuous_iv_late(df, y="y", treat="treat", instrument="z",
                                 n_quantiles=4, n_boot=10, seed=0)
    assert r.n_obs == n


# ─── kernel_iv ───────────────────────────────────────────────────────────

def test_continuous_late_constant_treatment_no_shift():
    # z varies (>=2 bins) but treatment is constant -> denom ~ 0 in every
    # adjacent-bin pair -> continue (92-93) then no atts -> nan (96-97).
    rng = np.random.default_rng(32)
    n = 300
    z = rng.uniform(-1, 1, size=n)
    df = pd.DataFrame({
        "y": rng.normal(size=n),
        "treat": np.ones(n),  # constant treatment, no first stage
        "z": z,
    })
    r = sp.iv.continuous_iv_late(df, y="y", treat="treat", instrument="z",
                                 n_quantiles=4, n_boot=10, seed=0)
    assert r.n_obs == n


def test_kernel_iv_summary_and_band():
    rng = np.random.default_rng(40)
    n = 400
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    d = 0.8 * z + v
    y = 1.0 + 1.5 * d - 0.2 * d ** 2 + 0.5 * v + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "z": z})
    r = sp.iv.kernel_iv(df, y="y", treat="d", instrument="z", n_boot=30, seed=0)
    assert np.all(r.ci_low <= r.ci_high + 1e-9)
    assert "Kernel IV" in r.summary()


def test_kernel_iv_far_grid_nan_weight():
    # grid points far outside the support with tiny bandwidth -> w_d.sum ~ 0
    # -> NaN h on those grid points (lines 109-110)
    rng = np.random.default_rng(41)
    n = 200
    z = rng.normal(size=n)
    d = 0.8 * z + rng.normal(size=n)
    y = 1.0 + d + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "z": z})
    grid = np.array([-1e6, 0.0, 1e6])
    r = sp.iv.kernel_iv(df, y="y", treat="d", instrument="z", grid=grid,
                        bandwidth=0.01, n_boot=10, seed=0)
    assert np.isnan(r.h_hat[0]) or np.isnan(r.h_hat[-1])


# ─── ivdml ───────────────────────────────────────────────────────────────

def test_ivdml_summary_and_estimate():
    rng = np.random.default_rng(50)
    n = 600
    Z = rng.normal(size=(n, 3))
    X = rng.normal(size=(n, 2))
    v = rng.normal(size=n)
    d = 0.6 * Z[:, 0] + 0.3 * X[:, 0] + v
    y = 1.0 + 2.0 * d + 0.4 * X[:, 1] + 0.5 * v + rng.normal(size=n)
    df = pd.DataFrame({
        "y": y, "d": d,
        **{f"z{i}": Z[:, i] for i in range(3)},
        **{f"x{i}": X[:, i] for i in range(2)},
    })
    r = sp.iv.ivdml(df, y="y", treat="d", instruments=["z0", "z1", "z2"],
                    covariates=["x0", "x1"], n_folds=3, seed=0)
    s = r.summary()
    assert "IV" in s and "DML" in s
    assert np.isfinite(r.estimate)
    assert r.n_obs == n


def test_ivdml_no_covariates():
    # exercises the "no X" outcome-model branch (line 103)
    rng = np.random.default_rng(51)
    n = 500
    Z = rng.normal(size=(n, 3))
    v = rng.normal(size=n)
    d = 0.6 * Z[:, 0] + v
    y = 1.0 + 2.0 * d + 0.5 * v + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, **{f"z{i}": Z[:, i] for i in range(3)}})
    r = sp.iv.ivdml(df, y="y", treat="d", instruments=["z0", "z1", "z2"],
                    n_folds=3, seed=0)
    assert np.isfinite(r.first_stage_F)
