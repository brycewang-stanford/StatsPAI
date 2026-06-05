"""Coverage tests for statspai.iv.weak_iv_ci (AR / CLR / K confidence sets)."""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp

_wci = importlib.import_module("statspai.iv.weak_iv_ci")


def _iv_df(n=400, seed=0, strong=True):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    coef = 0.9 if strong else 0.02
    d = coef * z + v
    u = 0.5 * v + rng.normal(0, 0.5, size=n)
    y = 1.0 + 2.0 * d + u
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": rng.normal(size=n)})


def test_ar_ci_summary_connected():
    df = _iv_df(seed=1, strong=True)
    cs = sp.iv.anderson_rubin_ci(y="y", endog="d", instruments="z", data=df,
                                 n_grid=201)
    s = cs.summary()
    assert "Anderson-Rubin" in s
    assert "confidence set" in s
    iv = cs.as_intervals()
    assert isinstance(iv, list)


def test_clr_ci_runs():
    df = _iv_df(seed=2, strong=True)
    cs = sp.iv.conditional_lr_ci(y="y", endog="d", instruments="z", data=df,
                                 n_grid=81, random_state=0)
    assert cs.method == "Moreira CLR"
    assert cs.beta_grid.shape[0] == 81


def test_k_test_ci_runs():
    df = _iv_df(seed=3, strong=True)
    cs = sp.iv.k_test_ci(y="y", endog="d", instruments="z", data=df, n_grid=81)
    assert cs.method == "Kleibergen K"
    assert cs.statistic.shape == cs.beta_grid.shape


def test_ar_ci_weak_instrument_unbounded_summary():
    # very weak instrument -> CI likely touches grid boundary / disconnected
    df = _iv_df(n=300, seed=4, strong=False)
    cs = sp.iv.anderson_rubin_ci(y="y", endog="d", instruments="z", data=df,
                                 n_grid=201)
    s = cs.summary()
    # exercises either unbounded note, disconnected, or single interval lines
    assert "confidence set" in s


def test_ar_ci_with_exog_and_multi_instruments():
    df = _iv_df(seed=5, strong=True).copy()
    rng = np.random.default_rng(5)
    df["z2"] = 0.6 * df["z"] + rng.normal(size=len(df))
    cs = sp.iv.anderson_rubin_ci(y="y", endog="d", instruments=["z", "z2"],
                                 exog="x", data=df, n_grid=121)
    assert cs.extra["df_num"] == 2


def test_ar_ci_custom_grid():
    df = _iv_df(seed=6, strong=True)
    grid = np.linspace(0.0, 4.0, 51)
    cs = sp.iv.anderson_rubin_ci(y="y", endog="d", instruments="z", data=df,
                                 beta_grid=grid)
    np.testing.assert_array_equal(cs.beta_grid, grid)


def test_ar_ci_empty_set_via_far_grid():
    # grid far from true beta=2 -> no point passes -> empty set (375-376)
    df = _iv_df(seed=9, strong=True)
    grid = np.linspace(50.0, 60.0, 41)
    cs = sp.iv.anderson_rubin_ci(y="y", endog="d", instruments="z", data=df,
                                 beta_grid=grid)
    assert cs.is_empty
    assert np.isnan(cs.lower) and np.isnan(cs.upper)


def test_as_intervals_empty_set():
    cs = _wci.WeakIVConfidenceSet(
        method="X", level=0.95, beta_grid=np.linspace(-1, 1, 5),
        statistic=np.ones(5), critical_value=np.ones(5),
        in_set=np.zeros(5, dtype=bool), lower=np.nan, upper=np.nan,
        is_empty=True, is_connected=False, is_unbounded=False, extra={},
    )
    assert cs.as_intervals() == []
    assert "EMPTY" in cs.summary()


def test_summary_disconnected_and_unbounded():
    grid = np.linspace(-2, 2, 7)
    in_set = np.array([True, False, True, True, False, False, True])
    cs = _wci.WeakIVConfidenceSet(
        method="X", level=0.9, beta_grid=grid,
        statistic=np.ones(7), critical_value=np.ones(7),
        in_set=in_set, lower=float(grid[0]), upper=float(grid[-1]),
        is_empty=False, is_connected=False, is_unbounded=True, extra={},
    )
    s = cs.summary()
    assert "disconnected" in s
    assert "unbounded" in s.lower()
    # multiple intervals recovered
    assert len(cs.as_intervals()) == 3


def test_prep_helpers():
    df = _iv_df(seed=7, strong=True)
    Yt, Dt, Zt, kW, n = _wci._prep(
        "y", "d", "z", "x", df, add_const=True,
    )
    assert n == len(df)
    assert kW == 2  # const + x
    # residualize with empty W passthrough
    M = np.arange(6.0).reshape(3, 2)
    np.testing.assert_array_equal(_wci._residualize(M, np.empty((3, 0))), M)
    arr = _wci._grab(np.array([1.0, 2.0]), None)
    np.testing.assert_array_equal(arr, [1.0, 2.0])


def test_default_grid_irrelevant_instrument():
    # instrument totally uncorrelated with D -> denom ~ 0 branch (148-149)
    rng = np.random.default_rng(8)
    n = 300
    Yt = rng.normal(size=n)
    Dt = rng.normal(size=n)
    # nonzero instrument that is (nearly) orthogonal to D -> D_hat @ Dt ~ 0
    Z = rng.normal(size=n)
    # Gram-Schmidt: remove the D-component so projection of D on Z is ~0
    Z = Z - (Z @ Dt) / (Dt @ Dt) * Dt
    Zt = Z.reshape(-1, 1)
    grid = _wci._default_grid(Yt, Dt, Zt, 51)
    assert grid.shape == (51,)
    assert np.isfinite(grid).all()
