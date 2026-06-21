"""Coverage tests for statspai.iv.mte (BMW 2017 polynomial MTE)."""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp

_mte = importlib.import_module("statspai.iv.mte")


def _binary_df(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    v = rng.normal(size=n)
    latent = 0.9 * z + 0.4 * x + v
    d = (latent > 0).astype(float)
    u = 0.5 * v + rng.normal(0, 0.5, size=n)
    y = 0.3 + 0.8 * d + 0.2 * x + u
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": x})


def test_mte_helpers():
    assert _mte._as_matrix(np.array([1.0, 2.0])).shape == (2, 1)
    assert _mte._as_matrix(np.zeros((2, 2))).shape == (2, 2)
    arr = _mte._grab(np.array([1.0, 2.0]), None)
    np.testing.assert_array_equal(arr, [1.0, 2.0])
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    out = _mte._grab(["a", "b"], df, cols=True)
    assert out.shape == (1, 2)
    P = _mte._poly_u(np.array([0.0, 1.0]), 2)
    assert P.shape == (2, 3)
    ip = _mte._int_poly_u(0.0, 1.0, 2)
    np.testing.assert_allclose(ip, [1.0, 0.5, 1.0 / 3.0])


def test_mte_logit_summary():
    df = _binary_df(seed=1)
    r = sp.iv.mte(
        y="y",
        treatment="d",
        instruments="z",
        exog="x",
        data=df,
        poly_degree=2,
        propensity_model="logit",
    )
    s = r.summary()
    assert "Marginal Treatment Effects" in s
    assert "ATE" in s and "ATT" in s
    assert r.mte_curve.shape[1] >= 3
    assert np.isfinite(r.ate)


def test_mte_probit_propensity():
    df = _binary_df(seed=2)
    r = sp.iv.mte(
        y="y",
        treatment="d",
        instruments="z",
        exog="x",
        data=df,
        poly_degree=1,
        propensity_model="probit",
    )
    assert np.isfinite(r.ate)


def test_mte_linear_propensity():
    df = _binary_df(seed=3)
    r = sp.iv.mte(
        y="y",
        treatment="d",
        instruments="z",
        exog="x",
        data=df,
        poly_degree=1,
        propensity_model="linear",
    )
    assert np.isfinite(r.ate)


def test_mte_no_exog_array_inputs():
    df = _binary_df(seed=4)
    # pass arrays (no data) to exercise reshape branches + no-exog path
    r = sp.iv.mte(
        y=df["y"].to_numpy(),
        treatment=df["d"].to_numpy(),
        instruments=df["z"].to_numpy(),
        poly_degree=1,
    )
    assert r.n_obs > 0


def test_mte_bootstrap_se():
    df = _binary_df(seed=5)
    r = sp.iv.mte(
        y="y",
        treatment="d",
        instruments="z",
        exog="x",
        data=df,
        poly_degree=1,
        bootstrap=25,
        random_state=0,
    )
    # bootstrap should populate att_se/atu_se or n_successful_draws
    assert (
        ("att_se" in r.extra)
        or ("n_successful_draws" in r.extra)
        or np.isfinite(r.ate_se)
    )


def test_mte_no_const_no_exog():
    # add_const=False and exog=None -> X_raw empty -> falls back to ones (180)
    df = _binary_df(seed=14)
    r = sp.iv.mte(
        y="y", treatment="d", instruments="z", data=df, poly_degree=1, add_const=False
    )
    assert np.isfinite(r.ate)


def test_mte_nonbinary_treatment_raises():
    df = _binary_df(seed=6).copy()
    df["d"] = df["d"] + 0.5
    with pytest.raises(ValueError, match="binary"):
        sp.iv.mte(y="y", treatment="d", instruments="z", data=df)


def test_mte_degree_too_high_raises():
    # very high poly degree relative to arm sizes -> "Not enough observations"
    df = _binary_df(n=40, seed=7)
    with pytest.raises(ValueError, match="Not enough observations"):
        sp.iv.mte(
            y="y", treatment="d", instruments="z", exog="x", data=df, poly_degree=8
        )


def test_fit_propensity_models():
    rng = np.random.default_rng(8)
    n = 300
    Z = np.column_stack([np.ones(n), rng.normal(size=n)])
    D = (Z[:, 1] + rng.normal(size=n) > 0).astype(float)
    for m in ("logit", "probit", "linear"):
        p = _mte._fit_propensity(D, Z, model=m)
        assert np.all((p > 0) & (p < 1))


def test_empirical_cdf_weight_empty_sample():
    u = np.linspace(0.01, 0.99, 11)
    w = _mte._empirical_cdf_weight(u, np.array([]), side="lower")
    assert w.shape == u.shape
    np.testing.assert_allclose(w.sum(), 1.0)
    # non-empty 'upper'
    w2 = _mte._empirical_cdf_weight(u, np.linspace(0.1, 0.9, 50), side="upper")
    assert np.all(w2 >= 0)


def test_wald_tsls_helper():
    rng = np.random.default_rng(9)
    n = 400
    X = np.ones((n, 1))
    p = rng.uniform(0.1, 0.9, size=n)
    D = (p + rng.normal(0, 0.1, n) > 0.5).astype(float)
    Y = 1.0 + 2.0 * D + rng.normal(size=n)
    beta = _mte._wald_tsls(Y, D, X, p)
    assert np.isfinite(beta)


def test_mte_point_only_too_small_raises():
    rng = np.random.default_rng(10)
    n = 20
    Z = rng.normal(size=(n, 1))
    X = np.ones((n, 1))
    D = (Z[:, 0] > 0).astype(float)
    Y = 1.0 + D + rng.normal(size=n)
    u = np.linspace(0.02, 0.98, 51)
    with pytest.raises(ValueError):
        _mte._mte_point_only(
            Y, D, Z, X, K=2, u_grid=u, propensity_model="logit", trim=0.01
        )
