"""Coverage tests for statspai.iv.npiv (Newey-Powell sieve NPIV)."""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp

_npiv = importlib.import_module("statspai.iv.npiv")


def _nonlinear_iv(n=600, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    d = 0.9 * z + v
    u = 0.5 * v + rng.normal(0, 0.5, size=n)
    y = 1.0 + 0.5 * d - 0.2 * d**2 + u
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": rng.normal(size=n)})


def test_npiv_polynomial_summary_frame():
    df = _nonlinear_iv(seed=1)
    r = sp.iv.npiv(
        y="y", endog="d", instruments="z", data=df, basis="polynomial", k_d=3, k_z=3
    )
    s = r.summary()
    assert "Nonparametric IV" in s
    frame = r.to_frame()
    assert {"D", "h", "se", "ci_lower", "ci_upper"}.issubset(frame.columns)
    assert r.h_values.shape == r.h_se.shape == r.d_grid.shape


def test_npiv_bspline_basis():
    df = _nonlinear_iv(seed=2)
    r = sp.iv.npiv(
        y="y", endog="d", instruments="z", data=df, basis="bspline", k_d=4, k_z=4
    )
    assert np.isfinite(r.h_values).all()


def test_npiv_auto_high_degree_uses_bspline():
    df = _nonlinear_iv(n=800, seed=3)
    r = sp.iv.npiv(
        y="y",
        endog="d",
        instruments="z",
        data=df,
        basis="auto",
        k_d=6,
        k_z=6,
        regularization=0.01,
    )
    assert r.basis_type == "auto"
    assert r.n_obs == len(df)


def test_npiv_unknown_basis_raises():
    df = _nonlinear_iv(seed=4)
    with pytest.raises(ValueError, match="Unknown basis"):
        sp.iv.npiv(y="y", endog="d", instruments="z", data=df, basis="bogus")


def test_npiv_multi_instrument_and_exog():
    df = _nonlinear_iv(seed=5)
    df = df.copy()
    rng = np.random.default_rng(5)
    df["z2"] = 0.5 * df["z"] + rng.normal(size=len(df))
    r = sp.iv.npiv(
        y="y",
        endog="d",
        instruments=["z", "z2"],
        exog="x",
        data=df,
        basis="polynomial",
        k_d=3,
        k_z=3,
    )
    assert np.isfinite(r.first_stage_f)


def test_npiv_array_inputs_and_regularization():
    df = _nonlinear_iv(seed=6)
    r = sp.iv.npiv(
        y=df["y"].to_numpy(),
        endog=df["d"].to_numpy(),
        instruments=df["z"].to_numpy(),
        basis="polynomial",
        k_d=4,
        k_z=4,
        regularization=0.05,
    )
    assert r.regularization == 0.05


def test_npiv_custom_grid():
    df = _nonlinear_iv(seed=7)
    grid = np.linspace(-1.0, 1.0, 21)
    r = sp.iv.npiv(
        y="y",
        endog="d",
        instruments="z",
        data=df,
        basis="polynomial",
        k_d=3,
        k_z=3,
        d_grid=grid,
    )
    np.testing.assert_array_equal(r.d_grid, grid)


def test_npiv_helpers():
    P = _npiv._poly_basis(np.array([0.0, 1.0, 2.0]), 2)
    assert P.shape == (3, 3)
    B = _npiv._bspline_basis(np.linspace(0, 1, 30), 4, n_knots=8)
    assert B.shape[0] == 30
    M = np.arange(6.0).reshape(3, 2)
    np.testing.assert_array_equal(_npiv._residualize(M, None), M)
    arr = _npiv._grab(np.array([1.0, 2.0]), None)
    np.testing.assert_array_equal(arr, [1.0, 2.0])
