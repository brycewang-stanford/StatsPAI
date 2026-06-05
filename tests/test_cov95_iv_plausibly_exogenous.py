"""Coverage tests for statspai.iv.plausibly_exogenous (CHR 2012 UCI / LTZ)."""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp

_pe = importlib.import_module("statspai.iv.plausibly_exogenous")


def _iv_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    z2 = rng.normal(size=n)
    v = rng.normal(size=n)
    d = 0.8 * z + 0.4 * z2 + v
    u = 0.5 * v + rng.normal(0, 0.5, size=n)
    y = 1.0 + 2.0 * d + u
    return pd.DataFrame({"y": y, "d": d, "z": z, "z2": z2,
                         "x": rng.normal(size=n)})


def test_as_matrix():
    assert _pe._as_matrix(np.array([1.0, 2.0])).shape == (2, 1)
    assert _pe._as_matrix(np.zeros((2, 2))).shape == (2, 2)


def test_uci_single_instrument_summary():
    df = _iv_df(seed=1)
    grid = np.linspace(-0.2, 0.2, 21)
    r = sp.iv.plausibly_exogenous_uci(y="y", endog="d", instruments="z",
                                      gamma_grid=grid, data=df)
    assert r.ci_lower <= r.ci_upper
    s = r.summary()
    assert "UCI" in s
    assert "Point estimate" in s
    assert r.beta_at_gamma.shape == (21,)


def test_uci_multi_instrument_flat_grid_reshape():
    df = _iv_df(seed=2)
    # 2 instruments; flat grid of length divisible by 2 -> reshaped (m, 2)
    grid = np.array([-0.1, -0.1, 0.0, 0.0, 0.1, 0.1])
    r = sp.iv.plausibly_exogenous_uci(
        y="y", endog="d", instruments=["z", "z2"], gamma_grid=grid, data=df,
    )
    assert r.gamma_grid.shape == (3, 2)


def test_uci_multi_instrument_bad_grid_length_raises():
    df = _iv_df(seed=3)
    with pytest.raises(ValueError, match="not divisible"):
        sp.iv.plausibly_exogenous_uci(
            y="y", endog="d", instruments=["z", "z2"],
            gamma_grid=np.array([0.0, 0.1, 0.2]), data=df,
        )


def test_uci_with_exog():
    df = _iv_df(seed=4)
    r = sp.iv.plausibly_exogenous_uci(
        y="y", endog="d", instruments="z", gamma_grid=np.linspace(-0.1, 0.1, 11),
        exog="x", data=df,
    )
    assert np.isfinite(r.beta_hat)


def test_ltz_scalar_var_summary_has_prior():
    df = _iv_df(seed=5)
    r = sp.iv.plausibly_exogenous_ltz(y="y", endog="d", instruments="z",
                                      gamma_mean=0.0, gamma_var=0.01, data=df)
    s = r.summary()
    assert "LTZ" in s
    assert "prior mean" in s
    assert "prior variance" in s
    # nonzero prior variance inflates SE relative to 2SLS
    assert r.se_at_gamma[0] >= r.se_hat - 1e-9


def test_ltz_zero_var_reproduces_2sls():
    df = _iv_df(seed=6)
    r = sp.iv.plausibly_exogenous_ltz(y="y", endog="d", instruments="z",
                                      gamma_var=0.0, data=df)
    # gamma_var=0 -> LTZ == 2SLS point/SE
    np.testing.assert_allclose(r.beta_at_gamma[0], r.beta_hat, atol=1e-9)
    np.testing.assert_allclose(r.se_at_gamma[0], r.se_hat, atol=1e-9)


def test_ltz_multi_instrument_diag_var():
    df = _iv_df(seed=7)
    # 1d gamma_var -> diagonal Omega
    r = sp.iv.plausibly_exogenous_ltz(
        y="y", endog="d", instruments=["z", "z2"],
        gamma_mean=np.array([0.0, 0.0]), gamma_var=np.array([0.01, 0.02]),
        data=df,
    )
    assert r.extra["gamma_var"].shape == (2, 2)


def test_ltz_matrix_var():
    df = _iv_df(seed=8)
    Omega = np.array([[0.01, 0.001], [0.001, 0.02]])
    r = sp.iv.plausibly_exogenous_ltz(
        y="y", endog="d", instruments=["z", "z2"],
        gamma_mean=np.array([0.0, 0.0]), gamma_var=Omega, data=df,
    )
    np.testing.assert_allclose(r.extra["gamma_var"], Omega)


def test_ltz_gamma_mean_wrong_length_raises():
    df = _iv_df(seed=9)
    with pytest.raises(ValueError, match="gamma_mean length"):
        sp.iv.plausibly_exogenous_ltz(
            y="y", endog="d", instruments="z",
            gamma_mean=np.array([0.0, 0.0]), gamma_var=0.0, data=df,
        )


def test_ltz_gamma_var_1d_wrong_length_raises():
    df = _iv_df(seed=10)
    with pytest.raises(ValueError, match="gamma_var length"):
        sp.iv.plausibly_exogenous_ltz(
            y="y", endog="d", instruments="z",
            gamma_var=np.array([0.01, 0.02]), data=df,
        )


def test_ltz_gamma_var_matrix_wrong_shape_raises():
    df = _iv_df(seed=11)
    with pytest.raises(ValueError, match="must be"):
        sp.iv.plausibly_exogenous_ltz(
            y="y", endog="d", instruments=["z", "z2"],
            gamma_var=np.eye(3), data=df,
        )


def test_prep_array_inputs():
    df = _iv_df(seed=12)
    Y, D, Z, W = _pe._prep(
        df["y"].to_numpy(), df["d"].to_numpy(), df["z"].to_numpy(),
        None, None, add_const=True,
    )
    assert D.shape[1] == 1 and Z.shape[1] == 1 and W.shape[1] == 1
