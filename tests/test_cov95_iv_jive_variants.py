"""Coverage tests for statspai.iv.jive_variants internal branches."""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp

_jv = importlib.import_module("statspai.iv.jive_variants")


def _iv_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 4))
    v = rng.normal(size=n)
    d = z @ np.array([0.6, 0.4, 0.3, 0.2]) + v
    u = 0.5 * v + rng.normal(0, 0.5, size=n)
    y = 1.0 + 2.0 * d + u
    cols = {f"z{i}": z[:, i] for i in range(4)}
    return pd.DataFrame({"y": y, "d": d, "x": rng.normal(size=n), **cols})


def test_jive1_summary_and_frame():
    df = _iv_df(seed=1)
    r = sp.iv.jive1(
        y="y", endog="d", instruments=["z0", "z1", "z2", "z3"], exog="x", data=df
    )
    s = r.summary()
    assert "JIVE1" in s
    assert "first-stage F" in s
    # endog string -> generic name "endog0"; just check the coef is finite
    assert np.isfinite(r.params.iloc[0])


def test_ujive_with_exog():
    df = _iv_df(seed=2)
    r = sp.iv.ujive(
        y="y", endog="d", instruments=["z0", "z1", "z2", "z3"], exog="x", data=df
    )
    assert np.isfinite(r.params.iloc[0])


def test_ujive_no_exog_no_const():
    df = _iv_df(seed=3)
    # no exog and no const -> the W.size == 0 ujive branch (183-185)
    r = sp.iv.ujive(
        y="y", endog="d", instruments=["z0", "z1", "z2", "z3"], data=df, add_const=False
    )
    assert np.isfinite(r.params.iloc[0])


def test_jive1_array_inputs_1d_instrument():
    df = _iv_df(seed=20)
    Y = df["y"].to_numpy()
    D = df["d"].to_numpy()
    Z = df["z0"].to_numpy()  # 1-D array -> reshape branch (96) + grab array (88)
    r = sp.iv.jive1(y=Y, endog=D, instruments=Z)
    assert np.isfinite(r.params.iloc[0])


def test_ijive_runs():
    df = _iv_df(seed=4)
    r = sp.iv.ijive(
        y="y", endog="d", instruments=["z0", "z1", "z2", "z3"], exog="x", data=df
    )
    assert np.isfinite(r.params.iloc[0])


def test_rjive_runs():
    df = _iv_df(seed=5)
    r = sp.iv.rjive(
        y="y",
        endog="d",
        instruments=["z0", "z1", "z2", "z3"],
        exog="x",
        data=df,
        ridge=0.5,
    )
    assert "RJIVE" in r.method


def test_rjive_zero_ridge_raises():
    df = _iv_df(seed=6)
    Y = df["y"].to_numpy()
    D = df["d"].to_numpy()
    Z = df[["z0", "z1"]].to_numpy()
    W = np.ones((len(df), 1))
    with pytest.raises(ValueError, match="ridge > 0"):
        _jv._jive_estimate(Y, D.reshape(-1, 1), Z, W, "rjive", ridge=0.0)


def test_jive_estimate_unknown_method_raises():
    df = _iv_df(seed=7)
    Y = df["y"].to_numpy()
    D = df["d"].to_numpy().reshape(-1, 1)
    Z = df[["z0", "z1"]].to_numpy()
    W = np.ones((len(df), 1))
    with pytest.raises(ValueError, match="Unknown JIVE method"):
        _jv._jive_estimate(Y, D, Z, W, "nope")


def test_jive_helpers():
    assert _jv._as_matrix(np.array([1.0, 2.0])).shape == (2, 1)
    # _names variants
    s = pd.Series([1.0], name="foo")
    assert _jv._names(s, "p", 1) == ["foo"]
    s2 = pd.Series([1.0])
    assert _jv._names(s2, "p", 1) == ["p0"]
    assert _jv._names(["a", "b"], "x", 2) == ["a", "b"]
    df = pd.DataFrame({"q": [1], "r": [2]})
    assert _jv._names(df, "x", 2) == ["q", "r"]
    assert _jv._names(np.zeros((3, 2)), "z", 2) == ["z0", "z1"]


def test_first_stage_f_helpers():
    rng = np.random.default_rng(8)
    n = 300
    Z = rng.normal(size=(n, 3))
    D = (Z @ np.array([0.5, 0.3, 0.2]) + rng.normal(size=n)).reshape(-1, 1)
    # no-W branch (rss_r via demeaning, line 130)
    f_noW = _jv._first_stage_f(D, Z, np.empty((n, 0)))
    assert np.isfinite(f_noW) and f_noW > 0
    # empty D -> nan (line 121)
    assert np.isnan(_jv._first_stage_f(np.empty((n, 0)), Z, np.empty((n, 0))))
