"""Coverage tests for statspai.rd.bandwidth (rdbwselect).

Exercises all eight MSE/CER bandwidth selectors, fuzzy / deriv / covs /
cluster variance paths, and the all=True grid. Real synthetic RD data;
bandwidth positivity and CER<=MSE properties asserted.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_sharp(n=2500, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    Z = rng.normal(0, 1, n)
    Y = 0.5 * X + 3.0 * (X >= 0) + 0.3 * Z + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": Y, "x": X, "z": Z})


def _make_fuzzy(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    prob = 0.15 + 0.7 * (X >= 0)
    D = (rng.uniform(0, 1, n) < prob).astype(float)
    Y = 0.5 * X + 2.0 * D + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": Y, "x": X, "d": D})


@pytest.mark.parametrize("bw", ["mserd", "msetwo", "cerrd", "certwo",
                                "msecomb1", "msecomb2", "cercomb1", "cercomb2"])
def test_rdbwselect_all_methods(bw):
    df = _make_sharp()
    out = sp.rdbwselect(df, y="y", x="x", c=0, bwselect=bw)
    assert isinstance(out, pd.DataFrame)
    assert (out["h_left"] > 0).all()
    assert (out["h_right"] > 0).all()


def test_rdbwselect_all_grid():
    df = _make_sharp()
    out = sp.rdbwselect(df, y="y", x="x", c=0, all=True)
    assert len(out) >= 8


def test_rdbwselect_fuzzy():
    df = _make_fuzzy()
    out = sp.rdbwselect(df, y="y", x="x", c=0, fuzzy="d")
    assert (out["h_left"] > 0).all()


def test_rdbwselect_deriv_rkd():
    df = _make_sharp()
    out = sp.rdbwselect(df, y="y", x="x", c=0, deriv=1, p=2)
    assert (out["h_left"] > 0).all()


def test_rdbwselect_covs():
    df = _make_sharp()
    out = sp.rdbwselect(df, y="y", x="x", c=0, covs=["z"])
    assert (out["h_left"] > 0).all()


def test_rdbwselect_cluster():
    df = _make_sharp()
    df["g"] = (np.arange(len(df)) // 25).astype(int)
    out = sp.rdbwselect(df, y="y", x="x", c=0, cluster="g")
    assert (out["h_left"] > 0).all()


def test_rdbwselect_cer_not_larger_than_mse():
    df = _make_sharp()
    mse = sp.rdbwselect(df, y="y", x="x", c=0, bwselect="mserd")["h_left"].iloc[0]
    cer = sp.rdbwselect(df, y="y", x="x", c=0, bwselect="cerrd")["h_left"].iloc[0]
    assert cer <= mse * 1.05


def test_rdbwselect_epanechnikov_kernel():
    df = _make_sharp()
    out = sp.rdbwselect(df, y="y", x="x", c=0, kernel="epanechnikov")
    assert (out["h_left"] > 0).all()
