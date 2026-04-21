"""Smoke tests for sp.bridge() — six bridging theorems."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def small_panel():
    rng = np.random.default_rng(0)
    n_units, n_time = 8, 10
    units = []
    for u in range(n_units):
        treated = u == 0
        for t in range(n_time):
            post = t >= 5
            y = (
                u * 0.5
                + t * 0.2
                + (3.0 if treated and post else 0.0)
                + rng.standard_normal()
            )
            units.append(
                {"unit": f"u{u}", "time": t, "y": y, "treated": int(treated)}
            )
    return pd.DataFrame(units)


@pytest.fixture
def cross_section():
    rng = np.random.default_rng(1)
    n = 400
    X = rng.standard_normal((n, 3))
    ps = 1.0 / (1.0 + np.exp(-X[:, 0]))
    D = (rng.uniform(size=n) < ps).astype(int)
    Y = X[:, 0] + X[:, 1] + 2.0 * D + rng.standard_normal(n)
    return pd.DataFrame({
        "y": Y, "treat": D,
        "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2],
    })


def test_bridge_did_sc(small_panel):
    res = sp.bridge(
        kind="did_sc", data=small_panel,
        y="y", unit="unit", time="time",
        treated_unit="u0", treatment_time=5,
    )
    assert isinstance(res, sp.BridgeResult)
    assert res.kind == "did_sc"
    # Both paths should be in the right ballpark (true effect = 3.0)
    assert 0.5 < res.estimate_dr < 5.5
    assert "AGREE" in res.summary() or "DISAGREE" in res.summary()


def test_bridge_cb_ipw(cross_section):
    res = sp.bridge(
        kind="cb_ipw", data=cross_section,
        y="y", treat="treat", covariates=["x1", "x2", "x3"],
        n_boot=50,
    )
    assert isinstance(res, sp.BridgeResult)
    # True ATE = 2.0
    assert 1.0 < res.estimate_dr < 3.0


def test_bridge_ewm_cate(cross_section):
    res = sp.bridge(
        kind="ewm_cate", data=cross_section,
        y="y", treat="treat", covariates=["x1", "x2", "x3"],
        n_boot=50,
    )
    assert isinstance(res, sp.BridgeResult)
    assert np.isfinite(res.estimate_dr)


def test_bridge_dr_calib(cross_section):
    res = sp.bridge(
        kind="dr_calib", data=cross_section,
        y="y", treat="treat", covariates=["x1", "x2", "x3"],
        n_boot=50,
    )
    assert isinstance(res, sp.BridgeResult)
    assert 1.0 < res.estimate_dr < 3.0


def test_bridge_surrogate_pci(cross_section):
    df = cross_section.copy()
    df["s1"] = df["y"] * 0.6 + np.random.default_rng(2).standard_normal(len(df))
    res = sp.bridge(
        kind="surrogate_pci", data=df,
        long_term="y", short_term=["s1"], treat="treat",
        covariates=["x1", "x2", "x3"], n_boot=50,
    )
    assert isinstance(res, sp.BridgeResult)
    assert np.isfinite(res.estimate_dr)


def test_bridge_kink_rdd():
    rng = np.random.default_rng(7)
    n = 800
    R = rng.uniform(-1, 1, size=n)
    Y = np.where(R > 0, 1.5 * R, 0.5 * R) + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({"y": Y, "r": R})
    res = sp.bridge(
        kind="kink_rdd", data=df,
        y="y", running="r", cutoff=0.0, bandwidth=0.6,
    )
    assert isinstance(res, sp.BridgeResult)
    # RKD slope-jump should be near 1.0 (= 1.5 - 0.5)
    assert 0.3 < abs(res.estimate_a) < 2.0


def test_bridge_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown bridge"):
        sp.bridge(kind="not_a_bridge", data=pd.DataFrame())
