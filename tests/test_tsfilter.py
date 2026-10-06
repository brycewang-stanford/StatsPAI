"""``tsfilter``: defining properties of each filter and argument checks.
Cross-language numbers are in ``reference_parity/test_tsfilter_parity.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.timeseries.tsfilter import hp_smoothing, tsfilter


@pytest.fixture(scope="module")
def y() -> pd.Series:
    rng = np.random.default_rng(21)
    t = np.arange(200)
    v = (
        100
        + 0.4 * t
        + 2 * np.sin(2 * np.pi * t / 20)
        + np.cumsum(rng.normal(0, 0.2, 200))
    )
    return pd.Series(
        v, index=pd.period_range("1970Q1", periods=200, freq="Q"), name="gdp"
    )


@pytest.mark.parametrize("method", ["hp", "bk", "cf", "bw", "hamilton"])
def test_trend_plus_cycle_is_the_series(y: pd.Series, method: str) -> None:
    res = tsfilter(y, method=method)
    ok = res.cycle.notna()
    np.testing.assert_allclose((res.trend + res.cycle)[ok], y[ok], rtol=1e-13)
    assert res.cycle.index.equals(y.index)
    assert res.trend.isna().equals(res.cycle.isna())
    assert list(res.to_frame().columns) == ["observed", "trend", "cycle"]
    assert str(len(y)) in res.summary()


def test_hp_solves_its_first_order_condition(y: pd.Series) -> None:
    lam = 1600.0
    res = tsfilter(y, method="hp", smooth=lam)
    n = len(y)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i : i + 3] = [1.0, -2.0, 1.0]
    trend = np.linalg.solve(np.eye(n) + lam * D.T @ D, y.to_numpy())
    np.testing.assert_allclose(res.trend, trend, rtol=1e-10)
    # a straight line is its own trend
    line = tsfilter(3.0 + 0.5 * np.arange(50), method="hp")
    np.testing.assert_allclose(line.cycle, 0.0, atol=1e-9)


def test_hp_smoothing_rule() -> None:
    assert hp_smoothing("annual") == 6.25
    assert hp_smoothing("quarterly") == 1600.0
    assert hp_smoothing("monthly") == 129600.0
    assert hp_smoothing(4) == 1600.0
    with pytest.raises(MethodIncompatibility):
        hp_smoothing("weekly")
    with pytest.raises(MethodIncompatibility):
        hp_smoothing(0)


def test_bk_undefined_ends_and_zero_sum(y: pd.Series) -> None:
    res = tsfilter(y, method="bk", K=12)
    assert res.cycle.isna().to_numpy().nonzero()[0].tolist() == list(range(12)) + list(
        range(188, 200)
    )
    b = res.params["weights"]
    assert b[0] + 2 * b[1:].sum() == pytest.approx(0.0, abs=1e-15)
    # zero-sum symmetric weights annihilate a line
    line = tsfilter(5.0 + 0.3 * np.arange(80), method="bk")
    np.testing.assert_allclose(line.cycle.dropna(), 0.0, atol=1e-12)
    # the truncated ideal weights do not
    st = tsfilter(y, method="bk", stationary=True)
    assert abs(st.cycle.mean()) > 5 * abs(res.cycle.mean())


def test_band_pass_filters_keep_the_20_period_wave(y: pd.Series) -> None:
    t = np.arange(200)
    wave = 2 * np.sin(2 * np.pi * t / 20)
    for method in ("bk", "cf"):
        c = tsfilter(y, method=method).cycle.to_numpy()[30:170]
        assert np.corrcoef(c, wave[30:170])[0, 1] > 0.9


def test_cf_drift_removes_a_line(y: pd.Series) -> None:
    line = 2.0 + 0.7 * np.arange(60)
    np.testing.assert_allclose(tsfilter(line, method="cf").cycle, 0.0, atol=1e-10)
    off = tsfilter(line, method="cf", drift=False).cycle
    assert np.abs(off).max() > 0.1


def test_gain(y: pd.Series) -> None:
    w = np.array([0.0, 2 * np.pi / 32, 2 * np.pi / 16, np.pi])
    hp = tsfilter(y, method="hp").gain(w)
    a = 4 * 1600 * (1 - np.cos(w)) ** 2
    np.testing.assert_allclose(hp, a / (1 + a), rtol=1e-14)
    assert hp.iloc[0] == 0.0 and hp.iloc[-1] > 0.9999
    bk = tsfilter(y, method="bk").gain(w)
    assert bk.iloc[0] == pytest.approx(0.0, abs=1e-14)
    assert bk.iloc[2] > 0.9 and bk.iloc[3] < 0.05
    bw = tsfilter(y, method="bw", high=32).gain(w)
    np.testing.assert_allclose(bw, [0.0, 0.5, bw.iloc[2], 1.0], atol=1e-12)
    assert len(tsfilter(y, method="hamilton").gain()) == 201
    with pytest.raises(MethodIncompatibility):
        tsfilter(y, method="cf").gain(w)


def test_gain_is_what_the_filter_does_to_a_wave() -> None:
    # away from the ends the HP and BK cycles of a pure wave are the wave
    # times the gain
    t = np.arange(600)
    w = 2 * np.pi / 24
    wave = np.sin(w * t)
    for method, tol in (("hp", 1e-6), ("bk", 1e-12)):
        res = tsfilter(wave, method=method)
        g = float(res.gain([w]).iloc[0])
        mid = slice(250, 350)
        np.testing.assert_allclose(res.cycle.to_numpy()[mid], g * wave[mid], atol=tol)


def test_hamilton_layout(y: pd.Series) -> None:
    res = tsfilter(y, method="hamilton", h=8, p=4)
    assert int(res.cycle.isna().sum()) == 11
    assert res.params["coef"].shape == (5,)
    assert abs(res.cycle.mean()) < 1e-8  # OLS residuals with a constant


def test_missing_ends_come_back_as_nan(y: pd.Series) -> None:
    padded = pd.concat([pd.Series([np.nan, np.nan]), y.reset_index(drop=True)])
    padded = padded.reset_index(drop=True)
    a = tsfilter(padded, method="hp")
    b = tsfilter(y, method="hp")
    assert a.cycle.iloc[:2].isna().all()
    np.testing.assert_allclose(a.cycle.iloc[2:], b.cycle, rtol=1e-12)
    frame = pd.DataFrame({"g": y})
    np.testing.assert_allclose(tsfilter(frame, "g").cycle, b.cycle)


def test_plot(y: pd.Series) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = tsfilter(y.reset_index(drop=True), method="bk").plot()
    assert len(axes) == 2
    plt.close("all")


@pytest.mark.parametrize(
    "kw",
    [
        {"method": "stl"},
        {"method": "hp", "smooth": -1},
        {"method": "hp", "smooth": "weekly"},
        {"method": "bk", "low": 32, "high": 6},
        {"method": "bk", "low": 1},
        {"method": "bk", "K": 0},
        {"method": "bw", "order": 0},
        {"method": "bw", "high": 2},
        {"method": "hamilton", "h": 0},
    ],
)
def test_bad_arguments(y: pd.Series, kw: dict) -> None:
    with pytest.raises(MethodIncompatibility):
        tsfilter(y, **kw)


def test_bad_data(y: pd.Series) -> None:
    with pytest.raises(MethodIncompatibility):
        tsfilter(pd.DataFrame({"a": y}), "b")
    gap = y.copy()
    gap.iloc[50] = np.nan
    with pytest.raises(MethodIncompatibility):
        tsfilter(gap)
    with pytest.raises(DataInsufficient):
        tsfilter(y.iloc[:20], method="bk")
    with pytest.raises(DataInsufficient):
        tsfilter(y.iloc[:14], method="hamilton")
    with pytest.raises(DataInsufficient):
        tsfilter([1.0, 2.0, 3.0], method="hp")
    with pytest.raises(DataInsufficient):
        tsfilter([np.nan, np.nan])
