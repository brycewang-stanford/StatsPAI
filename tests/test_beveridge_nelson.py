"""``sp.beveridge_nelson``: analytic cases, identities and argument checks.

The brute-force check in a second language is in
``tests/reference_parity/test_beveridge_nelson_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.timeseries.beveridge_nelson import BeveridgeNelsonResult, beveridge_nelson


def _arima(phi, n: int = 300, drift: float = 0.3, seed: int = 0) -> np.ndarray:
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    p = phi.size
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + 100)
    dy = np.zeros(n + 100)
    c = drift * (1 - phi.sum())
    for t in range(p, n + 100):
        dy[t] = c + float(phi @ dy[t - p : t][::-1]) + e[t]
    return 50.0 + np.cumsum(dy[100:])


def _brute_trend(y: np.ndarray, bn: BeveridgeNelsonResult, t: int, h: int) -> float:
    """Forecast of ``y[t+h]`` made at ``t``, net of ``h`` drifts."""
    p = bn.order
    dy = np.diff(y)
    hist = list(dy[t - p : t][::-1])  # dy[t-1] is the change ending at y[t]
    level = y[t]
    for _ in range(h):
        nxt = bn.intercept + float(bn.ar_coefs @ np.array(hist[:p]))
        level += nxt
        hist.insert(0, nxt)
    return level - h * bn.drift


def test_ar1_cycle_is_the_textbook_formula():
    y = _arima(0.5)
    bn = beveridge_nelson(y, order=1)
    phi = float(bn.ar_coefs[0])
    dy = np.diff(y)
    expected = -phi / (1 - phi) * (dy - bn.drift)
    # same expression, evaluated two ways
    assert np.allclose(bn.cycle.to_numpy()[1:], expected, rtol=0, atol=1e-12)
    assert np.isnan(bn.cycle.iloc[0]) and np.isnan(bn.trend.iloc[0])
    assert bn.long_run_multiplier == pytest.approx(1 / (1 - phi))
    assert bn.drift == pytest.approx(bn.intercept / (1 - phi))
    # psi(1)^2 / sum(psi_j^2) = (1 - phi^2) / (1 - phi)^2 for an AR(1)
    assert bn.variance_ratio == pytest.approx((1 + phi) / (1 - phi))
    assert 0.35 < phi < 0.65


@pytest.mark.parametrize("phi", [[0.5], [0.4, -0.3], [0.2, 0.1, -0.2, 0.3]])
def test_closed_form_equals_long_horizon_forecast(phi):
    y = _arima(phi, seed=1)
    bn = beveridge_nelson(y, order=len(phi))
    for t in (len(phi), 57, 150, y.size - 1):
        brute = _brute_trend(y, bn, t, 600)
        # geometric convergence: 600 steps leave no truncation error
        assert bn.trend.iloc[t] == pytest.approx(brute, rel=1e-10)


def test_random_walk_with_drift_has_no_cycle():
    rng = np.random.default_rng(2)
    y = np.cumsum(0.4 + rng.normal(size=200))
    bn = beveridge_nelson(y, order=0)
    assert (bn.cycle == 0.0).all()
    assert np.array_equal(bn.trend.to_numpy(), y)
    assert bn.drift == pytest.approx(np.diff(y).mean())
    assert bn.long_run_multiplier == 1.0 and bn.variance_ratio == 1.0
    assert bn.ar_coefs.size == 0
    # BIC finds the random walk, and the cycle is then identically zero
    auto = beveridge_nelson(y)
    assert auto.order == 0 and auto.ic == "bic"
    assert (auto.cycle == 0.0).all()


def test_trend_is_a_random_walk_driven_by_the_innovations():
    y = _arima([0.4, -0.3], seed=3)
    bn = beveridge_nelson(y, order=2)
    step = bn.trend.diff().to_numpy()[3:]
    resid = bn.residuals.to_numpy()[3:]
    assert np.allclose(step, bn.drift + bn.long_run_multiplier * resid, atol=1e-10)
    assert np.allclose((bn.trend + bn.cycle).to_numpy()[2:], y[2:])
    assert abs(bn.cycle.mean()) < 0.2
    assert bn.n_obs == y.size - 1 - 2


def test_order_selection_on_a_common_sample():
    y = _arima([0.4, -0.3], n=600, seed=4)
    bn = beveridge_nelson(y, max_order=6)
    assert bn.order == 2 and bn.ic == "bic"
    assert list(bn.selection.index) == list(range(7))
    assert bn.selection["bic"].idxmin() == 2
    dy = np.diff(y)
    target = dy[6:]
    design = np.column_stack([np.ones(target.size), dy[5:-1], dy[4:-2]])
    rss = float(np.sum((target - design @ np.linalg.lstsq(design, target)[0]) ** 2))
    m = target.size
    assert bn.selection.loc[2, "aic"] == pytest.approx(m * np.log(rss / m) + 6)
    assert bn.selection.loc[2, "bic"] == pytest.approx(
        m * np.log(rss / m) + 3 * np.log(m)
    )
    # the chosen order is refitted on every difference it can use
    assert np.allclose(bn.ar_coefs, beveridge_nelson(y, order=2).ar_coefs)
    assert beveridge_nelson(y, max_order=6, ic="aic").order >= 2


def test_index_and_names_follow_the_input():
    y = _arima(0.5, n=80, seed=5)
    idx = pd.period_range("2000Q1", periods=80, freq="Q")
    frame = pd.DataFrame({"lgdp": y}, index=idx)
    bn = beveridge_nelson("lgdp", data=frame, order=1)
    assert bn.trend.index.equals(idx) and bn.cycle.name == "lgdp_cycle"
    same = beveridge_nelson(frame["lgdp"], order=1)
    assert np.allclose(bn.cycle.dropna(), same.cycle.dropna())
    assert bn.to_dict()["order"] == 1


def test_summary_and_plot():
    y = _arima(0.5, n=120, seed=6)
    bn = beveridge_nelson(y)
    text = bn.summary()
    assert "psi(1)" in text and "chosen by BIC" in text
    assert "fixed" in beveridge_nelson(y, order=1).summary()
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    axes = bn.plot()
    assert len(axes) == 2
    matplotlib.pyplot.close("all")


def test_non_stationary_difference_is_refused():
    rng = np.random.default_rng(7)
    dy = np.zeros(60)
    for t in range(1, 60):
        dy[t] = 1.08 * dy[t - 1] + rng.normal()
    with pytest.raises(MethodIncompatibility, match="not stationary"):
        beveridge_nelson(np.cumsum(dy), order=1)


def test_bad_arguments_and_data():
    y = _arima(0.5, n=60, seed=8)
    with pytest.raises(MethodIncompatibility, match="ic="):
        beveridge_nelson(y, ic="hq")
    with pytest.raises(MethodIncompatibility, match="negative"):
        beveridge_nelson(y, order=-1)
    with pytest.raises(MethodIncompatibility, match="not a column"):
        beveridge_nelson("y", data=pd.DataFrame({"z": y}))
    gap = y.copy()
    gap[10] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing"):
        beveridge_nelson(gap)
    with pytest.raises(DataInsufficient):
        beveridge_nelson(y[:5])
    with pytest.raises(DataInsufficient, match="order=30"):
        beveridge_nelson(y, order=30)
    with pytest.raises(DataInsufficient, match="max_order"):
        beveridge_nelson(y, max_order=40)
