"""``sp.xcorr``: definition, sign convention, Haugh test and argument checks.

Agreement with R ``ccf`` / ``ar`` and Stata ``xcorr`` is in
``tests/reference_parity/test_xcorr_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.timeseries.xcorr import CrossCorrelogram, xcorr


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(size=n + 100)
    x = np.zeros(n + 100)
    for t in range(1, n + 100):
        x[t] = phi * x[t - 1] + e[t]
    return x[100:]


def _lead_pair(n: int = 300, seed: int = 0):
    """``y`` leads ``x`` by three periods."""
    rng = np.random.default_rng(seed)
    y = _ar1(n + 3, 0.6, rng)
    x = y[:-3] + 0.5 * rng.normal(size=n)
    return x, y[3:]


def test_definition_at_each_lag():
    x, y = _lead_pair(60)
    cc = xcorr(x, y, lags=4)
    dx, dy = x - x.mean(), y - y.mean()
    scale = np.sqrt((dx @ dx) * (dy @ dy))
    assert cc.table.loc[0, "xcorr"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    # h = 2: x two periods after y
    assert cc.table.loc[2, "xcorr"] == pytest.approx(float(dx[2:] @ dy[:-2]) / scale)
    assert cc.table.loc[-3, "xcorr"] == pytest.approx(float(dx[:-3] @ dy[3:]) / scale)
    assert list(cc.table.index) == list(range(-4, 5))
    assert cc.n_obs == 60


def test_positive_lag_means_y_leads():
    x, y = _lead_pair()
    for kwargs in (dict(), dict(prewhiten="ar"), dict(prewhiten="x")):
        cc = xcorr(x, y, lags=8, **kwargs)
        assert int(cc.table["xcorr"].abs().idxmax()) == 3
        assert bool(cc.table.loc[3, "outside"])


def test_swapping_the_series_mirrors_the_lags():
    x, y = _lead_pair(120, seed=1)
    a = xcorr(x, y, lags=6).table["xcorr"].to_numpy()
    b = xcorr(y, x, lags=6).table["xcorr"].to_numpy()
    assert np.allclose(a, b[::-1], atol=1e-14)


def test_band_and_alpha():
    x, y = _lead_pair(100, seed=2)
    cc = xcorr(x, y, lags=3)
    assert cc.table["upper"].iloc[0] == pytest.approx(1.959963984540054 / 10)
    assert (cc.table["lower"] == -cc.table["upper"]).all()
    wide = xcorr(x, y, lags=3, alpha=0.01)
    assert wide.table["upper"].iloc[0] == pytest.approx(2.5758293035489 / 10)


def test_haugh_statistics_are_the_sums_they_say():
    x, y = _lead_pair(150, seed=3)
    cc = xcorr(x, y, lags=5, prewhiten="ar", ar_order=2)
    rho = cc.table["xcorr"].to_numpy()
    n = cc.n_obs
    assert n == 148
    assert cc.haugh["statistic"] == pytest.approx(n * (rho**2).sum())
    h = np.abs(cc.table.index.to_numpy())
    assert cc.haugh["statistic_adj"] == pytest.approx(n**2 * (rho**2 / (n - h)).sum())
    assert cc.haugh["df"] == 11 and cc.haugh["lags"] == 5
    assert cc.haugh["statistic_adj"] > cc.haugh["statistic"]
    assert cc.haugh["pvalue"] < 1e-6
    assert xcorr(x, y, lags=5).haugh == {}


def test_haugh_size_for_unrelated_autocorrelated_series():
    """Raw correlations of unrelated AR(1) series leave the band far too
    often; after prewhitening the test has about its nominal size."""
    rng = np.random.default_rng(4)
    reps, reject, raw_out = 400, 0, 0.0
    for _ in range(reps):
        x = _ar1(200, 0.8, rng)
        y = _ar1(200, 0.8, rng)
        cc = xcorr(x, y, lags=6, prewhiten="ar", ar_order=1)
        reject += cc.haugh["pvalue_adj"] < 0.05
        raw_out += float(xcorr(x, y, lags=6).table["outside"].mean())
    # binomial se at 0.05 with 400 draws is 0.011: a 3.5 se window
    assert 0.012 < reject / reps < 0.09
    # share of raw correlations outside a nominal 5% band
    assert raw_out / reps > 0.25


def test_residuals_are_ols_ar_residuals():
    x, y = _lead_pair(90, seed=5)
    cc = xcorr(x, y, lags=4, prewhiten="ar", ar_order=(2, 1))
    design = np.column_stack([np.ones(88), x[1:-1], x[:-2]])
    beta = np.linalg.lstsq(design, x[2:], rcond=None)[0]
    resid = x[2:] - design @ beta
    assert np.allclose(cc.ar_coefs["x"], beta[1:])
    assert np.allclose(cc.residuals.iloc[:, 0], resid - resid.mean())
    assert cc.ar_orders == {"x": 2, "y": 1} and cc.n_obs == 88


def test_dataframe_input_trims_missing_ends():
    x, y = _lead_pair(80, seed=6)
    frame = pd.DataFrame({"gdp": x, "sentiment": y})
    full = xcorr("gdp", "sentiment", data=frame.iloc[5:70], lags=4)
    frame.loc[:4, "gdp"] = np.nan
    frame.loc[70:, "sentiment"] = np.nan
    cc = xcorr("gdp", "sentiment", data=frame, lags=4)
    assert cc.n_obs == 65 and cc.names == ("gdp", "sentiment")
    assert np.allclose(cc.table["xcorr"], full.table["xcorr"])
    frame.loc[30, "gdp"] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing values between"):
        xcorr("gdp", "sentiment", data=frame)


def test_summary_and_plot():
    x, y = _lead_pair(120, seed=7)
    cc = xcorr(x, y, lags=4, prewhiten="ar", ar_order=1)
    assert isinstance(cc, CrossCorrelogram)
    text = cc.summary()
    assert "y leads x" in text and "Haugh" in text and "AR(1)" in text
    assert "both filtered" in xcorr(x, y, lags=4, prewhiten="x").summary()
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = cc.plot()
    assert "leads" in ax.get_xlabel()
    matplotlib.pyplot.close("all")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(prewhiten="arma"), "prewhiten"),
        (dict(prewhiten="ar", ar_method="burg"), "ar_method"),
        (dict(prewhiten="ar", ic="hq"), "ic="),
        (dict(alpha=1.5), "alpha"),
        (dict(prewhiten="ar", ar_order=-1), "negative"),
    ],
)
def test_bad_arguments(kwargs, match):
    x, y = _lead_pair(100, seed=8)
    with pytest.raises(MethodIncompatibility, match=match):
        xcorr(x, y, **kwargs)


def test_bad_data():
    x, y = _lead_pair(100, seed=9)
    with pytest.raises(MethodIncompatibility, match="observations"):
        xcorr(x, y[:-1])
    with pytest.raises(MethodIncompatibility, match="not a column"):
        xcorr("a", "b", data=pd.DataFrame({"a": x}))
    with pytest.raises(DataInsufficient, match="lags"):
        xcorr(x, y, lags=99)
    with pytest.raises(DataInsufficient, match="constant"):
        xcorr(np.ones(50), y[:50])
    with pytest.raises(DataInsufficient, match="too few"):
        xcorr(x[:20], y[:20], prewhiten="ar", ar_order=8)
    with pytest.raises(DataInsufficient, match="no period"):
        xcorr(np.full(10, np.nan), y[:10])
