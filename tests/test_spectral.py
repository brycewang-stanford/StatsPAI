"""``periodogram`` and ``cumulative_periodogram_test``: definitions,
identities and argument checks. Cross-language numbers are in
``reference_parity/test_spectral_parity.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.timeseries.spectral import cumulative_periodogram_test, periodogram


@pytest.fixture(scope="module")
def x() -> np.ndarray:
    rng = np.random.default_rng(11)
    e = rng.normal(size=260)
    out = np.zeros(260)
    for t in range(2, 260):
        out[t] = 1.2 * out[t - 1] - 0.6 * out[t - 2] + e[t]
    return out[60:]


def test_textbook_periodogram_by_definition(x: np.ndarray) -> None:
    n = x.size
    res = periodogram(x, taper=0, detrend="mean", scale="radians")
    t = np.arange(1, n + 1)
    for j in (1, 7, n // 2):
        w = 2 * np.pi * j / n
        direct = abs(np.sum(x * np.exp(-1j * w * t))) ** 2 / (2 * np.pi * n)
        row = res.table.iloc[j - 1]
        assert row["omega"] == pytest.approx(w, rel=1e-14)
        assert row["cycle_length"] == pytest.approx(n / j, rel=1e-14)
        assert row["spectrum"] == pytest.approx(direct, rel=1e-11)
    assert res.df == 2.0


def test_mean_does_not_matter_at_fourier_frequencies(x: np.ndarray) -> None:
    a = periodogram(x + 50.0, taper=0, detrend="none").table["spectrum"]
    b = periodogram(x, taper=0, detrend="mean").table["spectrum"]
    np.testing.assert_allclose(a, b, rtol=1e-8)


def test_scales_differ_by_two_pi(x: np.ndarray) -> None:
    for kw in ({}, {"method": "smoothed", "spans": 5}, {"method": "ar"}):
        a = periodogram(x, scale="cycles", **kw)
        b = periodogram(x, scale="radians", **kw)
        np.testing.assert_allclose(
            a.table["spectrum"], 2 * np.pi * b.table["spectrum"], rtol=1e-14
        )
        if a.bandwidth is not None:
            assert a.bandwidth == b.bandwidth


def test_periodogram_adds_up_to_the_variance(x: np.ndarray) -> None:
    # Parseval: with even n the ordinates at j/n, j = 1..n/2, hold all of
    # the sum of squares, the last one (frequency 1/2) counted once
    spec = periodogram(x, taper=0, detrend="mean").table["spectrum"].to_numpy()
    total = (2 * spec[:-1].sum() + spec[-1]) / x.size
    assert total == pytest.approx(np.var(x), rel=1e-12)


def test_scipy_periodogram(x: np.ndarray) -> None:
    from scipy.signal import periodogram as sp_pgram

    f, p = sp_pgram(x, detrend="constant", scaling="density", return_onesided=False)
    res = periodogram(x, taper=0, detrend="mean")
    k = len(res.table)
    np.testing.assert_allclose(res.table["freq"], np.abs(f[1 : k + 1]), rtol=1e-14)
    np.testing.assert_allclose(res.table["spectrum"], p[1 : k + 1], rtol=1e-10)


def test_smoothing_is_a_weighted_average(x: np.ndarray) -> None:
    raw = periodogram(x, taper=0, detrend="mean").table["spectrum"].to_numpy()
    sm = periodogram(x, method="smoothed", spans=3, taper=0, detrend="mean")
    j = 20
    by_hand = 0.25 * raw[j - 1] + 0.5 * raw[j] + 0.25 * raw[j + 1]
    assert sm.table["spectrum"].iloc[j] == pytest.approx(by_hand, rel=1e-12)
    assert sm.df == pytest.approx(2 / (0.25**2 * 2 + 0.5**2))
    wide = periodogram(x, method="smoothed", spans=(7, 7))
    assert wide.df > sm.df and wide.bandwidth > sm.bandwidth
    assert (wide.table["lower"] < wide.table["spectrum"]).all()
    assert (wide.table["upper"] > wide.table["spectrum"]).all()


def test_ar_spectrum_recovers_the_cycle(x: np.ndarray) -> None:
    res = periodogram(x, method="ar", order=2)
    # true peak of 1.2, -0.6: cos(w) = 1.2 * 1.6 / (4 * 0.6)
    true_w = np.arccos(1.2 * 1.6 / 2.4)
    assert res.peak["freq"] * 2 * np.pi == pytest.approx(true_w, abs=0.1)
    np.testing.assert_allclose(res.ar_coef, [1.2, -0.6], atol=0.15)
    assert "lower" not in res.table.columns
    assert res.df is None and res.bandwidth is None
    # integrates to the variance of the fitted process
    area = 2 * np.trapezoid(res.table["spectrum"], res.table["freq"])
    assert area == pytest.approx(np.var(x), rel=0.05)


def test_input_forms_and_edges(x: np.ndarray) -> None:
    frame = pd.DataFrame({"g": np.r_[np.nan, np.nan, x]})
    a = periodogram(frame, "g").table
    b = periodogram(pd.Series(x)).table
    pd.testing.assert_frame_equal(a, b)
    assert "Periodogram" in periodogram(x).summary()
    assert "AR(1)" in periodogram(x, method="ar", order=1).summary()


def test_plot(x: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = periodogram(x, method="smoothed", spans=5).plot()
    assert ax.get_yscale() == "log"
    cumulative_periodogram_test(x).plot()
    plt.close("all")


@pytest.mark.parametrize(
    "kw",
    [
        {"method": "welch"},
        {"scale": "hz"},
        {"method": "smoothed"},
        {"spans": 3},
        {"method": "ar", "spans": 3},
        {"method": "smoothed", "spans": 0},
        {"taper": 0.7},
        {"detrend": "quadratic"},
        {"pad": -1},
        {"alpha": 1.5},
        {"method": "ar", "order": -1},
        {"method": "ar", "n_freq": 1},
    ],
)
def test_bad_arguments(x: np.ndarray, kw: dict) -> None:
    with pytest.raises(MethodIncompatibility):
        periodogram(x, **kw)


def test_bad_data(x: np.ndarray) -> None:
    with pytest.raises(MethodIncompatibility):
        periodogram(pd.DataFrame({"a": x}), "b")
    gap = x.copy()
    gap[30] = np.nan
    with pytest.raises(MethodIncompatibility):
        periodogram(gap)
    with pytest.raises(DataInsufficient):
        periodogram([1.0, 2.0, 3.0])
    with pytest.raises(DataInsufficient):
        periodogram(np.ones(40))
    with pytest.raises(DataInsufficient):
        periodogram(x[:10], method="ar", order=9)
    with pytest.raises(DataInsufficient):
        periodogram(x[:10], method="smoothed", spans=(9, 9))
    with pytest.raises(DataInsufficient):
        cumulative_periodogram_test(np.ones(40))
    with pytest.raises(MethodIncompatibility):
        cumulative_periodogram_test(x, alpha=0)


def test_cumulative_periodogram_test_size_and_power() -> None:
    rng = np.random.default_rng(5)
    rejections = sum(
        cumulative_periodogram_test(rng.normal(size=200)).reject for _ in range(400)
    )
    # 5% test; the Kolmogorov limit is conservative in finite samples
    assert rejections / 400 < 0.08
    e = rng.normal(size=200)
    ar = np.zeros(200)
    for t in range(1, 200):
        ar[t] = 0.5 * ar[t - 1] + e[t]
    res = cumulative_periodogram_test(ar)
    assert res.reject and res.pvalue < 0.01
    tab = res.table
    assert tab["cumulative"].iloc[-1] == pytest.approx(1.0)
    assert tab["cumulative"].is_monotonic_increasing
    width = np.abs(tab["cumulative"] - tab["expected"]).max()
    assert res.statistic == pytest.approx(np.sqrt(len(tab)) * width)
    assert "Bartlett" in res.summary()
