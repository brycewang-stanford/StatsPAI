"""Coverage tests for statspai.rd.rdit (Regression Discontinuity in Time).

Exercises datetime/numeric time axes, kernels, polynomial orders, donut,
seasonality, cluster argument, the optimal-bandwidth selector, HAC SEs,
and error paths. Real synthetic time-series RD data; properties asserted.

Note: a pandas-numeric (float) time column is parsed by ``pd.to_datetime``
as nanoseconds-since-epoch, so to exercise the genuinely numeric time axis
we use a string column of float-formatted numbers, which ``pd.to_datetime``
rejects while ``astype(float)`` accepts (this is the real numeric branch).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_numeric_rdit(n=400, tau=2.0, seed=0):
    """Numeric (non-datetime) time axis via float-formatted strings."""
    rng = np.random.default_rng(seed)
    t = np.arange(n).astype(float)
    x = t - n / 2.0  # cutoff at the middle
    D = (x >= 0).astype(float)
    y = 0.01 * x + tau * D + np.cumsum(rng.normal(0, 0.05, n)) + rng.normal(0, 0.5, n)
    # string-formatted floats: pd.to_datetime rejects, float() accepts
    return pd.DataFrame({"y": y, "t": [f"{v:.1f}" for v in t]})


def _make_datetime_rdit(n=400, tau=1.5, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    x = (dates - dates[n // 2]).days.values.astype(float)
    D = (x >= 0).astype(float)
    season = 0.5 * np.sin(2 * np.pi * dates.month.values / 12.0)
    y = 0.005 * x + tau * D + season + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, "date": dates})


def test_rdit_numeric_basic_properties():
    df = _make_numeric_rdit()
    res = sp.rdit(df, y="y", time="t", cutoff=200.0)
    assert res.se > 0
    assert 0.0 <= res.pvalue <= 1.0
    assert res.ci[0] < res.ci[1]
    assert res.model_info["bandwidth"] > 0
    assert res.model_info["bandwidth_auto"] is True
    assert res.model_info["n_eff"] > 0
    assert res.model_info["max_lag_hac"] >= 1
    assert abs(res.estimate - 2.0) < 2.0


def test_rdit_manual_bandwidth_and_quadratic():
    df = _make_numeric_rdit()
    res = sp.rdit(df, y="y", time="t", cutoff=200.0, h=60.0, p=2)
    assert res.model_info["bandwidth"] == pytest.approx(60.0)
    assert res.model_info["bandwidth_auto"] is False
    assert res.model_info["polynomial_order"] == 2
    assert res.detail is not None and len(res.detail) == 200


@pytest.mark.parametrize(
    "kernel", ["triangular", "epanechnikov", "uniform", "gaussian"]
)
def test_rdit_all_kernels(kernel):
    df = _make_numeric_rdit()
    res = sp.rdit(df, y="y", time="t", cutoff=200.0, h=80.0, kernel=kernel)
    assert res.se >= 0
    assert res.model_info["kernel"] == kernel


def test_rdit_donut_excludes_observations():
    df = _make_numeric_rdit()
    full = sp.rdit(df, y="y", time="t", cutoff=200.0, h=80.0, donut=0)
    donut = sp.rdit(df, y="y", time="t", cutoff=200.0, h=80.0, donut=10)
    assert donut.model_info["donut"] == 10
    assert donut.n_obs < full.n_obs


def test_rdit_datetime_with_seasonality():
    df = _make_datetime_rdit()
    res = sp.rdit(
        df, y="y", time="date", cutoff="2010-07-10", seasonality="month", h=120.0
    )
    assert res.model_info["seasonality"] == "month"
    assert res.se > 0
    assert abs(res.estimate - 1.5) < 2.0


@pytest.mark.parametrize("seas", ["month", "quarter", "dow"])
def test_rdit_seasonality_variants(seas):
    df = _make_datetime_rdit()
    res = sp.rdit(
        df, y="y", time="date", cutoff="2010-07-10", seasonality=seas, h=150.0
    )
    assert res.model_info["seasonality"] == seas


def test_rdit_datetime_string_column_autoconvert():
    df = _make_datetime_rdit()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")  # ISO date strings -> autoconvert
    res = sp.rdit(df, y="y", time="date", cutoff="2010-07-10", h=120.0)
    assert res.se > 0


def test_rdit_cluster_argument_accepted():
    df = _make_numeric_rdit()
    df["g"] = (np.arange(len(df)) // 20).astype(int)
    res = sp.rdit(df, y="y", time="t", cutoff=200.0, h=80.0, cluster="g")
    assert res.se >= 0


def test_rdit_errors():
    df = _make_numeric_rdit()
    with pytest.raises(ValueError, match="not found"):
        sp.rdit(df, y="nope", time="t", cutoff=200.0)
    with pytest.raises(ValueError, match="not found"):
        sp.rdit(df, y="y", time="nope", cutoff=200.0)
    with pytest.raises(ValueError, match="kernel"):
        sp.rdit(df, y="y", time="t", cutoff=200.0, kernel="bad")
    with pytest.raises(ValueError, match="Cluster"):
        sp.rdit(df, y="y", time="t", cutoff=200.0, cluster="missing")


def test_rdit_seasonality_requires_datetime():
    df = _make_numeric_rdit()
    with pytest.raises(ValueError, match="datetime"):
        sp.rdit(df, y="y", time="t", cutoff=200.0, seasonality="month")


def test_rdit_bad_seasonality_method():
    df = _make_datetime_rdit()
    with pytest.raises(ValueError, match="seasonality"):
        sp.rdit(df, y="y", time="date", cutoff="2010-07-10", seasonality="weekly")


def test_rdit_insufficient_after_donut():
    df = _make_numeric_rdit(n=60)
    with pytest.raises(ValueError):
        sp.rdit(df, y="y", time="t", cutoff=30.0, donut=1000)


def test_rdit_too_few_in_bandwidth():
    df = _make_numeric_rdit(n=400)
    with pytest.raises(ValueError, match="within bandwidth"):
        sp.rdit(df, y="y", time="t", cutoff=200.0, h=0.4)


def test_rdit_citation_available():
    df = _make_numeric_rdit()
    res = sp.rdit(df, y="y", time="t", cutoff=200.0, h=80.0)
    cite = res.cite()
    assert "hausman" in cite.lower()
