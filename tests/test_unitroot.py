"""sp.unitroot: augmented Dickey-Fuller and DF-GLS.

Evidence, by kind:

* ADF statistic, p-value, critical values and the lag an information
  criterion picks: against ``statsmodels.tsa.stattools.adfuller``, an
  independent implementation of the same definitions.
* DF-GLS statistic: against a from-the-definition computation written here,
  and against the ``arch`` package when it is installed.
* DF-GLS critical values: against a fresh simulation of the null, and
  against the asymptotic values the response surface must converge to.
* Size and power on known data-generating processes.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from statspai.timeseries._critvals import DFGLS_SURFACE, MACKINNON_2010, dfgls_cv

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT / "tests" / "reference_parity" / "_fixtures" / "dfgls_null_quantiles.json"
)


def _ar1(rng, n, phi, drift=0.0):
    y = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        y[t] = drift + phi * y[t - 1] + e[t]
    return y


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(20261002)
    walk = np.cumsum(rng.normal(size=180))
    return {
        "walk": walk,
        "ar": _ar1(rng, 180, 0.7),
        "trending": 0.05 * np.arange(180) + _ar1(rng, 180, 0.6),
    }


# ----------------------------------------------------------------- ADF
@pytest.mark.parametrize("name", ["walk", "ar", "trending"])
@pytest.mark.parametrize("trend", ["n", "c", "ct"])
@pytest.mark.parametrize("lags", [0, 1, 4])
def test_adf_fixed_lags_match_statsmodels(series, name, trend, lags):
    adfuller = pytest.importorskip("statsmodels.tsa.stattools").adfuller
    y = series[name]
    got = sp.unitroot(y, test="adf", trend=trend, lags=lags)
    stat, pval, used, nobs, crit = adfuller(
        y, maxlag=lags, regression=trend, autolag=None
    )
    # same OLS regression on both sides: agreement to rounding error
    np.testing.assert_allclose(got.statistic, stat, rtol=1e-9)
    assert got.lags == used and got.n_obs == nobs
    for key, value in crit.items():
        np.testing.assert_allclose(got.critical_values[key], value, rtol=1e-12)
    # statsmodels clips MacKinnon's approximation to [0, 1] outside its
    # fitted range; inside it the two are the same polynomial
    if 1e-6 < pval < 1 - 1e-6:
        np.testing.assert_allclose(got.pvalue, pval, rtol=1e-9)


@pytest.mark.parametrize("criterion", ["aic", "bic"])
@pytest.mark.parametrize("trend", ["c", "ct"])
def test_adf_lag_selection_matches_statsmodels(series, criterion, trend):
    adfuller = pytest.importorskip("statsmodels.tsa.stattools").adfuller
    for name, y in series.items():
        got = sp.unitroot(y, trend=trend, lags=criterion)
        stat, _, used, nobs = adfuller(y, regression=trend, autolag=criterion.upper())[
            :4
        ]
        assert got.lags == used, name
        assert got.n_obs == nobs, name
        np.testing.assert_allclose(got.statistic, stat, rtol=1e-9)
        assert got.lag_selection == criterion.upper()
        assert got.ic_table.index.max() == int(np.floor(12 * (len(y) / 100) ** 0.25))


def test_plain_dickey_fuller_by_hand():
    # dy = rho * y[-1] + e, no deterministics, no lags
    y = np.array([0.0, 1.0, 0.5, 1.5, 1.0, 2.0, 1.5, 2.5])
    dy, lag = np.diff(y), y[:-1]
    rho = (lag @ dy) / (lag @ lag)
    resid = dy - rho * lag
    se = np.sqrt(resid @ resid / (dy.size - 1) / (lag @ lag))
    got = sp.unitroot(y, trend="n", lags=0)
    np.testing.assert_allclose(got.rho, rho, rtol=1e-13)
    np.testing.assert_allclose(got.se, se, rtol=1e-13)
    np.testing.assert_allclose(got.statistic, rho / se, rtol=1e-13)
    assert got.n_obs == 7 and got.lag_selection == "fixed"


# --------------------------------------------------------------- DF-GLS
def _dfgls_from_definition(y, trend, lags):
    T = y.size
    a = 1 + (-7.0 if trend == "c" else -13.5) / T
    z = np.ones((T, 1)) if trend == "c" else np.c_[np.ones(T), np.arange(1, T + 1)]
    yq = np.r_[y[0], y[1:] - a * y[:-1]]
    zq = np.r_[z[:1], z[1:] - a * z[:-1]]
    yd = y - z @ np.linalg.solve(zq.T @ zq, zq.T @ yq)
    dy = np.diff(yd)
    rows = range(lags, T - 1)
    X = np.array([[yd[t]] + [dy[t - j] for j in range(1, lags + 1)] for t in rows])
    Y = dy[lags:]
    beta = np.linalg.solve(X.T @ X, X.T @ Y)
    resid = Y - X @ beta
    s2 = resid @ resid / (Y.size - X.shape[1])
    return beta[0] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])


@pytest.mark.parametrize("trend", ["c", "ct"])
@pytest.mark.parametrize("lags", [0, 2, 5])
def test_dfgls_statistic_matches_the_definition(series, trend, lags):
    for name, y in series.items():
        got = sp.unitroot(y, test="dfgls", trend=trend, lags=lags)
        want = _dfgls_from_definition(y, trend, lags)
        np.testing.assert_allclose(got.statistic, want, rtol=1e-9, err_msg=name)
        assert got.pvalue is None and got.test == "DF-GLS"
        assert got.n_obs == y.size - 1 - lags


@pytest.mark.parametrize("trend", ["c", "ct"])
def test_dfgls_statistic_matches_arch(series, trend):
    DFGLS = pytest.importorskip("arch.unitroot").DFGLS
    for name, y in series.items():
        for lags in (0, 1, 4):
            got = sp.unitroot(y, test="dfgls", trend=trend, lags=lags)
            ref = DFGLS(y, lags=lags, trend=trend)
            np.testing.assert_allclose(got.statistic, ref.stat, rtol=1e-9)
            # arch's critical values ignore the lag order; at lags=0 the two
            # surfaces are fits to the same distribution
            if lags == 0:
                for key, value in ref.critical_values.items():
                    assert abs(got.critical_values[key] - value) < 0.04, (name, key)


def test_surface_is_the_fit_to_the_committed_simulation():
    # the table in _critvals.py must be regenerable: refit the stored
    # quantiles and compare (the table is printed to six significant digits)
    import importlib.util

    script = ROOT / "scripts" / "simulate_dfgls_critical_values.py"
    spec = importlib.util.spec_from_file_location("simulate_dfgls", script)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    table = sim.fit(sim.load_points(raw))
    assert set(table) == set(DFGLS_SURFACE)
    for key, row in table.items():
        np.testing.assert_allclose(DFGLS_SURFACE[key], row["coef"], rtol=1e-12)
        # fitting error of the surface on its own grid
        assert row["rmse"] < 0.02 and row["max_abs_error"] < 0.08, (key, row)


def test_surface_reproduces_the_stored_quantiles_where_tests_are_run():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    worst = 0.0
    for key, quantiles in raw.items():
        trend, T, k = key.split("|")
        if int(T) < 50 or int(k) > 8:
            continue
        for level, q in zip((1, 5, 10), quantiles):
            worst = max(worst, abs(dfgls_cv(trend, level, int(T), int(k)) - q))
    assert worst < 0.045, worst


def test_dfgls_constant_case_converges_to_the_dickey_fuller_law():
    # With a constant the limit is the no-constant Dickey-Fuller
    # distribution, whose asymptotic critical values are MacKinnon's
    # beta_inf. An independent table, reached from finite-sample draws.
    for level in (1, 5, 10):
        limit = MACKINNON_2010["nc"][(1, level)][0]
        assert abs(dfgls_cv("c", level, 1e7, 0) - limit) < 0.01, level


@pytest.mark.parametrize(
    "trend, T, lags",
    [("c", 60, 0), ("c", 120, 3), ("ct", 60, 2), ("ct", 250, 6), ("ct", 1500, 0)],
)
def test_dfgls_critical_values_match_a_fresh_simulation(trend, T, lags):
    rng = np.random.default_rng(1000 * T + lags)
    reps = 20000
    stats = np.array(
        [
            sp.unitroot(
                np.cumsum(rng.normal(size=T)), test="dfgls", trend=trend, lags=lags
            ).statistic
            for _ in range(reps)
        ]
    )
    # Monte Carlo SE of the simulated quantile with 20,000 draws is about
    # 0.010 at 5% and 10% and 0.025 at 1%; the surface's own fitting error
    # is of the same order. The bands are about three combined SEs.
    for level, band in ((5, 0.045), (10, 0.045), (1, 0.10)):
        simulated = np.quantile(stats, level / 100)
        assert abs(dfgls_cv(trend, level, T, lags) - simulated) < band, level


def test_finite_sample_values_are_stricter_than_asymptotic_ones():
    # the reason the surface exists: at T = 50 the asymptotic 5% point
    # (-1.94) would reject far too often
    assert dfgls_cv("c", 5, 50, 0) < -2.15
    assert dfgls_cv("c", 5, 50, 0) < dfgls_cv("c", 5, 500, 0) < -1.9


# ------------------------------------------------------ size and power
@pytest.mark.parametrize("test", ["adf", "dfgls"])
def test_size_under_a_unit_root_and_power_under_stationarity(test):
    rng = np.random.default_rng(11)
    reps, T = 400, 150
    size = np.mean(
        [
            sp.unitroot(np.cumsum(rng.normal(size=T)), test=test, lags=1).reject
            for _ in range(reps)
        ]
    )
    power = np.mean(
        [sp.unitroot(_ar1(rng, T, 0.8), test=test, lags=1).reject for _ in range(reps)]
    )
    # nominal 5%; binomial SE with 400 draws is 0.011
    assert 0.02 <= size <= 0.09, size
    assert power > 0.6, power


def test_dfgls_has_more_power_than_adf_near_a_unit_root():
    rng = np.random.default_rng(3)
    draws = [_ar1(rng, 120, 0.92) for _ in range(500)]
    adf = np.mean([sp.unitroot(y, test="adf", lags=0).reject for y in draws])
    gls = np.mean([sp.unitroot(y, test="dfgls", lags=0).reject for y in draws])
    assert gls > adf + 0.1, (adf, gls)


# ----------------------------------------------------------- interface
def test_dataframe_series_and_time_ordering_agree(series):
    y = series["ar"]
    frame = pd.DataFrame({"t": np.arange(y.size), "y": y})
    shuffled = frame.sample(frac=1, random_state=0)
    base = sp.unitroot(y, lags=2)
    assert sp.unitroot(frame, "y", lags=2).statistic == base.statistic
    assert sp.unitroot(pd.Series(y), lags=2).statistic == base.statistic
    assert sp.unitroot(shuffled, "y", lags=2, time="t").statistic == base.statistic
    padded = np.r_[np.nan, np.nan, y, np.nan]
    assert sp.unitroot(padded, lags=2).statistic == base.statistic
    assert "Test statistic" in base.summary()


def test_alpha_moves_the_verdict_not_the_statistic(series):
    loose = sp.unitroot(series["walk"], lags=1, alpha=0.10)
    strict = sp.unitroot(series["walk"], lags=1, alpha=0.01)
    assert loose.statistic == strict.statistic
    assert loose.critical_values == strict.critical_values
    odd = sp.unitroot(series["ar"], lags=1, alpha=0.03)
    assert odd.reject == (odd.pvalue < 0.03)


def test_refusals(series):
    y = series["walk"]
    with pytest.raises(MethodIncompatibility, match="no trend='n'"):
        sp.unitroot(y, test="dfgls", trend="n")
    with pytest.raises(MethodIncompatibility, match="unknown test"):
        sp.unitroot(y, test="kpss")
    with pytest.raises(MethodIncompatibility, match="unknown trend"):
        sp.unitroot(y, trend="ctt")
    with pytest.raises(MethodIncompatibility, match="1%, 5%"):
        sp.unitroot(y, test="dfgls", alpha=0.03)
    with pytest.raises(MethodIncompatibility, match="unknown lag rule"):
        sp.unitroot(y, lags="hqic")
    gap = y.copy()
    gap[40] = np.nan
    with pytest.raises(MethodIncompatibility, match="gap"):
        sp.unitroot(gap)
    with pytest.raises(DataInsufficient):
        sp.unitroot(y[:8], lags=4)
    with pytest.warns(AssumptionWarning, match="extrapolated"):
        sp.unitroot(y[:20], test="dfgls", lags=0)
    with pytest.raises(MethodIncompatibility, match="y=<column>"):
        sp.unitroot(pd.DataFrame({"y": y}))


# ---------------------------------------------------------- translation
@pytest.mark.parametrize(
    "line, kwargs",
    [
        ("dfuller y", dict(lags=0)),
        ("dfuller y, lags(3)", dict(lags=3)),
        ("dfuller y, lags(2) trend", dict(lags=2, trend="ct")),
        ("dfuller y, noconstant", dict(lags=0, trend="n")),
    ],
)
def test_stata_dfuller_runs_the_same_test(series, line, kwargs):
    frame = pd.DataFrame({"y": series["trending"]})
    out = sp.from_stata(line)
    assert out["ok"] and out["untranslated_options"] == [], out
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = sp.stata(line, data=frame)
    want = sp.unitroot(frame, "y", test="adf", **kwargs)
    assert got.statistic == want.statistic and got.lags == want.lags


def test_stata_dfuller_drift_and_dfgls_are_not_run_as_something_else():
    assert sp.from_stata("dfuller y, drift")["untranslated_options"] == ["drift"]
    shown = sp.from_stata("dfuller y, regress")
    assert shown["ignored_display_options"] == ["regress"]
    assert shown["untranslated_options"] == []
    out = sp.from_stata("dfgls y, maxlag(4)")
    assert out["ok"] is False and "unitroot" in out["statspai_functions"]
