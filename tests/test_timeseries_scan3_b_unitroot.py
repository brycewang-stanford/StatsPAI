"""Options and degenerate inputs of ``sp.unitroot`` and
``sp.zivot_andrews``.

Reference computations: test regressions fitted by ``statsmodels`` OLS in
the test, ``statsmodels.tsa.stattools`` (``adfuller``, ``kpss``,
``zivot_andrews``) where it computes the same statistic, and the
Phillips-Perron corrections written out.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _walk(n: int = 120, seed: int = 1, drift: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n)) + drift * np.arange(n)


def _noise(n: int = 120, seed: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n)


def _deterministics(n: int, trend: str) -> np.ndarray:
    cols = []
    if trend in ("c", "ct"):
        cols.append(np.ones(n))
    if trend == "ct":
        cols.append(np.arange(1.0, n + 1.0))
    return np.column_stack(cols) if cols else np.empty((n, 0))


def _bartlett(u: np.ndarray, lags: int) -> float:
    n = len(u)
    out = u @ u / n
    for j in range(1, lags + 1):
        out += 2 * (1 - j / (lags + 1)) * (u[j:] @ u[:-j]) / n
    return out


class TestAugmentedDickeyFuller:
    @pytest.mark.parametrize("trend", ["n", "c", "ct"])
    @pytest.mark.parametrize("lags", [0, 2])
    def test_statistic_is_the_t_ratio_of_the_test_regression(self, trend, lags):
        y = _walk()
        dy = np.diff(y)
        rows = np.arange(lags, len(dy))
        cols = [y[rows]]
        if trend in ("c", "ct"):
            cols.append(np.ones(len(rows)))
        if trend == "ct":
            cols.append(np.arange(len(rows), dtype=float))
        for j in range(1, lags + 1):
            cols.append(dy[rows - j])
        ols = sm.OLS(dy[rows], np.column_stack(cols)).fit()
        out = sp.unitroot(y, trend=trend, lags=lags)
        assert out.statistic == pytest.approx(ols.tvalues[0], rel=1e-9)
        # For ADF, rho is the coefficient on the lagged level in the
        # regression of the first difference.
        assert out.rho == pytest.approx(ols.params[0], rel=1e-9)
        assert out.se == pytest.approx(ols.bse[0], rel=1e-9)
        assert out.n_obs == len(rows)
        assert out.lag_selection == "fixed" and out.ic_table is None

    def test_criterion_table_is_on_a_common_sample(self):
        y = _walk()
        out = sp.unitroot(y, lags="bic", max_lags=4)
        table = out.ic_table
        assert table.index.tolist() == [0, 1, 2, 3, 4]
        # Every order is fitted on the observations the largest allows.
        assert table["n_obs"].nunique() == 1
        assert table["n_obs"].iloc[0] == len(y) - 1 - 4
        assert out.lags == table["bic"].idxmin()
        assert sp.unitroot(y, lags="aic", max_lags=4).lags == table["aic"].idxmin()
        for p in (0, 3):
            dy = np.diff(y)
            rows = np.arange(4, len(dy))
            X = np.column_stack(
                [y[rows], np.ones(len(rows))] + [dy[rows - j] for j in range(1, p + 1)]
            )
            rss = sm.OLS(dy[rows], X).fit().ssr
            n, k = len(rows), X.shape[1]
            assert table.loc[p, "aic"] == pytest.approx(np.log(rss / n) + 2 * k / n)
            assert table.loc[p, "bic"] == pytest.approx(
                np.log(rss / n) + np.log(n) * k / n
            )

    def test_max_lags_is_capped_and_the_default_is_the_schwert_rule(self):
        y = _walk()
        assert len(sp.unitroot(y).ic_table) == int(12 * (120 / 100) ** 0.25) + 1
        assert len(sp.unitroot(y[:15], max_lags=10).ic_table) == (15 - 6) // 3 + 1
        assert sp.unitroot(y, max_lags=0).lags == 0

    def test_verdict_uses_the_tabulated_value_or_the_p_value(self):
        y = _noise()
        out = sp.unitroot(y, lags=1)
        assert out.reject == (out.statistic < out.critical_values["5%"])
        assert out.reject is True
        odd = sp.unitroot(y, lags=1, alpha=0.03)
        assert odd.statistic == out.statistic
        assert odd.reject == (odd.pvalue < 0.03)
        walk = sp.unitroot(_walk(), lags=1)
        assert walk.reject is False and walk.pvalue > 0.05

    def test_time_orders_a_frame(self):
        y = _walk()
        df = pd.DataFrame({"y": y[::-1], "t": np.arange(len(y))[::-1]})
        assert sp.unitroot(df, "y", time="t", lags=1).statistic == pytest.approx(
            sp.unitroot(y, lags=1).statistic
        )
        padded = np.r_[np.nan, y, np.nan, np.nan]
        assert sp.unitroot(padded, lags=1).statistic == sp.unitroot(y, lags=1).statistic


class TestDFGLS:
    @pytest.mark.parametrize("trend", ["c", "ct"])
    def test_statistic_from_the_quasi_differenced_regression(self, trend):
        y = _walk()
        T = len(y)
        a = 1 + {"c": -7.0, "ct": -13.5}[trend] / T
        z = _deterministics(T, trend)
        yq = np.r_[y[0], y[1:] - a * y[:-1]]
        zq = np.vstack([z[:1], z[1:] - a * z[:-1]])
        detrended = y - z @ sm.OLS(yq, zq).fit().params
        d = np.diff(detrended)
        rows = np.arange(2, len(d))
        X = np.column_stack([detrended[rows], d[rows - 1], d[rows - 2]])
        t = sm.OLS(d[rows], X).fit().tvalues[0]
        out = sp.unitroot(y, test="dfgls", trend=trend, lags=2)
        assert out.statistic == pytest.approx(t, rel=1e-9)
        assert out.pvalue is None
        assert out.reject == (out.statistic < out.critical_values["5%"])
        assert out.test == "DF-GLS"

    def test_short_series_warns_that_the_values_are_extrapolated(self):
        from statspai.exceptions import AssumptionWarning

        with pytest.warns(AssumptionWarning, match="extrapolated"):
            sp.unitroot(_walk(20), test="dfgls", lags=0)

    def test_refusals(self):
        with pytest.raises(MethodIncompatibility, match="no trend='n' version"):
            sp.unitroot(_walk(), test="dfgls", trend="n")
        with pytest.raises(MethodIncompatibility, match="1%, 5%"):
            sp.unitroot(_walk(), test="dfgls", alpha=0.025)


class TestPhillipsPerron:
    @pytest.mark.parametrize("trend", ["n", "c", "ct"])
    @pytest.mark.parametrize("lags", [0, 3])
    def test_statistics_from_the_corrections_written_out(self, trend, lags):
        y = _walk()
        dep, lag = y[1:], y[:-1]
        n = len(dep)
        X = np.column_stack([lag, _deterministics(n, trend)])
        ols = sm.OLS(dep, X).fit()
        u = ols.resid
        s2 = ols.scale
        se = ols.bse[0]
        g0 = u @ u / n
        lam2 = _bartlett(u, lags)
        t_rho = (ols.params[0] - 1) / se
        z_rho = n * (ols.params[0] - 1) - 0.5 * n**2 * se**2 / s2 * (lam2 - g0)
        z_t = np.sqrt(g0 / lam2) * t_rho - 0.5 * (lam2 - g0) * n * se / np.sqrt(
            lam2 * s2
        )
        out = sp.unitroot(y, test="pp", trend=trend, lags=lags)
        assert out.statistic == pytest.approx(z_t, rel=1e-9)
        assert out.z_rho == pytest.approx(z_rho, rel=1e-9)
        assert out.rho == pytest.approx(ols.params[0], rel=1e-10)
        assert out.n_obs == n
        assert "Z(rho)" in out.summary()

    @pytest.mark.parametrize("trend", ["n", "c", "ct"])
    def test_without_lags_it_is_the_dickey_fuller_test(self, trend):
        # With no autocovariance in the long-run variance the corrections
        # vanish.
        y = _walk()
        pp = sp.unitroot(y, test="pp", trend=trend, lags=0)
        df = sp.unitroot(y, test="adf", trend=trend, lags=0)
        assert pp.statistic == pytest.approx(df.statistic, rel=1e-9)
        assert pp.z_rho == pytest.approx(pp.n_obs * (pp.rho - 1), rel=1e-9)
        assert pp.pvalue == pytest.approx(df.pvalue, rel=1e-8)
        assert pp.critical_values == pytest.approx(df.critical_values)

    def test_rho_level_and_rho_diff_mean_the_same_thing_for_every_test(self):
        # ``rho`` is the coefficient of the levels regression for
        # Phillips-Perron (near 1) and of the differenced regression for ADF
        # and DF-GLS (near 0); ``rho_level`` and ``rho_diff`` name the two.
        y = _walk()
        pp = sp.unitroot(y, test="pp", lags=0)
        df = sp.unitroot(y, test="adf", lags=0)
        # The two fit the same regression when there are no lags.
        assert pp.rho_level == pytest.approx(df.rho_level, abs=1e-9)
        assert pp.rho_diff == pytest.approx(df.rho_diff, abs=1e-9)
        assert pp.se == pytest.approx(df.se, rel=1e-9)
        # Independent of the function: least squares of y[t] on y[t-1].
        X = np.column_stack([y[:-1], np.ones(len(y) - 1)])
        level = np.linalg.lstsq(X, y[1:], rcond=None)[0][0]
        assert pp.rho_level == pytest.approx(level, rel=1e-9)
        # ``rho`` keeps its documented meaning for each test.
        assert pp.rho == pp.rho_level and df.rho == df.rho_diff
        for out in (pp, df, sp.unitroot(y, test="dfgls", lags=1)):
            assert out.rho_level - out.rho_diff == pytest.approx(1.0, abs=1e-12)
            assert f"{out.rho_level:.4f} in levels" in out.summary()
        kpss = sp.unitroot(y, test="kpss")
        assert np.isnan(kpss.rho_level) and np.isnan(kpss.rho_diff)
        assert "in levels" not in kpss.summary()

    def test_default_lags_and_names(self):
        y = _walk()
        out = sp.unitroot(y, test="Phillips-Perron")
        assert out.lags == int(4 * (120 / 100) ** (2 / 9))
        assert out.lag_selection == "Newey-West rule"
        assert sp.unitroot(y, test="pperron").statistic == out.statistic
        assert out.null == "unit root"


class TestKPSS:
    @pytest.mark.parametrize("trend", ["c", "ct"])
    @pytest.mark.parametrize("lags", [0, 4, 9])
    def test_equals_statsmodels(self, trend, lags):
        from statsmodels.tsa.stattools import kpss

        for y in (_walk(), _noise()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = kpss(y, regression=trend, nlags=lags)
            out = sp.unitroot(y, test="kpss", trend=trend, lags=lags)
            # Same statistic: partial sums of the residuals over a
            # Bartlett long-run variance.
            assert out.statistic == pytest.approx(ref[0], rel=1e-10)
            assert out.critical_values == {
                k: pytest.approx(v) for k, v in ref[3].items()
            }

    def test_the_null_is_stationarity(self):
        walk = sp.unitroot(_walk(), test="kpss")
        noise = sp.unitroot(_noise(), test="kpss")
        assert walk.null == noise.null == "stationarity"
        # Rejecting is evidence of a unit root.
        assert walk.reject is True and noise.reject is False
        assert walk.reject == (walk.statistic > walk.critical_values["5%"])
        assert walk.pvalue is None and np.isnan(walk.rho)
        assert "H0: the series is stationary" in walk.summary()
        assert "Stationarity rejected" in walk.summary()
        assert walk.lags == int(4 * (120 / 100) ** 0.25)

    def test_alpha_picks_the_tabulated_level(self):
        y = _walk()
        out = sp.unitroot(y, test="kpss", alpha=0.025)
        assert out.reject == (out.statistic > out.critical_values["2.5%"])
        with pytest.raises(MethodIncompatibility, match="tabulated"):
            sp.unitroot(y, test="kpss", alpha=0.03)
        with pytest.raises(MethodIncompatibility, match="no trend='n' version"):
            sp.unitroot(y, test="kpss", trend="n")


class TestUnitRootRefusals:
    @pytest.mark.parametrize("test", ["pp", "kpss"])
    def test_long_run_lags_must_fit_the_sample(self, test):
        y = _walk(20)
        with pytest.raises(DataInsufficient, match="not defined"):
            sp.unitroot(y, test=test, lags=17)
        with pytest.raises(DataInsufficient, match="not defined"):
            sp.unitroot(y, test=test, lags=-1)

    def test_arguments(self):
        y = _walk()
        with pytest.raises(MethodIncompatibility, match="unknown test"):
            sp.unitroot(y, test="ers")
        with pytest.raises(MethodIncompatibility, match="unknown trend"):
            sp.unitroot(y, trend="ctt")
        with pytest.raises(MethodIncompatibility, match="alpha must be"):
            sp.unitroot(y, alpha=1.0)
        with pytest.raises(MethodIncompatibility, match="unknown lag rule"):
            sp.unitroot(y, lags="hqic")
        with pytest.raises(MethodIncompatibility, match="non-negative"):
            sp.unitroot(y, lags=-1)
        with pytest.raises(MethodIncompatibility, match="pass y="):
            sp.unitroot(pd.DataFrame({"y": y}))
        with pytest.raises(MethodIncompatibility, match="not in data"):
            sp.unitroot(pd.DataFrame({"y": y}), "z")

    def test_data(self):
        y = _walk()
        with pytest.raises(DataInsufficient, match="no non-missing"):
            sp.unitroot(np.full(10, np.nan))
        with pytest.raises(MethodIncompatibility, match="missing values between"):
            sp.unitroot(np.r_[y[:10], np.nan, y[11:]])
        with pytest.raises(DataInsufficient, match="too few"):
            sp.unitroot(y[:20], lags=15)
        with pytest.raises(DataInsufficient, match="singular"):
            sp.unitroot(np.ones(50))

    @pytest.mark.parametrize("test", ["pp", "kpss"])
    def test_constant_series_is_refused_by_the_long_run_tests(self, test):
        with pytest.raises(DataInsufficient):
            sp.unitroot(np.ones(50), test=test)

    def test_exact_trend_is_refused_by_phillips_perron(self):
        with pytest.raises(DataInsufficient):
            sp.unitroot(np.arange(50.0), test="pp", trend="ct")

    @pytest.mark.parametrize("test", ["pp", "kpss"])
    def test_infinite_value_is_refused_by_the_long_run_tests(self, test):
        y = _walk()
        y[10] = np.inf
        with pytest.raises((MethodIncompatibility, DataInsufficient)):
            sp.unitroot(y, test=test)

    @pytest.mark.parametrize("test", ["pp", "kpss"])
    def test_unknown_lag_rule_is_refused_by_the_long_run_tests(self, test):
        with pytest.raises(MethodIncompatibility, match="lag"):
            sp.unitroot(_walk(), test=test, lags="banana")
        # An information criterion does not choose a truncation lag.
        with pytest.raises(MethodIncompatibility, match="lag"):
            sp.unitroot(_walk(), test=test, lags="bic")
        # Unset (the function default) and 'auto' are the default rule.
        unset = sp.unitroot(_walk(), test=test)
        auto = sp.unitroot(_walk(), test=test, lags="auto")
        assert auto.lags == unset.lags and auto.statistic == unset.statistic

    def test_series_the_regression_fits_exactly_is_refused(self):
        # An exact trend around a constant: y[t] = y[t-1] + 1 leaves no
        # residual for Phillips-Perron, and KPSS around a trend has none.
        with pytest.raises(DataInsufficient, match="exactly"):
            sp.unitroot(np.arange(50.0), test="pp", trend="c")
        with pytest.raises(DataInsufficient, match="exactly"):
            sp.unitroot(np.arange(50.0), test="kpss", trend="ct")

    def test_infinite_value_is_refused_by_every_test(self):
        y = _walk()
        y[10] = -np.inf
        for test in ("adf", "dfgls"):
            with pytest.raises(MethodIncompatibility, match="infinite"):
                sp.unitroot(y, test=test)

    def test_unknown_time_column(self):
        with pytest.raises(MethodIncompatibility, match="time column"):
            sp.unitroot(pd.DataFrame({"y": _walk()}), "y", time="nope")

    def test_missing_time_is_refused(self):
        y = _walk()
        t = np.arange(len(y), dtype=float)
        t[3] = np.nan
        with pytest.raises(MethodIncompatibility, match="time"):
            sp.unitroot(pd.DataFrame({"y": y, "t": t}), "y", time="t")


# ----------------------------------------------------------------------
# Zivot-Andrews
# ----------------------------------------------------------------------
def _za_t(y, model, k, tb):
    """t ratio of rho - 1 with a break after observation ``tb``."""
    T = len(y)
    t = np.arange(1.0, T + 1.0)
    dy = np.diff(y)
    rows = np.arange(k + 1, T)
    cols = [np.ones(len(rows)), y[rows - 1], t[rows]]
    if model in ("intercept", "both"):
        cols.append((t[rows] > tb).astype(float))
    if model in ("trend", "both"):
        cols.append(np.where(t[rows] > tb, t[rows] - tb, 0.0))
    for j in range(1, k + 1):
        cols.append(dy[rows - 1 - j])
    fit = sm.OLS(y[rows], np.column_stack(cols)).fit()
    return (fit.params[1] - 1) / fit.bse[1]


class TestZivotAndrews:
    @pytest.mark.parametrize("model", ["intercept", "trend", "both"])
    @pytest.mark.parametrize("lags", [0, 2])
    def test_statistic_is_the_smallest_t_ratio_over_the_break_dates(self, model, lags):
        y = _walk(80)
        out = sp.zivot_andrews(y, model=model, lags=lags)
        lo = int(np.floor(0.15 * 80))
        cands = list(range(lo, 80 - lo + 1))
        stats_ = np.array([_za_t(y, model, lags, tb) for tb in cands])
        assert out.statistic == pytest.approx(stats_.min(), rel=1e-9)
        assert out.break_index == cands[int(np.argmin(stats_))]
        assert out.path.index.tolist() == cands
        np.testing.assert_allclose(out.path.to_numpy(), stats_, rtol=1e-8)
        assert out.lags == lags

    @pytest.mark.parametrize(
        "model, regression", [("intercept", "c"), ("trend", "t"), ("both", "ct")]
    )
    def test_equals_statsmodels(self, model, regression):
        from statsmodels.tsa.stattools import zivot_andrews

        y = _walk(80)
        ref = zivot_andrews(y, trim=0.15, maxlag=2, regression=regression, autolag=None)
        out = sp.zivot_andrews(y, model=model, lags=2)
        # Same regressions over the same candidate dates.
        # (Critical values are not compared: statsmodels uses its own
        # simulated table, this function the asymptotic one.)
        assert out.statistic == pytest.approx(ref[0], rel=1e-8)
        cv = out.critical_values
        assert cv["1%"] < cv["5%"] < cv["10%"] < 0

    def test_trim_zero_searches_every_date_that_gives_a_regression(self):
        y = _walk(80)
        full = sp.zivot_andrews(y, trim=0.0)
        trimmed = sp.zivot_andrews(y, trim=0.15)
        assert len(full.path) > len(trimmed.path)
        assert full.statistic <= trimmed.statistic
        # Where both search, they agree.
        np.testing.assert_allclose(
            full.path.loc[trimmed.path.index], trimmed.path, rtol=1e-9
        )

    def test_a_level_shift_is_dated_and_the_unit_root_rejected(self):
        rng = np.random.default_rng(5)
        y = rng.normal(size=120) + np.where(np.arange(120) >= 70, 6.0, 0.0)
        df = pd.DataFrame({"y": y, "year": 1900 + np.arange(120)})
        out = sp.zivot_andrews(df, "y", time="year")
        assert out.reject == {"1%": True, "5%": True, "10%": True}
        assert abs(out.break_index - 70) <= 2
        assert out.break_label == df["year"].iloc[out.break_index - 1]
        assert out.statistic < out.critical_values["1%"]

    def test_data_driven_lag_orders(self):
        y = _walk(80)
        fixed = {k: sp.zivot_andrews(y, lags=k).statistic for k in range(4)}
        for rule in ("aic", "bic", "ttest"):
            out = sp.zivot_andrews(y, lags=rule, max_lags=3)
            assert out.statistic == pytest.approx(fixed[out.lags], rel=1e-9)
        assert sp.zivot_andrews(y, lags="aic", max_lags=0).lags == 0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(model="x"), "model='x'"),
            (dict(trim=0.6), "trim must be"),
            (dict(lags=-1), "lags must be"),
            (dict(lags="hq"), "lags='hq'"),
            (dict(lag_selection="x"), "lag_selection='x'"),
            (dict(lag_rule="x"), "lag_rule='x'"),
            (dict(trim_rule="x"), "trim_rule='x'"),
        ],
    )
    def test_refusals(self, kwargs, match):
        with pytest.raises(MethodIncompatibility, match=match):
            sp.zivot_andrews(_walk(80), **kwargs)

    def test_degenerate_series(self):
        with pytest.raises(DataInsufficient, match="no break date"):
            sp.zivot_andrews(np.ones(60))
        frame = pd.DataFrame({"y": _walk(80), "year": np.arange(80.0)})
        with pytest.raises(MethodIncompatibility, match="time column"):
            sp.zivot_andrews(frame, "y", time="nope")
        frame.loc[4, "year"] = np.nan
        with pytest.raises(MethodIncompatibility, match="missing value"):
            sp.zivot_andrews(frame, "y", time="year")
        with pytest.raises(MethodIncompatibility, match="missing values inside"):
            sp.zivot_andrews(np.r_[_walk(20), np.nan, _walk(20)])
