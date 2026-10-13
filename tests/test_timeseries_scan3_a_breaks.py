"""Option and degenerate-input tests for ``sp.structural_break``,
``sp.chow_test`` and ``sp.cusum_test`` (scan round 3).

Numerical assertions compare with OLS algebra written here: split-sample
residual sums of squares, Wald statistics of the break dummies under each
covariance estimator, a brute-force search over break dates, recursive
residuals from expanding-window fits.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility, StatsPAIError
from statspai.timeseries.structural_break import _segment_rss_table

N = 120


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(5)
    x = rng.normal(size=N)
    w = rng.normal(size=N)
    y = 1 + 0.5 * x + 0.3 * w + rng.normal(size=N)
    y[70:] += 1.5 + 0.8 * x[70:]
    return pd.DataFrame({"y": y, "x": x, "w": w})


def _rss(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(((y - X @ b) ** 2).sum())


def _xy(frame, cols=("x", "w")):
    X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in cols])
    return X, frame["y"].to_numpy()


def _chow_f(X, y, t):
    k = X.shape[1]
    split = _rss(X[:t], y[:t]) + _rss(X[t:], y[t:])
    return ((_rss(X, y) - split) / k) / (split / (len(y) - 2 * k))


class TestSupF:
    def test_statistic_is_the_largest_chow_f_over_the_trimmed_dates(self, frame):
        X, y = _xy(frame)
        r = sp.structural_break(frame, y="y", x=["x", "w"], method="sup-f")
        dates = np.arange(int(0.15 * N), N - int(0.15 * N) + 1)
        path = np.array([_chow_f(X, y, t) for t in dates])
        np.testing.assert_array_equal(r.candidate_breaks, dates)
        np.testing.assert_allclose(r.f_path, path, rtol=1e-9)
        assert r.f_stats == pytest.approx(path.max(), rel=1e-9)
        assert r.sup_break == dates[path.argmax()]
        assert r.sup_wald == pytest.approx(3 * path.max(), rel=1e-9)
        assert r.break_dates == [r.sup_break] and r.n_breaks == 1
        assert r.rss_full == pytest.approx(_rss(X, y), rel=1e-10)

    def test_chow_is_an_alias_of_sup_f(self, frame):
        a = sp.structural_break(frame, y="y", x=["x"], method="chow")
        b = sp.structural_break(frame, y="y", x=["x"], method="sup-f")
        assert a.f_stats == b.f_stats and a.p_values == b.p_values
        assert a.test_type == "Chow" and b.test_type == "Sup-F"

    def test_naming_every_coefficient_reproduces_the_default(self, frame):
        a = sp.structural_break(frame, y="y", x=["x", "w"], method="sup-f")
        b = sp.structural_break(
            frame, y="y", x=["x", "w"], method="sup-f", break_vars=["const", "x", "w"]
        )
        # dummy-interaction Wald against the split-sample formula
        assert b.f_stats == pytest.approx(a.f_stats, rel=1e-9)
        assert b.sup_break == a.sup_break
        assert b.p_values == pytest.approx(a.p_values, rel=1e-6)

    @pytest.mark.parametrize("vce", ["nonrobust", "hc0", "hc1"])
    def test_partial_break_wald_under_each_covariance(self, frame, vce):
        X, y = _xy(frame)
        r = sp.structural_break(
            frame,
            y="y",
            x=["x", "w"],
            method="sup-f",
            break_vars=["const", "x"],
            vce=vce,
        )
        assert r.n_restrictions == 2 and r.vce == vce
        path = []
        for t in r.candidate_breaks:
            d = (np.arange(N) >= t).astype(float)
            W = np.column_stack([X, d, d * frame["x"].to_numpy()])
            bread = np.linalg.inv(W.T @ W)
            b = bread @ W.T @ y
            e = y - W @ b
            if vce == "nonrobust":
                V = bread * (e @ e) / (N - 5)
            else:
                V = bread @ ((W * (e**2)[:, None]).T @ W) @ bread
                if vce == "hc1":
                    V = V * N / (N - 5)
            path.append(b[3:] @ np.linalg.solve(V[3:, 3:], b[3:]) / 2)
        np.testing.assert_allclose(r.f_path, path, rtol=1e-8)
        assert r.f_stats == pytest.approx(max(path), rel=1e-8)
        assert r.sup_wald == pytest.approx(2 * max(path), rel=1e-8)

    def test_order_and_repeats_in_break_vars_do_not_matter(self, frame):
        a = sp.structural_break(
            frame, y="y", x=["x", "w"], method="sup-f", break_vars=["const", "x"]
        )
        b = sp.structural_break(
            frame, y="y", x=["x", "w"], method="sup-f", break_vars=["x", "const", "x"]
        )
        assert b.f_stats == pytest.approx(a.f_stats, rel=1e-9)
        assert b.n_restrictions == 2

    def test_single_candidate_date_uses_the_f_distribution(self, frame):
        X, y = _xy(frame, ("x",))
        r = sp.structural_break(frame, y="y", x=["x"], method="sup-f", min_segment=0.5)
        np.testing.assert_array_equal(r.candidate_breaks, [60])
        f = _chow_f(X, y, 60)
        assert r.f_stats == pytest.approx(f, rel=1e-9)
        assert r.p_values == pytest.approx(stats.f.sf(f, 2, N - 4), rel=1e-9)

    def test_simulated_p_value_is_a_probability_and_reproducible(self, frame):
        kw = dict(y="y", x=["x"], method="sup-f", pvalue_method="simulate")
        a = sp.structural_break(frame, **kw)
        b = sp.structural_break(frame, **kw)
        assert 0.0 <= a.p_values <= 1.0 and a.p_values == b.p_values
        hansen = sp.structural_break(frame, y="y", x=["x"], method="sup-f")
        assert a.f_stats == hansen.f_stats

    def test_mean_shift_without_regressors(self, frame):
        y = frame["y"].to_numpy()
        r = sp.structural_break(frame, y="y", method="sup-f")
        X = np.ones((N, 1))
        best = max(_chow_f(X, y, t) for t in r.candidate_breaks)
        assert r.f_stats == pytest.approx(best, rel=1e-9)

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"method": "zz"}, "method must be"),
            ({"method": "sup-f", "pvalue_method": "zz"}, "pvalue_method must be"),
            ({"method": "sup-f", "vce": "hc3"}, "vce must be"),
            ({"method": "sup-f", "break_vars": ["q"]}, "must be drawn from"),
            ({"method": "sup-f", "break_vars": []}, "must be drawn from"),
            ({"vce": "hc1"}, "apply to method='sup-f'"),
            ({"break_vars": ["x"]}, "apply to method='sup-f'"),
        ],
    )
    def test_option_errors(self, frame, kw, match):
        with pytest.raises(MethodIncompatibility, match=match):
            sp.structural_break(frame, y="y", x=["x"], **kw)

    def test_data_is_required(self):
        with pytest.raises(ValueError, match="data is required"):
            sp.structural_break(y="y")


class TestMultipleBreaks:
    def test_segment_rss_table_equals_direct_fits(self, frame):
        X, y = _xy(frame)
        table = _segment_rss_table(y, X, 18)
        for i in range(0, N - 18, 11):
            for j in range(i + 17, N, 13):
                # recursive least squares against a direct fit
                assert table[i, j] == pytest.approx(
                    _rss(X[i : j + 1], y[i : j + 1]), rel=1e-8
                )

    def test_sequential_first_step_is_the_ssr_minimising_split(self, frame):
        X, y = _xy(frame, ("x",))
        r = sp.structural_break(frame, y="y", x=["x"])
        h = int(np.floor(0.15 * N))
        assert r.min_segment_size == h
        split = {
            t: _rss(X[:t], y[:t]) + _rss(X[t:], y[t:]) for t in range(h, N - h + 1)
        }
        date = min(split, key=split.get)
        stat = (N - 4) * (_rss(X, y) - split[date]) / split[date]
        first = r.sequential_tests[0]
        assert first["date"] == date
        assert first["statistic"] == pytest.approx(stat, rel=1e-8)
        assert first["reject"] and r.break_dates[0] == date
        assert r.sequential_tests[-1]["reject"] is False
        assert r.p_values is None
        assert r.rss_segments == pytest.approx(split[date], rel=1e-8)
        assert r.bic == pytest.approx(
            N * np.log(split[date] / N) + 2 * 2 * np.log(N), rel=1e-8
        )
        assert "sup F(l+1|l)" in r.summary()

    def test_global_one_break_equals_brute_force(self, frame):
        X, y = _xy(frame, ("x",))
        r = sp.structural_break(frame, y="y", x=["x"], method="global", max_breaks=1)
        h = int(np.floor(0.15 * N))
        split = {
            t: _rss(X[:t], y[:t]) + _rss(X[t:], y[t:]) for t in range(h, N - h + 1)
        }
        date = min(split, key=split.get)
        assert r.breaks_by_m[1] == [date]
        np.testing.assert_allclose(
            r.rss_by_breaks, [_rss(X, y), split[date]], rtol=1e-8
        )
        bic = [
            N * (np.log(v) + 1 - np.log(N) + np.log(2 * np.pi))
            + 3 * (m + 1) * np.log(N)
            for m, v in enumerate(r.rss_by_breaks)
        ]
        np.testing.assert_allclose(r.bic_by_breaks, bic, rtol=1e-10)
        assert r.n_breaks == int(np.argmin(bic)) and r.break_dates == [date]

    def test_global_two_breaks_equal_brute_force(self):
        rng = np.random.default_rng(11)
        y = np.r_[rng.normal(0, 1, 20), rng.normal(4, 1, 20), rng.normal(-2, 1, 20)]
        data = pd.DataFrame({"y": y})
        r = sp.structural_break(data, y="y", method="global", min_segment=0.15)
        h = 9
        X = np.ones((60, 1))
        best, arg = np.inf, None
        for a in range(h, 60 - 2 * h + 1):
            for b in range(a + h, 60 - h + 1):
                v = _rss(X[:a], y[:a]) + _rss(X[a:b], y[a:b]) + _rss(X[b:], y[b:])
                if v < best:
                    best, arg = v, [a, b]
        assert r.breaks_by_m[2] == arg
        assert r.rss_by_breaks[2] == pytest.approx(best, rel=1e-8)
        # the number of breaks reported is the BIC minimiser
        assert r.n_breaks == int(np.argmin(r.bic_by_breaks))
        assert r.break_dates == r.breaks_by_m[r.n_breaks]

    def test_table_and_sample_size_limits(self, frame):
        with pytest.raises(MethodIncompatibility, match="critical-value tables"):
            sp.structural_break(frame, y="y", x=["x"], min_segment=0.3)
        with pytest.raises(MethodIncompatibility, match="critical-value tables"):
            sp.structural_break(frame, y="y", x=["x"], alpha=0.2)
        with pytest.raises(DataInsufficient, match="must exceed the number"):
            sp.structural_break(frame.iloc[:6], y="y", x=["x", "w"])
        with pytest.raises(DataInsufficient, match="smaller than half"):
            sp.structural_break(frame, y="y", x=["x"], method="global", min_segment=0.6)
        with pytest.raises(DataInsufficient, match="greater than the number"):
            sp.structural_break(frame.iloc[:6], y="y", x=["x", "w"], method="global")

    @pytest.mark.parametrize("method", ["bai-perron", "global"])
    def test_redundant_regressor_does_not_move_the_break_dates(self, frame, method):
        twin = frame.assign(x2=2 * frame["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = sp.structural_break(frame, y="y", x=["x"], method=method)
            try:
                got = sp.structural_break(twin, y="y", x=["x", "x2"], method=method)
            except StatsPAIError:
                return
        assert got.break_dates == base.break_dates

    def test_redundant_regressor_does_not_change_sup_f(self, frame):
        twin = frame.assign(x2=2 * frame["x"])
        base = sp.structural_break(frame, y="y", x=["x"], method="sup-f")
        try:
            got = sp.structural_break(twin, y="y", x=["x", "x2"], method="sup-f")
        except StatsPAIError:
            return
        assert got.f_stats == pytest.approx(base.f_stats, rel=1e-6)

    @pytest.mark.parametrize("kw", [{"vce": "hc1"}, {"break_vars": ["x"]}])
    def test_global_refuses_options_it_does_not_use(self, frame, kw):
        with pytest.raises(MethodIncompatibility, match="apply to method='sup-f'"):
            sp.structural_break(frame, y="y", x=["x"], method="global", **kw)

    def test_global_with_no_breaks_allowed_reports_none(self, frame):
        try:
            r = sp.structural_break(
                frame, y="y", x=["x"], method="global", max_breaks=0
            )
        except StatsPAIError:
            return
        assert r.break_dates == []


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestDegenerateBreakInputs:
    @pytest.mark.parametrize("kw", [{}, {"vce": "hc1"}])
    def test_sup_f_refuses_missing_values(self, frame, kw):
        hole = frame.copy()
        hole.loc[30, "y"] = np.nan
        with pytest.raises(StatsPAIError, match="(?i)missing|nan"):
            sp.structural_break(hole, y="y", x=["x"], method="sup-f", **kw)

    @pytest.mark.parametrize("method", ["bai-perron", "global"])
    def test_bai_perron_refuses_missing_values(self, frame, method):
        hole = frame.copy()
        hole.loc[30, "y"] = np.nan
        with pytest.raises(StatsPAIError, match="(?i)missing|nan"):
            sp.structural_break(hole, y="y", x=["x"], method=method)

    @pytest.mark.parametrize("rows, min_segment", [(6, 0.15), (N, 0.6)])
    def test_sup_f_without_candidate_dates_is_refused(self, frame, rows, min_segment):
        with pytest.raises(StatsPAIError):
            sp.structural_break(
                frame.iloc[:rows],
                y="y",
                x=["x", "w"],
                method="sup-f",
                min_segment=min_segment,
            )

    def test_constant_outcome_is_refused_by_the_sequential_procedure(self):
        with pytest.raises(StatsPAIError):
            sp.structural_break(pd.DataFrame({"y": np.ones(60)}), y="y")


class TestChowTest:
    def test_all_coefficients_is_the_split_sample_statistic(self, frame):
        X, y = _xy(frame)
        r = sp.chow_test(frame, "y", ["x", "w"], break_point=70)
        f = _chow_f(X, y, 70)
        assert r["statistic"] == pytest.approx(f, rel=1e-9)
        assert (r["df1"], r["df2"]) == (3, N - 6)
        assert r["pvalue"] == pytest.approx(stats.f.sf(f, 3, N - 6), rel=1e-8)
        assert r["chi2"] == pytest.approx(3 * f, rel=1e-9)
        assert r["chi2_pvalue"] == pytest.approx(stats.chi2.sf(3 * f, 3), rel=1e-8)
        assert r["rss_restricted"] == pytest.approx(_rss(X, y), rel=1e-10)
        assert r["rss_unrestricted"] == pytest.approx(
            _rss(X[:70], y[:70]) + _rss(X[70:], y[70:]), rel=1e-10
        )
        regimes = r["regimes"]
        np.testing.assert_allclose(
            regimes["regime_1"].to_numpy()[:3],
            np.linalg.lstsq(X[:70], y[:70], rcond=None)[0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            regimes["regime_2"].to_numpy()[:3],
            np.linalg.lstsq(X[70:], y[70:], rcond=None)[0],
            atol=1e-9,
        )
        assert list(regimes.loc["n_obs"]) == [70, 50]

    def test_two_breaks_jointly(self, frame):
        X, y = _xy(frame, ("x",))
        r = sp.chow_test(frame, "y", ["x"], break_point=[40, 80])
        split = _rss(X[:40], y[:40]) + _rss(X[40:80], y[40:80]) + _rss(X[80:], y[80:])
        f = ((_rss(X, y) - split) / 4) / (split / (N - 6))
        assert r["statistic"] == pytest.approx(f, rel=1e-9)
        assert r["break_point"] == [40, 80] and r["df1"] == 4

    @pytest.mark.parametrize("vce", ["nonrobust", "hc0", "hc1"])
    def test_partial_break_under_each_covariance(self, frame, vce):
        X, y = _xy(frame)
        r = sp.chow_test(
            frame, "y", ["x", "w"], break_point=70, break_vars=["x"], vce=vce
        )
        d = (np.arange(N) >= 70).astype(float)
        W = np.column_stack([X, d * frame["x"].to_numpy()])
        bread = np.linalg.inv(W.T @ W)
        b = bread @ W.T @ y
        e = y - W @ b
        if vce == "nonrobust":
            v = bread[3, 3] * (e @ e) / (N - 4)
        else:
            v = (bread @ ((W * (e**2)[:, None]).T @ W) @ bread)[3, 3]
            if vce == "hc1":
                v *= N / (N - 4)
        assert r["statistic"] == pytest.approx(b[3] ** 2 / v, rel=1e-9)
        assert (r["df1"], r["df2"]) == (1, N - 4)

    def test_break_point_by_date_and_by_index_label(self, frame):
        dated = frame.assign(t=pd.date_range("2000-01-01", periods=N, freq="MS"))
        by_int = sp.chow_test(frame, "y", ["x"], break_point=70)
        by_time = sp.chow_test(dated, "y", ["x"], break_point="2005-11-01", time="t")
        by_index = sp.chow_test(
            dated.set_index("t"), "y", ["x"], break_point="2005-11-01"
        )
        between = sp.chow_test(dated, "y", ["x"], break_point="2005-10-15", time="t")
        for other in (by_time, by_index, between):
            assert other["break_point"] == 70
            assert other["statistic"] == pytest.approx(by_int["statistic"], rel=1e-12)

    @pytest.mark.parametrize(
        "kw, exc, match",
        [
            ({"break_point": 0}, MethodIncompatibility, "both sides"),
            ({"break_point": N}, MethodIncompatibility, "both sides"),
            ({"break_point": [80, 40]}, MethodIncompatibility, "increasing order"),
            ({"break_point": [40, 40]}, MethodIncompatibility, "increasing order"),
            ({"break_point": []}, MethodIncompatibility, "is empty"),
            ({"break_point": 1}, DataInsufficient, "cannot identify"),
            ({"break_point": 60, "vce": "hc3"}, MethodIncompatibility, "vce must be"),
            ({"break_point": 60, "time": "nope"}, MethodIncompatibility, "not found"),
            (
                {"break_point": 60, "break_vars": ["q"]},
                MethodIncompatibility,
                "must be drawn from",
            ),
        ],
    )
    def test_argument_errors(self, frame, kw, exc, match):
        with pytest.raises(exc, match=match):
            sp.chow_test(frame, "y", ["x"], **kw)

    def test_missing_collinear_and_unknown_columns(self, frame):
        hole = frame.copy()
        hole.loc[3, "x"] = np.nan
        with pytest.raises(MethodIncompatibility, match="missing values"):
            sp.chow_test(hole, "y", ["x"], break_point=60)
        with pytest.raises(MethodIncompatibility, match="collinear within a regime"):
            sp.chow_test(
                frame.assign(x2=2 * frame["x"]), "y", ["x", "x2"], break_point=60
            )
        with pytest.raises(MethodIncompatibility, match="not found in data"):
            sp.chow_test(frame, "y", ["nope"], break_point=60)
        step = frame.assign(d=(np.arange(N) >= 60).astype(float))
        with pytest.raises(MethodIncompatibility, match="collinear within a regime"):
            sp.chow_test(step, "y", ["d"], break_point=60)

    def test_time_column_orders_the_observations(self, frame):
        dated = frame.assign(t=pd.date_range("2000-01-01", periods=N, freq="MS"))
        ordered = sp.chow_test(dated, "y", ["x"], break_point="2005-11-01", time="t")
        shuffled = dated.sample(frac=1, random_state=0)
        try:
            got = sp.chow_test(shuffled, "y", ["x"], break_point="2005-11-01", time="t")
        except StatsPAIError:
            return
        assert got["statistic"] == pytest.approx(ordered["statistic"], rel=1e-9)


class TestCusum:
    def test_ols_cusum_by_hand(self, frame):
        X, y = _xy(frame, ("x",))
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        path = np.cumsum(e) / (np.sqrt(e @ e / (N - 2)) * np.sqrt(N))
        r = sp.cusum_test(frame, y="y", x=["x"], method="ols", alpha=0.10)
        np.testing.assert_allclose(r["cusum"], path, atol=1e-10)
        stat = np.abs(path).max()
        assert r["statistic"] == pytest.approx(stat, rel=1e-10)
        assert r["p_value"] == pytest.approx(stats.kstwobign.sf(stat), rel=1e-10)
        assert r["critical_value"] == pytest.approx(stats.kstwobign.isf(0.10))
        assert r["reject"] == (stat > r["critical_value"])

    def test_recursive_cusum_by_hand(self, frame):
        X, y = _xy(frame, ("x",))
        k = 2
        w = []
        for t in range(k, N):
            bt = np.linalg.lstsq(X[:t], y[:t], rcond=None)[0]
            lever = X[t] @ np.linalg.inv(X[:t].T @ X[:t]) @ X[t]
            w.append((y[t] - X[t] @ bt) / np.sqrt(1 + lever))
        w = np.array(w)
        # identity: the recursive residuals carry the full-sample RSS
        assert (w**2).sum() == pytest.approx(_rss(X, y), rel=1e-8)
        m = len(w)
        path = np.cumsum(w) / (w.std(ddof=1) * np.sqrt(m))
        s = np.arange(1, m + 1)
        r = sp.cusum_test(frame, y="y", x=["x"])
        np.testing.assert_allclose(r["cusum"], path, atol=1e-8)
        assert r["statistic"] == pytest.approx(
            (np.abs(path) / (1 + 2 * s / m)).max(), rel=1e-8
        )
        assert r["max_cusum"] == pytest.approx(np.abs(path).max(), rel=1e-8)
        np.testing.assert_allclose(
            r["critical_value"], r["boundary_coef"] * (1 + 2 * s / m), rtol=1e-12
        )
        assert r["boundary_coef"] == pytest.approx(0.9479, abs=5e-4)
        assert r["reject"] == bool((np.abs(path) > r["critical_value"]).any())

    def test_option_errors(self, frame):
        with pytest.raises(MethodIncompatibility, match="method must be"):
            sp.cusum_test(frame, y="y", method="zz")
        for method in ("recursive", "ols"):
            with pytest.raises(MethodIncompatibility, match="alpha must lie"):
                sp.cusum_test(frame, y="y", x=["x"], method=method, alpha=3)

    @pytest.mark.parametrize("method", ["recursive", "ols"])
    def test_missing_value_is_refused(self, frame, method):
        hole = frame.copy()
        hole.loc[30, "y"] = np.nan
        with pytest.raises(StatsPAIError, match="(?i)missing|nan"):
            sp.cusum_test(hole, y="y", x=["x"], method=method)

    @pytest.mark.parametrize("method", ["recursive", "ols"])
    def test_constant_outcome_is_refused(self, method):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.cusum_test(pd.DataFrame({"y": np.ones(50)}), y="y", method=method)


class TestScanFixDetails:
    def test_break_in_an_omitted_coefficient_is_refused(self, frame):
        twin = frame.assign(x2=2 * frame["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(MethodIncompatibility, match="omitted as collinear"):
                sp.structural_break(
                    twin, y="y", x=["x", "x2"], method="sup-f", break_vars=["x2"]
                )
            robust = sp.structural_break(
                twin, y="y", x=["x", "x2"], method="sup-f", vce="hc1"
            )
            base = sp.structural_break(frame, y="y", x=["x"], method="sup-f", vce="hc1")
        assert robust.f_stats == pytest.approx(base.f_stats, rel=1e-12)

    def test_redundant_regressor_is_announced(self, frame):
        twin = frame.assign(x2=2 * frame["x"])
        with pytest.warns(UserWarning, match="x2 omitted"):
            got = sp.cusum_test(twin, y="y", x=["x", "x2"])
        base = sp.cusum_test(frame, y="y", x=["x"])
        assert got["statistic"] == pytest.approx(base["statistic"], rel=1e-12)

    def test_chow_refuses_an_undated_row(self, frame):
        dated = frame.assign(t=pd.date_range("2000-01-01", periods=N, freq="MS"))
        dated.loc[5, "t"] = pd.NaT
        with pytest.raises(MethodIncompatibility, match="time column"):
            sp.chow_test(dated, "y", ["x"], break_point="2005-11-01", time="t")


class TestExactFitWithinRegimes:
    """A noiseless step: the full-sample fit has residuals, every regime
    fits exactly at the true date, so the statistic would divide by zero."""

    @pytest.mark.parametrize("method", ["sup-f", "chow", "global", "bai-perron"])
    def test_noiseless_step_is_refused(self, method):
        df = pd.DataFrame({"y": np.r_[np.zeros(40), np.ones(40)]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no divide-by-zero on the way
            with pytest.raises(DataInsufficient, match="fit the outcome exactly"):
                sp.structural_break(df, y="y", method=method)

    @pytest.mark.parametrize("method", ["sup-f", "global"])
    def test_a_little_noise_is_enough(self, method):
        rng = np.random.default_rng(0)
        y = np.r_[np.zeros(40), np.ones(40)] + 1e-3 * rng.normal(size=80)
        res = sp.structural_break(pd.DataFrame({"y": y}), y="y", method=method)
        assert list(res.break_dates) == [40]


class TestTooFewObservations:
    def test_structural_break_needs_more_rows_than_coefficients(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.normal(size=(3, 3)), columns=["y", "x", "w"])
        with pytest.raises(DataInsufficient, match="3 observation.*3 coefficient"):
            sp.structural_break(df, y="y", x=["x", "w"])

    @pytest.mark.parametrize("method", ["ols", "recursive"])
    def test_cusum_needs_two_rows_beyond_the_coefficients(self, method):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.normal(size=(3, 2)), columns=["y", "x"])
        with pytest.raises(DataInsufficient, match="3 observation.*2 coefficient"):
            sp.cusum_test(df, y="y", x=["x"], method=method)
