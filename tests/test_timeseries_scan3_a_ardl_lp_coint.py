"""Option and degenerate-input tests for ``sp.ardl``,
``sp.local_projections``, ``sp.engle_granger``, ``sp.johansen`` and
``sp.johansen_lrtest`` (scan round 3).

Numerical assertions compare with algebra written here (OLS on hand-built
lags, forecast recursions, Newey-West sandwiches, canonical correlations by
QR and SVD) or with statsmodels where it computes the same quantity under
the same convention (``coint`` for Engle-Granger, ``coint_johansen`` for the
unrestricted-constant Johansen statistics).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility, StatsPAIError

# --------------------------------------------------------------------- ardl

N = 160


@pytest.fixture(scope="module")
def adl():
    rng = np.random.default_rng(0)
    x = rng.normal(size=N)
    z = rng.normal(size=N)
    y = np.zeros(N)
    for t in range(2, N):
        y[t] = (
            0.5
            + 0.4 * y[t - 1]
            - 0.2 * y[t - 2]
            + 0.8 * x[t - 1]
            + 0.3 * z[t - 2]
            + rng.normal()
        )
    return pd.DataFrame({"y": y, "x": x, "z": z})


def _adl_by_hand(frame, trend):
    """ADL(2; x: 1, z: 2) by OLS: coefficients, residual s.e., design."""
    y, x, z = (frame[c].to_numpy() for c in ("y", "x", "z"))
    cols = [y[1:-1], y[:-2], x[1:-1], z[1:-1], z[:-2]]
    if trend != "n":
        cols.append(np.ones(N - 2))
    if trend == "ct":
        cols.append(np.arange(3.0, N + 1))
    X = np.column_stack(cols)
    b = np.linalg.lstsq(X, y[2:], rcond=None)[0]
    e = y[2:] - X @ b
    return b, np.sqrt(e @ e / (len(e) - X.shape[1])), X


LAGGED = ["y_L1", "y_L2", "x_L1", "z_L1", "z_L2"]


class TestArdl:
    @pytest.mark.parametrize("trend", ["c", "ct", "n"])
    def test_coefficients_forecasts_and_long_run_by_hand(self, adl, trend):
        r = sp.ardl(
            adl, "y", ["x", "z"], lags=2, x_lags={"x": 1, "z": 2}, trend=trend,
            vce="nonrobust",
        )  # fmt: skip
        b, s, X = _adl_by_hand(adl, trend)
        np.testing.assert_allclose(r.params[LAGGED], b[:5], atol=1e-10)
        assert ("Intercept" in r.params.index) == (trend != "n")
        if trend == "ct":
            # the trend counts rows of the data from 1
            assert r.params["_trend"] == pytest.approx(b[-1], abs=1e-10)
        assert r.ser == pytest.approx(s, rel=1e-10)

        future = pd.DataFrame({"x": [1.0, 2.0], "z": [0.5, 0.6]})
        fc = r.forecast(steps=3, exog=future, alpha=0.10)
        ys = list(adl["y"])
        xs = list(adl["x"]) + [1.0, 2.0]
        zs = list(adl["z"]) + [0.5, 0.6]
        expect = []
        for h in range(1, 4):
            t = N - 1 + h
            row = [ys[t - 1], ys[t - 2], xs[t - 1], zs[t - 1], zs[t - 2]]
            row += [1.0] * (trend != "n") + [t + 1.0] * (trend == "ct")
            expect.append(float(np.array(row) @ b))
            ys.append(expect[-1])
        np.testing.assert_allclose(fc["forecast"], expect, atol=1e-9)
        psi = np.array([1.0, b[0], b[0] ** 2 + b[1]])
        rmsfe = s * np.sqrt(np.cumsum(psi**2))
        np.testing.assert_allclose(fc["rmsfe"], rmsfe, rtol=1e-9)
        z90 = stats.norm.ppf(0.95)
        np.testing.assert_allclose(fc["upper"], expect + z90 * rmsfe, rtol=1e-9)

        gap = 1 - b[0] - b[1]
        lr = r.long_run()
        assert lr.loc["x", "coef"] == pytest.approx(b[2] / gap, rel=1e-9)
        assert lr.loc["z", "coef"] == pytest.approx((b[3] + b[4]) / gap, rel=1e-9)
        assert lr.attrs["adjustment"] == pytest.approx(-gap, rel=1e-9)
        # delta method on the classical covariance s^2 (X'X)^{-1}
        V = s**2 * np.linalg.inv(X.T @ X)
        grad = np.zeros(X.shape[1])
        grad[[3, 4]] = 1 / gap
        grad[[0, 1]] = (b[3] + b[4]) / gap**2
        assert lr.loc["z", "std err"] == pytest.approx(
            np.sqrt(grad @ V @ grad), rel=1e-8
        )

    def test_criterion_ranks_every_order_on_the_same_rows(self, adl):
        r = sp.ardl(adl, "y", "x", lags="bic", x_lags=1, max_lags=4, sample=(20, 150))
        y, x = adl["y"].to_numpy(), adl["x"].to_numpy()
        n = 131
        for p in range(5):
            X = np.column_stack(
                [np.ones(n)]
                + [y[20 - k : 151 - k] for k in range(1, p + 1)]
                + [x[19:150]]
            )
            e = y[20:151] - X @ np.linalg.lstsq(X, y[20:151], rcond=None)[0]
            base = np.log(e @ e / n)
            assert r.ic_table.loc[p, "bic"] == pytest.approx(
                base + X.shape[1] * np.log(n) / n, rel=1e-9
            )
            assert r.ic_table.loc[p, "aic"] == pytest.approx(
                base + X.shape[1] * 2 / n, rel=1e-9
            )
            assert r.ic_table.loc[p, "nobs"] == n
        assert r.lags == int(r.ic_table["bic"].idxmin())

    def test_sample_fixes_the_window_and_takes_lags_from_earlier_rows(self, adl):
        r = sp.ardl(adl, "y", "x", lags=2, x_lags=1, sample=(20, 120), vce="nonrobust")
        y, x = adl["y"].to_numpy(), adl["x"].to_numpy()
        X = np.column_stack([np.ones(101), y[19:120], y[18:119], x[19:120]])
        b = np.linalg.lstsq(X, y[20:121], rcond=None)[0]
        np.testing.assert_allclose(
            r.params[["Intercept", "y_L1", "y_L2", "x_L1"]], b, atol=1e-10
        )
        # the forecast origin is the end of the sample, not of the data
        one = r.forecast()
        assert one["forecast"].iloc[0] == pytest.approx(
            b @ [1.0, y[120], y[119], x[120]], abs=1e-9
        )
        clipped = sp.ardl(adl, "y", "x", lags=3, sample=(0, 100))
        assert len(clipped._sample_index) == 98  # rows 3..100

    @pytest.mark.parametrize("window", ["expanding", "rolling"])
    def test_pseudo_out_of_sample_forecasts_by_hand(self, adl, window):
        r = sp.ardl(adl, "y", "x", lags=1)
        out = r.poos(start=100, window=window)
        y, x = adl["y"].to_numpy(), adl["x"].to_numpy()
        errors = []
        for target in (100, 130, 159):
            lo = 1 if window == "expanding" else target - 99
            rows = np.arange(max(lo, 1), target)
            X = np.column_stack([np.ones(len(rows)), y[rows - 1], x[rows - 1]])
            b = np.linalg.lstsq(X, y[rows], rcond=None)[0]
            point = b @ [1.0, y[target - 1], x[target - 1]]
            assert out.loc[target, "forecast"] == pytest.approx(point, abs=1e-9)
            errors.append(y[target] - point)
            assert out.loc[target, "error"] == pytest.approx(errors[-1], abs=1e-9)
        err = out["error"].to_numpy()
        assert out.attrs["n_forecasts"] == 60
        assert out.attrs["rmsfe"] == pytest.approx(np.sqrt(np.mean(err**2)))
        assert out.attrs["bias_se"] == pytest.approx(err.std(ddof=1) / np.sqrt(60))

    def test_granger_is_the_classical_f_test_of_the_lags_of_x(self, adl):
        r = sp.ardl(adl, "y", ["x", "z"], lags=1, x_lags=2, vce="nonrobust")
        y, x, z = (adl[c].to_numpy() for c in ("y", "x", "z"))
        full = np.column_stack(
            [np.ones(N - 2), y[1:-1], x[1:-1], x[:-2], z[1:-1], z[:-2]]
        )
        small = full[:, [0, 1, 4, 5]]
        rss = lambda M: float(  # noqa: E731
            ((y[2:] - M @ np.linalg.lstsq(M, y[2:], rcond=None)[0]) ** 2).sum()
        )
        f = ((rss(small) - rss(full)) / 2) / (rss(full) / (N - 2 - 6))
        got = r.granger("x")
        assert got["statistic"] == pytest.approx(f, rel=1e-8)
        assert got["pvalue"] == pytest.approx(stats.f.sf(f, 2, N - 8), rel=1e-6)

    def test_hac_with_zero_lags_is_the_white_estimator(self, adl):
        hac0 = sp.ardl(adl, "y", "x", vce="hac", hac_lags=0).model.std_errors
        hc0 = sp.ardl(adl, "y", "x", vce="hc0").model.std_errors
        np.testing.assert_allclose(hac0, hc0, rtol=1e-10)
        hac3 = sp.ardl(adl, "y", "x", vce="hac", hac_lags=3).model.std_errors
        assert not np.allclose(hac3, hc0, rtol=1e-6)
        with pytest.raises(MethodIncompatibility, match="only apply to robust='hac'"):
            sp.ardl(adl, "y", "x", vce="hc1", hac_lags=3)

    def test_time_column_sorts_the_rows(self, adl):
        dated = adl.assign(q=pd.period_range("1980Q1", periods=N, freq="Q"))
        shuffled = dated.sample(frac=1, random_state=3)
        a = sp.ardl(shuffled, "y", "x", lags=2, time="q")
        b = sp.ardl(adl, "y", "x", lags=2)
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-12)
        assert a._sample_index[0] == pd.Period("1980Q3", freq="Q")

    @pytest.mark.parametrize(
        "kw, exc, match",
        [
            ({"trend": "quadratic"}, MethodIncompatibility, "unknown trend"),
            ({"lags": "hqic"}, MethodIncompatibility, "unknown lag rule"),
            ({"lags": -1}, MethodIncompatibility, "non-negative"),
            ({"x_lags": "all"}, MethodIncompatibility, "unknown x_lags"),
            ({"x_lags": {"q": 1}}, MethodIncompatibility, "no entry for"),
            ({"lags": "bic", "x_lags": "aic"}, MethodIncompatibility, "different"),
            ({"sample": (100, 10)}, DataInsufficient, "sample is empty"),
            ({"sample": (0, 999)}, MethodIncompatibility, "not a date of the data"),
            ({"vce": "zzz"}, MethodIncompatibility, "Unknown robust option"),
        ],
    )
    def test_option_errors(self, adl, kw, exc, match):
        with pytest.raises(exc, match=match):
            sp.ardl(adl, "y", "x", **kw)

    def test_data_errors(self, adl):
        hole = adl.copy()
        hole.loc[80, "x"] = np.nan
        with pytest.raises(MethodIncompatibility, match="across a gap"):
            sp.ardl(hole, "y", "x")
        twice = pd.concat([adl, adl.iloc[[5]]])
        with pytest.raises(MethodIncompatibility, match="repeated values"):
            sp.ardl(twice, "y", "x")
        with pytest.raises(MethodIncompatibility, match="selects the lags of x"):
            sp.ardl(adl, "y", x_lags="bic")
        with pytest.raises(MethodIncompatibility, match="nothing to estimate"):
            sp.ardl(adl, "y", lags=0, trend="n")
        with pytest.raises(DataInsufficient, match="observations for"):
            sp.ardl(adl.iloc[:7], "y", "x", lags=3, x_lags=3)

    def test_forecast_errors(self, adl):
        r = sp.ardl(adl, "y", "x", lags=1, x_lags=1)
        with pytest.raises(MethodIncompatibility, match="steps must be at least 1"):
            r.forecast(steps=0)
        with pytest.raises(MethodIncompatibility, match="x after the sample"):
            r.forecast(steps=2)
        with pytest.raises(MethodIncompatibility, match="exog must have columns"):
            r.forecast(steps=3, exog=pd.DataFrame({"x": [1.0]}))
        with pytest.raises(MethodIncompatibility, match="missing values"):
            r.forecast(steps=2, exog=pd.DataFrame({"x": [np.nan]}))
        with pytest.raises(MethodIncompatibility, match="unknown window"):
            r.poos(start=100, window="zz")
        with pytest.raises(MethodIncompatibility, match="not a regressor"):
            r.granger("q")

    def test_explosive_model_has_no_long_run(self):
        t = np.arange(60.0)
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({"y": 1.05**t + rng.normal(0, 0.01, 60), "x": t % 3})
        r = sp.ardl(frame, "y", "x", lags=1)
        with pytest.raises(MethodIncompatibility, match="no long-run level"):
            r.long_run()

    def test_alpha_outside_the_unit_interval_is_refused(self, adl):
        with pytest.raises(StatsPAIError, match="alpha"):
            sp.ardl(adl, "y", "x", alpha=3).forecast()

    def test_column_names_that_are_not_identifiers(self, adl):
        renamed = adl.rename(columns={"y": "g rate"})
        got = sp.ardl(renamed, "g rate", "x", lags=1)
        ref = sp.ardl(adl, "y", "x", lags=1)
        np.testing.assert_allclose(got.params.to_numpy(), ref.params.to_numpy())

    def test_collinear_regressor_is_reported(self, adl):
        twin = adl.assign(w=2 * adl["x"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                r = sp.ardl(twin, "y", ["x", "w"], lags=1)
            except StatsPAIError:
                return
        said = any("collinear" in str(w.message).lower() for w in caught)
        assert said and set(r.regressors) <= set(r.params.index)


# ------------------------------------------------------- local projections

T_LP = 200


@pytest.fixture(scope="module")
def lp_data():
    rng = np.random.default_rng(3)
    s = rng.normal(size=T_LP)
    c = rng.normal(size=T_LP)
    y = np.zeros(T_LP)
    for t in range(1, T_LP):
        y[t] = 0.5 * y[t - 1] + 0.8 * s[t] + 0.3 * c[t] + rng.normal(0, 0.5)
    return pd.DataFrame({"y": y, "shock": s, "c": c})


def _lp_by_hand(frame, h, nw, auto=True, controls=True, cumulative=False):
    """Coefficient on the shock and its Newey-West standard error."""
    y, s, c = (frame[k].to_numpy() for k in ("y", "shock", "c"))
    idx = np.arange(1 if (auto or cumulative) else 0, T_LP - h)
    lhs = y[idx + h] - (y[idx - 1] if cumulative else 0.0)
    cols = [np.ones(len(idx)), s[idx]]
    if auto:
        cols += [y[idx - 1], s[idx - 1]]
    if controls:
        cols.append(c[idx])
    X = np.column_stack(cols)
    b = np.linalg.lstsq(X, lhs, rcond=None)[0]
    u = X * (lhs - X @ b)[:, None]
    meat = u.T @ u
    for lag in range(1, nw + 1):
        g = u[lag:].T @ u[:-lag]
        meat += (1 - lag / (nw + 1)) * (g + g.T)
    bread = np.linalg.inv(X.T @ X)
    return b[1], np.sqrt((bread @ meat @ bread)[1, 1]), len(idx)


class TestLocalProjections:
    def test_direct_projection_with_the_default_bandwidth(self, lp_data):
        r = sp.local_projections(lp_data, "y", "shock", controls=["c"], horizons=4)
        for h in range(5):
            b, se, n = _lp_by_hand(lp_data, h, nw=6)  # round(1.5 * 4)
            assert r.irf[h] == pytest.approx(b, abs=1e-10)
            assert r.se[h] == pytest.approx(se, rel=1e-9)
            assert r.n_obs_per_horizon[h] == n
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(r.ci_upper, r.irf + z * r.se, rtol=1e-12)
        assert list(r.to_frame().columns) == [
            "horizon", "irf", "se", "ci_lower", "ci_upper", "n",
        ]  # fmt: skip

    def test_auto_lag_false_is_the_bare_regression(self, lp_data):
        r = sp.local_projections(
            lp_data, "y", "shock", horizons=2, auto_lag=False, nw_lags=5
        )
        for h in range(3):
            b, se, n = _lp_by_hand(lp_data, h, nw=5, auto=False, controls=False)
            assert r.irf[h] == pytest.approx(b, abs=1e-10)
            assert r.se[h] == pytest.approx(se, rel=1e-9)
            assert n == T_LP - h

    def test_alias_names_and_alpha(self, lp_data):
        a = sp.local_projections(lp_data, y="y", shock="shock", horizons=2)
        b = sp.local_projections(lp_data, outcome="y", shock="shock", horizons=2)
        np.testing.assert_array_equal(a.irf, b.irf)
        wide = sp.local_projections(lp_data, "y", "shock", horizons=2, alpha=0.01)
        np.testing.assert_allclose(
            wide.ci_upper - wide.irf, stats.norm.ppf(0.995) * wide.se, rtol=1e-12
        )

    def test_cholesky_impact_is_the_ratio_of_the_var_factor(self, lp_data):
        r = sp.local_projections(
            lp_data, "y", "shock", horizons=2, identification="lpirfs_cholesky",
            endog_order=["shock", "y"],
        )  # fmt: skip
        Y = lp_data[["shock", "y"]].to_numpy()
        X = np.column_stack([np.ones(T_LP - 1), Y[:-1]])
        U = Y[1:] - X @ np.linalg.lstsq(X, Y[1:], rcond=None)[0]
        chol = np.linalg.cholesky(np.cov(U, rowvar=False, ddof=1))
        assert r.irf[0] == pytest.approx(chol[1, 0] / chol[0, 0], rel=1e-10)
        # horizon 1: projection coefficients mapped through the unit shock
        beta = np.linalg.lstsq(X, Y[1:, 1], rcond=None)[0][1:]
        assert r.irf[1] == pytest.approx(
            beta @ [1.0, chol[1, 0] / chol[0, 0]], rel=1e-10
        )

    def test_option_errors(self, lp_data):
        with pytest.raises(ValueError, match="identification must be"):
            sp.local_projections(lp_data, "y", "shock", identification="zz")
        kw = dict(identification="lpirfs_cholesky", horizons=2)
        with pytest.raises(ValueError, match="controls are not supported"):
            sp.local_projections(lp_data, "y", "shock", controls=["c"], **kw)
        with pytest.raises(ValueError, match="cumulative=True is not supported"):
            sp.local_projections(lp_data, "y", "shock", cumulative=True, **kw)
        with pytest.raises(ValueError, match="must include outcome"):
            sp.local_projections(lp_data, "y", "shock", endog_order=["shock"], **kw)
        with pytest.raises(ValueError, match="duplicate"):
            sp.local_projections(
                lp_data, "y", "shock", endog_order=["y", "shock", "y"], **kw
            )
        with pytest.raises(DataInsufficient, match="too few observations"):
            sp.local_projections(lp_data.iloc[:4], "y", "shock", **kw)
        hole = lp_data.copy()
        hole.loc[5, "y"] = np.nan
        with pytest.raises(ValueError, match="complete endog_order data"):
            sp.local_projections(hole, "y", "shock", **kw)

    def test_cumulative_response_is_the_change_since_before_the_shock(self, lp_data):
        r = sp.local_projections(lp_data, "y", "shock", horizons=3, cumulative=True)
        for h in range(4):
            b, _, _ = _lp_by_hand(lp_data, h, nw=4, controls=False, cumulative=True)
            assert r.irf[h] == pytest.approx(b, abs=1e-10)

    def test_nw_lags_is_the_truncation_lag_at_every_horizon(self, lp_data):
        r = sp.local_projections(
            lp_data, "y", "shock", controls=["c"], horizons=4, nw_lags=0
        )
        for h in range(5):
            _, se, _ = _lp_by_hand(lp_data, h, nw=0)
            assert r.se[h] == pytest.approx(se, rel=1e-9)

    def test_cholesky_band_is_ordered_under_negative_weights(self):
        rng = np.random.default_rng(3)
        rng.normal(size=2 * T_LP)  # the draws of the scratch reproduction
        L = np.array([[1.0, 0, 0], [-0.9, 1, 0], [-0.8, -0.5, 1]])
        u = rng.normal(size=(T_LP, 3)) @ L.T
        Y = np.zeros((T_LP, 3))
        for t in range(1, T_LP):
            Y[t] = 0.4 * Y[t - 1] + u[t]
        frame = pd.DataFrame(Y, columns=["a", "b", "cc"])
        r = sp.local_projections(
            frame, "cc", "a", horizons=4, identification="lpirfs_cholesky",
            endog_order=["a", "b", "cc"],
        )  # fmt: skip
        assert (r.se[1:] > 0).all()
        assert (r.ci_lower <= r.ci_upper).all()

    def test_alpha_outside_the_unit_interval_is_refused(self, lp_data):
        with pytest.raises(StatsPAIError, match="alpha"):
            sp.local_projections(lp_data, "y", "shock", horizons=2, alpha=4)

    def test_collinear_control_is_named(self, lp_data):
        twin = lp_data.assign(s2=2 * lp_data["shock"])
        with pytest.raises(StatsPAIError):
            sp.local_projections(twin, "y", "shock", controls=["s2"], horizons=2)


# ------------------------------------------------------------ cointegration

T_CI = 200


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(size=T_CI))
    other = np.cumsum(rng.normal(size=T_CI))
    return pd.DataFrame(
        {
            "a": common + rng.normal(scale=0.5, size=T_CI),
            "b": 0.8 * common + rng.normal(scale=0.5, size=T_CI),
            "c": other + 0.3 * common + rng.normal(scale=0.5, size=T_CI),
        }
    )


def _canonical(Y, lags, case):
    """Squared canonical correlations between the differences and the lagged
    levels after partialling out the short-run terms, by QR and SVD."""
    dY = np.diff(Y, axis=0)
    n = len(Y) - 1 - lags
    lhs, lev = dY[lags:], Y[lags:-1]
    tt = np.arange(lags + 2.0, len(Y) + 1)
    Z = [dY[lags - j : len(dY) - j] for j in range(1, lags + 1)]
    if case in ("c", "rt", "ct"):
        Z.append(np.ones((n, 1)))
    if case == "ct":
        Z.append(tt.reshape(-1, 1))
    if case == "rc":
        lev = np.column_stack([lev, np.ones(n)])
    if case == "rt":
        lev = np.column_stack([lev, tt])
    if Z:
        Zm = np.column_stack(Z)
        lhs = lhs - Zm @ np.linalg.lstsq(Zm, lhs, rcond=None)[0]
        lev = lev - Zm @ np.linalg.lstsq(Zm, lev, rcond=None)[0]
    q0, q1 = np.linalg.qr(lhs)[0], np.linalg.qr(lev)[0]
    rho = np.linalg.svd(q0.T @ q1, compute_uv=False)
    return rho[: Y.shape[1]] ** 2, n


class TestJohansen:
    @pytest.mark.parametrize("trend", ["n", "rc", "c", "rt", "ct"])
    @pytest.mark.parametrize("lags", [0, 2])
    def test_statistics_from_canonical_correlations(self, system, trend, lags):
        lam, n = _canonical(system.to_numpy(), lags, trend)
        trace = sp.johansen(system, lags=lags, trend=trend)
        maxeig = sp.johansen(system, lags=lags, trend=trend, test="maxeig")
        # eigenvalues of a symmetric problem against singular values
        np.testing.assert_allclose(trace.eigenvalues, lam, rtol=1e-8)
        np.testing.assert_allclose(maxeig.test_stats, -n * np.log(1 - lam), rtol=1e-8)
        np.testing.assert_allclose(
            trace.test_stats, np.cumsum((-n * np.log(1 - lam))[::-1])[::-1], rtol=1e-8
        )
        assert trace.n_used == n and trace.trend == trend

    @pytest.mark.parametrize("trend, det", [("n", -1), ("c", 0)])
    def test_matches_statsmodels(self, system, trend, det):
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        ref = coint_johansen(system.to_numpy(), det, 2)
        got = sp.johansen(system, lags=2, trend=trend)
        np.testing.assert_allclose(got.test_stats, ref.lr1, rtol=1e-9)
        np.testing.assert_allclose(got.eigenvalues, ref.eig, rtol=1e-9)

    @pytest.mark.parametrize("trend", ["rc", "c", "rt", "ct"])
    def test_invariant_to_a_nonsingular_recombination_of_the_series(
        self, system, trend
    ):
        A = np.array([[1, 0.5, 0], [0, 2, 0.3], [1, 0, -1.0]])
        mixed = pd.DataFrame(system.to_numpy() @ A * 100 + 5, columns=list("abc"))
        a = sp.johansen(system, lags=2, trend=trend)
        b = sp.johansen(mixed, lags=2, trend=trend)
        np.testing.assert_allclose(a.test_stats, b.test_stats, rtol=1e-8)
        assert a.rank == b.rank

    def test_rank_decision_and_stata_aliases(self, system):
        r = sp.johansen(system, lags=2)
        expect = 0
        for stat, cv in zip(r.test_stats, r.critical_values):
            if stat > cv:
                expect += 1
            else:
                break
        assert r.rank == expect == 1
        assert r.eigenvectors[0] == pytest.approx(np.ones(3))
        alias = sp.johansen(system, lags=2, trend="rconstant")
        short = sp.johansen(system, lags=2, trend="rc")
        np.testing.assert_array_equal(alias.test_stats, short.test_stats)
        strict = sp.johansen(system, lags=2, alpha=0.01)
        assert all(a > b for a, b in zip(strict.critical_values, r.critical_values))

    @pytest.mark.parametrize(
        "kw, exc, match",
        [
            ({"trend": "zz"}, MethodIncompatibility, "trend must be one of"),
            ({"test": "zz"}, MethodIncompatibility, "test must be"),
            ({"alpha": 0.10}, MethodIncompatibility, "alpha must be 0.05 or 0.01"),
            ({"lags": -1}, MethodIncompatibility, "lags must be >= 0"),
        ],
    )
    def test_option_errors(self, system, kw, exc, match):
        with pytest.raises(exc, match=match):
            sp.johansen(system, **kw)

    def test_too_few_observations(self, system):
        with pytest.raises(DataInsufficient, match="too few observations"):
            sp.johansen(system.iloc[:6], lags=3)

    def test_restriction_tests_from_the_reduced_eigenproblems(self, system):
        from statspai.timeseries.cointegration import _johansen_residuals

        R0, R1, n = _johansen_residuals(system.to_numpy(), 2, "c")

        def top(r0, r1):
            s00, s11, s01 = r0.T @ r0 / n, r1.T @ r1 / n, r0.T @ r1 / n
            m = np.linalg.solve(s11, s01.T @ np.linalg.solve(s00, s01))
            return np.sort(np.linalg.eigvals(m).real)[::-1][0]

        free = top(R0, R1)
        H = np.array([[1.0, 0], [0, 1], [0, 0]])
        on_beta = sp.johansen_lrtest(system, rank=1, lags=2, beta=H)
        lr = n * np.log((1 - top(R0, R1 @ H)) / (1 - free))
        assert on_beta.statistic == pytest.approx(lr, rel=1e-7)
        assert on_beta.df == 1
        assert on_beta.pvalue == pytest.approx(stats.chi2.sf(lr, 1), rel=1e-6)

        perp = np.array([[0.0], [0], [1]])
        cond = R0 @ perp
        M = np.eye(n) - cond @ np.linalg.solve(cond.T @ cond, cond.T)
        on_alpha = sp.johansen_lrtest(system, rank=1, lags=2, loading=H)
        lr_a = n * np.log((1 - top(M @ R0 @ H, M @ R1)) / (1 - free))
        assert on_alpha.statistic == pytest.approx(lr_a, rel=1e-7)
        assert on_alpha.df == 1

    def test_restriction_test_errors(self, system):
        with pytest.raises(MethodIncompatibility, match="exactly one of"):
            sp.johansen_lrtest(system, rank=1)
        with pytest.raises(MethodIncompatibility, match="rank must be between"):
            sp.johansen_lrtest(system, rank=3, beta=np.eye(3)[:, :2])
        with pytest.raises(MethodIncompatibility, match="fewer than 3"):
            sp.johansen_lrtest(system, rank=1, beta=np.eye(3))
        with pytest.raises(MethodIncompatibility, match="trend must be one of"):
            sp.johansen_lrtest(system, rank=1, trend="zz", beta=np.eye(3)[:, :2])

    def test_missing_row_is_refused(self, system):
        hole = system.copy()
        hole.loc[100, "b"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError, match="(?i)missing|gap|nan"):
                sp.johansen(hole, lags=1)

    def test_collinear_series_are_named(self, system):
        with pytest.raises(StatsPAIError):
            sp.johansen(system.assign(d=2 * system["a"]))


class TestEngleGranger:
    @pytest.mark.parametrize("trend", ["c", "ct", "ctt"])
    @pytest.mark.parametrize("lags", [0, 2])
    def test_matches_statsmodels_coint(self, system, trend, lags):
        from statsmodels.tsa.stattools import coint

        stat, p, cv = coint(
            system["a"], system[["b", "c"]], trend=trend, maxlag=lags, autolag=None
        )
        r = sp.engle_granger(system, variables=["a", "b", "c"], lags=lags, trend=trend)
        assert r.test_stats == pytest.approx(stat, rel=1e-9)
        assert r.pvalue == pytest.approx(p, rel=1e-6, abs=1e-12)
        # statsmodels evaluates the same response surface at n - 1
        np.testing.assert_allclose(r.critical_values, cv, rtol=1e-5)
        assert r.rank == int(stat < cv[1]) and r.lags == lags

    def test_default_lag_rule_and_alpha(self, system):
        r = sp.engle_granger(system, variables=["a", "b"])
        assert r.lags == int(np.floor(4 * (T_CI / 100) ** 0.25))
        strict = sp.engle_granger(system, variables=["a", "b"], alpha=0.01)
        assert strict.test_stats == r.test_stats
        assert "Cointegrated" in r.summary() or "Not cointegrated" in r.summary()

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"trend": "n"}, "trend must be"),
            ({"alpha": 0.2}, "alpha must be 0.01, 0.05 or 0.10"),
            ({"lags": -1}, "lags must be >= 0"),
            ({"variables": ["a"]}, r"needs 2\.\.12 variables"),
        ],
    )
    def test_option_errors(self, system, kw, match):
        args = {"variables": ["a", "b"], **kw}
        with pytest.raises(MethodIncompatibility, match=match):
            sp.engle_granger(system, **args)

    def test_missing_row_is_refused(self, system):
        hole = system.copy()
        hole.loc[100, "b"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError, match="(?i)missing|gap|nan"):
                sp.engle_granger(hole, variables=["a", "b"])

    def test_more_lags_than_observations_is_refused(self, system):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.engle_granger(system.iloc[:12], variables=["a", "b"], lags=50)

    def test_exactly_collinear_pair_is_refused(self, system):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.engle_granger(system.assign(d=2 * system["a"]), variables=["d", "a"])


class TestScanFixDetails:
    def test_negative_truncation_lag_is_refused(self, lp_data):
        with pytest.raises(MethodIncompatibility, match="nw_lags"):
            sp.local_projections(lp_data, "y", "shock", horizons=2, nw_lags=-1)

    def test_cholesky_band_keeps_the_width_of_the_reference_band(self, lp_data):
        # non-negative weights: nothing to reorder, se is the half-width
        r = sp.local_projections(
            lp_data, "y", "shock", horizons=3, identification="lpirfs_cholesky",
            endog_order=["shock", "y"],
        )  # fmt: skip
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(r.ci_upper - r.ci_lower, 2 * z * r.se, rtol=1e-12)

    def test_johansen_restriction_test_refuses_a_gap(self, system):
        hole = system.copy()
        hole.loc[100, "b"] = np.nan
        with pytest.raises(MethodIncompatibility, match="position 100"):
            sp.johansen_lrtest(hole, rank=1, beta_known=[[1], [-1], [0]])

    def test_cointegration_tests_trim_rows_missing_at_the_ends(self, system):
        ragged = system.copy()
        ragged.loc[[0, 1], "a"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = sp.engle_granger(ragged, variables=["a", "b"])
            ref = sp.engle_granger(system.iloc[2:], variables=["a", "b"])
            jg = sp.johansen(ragged, lags=1)
            jr = sp.johansen(system.iloc[2:], lags=1)
        assert got.test_stats == ref.test_stats
        np.testing.assert_array_equal(jg.test_stats, jr.test_stats)

    def test_ardl_with_quoted_names_keeps_its_methods(self, adl):
        renamed = adl.rename(columns={"y": "g rate", "x": "x-1"})
        got = sp.ardl(renamed, "g rate", "x-1", lags=2)
        ref = sp.ardl(adl, "y", "x", lags=2)
        assert list(got.params.index) == [
            "Intercept",
            "g rate_L1",
            "g rate_L2",
            "x-1_L1",
        ]
        np.testing.assert_allclose(got.forecast().to_numpy(), ref.forecast().to_numpy())
        assert got.granger()["statistic"] == pytest.approx(ref.granger()["statistic"])
        np.testing.assert_allclose(
            got.long_run().to_numpy(dtype=float), ref.long_run().to_numpy(dtype=float)
        )
        np.testing.assert_allclose(
            got.poos(start=120).to_numpy(), ref.poos(start=120).to_numpy()
        )

    def test_ardl_drops_the_collinear_term_from_its_regressors(self, adl):
        twin = adl.assign(w=2 * adl["x"])
        with pytest.warns(UserWarning, match="w_L1 omitted"):
            r = sp.ardl(twin, "y", ["x", "w"], lags=1)
        ref = sp.ardl(adl, "y", "x", lags=1)
        assert r.regressors == ref.regressors
        np.testing.assert_allclose(r.params.to_numpy(), ref.params.to_numpy())
        np.testing.assert_allclose(r.forecast().to_numpy(), ref.forecast().to_numpy())


class TestArdlOmittedRegressor:
    @pytest.fixture(scope="class")
    def fit(self, adl):
        twin = adl.assign(w=2 * adl["x"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sp.ardl(twin, "y", ["x", "w"], lags=1)

    def test_result_describes_the_model_that_was_estimated(self, fit, adl):
        ref = sp.ardl(adl, "y", "x", lags=1)
        assert fit.x == ["x"] and fit.x_lags == ref.x_lags
        assert fit.omitted == ["w_L1"] and fit.omitted_x == ["w"]
        assert fit.method == ref.method
        head, last = fit.summary().splitlines()[0], fit.summary().splitlines()[-1]
        assert head == ref.summary().splitlines()[0]
        assert "collinearity" in last and "w_L1" in last
        assert ref.omitted == [] and "collinearity" not in ref.summary()

    def test_granger_on_the_omitted_variable_says_why(self, fit, adl):
        with pytest.raises(MethodIncompatibility, match="omitted.*collinearity"):
            fit.granger("w")
        ref = sp.ardl(adl, "y", "x", lags=1)
        assert fit.granger("x")["pvalue"] == pytest.approx(
            ref.granger("x")["pvalue"], rel=1e-10
        )

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5, -0.1])
    def test_interval_level_out_of_range_is_refused(self, adl, alpha):
        fit = sp.ardl(adl, "y", "x", lags=1)
        with pytest.raises(MethodIncompatibility, match="ardl.forecast: alpha"):
            fit.forecast(alpha=alpha)
        with pytest.raises(MethodIncompatibility, match="ardl.long_run: alpha"):
            fit.long_run(alpha=alpha)
