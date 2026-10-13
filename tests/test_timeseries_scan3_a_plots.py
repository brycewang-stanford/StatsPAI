"""Plots and printed summaries of the time-series model results (scan
round 3): the data drawn must be the numbers the result reports.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402

N = 150


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(8)
    x = np.zeros(N)
    y = np.zeros(N)
    for t in range(1, N):
        x[t] = 0.5 * x[t - 1] + rng.normal()
        y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 1] + rng.normal()
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _fit_arima(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.arima(*args, **kw)


class TestArimaPlot:
    def test_lines_are_fitted_observed_and_forecast(self, frame):
        y = frame["y"].to_numpy() + 3
        r = _fit_arima(y, order=(1, 0, 0))
        ax = r.plot(horizon=5, alpha=0.10)
        fitted, observed, forecast = ax.get_lines()
        np.testing.assert_allclose(fitted.get_ydata(), r.fitted_values)
        np.testing.assert_allclose(observed.get_ydata(), y, atol=1e-10)
        fc = r.forecast(5, alpha=0.10)
        np.testing.assert_allclose(forecast.get_ydata(), fc["forecast"])
        np.testing.assert_array_equal(forecast.get_xdata(), np.arange(N, N + 5))
        band = ax.collections[0].get_paths()[0].vertices[:, 1]
        assert band.min() == pytest.approx(fc["lower"].min())
        assert band.max() == pytest.approx(fc["upper"].max())

    def test_model_with_regressors_draws_no_forecast(self, frame):
        r = _fit_arima("y", order=(1, 0, 0), exog=["x"], data=frame)
        _, ax = plt.subplots()
        assert r.plot(ax=ax) is ax
        assert len(ax.get_lines()) == 2 and not ax.collections
        text = r.summary()
        assert text.splitlines()[0] == "ARIMA(1, 0, 0) with regressors"
        assert "ar.L1" in text and repr(r) == text

    def test_observed_points_of_a_box_cox_fit_are_the_data(self, frame):
        level = np.exp(frame["y"].to_numpy() / 5 + 2)
        r = _fit_arima(level, order=(1, 0, 0), boxcox=0)
        observed = r.plot(horizon=3).get_lines()[1]
        np.testing.assert_allclose(observed.get_ydata(), level, rtol=1e-8)


class TestVarPlots:
    def test_irf_grid_draws_the_reported_paths(self, frame):
        v = sp.var(frame, variables=["y", "x"], lags=2)
        fig = v.plot_irf(periods=6, orthogonal=False)
        paths = v.irf(6, orthogonal=False)["irf"]
        axes = np.array(fig.axes).reshape(2, 2)
        for i, resp in enumerate(["y", "x"]):
            for j, imp in enumerate(["y", "x"]):
                ax = axes[i][j]
                assert ax.get_title() == f"{imp} -> {resp}"
                np.testing.assert_allclose(
                    ax.get_lines()[0].get_ydata(), paths[f"{imp} -> {resp}"]
                )
        assert "y" in v.summary()

    def test_structural_summary_reports_the_overidentification_test(self, frame):
        v = sp.var(frame, variables=["y", "x"], lags=1)
        res = sp.svar(v, B=[[np.nan, 0], [0, np.nan]])
        text = res.summary()
        assert f"chi2(1) = {res.overid['statistic']:.4f}" in text
        assert "Impact matrix" in text
        assert repr(res) == "SVARResult(short-run restrictions, K=2)"
        sign = sp.svar(v, sign={"s": {"y": "+"}}, n_draws=20, seed=0)
        assert "Admissible rotations kept: 20" in sign.summary()


class TestOtherResults:
    def test_break_plot_marks_the_break_dates(self):
        rng = np.random.default_rng(2)
        y = np.r_[rng.normal(0, 1, 60), rng.normal(3, 1, 60)]
        r = sp.structural_break(pd.DataFrame({"y": y}), y="y", method="sup-f")
        assert r.break_dates == [r.sup_break]
        ax = r.plot()
        assert [line.get_xdata()[0] for line in ax.get_lines()] == r.break_dates
        text = r.summary()
        assert f"F-statistic: {r.f_stats:.4f}" in text
        assert f"P-value: {r.p_values:.4f}" in text

    def test_local_projection_plot_and_summary(self, frame):
        r = sp.local_projections(frame, "y", "x", horizons=4)
        ax = r.plot()
        np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), r.irf)
        band = ax.collections[0].get_paths()[0].vertices[:, 1]
        assert band.min() == pytest.approx(r.ci_lower.min())
        assert band.max() == pytest.approx(r.ci_upper.max())
        assert ax.get_legend().get_texts()[1].get_text() == "95% CI"
        assert "Local Projections" in r.summary() and repr(r) == r.summary()

    def test_garch_and_cointegration_summaries(self, frame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = sp.garch(frame["y"].to_numpy(), p=0, q=1)
        assert g.summary().splitlines()[0] == "GARCH(0,1)"
        assert f"{g.params['alpha[1]']:.6f}" in repr(g)
        walk = frame.cumsum()
        eg = sp.engle_granger(walk, variables=["y", "x"], lags=1)
        assert f"ADF test statistic: {eg.test_stats:.4f}" in eg.summary()
        assert "Engle-Granger" in repr(eg)
        jo = sp.johansen(walk, lags=1)
        assert "Used: 148" in jo.summary()
        expect = f"Johansen (trace): rank={jo.rank}, n_vars=2, lags=1"
        assert repr(jo) == f"CointegrationResult({expect})"
