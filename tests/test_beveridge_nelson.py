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


# --- ARMA models by exact maximum likelihood ---------------------------------


def _arima11(phi, theta, n=400, drift=0.3, seed=0, sd=1.0):
    rng = np.random.default_rng(seed)
    e = sd * rng.normal(size=n + 200)
    x = np.zeros(n + 200)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t] + theta * e[t - 1]
    return np.concatenate([[50.0], 50.0 + np.cumsum(drift + x[200:])])


def _projection_cycle(bn, y, horizon=400):
    """``-sum_h E_t(x[t+h])`` from the autocovariances of the fitted ARMA.

    The conditional expectation of the next ``horizon`` demeaned
    differences given the first ``t`` is ``S21 S11^-1 x[:t]`` with the
    blocks of their joint covariance; no filter and no state space.
    """
    from statsmodels.tsa.arima_process import arma_acovf

    x = np.diff(y) - bn.drift
    n = x.size
    ar = np.r_[1.0, -bn.ar_coefs]
    ma = np.r_[1.0, bn.ma_coefs]
    gamma = arma_acovf(ar, ma, nobs=n + horizon, sigma2=bn.sigma2)
    lag = np.arange(n + horizon)
    S = gamma[np.abs(lag[:, None] - lag[None, :])]
    out = np.full(y.size, np.nan)
    for t in range(1, n + 1):
        ahead = S[t : t + horizon, :t] @ np.linalg.solve(S[:t, :t], x[:t])
        out[t] = -ahead.sum()
    return out


@pytest.mark.parametrize("order", [(0, 1), (1, 1), (2, 2), (2, 0), (0, 3)])
def test_arma_cycle_is_the_long_horizon_forecast(order):
    y = _arima11(0.5, 0.4, n=160, seed=3)
    bn = beveridge_nelson(y, order=order)
    assert (bn.method, bn.order, bn.ma_order) == ("mle", *order)
    # forecasts die out geometrically (largest root about 0.7): 400 steps
    # leave nothing; 1e-9 covers the solve of a 160 x 160 covariance
    np.testing.assert_allclose(
        bn.cycle.to_numpy()[1:], _projection_cycle(bn, y)[1:], atol=1e-9
    )
    assert np.isnan(bn.cycle.iloc[0]) and bn.cycle.notna().sum() == y.size - 1
    np.testing.assert_allclose((bn.trend + bn.cycle).to_numpy()[1:], y[1:], rtol=1e-13)
    psi1 = (1 + bn.ma_coefs.sum()) / (1 - bn.ar_coefs.sum())
    assert bn.long_run_multiplier == pytest.approx(psi1, rel=1e-13)
    assert bn.intercept == pytest.approx(bn.drift * (1 - bn.ar_coefs.sum()), rel=1e-13)


def test_arma_cycle_agrees_with_the_forecasts_of_sp_arima():
    import statspai as sp

    y = _arima11(0.5, 0.4, n=200, seed=5)
    bn = beveridge_nelson(y, order=(1, 1))
    fit = sp.arima(np.diff(y), order=(1, 0, 1), trend="c")
    # the same fitted model: its residuals, and its forecasts from the end
    np.testing.assert_allclose(bn.residuals.to_numpy()[1:], fit.residuals, atol=1e-10)
    assert bn.sigma2 == fit.sigma2 and bn.loglik == fit.log_likelihood
    path = np.asarray(fit.forecast(600)["forecast"], dtype=float)
    assert bn.cycle.iloc[-1] == pytest.approx(-(path - bn.drift).sum(), abs=1e-9)


def test_ima11_cycle_is_minus_theta_times_the_innovation():
    theta = -0.6
    y = _arima11(0.0, theta, n=300, seed=1)
    bn = beveridge_nelson(y, order=(0, 1))
    th = float(bn.ma_coefs[0])
    assert abs(th - theta) < 0.15
    assert bn.long_run_multiplier == pytest.approx(1 + th, rel=1e-14)
    assert bn.variance_ratio == pytest.approx((1 + th) ** 2 / (1 + th**2), rel=1e-12)
    v = bn.residuals.to_numpy()[1:]
    cycle = bn.cycle.to_numpy()[1:]
    # exactly: cycle = -theta E_t(e_t) = -theta sigma2 v_t / S_t, with the
    # prediction-error variance of an MA(1), S_1 = sigma2 (1 + theta^2),
    # S_{t+1} = sigma2 (1 + theta^2 - theta^2 sigma2 / S_t)
    S = np.empty(v.size)
    S[0] = bn.sigma2 * (1 + th**2)
    for t in range(1, v.size):
        S[t] = bn.sigma2 * (1 + th**2 - th**2 * bn.sigma2 / S[t - 1])
    np.testing.assert_allclose(cycle, -th * bn.sigma2 * v / S, atol=1e-12)
    # S_t -> sigma2 at rate theta^(2t): after 60 dates the textbook form
    np.testing.assert_allclose(cycle[60:], -th * v[60:], atol=1e-12)
    # and the trend is then a random walk with innovation psi(1) e_t
    step = np.diff(bn.trend.to_numpy()[1:])[60:]
    np.testing.assert_allclose(step, bn.drift + (1 + th) * v[61:], atol=1e-11)


def test_arima111_closed_form():
    y = _arima11(0.6, -0.3, n=400, seed=2)
    bn = beveridge_nelson(y, order=(1, 1))
    phi, th = float(bn.ar_coefs[0]), float(bn.ma_coefs[0])
    x = np.diff(y) - bn.drift
    v = bn.residuals.to_numpy()[1:]
    # sum_{j>=1} E_t x[t+j] = (phi x[t] + theta e[t]) / (1 - phi); the
    # prediction error is the innovation once the filter has settled
    closed = -(phi * x + th * v) / (1 - phi)
    np.testing.assert_allclose(bn.cycle.to_numpy()[1:][80:], closed[80:], atol=1e-11)
    assert bn.long_run_multiplier == pytest.approx((1 + th) / (1 - phi), rel=1e-13)
    psi = np.r_[1.0, (phi + th) * phi ** np.arange(2000)]
    assert bn.variance_ratio == pytest.approx(
        bn.long_run_multiplier**2 / np.sum(psi**2), rel=1e-10
    )


def test_ml_autoregression_is_close_to_but_not_the_ols_one():
    y = _arima([0.5, -0.2], n=400, seed=4)
    ols = beveridge_nelson(y, order=2)
    ml = beveridge_nelson(y, order=(2, 0))
    same = beveridge_nelson(y, order=2, arma=True)
    assert ols.method == "ols" and ml.method == "mle" and ols.ma_order == 0
    np.testing.assert_array_equal(ml.cycle.to_numpy(), same.cycle.to_numpy())
    # different estimators of the same model: sampling-error-sized gaps
    assert 1e-6 < np.max(np.abs(ml.ar_coefs - ols.ar_coefs)) < 0.02
    assert 1e-8 < abs(ml.drift - ols.drift) < 0.02
    gap = (ml.cycle - ols.cycle).abs()
    assert 1e-6 < gap.max() < 0.1
    # the ML path has a cycle at the two dates the OLS path cannot reach
    assert ols.cycle.isna().sum() == 2 and ml.cycle.isna().sum() == 1
    # sigma2: RSS / (N - p - 1) against the ML estimate
    assert ml.sigma2 < ols.sigma2 < ml.sigma2 * 1.03


def test_arma_order_search_and_its_table():
    y = _arima11(0.6, -0.3, n=300, seed=6)
    bn = beveridge_nelson(y, arma=True, max_order=(2, 1))
    table = bn.selection
    assert list(table.index.names) == ["p", "q"] and len(table) == 6
    assert (bn.order, bn.ma_order) == table["bic"].idxmin() and bn.ic == "bic"
    fixed = beveridge_nelson(y, order=(bn.order, bn.ma_order))
    np.testing.assert_array_equal(bn.cycle.to_numpy(), fixed.cycle.to_numpy())
    assert table.loc[(bn.order, bn.ma_order), "loglik"] == bn.loglik
    by_aic = beveridge_nelson(y, arma=True, max_order=1, ic="aic")
    assert len(by_aic.selection) == 4 and by_aic.ic == "aic"
    assert "by exact ML, chosen by BIC" in bn.summary()
    assert "theta[1]" in beveridge_nelson(y, order=(1, 1)).summary()
    # without arma=True nothing changes: least-squares autoregression
    default = beveridge_nelson(y)
    assert default.method == "ols" and default.selection.index.name == "order"


def test_arma_bad_arguments():
    y = _arima11(0.5, 0.4, n=120)
    with pytest.raises(MethodIncompatibility, match=r"\(p, q\)"):
        beveridge_nelson(y, order=(1, 1, 1))
    with pytest.raises(MethodIncompatibility, match="negative"):
        beveridge_nelson(y, order=(1, -1))
    with pytest.raises(MethodIncompatibility, match="arma=True"):
        beveridge_nelson(y, max_order=(2, 2))
    with pytest.raises(DataInsufficient):
        beveridge_nelson(y[:14], order=(3, 3))
    with pytest.raises(DataInsufficient):
        beveridge_nelson(y[:14], arma=True, max_order=4)


def test_a_fit_below_a_nested_model_is_flagged(monkeypatch):
    import importlib

    from statspai.exceptions import ConvergenceWarning

    module = importlib.import_module("statspai.timeseries.beveridge_nelson")

    real = module._fit_arma

    def stuck(dy, p, q):
        est = real(dy, p, q)
        if (p, q) == (1, 1):  # as if the optimiser had stopped early
            est["loglik"] -= 5.0
        return est

    monkeypatch.setattr(module, "_fit_arma", stuck)
    y = _arima11(0.6, -0.3, n=200, seed=6)
    with pytest.warns(ConvergenceWarning, match="local maximum"):
        bn = beveridge_nelson(y, arma=True, max_order=1)
    assert "local maximum" in bn.selection.loc[(1, 1), "note"]
    assert bn.selection.loc[(0, 1), "note"] == ""
