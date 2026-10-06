"""Forecasting functions against the R packages of the methods' authors.

Reference values come from ``forecast`` 9.0.2 (Hyndman et al.), ``stats``
(R 4.5.2) and ``hts`` 6.0.3 on simulated series; see
``_fixtures/_generate_forecasting_data.py`` and
``_fixtures/_generate_forecasting_R.R``. No R is needed to run the tests.

Tolerances, and why each is what it is:

* ``1e-9`` relative -- closed-form quantities and recursions run on the
  same numbers: the ETS filter, likelihood and forecast variances at
  given parameters, the benchmark methods, accuracy measures,
  cross-validation errors, portmanteau statistics, Fourier terms,
  classical decomposition, reconciliation.
* ``1e-7`` -- STL: the same loess recursion, in Fortran on the R side
  and a Cython port on ours. ``1e-4`` after the default fifteen
  robustness iterations, where rounding sends the two along different
  paths to the same fixed point (see ``test_stl_matches_r``).
* ``5e-4`` -- Box-Cox lambda: R's ``optimize`` stops at a tolerance of
  about ``1.2e-4``; ours minimises the same criterion to ``1e-10``.
* ARIMA and ETS *estimates* are optima of a likelihood. They are compared
  through the likelihood (for ETS ours must not be lower) and, where the
  optimum is well determined, through coefficients at ``2e-3``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.timeseries import _ets_core as core

FIX = Path(__file__).resolve().parent / "_fixtures"
R = json.loads((FIX / "forecasting_R.json").read_text(encoding="utf-8"))
_SER = pd.read_csv(FIX / "forecasting_series.csv")
SP_ETS = json.loads((FIX / "forecasting_sp_ets.json").read_text(encoding="utf-8"))

EXACT = 1e-9
_CODE = {"N": 0, "A": 1, "M": 2}


def series(name: str) -> np.ndarray:
    return _SER.loc[_SER["series"] == name, "y"].to_numpy(dtype=float)


def period(name: str) -> int:
    return int(_SER.loc[_SER["series"] == name, "period"].iloc[0])


def arr(x) -> np.ndarray:
    return np.array(
        (
            [[np.nan if v is None else v for v in row] for row in x]
            if x and isinstance(x[0], list)
            else [np.nan if v is None else v for v in x]
        ),
        dtype=float,
    )


def close(a, b, rtol: float = EXACT) -> None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    k = ~np.isnan(a)
    np.testing.assert_allclose(a[k], b[k], rtol=rtol, atol=rtol)


# ----------------------------------------------------------------------
# ETS
# ----------------------------------------------------------------------
def _run_at_r_parameters(tag: str):
    """Our filter and forecast formulas at the parameters R estimated."""
    r = R["ets"][tag]
    y = series(r["series"])
    err = int(r["model"][0] == "M")
    tr, se = _CODE[r["model"][1]], _CODE[r["model"][2]]
    m = r["period"] if se else 1
    par = r["par"]
    alpha = par["alpha"]
    beta = par.get("beta", 0.0)
    gamma = par.get("gamma", 0.0)
    phi = par.get("phi", 1.0)
    init = np.atleast_1d(np.asarray(r["init_state"], dtype=float))
    k = 1
    b0 = 0.0
    if tr:
        b0 = init[1]
        k = 2
    s0 = init[k:] if se else np.zeros(1)
    n = y.shape[0]
    states = np.zeros((n + 1, 2 + (m if se else 0)))
    fitted, resid = np.zeros(n), np.zeros(n)
    sse, sumlog, bad = core.ets_filter(
        y,
        m,
        err,
        tr,
        se,
        alpha,
        beta,
        gamma,
        phi,
        init[0],
        b0,
        s0,
        states,
        fitted,
        resid,
    )
    assert bad == 0
    return (
        r,
        (err, tr, se, m),
        (alpha, beta, gamma, phi),
        sse,
        sumlog,
        states,
        fitted,
        resid,
    )


@pytest.mark.parametrize("tag", sorted(R["ets"]))
def test_ets_filter_and_likelihood_at_r_parameters(tag):
    r, (err, *_), _, sse, sumlog, _, fitted, resid = _run_at_r_parameters(tag)
    n = len(fitted)
    close(fitted, r["fitted"])
    close(resid, r["residuals"])
    assert -0.5 * core.neg2loglik(sse, sumlog, n, err) == pytest.approx(
        r["loglik"], rel=EXACT
    )
    n_par = len(r["par"]) + 1
    assert sse / (n - n_par + 1) == pytest.approx(r["sigma2"], rel=EXACT)


#: R's forecast.ets puts gamma one lag late in the forecast variance of
#: the models with an additive season and no trend (ANA, MNA); see
#: test_ets_seasonal_no_trend_variance_is_the_textbook_formula.
_R_SHIFTED = {"N_A"}


@pytest.mark.parametrize("tag", sorted(R["ets"]))
def test_ets_forecast_mean_and_intervals_at_r_parameters(tag):
    r, (err, tr, se, m), (alpha, beta, gamma, phi), _, _, states, _, _ = (
        _run_at_r_parameters(tag)
    )
    h = len(r["mean"])
    last = states[-1]
    s2 = r["sigma2"]
    mu = core.point_forecast(h, m, tr, se, phi, last)
    if err == 0 and tr < 2 and se < 2:
        var = core.class1_variance(h, m, tr, se, alpha, beta, gamma, phi, s2)
    elif err == 1 and tr < 2 and se < 2:
        var = core.class2_variance(mu, m, tr, se, alpha, beta, gamma, phi, s2)
    elif err == 1 and tr < 2 and se == 2:
        mu, var = core.class3_moments(h, m, tr, alpha, beta, gamma, phi, s2, last)
    else:
        close(mu, r["mean"])  # simulated intervals are not comparable
        return
    close(mu, r["mean"])
    if f"{r['model'][1]}_{r['model'][2]}" in _R_SHIFTED:
        return
    for j, pct in enumerate((80, 95)):
        z = stats.norm.ppf(0.5 + pct / 200)
        close(mu - z * np.sqrt(var), arr(r["lower"])[:, j])
        close(mu + z * np.sqrt(var), arr(r["upper"])[:, j])


@pytest.mark.parametrize("tag", ["quarterly_ANA", "quarterly_MNA", "monthly_ANA"])
def test_r_seasonal_no_trend_intervals_have_gamma_one_lag_late(tag):
    """Documents the divergence: R's intervals for these models are
    reproduced exactly by entering gamma at lags m - 1, 2m - 1, ...
    instead of m, 2m, ... (Hyndman et al. 2008, Table 6.2)."""
    r, (err, tr, se, m), (alpha, beta, gamma, phi), _, _, states, _, _ = (
        _run_at_r_parameters(tag)
    )
    h = len(r["mean"])
    s2 = r["sigma2"]
    mu = np.asarray(r["mean"])
    j = np.arange(1, h + 1)
    c = alpha + gamma * (((j + 1) % m) == 0)
    if err == 0:
        var = s2 * (1 + np.concatenate([[0.0], np.cumsum(c[: h - 1] ** 2)]))
    else:
        theta = np.empty(h)
        theta[0] = mu[0] ** 2
        for k in range(1, h):
            theta[k] = mu[k] ** 2 + s2 * np.sum(c[:k] ** 2 * theta[k - 1 :: -1])
        var = (1 + s2) * theta - mu**2
    z = stats.norm.ppf(0.975)
    close(mu - z * np.sqrt(var), arr(r["lower"])[:, 1])


def test_ets_seasonal_no_trend_variance_is_the_textbook_formula():
    """Independent evidence for our side of that divergence: simulate
    ETS(A,N,A) with a large gamma and compare the variance of the
    simulated paths with the two candidate formulas."""
    m, alpha, gamma, sigma = 4, 0.3, 0.5, 1.0
    h = 2 * m + 1
    last = np.array([10.0, 0.0, 1.0, -2.0, 0.5, 0.5])
    rng = np.random.default_rng(0)
    innov = rng.normal(0, sigma, size=(400_000, h))
    paths = core.simulate_paths(h, m, 0, 0, 1, alpha, 0.0, gamma, 1.0, last, innov)
    mc = paths.var(axis=0)
    ours = core.class1_variance(h, m, 0, 1, alpha, 0.0, gamma, 1.0, sigma**2)
    j = np.arange(1, h + 1)
    c = alpha + gamma * (((j + 1) % m) == 0)
    shifted = sigma**2 * (1 + np.concatenate([[0.0], np.cumsum(c[: h - 1] ** 2)]))
    # Monte Carlo standard error of a variance of 400k normal draws is
    # about var * sqrt(2 / 400000) = 0.22%; allow five of them
    np.testing.assert_allclose(mc, ours, rtol=0.012)
    # at h = m the two formulas differ by (alpha + gamma)^2 - alpha^2 = 0.55
    assert abs(shifted[m - 1] - mc[m - 1]) > 0.3
    assert abs(ours[m - 1] - mc[m - 1]) < 0.03


@pytest.mark.parametrize("tag", sorted(R["ets"]))
def test_ets_estimates_reach_at_least_r_likelihood(tag):
    r = R["ets"][tag]
    mine = SP_ETS[tag]
    # (i) R's own likelihood code, at the parameters sp.ets estimated,
    #     returns the likelihood sp.ets reports: same objective function.
    assert r["loglik_at_sp_params"] == pytest.approx(mine["log_likelihood"], abs=1e-7)
    # (ii) and that value is not below what R's optimiser reached.
    assert mine["log_likelihood"] >= r["loglik"] - 1e-6


@pytest.mark.parametrize(
    "tag", ["level_ANN", "level_MNN", "trend_AAN", "trend_MAN", "quarterly_MNM"]
)
def test_ets_refit_matches_fixture_and_r_where_well_determined(tag):
    f = SP_ETS[tag]
    y = series(f["series"])
    fit = sp.ets(y, f["model"], period=f["period"], damped=f["damped"] or None)
    assert fit.log_likelihood == pytest.approx(f["log_likelihood"], abs=1e-5)
    r = R["ets"][tag]
    assert fit.log_likelihood >= r["loglik"] - 1e-6
    if fit.log_likelihood - r["loglik"] < 1e-4:
        for k, v in fit.params.items():
            assert v == pytest.approx(r["par"][k], abs=5e-3)


def test_ets_automatic_choice_scores_no_worse_than_r():
    for name, r in R["ets_auto"].items():
        fit = sp.ets(series(name), period=period(name))
        assert fit.aicc <= r["aicc"] + 1e-6, name
        cand = fit.candidates.set_index("model")
        # R's pick is among our candidates, scored at least as well
        assert r["method"] in cand.index
        assert cand.loc[r["method"], "aicc"] <= r["aicc"] + 1e-6


# ----------------------------------------------------------------------
# benchmark methods, accuracy, cross-validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "method, name", [("naive", "walk"), ("drift", "walk"), ("snaive", "quarterly")]
)
def test_simple_forecast_matches_r(method, name):
    r = R["simple"][method]
    fit = sp.simple_forecast(series(name), method, period=period(name))
    fc = fit.forecast(10, level=(80, 95))
    close(fc["forecast"], r["mean"])
    close(fc["lower_80"], arr(r["lower"])[:, 0])
    close(fc["upper_80"], arr(r["upper"])[:, 0])
    close(fc["lower_95"], arr(r["lower"])[:, 1])
    close(fc["upper_95"], arr(r["upper"])[:, 1])
    close(fit.residuals, arr(r["residuals"]))


def test_mean_method_matches_r_up_to_the_quantile():
    """forecast::meanf uses a Student-t quantile; the book's formula and
    fable::MEAN, like us, the normal one. Same standard deviation."""
    r = R["simple"]["mean"]
    y = series("quarterly")
    fc = sp.simple_forecast(y, "mean").forecast(10, level=95)
    close(fc["forecast"], r["mean"])
    sd = (fc["forecast"] - fc["lower_95"]) / stats.norm.ppf(0.975)
    t = stats.t.ppf(0.975, len(y) - 1)
    close(fc["forecast"] - t * sd, arr(r["lower"])[:, 1])


def test_forecast_accuracy_matches_r():
    y = series("quarterly")
    n_train = R["accuracy"]["n_train"]
    train, test = y[:n_train], y[n_train:]
    names = R["accuracy"]["names"]
    for method in ("snaive", "drift"):
        fc = sp.simple_forecast(train, method, period=4).forecast(len(test))
        acc = sp.forecast_accuracy(test, fc, train=train, period=4).iloc[0]
        ref = dict(zip(names, R["accuracy"][method]))
        for k in ("ME", "RMSE", "MAE", "MPE", "MAPE", "MASE", "ACF1"):
            assert acc[k] == pytest.approx(ref[k], rel=EXACT, abs=1e-12), (method, k)


def test_tscv_error_matrix_matches_r():
    walk = series("walk")
    ref = arr(R["tscv"]["drift_h3"])
    cv = sp.tscv(walk, "drift", horizon=3, initial=3)
    close(cv.errors.to_numpy(), ref[2:-1])  # R's row i is the origin after i obs
    q = series("quarterly")
    ref = arr(R["tscv"]["snaive_h4"])
    cv = sp.tscv(q, "snaive", horizon=4, initial=9, period=4)
    close(cv.errors.to_numpy(), ref[8:-1])


# ----------------------------------------------------------------------
# tests and tools
# ----------------------------------------------------------------------
def test_ljungbox_matches_box_test():
    dw = np.diff(series("walk"))
    out = sp.ljungbox(dw, lags=10)
    close(out.iloc[0].to_numpy(), R["box"]["lb10"])
    out = sp.ljungbox(dw, lags=10, method="box-pierce")
    close(out.iloc[0].to_numpy(), R["box"]["bp10"])
    out = sp.ljungbox(series("arma"), lags=12, model_df=3)
    close(out.iloc[0].to_numpy(), R["box"]["lb12_fitdf3"])


def test_ndiffs_and_nsdiffs_match_r():
    mo = series("monthly")
    got = {
        "level": sp.ndiffs(series("level")),
        "trend": sp.ndiffs(series("trend")),
        "arma": sp.ndiffs(series("arma")),
        "walk": sp.ndiffs(series("walk")),
        "monthly_sdiff": sp.ndiffs(mo[12:] - mo[:-12]),
    }
    assert got == {k: int(v) for k, v in R["ndiffs"].items()}
    for name, (y, m) in {
        "quarterly": (series("quarterly"), 4),
        "monthly": (mo, 12),
        "walk4": (series("walk"), 4),
    }.items():
        n_r, strength_r = R["nsdiffs"][name]
        assert sp.nsdiffs(y, m) == int(n_r)
        assert sp.stl(y, m).strength["seasonal"] == pytest.approx(strength_r, abs=1e-9)


def test_boxcox_lambda_matches_r_to_its_optimiser_tolerance():
    for name in ("quarterly", "monthly", "level"):
        lam = sp.boxcox_lambda(series(name), period(name))
        assert lam == pytest.approx(R["lambda"][name], abs=5e-4), name


@pytest.mark.parametrize(
    "key, kwargs, rtol",
    [
        ("default11", {}, 1e-7),
        (
            "robust_outer2",
            {"seasonal": 13, "trend": 21, "robust": True, "outer_iter": 2},
            1e-7,
        ),
        # Fifteen robustness iterations (the default with robust=True): the
        # two implementations part ways at about 1e-3 around the third to
        # fifth iteration and come back together as the weights converge
        # (1e-13 after two iterations, 2e-5 after fifteen on this series).
        # Rounding decides which side of a weight threshold an observation
        # falls on; the fixed point is the same.
        ("robust", {"seasonal": 13, "trend": 21, "robust": True}, 1e-4),
        ("periodic", {"seasonal": "periodic"}, 1e-7),
        (
            "exact",
            {
                "seasonal": 7,
                "seasonal_deg": 1,
                "inner_iter": 5,
                "seasonal_jump": 1,
                "trend_jump": 1,
                "low_pass_jump": 1,
            },
            1e-7,
        ),
    ],
)
def test_stl_matches_r(key, kwargs, rtol):
    ref = arr(R["stl"][key])  # seasonal, trend, remainder
    dec = sp.stl(series("monthly"), 12, **kwargs)
    close(dec.seasonal, ref[:, 0], rtol=rtol)
    close(dec.trend, ref[:, 1], rtol=rtol)
    close(dec.remainder, ref[:, 2], rtol=rtol)


def test_mstl_matches_r():
    names = R["mstl"]["names"]
    ref = pd.DataFrame(arr(R["mstl"]["values"]), columns=names)
    dec = sp.stl(series("multi"), [8, 40])
    close(dec.trend, ref["Trend"], rtol=1e-7)
    close(dec.seasonal_components[8], ref["Seasonal8"], rtol=1e-7)
    close(dec.seasonal_components[40], ref["Seasonal40"], rtol=1e-7)
    close(dec.remainder, ref["Remainder"], rtol=1e-6)


def test_stl_forecast_naive_matches_stlf():
    r = R["stlf_naive"]
    fc = sp.stl(series("monthly"), 12).forecast(24, level=(80, 95), method="naive")
    close(fc["forecast"], r["mean"], rtol=1e-7)
    close(fc["lower_95"], arr(r["lower"])[:, 1], rtol=1e-7)
    close(fc["upper_80"], arr(r["upper"])[:, 0], rtol=1e-7)


def test_classical_decomposition_matches_r():
    c = R["classical"]
    add = sp.classical_decompose(series("monthly"), 12)
    close(add.trend, arr(c["add_trend"]))
    close(add.seasonal, arr(c["add_seasonal"]))
    mult = sp.classical_decompose(series("quarterly"), 4, model="multiplicative")
    close(mult.trend, arr(c["mult_trend"]))
    close(mult.seasonal, arr(c["mult_seasonal"]))


def test_fourier_terms_match_r():
    f = R["fourier"]
    close(sp.fourier_terms(80, 4, 2).to_numpy(), arr(f["q_K2"]))
    close(sp.fourier_terms(144, 12, 3).to_numpy(), arr(f["m_K3"]))
    close(sp.fourier_terms(6, 12, 3, start=145).to_numpy(), arr(f["m_K3_future"]))


# ----------------------------------------------------------------------
# ARIMA
# ----------------------------------------------------------------------
_ARIMA_CASES = {
    "arma_201": ("arma", (2, 0, 1), None, None, False),
    "walk_011_drift": ("walk", (0, 1, 1), None, "c", False),
    "monthly_011_011": ("monthly", (0, 1, 1), (0, 1, 1, 12), None, False),
    "quarterly_log_100_011_drift": ("quarterly", (1, 0, 0), (0, 1, 1, 4), "c", True),
}
_R_NAME = {
    "ar1": "ar.L1",
    "ar2": "ar.L2",
    "ma1": "ma.L1",
    "sma1": None,
    "intercept": "const",
    "drift": "drift",
    "x": "x",
}


@pytest.mark.parametrize("tag", sorted(_ARIMA_CASES))
def test_arima_estimates_and_forecasts_match_r(tag):
    name, order, seasonal, trend, log = _ARIMA_CASES[tag]
    r = R["arima"][tag]
    y = series(name)
    if log:
        y = np.log(y)
    fit = sp.arima(y, order=order, seasonal_order=seasonal, trend=trend)
    # sp.arima maximises the exact likelihood of the differenced series;
    # R's Arima keeps the differences as states with a diffuse prior. The
    # two coincide without differencing and are within about 1e-3 here
    # (1.5e-2 on a series differenced both ways, where R's own arima on
    # the differenced series reproduces ours).
    assert fit.log_likelihood == pytest.approx(r["loglik"], abs=3e-3)
    for rname, val in r["coef"].items():
        mine = _R_NAME[rname]
        if mine is None:
            mine = f"ma.S.L{seasonal[3]}"
        assert fit.params[mine] == pytest.approx(val, abs=2e-3), rname
    h = len(r["mean"])
    fc = fit.forecast(h, level=(80, 95), dof_adjust=True)
    np.testing.assert_allclose(fc["forecast"], r["mean"], rtol=2e-3)
    np.testing.assert_allclose(fc["lower_95"], arr(r["lower"])[:, 1], rtol=5e-3)
    np.testing.assert_allclose(fc["upper_95"], arr(r["upper"])[:, 1], rtol=5e-3)


def test_arima_aicc_uses_observations_left_after_seasonal_differencing():
    r = R["arima"]["monthly_011_011"]
    fit = sp.arima(series("monthly"), order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
    k = 3  # ma1, sma1, sigma2
    n_star = 144 - 1 - 12
    assert fit.aicc == pytest.approx(
        fit.aic + 2 * k * (k + 1) / (n_star - k - 1), rel=1e-12
    )
    assert fit.aicc - fit.aic == pytest.approx(r["aicc"] - r["aic"], rel=1e-9)


def test_dynamic_regression_estimates_and_scenario_forecast_match_r():
    r = R["arima"]["dyn_100_x"]
    df = pd.DataFrame({"y": series("dyn_y"), "x": series("dyn_x")})
    fit = sp.arima("y", order=(1, 0, 0), exog=["x"], data=df)
    assert fit.exog_names == ("x",)
    assert fit.params["x"] == pytest.approx(r["coef"]["x"], abs=1e-4)
    assert fit.params["const"] == pytest.approx(r["coef"]["intercept"], abs=1e-4)
    assert fit.params["ar.L1"] == pytest.approx(r["coef"]["ar1"], abs=1e-4)
    assert fit.log_likelihood == pytest.approx(r["loglik"], abs=1e-5)
    xf = np.asarray(r["x_future"]).reshape(-1, 1)
    fc = fit.forecast(len(xf), level=(80, 95), exog=xf, dof_adjust=True)
    # two optimisers on one likelihood; the bounds pass near zero, hence
    # an absolute tolerance
    np.testing.assert_allclose(fc["forecast"], r["mean"], atol=1e-4)
    np.testing.assert_allclose(fc["lower_80"], arr(r["lower"])[:, 0], atol=1e-4)
    np.testing.assert_allclose(fc["upper_95"], arr(r["upper"])[:, 1], atol=1e-4)
    # a frame with named columns is matched by name
    fc2 = fit.forecast(len(xf), level=95, exog=pd.DataFrame({"x": xf.ravel()}))
    np.testing.assert_allclose(fc2["forecast"], fc["forecast"])


def _label(fit) -> tuple:
    so = fit.seasonal_order or (0, 0, 0, 1)
    const = any(k in fit.params.index for k in ("const", "drift"))
    return (*fit.order, so[0], so[1], so[2], const)


def _r_orders(r) -> tuple:
    """(p, d, q, P, D, Q, m) from R's arimaorder(), which has three
    entries for a non-seasonal model and seven for a seasonal one."""
    o = [int(v) for v in r["order"]]
    if len(o) == 3:
        return (*o, 0, 0, 0, 1)
    return tuple(o)


def _r_label(r) -> tuple:
    p, d, q, P, D, Q, _m = _r_orders(r)
    return (p, d, q, P, D, Q, ("drift" in r["label"]) or ("mean" in r["label"]))


@pytest.mark.parametrize("name", ["level", "trend", "walk"])
def test_auto_arima_stepwise_selects_r_model(name):
    r = R["auto_arima"][name]["stepwise"]
    fit = sp.arima(series(name), auto=True)
    assert _label(fit) == _r_label(r)
    assert fit.aicc == pytest.approx(r["aicc"], abs=2e-2)


@pytest.mark.parametrize("name", ["level", "trend", "walk"])
def test_auto_arima_full_search_selects_r_model(name):
    r = R["auto_arima"][name]["full"]
    fit = sp.arima(series(name), auto=True, stepwise=False)
    assert _label(fit) == _r_label(r)
    assert fit.aicc == pytest.approx(r["aicc"], abs=2e-2)


def test_auto_arima_differencing_orders_match_r_on_every_series():
    for name, both in R["auto_arima"].items():
        r = both["stepwise"]
        _, d, _, _, D, _, m = _r_orders(r)
        y = series(name)
        assert sp.ndiffs(y[m:] - y[:-m] if D else y) == d, name
        if m > 1:
            assert sp.nsdiffs(y, m) == D, name


def test_auto_arima_seasonal_search_scores_no_worse_than_r():
    r = R["auto_arima"]["quarterly"]["stepwise"]
    fit = sp.arima(series("quarterly"), auto=True, period=4)
    assert fit.seasonal_order is not None and fit.seasonal_order[1] == 1
    assert fit.aicc <= r["aicc"] + 2e-2
    assert fit.candidates is not None and len(fit.candidates) >= 5


# ----------------------------------------------------------------------
# reconciliation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "method", ["mint_shrink", "mint_cov", "ols", "wls_var", "wls_struct"]
)
def test_reconcile_matches_hts(method):
    S = np.loadtxt(FIX / "forecasting_hts_S.csv", delimiter=",")
    res = np.loadtxt(FIX / "forecasting_hts_res.csv", delimiter=",")
    base = np.loadtxt(FIX / "forecasting_hts_base.csv", delimiter=",")
    ids = [f"s{i}" for i in range(S.shape[0])]
    Sd = pd.DataFrame(S, index=ids, columns=ids[4:])
    rec = sp.reconcile(base, Sd, method=method, residuals=res)
    close(rec.forecasts.to_numpy(), arr(R["hts"][method]))
    f = rec.forecasts.to_numpy()
    np.testing.assert_allclose(f, f[:, 4:] @ S.T, atol=1e-9)


# ----------------------------------------------------------------------
# Box-Cox inside the forecasters, time series features
# ----------------------------------------------------------------------
BC = json.loads((FIX / "forecasting_boxcox_R.json").read_text(encoding="utf-8"))
TSF = json.loads((FIX / "forecasting_tsfeatures_R.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("biasadj, tag", [(False, "med"), (True, "adj")])
def test_boxcox_benchmark_forecasts_match_r(biasadj, tag):
    r = BC[f"drift_log_{tag}"]
    fc = sp.simple_forecast(
        series("walk"), "drift", boxcox=0, biasadj=biasadj
    ).forecast(10)
    close(fc["forecast"], r["mean"])
    close(fc["lower_80"], arr(r["lower"])[:, 0])
    close(fc["upper_95"], arr(r["upper"])[:, 1])
    r = BC[f"snaive_l3_{tag}"]
    fc = sp.simple_forecast(
        series("quarterly"), "snaive", period=4, boxcox=0.3, biasadj=biasadj
    ).forecast(8)
    close(fc["forecast"], r["mean"])
    close(fc["lower_95"], arr(r["lower"])[:, 1])
    close(fc["upper_80"], arr(r["upper"])[:, 0])


@pytest.mark.parametrize("biasadj, tag", [(False, "med"), (True, "adj")])
def test_boxcox_arima_forecasts_match_r(biasadj, tag):
    r = BC[f"arima_l5_{tag}"]
    fit = sp.arima(
        series("quarterly"),
        order=(1, 0, 0),
        seasonal_order=(0, 1, 1, 4),
        boxcox=0.5,
        biasadj=biasadj,
    )
    assert fit.log_likelihood == pytest.approx(r["loglik"], abs=3e-3)
    fc = fit.forecast(8, level=(80, 95), dof_adjust=True)
    # two optimisers on one likelihood, mapped through the inverse transform
    np.testing.assert_allclose(fc["forecast"], r["mean"], rtol=2e-3)
    np.testing.assert_allclose(fc["lower_80"], arr(r["lower"])[:, 0], rtol=5e-3)
    np.testing.assert_allclose(fc["upper_95"], arr(r["upper"])[:, 1], rtol=5e-3)


@pytest.mark.parametrize("name", sorted(TSF["features"]))
def test_ts_features_match_r_tsfeatures(name):
    ref = TSF["features"][name]
    got = sp.ts_features(series(name), period(name))
    rename = {"arch_lm": "ARCH.LM"}
    assert len(got) >= 18
    for key, val in got.items():
        target = ref[rename.get(key, key)]
        assert val == pytest.approx(target, rel=1e-9, abs=1e-10), key
