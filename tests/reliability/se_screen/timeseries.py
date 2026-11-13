"""SE calibration screen: time series.

Stationary designs with T = 500 (2,000 for GARCH), persistence 0.5, so
that the normal approximation is not a unit-root question. Targets are an
autoregressive or regression coefficient with true value 0.5, or an impulse
response whose true value follows from the design.

    python tests/reliability/se_screen/timeseries.py [--only name] [--confirm]
"""

# flake8: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import Row, coef, main, tidy_row, wald  # noqa: E402,F401

FAMILY = "timeseries"
T = 500
BURN = 100
PHI = 0.5
H = 2  # horizon of the impulse-response rows
A = np.array([[0.5, 0.2], [0.0, 0.5]])  # VAR(1) coefficient matrix
SIGMA = np.array([[1.0, 0.5], [0.5, 1.0]])  # VAR innovation covariance


def _sp():
    import statspai as sp

    return sp


def _ar(rng, n, phi, e=None):
    e = rng.normal(size=n + BURN) if e is None else e
    y = np.zeros(len(e))
    for t in range(1, len(e)):
        y[t] = phi * y[t - 1] + e[t]
    return y[BURN:]


# --------------------------------------------------------------------- DGPs
def d_ar1(rng):
    return pd.DataFrame({"y": 1.0 + _ar(rng, T, PHI)})


def d_arma11(rng):
    e = rng.normal(size=T + BURN)
    u = e.copy()
    u[1:] += 0.4 * e[:-1]
    return pd.DataFrame({"y": 1.0 + _ar(rng, T, PHI, u)})


def d_regar(rng):
    """Regression with AR(1) errors and an AR(1) regressor."""
    x = _ar(rng, T, 0.5)
    return pd.DataFrame(
        {"t": np.arange(T), "x1": x, "y": 1 + 0.5 * x + _ar(rng, T, 0.5)}
    )


def d_var(rng):
    e = rng.multivariate_normal(np.zeros(2), SIGMA, size=T + BURN)
    y = np.zeros((T + BURN, 2))
    for t in range(1, T + BURN):
        y[t] = A @ y[t - 1] + e[t]
    return pd.DataFrame(y[BURN:], columns=["y1", "y2"])


def d_lp(rng):
    """Observed iid shock s; y_t = 0.5 y_t-1 + s_t + e_t, so the response at h is 0.5^h."""
    s = rng.normal(size=T + BURN)
    y = _ar(rng, T, PHI, s + rng.normal(size=T + BURN))
    return pd.DataFrame({"y": y, "s": s[BURN:]})


def d_garch(rng, n=2000, omega=0.1, a=0.1, b=0.8):
    z = rng.normal(size=n + BURN)
    h = np.full(n + BURN, omega / (1 - a - b))
    e = np.zeros(n + BURN)
    for t in range(1, n + BURN):
        h[t] = omega + a * e[t - 1] ** 2 + b * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
    return pd.DataFrame({"y": 0.2 + e[BURN:]})


def d_ardl(rng):
    x = _ar(rng, T + 1, 0.5)
    y = np.zeros(T + 1)
    e = rng.normal(size=T + 1)
    for t in range(1, T + 1):
        y[t] = 0.3 + PHI * y[t - 1] + 0.8 * x[t - 1] + e[t]
    return pd.DataFrame({"y": y[1:], "x": x[1:]})


def d_its(rng, n=200):
    t = np.arange(n)
    y = 2 + 0.02 * t + 2.0 * (t >= n // 2) + _ar(rng, n, 0.3)
    return pd.DataFrame({"y": y, "month": t})


def d_vec(rng):
    """y1 - y2 is stationary; y2 is a random walk. Cointegrating vector (1, -1)."""
    y2 = rng.normal(size=T).cumsum()
    y1 = y2 + _ar(rng, T, 0.5)
    return pd.DataFrame({"y1": y1, "y2": y2})


def _true_oirf(h):
    P = np.linalg.cholesky(SIGMA)
    return np.linalg.matrix_power(A, h) @ P


# ------------------------------------------------------------------- fits
def _irf_key(d):
    """Key of the response of y1 to a y2 shock, whatever the key convention."""
    for k in d:
        s = str(k)
        if s.replace(" ", "") in ("y2->y1", "('y2','y1')", "y2_y1", "y2→y1"):
            return k
    raise KeyError(str(list(d.keys())))


def _first(index, *needles):
    for t in index:
        if all(n in str(t).lower() for n in needles):
            return t
    raise KeyError(f"no term with {needles} in {list(index)}")


def fit_arima(order, term, **kw):
    def fit(df):
        r = _sp().arima(df.y, order=order, **kw)
        name = _first(r.params.index, term)
        ci = r.conf_int(0.05)
        lo, hi = np.asarray(ci.loc[name])[:2]
        return float(r.params[name]), float(r.std_errors[name]), float(lo), float(hi)

    return fit


def fit_arimax(df):
    r = _sp().arima(df.y, order=(1, 0, 0), exog=df[["x1"]])
    name = _first(r.params.index, "x1")
    lo, hi = np.asarray(r.conf_int(0.05).loc[name])[:2]
    return float(r.params[name]), float(r.std_errors[name]), float(lo), float(hi)


def fit_newey(lags):
    def fit(df):
        return coef(_sp().regress("y ~ x1", data=df, vce="hac", hac_lags=lags), "x1")

    return fit


def fit_var(df):
    r = _sp().var(df, variables=["y1", "y2"], lags=1)
    tab = r.coefs["y1"]
    name = [t for t in tab.index if "y1" in str(t)][0]
    return wald(tab.loc[name, "coef"], tab.loc[name, "se"])  # no interval reported


def fit_irf(**kw):
    def fit(df):
        v = _sp().var(df, variables=["y1", "y2"], lags=1)
        out = _sp().irf(v, periods=6, **kw)
        key = _irf_key(out["irf"])
        return (
            out["irf"][key][H],
            out["se"][key][H],
            out["lower"][key][H],
            out["upper"][key][H],
        )

    return fit


def fit_lp(**kw):
    def fit(df):
        r = _sp().local_projections(df, outcome="y", shock="s", horizons=4, **kw)
        return r.irf[H], r.se[H], r.ci_lower[H], r.ci_upper[H]

    return fit


def fit_garch(term, **kw):
    def fit(df):
        r = _sp().garch(df.y, **kw)
        i = [k for k, nm in enumerate(r.param_names) if term in str(nm).lower()][0]
        return wald(r.coef[i], r.se_vec[i])  # no interval reported

    return fit


def fit_ardl(**kw):
    def fit(df):
        r = _sp().ardl(df, y="y", x="x", lags=1, x_lags=1, **kw)
        return coef(r.model, "y_L1")

    return fit


def fit_its(**kw):
    def fit(df):
        r = _sp().its(df, y="y", time="month", intervention=100, **kw)
        return r.level_change, r.se_level, r.ci_level[0], r.ci_level[1]

    return fit


def fit_vec(df):
    r = _sp().vec(df, variables=["y1", "y2"], lags=2, rank=1)
    return wald(
        r.beta.loc["y2", "_ce1"], r.beta_se.loc["y2", "_ce1"]
    )  # no interval reported


def R(name, estimator, call, dgp, fit, truth=PHI, n=T, B=None, design=""):
    return Row(name, estimator, call, n, truth, dgp, fit, B, design)


ROWS = [
    R(
        "arima/ar1",
        "arima",
        "arima(y, order=(1, 0, 0))",
        d_ar1,
        fit_arima((1, 0, 0), "ar"),
    ),
    R(
        "arima/ar1-innovations",
        "arima",
        'arima(y, order=(1, 0, 0), method="innovations_mle")',
        d_ar1,
        fit_arima((1, 0, 0), "ar", method="innovations_mle"),
    ),
    R(
        "arima/arma11-ar",
        "arima",
        "arima(y, order=(1, 0, 1))",
        d_arma11,
        fit_arima((1, 0, 1), "ar"),
        design="ARMA(1,1), MA coefficient 0.4",
    ),
    R(
        "arima/arma11-ma",
        "arima",
        "arima(y, order=(1, 0, 1))",
        d_arma11,
        fit_arima((1, 0, 1), "ma"),
        truth=0.4,
        design="ARMA(1,1); the MA coefficient",
    ),
    R(
        "arima/regression-ar1-errors",
        "arima",
        "arima(y, order=(1, 0, 0), exog=x1)",
        d_regar,
        fit_arimax,
        design="AR(1) error and regressor, rho 0.5",
    ),
    R(
        "regress/hac-4",
        "regress",
        'regress(..., vce="hac", hac_lags=4)',
        d_regar,
        fit_newey(4),
        design="AR(1) error and regressor, rho 0.5",
    ),
    R(
        "regress/hac-12",
        "regress",
        'regress(..., vce="hac", hac_lags=12)',
        d_regar,
        fit_newey(12),
        design="AR(1) error and regressor, rho 0.5",
    ),
    R(
        "var/coefficient",
        "var",
        "var(lags=1)",
        d_var,
        fit_var,
        design="bivariate VAR(1); own lag of y1",
    ),
    R(
        "irf/asymptotic",
        "irf",
        'irf(var, ci="asymptotic")',
        d_var,
        fit_irf(ci="asymptotic"),
        truth=float(_true_oirf(H)[0, 1]),
        design="orthogonalised response of y1 to y2 at horizon 2",
    ),
    R(
        "irf/bootstrap",
        "irf",
        'irf(var, ci="bootstrap", reps=200)',
        d_var,
        fit_irf(ci="bootstrap", reps=200, seed=1),
        truth=float(_true_oirf(H)[0, 1]),
        B=200,
        design="orthogonalised response of y1 to y2 at horizon 2; B = 200",
    ),
    R(
        "local_projections/h2",
        "local_projections",
        "local_projections(outcome, shock)",
        d_lp,
        fit_lp(),
        truth=PHI**H,
        design="response at horizon 2 to an observed iid shock",
    ),
    R(
        "garch/alpha-oim",
        "garch",
        "garch(y)",
        d_garch,
        fit_garch("alpha"),
        truth=0.1,
        n=2000,
        design="GARCH(1,1) with alpha 0.1, beta 0.8, T = 2000; the ARCH coefficient",
    ),
    R(
        "garch/alpha-robust",
        "garch",
        'garch(y, vce="robust")',
        d_garch,
        fit_garch("alpha", vce="robust"),
        truth=0.1,
        n=2000,
        design="GARCH(1,1) with alpha 0.1, beta 0.8, T = 2000; the ARCH coefficient",
    ),
    R(
        "ardl/lag-coefficient",
        "ardl",
        "ardl(y, x, lags=1, x_lags=1)",
        d_ardl,
        fit_ardl(),
        design="coefficient on the lagged outcome",
    ),
    R(
        "its/level-change",
        "its",
        "its(y, time, intervention)",
        d_its,
        fit_its(),
        truth=2.0,
        n=200,
        design="T = 200, AR(1) error rho 0.3, default HAC lag 4",
    ),
    R(
        "vec/beta",
        "vec",
        "vec(lags=2, rank=1)",
        d_vec,
        fit_vec,
        truth=-1.0,
        design="cointegrating coefficient on y2",
    ),
]


# ---------------------------------------------------------------- options
def d_arima110(rng):
    """Integrated AR(1): the first difference has AR coefficient 0.5."""
    return pd.DataFrame({"y": _ar(rng, T, PHI).cumsum()})


def d_ar1_zero_mean(rng):
    return pd.DataFrame({"y": _ar(rng, T, PHI)})


def d_garch_t(rng, n=2000, omega=0.1, a=0.1, b=0.8, nu=8.0):
    z = rng.standard_t(nu, size=n + BURN) * np.sqrt((nu - 2) / nu)
    h = np.full(n + BURN, omega / (1 - a - b))
    e = np.zeros(n + BURN)
    for t in range(1, n + BURN):
        h[t] = omega + a * e[t - 1] ** 2 + b * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
    return pd.DataFrame({"y": 0.2 + e[BURN:]})


def fit_var_kw(lags=1, **kw):
    def fit(df):
        r = _sp().var(df, variables=["y1", "y2"], lags=lags, **kw)
        tab = r.coefs["y1"]
        name = [t for t in tab.index if "y1" in str(t)][0]
        return wald(tab.loc[name, "coef"], tab.loc[name, "se"])

    return fit


def fit_lp_h(h, **kw):
    def fit(df):
        r = _sp().local_projections(df, outcome="y", shock="s", horizons=4, **kw)
        return r.irf[h], r.se[h], r.ci_lower[h], r.ci_upper[h]

    return fit


GARCH = "GARCH(1,1) with alpha 0.1, beta 0.8, T = 2000; the ARCH coefficient"

OPTION_ROWS = [
    R(
        "arima/ar1-no-constant",
        "arima",
        'arima(y, order=(1, 0, 0), trend="n")',
        d_ar1_zero_mean,
        fit_arima((1, 0, 0), "ar", trend="n"),
        design="zero-mean AR(1)",
    ),
    R(
        "arima/arima110",
        "arima",
        "arima(y, order=(1, 1, 0))",
        d_arima110,
        fit_arima((1, 1, 0), "ar"),
        design="integrated AR(1)",
    ),
    R(
        "var/se-df-r",
        "var",
        'var(lags=1, se_df="r")',
        d_var,
        fit_var_kw(se_df="r"),
        design="bivariate VAR(1); own lag of y1",
    ),
    R(
        "var/overfitted",
        "var",
        "var(lags=2)",
        d_var,
        fit_var_kw(lags=2),
        design="bivariate VAR(1) fitted with two lags; first own lag of y1",
    ),
    R(
        "irf/non-orthogonal",
        "irf",
        'irf(var, orthogonal=False, ci="asymptotic")',
        d_var,
        fit_irf(ci="asymptotic", orthogonal=False),
        truth=float(np.linalg.matrix_power(A, H)[0, 1]),
        design="response of y1 to a unit y2 innovation at horizon 2",
    ),
    R(
        "irf/bootstrap-efron",
        "irf",
        'irf(var, ci="bootstrap", boot="efron", reps=200)',
        d_var,
        fit_irf(ci="bootstrap", boot="efron", reps=200, seed=1),
        truth=float(_true_oirf(H)[0, 1]),
        B=200,
        design="orthogonalised response of y1 to y2 at horizon 2; B = 200",
    ),
    R(
        "irf/bootstrap-hall",
        "irf",
        'irf(var, ci="bootstrap", boot="hall", reps=200)',
        d_var,
        fit_irf(ci="bootstrap", boot="hall", reps=200, seed=1),
        truth=float(_true_oirf(H)[0, 1]),
        B=200,
        design="orthogonalised response of y1 to y2 at horizon 2; B = 200",
    ),
    R(
        "local_projections/h0",
        "local_projections",
        "local_projections(outcome, shock), h = 0",
        d_lp,
        fit_lp_h(0),
        truth=1.0,
        design="impact response to an observed iid shock",
    ),
    R(
        "local_projections/h4",
        "local_projections",
        "local_projections(outcome, shock), h = 4",
        d_lp,
        fit_lp_h(4),
        truth=PHI**4,
        design="response at horizon 4",
    ),
    R(
        "local_projections/robust",
        "local_projections",
        'local_projections(vce="robust")',
        d_lp,
        fit_lp_h(H, vce="robust"),
        truth=PHI**H,
        design="response at horizon 2; the shock is iid, so its score is uncorrelated",
    ),
    R(
        "garch/alpha-opg",
        "garch",
        'garch(y, vce="opg")',
        d_garch,
        fit_garch("alpha", vce="opg"),
        truth=0.1,
        n=2000,
        design=GARCH,
    ),
    R(
        "garch/t-innovations",
        "garch",
        'garch(y, dist="t")',
        d_garch_t,
        fit_garch("alpha", dist="t"),
        truth=0.1,
        n=2000,
        design=GARCH + "; t(8) innovations",
    ),
    R(
        "garch/normal-qmle-robust",
        "garch",
        'garch(y, vce="robust")',
        d_garch_t,
        fit_garch("alpha", vce="robust"),
        truth=0.1,
        n=2000,
        design=GARCH + "; t(8) innovations fitted by normal quasi-likelihood",
    ),
    R(
        "its/level-change-lag8",
        "its",
        "its(..., hac_lag=8)",
        d_its,
        fit_its(hac_lag=8),
        truth=2.0,
        n=200,
        design="T = 200, AR(1) error rho 0.3",
    ),
    R(
        "its/level-change-small-sample",
        "its",
        "its(..., hac_small_sample=True)",
        d_its,
        fit_its(hac_small_sample=True),
        truth=2.0,
        n=200,
        design="T = 200, AR(1) error rho 0.3, HAC lag 4",
    ),
]

ROWS = ROWS + OPTION_ROWS

MECHANISMS: dict = {
    "its/level-change": (
        "Known property of Newey-West at a fixed lag. The SE equals statsmodels HAC "
        "(maxlags 4) to 1e-12 on one draw. With an AR(1) error (rho 0.3) the Bartlett "
        "kernel at lag 4 recovers 0.936 of the true SE in population; the rest of the "
        "gap is the small-sample bias of HAC with estimated residuals at T = 200."
    ),
    "vec/beta": (
        "Heavy tails, not the SE: the median SE over the MAD-based spread is 1.01 and "
        "coverage is inside the band. The reduced-rank estimate of the cointegrating "
        "vector has occasional large errors that inflate the Monte Carlo SD."
    ),
    "its/level-change-lag8": (
        "Same mechanism as its/level-change, and more lags make it worse, not better: the "
        "kernel truncation matters less at lag 8 (0.96 of the true SE in population) but "
        "the small-sample downward bias of a HAC built from residuals grows with the lag "
        "when the regressors are a step and a trend."
    ),
    "its/level-change-small-sample": (
        "Same mechanism as its/level-change. The n / (n - k) factor of "
        "hac_small_sample=True is 1.02 at T = 200 and does not close a gap of 14%."
    ),
}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
