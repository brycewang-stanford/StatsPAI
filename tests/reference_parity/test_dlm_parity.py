"""``sp.dlm`` against R ``dlm`` and against an exact posterior.

Filter, smoother and likelihood are deterministic given the variances, so
they are compared digit for digit with the R package ``dlm`` (Petris 2010)
on the committed synthetic file ``_fixtures/dlm.csv``
(``_generate_dlm_data.py``; reference ``dlm_R.json``, R 4.5.2, dlm 1.1-6.1,
``_generate_dlm_R.R``).

Tolerances. ``EXACT`` (1e-9 relative) for the likelihood, the one-step
forecasts and, under a proper prior, every filtered and smoothed moment.
Under the diffuse prior ``C0 = 1e7 I`` the first ``k`` dates are
ill-conditioned by construction (a covariance of order 1e7 collapsing to
order one): the test compares from date ``k + 1`` on, and allows 1e-5 on
smoothed variances, where the recursion subtracts numbers of that size.
R uses a singular-value filter there, ours the Joseph form. ``OPTIM``
(1e-5) for maximum-likelihood variances: both sides stop an optimiser on
a flat surface, and the test also checks that our likelihood is at least
as high as R's.

The Gibbs sampler (forward filtering, backward sampling) has no
cross-language reference. It is compared with the exact posterior of the
two variances, integrated on a grid with a likelihood that does not use
the Kalman filter: the outcome vector is multivariate normal with a
covariance written out directly.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-9
OPTIM = 1e-5


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


@pytest.fixture(scope="module")
def R() -> dict:
    return json.loads((FIX / "dlm_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(FIX / "dlm.csv")


def _check(fit, ref, names, skip, var_tol):
    n, k = fit.n_obs, len(names)
    sd = [f"{c}_sd" for c in names]
    m, s = np.array(ref["m"]).reshape(n, k), np.array(ref["s"]).reshape(n, k)
    C, S = np.array(ref["C"]).reshape(n, k), np.array(ref["S"]).reshape(n, k)
    assert rel(fit.filtered[names].to_numpy()[skip:], m[skip:]) < EXACT
    assert rel(fit.smoothed[names].to_numpy()[skip:], s[skip:]) < EXACT
    assert rel(fit.filtered[sd].to_numpy()[skip:] ** 2, C[skip:]) < EXACT
    assert rel(fit.smoothed[sd].to_numpy()[skip:] ** 2, S[skip:]) < var_tol
    assert rel(fit.fitted.to_numpy()[skip:], np.array(ref["f"])[skip:]) < EXACT
    # dlmLL is the negative log likelihood without the 2 pi constant
    negll = -fit.loglik - 0.5 * n * np.log(2 * np.pi)
    assert negll == pytest.approx(ref["negll"], rel=EXACT)


def test_time_varying_regression_diffuse_prior(R, df):
    fit = sp.dlm("y ~ x", df, obs_var=0.3, state_var=[0.05, 0.02])
    _check(fit, R["tvp"], ["Intercept", "x"], skip=2, var_tol=1e-5)


def test_proper_prior_and_a_constant_coefficient(R, df):
    fit = sp.dlm(
        "y ~ x", df, obs_var=0.2, state_var=0.08, constant=["x"], m0=[1, 0.5], C0=[4, 1]
    )
    assert fit.variances.loc["state:x", "estimate"] == 0.0
    _check(fit, R["tvp_prior"], ["Intercept", "x"], skip=0, var_tol=EXACT)
    # a constant coefficient has one smoothed value
    assert np.ptp(fit.smoothed["x"].to_numpy()) < 1e-10


def test_local_level(R, df):
    fit = sp.dlm("level ~ 1", df, obs_var=1.2, state_var=0.1)
    _check(fit, R["level"], ["Intercept"], skip=1, var_tol=EXACT)


@pytest.mark.parametrize(
    "key, formula", [("mle_tvp", "y ~ x"), ("mle_level", "level ~ 1")]
)
def test_maximum_likelihood(R, df, key, formula):
    ref = R[key]
    assert ref["convergence"] == 0
    fit = sp.dlm(formula, df)
    assert rel(fit.variances["estimate"], ref["par"]) < OPTIM
    negll = -fit.loglik - 0.5 * fit.n_obs * np.log(2 * np.pi)
    assert negll <= ref["negll"] + 1e-9


def test_zero_state_variances_are_recursive_least_squares(df):
    """Known identity: with no state noise and a diffuse prior the last
    filtered (and every smoothed) coefficient is the OLS estimate."""
    fit = sp.dlm("y ~ x", df, state_var=0.0, C0=1e9)
    ols = sp.regress("y ~ x", df)
    for name in ("Intercept", "x"):
        assert fit.params[name] == pytest.approx(float(ols.params[name]), rel=1e-6)
        assert fit.smoothed[name].iloc[0] == pytest.approx(fit.params[name], rel=1e-8)
    # and the observation variance is the ML residual variance, up to the
    # two dates the diffuse prior absorbs
    resid = df["y"] - ols.predict(df)
    assert fit.variances.loc["obs", "estimate"] == pytest.approx(
        float(resid @ resid) / (len(df) - 2), rel=1e-3
    )


def test_gibbs_recovers_the_exact_posterior_of_the_variances():
    rng = np.random.default_rng(3)
    T = 40
    x = rng.normal(1.0, 1.0, size=T)
    slope = 0.5 + np.cumsum(0.3 * rng.normal(size=T))
    data = pd.DataFrame({"x": x, "y": 1.0 + slope * x + 0.5 * rng.normal(size=T)})
    a0, d0 = 3.0, 1.0
    m0, c0 = np.array([0.5, 0.0]), np.array([4.0, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.dlm(
            "y ~ x",
            data,
            method="gibbs",
            constant=["Intercept"],
            m0=m0,
            C0=c0,
            obs_var_prior=(a0, d0),
            state_var_prior=(a0, d0),
            draws=30000,
            burnin=3000,
            seed=2,
        )
    # y = b0 + x_t (b1_0 + w_1 + ... + w_t) + e_t is jointly normal:
    # Cov(y_t, y_s) = C0_0 + (C0_1 + min(t, s) W) x_t x_s + V 1{t = s}
    y = data["y"].to_numpy()
    steps = np.arange(1, T + 1)
    base = c0[0] * np.ones((T, T)) + c0[1] * np.outer(x, x)
    walk = np.outer(x, x) * np.minimum.outer(steps, steps)
    mean = m0[0] + m0[1] * x
    lv, lw = np.log(fit.draws["obs"]), np.log(fit.draws["state:x"])
    grid = 101
    av = np.linspace(lv.mean() - 8 * lv.std(), lv.mean() + 8 * lv.std(), grid)
    aw = np.linspace(lw.mean() - 8 * lw.std(), lw.mean() + 8 * lw.std(), grid)
    log_post = np.empty((grid, grid))
    last_slope = np.empty((grid, grid))
    for i, a in enumerate(av):
        for j, b in enumerate(aw):
            V, W = np.exp(a), np.exp(b)
            cov = V * np.eye(T) + base + W * walk
            log_post[i, j] = (
                stats.multivariate_normal.logpdf(y, mean, cov)
                + stats.invgamma.logpdf(V, a0 / 2, scale=d0 / 2)
                + a
                + stats.invgamma.logpdf(W, a0 / 2, scale=d0 / 2)
                + b
            )
            c_last = (c0[1] + W * steps) * x  # Cov(slope at T, y)
            last_slope[i, j] = m0[1] + c_last @ np.linalg.solve(cov, y - mean)
    w = np.exp(log_post - log_post.max())
    w /= w.sum()
    vg, wg = np.meshgrid(np.exp(av), np.exp(aw), indexing="ij")
    exact = np.array([(w * vg).sum(), (w * wg).sum()])
    exact_sd = np.sqrt(np.array([(w * vg**2).sum(), (w * wg**2).sum()]) - exact**2)
    t = fit.variances.loc[["obs", "state:x"]]
    mcse = t["sd"].to_numpy() / np.sqrt(t["ess"].to_numpy())
    z = (t["estimate"].to_numpy() - exact) / mcse
    assert np.abs(z).max() < 4.0, z
    assert np.abs(t["sd"].to_numpy() / exact_sd - 1).max() < 0.05
    # the smoothed state is the mixture of the conditional means
    sd_last = fit.smoothed["x_sd"].iloc[-1]
    assert fit.smoothed["x"].iloc[-1] == pytest.approx(
        (w * last_slope).sum(), abs=0.03 * sd_last
    )
    assert fit.variances.loc["state:Intercept", "estimate"] == 0.0
