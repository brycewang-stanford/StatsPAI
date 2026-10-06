"""``sp.bayes_arima`` and ``sp.stochvol`` against quantities known exactly.

* ARMA likelihood: the Kalman filter against the dense multivariate
  normal density built from the autocovariances.
* ARMA prior: the density placed on the partial autocorrelations must
  transform into a constant over the admissible region of the
  coefficients, equal to one over the region's volume.
* AR(1) and MA(1) posteriors: integrated on a grid from an integrand
  written here (means within four Monte Carlo standard errors, standard
  deviations within 5 percent, the marginal likelihood against the
  normalising constant).
* Stochastic volatility has no posterior that can be integrated on a
  grid. Its sampler is checked by the joint-distribution test instead:
  alternate the sampler's sweep with a fresh draw of the data given the
  current states, and the parameters must be distributed as the prior.
  Any error in a full conditional shows up as a wrong prior moment.
* The seven-normal mixture against the exact log chi-square density.
* A screen (not parity: both sides are simulation output) against long
  runs of the R package ``stochvol`` on a committed series.
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
from statspai.mcmc import arima as A
from statspai.mcmc import sv as SV

FIX = Path(__file__).parent / "_fixtures"


def _acov_matrix(phi, theta, sigma2, n):
    from statsmodels.tsa.arima_process import arma_acovf

    g = arma_acovf(
        np.r_[1.0, -np.asarray(phi)], np.r_[1.0, theta], nobs=n, sigma2=sigma2
    )
    return g[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]


@pytest.mark.parametrize(
    "phi, theta",
    [
        ([0.5, -0.3], [0.4]),
        ([], [0.6, 0.2]),
        ([0.2, 0.1, -0.4], []),
        ([0.9], [-0.5, 0.3]),
    ],
)
def test_arma_likelihood_is_the_dense_normal_density(phi, theta):
    rng = np.random.default_rng(1)
    n, mu, s2 = 45, 0.3, 1.7
    w = rng.normal(size=n)
    exact = stats.multivariate_normal(
        np.full(n, mu), _acov_matrix(phi, theta, s2, n)
    ).logpdf(w)
    mdl = A._ArmaModel(w, len(phi), len(theta), True, (0.0, 1e6), (0.001, 0.001))
    got = mdl.log_likelihood(mu, np.array(phi, float), np.array(theta, float), s2)
    # atol: two O(n) factorisations of the same quadratic form
    assert got == pytest.approx(exact, abs=1e-10)


def test_prior_is_uniform_over_the_stationary_region():
    rng = np.random.default_rng(2)
    # p = 2: the region is a triangle of area 4
    for _ in range(20):
        r = rng.uniform(-0.99, 0.99, 2)
        log_dens = A._log_pacf_prior(r) - np.log(1.0 - r[1])
        assert log_dens == pytest.approx(-np.log(4.0), abs=1e-12)
    # p = 3 and 4: the density of the coefficients is the same everywhere
    for p in (3, 4):
        vals = []
        for _ in range(25):
            r = rng.uniform(-0.95, 0.95, p)
            jac = np.empty((p, p))
            for j in range(p):
                e = np.zeros(p)
                e[j] = 1e-6
                jac[:, j] = (A.pacf_to_coefs(r + e) - A.pacf_to_coefs(r - e)) / 2e-6
            vals.append(A._log_pacf_prior(r) - np.linalg.slogdet(jac)[1])
            assert np.allclose(A.coefs_to_pacf(A.pacf_to_coefs(r)), r, atol=1e-12)
        # atol: central differences of a polynomial map
        assert np.ptp(vals) < 1e-6
    # and admissibility is exactly |pacf| < 1
    for _ in range(200):
        c = rng.uniform(-2.5, 2.5, 3)
        stationary = np.all(np.abs(np.roots(np.r_[1.0, -c])) < 1)
        pac = A.coefs_to_pacf(c)
        assert stationary == bool(np.all(np.abs(pac) < 1))


def _summaries(log_post, axes):
    w = np.exp(log_post - log_post.max())
    cell = np.prod([a[1] - a[0] for a in axes])
    log_norm = np.log(w.sum() * cell) + log_post.max()
    w = w / w.sum()
    out = []
    for i, a in enumerate(axes):
        marg = w.sum(axis=tuple(j for j in range(len(axes)) if j != i))
        out.append((a, marg))
    return log_norm, out


def _mean_sd(values, marg):
    m = marg @ values
    return m, np.sqrt(marg @ (values - m) ** 2)


def _check(fit, names, exact, sd_tol=0.05):
    t = fit.table.loc[names]
    mean = np.array([e[0] for e in exact])
    sd = np.array([e[1] for e in exact])
    z = (t["mean"].to_numpy() - mean) / t["mcse"].to_numpy()
    assert np.abs(z).max() < 4.0, dict(zip(names, z.round(2)))
    ratio = t["sd"].to_numpy() / sd
    assert np.abs(ratio - 1).max() < sd_tol, dict(zip(names, ratio.round(3)))


def test_ar1_exact_posterior():
    rng = np.random.default_rng(3)
    n = 40
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 + 0.5 * (y[t - 1] - 0.5) + rng.normal()
    v0, a0, d0 = 10.0, 4.0, 4.0
    mu = np.linspace(-4.0, 5.0, 181)
    phi = np.linspace(-0.9995, 0.9995, 400)
    ls = np.linspace(-2.0, 2.5, 181)
    dev = y[None, :] - mu[:, None]  # (M, n)
    e = dev[:, None, 1:] - phi[None, :, None] * dev[:, None, :-1]  # (M, P, n - 1)
    ss = (e**2).sum(axis=2) + (1 - phi[None, :] ** 2) * dev[:, 0][:, None] ** 2
    s2 = np.exp(ls)
    log_post = (
        -0.5 * n * np.log(2 * np.pi * s2)[None, None, :]
        + 0.5 * np.log(1 - phi**2)[None, :, None]
        - 0.5 * ss[:, :, None] / s2[None, None, :]
        + stats.norm(0.0, np.sqrt(v0)).logpdf(mu)[:, None, None]
        + np.log(0.5)
        + (stats.invgamma(a0 / 2, scale=d0 / 2).logpdf(s2) + ls)[None, None, :]
    )
    log_norm, margs = _summaries(log_post, [mu, phi, ls])
    exact = [
        _mean_sd(mu, margs[0][1]),
        _mean_sd(phi, margs[1][1]),
        _mean_sd(s2, margs[2][1]),
    ]
    fit = sp.bayes_arima(
        y,
        order=(1, 0, 0),
        mean_prior=(0.0, v0),
        sigma2_prior=(a0, d0),
        draws=60000,
        burnin=2000,
        seed=11,
    )
    _check(fit, ["const", "ar.L1", "sigma2"], exact)
    det = fit.marginal_likelihood_details()
    assert abs(det["log_marginal_likelihood"] - log_norm) < max(4 * det["mcse"], 0.03)


def test_ma1_exact_posterior():
    rng = np.random.default_rng(4)
    n = 35
    eps = rng.normal(size=n + 1)
    y = 1.0 + eps[1:] + 0.5 * eps[:-1]
    v0, a0, d0 = 10.0, 4.0, 4.0
    mu = np.linspace(-2.0, 4.0, 161)
    theta = np.linspace(-0.9995, 0.9995, 400)
    ls = np.linspace(-2.2, 2.2, 161)
    s2 = np.exp(ls)
    log_post = np.empty((mu.size, theta.size, ls.size))
    one = np.ones(n)
    for j, th in enumerate(theta):
        c = (1 + th * th) * np.eye(n) + th * (np.eye(n, k=1) + np.eye(n, k=-1))
        ci = np.linalg.inv(c)
        quad = y @ ci @ y - 2 * mu * (one @ ci @ y) + mu**2 * (one @ ci @ one)
        log_post[:, j, :] = (
            -0.5 * n * np.log(2 * np.pi * s2)[None, :]
            - 0.5 * np.linalg.slogdet(c)[1]
            - 0.5 * quad[:, None] / s2[None, :]
        )
    log_post += (
        stats.norm(0.0, np.sqrt(v0)).logpdf(mu)[:, None, None]
        + np.log(0.5)
        + (stats.invgamma(a0 / 2, scale=d0 / 2).logpdf(s2) + ls)[None, None, :]
    )
    log_norm, margs = _summaries(log_post, [mu, theta, ls])
    exact = [
        _mean_sd(mu, margs[0][1]),
        _mean_sd(theta, margs[1][1]),
        _mean_sd(s2, margs[2][1]),
    ]
    fit = sp.bayes_arima(
        y,
        order=(0, 0, 1),
        mean_prior=(0.0, v0),
        sigma2_prior=(a0, d0),
        draws=60000,
        burnin=2000,
        seed=12,
    )
    _check(fit, ["const", "ma.L1", "sigma2"], exact)
    det = fit.marginal_likelihood_details()
    assert abs(det["log_marginal_likelihood"] - log_norm) < max(4 * det["mcse"], 0.03)


def test_forecast_agrees_with_maximum_likelihood_under_a_vague_prior():
    rng = np.random.default_rng(5)
    T = 600
    e = rng.normal(size=T + 1)
    y = np.zeros(T)
    for t in range(T):
        y[t] = 2 + 0.7 * ((y[t - 1] - 2) if t else 0.0) + e[t + 1] + 0.4 * e[t]
    fit = sp.bayes_arima(y, order=(1, 0, 1), horizon=4, draws=8000, burnin=1000, seed=1)
    ml = sp.arima(y, order=(1, 0, 1))
    fc = fit.model_info["forecast"]
    ref = np.asarray(ml.forecast(4)["forecast"])
    # 2,000 predictive paths with standard deviation 1 to 1.8: the Monte
    # Carlo error of each mean is about 0.03 to 0.04
    assert np.abs(fc["mean"].to_numpy() - ref).max() < 0.2
    assert np.all(np.diff(fc["sd"].to_numpy()) > 0)
    for name in ("ar.L1", "ma.L1"):
        assert abs(fit.params[name] - ml.params[name]) < 0.02


# --------------------------------------------------------------------------
# Stochastic volatility
# --------------------------------------------------------------------------


def test_normal_mixture_approximates_log_chi_square():
    q, m, v = SV._MIX_Q, SV._MIX_M, SV._MIX_V
    assert q.sum() == pytest.approx(1.0, abs=1e-12)
    # exact mean digamma(1/2) + log 2 and variance pi^2 / 2
    assert q @ m == pytest.approx(-1.2703628454614782, abs=1e-4)
    assert q @ (v + m**2) - (q @ m) ** 2 == pytest.approx(np.pi**2 / 2, abs=1e-3)
    x = np.linspace(-15.0, 5.0, 4001)
    cdf = (q * stats.norm.cdf((x[:, None] - m) / np.sqrt(v))).sum(axis=1)
    assert np.abs(cdf - stats.chi2.cdf(np.exp(x), 1)).max() < 0.005


def test_stochvol_sweep_leaves_the_prior_invariant():
    q, m, v = SV._MIX_Q, SV._MIX_M, SV._MIX_V
    b_mu, B_mu, a0, b0, B_sig = prior = (-1.0, 0.25, 20.0, 1.5, 0.05)
    mb = a0 / (a0 + b0)
    vb = a0 * b0 / ((a0 + b0) ** 2 * (a0 + b0 + 1))
    truth = np.array(
        [
            b_mu,
            2 * mb - 1,
            B_sig,
            B_mu + b_mu**2,
            4 * vb + (2 * mb - 1) ** 2,
            3 * B_sig**2,
        ]
    )
    rng = np.random.default_rng(11)
    n, G, burn = 25, 80000, 2000
    theta = np.array([-1.0, 0.9, 0.05])
    h = np.full(n, -1.0)
    out = np.empty((G, 3))
    for g in range(G):
        s = rng.choice(7, size=n, p=q)
        ystar = h + m[s] + np.sqrt(v[s]) * rng.standard_normal(n)
        h, theta, _ = SV._sv_sweep(rng, ystar, h, theta, prior)
        out[g] = theta
    d = np.column_stack([out[burn:], out[burn:] ** 2])
    nb = 60
    L = len(d) // nb
    bm = d[: nb * L].reshape(nb, L, -1).mean(axis=1)
    z = (d.mean(axis=0) - truth) / (bm.std(axis=0, ddof=1) / np.sqrt(nb))
    assert np.abs(z).max() < 4.0, z.round(2)


def test_stochvol_screen_against_r_stochvol():
    ref = json.loads((FIX / "stochvol_R.json").read_text(encoding="utf-8"))
    y = pd.read_csv(FIX / "stochvol_returns.csv")["y"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.stochvol(y, draws=20000, burnin=3000, seed=1)
    runs = ref["runs"]
    for name in ("mu", "phi", "sigma"):
        r_mean = np.mean([r["mean"][name] for r in runs])
        r_sd = np.mean([r["sd"][name] for r in runs])
        # a fifth of a posterior standard deviation: several times the
        # spread of the three R runs, far below anything that matters
        assert abs(fit.params[name] - r_mean) < 0.2 * r_sd, name
        assert fit.std_errors[name] == pytest.approx(r_sd, rel=0.08), name
    vol = fit.model_info["volatility"]["mean"].to_numpy()[[0, 499, 999, 1499]]
    r_vol = np.mean([r["vol"] for r in runs], axis=0)
    assert np.allclose(vol, r_vol, rtol=0.03)
