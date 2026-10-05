"""``sp.bayes_mixed`` against the exact posterior of small hierarchical models.

Same idea as ``test_bayes_regress_exact_posterior.py``: with a random
intercept and two fixed effects the posterior can be computed without
simulation, by integrating analytically what is Gaussian and numerically
what is left.

* Gaussian outcome. Given ``(sigma2, D)`` the fixed and random effects
  integrate out in closed form: ``y ~ N(0, sigma2 I + D ZZ' + X B0 X')`` and
  ``E[beta | sigma2, D, y]`` is generalised least squares with the prior.
  A two-dimensional grid over ``(log sigma2, log D)`` does the rest.
* Logit and Poisson outcomes. Each group's random intercept is integrated
  out by Gauss-Hermite quadrature; a three-dimensional grid over
  ``(intercept, slope, log D)`` does the rest.

The integrands are written from ``scipy.stats`` and share no code with the
samplers. Acceptance: posterior means within four Monte Carlo standard
errors, posterior standard deviations within 5 percent.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import statspai as sp

G, T = 12, 4
V = 10.0  # prior variance of the fixed effects
A0, D0 = 3.0, 2.0  # sigma2 ~ InvGamma(A0 / 2, D0 / 2)
R0 = 3.0  # D ~ InvWishart(3, 3 * 1), i.e. InvGamma(3 / 2, 3 / 2)
KW = dict(
    group="id",
    prior_var=V,
    sigma2_prior=(A0, D0),
    re_prior=(R0, 1.0),
    draws=30000,
    burnin=3000,
    seed=5,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    g = np.repeat(np.arange(G), T)
    df = pd.DataFrame({"id": g, "x": rng.normal(size=G * T)})
    eta = 0.4 + 0.6 * df["x"] + rng.normal(scale=0.8, size=G)[g]
    df["y"] = eta + 0.7 * rng.normal(size=G * T)
    df["d"] = (rng.uniform(size=G * T) < special.expit(eta)).astype(int)
    df["c"] = rng.poisson(np.exp(0.5 * eta))
    return df


def log_prior_d(log_d: np.ndarray) -> np.ndarray:
    # density of D times the Jacobian of the log coordinate
    return stats.invgamma.logpdf(np.exp(log_d), R0 / 2, scale=R0 / 2) + log_d


def check(fit, exact_mean, exact_sd, names) -> None:
    t = fit.table.loc[names]
    z = (t["mean"].to_numpy() - exact_mean) / t["mcse"].to_numpy()
    assert np.abs(z).max() < 4.0, f"posterior means off by {z} Monte Carlo SEs"
    ratio = t["sd"].to_numpy() / exact_sd
    assert np.abs(ratio - 1.0).max() < 0.05, f"posterior sd ratio {ratio}"


def test_gaussian_random_intercept(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_mixed("y ~ x", panel, **KW)
    n = len(panel)
    y = panel["y"].to_numpy()
    X = np.column_stack([np.ones(n), panel["x"].to_numpy()])
    Z = np.zeros((n, G))
    Z[np.arange(n), panel["id"].to_numpy()] = 1.0
    logs = np.log(fit.draws[["sigma2", "var(Intercept)"]])
    c, s = logs.mean().to_numpy(), logs.std().to_numpy()
    m = 71
    ax = [np.linspace(ci - 8 * si, ci + 8 * si, m) for ci, si in zip(c, s)]
    lk = np.empty((m, m))
    cond_mean = np.empty((m, m, 2))
    cond_var = np.empty((m, m, 2))
    for i, ls2 in enumerate(ax[0]):
        for j, ld in enumerate(ax[1]):
            s2, dv = np.exp(ls2), np.exp(ld)
            vy = s2 * np.eye(n) + dv * Z @ Z.T
            lk[i, j] = (
                stats.multivariate_normal.logpdf(y, np.zeros(n), vy + V * X @ X.T)
                + stats.invgamma.logpdf(s2, A0 / 2, scale=D0 / 2)
                + ls2
                + log_prior_d(ld)
            )
            vi = np.linalg.inv(vy)
            bn = np.linalg.inv(np.eye(2) / V + X.T @ vi @ X)
            cond_mean[i, j] = bn @ X.T @ vi @ y
            cond_var[i, j] = np.diag(bn)
    w = np.exp(lk - lk.max())
    w /= w.sum()
    s2g, dg = np.meshgrid(np.exp(ax[0]), np.exp(ax[1]), indexing="ij")
    mean_beta = np.einsum("ij,ijk->k", w, cond_mean)
    # law of total variance for the fixed effects
    var_beta = np.einsum("ij,ijk->k", w, cond_var + cond_mean**2) - mean_beta**2
    mean_aux = np.array([(w * s2g).sum(), (w * dg).sum()])
    var_aux = np.array([(w * s2g**2).sum(), (w * dg**2).sum()]) - mean_aux**2
    check(
        fit,
        np.concatenate([mean_beta, mean_aux]),
        np.sqrt(np.concatenate([var_beta, var_aux])),
        ["Intercept", "x", "sigma2", "var(Intercept)"],
    )
    # the intraclass correlation is the same function of the same draws
    icc = (w * dg / (dg + s2g)).sum()
    assert fit.model_info["icc"]["mean"] == pytest.approx(icc, abs=0.01)


@pytest.mark.parametrize("family, outcome", [("logit", "d"), ("poisson", "c")])
def test_glmm_random_intercept(panel, family, outcome):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_mixed(f"{outcome} ~ x", panel, family=family, **KW)
    d = fit.draws
    y = panel[outcome].to_numpy()
    x = panel["x"].to_numpy()
    gid = panel["id"].to_numpy()
    coords = np.column_stack([d["Intercept"], d["x"], np.log(d["var(Intercept)"])])
    c, s = coords.mean(axis=0), coords.std(axis=0)
    m = 37
    ax = [np.linspace(ci - 7 * si, ci + 7 * si, m) for ci, si in zip(c, s)]
    grid = np.array(list(itertools.product(*ax)))
    nodes, wts = np.polynomial.hermite_e.hermegauss(32)
    log_w = np.log(wts / np.sqrt(2 * np.pi))
    sd_d = np.exp(0.5 * grid[:, 2])
    lk = np.zeros(len(grid))
    for g in range(G):
        rows = gid == g
        # linear index at [grid point, quadrature node, observation]
        eta = (
            grid[:, 0][:, None, None]
            + grid[:, 1][:, None, None] * x[rows][None, None, :]
            + (sd_d[:, None] * nodes[None, :])[:, :, None]
        )
        if family == "logit":
            ll = stats.bernoulli.logpmf(y[rows], special.expit(eta)).sum(axis=2)
        else:
            ll = stats.poisson.logpmf(y[rows], np.exp(eta)).sum(axis=2)
        lk += special.logsumexp(ll + log_w[None, :], axis=1)
    lk += stats.norm.logpdf(grid[:, :2], 0.0, np.sqrt(V)).sum(axis=1)
    lk += log_prior_d(grid[:, 2])
    w = np.exp(lk - lk.max())
    w /= w.sum()
    th = np.column_stack([grid[:, 0], grid[:, 1], np.exp(grid[:, 2])])
    mean = w @ th
    sd = np.sqrt(w @ (th - mean) ** 2)
    check(fit, mean, sd, ["Intercept", "x", "var(Intercept)"])


def test_random_slopes_agree_with_restricted_maximum_likelihood():
    """Many groups and a weak prior: the posterior sits on the REML fit."""
    rng = np.random.default_rng(11)
    n_g, n_t = 150, 8
    g = np.repeat(np.arange(n_g), n_t)
    df = pd.DataFrame({"id": g, "x": rng.normal(size=n_g * n_t)})
    cov = np.array([[0.6, 0.15], [0.15, 0.3]])
    b = rng.multivariate_normal([0, 0], cov, size=n_g)
    df["y"] = 1 + b[g, 0] + (0.5 + b[g, 1]) * df["x"] + 0.7 * rng.normal(size=len(df))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_mixed(
            "y ~ x", df, group="id", random=["x"], draws=6000, burnin=1000, seed=3
        )
        ref = sp.mixed(data=df, y="y", x_fixed=["x"], group="id", x_random=["x"])
    assert fit.params["x"] == pytest.approx(ref.params["x"], abs=0.01)
    assert fit.std_errors["x"] == pytest.approx(ref.std_errors["x"], rel=0.1)
    truth = {
        "sigma2": 0.49,
        "var(Intercept)": 0.6,
        "var(x)": 0.3,
        "cov(Intercept,x)": 0.15,
    }
    ci = fit.conf_int(level=0.99)
    for name, value in truth.items():
        assert ci.loc[name, "lower"] < value < ci.loc[name, "upper"], name
    # the group effects are recovered
    est = fit.random_effects[["Intercept", "x"]].to_numpy()
    assert np.corrcoef(est[:, 0], b[:, 0])[0, 1] > 0.85
    assert np.corrcoef(est[:, 1], b[:, 1])[0, 1] > 0.75
