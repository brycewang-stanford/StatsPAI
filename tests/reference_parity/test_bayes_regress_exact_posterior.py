"""Every sampler of ``sp.bayes_regress`` against its exact posterior.

MCMC output has no cross-language "same digits" reference: two correct
samplers with different random streams differ by Monte Carlo error. The
exact answer is still available when the model is small. With two
coefficients and at most one auxiliary parameter the posterior can be
integrated on a grid, which gives the posterior means, the posterior
standard deviations and the marginal likelihood to several digits without
any simulation.

The grid integrand is written here from ``scipy.stats`` densities and
shares no code with the samplers, so the comparison checks the sampler, the
prior parameterisation and the reported transformation at once:

* posterior mean within four Monte Carlo standard errors (the sampler's
  own ``mcse``) of the exact mean;
* posterior standard deviation within 5 percent;
* log marginal likelihood (Chib, Gelfand-Dey, Laplace) against the exact
  normalising constant.

The conjugate model has closed forms and is checked to 1e-9 against
independent expressions (the multivariate-t prior predictive, the ridge
form of the posterior mean), and the Savage-Dickey ratio is checked against
the ratio of two exact marginal likelihoods.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import statspai as sp

V = 10.0  # prior variance of each coefficient
A0, D0 = 3.0, 2.0  # sigma2 ~ InvGamma(A0 / 2, D0 / 2)
N = 40


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.normal(size=N)})
    eta = 0.3 + 0.7 * df["x"]
    df["y"] = eta + 0.8 * rng.normal(size=N)
    df["yb"] = (eta + rng.normal(size=N) > 0).astype(int)
    df["yl"] = (rng.uniform(size=N) < special.expit(eta)).astype(int)
    df["yc"] = rng.poisson(np.exp(eta))
    df["ynb"] = rng.negative_binomial(2.0, 2.0 / (2.0 + np.exp(eta)))
    df["ycens"] = np.maximum(df["y"], 0.0)
    df["yo"] = np.digitize(eta + rng.normal(size=N), [-0.2, 0.9])
    df["yt"] = eta + rng.standard_t(4, size=N)
    return df


def grid_posterior(log_kernel, report, center, spread, points, width=8.0):
    """Integrate ``exp(log_kernel)`` on a regular grid.

    ``log_kernel`` maps an (G, d) array of grid points to G log densities
    (likelihood x prior x Jacobian of the grid coordinates); ``report``
    maps the same points to the reported parameters. Returns the log
    normalising constant and the mean and sd of the reported parameters.
    """
    axes = [
        np.linspace(c - width * s, c + width * s, points)
        for c, s in zip(center, spread)
    ]
    cell = np.prod([a[1] - a[0] for a in axes])
    g = np.array(list(itertools.product(*axes)))
    lk = log_kernel(g)
    top = lk.max()
    w = np.exp(lk - top)
    log_norm = top + np.log(w.sum() * cell)
    # the grid must hold essentially all the mass (posteriors with an
    # unknown scale have Student-t tails, hence not a tighter bound)
    edge = np.zeros(len(g), dtype=bool)
    for j, a in enumerate(axes):
        edge |= (g[:, j] == a[0]) | (g[:, j] == a[-1])
    assert w[edge].max() < 1e-6
    w = w / w.sum()
    th = report(g)
    mean = w @ th
    sd = np.sqrt(w @ (th - mean) ** 2)
    return log_norm, mean, sd


def beta_prior(b):
    return stats.norm.logpdf(b, 0.0, np.sqrt(V)).sum(axis=1)


def s2_prior(log_s2):
    # density of sigma2 times the Jacobian of the log coordinate
    s2 = np.exp(log_s2)
    return stats.invgamma.logpdf(s2, A0 / 2.0, scale=D0 / 2.0) + log_s2


# Each case: outcome, bayes_regress kwargs, grid log-kernel and report map.
# Grid coordinates are (intercept, slope[, auxiliary on the log scale]).


def case_normal(df):
    y, x = df["y"].to_numpy(), df["x"].to_numpy()

    def lk(g):
        mu = g[:, [0]] + g[:, [1]] * x
        s = np.exp(0.5 * g[:, [2]])
        return (
            stats.norm.logpdf(y, mu, s).sum(axis=1)
            + beta_prior(g[:, :2])
            + s2_prior(g[:, 2])
        )

    return "y ~ x", {}, lk, lambda g: np.column_stack([g[:, :2], np.exp(g[:, 2])])


def case_t(df):
    y, x = df["yt"].to_numpy(), df["x"].to_numpy()

    def lk(g):
        mu = g[:, [0]] + g[:, [1]] * x
        s = np.exp(0.5 * g[:, [2]])
        return (
            stats.t.logpdf(y, 4.0, mu, s).sum(axis=1)
            + beta_prior(g[:, :2])
            + s2_prior(g[:, 2])
        )

    return (
        "yt ~ x",
        {"dof": 4.0},
        lk,
        lambda g: np.column_stack([g[:, :2], np.exp(g[:, 2])]),
    )


def case_probit(df):
    y, x = df["yb"].to_numpy(), df["x"].to_numpy()

    def lk(g):
        p = stats.norm.cdf(g[:, [0]] + g[:, [1]] * x)
        return stats.bernoulli.logpmf(y, p).sum(axis=1) + beta_prior(g)

    return "yb ~ x", {}, lk, lambda g: g


def case_logit(df):
    y, x = df["yl"].to_numpy(), df["x"].to_numpy()

    def lk(g):
        p = special.expit(g[:, [0]] + g[:, [1]] * x)
        return stats.bernoulli.logpmf(y, p).sum(axis=1) + beta_prior(g)

    return "yl ~ x", {}, lk, lambda g: g


def case_poisson(df):
    y, x = df["yc"].to_numpy(), df["x"].to_numpy()

    def lk(g):
        mu = np.exp(g[:, [0]] + g[:, [1]] * x)
        return stats.poisson.logpmf(y, mu).sum(axis=1) + beta_prior(g)

    return "yc ~ x", {}, lk, lambda g: g


def case_negbin(df):
    y, x = df["ynb"].to_numpy(), df["x"].to_numpy()

    def lk(g):
        # third coordinate: log(alpha); the Gamma(2, rate 1) prior is on
        # the size r = 1 / alpha, so the Jacobian is |dr / dlog(alpha)| = r
        mu = np.exp(g[:, [0]] + g[:, [1]] * x)
        r = np.exp(-g[:, [2]])
        ll = stats.nbinom.logpmf(y, r, r / (r + mu)).sum(axis=1)
        prior = stats.gamma.logpdf(r[:, 0], 2.0, scale=1.0) + np.log(r[:, 0])
        return ll + beta_prior(g[:, :2]) + prior

    kwargs = {"size_prior": (2.0, 1.0)}
    return "ynb ~ x", kwargs, lk, lambda g: np.column_stack([g[:, :2], np.exp(g[:, 2])])


def case_tobit(df):
    y, x = df["ycens"].to_numpy(), df["x"].to_numpy()
    cens = y <= 0.0

    def lk(g):
        mu = g[:, [0]] + g[:, [1]] * x
        s = np.exp(0.5 * g[:, [2]])
        ll = stats.norm.logpdf(y[~cens], mu[:, ~cens], s).sum(axis=1)
        ll += stats.norm.logcdf((0.0 - mu[:, cens]) / s).sum(axis=1)
        return ll + beta_prior(g[:, :2]) + s2_prior(g[:, 2])

    kwargs = {"lower": 0.0}
    return (
        "ycens ~ x",
        kwargs,
        lk,
        lambda g: np.column_stack([g[:, :2], np.exp(g[:, 2])]),
    )


def _ald_loglik(y, mu, sigma, p):
    u = (y - mu) / sigma
    return (np.log(p * (1 - p)) - np.log(sigma) - u * (p - (u < 0))).sum(axis=1)


def case_quantile(df):
    y, x = df["y"].to_numpy(), df["x"].to_numpy()

    def lk(g):
        mu = g[:, [0]] + g[:, [1]] * x
        sig = np.exp(g[:, [2]])
        prior = stats.invgamma.logpdf(sig[:, 0], A0 / 2.0, scale=D0 / 2.0) + g[:, 2]
        return _ald_loglik(y, mu, sig, 0.3) + beta_prior(g[:, :2]) + prior

    kwargs = {"quantile": 0.3, "scale_prior": (A0, D0)}
    return "y ~ x", kwargs, lk, lambda g: np.column_stack([g[:, :2], np.exp(g[:, 2])])


def case_quantile_fixed_scale(df):
    y, x = df["y"].to_numpy(), df["x"].to_numpy()

    def lk(g):
        mu = g[:, [0]] + g[:, [1]] * x
        return _ald_loglik(y, mu, 0.5, 0.7) + beta_prior(g)

    return "y ~ x", {"quantile": 0.7, "scale": 0.5}, lk, lambda g: g


def case_oprobit(df):
    y, x = df["yo"].to_numpy(), df["x"].to_numpy()

    def lk(g):
        # coordinates: slope, first cutpoint, log(second - first)
        b, c1, d = g[:, [0]], g[:, [1]], g[:, [2]]
        c2 = c1 + np.exp(d)
        eta = b * x
        p0 = stats.norm.cdf(c1 - eta)
        p1 = stats.norm.cdf(c2 - eta) - p0
        p2 = stats.norm.sf(c2 - eta)
        prob = np.where(y == 0, p0, np.where(y == 1, p1, p2))
        ll = np.log(prob).sum(axis=1)
        prior = (
            stats.norm.logpdf(g[:, 0], 0, np.sqrt(V))
            + stats.norm.logpdf(-g[:, 1], 0, np.sqrt(V))  # minus the first cutpoint
            + stats.norm.logpdf(g[:, 2], 0, 1.0)  # log increment
        )
        return ll + prior

    def report(g):
        return np.column_stack([g[:, 0], g[:, 1], g[:, 1] + np.exp(g[:, 2])])

    return "yo ~ x", {}, lk, report


CASES = {
    "normal": case_normal,
    "t": case_t,
    "probit": case_probit,
    "logit": case_logit,
    "poisson": case_poisson,
    "negbin": case_negbin,
    "tobit": case_tobit,
    "quantile": case_quantile,
    "quantile-fixed-scale": case_quantile_fixed_scale,
    "oprobit": case_oprobit,
}
# which marginal-likelihood estimators each model has, with the tolerance
# on the log scale (Laplace is an approximation, not a consistent estimator)
LML = {
    "normal": {"chib": 0.02, "gelfand-dey": 0.03, "laplace": 0.15},
    "t": {"gelfand-dey": 0.03, "laplace": 0.15},
    "probit": {"chib": 0.02, "gelfand-dey": 0.03, "laplace": 0.15},
    "logit": {"gelfand-dey": 0.04, "laplace": 0.15},
    "poisson": {"gelfand-dey": 0.04, "laplace": 0.15},
    "negbin": {"gelfand-dey": 0.05, "laplace": 0.2},
    "tobit": {"gelfand-dey": 0.03, "laplace": 0.2},
    "quantile": {"gelfand-dey": 0.03},
    "quantile-fixed-scale": {"gelfand-dey": 0.03},
    "oprobit": {"gelfand-dey": 0.04, "laplace": 0.15},
}


@pytest.mark.parametrize("name", list(CASES))
def test_sampler_recovers_the_exact_posterior(data, name):
    formula, kwargs, log_kernel, report = CASES[name](data)
    model = name.split("-")[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            formula,
            data,
            model=model,
            prior_var=V,
            sigma2_prior=(A0, D0),
            draws=30000,
            burnin=2000,
            seed=11,
            **kwargs,
        )
    # place the grid from the draws, in the grid's own coordinates
    d = fit.draws.to_numpy()
    if d.shape[1] == 3 and model != "oprobit":
        coords = np.column_stack([d[:, :2], np.log(d[:, 2])])
    elif model == "oprobit":
        coords = np.column_stack([d[:, 0], d[:, 1], np.log(d[:, 2] - d[:, 1])])
    else:
        coords = d
    points = 61 if coords.shape[1] == 3 else 241
    log_norm, mean, sd = grid_posterior(
        log_kernel, report, coords.mean(axis=0), coords.std(axis=0), points
    )
    t = fit.table
    z = (t["mean"].to_numpy() - mean) / t["mcse"].to_numpy()
    assert np.abs(z).max() < 4.0, f"posterior means off by {z} Monte Carlo SEs"
    ratio = t["sd"].to_numpy() / sd
    assert np.abs(ratio - 1.0).max() < 0.05, f"posterior sd ratio {ratio}"
    for method, tol in LML[name].items():
        got = fit.log_marginal_likelihood(method)
        assert abs(got - log_norm) < tol, f"{method}: {got} vs exact {log_norm}"


# --------------------------------------------------------------------------
# conjugate model: closed forms
# --------------------------------------------------------------------------


def test_conjugate_closed_forms(data):
    fit = sp.bayes_regress(
        "y ~ x", data, model="conjugate", prior_var=V, sigma2_prior=(A0, D0), seed=1
    )
    y = data["y"].to_numpy()
    X = np.column_stack([np.ones(N), data["x"].to_numpy()])
    # posterior mean: ridge with penalty 1 / V
    ridge = np.linalg.solve(X.T @ X + np.eye(2) / V, X.T @ y)
    assert np.allclose(fit.params.to_numpy()[:2], ridge, rtol=1e-10)
    # marginal likelihood: the prior predictive of y is multivariate t
    scale = (D0 / A0) * (np.eye(N) + V * X @ X.T)
    exact = stats.multivariate_t.logpdf(y, loc=np.zeros(N), shape=scale, df=A0)
    assert fit.log_marginal_likelihood() == pytest.approx(exact, rel=1e-10)
    # posterior of sigma2: InvGamma((A0 + N) / 2, dn / 2)
    e = y - X @ ridge
    dn = D0 + e @ e + ridge @ ridge / V
    assert fit.model_info["sigma2_rate"] == pytest.approx(dn / 2, rel=1e-10)
    assert fit.params["sigma2"] == pytest.approx(dn / (A0 + N - 2), rel=1e-10)
    # the reported interval is the Student-t interval
    Bn = np.linalg.inv(X.T @ X + np.eye(2) / V)
    half = stats.t.ppf(0.975, A0 + N) * np.sqrt(dn / (A0 + N) * np.diag(Bn))
    ci = fit.conf_int()
    assert np.allclose(ci["upper"].to_numpy()[:2], ridge + half, rtol=1e-10)
    assert np.allclose(ci["lower"].to_numpy()[:2], ridge - half, rtol=1e-10)
    # and the independent draws agree with it
    q = fit.draws["x"].quantile([0.025, 0.975]).to_numpy()
    assert np.allclose(q, [ridge[1] - half[1], ridge[1] + half[1]], atol=0.01)


def test_savage_dickey_is_the_ratio_of_exact_marginal_likelihoods():
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = 1 + 0.6 * df["x"] + 0.12 * df["z"] + rng.normal(size=n)
    full = sp.bayes_regress(
        "y ~ x + z",
        df,
        model="conjugate",
        prior_var=4.0,
        sigma2_prior=(4.0, 3.0),
        seed=2,
    )
    # conditioning on z = 0 adds one degree of freedom to the sigma2 prior
    rest = sp.bayes_regress(
        "y ~ x", df, model="conjugate", prior_var=4.0, sigma2_prior=(5.0, 3.0), seed=2
    )
    sd = sp.savage_dickey(full, "z")
    assert sd["estimator"] == "exact"
    assert sd["log_bf01"] == pytest.approx(
        sp.bayes_factor(rest, full).log_bf, abs=1e-10
    )


def test_savage_dickey_rao_blackwell_agrees_with_chib():
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = 1 + 0.6 * df["x"] + 0.12 * df["z"] + rng.normal(size=n)
    df["d"] = (0.2 + 0.6 * df["x"] + 0.2 * df["z"] + rng.normal(size=n) > 0).astype(int)
    for outcome, model in (("y", "normal"), ("d", "probit")):
        kw = dict(
            model=model, prior_var=4.0, sigma2_prior=(4.0, 3.0), draws=20000, seed=2
        )
        full = sp.bayes_regress(f"{outcome} ~ x + z", df, **kw)
        rest = sp.bayes_regress(f"{outcome} ~ x", df, **kw)
        sd = sp.savage_dickey(full, "z")
        assert sd["estimator"] == "rao-blackwell"
        assert sd["log_bf01"] == pytest.approx(
            sp.bayes_factor(rest, full).log_bf, abs=0.03
        )


def test_bayesian_bootstrap_of_a_mean_has_rubins_variance():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({"y": rng.gamma(2.0, size=120)})
    y = df["y"].to_numpy()
    n = len(y)
    fit = sp.bayes_bootstrap("y ~ 1", df, draws=40000, seed=1)
    exact_sd = np.sqrt(((y - y.mean()) ** 2).sum() / (n * (n + 1)))
    assert fit.params["Intercept"] == pytest.approx(y.mean(), abs=4 * exact_sd / 200)
    # sd of an sd estimate from 40,000 independent draws: about 0.35 percent
    assert fit.std_errors["Intercept"] == pytest.approx(exact_sd, rel=0.015)
    # the callable form is the same calculation
    same = sp.bayes_bootstrap(
        data=df,
        statistic=lambda d, w: float(w @ d["y"].to_numpy()),
        draws=40000,
        seed=1,
    )
    assert same.params["statistic"] == pytest.approx(fit.params["Intercept"], rel=1e-12)
