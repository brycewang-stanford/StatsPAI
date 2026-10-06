"""The Bayesian workflow pieces against answers that need no sampler.

Companion of ``test_bayes_regress_exact_posterior.py`` for what the
Gelman-Hill-Vehtari pass added:

* the weakly informative prior (``prior='weakly_informative'``): each
  model's posterior integrated on a grid, the integrand written here from
  ``scipy.stats`` in the *centred* parameterisation the prior is stated
  in, while the sampler works on raw coefficients with a correlated prior.
  Agreement checks the prior transformation and, for the Gaussian model,
  the independence Metropolis step for ``sigma ~ Exponential``;
* offsets, against the same kind of grid;
* the horseshoe Gibbs sampler with one regressor, where the two
  half-Cauchy scales collapse into their product and the posterior is
  three-dimensional;
* PSIS-LOO against exact leave-one-out, available in closed form for the
  conjugate normal model (a Student-t predictive density);
* the pointwise log-likelihood of every model against the likelihood its
  own sampler uses, and the predictive draws against their moments.

Tolerances: posterior means within four Monte Carlo standard errors
(the sampler's own ``mcse``), standard deviations within 6 percent, as in
the companion file.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

import statspai as sp

N = 40


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": rng.normal(3.0, 2.0, size=N)})
    eta = -0.9 + 0.5 * df["x"]
    df["y"] = 4.0 + 1.5 * df["x"] + 2.0 * rng.normal(size=N)
    df["yl"] = (rng.uniform(size=N) < special.expit(eta)).astype(int)
    df["yb"] = (eta + rng.normal(size=N) > 0).astype(int)
    df["expo"] = rng.uniform(0.5, 3.0, size=N)
    df["yc"] = rng.poisson(df["expo"] * np.exp(-0.5 + 0.3 * df["x"]))
    df["ynb"] = rng.negative_binomial(2.0, 2.0 / (2.0 + np.exp(eta + 1.0)))
    df["yt"] = eta + rng.standard_t(4, size=N)
    df["ycens"] = np.maximum(df["y"] - 8.0, 0.0)
    df["yo"] = np.digitize(eta + rng.normal(size=N), [-0.2, 0.9])
    df["ym"] = rng.integers(0, 3, size=N)
    return df


def grid_posterior(log_kernel, report, center, spread, points, width=8.0):
    axes = [
        np.linspace(c - width * s, c + width * s, points)
        for c, s in zip(center, spread)
    ]
    g = np.array(list(itertools.product(*axes)))
    lk = log_kernel(g)
    w = np.exp(lk - lk.max())
    edge = np.zeros(len(g), dtype=bool)
    for j, a in enumerate(axes):
        edge |= (g[:, j] == a[0]) | (g[:, j] == a[-1])
    assert w[edge].max() < 1e-5, "the grid cuts off posterior mass"
    w = w / w.sum()
    th = report(g)
    mean = w @ th
    return mean, np.sqrt(w @ (th - mean) ** 2)


def check(fit, mean, sd, names):
    for j, name in enumerate(names):
        row = fit.table.loc[name]
        assert abs(row["mean"] - mean[j]) < 4 * row["mcse"] + 1e-3 * sd[j], (
            name,
            row["mean"],
            mean[j],
        )
        assert row["sd"] == pytest.approx(sd[j], rel=0.06), (name, row["sd"], sd[j])


# ---------------------------------------------------------------------
# weakly informative priors, written in the centred parameterisation
# ---------------------------------------------------------------------


def test_weakly_informative_gaussian_posterior(data):
    y, x = data["y"].to_numpy(), data["x"].to_numpy()
    s_y, s_x, xbar = y.std(ddof=1), x.std(ddof=1), x.mean()

    def lk(g):
        a, b, log_s = g[:, [0]], g[:, [1]], g[:, 2]
        s = np.exp(log_s)
        centred = a[:, 0] + b[:, 0] * xbar
        return (
            stats.norm.logpdf(y, a + b * x, s[:, None]).sum(axis=1)
            + stats.norm.logpdf(centred, y.mean(), 2.5 * s_y)
            + stats.norm.logpdf(b[:, 0], 0.0, 2.5 * s_y / s_x)
            + stats.expon.logpdf(s, scale=s_y)
            + log_s  # Jacobian of the log coordinate
        )

    ols = np.polyfit(x, y, 1)
    resid = y - np.polyval(ols, x)
    s_hat = np.sqrt(resid @ resid / (N - 2))
    se_b = s_hat / (s_x * np.sqrt(N - 1))
    se_a = s_hat * np.sqrt(1 / N + xbar**2 / ((N - 1) * s_x**2))
    mean, sd = grid_posterior(
        lk,
        lambda g: np.column_stack([g[:, :2], np.exp(2 * g[:, 2])]),
        [ols[1], ols[0], np.log(s_hat)],
        [se_a, se_b, 1 / np.sqrt(2 * (N - 2))],
        points=61,
        width=7.0,
    )
    fit = sp.bayes_regress(
        "y ~ x", data, prior="weakly_informative", draws=40000, burnin=1000, seed=3
    )
    check(fit, mean, sd, ["Intercept", "x", "sigma2"])
    assert fit.prior["kind"] == "weakly_informative"
    # the Metropolis step for sigma accepts nearly always
    assert fit._extras["sigma_accept"] > 0.8


def _glm_grid(data, col, loglik_fn, extra_prior=None, offset=0.0, points=121):
    y, x = data[col].to_numpy(dtype=float), data["x"].to_numpy()
    s_x, xbar = x.std(ddof=1), x.mean()

    def lk(g):
        a, b = g[:, [0]], g[:, [1]]
        eta = a + b * x + offset
        centred = a[:, 0] + b[:, 0] * xbar
        val = (
            loglik_fn(y, eta, g).sum(axis=1)
            + stats.norm.logpdf(centred, 0.0, 2.5)
            + stats.norm.logpdf(b[:, 0], 0.0, 2.5 / s_x)
        )
        return val if extra_prior is None else val + extra_prior(g)

    return lk


@pytest.mark.parametrize("model, col", [("logit", "yl"), ("probit", "yb")])
def test_weakly_informative_binary_posterior(data, model, col):
    cdf = special.expit if model == "logit" else special.ndtr

    def ll(y, eta, g):
        p = np.clip(cdf(eta), 1e-300, 1 - 1e-16)
        return y * np.log(p) + (1 - y) * np.log1p(-p)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            f"{col} ~ x",
            data,
            model=model,
            prior="weakly_informative",
            draws=40000,
            burnin=2000,
            thin=2,
            seed=5,
        )
    center = fit.table["mean"].to_numpy()[:2]
    spread = fit.table["sd"].to_numpy()[:2]
    mean, sd = grid_posterior(
        _glm_grid(data, col, ll), lambda g: g, center, spread, points=161, width=7.0
    )
    check(fit, mean, sd, ["Intercept", "x"])


def test_poisson_posterior_with_exposure_and_weak_prior(data):
    offset = np.log(data["expo"].to_numpy())

    def ll(y, eta, g):
        return y * eta - np.exp(eta) - special.gammaln(y + 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            "yc ~ x",
            data,
            model="poisson",
            exposure="expo",
            prior="weakly_informative",
            draws=40000,
            burnin=2000,
            thin=2,
            seed=7,
        )
    center = fit.table["mean"].to_numpy()[:2]
    spread = fit.table["sd"].to_numpy()[:2]
    mean, sd = grid_posterior(
        _glm_grid(data, "yc", ll, offset=offset),
        lambda g: g,
        center,
        spread,
        points=161,
        width=7.0,
    )
    check(fit, mean, sd, ["Intercept", "x"])
    # offset= with the logged column is the same model
    logged = data.assign(lexpo=offset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        same = sp.bayes_regress(
            "yc ~ x",
            logged,
            model="poisson",
            offset="lexpo",
            prior="weakly_informative",
            draws=40000,
            burnin=2000,
            thin=2,
            seed=7,
        )
    np.testing.assert_allclose(same.draws.to_numpy(), fit.draws.to_numpy())
    # without the exposure the intercept absorbs its mean: a different fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.bayes_regress(
            "yc ~ x", data, model="poisson", draws=4000, burnin=1000, seed=7
        )
    assert abs(plain.params["Intercept"] - fit.params["Intercept"]) > 0.2


def test_negative_binomial_weak_prior_puts_exponential_on_the_size(data):
    def ll(y, eta, g):
        r = np.exp(g[:, [2]])
        mu = np.exp(eta)
        return (
            special.gammaln(y + r)
            - special.gammaln(r)
            - special.gammaln(y + 1)
            + r * np.log(r)
            + y * eta
            - (y + r) * np.log(r + mu)
        )

    def size_prior(g):
        # r ~ Exponential(1), grid coordinate log r
        return -np.exp(g[:, 2]) + g[:, 2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            "ynb ~ x",
            data,
            model="negbin",
            prior="weakly_informative",
            draws=60000,
            burnin=3000,
            thin=2,
            seed=9,
        )
    log_r = -np.log(fit.draws["alpha"].to_numpy())
    center = list(fit.table["mean"].to_numpy()[:2]) + [log_r.mean()]
    spread = list(fit.table["sd"].to_numpy()[:2]) + [log_r.std()]
    mean, sd = grid_posterior(
        _glm_grid(data, "ynb", ll, extra_prior=size_prior),
        lambda g: np.column_stack([g[:, :2], np.exp(-g[:, 2])]),
        center,
        spread,
        points=61,
        width=6.0,
    )
    check(fit, mean, sd, ["Intercept", "x", "alpha"])
    assert "Gamma(1, rate 1)" in fit.prior["size"]


# ---------------------------------------------------------------------
# horseshoe, one regressor
# ---------------------------------------------------------------------


def _product_of_half_cauchy_logpdf(log_rho: np.ndarray, tau0: float) -> np.ndarray:
    """Log density of ``log(lam * tau)``, lam ~ C+(0, 1), tau ~ C+(0, tau0)."""
    out = np.empty_like(log_rho)
    for i, lr in enumerate(log_rho):
        rho = np.exp(lr)

        def f(t):  # t = log(lam)
            lam = np.exp(t)
            return (
                stats.halfcauchy.pdf(lam)
                * stats.halfcauchy.pdf(rho / lam, scale=tau0)
                / lam
                * lam
            )

        val, _ = integrate.quad(f, -40, 40, points=[0.0, lr], limit=400)
        out[i] = np.log(val) + lr
    return out


@pytest.mark.parametrize("tau0", [1.0, 0.2])
def test_horseshoe_posterior_with_one_regressor(tau0):
    rng = np.random.default_rng(21)
    n = 30
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": 0.5 + 0.35 * x + rng.normal(size=n)})
    a0, d0 = 4.0, 3.0
    z = (x - x.mean()) / x.std(ddof=1)
    yc = df["y"].to_numpy() - df["y"].mean()
    zz, zy, yy = z @ z, z @ yc, yc @ yc

    # The coefficient is integrated out analytically: given (sigma2, rho) it
    # is N(zy / A, sigma2 / A) with A = zz + 1 / rho^2. A grid over it could
    # not resolve the spike at zero that a small rho produces.
    log_rho_axis = np.linspace(-14.0, 9.0, 461)
    rho_prior = _product_of_half_cauchy_logpdf(log_rho_axis, tau0)
    b_hat = zy / zz
    s2_hat = (yy - b_hat * zy) / (n - 2)
    ls_axis = np.linspace(np.log(s2_hat) - 2.8, np.log(s2_hat) + 2.8, 281)
    LS, LR = np.meshgrid(ls_axis, log_rho_axis, indexing="ij")
    s2 = np.exp(LS)
    rho2 = np.exp(2 * LR)
    A = zz + 1.0 / rho2
    lk = (
        # flat intercept integrated out: n - 1 residual degrees of freedom
        -0.5 * (n - 1) * LS
        - 0.5 * (yy - zy**2 / A) / s2
        - 0.5 * np.log(rho2 * A)
        + stats.invgamma.logpdf(s2, a0 / 2, scale=d0 / 2)
        + LS
        + rho_prior[None, :]
    )
    w = np.exp(lk - lk.max())
    for edge in (w[0], w[-1], w[:, 0], w[:, -1]):
        assert edge.max() < 1e-4
    w = w / w.sum()
    kappa = 1.0 / (1.0 + n * (zz / n) * rho2)
    cond_mean = zy / A
    m_bz = float((w * cond_mean).sum())
    var_bz = float((w * (s2 / A + (cond_mean - m_bz) ** 2)).sum())

    def moments(v):
        m = float((w * v).sum())
        return m, float(np.sqrt((w * (v - m) ** 2).sum()))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_shrink(
            "y ~ x",
            df,
            prior="horseshoe",
            global_scale=tau0,
            sigma2_prior=(a0, d0),
            draws=60000,
            burnin=3000,
            thin=2,
            seed=4,
        )
    # the coefficient is reported on the original scale of x
    m_b, sd_b = m_bz / x.std(ddof=1), np.sqrt(var_bz) / x.std(ddof=1)
    m_s, sd_s = moments(s2)
    row = fit.table.loc["x"]
    assert abs(row["mean"] - m_b) < 4 * row["mcse"] + 0.003 * sd_b
    assert row["sd"] == pytest.approx(sd_b, rel=0.06)
    row = fit.table.loc["sigma2"]
    assert abs(row["mean"] - m_s) < 4 * row["mcse"] + 0.003 * sd_s
    assert row["sd"] == pytest.approx(sd_s, rel=0.06)
    assert fit.table.loc["x", "shrinkage"] == pytest.approx(moments(kappa)[0], abs=0.02)
    assert fit.model_info["m_eff"] == pytest.approx(1 - moments(kappa)[0], abs=0.02)


def test_horseshoe_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    n, p = 120, 25
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:3] = [2.0, -1.5, 1.0]
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["y"] = 1 + X @ beta + rng.normal(size=n)
    formula = "y ~ " + " + ".join(df.columns[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hs = sp.bayes_shrink(
            formula, df, prior="horseshoe", p0=3, draws=3000, burnin=1000, seed=1
        )
        flat = sp.bayes_regress(
            formula, df, prior="weakly_informative", draws=3000, burnin=500, seed=1
        )
    shrink = hs.table["shrinkage"].dropna()
    assert shrink.iloc[:3].max() < 0.1
    assert shrink.iloc[3:].min() > 0.3
    err_hs = np.sqrt(((hs.params.iloc[1 : p + 1].to_numpy() - beta) ** 2).mean())
    err_flat = np.sqrt(((flat.params.iloc[1 : p + 1].to_numpy() - beta) ** 2).mean())
    assert err_hs < 0.75 * err_flat
    # and it predicts better out of sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = sp.loo_compare(sp.loo(hs), sp.loo(flat), names=["horseshoe", "weak"])
    assert cmp.index[0] == "horseshoe"
    assert cmp.loc["weak", "elpd_diff"] < -3


# ---------------------------------------------------------------------
# PSIS-LOO against exact leave-one-out
# ---------------------------------------------------------------------


def test_loo_matches_exact_leave_one_out_of_the_conjugate_model(data):
    """With the conjugate prior the leave-one-out predictive density is a
    Student-t in closed form. PSIS from one fit of 20,000 independent
    posterior draws must reproduce the n exact values."""
    y, x = data["y"].to_numpy(), data["x"].to_numpy()
    X = np.column_stack([np.ones(N), x])
    v, a0, d0 = 4.0, 3.0, 2.0
    exact = np.empty(N)
    for i in range(N):
        keep = np.arange(N) != i
        Xi, yi = X[keep], y[keep]
        prec = np.eye(2) / v + Xi.T @ Xi
        Bn = np.linalg.inv(prec)
        bn = Bn @ (Xi.T @ yi)
        an = a0 + (N - 1)
        dn = d0 + yi @ yi - bn @ prec @ bn
        scale2 = dn / an * (1.0 + X[i] @ Bn @ X[i])
        exact[i] = stats.t.logpdf(y[i], an, loc=X[i] @ bn, scale=np.sqrt(scale2))
    fit = sp.bayes_regress(
        "y ~ x",
        data,
        model="conjugate",
        prior_var=v,
        sigma2_prior=(a0, d0),
        draws=20000,
        seed=2,
    )
    out = sp.loo(fit)
    assert out.pareto_k.max() < 0.5
    np.testing.assert_allclose(out.pointwise["elpd"], exact, atol=0.02)
    assert out.elpd == pytest.approx(exact.sum(), abs=4 * out.mcse_elpd + 0.02)
    # K-fold with one observation per fold is leave-one-out by refitting
    kf = sp.kfold(fit, folds=np.arange(N))
    np.testing.assert_allclose(kf.pointwise["elpd"], exact, atol=0.02)
    # and WAIC is close to both, as the theory says
    assert sp.waic(fit).elpd == pytest.approx(exact.sum(), abs=0.3)


# ---------------------------------------------------------------------
# pointwise likelihood and predictive draws, model by model
# ---------------------------------------------------------------------

CASES = [
    ("normal", "y ~ x", {}),
    ("conjugate", "y ~ x", {}),
    ("t", "yt ~ x", {"dof": 4}),
    ("logit", "yl ~ x", {}),
    ("probit", "yb ~ x", {}),
    ("poisson", "yc ~ x", {"exposure": "expo"}),
    ("negbin", "ynb ~ x", {}),
    ("tobit", "ycens ~ x", {"lower": 0.0}),
    ("quantile", "y ~ x", {"quantile": 0.3}),
    ("oprobit", "yo ~ x", {}),
    ("mlogit", "ym ~ x", {}),
]


@pytest.fixture(scope="module")
def fits(data):
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model, formula, kw in CASES:
            out[model] = sp.bayes_regress(
                formula, data, model=model, draws=600, burnin=300, seed=1, **kw
            )
    return out


@pytest.mark.parametrize("model", [c[0] for c in CASES if c[0] != "conjugate"])
def test_pointwise_log_lik_sums_to_the_sampler_likelihood(fits, model):
    """Each sampler is driven by its own scalar log-likelihood. The
    draws-by-observations matrix, written separately, must add up to it."""
    fit = fits[model]
    ll = fit.log_lik()
    assert ll.shape == (600, N)
    mdl = fit._model
    theta = fit.draws.to_numpy()
    for s in (0, 137, 599):
        total = mdl.log_lik(theta[s])
        assert ll[s].sum() == pytest.approx(total, rel=1e-9, abs=1e-8)


def test_pointwise_log_lik_against_scipy(fits, data):
    d = fits["normal"].draws.to_numpy()
    x, y = data["x"].to_numpy(), data["y"].to_numpy()
    want = stats.norm.logpdf(y, d[:, [0]] + d[:, [1]] * x, np.sqrt(d[:, [2]]))
    np.testing.assert_allclose(fits["normal"].log_lik(), want, rtol=1e-11)
    d = fits["conjugate"].draws.to_numpy()
    want = stats.norm.logpdf(y, d[:, [0]] + d[:, [1]] * x, np.sqrt(d[:, [2]]))
    np.testing.assert_allclose(fits["conjugate"].log_lik(), want, rtol=1e-11)
    d = fits["poisson"].draws.to_numpy()
    mu = data["expo"].to_numpy() * np.exp(d[:, [0]] + d[:, [1]] * x)
    want = stats.poisson.logpmf(data["yc"].to_numpy(), mu)
    np.testing.assert_allclose(fits["poisson"].log_lik(), want, rtol=1e-9, atol=1e-10)
    d = fits["negbin"].draws.to_numpy()
    r = 1 / d[:, [2]]
    mu = np.exp(d[:, [0]] + d[:, [1]] * x)
    want = stats.nbinom.logpmf(data["ynb"].to_numpy(), r, r / (r + mu))
    np.testing.assert_allclose(fits["negbin"].log_lik(), want, rtol=1e-9, atol=1e-10)
    d = fits["t"].draws.to_numpy()
    want = stats.t.logpdf(
        data["yt"].to_numpy(),
        4,
        loc=d[:, [0]] + d[:, [1]] * x,
        scale=np.sqrt(d[:, [2]]),
    )
    np.testing.assert_allclose(fits["t"].log_lik(), want, rtol=1e-10)


def test_log_lik_on_new_data_is_the_same_function(fits, data):
    for model in ("normal", "logit", "poisson", "oprobit", "mlogit", "tobit"):
        fit = fits[model]
        np.testing.assert_allclose(fit.log_lik(data), fit.log_lik(), rtol=1e-10)
        part = fit.log_lik(data.iloc[5:12])
        np.testing.assert_allclose(part, fit.log_lik()[:, 5:12], rtol=1e-10)


@pytest.mark.parametrize(
    "model", ["normal", "t", "logit", "probit", "poisson", "negbin", "quantile"]
)
def test_predictive_draws_have_the_model_moments(data, model):
    formula, kw = next((f, k) for m, f, k in CASES if m == model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            formula, data, model=model, draws=20000, burnin=500, seed=3, **kw
        )
    y_rep = fit.posterior_predict(seed=8)
    assert y_rep.shape == (20000, N)
    mu = fit.posterior_epred()
    d = fit.draws.to_numpy()
    if model == "quantile":
        # the conditional quantile, not the mean, is what the model fits:
        # the share of predictive draws below it is the quantile
        below = (y_rep < mu).mean()
        assert below == pytest.approx(0.3, abs=0.005)
        return
    # mean of the predictive draws = mean of the expected outcome
    se = y_rep.std(axis=0) / np.sqrt(20000)
    assert np.all(np.abs(y_rep.mean(axis=0) - mu.mean(axis=0)) < 5 * se + 1e-9)
    if model in ("logit", "probit"):
        # a 0 / 1 draw is determined by its mean
        assert set(np.unique(y_rep)) <= {0.0, 1.0}
        return
    # variance by the law of total variance
    if model == "normal":
        within = d[:, -1].mean()
    elif model == "t":
        within = (d[:, -1] * 4 / (4 - 2)).mean()
    elif model == "poisson":
        within = mu.mean(axis=0)
    else:
        within = (mu + d[:, [-1]] * mu**2).mean(axis=0)
    total = within + mu.var(axis=0)
    rel = 0.12 if model in ("t", "negbin") else 0.06
    np.testing.assert_allclose(y_rep.var(axis=0), total, rtol=rel)


def test_ordered_and_multinomial_predictions_follow_the_probabilities(fits, data):
    for model in ("oprobit", "mlogit"):
        fit = fits[model]
        y_rep = fit.posterior_predict(seed=5)
        probs = fit.predict(what="probabilities").to_numpy()
        for level in range(3):
            share = (y_rep == level).mean(axis=0)
            assert np.max(np.abs(share - probs[:, level])) < 0.08


def test_tobit_predictions_are_censored(fits):
    y_rep = fits["tobit"].posterior_predict(seed=5)
    assert y_rep.min() == 0.0
    assert (y_rep == 0.0).mean() > 0.05
