"""Probit systems and mixtures against posteriors computed another way.

``sp.bayes_mvprobit`` and ``sp.bayes_mnprobit``

* The joint-distribution test: alternate the sampler's sweep with a fresh
  draw of the latent variables and outcomes given the current parameters.
  The coefficients and the error covariance must then be distributed as
  their prior, whose moments are known.
* Importance sampling from the prior. With discrete regressors the
  likelihood is a product of a few cell probabilities, each a bivariate
  normal rectangle computed here by quadrature. Weighted prior draws give
  the posterior of the identified quantities without any Markov chain.

``sp.bayes_mixture``

* With eight observations every partition can be enumerated (4,140 of
  them). The posterior probability of each is the prior of the partition
  times the product of the components' marginal likelihoods, a
  multivariate t density from ``scipy``. The sampler must reproduce the
  distribution of the number of clusters, the co-clustering
  probabilities and, with a Gamma prior, the posterior mean of the
  concentration.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import statspai as sp
from statspai.mcmc import mprobit as MP
from statspai.mcmc._core import normal_prior

# --------------------------------------------------------------------------
# Probit systems
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["mv", "mn"])
def test_probit_sweep_leaves_the_prior_invariant(kind):
    rng = np.random.default_rng(1)
    n, m, G, burn = 12, 2, 50000, 1000
    x = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    Xl = [x, x]
    off = np.array([0, 2, 4])
    b0 = np.array([0.2, -0.3, 0.1, 0.4])
    B0 = 0.5
    _, _, A = normal_prior(4, b0, B0, list("abcd"))
    nu0 = 9.0
    V0 = np.eye(2) * 6.0 + 1.5 * (np.ones((2, 2)) - np.eye(2))
    prior = (A, A @ b0, nu0, V0)
    XX = [[Xl[a].T @ Xl[b] for b in range(m)] for a in range(m)]
    mean_sigma = V0 / (nu0 - m - 1)
    beta, Sigma = b0.copy(), mean_sigma.copy()
    out = np.empty((G, 11))
    for g in range(G):
        mu = np.column_stack([x @ beta[:2], x @ beta[2:]])
        w = mu + rng.standard_normal((n, m)) @ np.linalg.cholesky(Sigma).T
        if kind == "mv":
            y = (w > 0).astype(float)
        else:
            y = np.where(w.max(axis=1) < 0, 0, w.argmax(axis=1) + 1)
        w, beta, Sigma = MP._sweep(rng, w, beta, Sigma, y, Xl, XX, off, prior, kind)
        out[g] = np.r_[beta, Sigma[0, 0], Sigma[0, 1], Sigma[1, 1], beta**2]
    truth = np.r_[b0, mean_sigma[0, 0], mean_sigma[0, 1], mean_sigma[1, 1], B0 + b0**2]
    d = out[burn:]
    nb = 70
    L = len(d) // nb
    bm = d[: nb * L].reshape(nb, L, -1).mean(axis=1)
    z = (d.mean(axis=0) - truth) / (bm.std(axis=0, ddof=1) / np.sqrt(nb))
    assert np.abs(z).max() < 4.0, z.round(2)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def _binorm_cdf(a, b, rho):
    """P(Z1 < a, Z2 < b) for standard normals with correlation rho."""
    top = special.ndtr(a)
    u = 0.5 * top[..., None] * (_GL_X + 1.0)
    z = special.ndtri(np.clip(u, 1e-300, 1 - 1e-16))
    inner = special.ndtr(
        (b[..., None] - rho[..., None] * z) / np.sqrt(1 - rho[..., None] ** 2)
    )
    return 0.5 * top * (inner * _GL_W).sum(axis=-1)


def test_bivariate_normal_quadrature():
    rng = np.random.default_rng(0)
    a, b, r = rng.normal(size=8), rng.normal(size=8), rng.uniform(-0.9, 0.9, 8)
    exact = [
        stats.multivariate_normal([0, 0], [[1, rr], [rr, 1]]).cdf([aa, bb])
        for aa, bb, rr in zip(a, b, r)
    ]
    # scipy's own routine is accurate to about 1e-5 here
    assert np.abs(_binorm_cdf(a, b, r) - exact).max() < 2e-5


def _compare(fit, logw, funcs, sd_tol):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    assert 1.0 / (w**2).sum() > 1500  # the importance sample is informative
    for name, f in funcs.items():
        mean = w @ f
        sd = np.sqrt(w @ (f - mean) ** 2)
        se = np.sqrt((w**2) @ (f - mean) ** 2)
        row = fit.table.loc[name]
        z = (row["mean"] - mean) / np.hypot(se, row["mcse"])
        assert abs(z) < 4.0, (name, round(z, 2))
        assert row["sd"] == pytest.approx(sd, rel=sd_tol), name


N_IS = 400000
B0, NU0 = 1.0, 6.0
V0 = 6.0 * np.eye(2)


def test_mvprobit_against_importance_sampling():
    rng = np.random.default_rng(5)
    n = 40
    x = (np.arange(n) % 2).astype(float)
    e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "y1": (0.2 + 0.6 * x + e[:, 0] > 0) * 1,
            "y2": (-0.1 + e[:, 1] > 0) * 1,
        }
    )
    beta = rng.normal(0, np.sqrt(B0), size=(N_IS, 3))
    Sig = stats.invwishart(NU0, V0).rvs(N_IS, random_state=rng)
    s1, s2 = np.sqrt(Sig[:, 0, 0]), np.sqrt(Sig[:, 1, 1])
    rho = Sig[:, 0, 1] / (s1 * s2)
    logw = np.zeros(N_IS)
    for xv in (0, 1):
        m1 = (beta[:, 0] + beta[:, 1] * xv) / s1
        m2 = beta[:, 2] / s2
        for y1 in (0, 1):
            for y2 in (0, 1):
                c = int(((df.x == xv) & (df.y1 == y1) & (df.y2 == y2)).sum())
                q1, q2 = 2 * y1 - 1, 2 * y2 - 1
                logw += c * np.log(_binorm_cdf(q1 * m1, q2 * m2, q1 * q2 * rho))
    funcs = {
        "y1:Intercept": beta[:, 0] / s1,
        "y1:x": beta[:, 1] / s1,
        "y2:Intercept": beta[:, 2] / s2,
        "corr(y1,y2)": rho,
    }
    fit = sp.bayes_mvprobit(
        ["y1 ~ x", "y2 ~ 1"],
        df,
        prior_var=B0,
        sigma_prior=(NU0, V0),
        draws=40000,
        burnin=2000,
        seed=3,
    )
    _compare(fit, logw, funcs, sd_tol=0.05)


def test_mnprobit_against_importance_sampling():
    rng = np.random.default_rng(6)
    n = 45
    x = (np.arange(n) % 2).astype(float)
    u = np.column_stack([0.3 + 0.5 * x, -0.2 + 0.4 * x]) + rng.multivariate_normal(
        [0, 0], [[1, 0.4], [0.4, 1]], size=n
    )
    ch = np.where(u.max(axis=1) < 0, 0, u.argmax(axis=1) + 1)
    df = pd.DataFrame({"x": x, "choice": np.array(["a", "b", "c"])[ch]})
    beta = rng.normal(0, np.sqrt(B0), size=(N_IS, 4))
    Sig = stats.invwishart(NU0, V0).rvs(N_IS, random_state=rng)
    s11, s22, s12 = Sig[:, 0, 0], Sig[:, 1, 1], Sig[:, 0, 1]
    logw = np.zeros(N_IS)
    for xv in (0, 1):
        mu1 = beta[:, 0] + beta[:, 1] * xv
        mu2 = beta[:, 2] + beta[:, 3] * xv
        # base: both utility differences negative
        p0 = _binorm_cdf(
            -mu1 / np.sqrt(s11), -mu2 / np.sqrt(s22), s12 / np.sqrt(s11 * s22)
        )
        # first alternative: w1 > 0 and w1 > w2
        vd = s11 + s22 - 2 * s12
        p1 = _binorm_cdf(
            mu1 / np.sqrt(s11),
            (mu1 - mu2) / np.sqrt(vd),
            (s11 - s12) / np.sqrt(s11 * vd),
        )
        p2 = np.clip(1 - p0 - p1, 1e-300, None)
        for lev, p in enumerate((p0, p1, p2)):
            logw += int(((x == xv) & (ch == lev)).sum()) * np.log(p)
    sc = np.sqrt(s11)
    funcs = {
        "b:Intercept": beta[:, 0] / sc,
        "b:x": beta[:, 1] / sc,
        "c:Intercept": beta[:, 2] / sc,
        "c:x": beta[:, 3] / sc,
        "var(c)": s22 / s11,
        "cov(b,c)": s12 / s11,
    }
    fit = sp.bayes_mnprobit(
        "choice ~ x",
        df,
        prior_var=B0,
        sigma_prior=(NU0, V0),
        draws=80000,
        burnin=3000,
        seed=4,
    )
    # the relative variance has a heavy right tail under both methods
    _compare(fit, logw, funcs, sd_tol=0.10)


# --------------------------------------------------------------------------
# Mixtures
# --------------------------------------------------------------------------


def _partitions(n):
    def rec(prefix, top):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for v in range(top + 2):
            yield from rec(prefix + [v], max(top, v))

    yield from rec([0], 0)


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(2)
    n = 8
    x = rng.normal(size=n)
    y = np.r_[rng.normal(-1.5, 0.6, 4), rng.normal(1.5, 0.6, 4)] + 0.5 * x
    X = np.column_stack([np.ones(n), x])
    b0 = np.array([0.0, 0.3])
    B0m = np.array([[4.0, 0.5], [0.5, 2.0]])
    a0, d0 = 5.0, 3.0
    cache = {}

    def logm(idx):
        if idx not in cache:
            Xs = X[list(idx)]
            shape = (d0 / a0) * (np.eye(len(idx)) + Xs @ B0m @ Xs.T)
            cache[idx] = stats.multivariate_t(loc=Xs @ b0, shape=shape, df=a0).logpdf(
                y[list(idx)]
            )
        return cache[idx]

    parts = list(_partitions(n))
    blocks = [
        [tuple(i for i in range(n) if p[i] == c) for c in range(max(p) + 1)]
        for p in parts
    ]
    like = np.array([sum(logm(b) for b in bl) for bl in blocks])
    sizes = np.array([sum(math.lgamma(len(b)) for b in bl) for bl in blocks])
    ks = np.array([len(bl) for bl in blocks])
    return dict(
        df=pd.DataFrame({"y": y, "x": x}),
        n=n,
        parts=np.array(parts),
        blocks=blocks,
        like=like,
        sizes=sizes,
        ks=ks,
        prior=dict(prior_mean=b0, prior_scale=B0m, sigma2_prior=(a0, d0)),
    )


def _exact(tiny, logw):
    w = np.exp(logw - logw[np.isfinite(logw)].max())
    w /= w.sum()
    pk = np.array([w[tiny["ks"] == k].sum() for k in range(1, tiny["n"] + 1)])
    P = tiny["parts"]
    co = np.einsum("p,pij->ij", w, (P[:, :, None] == P[:, None, :]).astype(float))
    return pk, co


def _check_partition_posterior(fit, pk, co, n):
    got = fit.model_info["n_clusters"].reindex(range(1, n + 1), fill_value=0.0)
    # 40,000 thinned draws: binomial error at most 0.0025 if independent
    assert np.abs(got.to_numpy() - pk).max() < 0.012
    row = fit.table.loc["n_clusters"]
    if row["sd"] > 0:
        assert abs(row["mean"] - pk @ np.arange(1, n + 1)) < 4 * row["mcse"]
    # 4,000 partitions enter the similarity matrix
    assert np.abs(fit.model_info["similarity"] - co).max() < 0.04


def test_dirichlet_process_mixture_by_enumeration(tiny):
    alpha = 0.8
    logw = tiny["ks"] * math.log(alpha) + tiny["sizes"] + tiny["like"]
    pk, co = _exact(tiny, logw)
    fit = sp.bayes_mixture(
        "y ~ x",
        tiny["df"],
        components="dp",
        alpha=alpha,
        draws=40000,
        burnin=500,
        thin=2,
        seed=1,
        **tiny["prior"],
    )
    _check_partition_posterior(fit, pk, co, tiny["n"])


def test_finite_mixture_by_enumeration(tiny):
    e0, K = 0.7, 3
    logw = np.full(len(tiny["ks"]), -np.inf)
    for j, bl in enumerate(tiny["blocks"]):
        k = len(bl)
        if k <= K:
            # labelled allocations that give this partition: K! / (K - k)!
            logw[j] = (
                math.log(math.factorial(K) / math.factorial(K - k))
                + sum(math.lgamma(len(b) + e0) for b in bl)
                + (K - k) * math.lgamma(e0)
            )
    pk, co = _exact(tiny, logw + tiny["like"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        fit = sp.bayes_mixture(
            "y ~ x",
            tiny["df"],
            components=K,
            alpha=e0,
            draws=40000,
            burnin=500,
            thin=2,
            seed=1,
            **tiny["prior"],
        )
    _check_partition_posterior(fit, pk, co, tiny["n"])


def test_dirichlet_process_concentration_by_enumeration(tiny):
    sh, rt, n = 2.0, 2.0, tiny["n"]
    ag = np.linspace(0.005, 14.0, 2800)
    log_alpha = (
        stats.gamma(sh, scale=1 / rt).logpdf(ag)
        + special.gammaln(ag)
        - special.gammaln(ag + n)
    )
    joint = (
        (tiny["sizes"] + tiny["like"])[:, None]
        + tiny["ks"][:, None] * np.log(ag)[None, :]
        + log_alpha[None, :]
    )
    w = np.exp(joint - joint.max())
    w /= w.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        fit = sp.bayes_mixture(
            "y ~ x",
            tiny["df"],
            components="dp",
            alpha_prior=(sh, rt),
            draws=40000,
            burnin=500,
            thin=2,
            seed=1,
            **tiny["prior"],
        )
    t = fit.table
    assert abs(t.loc["alpha", "mean"] - w.sum(axis=0) @ ag) < 4 * t.loc["alpha", "mcse"]
    assert (
        abs(t.loc["n_clusters", "mean"] - w.sum(axis=1) @ tiny["ks"])
        < 4 * t.loc["n_clusters", "mcse"]
    )


def test_mixture_density_and_components_recover_a_known_mixture():
    rng = np.random.default_rng(0)
    y = np.r_[
        rng.normal(-2, 0.5, 150), rng.normal(1, 1.0, 250), rng.normal(5, 0.4, 100)
    ]
    df = pd.DataFrame({"y": y})
    fit = sp.bayes_mixture("y ~ 1", df, components=3, draws=1500, burnin=500, seed=1)
    p = fit.params
    assert p[["weight[1]", "weight[2]", "weight[3]"]].to_numpy() == pytest.approx(
        [0.3, 0.5, 0.2], abs=0.03
    )
    got = p[["comp1:Intercept", "comp2:Intercept", "comp3:Intercept"]].to_numpy()
    assert got == pytest.approx([-2, 1, 5], abs=0.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        dpm = sp.bayes_mixture(
            "y ~ 1", df, components="dp", draws=1500, burnin=500, seed=1
        )
    d = dpm.model_info["density"]
    true = (
        0.3 * stats.norm(-2, 0.5).pdf(d.y)
        + 0.5 * stats.norm(1, 1).pdf(d.y)
        + 0.2 * stats.norm(5, 0.4).pdf(d.y)
    )
    step = d.y.iloc[1] - d.y.iloc[0]
    assert d.density.sum() * step == pytest.approx(1.0, abs=0.01)
    assert np.abs(d.density - true).sum() * step < 0.15
    # the partition recovers the three groups: share of pairs classified
    # alike (the Rand index). The components overlap, so it is not 1.
    truth = np.repeat([1, 2, 3], [150, 250, 100])
    c = dpm.model_info["cluster"].to_numpy()
    same = (truth[:, None] == truth[None, :]) == (c[:, None] == c[None, :])
    assert same.mean() > 0.93
