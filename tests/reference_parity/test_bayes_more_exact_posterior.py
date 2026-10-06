"""More samplers of ``statspai.mcmc`` against exact posteriors.

Same method as ``test_bayes_regress_exact_posterior.py``: the posterior of
a small model is integrated on a grid from an integrand written here, and
the sampler must reproduce its means (within four Monte Carlo standard
errors) and standard deviations (within 5 percent).

* ``sp.bayes_sur``: the inverse-Wishart prior integrates the error
  covariance out, leaving ``prior * |V0 + E'E|^{-(nu0 + n) / 2}`` over the
  four coefficients of two equations.
* ``sp.bayes_regress(model='mlogit')``: three categories, an intercept
  and a slope, four coefficients.
* ``sp.bayes_shrink``: with the flat intercept integrated out, the Laplace
  prior (lasso, fixed penalty) and the two-normal mixture (SSVS) are
  explicit marginal priors on two slopes. For SSVS the exact inclusion
  probability is the posterior mean of the slab's share of the mixture.
* ``inference='vb'``: the fixed point of the coordinate ascent, the
  evidence lower bound against the exact marginal likelihood.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import statspai as sp


def _grid(center, spread, points, width=7.0):
    axes = [
        np.linspace(c - width * s, c + width * s, m)
        for c, s, m in zip(center, spread, points)
    ]
    return np.array(list(itertools.product(*axes)))


def _moments(log_post, values):
    w = np.exp(log_post - log_post.max())
    w /= w.sum()
    mean = w @ values
    return w, mean, np.sqrt(w @ (values - mean) ** 2)


def _check(table, names, mean, sd, sd_tol=0.05):
    t = table.loc[names]
    z = (t["mean"].to_numpy() - mean) / t["mcse"].to_numpy()
    assert np.abs(z).max() < 4.0, dict(zip(names, z.round(2)))
    ratio = t["sd"].to_numpy() / sd
    assert np.abs(ratio - 1).max() < sd_tol, dict(zip(names, ratio.round(3)))


def test_sur_gibbs():
    rng = np.random.default_rng(6)
    n = 50
    x, z = rng.normal(size=n), rng.normal(size=n)
    e = rng.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 0.8]], size=n)
    df = pd.DataFrame(
        {"x": x, "z": z, "y1": 1 + 0.5 * x + e[:, 0], "y2": -1 + 0.8 * z + e[:, 1]}
    )
    pv, nu0, v0 = 10.0, 4.0, 0.5
    fit = sp.bayes_sur(
        ["y1 ~ x", "y2 ~ z"],
        df,
        prior_var=pv,
        sigma_prior=(nu0, v0),
        draws=60000,
        burnin=2000,
        seed=4,
    )
    names = ["y1:Intercept", "y1:x", "y2:Intercept", "y2:z"]
    d = fit.draws[names].to_numpy()
    g = _grid(d.mean(axis=0), d.std(axis=0), [41] * 4)
    one = np.ones(n)

    def cross(a1, a2):
        return sum(ci * cj * float(xi @ xj) for ci, xi in a1 for cj, xj in a2)

    e1 = [(1.0, df["y1"].to_numpy()), (-g[:, 0], one), (-g[:, 1], x)]
    e2 = [(1.0, df["y2"].to_numpy()), (-g[:, 2], one), (-g[:, 3], z)]
    s11, s12, s22 = cross(e1, e1), cross(e1, e2), cross(e2, e2)
    log_post = -0.5 * (nu0 + n) * np.log((v0 + s11) * (v0 + s22) - s12**2)
    log_post += stats.norm.logpdf(g, 0.0, np.sqrt(pv)).sum(axis=1)
    w, mean, sd = _moments(log_post, g)
    _check(fit.table, names, mean, sd)
    # E[Sigma | coefficients] = (V0 + E'E) / (nu0 + n - 3)
    for name, value in (
        ("var(y1)", w @ (v0 + s11)),
        ("cov(y1,y2)", w @ s12),
        ("var(y2)", w @ (v0 + s22)),
    ):
        exact = value / (nu0 + n - 3)
        allow = 4 * fit.table.loc[name, "mcse"] + 0.004 * abs(exact)
        assert abs(fit.params[name] - exact) < allow, name
    # the second equation's error helps the first: tighter than OLS alone
    alone = sp.bayes_regress("y1 ~ x", df, prior_var=pv, draws=20000, seed=1)
    assert fit.std_errors["y1:x"] < alone.std_errors["x"]


def test_multinomial_logit_metropolis():
    rng = np.random.default_rng(2)
    n = 60
    x = rng.normal(size=n)
    eta = np.column_stack([np.zeros(n), 0.3 + 0.8 * x, -0.4 - 0.6 * x])
    p = special.softmax(eta, axis=1)
    y = np.array([rng.choice(3, p=row) for row in p])
    df = pd.DataFrame({"y": y, "x": x})
    pv = 4.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            "y ~ x",
            df,
            model="mlogit",
            prior_var=pv,
            draws=30000,
            burnin=2000,
            thin=4,
            seed=5,
        )
    names = ["1:Intercept", "1:x", "2:Intercept", "2:x"]
    assert list(fit.params.index) == names
    d = fit.draws.to_numpy()
    g = _grid(d.mean(axis=0), d.std(axis=0), [37] * 4, width=6.5)
    e1 = g[:, [0]] + g[:, [1]] * x
    e2 = g[:, [2]] + g[:, [3]] * x
    lse = np.logaddexp(0.0, np.logaddexp(e1, e2))
    ll = (np.where(y == 1, e1, 0.0) + np.where(y == 2, e2, 0.0) - lse).sum(axis=1)
    _, mean, sd = _moments(ll + stats.norm.logpdf(g, 0.0, np.sqrt(pv)).sum(axis=1), g)
    _check(fit.table, names, mean, sd)
    probs = fit.predict(what="probabilities")
    assert np.allclose(probs.sum(axis=1), 1.0) and list(probs.columns) == [
        "0",
        "1",
        "2",
    ]


@pytest.fixture(scope="module")
def two_slopes():
    rng = np.random.default_rng(9)
    n = 40
    X = rng.normal(size=(n, 2))
    y = 0.5 + 0.9 * X[:, 0] + 0.08 * X[:, 1] + 0.7 * rng.normal(size=n)
    return pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "y": y})


def _centred_loglik(df, g):
    """Normal likelihood of (slopes, log sigma2) with the flat intercept
    integrated out: n - 1 degrees of freedom."""
    n = len(df)
    yc = (df["y"] - df["y"].mean()).to_numpy()
    Z = (df[["a", "b"]] - df[["a", "b"]].mean()).to_numpy()
    resid = yc[None, :] - g[:, :2] @ Z.T
    s2 = np.exp(g[:, 2])
    return -0.5 * (n - 1) * np.log(s2) - 0.5 * (resid**2).sum(axis=1) / s2


def test_bayesian_lasso_gibbs(two_slopes):
    a0, d0, lam = 3.0, 2.0, 1.5
    fit = sp.bayes_shrink(
        "y ~ a + b",
        two_slopes,
        prior="lasso",
        lam=lam,
        standardize=False,
        sigma2_prior=(a0, d0),
        draws=60000,
        burnin=2000,
        seed=3,
    )
    d = fit.draws
    co = np.column_stack([d["a"], d["b"], np.log(d["sigma2"])])
    g = _grid(co.mean(axis=0), co.std(axis=0), [81, 81, 61])
    s = np.exp(0.5 * g[:, 2])
    log_post = _centred_loglik(two_slopes, g)
    log_post += stats.laplace.logpdf(g[:, :2], 0.0, (s / lam)[:, None]).sum(axis=1)
    log_post += stats.invgamma.logpdf(s * s, a0 / 2, scale=d0 / 2) + g[:, 2]
    vals = np.column_stack([g[:, 0], g[:, 1], s * s])
    _, mean, sd = _moments(log_post, vals)
    _check(fit.table, ["a", "b", "sigma2"], mean, sd)
    # shrinkage: both slopes are pulled toward zero relative to OLS
    ols = sp.regress("y ~ a + b", two_slopes)
    assert abs(fit.params["a"]) < abs(float(ols.params["a"]))


def test_ssvs_gibbs_and_inclusion_probabilities(two_slopes):
    a0, d0, spike, slab, incl = 3.0, 2.0, 0.1, 1.0, 0.4
    fit = sp.bayes_shrink(
        "y ~ a + b",
        two_slopes,
        prior="ssvs",
        spike_sd=spike,
        slab_sd=slab,
        inclusion=incl,
        standardize=False,
        sigma2_prior=(a0, d0),
        draws=80000,
        burnin=2000,
        seed=3,
    )
    sy = two_slopes["y"].std(ddof=1)
    v0, v1 = (spike * sy) ** 2, (slab * sy) ** 2
    d = fit.draws
    co = np.column_stack([d["a"], d["b"], np.log(d["sigma2"])])
    g = _grid(co.mean(axis=0), co.std(axis=0), [141, 141, 41])
    slab_part = np.log(incl) + stats.norm.logpdf(g[:, :2], 0.0, np.sqrt(v1))
    spike_part = np.log1p(-incl) + stats.norm.logpdf(g[:, :2], 0.0, np.sqrt(v0))
    mix = np.logaddexp(slab_part, spike_part)
    s2 = np.exp(g[:, 2])
    log_post = _centred_loglik(two_slopes, g) + mix.sum(axis=1)
    log_post += stats.invgamma.logpdf(s2, a0 / 2, scale=d0 / 2) + g[:, 2]
    w, mean, sd = _moments(log_post, np.column_stack([g[:, 0], g[:, 1], s2]))
    _check(fit.table, ["a", "b", "sigma2"], mean, sd)
    pip = w @ np.exp(slab_part - mix)
    got = fit.table.loc[["a", "b"], "pip"].to_numpy()
    # a draw of the indicator is Bernoulli: se of the mean below 0.006 here
    assert np.abs(got - pip).max() < 0.02, (got, pip)
    assert got[0] > 0.95 and got[1] < 0.5


def test_variational_bayes_fixed_point_and_bound(two_slopes):
    kw = dict(prior_var=10.0, sigma2_prior=(3.0, 2.0))
    vb = sp.bayes_regress(
        "y ~ a + b", two_slopes, inference="vb", draws=2000, seed=1, **kw
    )
    info = vb.model_info
    y = two_slopes["y"].to_numpy()
    X = np.column_stack([np.ones(len(y)), two_slopes[["a", "b"]].to_numpy()])
    m, S = info["q_beta_mean"], info["q_beta_cov"]
    an, dn = 2 * info["q_sigma2_shape"], 2 * info["q_sigma2_rate"]
    r = an / dn
    # the coordinate-ascent equations hold at the solution
    assert np.allclose(S, np.linalg.inv(np.eye(3) / 10.0 + r * X.T @ X), rtol=1e-8)
    assert np.allclose(m, S @ (r * X.T @ y), rtol=1e-8)
    e = y - X @ m
    assert dn == pytest.approx(2.0 + e @ e + np.trace(X.T @ X @ S), rel=1e-8)
    assert an == 3.0 + len(y)
    # the bound is below the marginal likelihood, and tight here
    exact = sp.bayes_regress("y ~ a + b", two_slopes, draws=40000, seed=1, **kw)
    lml = exact.log_marginal_likelihood("chib")
    assert lml - 0.2 < info["elbo"] < lml + 0.01
    # means agree; mean-field standard deviations are not larger
    se = exact.std_errors
    assert np.all(
        np.abs(vb.params - exact.params)[["Intercept", "a", "b"]]
        < 0.05 * se[["Intercept", "a", "b"]]
    )
    assert np.all(vb.std_errors[["a", "b"]] < 1.02 * se[["a", "b"]])
    with pytest.raises(sp.MethodIncompatibility, match="lower bound"):
        vb.log_marginal_likelihood()
    with pytest.raises(sp.MethodIncompatibility, match="independent draws"):
        vb.diagnostics()
    with pytest.raises(sp.MethodIncompatibility, match="model='normal' only"):
        sp.bayes_regress("y ~ a", two_slopes, model="t", inference="vb")
