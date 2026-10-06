"""``sp.gp_regress``, ``sp.abc`` and ``sp.bart`` against known answers.

Gaussian process regression

* At fixed hyperparameters and a fixed mean, the posterior mean, standard
  deviation and log marginal likelihood against scikit-learn (1e-10).
* With the constant estimated, against the textbook formulas written
  with explicit inverses.
* The optimiser against scikit-learn's best of nine starts.

Approximate Bayesian computation

* Rejection with a fixed tolerance on a normal mean: the accepted draws
  are an independent sample from ``prior * P(|s - s_obs| < tol | theta)``,
  which is computed on a grid.
* The regression adjustment is exact when parameter and summary are
  jointly normal: at a tolerance where plain rejection more than doubles
  the posterior standard deviation, the adjusted draws have the exact
  posterior mean and standard deviation.
* Synthetic likelihood with sufficient, nearly normal summaries against
  the exact posterior on a grid.

BART

* One tree, one regressor, two cutpoints: five possible trees. Their
  posterior probabilities, the posterior mean of the error variance and
  of the fit are integrated exactly (the error variance on a grid).
* The joint-distribution test for a sum of three trees: redraw the data
  from the current state each sweep, and the trees must follow their
  prior, which a separate recursive generator written here simulates.
  Run with and without a minimum leaf size.
* A screen against five runs of R ``dbarts`` on a committed sample.
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
from statspai.mcmc import gp as GP
from statspai.mcmc import trees as B

FIX = Path(__file__).parent / "_fixtures"

# --------------------------------------------------------------------------
# Gaussian process regression
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gp_data():
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 2)) * [1.0, 3.0]
    y = np.sin(X[:, 0]) + 0.1 * X[:, 1] + 0.2 * rng.normal(size=n)
    df = pd.DataFrame({"y": y, "a": X[:, 0], "b": X[:, 1]})
    new = pd.DataFrame({"a": rng.normal(size=7), "b": rng.normal(size=7) * 3})
    return X, y, df, new


LS, SF, SN = np.array([0.8, 2.5]), 1.3, 0.07


@pytest.mark.parametrize("kind", ["rbf", "matern32", "matern52"])
def test_gp_matches_scikit_learn_at_fixed_hyperparameters(gp_data, kind):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    X, y, df, new = gp_data
    base = {
        "rbf": RBF(LS),
        "matern32": Matern(LS, nu=1.5),
        "matern52": Matern(LS, nu=2.5),
    }
    sk = GaussianProcessRegressor(
        ConstantKernel(SF) * base[kind] + WhiteKernel(SN), optimizer=None, alpha=0.0
    ).fit(X, y)
    mu, sd = sk.predict(new[["a", "b"]].to_numpy(), return_std=True)
    fit = sp.gp_regress(
        "y ~ a + b",
        df,
        kernel=kind,
        length_scale=LS,
        signal_var=SF,
        noise_var=SN,
        mean=0.0,
        optimize_hyper=False,
    )
    pred = fit.predict(new, noise=True)
    # two Cholesky solves of the same 60 x 60 system
    assert np.abs(pred["mean"] - mu).max() < 1e-10
    assert np.abs(pred["sd"] - sd).max() < 1e-10
    assert fit.loglik == pytest.approx(sk.log_marginal_likelihood_value_, abs=1e-9)


def test_gp_estimated_constant_is_generalised_least_squares(gp_data):
    X, y, df, new = gp_data
    n = len(y)
    K = GP._kernel("rbf", X, X, LS, SF) + SN * np.eye(n)
    Ki = np.linalg.inv(K)
    one = np.ones(n)
    m = one @ Ki @ y / (one @ Ki @ one)
    Xs = new[["a", "b"]].to_numpy()
    ks = GP._kernel("rbf", Xs, X, LS, SF)
    mu = m + ks @ Ki @ (y - m)
    var = (
        SF
        - np.einsum("ij,jk,ik->i", ks, Ki, ks)
        + (1 - ks @ Ki @ one) ** 2 / (one @ Ki @ one)
    )
    fit = sp.gp_regress(
        "y ~ a + b",
        df,
        length_scale=LS,
        signal_var=SF,
        noise_var=SN,
        optimize_hyper=False,
    )
    pred = fit.predict(new)
    assert fit.params["mean"] == pytest.approx(m, abs=1e-10)
    assert np.abs(pred["mean"] - mu).max() < 1e-10
    assert np.abs(pred["sd"] - np.sqrt(var)).max() < 1e-9
    # the restricted likelihood, written out
    r = y - m
    reml = (
        -0.5 * r @ Ki @ r
        - 0.5 * np.linalg.slogdet(K)[1]
        - 0.5 * np.log(one @ Ki @ one)
        - 0.5 * (n - 1) * np.log(2 * np.pi)
    )
    assert fit.loglik == pytest.approx(reml, abs=1e-9)


def test_gp_optimiser_finds_scikit_learns_maximum(gp_data):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    X, y, df, _ = gp_data
    sk = GaussianProcessRegressor(
        ConstantKernel(1.0) * RBF([1.0, 1.0]) + WhiteKernel(0.1),
        n_restarts_optimizer=8,
        random_state=0,
    ).fit(X, y)
    fit = sp.gp_regress("y ~ a + b", df, mean=0.0, seed=1)
    # both are numerical maxima of the same function
    assert fit.loglik == pytest.approx(sk.log_marginal_likelihood_value_, abs=1e-5)


def test_gp_bands_cover_a_known_function():
    cover = []
    for r in range(40):
        rng = np.random.default_rng(100 + r)
        x = np.sort(rng.uniform(0, 6, 120))
        y = np.sin(1.5 * x) + 0.3 * rng.normal(size=120)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sp.ConvergenceWarning)
            fit = sp.gp_regress(
                "y ~ x", pd.DataFrame({"x": x, "y": y}), restarts=2, seed=r
            )
        g = np.linspace(0.3, 5.7, 40)
        band = fit.predict(pd.DataFrame({"x": g}))
        truth = np.sin(1.5 * g)
        cover.append(((band.lower <= truth) & (truth <= band.upper)).mean())
    # 1,600 correlated points: nominal 0.95
    assert 0.90 < np.mean(cover) < 0.99


# --------------------------------------------------------------------------
# Approximate Bayesian computation
# --------------------------------------------------------------------------

N_OBS, S_OBS, PRIOR_SD = 20, 0.7, 2.0


def _sim_mean(theta, rng):
    return rng.normal(theta[:, 0], 1 / np.sqrt(N_OBS))[:, None]


def test_abc_rejection_samples_the_tolerance_posterior():
    eps = 0.3
    g = np.linspace(-3, 4, 20001)
    dens = stats.norm(0, PRIOR_SD).pdf(g) * (
        stats.norm.cdf((S_OBS + eps - g) * np.sqrt(N_OBS))
        - stats.norm.cdf((S_OBS - eps - g) * np.sqrt(N_OBS))
    )
    dens /= dens.sum()
    mean = dens @ g
    sd = np.sqrt(dens @ (g - mean) ** 2)
    fit = sp.abc(
        _sim_mean,
        [S_OBS],
        [stats.norm(0, PRIOR_SD)],
        n_sim=300000,
        tol=eps,
        scale=None,
        vectorized=True,
        seed=1,
    )
    row = fit.table.loc["theta1"]
    assert abs(row["mean"] - mean) < 4 * row["mcse"]
    assert row["sd"] == pytest.approx(sd, rel=0.02)
    cdf = np.cumsum(dens)
    ks = stats.kstest(fit.draws["theta1"], lambda v: np.interp(v, g, cdf))
    assert ks.pvalue > 0.001
    # the tolerance posterior is wider than the posterior given the data
    assert sd > np.sqrt(1 / (1 / PRIOR_SD**2 + N_OBS))


def test_abc_linear_adjustment_is_exact_for_a_normal_model():
    post_var = 1 / (1 / PRIOR_SD**2 + N_OBS)
    post_mean = post_var * N_OBS * S_OBS
    kw = dict(n_sim=300000, quantile=0.3, vectorized=True, seed=2)
    plain = sp.abc(_sim_mean, [S_OBS], [stats.norm(0, PRIOR_SD)], **kw)
    fit = sp.abc(_sim_mean, [S_OBS], [stats.norm(0, PRIOR_SD)], adjust="linear", **kw)
    assert plain.table.loc["theta1", "sd"] > 2 * np.sqrt(post_var)
    row = fit.table.loc["theta1"]
    assert abs(row["mean"] - post_mean) < 4 * row["mcse"]
    assert row["sd"] == pytest.approx(np.sqrt(post_var), rel=0.02)


def test_synthetic_likelihood_against_the_exact_posterior():
    rng = np.random.default_rng(3)
    y = rng.normal(1.0, 1.5, size=40)
    s_obs = [y.mean(), np.log(y.var(ddof=1))]

    def simulate(theta, rng):
        z = rng.normal(theta[:, [0]], np.exp(theta[:, [1]]), size=(len(theta), 40))
        return np.column_stack([z.mean(axis=1), np.log(z.var(axis=1, ddof=1))])

    priors = [stats.norm(0, 3), stats.norm(0, 1)]
    fit = sp.abc(
        simulate,
        s_obs,
        priors,
        method="synthetic",
        n_synthetic=200,
        draws=12000,
        burnin=2000,
        vectorized=True,
        names=["mu", "log_sd"],
        seed=4,
    )
    mu = np.linspace(-1, 3, 401)
    ls = np.linspace(-0.6, 1.2, 401)
    MU, LSG = np.meshgrid(mu, ls, indexing="ij")
    ll = (
        -40 * LSG
        - 0.5 * ((y[None, None, :] - MU[..., None]) ** 2).sum(axis=-1) / np.exp(2 * LSG)
        + priors[0].logpdf(MU)
        + priors[1].logpdf(LSG)
    )
    w = np.exp(ll - ll.max())
    w /= w.sum()
    for name, axis, grid in (("mu", 1, mu), ("log_sd", 0, ls)):
        marg = w.sum(axis=axis)
        mean = marg @ grid
        sd = np.sqrt(marg @ (grid - mean) ** 2)
        row = fit.table.loc[name]
        # the log variance is only approximately normal, and the mean and
        # covariance of the summaries are estimated from 200 simulations:
        # a tenth of a posterior standard deviation covers both
        assert abs(row["mean"] - mean) < 4 * row["mcse"] + 0.1 * sd, name
        assert row["sd"] == pytest.approx(sd, rel=0.08), name


# --------------------------------------------------------------------------
# BART
# --------------------------------------------------------------------------


def _batch_z(draws, truth, nb=60):
    L = len(draws) // nb
    bm = draws[: nb * L].reshape(nb, L, -1).mean(axis=1)
    return (draws.mean(axis=0) - truth) / (bm.std(axis=0, ddof=1) / np.sqrt(nb))


def test_single_tree_posterior_by_enumeration():
    rng = np.random.default_rng(1)
    x = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    n = len(x)
    y = np.array([-0.4, -0.1, -0.5, 0.1, 0.3, 0.0, 0.5, 0.2, 0.6, 0.4])
    y = y + 0.1 * rng.normal(size=n)
    tau2, base, power, nu, lam = 0.3**2, 0.8, 1.0, 4.0, 0.05
    ps0, ps1 = base, base * 2.0**-power
    # the five trees: their leaves as sets of x values, and their prior
    trees = [
        ([[0, 1, 2]], 1 - ps0),
        ([[0], [1, 2]], ps0 * 0.5 * (1 - ps1)),
        ([[0, 1], [2]], ps0 * 0.5 * (1 - ps1)),
        ([[0], [1], [2]], ps0 * 0.5 * ps1),  # cut 0, then cut 1 on the right
        ([[0], [1], [2]], ps0 * 0.5 * ps1),  # cut 1, then cut 0 on the left
    ]
    s2g = np.exp(np.linspace(np.log(0.003), np.log(1.5), 1200))
    lp = np.empty((5, s2g.size))
    fit = np.empty((5, s2g.size, n))
    for a, (groups, prior) in enumerate(trees):
        for b, s2 in enumerate(s2g):
            ll = 0.0
            for gset in groups:
                idx = np.isin(x, gset)
                k = int(idx.sum())
                cov = s2 * np.eye(k) + tau2 * np.ones((k, k))
                ll += stats.multivariate_normal(np.zeros(k), cov).logpdf(y[idx])
                fit[a, b, idx] = tau2 * y[idx].sum() / (s2 + k * tau2)
            lp[a, b] = (
                np.log(prior)
                + ll
                + stats.invgamma(nu / 2, scale=nu * lam / 2).logpdf(s2)
                + np.log(s2)
            )
    w = np.exp(lp - lp.max())
    w /= w.sum()
    truth = np.r_[w.sum(axis=1), (w * s2g).sum(), np.einsum("ab,abi->i", w, fit)]

    forest = B._Forest(x[:, None], np.array([2]), 1, tau2, base, power, 4)
    forest.k["seed"](123)
    r = np.random.default_rng(5)
    G, burn = 160000, 1000
    out = np.zeros((G, 6 + n))
    s2 = 0.1
    for g in range(G):
        forest.sweep(y, s2)
        s2 = (nu * lam + ((y - forest.total) ** 2).sum()) / r.chisquare(nu + n)
        v, c = forest.var[0], forest.cut[0, 0]
        if v[0] < 0:
            key = 0
        elif c == 0:
            key = 1 if v[2] < 0 else 3
        else:
            key = 2 if v[1] < 0 else 4
        out[g, key] = 1.0
        out[g, 5] = s2
        out[g, 6:] = forest.total
    z = _batch_z(out[burn:], truth)
    assert np.abs(z).max() < 4.0, z.round(2)
    assert np.abs(out[burn:, :5].mean(axis=0) - truth[:5]).max() < 0.02


def _prior_tree(rs, ncut, base, power, maxdepth):
    """Leaves (boxes of cut indices) of one tree drawn from the prior."""
    leaves = []

    def rec(box, d):
        avail = [v for v in range(len(ncut)) if box[v][1] > box[v][0]]
        if d < maxdepth and avail and rs.random() < base * (1 + d) ** -power:
            v = avail[rs.integers(len(avail))]
            c = rs.integers(box[v][0], box[v][1])
            left, right = list(box), list(box)
            left[v] = (box[v][0], c)
            right[v] = (c + 1, box[v][1])
            rec(left, d + 1)
            rec(right, d + 1)
        else:
            leaves.append(box)

    rec([(0, k) for k in ncut], 0)
    return leaves


@pytest.mark.parametrize("min_leaf", [0, 2])
def test_bart_sweep_leaves_the_prior_invariant(min_leaf):
    rs = np.random.default_rng(7)
    n = 12
    xc = np.column_stack([rs.integers(0, 4, n), rs.integers(0, 3, n)]).astype(np.int64)
    ncut = np.array([3, 2])
    m, tau2, base, power, md, nu, lam = 3, 0.25, 0.9, 1.2, 3, 6.0, 0.4

    def leaf_of(L, i):
        return next(
            k
            for k, b in enumerate(L)
            if all(b[v][0] <= xc[i, v] <= b[v][1] for v in range(2))
        )

    sizes, same = [], []
    while len(sizes) < 40000:
        L = _prior_tree(rs, ncut, base, power, md)
        where = [leaf_of(L, i) for i in range(n)]
        # the minimum leaf size truncates the prior to trees that respect it
        if min_leaf and np.bincount(where, minlength=len(L)).min() < min_leaf:
            continue
        sizes.append(len(L))
        same.append(where[0] == where[n - 1])
    sizes, same = np.array(sizes), np.array(same)
    truth = np.array(
        [
            sizes.mean(),
            (sizes == 1).mean(),
            0.0,
            m * tau2,
            m * tau2 * same.mean(),
            nu * lam / (nu - 2),
        ]
    )
    forest = B._Forest(xc, ncut, m, tau2, base, power, md, min_leaf)
    forest.k["seed"](99)
    r = np.random.default_rng(8)
    G, burn = 200000, 2000
    out = np.empty((G, 6))
    s2 = 0.3
    for g in range(G):
        y = forest.total + np.sqrt(s2) * r.standard_normal(n)
        forest.sweep(y, s2)
        s2 = (nu * lam + ((y - forest.total) ** 2).sum()) / r.chisquare(nu + n)
        t = forest.total
        out[g] = [
            forest.nleaf.mean(),
            (forest.nleaf == 1).mean(),
            t[0],
            t[0] ** 2,
            t[0] * t[n - 1],
            s2,
        ]
    d = out[burn:]
    L = len(d) // 60
    bm = d[: 60 * L].reshape(60, L, -1).mean(axis=1)
    se = bm.std(axis=0, ddof=1) / np.sqrt(60)
    # the prior moments of the tree shape are themselves simulated
    p_root = truth[1]
    se_truth = np.array(
        [
            sizes.std() / np.sqrt(len(sizes)),
            np.sqrt(p_root * (1 - p_root) / len(sizes)),
            0.0,
            0.0,
            m * tau2 * same.std() / np.sqrt(len(sizes)),
            0.0,
        ]
    )
    z = (d.mean(axis=0) - truth) / np.hypot(se, se_truth)
    assert np.abs(z).max() < 4.0, z.round(2)


def test_bart_screen_against_dbarts():
    ref = json.loads((FIX / "bart_R.json").read_text(encoding="utf-8"))["runs"]
    d = pd.read_csv(FIX / "bart_friedman.csv")
    tr, te = d[d.test == 0], d[d.test == 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.ConvergenceWarning)
        fit = sp.bart(
            "y ~ x0 + x1 + x2 + x3 + x4 + x5", tr, draws=1000, burnin=1000, seed=3
        )
    pred = fit.predict(te)
    rmse = np.sqrt(np.mean((pred["mean"] - te.f) ** 2))
    cover = ((pred.lower <= te.f) & (te.f <= pred.upper)).mean()
    r_rmse = np.array([r["rmse"] for r in ref])
    r_sigma = np.array([r["sigma"] for r in ref])
    # a screen, not parity: dbarts' own seeds differ by 8 percent in RMSE
    assert rmse < 1.12 * r_rmse.mean()
    assert fit.params["sigma"] == pytest.approx(r_sigma.mean(), rel=0.12)
    assert cover > 0.93
    # the regressor that does not enter the function is used least
    assert fit.variable_importance.idxmin() == "x5"
