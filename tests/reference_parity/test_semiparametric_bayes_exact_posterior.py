"""``sp.semiparametric_bayes`` against the exact posterior of a small model.

The augmented likelihood of Mueller and Norets is
``q(y | theta, delta) = c(theta, delta) k0(delta * ldot_d(y)) p(y | theta)``
with a normalising constant ``c`` that the sampler never computes: it
replaces ``c`` by the rejected proposals of an acceptance sampler (Rao, Lin
and Dunson). The check here computes ``c`` by quadrature instead.

Model: Weibull durations with an intercept only, ``theta = (b, log alpha)``
and a scalar ``delta``. Under the baseline ``w = y / lambda`` is a standard
Weibull(alpha) variable, so ``c`` depends on ``(alpha, delta)`` only and is
one Gauss-Legendre integral over ``w`` per grid cell. The posterior of
``(b, log alpha, delta)`` is integrated on a 3-d grid written from the
paper's formulas, sharing no code with the sampler.

``delta_prior_sd = 1`` lets ``delta`` reach the region where ``k0`` is
nonlinear. There ``c`` matters: the posterior that drops it differs from
the exact one by more than ten Monte Carlo standard errors in every
coordinate, so the test would catch a sampler that ignored the rejected
proposals. (With the paper's ``delta_prior_sd = 0.2`` the two agree to
Monte Carlo error, because ``E[ldot_d] = 0`` under the baseline makes
``log c`` of order ``delta^3``.)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special

import statspai as sp

N = 300
DELTA_SD = 1.0
PRIOR_SD = 10.0


def _cw(a):
    return np.exp(special.gammaln(1 + 2 / a) - 2 * special.gammaln(1 + 1 / a)) - 1


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(3)
    # log-normal durations: a Weibull baseline is misspecified, so the
    # posterior of delta moves well away from zero
    y = np.exp(0.3 + 1.5 * rng.normal(size=N))
    df = pd.DataFrame({"y": y})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.semiparametric_bayes(
            "y ~ 1",
            df,
            model="weibull",
            delta_prior_sd=DELTA_SD,
            draws=8000,
            burnin=2000,
            seed=11,
            baseline=False,
        )

    # grid fixed in advance (not read off the sampler); edge mass checked
    b_g = np.linspace(0.7, 2.6, 45)
    la_g = np.linspace(-0.6, -0.08, 45)
    d_g = np.linspace(-7.5, 2.5, 67)
    B, LA, D = np.meshgrid(b_g, la_g, d_g, indexing="ij")
    A = np.exp(LA)
    loglam = B - special.gammaln(1 + 1 / A)
    cA = _cw(A)
    with_c = np.zeros(B.shape)
    for yi in y:
        dd = np.log(yi) - loglam
        z = np.exp(A * dd)
        with_c += LA - loglam + (A - 1) * dd - z
        ldot = A * (z - 1) - (yi * np.exp(-B) - 1) / cA
        with_c += np.log(2.0) - np.logaddexp(0.0, -2.0 * D * ldot)
    # log E_w[k0(delta ldot_d(w))], w ~ Weibull(alpha), by u = F(w)
    x, wts = np.polynomial.legendre.leggauss(600)
    u, wts = (x + 1) / 2, wts / 2
    log_e = np.empty((la_g.size, d_g.size))
    for i, la in enumerate(la_g):
        a = np.exp(la)
        w = (-np.log1p(-u)) ** (1 / a)
        ld = a * (w**a - 1) - (w / special.gamma(1 + 1 / a) - 1) / _cw(a)
        for j, d in enumerate(d_g):
            log_e[i, j] = np.log(np.sum(wts * 2 * special.expit(2 * d * ld)))
    prior = -0.5 * (B / PRIOR_SD) ** 2 - 0.5 * (LA / PRIOR_SD) ** 2
    prior += -0.5 * (D / DELTA_SD) ** 2
    without_c = with_c + prior
    with_c = with_c - N * log_e[None, :, :] + prior
    return fit, (B, A, D), with_c, without_c


def _moments(logk, G):
    P = np.exp(logk - logk.max())
    P /= P.sum()
    mu = float((P * G).sum())
    return mu, float(np.sqrt((P * (G - mu) ** 2).sum())), P


def test_grid_covers_the_posterior(setup):
    _, (B, _, _), with_c, _ = setup
    _, _, P = _moments(with_c, B)
    edge = max(P[[0, -1]].sum(), P[:, [0, -1]].sum(), P[:, :, [0, -1]].sum())
    assert edge < 1e-5  # the grid truncates a negligible tail


@pytest.mark.parametrize(
    "name, idx", [("Intercept", 0), ("alpha", 1), ("delta:Intercept", 2)]
)
def test_sampler_matches_exact_posterior(setup, name, idx):
    fit, grids, with_c, _ = setup
    mu, sd, _ = _moments(with_c, grids[idx])
    row = fit.table.loc[name]
    # mean within four of the sampler's own Monte Carlo standard errors;
    # sd within 6 percent (ESS ~ 500-1000, so the sd itself is estimated to
    # about 3 percent)
    assert abs(row["mean"] - mu) < 4 * row["mcse"]
    assert abs(row["sd"] / sd - 1) < 0.06


def test_normalising_constant_matters_here(setup):
    """The configuration discriminates: dropping c moves every mean."""
    fit, grids, _, without_c = setup
    for name, idx in (("Intercept", 0), ("alpha", 1), ("delta:Intercept", 2)):
        mu, _, _ = _moments(without_c, grids[idx])
        row = fit.table.loc[name]
        assert abs(row["mean"] - mu) > 6 * row["mcse"]
