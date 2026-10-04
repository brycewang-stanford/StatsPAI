"""Standard errors and the uniform band of ``sp.cate_eval``'s TOC curve.

The CATE-evaluation notebook of *Applied Causal Inference Powered by ML
and AI* draws the targeting-operator-characteristic curve with pointwise
and uniform bands. Its influence function takes the ranking thresholds
from a separate sample. ``sp.cate_eval`` ranks within the evaluation
sample, as ``grf`` does, so the influence function has a rank term.

No reference package reports these standard errors (``grf`` bootstraps
the scalar RATE only), so the evidence is a known-truth simulation:

* the TOC values themselves are ``grf``'s (pinned elsewhere at 1e-10);
* the average standard error equals the Monte Carlo standard deviation of
  the estimate within 12% at five points of the curve (300 replications:
  the sd itself has a relative standard error of 4%);
* pointwise 95% intervals cover the true curve between 0.91 and 0.985
  (binomial 3-sd band 0.038), the uniform band covers the whole curve at
  least 93% of the time;
* without the rank term the standard error at q = 0.1 is more than 10%
  larger.

References
----------
[@yadlowsky2025evaluating]
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp
from statspai.forest.forest_inference import _tie_averaged_sorted

Q_GRID = 20  # q = 0.05, 0.10, ..., 1.00
CHECK = [1, 5, 9, 15, 18]  # q = 0.10, 0.30, 0.50, 0.80, 0.95


def _draw(seed: int, n: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    e = 1.0 / (1.0 + np.exp(-0.5 * x[:, 1]))
    t = (rng.uniform(size=n) < e).astype(float)
    tau = 1.0 + x[:, 0]
    mu0 = x[:, 1]
    y = mu0 + t * tau + rng.normal(size=n)
    priority = tau + rng.normal(size=n)  # an imperfect CATE estimate
    return y, t, e, mu0, mu0 + tau, priority, tau


@pytest.fixture(scope="module")
def truth() -> np.ndarray:
    *_, priority, tau = _draw(0, 2_000_000)
    ordered = tau[np.argsort(-priority)]
    cum = np.cumsum(ordered)
    q = np.linspace(1 / Q_GRID, 1.0, Q_GRID)
    k = np.round(q * len(tau)).astype(int)
    return cum[k - 1] / k - ordered.mean()


@pytest.fixture(scope="module")
def replications(truth):
    est, se, lo, hi = [], [], [], []
    for r in range(300):
        y, t, e, m0, m1, priority, _ = _draw(100 + r, 2000)
        res = sp.cate_eval(
            priority, y, t, e_hat=e, mu1_hat=m1, mu0_hat=m0,
            clip=0.0, q_grid=Q_GRID, random_state=r,
        )  # fmt: skip
        curve = res.toc_curve
        est.append(curve["toc"].to_numpy())
        se.append(curve["se"].to_numpy())
        lo.append(curve["band_lower"].to_numpy())
        hi.append(curve["band_upper"].to_numpy())
    return tuple(np.array(a) for a in (est, se, lo, hi))


def test_standard_errors_match_the_sampling_spread(replications):
    est, se, _, _ = replications
    ratio = se.mean(axis=0)[CHECK] / est.std(axis=0, ddof=1)[CHECK]
    assert np.all(np.abs(ratio - 1.0) < 0.12), ratio


def test_pointwise_and_uniform_coverage(replications, truth):
    est, se, lo, hi = replications
    z = 1.959963984540054
    cover = (np.abs(est - truth) <= z * se)[:, CHECK].mean(axis=0)
    assert np.all((cover >= 0.91) & (cover <= 0.985)), cover
    whole = ((lo[:, :-1] <= truth[:-1]) & (truth[:-1] <= hi[:, :-1])).all(axis=1)
    assert whole.mean() >= 0.93
    pointwise_whole = (np.abs(est - truth) <= z * se)[:, :-1].all(axis=1)
    assert pointwise_whole.mean() < whole.mean()


def test_rank_term_matters_at_the_top_of_the_curve(replications):
    """The fixed-threshold influence function is too wide at q = 0.1."""
    _, se, _, _ = replications
    naive = []
    for r in range(300):
        y, t, e, m0, m1, priority, _ = _draw(100 + r, 2000)
        dr = (m1 - m0) + (t - e) * (y - t * m1 - (1 - t) * m0) / (e * (1 - e))
        s, _ = _tie_averaged_sorted(dr, priority)
        n, k = len(s), 200  # q = 0.1
        top = (np.arange(n) < k).astype(float)
        toc = s[:k].mean() - s.mean()
        psi = (s - s.mean()) * (top / (k / n) - 1.0) - toc
        naive.append(np.sqrt(np.mean(psi**2) / n))
    assert np.mean(naive) / se[:, 1].mean() > 1.10


def test_curve_ends_at_zero_and_the_band_is_wider():
    y, t, e, m0, m1, priority, _ = _draw(7, 1500)
    res = sp.cate_eval(priority, y, t, e_hat=e, mu1_hat=m1, mu0_hat=m0, clip=0.0)
    curve = res.toc_curve
    last = curve.iloc[-1]
    assert last["q"] == 1.0 and abs(last["toc"]) < 1e-12 and last["se"] == 0.0
    crit = curve.attrs["uniform_critical_value"]
    assert 1.96 < crit < 4.0
    inner = curve.iloc[:-1]
    assert (inner["band_lower"] < inner["ci_lower"]).all()
    assert (inner["band_lower"] > 0).any()  # heterogeneity is detected here


def test_m_hat_is_not_needed_with_arm_specific_means():
    y, t, e, m0, m1, priority, _ = _draw(8, 800)
    a = sp.cate_eval(priority, y, t, e_hat=e, mu1_hat=m1, mu0_hat=m0)
    b = sp.cate_eval(priority, y, t, e_hat=e, mu1_hat=m1, mu0_hat=m0, m_hat=m0 + 9.0)
    assert a.autoc == b.autoc and a.qini_se == b.qini_se
