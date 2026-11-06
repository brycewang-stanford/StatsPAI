"""``sp.expected_instrument``: closed forms of the expected instrument.

``sp.expected_instrument`` estimates ``mu_i = E[z_i]`` by Monte Carlo over a
known shock-assignment process and reports a Monte Carlo SE. No R or Stata
command takes a user instrument function and a shock sampler, so the
evidence is known truth (T1), on designs where ``mu`` has a closed form:

* a Bernoulli share instrument (share of girls among a parent's children)
  has ``mu_i`` = mean of the relevant probabilities and
  ``Var(z_i) = sum p(1-p) / m_i^2``; the simulated ``mu`` converges to it at
  the ``1/sqrt(S)`` rate and the reported MC SE estimates ``sqrt(Var/S)``;
* fair coins give a constant ``mu`` = 0.5 (flagged);
* within-stratum permutation gives the stratum mean, and every draw keeps
  the per-stratum multiset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _share_design(seed, n=300):
    """Each adult has a disjoint set of m_i children with girl probs p_k.

    For a man the instrument is the share of girls, for a woman the share of
    boys, so mu_i = mean of q_k with q_k = p_k (men) or 1 - p_k (women), and
    Var(z_i) = sum_k p_k (1 - p_k) / m_i^2.
    """
    rng = np.random.default_rng(seed)
    m = rng.integers(3, 30, n)
    owner = np.repeat(np.arange(n), m)
    p = rng.uniform(0.3, 0.7, owner.size)
    female = rng.integers(0, 2, n)
    data = pd.DataFrame({"female": female, "m": m})

    def instrument_fn(d, sex):
        girls = np.bincount(owner, weights=sex, minlength=n) / m
        return np.where(d["female"].to_numpy() == 1, 1 - girls, girls)

    pbar = np.bincount(owner, weights=p, minlength=n) / m
    mu = np.where(female == 1, 1 - pbar, pbar)
    var = np.bincount(owner, weights=p * (1 - p), minlength=n) / m**2
    sampler = {"kind": "bernoulli", "p": p}
    return data, instrument_fn, sampler, mu, var


@pytest.mark.parametrize("S", [400, 6400])
def test_simulated_mu_converges_to_closed_form_at_mc_rate(S):
    data, fn, sampler, mu, var = _share_design(11)
    res = sp.expected_instrument(data, fn, sampler, n_draws=S, seed=S)
    zstat = (res.mu - mu) / np.sqrt(var / S)
    n = zstat.size
    # Rows have disjoint children, so the z-scores are independent and
    # approximately N(0, 1): mean square ~ chi2_n / n. 5 SDs of chi2_n / n.
    assert abs(np.mean(zstat**2) - 1.0) < 5 * np.sqrt(2.0 / n)
    assert abs(np.mean(zstat)) < 5 / np.sqrt(n)
    # The reported MC SE estimates sqrt(Var/S): relative error of an SD
    # estimate from S draws is ~ 1/sqrt(2S) (<= 3.6% at S=400); allow 15%.
    ratio = res.mc_se / np.sqrt(var / S)
    assert np.all(np.abs(ratio - 1) < 0.15)
    assert np.median(np.abs(ratio - 1)) < 3 / np.sqrt(2 * S)


def test_fair_coin_share_has_constant_mu():
    data, fn, _, _, _ = _share_design(12)
    n_child = int(data["m"].sum())
    res = sp.expected_instrument(
        data, fn, {"kind": "bernoulli", "p": 0.5, "size": n_child}, n_draws=800, seed=1
    )
    # Fair coins: mu = 0.5 whatever the exposure, and the diagnostic says so.
    assert np.all(np.abs(res.mu - 0.5) < 6 * res.mc_se)
    assert res.diagnostics["mu_constant"]


def test_permutation_sampler_mu_is_stratum_mean():
    """Permuting a vector within strata: E[z_i] = stratum mean; the simulation
    is within 6 MC SEs of it for every row."""
    rng = np.random.default_rng(5)
    g = np.repeat([0, 1, 2], [40, 25, 35])
    shocks = rng.gamma(2.0, 1.0, g.size) * (1 + g)
    data = pd.DataFrame({"g": g})
    res = sp.expected_instrument(
        data,
        lambda d, sh: sh,
        {"kind": "permute", "shocks": shocks, "strata": g},
        n_draws=2000,
        seed=2,
    )
    target = pd.Series(shocks).groupby(g).transform("mean").to_numpy()
    assert np.all(np.abs(res.mu - target) < 6 * res.mc_se)
    # Every draw is a within-stratum permutation: the multiset per stratum
    # is kept, so the stratum mean of every draw is exact.
    for k in range(3):
        drawn = np.sort(res.draws[g == k, 7])
        assert np.array_equal(drawn, np.sort(shocks[g == k]))
    np.testing.assert_allclose(
        pd.DataFrame(res.draws).groupby(g).mean().to_numpy(),
        np.repeat(pd.Series(shocks).groupby(g).mean().to_numpy()[:, None], 2000, 1),
        rtol=1e-12,
    )
