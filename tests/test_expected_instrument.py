"""Tests for sp.expected_instrument (Borusyak and Hull 2023, expected instrument).

Evidence map:

* closed form: a Bernoulli share instrument has ``mu_i`` = mean of the
  relevant probabilities and ``Var(z_i) = sum p(1-p) / m_i^2``; the simulated
  ``mu`` converges to it at the ``1/sqrt(S)`` rate and the reported MC SE
  estimates ``sqrt(Var/S)``;
* fair coins give a constant ``mu`` (flagged); within-stratum permutation
  gives the stratum mean, and every draw keeps the per-stratum multiset;
* reproducibility: seeded calls are identical, ``seed_entropy`` replays an
  unseeded call, and reordering rows permutes ``mu``;
* streaming (``keep_draws=False``) gives the same moments as the kept draws;
* end to end: the draws fed to ``sp.recentered_iv(counterfactual=...)``
  recover a known ``beta`` where the naive IV is biased.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)

# ----------------------------------------------------------------------
# DGPs
# ----------------------------------------------------------------------


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


def _exposure_dgp(seed, n=600, K=200, k_exp=4, beta=1.0, gamma=1.0):
    """z_i = sum_k w_ik g_k, g_k ~ N(1, 1), so mu_i = sum_k w_ik = s_i.

    The error loads on total exposure s_i, so the realised instrument is
    correlated with it (Cov(z, u) = gamma Var(s)) while z - s is not.
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((n, K))
    scale = rng.uniform(0, 2, n)
    for i in range(n):
        idx = rng.choice(K, k_exp, replace=False)
        W[i, idx] = scale[i] * rng.uniform(0.5, 1.5, k_exp)
    g = rng.normal(1.0, 1.0, K)
    s = W.sum(axis=1)
    u = gamma * (s - s.mean()) + rng.normal(size=n)
    z = W @ g
    d = 0.5 * z + 0.8 * u + rng.normal(size=n)
    df = pd.DataFrame({"y": beta * d + u, "d": d, "z": z, "s": s})
    return df, W, K, g


# ----------------------------------------------------------------------
# Closed forms
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# Reproducibility, row order, storage
# ----------------------------------------------------------------------


def test_seeded_results_are_reproducible_and_row_order_free():
    df, W, K, _ = _exposure_dgp(9, n=300, K=150)

    def sampler(rng):
        return rng.normal(1.0, 1.0, K)

    a = sp.expected_instrument(df, lambda d, g: W @ g, sampler, n_draws=200, seed=11)
    b = sp.expected_instrument(df, lambda d, g: W @ g, sampler, n_draws=200, seed=11)
    assert np.array_equal(a.mu, b.mu) and np.array_equal(a.draws, b.draws)
    # An unseeded call records its entropy, which replays it exactly.
    u = sp.expected_instrument(df, lambda d, g: W @ g, sampler, n_draws=50)
    v = sp.expected_instrument(
        df, lambda d, g: W @ g, sampler, n_draws=50, seed=u.seed_entropy
    )
    assert np.array_equal(u.draws, v.draws)
    # Same shocks, rows reordered: mu permutes (row sums are order-free) and
    # the index travels with it.
    perm = np.random.default_rng(0).permutation(len(df))
    dfp = df.iloc[perm]
    Wp = W[perm]
    c = sp.expected_instrument(dfp, lambda d, g: Wp @ g, sampler, n_draws=200, seed=11)
    np.testing.assert_allclose(c.mu, a.mu[perm], rtol=1e-12)
    assert c.to_frame().index.equals(dfp.index)
    pd.testing.assert_frame_equal(
        c.counterfactual(), a.counterfactual().loc[dfp.index], rtol=1e-12
    )


def test_keep_draws_false_streams_same_moments():
    data, fn, sampler, _, _ = _share_design(15, n=50)
    kept = sp.expected_instrument(data, fn, sampler, n_draws=300, seed=4)
    streamed = sp.expected_instrument(
        data, fn, sampler, n_draws=300, seed=4, keep_draws=False
    )
    assert streamed.draws is None and kept.draws.shape == (50, 300)
    assert not streamed.diagnostics["draws_kept"]
    np.testing.assert_allclose(streamed.mu, kept.draws.mean(axis=1), rtol=1e-12)
    np.testing.assert_allclose(streamed.sd, kept.draws.std(axis=1, ddof=1), rtol=1e-10)
    with pytest.raises(MethodIncompatibility, match="not kept"):
        streamed.counterfactual()


def test_counterfactual_frame_is_the_draws_matrix():
    data, fn, sampler, _, _ = _share_design(16, n=40)
    data.index = pd.Index([f"u{i}" for i in range(40)])
    res = sp.expected_instrument(data, fn, sampler, n_draws=25, seed=3)
    cf = res.counterfactual()
    assert cf.shape == (40, 25)
    assert list(cf.columns) == [f"draw_{s}" for s in range(25)]
    assert cf.index.equals(data.index)
    assert np.array_equal(cf.to_numpy(), res.draws)


def test_observed_shocks_give_the_observed_instrument():
    data, fn, sampler, _, _ = _share_design(17, n=60)
    sex = (np.random.default_rng(1).random(int(data["m"].sum())) < 0.5).astype(float)
    res = sp.expected_instrument(
        data, fn, sampler, n_draws=50, seed=0, observed_shocks=sex
    )
    assert np.array_equal(res.observed, fn(data, sex))
    frame = res.to_frame()
    np.testing.assert_array_equal(frame["recentered"], res.observed - res.mu)


# ----------------------------------------------------------------------
# End to end with sp.recentered_iv
# ----------------------------------------------------------------------


def test_draws_feed_recentered_iv_and_recover_known_truth():
    """Exposure s_i is non-random and loads on the error. The naive IV is
    biased; feeding the simulated draws to sp.recentered_iv recovers beta = 1
    (mean over R = 40 replications within 4 MC SEs) and the simulated mu
    tracks the closed form s_i row by row."""
    R = 40
    est, naive = [], []
    for r in range(R):
        df, W, K, g = _exposure_dgp(5000 + r)
        ei = sp.expected_instrument(
            df,
            lambda d, sh: W @ sh,
            lambda rng: rng.normal(1.0, 1.0, K),
            n_draws=200,
            seed=r,
            observed_shocks=g,
        )
        assert np.allclose(ei.observed, df["z"].to_numpy(), rtol=1e-12)
        assert np.all(np.abs(ei.mu - df["s"].to_numpy()) < 6 * ei.mc_se)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sp.recentered_iv(df, "y", "d", "z", counterfactual=ei.draws, ri=False)
        est.append(out.estimate)
        naive.append(out.diagnostics["unadjusted_estimate"])
    est, naive = np.asarray(est), np.asarray(naive)
    assert abs(est.mean() - 1.0) < 4 * est.std(ddof=1) / np.sqrt(R)
    # plim of the naive IV: 1 + Cov(z, u) / Cov(z, d), above 1.3 here.
    assert naive.mean() - 1.0 > 0.3
    assert naive.mean() - 1.0 > 20 * naive.std(ddof=1) / np.sqrt(R)


def test_counterfactual_frame_equals_draws_array_in_recentered_iv():
    df, W, K, g = _exposure_dgp(21, n=300, K=150)
    ei = sp.expected_instrument(
        df,
        lambda d, sh: W @ sh,
        lambda rng: rng.normal(1.0, 1.0, K),
        n_draws=99,
        seed=2,
    )
    a = sp.recentered_iv(df, "y", "d", "z", ei.draws)
    b = sp.recentered_iv(df, "y", "d", "z", ei.counterfactual())
    assert a.estimate == b.estimate and a.se == b.se
    assert a.diagnostics["ri_pvalue"] == b.diagnostics["ri_pvalue"]
    assert a.diagnostics["ri_ci"] == b.diagnostics["ri_ci"]


def test_structurally_undefined_rows_are_nan_and_dropped_downstream():
    df, W, K, g = _exposure_dgp(16, n=200, K=100)

    def fn(data, shocks):
        z = W @ shocks
        z[:5] = np.nan  # no exposure: undefined whatever the shocks
        return z

    ei = sp.expected_instrument(
        df, fn, lambda rng: rng.normal(1.0, 1.0, K), n_draws=60, seed=1
    )
    assert ei.diagnostics["n_undefined"] == 5
    assert np.all(np.isnan(ei.mu[:5])) and np.all(np.isnan(ei.draws[:5]))
    assert np.all(np.isfinite(ei.draws[5:]))
    with pytest.warns(sp.AssumptionWarning, match="dropped 5 rows"):
        out = sp.recentered_iv(df, "y", "d", "z", ei.draws, ri=False)
    assert out.n_obs == 195


# ----------------------------------------------------------------------
# Loud failures
# ----------------------------------------------------------------------


def test_instrument_fn_wrong_length_raises():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(MethodIncompatibility, match="returned 2 values"):
        sp.expected_instrument(
            df, lambda d, s: s[:2], lambda rng: rng.random(4), n_draws=5
        )


def test_instrument_undefined_in_some_draws_raises():
    df = pd.DataFrame({"a": [1.0, 2.0]})

    def fn(d, s):
        out = np.ones(2)
        out[0] = np.nan if s[0] > 0.5 else 1.0
        return out

    with pytest.raises(NumericalInstability, match="undefined"):
        sp.expected_instrument(df, fn, lambda rng: rng.random(1), n_draws=50, seed=0)


def test_infinite_instrument_raises():
    df = pd.DataFrame({"a": [1.0, 0.0]})

    def fn(d, s):
        with np.errstate(divide="ignore"):
            return s.sum() / d["a"].to_numpy()

    with pytest.raises(NumericalInstability, match="infinite"):
        sp.expected_instrument(df, fn, lambda rng: rng.random(3) + 1, n_draws=5, seed=0)


@pytest.mark.parametrize(
    "kwargs,err,msg",
    [
        ({"n_draws": 1}, DataInsufficient, "at least 2"),
        ({"n_draws": 10.0}, MethodIncompatibility, "integer"),
        ({"keep_draws": "yes"}, MethodIncompatibility, "keep_draws"),
    ],
)
def test_bad_options_raise(kwargs, err, msg):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(err, match=msg):
        sp.expected_instrument(
            df,
            lambda d, s: d["a"].to_numpy() * s.sum(),
            lambda rng: rng.random(3),
            **kwargs,
        )


@pytest.mark.parametrize(
    "sampler,msg",
    [
        ({"kind": "bernoulli", "p": 0.5}, "needs 'size'"),
        ({"kind": "bernoulli", "p": [0.2, 1.5]}, r"in \[0, 1\]"),
        ({"kind": "bernoulli", "p": [0.2, 0.5], "size": 3}, "disagrees"),
        ({"kind": "permute"}, "observed 'shocks'"),
        ({"kind": "permute", "shocks": [1.0, 2.0], "strata": [0]}, "strata"),
        ({"kind": "poisson"}, "Unknown shock sampler"),
        (42, "callable"),
    ],
)
def test_malformed_sampler_specs_raise(sampler, msg):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(MethodIncompatibility, match=msg):
        sp.expected_instrument(
            df, lambda d, s: d["a"].to_numpy() * np.sum(s), sampler, n_draws=5
        )


def test_bad_data_and_fn_raise():
    with pytest.raises(MethodIncompatibility, match="DataFrame"):
        sp.expected_instrument(np.ones(3), lambda d, s: s, lambda rng: rng.random(3))
    with pytest.raises(MethodIncompatibility, match="callable"):
        sp.expected_instrument(
            pd.DataFrame({"a": [1.0]}), "not a function", lambda rng: rng.random(1)
        )
    with pytest.raises(DataInsufficient, match="no rows"):
        sp.expected_instrument(
            pd.DataFrame({"a": []}), lambda d, s: s, lambda rng: rng.random(0)
        )
