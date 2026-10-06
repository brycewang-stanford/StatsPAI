"""Behaviour of the functions added in the pass over Rosenbaum, *An
Introduction to the Theory of Observational Studies* (2026-10).

``reference_parity/test_rosenbaum_itos_parity.py`` pins the numbers against
the R packages. This file checks the same functions against what they
claim, with no reference implementation in the loop: level under
randomization, validity under the worst hidden bias the analysis allows,
brute-force enumeration on problems small enough to enumerate, and the
edges of the input.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.diagnostics import _sens_engine as eng

# ------------------------------------------------------------------ engine


def test_fnch_moments_match_scipy():
    for n_top, n_total, n_drawn, gamma in [
        (3, 10, 4, 2.5),
        (7, 12, 5, 1.0),
        (1, 4, 1, 6.0),
    ]:
        ea, ea2 = eng.fnch_moments(np.array([n_top]), n_total, n_drawn, gamma)
        dist = stats.nchypergeom_fisher(n_total, n_top, n_drawn, gamma)
        assert ea[0] == pytest.approx(dist.mean(), rel=1e-10)
        assert ea2[0] - ea[0] ** 2 == pytest.approx(dist.var(), rel=1e-9)


def _enumerate_moments(q, n_treated, u, gamma):
    """Mean and variance of the treated score sum by listing assignments."""
    idx = list(itertools.combinations(range(len(q)), n_treated))
    w = np.array([gamma ** sum(u[i] for i in s) for s in idx])
    w = w / w.sum()
    t = np.array([sum(q[i] for i in s) for s in idx])
    mean = float(w @ t)
    return mean, float(w @ (t - mean) ** 2)


def test_moment_tables_match_enumeration_and_bound_every_u():
    rng = np.random.default_rng(3)
    q = np.sort(rng.normal(size=7).round(1))  # rounding makes a tie likely
    n_treated, gamma = 3, 2.7
    mu, nu = eng.moment_tables(q[None, :], n_treated, gamma)
    for m in range(1, 7):
        u = np.r_[np.zeros(7 - m), np.ones(m)]
        mean, var = _enumerate_moments(q, n_treated, u, gamma)
        assert mu[0, m - 1] == pytest.approx(mean, rel=1e-12)
        assert nu[0, m - 1] == pytest.approx(var, rel=1e-10)
    # Rosenbaum and Krieger: no binary u gives a larger expectation
    best = max(
        _enumerate_moments(q, n_treated, np.array(u), gamma)[0]
        for u in itertools.product([0, 1], repeat=7)
    )
    assert mu.max() == pytest.approx(best, rel=1e-12)


# ---------------------------------------------------- weighted rank: level


def _biased_blocks(rng, n_blocks, size, gamma, effect=0.0):
    """Blocks in which the unit with the largest response is ``gamma``
    times as likely to be the treated one: the bias the bound allows."""
    y = rng.normal(size=(n_blocks, size))
    top = y.argmax(axis=1)
    p_top = gamma / (gamma + size - 1)
    others = (top[:, None] + rng.integers(1, size, n_blocks)[:, None]) % size
    pick = np.where(rng.random(n_blocks) < p_top, top, others[:, 0])
    rows = np.arange(n_blocks)
    y[rows, pick] += effect
    # put the treated unit in column 0
    y[rows, 0], y[rows, pick] = y[rows, pick].copy(), y[rows, 0].copy()
    return y


@pytest.mark.parametrize("phi", ["wilcoxon", "quade", "u868"])
def test_weighted_rank_has_its_level_under_randomization(phi):
    rng = np.random.default_rng(11)
    reject = np.mean(
        [
            sp.weighted_rank(rng.normal(size=(150, 3)), phi=phi).pvalue <= 0.05
            for _ in range(600)
        ]
    )
    assert 0.03 <= reject <= 0.075  # binomial se 0.009 around 0.05


def test_weighted_rank_bound_is_not_exceeded_under_the_bias_it_allows():
    """With a bias of Gamma = 2 and no effect, the Gamma = 2 analysis
    rejects no more than 5% of the time while the randomization test is
    badly fooled."""
    rng = np.random.default_rng(5)
    at_two, at_one = [], []
    for _ in range(500):
        y = _biased_blocks(rng, 200, 3, gamma=2.0)
        res = sp.weighted_rank(y, gamma=[1.0, 2.0], phi="wilcoxon")
        at_one.append(res.detail["pvalue"][0] <= 0.05)
        at_two.append(res.detail["pvalue"][1] <= 0.05)
    assert np.mean(at_two) <= 0.08
    assert np.mean(at_one) >= 0.9


def test_gamma_critical_is_where_the_bound_crosses_alpha():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(200, 3))
    y[:, 0] += 1.0
    res = sp.weighted_rank(y, phi="u878", alpha=0.01)
    assert res.gamma_critical > 1
    at = sp.weighted_rank(y, phi="u878", gamma=res.gamma_critical).pvalue
    assert at == pytest.approx(0.01, rel=1e-4)
    # no effect: sensitive already at Gamma = 1
    null = sp.weighted_rank(rng.normal(size=(200, 3)), phi="u878")
    assert null.gamma_critical == 1.0 or null.pvalue <= 0.05


def test_pairs_with_equal_weights_reproduce_the_signed_rank_interval():
    rng = np.random.default_rng(2)
    d = rng.normal(0.4, 1.0, 80)
    y = np.column_stack([d, np.zeros(80)])
    res = sp.weighted_rank(y, phi="wilcoxon", estimates=True, alternative="two-sided")
    walsh = np.add.outer(d, d)[np.triu_indices(80)] / 2.0
    # Within a pair the ranks are 1 and 2, so the test is the sign test
    # of the differences and its Hodges-Lehmann estimate their median.
    assert res.estimate[0] == pytest.approx(np.median(d), abs=1e-8)
    assert res.estimate[0] == pytest.approx(res.estimate[1], abs=1e-8)
    assert res.conf_int[0] < np.median(d) < res.conf_int[1]
    assert walsh.min() < res.conf_int[0]


def test_estimates_widen_with_gamma_and_bracket_the_truth():
    rng = np.random.default_rng(4)
    y = rng.normal(size=(300, 4))
    y[:, 0] += 0.7
    res = sp.weighted_rank(
        y, gamma=[1.0, 1.5, 2.5], estimates=True, alternative="two-sided"
    )
    d = res.detail
    assert (
        d["hl_lower"].is_monotonic_decreasing and d["hl_upper"].is_monotonic_increasing
    )
    assert (
        d["ci_lower"].is_monotonic_decreasing and d["ci_upper"].is_monotonic_increasing
    )
    assert d["hl_lower"][0] == pytest.approx(d["hl_upper"][0], abs=1e-6)
    assert d["ci_lower"][0] < 0.7 < d["ci_upper"][0]
    assert res.estimate == (d["hl_lower"][0], d["hl_upper"][0])
    one = sp.weighted_rank(y, estimates=True)
    assert np.isinf(one.conf_int[1]) and one.conf_int[0] > d["ci_lower"][0]


def test_weighted_rank_is_invariant_to_what_should_not_matter():
    rng = np.random.default_rng(8)
    y = rng.normal(size=(120, 4))
    y[:, 0] += 0.5
    base = sp.weighted_rank(y, gamma=1.7)
    # order of the controls, monotone transformation that keeps the ranks
    # of the ranges (a common scale), the labels of the blocks
    assert sp.weighted_rank(y[:, [0, 3, 1, 2]], gamma=1.7).deviate == pytest.approx(
        base.deviate, rel=1e-12
    )
    assert sp.weighted_rank(3.0 * y + 10.0, gamma=1.7).deviate == pytest.approx(
        base.deviate, rel=1e-12
    )
    long = pd.DataFrame(
        {
            "y": y.ravel(),
            "z": np.tile([1, 0, 0, 0], 120),
            "b": np.repeat(rng.permutation(120), 4),
        }
    ).sample(frac=1.0, random_state=1)
    assert sp.weighted_rank(
        "y", data=long, treat="z", block="b", gamma=1.7
    ).deviate == pytest.approx(base.deviate, rel=1e-12)
    z = np.zeros_like(y)
    z[:, 0] = 1
    assert sp.weighted_rank(y, treated=z, gamma=1.7).pvalue == base.pvalue
    # negating the outcomes swaps the tails
    less = sp.weighted_rank(-y, gamma=1.7, alternative="less")
    assert less.pvalue == pytest.approx(base.pvalue, rel=1e-10)
    two = sp.weighted_rank(y, gamma=1.7, alternative="two-sided")
    assert two.pvalue == pytest.approx(min(1.0, 2 * base.pvalue), rel=1e-12)


def test_conditional_test_has_its_level_and_ignores_the_middle():
    rng = np.random.default_rng(21)
    reject = np.mean(
        [
            sp.weighted_rank(rng.normal(size=(200, 4)), conditional=True).pvalue <= 0.05
            for _ in range(500)
        ]
    )
    assert 0.03 <= reject <= 0.08
    y = rng.normal(size=(150, 5))
    y[:, 0] += 0.8
    a = sp.weighted_rank(y, gamma=2, conditional=True)
    # moving the three middle responses of each block changes nothing
    s = np.sort(y, axis=1)
    middle = (y > s[:, [0]]) & (y < s[:, [-1]])
    y2 = np.where(middle, (s[:, [0]] + s[:, [-1]]) / 2, y)
    b = sp.weighted_rank(y2, gamma=2, conditional=True)
    assert a.deviate == pytest.approx(b.deviate, rel=1e-12)
    assert a.diagnostics["n_decisive"] < 150


def test_weighted_rank_input_errors():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(30, 3))
    with pytest.raises(ValueError, match="gamma must be >= 1"):
        sp.weighted_rank(y, gamma=0.5)
    with pytest.raises(ValueError, match="unknown phi"):
        sp.weighted_rank(y, phi="u999")
    with pytest.raises(ValueError, match="1 <= m1 <= m2 <= m"):
        sp.weighted_rank(y, phi=(8, 9, 8))
    with pytest.raises(ValueError, match="non-decreasing"):
        sp.weighted_rank(y, scores=(3, 2, 1))
    with pytest.raises(ValueError, match="missing or infinite"):
        sp.weighted_rank(np.where(y > 2, np.nan, y))
    with pytest.raises(sp.MethodIncompatibility, match="conditional=True"):
        sp.weighted_rank(y, conditional=True, estimates=True)
    with pytest.raises(sp.MethodIncompatibility, match="single phi"):
        sp.weighted_rank(y, phi=["u868", "u878"], estimates=True)
    with pytest.raises(sp.DataInsufficient, match="no variance"):
        sp.weighted_rank(np.ones((10, 3)))
    ragged = pd.DataFrame(
        {
            "y": rng.normal(size=7),
            "z": [1, 0, 1, 0, 0, 1, 0],
            "b": [1, 1, 2, 2, 2, 3, 3],
        }
    )
    with pytest.raises(sp.MethodIncompatibility, match="same size") as err:
        sp.weighted_rank("y", data=ragged, treat="z", block="b")
    assert "sp.rosenbaum_stratified" in err.value.alternative_functions
    z = np.zeros_like(y)
    z[:5, 0] = 1  # 25 blocks without a treated unit
    with pytest.warns(UserWarning, match="25 blocks"):
        res = sp.weighted_rank(y, treated=z)
    assert res.n_blocks == 5


# ------------------------------------------------------------------- power


def test_power_estimate_tracks_simulated_power():
    """A pilot of 4000 blocks pins the mean and variance, so the estimate
    for a study of 150 blocks can be compared with the rejection rate in
    simulated studies of 150 blocks. This is also what decides between
    scaling the bounding standard deviation by the square root of the
    sample ratio (here) and by the ratio itself (``weightedRank::estPower``
    0.7.0): the second puts the power at zero where it is 0.2 to 0.9."""
    rng = np.random.default_rng(9)

    def blocks(n):
        y = rng.normal(size=(n, 3))
        y[:, 0] += 0.6
        return y

    gammas = [1.8, 2.3, 2.8]
    pilot = blocks(4000)
    ratio = 150 / 4000
    est = sp.weighted_rank_power(pilot, gammas, sample_ratio=ratio)
    hits = np.zeros(3)
    n_rep = 400
    for _ in range(n_rep):
        res = sp.weighted_rank(blocks(150), gamma=gammas)
        hits += res.detail["pvalue"].to_numpy() <= 0.05
    truth = hits / n_rep
    np.testing.assert_allclose(est.power["power"], truth, atol=0.1)
    assert truth[0] > 0.8 and 0.1 < truth[2] < 0.4

    crit = stats.norm.isf(0.05)
    by_ratio = []
    for g in gammas:
        w = sp.weighted_rank(pilot, gamma=g)
        e_bar, sd_bar = w.expectation / 4000, np.sqrt(w.variance) / 4000
        z = (e_bar - est.mean + crit * sd_bar / ratio) / np.sqrt(est.variance / ratio)
        by_ratio.append(stats.norm.sf(z))
    assert max(by_ratio) < 0.01


# ------------------------------------------------------ stratified, others


def test_rosenbaum_stratified_level_and_agreement_with_ranksum():
    rng = np.random.default_rng(3)
    n = 400
    s = rng.integers(0, 5, n)
    df = pd.DataFrame({"s": s, "z": rng.binomial(1, 0.4, n)})
    df["y"] = 0.5 * s + rng.normal(size=n)
    one = sp.rosenbaum_stratified(df, "y", "z", alternative="two-sided")
    u = stats.mannwhitneyu(
        df.y[df.z == 1], df.y[df.z == 0], method="asymptotic", use_continuity=False
    )
    assert one.pvalue == pytest.approx(u.pvalue, rel=1e-9)
    reject = []
    for _ in range(400):
        df["z"] = rng.binomial(1, 0.4, n)
        reject.append(
            sp.rosenbaum_stratified(df, "y", "z", "s", score="aligned_rank").pvalue
            <= 0.05
        )
    assert 0.03 <= np.mean(reject) <= 0.08


def test_rosenbaum_stratified_bounds_order_and_pairs_special_case():
    rng = np.random.default_rng(9)
    t, c = rng.normal(0.5, 1, 60), rng.normal(0, 1, 60)
    long = pd.DataFrame(
        {
            "y": np.r_[t, c],
            "z": np.r_[np.ones(60), np.zeros(60)],
            "p": np.r_[0:60, 0:60],
        }
    )
    res = sp.rosenbaum_stratified(
        long, "y", "z", "p", gamma=[1, 1.5, 2], score="stratum_rank"
    )
    d = res.detail
    assert d["pvalue"].is_monotonic_increasing
    assert (d["pvalue_taylor"] >= d["pvalue_separable"] - 1e-15).all()
    # within-pair ranks: the sign test, by its normal approximation
    n_pos = int(np.sum(t > c))
    z = (n_pos - 60 * 2 / 3) / np.sqrt(60 * 2 / 9)
    assert d["pvalue_separable"][2] == pytest.approx(stats.norm.sf(z), rel=1e-9)
    with pytest.raises(sp.DataInsufficient):
        sp.rosenbaum_stratified(long[long.z == 1], "y", "z", "p")
    with pytest.raises(ValueError, match="score must be"):
        sp.rosenbaum_stratified(long, "y", "z", "p", score="median")


def test_noether_edges_and_truncated_product_edges():
    d = np.array([0.0, 0.0, 1.0, -2.0, 3.0, 4.0])
    res = sp.noether_test(d, f=0)
    assert res.diagnostics["n_pairs_used"] == 4 and res.statistic == 3
    assert res.pvalue == pytest.approx(stats.binomtest(3, 4, 0.5, "greater").pvalue)
    assert sp.noether_test(d, np.zeros(6), f=0).pvalue == res.pvalue
    with pytest.raises(sp.DataInsufficient):
        sp.noether_test(np.zeros(5))
    with pytest.raises(ValueError, match="0 <= f < 1"):
        sp.noether_test(d, f=1.0)
    assert sp.truncated_product([0.3]) == 1.0
    assert sp.truncated_product([0.03]) == pytest.approx(0.03)
    assert sp.truncated_product([0.0, 0.5]) == 0.0
    with pytest.raises(ValueError):
        sp.truncated_product([1.2])
    with pytest.raises(ValueError):
        sp.truncated_product([0.1], trunc=0.0)
    # uniform p-values give a uniform combined p-value
    rng = np.random.default_rng(1)
    combined = np.array([sp.truncated_product(rng.random(4)) for _ in range(4000)])
    for level in (0.05, 0.2):
        assert np.mean(combined <= level) == pytest.approx(level, abs=0.015)


def test_amplify_identity_and_errors():
    gamma, lam = 2.5, np.array([3.0, 5.0, 40.0])
    delta = sp.amplify(gamma, lam)
    np.testing.assert_allclose((lam * delta + 1) / (lam + delta), gamma, rtol=1e-13)
    assert sp.amplify(gamma, 1e12) == pytest.approx(gamma, rel=1e-9)
    with pytest.raises(ValueError):
        sp.amplify(1.0, 3.0)
    with pytest.raises(ValueError):
        sp.amplify(2.0, [3.0, 1.5])


def test_evidence_factors_are_nearly_independent_under_the_null():
    rng = np.random.default_rng(13)
    p1, p2, comb = [], [], []
    for _ in range(600):
        res = sp.evidence_factors(rng.normal(size=(120, 3)))
        p1.append(res.pvalue_treated_vs_control1)
        p2.append(res.pvalue_control2_vs_others)
        comb.append(res.pvalue)
    assert abs(stats.spearmanr(p1, p2).statistic) < 0.1
    assert np.mean(np.array(comb) <= 0.05) == pytest.approx(0.05, abs=0.025)
    with pytest.raises(ValueError, match="three columns"):
        sp.evidence_factors(rng.normal(size=(10, 4)))


# ---------------------------------------------------------------- matching


def _brute_force(pair, balance, use, ratio):
    """Best objective over every choice of controls and both assignments."""
    n_t, n_c = pair.shape
    best = np.inf
    slots = [t for t in range(n_t) for _ in range(ratio)]
    for chosen in itertools.combinations(range(n_c), n_t * ratio):
        perms = list(itertools.permutations(chosen))
        left = min(sum(pair[t, c] for t, c in zip(slots, perm)) for perm in perms)
        right = min(sum(balance[t, c] for t, c in zip(slots, perm)) for perm in perms)
        best = min(best, left + right + sum(use[c] for c in chosen))
    return best


@pytest.mark.parametrize("ratio, n_t, n_c", [(1, 3, 6), (2, 2, 6), (1, 4, 5)])
def test_two_criteria_match_is_optimal_by_enumeration(ratio, n_t, n_c):
    rng = np.random.default_rng(100 * ratio + n_t)
    for _ in range(4):
        pair = rng.integers(0, 9, (n_t, n_c)).astype(float) + rng.random((n_t, n_c))
        balance = rng.integers(0, 4, (n_t, n_c)) * 5.0
        use = rng.integers(0, 3, n_c).astype(float)
        df = pd.DataFrame({"z": np.r_[np.ones(n_t), np.zeros(n_c)]})
        fit = sp.two_criteria_match(
            df, "z", pair=pair, balance=balance, control_cost=use, ratio=ratio
        )
        assert fit.total_cost == pytest.approx(
            _brute_force(pair, balance, use, ratio), rel=1e-12
        )
        assert fit.pairs["control"].is_unique
        assert fit.matched.groupby("mset").size().eq(ratio + 1).all()


def _reservoir(seed=0, n=300, centre=58):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, n).round(),
            "female": rng.integers(0, 2, n),
            "smoke": rng.integers(1, 4, n),
        }
    )
    lin = (df.age - centre) / 6 + 0.8 * (df.smoke == 1)
    df["z"] = rng.binomial(1, 1 / (1 + np.exp(-lin)))
    return df


def test_fine_balance_equalises_margins_without_pairing_on_them():
    df = _reservoir()
    fit = sp.two_criteria_match(
        df,
        "z",
        pair=[{"type": "mahalanobis", "on": ["age"]}],
        balance=[{"type": "near_exact", "on": "smoke", "penalty": 1000}],
    )
    m = fit.matched
    counts = pd.crosstab(m["z"], m["smoke"])
    assert (counts.loc[0] == counts.loc[1]).all()
    assert fit.balance_cost == 0
    # ... while pairs are free to differ on it
    within = m.groupby("mset")["smoke"].nunique()
    assert (within > 1).any()
    # the same covariate in pair= forces agreement within pairs instead
    exact = sp.two_criteria_match(
        df, "z", pair=[{"type": "near_exact", "on": "smoke", "penalty": 1000},
                       {"type": "mahalanobis", "on": ["age"]}],
    )  # fmt: skip
    assert exact.matched.groupby("mset")["smoke"].nunique().eq(1).all()
    assert exact.pair_cost >= fit.pair_cost


def test_subset_cost_trades_treated_units_for_closeness():
    df = _reservoir(seed=3)
    terms = dict(pair=[{"type": "caliper", "on": "age", "width": 2, "penalty": 50}])
    full = sp.two_criteria_match(df, "z", **terms)
    assert full.n_unmatched == 0 and full.pair_cost > 0
    part = sp.two_criteria_match(df, "z", subset_cost=10.0, **terms)
    assert part.n_unmatched > 0
    assert part.pair_cost == 0  # every violation costs 50, dropping costs 10
    assert part.total_cost == pytest.approx(10.0 * part.n_unmatched)
    assert part.n_sets + part.n_unmatched == full.n_treated
    assert part.matched["mset"].max() == part.n_sets
    never = sp.two_criteria_match(df, "z", subset_cost=1e6, **terms)
    assert never.n_unmatched == 0
    assert never.total_cost == pytest.approx(full.total_cost)


def test_two_criteria_match_input_errors():
    df = _reservoir()
    with pytest.raises(ValueError, match="at least one of pair= and balance="):
        sp.two_criteria_match(df, "z")
    with pytest.raises(ValueError, match="unknown cost term type"):
        sp.two_criteria_match(df, "z", pair=[{"type": "exact", "on": "female"}])
    with pytest.raises(sp.DataInsufficient, match="Missing columns"):
        sp.two_criteria_match(df, "z", pair=[{"type": "integer", "on": "nope"}])
    with pytest.raises(ValueError, match="takes the keys"):
        sp.two_criteria_match(
            df, "z", pair=[{"type": "near_exact", "on": "female", "width": 2}]
        )
    with pytest.raises(sp.MethodIncompatibility, match="ratio=1"):
        sp.two_criteria_match(
            df, "z", pair=[{"type": "integer", "on": "age"}], ratio=2, subset_cost=1
        )
    few = pd.concat([df[df.z == 1], df[df.z == 0].head(5)])
    with pytest.raises(sp.DataInsufficient, match="distinct controls"):
        sp.two_criteria_match(few, "z", pair=[{"type": "integer", "on": "age"}])
    with pytest.raises(ValueError, match="shape"):
        sp.two_criteria_match(df, "z", pair=np.zeros((2, 2)))
    with pytest.raises(sp.DataInsufficient):
        sp.two_criteria_match(
            df[df.z == 1], "z", pair=[{"type": "integer", "on": "age"}]
        )
    df2 = df.copy()
    df2.loc[3, "age"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        sp.two_criteria_match(df2, "z", pair=[{"type": "integer", "on": "age"}])


def test_matched_frame_feeds_the_outcome_analysis():
    df = _reservoir(seed=5, n=900, centre=66)
    rng = np.random.default_rng(1)
    df["y"] = 0.8 * df.z + rng.normal(size=len(df))
    fit = sp.two_criteria_match(
        df, "z", ps=["age", "female", "smoke"], ratio=2,
        pair=[{"type": "mahalanobis", "on": ["age", "female", "smoke"]},
              {"type": "caliper", "on": "pscore", "penalty": 100}],
    )  # fmt: skip
    assert "pscore" in fit.matched and fit.matched.index.isin(df.index).all()
    assert fit.matched.groupby("mset")["z"].first().eq(1).all()
    out = sp.weighted_rank(
        "y", data=fit.matched, treat="z", block="mset", estimates=True
    )
    assert out.block_size == 3 and out.pvalue < 0.01
    assert out.conf_int[0] < 0.8
    assert isinstance(fit.summary(), str) and "Pairing cost" in fit.summary()
    assert set(fit.to_dict()) >= {"pair_cost", "balance_cost", "n_sets"}


def test_tighten_blocks_keeps_blocks_and_balances():
    rng = np.random.default_rng(0)
    n_blocks = 90
    df = pd.DataFrame(
        {
            "block": np.repeat(np.arange(n_blocks), 4),
            "z": np.tile([1, 0, 0, 0], n_blocks),
            "bmi": rng.normal(27, 4, 4 * n_blocks),
        }
    )
    df["cat"] = np.where(
        df.z == 1, rng.integers(0, 2, len(df)), rng.integers(0, 3, len(df))
    )
    tight = sp.tighten_blocks(
        df, "z", "block", covariates=["bmi"], fine_balance=["cat"], ratio=1
    )
    m = tight.matched
    assert m.groupby("mset")["block"].nunique().eq(1).all()
    assert m.groupby("mset").size().eq(2).all() and tight.n_sets == n_blocks
    before = abs(df[df.z == 0]["cat"].eq(2).mean() - df[df.z == 1]["cat"].eq(2).mean())
    after = abs(m[m.z == 0]["cat"].eq(2).mean() - m[m.z == 1]["cat"].eq(2).mean())
    assert after < before / 2
    dropped = sp.tighten_blocks(
        df, "z", "block", covariates=["bmi"], fine_balance=["cat"], subset_cost=1.0
    )
    assert 0 < dropped.n_unmatched < n_blocks
    bad = df.copy()
    bad.loc[1, "z"] = 1
    with pytest.raises(sp.MethodIncompatibility, match="exactly one treated"):
        sp.tighten_blocks(bad, "z", "block", covariates=["bmi"])
    with pytest.raises(sp.DataInsufficient, match="fewer than 4 controls"):
        sp.tighten_blocks(df, "z", "block", covariates=["bmi"], ratio=4)


# ------------------------------------------------------------ balance check


def test_balance_vs_randomization_tests_match_scipy_and_calibrate():
    rng = np.random.default_rng(6)
    n = 120
    df = pd.DataFrame(
        {
            "z": rng.permutation(np.r_[np.ones(40), np.zeros(80)]),
            "x": rng.normal(size=n),
            "k": rng.integers(0, 2, n),
            "g": rng.integers(0, 4, n),
        }
    )
    t, c = df[df.z == 1], df[df.z == 0]
    res = sp.balance_vs_randomization(
        df, "z", ["x", "k", "g"], n_sim=400, random_state=0
    )
    assert res.tests_used == {"x": "wilcoxon", "k": "chi2", "g": "wilcoxon"}
    # 40 and 80 without ties: R's rule sends this to the normal approximation
    assert res.table.loc["x", "actual"] == pytest.approx(
        stats.mannwhitneyu(t.x, c.x, method="asymptotic").pvalue, rel=1e-10
    )
    assert res.table.loc["k", "actual"] == pytest.approx(
        stats.chi2_contingency(pd.crosstab(df.k, df.z)).pvalue, rel=1e-10
    )
    as_t = sp.balance_vs_randomization(
        df, "z", ["x"], n_sim=20, test="t", random_state=0
    )
    assert as_t.table.loc["x", "actual"] == pytest.approx(
        stats.ttest_ind(t.x, c.x, equal_var=False).pvalue, rel=1e-10
    )
    levels = sp.balance_vs_randomization(
        df, "z", ["g"], n_sim=20, max_levels=4, random_state=0
    )
    assert levels.table.loc["g", "actual"] == pytest.approx(
        stats.chi2_contingency(pd.crosstab(df.g, df.z), correction=False).pvalue,
        rel=1e-10,
    )
    # a randomized sample sits in the middle of its own benchmark
    assert 0.02 < res.table.loc["min_p", "share_better"] < 0.98
    assert res.sim.shape == (400, 6)
    again = sp.balance_vs_randomization(
        df, "z", ["x", "k", "g"], n_sim=400, random_state=0
    )
    pd.testing.assert_frame_equal(res.table, again.table)


def test_balance_vs_randomization_small_untied_groups_use_the_exact_test():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"z": np.r_[np.ones(12), np.zeros(15)], "x": rng.normal(size=27)})
    res = sp.balance_vs_randomization(df, "z", ["x"], n_sim=10, random_state=0)
    exact = stats.mannwhitneyu(df.x[df.z == 1], df.x[df.z == 0], method="exact").pvalue
    assert res.table.loc["x", "actual"] == pytest.approx(exact, rel=1e-12)
    with pytest.raises(sp.DataInsufficient):
        sp.balance_vs_randomization(df.iloc[11:], "z", ["x"])
    with pytest.raises(sp.DataInsufficient, match="Missing columns"):
        sp.balance_vs_randomization(df, "z", ["nope"])


def test_a_good_match_beats_most_randomized_experiments():
    rng = np.random.default_rng(8)
    n = 1500
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, n).round(),
            "female": rng.integers(0, 2, n),
            "smoke": rng.integers(1, 4, n),
        }
    )
    lin = -2.2 + 0.05 * (df.age - 50) + 0.7 * (df.smoke == 1) - 0.5 * df.female
    df["z"] = rng.binomial(1, 1 / (1 + np.exp(-lin)))
    fit = sp.two_criteria_match(
        df, "z",
        pair=[{"type": "mahalanobis", "on": ["age", "female", "smoke"]}],
        balance=[{"type": "near_exact", "on": "smoke"},
                 {"type": "near_exact", "on": "female"}],
    )  # fmt: skip
    cov = ["age", "female", "smoke"]
    raw = sp.balance_vs_randomization(df, "z", cov, n_sim=300, random_state=1)
    matched = sp.balance_vs_randomization(
        fit.matched, "z", cov, n_sim=300, random_state=1
    )
    # before matching nearly every randomized experiment is better balanced;
    # after it, few are
    assert raw.table.loc["min_p", "share_better"] > 0.95
    assert matched.table.loc["min_p", "share_better"] < 0.2
    assert matched.table.loc["n_below_alpha", "actual"] == 0
