"""Functions added or rebuilt while auditing Wager, *Causal Inference: A
Statistical Learning Approach* (draft of September 2026).

Each block states what is being checked and why the tolerance is what it
is. Review notes: ``docs/dev/2026-10-07-wager-causal-inference-review.md``.
"""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from statspai.experimental.adaptive import _prob_best_beta, _prob_best_gaussian
from statspai.interference.network_exposure import (
    _as4_mapping,
    _dependency_graph,
    _fraction_mapping,
)

# ----------------------------------------------------------------------
# sp.network_exposure
# ----------------------------------------------------------------------


def _ring_plus(n, step=7, every=3):
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
        if i % every == 0:
            A[i, (i + step) % n] = A[(i + step) % n, i] = 1
    return A


def test_exposure_probabilities_are_exact():
    """The closed-form probabilities equal a full enumeration.

    Seven units, p = 0.3: all 128 assignments are enumerated with their
    probabilities, which gives each unit's exposure distribution exactly.
    The Horvitz-Thompson mean divides by that probability, so the check
    is that sum_i 1{H_i = h} Y_i / (n * mean) reproduces it.
    """
    rng = np.random.default_rng(0)
    n, p = 7, 0.3
    A = np.zeros((n, n), dtype=int)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 3), (2, 6)]:
        A[i, j] = A[j, i] = 1
    for mapping, fn in (("as4", _as4_mapping), ("fraction", _fraction_mapping)):
        exact = {}
        for bits in range(2**n):
            z = np.array([(bits >> k) & 1 for k in range(n)])
            pr = p ** z.sum() * (1 - p) ** (n - z.sum())
            for i, lab in enumerate(fn(z, A)):
                exact.setdefault(lab, np.zeros(n))[i] += pr
        Z = np.array([1, 0, 0, 1, 0, 1, 0])
        Y = rng.normal(size=n) + 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.network_exposure(
                Y, Z, A, mapping=mapping, p_treat=p, estimator="ht", variance="hac"
            )
        labels = fn(Z, A)
        for _, row in res.estimates.iterrows():
            lev = row["exposure"]
            assert row["min_prob"] == pytest.approx(exact[lev].min(), abs=1e-14)
            ind = labels == lev
            if ind.any():
                by_hand = np.sum(Y[ind] / exact[lev][ind]) / n
                assert row["mean_Y(d)"] == pytest.approx(by_hand, abs=1e-12)


def test_horvitz_thompson_is_unbiased_and_variance_is_exact_in_expectation():
    """Full enumeration of a small Bernoulli design.

    With n = 10 every one of the 1,024 assignments is evaluated, so
    expectations over the design are exact (no simulation error).

    * The Horvitz-Thompson contrast is exactly unbiased for the average
      of Y_i(h') - Y_i(h).
    * The reported variance is the quadratic form n^-2 v' G v, floored
      at zero (G is not positive semidefinite, so the raw form can be
      negative on a network this small).
    * The expectation of the raw form equals the true randomisation
      variance plus n^-2 * delta' G delta, where delta is the vector of
      unit-level effects: the identity behind the conservativeness
      argument.
    * The PSD-adjusted estimate has an expectation at least as large as
      the true variance.
    """
    n, p = 10, 0.5
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    rng = np.random.default_rng(1)
    y_h = {
        h: rng.normal(size=n) + k for k, h in enumerate(["c00", "c01", "c10", "c11"])
    }
    G = _dependency_graph(A).toarray()
    lam, U = np.linalg.eigh(G)
    assert lam.min() < 0
    e = {"c00": (1 - p) ** 3, "c01": (1 - p) * (1 - (1 - p) ** 2)}
    key = "spillover (c01 - c00)"
    est, raw, psd, prob = [], [], [], []
    for bits in range(2**n):
        z = np.array([(bits >> k) & 1 for k in range(n)])
        lab = _as4_mapping(z, A)
        y = np.array([y_h[lab[i]][i] for i in range(n)])
        v = ((lab == "c01") / e["c01"] - (lab == "c00") / e["c00"]) * y
        est.append(v.mean())
        raw.append(v @ G @ v / n**2)
        psd.append(((U.T @ v) ** 2 * np.clip(lam, 0, None)).sum() / n**2)
        prob.append(p ** z.sum() * (1 - p) ** (n - z.sum()))
        if bits % 37 == 0 and (lab == "c01").any() and (lab == "c00").any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r1 = sp.network_exposure(
                    y, z, A, p_treat=p, estimator="ht", variance="hac"
                )
                r2 = sp.network_exposure(y, z, A, p_treat=p, estimator="ht")
            c1 = r1.contrasts.set_index("contrast").loc[key]
            c2 = r2.contrasts.set_index("contrast").loc[key]
            assert c1["estimate"] == pytest.approx(est[-1], abs=1e-12)
            assert c1["se"] ** 2 == pytest.approx(max(raw[-1], 0.0), abs=1e-10)
            assert c2["se"] ** 2 == pytest.approx(psd[-1], abs=1e-10)
    est, raw, psd, prob = map(np.asarray, (est, raw, psd, prob))
    delta = y_h["c01"] - y_h["c00"]
    mean = prob @ est
    true_var = prob @ (est - mean) ** 2
    assert mean == pytest.approx(delta.mean(), abs=1e-12)
    assert prob @ raw == pytest.approx(true_var + delta @ G @ delta / n**2, abs=1e-10)
    assert prob @ psd >= true_var - 1e-12


def test_hajek_is_translation_invariant_and_contrasts_are_differences():
    rng = np.random.default_rng(2)
    n = 300
    A = _ring_plus(n)
    Z = (rng.random(n) < 0.5).astype(int)
    Y = 1 + 2 * Z + ((A @ Z) > 0) + rng.normal(size=n)
    a = sp.network_exposure(Y, Z, A, p_treat=0.5)
    b = sp.network_exposure(Y + 100.0, Z, A, p_treat=0.5)
    np.testing.assert_allclose(
        a.contrasts["estimate"], b.contrasts["estimate"], atol=1e-10
    )
    np.testing.assert_allclose(a.contrasts["se"], b.contrasts["se"], rtol=1e-9)
    m = a.estimates.set_index("exposure")["mean_Y(d)"]
    c = a.contrasts.set_index("contrast")["estimate"]
    assert c["composite (c11 - c00)"] == pytest.approx(m["c11"] - m["c00"], abs=1e-12)
    assert a.estimator == "hajek" and a.variance == "hac_psd"
    assert a.n_eligible == n
    # The PSD adjustment can only raise the variance.
    h = sp.network_exposure(Y, Z, A, p_treat=0.5, variance="hac")
    assert (a.contrasts["se"].to_numpy() >= h.contrasts["se"].to_numpy() - 1e-12).all()


def test_network_exposure_coverage_on_a_well_overlapped_network():
    """Fixed potential outcomes, 300 re-randomisations.

    The estimand is the finite-population contrast, known exactly.
    Probed over 1,500 draws: the HAC interval covers 93-95% and the PSD
    interval 98%, the bias of the Hajek contrast is below 0.01. With 300
    draws the Monte Carlo error of a coverage rate is about 0.013, so
    the bounds below leave three of those.
    """
    rng = np.random.default_rng(3)
    n = 400
    A = _ring_plus(n)
    base, het = 5 + rng.normal(size=n), rng.normal(size=n)

    def po(own, has):
        return base + (2 + het) * own + (1 + 0.5 * het) * has

    truth = float(np.mean(po(0, 1) - po(0, 0)))
    draws = np.random.default_rng(4)
    hit_hac, hit_psd, est = [], [], []
    for _ in range(300):
        Z = (draws.random(n) < 0.5).astype(int)
        Y = po(Z, ((A @ Z) > 0).astype(int))
        for variance, hits in (("hac", hit_hac), ("hac_psd", hit_psd)):
            r = sp.network_exposure(Y, Z, A, p_treat=0.5, variance=variance)
            row = r.contrasts.set_index("contrast").loc["spillover (c01 - c00)"]
            hits.append(row["ci_lo"] <= truth <= row["ci_hi"])
        est.append(row["estimate"])
    assert abs(np.mean(est) - truth) < 0.04
    assert 0.89 <= np.mean(hit_hac) <= 0.985
    assert np.mean(hit_psd) >= np.mean(hit_hac)
    assert np.mean(hit_psd) >= 0.94


def test_network_exposure_edge_cases():
    rng = np.random.default_rng(5)
    n = 60
    A = _ring_plus(n)
    A[0, :] = 0
    A[:, 0] = 0  # an isolated unit can never have a treated neighbour
    Z = (rng.random(n) < 0.5).astype(int)
    Y = rng.normal(size=n)
    with pytest.warns(AssumptionWarning, match="excluded"):
        r = sp.network_exposure(Y, Z, A, p_treat=0.5)
    assert r.n_eligible == n - 1
    with pytest.raises(MethodIncompatibility):
        sp.network_exposure(Y, Z, A, estimator="nope")
    with pytest.raises(MethodIncompatibility):
        sp.network_exposure(Y, Z * 2, A)
    with pytest.raises(NotImplementedError):
        sp.network_exposure(Y, Z, A, design="complete")
    with pytest.raises(DataInsufficient):
        sp.network_exposure(Y, Z, np.zeros((n, n), dtype=int), p_treat=0.5)
    # A callable mapping reproduces the built-in one up to simulation error
    # in the exposure probabilities.
    B = _ring_plus(n)
    exact = sp.network_exposure(Y, Z, B, p_treat=0.5, estimator="ht")
    sim = sp.network_exposure(
        Y, Z, B, mapping=_as4_mapping, p_treat=0.5, estimator="ht", n_sim=4000
    )
    assert sim.detail["exposure_probabilities"] == "simulated"
    np.testing.assert_allclose(
        sim.estimates["mean_Y(d)"], exact.estimates["mean_Y(d)"], atol=0.15
    )
    custom = sp.network_exposure(
        Y, Z, B, mapping="fraction", p_treat=0.5, contrasts=[("z1_b1", "z0_b1")]
    )
    assert custom.contrasts["contrast"].tolist() == ["z1_b1 - z0_b1"]


# ----------------------------------------------------------------------
# sp.interference_test
# ----------------------------------------------------------------------


def test_interference_test_enumeration_matches_by_hand():
    """Fourteen units, seven focal: the 21 admissible assignments are
    enumerated and the p-value is recomputed here from the definition."""
    rng = np.random.default_rng(6)
    n = 14
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    Z = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    Y = rng.normal(size=n) + Z
    focal = np.array([0, 2, 4, 6, 8, 10, 12])
    r = sp.interference_test(Y, Z, A, focal=focal, n_perm=5000)
    assert r.exact and r.n_focal == 7
    free = np.setdiff1d(np.arange(n), focal)
    k = int(Z[free].sum())

    def stat(z):
        share = (A @ z) / A.sum(axis=1)
        Xf = np.column_stack([np.ones(focal.size), z[focal], share[focal]])
        return np.linalg.lstsq(Xf, Y[focal], rcond=None)[0][-1]

    t0, count, total = stat(Z), 0, 0
    for treated in combinations(free, k):
        z = Z.copy()
        z[free] = 0
        z[list(treated)] = 1
        total += 1
        count += abs(stat(z)) >= abs(t0) - 1e-12
    assert r.n_perm == total
    assert r.pvalue == pytest.approx(count / total, abs=1e-12)
    assert r.statistic == pytest.approx(t0, abs=1e-12)


def test_interference_test_size_and_power():
    """Under no spillover but a large heterogeneous direct effect the test
    must not over-reject; with a spillover it must reject often.

    300 replications: the Monte Carlo error of a 5% rejection rate is
    0.0126, so 0.09 is three errors above nominal. Probed rates: 0.04
    under the null and above 0.9 under the alternative below.
    """
    rng = np.random.default_rng(7)
    n = 160
    A = _ring_plus(n)
    base, het = rng.normal(size=n), rng.normal(size=n)
    rej = {0.0: [], 1.5: []}
    for spill in rej:
        for rep in range(300 if spill == 0 else 100):
            Z = (rng.random(n) < 0.4).astype(int)
            # Heterogeneous direct effects under the null (the hard case
            # for size); a homogeneous one under the alternative.
            direct = 3 + 2 * het if spill == 0 else 3.0
            Y = base + direct * Z + spill * ((A @ Z) > 0)
            Y = Y + 0.3 * rng.normal(size=n)
            p = sp.interference_test(Y, Z, A, n_perm=199, seed=rep).pvalue
            rej[spill].append(p <= 0.05)
    assert np.mean(rej[0.0]) <= 0.09
    assert np.mean(rej[1.5]) >= 0.6


def test_interference_test_sharp_null_and_errors():
    rng = np.random.default_rng(8)
    n = 12
    Z = np.array([1] * 6 + [0] * 6)
    Y = rng.normal(size=n) + 3 * Z
    r = sp.interference_test(Y, Z, null="no_effect", n_perm=2000)
    # 924 assignments: enumerated, so this is Fisher's exact p-value.
    assert r.exact and r.n_perm == 924
    diffs = []
    for treated in combinations(range(n), 6):
        z = np.zeros(n, dtype=int)
        z[list(treated)] = 1
        diffs.append(Y[z == 1].mean() - Y[z == 0].mean())
    t0 = Y[Z == 1].mean() - Y[Z == 0].mean()
    assert r.pvalue == pytest.approx(
        np.mean(np.abs(diffs) >= abs(t0) - 1e-12), abs=1e-12
    )
    with pytest.raises(MethodIncompatibility):
        sp.interference_test(Y, Z)  # no adjacency for the spillover null
    with pytest.raises(MethodIncompatibility):
        sp.interference_test(Y, Z, null="other")
    with pytest.raises(DataInsufficient):
        sp.interference_test(Y, np.ones(n, dtype=int), null="no_effect")


# ----------------------------------------------------------------------
# Adaptive experiments
# ----------------------------------------------------------------------


def test_probability_of_being_best():
    """Quadrature against the closed form and against 400,000 draws
    (Monte Carlo error of a probability is at most 0.0008)."""
    from scipy import stats

    two = _prob_best_gaussian(np.array([0.2, 0.0]), np.array([0.3, 0.4]))
    assert two[0] == pytest.approx(stats.norm.cdf(0.2 / 0.5), abs=1e-14)
    rng = np.random.default_rng(9)
    m, s = np.array([0.1, 0.3, 0.0]), np.array([0.2, 0.3, 0.1])
    mc = np.bincount(rng.normal(m, s, size=(400000, 3)).argmax(1)) / 400000
    np.testing.assert_allclose(_prob_best_gaussian(m, s), mc, atol=4e-3)
    a, b = np.array([3.0, 5.0, 2.0]), np.array([4.0, 6.0, 2.0])
    mc = np.bincount(rng.beta(a, b, size=(400000, 3)).argmax(1)) / 400000
    np.testing.assert_allclose(_prob_best_beta(a, b), mc, atol=4e-3)


def test_bandit_allocate_rules():
    df = pd.DataFrame({"arm": list("aabbcc"), "y": [1.0, 1.2, 0.1, 0.3, 0.6, 0.4]})
    p = sp.bandit_allocate(df, "y", "arm", sigma=1.0)
    assert p.sum() == pytest.approx(1.0) and p["a"] > p["c"] > p["b"]
    floor = sp.bandit_allocate(df, "y", "arm", sigma=0.01, prob_floor=0.1)
    assert floor.min() == pytest.approx(0.1) and floor.sum() == pytest.approx(1.0)
    ucb = sp.bandit_allocate(df, "y", "arm", algorithm="ucb", sigma=1.0, horizon=100)
    assert sorted(ucb.tolist()) == [0.0, 0.0, 1.0] and ucb["a"] == 1.0
    eps = sp.bandit_allocate(df, "y", "arm", algorithm="epsilon_greedy", epsilon=0.3)
    np.testing.assert_allclose(eps.loc[["a", "b", "c"]], [0.8, 0.1, 0.1])
    # An untried arm is drawn before any adaptation.
    first = sp.bandit_allocate(df, "y", "arm", arms=["a", "b", "c", "d"])
    assert first["d"] == 1.0
    bern = pd.DataFrame({"arm": list("aabb"), "y": [1, 1, 0, 1]})
    pb = sp.bandit_allocate(bern, "y", "arm", model="bernoulli")
    assert pb["a"] > 0.5
    with pytest.raises(MethodIncompatibility):
        sp.bandit_allocate(df, "y", "arm", model="bernoulli")
    with pytest.raises(MethodIncompatibility):
        sp.bandit_allocate(df, "y", "arm", prob_floor=0.5)


def test_bandit_experiment_regret_and_bookkeeping():
    means = [0.0, 1.0, 0.2]
    kw = dict(n_arms=3, sigma=1.0, true_means=means, seed=0)

    def reward(k, rng):
        return means[k] + rng.normal()

    ts = sp.bandit_experiment(reward, 1500, **kw)
    ucb = sp.bandit_experiment(reward, 1500, algorithm="ucb", **kw)
    uni = sp.bandit_experiment(reward, 1500, algorithm="uniform", **kw)
    # Uniform assignment has regret T * mean gap = 1500 * 0.6; the two
    # adaptive rules must do far better (probed: about 15 and 60).
    assert uni.regret == pytest.approx(900, rel=0.1)
    assert ts.regret < 150 and ucb.regret < 300
    d = ts.data
    np.testing.assert_allclose(d[["prob_0", "prob_1", "prob_2"]].sum(axis=1), 1.0)
    picked = d[["prob_0", "prob_1", "prob_2"]].to_numpy()[np.arange(1500), d["arm"]]
    np.testing.assert_allclose(picked, d["prob"])
    assert ts.arms["pulls"].sum() == 1500
    # A table of potential outcomes replays deterministically given the seed.
    rng = np.random.default_rng(1)
    table = pd.DataFrame(rng.normal(size=(200, 2)) + [0.0, 0.5], columns=["a", "b"])
    one = sp.bandit_experiment(table, prob_floor=0.05, seed=3)
    two = sp.bandit_experiment(table, prob_floor=0.05, seed=3)
    pd.testing.assert_frame_equal(one.data, two.data)
    assert set(one.data["arm"]) == {"a", "b"} and one.data["prob"].min() >= 0.05 - 1e-12
    assert one.regret is not None


def test_adaptive_weighting_matches_the_formula():
    """The estimate and its variance are the two displayed formulas:
    sum(Y/sqrt(e)) / sum(1/sqrt(e)) over the arm's observations and
    sum((Y - est)^2 / e) / (sum(1/sqrt(e)))^2."""
    rng = np.random.default_rng(10)
    T = 50
    arm = rng.integers(0, 2, size=T)
    e = rng.uniform(0.1, 0.9, size=T)
    y = rng.normal(size=T)
    df = pd.DataFrame({"y": y, "arm": arm, "e": e})
    r = sp.adaptive_inference(df, "y", "arm", "e")
    for k in (0, 1):
        s = arm == k
        w = 1 / np.sqrt(e[s])
        est = (w * y[s]).sum() / w.sum()
        var = ((y[s] - est) ** 2 / e[s]).sum() / w.sum() ** 2
        row = r.estimates.iloc[k]
        assert row["estimate"] == pytest.approx(est, abs=1e-13)
        assert row["se"] == pytest.approx(np.sqrt(var), abs=1e-13)
    c = r.contrasts.iloc[0]
    assert c["se"] == pytest.approx(np.hypot(*r.estimates["se"]), abs=1e-13)
    # With a constant probability the weights cancel: the sample mean.
    df["half"] = 0.5
    r2 = sp.adaptive_inference(df, "y", "arm", "half")
    assert r2.estimates["estimate"].iloc[0] == pytest.approx(y[arm == 0].mean())


@pytest.mark.slow
def test_adaptive_inference_restores_coverage_under_thompson_sampling():
    """Two arms with equal means, the hardest case for the sample mean.

    400 experiments of 600 subjects with Thompson sampling and a 2%
    probability floor. Probed with 600 x 1000: the adaptively weighted
    interval covers 95.5%, the augmented one 94.2%, the sample-mean
    interval 92.7% with a negative bias of 0.2 standard errors. The
    Monte Carlo error of a coverage rate over 400 runs is 0.011.
    """
    means = [0.0, 0.0]
    z = {"aw": [], "aipw": [], "mean": []}
    for seed in range(400):
        exp = sp.bandit_experiment(
            lambda k, rng: means[k] + rng.normal(),
            600,
            n_arms=2,
            sigma=1.0,
            prob_floor=0.02,
            min_pulls=1,
            seed=seed,
        )
        for method in z:
            r = sp.adaptive_inference(
                exp.data,
                "reward",
                "arm",
                "prob",
                probs=["prob_0", "prob_1"],
                method=method,
            )
            z[method].append((r.estimates["estimate"] / r.estimates["se"]).iloc[0])
    cover = {m: np.mean(np.abs(v) < 1.96) for m, v in z.items()}
    assert cover["aw"] >= 0.92 and cover["aipw"] >= 0.91
    # The sample mean is biased downwards under Thompson sampling.
    assert np.mean(z["mean"]) < -0.08
    assert abs(np.mean(z["aw"])) < 0.12


def test_adaptive_inference_refuses_what_it_cannot_do():
    rng = np.random.default_rng(11)

    def reward(k, g):
        return [0.0, 0.5][k] + g.normal()

    ucb = sp.bandit_experiment(
        reward, 200, n_arms=2, algorithm="ucb", sigma=1.0, seed=0
    )
    with pytest.raises(MethodIncompatibility, match="deterministic"):
        sp.adaptive_inference(ucb.data, "reward", "arm", "prob")
    ts = sp.bandit_experiment(reward, 200, n_arms=2, sigma=1.0, prob_floor=0.1, seed=0)
    with pytest.raises(MethodIncompatibility, match="probs="):
        sp.adaptive_inference(ts.data, "reward", "arm", "prob", method="aipw")
    with pytest.raises(MethodIncompatibility):
        sp.adaptive_inference(ts.data, "reward", "arm")
    bad = ts.data.assign(prob=0.0)
    with pytest.raises(MethodIncompatibility, match="positive"):
        sp.adaptive_inference(bad, "reward", "arm", "prob")
    ok = sp.adaptive_inference(
        ts.data, "reward", "arm", probs={0: "prob_0", 1: "prob_1"}, method="aipw"
    )
    assert ok.detail["valid_under_adaptivity"]
    assert not sp.adaptive_inference(
        ts.data, "reward", "arm", "prob", method="mean"
    ).detail["valid_under_adaptivity"]
    del rng


# ----------------------------------------------------------------------
# sp.residual_balance
# ----------------------------------------------------------------------


def _sparse_design(rng, n=300, p=120):
    X = rng.normal(size=(n, p))
    e = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.6 * X[:, 1])))
    W = rng.binomial(1, e)
    tau = 1 + 0.5 * X[:, 0]
    Y = 2 * X[:, 0] + X[:, 1] - X[:, 2] + W * tau + rng.normal(size=n)
    cols = [f"x{j}" for j in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["y"], df["w"] = Y, W
    return df, cols, tau


def test_exact_balance_limit_is_interacted_ols():
    """With few covariates, weights allowed to be negative and zeta near
    one, balance is exact and the weighting estimator is the interacted
    regression estimator (the identity noted in the book's chapter 7).
    The agreement is limited by how close zeta = 1 - 1e-9 is to the
    limit, hence 1e-6."""
    rng = np.random.default_rng(12)
    n, p = 200, 3
    X = rng.normal(size=(n, p))
    W = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    Y = 1 + X @ [1.0, -1.0, 0.5] + W * (2 + X[:, 0]) + rng.normal(size=n)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["y"], df["w"] = Y, W
    r = sp.residual_balance(
        df,
        "y",
        "w",
        ["a", "b", "c"],
        outcome_model="none",
        zeta=1 - 1e-9,
        allow_negative_weights=True,
    )
    assert r.model_info["imbalance_treated"] < 1e-8
    lin = sp.lm_lin(df, "y", "w", ["a", "b", "c"])
    assert r.estimate == pytest.approx(lin.estimate, abs=1e-6)


def test_residual_balance_recovers_the_effect_where_weighting_alone_does_not():
    """Sparse linear outcome, p = 120, n = 300, 40 replications.

    The target is the average conditional effect at the observed
    covariates, which is what the standard error is for. Probed over
    150 replications: bias -0.008 (sd 0.138, mean SE 0.135, coverage
    93%); pure weighting is biased by +0.13. Bounds are three Monte
    Carlo errors.
    """
    rng = np.random.default_rng(13)
    err, se, err_w, cover = [], [], [], []
    for _ in range(40):
        df, cols, tau = _sparse_design(rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sp.residual_balance(df, "y", "w", cols, cv=5)
            w = sp.residual_balance(df, "y", "w", cols, outcome_model="none")
        err.append(r.estimate - tau.mean())
        se.append(r.se)
        cover.append(abs(err[-1]) < 1.96 * r.se)
        err_w.append(w.estimate - tau.mean())
    assert abs(np.mean(err)) < 0.07
    assert 0.7 < np.mean(se) / np.std(err) < 1.4
    assert np.mean(cover) >= 0.82
    assert np.mean(err_w) > 0.05


def test_residual_balance_options_and_errors():
    rng = np.random.default_rng(14)
    df, cols, tau = _sparse_design(rng, n=160, p=30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        att = sp.residual_balance(df, "y", "w", cols, estimand="ATT", cv=5)
        lasso = sp.residual_balance(df, "y", "w", cols, outcome_model="lasso", cv=5)
    assert att.estimand == "ATT" and "imbalance_treated" not in att.model_info
    treated = df["w"].to_numpy() == 1
    # Under ATT the treated keep uniform weights.
    np.testing.assert_allclose(att.model_info["weights"][treated], 1 / treated.sum())
    assert att.model_info["weights"][~treated].sum() == pytest.approx(1.0)
    assert np.isfinite(lasso.se) and lasso.se > 0
    # Categorical covariates are expanded like everywhere else.
    df["g"] = pd.Categorical(rng.choice(list("abc"), size=len(df)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat = sp.residual_balance(df, "y", "w", cols[:5] + ["g"], cv=5)
    assert cat.model_info["n_covariates"] == 7
    with pytest.raises(MethodIncompatibility):
        sp.residual_balance(df, "y", "w", cols, zeta=1.0)
    with pytest.raises(MethodIncompatibility):
        sp.residual_balance(df, "y", "w", cols, estimand="LATE")
    with pytest.raises(MethodIncompatibility):
        sp.residual_balance(df, "y", "w", cols, outcome_model="forest")
    three = df.assign(w=rng.integers(0, 3, size=len(df)))
    with pytest.raises(Exception):
        sp.residual_balance(three, "y", "w", cols)
    del tau


# ----------------------------------------------------------------------
# sp.iv.mte: aggregate parameters and their standard errors
# ----------------------------------------------------------------------


def _roy_sample(rng, n):
    """Selection on a uniform resistance U with MTE(u) = 2 - 2u.

    D = 1{P(Z) > U}, so ATE = 1, ATT = 2 - E[P^2]/E[P] and
    ATU = 2 - (1 - E[P^2]) / (1 - E[P]).
    """
    from scipy import stats

    Z = rng.normal(size=n)
    U = rng.random(n)
    P = stats.norm.cdf(0.2 + Z)
    D = (P > U).astype(int)
    Y0 = 1 + 0.5 * (U - 0.5) + 0.3 * rng.normal(size=n)
    Y = np.where(D == 1, Y0 + 2 - 2 * U, Y0)
    return pd.DataFrame({"y": Y, "d": D, "z": Z}), P


def test_mte_att_and_atu_weight_the_whole_sample():
    """With a linear MTE the treated are the units with U < P, so the
    ATT weights each unit by its propensity. Before October 2026 the
    weights were built from the propensity distribution among the
    treated only, which gave 1.22 against a true 1.30 on this design.

    60 samples of 4,000: the standard deviation of each estimate is
    0.023-0.038, so the mean is known to about 0.005; the trimming of
    propensities outside (0.01, 0.99) moves the ATU by about 0.008.
    """
    rng = np.random.default_rng(15)
    from statspai.iv.mte import mte

    rows, P_all = [], []
    for _ in range(60):
        df, P = _roy_sample(rng, 4000)
        m = mte("y", "d", ["z"], data=df, poly_degree=1, propensity_model="probit")
        rows.append(
            (m.ate, m.ate_se, m.att, m.extra["att_se"], m.atu, m.extra["atu_se"])
        )
        P_all.append(P)
    P = np.concatenate(P_all)
    truth = (
        1.0,
        2 - (P**2).mean() / P.mean(),
        2 - (1 - (P**2).mean()) / (1 - P.mean()),
    )
    o = np.array(rows)
    for k, tol in zip(range(3), (0.015, 0.02, 0.025)):
        est, se = o[:, 2 * k], o[:, 2 * k + 1]
        assert abs(est.mean() - truth[k]) < tol
        # The reported standard error tracks the sampling spread; before
        # the fix the ATE standard error was three times too large.
        assert 0.6 < se.mean() / est.std(ddof=1) < 1.5


def test_mte_bootstrap_resamples_aligned_rows():
    """Trimming used to shorten the outcome but not the instrument, so a
    bootstrap draw paired outcomes with other units' instruments and the
    standard errors came out in the hundreds of thousands."""
    from statspai.iv.mte import mte

    rng = np.random.default_rng(16)
    df, _ = _roy_sample(rng, 3000)
    df["z"] = df["z"] * 1.6  # wider propensity range, so rows are trimmed
    kw = dict(data=df, poly_degree=1, propensity_model="probit")
    a = mte("y", "d", ["z"], **kw)
    b = mte("y", "d", ["z"], bootstrap=60, random_state=0, **kw)
    assert a.n_obs < len(df)
    assert b.extra["se_method"] == "bootstrap"
    assert 0.5 < b.ate_se / a.ate_se < 2.5
    assert 0.5 < b.extra["att_se"] / a.extra["att_se"] < 2.5


# ----------------------------------------------------------------------
# R- and DR-learner: conditional effects with the default final stage
# ----------------------------------------------------------------------


def test_r_and_dr_learner_default_final_stage_does_not_chase_outliers():
    """True effects lie between about -2 and 4. With no minimum leaf size
    the default final stage returned fitted effects of -24 and +27 and a
    root-mean-square error of 0.76 (R) and 1.01 (DR), no better than the
    constant (1.03). With the minimum leaf size: 0.46 and 0.59, largest
    absolute value 6 and 13 over three seeds. The bounds leave room for
    another seed and are still far from the old behaviour."""
    rng = np.random.default_rng(100)
    n = 2000
    X = rng.normal(size=(n, 4))
    W = rng.binomial(1, 1 / (1 + np.exp(-0.8 * X[:, 0])))
    tau = 1 + X[:, 0] + 0.5 * (X[:, 1] > 0)
    Y = np.sin(X[:, 0]) + X[:, 2] + W * tau + rng.normal(size=n)
    cols = ["x1", "x2", "x3", "x4"]
    df = pd.DataFrame(X, columns=cols)
    df["y"], df["w"] = Y, W
    limits = {"r": (0.62, 12.0), "dr": (0.80, 20.0)}
    for learner, (rmse_max, abs_max) in limits.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.metalearner(df, "y", "w", cols, learner=learner)
        cate = np.asarray(fit.model_info["cate"])
        assert np.sqrt(np.mean((cate - tau) ** 2)) < rmse_max
        assert np.abs(cate).max() < abs_max
        assert np.corrcoef(cate, tau)[0, 1] > 0.8
        # The average effect does not come from the final stage.
        assert abs(fit.estimate - tau.mean()) < 4 * fit.se


# ----------------------------------------------------------------------
# sp.rd_optimized
# ----------------------------------------------------------------------


def _rd_sample(rng, n=500, M=4.0, discrete=False):
    x = rng.uniform(-1, 1, n)
    if discrete:
        x = (np.floor(x * 8) + 0.5) / 8  # sixteen support points
    mu = np.where(x >= 0, 0.5 * M * x**2, -0.5 * M * x**2)
    y = 1.0 * (x >= 0) + mu + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x})


def test_worst_case_bias_matches_the_closed_form_for_local_linear_weights():
    """For local linear weights the kernel G(t) does not change sign and
    the worst-case bias has the closed form M/2 * |sum w x^2| used by
    RDHonest (and sp.rd_honest, which equals RDHonest to all digits). The
    general formula must reduce to it."""
    from statspai.rd._rdhonest import honest_bias, honest_weights
    from statspai.rd.optimized import worst_case_bias

    rng = np.random.default_rng(20)
    x = np.sort(rng.normal(scale=10, size=1500))
    for h in (4.0, 9.0, 25.0):
        for kernel in ("triangular", "uniform"):
            w = honest_weights(x, 0.0, h, kernel)
            assert worst_case_bias(x, w, 0.1) == pytest.approx(
                honest_bias(w, x, 0.0, 0.1, "H"), rel=1e-10
            )
    # Weights that violate a moment constraint have unbounded bias.
    assert worst_case_bias(x, w * 1.01, 0.1) == np.inf


def test_worst_case_bias_is_attained_by_an_explicit_function():
    """Build the function whose second derivative is M * sign(G) on each
    side, evaluate it at the data and apply the weights. The realised bias
    must equal the reported bound. The function is integrated twice on a
    grid of 200,001 points, which limits the agreement to about 1e-7."""
    rng = np.random.default_rng(21)
    df = _rd_sample(rng, n=300, M=2.0)
    r = sp.rd_optimized(df, "y", "x", M=2.0)
    mi = r.model_info
    g = mi["weights"]
    xo = df.loc[mi["index"], "x"].to_numpy()
    grid = np.linspace(0.0, 1.0, 200001)
    step = grid[1] - grid[0]
    realised = 0.0
    for sign in (1.0, -1.0):
        side = xo >= 0 if sign > 0 else xo < 0
        u, gs = np.abs(xo[side]), sign * g[side]
        G = np.array([np.sum(gs * np.clip(u - t, 0, None)) for t in grid[::50]])
        G = np.interp(grid, grid[::50], G)  # G is piecewise linear in t
        second = 2.0 * np.sign(G)
        mu = np.cumsum(np.cumsum(second) * step) * step
        realised += np.sum(gs * np.interp(u, grid, mu))
    assert realised == pytest.approx(mi["max_bias"], rel=2e-3)
    # The four moment conditions hold, so a level and a slope on either
    # side leave the estimate unchanged.
    right = xo >= 0
    assert g[right].sum() == pytest.approx(1.0, abs=1e-10)
    assert g[~right].sum() == pytest.approx(-1.0, abs=1e-10)
    assert g[right] @ xo[right] == pytest.approx(0.0, abs=1e-10)
    assert g[~right] @ xo[~right] == pytest.approx(0.0, abs=1e-10)
    assert r.estimate == pytest.approx(g @ df.loc[mi["index"], "y"].to_numpy())


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("criterion", ["mse", "flci"])
def test_rd_optimized_is_no_worse_than_local_linear(criterion, discrete):
    """The weights minimise the criterion over all linear estimators, so
    on the same data, with the same M and the same variance estimates,
    they cannot do worse than local linear regression at its own optimal
    bandwidth. The 0.5% slack covers the grid of the search."""
    rng = np.random.default_rng(22)
    df = _rd_sample(rng, n=600, discrete=discrete)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.rd_optimized(df, "y", "x", M=4.0, criterion=criterion)
    mi = r.model_info
    ll = mi["local_linear"]
    if criterion == "flci":
        assert mi["half_length"] <= ll["half_length"] * 1.005
    else:
        assert (
            mi["max_bias"] ** 2 + r.se**2
            <= (ll["max_bias"] ** 2 + ll["se"] ** 2) * 1.005
        )
    assert r.ci[0] < r.estimate < r.ci[1]
    assert mi["critical_value"] >= 1.96
    # The dispatcher reaches the same function.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via = sp.rd(
            df, y="y", x="x", c=0, method="optimized", M=4.0, criterion=criterion
        )
    assert via.estimate == r.estimate


def test_rd_optimized_errors():
    rng = np.random.default_rng(23)
    df = _rd_sample(rng, n=200)
    with pytest.raises(MethodIncompatibility):
        sp.rd_optimized(df, "y", "x", M=4.0, criterion="oci")
    with pytest.raises(MethodIncompatibility):
        sp.rd_optimized(df, "y", "x", M=-1.0)
    with pytest.raises(MethodIncompatibility):
        sp.rd_optimized(df, "y", "x", M=4.0, sigma2=0.0)
    with pytest.raises(DataInsufficient):
        sp.rd_optimized(df[df["x"] > 0], "y", "x", M=4.0)
    two = df.assign(x=np.where(df["x"] >= 0, 0.5, -0.5))
    with pytest.raises(DataInsufficient):
        sp.rd_optimized(two, "y", "x", M=4.0)
