"""Known-truth anchors for functions that had no numerical evidence.

Each test calls one public function and compares its output with an answer
obtained without it. The answers are of four kinds: a closed form (the
inverse logit, the two-arm Thompson probability, lattice sizes); an exact
enumeration (every assignment of a small network, every regular switchback
design of a short horizon); an algebraic identity recomputed here from
numpy and scipy alone (the fitted Bellman equation, the cross-fitted
synthetic control blocks, the exact leave-one-out posterior by quadrature);
and a simulation with a planted answer, where the tolerance is stated in
Monte Carlo standard errors before the numbers are looked at and the seeds
are spaced far apart (``1000 + 100000 k``).
"""

from __future__ import annotations

import itertools
import math
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, signal, stats

import statspai as sp
from statspai.exceptions import DataInsufficient

pytestmark = pytest.mark.filterwarnings("ignore")


def _seeds(n):
    """Seeds far enough apart that no two streams share a neighbourhood."""
    return [1000 + 100000 * k for k in range(n)]


# --------------------------------------------------------------------- #
#  invlogit
# --------------------------------------------------------------------- #


def test_invlogit_is_the_logistic_function():
    # Truth: 1 / (1 + exp(-x)), written in the form that does not overflow
    # on either side (exp(x) / (1 + exp(x)) for x < 0). Both forms are a
    # handful of correctly rounded operations, so 4 ulp is the budget.
    x = np.concatenate([np.linspace(-700.0, 700.0, 2801), [-1e-12, 0.0, 1e-12]])
    pos = x >= 0
    truth = np.empty_like(x)
    truth[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    truth[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    np.testing.assert_allclose(sp.invlogit(x), truth, rtol=4 * np.finfo(float).eps)
    # Exact values: the midpoint, and the two limits, which must be reached
    # without a NaN or an overflow where exp(-x) is not representable.
    assert sp.invlogit(0.0) == 0.5
    assert sp.invlogit(-800.0) == 0.0
    assert sp.invlogit(800.0) == 1.0
    # Inverse of the logit: log(p / (1 - p)) maps back to p. The logit
    # amplifies a rounding error in p by 1 / (p (1 - p)) and the inverse
    # shrinks it by the same factor, so the round trip is exact to a few ulp
    # of p; 1e-15 absolute covers that on (0.001, 0.999).
    p = np.linspace(0.001, 0.999, 999)
    np.testing.assert_allclose(sp.invlogit(np.log(p / (1.0 - p))), p, atol=1e-15)


# --------------------------------------------------------------------- #
#  mixture_design
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("q, m", [(2, 1), (3, 2), (4, 3), (5, 4), (6, 2)])
def test_simplex_lattice_is_every_composition(q, m):
    # Truth: the {q, m} lattice is the set of blends whose shares are
    # multiples of 1 / m, i.e. the compositions of m into q non-negative
    # parts, of which there are C(q + m - 1, m). Exact: the shares times m
    # are integers.
    X = sp.mixture_design(q, degree=m).design.to_numpy()
    counts = np.round(X * m).astype(int)
    assert np.array_equal(counts, X * m)
    assert (counts >= 0).all() and (counts.sum(axis=1) == m).all()
    assert len({tuple(r) for r in counts}) == len(counts) == math.comb(q + m - 1, m)


def test_simplex_lattice_supports_the_polynomial_of_its_degree():
    # Truth: a planted second-degree canonical mixture polynomial (q linear
    # and C(q, 2) cross-product terms, no constant because the shares sum to
    # one) has as many coefficients as the {q, 2} lattice has points, and is
    # recovered exactly from noise-free responses. The model matrix has
    # condition number about 10, so 1e-12 is three orders above rounding.
    q = 4
    X = sp.mixture_design(q, degree=2).design.to_numpy()
    cross = [X[:, i] * X[:, j] for i, j in itertools.combinations(range(q), 2)]
    M = np.column_stack([X] + cross)
    beta = np.random.default_rng(0).normal(size=M.shape[1])
    assert M.shape[0] == M.shape[1]
    np.testing.assert_allclose(np.linalg.solve(M, M @ beta), beta, atol=1e-12)


@pytest.mark.parametrize("q", [2, 3, 5])
def test_simplex_centroid_is_every_equal_blend(q):
    # Truth: one run per non-empty subset of the components, with equal
    # shares on the subset: 2^q - 1 runs, the run of a subset of size s
    # having s entries equal to 1 / s. Exact up to the division.
    X = sp.mixture_design(q, kind="simplex_centroid").design.to_numpy()
    subsets = {tuple(np.flatnonzero(r)) for r in X}
    assert len(subsets) == len(X) == 2**q - 1
    for r in X:
        np.testing.assert_allclose(r[r > 0], 1.0 / np.count_nonzero(r), rtol=1e-15)


def test_mixture_lower_bounds_map_the_lattice_to_the_sub_simplex():
    # Truth: with lower bounds L and total S the design is the lattice in
    # pseudo-components, x = L + (S - sum L) z. For L = (10, 20, 30), S = 100
    # and degree 2, z takes the values 0, 1/2, 1 and x = L + 40 z, so every
    # entry is an integer: exact up to one rounding of 40 z.
    lower = np.array([10.0, 20.0, 30.0])
    X = sp.mixture_design(
        ["a", "b", "c"], degree=2, lower=lower, total=100.0
    ).design.to_numpy()
    z = sp.mixture_design(3, degree=2).design.to_numpy()
    np.testing.assert_allclose(X, lower + 40.0 * z, rtol=1e-14)
    np.testing.assert_allclose(X.sum(axis=1), 100.0, rtol=1e-14)
    assert (X >= lower).all()


# --------------------------------------------------------------------- #
#  sequential_design
# --------------------------------------------------------------------- #


def _branin(d):
    a, b = d["a"].to_numpy(), d["b"].to_numpy()
    curve = b - 5.1 / (4 * math.pi**2) * a**2 + 5 / math.pi * a - 6
    return curve**2 + 10 * (1 - 1 / (8 * math.pi)) * np.cos(a) + 10


@pytest.mark.parametrize("seed", _seeds(3))
def test_sequential_design_finds_a_planted_minimiser(seed):
    # Truth: (x - 0.3)^2 is minimised at x = 0.3. Each step scores at least
    # 2^11 quasi-random candidates on the unit interval, so the minimiser is
    # located up to the candidate spacing 1 / 2048; the tolerance is two
    # spacings (the probe saw 1.2e-5 at worst, because candidates are also
    # drawn near the best runs).
    res = sp.sequential_design(
        lambda d: (d["x"] - 0.3) ** 2, {"x": (0, 1)}, n_new=8, seed=seed
    )
    assert abs(res.best["x"] - 0.3) < 2 / 2048


@pytest.mark.parametrize("seed", _seeds(3))
def test_sequential_design_reaches_the_global_minimum_of_a_known_surface(seed):
    # Truth: on [-5, 10] x [0, 15] the function below has global minimum
    # 10 / (8 pi), attained where the squared term vanishes and cos(a) = -1
    # (a = pi, b = 2.275 among others). The initial design leaves a gap of
    # 1.6 to 4.9 above it; the tolerance is 0.01, i.e. at least 99% of the
    # initial gap closed in 20 runs, ten times the largest gap the probe saw
    # over these seeds (9.7e-4).
    truth = 10 / (8 * math.pi)
    at_minimum = _branin(pd.DataFrame({"a": [math.pi], "b": [2.275]}))[0]
    assert at_minimum == pytest.approx(truth, rel=1e-12)
    res = sp.sequential_design(
        _branin, {"a": (-5, 10), "b": (0, 15)}, n_new=20, seed=seed
    )
    initial = res.design.loc[res.design["stage"] == "initial", "y"].min()
    assert initial - truth > 1.0
    assert 0.0 <= res.best["y"] - truth < 0.01
    # The mirrored problem is the same search with the sign changed.
    up = sp.sequential_design(
        lambda d: -_branin(d),
        {"a": (-5, 10), "b": (0, 15)},
        n_new=20,
        seed=seed,
        goal="maximize",
    )
    assert up.best["y"] == pytest.approx(-res.best["y"], abs=1e-9)


@pytest.mark.parametrize("criterion", ["variance", "alc"])
def test_sequential_design_emulator_reproduces_the_function(criterion):
    # Truth: the function itself, evaluated at 500 fresh points. It is
    # smooth and noise-free, so an interpolating surrogate on 25 runs should
    # leave an error far below the function's spread; the tolerance is a
    # root mean squared error of 1% of the standard deviation of the
    # function over the region (probe: 0.06% to 0.1%).
    def func(d):
        return np.sin(3 * d["a"]) + (d["b"] - 0.5) ** 2

    res = sp.sequential_design(
        func,
        {"a": (0, 3), "b": (0, 1)},
        n_new=15,
        goal="emulate",
        criterion=criterion,
        seed=1000,
    )
    rng = np.random.default_rng(0)
    new = pd.DataFrame({"a": rng.uniform(0, 3, 500), "b": rng.uniform(0, 1, 500)})
    truth = func(new).to_numpy()
    error = res.predict(new)["mean"].to_numpy() - truth
    assert math.sqrt(np.mean(error**2)) < 0.01 * truth.std()


# --------------------------------------------------------------------- #
#  interference_test
# --------------------------------------------------------------------- #


def _small_network(n=12, chords=((1, 6), (3, 10), (5, 9))):
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    for i, j in chords:
        A[i, j] = A[j, i] = 1
    return A


def _assignments(n, m):
    for treated in itertools.combinations(range(n), m):
        z = np.zeros(n, dtype=int)
        z[list(treated)] = 1
        yield z


def test_interference_test_no_effect_pvalue_is_the_exact_tail_share():
    # Truth: under "treatment affects no one" the outcomes are fixed, and
    # the p-value is the share of the C(10, 5) = 252 equally likely
    # assignments whose difference in means is at least as extreme as the
    # realised one. Enumerated here; a share of 252 is exact in floating
    # point up to one division.
    rng = np.random.default_rng(0)
    Z = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    Y = rng.normal(size=10) + Z
    draws = np.array([Y[z == 1].mean() - Y[z == 0].mean() for z in _assignments(10, 5)])
    observed = Y[Z == 1].mean() - Y[Z == 0].mean()
    truth = {
        "two-sided": np.mean(np.abs(draws) >= abs(observed) - 1e-12),
        "greater": np.mean(draws >= observed - 1e-12),
        "less": np.mean(draws <= observed + 1e-12),
    }
    for alternative, share in truth.items():
        res = sp.interference_test(
            Y, Z, null="no_effect", alternative=alternative, n_perm=5000
        )
        assert res.exact and res.n_perm == 252
        assert res.statistic == pytest.approx(observed, rel=1e-13)
        assert res.pvalue == pytest.approx(share, rel=1e-13)


def test_interference_test_no_effect_size_is_the_nominal_level():
    # Truth: with outcomes fixed and 4 of 8 units treated, the difference in
    # means changes sign when the assignment is complemented, so the 70
    # assignments form 35 pairs with a common absolute statistic. With no
    # other ties the two-sided p-values are 2/70, 4/70, ..., 1, each taken
    # by one pair, and the test of level alpha rejects for exactly
    # floor(35 alpha) of the 35 pairs. Exact.
    Y = np.random.default_rng(3).normal(size=8)
    pvalues = np.array(
        [
            sp.interference_test(Y, z, null="no_effect", n_perm=5000).pvalue
            for z in _assignments(8, 4)
        ]
    )
    np.testing.assert_allclose(
        np.sort(pvalues), np.repeat(np.arange(1, 36), 2) / 35, rtol=1e-13
    )
    for alpha in (0.05, 0.10, 0.20, 0.50):
        assert np.sum(pvalues <= alpha + 1e-12) == 2 * math.floor(35 * alpha + 1e-9)


def test_interference_test_no_spillover_pvalue_is_the_exact_tail_share():
    # Truth: with the focal units and their treatments held fixed, the
    # admissible assignments are the C(5, k) placements of the k treated
    # non-focal units; the p-value is the share of them whose statistic (the
    # coefficient on the share of treated neighbours in a regression of
    # focal outcomes on own treatment and that share, recomputed here) is at
    # least as extreme as the realised one. Exact up to the least squares
    # solve: 1e-9 on the statistic, and the p-value is a ratio of counts.
    A = _small_network(10, chords=((0, 5), (2, 7)))
    rng = np.random.default_rng(0)
    Z = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    Y = rng.normal(size=10) + Z + 0.8 * (A @ Z)
    focal = np.array([0, 2, 4, 6, 8])
    rest = np.setdiff1d(np.arange(10), focal)

    def statistic(z):
        share = (A @ z) / A.sum(axis=1)
        X = np.column_stack([np.ones(focal.size), z[focal], share[focal]])
        return np.linalg.lstsq(X, Y[focal], rcond=None)[0][-1]

    draws = []
    for treated in itertools.combinations(rest, int(Z[rest].sum())):
        z = Z.copy()
        z[rest] = 0
        z[list(treated)] = 1
        draws.append(statistic(z))
    observed = statistic(Z)
    share = np.mean(np.abs(draws) >= abs(observed) - 1e-12)
    res = sp.interference_test(Y, Z, A, focal=focal, n_perm=5000)
    assert res.exact and res.n_perm == len(draws)
    assert res.statistic == pytest.approx(observed, rel=1e-9)
    assert res.pvalue == pytest.approx(share, rel=1e-13)


_A12 = _small_network()
_UNIT = np.random.default_rng(3).normal(size=(3, 12))
_DEG = _A12.sum(axis=1)


@pytest.mark.parametrize(
    "null, focal, outcome",
    [
        # own treatment only
        ("no_spillover", [0, 2, 5, 7, 9, 11], lambda z: _UNIT[0] + _UNIT[1] * z),
        # own treatment and the neighbours', through two different channels
        (
            "no_higher_order",
            [0, 4],
            lambda z: _UNIT[0]
            + _UNIT[1] * z
            + _UNIT[2] * (_A12 @ z) / _DEG
            + 0.5 * z * (_A12 @ z),
        ),
        # own treatment and the share of treated neighbours, non-linearly
        (
            "anonymous",
            [0, 4, 8],
            lambda z: _UNIT[0] + _UNIT[1] * z + _UNIT[2] * ((_A12 @ z) / _DEG) ** 2,
        ),
    ],
)
def test_interference_test_is_exactly_valid_under_each_null(null, focal, outcome):
    # Truth: finite-sample validity. Potential outcomes are built to satisfy
    # the null and to violate every stricter one; the design treats 6 of 12
    # units, all C(12, 6) = 924 assignments equally likely. For each of them
    # the test is run on the outcomes that assignment would produce, and the
    # share of assignments with p <= alpha must not exceed alpha for any
    # alpha. Checked at every p-value the test can return, which covers all
    # alpha. Exact, no simulation; an assignment that leaves nothing to
    # permute cannot reject.
    pvalues = []
    for z in _assignments(12, 6):
        try:
            res = sp.interference_test(
                outcome(z), z, _A12, null=null, focal=np.array(focal), n_perm=10**6
            )
            assert res.exact
            pvalues.append(res.pvalue)
        except DataInsufficient:
            pvalues.append(np.inf)
    pvalues = np.array(pvalues)
    levels = np.unique(pvalues[np.isfinite(pvalues)])
    assert levels.size >= 4 and levels.min() < 0.3
    for alpha in levels:
        assert np.mean(pvalues <= alpha + 1e-12) <= alpha + 1e-12


# --------------------------------------------------------------------- #
#  bandit_allocate
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("sigma", [1.0, 2.5, None])
def test_two_arm_thompson_probability_is_a_normal_cdf(sigma):
    # Truth: with flat priors the posteriors of the two means are
    # independent N(ybar_k, s^2 / n_k), so the probability that arm a has
    # the larger mean is Phi((ybar_a - ybar_b) / (s sqrt(1/n_a + 1/n_b))),
    # s being the known sigma or the pooled within-arm standard deviation.
    # Closed form: 1e-13 covers the rounding of the mean and the CDF.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"arm": ["a"] * 5 + ["b"] * 8, "y": rng.normal(size=13)})
    ya, yb = df.y[df.arm == "a"].to_numpy(), df.y[df.arm == "b"].to_numpy()
    if sigma is None:
        rss = ((ya - ya.mean()) ** 2).sum() + ((yb - yb.mean()) ** 2).sum()
        s = math.sqrt(rss / (13 - 2))
    else:
        s = sigma
    truth = stats.norm.cdf((ya.mean() - yb.mean()) / (s * math.sqrt(1 / 5 + 1 / 8)))
    p = sp.bandit_allocate(df, "y", "arm", sigma=sigma)
    assert p["a"] == pytest.approx(truth, abs=1e-13)
    assert p["b"] == pytest.approx(1.0 - truth, abs=1e-13)


def test_three_arm_thompson_probability_is_the_integral_it_stands_for():
    # Truth: P(arm k best) = integral of the density of arm k's posterior
    # times the CDFs of the other two, evaluated by adaptive quadrature
    # (scipy.integrate.quad, tolerance 1e-13). The function uses a fixed
    # 80-node Gauss-Hermite rule on a smooth integrand; 1e-10 is the budget
    # for that rule (probe: 6e-17).
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"arm": list("abc") * 6, "y": rng.normal(size=18) + np.tile([0, 0.3, 0.6], 6)}
    )
    mean = df.groupby("arm")["y"].mean().to_numpy()
    sd = 1.0 / math.sqrt(6)
    truth = []
    for k in range(3):
        others = [j for j in range(3) if j != k]

        def integrand(x, k=k, others=others):
            cdfs = [stats.norm.cdf(x, mean[j], sd) for j in others]
            return stats.norm.pdf(x, mean[k], sd) * cdfs[0] * cdfs[1]

        truth.append(
            integrate.quad(integrand, -10, 10, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
        )
    p = sp.bandit_allocate(df, "y", "arm", sigma=1.0)
    np.testing.assert_allclose(p.to_numpy(), truth, atol=1e-10)
    # Exchangeable arms: identical data in every arm give 1 / 3 each.
    same = pd.DataFrame({"arm": list("abc") * 4, "y": np.repeat([0.1, 0.7, 0.2, 1], 3)})
    np.testing.assert_allclose(
        sp.bandit_allocate(same, "y", "arm", sigma=1.0).to_numpy(), 1 / 3, atol=1e-12
    )


def _beta_fn(a, b):
    return Fraction(
        math.factorial(a - 1) * math.factorial(b - 1), math.factorial(a + b - 1)
    )


def test_bernoulli_thompson_probability_is_an_exact_rational():
    # Truth: with uniform priors the posteriors are Beta(1 + s, 1 + f).
    # For integer parameters P(Y < x) for Y ~ Beta(c, d) is the binomial
    # tail sum_{j >= c} C(n, j) x^j (1 - x)^(n - j) with n = c + d - 1, and
    # E[X^j (1 - X)^(n - j)] for X ~ Beta(a, b) is B(a + j, b + n - j) /
    # B(a, b); so P(X > Y) is a finite sum of ratios of factorials, computed
    # here in exact rational arithmetic. The function integrates with 400
    # Gauss-Legendre nodes, exact for these polynomial integrands: 1e-12.
    df = pd.DataFrame(
        {
            "arm": ["a"] * 7 + ["b"] * 9,
            "y": [1, 1, 0, 1, 0, 1, 1] + [0, 1, 0, 0, 1, 0, 0, 1, 0],
        }
    )
    a, b = 1 + 5, 1 + 2  # arm a: 5 successes, 2 failures
    c, d = 1 + 3, 1 + 6  # arm b: 3 successes, 6 failures
    n = c + d - 1
    truth = sum(
        math.comb(n, j) * _beta_fn(a + j, b + n - j) / _beta_fn(a, b)
        for j in range(c, n + 1)
    )
    p = sp.bandit_allocate(df, "y", "arm", model="bernoulli")
    assert p["a"] == pytest.approx(float(truth), abs=1e-12)
    assert p["b"] == pytest.approx(float(1 - truth), abs=1e-12)


def test_deterministic_allocation_rules_and_the_floor():
    # Truth, all exact by definition of the rule. Epsilon-greedy: epsilon / K
    # on every arm plus 1 - epsilon on the arm with the largest mean. Upper
    # confidence bound: probability one on the largest
    # mean + c * sigma * sqrt(log(horizon) / n). Floor: a two-arm
    # probability below the floor is raised to it and the other arm gets
    # the rest.
    df = pd.DataFrame(
        {
            "arm": list("abc") * 2 + ["c"] * 6,
            "y": [0.0, 1.0, 0.8, 0.2, 1.2, 0.9] + [1.0] * 6,
        }
    )
    # means: a 0.1, b 1.1, c 0.9625; counts 2, 2, 8
    p = sp.bandit_allocate(df, "y", "arm", algorithm="epsilon_greedy", epsilon=0.3)
    np.testing.assert_allclose(p.to_numpy(), [0.1, 0.8, 0.1], rtol=1e-15)
    index = np.array([0.1, 1.1, 0.9625]) + 2.0 * 1.0 * np.sqrt(
        math.log(50) / np.array([2, 2, 8])
    )
    assert index.argmax() == 1
    p = sp.bandit_allocate(df, "y", "arm", algorithm="ucb", sigma=1.0, horizon=50)
    assert p.to_numpy().tolist() == [0.0, 1.0, 0.0]
    two = df[df.arm != "c"]
    raw = sp.bandit_allocate(two, "y", "arm", sigma=0.2)
    assert raw["a"] < 1e-6
    floored = sp.bandit_allocate(two, "y", "arm", sigma=0.2, prob_floor=0.1)
    np.testing.assert_allclose(floored.to_numpy(), [0.1, 0.9], rtol=1e-15)


# --------------------------------------------------------------------- #
#  bandit_experiment
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rule",
    [
        dict(algorithm="thompson", sigma=1.0),
        dict(algorithm="thompson"),
        dict(algorithm="thompson", model="bernoulli"),
        dict(algorithm="thompson", sigma=1.0, prob_floor=0.1),
        dict(algorithm="epsilon_greedy", epsilon=0.2),
    ],
)
def test_bandit_experiment_records_the_probabilities_of_its_rule(rule):
    # Truth: the probabilities recorded for period t are those the rule
    # assigns given the first t - 1 rows, i.e. sp.bandit_allocate on that
    # history (itself checked against closed forms above). The same
    # arithmetic on the same sums in a different order: 1e-12. Regret is the
    # sum over periods of the best mean minus the mean of the arm played.
    means = np.array([0.2, 0.5, 0.4])
    if rule.get("model") == "bernoulli":

        def draw(k, rng):
            return float(rng.random() < means[k])

    else:

        def draw(k, rng):
            return means[k] + rng.normal()

    T = 80
    exp = sp.bandit_experiment(draw, T, n_arms=3, true_means=means, seed=3, **rule)
    data = exp.data
    cols = ["prob_0", "prob_1", "prob_2"]
    for t in range(T):
        truth = sp.bandit_allocate(
            data.iloc[:t], "reward", "arm", arms=[0, 1, 2], horizon=T, **rule
        )
        np.testing.assert_allclose(
            data.loc[t, cols].to_numpy(dtype=float), truth.to_numpy(), atol=1e-12
        )
    played = data["arm"].to_numpy()
    recorded = data[cols].to_numpy()
    assert np.array_equal(data["prob"].to_numpy(), recorded[np.arange(T), played])
    assert exp.regret == pytest.approx((means.max() - means[played]).sum(), rel=1e-12)


def test_bandit_experiment_reads_rewards_from_the_potential_outcome_table():
    # Truth: with a table of potential outcomes the reward of period t is
    # the entry of the arm played, exactly, and regret is measured against
    # the column means.
    table = np.random.default_rng(0).normal(size=(50, 2)) + [0.0, 1.0]
    exp = sp.bandit_experiment(table, sigma=1.0, seed=1)
    played = exp.data["arm"].to_numpy()
    assert np.array_equal(exp.data["reward"].to_numpy(), table[np.arange(50), played])
    mu = table.mean(axis=0)
    assert exp.regret == pytest.approx((mu.max() - mu[played]).sum(), rel=1e-12)


def test_bandit_experiment_draws_arms_with_the_recorded_probabilities():
    # Truth: if arm 1 is drawn in period t with the recorded probability
    # p_t, then sum_t (1{arm_t = 1} - p_t) is a martingale with variance
    # sum_t p_t (1 - p_t), and the ratio Z is standard normal up to the
    # martingale central limit theorem (T = 60, floor 0.05). Over S = 300
    # experiments: |mean Z| <= 3 / sqrt(S) and |sd Z - 1| <= 3 / sqrt(2 S),
    # three Monte Carlo standard errors each (probe at S = 400: -0.018 and
    # 0.981).
    S, T = 300, 60
    z = np.empty(S)
    for i, seed in enumerate(_seeds(S)):
        data = sp.bandit_experiment(
            lambda k, rng: [0.0, 0.3][k] + rng.normal(),
            T,
            n_arms=2,
            sigma=1.0,
            prob_floor=0.05,
            seed=seed,
        ).data
        p1 = data["prob_1"].to_numpy()
        z[i] = ((data["arm"].to_numpy() == 1) - p1).sum() / math.sqrt(
            (p1 * (1 - p1)).sum()
        )
    assert abs(z.mean()) <= 3 / math.sqrt(S)
    assert abs(z.std(ddof=1) - 1.0) <= 3 / math.sqrt(2 * S)


# --------------------------------------------------------------------- #
#  contextual_bandit
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "options",
    [
        dict(sigma=1.0),
        dict(sigma=1.0, prob_floor=0.05, prior_precision=2.0),
        dict(sigma=0.7, batch_size=5),
    ],
)
def test_contextual_bandit_probabilities_are_the_posterior_probabilities(options):
    # Truth: each arm has a Gaussian linear model with prior N(0, I / lam)
    # on the coefficients and known error sd s. Given the rows of arm k up
    # to the last update, the posterior is N(b_k, s^2 V_k) with
    # V_k = (lam I + X_k'X_k)^-1 and b_k = V_k X_k'y_k, so the expected
    # reward at x is N(x'b_k, s^2 x'V_k x) and, with two arms, arm 0 is
    # assigned with probability Phi((x'b_0 - x'b_1) / sqrt(v_0 + v_1)),
    # clipped to the floor. Recomputed here from the recorded rows by
    # ordinary linear algebra; 3 x 3 well-conditioned systems, so 1e-12.
    rng = np.random.default_rng(5)
    T = 120
    X = rng.normal(size=(T, 2))
    exp = sp.contextual_bandit(
        lambda k, x, g: (x[0] if k == 1 else -x[0]) + 0.5 * x[1] + g.normal(),
        X,
        n_arms=2,
        seed=2,
        **options,
    )
    lam = options.get("prior_precision", 1.0)
    s = options["sigma"]
    batch = options.get("batch_size", 1)
    floor = options.get("prob_floor", 0.0)
    D = np.column_stack([np.ones(T), X])
    arm = exp.data["arm"].to_numpy()
    reward = exp.data["reward"].to_numpy()

    def posterior(rows):
        V = np.linalg.inv(lam * np.eye(3) + D[rows].T @ D[rows])
        return V @ D[rows].T @ reward[rows], V

    truth = np.empty(T)
    for t in range(T):
        seen = (t // batch) * batch
        fits = [posterior(np.flatnonzero(arm[:seen] == k)) for k in range(2)]
        mean = [D[t] @ b for b, _ in fits]
        var = [s**2 * D[t] @ V @ D[t] for _, V in fits]
        p0 = stats.norm.cdf((mean[0] - mean[1]) / math.sqrt(var[0] + var[1]))
        truth[t] = min(max(p0, floor), 1.0 - floor)
    np.testing.assert_allclose(exp.data["prob_0"].to_numpy(), truth, atol=1e-12)
    np.testing.assert_allclose(exp.data["prob_1"].to_numpy(), 1 - truth, atol=1e-12)
    # The reported coefficients are the posterior means on all the data.
    for k in range(2):
        b, _ = posterior(np.flatnonzero(arm == k))
        np.testing.assert_allclose(
            exp.detail["coefficients"].loc[k].to_numpy(), b, atol=1e-12
        )


# --------------------------------------------------------------------- #
#  adaptive_inference
# --------------------------------------------------------------------- #


def test_adaptive_weights_reduce_to_the_sample_mean_under_a_fixed_design():
    # Truth: with a constant assignment probability the square-root weights
    # are constant, so the weighted mean of an arm is its sample mean and
    # the variance sum w^2 (y - est)^2 / (sum w)^2 is
    # sum (y - ybar)^2 / n^2. Algebraic identity: 1e-13.
    rng = np.random.default_rng(1)
    arm = rng.integers(0, 3, 200)
    y = rng.normal(size=200) + arm
    df = pd.DataFrame({"y": y, "arm": arm, "p": 1 / 3})
    res = sp.adaptive_inference(df, "y", "arm", "p")
    var = []
    for k in range(3):
        yk = y[arm == k]
        var.append(((yk - yk.mean()) ** 2).sum() / yk.size**2)
        assert res.estimates["estimate"][k] == pytest.approx(yk.mean(), abs=1e-13)
        assert res.estimates["se"][k] == pytest.approx(math.sqrt(var[k]), abs=1e-13)
    # A contrast is the difference of two arms and its variance their sum.
    row = res.contrasts.iloc[1]
    assert row["estimate"] == pytest.approx(
        y[arm == 2].mean() - y[arm == 0].mean(), abs=1e-13
    )
    assert row["se"] == pytest.approx(math.sqrt(var[2] + var[0]), abs=1e-13)


def test_adaptive_inference_is_calibrated_after_thompson_sampling():
    # Truth: arm means 0 and 0.5 (planted), unit-variance normal rewards,
    # T = 400, Thompson sampling with a probability floor of 0.05. The claim
    # of the method is that the studentised contrast (estimate - 0.5) / se
    # is asymptotically standard normal although assignment depended on
    # earlier outcomes. Over S = 400 experiments, three Monte Carlo standard
    # errors each: |mean Z| <= 3 / sqrt(S) = 0.15, |sd Z - 1| <=
    # 3 / sqrt(2 S) = 0.106, and the coverage of the 95% interval within
    # 3 sqrt(0.05 * 0.95 / S) = 0.033 of 0.95. (Probe at S = 600: 0.040,
    # 1.021, 0.945.) The estimate itself is not unbiased in finite samples,
    # which is why the statistic and not the bias is tested.
    S, T, mu = 400, 400, [0.0, 0.5]
    z = np.empty(S)
    for i, seed in enumerate(_seeds(S)):
        data = sp.bandit_experiment(
            lambda k, rng: mu[k] + rng.normal(),
            T,
            n_arms=2,
            sigma=1.0,
            prob_floor=0.05,
            seed=seed,
        ).data
        row = sp.adaptive_inference(data, "reward", "arm", "prob").contrasts.iloc[0]
        z[i] = (row["estimate"] - 0.5) / row["se"]
    assert abs(z.mean()) <= 3 / math.sqrt(S)
    assert abs(z.std(ddof=1) - 1.0) <= 3 / math.sqrt(2 * S)
    cover = np.mean(np.abs(z) < stats.norm.ppf(0.975))
    assert abs(cover - 0.95) <= 3 * math.sqrt(0.05 * 0.95 / S)


# --------------------------------------------------------------------- #
#  mdp_policy_value
# --------------------------------------------------------------------- #

# Three states, two actions. P[a, s] is the distribution of the next state
# and R[s, a] the mean reward.
_P = np.array(
    [
        [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]],
        [[0.2, 0.3, 0.5], [0.1, 0.3, 0.6], [0.1, 0.2, 0.7]],
    ]
)
_R = np.array([[0.0, 1.0], [0.5, 0.2], [2.0, 1.0]])
_E1 = np.array([0.3, 0.5, 0.7])  # probability of action 1 in the data, by state
_TARGET = np.array([0, 0, 1])  # the policy evaluated: act only in state 2


def _stationary_mean(transition, reward):
    """Long-run average reward of a chain: stationary distribution times reward."""
    k = transition.shape[0]
    lhs = np.vstack([transition.T - np.eye(k), np.ones(k)])
    rhs = np.r_[np.zeros(k), 1.0]
    return float(np.linalg.lstsq(lhs, rhs, rcond=None)[0] @ reward)


def _trajectory(T, seed, e1=_E1):
    rng = np.random.default_rng(seed)
    s, rows = 0, []
    for _ in range(T):
        a = int(rng.random() < e1[s])
        rows.append((s, a, _R[s, a] + rng.normal(), e1[s] if a else 1 - e1[s]))
        s = int(rng.choice(3, p=_P[a, s]))
    return pd.DataFrame(rows, columns=["s", "a", "y", "e"])


def _fitted_model(df, target, weight):
    """Transition matrix and mean reward under `target`, fitted by weighted
    frequencies over the transitions on which the action agrees with it."""
    s, a, y = df["s"].to_numpy(), df["a"].to_numpy(), df["y"].to_numpy()
    cur = np.arange(len(df) - 1)
    w = np.where(a[cur] == target[s[cur]], weight[cur], 0.0)
    transition, reward = np.zeros((3, 3)), np.zeros(3)
    for i in range(3):
        here = s[cur] == i
        for j in range(3):
            transition[i, j] = w[here & (s[cur + 1] == j)].sum() / w[here].sum()
        reward[i] = (w[here] * y[cur][here]).sum() / w[here].sum()
    return transition, reward


@pytest.mark.parametrize("known_propensity", [True, False])
def test_mdp_policy_value_is_the_value_of_the_fitted_model(known_propensity):
    # Truth: with one indicator per state the estimate equals (i) the
    # stationary mean reward of the transition model fitted to the data and
    # (ii) the constant eta solving the fitted Bellman equation
    # eta + Q(s) - sum_s' P(s'|s) Q(s') = r(s). Both are recomputed here
    # from weighted frequencies. Linear algebra on 3 x 3 systems: 1e-10.
    df = _trajectory(3000, 7)
    if known_propensity:
        weight = 1.0 / df["e"].to_numpy()
        propensity = "e"
    else:
        cell = df.groupby(["s", "a"])["a"].transform("size")
        weight = (df.groupby("s")["a"].transform("size") / cell).to_numpy()
        propensity = None
    res = sp.mdp_policy_value(
        df,
        "y",
        "a",
        ["s"],
        policy=lambda S: (S["s"].to_numpy() == 2).astype(int),
        propensity=propensity,
    )
    transition, reward = _fitted_model(df, _TARGET, weight)
    assert res.value == pytest.approx(_stationary_mean(transition, reward), abs=1e-10)
    bellman = np.column_stack([np.ones(3), (np.eye(3) - transition)[:, 1:]])
    assert res.value == pytest.approx(np.linalg.solve(bellman, reward)[0], abs=1e-10)
    # A contrast of two policies is the difference of their two values.
    both = sp.mdp_policy_value(
        df, "y", "a", ["s"], policy=1, baseline=0, propensity=propensity
    )
    values = [
        _stationary_mean(*_fitted_model(df, np.full(3, action), weight))
        for action in (1, 0)
    ]
    assert both.value_policy == pytest.approx(values[0], abs=1e-10)
    assert both.value_baseline == pytest.approx(values[1], abs=1e-10)
    assert both.value == pytest.approx(values[0] - values[1], abs=1e-10)


def test_mdp_policy_value_recovers_the_stationary_mean_of_the_true_chain():
    # Truth: the long-run value of the policy in the chain that generated
    # the data, from the true transition matrix (0.5735...). Over S = 200
    # trajectories of 1,500 periods, three Monte Carlo standard errors
    # each: |bias| <= 3 sd / sqrt(S); mean reported se within
    # 3 / sqrt(2 S) = 15% of the across-seed sd; coverage of the 95%
    # interval within 3 sqrt(0.05 * 0.95 / S) = 0.046 of 0.95. (Probe: bias
    # 1.4 Monte Carlo se, ratio 1.02, coverage 0.95.)
    truth = _stationary_mean(
        np.array([_P[_TARGET[s], s] for s in range(3)]),
        np.array([_R[s, _TARGET[s]] for s in range(3)]),
    )
    S = 200
    est, se = np.empty(S), np.empty(S)
    for i, seed in enumerate(_seeds(S)):
        res = sp.mdp_policy_value(
            _trajectory(1500, seed),
            "y",
            "a",
            ["s"],
            policy=lambda X: (X["s"].to_numpy() == 2).astype(int),
            propensity="e",
        )
        est[i], se[i] = res.value, res.se
    sd = est.std(ddof=1)
    assert abs(est.mean() - truth) <= 3 * sd / math.sqrt(S)
    assert abs(se.mean() / sd - 1.0) <= 3 / math.sqrt(2 * S)
    cover = np.mean(np.abs(est - truth) < stats.norm.ppf(0.975) * se)
    assert abs(cover - 0.95) <= 3 * math.sqrt(0.05 * 0.95 / S)


# --------------------------------------------------------------------- #
#  marginal_policy_effect
# --------------------------------------------------------------------- #


def _carryover_series(n, seed, share=0.3):
    """y_t = w_t + u_t + noise with u_t = 0.5 u_{t-1} + w_{t-1} + noise: one
    treatment adds 1 now, 1 next period, then 0.5, 0.25, ..."""
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, share, n)
    shock = np.r_[0.0, w[:-1]] + rng.normal(size=n)
    shock[0] = 0.0
    u = signal.lfilter([1.0], [1.0, -0.5], shock)
    return pd.DataFrame({"y": w + u + rng.normal(size=n), "w": w})


@pytest.mark.parametrize(
    "horizon, per_treatment", [(0, 1.0), (1, 2.0), (2, 2.5), (4, 2.875)]
)
def test_marginal_policy_effect_recovers_the_planted_forward_effect(
    horizon, per_treatment
):
    # Truth: treatment is randomised with probability 0.3 and one treatment
    # raises the sum of the outcomes of the current and the next `horizon`
    # periods by 1 + (1 + 0.5 + ... + 0.5^(horizon - 1)), so
    # theta = 0.3 * that. Over S = 300 series of 2,000 periods, three Monte
    # Carlo standard errors each: |bias| <= 3 sd / sqrt(S), and the mean
    # reported (HAC) se within 3 / sqrt(2 S) = 12.2% of the across-seed sd.
    # (Probe: bias 0.2 to 0.4 Monte Carlo se; ratio 0.97 to 1.06.)
    truth = 0.3 * per_treatment
    S = 300
    est, se = np.empty(S), np.empty(S)
    for i, seed in enumerate(_seeds(S)):
        res = sp.marginal_policy_effect(
            _carryover_series(2000, seed),
            "y",
            "w",
            [],
            horizon=horizon,
            propensity=0.3,
        )
        est[i], se[i] = res.estimate, res.se
    sd = est.std(ddof=1)
    assert abs(est.mean() - truth) <= 3 * sd / math.sqrt(S)
    assert abs(se.mean() / sd - 1.0) <= 3 / math.sqrt(2 * S)


# --------------------------------------------------------------------- #
#  synth_ttest
# --------------------------------------------------------------------- #


def _factor_panel(seed, T0=60, T1=20, n_donors=8, effect=2.0, noise=1.0):
    """Treated unit = 0.5, 0.3, 0.2 mix of the first three donors, plus
    noise, plus `effect` from period T0 on."""
    rng = np.random.default_rng(seed)
    T = T0 + T1
    Y0 = rng.normal(size=(T, 3)) @ rng.normal(size=(n_donors, 3)).T + 5.0
    Y0 = Y0 + noise * rng.normal(size=(T, n_donors))
    y1 = Y0[:, :3] @ np.array([0.5, 0.3, 0.2]) + noise * rng.normal(size=T)
    y1[T0:] += effect
    rows = [(0, t, y1[t]) for t in range(T)]
    rows += [(j + 1, t, Y0[t, j]) for j in range(n_donors) for t in range(T)]
    return pd.DataFrame(rows, columns=["unit", "time", "y"]), y1, Y0


_SYNTH = dict(outcome="y", unit="unit", time="time", treated_unit=0)


def _simplex_least_squares(y, X):
    J = X.shape[1]
    fit = optimize.minimize(
        lambda w: ((y - X @ w) ** 2).sum(),
        np.full(J, 1.0 / J),
        jac=lambda w: -2 * X.T @ (y - X @ w),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * J,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    return fit.x


def test_synth_ttest_is_the_cross_fitted_formula():
    # Truth: the definition, recomputed with another solver (scipy SLSQP).
    # The last K r pre-periods form K blocks; weights are fitted without a
    # block, tau_k is the post-period gap minus the gap on the block, the
    # estimate is their mean, the se sqrt(1 + K r / T1) sd(tau) / sqrt(K)
    # and the interval uses t with K - 1 degrees of freedom. Two solvers of
    # one strictly convex programme agree to their stopping tolerances:
    # 1e-8 (probe: 2e-14 on the estimate, 6e-13 on the se).
    T0, T1, K = 60, 20, 3
    df, y1, Y0 = _factor_panel(1)
    res = sp.synth_ttest(df, treatment_time=T0, n_folds=K, **_SYNTH)
    r = min(T0 // K, T1)
    tau = []
    for k in range(K):
        hold = np.arange(T0 - r * K + k * r, T0 - r * K + (k + 1) * r)
        keep = np.setdiff1d(np.arange(T0), hold)
        w = _simplex_least_squares(y1[keep], Y0[keep])
        gap = y1 - Y0 @ w
        tau.append(gap[T0:].mean() - gap[hold].mean())
    est = np.mean(tau)
    se = math.sqrt(1 + K * r / T1) * np.std(tau, ddof=1) / math.sqrt(K)
    crit = stats.t.ppf(0.975, K - 1)
    np.testing.assert_allclose(res.detail["tau"].to_numpy(), tau, atol=1e-8)
    assert res.estimate == pytest.approx(est, abs=1e-8)
    assert res.se == pytest.approx(se, abs=1e-8)
    assert res.ci == pytest.approx((est - crit * se, est + crit * se), abs=1e-7)
    assert res.pvalue == pytest.approx(2 * stats.t.sf(abs(est / se), K - 1), abs=1e-8)


def test_synth_ttest_is_exact_without_noise():
    # Truth: when the treated unit is exactly a convex combination of donors
    # every block's weights reproduce it, each gap is zero before treatment
    # and equal to the effect after, so every tau_k is the effect (2.0) and
    # their spread is zero. Limited by the weight solver: 1e-10 (probe:
    # 2e-15).
    df, _, _ = _factor_panel(2, noise=0.0)
    res = sp.synth_ttest(df, treatment_time=60, **_SYNTH)
    np.testing.assert_allclose(res.detail["tau"].to_numpy(), 2.0, atol=1e-10)
    assert res.estimate == pytest.approx(2.0, abs=1e-10)
    assert res.se < 1e-10


def test_synth_ttest_recovers_a_planted_effect_with_a_calibrated_interval():
    # Truth: effect 2.0 (planted), 60 pre- and 20 post-periods, K = 3. Over
    # S = 400 panels: |bias| <= 3 sd / sqrt(S), and the coverage of the 95%
    # t interval within 3 sqrt(0.05 * 0.95 / S) = 0.033 of 0.95; the
    # reference distribution is an approximation that needs long pre- and
    # post-periods, which this design gives it. (Probe: bias 1.0 Monte Carlo
    # se, coverage 0.955.)
    S = 400
    est, cover = np.empty(S), np.empty(S, dtype=bool)
    for i, seed in enumerate(_seeds(S)):
        df, _, _ = _factor_panel(seed)
        res = sp.synth_ttest(df, treatment_time=60, **_SYNTH)
        est[i] = res.estimate
        cover[i] = res.ci[0] < 2.0 < res.ci[1]
    assert abs(est.mean() - 2.0) <= 3 * est.std(ddof=1) / math.sqrt(S)
    assert abs(cover.mean() - 0.95) <= 3 * math.sqrt(0.05 * 0.95 / S)


# --------------------------------------------------------------------- #
#  loo_predict
# --------------------------------------------------------------------- #


def test_loo_predict_matches_the_exact_leave_one_out_posterior_mean():
    # Truth: for y ~ N(X b, s2), b ~ N(0, V I), s2 ~ InvGamma(A0/2, D0/2),
    # the posterior of b without observation i, given s2, is normal with
    # mean (X'X / s2 + I / V)^-1 X'y / s2, and the posterior of s2 is
    # proportional to its prior times N(y; 0, s2 I + V X X'). The expected
    # outcome of observation i under that posterior is a one-dimensional
    # integral over s2, done here on a grid in log s2 (written from
    # scipy.stats, sharing no code with the sampler or with the importance
    # weights). The function reweights S posterior draws instead, so it
    # carries Monte Carlo error: the tolerance is 4 standard errors, with
    # standard error = posterior sd of the prediction / sqrt(effective
    # sample size of the weights for that observation). (Probe: 2.5 at
    # worst over the 30 observations.)
    V, A0, D0, N = 10.0, 3.0, 2.0, 30
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"x": rng.normal(size=N)})
    df.loc[0, "x"] = 3.5  # one point with high leverage
    df["y"] = 0.3 + 0.7 * df["x"] + 0.8 * rng.normal(size=N)
    X = np.column_stack([np.ones(N), df["x"]])
    y = df["y"].to_numpy()
    grid = np.linspace(math.log(0.05), math.log(20.0), 1201)

    def exact(i):
        keep = np.arange(N) != i
        Xk, yk = X[keep], y[keep]
        eig, U = np.linalg.eigh(V * Xk @ Xk.T)
        eig, proj = np.clip(eig, 0.0, None), U.T @ yk
        log_post, mean = np.empty(grid.size), np.empty(grid.size)
        for g, s2 in enumerate(np.exp(grid)):
            log_post[g] = (
                stats.invgamma.logpdf(s2, A0 / 2, scale=D0 / 2)
                - 0.5 * np.log(eig + s2).sum()
                - 0.5 * (proj**2 / (eig + s2)).sum()
            )
            B = np.linalg.inv(Xk.T @ Xk / s2 + np.eye(2) / V)
            mean[g] = X[i] @ B @ Xk.T @ yk / s2
        weight = np.exp(log_post - log_post.max() + grid)  # ds2 = s2 dlog(s2)
        return integrate.trapezoid(weight * mean, grid) / integrate.trapezoid(
            weight, grid
        )

    truth = np.array([exact(i) for i in range(N)])
    fit = sp.bayes_regress(
        "y ~ x",
        df,
        prior_mean=0.0,
        prior_var=V,
        sigma2_prior=(A0, D0),
        draws=20000,
        burnin=1000,
        seed=1,
    )
    loo = sp.loo(fit)
    pred = sp.loo_predict(fit, loo)
    draws = np.asarray(fit.predict(what="draws"), dtype=float)
    mc_se = draws.std(axis=0) / np.sqrt(loo.pointwise["n_eff"].to_numpy())
    assert np.all(np.abs(pred - truth) <= 4 * mc_se)
    # The check has bite: at the high-leverage point the in-sample fitted
    # value misses the exact answer by 0.098, about 20 of these standard
    # errors, so a function that returned fitted values would fail by a
    # factor of five; three times the tolerance is asserted.
    fitted = draws.mean(axis=0)
    assert abs(fitted[0] - truth[0]) > 3 * 4 * mc_se[0]
    assert abs(pred[0] - truth[0]) < 0.25 * abs(fitted[0] - truth[0])


# --------------------------------------------------------------------- #
#  switchback_design
# --------------------------------------------------------------------- #


def _variance_at_the_bound(points, m):
    """Variance of the Horvitz-Thompson estimator of the lag-m effect under
    a fair coin when every potential outcome equals 1, by enumerating every
    outcome of the coins."""
    T = points.size
    block = np.cumsum(points) - 1
    n_coins = int(points.sum())
    coins = np.array(list(itertools.product((0, 1), repeat=n_coins)))
    paths = coins[:, block]
    est = np.zeros(len(coins))
    for t in range(m, T):
        window = paths[:, t - m : t + 1].sum(axis=1)
        scale = 2.0 ** (block[t] - block[t - m] + 1)
        est += scale * ((window == m + 1).astype(float) - (window == 0))
    est /= T - m
    return float(np.mean(est**2) - np.mean(est) ** 2)


@pytest.mark.parametrize("T, m", [(8, 1), (9, 1), (8, 2), (10, 2), (12, 3)])
def test_optimal_switchback_design_minimises_the_worst_case_variance(T, m):
    # Truth: exhaustive search. A regular design is a set of periods at
    # which a fair coin is flipped, the first period always among them:
    # 2^(T - 1) designs. For each, the variance of the estimator when every
    # potential outcome sits at the bound is computed by enumerating all
    # coin outcomes. The design returned as 'optimal' must be the unique
    # minimiser. Variances are sums of dyadic rationals over at most 2^12
    # paths; distinct designs differ by far more than the 1e-9 margin.
    plan = sp.switchback_design(T, m=m, seed=0)
    chosen = plan["randomize"].to_numpy()
    variances = {}
    for rest in itertools.product((False, True), repeat=T - 1):
        points = np.array((True,) + rest)
        variances[tuple(points)] = _variance_at_the_bound(points, m)
    best = min(variances.values())
    assert variances[tuple(chosen)] == pytest.approx(best, abs=1e-12)
    ties = [k for k, v in variances.items() if v <= best + 1e-9]
    assert ties == [tuple(chosen)]


def test_switchback_design_without_carryover_flips_every_period():
    # Truth: with m = 0 the same exhaustive search (T = 8) is won by the
    # design that flips the coin in every period, whose estimator is a mean
    # of T independent terms of variance 4: variance 4 / T exactly.
    plan = sp.switchback_design(8, m=0, seed=0)
    assert plan["randomize"].all()
    variances = [
        _variance_at_the_bound(np.array((True,) + rest), 0)
        for rest in itertools.product((False, True), repeat=7)
    ]
    assert min(variances) == pytest.approx(0.5, abs=1e-12)
    chosen = _variance_at_the_bound(plan["randomize"].to_numpy(), 0)
    assert chosen == pytest.approx(min(variances), abs=1e-12)


def test_switchback_design_draws_one_coin_per_randomization_point():
    # Truth: the assignment is constant between randomization points
    # (exact), and the coins are independent with success probability p.
    # Over S = 2,000 seeds, with p = 0.3 and four coins: each coin's
    # frequency within 4 binomial standard errors of p, and the frequency
    # of two given coins both showing treatment within 4 standard errors of
    # p^2 (six pairs and four coins, so 4 rather than 3).
    S, p = 2000, 0.3
    coins = np.empty((S, 4), dtype=int)
    for i, seed in enumerate(_seeds(S)):
        plan = sp.switchback_design(12, m=2, p=p, seed=seed)
        points = plan["randomize"].to_numpy()
        assert (np.flatnonzero(points) + 1).tolist() == [1, 5, 7, 9]
        treat = plan["treat"].to_numpy()
        block = np.cumsum(points) - 1
        coins[i] = treat[points]
        assert np.array_equal(treat, coins[i][block])
    tol = 4 * math.sqrt(p * (1 - p) / S)
    assert np.all(np.abs(coins.mean(axis=0) - p) <= tol)
    tol2 = 4 * math.sqrt(p**2 * (1 - p**2) / S)
    for a, b in itertools.combinations(range(4), 2):
        assert abs(np.mean(coins[:, a] * coins[:, b]) - p**2) <= tol2


@pytest.mark.parametrize("n, k", [(10, 5), (23, 4), (101, 10)])
def test_kfold_split_is_a_balanced_partition(n, k):
    # Truth (exact, combinatorial): every observation gets one fold in
    # 0 .. k - 1 and fold sizes differ by at most one. With groups= no
    # group is split and the number of groups per fold differs by at most
    # one; with stratify= every level is spread so that its counts per
    # fold differ by at most one. No tolerance: these are integer facts.
    for seed in _seeds(5):
        folds = sp.kfold_split(n, k=k, seed=seed)
        assert sorted(set(folds.tolist())) == list(range(k))
        sizes = np.bincount(folds, minlength=k)
        assert sizes.sum() == n and sizes.max() - sizes.min() <= 1

        groups = np.arange(n) // 2
        gfolds = sp.kfold_split(n, k=k, seed=seed, groups=groups)
        per_group = pd.Series(gfolds).groupby(groups).nunique()
        assert (per_group == 1).all()
        n_groups = np.bincount(
            pd.Series(gfolds).groupby(groups).first().to_numpy(), minlength=k
        )
        assert n_groups.max() - n_groups.min() <= 1

        strata = np.arange(n) % 3
        sfolds = sp.kfold_split(n, k=k, seed=seed, stratify=strata)
        for level in range(3):
            counts = np.bincount(sfolds[strata == level], minlength=k)
            assert counts.max() - counts.min() <= 1
        total = np.bincount(sfolds, minlength=k)
        assert total.max() - total.min() <= 1


def test_mswitch_lrtest_does_not_depend_on_the_unit_of_y():
    # Truth: the likelihood-ratio statistic of one regime against two is a
    # function of y that an affine change of units leaves alone, and the
    # bootstrap series of a + b y are a + b times those of y. So the
    # statistic, every replicate and the p-value must be the same for y,
    # 100 y, 0.01 y and -2 + 0.01 y. Before the search was run on the
    # standardised series, 0.01 y sent 4 of 19 replicates to another local
    # maximum (largest change 5.16) and moved the p-value from 0.20 to 0.15.
    # Tolerance 1e-6 on a statistic of order 1 to 10: the stopping rule of
    # the optimiser (observed 5e-9).
    y = np.random.default_rng(11).normal(size=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = sp.mswitch_lrtest(y, states=2, reps=19, starts=3, seed=4)
        for shift, scale in ((0.0, 100.0), (0.0, 0.01), (-2.0, 0.01), (1000.0, 1.0)):
            other = sp.mswitch_lrtest(
                shift + scale * y, states=2, reps=19, starts=3, seed=4
            )
            assert other.statistic == pytest.approx(base.statistic, abs=1e-6)
            assert other.pvalue == base.pvalue
            np.testing.assert_allclose(
                other.replicates["lr"].to_numpy(),
                base.replicates["lr"].to_numpy(),
                atol=1e-6,
            )


def test_mswitch_lrtest_null_likelihood_and_exact_p_value():
    # Truth 1 (closed form): with one regime and a constant the Gaussian
    # maximum likelihood is -n/2 (log(2 pi s2) + 1) with s2 the mean squared
    # deviation; rtol 1e-12, rounding only.
    # Truth 2 (exact): two regimes five standard deviations apart give a
    # statistic above every bootstrap replicate drawn under one regime, so
    # the p-value is exactly 1 / (reps + 1).
    rng = np.random.default_rng(3)
    state = (np.arange(80) // 20) % 2
    y = np.where(state == 1, 2.5, -2.5) + rng.normal(size=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.mswitch_lrtest(y, states=2, reps=9, starts=3, seed=11)
    s2 = float(np.mean((y - y.mean()) ** 2))
    closed = -0.5 * len(y) * (math.log(2.0 * math.pi * s2) + 1.0)
    assert res.loglik_null == pytest.approx(closed, rel=1e-12)
    assert res.statistic == pytest.approx(2.0 * (res.loglik_alt - closed), rel=1e-12)
    assert res.pvalue == pytest.approx(1.0 / 10.0, abs=1e-15)
