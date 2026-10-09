"""Coverage gaps in ``statspai.did._arp`` (conditional moment-inequality tests).

The linear-programming fallbacks are only used when vertex enumeration is
refused (too many bases) and the single-post-period path only when
``n_post == 1``; neither is reached by the Honest DiD tests that run in CI.
Each fallback is checked against the path that *is* exercised there: LP
duality says the primal value of the eta program equals the maximum of
``gamma' y`` over the dual polytope's vertices, so the two computations
must agree to solver precision.
"""

import importlib

import numpy as np
import pytest
from scipy import stats

arp = importlib.import_module("statspai.did._arp")


@pytest.fixture(scope="module")
def problem():
    """A bounded eta program with 6 moments and 2 nuisance parameters."""
    rng = np.random.default_rng(0)
    m, k = 6, 2
    x = rng.normal(size=(m, k))
    x -= x.mean(axis=0)
    root = rng.normal(size=(m, m))
    sigma = root @ root.T / m + 0.2 * np.eye(m)
    y = rng.normal(size=m)
    sd = np.sqrt(np.diag(sigma))
    w_t = np.column_stack([sd, x])
    verts = arp._polytope_vertices(w_t)
    return {"x": x, "sigma": sigma, "y": y, "sd": sd, "w_t": w_t, "verts": verts}


# ---------------------------------------------------------------------------
#  Linear programs
# ---------------------------------------------------------------------------


def test_eta_lp_matches_the_dual_vertex_maximum(problem):
    y, x, sigma = problem["y"], problem["x"], problem["sigma"]
    eta, delta, lam, ok = arp._eta_lp(y, x, sigma)
    assert ok
    # Strong duality: primal value == max of gamma'y over the dual vertices.
    assert eta == pytest.approx(float(np.max(problem["verts"] @ y)), abs=1e-10)
    # The multipliers are a point of the dual polytope {lam >= 0, W'lam = e1}.
    assert lam.min() >= -1e-12
    assert float(lam @ problem["sd"]) == pytest.approx(1.0, abs=1e-10)
    assert np.allclose(lam @ x, 0.0, atol=1e-10)
    # ... and delta is primal feasible: y - X delta <= eta * sd.
    assert np.all(y - x @ delta <= eta * problem["sd"] + 1e-9)


def test_eta_lp_reports_an_unbounded_program(problem):
    # A strictly positive regressor lets delta push every moment to -inf.
    m = problem["y"].size
    eta, delta, lam, ok = arp._eta_lp(problem["y"], np.ones((m, 1)), problem["sigma"])
    assert not ok
    assert np.isnan(eta) and np.isnan(delta).all()
    assert lam.shape == (m,) and not lam.any()


def _dual_inputs(problem):
    y, x, sigma = problem["y"], problem["x"], problem["sigma"]
    eta, _, gamma, _ = arp._eta_lp(y, x, sigma)
    s2 = float(gamma @ sigma @ gamma)
    s_t = y - (sigma @ gamma) * float(gamma @ y) / s2
    return eta, gamma, s_t, (sigma @ gamma) / s2


@pytest.mark.parametrize("c", [-1.0, 0.3, 2.0])
def test_max_program_equals_the_vertex_maximum(problem, c):
    eta, gamma, s_t, b = _dual_inputs(problem)
    value, sol = arp._max_program(s_t, gamma, problem["sigma"], problem["w_t"], c)
    assert value == pytest.approx(
        float(np.max(problem["verts"] @ (s_t + b * c))), abs=1e-9
    )
    assert sol.min() >= -1e-12
    assert np.allclose(problem["w_t"].T @ sol, np.r_[1.0, 0.0, 0.0], atol=1e-9)


def test_max_program_infeasible_returns_nan(problem):
    # With every "standard deviation" negative, W'x = e1 has no x >= 0.
    eta, gamma, s_t, _ = _dual_inputs(problem)
    value, sol = arp._max_program(s_t, gamma, problem["sigma"], -problem["w_t"], 0.1)
    assert np.isnan(value)
    assert sol.shape == (problem["y"].size,) and np.isnan(sol).all()


def test_polytope_vertices_refuses_empty_and_oversized_problems(problem, monkeypatch):
    assert arp._polytope_vertices(np.zeros((4, 2))) is None  # rank 0
    assert arp._polytope_vertices(-problem["w_t"]) is None  # empty polytope
    monkeypatch.setattr(arp, "_MAX_BASES", 1)
    assert arp._polytope_vertices(problem["w_t"]) is None  # too many bases


# ---------------------------------------------------------------------------
#  Truncation points
# ---------------------------------------------------------------------------


def test_truncation_points_agree_between_lp_and_vertex_paths(problem, monkeypatch):
    eta, gamma, s_t, _ = _dual_inputs(problem)
    sigma, w_t = problem["sigma"], problem["w_t"]
    given = arp._vlo_vup_dual(eta, s_t, gamma, sigma, w_t, problem["verts"])
    built = arp._vlo_vup_dual(eta, s_t, gamma, sigma, w_t)  # enumerates itself
    assert built == given
    assert given[0] <= eta <= given[1]
    monkeypatch.setattr(arp, "_MAX_BASES", 0)  # force one LP per evaluation
    by_lp = arp._vlo_vup_dual(eta, s_t, gamma, sigma, w_t)
    assert by_lp == pytest.approx(given, abs=1e-5)


def test_truncation_points_at_a_non_fixed_point(problem):
    # eta + 0.5 is not the value of the dual program at c = eta + 0.5, so
    # the reference rule returns the degenerate interval [eta, inf).
    eta, gamma, s_t, _ = _dual_inputs(problem)
    lo, hi = arp._vlo_vup_dual(
        eta + 0.5, s_t, gamma, problem["sigma"], problem["w_t"], problem["verts"]
    )
    assert lo == eta + 0.5
    assert hi == float("inf")


# ---------------------------------------------------------------------------
#  Conditional test
# ---------------------------------------------------------------------------


def test_conditional_test_is_the_same_with_and_without_vertices(problem):
    x, sigma, verts = problem["x"], problem["sigma"], problem["verts"]
    primal, dual = [], []
    for shift in np.linspace(-3.0, 6.0, 40):
        y = problem["y"] + shift
        primal.append(arp.conditional_test(y, x, sigma, 0.05))
        dual.append(arp.conditional_test(y, x, sigma, 0.05, vertices=verts))
    assert primal == dual
    # Shifting every moment up moves from "no violation" to "violation".
    assert primal[0] is False and primal[-1] is True
    assert primal == sorted(primal)


def test_conditional_test_does_not_reject_when_the_lp_fails(problem):
    ones = np.ones((problem["y"].size, 1))
    assert arp.conditional_test(problem["y"] + 50.0, ones, problem["sigma"], 0.05) is (
        False
    )


def test_conditional_test_with_a_zero_variance_statistic():
    # Two perfectly negatively correlated moments with opposite regressors:
    # eta = (y1 + y2) / 2 has variance gamma' Sigma gamma = 0, so the test
    # reduces to the sign of eta.
    x = np.array([[1.0, 0.0], [-1.0, 0.0]])
    sigma = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert arp.conditional_test(np.array([1.0, 2.0]), x, sigma, 0.05)
    assert not arp.conditional_test(np.array([-3.0, 1.0]), x, sigma, 0.05)


# ---------------------------------------------------------------------------
#  Least-favourable critical value
# ---------------------------------------------------------------------------


def test_least_favorable_cv_without_nuisance_is_a_normal_quantile():
    cv = arp._least_favorable_cv(None, np.eye(1), 0.1)
    # One standardised moment: the 0.9 quantile of N(0, 1), up to the Monte
    # Carlo error of 1,000 draws (sd of that sample quantile is about 0.05).
    assert cv == pytest.approx(stats.norm.ppf(0.9), abs=0.15)
    # Standardising the draws makes the value free of the moment's scale.
    assert arp._least_favorable_cv(None, 4.0 * np.eye(1), 0.1) == pytest.approx(
        cv, abs=1e-12
    )


def test_least_favorable_cv_lp_fallback_matches_vertex_enumeration(
    problem, monkeypatch
):
    by_vertices = arp._least_favorable_cv(problem["x"], problem["sigma"], 0.1, sims=60)
    monkeypatch.setattr(arp, "_MAX_BASES", 0)
    by_lp = arp._least_favorable_cv(problem["x"], problem["sigma"], 0.1, sims=60)
    assert by_lp == pytest.approx(by_vertices, abs=1e-8)


# ---------------------------------------------------------------------------
#  Delta^RM confidence set with a single post-treatment period
# ---------------------------------------------------------------------------

BETA = np.array([0.05, -0.02, 1.0])
SIGMA = np.diag([0.01, 0.01, 0.04])


@pytest.mark.parametrize("method", ["Conditional", "C-LF"])
def test_single_post_period_set_is_nested_in_mbar(method):
    lo0, hi0, grid, acc0 = arp.rm_confidence_set(
        BETA, SIGMA, 2, 1, 0.0, method=method, grid_points=201
    )
    lo1, hi1, _, acc1 = arp.rm_confidence_set(
        BETA, SIGMA, 2, 1, 1.0, method=method, grid_points=201
    )
    # Default grid: +/- 20 sd of the target (sd = 0.2).
    assert grid[0] == pytest.approx(-4.0) and grid[-1] == pytest.approx(4.0)
    # Mbar = 0 forbids any post-period violation: the set is centred on the
    # estimate and roughly a 95% interval for it.
    assert lo0 < BETA[2] < hi0
    assert 0.5 * (lo0 + hi0) == pytest.approx(BETA[2], abs=0.05)
    assert 2 * 1.5 * 0.2 < hi0 - lo0 < 2 * 2.5 * 0.2
    # A larger Mbar is a weaker restriction, so the set can only grow.
    assert lo1 <= lo0 and hi1 >= hi0
    assert hi1 - lo1 > hi0 - lo0
    assert np.all(acc1 >= acc0)


def test_single_post_period_set_is_translation_equivariant():
    kw = dict(method="Conditional", grid_points=81)
    _, _, grid, acc = arp.rm_confidence_set(
        BETA, SIGMA, 2, 1, 0.5, grid_lb=0.0, grid_ub=2.0, **kw
    )
    shifted = BETA + np.array([0.0, 0.0, 3.0])
    lo, hi, grid_s, acc_s = arp.rm_confidence_set(
        shifted, SIGMA, 2, 1, 0.5, grid_lb=3.0, grid_ub=5.0, **kw
    )
    assert np.allclose(grid_s, grid + 3.0)
    assert np.array_equal(acc_s, acc)
    assert lo == pytest.approx(grid[acc == 1].min() + 3.0)
    assert hi == pytest.approx(grid[acc == 1].max() + 3.0)


def test_empty_set_progress_callback_and_method_validation():
    ticks = []
    lo, hi, grid, acc = arp.rm_confidence_set(
        BETA,
        SIGMA,
        2,
        1,
        0.0,
        method="arp",
        grid_points=25,
        grid_lb=20.0,
        grid_ub=30.0,
        progress=ticks.append,
    )
    # Every grid value is 95+ standard deviations from the estimate.
    assert np.isnan(lo) and np.isnan(hi)
    assert grid.size == 25 and not acc.any()
    # One tick per polyhedral piece: n_pre locations x 2 signs.
    assert ticks == [1, 1, 1, 1]
    with pytest.raises(ValueError, match="method must be 'C-LF' or 'Conditional'"):
        arp.rm_confidence_set(BETA, SIGMA, 2, 1, 0.0, method="flci")


def test_union_is_the_pointwise_maximum_of_its_pieces():
    grid = np.linspace(0.0, 2.0, 41)
    pieces = [
        arp._ci_fixed_s(
            BETA, SIGMA, 2, 1, np.ones(1), 0.5, s, pos, 0.05, "ARP", 0.005, grid, 0
        )
        for s in (-1, 0)
        for pos in (True, False)
    ]
    _, _, _, acc = arp.rm_confidence_set(
        BETA,
        SIGMA,
        2,
        1,
        0.5,
        method="Conditional",
        grid_points=41,
        grid_lb=0.0,
        grid_ub=2.0,
    )
    assert np.array_equal(np.max(pieces, axis=0), acc)
    assert all(set(np.unique(p)) <= {0.0, 1.0} for p in pieces)
