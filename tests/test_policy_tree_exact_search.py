"""Exactness of the depth<=2 policy-tree search.

``sp.policy_tree`` claims a *welfare-maximising* tree for depth <= 2.
These tests hold that claim to account against an independent
brute-force enumeration of the same objective, and pin the behaviour of
the ``search`` / ``split_step`` / ``scores`` knobs added alongside it.

Historical note
---------------
Before v1.21 the search was greedy at every depth (the root split was
scored as if its children were terminal) *and* subsampled candidate
thresholds to at most 50 quantiles per feature, while the docstring
advertised exhaustive depth-2 search. On the Track A module-70 fixture
the greedy tree falls 0.70% short of the welfare optimum and assigns 78
of 1200 units to the wrong arm.  See ``docs/dev/ml_causal_parity_plan.md``.

Reference
---------
[@athey2021policy], [@zhou2023offline]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.policy_learning._exact_tree import exact_policy_tree

# ---------------------------------------------------------------------------
# Independent brute-force reference for the same objective
# ---------------------------------------------------------------------------


def _brute_force_value(X: np.ndarray, s: np.ndarray, depth: int, min_leaf: int):
    """Optimal value of sum_i Gamma_i pi(X_i) over depth-<=d trees.

    Deliberately naive: full recursion over every distinct split of every
    feature, with no sharing of work with the implementation under test.
    """
    n = X.shape[0]
    p = X.shape[1]

    def best(idx: np.ndarray, d: int):
        options = []
        if idx.size >= min_leaf:
            options.append(max(float(s[idx].sum()), 0.0))
        if d > 0:
            for k in range(p):
                vals = np.unique(X[idx, k])
                for t in vals[:-1]:
                    mask = X[idx, k] <= t
                    left, right = idx[mask], idx[~mask]
                    lo, hi = best(left, d - 1), best(right, d - 1)
                    if lo is not None and hi is not None:
                        options.append(lo + hi)
        return max(options) if options else None

    return best(np.arange(n), depth)


def _tree_value(tree: dict, X: np.ndarray, s: np.ndarray) -> float:
    total = 0.0
    for i in range(X.shape[0]):
        node = tree
        while node["type"] == "split":
            node = (
                node["left"]
                if X[i, node["feature"]] <= node["threshold"]
                else node["right"]
            )
        total += s[i] * node["action"]
    return float(total)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("min_leaf", [1, 3])
def test_exact_search_attains_brute_force_optimum(depth, min_leaf):
    """Randomised sweep: the exact search never scores below brute force.

    Covariates are rounded so ties are common — tie handling is where an
    off-by-one in the ``x <= t`` split convention would surface.
    """
    rng = np.random.default_rng(20260731)
    for _ in range(40):
        n = int(rng.integers(8, 30))
        p = int(rng.integers(1, 4))
        X = np.round(rng.normal(size=(n, p)), 1)
        s = np.round(rng.normal(size=n), 2)
        tree = exact_policy_tree(X, s, max_depth=depth, min_leaf_size=min_leaf)
        got = _tree_value(tree, X, s)
        want = _brute_force_value(X, s, depth, min_leaf)
        want = 0.0 if want is None else want
        assert got == pytest.approx(want, abs=1e-9), (
            f"exact search returned {got:.10f} but the optimum is "
            f"{want:.10f} (n={n}, p={p}, depth={depth}, min_leaf={min_leaf})"
        )


def test_exact_beats_greedy_on_a_two_dimensional_rule():
    """The exact/greedy gap is real, not a rounding artefact.

    ``tau(x) = x1 + x2`` needs both covariates, so the root split that
    looks best with terminal children is not the root split that admits
    the best depth-1 subtrees — precisely the case greedy search misses.
    """
    rng = np.random.default_rng(42)
    n, k = 1200, 3
    X = rng.normal(size=(n, k))
    d = rng.integers(0, 2, size=n)
    y = 0.5 * X[:, 2] + (X[:, 0] + X[:, 1]) * d + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["d"], df["y"] = d, y

    design = np.column_stack([np.ones(n), X])
    mu = {}
    for arm in (0, 1):
        m = d == arm
        beta, *_ = np.linalg.lstsq(design[m], y[m], rcond=None)
        mu[arm] = design @ beta
    gamma = (mu[1] - mu[0]) + d * (y - mu[1]) / 0.5 - (1 - d) * (y - mu[0]) / 0.5

    kwargs = dict(
        y="y",
        treat="d",
        covariates=["x1", "x2", "x3"],
        max_depth=2,
        min_leaf_size=1,
        scores=gamma,
    )
    exact = sp.policy_tree(df, search="exact", **kwargs)
    greedy = sp.policy_tree(df, search="greedy", **kwargs)

    assert exact["search_mode"] == "exact"
    assert greedy["search_mode"] == "greedy"
    assert exact["value_policy"] > greedy["value_policy"], (
        "the exact search must not score below the greedy search on the "
        "shared objective"
    )
    n_diff = int(
        (np.asarray(exact["policy"], int) != np.asarray(greedy["policy"], int)).sum()
    )
    assert n_diff > 0, "this fixture is supposed to separate the two searches"


def test_supplied_scores_bypass_the_cross_fitted_aipw_step():
    """``scores=`` must be used verbatim, not recomputed."""
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "d": rng.integers(0, 2, n), "y": rng.normal(size=n)})
    gamma = np.where(x > 0, 1.0, -1.0)
    res = sp.policy_tree(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        max_depth=1,
        min_leaf_size=1,
        scores=gamma,
    )
    np.testing.assert_allclose(np.asarray(res["scores"]), gamma)
    # With Gamma = sign(x), the optimal stump is exactly 1{x > 0}.
    np.testing.assert_array_equal(np.asarray(res["policy"], int), (x > 0).astype(int))
    assert res["value_policy"] == pytest.approx(float(np.mean(np.abs(gamma) * (x > 0))))


def test_scores_length_mismatch_fails_loudly():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "d": [0, 1, 0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(Exception, match="len\\(scores\\)"):
        sp.policy_tree(df, y="y", treat="d", covariates=["x"], scores=np.zeros(2))


@pytest.mark.parametrize("bad", ["exhaustive", "", "EXACT"])
def test_unknown_search_mode_is_rejected(bad):
    df = pd.DataFrame({"x": [0.0, 1.0], "d": [0, 1], "y": [1.0, 2.0]})
    with pytest.raises(Exception, match="search="):
        sp.policy_tree(df, y="y", treat="d", covariates=["x"], search=bad)


def test_split_step_thins_the_candidate_grid_but_stays_admissible():
    """``split_step>1`` is an approximation, so it may only lose value."""
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(size=(n, 2))
    s = X[:, 0] + 0.5 * X[:, 1]
    full = exact_policy_tree(X, s, max_depth=2, min_leaf_size=1, split_step=1)
    thin = exact_policy_tree(X, s, max_depth=2, min_leaf_size=1, split_step=10)
    assert _tree_value(full, X, s) >= _tree_value(thin, X, s) - 1e-9


def test_depth_three_reports_greedy_search_mode():
    """Depth >= 3 has no exact route; the result must say so."""
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "d": rng.integers(0, 2, n),
            "y": rng.normal(size=n),
        }
    )
    res = sp.policy_tree(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        max_depth=3,
        min_leaf_size=10,
        scores=rng.normal(size=n),
    )
    assert res["search_mode"] == "greedy"


def test_auto_warns_before_falling_back_to_greedy():
    """A silent downgrade from exact to greedy would be a hidden defect."""
    from statspai.policy_learning import _exact_tree as et

    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "d": rng.integers(0, 2, n),
            "y": rng.normal(size=n),
        }
    )
    original = et._AUTO_EXACT_BUDGET
    try:
        et._AUTO_EXACT_BUDGET = 1.0  # force the budget to bite
        with pytest.warns(UserWarning, match="exact depth-2 search"):
            res = sp.policy_tree(
                df,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                max_depth=2,
                min_leaf_size=10,
                scores=rng.normal(size=n),
                search="auto",
            )
        assert res["search_mode"] == "greedy"
    finally:
        et._AUTO_EXACT_BUDGET = original
