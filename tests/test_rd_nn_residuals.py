"""Nearest-neighbour residuals for the RD variance (``rd/_nn_variance.py``).

Two kinds of evidence:

* the vectorised ``nn_residuals`` against a direct O(n^2) transcription of
  the published definition (CCT 2014 §5 with the CCFT 2017 §2.2 tie rule:
  every neighbour at the J-th distance is used);
* conventional standard errors of R rdrobust 4.0.0 (``vce="nn"``, black
  box) reproduced from these residuals on committed data
  (``tests/reference_parity/_fixtures/_generate_rd_nn_residuals_R.R``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from statspai.rd._nn_variance import nn_residuals

FIXTURE = (
    Path(__file__).parent / "reference_parity" / "_fixtures" / "rd_nn_residuals_R.json"
)


def _tie_runs(x):
    """dups / dupsid as described in the nn_residuals signature."""
    n = len(x)
    starts = np.concatenate(([0], np.flatnonzero(x[1:] != x[:-1]) + 1))
    counts = np.diff(np.append(starts, n))
    dups = np.repeat(counts, counts)
    dupsid = np.arange(n) - np.repeat(starts, counts) + 1
    return dups, dupsid


def _call(x, y, matches):
    dups, dupsid = _tie_runs(x)
    return nn_residuals(x, y, dups, dupsid, matches)


def _brute_force(x, y, matches):
    """Definition, unit by unit: N_i = {j != i : |x_j - x_i| <= d_i}."""
    n = len(x)
    J = min(matches, n - 1)
    y2 = np.asarray(y, dtype=float).reshape(n, -1)
    out = np.empty_like(y2)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        dist = np.abs(x[others] - x[i])
        d_i = np.sort(dist)[J - 1]
        nb = others[dist <= d_i]
        J_i = len(nb)
        out[i] = np.sqrt(J_i / (J_i + 1.0)) * (y2[i] - y2[nb].mean(axis=0))
    return out.reshape(np.shape(y))


def _design(kind, n, rng):
    if kind == "continuous":
        x = rng.normal(size=n)
    elif kind == "many_ties":
        x = rng.integers(0, max(2, n // 6), size=n).astype(float)
    elif kind == "three_values":
        x = rng.choice([-0.3, 0.1, 0.7], size=n)
    elif kind == "rounded":
        x = np.round(rng.uniform(-1, 1, size=n), 1)
    elif kind == "dyadic_sparse":  # exact equidistant neighbours
        x = rng.integers(-64, 65, size=n) / 64.0
    else:  # pragma: no cover - test parametrisation error
        raise AssertionError(kind)
    return np.sort(x)


@pytest.mark.parametrize("n", [2, 3, 9, 57, 400])
@pytest.mark.parametrize(
    "kind",
    ["continuous", "many_ties", "three_values", "rounded", "dyadic_sparse"],
)
@pytest.mark.parametrize("matches", [1, 3, 5])
@pytest.mark.parametrize("ncol", [None, 3])
def test_matches_definition(n, kind, matches, ncol):
    seed = [n, matches, 0 if ncol is None else ncol, len(kind)]
    rng = np.random.default_rng(seed)
    x = _design(kind, n, rng)
    y = rng.normal(size=n if ncol is None else (n, ncol))
    got = _call(x, y, matches)
    want = _brute_force(x, y, matches)
    assert got.shape == y.shape
    # Same neighbour sets, different summation order (window sums are
    # accumulated per distinct value): agreement to rounding, rtol 1e-12;
    # atol covers residuals that are zero up to rounding.
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


def test_columns_are_independent_problems():
    rng = np.random.default_rng(7)
    x = np.sort(np.round(rng.normal(size=200), 1))
    y = rng.normal(size=(200, 2))
    both = _call(x, y, 3)
    # Column-wise calls do the identical arithmetic on each column.
    np.testing.assert_array_equal(both[:, 0], _call(x, y[:, 0], 3))
    np.testing.assert_array_equal(both[:, 1], _call(x, y[:, 1], 3))


def test_known_values_small_case():
    # x = 0, 0, 1, 3 with J = 1: unit 0 and 1 are each other's only
    # neighbour (distance 0); unit 2's nearest are both zeros (distance 1,
    # tied), so J_2 = 2; unit 3's nearest is unit 2.
    x = np.array([0.0, 0.0, 1.0, 3.0])
    y = np.array([1.0, 3.0, 5.0, 11.0])
    want = np.array(
        [
            np.sqrt(1 / 2) * (1 - 3),
            np.sqrt(1 / 2) * (3 - 1),
            np.sqrt(2 / 3) * (5 - 2),
            np.sqrt(1 / 2) * (11 - 5),
        ]
    )
    # rtol 1e-14: the expected values are closed forms, the function sums.
    np.testing.assert_allclose(_call(x, y, 1), want, rtol=1e-14)


def test_matches_capped_at_n_minus_one():
    x = np.array([0.0, 1.0, 5.0])
    y = np.array([2.0, 4.0, 9.0])
    # J = min(10, 2) = 2: every unit uses the other two.
    want = np.array(
        [
            np.sqrt(2 / 3) * (2 - 6.5),
            np.sqrt(2 / 3) * (4 - 5.5),
            np.sqrt(2 / 3) * (9 - 3.0),
        ]
    )
    np.testing.assert_allclose(_call(x, y, 10), want, rtol=1e-14)


def test_large_n_runs_and_matches_definition_on_a_subset():
    # n = 10^5 with mass points: the vectorised path must finish quickly
    # and agree with the definition at units whose neighbour sets are
    # small enough to check directly.
    rng = np.random.default_rng(11)
    n = 100_000
    x = np.sort(np.round(rng.normal(size=n), 3))
    y = rng.normal(size=n)
    got = _call(x, y, 3)
    for i in rng.choice(n, size=25, replace=False):
        lo, hi = max(0, i - 400), min(n, i + 401)
        # The J-th neighbour of i is within a few hundred positions at
        # this density, so a local slice gives the same neighbour set,
        # unless the slice is cut by the array ends.
        xs, ys = x[lo:hi], y[lo:hi]
        local = _brute_force(xs, ys, 3)[i - lo]
        np.testing.assert_allclose(got[i], local, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize(
    "x, y, matches, msg",
    [
        (np.array([1.0, 0.0]), np.zeros(2), 1, "sorted"),
        (np.array([0.0]), np.zeros(1), 1, "at least two"),
        (np.array([0.0, 1.0]), np.zeros(2), 0, "matches"),
        (np.array([0.0, 1.0]), np.zeros(3), 1, "shape"),
        (np.array([0.0, np.nan]), np.zeros(2), 1, "non-finite"),
    ],
)
def test_invalid_input_raises(x, y, matches, msg):
    with pytest.raises(ValueError, match=msg):
        nn_residuals(x, y, np.ones(len(x)), np.ones(len(x)), matches)


# ---------------------------------------------------------------------------
# R rdrobust 4.0.0 black-box fixture
# ---------------------------------------------------------------------------


def _side_variance(x, y, h, matches):
    """Conventional variance of the side's intercept, p = 1, triangular.

    Black-box finding pinned by the fixture: the neighbours are searched
    only among observations with |x - c| < h (strictly inside the window;
    observations with zero kernel weight at |x| = h are not neighbours),
    not on the whole side of the cutoff.
    """
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    keep = np.abs(x) < h
    x, y = x[keep], y[keep]
    r = _call(x, y, matches)
    w = (1.0 - np.abs(x / h)) / h
    R = np.column_stack([np.ones_like(x), x])
    G_inv = np.linalg.inv(R.T @ (w[:, None] * R))
    score = R * (w * r)[:, None]
    V = G_inv @ (score.T @ score) @ G_inv
    return V[0, 0]


def _fixture_cases():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["cases"]


@pytest.mark.parametrize("case", _fixture_cases(), ids=lambda c: c["name"])
def test_reproduces_rdrobust_nn_standard_error(case):
    x = np.asarray(case["x"], dtype=float)
    y = np.asarray(case["y"], dtype=float)
    h, J = case["h"], int(case["nnmatch"])
    left, right = x < 0, x >= 0
    v_l = _side_variance(x[left], y[left], h, J)
    v_r = _side_variance(x[right], y[right], h, J)
    # 1e-10 relative: same estimator, different summation order and a
    # 2x2 inverse; observed agreement is ~1e-14.
    np.testing.assert_allclose(v_l, case["V_cl_l"], rtol=1e-10)
    np.testing.assert_allclose(v_r, case["V_cl_r"], rtol=1e-10)
    np.testing.assert_allclose(np.sqrt(v_l + v_r), case["se_conventional"], rtol=1e-10)
