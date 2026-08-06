"""`_within_group_self_outcome` fast path == the literal definition.

The Abadie-Imbens variance needs, for every unit, the mean outcome of its
``J`` nearest same-arm neighbours by propensity distance.  The rule is a
stable sort — ``(distance ascending, original index ascending)`` — and the
literal implementation of that is ``O(m² log m)``.  Since v1.22 this runs on
**every default** ``sp.match`` call, so the shipped implementation sorts each
arm once and walks outward in ``O(m log m + m·J)``.

That optimisation is only admissible if it is *bit-identical*: the quantity
it feeds is pinned against Stata in
``tests/reference_parity/test_psmatch2_parity.py``.  So the literal version
is kept as :func:`_within_group_self_outcome_reference` and this module
compares the two across the tie structures that make the rule subtle.

Two traps the fast path has to get right, both found by these tests during
development:

1. **Ties straddle the origin.**  Units at equal distance can sit on either
   side in sorted order, and the rule ranks them by original index across
   both sides.  Walking outward visits the left-hand ties in *descending*
   index order, so a left/right frontier comparison picks the wrong set —
   the first attempt did exactly that and disagreed on 180 of 300 cases.
2. **Summation order is part of the contract.**  Accumulating with ``+=``
   instead of ``np.mean`` over the selected values in reference order left
   residuals of ~4e-16.  Small, but "bit-identical" has to mean it.
"""

from __future__ import annotations

import numpy as np
import pytest

from statspai.matching._matched_frame import (
    _within_group_self_outcome,
    _within_group_self_outcome_reference,
)

#: Propensity-score generators, chosen for their tie structure.
SCORE_KINDS = {
    "continuous": lambda rng, n: rng.normal(size=n),
    "integer_ties": lambda rng, n: rng.integers(0, 3, n).astype(float),
    "all_identical": lambda rng, n: np.zeros(n),
    "rounded_1dp": lambda rng, n: np.round(rng.normal(size=n), 1),
    "binary": lambda rng, n: rng.choice([0.0, 0.5], size=n),
    "one_outlier": lambda rng, n: np.r_[np.zeros(n - 1), 99.0],
}


def _arms_ok(t):
    return t.sum() >= 2 and (1 - t).sum() >= 2


def _assert_bit_identical(y, t, ps, j):
    fast = _within_group_self_outcome(y, t, ps, j)
    ref = _within_group_self_outcome_reference(y, t, ps, j)
    np.testing.assert_array_equal(np.isfinite(fast), np.isfinite(ref))
    m = np.isfinite(ref)
    # Not approx: exactly equal, including the last bit.
    np.testing.assert_array_equal(fast[m], ref[m])


@pytest.mark.parametrize("kind", sorted(SCORE_KINDS))
@pytest.mark.parametrize("j", [1, 2, 5])
def test_bit_identical_across_tie_structures(kind, j):
    rng = np.random.default_rng(hash(kind) % 2**32)
    checked = 0
    for _ in range(60):
        n = int(rng.integers(4, 90))
        t = rng.integers(0, 2, n)
        if not _arms_ok(t):
            continue
        ps = SCORE_KINDS[kind](rng, n)
        y = rng.normal(size=n)
        _assert_bit_identical(y, t, ps, j)
        checked += 1
    assert checked >= 20, "the generator produced too few usable arms"


class TestTieStraddlingTheOrigin:
    """The failure mode a left/right frontier comparison gets wrong."""

    def test_equal_distance_ranks_by_original_index_not_by_side(self):
        # Arm members at indices 0..4, all with the same propensity score.
        # Unit 2's neighbours are all at distance 0, so the rule takes the
        # two smallest original indices: 0 and 1 — not the two adjacent
        # ones (1 and 3).
        t = np.ones(5, dtype=int)
        ps = np.zeros(5)
        y = np.array([10.0, 20.0, 999.0, 40.0, 50.0])
        out = _within_group_self_outcome(y, t, ps, 2)
        assert out[2] == pytest.approx((10.0 + 20.0) / 2)
        _assert_bit_identical(y, t, ps, 2)

    def test_symmetric_distance_picks_the_smaller_index(self):
        # Unit 1 sits midway between units 0 and 2: both at distance 1.
        # Ascending original index breaks the tie toward unit 0.
        t = np.ones(3, dtype=int)
        ps = np.array([0.0, 1.0, 2.0])
        y = np.array([7.0, 999.0, 11.0])
        out = _within_group_self_outcome(y, t, ps, 1)
        assert out[1] == pytest.approx(7.0)
        _assert_bit_identical(y, t, ps, 1)

    def test_tie_group_spanning_both_sides(self):
        # Unit 3 (ps = 2.0) has four neighbours at distance 1.0: units 1, 2
        # on the left and 4, 5 on the right. Unit 0 sits far away so it
        # cannot join the tie group. With J=3 the rule ranks the tied four
        # by original index across BOTH sides and takes 1, 2, 4 — an
        # outward walk would have taken 2, 4, 1.
        t = np.ones(6, dtype=int)
        ps = np.array([7.0, 1.0, 1.0, 2.0, 3.0, 3.0])
        y = np.array([100.0, 1.0, 2.0, 999.0, 4.0, 5.0])
        out = _within_group_self_outcome(y, t, ps, 3)
        assert out[3] == pytest.approx((1.0 + 2.0 + 4.0) / 3)
        _assert_bit_identical(y, t, ps, 3)

    def test_all_neighbours_tied_takes_lowest_indices(self):
        """Every other unit at the same distance -> the J smallest indices."""
        t = np.ones(6, dtype=int)
        ps = np.array([3.0, 1.0, 1.0, 2.0, 3.0, 3.0])  # all |ps - 2| == 1
        y = np.array([100.0, 1.0, 2.0, 999.0, 4.0, 5.0])
        out = _within_group_self_outcome(y, t, ps, 3)
        assert out[3] == pytest.approx((100.0 + 1.0 + 2.0) / 3)
        _assert_bit_identical(y, t, ps, 3)


class TestDegenerateArms:
    def test_arm_of_two_units(self):
        t = np.array([1, 1, 0, 0])
        ps = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        _assert_bit_identical(y, t, ps, 1)

    def test_j_larger_than_the_arm(self):
        """J is clipped to m - 1; both implementations must clip the same."""
        t = np.array([1, 1, 1, 0, 0, 0])
        ps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        y = np.arange(6, dtype=float)
        _assert_bit_identical(y, t, ps, 99)

    def test_singleton_arm_is_left_missing(self):
        t = np.array([1, 0, 0, 0])
        ps = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.arange(4, dtype=float)
        fast = _within_group_self_outcome(y, t, ps, 1)
        assert np.isnan(fast[0])
        _assert_bit_identical(y, t, ps, 1)

    def test_one_extreme_outlier(self):
        t = np.ones(6, dtype=int)
        ps = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1e9])
        y = np.arange(6, dtype=float)
        _assert_bit_identical(y, t, ps, 2)


class TestScaling:
    """The reason the fast path exists at all."""

    def test_large_arm_is_still_bit_identical(self):
        rng = np.random.default_rng(7)
        n = 1200
        t = rng.integers(0, 2, n)
        ps = np.round(rng.normal(size=n), 2)  # plenty of ties
        y = rng.normal(size=n)
        _assert_bit_identical(y, t, ps, 3)
