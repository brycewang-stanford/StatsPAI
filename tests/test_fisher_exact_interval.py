"""Confidence interval of ``sp.fisher_exact``: inversion of the test.

The interval is the set of constant effects the randomization test does not
reject. For the difference in means it is inverted on the draws that gave
the p-value (see ``tests/test_ding_first_course_fixes.py``); the tests here
add a brute-force check on an enumerated design, coverage, and the
statistics that are not linear in the outcome (``'t'``, ``'ks'``,
``'rank_sum'``), whose interval through 1.38.0 was the percentiles of the
null distribution of the statistic.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.inference.randomization import _get_stat_fn, _stat_all


def _frame(y, d):
    return pd.DataFrame({"y": y, "d": d})


def test_exact_interval_equals_brute_force_on_an_enumerated_design():
    """8 choose 3 = 56 assignments: compare with a fine grid of nulls."""
    rng = np.random.default_rng(3)
    d = np.r_[np.ones(3), np.zeros(5)]
    y = d + rng.standard_normal(8)
    res = sp.fisher_exact(_frame(y, d), y="y", treatment="d", alpha=0.1)

    assign = np.array(
        [[1.0 if i in c else 0.0 for i in range(8)] for c in combinations(range(8), 3)]
    )

    def p_value(tau):
        ya = y - tau * d
        perm = (assign @ ya) / 3 - ((1 - assign) @ ya) / 5
        obs = ya[d == 1].mean() - ya[d == 0].mean()
        return np.mean(np.abs(perm) >= abs(obs) - 1e-12)

    grid = np.linspace(-6, 8, 14001)  # step 1e-3
    accepted = grid[np.array([p_value(t) for t in grid]) >= 0.1]
    # The grid can only under-shoot each end, by at most one step.
    assert res.ci[0] == pytest.approx(accepted.min(), abs=1e-3)
    assert res.ci[1] == pytest.approx(accepted.max(), abs=1e-3)
    assert res.ci[0] <= accepted.min() and res.ci[1] >= accepted.max()


def test_too_few_assignments_gives_the_whole_line():
    """6 choose 3 = 20: the smallest p-value is 2/20, nothing is rejected
    at 5% or 10%, and the interval says so instead of inventing ends."""
    df = _frame(np.r_[3, 4, 5, 1, 2, 2.5], np.r_[1, 1, 1, 0, 0, 0.0])
    for alpha in (0.05, 0.10):
        with pytest.warns(UserWarning, match="unbounded"):
            res = sp.fisher_exact(df, y="y", treatment="d", alpha=alpha)
        assert res.ci == (-np.inf, np.inf)
    lo, hi = sp.fisher_exact(df, y="y", treatment="d", alpha=0.2).ci
    assert np.isfinite(lo) and np.isfinite(hi) and lo < 2.1667 < hi


@pytest.mark.parametrize("statistic", ["ate", "t", "rank_sum", "ks"])
def test_interval_is_dual_to_the_test(statistic):
    """Zero is outside the interval exactly when the test rejects."""
    rng = np.random.default_rng(11)
    agree = 0
    reps = 12
    for r in range(reps):
        d = rng.permutation(np.r_[np.ones(10), np.zeros(30)])
        y = 0.6 * d + rng.standard_normal(40)
        res = sp.fisher_exact(
            _frame(y, d), y="y", treatment="d", statistic=statistic, n_perm=499, seed=r
        )
        lo, hi = res.ci
        assert lo < hi
        agree += (res.p_value < 0.05) == (not lo <= 0.0 <= hi)
    assert agree == reps


def test_interval_covers_a_constant_effect():
    """A randomization interval is valid in finite samples: coverage of a
    constant effect is at least nominal. 150 draws, so the binomial 3-sigma
    band around 0.95 is about +/- 0.053; the grid interval of 1.38.0 sat at
    0.927 over 300 draws."""
    rng = np.random.default_rng(1)
    reps, cover = 150, 0
    for r in range(reps):
        d = rng.permutation(np.r_[np.ones(12), np.zeros(48)])
        y = 0.75 * d + rng.standard_normal(60)
        lo, hi = sp.fisher_exact(
            _frame(y, d), y="y", treatment="d", n_perm=499, seed=r
        ).ci
        cover += lo <= 0.75 <= hi
    assert cover / reps >= 0.93


def test_interval_is_in_outcome_units_for_every_statistic():
    """The rank-sum / KS interval used to be percentiles of the null
    distribution of the statistic (centred on zero whatever the effect)."""
    rng = np.random.default_rng(5)
    d = rng.permutation(np.r_[np.ones(40), np.zeros(40)])
    y = 3.0 * d + rng.standard_normal(80)
    for statistic in ("t", "rank_sum", "ks"):
        lo, hi = sp.fisher_exact(
            _frame(y, d), y="y", treatment="d", statistic=statistic, n_perm=499, seed=0
        ).ci
        assert 2.0 < lo < 3.0 < hi < 4.0


def test_vectorised_statistics_match_scipy():
    rng = np.random.default_rng(0)
    y = np.round(rng.standard_normal(40), 1)  # ties
    assign = np.array(
        [rng.permutation(np.r_[np.ones(15), np.zeros(25)]) for _ in range(50)]
    )
    for statistic in ("ate", "t", "ks", "rank_sum"):
        fn = _get_stat_fn(statistic)
        np.testing.assert_allclose(
            _stat_all(assign, y, statistic),
            [fn(y, a) for a in assign],
            rtol=0,
            atol=1e-12,
        )
    assert stats.ks_2samp(y[:15], y[15:]).statistic >= 0
