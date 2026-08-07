"""The weighted KS statistic must be the same statistic as the unweighted one.

`_ks_stat` delegates the unweighted case to `scipy.stats.ks_2samp`, which is
exact, and used to compute the weighted case by linearly interpolating the
cumulative weights between order statistics. An empirical CDF is a *step*
function, so interpolation reports a smaller maximum gap than exists: on
random samples it understated the true statistic by up to 0.066 absolute and
25.8% relative at n = 30.

That matters because the number is a balance *diagnostic* — a user asking
"is my weighted KS under 0.1?" was being told 0.19 where the answer was
0.256 — and because it made one branch of one function less accurate than
the other.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from statspai.matching.ps_diagnostics import _ks_stat


def _exact_weighted_ks(x_t, x_c, w_t, w_c):
    """F(v) = sum(w_i 1[x_i <= v]) / sum(w), evaluated on the pooled support."""
    v = np.sort(np.concatenate([x_t, x_c]))
    st, sc = np.argsort(x_t), np.argsort(x_c)
    xt, wt = x_t[st], w_t[st]
    xc, wc = x_c[sc], w_c[sc]
    ct = np.concatenate([[0.0], np.cumsum(wt)])
    cc = np.concatenate([[0.0], np.cumsum(wc)])
    ft = ct[np.searchsorted(xt, v, side="right")] / ct[-1]
    fc = cc[np.searchsorted(xc, v, side="right")] / cc[-1]
    return float(np.max(np.abs(ft - fc)))


class TestAgreesWithScipyWhenUnweighted:
    def test_equal_weights_reproduce_ks_2samp(self):
        """The weighted branch with w=1 must be the unweighted statistic."""
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(200):
            n = int(rng.integers(10, 200))
            x_t = rng.normal(size=n)
            x_c = rng.normal(loc=0.3, size=n)
            got = _ks_stat(x_t, x_c, np.ones(n), np.ones(n))
            want = float(stats.ks_2samp(x_t, x_c).statistic)
            worst = max(worst, abs(got - want))
        assert worst < 1e-12

    def test_the_unweighted_branch_still_delegates_to_scipy(self):
        rng = np.random.default_rng(1)
        x_t, x_c = rng.normal(size=80), rng.normal(loc=0.4, size=95)
        assert _ks_stat(x_t, x_c) == pytest.approx(
            float(stats.ks_2samp(x_t, x_c).statistic), rel=0, abs=0
        )


class TestExactStepFunction:
    def test_matches_the_step_function_definition(self):
        rng = np.random.default_rng(2)
        for _ in range(200):
            n = int(rng.integers(10, 150))
            x_t, x_c = rng.normal(size=n), rng.normal(loc=0.3, size=n)
            w_t, w_c = rng.random(n) + 0.1, rng.random(n) + 0.1
            assert _ks_stat(x_t, x_c, w_t, w_c) == pytest.approx(
                _exact_weighted_ks(x_t, x_c, w_t, w_c), rel=0, abs=1e-12
            )

    def test_interpolation_would_understate(self):
        """Pin the direction and size of the old defect."""
        rng = np.random.default_rng(3)
        n = 30
        x_t, x_c = rng.normal(size=n), rng.normal(loc=0.3, size=n)
        w_t, w_c = rng.random(n) + 0.1, rng.random(n) + 0.1

        v = np.sort(np.concatenate([x_t, x_c]))
        st, sc = np.argsort(x_t), np.argsort(x_c)
        ct, cc = np.cumsum(w_t[st]), np.cumsum(w_c[sc])
        old = float(
            np.max(
                np.abs(
                    np.interp(v, x_t[st], ct / ct[-1], left=0, right=1)
                    - np.interp(v, x_c[sc], cc / cc[-1], left=0, right=1)
                )
            )
        )
        new = _ks_stat(x_t, x_c, w_t, w_c)
        assert new >= old  # a step function is never below its interpolant here
        assert new - old > 1e-3  # and the gap is not negligible at this n


class TestDegenerateInputs:
    def test_zero_total_weight_is_nan_not_a_crash(self):
        x = np.array([0.0, 1.0, 2.0])
        assert np.isnan(_ks_stat(x, x, np.zeros(3), np.ones(3)))

    def test_identical_samples_give_zero(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=50)
        w = rng.random(50) + 0.1
        assert _ks_stat(x, x, w, w) == pytest.approx(0.0, abs=1e-15)

    def test_disjoint_supports_give_one(self):
        assert _ks_stat(
            np.array([0.0, 1.0]),
            np.array([10.0, 11.0]),
            np.ones(2),
            np.ones(2),
        ) == pytest.approx(1.0)
