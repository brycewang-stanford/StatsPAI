"""sp.ttest against closed forms and an independent implementation.

T1 evidence (known answers, no cross-language reference): every quantity of
the three two-sample variants is worked out by hand on two small samples,
and the one-sample, paired and two-sample statistics are compared with
``scipy.stats`` on simulated data. The Stata comparison (Table 6.1 of
Stock & Watson, from the log shipped with the book's replication files) is
in ``tests/external_parity/test_stock_watson_4e_logs.py``.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp


def test_two_sample_variants_by_hand():
    # a = 1, 2, 3, 4      mean 2.5, variance 5/3, n = 4
    # b = 2, 4, 6, 8, 10  mean 6,   variance 10,  n = 5
    data = pd.DataFrame(
        {"v": [1, 2, 3, 4, 2, 4, 6, 8, 10], "g": [0, 0, 0, 0, 1, 1, 1, 1, 1]}
    )
    qa, qb = (5 / 3) / 4, 10 / 5

    pooled = sp.ttest(data, "v", by="g")
    s2 = (3 * (5 / 3) + 4 * 10) / 7
    assert pooled.estimate == -3.5 and pooled.df == 7
    # rtol: closed-form arithmetic, exact up to rounding
    np.testing.assert_allclose(pooled.se, np.sqrt(s2 * (1 / 4 + 1 / 5)), rtol=1e-14)
    np.testing.assert_allclose(
        pooled.pvalue, 2 * stats.t.sf(3.5 / pooled.se, 7), rtol=1e-12
    )

    satt = sp.ttest(data, "v", by="g", unequal=True)
    np.testing.assert_allclose(satt.se, np.sqrt(qa + qb), rtol=1e-14)
    np.testing.assert_allclose(
        satt.df, (qa + qb) ** 2 / (qa**2 / 3 + qb**2 / 4), rtol=1e-14
    )

    welch = sp.ttest(data, "v", by="g", welch=True)
    np.testing.assert_allclose(
        welch.df, -2 + (qa + qb) ** 2 / (qa**2 / 5 + qb**2 / 6), rtol=1e-14
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_statistics_match_scipy(seed):
    rng = np.random.default_rng(seed)
    n = 150
    data = pd.DataFrame({"g": rng.integers(0, 2, n)})
    data["y"] = 1 + 0.4 * data.g + rng.normal(scale=1 + data.g, size=n)
    data["x"] = data.y + rng.normal(loc=0.1, size=n)
    a, b = data.y[data.g == 0], data.y[data.g == 1]

    one = sp.ttest(data, "y", mu=1.0)
    ref = stats.ttest_1samp(data.y, 1.0)
    np.testing.assert_allclose(
        [one.statistic, one.pvalue], [ref.statistic, ref.pvalue], rtol=1e-12
    )

    paired = sp.ttest(data, "y", other="x")
    ref = stats.ttest_rel(data.y, data.x)
    np.testing.assert_allclose(
        [paired.statistic, paired.pvalue], [ref.statistic, ref.pvalue], rtol=1e-12
    )

    for unequal in (False, True):
        two = sp.ttest(data, "y", by="g", unequal=unequal)
        ref = stats.ttest_ind(a, b, equal_var=not unequal)
        np.testing.assert_allclose(
            [two.statistic, two.pvalue, two.df],
            [ref.statistic, ref.pvalue, ref.df],
            rtol=1e-12,
        )


def test_size_of_the_welch_test_under_unequal_variances():
    # equal means, variance ratio 9, unbalanced groups: the pooled test
    # over-rejects, the unequal-variance test holds its level
    rng = np.random.default_rng(42)
    reps = 2000
    pooled = welch = 0
    for _ in range(reps):
        data = pd.DataFrame(
            {
                "g": [0] * 40 + [1] * 10,
                "y": np.r_[rng.normal(size=40), 3 * rng.normal(size=10)],
            }
        )
        pooled += sp.ttest(data, "y", by="g").pvalue < 0.05
        welch += sp.ttest(data, "y", by="g", unequal=True).pvalue < 0.05
    # binomial SE at 2,000 draws is 0.005
    assert pooled / reps > 0.15
    assert 0.03 < welch / reps < 0.075
