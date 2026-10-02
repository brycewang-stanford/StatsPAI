"""sp.ttest: one-sample, paired and two-sample t tests.

References are scipy.stats (an independent implementation) and, for the
quantities scipy does not report, closed forms worked out by hand on a
five-number sample.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(7)
    n = 120
    out = pd.DataFrame({"g": rng.integers(0, 2, n)})
    out["y"] = 2.0 + 0.6 * out.g + rng.normal(scale=1 + out.g, size=n)
    out["x"] = out.y + rng.normal(loc=0.2, size=n)
    out.loc[[3, 17], "y"] = np.nan
    out.loc[[5], "x"] = np.nan
    return out


def test_one_sample_matches_scipy(df):
    res = sp.ttest(df, "y", mu=2.0)
    ref = stats.ttest_1samp(df.y.dropna(), 2.0)
    assert res.method == "One-sample t test"
    np.testing.assert_allclose(res.statistic, ref.statistic, rtol=1e-12)
    np.testing.assert_allclose(res.pvalue, ref.pvalue, rtol=1e-12)
    np.testing.assert_allclose(res.df, df.y.notna().sum() - 1)
    lo, hi = ref.confidence_interval(0.95)
    np.testing.assert_allclose(res.ci, (lo, hi), rtol=1e-12)


def test_paired_uses_rows_where_both_are_observed(df):
    res = sp.ttest(df, "y", other="x")
    both = df.dropna(subset=["y", "x"])
    ref = stats.ttest_rel(both.y, both.x)
    assert res.method == "Paired t test"
    assert res.n_obs == len(both)
    np.testing.assert_allclose(res.statistic, ref.statistic, rtol=1e-12)
    np.testing.assert_allclose(res.pvalue, ref.pvalue, rtol=1e-12)
    np.testing.assert_allclose(res.estimate, (both.y - both.x).mean(), rtol=1e-12)


@pytest.mark.parametrize("unequal", [False, True])
def test_two_sample_by_matches_scipy(df, unequal):
    res = sp.ttest(df, "y", by="g", unequal=unequal)
    a = df.loc[df.g == 0, "y"].dropna()
    b = df.loc[df.g == 1, "y"].dropna()
    ref = stats.ttest_ind(a, b, equal_var=not unequal)
    np.testing.assert_allclose(res.statistic, ref.statistic, rtol=1e-12)
    np.testing.assert_allclose(res.pvalue, ref.pvalue, rtol=1e-12)
    np.testing.assert_allclose(res.df, ref.df, rtol=1e-12)
    # lower group minus higher group, as Stata
    np.testing.assert_allclose(res.estimate, a.mean() - b.mean(), rtol=1e-12)
    assert list(res.groups.index) == ["0", "1", "combined"]
    assert res.groups.loc["combined", "n"] == len(a) + len(b)


def test_unpaired_columns_use_each_columns_own_rows(df):
    res = sp.ttest(df, "y", other="x", paired=False, unequal=True)
    ref = stats.ttest_ind(df.y.dropna(), df.x.dropna(), equal_var=False)
    np.testing.assert_allclose(res.statistic, ref.statistic, rtol=1e-12)
    assert res.groups.loc["y", "n"] == df.y.notna().sum()
    assert res.groups.loc["x", "n"] == df.x.notna().sum()


def test_hand_worked_example():
    # a = 1..4 (mean 2.5, var 5/3), b = 2,4,6,8,10 (mean 6, var 10)
    data = pd.DataFrame(
        {"v": [1, 2, 3, 4, 2, 4, 6, 8, 10], "g": [0, 0, 0, 0, 1, 1, 1, 1, 1]}
    )
    qa, qb = (5 / 3) / 4, 10 / 5
    se = np.sqrt(qa + qb)

    satt = sp.ttest(data, "v", by="g", unequal=True)
    np.testing.assert_allclose(satt.estimate, -3.5)
    np.testing.assert_allclose(satt.se, se, rtol=1e-14)
    np.testing.assert_allclose(
        satt.df, (qa + qb) ** 2 / (qa**2 / 3 + qb**2 / 4), rtol=1e-14
    )

    welch = sp.ttest(data, "v", by="g", welch=True)
    np.testing.assert_allclose(welch.se, se, rtol=1e-14)
    np.testing.assert_allclose(
        welch.df, -2 + (qa + qb) ** 2 / (qa**2 / 5 + qb**2 / 6), rtol=1e-14
    )
    assert welch.df != satt.df

    pooled = sp.ttest(data, "v", by="g")
    sp2 = (3 * (5 / 3) + 4 * 10) / 7
    np.testing.assert_allclose(pooled.se, np.sqrt(sp2 * (1 / 4 + 1 / 5)), rtol=1e-14)
    assert pooled.df == 7


def test_one_sided_pvalues_and_level(df):
    res = sp.ttest(df, "y", by="g", alpha=0.10)
    np.testing.assert_allclose(res.pvalue_less + res.pvalue_greater, 1.0)
    np.testing.assert_allclose(res.pvalue, 2 * min(res.pvalue_less, res.pvalue_greater))
    wide = sp.ttest(df, "y", by="g", alpha=0.01)
    assert wide.ci[0] < res.ci[0] < res.ci[1] < wide.ci[1]
    assert "t =" in res.summary()


def test_refuses_what_it_cannot_test(df):
    three = df.assign(g3=np.arange(len(df)) % 3)
    with pytest.raises(MethodIncompatibility, match="exactly two groups"):
        sp.ttest(three, "y", by="g3")
    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.ttest(df, "y", by="g", other="x")
    with pytest.raises(MethodIncompatibility, match="standard error is zero"):
        sp.ttest(pd.DataFrame({"c": [1.0, 1.0, 1.0]}), "c")
    with pytest.raises(MethodIncompatibility, match="fewer than two"):
        sp.ttest(pd.DataFrame({"c": [1.0, np.nan]}), "c")
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.ttest(df, "nope")


# ------------------------------------------------------------ translation
@pytest.mark.parametrize(
    "line, direct",
    [
        ("ttest y == 2", dict(mu=2.0)),
        ("ttest y = 2", dict(mu=2.0)),
        ("ttest y, by(g)", dict(by="g")),
        ("ttest y, by(g) unequal", dict(by="g", unequal=True)),
        ("ttest y, by(g) une unpaired", dict(by="g", unequal=True)),
        ("ttest y, by(g) welch", dict(by="g", welch=True)),
        ("ttest y == x", dict(other="x")),
        ("ttest y=x, unp une", dict(other="x", paired=False, unequal=True)),
        ("ttest y, by(g) level(90)", dict(by="g", alpha=0.10)),
    ],
)
def test_stata_ttest_runs_the_same_test(df, line, direct):
    out = sp.from_stata(line)
    assert out["ok"] and out["untranslated_options"] == [], out
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = sp.stata(line, data=df)
    want = sp.ttest(df, "y", **direct)
    assert got.method == want.method
    assert got.statistic == want.statistic and got.df == want.df
    assert got.ci == want.ci


def test_stata_ttest_does_not_drop_an_option_silently():
    # unequal has no meaning for a paired test: reported, and sp.stata refuses
    out = sp.from_stata("ttest y == x, unequal")
    assert out["untranslated_options"] == ["unequal"]
    assert sp.from_stata("ttest y x")["ok"] is False
