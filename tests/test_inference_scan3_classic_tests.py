"""Classic tests: rank tests, t / z / variance / proportion tests.

Checked against scipy, statsmodels and formulas written out here, on data
with ties. Where Stata's convention differs from scipy's default (no
continuity correction in ``ranksum``, a continuity correction in ``ktau``,
the Pratt treatment of zeros in ``signrank``) the scipy call or the formula
below uses Stata's.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _tied() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 40
    y = np.round(rng.normal(size=n) * 2)
    return pd.DataFrame(
        {
            "y": y,
            "x": np.round(rng.normal(size=n) + 0.3 * y),
            "g": np.arange(n) % 2,
            "g3": np.arange(n) % 3,
        }
    )


class TestRankTests:
    def test_ranksum_is_mann_whitney_without_continuity_correction(self):
        df = _tied()
        a, b = df.loc[df.g == 0, "y"], df.loc[df.g == 1, "y"]
        ref = stats.mannwhitneyu(a, b, use_continuity=False, method="asymptotic")
        res = sp.ranksum(df, "y", by="g")
        assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
        n1, n2 = len(a), len(b)
        # W = U + n1 (n1 + 1) / 2, and porder is U / (n1 n2)
        w = float(res.table["rank_sum"].iloc[0])
        assert w - n1 * (n1 + 1) / 2 == pytest.approx(ref.statistic, abs=1e-10)
        assert res.estimates["porder"] == pytest.approx(
            ref.statistic / (n1 * n2), abs=1e-12
        )
        n = n1 + n2
        assert res.table["rank_sum"].iloc[:2].sum() == pytest.approx(n * (n + 1) / 2)
        # tie-corrected variance written out
        ties = np.unique(df["y"], return_counts=True)[1]
        var = n1 * n2 / 12 * ((n + 1) - np.sum(ties**3 - ties) / (n * (n - 1)))
        assert res.estimates["adjusted_variance"] == pytest.approx(var, rel=1e-12)
        assert res.statistic == pytest.approx(
            (w - n1 * (n + 1) / 2) / np.sqrt(var), rel=1e-12
        )

    def test_ranksum_does_not_depend_on_group_labels_or_row_order(self):
        df = _tied()
        base = sp.ranksum(df, "y", by="g")
        relabelled = df.assign(g=np.where(df["g"] == 1, "b", "a"))
        shuffled = df.sample(frac=1.0, random_state=0)
        assert sp.ranksum(relabelled, "y", by="g").statistic == base.statistic
        assert sp.ranksum(shuffled, "y", by="g").statistic == pytest.approx(
            base.statistic, rel=1e-12
        )

    def test_kwallis_reports_both_the_plain_and_the_tie_adjusted_statistic(self):
        df = _tied()
        parts = [df.loc[df.g3 == k, "y"] for k in range(3)]
        ref = stats.kruskal(*parts)  # tie-adjusted
        res = sp.kwallis(df, "y", by="g3")
        assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)
        assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
        n = len(df)
        ties = np.unique(df["y"], return_counts=True)[1]
        factor = 1 - np.sum(ties**3 - ties) / (n**3 - n)
        assert res.estimates["chi2_unadjusted"] == pytest.approx(
            ref.statistic * factor, rel=1e-12
        )

    def test_spearman_and_kendall(self):
        df = _tied()
        rho = stats.spearmanr(df["x"], df["y"])
        res = sp.spearman(df, "x", "y")
        assert res.statistic == pytest.approx(rho.statistic, rel=1e-12)
        assert res.pvalue == pytest.approx(rho.pvalue, rel=1e-10)

        tau = sp.ktau(df, "x", "y")
        ref = stats.kendalltau(df["x"], df["y"], variant="b")
        assert tau.statistic == pytest.approx(ref.statistic, rel=1e-12)
        # score, tau-a and the tie-corrected variance written out
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        n = len(x)
        iu = np.triu_indices(n, 1)
        score = np.sum(
            np.sign(x[:, None] - x[None, :])[iu] * np.sign(y[:, None] - y[None, :])[iu]
        )
        assert tau.estimates["score"] == pytest.approx(score)
        assert tau.estimates["tau_a"] == pytest.approx(score / (n * (n - 1) / 2))
        tx = np.unique(x, return_counts=True)[1].astype(float)
        ty = np.unique(y, return_counts=True)[1].astype(float)
        v0 = n * (n - 1) * (2 * n + 5)
        vt = np.sum(tx * (tx - 1) * (2 * tx + 5))
        vu = np.sum(ty * (ty - 1) * (2 * ty + 5))
        v1 = np.sum(tx * (tx - 1)) * np.sum(ty * (ty - 1)) / (2 * n * (n - 1))
        v2 = (
            np.sum(tx * (tx - 1) * (tx - 2))
            * np.sum(ty * (ty - 1) * (ty - 2))
            / (9 * n * (n - 1) * (n - 2))
        )
        se = np.sqrt((v0 - vt - vu) / 18 + v1 + v2)
        assert tau.estimates["se_score"] == pytest.approx(se, rel=1e-12)
        # Stata's continuity correction: |score| - 1
        z = (abs(score) - 1) / se
        assert tau.pvalue == pytest.approx(2 * stats.norm.sf(z), rel=1e-10)

    def test_signrank_is_wilcoxon_with_pratt_zeros(self):
        rng = np.random.default_rng(4)
        df = pd.DataFrame(
            {"a": np.round(rng.normal(size=25)), "b": np.round(rng.normal(0.4, 1, 25))}
        )
        ref = stats.wilcoxon(
            df["a"], df["b"], zero_method="pratt", correction=False, method="approx"
        )
        res = sp.signrank(df, "a", "b")
        assert (df["a"] == df["b"]).any()  # the design has zero differences
        assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-10)
        # the paired form is the one-sample form on the differences
        one = sp.signrank(df.assign(d=df["a"] - df["b"]), "d")
        assert one.statistic == pytest.approx(res.statistic, rel=1e-12)
        shifted = sp.signrank(df.assign(d=df["a"] - df["b"] + 2.0), "d", value=2.0)
        assert shifted.statistic == pytest.approx(res.statistic, rel=1e-12)
        total = res.table["rank_sum"].iloc[:3].sum()
        assert total == pytest.approx(25 * 26 / 2)

    def test_ksmirnov_statistic_and_one_sided_parts(self):
        df = _tied()
        a, b = df.loc[df.g == 0, "y"], df.loc[df.g == 1, "y"]
        res = sp.ksmirnov(df, "y", by="g")
        assert res.statistic == pytest.approx(stats.ks_2samp(a, b).statistic, abs=1e-12)
        grid = np.unique(df["y"])
        fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
        fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
        d_plus, d_minus = np.max(fa - fb), np.min(fa - fb)
        assert res.table["D"].iloc[0] == pytest.approx(d_plus, abs=1e-12)
        assert res.table["D"].iloc[1] == pytest.approx(d_minus, abs=1e-12)
        m = len(a) * len(b) / (len(a) + len(b))
        # the one-sided asymptotic p-value exp(-2 m D^2)
        assert res.table["pvalue"].iloc[0] == pytest.approx(
            np.exp(-2 * m * d_plus**2), rel=1e-10
        )

    def test_median_test_is_pearson_chi2_on_the_above_median_table(self):
        df = _tied()
        parts3 = [df.loc[df.g3 == k, "y"] for k in range(3)]
        ref3 = stats.median_test(*parts3, correction=False, ties="below")
        res3 = sp.median_test(df, "y", by="g3")
        assert res3.statistic == pytest.approx(ref3.statistic, rel=1e-12)
        assert res3.pvalue == pytest.approx(ref3.pvalue, rel=1e-12)
        assert res3.df == 2
        rng = np.random.default_rng(1)
        two = pd.DataFrame(
            {"y": rng.normal(size=31) + (np.arange(31) % 2), "g": np.arange(31) % 2}
        )
        parts = [two.loc[two.g == k, "y"] for k in range(2)]
        plain = stats.median_test(*parts, correction=False, ties="below")
        yates = stats.median_test(*parts, correction=True, ties="below")
        res = sp.median_test(two, "y", by="g")
        assert res.statistic == pytest.approx(plain.statistic, rel=1e-12)
        assert res.estimates["chi2_corrected"] == pytest.approx(
            yates.statistic, rel=1e-12
        )
        assert res.estimates["median"] == pytest.approx(two["y"].median())

    def test_robvar_centres(self):
        df = _tied()
        parts = [df.loc[df.g3 == k, "y"].to_numpy() for k in range(3)]
        res = sp.robvar(df, "y", by="g3")
        w0 = stats.levene(*parts, center="mean")
        w50 = stats.levene(*parts, center="median")
        assert res.statistic == pytest.approx(w0.statistic, rel=1e-12)
        assert res.pvalue == pytest.approx(w0.pvalue, rel=1e-10)
        assert res.estimates["W50"] == pytest.approx(w50.statistic, rel=1e-12)
        assert res.estimates["p_W50"] == pytest.approx(w50.pvalue, rel=1e-10)
        assert res.df == (2, len(df) - 3)

    def test_oneway_anova_bartlett_and_comparisons(self):
        df = _tied()
        parts = [df.loc[df.g3 == k, "y"].to_numpy() for k in range(3)]
        ref = stats.f_oneway(*parts)
        res = sp.oneway(df, "y", by="g3", compare="bonferroni")
        assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)
        assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-10)
        est = res.estimates
        assert est["ss_total"] == pytest.approx(
            np.sum((df["y"] - df["y"].mean()) ** 2), rel=1e-12
        )
        bart = stats.bartlett(*parts)
        got = [v for k, v in est.items() if "bartlett" in k.lower()]
        assert any(v == pytest.approx(bart.statistic, rel=1e-10) for v in got)
        # pairwise comparisons: pooled-variance t with n - k df
        n, k = len(df), 3
        mse = est["ms_within"]
        for compare in ("bonferroni", "sidak", "scheffe"):
            out = sp.oneway(df, "y", by="g3", compare=compare).estimates
            for i, j in [(1, 0), (2, 0), (2, 1)]:
                diff = parts[i].mean() - parts[j].mean()
                se = np.sqrt(mse * (1 / len(parts[i]) + 1 / len(parts[j])))
                p = 2 * stats.t.sf(abs(diff / se), n - k)
                if compare == "bonferroni":
                    adj = min(1.0, 3 * p)
                elif compare == "sidak":
                    adj = 1 - (1 - p) ** 3
                else:
                    adj = stats.f.sf((diff / se) ** 2 / (k - 1), k - 1, n - k)
                levels = out["differences"].index
                assert out["differences"].loc[levels[i], levels[j]] == pytest.approx(
                    diff, abs=1e-12
                )
                assert out["pvalues"].loc[levels[i], levels[j]] == pytest.approx(
                    adj, abs=1e-10
                )

    def test_missing_values_are_dropped_pairwise(self):
        df = _tied()
        holes = df.assign(
            y=df["y"].where(df.index >= 4), g=df["g"].where(df.index < 36)
        )
        clean = holes.dropna(subset=["y", "g"])
        a = sp.ranksum(holes, "y", by="g")
        b = sp.ranksum(clean, "y", by="g")
        assert a.n_obs == b.n_obs == 32
        assert a.statistic == pytest.approx(b.statistic, rel=1e-12)

    @pytest.mark.parametrize(
        "call, exc",
        [
            (lambda d: sp.ranksum(d, "y", by="g3"), MethodIncompatibility),
            (lambda d: sp.ranksum(d.assign(y=1.0), "y", by="g"), DataInsufficient),
            (lambda d: sp.kwallis(d.assign(y=1.0), "y", by="g3"), DataInsufficient),
            (lambda d: sp.ktau(d.assign(y=1.0), "x", "y"), DataInsufficient),
            (lambda d: sp.ktau(d.iloc[:2], "x", "y"), DataInsufficient),
            (lambda d: sp.spearman(d.iloc[:2], "x", "y"), DataInsufficient),
            (lambda d: sp.signrank(d.assign(x=d["y"]), "y", "x"), DataInsufficient),
            (lambda d: sp.median_test(d.assign(y=1.0), "y", by="g"), DataInsufficient),
            (
                lambda d: sp.robvar(d.assign(g=[0] * 39 + [1]), "y", by="g"),
                DataInsufficient,
            ),
            (
                lambda d: sp.oneway(d, "y", by="g3", compare="tukey"),
                MethodIncompatibility,
            ),
            (lambda d: sp.ranksum(d, "nope", by="g"), MethodIncompatibility),
            (lambda d: sp.ranksum(d.to_numpy(), "y", by="g"), MethodIncompatibility),
        ],
    )
    def test_degenerate_inputs_are_refused(self, call, exc):
        with pytest.raises(exc):
            call(_tied())

    def test_spearman_refuses_a_constant_variable(self):
        with pytest.raises(DataInsufficient):
            sp.spearman(_tied().assign(y=1.0), "x", "y")

    def test_oneway_refuses_a_constant_outcome(self):
        with pytest.raises(DataInsufficient):
            sp.oneway(_tied().assign(y=2.0), "y", by="g3")


# --------------------------------------------------------------------- #
#  t, z, variance and proportion tests
# --------------------------------------------------------------------- #


def _two() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 30
    return pd.DataFrame(
        {"a": rng.normal(1, 2, n), "b": rng.normal(1.5, 1, n), "g": np.arange(n) % 2}
    )


class TestTTest:
    def test_one_sample_and_its_one_sided_p_values(self):
        df = _two()
        res = sp.ttest(df, "a", mu=0.5)
        ref = stats.ttest_1samp(df["a"], 0.5)
        assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)
        assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
        less = stats.ttest_1samp(df["a"], 0.5, alternative="less").pvalue
        greater = stats.ttest_1samp(df["a"], 0.5, alternative="greater").pvalue
        assert res.pvalue_less == pytest.approx(less, rel=1e-12)
        assert res.pvalue_greater == pytest.approx(greater, rel=1e-12)
        assert res.pvalue_less + res.pvalue_greater == pytest.approx(1.0)
        ci = ref.confidence_interval(0.95)
        assert res.ci == pytest.approx((ci.low, ci.high), rel=1e-12)

    def test_paired_is_the_one_sample_test_on_differences(self):
        df = _two()
        paired = sp.ttest(df, "a", other="b", mu=0.3)
        one = sp.ttest(df.assign(d=df["a"] - df["b"]), "d", mu=0.3)
        assert paired.statistic == pytest.approx(one.statistic, rel=1e-12)
        assert paired.df == one.df == 29
        assert paired.ci == pytest.approx(one.ci, rel=1e-12)
        ref = stats.ttest_rel(df["a"], df["b"])
        assert sp.ttest(df, "a", other="b").pvalue == pytest.approx(ref.pvalue)

    def test_two_sample_pooled_unequal_and_welch(self):
        df = _two()
        a, b = df.loc[df.g == 0, "a"], df.loc[df.g == 1, "a"]
        pooled = sp.ttest(df, "a", by="g")
        ref = stats.ttest_ind(a, b)
        assert pooled.statistic == pytest.approx(ref.statistic, rel=1e-12)
        assert pooled.df == 28
        sat = sp.ttest(df, "a", by="g", unequal=True)
        ref_u = stats.ttest_ind(a, b, equal_var=False)
        assert sat.statistic == pytest.approx(ref_u.statistic, rel=1e-12)
        assert sat.df == pytest.approx(ref_u.df, rel=1e-12)
        assert sat.pvalue == pytest.approx(ref_u.pvalue, rel=1e-12)
        # Welch's own formula for the degrees of freedom
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        welch_df = -2 + (va + vb) ** 2 / (va**2 / (len(a) + 1) + vb**2 / (len(b) + 1))
        welch = sp.ttest(df, "a", by="g", welch=True)
        assert welch.df == pytest.approx(welch_df, rel=1e-12)
        assert welch.statistic == pytest.approx(sat.statistic, rel=1e-12)
        # unpaired comparison of two columns
        unp = sp.ttest(df, "a", other="b", paired=False)
        ref2 = stats.ttest_ind(df["a"], df["b"])
        assert unp.statistic == pytest.approx(ref2.statistic, rel=1e-12)
        assert unp.df == 58

    def test_null_value_shifts_the_statistic_not_the_interval(self):
        df = _two()
        base = sp.ttest(df, "a", by="g")
        shifted = sp.ttest(df, "a", by="g", mu=1.0)
        assert shifted.ci == pytest.approx(base.ci, rel=1e-12)
        assert shifted.statistic == pytest.approx(
            (base.estimate - 1.0) / base.se, rel=1e-12
        )

    def test_summary_statistics_reproduce_the_data_call(self):
        df = _two()
        a, b = df.loc[df.g == 0, "a"], df.loc[df.g == 1, "a"]
        for kw in ({}, {"unequal": True}, {"welch": True}):
            data = sp.ttest(df, "a", by="g", **kw)
            summ = sp.ttest(
                n=[len(a), len(b)],
                mean=[a.mean(), b.mean()],
                sd=[a.std(ddof=1), b.std(ddof=1)],
                **kw,
            )
            assert summ.statistic == pytest.approx(data.statistic, rel=1e-10)
            assert summ.df == pytest.approx(data.df, rel=1e-10)
            assert summ.ci == pytest.approx(data.ci, rel=1e-10)

    def test_missing_values(self):
        df = _two()
        holes = df.assign(b=df["b"].where(df.index >= 3))
        assert sp.ttest(holes, "a", other="b").n_obs == 27  # pairs
        unp = sp.ttest(holes, "a", other="b", paired=False)
        assert list(unp.groups["n"].iloc[:2]) == [30, 27]  # each column its own
        by = sp.ttest(df.assign(g=df["g"].where(df.index >= 3)), "a", by="g")
        assert by.n_obs == 27

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(y="a", alpha=2),
            dict(y="a", alpha=0),
            dict(y="a", by="g", other="b"),
            dict(y="c1"),  # constant
            dict(y="a", by="g3"),
            dict(y="a", by="solo"),
        ],
    )
    def test_invalid_requests_are_refused(self, kwargs):
        df = _two().assign(c1=1.0, g3=np.arange(30) % 3, solo=[0] * 29 + [1])
        with pytest.raises(MethodIncompatibility):
            sp.ttest(df, **kwargs)


class TestVarianceAndZTests:
    def test_one_sample_chi2(self):
        df = _two()
        res = sp.sdtest(df, "a", sd0=1.5)
        chi2 = 29 * df["a"].var(ddof=1) / 1.5**2
        assert res.statistic == pytest.approx(chi2, rel=1e-12)
        assert res.pvalue_greater == pytest.approx(stats.chi2.sf(chi2, 29), rel=1e-10)
        assert res.pvalue_less == pytest.approx(stats.chi2.cdf(chi2, 29), rel=1e-10)
        assert res.pvalue == pytest.approx(
            2 * min(stats.chi2.sf(chi2, 29), stats.chi2.cdf(chi2, 29)), rel=1e-10
        )
        s2 = df["a"].var(ddof=1)
        lo = np.sqrt(29 * s2 / stats.chi2.ppf(0.975, 29))
        hi = np.sqrt(29 * s2 / stats.chi2.ppf(0.025, 29))
        assert tuple(res.ci) == pytest.approx((lo, hi), rel=1e-10)

    def test_two_sample_variance_ratio(self):
        df = _two()
        a, b = df.loc[df.g == 0, "a"], df.loc[df.g == 1, "a"]
        f = a.var(ddof=1) / b.var(ddof=1)
        res = sp.sdtest(df, "a", by="g")
        assert res.statistic == pytest.approx(f, rel=1e-12)
        assert res.estimate == pytest.approx(np.sqrt(f), rel=1e-12)
        assert res.pvalue_less == pytest.approx(stats.f.cdf(f, 14, 14), rel=1e-10)
        assert res.pvalue_greater == pytest.approx(stats.f.sf(f, 14, 14), rel=1e-10)
        assert res.pvalue == pytest.approx(
            2 * min(stats.f.cdf(f, 14, 14), stats.f.sf(f, 14, 14)), rel=1e-10
        )
        summ = sp.sdtest(n=[15, 15], sd=[a.std(ddof=1), b.std(ddof=1)])
        assert summ.statistic == pytest.approx(f, rel=1e-10)
        other = sp.sdtest(df, "a", other="b")
        assert other.statistic == pytest.approx(
            df["a"].var(ddof=1) / df["b"].var(ddof=1), rel=1e-12
        )

    def test_ztest_uses_the_known_standard_deviations(self):
        df = _two()
        one = sp.ztest(df, "a", mu=0.5, sd=2)
        z = (df["a"].mean() - 0.5) / (2 / np.sqrt(30))
        assert one.statistic == pytest.approx(z, rel=1e-12)
        assert one.pvalue == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-10)
        a, b = df.loc[df.g == 0, "a"], df.loc[df.g == 1, "a"]
        two = sp.ztest(df, "a", by="g", sd=[2, 1])
        se = np.sqrt(4 / 15 + 1 / 15)
        assert two.se == pytest.approx(se, rel=1e-12)
        assert two.statistic == pytest.approx((a.mean() - b.mean()) / se, rel=1e-12)
        same = sp.ztest(df, "a", by="g", sd=2)
        assert same.se == pytest.approx(np.sqrt(8 / 15), rel=1e-12)
        crit = stats.norm.ppf(0.975)
        assert two.ci[1] == pytest.approx(two.estimate + crit * se, rel=1e-12)

    @pytest.mark.parametrize(
        "call",
        [
            lambda d: sp.sdtest(d, "a", sd0=-1),
            lambda d: sp.sdtest(d, "a"),
            lambda d: sp.sdtest(d.assign(a=1.0), "a", by="g"),
            lambda d: sp.sdtest(d, "a", sd0=1, alpha=1.0),
            lambda d: sp.ztest(d, "a", sd=0),
            lambda d: sp.ztest(d, "a", sd=[1, 2]),
        ],
    )
    def test_invalid_requests_are_refused(self, call):
        with pytest.raises(MethodIncompatibility):
            call(_two())


class TestProportions:
    def _frame(self):
        df = _two()
        return pd.DataFrame(
            {
                "u": (df["a"] > 1).astype(int),
                "v": (df["b"] > 1).astype(int),
                "g": df["g"],
            }
        )

    def test_one_sample_uses_the_null_variance_for_the_statistic(self):
        df = self._frame()
        k = int(df["u"].sum())
        z, p = proportions_ztest(k, 30, 0.3, prop_var=0.3)
        res = sp.prtest(df, "u", p=0.3)
        assert res.statistic == pytest.approx(z, rel=1e-12)
        assert res.pvalue == pytest.approx(p, rel=1e-10)
        phat = k / 30
        se = np.sqrt(phat * (1 - phat) / 30)  # the interval uses the sample one
        crit = stats.norm.ppf(0.975)
        assert res.ci == pytest.approx((phat - crit * se, phat + crit * se), rel=1e-10)

    def test_two_sample_pools_under_the_null(self):
        df = self._frame()
        counts = [int(df.loc[df.g == k, "u"].sum()) for k in (0, 1)]
        z, p = proportions_ztest(counts, [15, 15])
        res = sp.prtest(df, "u", by="g")
        assert res.statistic == pytest.approx(z, rel=1e-12)
        assert res.pvalue == pytest.approx(p, rel=1e-10)
        summ = sp.prtest(n=[15, 15], proportion=[counts[0] / 15, counts[1] / 15])
        assert summ.statistic == pytest.approx(z, rel=1e-10)
        cols = sp.prtest(df, "u", other="v")
        z2, _ = proportions_ztest([df["u"].sum(), df["v"].sum()], [30, 30])
        assert cols.statistic == pytest.approx(z2, rel=1e-12)

    def test_bitest_is_the_exact_binomial_test(self):
        df = self._frame()
        k = int(df["u"].sum())
        res = sp.bitest(df, "u", p=0.3)
        ref = stats.binomtest(k, 30, 0.3)
        assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-10)
        assert res.pvalue_greater == pytest.approx(
            stats.binomtest(k, 30, 0.3, alternative="greater").pvalue, rel=1e-10
        )
        assert res.pvalue_less == pytest.approx(
            stats.binomtest(k, 30, 0.3, alternative="less").pvalue, rel=1e-10
        )
        cp = ref.proportion_ci(0.95, method="exact")
        assert res.ci == pytest.approx((cp.low, cp.high), rel=1e-8)
        summ = sp.bitest(n=30, successes=k, p=0.3)
        assert summ.pvalue == pytest.approx(res.pvalue, rel=1e-12)

    @pytest.mark.parametrize(
        "call",
        [
            lambda d: sp.prtest(d.assign(u=d["u"] + 1), "u"),
            lambda d: sp.prtest(d, "u", p=0),
            lambda d: sp.prtest(d.assign(u=1), "u", by="g"),
            lambda d: sp.prtest(n=[15, 15], proportion=[0.4, 1.6]),
            lambda d: sp.bitest(n=10, successes=11),
            lambda d: sp.bitest(n=10, successes=3, p=1.5),
        ],
    )
    def test_invalid_requests_are_refused(self, call):
        with pytest.raises(MethodIncompatibility):
            call(self._frame())
