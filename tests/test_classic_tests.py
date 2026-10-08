"""The tests for comparing distributions and the influence statistics:
each against an independent implementation (scipy / statsmodels) or a
hand computation, on data with ties and missing values.

The Stata reference numbers for the same functions are pinned in
``tests/reference_parity/test_kohler_kreuter_stata_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 240
    out = pd.DataFrame(
        {
            "g": rng.integers(1, 4, size=n),
            "f": rng.integers(0, 2, size=n),
            "x": np.round(rng.normal(10, 3, size=n)),
        }
    )
    out["y"] = np.round(2 + 0.5 * out["x"] + out["g"] + 2 * rng.normal(size=n))
    out["z"] = np.round(out["y"] + rng.normal(0, 2, size=n))
    out["d"] = (out["x"] + 3 * rng.normal(size=n) > 10).astype(int)
    out.loc[rng.choice(n, 15, replace=False), "y"] = np.nan
    return out


def test_ranksum_is_the_normal_approximation_without_continuity(df):
    res = sp.ranksum(df, "y", by="f")
    d = df.dropna(subset=["y"])
    a, b = d.y[d.f == 0], d.y[d.f == 1]
    ref = stats.mannwhitneyu(a, b, use_continuity=False, method="asymptotic")
    assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-12)
    # the rank sum of the first group and its share of "wins"
    assert res.table["rank_sum"].iloc[0] == pytest.approx(
        ref.statistic + len(a) * (len(a) + 1) / 2
    )
    assert res.estimates["porder"] == pytest.approx(ref.statistic / (len(a) * len(b)))
    assert res.n_obs == len(d)


def test_kwallis_with_ties_is_scipy_kruskal(df):
    res = sp.kwallis(df, "y", by="g")
    d = df.dropna(subset=["y"])
    ref = stats.kruskal(*[d.y[d.g == k] for k in (1, 2, 3)])
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-10)
    assert res.estimates["chi2_unadjusted"] < res.statistic  # ties raise it
    assert res.df == 2


def test_rank_correlations(df):
    d = df.dropna(subset=["y"])
    rho = sp.spearman(df, "y", "x")
    ref = stats.spearmanr(d.y, d.x)
    assert rho.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert rho.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    tau = sp.ktau(df, "y", "x")
    assert tau.statistic == pytest.approx(
        stats.kendalltau(d.y, d.x).statistic, rel=1e-12
    )
    # tau-a counts every pair, tied or not
    conc = disc = 0
    yy, xx = d.y.to_numpy(), d.x.to_numpy()
    for i in range(len(d)):
        s = np.sign(yy[i] - yy[i + 1 :]) * np.sign(xx[i] - xx[i + 1 :])
        conc, disc = conc + int((s > 0).sum()), disc + int((s < 0).sum())
    assert tau.estimates["score"] == conc - disc
    assert tau.estimates["tau_a"] == pytest.approx(
        (conc - disc) / (len(d) * (len(d) - 1) / 2)
    )


def test_ksmirnov_statistic_and_median_test(df):
    d = df.dropna(subset=["y"])
    ks = sp.ksmirnov(df, "y", by="f")
    assert ks.statistic == pytest.approx(
        stats.ks_2samp(d.y[d.f == 0], d.y[d.f == 1]).statistic, rel=1e-12
    )
    assert ks.table["D"].iloc[0] >= 0 >= ks.table["D"].iloc[1]
    med = sp.median_test(df, "y", by="g")
    ref = stats.median_test(*[d.y[d.g == k] for k in (1, 2, 3)], ties="below",
                            correction=False)  # fmt: skip
    assert med.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert med.table.loc["total", "total"] == len(d)
    two = sp.median_test(df, "y", by="f")
    assert two.estimates["chi2_corrected"] < two.statistic


def test_robvar_and_oneway(df):
    d = df.dropna(subset=["y"])
    parts = [d.y[d.g == k].to_numpy() for k in (1, 2, 3)]
    rob = sp.robvar(df, "y", by="g")
    assert rob.statistic == pytest.approx(stats.levene(*parts, center="mean").statistic)
    assert rob.estimates["W50"] == pytest.approx(
        stats.levene(*parts, center="median").statistic
    )
    one = sp.oneway(df, "y", by="g", compare="bonferroni")
    ref = stats.f_oneway(*parts)
    assert one.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert one.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    assert one.estimates["bartlett_chi2"] == pytest.approx(
        stats.bartlett(*parts).statistic
    )
    diff = one.estimates["differences"]
    assert diff.loc[3, 1] == pytest.approx(parts[2].mean() - parts[0].mean())
    # pandas 3 keeps the empty upper triangle when stacking.
    assert (one.estimates["pvalues"].stack().dropna() <= 1).all()


def test_signrank_counts_zeros_and_ties(df):
    res = sp.signrank(df, "y", other="z")
    d = (df.y - df.z).dropna()
    assert res.n_obs == len(d)
    table = res.table
    assert table.loc["zero", "obs"] == int((d == 0).sum())
    assert table["rank_sum"].iloc[:3].sum() == pytest.approx(len(d) * (len(d) + 1) / 2)
    # symmetric about zero when the pair is swapped
    assert sp.signrank(df, "z", other="y").statistic == pytest.approx(-res.statistic)


def test_tests_refuse_what_they_cannot_answer(df):
    with pytest.raises(MethodIncompatibility, match="exactly 2 groups"):
        sp.ranksum(df, "y", by="g")
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.kwallis(df, "nope", by="g")
    with pytest.raises(DataInsufficient):
        sp.oneway(df.assign(one=1), "y", by="one")
    with pytest.raises(MethodIncompatibility, match="compare"):
        sp.oneway(df, "y", by="g", compare="tukey")


def test_influence_measures_match_statsmodels(df):
    import statsmodels.api as sm

    d = df.dropna(subset=["y"])
    fit = sp.regress("y ~ x + f", data=d)
    ours = sp.influence_measures(fit)
    ref = sm.OLS(d.y, sm.add_constant(d[["x", "f"]])).fit().get_influence()
    assert np.allclose(ours["leverage"], ref.hat_matrix_diag)
    assert np.allclose(ours["rstandard"], ref.resid_studentized_internal)
    assert np.allclose(ours["rstudent"], ref.resid_studentized_external)
    assert np.allclose(ours["cooksd"], ref.cooks_distance[0])
    assert np.allclose(ours["dfits"], ref.dffits[0])
    assert np.allclose(ours["covratio"], ref.cov_ratio)
    assert np.allclose(ours[["dfbeta_x", "dfbeta_f"]], ref.dfbetas[:, 1:])
    assert ours["leverage"].sum() == pytest.approx(3.0)
    with pytest.raises(MethodIncompatibility, match="linear regression"):
        sp.influence_measures(sp.logit("d ~ x", data=d))


def test_logit_diagnostics_by_covariate_pattern(df):
    fit = sp.logit("d ~ x + f", data=df)
    infl = sp.logit_influence(fit)
    n_patterns = df[["x", "f"]].drop_duplicates().shape[0]
    assert infl["pattern"].nunique() == n_patterns
    # one value per pattern, and the leverages add up to the parameters
    per_pattern = infl.groupby("pattern").first()
    assert per_pattern["hat"].sum() == pytest.approx(3.0, rel=1e-6)
    gof = sp.logit_gof(fit)
    assert gof["n_patterns"] == n_patterns and gof["df"] == n_patterns - 3
    assert gof["statistic"] == pytest.approx(
        float((per_pattern["residual"] ** 2).sum())
    )
    hl = sp.logit_gof(fit, groups=6)
    assert hl["df"] == hl["n_groups"] - 2
    assert hl["table"]["total"].sum() == len(df)
    assert hl["table"]["obs_1"].sum() == df["d"].sum()
    with pytest.raises(MethodIncompatibility, match="at least 3"):
        sp.logit_gof(fit, groups=2)


def test_sumstats_analytic_weights_and_total(df):
    d = df.dropna(subset=["y"]).assign(w=lambda t: 1.0 + (t.x % 3))
    out = sp.sumstats(d, vars=["y"], stats=["n", "mean", "sd", "sum"], weights="w",
                      output="numeric")  # fmt: skip
    mean = np.average(d.y, weights=d.w)
    var = np.average((d.y - mean) ** 2, weights=d.w) * len(d) / (len(d) - 1)
    assert out.loc["y", "Mean"] == pytest.approx(mean)
    assert out.loc["y", "Std. Dev."] == pytest.approx(np.sqrt(var))
    assert out.loc["y", "Sum"] == pytest.approx(len(d) * mean)
    # equal weights change nothing
    same = sp.sumstats(d.assign(w=2.0), vars=["y"], stats=["mean", "sd", "p25", "iqr"],
                       weights="w", percentile_method="stata", output="numeric")  # fmt: skip
    plain = sp.sumstats(d, vars=["y"], stats=["mean", "sd", "p25", "iqr"],
                        percentile_method="stata", output="numeric")  # fmt: skip
    assert np.allclose(same.to_numpy(dtype=float), plain.to_numpy(dtype=float))
    with pytest.raises(MethodIncompatibility, match="negative"):
        sp.sumstats(d.assign(w=-1.0), vars=["y"], weights="w")
    by = sp.sumstats(d, vars=["y"], by="g", stats=["n", "mean"], total=True,
                     output="numeric", by_labels={})  # fmt: skip
    assert by[("Total", "N")].iloc[0] == len(d)
    assert by[("Total", "Mean")].iloc[0] == pytest.approx(d.y.mean())


def test_regression_reads_python_keywords_and_factor_levels_like_stata():
    rng = np.random.default_rng(3)
    n = 200
    d = pd.DataFrame({"class": rng.integers(1, 4, n), "yield": rng.uniform(1, 2, n)})
    d["y"] = d["yield"] + (d["class"] == 3) + rng.normal(size=n)
    fit = sp.regress("y ~ C(class) + np.log(yield)", data=d)
    alias = d.rename(columns={"class": "k", "yield": "v"})
    ref = sp.regress("y ~ C(k) + np.log(v)", data=alias)
    assert list(fit.params.index) == ["Intercept", "C(class)[T.2]", "C(class)[T.3]",
                                      "np.log(yield)"]  # fmt: skip
    assert np.allclose(fit.params.to_numpy(), ref.params.to_numpy())
    # a level that never occurs with an observed outcome is not the base
    d.loc[d["class"] == 1, "y"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        part = sp.regress("y ~ C(class)", data=d)
        only = sp.regress("y ~ C(class)", data=d.dropna())
    assert list(part.params.index) == ["Intercept", "C(class)[T.3]"]
    assert np.allclose(part.params.to_numpy(), only.params.to_numpy())


def test_a_constant_regressor_without_an_intercept_is_the_mean():
    d = pd.DataFrame({"y": [1.0, 2.0, 4.0, 5.0], "one": 1.0})
    fit = sp.regress("y ~ one - 1", data=d)
    assert fit.params["one"] == pytest.approx(3.0)


def test_reset_on_the_regressors_survives_large_scales():
    rng = np.random.default_rng(5)
    n = 300
    d = pd.DataFrame({"a": rng.uniform(100, 5000, n), "b": rng.uniform(1e4, 9e4, n)})
    d["y"] = 3 + 0.01 * d.a + 1e-4 * d.b + rng.normal(size=n)
    out = sp.estat(sp.regress("y ~ a + b", data=d), "reset", rhs=True)
    scaled = d.assign(a=d.a / 1000, b=d.b / 1e4)
    ref = sp.estat(sp.regress("y ~ a + b", data=scaled), "reset", rhs=True)
    assert out["statistic"] > 0
    assert out["statistic"] == pytest.approx(ref["statistic"], rel=1e-6)


def test_association_measures_for_ordered_tables():
    from statspai.output.tab import association_tests

    rng = np.random.default_rng(11)
    a = rng.integers(1, 5, 400)
    b = np.clip(a + rng.integers(-1, 2, 400), 1, 4)
    out = association_tests(pd.crosstab(a, b))
    assert out["taub"] == pytest.approx(stats.kendalltau(a, b).statistic, rel=1e-12)
    conc = disc = 0
    for i in range(len(a)):
        s = np.sign(a[i] - a[i + 1 :]) * np.sign(b[i] - b[i + 1 :])
        conc, disc = conc + int((s > 0).sum()), disc + int((s < 0).sum())
    assert out["gamma"] == pytest.approx((conc - disc) / (conc + disc))
    assert 0 < out["gamma_ase"] < 0.1 and 0 < out["taub_ase"] < 0.1
