"""Regression tests for the October 2026 review of the classic toolbox.

Each test states a proposition that was false before the fix: an identity,
a number computed independently with numpy / scipy / pandas, or the class
of the exception or warning a degenerate input must produce.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)


# --------------------------------------------------------------- 1. .dta
def test_categorical_column_keeps_its_labels_in_dta(tmp_path):
    df = pd.DataFrame(
        {
            "region": pd.Categorical(["north", "south", None, "west"]),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    path = sp.write_data(df, tmp_path / "cat.dta")
    # independent reader: the label table is in the file
    with pd.io.stata.StataReader(str(path)) as reader:
        tables = reader.value_labels()
    assert {int(k): v for k, v in tables["region"].items()} == {
        0: "north",
        1: "south",
        2: "west",
    }
    back = sp.read_data(path)
    assert back.attrs["_value_labels"]["region"] == {
        0: "north",
        1: "south",
        2: "west",
    }
    decoded = sp.decode(back)["region"]
    assert decoded.astype(object).where(decoded.notna(), None).tolist() == [
        "north",
        "south",
        None,
        "west",
    ]


def test_decoded_frame_round_trips_its_texts_through_dta(tmp_path):
    df = pd.DataFrame({"edu": [3, 1, 2, 1], "w": [1.5, 2.5, 3.5, 4.5]})
    sp.label_values(df, "edu", {1: "primary", 2: "secondary", 3: "tertiary"})
    texts = sp.decode(df)["edu"].tolist()
    path = sp.write_data(sp.decode(df), tmp_path / "decoded.dta")
    assert sp.decode(sp.read_data(path))["edu"].tolist() == texts


def test_numeric_value_labels_are_written_as_before(tmp_path):
    df = pd.DataFrame({"edu": [3, 1, 2, 1]})
    sp.label_values(df, "edu", {1: "primary", 2: "secondary", 3: "tertiary"})
    back = sp.read_data(sp.write_data(df, tmp_path / "num.dta"))
    assert back["edu"].tolist() == [3, 1, 2, 1]
    assert back.attrs["_value_labels"]["edu"] == {
        1: "primary",
        2: "secondary",
        3: "tertiary",
    }


# ------------------------------------------------- 2. feather, tsv, types
def test_feather_keeps_value_labels_with_integer_codes(tmp_path):
    df = pd.DataFrame({"edu": [3, 1, 2, 1]})
    labels = {1: "primary", 2: "secondary", 3: "tertiary"}
    sp.label_values(df, "edu", labels)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "labels are not stored" warning
        path = sp.write_data(df, tmp_path / "t.feather")
    back = sp.read_data(path)
    assert back.attrs["_value_labels"] == {"edu": labels}
    assert sp.decode(back)["edu"].tolist() == [
        "tertiary",
        "primary",
        "secondary",
        "primary",
    ]


def test_feather_matches_parquet_for_labels_passed_as_arguments(tmp_path):
    df = pd.DataFrame({"female": [0, 1, 1]})
    kwargs = dict(
        labels={"female": "Sex"}, value_labels={"female": {0: "male", 1: "female"}}
    )
    a = sp.read_data(sp.write_data(df, tmp_path / "a.feather", **kwargs))
    b = sp.read_data(sp.write_data(df, tmp_path / "a.parquet", **kwargs))
    assert a.attrs["_value_labels"] == b.attrs["_value_labels"]
    assert a.attrs["_labels"] == b.attrs["_labels"] == {"female": "Sex"}


def test_tsv_round_trip_is_tab_separated(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5], "c": ["x, y", "z"]})
    back = sp.read_data(sp.write_data(df, tmp_path / "t.tsv"))
    pd.testing.assert_frame_equal(back, df)


def test_csv_still_warns_that_labels_are_lost(tmp_path):
    df = pd.DataFrame({"edu": [1, 2]})
    sp.label_values(df, "edu", {1: "a", 2: "b"})
    with pytest.warns(UserWarning, match="cannot store"):
        sp.write_data(df, tmp_path / "t.csv")


def test_write_data_rejects_what_is_not_a_frame(tmp_path):
    with pytest.raises(TypeError, match="DataFrame"):
        sp.write_data([1, 2, 3], tmp_path / "x.csv")


# ------------------------------------------- 3. unseen factor levels
@pytest.fixture(scope="module")
def poll():
    rng = np.random.default_rng(19)
    n = 600
    df = pd.DataFrame(
        {
            "pid": rng.choice(["R", "D", "I"], n, p=[0.5, 0.3, 0.2]),
            "k": rng.integers(1, 4, n),
        }
    )
    share = df["pid"].map({"R": 0.8, "D": 0.1, "I": 0.4})
    df["vote"] = (rng.uniform(size=n) < share).astype(int)
    return df


@pytest.mark.parametrize("fit", ["logit", "probit", "regress"])
def test_predict_rejects_a_level_the_model_did_not_see(poll, fit):
    model = getattr(sp, fit)("vote ~ C(pid)", data=poll)
    with pytest.raises(MethodIncompatibility, match=r"'pid'.*\['X'\]"):
        model.predict(pd.DataFrame({"pid": ["R", "X"]}))


@pytest.mark.parametrize("fit", ["logit", "probit", "regress"])
def test_predict_on_known_levels_is_the_cell_mean(poll, fit):
    # a saturated model reproduces the cell means, whatever the link
    model = getattr(sp, fit)("vote ~ C(pid)", data=poll)
    means = poll.groupby("pid")["vote"].mean()
    full = np.asarray(model.predict(pd.DataFrame({"pid": ["D", "I", "R"]})))
    np.testing.assert_allclose(full, means[["D", "I", "R"]], atol=1e-6)
    # a subset of the levels, the reference level included or not
    for subset in (["R"], ["D"], ["I", "D"]):
        got = np.asarray(model.predict(pd.DataFrame({"pid": subset})))
        np.testing.assert_allclose(got, means[subset], atol=1e-6)


def test_numeric_factor_levels_match_whatever_their_dtype(poll):
    model = sp.regress("vote ~ C(k)", data=poll)
    means = poll.groupby("k")["vote"].mean()
    got = model.predict(pd.DataFrame({"k": [1.0, 3.0]}))
    np.testing.assert_allclose(got, means[[1, 3]], atol=1e-10)
    with pytest.raises(MethodIncompatibility, match=r"'k'.*\[7\]"):
        model.predict(pd.DataFrame({"k": [1, 7]}))


def test_poststratify_refuses_a_cell_the_model_cannot_predict(poll):
    model = sp.logit("vote ~ C(pid)", data=poll)
    with pytest.raises(MethodIncompatibility, match="not fitted on"):
        sp.poststratify(model, pd.DataFrame({"pid": ["X"], "N": [1]}))
    cells = pd.DataFrame({"pid": ["R", "D", "I"], "N": [2.0, 1.0, 1.0]})
    means = poll.groupby("pid")["vote"].mean()
    expected = (2 * means["R"] + means["D"] + means["I"]) / 4
    assert sp.poststratify(model, cells).estimate == pytest.approx(expected, abs=1e-6)


# ------------------------------------------------------ 4. poststratify
def test_poststratify_rejects_missing_predictions_and_text_counts():
    cells = pd.DataFrame({"N": [1, 2]})
    with pytest.raises(MethodIncompatibility, match="missing or infinite"):
        sp.poststratify(np.array([np.nan, 0.2]), cells)
    with pytest.raises(MethodIncompatibility, match="non-negative, finite"):
        sp.poststratify([0.1, 0.2], pd.DataFrame({"N": [1.0, np.nan]}))
    with pytest.raises(MethodIncompatibility, match="must be numeric"):
        sp.poststratify([0.1, 0.2], pd.DataFrame({"N": ["a", "b"]}))
    out = sp.poststratify([0.2, 0.6], cells)
    assert out.estimate == pytest.approx((0.2 + 2 * 0.6) / 3)


# -------------------------------------------------- 5. influence_measures
def test_influence_measures_rejects_a_weighted_fit():
    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["y"] = 1 + df.x + rng.normal(size=n)
    df["w"] = rng.random(n) * 3 + 0.2
    with pytest.raises(MethodIncompatibility, match="weighted"):
        sp.influence_measures(sp.regress("y ~ x", data=df, weights="w"))
    out = sp.influence_measures(sp.regress("y ~ x", data=df))
    X = np.column_stack([np.ones(n), df.x])
    hat = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
    np.testing.assert_allclose(out["leverage"], hat, atol=1e-12)


# -------------------------------------------------------------- 6. nlcom
def test_nlcom_reads_a_name_with_brackets_of_its_own():
    rng = np.random.default_rng(9)
    n = 60
    df = pd.DataFrame({"x": rng.normal(size=n), "race": rng.integers(1, 4, n)})
    df["y"] = 1 + 2 * df.x + 0.5 * df.race + rng.normal(size=n)
    fit = sp.regress("y ~ x + C(race)", data=df)
    b = fit.params
    V = pd.DataFrame(
        fit.data_info["var_cov"], index=b.index, columns=b.index, dtype=float
    )
    num, den = "C(race)[T.2]", "x"
    ratio = b[num] / b[den]
    grad = pd.Series(0.0, index=b.index)
    grad[num], grad[den] = 1 / b[den], -b[num] / b[den] ** 2
    se = float(np.sqrt(grad @ V @ grad))
    for text in (
        "_b[C(race)[T.2]] / _b[x]",
        "_b[2.race] / _b[x]",
        "_b[ C(race)[T.2] ] / x",
    ):
        out = sp.nlcom(fit, text)
        assert out["estimate"] == pytest.approx(ratio, rel=1e-12)
        assert out["se"] == pytest.approx(se, rel=1e-10)
    with pytest.raises(MethodIncompatibility, match="never closed"):
        sp.nlcom(fit, "_b[C(race)[T.2] / _b[x]")


# ------------------------------------------- 7. zero standard errors
def test_ttest_refuses_a_constant_paired_difference():
    rng = np.random.default_rng(23)
    df = pd.DataFrame({"y": rng.normal(size=30)})
    df["x"] = df.y + 1  # the difference is -1 up to rounding, not exactly
    assert np.ptp((df.y - df.x).to_numpy()) > 0
    with pytest.raises(MethodIncompatibility, match="standard error is zero"):
        sp.ttest(df, "y", other="x")
    with pytest.raises(MethodIncompatibility, match="standard error is zero"):
        sp.ttest(pd.DataFrame({"y": [1e9 + 0.1] * 10}), "y", mu=0)


def test_ttest_matches_scipy_whatever_the_units():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40)})
    df["g"] = np.arange(40) % 2
    for scale in (1.0, 1e-12, 1e12):
        d = df.assign(y=df.y * scale, x=df.x * scale)
        paired = stats.ttest_rel(d.y, d.x)
        out = sp.ttest(d, "y", other="x")
        assert out.statistic == pytest.approx(paired.statistic, rel=1e-9)
        assert out.pvalue == pytest.approx(paired.pvalue, rel=1e-9)
        two = stats.ttest_ind(d.y[d.g == 0], d.y[d.g == 1])
        assert sp.ttest(d, "y", by="g").pvalue == pytest.approx(two.pvalue, rel=1e-9)
        one = stats.ttest_1samp(d.y, 0.0)
        assert sp.ttest(d, "y").pvalue == pytest.approx(one.pvalue, rel=1e-9)


def test_difference_in_means_warns_when_the_outcome_is_constant():
    df = pd.DataFrame({"y": np.ones(30), "g": np.arange(30) % 2})
    with pytest.warns(UserWarning, match="standard error is zero"):
        out = sp.difference_in_means(df, "y", "g")
    assert out.estimate == 0.0 and np.isnan(out.pvalue)
    # arms that differ but do not vary: the same, at any offset
    shifted = df.assign(y=0.1 + 0.2 * df.g)
    with pytest.warns(UserWarning, match="standard error is zero"):
        out = sp.difference_in_means(shifted, "y", "g")
    assert out.estimate == pytest.approx(0.2) and np.isnan(out.pvalue)


def test_difference_in_means_is_welch_on_ordinary_data():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"y": rng.normal(size=50), "g": np.arange(50) % 2})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sp.difference_in_means(df, "y", "g")
    ref = stats.ttest_ind(df.y[df.g == 1], df.y[df.g == 0], equal_var=False)
    assert out.pvalue == pytest.approx(ref.pvalue, rel=1e-10)


def test_jackknife_warns_when_the_statistic_never_moves():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(size=30)})
    with pytest.warns(RuntimeWarning, match="same value in every"):
        out = sp.jackknife(df, lambda d: 1.0)
    assert out.se == 0.0 and np.isnan(out.pvalue)
    # the mean: the jackknife standard error is s / sqrt(n) exactly
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean = sp.jackknife(df, lambda d: float(d.y.mean()))
    assert mean.se == pytest.approx(df.y.std(ddof=1) / np.sqrt(30), rel=1e-10)


# --------------------------------------------------------------- 8. ndcg
def test_ndcg_needs_a_binary_outcome():
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    with pytest.raises(MethodIncompatibility, match="0/1 outcome"):
        sp.ndcg([2, 0, 2, 0, 0], scores, k=3)
    with pytest.raises(MethodIncompatibility, match="0/1 outcome"):
        sp.ndcg([0.5, 0, 1, 0, 0], scores, k=3)
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    ideal = 1 / np.log2(2) + 1 / np.log2(3)
    for y in ([1, 0, 1, 0, 0], [True, False, True, False, False]):
        assert sp.ndcg(y, scores, k=3) == pytest.approx(dcg / ideal)


# ----------------------------------------------------------- 9. cor_test
def test_cor_test_counts_the_rank_of_the_covariates():
    rng = np.random.default_rng(23)
    n = 30
    e = pd.DataFrame(
        {"y": rng.normal(size=n), "x": rng.normal(size=n), "z": rng.normal(size=n)}
    )
    e["z2"], e["c"] = 2 * e.z, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base = sp.cor_test(e, "x", "y", covariates=["z"])
    assert base.df == n - 3
    with pytest.warns(UserWarning, match="z2 is redundant"):
        twice = sp.cor_test(e, "x", "y", covariates=["z", "z2"])
    assert twice.df == base.df
    assert twice.statistic == pytest.approx(base.statistic, abs=1e-12)
    assert twice.pvalue == pytest.approx(base.pvalue, rel=1e-9)
    with pytest.warns(UserWarning, match="c is redundant"):
        const = sp.cor_test(e, "x", "y", covariates=["c"])
    plain = stats.pearsonr(e.x, e.y)
    assert const.df == n - 2
    assert const.pvalue == pytest.approx(plain.pvalue, rel=1e-9)


# ------------------------------------------- 10. constant and infinite data
def test_constant_outcomes_are_refused():
    df = pd.DataFrame({"y": np.ones(30), "g": np.arange(30) % 2})
    with pytest.raises(DataInsufficient, match="single value"):
        sp.robvar(df, "y", "g")
    with pytest.raises(DataInsufficient, match="single value"):
        sp.loneway(df, "y", "g")


def test_infinite_values_are_refused():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": rng.normal(size=30), "x": rng.normal(size=30)})
    df["g"] = np.arange(30) % 2
    bad = df.assign(y=df.y.where(df.index > 0, np.inf))
    for call in (
        lambda: sp.swilk(bad, "y"),
        lambda: sp.oneway(bad, "y", "g"),
        lambda: sp.robvar(bad, "y", "g"),
        lambda: sp.loneway(bad, "y", "g"),
    ):
        with pytest.raises(DataInsufficient, match="infinite"):
            call()
    with pytest.raises(MethodIncompatibility, match="infinite"):
        sp.ttest(bad, "y")
    with pytest.raises(MethodIncompatibility, match="infinite"):
        sp.pca(bad, ["y", "x"])


def test_robvar_w0_is_levene_centred_at_the_mean():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"y": rng.normal(size=40), "g": np.arange(40) % 2})
    ref = stats.levene(df.y[df.g == 0], df.y[df.g == 1], center="mean")
    out = sp.robvar(df, "y", "g")
    assert out.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert out.pvalue == pytest.approx(ref.pvalue, rel=1e-10)


# ------------------------------------------------------------ 11. factor
@pytest.fixture(scope="module")
def six_variables():
    f = np.random.default_rng(1).normal(size=(200, 2))
    load = np.random.default_rng(2).normal(size=(2, 6))
    noise = np.random.default_rng(3).normal(size=(200, 6))
    return pd.DataFrame(f @ load + noise, columns=[f"x{i}" for i in range(6)])


def test_ipf_says_when_it_stopped_short(six_variables):
    with pytest.warns(ConvergenceWarning, match="did not converge in 2"):
        short = sp.factor(six_variables, method="ipf", n_factors=2, maxiter=2)
    assert short.converged is False and short.n_iter == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = sp.factor(six_variables, method="ipf", n_factors=2)
    assert full.converged is True and 2 < full.n_iter < 1000
    # converged: the communalities are a fixed point of the iteration
    R = six_variables.corr().to_numpy()
    reduced = R.copy()
    np.fill_diagonal(reduced, 1 - full.uniqueness.to_numpy())
    vals, vecs = np.linalg.eigh(reduced)
    top = np.argsort(vals)[::-1][:2]
    again = ((vecs[:, top] * np.sqrt(vals[top])) ** 2).sum(axis=1)
    np.testing.assert_allclose(again, 1 - full.uniqueness.to_numpy(), atol=1e-6)


def test_methods_without_iterations_report_convergence(six_variables):
    for method in ("pf", "pcf"):
        res = sp.factor(six_variables, method=method, n_factors=2)
        assert res.converged is True and res.n_iter == 0
    ml = sp.factor(six_variables, method="ml", n_factors=2)
    assert ml.converged is True and ml.n_iter > 0


# ------------------------------------------------------------ 12. sdtest
def test_sdtest_lower_tail_is_the_chi2_cdf():
    res = sp.sdtest(n=10, sd=1.3**0.5, sd0=2)
    assert res.pvalue_less == pytest.approx(stats.chi2.cdf(9 * 1.3 / 4, 9), rel=1e-12)
    assert round(res.pvalue_less, 4) == 0.0328


# ------------------------------------------------- 13. small validations
def test_bitest_validates_alpha_and_counts():
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.bitest(n=10, successes=3, alpha=1.5)
    with pytest.raises(MethodIncompatibility, match="whole number"):
        sp.bitest(n=10.5, successes=3)
    res = sp.bitest(n=10, successes=3)
    lower = stats.beta.ppf(0.025, 3, 8)
    upper = stats.beta.ppf(0.975, 4, 7)
    assert res.ci == pytest.approx((lower, upper), rel=1e-12)


def test_pca_reads_a_string_as_one_name(six_variables):
    with pytest.raises(MethodIncompatibility, match="at least two variables"):
        sp.pca(six_variables, "x0")
    with pytest.raises(MethodIncompatibility, match="at least two variables"):
        sp.factor(six_variables, "x0")
    with pytest.raises(MethodIncompatibility, match="whole number"):
        sp.pca(six_variables, n_components=2.5)
    res = sp.pca(six_variables, n_components=2)
    values = np.sort(np.linalg.eigvalsh(six_variables.corr().to_numpy()))[::-1]
    np.testing.assert_allclose(res.eigenvalues["eigenvalue"], values, atol=1e-10)
    assert res.loadings.shape == (6, 2)


def test_lm_lin_rejects_the_treatment_among_the_covariates():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40)})
    df["d"] = np.arange(40) % 2
    for covariates in (["d"], ["x", "y"], ["x", "x"]):
        with pytest.raises(MethodIncompatibility, match="covariates must be"):
            sp.lm_lin(df, "y", "d", covariates)
    # one covariate named as a string is that covariate
    a = sp.lm_lin(df, "y", "d", "x")
    b = sp.lm_lin(df, "y", "d", ["x"])
    assert (a.estimate, a.se) == (b.estimate, b.se)


@pytest.fixture(scope="module")
def strata_frame():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 6, 300)
    z = rng.binomial(1, 0.3 + 0.05 * s)
    y = 0.6 * z + 0.3 * s + rng.normal(size=300)
    return pd.DataFrame({"y": y, "z": z, "s": s})


def test_rosenbaum_stratified_names_a_text_treatment(strata_frame):
    text = strata_frame.assign(z=strata_frame.z.map({0: "c", 1: "t"}))
    with pytest.raises(MethodIncompatibility, match="coded 0/1"):
        sp.rosenbaum_stratified(text, "y", "z", "s")


def test_rosenbaum_stratified_counts_the_rows_it_drops(strata_frame):
    holed = strata_frame.copy()
    holed.loc[[3, 10, 50], "y"] = np.nan
    with pytest.warns(UserWarning, match="dropped 3 rows"):
        got = sp.rosenbaum_stratified(holed, "y", "z", "s")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = sp.rosenbaum_stratified(holed.dropna(), "y", "z", "s")
    assert got.pvalue == ref.pvalue and got.statistic == ref.statistic


def test_noether_test_counts_the_pairs_it_drops():
    d = np.random.default_rng(1).normal(0.5, 1.0, 100)
    holed = np.r_[d, np.nan, np.inf]
    with pytest.warns(UserWarning, match="dropped 2 pairs"):
        got = sp.noether_test(holed, f=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = sp.noether_test(d, f=0)
    assert got.pvalue == ref.pvalue
    # f = 0 is the sign test
    exact = stats.binom.sf(int((d > 0).sum()) - 1, int((d != 0).sum()), 0.5)
    assert ref.pvalue == pytest.approx(exact, rel=1e-12)


def test_binned_residuals_counts_what_it_drops():
    rng = np.random.default_rng(8)
    x, r = rng.normal(size=100), rng.normal(size=100)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = sp.binned_residuals(x, r)
    with pytest.warns(UserWarning, match="dropped 2 observations"):
        got = sp.binned_residuals(np.r_[x, np.nan, 0.0], np.r_[r, 0.0, np.nan])
    pd.testing.assert_frame_equal(got, ref)


def test_confounder_tip_has_no_answer_for_a_confounder_without_effect():
    with pytest.warns(UserWarning, match="no confounder"):
        out = sp.confounder_tip(1.0, confounder_outcome_effect=0)
    assert np.isnan(out["exposure_confounder_effect"].iloc[0])
    assert np.isnan(out["n_unmeasured_confounders"].iloc[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tip = sp.confounder_tip(6.58, confounder_outcome_effect=-7)
    assert tip["exposure_confounder_effect"].iloc[0] == pytest.approx(6.58 / -7)


def test_hausman_warns_about_a_robust_covariance():
    rng = np.random.default_rng(0)
    n = 2000
    z, u = rng.normal(size=n), rng.normal(size=n)
    d = z + 0.8 * u + rng.normal(size=n)
    df = pd.DataFrame({"y": 1 + 0.5 * d + u, "d": d, "z": z})
    df["g"] = np.arange(n) % 50
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        iv = sp.iv("y ~ (d ~ z)", data=df, small=False)
        ols = sp.regress("y ~ d", data=df)
        plain = sp.hausman(iv, ols, sigmamore=True)
    # the statistic is the quadratic form in the covariances as given
    robust_iv = sp.iv("y ~ (d ~ z)", data=df, small=False, robust="hc1")
    robust_ols = sp.regress("y ~ d", data=df, robust="hc1")
    with pytest.warns(AssumptionWarning, match="robust covariance"):
        out = sp.hausman(robust_iv, robust_ols)
    diff = robust_iv.params["d"] - robust_ols.params["d"]
    var = robust_iv.std_errors["d"] ** 2 - robust_ols.std_errors["d"] ** 2
    assert out["statistic"] == pytest.approx(diff**2 / var, rel=1e-8)
    with pytest.warns(AssumptionWarning, match="clustered by g"):
        sp.hausman(iv, sp.regress("y ~ d", data=df, cluster="g"))
    assert plain["df"] == 1 and plain["pvalue"] < 0.05
