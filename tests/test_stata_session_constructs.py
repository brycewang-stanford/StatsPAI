"""Everyday do-file constructs in ``sp.stata``, and three translations that
used to run something other than the Stata command.

Found by replaying the Stata chapters of Clarke's *Applied
Microeconometrics* (``docs/dev/2026-10-04-clarke-applied-microeconometrics-review.md``).
Every construct here is written from the Stata manual and tested on
synthetic commands, not on the book's files.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession
from statspai.agent._translation._stata_session import stata_percentile
from statspai.exceptions import MethodIncompatibility


@pytest.fixture()
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = pd.DataFrame({"g": np.repeat(np.arange(12), 30)})
    out["x"] = rng.normal(size=360)
    out["w"] = rng.normal(size=360)
    out["y"] = 0.3 * out.x + rng.normal(size=360) + rng.normal(size=12)[out.g]
    return out


def run(commands: str, data: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(commands, data=data)


# ------------------------------------------------------------------ blocks
@pytest.mark.parametrize("opener", ["quietly {", "quietly{", "qui {", "noisily {"])
def test_quiet_block_runs_its_body(df, opener):
    direct = run("gen x2 = x^2\nreg y x x2\ndisplay _b[x2]", df)
    assert (
        run(f"{opener}\n gen x2 = x^2\n reg y x x2\n}}\ndisplay _b[x2]", df) == direct
    )


def test_a_loop_runs_its_body_inside_a_quiet_block(df):
    out = run(
        "quietly {\n forvalues i = 1/3 {\n  gen z`i' = x * `i'\n }\n}\n"
        "su z3\ndisplay r(N)",
        df,
    )
    assert out == 360


# ----------------------------------------------------------------- capture
def test_capture_swallows_what_stata_would_raise(df):
    assert run("capture drop no_such_variable\ncount", df) == 360
    assert run("capture restore\ncount", df) == 360
    assert run("capture gen x = 1\ncount", df) == 360  # already defined


def test_capture_does_not_hide_an_untranslated_command(df):
    with pytest.raises(MethodIncompatibility, match="frobnicate"):
        run("capture frobnicate x", df)


def test_quietly_prefix_on_a_data_step(df):
    assert run("qui gen x3 = x^3\nqui replace x3 = 0\nsu x3\ndisplay r(max)", df) == 0


# ---------------------------------------------------------------- by-group
def test_bysort_generate_counts_within_group(df):
    assert run("bys g: gen t = _n + 1999\nsu t\ndisplay r(max)", df) == 2029
    assert run("bysort g: gen n = _N\nsu n\ndisplay r(min)", df) == 30


def test_bysort_with_order_variable_and_subscript(df):
    out = run(
        "bysort g (x): gen double first = x[1]\ngen double d = x - first\n"
        "su d\ndisplay r(min)",
        df,
    )
    assert out == 0
    session = StataSession(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.run("bysort g (x): gen double lowest = x[1]")
        session.run("bysort g (x): gen double highest = x[_N]")
    held = session.data
    np.testing.assert_array_equal(
        held["lowest"], held.groupby("g")["x"].transform("min")
    )
    np.testing.assert_array_equal(
        held["highest"], held.groupby("g")["x"].transform("max")
    )


def test_bysort_replace_with_condition(df):
    out = run("gen z = 0\nbys g: replace z = _n if _n <= 2\nsu z\ndisplay r(sum)", df)
    assert out == 12 * 3


def test_plain_by_needs_sorted_data(df):
    with pytest.raises(MethodIncompatibility, match="not sorted"):
        run("gen u = mod(_n, 3)\nby u: gen k = _n", df)


def test_by_prefix_runs_an_estimation_command_group_by_group(df):
    out = run("bysort g: regress y x", df)
    assert set(out) == set(df["g"].unique())
    for level, fit in out.items():
        alone = sp.regress("y ~ x", data=df[df["g"] == level])
        assert np.allclose(fit.params.to_numpy(), alone.params.to_numpy())


def test_panel_built_from_scratch():
    out = run(
        "clear\nset obs 30\ngen unit = ceil(_n/10)\n"
        "bys unit: gen year = _n + 1999\nsu year\ndisplay r(max)",
        pd.DataFrame({"a": [1.0]}),
    )
    assert out == 2009


# ----------------------------------------------- count, percentiles, scalars
def test_count(df):
    assert run("count", df) == 360
    assert run("count if x > 0\ndisplay r(N)", df) == float((df.x > 0).sum())
    assert run("count if x > 0 in 1/100\ndisplay r(N)", df) == float(
        (df.x.iloc[:100] > 0).sum()
    )


def test_stata_percentile_definition():
    # [R] summarize: P = n p / 100; the mean of x[P] and x[P+1] when P is an
    # integer, x[ceil(P)] otherwise
    x = np.arange(1.0, 11.0)
    assert stata_percentile(x, 50) == 5.5
    assert stata_percentile(x, 25) == 3.0
    assert stata_percentile(x, 75) == 8.0
    assert stata_percentile(x, 10) == 1.5
    assert stata_percentile(x, 99) == 10.0
    assert stata_percentile(np.arange(1.0, 10.0), 50) == 5.0
    assert np.isnan(stata_percentile(np.array([]), 50))


def test_pctile_and_summarize_detail_leave_percentiles(df):
    x = df.x.to_numpy()
    assert run(
        "_pctile x, percentiles(2.5 97.5)\ndisplay r(r1)", df
    ) == stata_percentile(x, 2.5)
    assert run("_pctile x, p(2.5 97.5)\ndisplay r(r2)", df) == stata_percentile(x, 97.5)
    assert run("_pctile x\ndisplay r(r1)", df) == stata_percentile(x, 50)
    assert run("summarize x, detail\ndisplay r(p50)", df) == stata_percentile(x, 50)
    dev = x - x.mean()
    skew = float(np.mean(dev**3) / np.mean(dev**2) ** 1.5)
    assert run("summarize x, detail\ndisplay r(skewness)", df) == pytest.approx(skew)


def test_scalar_function_and_scalar_without_data(df):
    assert run("scalar a = 2\ngen z = x * scalar(a)\nsu z\ndisplay r(mean)", df) == (
        pytest.approx(2 * float(np.float32(1) * df.x.mean()), rel=1e-6)
    )
    assert run("clear\nscalar third = 1/3\ndisplay third", df) == 1 / 3
    with pytest.raises(MethodIncompatibility, match="not defined"):
        run("gen z = scalar(nope)", df)


def test_quoted_coefficient_name(df):
    assert run('reg y x w\ndisplay _b["x"]', df) == run("reg y x w\ndisplay _b[x]", df)


# ---------------------------------------------------- duplicates and bsample
def test_duplicates_drop():
    d = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [1, 1, 2, 3, 3]})
    assert run("duplicates drop\ncount", d) == 4
    assert run("duplicates drop a, force\ncount", d) == 3
    with pytest.raises(MethodIncompatibility, match="force"):
        run("duplicates drop a", d)


def test_bsample_resamples_and_marks_the_data_as_random(df):
    session = StataSession(df)
    with pytest.warns(UserWarning, match="random numbers"):
        session.run("bsample")
    assert session.simulated and len(session.data) == len(df)
    assert set(session.data["g"]).issubset(set(df["g"]))

    session = StataSession(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.run("set seed 1")
        session.run("bsample, cluster(g)")
    sizes = session.data.groupby("g").size()
    assert (sizes % 30 == 0).all() and len(session.data) == len(df)


# ------------------------------------------------- predict with i. variables
def test_predict_after_a_regression_with_factor_variables(df):
    session = StataSession(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.run("reg y x i.g")
        session.run("predict double yhat")
        session.run("predict double e, residuals")
    fit = sp.regress("y ~ x + C(g)", data=df)
    held = session.data
    fitted = np.asarray(fit.fitted_values(), dtype=float)
    np.testing.assert_allclose(held["yhat"], fitted, atol=1e-10)
    np.testing.assert_allclose(held["e"], df.y - fitted, atol=1e-10)


# ----------------------------------------------- psmatch2 leaves its variables
def test_psmatch2_leaves_its_variables():
    d = sp.datasets.nsw_lalonde()
    session = StataSession(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.run("psmatch2 treat (age educ black married), outcome(re78) logit")
        session.run("gen matched = (_weight != .)")
    held = session.data
    for name in ("_pscore", "_treated", "_support", "_weight"):
        assert name in held.columns
    assert held["matched"].sum() >= (d.treat == 1).sum()
    assert held["_pscore"].between(0, 1).all()


# ------------------------------------------------------------ translations
def test_teffects_ipw_asks_for_the_standard_error_stata_reports():
    out = sp.from_stata("teffects ipw (y) (d x1 x2), atet")
    assert out["tool"] == "ipw"
    assert out["arguments"]["se_method"] == "sandwich"
    assert out["arguments"]["estimand"] == "ATT"
    assert "ps_model" not in out["arguments"]
    assert "se_method='sandwich'" in out["python_code"]

    probit = sp.from_stata("teffects ipw (y) (d x1 x2, probit)")
    assert probit["arguments"]["ps_model"] == "probit"
    assert "ps_model='probit'" in probit["python_code"]


def test_teffects_aipw_is_not_cross_fitted():
    out = sp.from_stata("teffects aipw (y x1 x2) (d x1 x2)")
    assert out["arguments"]["cross_fit"] is False
    assert out["arguments"]["se_method"] == "sandwich"
    assert not sp.from_stata("teffects aipw (y x1) (d x1), atet")["ok"]
    assert not sp.from_stata("teffects aipw (y x1) (d x1, probit)")["ok"]


def test_teffects_psmatch_probit():
    out = sp.from_stata("teffects psmatch (y) (d x1 x2, probit), atet")
    assert out["arguments"]["ps_model"] == "probit"
    assert out["arguments"]["se_method"] == "abadie_imbens_2016"
    assert not sp.from_stata("teffects psmatch (y) (d x1, hetprobit(x1))")["ok"]


def test_psmatch2_default_is_probit():
    default = sp.from_stata("psmatch2 d x1 x2, outcome(y)")
    assert default["arguments"]["ps_model"] == "probit"
    assert default["untranslated_options"] == []
    logit = sp.from_stata("psmatch2 d x1 x2, outcome(y) logit")
    assert "ps_model" not in logit["arguments"]
    # Stata reads the parentheses as part of the variable list
    paren = sp.from_stata("psmatch2 d (x1 x2), outcome(y)")
    assert paren["arguments"]["covariates"] == ["x1", "x2"]


def test_translated_teffects_run_as_the_direct_calls():
    d = sp.datasets.nsw_lalonde()
    covariates = ["age", "educ", "black", "married"]
    ipw = run("teffects ipw (re78) (treat age educ black married, probit), atet", d)
    direct = sp.ipw(
        d, y="re78", treat="treat", covariates=covariates, estimand="ATT",
        se_method="sandwich", ps_model="probit",
    )  # fmt: skip
    assert ipw.estimate == direct.estimate and ipw.se == direct.se
    aipw = run(
        "teffects aipw (re78 age educ black married) (treat age educ black married)", d
    )
    direct = sp.aipw(
        d, y="re78", treat="treat", covariates=covariates, cross_fit=False,
        se_method="sandwich",
    )  # fmt: skip
    assert aipw.estimate == direct.estimate and aipw.se == direct.se


# ---------------------------------------------------------------- boottest
REG = "reg y x w, cluster(g)"


def test_boottest_translation_names_a_function_that_exists():
    out = sp.from_stata("boottest x = 0.1, reps(999) weight(webb) seed(3) nograph")
    assert out["tool"] == "wild_cluster_boot"
    assert out["arguments"] == {
        "variable": "x", "n_boot": 999, "weight_type": "webb", "seed": 3, "h0": 0.1,
        "confidence_set": True,
    }  # fmt: skip
    assert out["untranslated_options"] == []
    assert "confidence_set" not in sp.from_stata("boottest x, noci")["arguments"]
    assert out["python_code"].startswith("sp.wild_cluster_boot(result, data=df")


def test_boottest_runs_on_the_regression_before_it(df):
    out = run(REG + "\nboottest x, reps(99999)", df)
    fit = sp.regress("y ~ x + w", data=df, cluster="g")
    direct = sp.wild_cluster_boot(fit, data=df, cluster="g", variable="x", n_boot=99999)
    # 12 clusters: the 4,096 Rademacher draws are enumerated, so the p-value
    # does not depend on a seed
    assert out["enumerated"] and out["n_boot"] == 4096
    assert out["p_boot"] == direct["p_boot"]
    assert out["t_stat"] == pytest.approx(float(fit.tvalues["x"]), rel=1e-10)


def test_boottest_null_value(df):
    out = run(REG + "\nboottest x = 0.3, reps(99999)", df)
    fit = sp.regress("y ~ x + w", data=df, cluster="g")
    t = (float(fit.params["x"]) - 0.3) / float(fit.std_errors["x"])
    assert out["h0"] == 0.3
    assert out["t_stat"] == pytest.approx(t, rel=1e-10)
    # the same test through the data-first entry point
    other = sp.wild_cluster_bootstrap(
        df, y="y", x=["x", "w"], cluster="g", test_var="x", h0=0.3, n_boot=99999
    )
    assert out["p_boot"] == pytest.approx(other["p_boot"], abs=1e-12)


def test_boottest_confidence_set_is_the_inverted_test(df):
    out = run(REG + "\nboottest x, reps(99999)", df)
    lo, hi = out["ci_inverted"]
    assert lo < out["beta_hat"] < hi
    fit = sp.regress("y ~ x + w", data=df, cluster="g")
    kw = dict(data=df, cluster="g", variable="x", n_boot=99999)
    width = hi - lo
    # just inside each endpoint the null is not rejected, just outside it is
    for inside, outside in ((lo + 1e-9 * width, lo - 1e-9 * width),
                            (hi - 1e-9 * width, hi + 1e-9 * width)):  # fmt: skip
        assert sp.wild_cluster_boot(fit, h0=inside, **kw)["p_boot"] >= 0.05
        assert sp.wild_cluster_boot(fit, h0=outside, **kw)["p_boot"] < 0.05
    # the data-first function finds the same set
    other = sp.wild_cluster_ci_inv(
        df, y="y", x=["x", "w"], cluster="g", test_var="x", n_boot=99999,
        weight_type="rademacher",
    )  # fmt: skip
    assert other["ci"] == pytest.approx((lo, hi), abs=1e-9)
    # not asked for, not computed
    assert "ci_inverted" not in sp.wild_cluster_boot(fit, **kw)
    assert "ci_inverted" not in run(REG + "\nboottest x, reps(999) noci", df)


def test_boottest_weights_and_bootcluster(df):
    out = run(REG + "\nboottest x, reps(199) weight(webb) bootcluster(g) seed(1)", df)
    assert out["weight_type"] == "webb" and out["n_boot"] == 199


@pytest.mark.parametrize(
    "line, message",
    [
        ("boottest x, bootcluster(w)", "bootstraps 'w'"),
        ("boottest x, weight(gamma)", "weight"),
        ("boottest x w", "single-coefficient"),
        ("boottest x, nonull", "nonull"),
    ],
)
def test_boottest_refuses_what_it_does_not_translate(df, line, message):
    with pytest.raises(MethodIncompatibility, match=message):
        run(REG + "\n" + line, df)


def test_boottest_needs_a_clustered_regression(df):
    with pytest.raises(MethodIncompatibility, match="not clustered"):
        run("reg y x w\nboottest x", df)


def test_wild_cluster_boot_h0_default_is_unchanged(df):
    fit = sp.regress("y ~ x + w", data=df, cluster="g")
    kw = dict(data=df, cluster="g", variable="x", n_boot=99999)
    a = sp.wild_cluster_boot(fit, **kw)
    b = sp.wild_cluster_boot(fit, h0=0.0, **kw)
    assert a["p_boot"] == b["p_boot"] and a["t_stat"] == b["t_stat"]
    assert a["h0"] == 0.0
