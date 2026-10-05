"""Behaviour of the accounting-research tools away from the reference data:
edge cases, refusals, and the R translations that reach them."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n, t = 60, 8
    df = pd.DataFrame(
        {
            "firm": np.repeat(np.arange(n), t),
            "year": np.tile(np.arange(2000, 2000 + t), n),
        }
    )
    df["x"] = rng.normal(size=len(df))
    df["g"] = rng.integers(1, 4, len(df))
    shock = rng.normal(size=t)
    df["y"] = 1 + 0.5 * df["x"] + shock[df["year"] - 2000] + rng.normal(size=len(df))
    return df


# ---------------------------------------------------------------- fama_macbeth


def test_fama_macbeth_is_the_mean_of_the_cross_sections(panel):
    res = sp.fama_macbeth("y ~ x", panel, time="year")
    per = [sp.regress("y ~ x", d).params["x"] for _, d in panel.groupby("year")]
    assert res.params["x"] == pytest.approx(np.mean(per), rel=1e-12)
    assert res.std_errors["x"] == pytest.approx(
        np.std(per, ddof=1) / np.sqrt(len(per)), rel=1e-12
    )
    coefs = res.model_info["period_coefs"]
    assert list(coefs.index) == sorted(panel["year"].unique())
    assert coefs["nobs"].sum() == len(panel) == res.data_info["nobs"]
    # t(T - 1) inference
    assert res.data_info["df_resid"] == panel["year"].nunique() - 1


def test_fama_macbeth_lags_zero_is_the_plain_estimator(panel):
    a = sp.fama_macbeth("y ~ x", panel, time="year")
    b = sp.fama_macbeth("y ~ x", panel, time="year", lags=0)
    assert np.array_equal(a.std_errors.to_numpy(), b.std_errors.to_numpy())
    c = sp.fama_macbeth("y ~ x", panel, time="year", lags=2)
    assert not np.allclose(a.std_errors, c.std_errors)
    assert np.allclose(a.params, c.params)


def test_fama_macbeth_skips_a_period_it_cannot_estimate(panel):
    thin = pd.concat([panel, panel.iloc[:2].assign(year=2050)])
    with pytest.warns(UserWarning, match="left out"):
        res = sp.fama_macbeth("y ~ x + C(g)", thin, time="year")
    assert res.model_info["skipped_periods"] == [2050]
    assert res.model_info["n_periods"] == panel["year"].nunique()


def test_fama_macbeth_refusals(panel):
    with pytest.raises(MethodIncompatibility, match="not a column"):
        sp.fama_macbeth("y ~ x", panel, time="period")
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.fama_macbeth("y ~ x", panel, time="year", lags=-1)
    with pytest.raises(DataInsufficient, match="lags"):
        sp.fama_macbeth("y ~ x", panel, time="year", lags=8)
    with pytest.raises(DataInsufficient, match="at least two"):
        sp.fama_macbeth("y ~ x", panel[panel["year"] == 2000], time="year")


# ---------------------------------------------------------------- panel HAC


def test_hac_panel_with_zero_lags_is_hc0(panel):
    hac = sp.regress(
        "y ~ x", panel, robust="hac", hac_lags=0, hac_panel=("firm", "year")
    )
    hc0 = sp.regress("y ~ x", panel, robust="hc0")
    assert np.allclose(hac.std_errors, hc0.std_errors, rtol=1e-12)
    assert hac.model_info["hac_panel"] == ["firm", "year"]


def test_hac_panel_default_lags_use_the_number_of_periods(panel):
    res = sp.regress("y ~ x", panel, robust="hac", hac_panel=("firm", "year"))
    assert res.model_info["hac_lags"] == int(np.floor(4 * (8 / 100) ** (2 / 9)))


def test_hac_panel_accepts_dates_and_ignores_row_order(panel):
    dated = panel.assign(date=pd.to_datetime(panel["year"].astype(str) + "-12-31"))
    a = sp.regress("y ~ x", panel, robust="hac", hac_lags=2, hac_panel=("firm", "year"))
    b = sp.regress(
        "y ~ x",
        dated.sample(frac=1.0, random_state=0),
        robust="hac",
        hac_lags=2,
        hac_panel=("firm", "date"),
    )
    assert np.allclose(a.std_errors, b.std_errors, rtol=1e-12)


def test_hac_panel_refusals(panel):
    with pytest.raises(MethodIncompatibility, match="only apply"):
        sp.regress("y ~ x", panel, hac_panel=("firm", "year"))
    with pytest.raises(MethodIncompatibility, match="panel and the time column"):
        sp.regress("y ~ x", panel, robust="hac", hac_panel=("firm",))
    with pytest.raises(MethodIncompatibility, match="not in the data"):
        sp.regress("y ~ x", panel, robust="hac", hac_panel=("firm", "month"))
    dup = pd.concat([panel, panel.iloc[:3]])
    with pytest.raises(MethodIncompatibility, match="do not identify"):
        sp.regress("y ~ x", dup, robust="hac", hac_panel=("firm", "year"))


# ---------------------------------------------------------------- robreg


@pytest.fixture(scope="module")
def dirty() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n), "w": rng.normal(size=n)})
    df["y"] = 2 + 1.5 * df["x"] - 0.5 * df["w"] + rng.normal(size=n)
    df.loc[:19, "y"] += 40
    df.loc[:19, "x"] = 6  # bad leverage points
    return df


def test_mm_resists_leverage_points_that_a_huber_fit_does_not(dirty):
    ols = sp.regress("y ~ x + w", dirty)
    huber = sp.robreg("y ~ x + w", dirty, method="m")
    mm = sp.robreg("y ~ x + w", dirty)
    assert abs(ols.params["x"] - 1.5) > 2
    assert abs(huber.params["x"] - 1.5) > 0.5  # monotone M: no leverage protection
    assert abs(mm.params["x"] - 1.5) < 0.2
    w = mm.model_info["weights"]
    assert (w.iloc[:20] == 0).all() and w.iloc[20:].median() > 0.9
    assert w.index.equals(dirty.index)


def test_robreg_on_clean_data_is_close_to_ols():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"x": rng.normal(size=1000)})
    df["y"] = 1 + df["x"] + rng.normal(size=1000)
    ols = sp.regress("y ~ x", df)
    for kwargs in (dict(), dict(method="m"), dict(efficiency=0.95)):
        rob = sp.robreg("y ~ x", df, **kwargs)
        assert abs(rob.params["x"] - ols.params["x"]) < 0.03
        # the price of robustness: a standard error up to 1 / sqrt(efficiency)
        ratio = rob.std_errors["x"] / ols.std_errors["x"]
        assert 0.9 < ratio < 1.25


def test_robreg_tuning_constants_are_the_textbook_ones():
    from statspai.regression.robreg import _tuning_for_breakdown, _tuning_for_efficiency

    assert _tuning_for_efficiency("huber", 0.95) == pytest.approx(1.345, abs=1e-3)
    assert _tuning_for_efficiency("bisquare", 0.95) == pytest.approx(4.685, abs=1e-3)
    assert _tuning_for_efficiency("bisquare", 0.85) == pytest.approx(3.4437, abs=1e-4)
    assert _tuning_for_breakdown(0.5) == pytest.approx(1.547645, abs=1e-6)


def test_robreg_reads_factor_terms(dirty):
    df = dirty.assign(g=np.arange(len(dirty)) % 4)
    res = sp.robreg("y ~ x + w + factor(g)", df)
    assert "C(g)[T.3]" in res.params.index
    assert abs(res.params["x"] - 1.5) < 0.2


def test_robreg_refusals(dirty):
    with pytest.raises(MethodIncompatibility, match="method must be"):
        sp.robreg("y ~ x", dirty, method="lts")
    with pytest.raises(MethodIncompatibility, match="uses the bisquare"):
        sp.robreg("y ~ x", dirty, method="mm", psi="huber")
    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.robreg("y ~ x", dirty, efficiency=0.9, tuning=4.0)
    with pytest.raises(MethodIncompatibility, match="belong to method='m'"):
        sp.robreg("y ~ x", dirty, init="lad")
    with pytest.raises(MethodIncompatibility, match="collinear"):
        sp.robreg("y ~ x + I(2 * x)", dirty)
    with pytest.raises(DataInsufficient):
        sp.robreg("y ~ x + w", dirty.iloc[:3])


# ---------------------------------------------------------------- itcv


def test_itcv_of_a_non_significant_estimate_is_what_it_takes_to_get_there():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(size=200), "z": rng.normal(size=200)})
    df["y"] = 0.02 * df["x"] + df["z"] + rng.normal(size=200)
    out = sp.itcv(sp.regress("y ~ x + z", df), "x")
    assert not out["significant"]
    assert abs(out["r_obs"]) < abs(out["r_crit"])
    assert 0 < out["percent_bias"] < 1


def test_itcv_sign_follows_the_estimate():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.normal(size=300), "z": rng.normal(size=300)})
    df["y"] = -0.6 * df["x"] + 0.5 * df["z"] + rng.normal(size=300)
    out = sp.itcv(sp.regress("y ~ x + z", df), "x")
    assert out["significant"] and out["itcv"] < 0 and out["r_crit"] < 0
    assert out["beta_threshold"] < 0
    # with a negative estimate the threatening impacts are the negative ones
    assert out["benchmark"] == out["impacts"]["impact"].min()


def test_itcv_refusals():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"x": rng.normal(size=100)})
    df["y"] = df["x"] + rng.normal(size=100)
    fit = sp.regress("y ~ x", df)
    with pytest.raises(MethodIncompatibility, match="not a coefficient"):
        sp.itcv(fit, "w")
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.itcv(fit, "x", alpha=1.5)
    df["d"] = (df["y"] > 0).astype(int)
    with pytest.raises(MethodIncompatibility, match="linear regression"):
        sp.itcv(sp.logit("d ~ x", df), "x")
    # nothing to benchmark against: the threshold alone
    assert "impacts" not in sp.itcv(fit, "x")


# ---------------------------------------------------------------- ndcg, trim


def test_ndcg_bounds_and_counts():
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    perfect = np.array([0.1, 0.9, 0.2, 0.3, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert sp.ndcg(y, perfect, k=2) == 1.0
    assert sp.ndcg(y, -perfect, k=2) == 0.0
    assert sp.ndcg(y, perfect, k=0.2) == sp.ndcg(y, perfect, k=2)
    assert sp.ndcg(np.zeros(10), perfect, k=3) == 0.0
    # more slots than true cases: the ideal ranking fills what it can
    assert sp.ndcg(y, perfect, k=5) == 1.0
    with pytest.raises(MethodIncompatibility, match="selects no case"):
        sp.ndcg(y, perfect, k=0.01)
    with pytest.raises(MethodIncompatibility, match="equally long"):
        sp.ndcg(y, perfect[:5])


def test_trim_sets_the_tails_to_missing_and_keeps_the_cut_values():
    df = pd.DataFrame({"v": np.arange(1.0, 201.0), "g": np.repeat([0, 1], 100)})
    out = sp.winsor(df, ["v"], cuts=(1, 99), trim=True)
    assert list(out.columns) == ["v", "g", "v_tr"]
    # the 1st percentile of 1..200 is 2.5: 1 and 2 go, 3 stays
    assert out["v_tr"].isna().sum() == 4
    assert out["v_tr"].min() == 3 and out["v_tr"].max() == 198
    by = sp.winsor(df, ["v"], cuts=(1, 99), trim=True, by="g", replace=True)
    assert by["v"].isna().sum() == 4 and by.loc[99, "v"] != by.loc[99, "v"]
    named = sp.winsor(df, ["v"], trim=True, suffix="_cut")
    assert "v_cut" in named


# ---------------------------------------------------------------- formulas


@pytest.mark.parametrize(
    "fit",
    [
        lambda d: sp.regress("y ~ x + factor(g)", d),
        lambda d: sp.regress("y ~ x + as.factor(g)", d),
        lambda d: sp.feols("y ~ x + factor(g) | firm", d),
        lambda d: sp.glm("y ~ x + factor(g)", d),
        lambda d: sp.qreg("y ~ x + factor(g)", d),
        lambda d: sp.panel(d, "y ~ x + factor(g)", entity="firm", time="year"),
        lambda d: sp.fama_macbeth("y ~ x + factor(g)", d, time="year"),
    ],
)
def test_factor_is_c_in_every_entry_point(panel, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit(panel)
    labels = [str(n) for n in res.params.index]
    assert not any("factor" in n for n in labels)
    assert sum("g" in n for n in labels) == 2  # three levels, one reference


def test_a_column_named_factor_is_left_alone(panel):
    df = panel.rename(columns={"x": "factor"})
    res = sp.regress("y ~ factor + my_factor", df.assign(my_factor=df["g"]))
    assert list(res.params.index) == ["Intercept", "factor", "my_factor"]


# ---------------------------------------------------------------- Poisson


def test_poisson_from_a_far_start_and_without_separation_is_unchanged():
    """Counts in the thousands next to zeros used to send the first Newton
    step out of range; the estimates are the ones sp.glm finds."""
    rng = np.random.default_rng(8)
    n = 500
    df = pd.DataFrame({"x": rng.normal(size=n), "d": rng.integers(0, 2, n)})
    df["y"] = rng.poisson(np.exp(0.2 + 0.9 * df["x"] + 6 * df["d"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.poisson("y ~ x + d", df)
    ref = sp.glm("y ~ x + d", df, family="poisson")
    assert "separated_terms" not in res.model_info
    assert np.allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-9)
    assert np.allclose(res.std_errors.to_numpy(), ref.std_errors.to_numpy(), rtol=1e-7)


# ---------------------------------------------------------------- from_r


def _run(code: str, df: pd.DataFrame, result=None):
    return eval(code, {"sp": sp, "df": df, "result": result})  # noqa: S307


def test_from_r_feols_keeps_the_clustering(panel):
    """A positional `~ a + b` after the formula, or `vcov = ~ a + b`, is the
    covariance. It used to be dropped without a note."""
    for line in (
        "feols(y ~ x | firm, ~ firm + year, data = d)",
        "feols(y ~ x | firm, vcov = ~ firm + year, data = d)",
        "feols(y ~ x | firm, d, ~ firm + year)",
    ):
        out = sp.from_r(line)
        assert out["arguments"]["vcov"] == {"CRV1": "firm + year"}, line
        assert "untranslated_arguments" not in out
        two = _run(out["python_code"], panel)
        ref = sp.feols("y ~ x | firm", panel, vcov={"CRV1": "firm + year"})
        assert np.allclose(two.std_errors, ref.std_errors)
    one = sp.from_r("feols(y ~ x | firm, vcov = ~ firm, data = d)")
    assert one["arguments"]["cluster"] == "firm" and "vcov" not in one["arguments"]
    iid = sp.from_r('feols(y ~ x | firm, vcov = "iid", data = d)')
    assert iid["arguments"]["vcov"] == "iid"
    assert "vcov='iid'" in iid["python_code"]
    odd = sp.from_r('feols(y ~ x | firm, vcov = "twoway", data = d)')
    assert odd["untranslated_arguments"] == ["vcov"]
    assert "vcov" not in odd["arguments"]


def test_from_r_pmg_lmrob_rlm(panel, dirty):
    out = sp.from_r('pmg(y ~ x, data = test, index = "year")')
    assert out["tool"] == "fama_macbeth"
    res = _run(out["python_code"], panel)
    assert res.model_info["n_periods"] == 8
    assert not sp.from_r('pmg(y ~ x, data = d, index = "year", model = "cmg")')["ok"]

    out = sp.from_r(
        'lmrob(y ~ x + w, data = d, method = "MM", '
        "control = lmrob.control(tuning.psi = 3.4437))"
    )
    assert out["arguments"]["tuning"] == 3.4437 and out["arguments"]["small"] is False
    assert "untranslated_arguments" not in out
    assert abs(_run(out["python_code"], dirty).params["x"] - 1.5) < 0.2
    kept = sp.from_r(
        "lmrob(y ~ x, data = d, control = lmrob.control(tuning.psi = 3.4437, "
        "max.it = 100))"
    )
    assert kept["untranslated_arguments"] == ["control"]

    out = sp.from_r("rlm(y ~ x + w, data = d, psi = psi.bisquare)")
    assert out["arguments"] == {
        "formula": "y ~ x + w",
        "method": "m",
        "psi": "bisquare",
        "tuning": 4.685,
        "vce": "huber",
    }
    assert _run(out["python_code"], dirty).model_info["psi"] == "bisquare"
    assert not sp.from_r('rlm(y ~ x, data = d, method = "MM")')["ok"]


def test_from_r_rdrobust_binom_test_linear_hypothesis(panel):
    out = sp.from_r('rdrobust(d$sox, d$float2004, c = 75, masspoints = "off")')
    assert out["arguments"] == {
        "y": "sox",
        "x": "float2004",
        "c": 75.0,
        "masspoints": "off",
    }
    var = sp.from_r("rdrobust(d$y, d$x, fuzzy = d$t, c = cutoff)")
    assert var["untranslated_arguments"] == ["c"] and var["arguments"]["fuzzy"] == "t"
    assert not sp.from_r("rdrobust(y, log(x))")["ok"]

    out = sp.from_r("binom.test(x = 10, n = 1000, p = 0.05)")
    assert _run(out["python_code"], panel).pvalue == pytest.approx(
        sp.bitest(successes=10, n=1000, p=0.05).pvalue
    )
    assert sp.from_r("binom.test(90, 1000, 0.05)")["arguments"]["successes"] == 90

    fit = sp.regress("y ~ x + C(g) - 1", panel)
    out = sp.from_r('linearHypothesis(fm, "C(g)[1] = C(g)[3]")')
    got = _run(out["python_code"], panel, result=fit)
    assert got["pvalue"] == pytest.approx(sp.test(fit, "C(g)[1] = C(g)[3]")["pvalue"])


def test_from_r_matchit_negated_treatment_and_caliper():
    out = sp.from_r(
        "matchit(!big4 ~ size, data = d, caliper = 0.03, std.caliper = FALSE)"
    )
    assert out["arguments"]["treat"] == "not_big4"
    assert out["arguments"]["caliper"] == 0.03
    assert out["arguments"]["caliper_scale"] == "raw"
    assert "df.assign(not_big4=1 - df['big4'].astype(int))" in out["python_code"]
    assert "untranslated_arguments" not in out
    std = sp.from_r("matchit(treat ~ size, data = d, caliper = 0.2)")
    assert std["arguments"]["caliper_scale"] == "sd"
    assert "data=df," in std["python_code"]


# ---------------------------------------------------------------- from_stata


def test_from_stata_xtfmb_robreg_and_panel_newey(panel):
    out = sp.from_stata("xtfmb y x, lag(2) i(firm) t(year)")
    assert out["arguments"] == {"formula": "y ~ x", "time": "year", "lags": 2}
    assert not out["untranslated_options"]
    alone = sp.from_stata("xtfmb y x")
    assert alone["arguments"]["time"] == "<time>" and alone["notes"]

    out = sp.from_stata("robreg mm y x, efficiency(95) bp(25)")
    assert out["arguments"] == {
        "formula": "y ~ x",
        "method": "mm",
        "efficiency": 0.95,
        "breakdown": 0.25,
    }
    m = sp.from_stata("robreg m y x, biweight")
    assert m["arguments"]["psi"] == "bisquare"
    assert m["arguments"]["init"] == "lad" and m["arguments"]["scale"] == "fixed"
    assert not sp.from_stata("robreg lts y x")["ok"]
    assert sp.from_stata("robreg mm y x, nor2")["untranslated_options"] == ["nor2"]

    nw = sp.from_stata("newey y x, lag(2) force i(firm) t(year)")
    assert nw["arguments"]["hac_panel"] == ["firm", "year"]
    plain = sp.from_stata("newey y x, lag(2)")
    assert "hac_panel" not in plain["arguments"] and not plain["notes"]
    forced = sp.from_stata("newey y x, lag(2) force")
    assert forced["untranslated_options"] == ["force"]
    assert any("hac_panel" in note for note in forced["notes"])


def test_stata_script_fills_the_panel_from_xtset(panel):
    fm = sp.stata("xtset firm year\nxtfmb y x, lag(1)", panel)
    ref = sp.fama_macbeth("y ~ x", panel, time="year", lags=1)
    assert np.allclose(fm.std_errors, ref.std_errors)
    rob = sp.stata("robreg mm y x", panel)
    assert rob.model_info["estimator"] == "mm"


# ---------------------------------------------------------------- two-way PSD


@pytest.fixture(scope="module")
def few_years() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    n, t = 80, 6
    df = pd.DataFrame(
        {"firm": np.repeat(np.arange(n), t), "year": np.tile(np.arange(t), n)}
    )
    for j in range(4):
        df[f"x{j}"] = rng.normal(size=len(df))
    df["y"] = df["x0"] + rng.normal(size=t)[df["year"]] + rng.normal(size=len(df))
    return df


F_YEARS = "y ~ C(year) * (x0 + x1 + x2 + x3)"


def test_two_way_cluster_with_few_years_is_adjusted_and_says_so(few_years):
    with pytest.warns(RuntimeWarning, match="not positive semi-definite"):
        res = sp.regress(F_YEARS, few_years, cluster=["firm", "year"])
    V = res.vcov().to_numpy()
    assert res.diagnostics["Two-way VCOV negative eigenvalues"] >= 1
    assert np.linalg.eigvalsh((V + V.T) / 2).min() > -1e-12 * np.abs(V).max()
    assert np.allclose(np.sqrt(np.diag(V)), res.std_errors)
    assert (res.std_errors > 0).all()
    # sp.feols reports the matrix as computed, and says so
    with pytest.warns(RuntimeWarning, match="not positive semi-definite"):
        fe = sp.feols(F_YEARS, few_years, vcov={"CRV1": "firm + year"})
    assert fe.diagnostics["Multiway VCOV negative eigenvalues"] >= 1
    # where its variance is negative pyfixest has no standard error at all
    assert np.isnan(fe.std_errors.to_numpy()).any()


def test_two_way_cluster_that_is_psd_is_left_alone(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = sp.regress("y ~ x", panel, cluster=["firm", "year"])
        sp.feols("y ~ x", panel, vcov={"CRV1": "firm + year"})
    assert "Two-way VCOV negative eigenvalues" not in (res.diagnostics or {})
    assert np.allclose(np.sqrt(np.diag(res.vcov())), res.std_errors)
