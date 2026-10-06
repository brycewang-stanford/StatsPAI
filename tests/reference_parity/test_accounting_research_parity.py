"""Tools of empirical accounting research against R and Stata.

What a pass over Gow and Ding, *Empirical Research in Accounting: Tools and
Methods* (2024) added or changed: Fama-MacBeth regression, panel
Newey-West, robust (M / S / MM) regression, the impact threshold for a
confounding variable, trimming, NDCG, ``factor()`` in formulas and a
Poisson fit that survives quasi-separation.

The two data files are synthetic
(``_fixtures/_generate_accounting_research_data.py``). The references are
``accounting_research_R.json`` (R 4.5.2: plm, sandwich, fixest, robustbase,
MASS; ``_generate_accounting_research_R.R``) and
``accounting_research_Stata.csv`` (Stata 18: xtfmb, newey, robreg,
pkonfound, winsor2; ``_generate_accounting_research_Stata.do``), stored as
the programs wrote them. The do-file reads the data as doubles; read as
floats, ``newey`` agrees to 2.5e-9 only and ``winsor2, trim`` also removes
the two observations that sit exactly on the cut-offs.

Tolerances. ``EXACT`` (1e-9 relative) where both sides evaluate a closed
form. ``ITER`` (1e-6) where the reference stops an iteration at its own
tolerance (``lmrob``, ``robreg``, ``glm``) or stores intermediate results
in single precision (``xtfmb`` keeps the per-period coefficients as
floats). One comparison is *not* a parity claim and says so where it is
made: ``lmrob``'s standard errors.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-9
ITER = 1e-6
R_ORDER = ["Intercept", "x", "z"]
ROB = "y_out ~ x1 + x2 + x3"
ROB_R = ["Intercept", "x1", "x2", "x3"]
ROB_S = ["x1", "x2", "x3", "_cons"]


def rel(got, ref) -> float:
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(FIX / "accounting_research_panel.csv")


@pytest.fixture(scope="module")
def cross() -> pd.DataFrame:
    return pd.read_csv(FIX / "accounting_research_cross.csv")


@pytest.fixture(scope="module")
def R() -> dict:
    text = (FIX / "accounting_research_R.json").read_text(encoding="utf-8")
    return json.loads(text)


@pytest.fixture(scope="module")
def stata() -> pd.Series:
    table = pd.read_csv(FIX / "accounting_research_Stata.csv")
    return table.set_index(["model", "term"])["value"]


def stata_names(index) -> list:
    return ["_cons" if n == "Intercept" else n for n in index]


# ---------------------------------------------------------------------------
# Fama-MacBeth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key, gap", [("fm", False), ("fm_gap", True)])
def test_fama_macbeth_matches_plm_pmg(panel, R, key, gap):
    data = panel[panel["gap"] == 0] if gap else panel
    res = sp.fama_macbeth("y ~ x + z", data, time="year")
    ref = R[key]
    assert rel(res.params[R_ORDER], ref["coef"]) < EXACT
    assert rel(res.std_errors[R_ORDER], ref["se"]) < EXACT
    assert rel(res.vcov().loc[R_ORDER, R_ORDER], ref["vcov"]) < EXACT
    assert rel(res.model_info["avg_r2"], R[key + "_r2"]) < EXACT


@pytest.mark.parametrize("key, gap", [("fm", False), ("fm_gap", True)])
@pytest.mark.parametrize("lags", [1, 3])
def test_fama_macbeth_newey_west_matches_sandwich(panel, R, key, gap, lags):
    """NeweyWest(lm(b_t ~ 1), lag, prewhite = FALSE, adjust = TRUE) on each
    coefficient series, the construction in the book."""
    data = panel[panel["gap"] == 0] if gap else panel
    res = sp.fama_macbeth("y ~ x + z", data, time="year", lags=lags)
    ref = R[f"{key}_nw_{lags}"]
    names = ["Intercept" if n == "(Intercept)" else n for n in ref["names"]]
    assert rel(res.std_errors[names], ref["se"]) < EXACT


@pytest.mark.parametrize("lags", [0, 2])
def test_fama_macbeth_matches_stata_xtfmb(panel, stata, lags):
    res = sp.fama_macbeth("y ~ x + z", panel, time="year", lags=lags)
    ref = stata[f"xtfmb_lag{lags}"]
    V = res.vcov()
    for name in R_ORDER:
        s = "_cons" if name == "Intercept" else name
        assert rel(res.params[name], ref[f"b_{s}"]) < ITER
        assert rel(V.loc[name, name], ref[f"V_{s}"]) < 5 * ITER
    assert rel(res.model_info["avg_r2"], ref["r2"]) < ITER
    # xtfmb refers to t(T - 1) as well
    assert res.data_info["df_resid"] == ref["df_r"]


def test_fama_macbeth_unbalanced_matches_stata(panel, stata):
    res = sp.fama_macbeth("y ~ x + z", panel[panel["gap"] == 0], time="year")
    ref = stata["xtfmb_gap"]
    for name in R_ORDER:
        s = "_cons" if name == "Intercept" else name
        assert rel(res.params[name], ref[f"b_{s}"]) < ITER
        assert rel(res.std_errors[name], ref[f"se_{s}"]) < ITER


# ---------------------------------------------------------------------------
# the standard errors the book lines up next to Fama-MacBeth
# ---------------------------------------------------------------------------


def test_ols_white_and_clustered_match_r(panel, R):
    assert (
        rel(sp.regress("y ~ x + z", panel).std_errors[R_ORDER], R["ols"]["se"]) < EXACT
    )
    hc1 = sp.regress("y ~ x + z", panel, robust="hc1")
    assert rel(hc1.std_errors[R_ORDER], R["hc1"]["se"]) < EXACT
    for key, cluster in [
        ("cl_firm", "firm"),
        ("cl_year", "year"),
        ("cl_two", ["firm", "year"]),
    ]:
        res = sp.regress("y ~ x + z", panel, cluster=cluster)
        assert rel(res.std_errors[R_ORDER], R[key]["se"]) < EXACT
    two = sp.feols("y ~ x + z | firm + year", panel, vcov={"CRV1": "firm + year"})
    assert rel(two.params[["x", "z"]], R["fe_two"]["coef"]) < EXACT
    assert rel(two.std_errors[["x", "z"]], R["fe_two"]["se"]) < EXACT


@pytest.mark.parametrize("lags", [1, 3])
def test_panel_newey_west_matches_plm_and_stata(panel, R, stata, lags):
    shuffled = panel.sample(frac=1.0, random_state=3)  # row order must not matter
    res = sp.regress(
        "y ~ x + z", shuffled, robust="hac", hac_lags=lags, hac_panel=("firm", "year")
    )
    assert rel(res.std_errors[R_ORDER], R[f"panel_nw_{lags}"]["se"]) < EXACT
    small = sp.regress(
        "y ~ x + z",
        shuffled,
        robust="hac",
        hac_lags=lags,
        hac_panel=("firm", "year"),
        hac_small=True,
    )
    ref = stata[f"newey_lag{lags}"]
    got = [small.std_errors[n] for n in ["x", "z", "Intercept"]]
    assert rel(got, [ref["se_x"], ref["se_z"], ref["se__cons"]]) < EXACT


def test_panel_newey_west_skips_pairs_across_a_gap(panel, stata):
    """Stata pairs an observation with the one exactly j periods earlier;
    a missing year in between leaves no pair at that lag."""
    data = panel[panel["gap"] == 0]
    res = sp.regress(
        "y ~ x + z",
        data,
        robust="hac",
        hac_lags=2,
        hac_panel=("firm", "year"),
        hac_small=True,
    )
    ref = stata["newey_gap_lag2"]
    got = [res.std_errors[n] for n in ["x", "z", "Intercept"]]
    assert rel(got, [ref["se_x"], ref["se_z"], ref["se__cons"]]) < EXACT


def test_time_series_hac_on_stacked_panel_is_a_different_number(panel, R):
    """Without hac_panel the last year of one firm is paired with the first
    year of the next. The option exists because that is not Newey-West."""
    naive = sp.regress("y ~ x + z", panel, robust="hac", hac_lags=1)
    assert rel(naive.std_errors[R_ORDER], R["panel_nw_1"]["se"]) > 1e-4


# ---------------------------------------------------------------------------
# robust regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("robreg_mm85", dict(method="mm")),
        ("robreg_mm95", dict(method="mm", efficiency=0.95)),
        ("robreg_s", dict(method="s")),
        ("robreg_m_huber", dict(method="m", init="lad", scale="fixed")),
        (
            "robreg_m_biweight",
            dict(method="m", psi="bisquare", init="lad", scale="fixed"),
        ),
    ],
)
def test_robreg_matches_stata_robreg(cross, stata, key, kwargs):
    ref = stata[key]
    if kwargs["method"] == "m":
        # the M step and its covariance at robreg's scale; the scale rule
        # itself is the subject of the next test
        kwargs = {**kwargs, "scale": float(ref["scale"])}
    res = sp.robreg(ROB, cross, **kwargs)
    V = res.vcov()
    for name in ROB_R:
        s = "_cons" if name == "Intercept" else name
        assert rel(res.params[name], ref[f"b_{s}"]) < ITER
        assert rel(V.loc[name, name], ref[f"V_{s}"]) < ITER
    assert rel(res.model_info["scale"], ref["scale"]) < ITER
    assert rel(res.model_info["tuning"], ref["k"]) < ITER


def test_lad_start_scale_drops_the_interpolated_points(cross, stata):
    """An L1 fit passes through p observations. Their residuals are zero up
    to rounding and say nothing about the spread, so the scale is the
    normalised median of the other n - p absolute residuals. robreg drops
    the residuals that are *exactly* zero, which is the same set only when
    rounding leaves all p at 0.0; on this file it leaves three of the four
    (one is 4e-16), and its scale is the neighbouring order statistic."""
    res = sp.robreg(ROB, cross, method="m", init="lad", scale="fixed")
    ours = res.model_info["scale"]
    theirs = float(stata["robreg_m_huber"]["scale"])
    assert abs(ours / theirs - 1) < 2e-3
    X = np.column_stack([np.ones(len(cross))] + [cross[c] for c in ("x1", "x2", "x3")])
    from statspai.regression.robreg import _MADN, _lad

    a = np.sort(
        np.abs(cross["y_out"].to_numpy() - X @ _lad(X, cross["y_out"].to_numpy()))
    )
    assert (a[:4] < 1e-12).all() and a[4] > 1e-6
    assert rel(ours, np.median(a[4:]) / _MADN) < 1e-12
    assert rel(theirs, np.median(a[3:]) / _MADN) < 1e-7


@pytest.mark.parametrize("psi", [3.4437, 4.685061])
def test_robreg_mm_matches_lmrob(cross, R, psi):
    """Coefficients, S start, scale and robustness weights of lmrob, with
    its rounded S constant."""
    ref = R[f"lmrob_{psi}"]
    res = sp.robreg(ROB, cross, tuning=psi, tuning_s=1.54764, small=False)
    assert rel(res.params[ROB_R], ref["coef"]) < ITER
    assert rel(res.model_info["s_coefficients"][ROB_R], ref["s_coef"]) < ITER
    assert rel(res.model_info["scale"], ref["scale"]) < EXACT
    assert np.max(np.abs(res.model_info["weights"].to_numpy() - ref["weights"])) < ITER


@pytest.mark.parametrize("psi", [3.4437, 4.685061])
def test_robreg_mm_standard_errors_are_close_to_lmrob_but_not_claimed_equal(
    cross, R, psi
):
    """Not a parity claim. lmrob's default covariance (.vcov.avar1) and the
    stacked sandwich here agree to about 1e-5 once the n / (n - k) factor is
    dropped; the remaining gap is not explained (review document, open
    items). The same sandwich matches Stata robreg to 1e-8 above."""
    ref = R[f"lmrob_{psi}"]
    res = sp.robreg(ROB, cross, tuning=psi, tuning_s=1.54764, small=False)
    assert rel(res.std_errors[ROB_R], ref["se"]) < 5e-4


@pytest.mark.parametrize(
    "key, psi, tuning",
    [("rlm_huber", "huber", 1.345), ("rlm_bisquare", "bisquare", 4.685)],
)
def test_robreg_m_matches_mass_rlm(cross, R, key, psi, tuning):
    ref = R[key]
    res = sp.robreg(ROB, cross, method="m", psi=psi, tuning=tuning, vce="huber")
    assert rel(res.params[ROB_R], ref["coef"]) < ITER
    assert rel(res.std_errors[ROB_R], ref["se"]) < ITER
    assert rel(res.model_info["scale"], ref["scale"]) < ITER


def test_robreg_recovers_the_clean_fit_where_ols_does_not(cross):
    truth = np.array([0.8, 0.4, 0.2])
    names = ["x1", "x2", "x3"]
    ols = sp.regress(ROB, cross)
    mm = sp.robreg(ROB, cross)
    clean = sp.regress("y ~ x1 + x2 + x3", cross)
    # 5% of the outcomes are shifted by about 12 residual standard deviations
    assert abs(ols.params["Intercept"] - clean.params["Intercept"]) > 0.4
    assert abs(mm.params["Intercept"] - clean.params["Intercept"]) < 0.06
    assert np.max(np.abs(mm.params[names].to_numpy() - truth)) < 0.12
    assert mm.std_errors["x1"] < 0.5 * ols.std_errors["x1"]
    assert mm.model_info["n_zero_weight"] >= 35


def test_robreg_seed_does_not_move_the_s_estimate(cross):
    a = sp.robreg(ROB, cross, method="s", random_state=1)
    b = sp.robreg(ROB, cross, method="s", random_state=99)
    assert rel(a.params, b.params) < 1e-7
    assert rel(a.model_info["scale"], b.model_info["scale"]) < 1e-9


# ---------------------------------------------------------------------------
# ITCV
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def itcv_fit(cross):
    return sp.regress("y ~ x1 + x2 + x3 + factor(ind)", cross)


@pytest.mark.parametrize("i, alpha", [(0, 0.01), (1, 0.05), (2, 0.1)])
def test_itcv_matches_the_book_formula(itcv_fit, R, i, alpha):
    out = sp.itcv(itcv_fit, "x2", alpha=alpha)
    r_obs, r_crit, value = R["itcv"]["x2_iid"][i]
    assert out["df"] == R["itcv"]["df"]
    assert rel(out["r_obs"], r_obs) < EXACT
    assert rel(out["r_crit"], r_crit) < EXACT
    assert rel(out["itcv"], value) < EXACT


def test_itcv_uses_the_reported_t_statistic(cross, R):
    robust = sp.regress("y ~ x1 + x2 + x3 + factor(ind)", cross, robust="hc1")
    out = sp.itcv(robust, "x2")
    assert rel(out["itcv"], R["itcv"]["x2_hc1"][2]) < EXACT


def test_itcv_benchmark_impacts_match_partial_correlations(itcv_fit, R):
    out = sp.itcv(itcv_fit, "x2")
    imp = out["impacts"]
    rename = {f"factor(ind){k}": f"C(ind)[T.{k}]" for k in range(2, 6)}
    names = [rename.get(c, c) for c in R["itcv"]["controls"]]
    assert rel(imp.loc[names, "r_yz"], R["itcv"]["r_yz"]) < 1e-8
    assert rel(imp.loc[names, "r_xz"], R["itcv"]["r_xz"]) < 1e-8
    assert out["benchmark"] == imp["impact"].iloc[0] == imp["impact"].max()


@pytest.mark.parametrize("var", ["x2", "x3"])
def test_itcv_and_rir_match_stata_pkonfound(itcv_fit, stata, var):
    """pkonfound counts degrees of freedom as n - ncov - 2 with ncov the
    covariates other than the focal one, one fewer than the regression's
    own when ncov is passed as df_m - 1; the comparison is made at
    pkonfound's df."""
    ref = stata[f"pkonfound_{var}"]
    n = itcv_fit.data_info["nobs"]
    ncov = len(itcv_fit.params) - 2
    like_pkonfound = SimpleNamespace(
        params=itcv_fit.params,
        std_errors=itcv_fit.std_errors,
        data_info={"df_resid": n - ncov - 2, "nobs": n},
        model_info={},
    )
    out = sp.itcv(like_pkonfound, var)
    assert rel(out["r_obs"], ref["r_obs"]) < ITER
    assert rel(out["r_crit"], ref["r_crit"]) < ITER
    assert rel(out["itcv"], ref["itcv"]) < ITER
    assert rel(out["beta_threshold"], ref["beta_threshold"]) < ITER
    assert rel(100 * out["percent_bias"], ref["percent_bias"]) < ITER
    assert out["rir"] == ref["rir"]


# ---------------------------------------------------------------------------
# data handling and prediction scores
# ---------------------------------------------------------------------------


def test_winsor_and_trim_match_r_type2_quantile(cross, R):
    ref = R["winsor"]
    w = sp.winsor(cross, ["tail"], cuts=(1, 99))["tail_w"]
    tr = sp.winsor(cross, ["tail"], cuts=(1, 99), trim=True)["tail_tr"]
    want_w = pd.Series(ref["w"], dtype=float)
    want_tr = pd.Series(ref["tr"], dtype=float)
    assert (w.isna() == want_w.isna()).all() and (tr.isna() == want_tr.isna()).all()
    assert rel(w.dropna(), want_w.dropna()) < EXACT
    assert rel(tr.dropna(), want_tr.dropna()) < EXACT
    assert rel([w.min(), w.max()], ref["cuts"]) < EXACT


def test_winsor_and_trim_match_stata_winsor2(cross, stata):
    ref = stata["winsor2"]
    cols = {
        "tail_w": sp.winsor(cross, ["tail"], cuts=(1, 99))["tail_w"],
        "tail_tr": sp.winsor(cross, ["tail"], cuts=(1, 99), trim=True)["tail_tr"],
        "tail_tr5": sp.winsor(cross, ["tail"], cuts=(5, 95), trim=True, suffix="_tr5")[
            "tail_tr5"
        ],
    }
    for name, col in cols.items():
        assert col.notna().sum() == ref[f"{name}_n"]
        assert rel(col.sum(), ref[f"{name}_sum"]) < EXACT
        assert (
            rel([col.min(), col.max()], [ref[f"{name}_min"], ref[f"{name}_max"]])
            < EXACT
        )


def test_ndcg_and_auc_match_farr(cross, R):
    ref = R["ranking"]
    assert rel(sp.ndcg(cross["event"], cross["score"], 0.01), ref["ndcg_01"]) < EXACT
    assert rel(sp.ndcg(cross["event"], cross["score"], 0.05), ref["ndcg_05"]) < EXACT
    assert rel(sp.ndcg(cross["event"], cross["score"], 0.2), ref["ndcg_20"]) < EXACT
    assert rel(sp.auc(cross["event"], cross["score"]), ref["auc"]) < EXACT


# ---------------------------------------------------------------------------
# formulas and the Poisson fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, formula",
    [("lm_factor", "y ~ x1 + factor(ind)"), ("lm_factor_int", "y ~ factor(ind) * x1")],
)
def test_factor_in_a_formula_is_read_as_in_r(cross, R, key, formula):
    res = sp.regress(formula, cross)
    ref = R[key]
    # R: factor(ind)3 and factor(ind)3:x1; here C(ind)[T.3] and C(ind)[T.3]:x1
    ours = {}
    for name in res.params.index:
        r_name = name.replace("Intercept", "(Intercept)")
        for k in range(2, 6):
            r_name = r_name.replace(f"C(ind)[T.{k}]", f"factor(ind){k}")
        ours[r_name] = name
    assert set(ours) == set(ref["names"])
    names = [ours[n] for n in ref["names"]]
    assert rel(res.params[names], ref["coef"]) < EXACT
    assert rel(res.std_errors[names], ref["se"]) < EXACT
    same = sp.regress(formula.replace("factor(", "C("), cross)
    assert list(same.params.index) == list(res.params.index)
    assert rel(same.params, res.params) == 0.0


def test_poisson_with_a_separated_regressor_converges_to_glm(cross, R):
    """`rare` picks out only zero counts, so its coefficient has no finite
    estimate. The fit used to walk it to minus infinity until the weights
    underflowed and the routine failed in an SVD."""
    ref = R["poisson_sep"]
    with pytest.warns(RuntimeWarning, match="quasi-separation"):
        res = sp.poisson("fines ~ x1 + x2 + rare + factor(ind)", cross)
    assert res.model_info["separated_terms"] == ["rare"]
    assert res.model_info["converged"]
    assert res.params["rare"] < -15
    rename = {"(Intercept)": "_cons"}
    names = []
    for n in ref["names"]:
        n = rename.get(n, n)
        for k in range(2, 6):
            n = n.replace(f"factor(ind){k}", f"C(ind)[T.{k}]")
        names.append(n)
    assert rel(res.params[names], ref["coef"]) < ITER
    assert rel(res.std_errors[names], ref["se"]) < ITER
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        robust = sp.poisson("fines ~ x1 + x2 + rare + factor(ind)", cross, robust="hc1")
    assert rel(robust.std_errors[names], ref["se_hc1"]) < ITER


# ---------------------------------------------------------------------------
# the translated matchit() call gives MatchIt's estimate
# ---------------------------------------------------------------------------

MATCHIT_CALLS = {
    "default": "matchit(d ~ x1 + x2, data = d)",
    "caliper_sd": "matchit(d ~ x1 + x2, data = d, caliper = 0.2)",
    "caliper_raw": "matchit(d ~ x1 + x2, data = d, caliper = 0.03, std.caliper = FALSE)",
    "smallest": 'matchit(d ~ x1 + x2, data = d, m.order = "smallest")',
    "data": 'matchit(d ~ x1 + x2, data = d, m.order = "data")',
    "replace": "matchit(d ~ x1 + x2, data = d, replace = TRUE)",
}


@pytest.mark.parametrize("name", list(MATCHIT_CALLS))
def test_translated_matchit_call_reproduces_matchit(name):
    """Without replacement the pairs depend on the order in which treated
    units choose. MatchIt starts from the largest propensity score and
    sp.match does not by default, so the translation that left the order
    out gave a different estimate (0.533 against 0.542 on this file) and
    said nothing."""
    df = pd.read_csv(FIX / "matchit_translation.csv")
    ref = json.loads((FIX / "matchit_translation_R.json").read_text(encoding="utf-8"))
    out = sp.from_r(MATCHIT_CALLS[name])
    assert out["ok"] and "untranslated_arguments" not in out
    code = out["python_code"].replace("sp.match(", "sp.match(y='y', ", 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = eval(code, {"sp": sp, "df": df})  # noqa: S307
    assert rel(res.estimate, ref[name]["att"]) < EXACT
    if name != "replace":
        assert out["arguments"]["m_order"] in ("largest", "smallest", "data")


def test_matchit_order_without_a_counterpart_is_reported():
    out = sp.from_r('matchit(d ~ x1 + x2, data = d, m.order = "random")')
    assert out["untranslated_arguments"] == ["m.order"]
    assert "m_order" not in out["arguments"]
