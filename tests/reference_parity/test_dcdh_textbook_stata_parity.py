"""The de Chaisemartin-D'Haultfoeuille estimators against the authors' Stata
commands, on a panel shaped like their textbook's applications.

The textbook's four applications need things the earlier fixtures did not
exercise: a count treatment that moves both ways from different starting
levels, periods four years apart, a first-difference regression, other
treatments in the regression, an unbalanced binary panel with binned event
times, time-varying observation weights. ``dcdh_textbook_data.csv`` has all
of them; ``dcdh_textbook_Stata.json`` holds what Stata 18 returns for each
command (``_fixtures/_generate_dcdh_textbook_Stata.do``).

Tolerances. Stata's ``did_multiplegt_dyn`` and ``twowayfeweights`` hold their
working variables in single precision, so agreement is to about 1e-7; the
bounds below are 1e-6 relative for estimates and standard errors (the Track
A default) and 1e-5 absolute for p-values. ``eventstudyinteract`` and
``regress`` work in double and are held to 1e-8.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
RTOL = 1e-6


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(FIX / "dcdh_textbook_data.csv")


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "dcdh_textbook_Stata.json").read_text(encoding="utf-8"))


def _check_dyn(
    result, ref, tag, effects, placebos, joint=True, weighted=False, se_rtol=RTOL
):
    es = result.model_info["event_study"].set_index("relative_time").copy()
    if weighted:
        # with weights the command reports the weighted number of switchers
        detail = result.detail.set_index("horizon")
        es["n_switchers"] = detail["w_switchers"].reindex(es.index)
    for ell in range(1, effects + 1):
        row = es.loc[ell - 1]
        assert row["att"] == pytest.approx(ref[f"{tag}_effect_{ell}"], rel=RTOL)
        assert row["se"] == pytest.approx(ref[f"{tag}_se_effect_{ell}"], rel=se_rtol)
        assert row["n_switchers"] == pytest.approx(
            ref[f"{tag}_n_switchers_{ell}"], rel=RTOL
        )
    for ell in range(1, placebos + 1):
        row = es.loc[-ell]
        assert row["att"] == pytest.approx(ref[f"{tag}_placebo_{ell}"], rel=RTOL)
        assert row["se"] == pytest.approx(ref[f"{tag}_se_placebo_{ell}"], rel=RTOL)
        assert row["n_switchers"] == pytest.approx(
            ref[f"{tag}_n_switchers_placebo_{ell}"], rel=RTOL
        )
    assert result.estimate == pytest.approx(ref[f"{tag}_av_tot"], rel=RTOL)
    assert result.se == pytest.approx(ref[f"{tag}_se_av_tot"], rel=se_rtol)
    if joint and effects > 1:
        assert result.model_info["joint_effects_test"]["pvalue"] == pytest.approx(
            ref[f"{tag}_p_joint_effects"], abs=1e-5
        )
    if joint and placebos > 1:
        assert result.model_info["joint_placebo_test"]["pvalue"] == pytest.approx(
            ref[f"{tag}_p_joint_placebo"], abs=1e-5
        )


def _dyn(df, **kwargs):
    return sp.did_multiplegt_dyn(
        df,
        "y",
        group="g",
        time="year",
        treatment="d",
        se_method="analytic",
        aggregation="switchers",
        **kwargs,
    )


def test_dyn_count_treatment(df, ref):
    """Effects, placebos, analytic SEs, Av_tot_eff and the three Wald tests."""
    r = _dyn(df, dynamic=2, placebo=2, effects_equal=True)
    _check_dyn(r, ref, "dyn", 3, 2)
    assert r.model_info["effects_equal_test"]["pvalue"] == pytest.approx(
        ref["dyn_p_equal"], abs=1e-5
    )
    # every count level is a baseline with its own controls
    assert r.model_info["n_dropped_bidirectional"] > 0


def test_dyn_normalized(df, ref):
    r = _dyn(df, dynamic=2, placebo=2, normalized=True, effects_equal=True)
    _check_dyn(r, ref, "norm", 3, 2)
    assert r.model_info["effects_equal_test"]["pvalue"] == pytest.approx(
        ref["norm_p_equal"], abs=1e-5
    )


def test_dyn_same_switchers(df, ref):
    """same_switchers keeps the switchers with every effect *estimable*."""
    r = _dyn(df, dynamic=2, same_switchers=True)
    _check_dyn(r, ref, "same", 3, 0)
    es = r.model_info["event_study"]
    assert es["n_switchers"].nunique() == 1


def test_dyn_weights_and_cluster(df, ref):
    r = _dyn(df, dynamic=1, placebo=1, weights="wt", cluster="state")
    _check_dyn(r, ref, "wcl", 2, 1, weighted=True)


CONTROLS_CASES = {
    # tag: (outcome, time, treatment, effects, placebos, weighted, options)
    "ctl": ("y", "year", "d", 2, 2, False, {}),
    "ctlw": ("y", "year", "d", 2, 1, True, {"weights": "wt", "cluster": "state"}),
    "ctln": ("y", "year", "d", 2, 0, False, {"normalized": True}),
    "ctlt": ("y", "year", "d", 2, 0, False, {"trends_nonparam": ["cohort"]}),
    "inx": ("y", "year", "d", 2, 0, False, {"switchers": "in"}),
    "ctlb": ("y2", "t", "d2", 2, 1, False, {}),
}


@pytest.mark.parametrize("tag", sorted(CONTROLS_CASES))
def test_dyn_with_controls(df, ref, tag):
    """Estimates and the variance with its slope-estimation term.

    The covariate slopes are estimated, per period-one treatment, on the
    not-yet-switched cells; the command's variance carries the term for
    that (``U^{var,X}`` of the companion paper) and so does the analytic
    variance here. Weighted and clustered, within ``trends_nonparam``
    cells, normalized, one direction only, and on an unbalanced panel.
    """
    y, time, treat, effects, placebos, weighted, options = CONTROLS_CASES[tag]
    data = df.dropna(subset=[y])
    r = sp.did_multiplegt_dyn(
        data,
        y,
        group="g",
        time=time,
        treatment=treat,
        dynamic=effects - 1,
        placebo=placebos,
        controls=["x"],
        se_method="analytic",
        aggregation="switchers",
        **options,
    )
    _check_dyn(r, ref, tag, effects, placebos, weighted=weighted)


def test_controls_variance_term_matters(df, ref):
    """Without the slope-estimation term the SE is visibly off."""
    module = __import__("sys").modules["statspai.did.did_multiplegt_dyn"]
    original = module._residualise_on_controls

    def no_term(*args, **kwargs):
        work, name, _ = original(*args, **kwargs)
        return work, name, {}

    kwargs = dict(group="g", time="year", treatment="d", dynamic=1, controls=["x"])
    module._residualise_on_controls = no_term
    try:
        naive = sp.did_multiplegt_dyn(df, "y", se_method="analytic", **kwargs)
    finally:
        module._residualise_on_controls = original
    es = naive.model_info["event_study"].set_index("relative_time")
    assert es.loc[0, "att"] == pytest.approx(ref["ctl_effect_1"], rel=RTOL)
    assert abs(es.loc[0, "se"] / ref["ctl_se_effect_1"] - 1) > 1e-3


def test_dyn_controls_on_a_panel_with_holes(df):
    """A covariate change across a hole is not a one-period change.

    Stata ``did_multiplegt_dyn y2 g t d2, effects(2) controls(x)`` on the
    rows with a non-missing outcome: Effect_1 = 0.403597479543 (101
    switchers), Effect_2 = 0.834774954927. Before the adjustment was written
    in levels the first differences were taken between consecutive rows,
    whatever the gap between them, and the estimates were 0.40304 and
    0.82877.
    """
    r = sp.did_multiplegt_dyn(
        df.dropna(subset=["y2"]),
        "y2",
        group="g",
        time="t",
        treatment="d2",
        dynamic=1,
        controls=["x"],
        se_method="analytic",
    )
    es = r.model_info["event_study"].set_index("relative_time")
    assert es.loc[0, "att"] == pytest.approx(0.403597479543, rel=RTOL)
    assert es.loc[1, "att"] == pytest.approx(0.834774954927, rel=RTOL)
    assert es.loc[0, "n_switchers"] == 101


def test_dyn_by_path(df, ref):
    r = _dyn(df, dynamic=1, by_path=2, design=1.0)
    paths = r.model_info["by_path"]
    assert [p["path"] for p in paths] == [(0.0, 1.0, 1.0), (0.0, 2.0, 2.0)]
    for tag, res in zip(("path1", "path2"), paths):
        es = res["event_study"]
        for ell in (1, 2):
            assert es["att"].iloc[ell - 1] == pytest.approx(
                ref[f"{tag}_effect_{ell}"], rel=RTOL
            )
            assert es["se"].iloc[ell - 1] == pytest.approx(
                ref[f"{tag}_se_effect_{ell}"], rel=RTOL
            )
            assert es["n_switchers"].iloc[ell - 1] == ref[f"{tag}_n_switchers_{ell}"]
        assert res["estimate"] == pytest.approx(ref[f"{tag}_av_tot"], rel=RTOL)
        assert res["se"] == pytest.approx(ref[f"{tag}_se_av_tot"], rel=RTOL)
        assert res["joint_effects_test"]["pvalue"] == pytest.approx(
            ref[f"{tag}_p_joint_effects"], abs=1e-5
        )
    design = r.model_info["design"]
    assert design["n_groups"].sum() == design.attrs["n_groups"]
    assert design["n_groups"].iloc[0] == paths[0]["n_switchers"]


def test_dyn_period_labels_do_not_matter(df):
    """Periods are ranked: year (steps of four) and t (steps of one) agree."""
    a = _dyn(df, dynamic=2, placebo=1)
    b = sp.did_multiplegt_dyn(
        df,
        "y",
        group="g",
        time="t",
        treatment="d",
        dynamic=2,
        placebo=1,
        se_method="analytic",
        aggregation="switchers",
    )
    ea, eb = a.model_info["event_study"], b.model_info["event_study"]
    np.testing.assert_allclose(ea["att"], eb["att"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ea["se"], eb["se"], rtol=0, atol=1e-12)


def test_didm_count_treatment(df, ref):
    """DID_M per unit of treatment; the command's placebos are forward differences."""
    r = sp.did_multiplegt(
        df, "y", "g", "year", "d", placebo=2, n_boot=2, seed=0, placebo_sign="r"
    )
    assert r.estimate == pytest.approx(ref["didm_effect"], rel=RTOL)
    assert r.model_info["n_switchers"] == ref["didm_n_switchers"]
    placebos = {p["lag"]: p["estimate"] for p in r.model_info["placebo"]}
    assert placebos[-1] == pytest.approx(ref["didm_placebo_1"], rel=RTOL)
    assert placebos[-2] == pytest.approx(ref["didm_placebo_2"], rel=RTOL)


def _check_weights(result, ref, tag):
    mi = result.model_info
    assert result.estimate == pytest.approx(ref[f"{tag}_beta"], rel=RTOL)
    assert mi["n_positive"] == ref[f"{tag}_n_plus"]
    assert mi["n_negative"] == ref[f"{tag}_n_minus"]
    # the command returns these sums in single precision
    assert mi["sum_positive"] == pytest.approx(ref[f"{tag}_sum_plus"], abs=1e-6)
    assert mi["sum_negative"] == pytest.approx(ref[f"{tag}_sum_minus"], abs=1e-6)


def test_twowayfeweights_fe(df, ref):
    w = sp.twowayfeweights(
        df,
        "y",
        "g",
        "year",
        "d",
        covariates=["x"],
        test_random_weights=["year"],
        weights="wt",
    )
    _check_weights(w, ref, "fe")
    assert w.model_info["sigma_fe"] == pytest.approx(ref["fe_sigma"], rel=RTOL)
    assert w.model_info["sigma_fe_2"] == pytest.approx(ref["fe_sigma2"], rel=RTOL)
    rw = w.model_info["random_weights"].loc["year"]
    assert rw["coef"] == pytest.approx(ref["fe_rw_coef"], rel=RTOL)
    assert rw["se"] == pytest.approx(ref["fe_rw_se"], rel=RTOL)
    assert rw["correlation"] == pytest.approx(ref["fe_rw_corr"], rel=RTOL)
    # the weights are a decomposition of the coefficient: they add up to one
    assert w.detail["weight"].sum() == pytest.approx(1.0, abs=1e-10)


def test_twowayfeweights_fd(df, ref):
    w = sp.twowayfeweights(
        df,
        "dy",
        "g",
        "year",
        "dd",
        type="fdTR",
        treat_level="d",
        controls=["x"],
        test_random_weights=["t"],
    )
    _check_weights(w, ref, "fd")
    assert w.model_info["sigma_fe"] == pytest.approx(ref["fd_sigma"], rel=RTOL)
    assert w.model_info["sigma_fe_2"] == pytest.approx(ref["fd_sigma2"], rel=RTOL)
    rw = w.model_info["random_weights"].loc["t"]
    assert rw["coef"] == pytest.approx(ref["fd_rw_coef"], rel=RTOL)
    assert rw["se"] == pytest.approx(ref["fd_rw_se"], rel=RTOL)
    assert rw["correlation"] == pytest.approx(ref["fd_rw_corr"], rel=RTOL)


def test_twowayfeweights_other_treatments(df, ref):
    w = sp.twowayfeweights(df, "y", "g", "year", "d", other_treatments=["d2"])
    _check_weights(w, ref, "ot")
    other = w.model_info["other_treatments"]["d2"]
    assert other["n_positive"] == ref["ot_other_n_plus"]
    assert other["n_negative"] == ref["ot_other_n_minus"]
    assert other["sum_positive"] == pytest.approx(ref["ot_other_sum_plus"], abs=1e-6)
    assert other["sum_negative"] == pytest.approx(ref["ot_other_sum_minus"], abs=1e-6)
    # the other treatment's weights add up to zero: pure contamination
    assert w.detail["weight_d2"].sum() == pytest.approx(0.0, abs=1e-10)
    assert np.isnan(w.model_info["sigma_fe"])


def test_twowayfeweights_is_the_twfe_decomposition_on_a_staggered_design():
    d = sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=3)
    d["d"] = ((d["first_treat"] > 0) & (d["time"] >= d["first_treat"])) * 1.0
    w = sp.twowayfeweights(d, y="y", group="unit", time="time", treat="d")
    t = sp.twfe_decomposition(d, "y", "unit", "time", "first_treat")
    assert w.estimate == pytest.approx(t.estimate, rel=1e-12)
    assert w.se == pytest.approx(t.se, rel=1e-10)
    assert w.model_info["n_negative"] == t.model_info["n_negative_weights_dcdh"]
    fe = sp.feols("y ~ d | unit + time", data=d, cluster="unit")
    assert w.estimate == pytest.approx(float(fe.params["d"]), rel=1e-10)
    assert w.se == pytest.approx(float(fe.std_errors["d"]), rel=1e-8)


def test_twowayfeweights_refusals(df):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="treat_level"):
        sp.twowayfeweights(df, "dy", "g", "year", "dd", type="fdTR")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="feTR"):
        sp.twowayfeweights(df, "y", "g", "year", "d", type="feS")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="only available"):
        sp.twowayfeweights(
            df,
            "dy",
            "g",
            "year",
            "dd",
            type="fdTR",
            treat_level="d",
            other_treatments=["d2"],
        )
    zero = df.assign(d=0.0)
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.twowayfeweights(zero, "y", "g", "year", "d")


def test_regress_hc2_dfadjust(df, ref):
    """Bell-McCaffrey degrees of freedom: Stata 18 ``vce(hc2, dfadjust)``."""
    last = df[df["t"] == 8]
    r = sp.regress("y ~ x + d", data=last, robust="hc2", dfadjust=True)
    ci = r.conf_int()
    assert r.params["x"] == pytest.approx(ref["hc2_b_x"], rel=1e-8)
    assert r.std_errors["x"] == pytest.approx(ref["hc2_se_x"], rel=1e-8)
    assert r.pvalues["x"] == pytest.approx(ref["hc2_p_x"], rel=1e-7)
    assert r.pvalues["d"] == pytest.approx(ref["hc2_p_d"], rel=1e-7)
    assert ci.loc["x"].iloc[0] == pytest.approx(ref["hc2_lo_x"], rel=1e-7)
    assert ci.loc["x"].iloc[1] == pytest.approx(ref["hc2_hi_x"], rel=1e-7)
    joint = sp.test(r, "x d")
    assert joint["statistic"] == pytest.approx(ref["hc2_F"], rel=1e-8)
    assert joint["pvalue"] == pytest.approx(ref["hc2_F_p"], rel=1e-6)
    # the standard errors are plain HC2; only the reference distribution moves
    plain = sp.regress("y ~ x + d", data=last, robust="hc2")
    assert r.std_errors["x"] == pytest.approx(plain.std_errors["x"], rel=1e-12)
    assert r.pvalues["x"] > plain.pvalues["x"]


def test_regress_cr2_dfadjust(df, ref):
    r = sp.regress("y ~ x + d", data=df, vce="cr2", cluster="state", dfadjust=True)
    ci = r.conf_int()
    assert r.std_errors["d"] == pytest.approx(ref["cr2_se_d"], rel=1e-8)
    assert r.pvalues["d"] == pytest.approx(ref["cr2_p_d"], rel=1e-6)
    assert ci.loc["d"].iloc[0] == pytest.approx(ref["cr2_lo_d"], rel=1e-7)
    assert ci.loc["d"].iloc[1] == pytest.approx(ref["cr2_hi_d"], rel=1e-7)
    joint = sp.test(r, "x d")
    assert joint["statistic"] == pytest.approx(ref["cr2_F"], rel=1e-8)
    assert joint["pvalue"] == pytest.approx(ref["cr2_F_p"], rel=1e-6)
    # without dfadjust the same variance now also supports a joint test
    plain = sp.regress("y ~ x + d", data=df, vce="cr2", cluster="state")
    assert sp.test(plain, "x d")["statistic"] == pytest.approx(ref["cr2_F"], rel=1e-8)


def test_regress_dfadjust_refusals(df):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="Bell-McCaffrey"):
        sp.regress("y ~ x", data=df, robust="hc1", dfadjust=True)
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.regress("y ~ x", data=df, robust="hc2", dfadjust=True, weights="w")


def _sa(df, **kwargs):
    d = df.dropna(subset=["y2"])
    return sp.sun_abraham(
        d,
        "y2",
        g="cohort",
        t="t",
        i="g",
        cluster="g",
        event_window=(-3, 3),
        window_rule="bin",
        **kwargs,
    )


@pytest.mark.parametrize("weights, tag", [(None, "sa"), ("wt", "saw")])
def test_sun_abraham_binned_unbalanced(df, ref, weights, tag):
    """Interaction weights are observation shares; ends pooled into bins."""
    es = _sa(df, weights=weights).model_info["event_study"].set_index("relative_time")
    for name, e in (("L0", 0), ("L1", 1), ("L2", 2), ("L3", 3), ("F2", -2), ("F3", -3)):
        # abs: one of these coefficients is 5e-5
        assert es.loc[e, "att"] == pytest.approx(
            ref[f"{tag}_{name}"], rel=1e-8, abs=1e-9
        )
        assert es.loc[e, "se"] == pytest.approx(ref[f"{tag}_se_{name}"], rel=1e-8)


def test_sun_abraham_window_only_selects_what_is_reported(df):
    d = df.dropna(subset=["y2"])
    full = sp.sun_abraham(d, "y2", g="cohort", t="t", i="g")
    cut = sp.sun_abraham(d, "y2", g="cohort", t="t", i="g", event_window=(-2, 2))
    a = full.model_info["event_study"].set_index("relative_time")
    b = cut.model_info["event_study"].set_index("relative_time")
    assert list(b.index) == [-2, 0, 1, 2]
    np.testing.assert_allclose(b["att"], a.loc[b.index, "att"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(b["se"], a.loc[b.index, "se"], rtol=0, atol=1e-12)
    # the earlier behaviour: left-out relative times join the reference
    old = sp.sun_abraham(
        d, "y2", g="cohort", t="t", i="g", event_window=(-2, 2), window_rule="reference"
    )
    c = old.model_info["event_study"].set_index("relative_time")
    assert abs(c.loc[0, "att"] - a.loc[0, "att"]) > 0.05


def test_sun_abraham_window_recovers_a_known_effect():
    """Effects grow with exposure; a window cut at 1 must not move effect 0."""
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(300):
        cohort = int(rng.choice([0, 3, 5]))
        a = rng.normal()
        for t in range(1, 11):
            e = t - cohort
            tau = 1.0 + e if cohort and e >= 0 else 0.0
            rows.append((unit, t, cohort, a + 0.2 * t + tau + rng.normal(scale=0.1)))
    d = pd.DataFrame(rows, columns=["i", "t", "g", "y"])
    kept = sp.sun_abraham(d, "y", g="g", t="t", i="i", event_window=(-2, 1))
    es = kept.model_info["event_study"].set_index("relative_time")
    assert es.loc[0, "att"] == pytest.approx(1.0, abs=0.05)
    assert es.loc[1, "att"] == pytest.approx(2.0, abs=0.05)
    old = sp.sun_abraham(
        d, "y", g="g", t="t", i="i", event_window=(-2, 1), window_rule="reference"
    )
    biased = old.model_info["event_study"].set_index("relative_time")
    # the reference then holds exposures 2 to 7, whose effects are 3 to 8
    assert biased.loc[0, "att"] < 0.0


def test_sun_abraham_window_rule_checks(df):
    d = df.dropna(subset=["y2"])
    with pytest.raises(ValueError, match="window_rule"):
        sp.sun_abraham(d, "y2", g="cohort", t="t", i="g", window_rule="drop")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="end bins"):
        sp.sun_abraham(
            d, "y2", g="cohort", t="t", i="g", event_window=(-1, 2), window_rule="bin"
        )
