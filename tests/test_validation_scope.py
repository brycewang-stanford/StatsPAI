"""``sp.validation_scope``: configuration- and output-level evidence.

The registry tier is per function; the evidence is per configuration and
per output. These tests keep the map honest in both directions:

* every row is well formed -- every dimension enumerated, no wildcard, every
  value inside its domain, every artifact present and calling the entry
  point the row credits;
* the configurations the reviewers' probes used are graded correctly -- a
  point-estimate row never vouches for a standard error it did not
  compare, an adjacent legal option that no artifact ran is not covered,
  and a value outside a dimension's domain is refused;
* the status vocabulary keeps disclosure (T4) apart from stochastic
  evidence (T3 / S / B).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.validation_scope import KINDS, OUTPUTS, SCOPES

ROOT = Path(__file__).resolve().parents[1]
CARD_IV = "lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4)"


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# --------------------------------------------------------------------------- #
#  The map itself
# --------------------------------------------------------------------------- #


def test_every_artifact_exists():
    missing = sorted(
        {row.artifact for s in SCOPES.values() for row in s.rows}
        - {
            p
            for p in {row.artifact for s in SCOPES.values() for row in s.rows}
            if (ROOT / p).exists()
        }
    )
    assert not missing


@pytest.mark.parametrize("name", sorted(SCOPES))
def test_rows_enumerate_every_dimension_inside_its_domain(name):
    scope = SCOPES[name]
    for dim, domain in scope.domains.items():
        assert domain and len(set(domain)) == len(domain), (name, dim)
    for row in scope.rows:
        assert row.kind in KINDS, (name, row)
        assert set(row.config) == set(scope.domains), (name, row.artifact)
        for dim, values in row.config.items():
            assert values, (name, row.artifact, dim)
            assert values <= set(scope.domains[dim]), (name, row.artifact, dim, values)
            assert "*" not in values
        assert row.outputs and set(row.outputs) <= set(OUTPUTS), (name, row.artifact)
    for dim, (outs, reason) in scope.invariant.items():
        assert dim in scope.domains and set(outs) <= set(OUTPUTS) and reason, (
            name,
            dim,
        )


@pytest.mark.parametrize("name", sorted(SCOPES))
def test_each_artifact_calls_the_entry_point_it_is_credited_to(name):
    """A row may only claim an entry point its artifact actually calls."""
    for row in SCOPES[name].rows:
        if row.artifact.endswith(".json"):
            src = (ROOT / "tests/coverage_monte_carlo/run_b1000.py").read_text(
                encoding="utf-8"
            )
        else:
            src = (ROOT / row.artifact).read_text(encoding="utf-8")
        call = re.match(r"(sp(?:\.\w+)+)\(", row.entry_point).group(1)
        assert call + "(" in src, (name, row.artifact, call)


def test_coverage_rows_exist_in_the_coverage_artifact():
    rows = json.loads(
        (
            ROOT / "tests/coverage_monte_carlo/results_b1000/coverage_b1000.json"
        ).read_text(encoding="utf-8")
    )
    names = " ".join(r["name"] for r in rows).lower()
    for token in (
        "regress",
        "ivreg",
        "callaway",
        "sun_abraham",
        "fast.feols",
        "rdrobust",
        "sdid",
        "plr",
        "irm",
        "causal_forest",
    ):
        assert token in names, token


# --------------------------------------------------------------------------- #
#  Output-level grading (the reviewers' probes)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def card():
    return sp.datasets.card_1995()


@pytest.mark.parametrize("robust", ["hc2", "hc3", "hc0"])
def test_untested_iv_covariance_does_not_borrow_the_card_row(card, robust):
    scope = sp.validation_scope(_quiet(sp.iv, CARD_IV, data=card, robust=robust))
    assert scope["configuration"]["vce"] == robust
    assert scope["status"] == "estimate_only"
    assert scope["outputs"]["estimate"]["status"] == "reference"
    assert scope["outputs"]["se"]["status"] == "not_covered"
    assert not scope["outputs"]["se"]["evidence"]


@pytest.mark.parametrize("robust,expected", [(None, "covered"), ("hc1", "covered")])
def test_tested_iv_covariance_is_covered(card, robust, expected):
    kw = {} if robust is None else {"robust": robust}
    scope = sp.validation_scope(_quiet(sp.iv, CARD_IV, data=card, **kw))
    assert scope["status"] == expected
    assert any(
        e["artifact"].endswith("test_iv_card_aer_parity.py")
        for e in scope["outputs"]["se"]["evidence"]
    )


def test_iv_small_sample_cluster_vce_is_not_the_cr1_row(card):
    df = card.assign(cl=np.arange(len(card)) % 40)
    scope = sp.validation_scope(
        _quiet(sp.iv, CARD_IV, data=df, vce="cr2", cluster="cl")
    )
    assert scope["configuration"]["vce"] == "cr2"
    assert scope["outputs"]["se"]["status"] == "not_covered"


def test_values_outside_a_domain_are_refused():
    with pytest.raises(MethodIncompatibility):
        sp.validation_scope(function="iv", vce="invented_vce")
    with pytest.raises(MethodIncompatibility):
        sp.validation_scope(function="causal_forest", trees=">=2000")
    with pytest.raises(MethodIncompatibility):
        sp.validation_scope(function="dml", learners="flexible")


def test_unknown_function_and_dimension_raise():
    with pytest.raises(MethodIncompatibility):
        sp.validation_scope(function="ols_but_fancier")
    with pytest.raises(MethodIncompatibility):
        sp.validation_scope(function="dml", learner="lasso")


def test_regress_hc1_is_covered_and_names_its_coverage_row(card):
    scope = sp.validation_scope(sp.regress("lwage ~ educ", data=card, robust="hc1"))
    assert scope["configuration"] == {"vce": "hc1", "weights": "none"}
    assert scope["status"] == "covered"
    assert scope["outputs"]["coverage"]["status"] == "coverage_simulation"


def test_regress_cr2_has_a_stata_reference_and_hc0_does_not(card):
    df = card.assign(cl=np.arange(len(card)) % 40)
    scope = sp.validation_scope(
        sp.regress("lwage ~ educ", data=df, vce="cr2", cluster="cl")
    )
    assert scope["configuration"]["vce"] == "cr2"
    assert scope["status"] == "covered"
    # Stata has no HC0 and no R row runs it: the SE has no reference.
    scope = sp.validation_scope(sp.regress("lwage ~ educ", data=df, vce="hc0"))
    assert scope["configuration"]["vce"] == "hc0"
    assert scope["status"] == "estimate_only"


def test_a_weighted_regress_is_its_own_cell(card):
    df = card.assign(w=1.0 + (np.arange(len(card)) % 5))
    scope = sp.validation_scope(sp.regress("lwage ~ educ", data=df, weights="w"))
    assert scope["configuration"]["weights"] == "set"
    assert scope["status"] == "covered"
    hac = sp.validation_scope(
        sp.regress("lwage ~ educ", data=df, weights="w", vce="hac", hac_lags=2)
    )
    assert hac["status"] == "estimate_only"


def test_default_learner_dml_is_stochastic_only_and_the_linear_row_is_a_near_miss(card):
    fit = _quiet(sp.dml, card, y="lwage", d="educ", X=["exper", "black"], model="plr")
    scope = sp.validation_scope(fit)
    assert scope["configuration"]["learners"] == "default"
    assert scope["status"] == "stochastic_only"
    assert scope["outputs"]["estimate"]["status"] == "not_covered"
    assert scope["outputs"]["coverage"]["status"] == "coverage_simulation"
    assert any(
        n["artifact"].endswith("08_dml.py") and n["differs_in"] == "learners"
        for n in scope["near_misses"]
    )


def test_user_supplied_boosting_learner_is_not_the_default_learner_row(card):
    from sklearn.ensemble import GradientBoostingRegressor

    fit = _quiet(
        sp.dml,
        card,
        y="lwage",
        d="educ",
        X=["exper", "black"],
        model="plr",
        model_y=GradientBoostingRegressor(),
        model_d=GradientBoostingRegressor(),
    )
    scope = sp.validation_scope(fit)
    assert scope["configuration"]["learners"] == "other"
    assert scope["status"] == "not_covered"


def test_linear_learner_dml_on_other_folds_is_not_covered(card):
    from sklearn.linear_model import LinearRegression

    fit = _quiet(
        sp.dml,
        card,
        y="lwage",
        d="educ",
        X=["exper", "black"],
        model="plr",
        model_y=LinearRegression(),
        model_d=LinearRegression(),
        n_folds=3,
    )
    scope = sp.validation_scope(fit)
    assert scope["configuration"]["n_folds"] == "3"
    assert scope["status"] == "not_covered"


def test_callaway_santanna_default_dr_is_covered_by_the_grid_not_module_04():
    m = sp.datasets.mpdta()
    fit = _quiet(
        sp.callaway_santanna, m, y="lemp", g="first_treat", t="year", i="countyreal"
    )
    scope = sp.validation_scope(fit)
    assert scope["configuration"]["estimator"] == "dr"
    assert scope["status"] == "covered"
    assert {e["artifact"] for e in scope["outputs"]["se"]["evidence"]} == {
        "tests/reference_parity/test_cs_weighted_parity.py"
    }


def test_callaway_santanna_bootstrap_se_is_a_stochastic_screen():
    m = sp.datasets.mpdta()
    fit = _quiet(
        sp.callaway_santanna,
        m,
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        bstrap=True,
        biters=99,
        random_state=0,
    )
    scope = sp.validation_scope(fit)
    assert scope["outputs"]["estimate"]["status"] == "reference"
    assert scope["outputs"]["se"]["status"] == "stochastic_screen"
    assert scope["status"] == "estimate_only"


def test_callaway_santanna_anticipation_is_not_covered():
    m = sp.datasets.mpdta()
    fit = _quiet(
        sp.callaway_santanna,
        m,
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        anticipation=1,
    )
    assert sp.validation_scope(fit)["status"] == "not_covered"


def test_forest_rows_are_specific_to_the_tree_counts_that_ran():
    base = dict(
        function="causal_forest",
        treatment="binary",
        tuning="grf_defaults",
        design="iid",
    )
    s2000 = sp.validation_scope(trees="2000", **base)
    assert s2000["status"] == "stochastic_only"
    assert s2000["outputs"]["estimate"]["status"] == "seed_equivalence"
    s4000 = sp.validation_scope(trees="4000", **base)
    assert s4000["status"] == "not_covered"
    assert any(n["differs_in"] == "trees" for n in s4000["near_misses"])
    custom = sp.validation_scope(
        function="causal_forest",
        treatment="binary",
        trees="2000",
        tuning="custom",
        design="iid",
    )
    assert custom["status"] == "not_covered"


def test_default_forest_fit_reads_as_grf_defaults():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 2))
    d = rng.binomial(1, 0.5, size=n)
    df = pd.DataFrame(
        {"y": x[:, 0] + d + rng.normal(size=n), "d": d, "x1": x[:, 0], "x2": x[:, 1]}
    )
    cf = _quiet(
        sp.causal_forest, "y ~ d | x1 + x2", data=df, n_estimators=500, random_state=0
    )
    cfg = sp.validation_scope(cf)["configuration"]
    assert cfg == {
        "treatment": "binary",
        "trees": "500",
        "tuning": "grf_defaults",
        "design": "iid",
    }
    cf2 = _quiet(
        sp.causal_forest,
        "y ~ d | x1 + x2",
        data=df,
        n_estimators=500,
        min_samples_leaf=10,
        random_state=0,
    )
    assert sp.validation_scope(cf2)["configuration"]["tuning"] == "custom"


def test_scm_disclosure_is_not_stochastic_evidence():
    scope = sp.validation_scope(
        function="synth", v_method="nested", weights="nonunique", code_path="native"
    )
    assert scope["status"] == "disclosure_only"
    assert scope["outputs"]["estimate"]["status"] == "disclosure"
    ok = sp.validation_scope(
        function="synth", v_method="equal", weights="unique", code_path="native"
    )
    assert ok["status"] == "covered"


def test_sdid_placebo_se_is_not_part_of_the_parity_row():
    scope = sp.validation_scope(
        function="sdid", method="sdid", se_method="placebo", code_path="native"
    )
    assert scope["status"] == "estimate_only"
    assert scope["outputs"]["coverage"]["status"] == "coverage_simulation"
    jk = sp.validation_scope(
        function="sdid", method="sdid", se_method="jackknife", code_path="native"
    )
    assert jk["outputs"]["estimate"]["status"] == "reference"
    assert jk["outputs"]["coverage"]["status"] == "not_covered"


def test_psm_default_se_has_no_reference_row():
    df = sp.datasets.nsw_dw()
    X = ["age", "education", "black", "hispanic", "married", "re74", "re75"]
    fit = _quiet(sp.psm, df, y="re78", d="treat", X=X, method="nn")
    scope = sp.validation_scope(fit)
    assert scope["status"] == "estimate_only"
    fit16 = _quiet(
        sp.psm,
        df,
        y="re78",
        d="treat",
        X=X,
        method="nn",
        se_method="abadie_imbens_2016",
    )
    assert sp.validation_scope(fit16)["status"] == "covered"


def test_fast_feols_legacy_ssc_has_no_reference_row():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"i": np.repeat(np.arange(40), 5), "t": np.tile(np.arange(5), 40)}
    )
    df["x"] = rng.normal(size=len(df))
    df["y"] = df["x"] + rng.normal(size=len(df))
    default = sp.fast.feols("y ~ x | i + t", data=df, vcov="cr1", cluster="i")
    assert sp.validation_scope(default)["status"] == "covered"
    legacy = sp.fast.feols(
        "y ~ x | i + t", data=df, vcov="cr1", cluster="i", ssc="statspai"
    )
    assert sp.validation_scope(legacy)["status"] == "estimate_only"


def test_causal_question_result_is_unwrapped():
    rng = np.random.default_rng(0)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-0.4 * x1)))
    df = pd.DataFrame({"y": d + x1 + rng.normal(size=n), "d": d, "x1": x1, "x2": x2})
    r = _quiet(
        lambda: sp.causal_question(
            treatment="d", outcome="y", design="dml", covariates=["x1", "x2"], data=df
        ).estimate()
    )
    scope = sp.validation_scope(r)
    assert scope["function"] == "dml"
    assert scope["configuration"]["model"] == "irm"


def test_unrecorded_dimensions_are_listed_as_unchecked():
    scope = sp.validation_scope(function="iv", estimator="2sls", vce="hc1")
    assert set(scope["unchecked"]) == {"identification", "absorb"}
    assert scope["status"] == "not_covered"


def test_renders_readably(card):
    text = str(sp.validation_scope(sp.regress("lwage ~ educ", data=card, robust="hac")))
    assert text.startswith("Validation scope: sp.regress  [covered]")
    assert "51_newey.py" in text


# --------------------------------------------------------------------------- #
#  Joint outputs: covariance matrices and joint tests (review 2026-09 §4.1)
# --------------------------------------------------------------------------- #


def test_joint_outputs_are_graded_separately_from_se():
    cs_dr = sp.validation_scope(
        function="callaway_santanna",
        estimator="dr",
        control_group="nevertreated",
        weights="none",
        covariates="none",
        inference="analytic",
        base_period="universal",
        anticipation="0",
        clustering="none",
    )
    assert cs_dr["outputs"]["vcov"]["status"] == "reference"
    cs_reg = sp.validation_scope(
        function="callaway_santanna",
        estimator="reg",
        control_group="nevertreated",
        weights="none",
        covariates="none",
        inference="analytic",
        base_period="universal",
        anticipation="0",
        clustering="none",
    )
    # the SE rows cover reg, but no artifact compared its joint covariance
    assert cs_reg["outputs"]["se"]["status"] == "reference"
    assert cs_reg["outputs"]["vcov"]["status"] == "not_covered"

    cr1 = sp.validation_scope(function="regress", vce="cr1", weights="none")
    assert cr1["outputs"]["joint_test"]["status"] == "reference"
    # hc3 gained a joint-test row in 2026-10 (test_joint_wald_stata_parity);
    # Newey-West has an SE row and still no joint-test row.
    hc3 = sp.validation_scope(function="regress", vce="hc3", weights="none")
    assert hc3["outputs"]["joint_test"]["status"] == "reference"
    hac = sp.validation_scope(function="regress", vce="hac", weights="none")
    assert hac["outputs"]["joint_test"]["status"] == "not_covered"
    assert hac["outputs"]["se"]["status"] == "reference"


def test_panel_scope_separates_the_default_and_the_reference_conventions():
    import pandas as pd

    df = pd.read_csv(ROOT / "tests/reference_parity/_fixtures/panel_ssc_data.csv")
    kw = dict(entity="id", time="t", cluster="st")
    default = sp.validation_scope(sp.panel(df, "y ~ x1 + x2", **kw))
    stata = sp.validation_scope(sp.panel(df, "y ~ x1 + x2", ssc="stata", **kw))
    assert default["configuration"]["ssc"] == "linearmodels"
    assert stata["configuration"]["ssc"] == "stata"
    assert stata["status"] == "covered"
    assert stata["outputs"]["se"]["status"] == "reference"
    # linearmodels' own clustered scaling has no reference row.
    assert default["status"] == "estimate_only"
    assert default["outputs"]["se"]["status"] == "not_covered"


def test_sdid_treat_path_has_its_own_rows():
    # A fitted staggered result reports interface / design / covariates itself.
    staggered = sp.validation_scope(
        function="sdid",
        method="sdid",
        se_method="jackknife",
        code_path="native",
        interface="treat",
        design="staggered",
        covariates="projected",
    )
    assert staggered["status"] == "covered"
    # A by-name query that predates those dimensions means the treated_unit=
    # block design without covariates, and keeps its old answer.
    legacy = sp.validation_scope(
        function="sdid", method="sdid", se_method="placebo", code_path="native"
    )
    assert legacy["configuration"]["interface"] == "treated_unit"
    assert legacy["status"] == "estimate_only"
    # Bootstrap on the treat= path is screened, not a reference row.
    boot = sp.validation_scope(
        function="sdid",
        method="sdid",
        se_method="bootstrap",
        code_path="native",
        interface="treat",
        design="staggered",
        covariates="none",
    )
    assert boot["outputs"]["se"]["status"] == "not_covered"


def test_sun_abraham_joint_covariance_has_a_reference_only_with_fixed_shares():
    """The vcov row ran ``share_variance=False``; the default must not borrow it.

    fixest holds cohort shares fixed, so its ``A V A'`` pins the full
    event-study matrix of that setting only. Under the default (the
    Sun-Abraham Prop. 3 share term) the diagonal is pinned against
    ``eventstudyinteract`` by module 05, the off-diagonal blocks by nothing.
    """
    mp = pd.read_csv(ROOT / "tests/orig_parity/data/02_mpdta_original.csv")
    keys = dict(y="lemp", g="first_treat", t="year", i="countyreal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = sp.validation_scope(sp.sun_abraham(mp, **keys))
        fixed = sp.validation_scope(sp.sun_abraham(mp, **keys, share_variance=False))
    assert default["configuration"]["share_variance"] == "estimated"
    assert default["outputs"]["vcov"]["status"] == "not_covered"
    assert default["outputs"]["se"]["status"] == "reference"
    assert fixed["outputs"]["vcov"]["status"] == "reference"
    assert fixed["outputs"]["vcov"]["evidence"][0]["artifact"].endswith(
        "test_event_study_vcov_R_parity.py"
    )
    # The coverage simulation ran the default only.
    assert default["outputs"]["coverage"]["status"] == "coverage_simulation"
    assert fixed["outputs"]["coverage"]["status"] == "not_covered"
    # A by-name query that omits the new dimension means the default.
    by_name = sp.validation_scope(
        function="sun_abraham", control_group="nevertreated", aggregation="event_time"
    )
    assert by_name["configuration"]["share_variance"] == "estimated"


@pytest.fixture(scope="module")
def mpdta():
    return pd.read_csv(ROOT / "tests/orig_parity/data/02_mpdta_original.csv")


_BJS = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")


def _scope_of(fit):
    scope = sp.validation_scope(fit)
    return scope, {k: v["status"] for k, v in scope["outputs"].items()}


def test_did_imputation_scope_follows_the_y0_specification(mpdta):
    """Each Y(0)-model option has its own evidence; none borrows the default's."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = sp.did_imputation(mpdta, **_BJS)
        time_only = sp.did_imputation(mpdta, **_BJS, fe=["year"])
        custom = sp.did_imputation(
            mpdta.assign(st=mpdta["countyreal"] // 1000),
            **_BJS,
            fe=["countyreal", "st#year"],
        )
        horizons = sp.did_imputation(mpdta, **_BJS, horizon=[0, 1, 2, 3])
    scope, out = _scope_of(default)
    assert scope["function"] == "did_imputation" and scope["unchecked"] == []
    assert scope["configuration"] == {
        "fe": "unit_time",
        "covariates": "none",
        "vce": "analytic",
        "weights": "none",
        "horizon": "overall",
    }
    assert (out["estimate"], out["se"], out["vcov"]) == (
        "reference",
        "reference",
        "not_covered",
    )
    assert _scope_of(time_only)[0]["status"] == "covered"
    assert _scope_of(custom)[0]["status"] == "not_covered"
    # The joint covariance is pinned only where horizons were requested.
    assert _scope_of(horizons)[1]["vcov"] == "reference"


def test_did_imputation_unit_slopes_are_a_disclosure_not_a_match(mpdta):
    """Stata's unitcontrols() ATT is not stable to 1e-6; say so, not 'aligned'."""
    untreated = (mpdta["first_treat"] == 0) | (mpdta["year"] < mpdta["first_treat"])
    keep = untreated.groupby(mpdta["countyreal"]).transform("sum") >= 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.did_imputation(mpdta[keep].copy(), **_BJS, unit_covariates=["year"])
    scope, out = _scope_of(fit)
    assert out["estimate"] == "disclosure"
    assert out["se"] == "reference"
    assert scope["status"] != "covered"


def test_gardner_scope_separates_the_corrected_and_the_legacy_variance(mpdta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = sp.gardner_did(mpdta, **_BJS)
        stage2 = sp.gardner_did(mpdta, **_BJS, vce="stage2")
        dynamic = sp.gardner_did(mpdta, **_BJS, event_study=True)
        controls = sp.gardner_did(mpdta, **_BJS, controls=["lpop"])
    scope, out = _scope_of(default)
    assert scope["function"] == "gardner_did" and scope["status"] == "covered"
    assert out["vcov"] == "not_covered"
    # Same point estimate, but module 73 pins the corrected variance only.
    scope, out = _scope_of(stage2)
    assert scope["status"] == "estimate_only" and out["se"] == "not_covered"
    assert _scope_of(dynamic)[1]["vcov"] == "reference"
    assert _scope_of(controls)[0]["status"] == "not_covered"


def test_twfe_event_study_is_covered_only_where_module_85_ran(mpdta):
    """A staggered panel, or another window, is not what the reference pinned."""
    twfe = pd.read_csv(ROOT / "tests/r_parity/data/85_twfe_event_study.csv")
    twfe["g_nan"] = twfe["g"].where(twfe["g"] > 0)
    keys = dict(y="y", treat_time="g_nan", time="time", unit="unit")
    staggered = mpdta.assign(g=mpdta["first_treat"].where(mpdta["first_treat"] > 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ran = sp.event_study(twfe, **keys, window=(-4, 4), cluster="unit")
        narrow = sp.event_study(twfe, **keys, window=(-3, 3))
        stag = sp.event_study(
            staggered, y="lemp", treat_time="g", time="year", unit="countyreal"
        )
    scope, out = _scope_of(ran)
    assert scope["function"] == "event_study" and scope["unchecked"] == []
    assert (out["estimate"], out["se"], out["vcov"]) == ("reference",) * 3
    assert _scope_of(narrow)[0]["configuration"]["window"] == "other"
    assert _scope_of(narrow)[0]["status"] == "not_covered"
    scope, _ = _scope_of(stag)
    assert scope["configuration"]["adoption"] == "staggered"
    assert scope["status"] == "not_covered"


def test_etwfe_headline_se_is_a_disclosure_and_the_default_call_is_attached():
    data = pd.read_csv(ROOT / "tests/r_parity/data/17_etwfe.csv")
    keys = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rcs = sp.etwfe(data, panel=False, cluster="countyreal", **keys)
        default = sp.etwfe(data, **keys)
        never = sp.etwfe(data, cgroup="nevertreated", **keys)
        unit_w = sp.etwfe(data, agg_weights="unit", **keys)
        logit = sp.etwfe(
            data.assign(yb=(data["lemp"] > data["lemp"].median()).astype(int)),
            y="yb",
            group="countyreal",
            time="year",
            first_treat="first_treat",
            family="logit",
        )
    for fit in (rcs, default, never):
        scope, out = _scope_of(fit)
        assert scope["function"] == "etwfe"
        assert out["estimate"] == "reference" and out["vcov"] == "reference"
        # 2e-6 from etwfe::emfx, 6e-4 from jwdid: documented, not same-byte.
        assert out["se"] == "disclosure"
        assert scope["status"] == "estimate_only"
    attached = _scope_of(default)[0]["outputs"]["estimate"]["evidence"]
    assert [e["artifact"] for e in attached] == [
        "tests/reference_parity/test_validation_entry_points.py"
    ]
    assert _scope_of(unit_w)[0]["status"] == "not_covered"
    # A GLM family is routed to its own map rather than failing this one.
    scope, _ = _scope_of(logit)
    assert scope["function"] == "etwfe_glm" and scope["status"] == "estimate_only"


def test_wooldridge_did_does_not_borrow_the_etwfe_map():
    """Different aggregation (see MIGRATION): it must not read as etwfe."""
    data = pd.read_csv(ROOT / "tests/r_parity/data/17_etwfe.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.wooldridge_did(
            data=data,
            y="lemp",
            group="countyreal",
            time="year",
            first_treat="first_treat",
        )
    with pytest.raises(MethodIncompatibility):
        sp.validation_scope(fit)


def test_nonlinear_etwfe_has_its_own_map():
    """family='poisson' is routed to the GLM map, whose option axes differ."""
    panel = pd.read_csv(
        ROOT / "tests/reference_parity/_fixtures/etwfe_poisson_panel.csv"
    )
    keys = dict(y="y", group="id", time="year", first_treat="g")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poisson = sp.etwfe(panel, **keys, family="poisson")
        unit_fe = sp.etwfe(panel, **keys, family="poisson", fe="unit")
        linear = sp.etwfe(panel, **keys)
    scope, out = _scope_of(poisson)
    assert scope["function"] == "etwfe_glm" and scope["unchecked"] == []
    assert out["estimate"] == "reference"
    # ~1e-5 from etwfe::emfx: a finite-sample convention, not same-byte.
    assert out["se"] == "disclosure"
    # Unit effects: Stata jwdid is the reference. The response-scale SE
    # under the default response_se='profile' is not compared with it; on
    # the link scale both the estimate and the SE are.
    assert _scope_of(unit_fe)[0]["status"] == "estimate_only"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        link = sp.etwfe(panel, **keys, family="poisson", fe="unit", scale="link")
        never = sp.etwfe(panel, **keys, fe="unit", cgroup="nevertreated")
    assert _scope_of(link)[0]["status"] == "covered"
    # The linear fe='unit' path records Stata's 'never'; the map reads it.
    never_scope = _scope_of(never)[0]
    assert never_scope["configuration"]["cgroup"] == "nevertreated"
    assert never_scope["configuration"]["family"] == "gaussian"
    assert never_scope["status"] == "covered"
    assert _scope_of(linear)[0]["function"] == "etwfe"
    # The result card goes through the same routing.
    assert sp.result_card(poisson)["evidence"]["outputs"]["se"] == "disclosure"


def test_panel_clustered_joint_test_depends_on_the_small_sample_convention():
    """Same fit, same hypothesis: a match under ssc='stata', a disclosure by default."""
    panel = pd.read_csv(ROOT / "tests/stata_translation_holdout/holdout_panel.csv")
    keys = dict(entity="id", time="t", method="fe", cluster="id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = sp.validation_scope(sp.panel(panel, "y ~ x", **keys))
        stata = sp.validation_scope(sp.panel(panel, "y ~ x", ssc="stata", **keys))
    assert default["outputs"]["joint_test"]["status"] == "disclosure"
    assert stata["outputs"]["joint_test"]["status"] == "reference"
    assert "ssc='stata'" in default["outputs"]["joint_test"]["evidence"][0]["compares"]


def test_iv_joint_test_is_covered_where_the_variance_is():
    cross = pd.read_csv(ROOT / "tests/stata_translation_holdout/holdout_cross.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hc1 = sp.validation_scope(
            sp.ivreg("y ~ x1 + (x2 ~ z + z2)", data=cross, robust="hc1")
        )
        hc3 = sp.validation_scope(
            sp.ivreg("y ~ x1 + (x2 ~ z + z2)", data=cross, robust="hc3")
        )
    assert hc1["outputs"]["joint_test"]["status"] == "reference"
    assert hc3["outputs"]["joint_test"]["status"] == "not_covered"


def test_nnmatch_has_its_own_map_and_matchit_rows_reach_psm():
    """``sp.match`` is a dispatcher: two estimators, two maps.

    ``method='nnmatch'`` (Abadie-Imbens covariate matching) used to be
    routed to the propensity-score map, where none of its options exist,
    so every fit read as "no evidence" although nine Stata commands pin it.
    ``method='nearest'`` is the same fit as ``sp.psm``, which is why the
    MatchIt comparisons made through ``sp.match`` count for that map.
    """
    df = sp.datasets.nsw_dw()
    covs = ["age", "education", "black", "hispanic", "married", "nodegree"]
    keys = dict(y="re78", treat="treat", covariates=covs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nn = sp.match(df, method="nnmatch", **keys)
        nn5 = sp.match(df, method="nnmatch", n_matches=5, **keys)
        via_match = sp.match(df, method="nearest", **keys)
        via_psm = sp.psm(df, **keys)
        no_replace = sp.match(df, method="nearest", replace=False, **keys)
    scope = sp.validation_scope(nn)
    assert scope["function"] == "nnmatch" and scope["status"] == "covered"
    assert scope["configuration"]["estimand"] == "att"
    assert sp.validation_scope(nn5)["status"] == "not_covered"
    # the dispatcher and the direct entry point are one fit
    assert via_match.estimate == via_psm.estimate and via_match.se == via_psm.se
    assert sp.validation_scope(via_match)["function"] == "psm"
    out = sp.validation_scope(no_replace)["outputs"]
    assert out["estimate"]["status"] == "reference"
    assert any("MatchIt" in e["compares"] for e in out["estimate"]["evidence"])
