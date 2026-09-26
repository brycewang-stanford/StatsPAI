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


def test_regress_cr2_is_estimate_only(card):
    df = card.assign(cl=np.arange(len(card)) % 40)
    scope = sp.validation_scope(
        sp.regress("lwage ~ educ", data=df, vce="cr2", cluster="cl")
    )
    assert scope["configuration"]["vce"] == "cr2"
    assert scope["status"] == "estimate_only"


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
    hc3 = sp.validation_scope(function="regress", vce="hc3", weights="none")
    assert hc3["outputs"]["joint_test"]["status"] == "not_covered"
    assert hc3["outputs"]["se"]["status"] == "reference"


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
