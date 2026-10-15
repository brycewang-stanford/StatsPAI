"""The agent cards of the 70 most-used entry points say true things.

``scripts/agent_card_audit.py`` checks each card against a real call
(review item A1). The enum sweep takes about ten minutes, so it runs
offline and its result is committed as ``docs/dev/agent_card_audit.json``;
this file runs the fast layer on every test run and pins the committed
result to the schemas it was computed from.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

import statspai as sp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agent_card_audit.py"
REPORT = ROOT / "docs" / "dev" / "agent_card_audit.json"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="source checkout only (scripts/ not installed)"
)


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("agent_card_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_seventy_functions_each_with_a_call(audit):
    assert len(audit.TOP_30) == len(set(audit.TOP_30)) == 30
    assert len(audit.AUDITED) == len(set(audit.AUDITED)) == 70
    assert set(audit.AUDITED) <= set(audit.CALLS)
    registered = set(sp.list_functions())
    assert set(audit.AUDITED) <= registered


@pytest.mark.parametrize("name", ["feols", "fepois", "feglm"])
def test_multiple_estimation_entry_points_declare_the_single_result(name):
    """``Union[EconometricResults, List[...]]``: a plain formula returns one."""
    assert sp.describe_function(name)["result_class"] == "EconometricResults"


def test_fast_layer_finds_no_defect(audit):
    """Required arguments, result class and alternatives, by real calls."""
    pytest.importorskip("pyfixest")  # the audit calls sp.feols itself
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = audit.build(enums=False)
    defects = {f["function"]: f["defects"] for f in fresh["functions"] if f["defects"]}
    assert not defects, defects
    for f in fresh["functions"]:
        assert f["base_call"] == "ok"
        assert f["result_class"]["matches"] is True


def test_committed_enum_sweep_has_no_rejected_value(report):
    assert report["n_functions"] == 70
    assert report["n_with_defects"] == 0
    assert report["enum_values"]["rejected"] == 0
    assert report["enum_values"]["ok"] > 200


def test_enum_values_added_since_the_sweep_are_accepted(audit, report):
    """The committed sweep may lag the schemas; a new value is tried here.

    Re-running the whole sweep takes about ten minutes, so a commit that
    adds an enum value to one of the seventy is not asked to. Instead every
    value the committed report has not seen is called for real, now: it may
    need a precondition, it may not be refused. Regenerate the report
    (``python scripts/agent_card_audit.py``) when convenient.
    """
    committed = {
        (f["function"], arg, value)
        for f in report["functions"]
        for arg, per_value in (f.get("enums") or {}).items()
        for value in per_value
    }
    fresh = []
    for name in audit.AUDITED:
        props = sp.function_schema(name)["parameters"]["properties"]
        for arg, spec in props.items():
            for value in spec.get("enum") or []:
                if (name, arg, str(value)) not in committed:
                    fresh.append((name, arg, value))
    assert len(fresh) <= 25, (
        f"{len(fresh)} enum values are newer than the committed sweep; "
        "run python scripts/agent_card_audit.py"
    )
    rejected = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, arg, value in fresh:
            outcome = audit.try_enum_value(name, arg, value)
            if outcome["status"] == "rejected":
                rejected.append((name, arg, value, outcome["error"]))
    assert not rejected, rejected


def test_every_precondition_carries_its_reason(report):
    for f in report["functions"]:
        for per_value in (f.get("enums") or {}).values():
            for outcome in per_value.values():
                if outcome["status"] == "precondition":
                    assert len(outcome["error"]) > 20


# ---------------------------------------------------------------------------
# What the cards say (read by hand on 2026-10-03; see VARIANT_OVERRIDES)
# ---------------------------------------------------------------------------


def _assumptions(name):
    return " | ".join(sp.describe_function(name)["assumptions"])


def _symptoms(name):
    return " | ".join(f["symptom"] for f in sp.describe_function(name)["failure_modes"])


def test_binary_models_do_not_list_ordered_or_multinomial_assumptions():
    for name in ("logit", "probit"):
        text = _assumptions(name)
        assert "independence of irrelevant alternatives" not in text
        assert "proportional odds" not in text
        assert "link function" in text  # the family statement that does apply
    # ...and the models those statements are about keep them.
    assert "independence of irrelevant alternatives" in _assumptions("mlogit")
    assert "proportional odds" in _assumptions("ologit")
    assert "proportional odds" not in _assumptions("mlogit")
    assert "independence of irrelevant alternatives" not in _assumptions("ologit")


def test_robust_did_estimators_are_not_told_to_use_themselves():
    for name in ("callaway_santanna", "sun_abraham", "did_imputation", "etwfe"):
        assert "use CS or SA" not in _assumptions(name), name
        assert "TWFE method" not in _symptoms(name), name
        assert "Parallel trends" in _assumptions(name)
        assert _assumptions(name).count("SUTVA") == 1, name
    # The advice stays where it belongs: on the dispatcher that can run TWFE.
    assert "use CS or SA" in _assumptions("did")
    assert "TWFE method" in _symptoms("did")


def test_rdrobust_card_covers_the_fuzzy_and_kink_designs():
    text = _assumptions("rdrobust")
    assert "Continuity of potential outcomes" in text
    assert "fuzzy=" in text and "monotonicity" in text and "compliers" in text
    assert "deriv=1 (kink)" in text
    # Continuity-based local polynomials, not the local-randomization framework.
    assert "Local randomization only" not in text


def test_point_treatment_ipw_is_not_described_as_a_longitudinal_method():
    text = _assumptions("ipw")
    assert "Sequential exchangeability" not in text and "given the past" not in text
    assert "point treatment" in text and "propensity model" in text


def test_dml_card_names_the_parameter_it_estimates():
    text = _assumptions("dml")
    assert "√n CATE" not in text
    assert "√n-consistent" in text


def test_overrides_only_name_real_functions_and_real_statements():
    """An override that matches nothing is a typo or a stale entry."""
    from statspai._family_cards import VARIANT_OVERRIDES
    from statspai.registry import _REGISTRY

    for name, override in VARIANT_OVERRIDES.items():
        assert name in _REGISTRY, name
        raw = _REGISTRY[name].agent_card()  # after overrides
        for field, texts in override.get("add", {}).items():
            for text in texts:
                assert text in raw[field], (name, text[:40])
        for field, prefixes in override.get("drop", {}).items():
            values = [v["symptom"] if isinstance(v, dict) else v for v in raw[field]]
            added = override.get("add", {}).get(field, [])
            for prefix in prefixes:
                left = [v for v in values if v.startswith(prefix) and v not in added]
                assert not left, (name, prefix)


# ---------------------------------------------------------------------------
# Family statements reach only the members they name (all 30 family cards)
# ---------------------------------------------------------------------------


def test_scope_table_names_real_statements_and_real_members():
    from statspai._family_cards import FAMILY_CARDS, STATEMENT_SCOPE

    for family, table in STATEMENT_SCOPE.items():
        statements = FAMILY_CARDS[family]["assumptions"]
        members = set(FAMILY_CARDS[family]["members"])
        for prefix, scoped in table.items():
            hits = [a for a in statements if a.startswith(prefix)]
            assert len(hits) == 1, (family, prefix, len(hits))
            assert set(scoped) <= members, (family, prefix, set(scoped) - members)
            assert 0 < len(scoped) < len(members), (family, prefix)


@pytest.mark.parametrize(
    "name,absent,present",
    [
        ("kaplan_meier", "proportional hazards", "censoring"),
        ("logrank_test", "Frailty models", "censoring"),
        ("cox", "baseline distribution", "proportional hazards"),
        ("aft", "proportional hazards", "baseline distribution"),
        ("bonferroni", "Romano-Wolf", "family-wise error rate"),
        ("romano_wolf", "Bonferroni / Holm", "bootstrap"),
        ("heckman", "Rosenbaum bounds", "exclusion restriction"),
        ("rosenbaum_bounds", "joint normality", "unobserved confounder"),
        ("granger_causality", "Cholesky", "predictive precedence"),
        ("mice", "sharp null", "missing at random"),
        ("ri_test", "missing at random", "sharp null"),
        ("gmm", "SUR / 3SLS", "moment conditions"),
        ("hausman_test", "Marginal effects are averages", "efficient under the null"),
        ("peer_effects", "QAP permutations", "reflection problem"),
        ("meta_analysis", "genetic instruments", "exchangeable"),
    ],
)
def test_member_card_keeps_its_own_statement_and_loses_its_sibling_s(
    name, absent, present
):
    text = _assumptions(name)
    assert absent not in text, f"sp.{name} still lists a sibling's assumption"
    assert present in text, f"sp.{name} lost the statement that is about it"


def test_methods_left_without_a_family_statement_have_their_own():
    """Scoping emptied these; a method with no stated assumption is a gap."""
    assert "moderator" in _assumptions("interflex")
    assert "random-intercept" in _assumptions("icc")
    assert "selection-by-time" in _assumptions("negd")
    # None of them keeps the unrelated statement it used to inherit.
    assert "g-formula" not in _assumptions("interflex")
    assert "inefficiency distribution" not in _assumptions("icc")


def test_utilities_carry_no_borrowed_assumption():
    """A weights constructor or an exporter has nothing to assume; say nothing."""
    for name in ("W", "scdata", "influence_functions", "validation_scope"):
        assert sp.describe_function(name)["assumptions"] == [], name


# ---------------------------------------------------------------------------
# The same for failure modes
# ---------------------------------------------------------------------------


def _symptoms(name):
    return " | ".join(f["symptom"] for f in sp.describe_function(name)["failure_modes"])


def test_failure_scope_names_real_symptoms_and_real_members():
    from statspai._family_cards import FAILURE_SCOPE, FAMILY_CARDS

    n_scoped = 0
    for family, table in FAILURE_SCOPE.items():
        symptoms = [f["symptom"] for f in FAMILY_CARDS[family]["failure_modes"]]
        members = set(FAMILY_CARDS[family]["members"])
        for prefix, scoped in table.items():
            hits = [s for s in symptoms if s.startswith(prefix)]
            assert len(hits) == 1, (family, prefix, len(hits))
            assert set(scoped) <= members, (family, prefix, set(scoped) - members)
            assert 0 < len(scoped) < len(members), (family, prefix)
            n_scoped += 1
    assert n_scoped == 28


@pytest.mark.parametrize(
    "name,absent,present",
    [
        ("kaplan_meier", "Proportional-hazards test", "Competing events"),
        ("logrank_test", "events per covariate", "curves cross"),
        ("cox", "curves cross", "Proportional-hazards test"),
        ("romano_wolf", "All adjusted p-values become 1", "bootstrap replications"),
        ("bonferroni", "bootstrap replications", "All adjusted p-values become 1"),
        ("johansen", "GARCH", "integrated of order one"),
        ("garch", "integrated of order one", "GARCH"),
        ("lincom", "Hausman statistic negative", "Few clusters"),
        ("hausman_test", "RESET rejects", "Hausman statistic negative"),
        ("mice", "Many subgroups", "Imputation model omits"),
        ("frontdoor", "mediator-outcome confounding (rho)", "direct path"),
        ("evalue_rr", "Heckman", "without a benchmark"),
        ("power_rct", "ICC unknown", "randomised in clusters"),
    ],
)
def test_member_card_keeps_its_own_failure_mode_and_loses_its_sibling_s(
    name, absent, present
):
    text = _symptoms(name)
    assert absent not in text, f"sp.{name} still lists a sibling's failure mode"
    assert present in text, f"sp.{name} lost the failure mode that is about it"


def test_member_failure_modes_name_real_functions_and_real_alternatives():
    from statspai._family_cards import MEMBER_FAILURE_MODES, family_members

    known = set(sp.list_functions())
    in_a_family = {m for members in family_members().values() for m in members}
    for members, mode in MEMBER_FAILURE_MODES:
        assert set(members) <= known & in_a_family, set(members) - known
        assert mode["symptom"] and mode["remedy"] and mode["exception"]
        alt = mode.get("alternative", "")
        if alt:
            assert alt.startswith("sp.") and alt[3:] in known, alt
            assert alt[3:] not in members, f"{alt} recommended to itself"


def test_datasets_carry_no_borrowed_failure_mode():
    for name in ("karate_club", "florentine_families", "validation_scope"):
        assert sp.describe_function(name)["failure_modes"] == [], name


def test_no_card_lists_an_alternative_twice_or_itself():
    """Every registered function, not only the audited seventy.

    Until 2026-10-05, 69 cards listed one alternative under two spellings
    (``did_imputation`` from the family card, ``sp.did_imputation`` from
    the curated one) and 12 named the function itself as its alternative.
    """
    import statspai as sp
    from statspai.registry import _provenance_key

    offenders = []
    for name in sp.list_functions():
        keys = [
            _provenance_key("alternatives", a)
            for a in sp.agent_card(name)["alternatives"]
        ]
        if len(keys) != len(set(keys)) or name in keys:
            offenders.append(name)
    assert offenders == []
    # the merge keeps the alternatives a card had, once each
    alts = sp.agent_card("lp_did")["alternatives"]
    assert [_provenance_key("alternatives", a) for a in alts] == [
        "callaway_santanna",
        "sun_abraham",
        "did_imputation",
        "gardner_did",
        "did_multiplegt",
    ]


# ---------------------------------------------------------------------------
# The fifth ten (read and probed on 2026-10-15)
# ---------------------------------------------------------------------------


def _pre_conditions(name):
    return " | ".join(sp.describe_function(name)["pre_conditions"])


def _exception_of(name, symptom_prefix):
    for fm in sp.describe_function(name)["failure_modes"]:
        if fm["symptom"].startswith(symptom_prefix):
            return fm.get("exception", "")
    raise AssertionError(f"sp.{name}: no failure mode {symptom_prefix!r}")


def test_arguments_with_a_second_route_are_not_marked_required():
    """``y=`` / ``x=`` replace the formula, ``| id`` the entity, ``pscore=``
    the covariates: the schema used to call all four required."""
    assert sp.function_schema("xtnbreg")["parameters"]["required"] == ["data"]
    assert "covariates" not in sp.function_schema("psmatch2")["parameters"]["required"]
    for name in ("xtnbreg", "psmatch2"):
        props = sp.function_schema(name)["parameters"]["properties"]
        assert "y" in props, name  # the second route is visible to the agent


def test_quoted_none_in_a_docstring_set_is_a_choice():
    """``{'auto', 'parametric', 'nonparametric', 'none'}`` lost ``'none'``."""
    enum = sp.function_schema("gsynth")["parameters"]["properties"]["inference"]["enum"]
    assert "none" in enum
    from statspai.registry import _enum_from_text

    assert _enum_from_text("{'a', 'none'}") == ["a", "none"]
    assert _enum_from_text("{a, b, None}") == ["a", "b"]  # bare None is notation


def test_point_treatment_g_formula_is_not_described_as_a_longitudinal_method():
    for name in ("g_computation", "ipw"):
        pre = _pre_conditions(name)
        assert "Sequentially measured" not in pre and "point treatment" in pre, name
    text = _assumptions("g_computation")
    assert "Sequential exchangeability" not in text and "point treatment" in text
    assert "Stabilized weights" not in _symptoms("g_computation")
    assert _exception_of("g_computation", "estimand='dose_response'") == "ValueError"


def test_dcdh_cards_do_not_inherit_the_twfe_dispatcher_s_statements():
    for name in ("did_multiplegt", "did_multiplegt_dyn"):
        assert "use CS or SA" not in _assumptions(name), name
        assert "TWFE method" not in _symptoms(name), name
        pre = _pre_conditions(name)
        assert "first-treatment-period" not in pre, name
        assert "discrete" in pre, name  # both take non-binary treatments
        # the joint placebo test is reported, never raised
        assert _exception_of(name, "Joint placebo test rejects").startswith("(none")
    # stale statements: switch-offs are handled, the event study exists
    assert "silently ignores switch-off" not in str(
        sp.describe_function("did_multiplegt_dyn")
    )
    assert "on the roadmap" not in sp.describe_function("did_multiplegt")["description"]
    assert _exception_of("did_multiplegt", "No switching cell").startswith("(none")
    assert (
        _exception_of("did_multiplegt_dyn", "No switcher can be matched")
        == "DataInsufficient"
    )


def test_synthetic_control_template_is_scoped_off_gsynth():
    text = _assumptions("gsynth")
    assert "convex (or regularized) combination" not in text
    assert "interactive fixed-effects" in text
    assert "Large pre-treatment RMSPE" not in _symptoms("gsynth")
    assert "min_T0" in _pre_conditions("gsynth")


def test_diagnostic_bundle_reports_a_weak_first_stage_instead_of_raising():
    assert "A strong first stage" not in _pre_conditions("iv_diag")
    assert _exception_of("iv_diag", "Weak first stage").startswith("(none")


def test_exceptions_named_on_the_fifth_ten_are_the_ones_raised():
    assert _exception_of("aggte", "Empty event-time aggregation") == "ValueError"
    assert (
        _exception_of("honest_did", "Passed result lacks event_study")
        == "MethodIncompatibility"
    )
    assert _exception_of("drdid", "Propensity score near 0/1").startswith("(none")
    assert "staggered=True" not in _pre_conditions("aggte")
    # xtnbreg's default is the conditional (HHG) model, not entity dummies
    assert "adds explicit entity dummies" not in _assumptions("xtnbreg")
    assert "Hausman-Hall-Griliches" in _assumptions("xtnbreg")


def test_spec_curve_exposes_its_standard_error_axis():
    props = sp.function_schema("spec_curve")["parameters"]["properties"]
    assert props["se_types"]["items"]["enum"] == [
        "nonrobust",
        "hc1",
        "robust",
        "cluster",
    ]
    assert "cluster_var" in props
