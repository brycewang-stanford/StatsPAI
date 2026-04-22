"""Tests for the agent-native registry schema extensions.

Covers the *mechanics* of FunctionSpec agent-native fields
(assumptions / pre_conditions / failure_modes / alternatives /
typical_n_min) and their public accessors (sp.agent_card /
sp.agent_cards). Per-estimator population is tested elsewhere once
flagship families are filled in.
"""

import pytest

import statspai as sp
from statspai.registry import (
    FailureMode,
    FunctionSpec,
    ParamSpec,
    _REGISTRY,
    register,
)


@pytest.fixture
def scratch_spec():
    """Register a synthetic spec then clean up."""
    name = "__test_agent_scratch_fn__"
    spec = FunctionSpec(
        name=name,
        category="causal",
        description="Synthetic test estimator for agent-native schema.",
        params=[ParamSpec("data", "DataFrame", True, description="input")],
        returns="CausalResult",
        tags=["test"],
        pre_conditions=[
            "data is a pandas DataFrame",
            "treatment column is binary",
        ],
        assumptions=[
            "Parallel trends",
            "No anticipation",
        ],
        failure_modes=[
            FailureMode(
                symptom="Pre-trend test rejects at 5%",
                exception="statspai.AssumptionViolation",
                remedy="Try sp.callaway_santanna or honest_did",
                alternative="sp.callaway_santanna",
            ),
        ],
        alternatives=["callaway_santanna", "did_imputation"],
        typical_n_min=50,
    )
    register(spec)
    yield spec
    _REGISTRY.pop(name, None)


class TestFunctionSpecAgentFields:
    def test_defaults_are_empty(self):
        spec = FunctionSpec(name="foo", category="causal", description="x")
        assert spec.assumptions == []
        assert spec.pre_conditions == []
        assert spec.failure_modes == []
        assert spec.alternatives == []
        assert spec.typical_n_min is None

    def test_to_dict_includes_new_fields(self, scratch_spec):
        d = scratch_spec.to_dict()
        assert "assumptions" in d
        assert "pre_conditions" in d
        assert "failure_modes" in d
        assert "alternatives" in d
        assert "typical_n_min" in d
        assert d["typical_n_min"] == 50
        # failure_modes nested asdict
        assert d["failure_modes"][0]["symptom"].startswith("Pre-trend")
        assert d["failure_modes"][0]["exception"] == "statspai.AssumptionViolation"

    def test_agent_card_shape(self, scratch_spec):
        card = scratch_spec.agent_card()
        for key in (
            "name", "category", "description", "signature",
            "pre_conditions", "assumptions", "failure_modes",
            "alternatives", "typical_n_min", "reference", "example",
        ):
            assert key in card, f"agent_card missing {key!r}"
        assert card["signature"]["name"] == scratch_spec.name
        assert card["typical_n_min"] == 50
        assert card["alternatives"] == ["callaway_santanna", "did_imputation"]
        assert card["failure_modes"][0]["alternative"] == "sp.callaway_santanna"


class TestPublicAgentCardAPI:
    def test_sp_agent_card_roundtrip(self, scratch_spec):
        card = sp.agent_card(scratch_spec.name)
        assert card["name"] == scratch_spec.name
        assert len(card["assumptions"]) == 2

    def test_sp_agent_card_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown function"):
            sp.agent_card("definitely_no_such_fn_xyz")

    def test_sp_agent_cards_filters_empty(self, scratch_spec):
        # All cards include only specs with at least one agent-native
        # field populated. Scratch spec has them; most auto-registered
        # specs do not.
        cards = sp.agent_cards()
        names = {c["name"] for c in cards}
        assert scratch_spec.name in names

    def test_sp_agent_cards_category_filter(self, scratch_spec):
        causal = sp.agent_cards(category="causal")
        causal_names = {c["name"] for c in causal}
        assert scratch_spec.name in causal_names

        reg_cards = sp.agent_cards(category="regression")
        reg_names = {c["name"] for c in reg_cards}
        assert scratch_spec.name not in reg_names


class TestFailureMode:
    def test_to_dict(self):
        fm = FailureMode(
            symptom="Weak instrument",
            exception="ValueError",
            remedy="Use Anderson-Rubin CI",
            alternative="sp.anderson_rubin_ci",
        )
        d = fm.to_dict()
        assert d == {
            "symptom": "Weak instrument",
            "exception": "ValueError",
            "remedy": "Use Anderson-Rubin CI",
            "alternative": "sp.anderson_rubin_ci",
        }

    def test_default_alternative(self):
        fm = FailureMode("symptom", "Exc", "remedy")
        assert fm.alternative == ""


class TestBackwardCompatibility:
    def test_existing_specs_still_work(self):
        # Sanity: no field we added breaks existing describe/schema output
        info = sp.describe_function("regress")
        assert info["name"] == "regress"
        assert "params" in info
        schema = sp.function_schema("regress")
        assert schema["name"] == "regress"
        assert "parameters" in schema


class TestFlagshipPopulated:
    """Hand-written flagship families must carry agent-native metadata."""

    FLAGSHIPS = ("regress", "iv", "did", "callaway_santanna", "rdrobust", "synth")

    @pytest.mark.parametrize("name", FLAGSHIPS)
    def test_has_assumptions(self, name):
        card = sp.agent_card(name)
        assert card["assumptions"], f"{name} must declare assumptions"

    @pytest.mark.parametrize("name", FLAGSHIPS)
    def test_has_pre_conditions(self, name):
        card = sp.agent_card(name)
        assert card["pre_conditions"], f"{name} must declare pre-conditions"

    @pytest.mark.parametrize("name", FLAGSHIPS)
    def test_has_failure_modes(self, name):
        card = sp.agent_card(name)
        fms = card["failure_modes"]
        assert fms, f"{name} must declare failure modes"
        for fm in fms:
            assert fm["symptom"]
            assert fm["exception"]
            assert fm["remedy"]

    @pytest.mark.parametrize("name", FLAGSHIPS)
    def test_has_typical_n_min(self, name):
        card = sp.agent_card(name)
        assert card["typical_n_min"] is not None, (
            f"{name} must declare typical_n_min"
        )
        assert isinstance(card["typical_n_min"], int)
        assert card["typical_n_min"] > 0

    @pytest.mark.parametrize("name", FLAGSHIPS)
    def test_renders_nonempty_block(self, name):
        block = sp.render_agent_block(name)
        assert block.startswith("## For Agents")
        assert "**Pre-conditions**" in block
        assert "**Identifying assumptions**" in block
        assert "**Failure modes → recovery**" in block
