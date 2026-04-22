"""Tests for the ## For Agents markdown renderer."""

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
def demo_spec():
    name = "__test_doc_scratch_fn__"
    spec = FunctionSpec(
        name=name,
        category="causal",
        description="Scratch spec for agent doc renderer.",
        params=[ParamSpec("data", "DataFrame", True)],
        assumptions=["Parallel trends", "No anticipation"],
        pre_conditions=["Panel: unit x time", "Binary treatment column"],
        failure_modes=[
            FailureMode(
                symptom="pretrend p < 0.05",
                exception="statspai.AssumptionViolation",
                remedy="Try honest_did / sensitivity_rr",
                alternative="sp.sensitivity_rr",
            ),
        ],
        alternatives=["callaway_santanna", "did_imputation"],
        typical_n_min=50,
    )
    register(spec)
    yield spec
    _REGISTRY.pop(name, None)


class TestRenderAgentBlock:
    def test_empty_for_unpopulated_entry(self):
        # qreg (quantile regression) has no agent-native fields populated
        block = sp.render_agent_block("qreg")
        assert block == ""

    def test_renders_header_by_default(self, demo_spec):
        block = sp.render_agent_block(demo_spec.name)
        assert block.startswith("## For Agents")

    def test_no_header_when_requested(self, demo_spec):
        block = sp.render_agent_block(demo_spec.name, header=False)
        assert not block.startswith("## For Agents")
        # still has section labels
        assert "**Pre-conditions**" in block

    def test_all_sections_rendered(self, demo_spec):
        block = sp.render_agent_block(demo_spec.name)
        assert "**Pre-conditions**" in block
        assert "**Identifying assumptions**" in block
        assert "**Failure modes → recovery**" in block
        assert "**Alternatives (ranked)**" in block
        assert "**Typical minimum N**: 50" in block

    def test_failure_table_structure(self, demo_spec):
        block = sp.render_agent_block(demo_spec.name)
        assert "| Symptom | Exception | Remedy | Try next |" in block
        assert "| --- | --- | --- | --- |" in block
        assert "`statspai.AssumptionViolation`" in block
        assert "`sp.sensitivity_rr`" in block

    def test_alternatives_wrapped_as_code(self, demo_spec):
        block = sp.render_agent_block(demo_spec.name)
        assert "- `sp.callaway_santanna`" in block
        assert "- `sp.did_imputation`" in block

    def test_pipe_in_symptom_is_escaped(self):
        name = "__test_pipe_escape__"
        register(FunctionSpec(
            name=name,
            category="causal",
            description="",
            failure_modes=[
                FailureMode(
                    symptom="contains | pipe character",
                    exception="ValueError",
                    remedy="nothing",
                ),
            ],
        ))
        try:
            block = sp.render_agent_block(name)
            assert "\\|" in block
            # Markdown table must still have exactly 4 delimiter pipes
            # on the row (outer two + three inner).
            data_rows = [ln for ln in block.splitlines()
                         if ln.startswith("| contains")]
            assert len(data_rows) == 1
        finally:
            _REGISTRY.pop(name, None)

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            sp.render_agent_block("no_such_function_xyz_123")


class TestRenderAgentBlocks:
    def test_empty_when_no_matches(self):
        # Filter by category that has nothing populated
        out = sp.render_agent_blocks(category="datasets")
        assert out == ""

    def test_includes_registered_demo_spec(self, demo_spec):
        out = sp.render_agent_blocks(category="causal")
        assert out.startswith("## For Agents")
        assert f"### `sp.{demo_spec.name}`" in out
        assert "**Identifying assumptions**" in out

    def test_explicit_names_filter(self, demo_spec):
        out = sp.render_agent_blocks(names=[demo_spec.name])
        assert f"### `sp.{demo_spec.name}`" in out
