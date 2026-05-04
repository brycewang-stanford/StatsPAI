"""Tests for agent/auto_tools.py — auto-generated MCP tool manifest.

Covers
------
* ``_is_agent_safe`` — private/internal/class/deny-list rejection.
* ``_enrich_description`` — agent-card blurb appending with truncation.
* ``_json_safe_value`` / ``_clean_properties`` — JSON round-trip safety.
* ``_schema_to_mcp_tool`` — OpenAI → MCP schema conversion.
* ``auto_tool_manifest`` — full manifest generation (with mocked registry).
* ``merged_tool_manifest`` — hand-curated + auto merge with deduplication.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

import statspai

from statspai.agent.auto_tools import (
    DEFAULT_EXCLUDE,
    DEFAULT_WHITELIST,
    _clean_properties,
    _enrich_description,
    _is_agent_safe,
    _json_safe_value,
    _schema_to_mcp_tool,
    auto_tool_manifest,
    merged_tool_manifest,
)


# ======================================================================
#  _is_agent_safe
# ======================================================================


class MockSpec:
    """Minimal stand-in for a RegistryEntry."""

    def __init__(self, name="some_func", category="causal", tags=None):
        self.name = name
        self.category = category
        self.tags = tags or []


def test_agent_safe_private():
    spec = MockSpec("_private_fn")
    assert not _is_agent_safe("_private_fn", spec)


def test_agent_safe_deny_listed():
    spec = MockSpec("deepiv")
    assert not _is_agent_safe("deepiv", spec)


def test_agent_safe_internal_tag():
    spec = MockSpec("internal_fn", tags=["internal"])
    assert not _is_agent_safe("internal_fn", spec)


def test_agent_safe_class_heuristic():
    """PascalCase names that resolve to classes are rejected."""

    class MyClass:
        pass

    with patch.object(statspai, "MyClass", MyClass, create=True):
        assert not _is_agent_safe("MyClass", MockSpec("MyClass"))


def test_agent_safe_normal_function():
    """Normal function names are accepted."""
    spec = MockSpec("did", category="causal")
    assert _is_agent_safe("did", spec)


# ======================================================================
#  _enrich_description
# ======================================================================


def test_enrich_empty_card():
    desc = _enrich_description("Base description.", None)
    assert desc == "Base description."


def test_enrich_with_assumptions():
    card = {"assumptions": ["Linear model", "No multicollinearity"]}
    desc = _enrich_description("Base.", card)
    assert "Base." in desc
    assert "Assumptions:" in desc
    assert "Linear model" in desc


def test_enrich_with_failure_modes():
    card = {
        "failure_modes": [
            {"symptom": "Not converging", "remedy": "Increase iterations"},
        ],
    }
    desc = _enrich_description("Base.", card)
    assert "Failure modes:" in desc
    assert "Not converging" in desc


def test_enrich_with_alternatives():
    card = {"alternatives": ["did", "rd"]}
    desc = _enrich_description("Base.", card)
    assert "Alternatives:" in desc
    assert "sp.did" in desc


def test_enrich_truncation():
    """Overly long descriptions are truncated to MAX_DESCRIPTION_LEN."""
    base = "x" * 1100
    card = {"assumptions": [("y" * 200)]}
    desc = _enrich_description(base, card)
    assert len(desc) <= 1200


# ======================================================================
#  _json_safe_value
# ======================================================================


def test_json_safe_primitives():
    assert _json_safe_value(None) is None
    assert _json_safe_value(True) is True
    assert _json_safe_value(42) == 42
    assert _json_safe_value(3.14) == 3.14
    assert _json_safe_value("hello") == "hello"


def test_json_safe_list():
    result = _json_safe_value([1, "two", 3.0])
    assert result == [1, "two", 3.0]


def test_json_safe_unsafe_default():
    """Callables / sentinels → None."""
    assert _json_safe_value(object()) is None
    assert _json_safe_value(lambda: 42) is None


# ======================================================================
#  _clean_properties
# ======================================================================


def test_clean_properties_removes_bad_default():
    props = {
        "x": {"type": "number", "default": 5},
        "y": {"type": "string", "default": object()},  # unsafe
    }
    cleaned = _clean_properties(props)
    assert "default" in cleaned["x"]
    assert cleaned["x"]["default"] == 5
    assert "default" not in cleaned["y"]


def test_clean_properties_skips_non_dict():
    cleaned = _clean_properties({"x": "not_a_dict"})
    assert "x" not in cleaned


# ======================================================================
#  _schema_to_mcp_tool
# ======================================================================


def test_schema_to_mcp_tool_basic():
    schema = {
        "name": "test_function",
        "description": "A test function.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "data": {"type": "string"},  # should be removed
            },
            "required": ["x", "data"],
        },
    }
    tool = _schema_to_mcp_tool(schema, None)
    assert tool["name"] == "test_function"
    assert "description" in tool
    assert "input_schema" in tool
    # data param removed
    assert "data" not in tool["input_schema"]["properties"]
    assert "data" not in tool["input_schema"]["required"]
    assert tool["input_schema"]["required"] == ["x"]


def test_schema_to_mcp_tool_enriches():
    card = {"assumptions": ["Linear"]}
    schema = {
        "name": "fn",
        "description": "Desc.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    tool = _schema_to_mcp_tool(schema, card)
    assert "Assumptions:" in tool["description"]


# ======================================================================
#  auto_tool_manifest — mock-registry integration
# ======================================================================


def _make_registry_entry(name: str, category: str, tags: Optional[List[str]] = None):
    """Build a mock RegistryEntry that behaves like the real one."""
    entry = MagicMock()
    entry.category = category
    entry.tags = tags or []
    entry.to_openai_schema.return_value = {
        "name": name,
        "description": f"Auto-generated tool for {name}.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    entry.agent_card.return_value = {
        "assumptions": ["Standard assumption set."],
        "pre_conditions": ["Data must be loaded."],
    }
    return entry


class TestAutoToolManifest:
    def test_empty_registry(self):
        """Empty registry → empty manifest."""
        with patch("statspai.registry._REGISTRY", {}), \
             patch("statspai.registry._ensure_full_registry") as mock_ensure:
            result = auto_tool_manifest(max_tools=500, warn_on_truncate=False)
            assert result == []

    def test_single_tool(self):
        """One eligible entry → one tool in manifest."""
        registry = {
            "sp_did": _make_registry_entry("sp_did", "causal"),
        }
        with patch("statspai.registry._REGISTRY", registry), \
             patch("statspai.registry._ensure_full_registry"):
            result = auto_tool_manifest(max_tools=500, warn_on_truncate=False)
            assert len(result) == 1
            assert result[0]["name"] == "sp_did"

    def test_filter_by_category(self):
        """Only whitelisted categories appear."""
        registry = {
            "causal_fn": _make_registry_entry("causal_fn", "causal"),
            "plot_fn": _make_registry_entry("plot_fn", "plots"),
        }
        with patch("statspai.registry._REGISTRY", registry), \
             patch("statspai.registry._ensure_full_registry"):
            result = auto_tool_manifest(max_tools=500, warn_on_truncate=False)
            names = [t["name"] for t in result]
            assert "causal_fn" in names
            assert "plot_fn" not in names

    def test_filter_by_exclude(self):
        """Entries in DEFAULT_EXCLUDE are skipped."""
        registry = {
            "deepiv": _make_registry_entry("deepiv", "causal"),
            "did": _make_registry_entry("did", "causal"),
        }
        with patch("statspai.registry._REGISTRY", registry), \
             patch("statspai.registry._ensure_full_registry"):
            result = auto_tool_manifest(max_tools=500, warn_on_truncate=False)
            names = [t["name"] for t in result]
            assert "did" in names
            assert "deepiv" not in names  # in DEFAULT_EXCLUDE

    def test_stable_ordering(self):
        """Results sorted alphabetically by name."""
        registry = {
            "z_fn": _make_registry_entry("z_fn", "causal"),
            "a_fn": _make_registry_entry("a_fn", "causal"),
        }
        with patch("statspai.registry._REGISTRY", registry), \
             patch("statspai.registry._ensure_full_registry"):
            result = auto_tool_manifest(max_tools=500, warn_on_truncate=False)
            assert result[0]["name"] == "a_fn"
            assert result[1]["name"] == "z_fn"

    def test_truncation_warning(self):
        """When eligible > max_tools, a RuntimeWarning fires."""
        registry = {
            f"fn_{i:03d}": _make_registry_entry(f"fn_{i:03d}", "causal")
            for i in range(20)
        }
        with patch("statspai.registry._REGISTRY", registry), \
             patch("statspai.registry._ensure_full_registry"):
            with pytest.warns(RuntimeWarning, match="truncated"):
                result = auto_tool_manifest(max_tools=5, warn_on_truncate=True)
            assert len(result) == 5


# ======================================================================
#  merged_tool_manifest
# ======================================================================


class TestMergedToolManifest:
    def test_hand_curated_first(self):
        """Hand-curated entries appear before auto-generated ones, sorted."""
        hand = [
            {"name": "z_hand", "description": "Hand", "input_schema": {}},
            {"name": "a_hand", "description": "Hand A", "input_schema": {}},
        ]
        # One auto entry
        registry = {
            "auto_fn": _make_registry_entry("auto_fn", "causal"),
        }
        with patch("statspai.registry._REGISTRY", registry), \
             patch("statspai.registry._ensure_full_registry"):
            result = merged_tool_manifest(hand)
            assert len(result) == 3
            assert result[0]["name"] == "a_hand"
            assert result[1]["name"] == "auto_fn"
            assert result[2]["name"] == "z_hand"

    def test_hand_curated_wins_collision(self):
        """On name collision, hand-curated entry takes precedence."""
        hand = [
            {"name": "did", "description": "Hand-curated DID", "input_schema": {}},
        ]
        registry = {
            "did": _make_registry_entry("did", "causal"),
        }
        with patch("statspai.registry._REGISTRY", registry), \
             patch("statspai.registry._ensure_full_registry"):
            result = merged_tool_manifest(hand)
            assert len(result) == 1
            assert result[0]["description"] == "Hand-curated DID"

    def test_dedup_skips_duplicate_hand(self):
        """Duplicate names in hand-curated are skipped."""
        hand = [
            {"name": "fn", "description": "First", "input_schema": {}},
            {"name": "fn", "description": "Dup", "input_schema": {}},
        ]
        with patch("statspai.registry._REGISTRY", {}), \
             patch("statspai.registry._ensure_full_registry"):
            result = merged_tool_manifest(hand)
            assert len(result) == 1
            assert result[0]["description"] == "First"

    def test_empty_hand(self):
        """Empty hand-curated list → only auto tools."""
        registry = {
            "fn": _make_registry_entry("fn", "causal"),
        }
        with patch("statspai.registry._REGISTRY", registry), \
             patch("statspai.registry._ensure_full_registry"):
            result = merged_tool_manifest([])
            assert len(result) == 1


# ======================================================================
#  Constants smoke test
# ======================================================================


def test_default_whitelist_is_nonempty():
    assert len(DEFAULT_WHITELIST) >= 5


def test_default_exclude_is_nonempty():
    assert len(DEFAULT_EXCLUDE) >= 5
