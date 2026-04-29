"""Token-budget guards on the unified ``to_dict(detail=...)`` surface.

These tests pin the size of the agent-facing JSON payload so future
changes don't accidentally bloat the LLM tool-result channel.

Char budget targets (tokens ≈ chars / 4):

* ``CausalResult``:
    - ``minimal``  < 600  chars (< 150  tokens)
    - ``standard`` < 4000 chars (< 1000 tokens)
    - ``agent``    < 8000 chars (< 2000 tokens)

* ``EconometricResults``:
    - ``minimal``  < 600   chars
    - ``standard`` < 8000  chars (depends on n_terms)
    - ``agent``    < 12000 chars

Backward-compat is also pinned: ``to_dict()`` with no args == old
``to_dict()``, and ``for_agent()`` == ``to_dict(detail="agent")``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def causal_result():
    """A populated ``CausalResult`` from a real DID estimator."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(200):
        tr = 1 if i < 100 else 0
        for t in (0, 1):
            y = (1.0 + 0.3 * t + 0.5 * tr + 2.0 * tr * t
                 + rng.normal(scale=0.5))
            rows.append({"i": i, "t": t, "treated": tr,
                         "post": t, "y": y})
    df = pd.DataFrame(rows)
    return sp.did(df, y="y", treat="treated", time="t", post="post")


@pytest.fixture(scope="module")
def regress_result():
    """A populated ``EconometricResults`` from sp.regress."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=300),
        "x1": rng.normal(size=300),
        "x2": rng.normal(size=300),
    })
    return sp.regress("y ~ x1 + x2", data=df)


# ---------------------------------------------------------------------------
#  CausalResult.to_dict(detail=...)
# ---------------------------------------------------------------------------

class TestCausalDetailLevels:

    def test_minimal_has_only_core_keys(self, causal_result):
        d = causal_result.to_dict(detail="minimal")
        for k in ("method", "estimand", "estimate", "se", "pvalue",
                  "ci", "alpha", "n_obs", "citation_key"):
            assert k in d, f"minimal missing {k!r}"
        # Minimal must NOT include heavy fields.
        for absent in ("diagnostics", "detail_head", "violations",
                       "next_steps", "suggested_functions", "warnings"):
            assert absent not in d, (
                f"minimal leaked {absent!r}; payload must stay tiny")

    def test_minimal_under_token_budget(self, causal_result):
        s = json.dumps(causal_result.to_dict(detail="minimal"),
                       default=str)
        assert len(s) < 600, (
            f"minimal payload {len(s)} chars exceeds budget; "
            "agents need this < ~150 tokens")

    def test_standard_extends_minimal(self, causal_result):
        m = set(causal_result.to_dict(detail="minimal").keys())
        s = set(causal_result.to_dict(detail="standard").keys())
        assert m.issubset(s)
        assert "diagnostics" in s

    def test_standard_under_token_budget(self, causal_result):
        s = json.dumps(causal_result.to_dict(detail="standard"),
                       default=str)
        assert len(s) < 4000, (
            f"standard payload {len(s)} chars exceeds 1K-token budget")

    def test_agent_extends_standard(self, causal_result):
        s = set(causal_result.to_dict(detail="standard").keys())
        a = set(causal_result.to_dict(detail="agent").keys())
        assert s.issubset(a)
        for k in ("violations", "warnings", "next_steps",
                  "suggested_functions"):
            assert k in a, f"agent missing {k!r}"

    def test_agent_under_token_budget(self, causal_result):
        s = json.dumps(causal_result.to_dict(detail="agent"),
                       default=str)
        assert len(s) < 8000, (
            f"agent payload {len(s)} chars exceeds 2K-token budget")

    def test_default_is_standard(self, causal_result):
        # Backward compat: ``to_dict()`` with no kwargs is the legacy
        # standard shape, byte-for-byte.
        assert (causal_result.to_dict()
                == causal_result.to_dict(detail="standard"))

    def test_for_agent_alias(self, causal_result):
        # Backward compat: ``for_agent()`` is now a thin alias.
        assert (causal_result.for_agent()
                == causal_result.to_dict(detail="agent"))

    def test_invalid_detail_raises(self, causal_result):
        with pytest.raises(ValueError):
            causal_result.to_dict(detail="bogus")

    def test_detail_is_keyword_only(self, causal_result):
        # Closes the trap where a caller could write
        # ``result.to_dict("agent")`` expecting ``detail="agent"`` and
        # silently get ``detail_head="agent"`` instead. Both ``detail``
        # and ``detail_head`` are keyword-only on CausalResult.
        with pytest.raises(TypeError):
            causal_result.to_dict("agent")  # type: ignore[misc]

    def test_all_levels_json_serialisable(self, causal_result):
        for level in ("minimal", "standard", "agent"):
            json.dumps(causal_result.to_dict(detail=level), default=str)

    def test_detail_head_zero_still_works(self, causal_result):
        d = causal_result.to_dict(detail_head=0, detail="standard")
        assert "detail_head" not in d


# ---------------------------------------------------------------------------
#  EconometricResults.to_dict(detail=...)
# ---------------------------------------------------------------------------

class TestEconometricDetailLevels:

    def test_minimal_strips_coefficients(self, regress_result):
        d = regress_result.to_dict(detail="minimal")
        assert "method" in d and "n_obs" in d
        assert "coefficients" not in d
        assert "glance" not in d

    def test_minimal_under_token_budget(self, regress_result):
        s = json.dumps(regress_result.to_dict(detail="minimal"),
                       default=str)
        assert len(s) < 600, (
            f"minimal payload {len(s)} chars exceeds budget")

    def test_minimal_includes_fit_stats_when_available(self, regress_result):
        d = regress_result.to_dict(detail="minimal")
        # OLS glance should expose at least one of these.
        if "fit_stats" in d:
            assert isinstance(d["fit_stats"], dict)
            assert any(
                k in d["fit_stats"]
                for k in ("r_squared", "r.squared", "r2",
                          "f_statistic", "f.statistic", "aic", "AIC")
            )

    def test_standard_has_full_coef_table(self, regress_result):
        d = regress_result.to_dict(detail="standard")
        assert "coefficients" in d
        assert "glance" in d
        assert "diagnostics" in d
        # Each coefficient must round-trip through json.dumps.
        for term, cell in d["coefficients"].items():
            for k in ("estimate", "std_error", "p_value"):
                assert k in cell

    def test_standard_under_token_budget(self, regress_result):
        # ~ 50 chars × n_terms; with 3 terms (Intercept, x1, x2) this
        # should be tight. The 8 000-char ceiling is a regression guard
        # against schema bloat, not a tight upper bound.
        s = json.dumps(regress_result.to_dict(detail="standard"),
                       default=str)
        assert len(s) < 8000, (
            f"standard payload {len(s)} chars exceeds 2K-token budget")

    def test_default_is_standard(self, regress_result):
        assert (regress_result.to_dict()
                == regress_result.to_dict(detail="standard"))

    def test_agent_extends_standard(self, regress_result):
        a = regress_result.to_dict(detail="agent")
        for k in ("violations", "warnings", "next_steps",
                  "suggested_functions"):
            assert k in a, f"agent missing {k!r}"

    def test_agent_under_token_budget(self, regress_result):
        s = json.dumps(regress_result.to_dict(detail="agent"),
                       default=str)
        assert len(s) < 12000, (
            f"agent payload {len(s)} chars exceeds 3K-token budget")

    def test_for_agent_alias(self, regress_result):
        assert (regress_result.for_agent()
                == regress_result.to_dict(detail="agent"))

    def test_invalid_detail_raises(self, regress_result):
        with pytest.raises(ValueError):
            regress_result.to_dict(detail="bogus")

    def test_all_levels_json_serialisable(self, regress_result):
        for level in ("minimal", "standard", "agent"):
            json.dumps(regress_result.to_dict(detail=level), default=str)


# ---------------------------------------------------------------------------
#  MCP-layer integration: _default_serializer reaches detail="agent"
# ---------------------------------------------------------------------------

class TestMcpSerializerUsesAgentLevel:
    """``execute_tool`` should surface ``detail="agent"`` to the LLM
    so violations + next_steps land in the tool result automatically."""

    def test_did_tool_result_contains_agent_extras(self):
        from statspai.agent import execute_tool
        rng = np.random.default_rng(5)
        rows = []
        for i in range(200):
            tr = 1 if i < 100 else 0
            for t in (0, 1):
                y = (1.0 + 0.3 * t + 0.5 * tr + 2.0 * tr * t
                     + rng.normal(scale=0.5))
                rows.append({"i": i, "t": t, "treated": tr,
                             "post": t, "y": y})
        df = pd.DataFrame(rows)
        out = execute_tool(
            "did",
            {"y": "y", "treat": "treated", "time": "t", "post": "post"},
            data=df,
        )
        assert "estimate" in out
        # Agent-level keys must be present.
        for k in ("violations", "warnings", "next_steps",
                  "suggested_functions"):
            assert k in out, (
                f"MCP serializer dropped agent-level field {k!r}")

    def test_regress_tool_result_contains_agent_extras(self):
        from statspai.agent import execute_tool
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "y": rng.normal(size=200),
            "x": rng.normal(size=200),
        })
        out = execute_tool("regress", {"formula": "y ~ x"}, data=df)
        assert "coefficients" in out
        for k in ("violations", "warnings", "next_steps",
                  "suggested_functions"):
            assert k in out, (
                f"MCP serializer dropped agent-level field {k!r}")
