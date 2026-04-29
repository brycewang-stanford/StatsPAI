"""End-to-end JSON-RPC 2.0 protocol smoke tests for the StatsPAI MCP server.

These tests run the full ``handle_request`` codepath that
``serve_stdio`` uses, so the wire format actually exercised matches
what Claude Desktop / Cursor / the Anthropic MCP CLI sees.

Coverage:

* ``initialize``                              — protocol handshake
* ``tools/list``                              — manifest enumeration
* ``tools/call`` happy + structured error    — round-trip via subprocess CSV
* ``resources/list``                          — both top-level URIs surface
* ``resources/read``                          — catalog, functions index,
                                                 per-function agent cards
* ``console_script``                          — ``statspai-mcp`` is wired
                                                 in pyproject so ``pip
                                                 install`` exposes it
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from statspai.agent.mcp_server import (
    MCP_PROTOCOL_VERSION,
    SERVER_NAME,
    SERVER_VERSION,
    handle_request,
)


def _rpc(method: str, params: dict, request_id: int = 1) -> dict:
    """Round-trip a JSON-RPC request through ``handle_request``."""
    raw = json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    })
    response = handle_request(raw)
    assert response is not None, f"{method} returned no response"
    return json.loads(response)


# ---------------------------------------------------------------------------
#  initialize handshake
# ---------------------------------------------------------------------------

class TestInitialize:

    def test_handshake_returns_protocol_version(self):
        msg = _rpc("initialize", {})
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 1
        result = msg["result"]
        assert result["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert result["serverInfo"]["name"] == SERVER_NAME
        assert result["serverInfo"]["version"] == SERVER_VERSION

    def test_capabilities_include_tools_and_resources(self):
        msg = _rpc("initialize", {})
        caps = msg["result"]["capabilities"]
        assert "tools" in caps
        assert "resources" in caps


# ---------------------------------------------------------------------------
#  tools/list — manifest enumeration
# ---------------------------------------------------------------------------

class TestToolsList:

    def test_manifest_has_curated_tools(self):
        msg = _rpc("tools/list", {})
        tools = msg["result"]["tools"]
        names = {t["name"] for t in tools}
        for canonical in ("regress", "did", "rdrobust", "ivreg",
                          "callaway_santanna", "ebalance"):
            assert canonical in names

    def test_every_tool_has_data_path_param(self):
        # Every tool must EXPOSE the ``data_path`` property so MCP
        # clients can load a CSV server-side. ``required`` membership
        # is conditional — dataless tools (honest_did, sensitivity)
        # leave it optional so strict-schema clients don't refuse to
        # dispatch when no CSV is supplied.
        from statspai.agent.mcp_server import _DATALESS_TOOLS
        msg = _rpc("tools/list", {})
        for t in msg["result"]["tools"]:
            schema = t["inputSchema"]
            assert "data_path" in schema["properties"]
            if t["name"] in _DATALESS_TOOLS:
                assert "data_path" not in schema["required"], (
                    f"{t['name']!r} is dataless but its schema still "
                    "marks data_path required")
            else:
                assert "data_path" in schema["required"]

    def test_dataless_tools_omit_data_path_required(self):
        # Regression guard for the 1.9.1 schema fix: honest_did and
        # sensitivity must dispatch through MCP without a CSV path.
        msg = _rpc("tools/list", {})
        by_name = {t["name"]: t for t in msg["result"]["tools"]}
        for n in ("honest_did", "sensitivity"):
            if n not in by_name:
                continue  # not in the merged manifest in this env
            assert "data_path" not in (
                by_name[n]["inputSchema"]["required"]), (
                    f"{n!r} schema must NOT require data_path")

    def test_tools_call_missing_name_returns_invalid_params(self):
        # Regression guard for the 1.9.1 typed-error fix: a missing
        # `name` field should return -32602 (invalid params), not
        # -32000 (generic server fallback).
        msg = _rpc("tools/call", {"arguments": {}}, request_id=99)
        assert "error" in msg
        assert msg["error"]["code"] == -32602


# ---------------------------------------------------------------------------
#  tools/call — happy + structured error round-trip
# ---------------------------------------------------------------------------

class TestToolsCall:

    @pytest.fixture
    def sample_csv(self, tmp_path):
        rng = np.random.default_rng(1)
        n = 300
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
        })
        path = tmp_path / "sample.csv"
        df.to_csv(path, index=False)
        return path

    def test_happy_path_runs_estimator_and_returns_text_block(
            self, sample_csv):
        msg = _rpc("tools/call", {
            "name": "regress",
            "arguments": {
                "formula": "y ~ x",
                "data_path": str(sample_csv),
            },
        })
        result = msg["result"]
        assert result["isError"] is False
        text = result["content"][0]["text"]
        payload = json.loads(text)
        # Agent-detail-level keys present: violations + next_steps.
        assert "coefficients" in payload
        assert "next_steps" in payload

    def test_detail_minimal_returns_smaller_payload(self, sample_csv):
        # Verify the MCP-level detail control yields a strictly smaller
        # payload than the agent default — agents trade richness for
        # token cost on a per-call basis.
        msg_min = _rpc("tools/call", {
            "name": "regress",
            "arguments": {"formula": "y ~ x",
                           "data_path": str(sample_csv),
                           "detail": "minimal"},
        }, request_id=11)
        msg_agent = _rpc("tools/call", {
            "name": "regress",
            "arguments": {"formula": "y ~ x",
                           "data_path": str(sample_csv),
                           "detail": "agent"},
        }, request_id=12)
        text_min = msg_min["result"]["content"][0]["text"]
        text_agent = msg_agent["result"]["content"][0]["text"]
        assert len(text_min) < len(text_agent), (
            f"minimal payload ({len(text_min)} chars) should be "
            f"smaller than agent payload ({len(text_agent)} chars)"
        )
        # minimal must NOT carry the agent-level fields.
        payload_min = json.loads(text_min)
        for k in ("violations", "next_steps", "suggested_functions"):
            assert k not in payload_min, (
                f"minimal leaked agent-level field {k!r}")

    def test_detail_standard_excludes_agent_extras(self, sample_csv):
        msg = _rpc("tools/call", {
            "name": "regress",
            "arguments": {"formula": "y ~ x",
                           "data_path": str(sample_csv),
                           "detail": "standard"},
        }, request_id=13)
        payload = json.loads(msg["result"]["content"][0]["text"])
        assert "coefficients" in payload  # standard has coef table
        for k in ("violations", "next_steps", "suggested_functions"):
            assert k not in payload

    def test_detail_default_is_agent(self, sample_csv):
        # Backward compat: omitting ``detail`` falls back to the
        # agent-rich payload, same shape as Phase 1.
        msg = _rpc("tools/call", {
            "name": "regress",
            "arguments": {"formula": "y ~ x",
                           "data_path": str(sample_csv)},
        }, request_id=14)
        payload = json.loads(msg["result"]["content"][0]["text"])
        for k in ("violations", "next_steps", "suggested_functions"):
            assert k in payload, (
                f"default tools/call must keep agent-level field {k!r}")

    def test_invalid_detail_returns_invalid_params(self, sample_csv):
        msg = _rpc("tools/call", {
            "name": "regress",
            "arguments": {"formula": "y ~ x",
                           "data_path": str(sample_csv),
                           "detail": "verbose"},  # not a valid level
        }, request_id=15)
        assert "error" in msg
        assert msg["error"]["code"] == -32602

    def test_detail_in_tool_input_schema(self):
        msg = _rpc("tools/list", {}, request_id=16)
        for tool in msg["result"]["tools"]:
            schema = tool["inputSchema"]
            assert "detail" in schema["properties"], (
                f"{tool['name']} schema missing the detail enum")
            d = schema["properties"]["detail"]
            assert d["enum"] == ["minimal", "standard", "agent"]
            assert d["default"] == "agent"
            # detail is OPTIONAL — agents shouldn't be forced to
            # supply it (the default preserves Phase 1 behaviour).
            assert "detail" not in schema["required"]

    def test_structured_error_routes_through_isError(self, tmp_path):
        # IV under-identification → MethodIncompatibility.
        rng = np.random.default_rng(0)
        n = 200
        z = rng.normal(size=n)
        d1 = 0.5 * z + rng.normal(size=n)
        d2 = rng.normal(size=n)
        y = d1 + d2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d1": d1, "d2": d2, "z": z})
        csv = tmp_path / "iv.csv"
        df.to_csv(csv, index=False)
        msg = _rpc("tools/call", {
            "name": "ivreg",
            "arguments": {
                "formula": "y ~ (d1 + d2 ~ z)",
                "data_path": str(csv),
            },
        })
        result = msg["result"]
        assert result["isError"] is True
        payload = json.loads(result["content"][0]["text"])
        assert payload["error_kind"] == "method_incompatibility"


# ---------------------------------------------------------------------------
#  resources/list + resources/read
# ---------------------------------------------------------------------------

class TestResources:

    def test_list_returns_catalog_and_functions_index(self):
        msg = _rpc("resources/list", {})
        uris = {r["uri"] for r in msg["result"]["resources"]}
        assert "statspai://catalog" in uris
        assert "statspai://functions" in uris

    def test_read_catalog_returns_markdown(self):
        msg = _rpc("resources/read",
                   {"uri": "statspai://catalog"})
        content = msg["result"]["contents"][0]
        assert content["mimeType"] == "text/markdown"
        assert "StatsPAI tool catalog" in content["text"]
        # Must document the per-function URI pattern so agents discover it.
        assert "statspai://function/" in content["text"]

    def test_read_functions_index_returns_json_array(self):
        msg = _rpc("resources/read",
                   {"uri": "statspai://functions"})
        content = msg["result"]["contents"][0]
        assert content["mimeType"] == "application/json"
        index = json.loads(content["text"])
        assert isinstance(index, list)
        assert len(index) >= 50
        for entry in index[:5]:
            assert "name" in entry
            assert "description" in entry

    def test_read_per_function_returns_agent_card(self):
        msg = _rpc("resources/read",
                   {"uri": "statspai://function/regress"})
        content = msg["result"]["contents"][0]
        assert content["mimeType"] == "application/json"
        card = json.loads(content["text"])
        # Rich agent-card fields must round-trip.
        assert card["name"] == "regress"
        assert card["description"]
        assert isinstance(card["assumptions"], list)
        assert len(card["assumptions"]) >= 1
        assert isinstance(card["failure_modes"], list)
        assert isinstance(card["alternatives"], list)
        assert "signature" in card

    def test_read_unknown_function_returns_error(self):
        msg = _rpc("resources/read",
                   {"uri": "statspai://function/__not_a_real_tool__"})
        # Per JSON-RPC spec, server errors come back as ``error``, not
        # ``result``.
        assert "error" in msg
        # MCP 2024-11-05 reserves -32002 for resource-not-found so
        # clients can prompt "did you mean" without regexing the
        # message.
        assert msg["error"]["code"] == -32002
        assert "Unknown StatsPAI tool" in msg["error"]["message"]

    def test_read_function_uri_with_embedded_slash_is_invalid_params(self):
        # Embedded slash → malformed URI under the {name} template.
        # Should return -32602 (invalid params), NOT -32002 (which
        # would mislead clients into a "did you mean" retry loop).
        msg = _rpc("resources/read",
                   {"uri": "statspai://function/foo/bar"})
        assert "error" in msg
        assert msg["error"]["code"] == -32602

    def test_read_function_uri_empty_name_is_invalid_params(self):
        msg = _rpc("resources/read", {"uri": "statspai://function/"})
        assert "error" in msg
        assert msg["error"]["code"] == -32602

    def test_read_unknown_uri_scheme_errors(self):
        msg = _rpc("resources/read",
                   {"uri": "https://example.com/no"})
        assert "error" in msg
        assert msg["error"]["code"] == -32002

    def test_templates_list_exposes_per_function_uri_pattern(self):
        msg = _rpc("resources/templates/list", {})
        templates = msg["result"]["resourceTemplates"]
        uris = {t["uriTemplate"] for t in templates}
        assert "statspai://function/{name}" in uris
        # Each template carries the standard MCP fields.
        for t in templates:
            assert "name" in t
            assert "description" in t
            assert "mimeType" in t


# ---------------------------------------------------------------------------
#  tools/call: unknown tool name
# ---------------------------------------------------------------------------

class TestToolsCallUnknownName:

    def test_unknown_tool_returns_structured_error_dict(self, tmp_path):
        # The MCP server forwards the args to ``execute_tool``, which
        # returns ``{"error": ..., "available_tools": [...]}`` for an
        # unknown name. That dict surfaces as the ``content[0].text``
        # of the tools/call result with ``isError=true``.
        csv = tmp_path / "empty.csv"
        pd.DataFrame({"y": [1.0]}).to_csv(csv, index=False)
        msg = _rpc("tools/call", {
            "name": "__not_a_real_tool__",
            "arguments": {"data_path": str(csv)},
        })
        result = msg["result"]
        assert result["isError"] is True
        payload = json.loads(result["content"][0]["text"])
        assert "error" in payload
        assert "available_tools" in payload


# ---------------------------------------------------------------------------
#  Notifications (no id) must not produce a response per JSON-RPC 2.0
# ---------------------------------------------------------------------------

class TestPrompts:
    """``prompts/list`` + ``prompts/get`` workflow templates."""

    def test_initialize_advertises_prompts_capability(self):
        msg = _rpc("initialize", {}, request_id=80)
        caps = msg["result"]["capabilities"]
        assert "prompts" in caps

    def test_list_returns_curated_set(self):
        msg = _rpc("prompts/list", {}, request_id=81)
        prompts = msg["result"]["prompts"]
        names = {p["name"] for p in prompts}
        # At least the 3 curated workflows must be present.
        for n in ("audit_did_result", "design_then_estimate",
                  "robustness_followup"):
            assert n in names, f"prompt {n!r} missing from prompts/list"

    def test_each_prompt_has_required_metadata(self):
        msg = _rpc("prompts/list", {}, request_id=82)
        for p in msg["result"]["prompts"]:
            assert "name" in p and "description" in p
            assert isinstance(p.get("arguments"), list)
            for arg in p["arguments"]:
                assert "name" in arg and "description" in arg

    def test_get_renders_template_with_arguments(self):
        msg = _rpc("prompts/get", {
            "name": "audit_did_result",
            "arguments": {
                "data_path": "/abs/path.csv",
                "y": "outcome",
                "treat": "treated",
                "time": "year",
            },
        }, request_id=83)
        result = msg["result"]
        assert "description" in result
        text = result["messages"][0]["content"]["text"]
        # All four arguments should be substituted into the template.
        assert "/abs/path.csv" in text
        assert "y=outcome" in text
        assert "treat=treated" in text
        assert "time=year" in text
        # Literal placeholders must NOT leak through.
        assert "{data_path}" not in text
        assert "{y}" not in text

    def test_get_missing_required_argument_returns_invalid_params(self):
        msg = _rpc("prompts/get", {
            "name": "audit_did_result",
            "arguments": {"data_path": "/x.csv"},  # missing y/treat/time
        }, request_id=84)
        assert "error" in msg
        assert msg["error"]["code"] == -32602

    def test_get_with_brace_in_user_value_does_not_substitute_again(self):
        # Defense-in-depth: a user value containing literal ``{name}``
        # must NOT be re-rendered against the template's placeholders.
        # Regression guard for the brace-escape pass in _handle_prompts_get.
        msg = _rpc("prompts/get", {
            "name": "audit_did_result",
            "arguments": {
                "data_path": "/data/runs/{y}/panel.csv",  # literal {y}
                "y": "earnings",
                "treat": "treated",
                "time": "year",
            },
        }, request_id=86)
        assert "result" in msg
        text = msg["result"]["messages"][0]["content"]["text"]
        # The literal ``{y}`` inside data_path must survive verbatim.
        assert "/data/runs/{y}/panel.csv" in text
        # And the actual {y} placeholder ELSEWHERE in the template
        # must still be substituted with "earnings".
        assert "y=earnings" in text

    def test_get_unknown_prompt_returns_resource_not_found(self):
        msg = _rpc("prompts/get", {
            "name": "totally_made_up_prompt",
        }, request_id=85)
        assert "error" in msg
        assert msg["error"]["code"] == -32002


class TestNotifications:

    def test_notification_returns_none(self):
        # No "id" field → notification, server must stay silent.
        raw = json.dumps({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {},
        })
        assert handle_request(raw) is None

    def test_notifications_initialized_silenced_even_with_id(self):
        # Claude Desktop / Cursor send ``notifications/initialized``
        # right after the handshake. Per MCP spec the server MUST NOT
        # respond — even when the client erroneously included an
        # ``id`` field, the ``notifications/`` prefix should suppress
        # any response (which would otherwise be -32601 noise on every
        # session).
        raw = json.dumps({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "notifications/initialized",
            "params": {},
        })
        assert handle_request(raw) is None

    def test_notifications_cancelled_silenced(self):
        raw = json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 1, "reason": "user"},
        })
        assert handle_request(raw) is None


# ---------------------------------------------------------------------------
#  Console script wiring (pip install statspai → ``statspai-mcp`` on PATH)
# ---------------------------------------------------------------------------

class TestConsoleScript:

    def test_pyproject_declares_statspai_mcp_entry_point(self):
        # Read pyproject.toml directly and confirm the entry point is
        # registered. Tomllib lives in stdlib from 3.11; fall back to
        # toml for 3.9 / 3.10.
        try:
            import tomllib  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover — Py 3.9/3.10
            import tomli as tomllib  # type: ignore

        # The repo root is two parents up from this test file.
        pyproject = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "pyproject.toml",
        )
        with open(pyproject, "rb") as f:
            cfg = tomllib.load(f)
        scripts = cfg.get("project", {}).get("scripts", {})
        assert scripts.get("statspai-mcp") == (
            "statspai.agent.mcp_server:main"
        ), (
            "statspai-mcp console script not wired in pyproject.toml; "
            "agents won't see it on PATH after pip install."
        )

    def test_main_callable_exists(self):
        from statspai.agent import mcp_server
        assert callable(getattr(mcp_server, "main", None))
