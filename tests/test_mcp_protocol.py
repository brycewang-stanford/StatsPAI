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

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from statspai.agent.mcp_server import (
    MCP_PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
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

    def test_preferred_version_is_latest_supported(self):
        # The advertised preferred revision must be the newest entry in the
        # supported set — otherwise a no-version client would be offered a
        # stale revision.
        assert MCP_PROTOCOL_VERSION == SUPPORTED_PROTOCOL_VERSIONS[0]
        assert "2024-11-05" in SUPPORTED_PROTOCOL_VERSIONS  # original revision

    def test_negotiation_echoes_supported_client_version(self):
        # Per spec: when the client requests a revision the server supports,
        # the server MUST reply with that exact revision (not its own latest).
        for requested in SUPPORTED_PROTOCOL_VERSIONS:
            msg = _rpc("initialize", {"protocolVersion": requested})
            assert msg["result"]["protocolVersion"] == requested, (
                f"requested {requested!r} should be echoed verbatim")

    def test_negotiation_falls_back_for_unknown_version(self):
        # An unknown / unsupported revision → server offers its latest.
        msg = _rpc("initialize", {"protocolVersion": "1999-01-01"})
        assert msg["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION

    def test_negotiation_falls_back_for_missing_version(self):
        # No ``protocolVersion`` at all → server offers its latest.
        msg = _rpc("initialize", {})
        assert msg["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION


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
        # clients can load a data file server-side. ``required``
        # membership is conditional — dataless tools (honest_did,
        # sensitivity, audit_result, brief_result, bibtex, …) leave
        # it optional so strict-schema clients don't refuse to dispatch
        # when no path is supplied. The dataless set is auto-derived
        # from the registry (any spec without a required ``data`` param)
        # plus a small hand-curated override list — see
        # ``_dataless_tool_names``.
        from statspai.agent.mcp_server import _dataless_tool_names
        dataless = _dataless_tool_names()
        msg = _rpc("tools/list", {})
        for t in msg["result"]["tools"]:
            schema = t["inputSchema"]
            assert "data_path" in schema["properties"]
            if t["name"] in dataless:
                assert "data_path" not in schema["required"], (
                    f"{t['name']!r} is dataless but its schema still "
                    "marks data_path required")
            else:
                assert "data_path" in schema["required"], (
                    f"{t['name']!r} is data-bound (registry has a "
                    "required ``data`` param) but its schema does not "
                    "mark data_path required")

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
#  Tool annotations + structured output (MCP 2025-03-26 / 2025-06-18)
# ---------------------------------------------------------------------------

class TestAnnotationsAndOutputSchema:

    def test_every_tool_is_annotated_read_only(self):
        # StatsPAI tools are estimators / diagnostics: read-only and
        # closed-world. A client can use readOnlyHint to auto-approve.
        msg = _rpc("tools/list", {})
        tools = msg["result"]["tools"]
        assert tools, "manifest unexpectedly empty"
        for t in tools:
            ann = t.get("annotations")
            assert isinstance(ann, dict), f"{t['name']} missing annotations"
            assert ann.get("readOnlyHint") is True, (
                f"{t['name']} should be readOnlyHint=true")
            assert ann.get("openWorldHint") is False, (
                f"{t['name']} should be openWorldHint=false")

    def test_every_tool_declares_compact_output_schema(self):
        # Every tool advertises a *compact* outputSchema (valid object
        # schema so structuredContent validates), and points to the
        # shared resource for the full field reference — we deliberately
        # do NOT inline the ~2.7 KB documented schema 480x.
        from statspai.agent.mcp_server import RESULT_SCHEMA_URI
        msg = _rpc("tools/list", {})
        for t in msg["result"]["tools"]:
            schema = t.get("outputSchema")
            assert isinstance(schema, dict), (
                f"{t['name']} missing outputSchema")
            assert schema.get("type") == "object"
            assert schema.get("additionalProperties") is True
            assert RESULT_SCHEMA_URI in schema.get("description", ""), (
                f"{t['name']} outputSchema should point to the schema "
                "resource")
            assert "properties" not in schema, (
                f"{t['name']} inlines the full property table — that is "
                "the 1.3 MB duplication we avoid")

    def test_output_schema_not_duplicated_across_manifest(self):
        # Efficiency guard: the per-tool outputSchema must be small. If a
        # future change re-inlines the full documented envelope, the
        # tools/list payload roughly doubles — fail loudly here first.
        from statspai.agent.mcp_server import _RESULT_OUTPUT_SCHEMA_COMPACT
        compact_bytes = len(json.dumps(_RESULT_OUTPUT_SCHEMA_COMPACT))
        assert compact_bytes < 600, (
            f"compact outputSchema is {compact_bytes} bytes — too large to "
            "inline across ~480 tools; keep the full schema in the resource")

    def test_result_schema_resource_is_listed_and_readable(self):
        from statspai.agent.mcp_server import RESULT_SCHEMA_URI
        listing = _rpc("resources/list", {})
        uris = {r["uri"] for r in listing["result"]["resources"]}
        assert RESULT_SCHEMA_URI in uris, "schema resource not enumerated"
        read = _rpc("resources/read", {"uri": RESULT_SCHEMA_URI})
        content = read["result"]["contents"][0]
        assert content["mimeType"] == "application/json"
        schema = json.loads(content["text"])
        assert schema["type"] == "object"
        # The full envelope IS documented here (just not per-tool).
        for key in ("estimate", "std_error", "violations", "error_kind"):
            assert key in schema["properties"]

    def test_output_schema_documents_only_real_fields(self):
        # Guard against documenting invented keys: every documented
        # property must be one the serializer / enrichment can emit.
        from statspai.agent.mcp_server import _RESULT_OUTPUT_SCHEMA
        documented = set(_RESULT_OUTPUT_SCHEMA["properties"])
        real = {
            "estimate", "std_error", "p_value", "conf_low", "conf_high",
            "estimand", "method", "n_obs", "coefficients", "diagnostics",
            "violations", "warnings", "next_steps", "suggested_functions",
            "next_calls", "citations", "narrative", "result_id",
            "result_uri", "data_provenance", "error", "error_kind",
            "remediation",
        }
        assert documented <= real, f"undocumented-key drift: {documented - real}"


class TestStructuredContent:

    @pytest.fixture
    def sample_csv(self, tmp_path):
        rng = np.random.default_rng(2)
        n = 300
        df = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n)})
        path = tmp_path / "sc.csv"
        df.to_csv(path, index=False)
        return path

    def test_tools_call_returns_structured_content(self, sample_csv):
        msg = _rpc("tools/call", {
            "name": "regress",
            "arguments": {"formula": "y ~ x", "data_path": str(sample_csv)},
        })
        result = msg["result"]
        assert "structuredContent" in result
        sc = result["structuredContent"]
        assert isinstance(sc, dict)
        # structuredContent must be the machine twin of the text block.
        text_payload = json.loads(result["content"][0]["text"])
        assert sc == text_payload

    def test_structured_content_is_nan_free(self, sample_csv):
        # Same strict-JSON guarantee as the text block: no NaN/Infinity
        # tokens reach the wire (the raw response string is parsed back).
        raw = handle_request(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "regress",
                       "arguments": {"formula": "y ~ x",
                                     "data_path": str(sample_csv)}},
        }))
        assert "NaN" not in raw and "Infinity" not in raw
        # round-trips through a strict parser without error
        json.loads(raw)

    def test_error_envelope_also_structured(self, tmp_path):
        # A failed call still returns structuredContent (the error object),
        # so agents can branch on error_kind without parsing text.
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        msg = _rpc("tools/call", {
            "name": "regress",
            "arguments": {"formula": "y ~ nonexistent_col",
                           "data_path": str(path)},
        })
        result = msg["result"]
        assert result["isError"] is True
        assert isinstance(result.get("structuredContent"), dict)
        assert "error" in result["structuredContent"]

    def test_local_data_path_provenance_reaches_result_resource(self, sample_csv):
        expected_sha = hashlib.sha256(sample_csv.read_bytes()).hexdigest()
        msg = _rpc("tools/call", {
            "name": "regress",
            "arguments": {
                "formula": "y ~ x",
                "data_path": str(sample_csv),
                "data_columns": ["y", "x"],
                "as_handle": True,
            },
        })
        payload = msg["result"]["structuredContent"]
        prov = payload["data_provenance"]
        assert prov["source"] == str(sample_csv)
        assert prov["format"] == "csv"
        assert prov["columns_requested"] == ["y", "x"]
        assert prov["size_bytes"] == sample_csv.stat().st_size
        assert prov["sha256"] == expected_sha

        rid = payload["result_id"]
        read = _rpc("resources/read", {"uri": f"statspai://result/{rid}"})
        body = json.loads(read["result"]["contents"][0]["text"])
        cached_prov = body["provenance"]["arguments"]["_mcp_data_provenance"]
        assert cached_prov["sha256"] == expected_sha
        assert cached_prov["source"] == str(sample_csv)

    def test_remote_data_provenance_strips_query_tokens(self):
        from statspai.agent._data_loader import data_provenance

        prov = data_provenance(
            "https://data.example.org/panel.csv?token=secret#frag",
            columns=["y"],
            sample_n=10,
        )
        assert prov["source"] == "https://data.example.org/panel.csv"
        assert prov["hash_status"] == "not_hashed_remote"
        assert prov["columns_requested"] == ["y"]
        assert prov["sample_seed"] == 0


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

    def test_cross_validate_tool_keeps_claim_flag_and_data_provenance(
            self, sample_csv):
        msg = _rpc("tools/call", {
            "name": "cross_validate",
            "arguments": {
                "estimand": "ols",
                "formula": "y ~ x",
                "treatment": "x",
                "engines": ["statspai", "definitely_not_an_engine"],
                "data_path": str(sample_csv),
            },
        })
        payload = msg["result"]["structuredContent"]
        assert payload["verdict"] == "INSUFFICIENT"
        assert payload["can_claim_cross_engine_agreement"] is False
        assert payload["engine_status_counts"]["ok"] == 1
        assert payload["engine_status_counts"]["unavailable"] == 1
        assert payload["data_provenance"]["source"] == str(sample_csv)

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
        assert "statspai://parity/track-a-summary" in uris

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

    def test_read_track_a_parity_summary_returns_compact_json(self):
        msg = _rpc(
            "resources/read",
            {"uri": "statspai://parity/track-a-summary"},
        )
        content = msg["result"]["contents"][0]
        assert content["mimeType"] == "application/json"
        summary = json.loads(content["text"])
        assert summary["available"] is True
        assert summary["artifact"].endswith("parity_table_3way.md")
        assert summary["strictness_tiers"]["machine"] >= 1
        assert summary["module_count"] >= 50
        by_module = {m["module"]: m for m in summary["modules"]}
        assert by_module["04_csdid"]["strictness_tier"] == "machine"
        assert "csdid" in by_module["04_csdid"]["stata_command"]
        assert by_module["07_scm"]["strictness_tier"] == "methodological"
        evidence = summary["tool_evidence"]
        assert evidence["callaway_santanna"][0]["module"] == "04_csdid"
        assert evidence["rdrobust"][0]["module"] == "06_rd"
        assert any(row["module"] == "03_hdfe" for row in evidence["fixest"])
        assert evidence["match"][0]["module"] == "11_psm"
        assert "not a live Stata/R execution" in summary["caution"]

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
                  "robustness_followup", "stata_command_workflow",
                  "r_command_workflow", "cross_language_command_check"):
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

    def test_stata_command_workflow_prompt_renders_dispatch_contract(self):
        msg = _rpc("prompts/get", {
            "name": "stata_command_workflow",
            "arguments": {
                "data_path": "/abs/panel.dta",
                "command": "reghdfe y x, absorb(id year) cluster(id)",
            },
        }, request_id=87)
        text = msg["result"]["messages"][0]["content"]["text"]
        assert "from_stata" in text
        assert "/abs/panel.dta" in text
        assert "as_handle=true" in text
        assert "audit_result" in text
        assert "{command}" not in text

    def test_cross_language_command_check_prompt_warns_no_live_external_run(self):
        msg = _rpc("prompts/get", {
            "name": "cross_language_command_check",
            "arguments": {
                "data_path": "/abs/panel.dta",
                "stata_command": "csdid y, ivar(id) time(year) gvar(g)",
                "r_expression": (
                    'att_gt(yname="y", tname="year", '
                    'idname="id", gname="g", data=df)'
                ),
            },
        }, request_id=88)
        text = msg["result"]["messages"][0]["content"]["text"]
        assert "from_stata" in text
        assert "from_r" in text
        assert "convention mismatch" in text
        assert "can_claim_cross_engine_agreement" in text
        assert "statspai://parity/track-a-summary" in text
        assert "tool_evidence" in text
        assert "do not claim live Stata or R execution" in text

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
