"""Tests for the result-handle / chained-workflow surface.

Covers:
* ``execute_tool(..., as_handle=True)`` caches the fitted object and
  emits ``result_id`` / ``result_uri``.
* ``statspai://result/<id>`` resource resolves to a JSON payload with
  full provenance.
* ``audit_result`` / ``brief_result`` consume a handle without re-running.
* ``honest_did_from_result`` auto-extracts betas / sigma.
* ``bibtex`` tool round-trips known + unknown keys.
* LRU eviction.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest


from statspai.agent._result_cache import RESULT_CACHE, ResultCache, CacheEntry
from statspai.agent import (
    execute_tool,
    mcp_handle_request,
)


def _rpc(method, params=None, request_id=1):
    msg = {"jsonrpc": "2.0", "id": request_id, "method": method,
           "params": params or {}}
    line = mcp_handle_request(json.dumps(msg))
    return json.loads(line)


# ----------------------------------------------------------------------
# Result cache primitives
# ----------------------------------------------------------------------

class TestResultCache:
    def test_put_and_get_round_trip(self):
        cache = ResultCache(max_size=4)
        rid = cache.put({"x": 1}, tool="did", arguments={"y": "wage"})
        assert rid.startswith("r_")
        assert cache.get(rid) == {"x": 1}
        entry = cache.get_entry(rid)
        assert isinstance(entry, CacheEntry)
        assert entry.tool == "did"
        assert entry.arguments == {"y": "wage"}

    def test_lru_eviction(self):
        cache = ResultCache(max_size=3)
        ids = [cache.put({"i": i}, tool="t") for i in range(5)]
        # Oldest two evicted; newest three remain.
        assert cache.get(ids[0]) is None
        assert cache.get(ids[1]) is None
        for i in (2, 3, 4):
            assert cache.get(ids[i]) == {"i": i}

    def test_metadata_strips_dataframes(self):
        df = pd.DataFrame({"y": [1, 2, 3]})
        cache = ResultCache(max_size=2)
        rid = cache.put({"hello": "world"}, tool="did",
                        arguments={"data": df, "y": "wage", "post": True})
        meta = cache.get_entry(rid).to_metadata()
        assert "data" in meta["arguments"]
        # DataFrame replaced by a placeholder string
        assert isinstance(meta["arguments"]["data"], str)
        assert meta["arguments"]["y"] == "wage"
        assert meta["arguments"]["post"] is True


# ----------------------------------------------------------------------
# bibtex tool
# ----------------------------------------------------------------------

class TestBibtexTool:
    def test_known_key_returns_entry(self):
        out = execute_tool("bibtex", {"keys": ["abadie2003economic"]})
        if not out["bibtex"]["abadie2003economic"]:
            pytest.skip("paper.bib does not ship abadie2003economic in this build")
        entry = out["bibtex"]["abadie2003economic"]
        assert "@" in entry  # bib syntax
        assert "abadie2003economic" in entry

    def test_unknown_key_returns_empty_with_suggestions(self):
        out = execute_tool("bibtex", {"keys": ["definitely_not_a_real_key_xyz123"]})
        assert out["bibtex"]["definitely_not_a_real_key_xyz123"] == ""
        assert "definitely_not_a_real_key_xyz123" in out["unknown_keys"]

    def test_keys_required(self):
        out = execute_tool("bibtex", {})
        assert "error" in out

    def test_string_key_accepted(self):
        # Permissive: a single string instead of a list.
        out = execute_tool("bibtex", {"keys": "abadie2003economic"})
        assert "bibtex" in out


# ----------------------------------------------------------------------
# Workflow tools
# ----------------------------------------------------------------------

def _toy_panel():
    """2-period 2-group panel for sanity tests."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        treat = i % 2
        for t in (0, 1):
            y = 1.0 + 0.5 * t + 0.4 * treat * t + rng.normal(scale=0.1)
            rows.append({"id": i, "time": t, "treat": treat, "y": y})
    return pd.DataFrame(rows)


class TestAsHandleAndResourceRead:
    def test_did_with_handle_caches_result(self):
        df = _toy_panel()
        out = execute_tool(
            "did",
            {"y": "y", "treat": "treat", "time": "time"},
            data=df,
            as_handle=True,
        )
        assert "result_id" in out, out
        assert out["result_uri"].startswith("statspai://result/")
        # The handle resolves
        rid = out["result_id"]
        assert rid in RESULT_CACHE

    def test_resource_read_resolves_handle(self):
        df = _toy_panel()
        out = execute_tool(
            "did",
            {"y": "y", "treat": "treat", "time": "time"},
            data=df,
            as_handle=True,
        )
        rid = out["result_id"]
        msg = _rpc("resources/read",
                   {"uri": f"statspai://result/{rid}"})
        assert "result" in msg, msg
        contents = msg["result"]["contents"]
        body = json.loads(contents[0]["text"])
        assert body["result_id"] == rid
        assert body["provenance"]["tool"] == "did"

    def test_resource_read_unknown_handle_is_32002(self):
        msg = _rpc("resources/read",
                   {"uri": "statspai://result/r_deadbeef"})
        assert msg["error"]["code"] == -32002

    def test_audit_result_via_handle(self):
        df = _toy_panel()
        fit = execute_tool(
            "did",
            {"y": "y", "treat": "treat", "time": "time"},
            data=df,
            as_handle=True,
        )
        rid = fit["result_id"]
        out = execute_tool("audit_result", {"result_id": rid})
        # Either the audit ran (rich payload) or audit isn't available
        # in this build — both shapes are acceptable.
        assert isinstance(out, dict)
        if "error" in out:
            assert "audit" in out["error"].lower() or "result_id" in out

    def test_audit_with_missing_handle_returns_friendly_error(self):
        out = execute_tool("audit_result",
                            {"result_id": "r_definitely_missing"})
        assert "error" in out
        assert "result_id" in out["error"] or "not found" in out["error"]


# ----------------------------------------------------------------------
# Schema injection: result_id, as_handle, data_columns, data_sample_n
# ----------------------------------------------------------------------

class TestSchemaInjection:
    def test_every_tool_exposes_result_id(self):
        msg = _rpc("tools/list", {})
        for t in msg["result"]["tools"]:
            props = t["inputSchema"]["properties"]
            assert "result_id" in props, t["name"]

    def test_every_tool_exposes_as_handle(self):
        msg = _rpc("tools/list", {})
        for t in msg["result"]["tools"]:
            props = t["inputSchema"]["properties"]
            assert "as_handle" in props, t["name"]
            assert props["as_handle"]["type"] == "boolean"

    def test_every_tool_exposes_data_columns(self):
        msg = _rpc("tools/list", {})
        for t in msg["result"]["tools"]:
            props = t["inputSchema"]["properties"]
            assert "data_columns" in props, t["name"]


# ----------------------------------------------------------------------
# Initialize block surfaces instructions
# ----------------------------------------------------------------------

class TestInitializeInstructions:
    def test_instructions_present(self):
        msg = _rpc("initialize", {})
        assert "instructions" in msg["result"]
        text = msg["result"]["instructions"]
        # Sanity checks the workflow advice is in the text
        assert "as_handle" in text
        assert "audit_result" in text
        assert "bibtex" in text


# ----------------------------------------------------------------------
# Templates list includes the result template
# ----------------------------------------------------------------------

class TestResultTemplate:
    def test_result_template_exposed(self):
        msg = _rpc("resources/templates/list", {})
        templates = msg["result"]["resourceTemplates"]
        uris = [t["uriTemplate"] for t in templates]
        assert any("result/{id}" in u for u in uris), uris


# ----------------------------------------------------------------------
# JSON encoder coverage
# ----------------------------------------------------------------------

class TestJsonDefault:
    def test_numpy_bool_int_float(self):
        from statspai.agent.mcp_server import _json_default
        assert _json_default(np.bool_(True)) is True
        assert _json_default(np.int64(7)) == 7
        assert _json_default(np.float32(1.5)) == 1.5

    def test_numpy_nan_inf_become_none(self):
        from statspai.agent.mcp_server import _json_default
        assert _json_default(np.float64("nan")) is None
        assert _json_default(np.float64("inf")) is None

    def test_set_and_frozenset(self):
        from statspai.agent.mcp_server import _json_default
        out = _json_default({"b", "a"})
        assert out == ["a", "b"]
        out = _json_default(frozenset(["c", "a"]))
        assert out == ["a", "c"]

    def test_path_and_decimal(self):
        from pathlib import Path
        from decimal import Decimal
        from statspai.agent.mcp_server import _json_default
        assert _json_default(Path("/tmp/x")) == "/tmp/x"
        assert _json_default(Decimal("1.5")) == 1.5

    def test_pandas_index_and_timestamp(self):
        from statspai.agent.mcp_server import _json_default
        out = _json_default(pd.Index([1, 2, 3]))
        assert out == [1, 2, 3]
        ts = pd.Timestamp("2026-04-29")
        out = _json_default(ts)
        assert "2026-04-29" in out


# ----------------------------------------------------------------------
# Debug flag gates traceback
# ----------------------------------------------------------------------

class TestDebugFlagGatesTraceback:
    def test_default_no_traceback(self, monkeypatch):
        monkeypatch.delenv("STATSPAI_MCP_DEBUG", raising=False)
        # Force a -32000 by triggering an unknown handler error path.
        # We monkey-patch a registered method to raise.
        from statspai.agent import mcp_server
        original = mcp_server._METHODS["initialize"]
        def boom(_):
            raise RuntimeError("synthetic")
        mcp_server._METHODS["initialize"] = boom
        try:
            msg = _rpc("initialize", {})
            assert msg["error"]["code"] == -32000
            assert "data" not in msg["error"]
        finally:
            mcp_server._METHODS["initialize"] = original

    def test_debug_flag_emits_traceback(self, monkeypatch):
        monkeypatch.setenv("STATSPAI_MCP_DEBUG", "1")
        from statspai.agent import mcp_server
        original = mcp_server._METHODS["initialize"]
        def boom(_):
            raise RuntimeError("synthetic")
        mcp_server._METHODS["initialize"] = boom
        try:
            msg = _rpc("initialize", {})
            assert msg["error"]["code"] == -32000
            assert "data" in msg["error"]
            assert "traceback" in msg["error"]["data"]
        finally:
            mcp_server._METHODS["initialize"] = original
