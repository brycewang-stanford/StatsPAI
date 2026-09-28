"""MCP server hardening: errors-as-results, handles, budgets, security.

Covers the 2026-09-28 audit items for ``statspai.agent.mcp_server``:

* tool-execution failures are ``isError`` results with structured
  payloads; protocol faults stay JSON-RPC errors; ``ping`` works;
* stale ``result_id`` is an error on every dispatch path (curated,
  auto-registered, workflow) and is threaded through the auto path;
* output byte budget + ``truncated`` ledger, ``_nonfinite`` paths,
  compact text block, ``replay`` strings;
* data roots allowlist, remote opt-in + byte cap, ``openWorldHint``,
  schema-derived file writers, ``transform_data`` expression allowlist;
* data-cache byte budget; unified default profile.

Cancellation / concurrency over the stdio loop lives in
``tests/test_mcp_hardening_stdio.py``.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import mcp_server
from statspai.agent.mcp_server import handle_request


def _rpc(method, params=None, request_id=1):
    raw = json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
    )
    out = handle_request(raw)
    assert out is not None
    return json.loads(out)


def _call(name, **arguments):
    return _rpc("tools/call", {"name": name, "arguments": arguments})


@pytest.fixture
def ols_csv(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200)})
    df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(size=200)
    path = tmp_path / "ols.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def fresh_tools():
    mcp_server._clear_mcp_caches()
    yield
    mcp_server._clear_mcp_caches()


# ---------------------------------------------------------------------------
# Protocol vs tool errors
# ---------------------------------------------------------------------------


class TestProtocolErrors:
    def test_ping_returns_empty_result(self):
        assert _rpc("ping", {}, request_id=7) == {
            "jsonrpc": "2.0",
            "id": 7,
            "result": {},
        }

    def test_unknown_method(self):
        err = _rpc("no/such/method")["error"]
        assert err["code"] == -32601
        assert "no/such/method" in err["message"]

    def test_params_not_object(self):
        raw = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1]}
        )
        err = json.loads(handle_request(raw))["error"]
        assert err["code"] == -32602
        assert "`params` must be a JSON object" in err["message"]

    def test_arguments_not_object(self):
        msg = _rpc("tools/call", {"name": "regress", "arguments": [1, 2]})
        assert msg["error"]["code"] == -32602
        assert "`arguments` must be a JSON object" in msg["error"]["message"]

    def test_request_not_object(self):
        err = json.loads(handle_request("[1, 2]"))["error"]
        assert err["code"] == -32600
        assert "expected a JSON object" in err["message"]

    def test_unknown_tool_is_protocol_error(self):
        msg = _call("definitely_not_a_tool_xyz")
        assert msg["error"]["code"] == -32602
        assert "definitely_not_a_tool_xyz" in msg["error"]["message"]

    def test_call_function_unknown_inner_is_tool_error(self):
        msg = _call("call_function", function="definitely_not_a_tool_xyz")
        assert msg["result"]["isError"] is True
        sc = msg["result"]["structuredContent"]
        assert sc["error_kind"] == "unknown_tool"
        assert "definitely_not_a_tool_xyz" in sc["error"]


class TestToolErrorsAreResults:
    def test_missing_file_is_structured_is_error(self, tmp_path):
        msg = _call("regress", formula="y ~ x", data_path=str(tmp_path / "nope.csv"))
        res = msg["result"]
        assert res["isError"] is True
        sc = res["structuredContent"]
        assert sc["error_kind"] == "file_not_found"
        assert "nope.csv" in sc["message"]
        # The text block is the compact JSON twin of structuredContent.
        assert json.loads(res["content"][0]["text"]) == sc

    def test_relative_path_is_data_load_error(self):
        msg = _call("regress", formula="y ~ x", data_path="relative.csv")
        assert msg["result"]["structuredContent"]["error_kind"] == "data_load_error"

    def test_bad_budget_argument(self, ols_csv):
        msg = _call(
            "regress", formula="y ~ x", data_path=str(ols_csv), max_output_bytes=-5
        )
        assert msg["result"]["structuredContent"]["error_kind"] == "invalid_arguments"

    def test_escaped_exception_is_internal_error(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("synthetic dispatch bug")

        monkeypatch.setattr(mcp_server, "execute_tool", boom)
        monkeypatch.delenv("STATSPAI_MCP_DEBUG", raising=False)
        res = _call("regress", formula="y ~ x")["result"]
        assert res["isError"] is True
        sc = res["structuredContent"]
        assert sc["error_kind"] == "internal_error"
        assert "synthetic dispatch bug" in sc["message"]
        assert "traceback" not in sc


# ---------------------------------------------------------------------------
# result_id
# ---------------------------------------------------------------------------


class TestResultHandles:
    def test_stale_result_id_mcp(self):
        res = _call("audit_result", result_id="r_deadbeef")["result"]
        assert res["isError"] is True
        sc = res["structuredContent"]
        assert sc["error_kind"] == "missing_result_handle"
        assert sc["miss_reason"] == "unknown"
        assert "as_handle" in sc["hint"]

    def test_stale_result_id_auto_tool_mcp(self):
        res = _call("evalue_from_result", result_id="r_deadbeef")["result"]
        assert res["structuredContent"]["error_kind"] == "missing_result_handle"

    def test_stale_result_id_python_paths(self):
        from statspai.agent.tools import execute_tool

        for tool in ("honest_did", "evalue_from_result", "audit_result"):
            out = execute_tool(tool, {}, result_id="r_deadbeef")
            assert out["error_kind"] == "missing_result_handle", (tool, out)

    def test_auto_path_receives_result(self, ols_csv):
        fit = _call("regress", formula="y ~ x", data_path=str(ols_csv), as_handle=True)
        rid = fit["result"]["structuredContent"]["result_id"]
        out = _call("evalue_from_result", result_id=rid)["result"]["structuredContent"]
        # The fitted OLS object reached the function (which then explains
        # that it needs a single causal estimate) — before the fix the
        # handle was dropped and the call failed for a missing argument.
        assert "EconometricResults" in out.get("error", ""), out

    def test_unconsumed_result_id_is_reported(self, ols_csv):
        fit = _call("regress", formula="y ~ x", data_path=str(ols_csv), as_handle=True)
        rid = fit["result"]["structuredContent"]["result_id"]
        out = _call("adjust_pvalues", pvalues=[0.01, 0.04], result_id=rid)
        assert "result_id" in out["result"]["structuredContent"]["_unsupported_args"]


# ---------------------------------------------------------------------------
# Output shaping
# ---------------------------------------------------------------------------


def _fake_tool(payload):
    def _tool(
        name, arguments, *, data=None, detail="agent", result_id=None, as_handle=False
    ):
        return payload() if callable(payload) else payload

    return _tool


class TestOutputShaping:
    def test_text_block_is_compact(self, ols_csv):
        res = _call("regress", formula="y ~ x", data_path=str(ols_csv))["result"]
        text = res["content"][0]["text"]
        assert "\n" not in text and ": " not in text[:200]
        assert json.loads(text) == res["structuredContent"]

    def test_budget_truncates_largest_list_keeps_headline(self, monkeypatch):
        big = {
            "estimate": 1.25,
            "std_error": 0.5,
            "ci": [0.2, 2.3],
            "p_value": 0.01,
            "weights": list(range(50_000)),
            "draws": [0.5] * 100,
        }
        monkeypatch.setattr(mcp_server, "execute_tool", _fake_tool(lambda: dict(big)))
        res = _call("regress", formula="y ~ x", max_output_bytes=4096)["result"]
        sc = res["structuredContent"]
        assert len(res["content"][0]["text"]) <= 4096
        assert sc["estimate"] == 1.25 and sc["std_error"] == 0.5
        assert sc["ci"] == [0.2, 2.3] and sc["p_value"] == 0.01
        rec = {r["path"]: r for r in sc["truncated"]}
        assert rec["/weights"]["total"] == 50_000
        assert rec["/weights"]["shown"] == len(sc["weights"]) < 50_000

    def test_budget_default_and_disable(self, monkeypatch):
        big = {
            "estimate": 1.0,
            "rows": [{"a": i, "b": "x" * 20} for i in range(20_000)],
        }
        monkeypatch.setattr(
            mcp_server, "execute_tool", _fake_tool(lambda: json.loads(json.dumps(big)))
        )
        sc = _call("regress", formula="y ~ x")["result"]["structuredContent"]
        assert "truncated" in sc
        assert len(json.dumps(sc, separators=(",", ":"))) <= 256 * 1024
        monkeypatch.setenv("STATSPAI_MCP_MAX_OUTPUT_BYTES", "0")
        sc = _call("regress", formula="y ~ x")["result"]["structuredContent"]
        assert "truncated" not in sc and len(sc["rows"]) == 20_000
        sc = _call("regress", formula="y ~ x", max_output_bytes=2000)["result"][
            "structuredContent"
        ]
        assert "truncated" in sc

    def test_budget_unit_nested_and_strings(self):
        from statspai.agent._output_budget import apply_budget, json_size

        obj = {
            "estimate": 2.0,
            "table": {f"k{i}": {"v": i} for i in range(500)},
            "note": "z" * 20_000,
        }
        out, rec = apply_budget(obj, 3000)
        assert json_size(out) <= 3000
        paths = {r["path"] for r in rec}
        assert {"/table", "/note"} <= paths
        assert out["estimate"] == 2.0

    def test_nonfinite_paths(self, monkeypatch):
        payload = {
            "estimate": 1.0,
            "std_error": float("inf"),
            "diagnostics": {"f/stat": float("nan")},
            "draws": np.array([1.0, -np.inf]),
            "frame": pd.DataFrame({"a": [np.nan, 1.0]}),
        }
        monkeypatch.setattr(
            mcp_server, "execute_tool", _fake_tool(lambda: dict(payload))
        )
        sc = _call("regress", formula="y ~ x")["result"]["structuredContent"]
        assert sc["std_error"] is None
        got = {d["path"]: d["value"] for d in sc["_nonfinite"]}
        assert got == {
            "/std_error": "Infinity",
            "/diagnostics/f~1stat": "NaN",
            "/draws/1": "-Infinity",
            "/frame/a/0": "NaN",
        }

    def test_no_nonfinite_key_when_clean(self, ols_csv):
        sc = _call("regress", formula="y ~ x", data_path=str(ols_csv))["result"][
            "structuredContent"
        ]
        assert "_nonfinite" not in sc

    def test_replay_string(self, ols_csv):
        sc = _call("regress", formula="y ~ x", data_path=str(ols_csv), as_handle=True)[
            "result"
        ]["structuredContent"]
        assert sc["replay"].startswith("sp.regress(data=data, formula='y ~ x')")
        assert str(ols_csv) in sc["replay"]
        read = _rpc("resources/read", {"uri": f"statspai://result/{sc['result_id']}"})
        body = json.loads(read["result"]["contents"][0]["text"])
        assert body["provenance"]["replay"] == sc["replay"]

    def test_replay_reproduces_the_fit(self, ols_csv):
        import statspai as sp

        sc = _call("regress", formula="y ~ x", data_path=str(ols_csv))["result"][
            "structuredContent"
        ]
        call = sc["replay"].split("  #", 1)[0]
        data = pd.read_csv(ols_csv)
        res = eval(call, {"sp": sp, "data": data})  # noqa: S307 — test of replay
        assert float(res.params["x"]) == pytest.approx(
            sc["coefficients"]["x"]["estimate"], rel=1e-12
        )


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestDataRoots:
    def test_outside_root_refused_inside_allowed(self, tmp_path, ols_csv, monkeypatch):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        inside = allowed / "d.csv"
        inside.write_text(ols_csv.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setenv("STATSPAI_MCP_DATA_ROOTS", str(allowed))
        res = _call("regress", formula="y ~ x", data_path=str(ols_csv))["result"]
        assert res["structuredContent"]["error_kind"] == "path_not_allowed"
        res = _call("regress", formula="y ~ x", data_path=str(inside))["result"]
        assert res["isError"] is False
        local = sp.regress("y ~ x", data=pd.read_csv(ols_csv))
        assert res["structuredContent"]["coefficients"]["x"][
            "estimate"
        ] == pytest.approx(float(local.params["x"]), rel=1e-12)
        res = _call("regress", formula="y ~ x", data_path=inside.as_uri())["result"]
        assert res["isError"] is False
        res = _call("regress", formula="y ~ x", data_path=ols_csv.as_uri())["result"]
        assert res["structuredContent"]["error_kind"] == "path_not_allowed"

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_symlink_escape_refused(self, tmp_path, ols_csv, monkeypatch):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        link = allowed / "link.csv"
        link.symlink_to(ols_csv)
        monkeypatch.setenv("STATSPAI_MCP_DATA_ROOTS", str(allowed))
        res = _call("regress", formula="y ~ x", data_path=str(link))["result"]
        assert res["structuredContent"]["error_kind"] == "path_not_allowed"

    def test_load_data_also_checked(self, tmp_path, ols_csv, monkeypatch):
        monkeypatch.setenv("STATSPAI_MCP_DATA_ROOTS", str(tmp_path / "elsewhere"))
        res = _call("load_data", data_path=str(ols_csv))["result"]
        assert res["structuredContent"]["error_kind"] == "path_not_allowed"


class _FakeResp(io.BytesIO):
    def __init__(self, body: bytes, length=True):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))} if length else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRemote:
    URL = "https://data.example.org/panel.csv?token=secret"

    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("STATSPAI_MCP_ALLOW_REMOTE", raising=False)
        sc = _call("regress", formula="y ~ x", data_path=self.URL)["result"][
            "structuredContent"
        ]
        assert sc["error_kind"] == "remote_disabled"
        assert "STATSPAI_MCP_ALLOW_REMOTE" in sc["hint"]
        assert "secret" not in json.dumps(sc["message"])

    def test_enabled_loads_under_cap(self, monkeypatch, ols_csv):
        import urllib.request

        body = ols_csv.read_bytes()
        monkeypatch.setenv("STATSPAI_MCP_ALLOW_REMOTE", "1")
        monkeypatch.setattr(
            urllib.request, "urlopen", lambda url, timeout=0: _FakeResp(body)
        )
        res = _call("regress", formula="y ~ x", data_path=self.URL)["result"]
        assert res["isError"] is False
        sc = res["structuredContent"]
        assert sc["data_provenance"]["source_type"] == "remote"
        local = sp.regress("y ~ x", data=pd.read_csv(ols_csv))
        assert sc["coefficients"]["x"]["estimate"] == pytest.approx(
            float(local.params["x"]), rel=1e-12
        )

    @pytest.mark.parametrize("with_length", [True, False])
    def test_remote_byte_cap(self, monkeypatch, ols_csv, with_length):
        import urllib.request

        body = ols_csv.read_bytes()
        monkeypatch.setenv("STATSPAI_MCP_ALLOW_REMOTE", "1")
        monkeypatch.setenv("STATSPAI_MCP_MAX_DATA_BYTES", "100")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda url, timeout=0: _FakeResp(body, length=with_length),
        )
        sc = _call("regress", formula="y ~ x", data_path=self.URL)["result"][
            "structuredContent"
        ]
        assert sc["error_kind"] == "data_load_error"
        assert "STATSPAI_MCP_MAX_DATA_BYTES" in sc["message"]


class TestAnnotations:
    def test_open_world_follows_remote_opt_in(self, monkeypatch, fresh_tools):
        monkeypatch.delenv("STATSPAI_MCP_ALLOW_REMOTE", raising=False)
        tools = _rpc("tools/list")["result"]["tools"]
        assert all(t["annotations"]["openWorldHint"] is False for t in tools)
        monkeypatch.setenv("STATSPAI_MCP_ALLOW_REMOTE", "1")
        mcp_server._clear_mcp_caches()
        tools = _rpc("tools/list")["result"]["tools"]
        assert all(t["annotations"]["openWorldHint"] is True for t in tools)

    def test_schema_derived_file_writers(self, fresh_tools):
        tools = {t["name"]: t for t in mcp_server._build_mcp_tools("full")}
        assert tools["synth_report"]["annotations"]["readOnlyHint"] is False
        assert tools["synth_report_to_file"]["annotations"]["readOnlyHint"] is False
        assert tools["did"]["annotations"]["readOnlyHint"] is True
        assert mcp_server._is_file_writing_tool(
            "x", {"properties": {"output_path": {}}}
        )
        assert not mcp_server._is_file_writing_tool("x", {"properties": {"y": {}}})

    def test_max_output_bytes_in_schema(self, fresh_tools):
        tools = _rpc("tools/list")["result"]["tools"]
        assert all("max_output_bytes" in t["inputSchema"]["properties"] for t in tools)


class TestTransformExpressions:
    def test_unsafe_expression_rejected(self, ols_csv):
        did = _call("load_data", data_path=str(ols_csv))["result"]["structuredContent"][
            "data_id"
        ]
        for expr in ("x.__class__", "x.values.sum() > 0", "x[0] > 1", "@a > 1"):
            sc = _call(
                "transform_data",
                data_id=did,
                operations=[{"op": "query", "expr": expr}],
            )["result"]["structuredContent"]
            assert sc["error_kind"] == "unsafe_expression", (expr, sc)
        sc = _call(
            "transform_data",
            data_id=did,
            operations=[{"op": "assign", "column": "z", "expr": "y.__class__"}],
        )["result"]["structuredContent"]
        assert sc["error_kind"] == "unsafe_expression"

    def test_ordinary_expressions_work(self, ols_csv):
        did = _call("load_data", data_path=str(ols_csv))["result"]["structuredContent"][
            "data_id"
        ]
        sc = _call(
            "transform_data",
            data_id=did,
            operations=[
                {"op": "query", "expr": "x > 0 and y.notnull()"},
                {"op": "assign", "column": "ly", "expr": "log(abs(y) + 1)"},
            ],
        )["result"]["structuredContent"]
        assert "error" not in sc, sc
        assert 0 < sc["n_rows"] < 200 and "ly" in sc["columns"]


# ---------------------------------------------------------------------------
# Data cache byte budget, profile default, doc numbers
# ---------------------------------------------------------------------------


class TestDataCacheBytes:
    def test_bytes_budget_evicts_lru(self):
        from statspai.agent._data_cache import DataCache

        frame = pd.DataFrame({"a": np.zeros(1000)})  # ~8 KB
        cache = DataCache(max_size=16, max_bytes=20_000)
        ids = [cache.put(frame.copy(), tool="t") for _ in range(4)]
        assert cache.get(ids[0]) is None and cache.get(ids[-1]) is not None
        assert cache.miss_reason(ids[0]) == "bytes"
        assert cache.stats()["total_bytes"] <= 20_000

    def test_newest_kept_even_if_over_budget(self):
        from statspai.agent._data_cache import DataCache

        cache = DataCache(max_size=4, max_bytes=10)
        rid = cache.put(pd.DataFrame({"a": np.zeros(100)}), tool="t")
        assert cache.get(rid) is not None

    def test_env_default(self, monkeypatch):
        from statspai.agent._data_cache import DEFAULT_DATA_CACHE_BYTES, DataCache

        monkeypatch.delenv("STATSPAI_MCP_DATA_CACHE_BYTES", raising=False)
        assert DataCache().stats()["max_bytes"] == DEFAULT_DATA_CACHE_BYTES
        monkeypatch.setenv("STATSPAI_MCP_DATA_CACHE_BYTES", "0")
        assert DataCache().stats()["max_bytes"] is None

    def test_bytes_miss_hint(self, monkeypatch):
        from statspai.agent import _data_cache

        monkeypatch.setattr(_data_cache.DATA_CACHE, "miss_reason", lambda rid: "bytes")
        err = _data_cache.missing_handle_error("d_x")
        assert "STATSPAI_MCP_DATA_CACHE_BYTES" in err["hint"]


def test_default_profile_is_curated(monkeypatch):
    monkeypatch.delenv("STATSPAI_MCP_PROFILE", raising=False)
    assert (
        mcp_server._normalise_profile(None) == "curated" == mcp_server.DEFAULT_PROFILE
    )


def test_docs_tool_counts_match_reality():
    """Any "N curated / core tools" claim in the MCP docs must be current."""
    from statspai.agent.tools import tool_manifest

    curated = len(tool_manifest(curated_only=True))
    core = len(
        [
            n
            for n in (t["name"] for t in tool_manifest(curated_only=True))
            if n in mcp_server._CORE_PROFILE_TOOLS
        ]
    )
    root = Path(__file__).resolve().parents[1]
    docs = [
        root / "src/statspai/agent/_skill/references/mcp-and-cli.md",
        root / "docs/guides/agent_api.md",
        root / "docs/guides/economist_mcp_workflow.md",
    ]
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for m in re.finditer(r"~?(\d+)\s+curated tools", text):
            assert int(m.group(1)) == curated, (doc.name, m.group(0), curated)
        for m in re.finditer(r"~?(\d+)\s+core tools", text):
            assert int(m.group(1)) == core, (doc.name, m.group(0), core)
