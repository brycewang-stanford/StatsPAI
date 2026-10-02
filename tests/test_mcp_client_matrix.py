"""What kinds of client the stdio server is tested against (review item M4).

Each row is a real ``python -m statspai.agent.mcp_server`` subprocess
driven the way that kind of client drives it. The claims are narrow on
purpose: these are the combinations the server says it supports, nothing
about transports or clients it does not ship (no HTTP, no OAuth).

=========================  ==============================================
client                     what is asserted
=========================  ==============================================
each protocol revision     negotiated back verbatim; analysis works
unknown / missing version  server's preferred revision is offered
text-only                  the text block alone carries result and error
no sampling capability     interpret_result answers without the client's
                           LLM instead of waiting for it
sends a cursor             lists come back whole, with no nextCursor
restarts the server        old handles are errors that say how to recover
skips ``initialize``       calls still work (stateless handshake)
unknown method             -32601, and the session survives
=========================  ==============================================
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from statspai.agent.mcp_server import SUPPORTED_PROTOCOL_VERSIONS


class _Client:
    def __init__(self) -> None:
        env = dict(os.environ)
        env.setdefault("STATSPAI_SKIP_RUST", "1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "statspai.agent.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
            close_fds=False,  # posix_spawn; see test_mcp_stdio_subprocess
        )
        self._next = 1

    def rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        rid = self._next
        self._next += 1
        self.proc.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
            )
            + "\n"
        )
        self.proc.stdin.flush()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("server exited")
            msg = json.loads(line)
            if msg.get("id") == rid:
                return msg
        raise TimeoutError(method)

    def initialize(self, version: Optional[str], **capabilities: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "capabilities": capabilities,
            "clientInfo": {"name": "matrix", "version": "0"},
        }
        if version is not None:
            params["protocolVersion"] = version
        return self.rpc("initialize", params)["result"]

    def tool(self, name: str, **arguments: Any) -> Dict[str, Any]:
        return self.rpc("tools/call", {"name": name, "arguments": arguments})["result"]

    def close(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        return self.proc.wait(timeout=30)


@pytest.fixture
def client():
    c = _Client()
    try:
        yield c
    finally:
        assert c.close() == 0


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(size=150)})
    frame["y"] = 1 + 2 * frame["x"] + rng.normal(size=150)
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    slope = float(np.polyfit(frame["x"], frame["y"], 1)[0])
    return str(path), slope


@pytest.mark.parametrize("version", SUPPORTED_PROTOCOL_VERSIONS)
def test_each_supported_revision_is_negotiated_and_usable(version, csv):
    path, slope = csv
    c = _Client()
    try:
        assert c.initialize(version)["protocolVersion"] == version
        res = c.tool("regress", formula="y ~ x", data_path=path)
        assert res["isError"] is False
        got = res["structuredContent"]["coefficients"]["x"]["estimate"]
        assert got == pytest.approx(slope, rel=1e-9)
    finally:
        assert c.close() == 0


@pytest.mark.parametrize("version", ["1999-01-01", None])
def test_unknown_or_missing_revision_gets_the_preferred_one(client, version):
    offered = client.initialize(version)["protocolVersion"]
    assert offered == SUPPORTED_PROTOCOL_VERSIONS[0]


def test_text_only_client_gets_the_whole_result_and_the_whole_error(client, csv):
    """A client that never reads structuredContent loses nothing."""
    path, slope = csv
    client.initialize("2024-11-05")
    ok = client.tool("regress", formula="y ~ x", data_path=path)
    text = json.loads(ok["content"][0]["text"])
    assert text["coefficients"]["x"]["estimate"] == pytest.approx(slope, rel=1e-9)
    assert text["n_obs"] == 150 and "replay" in text
    bad = client.tool("regress", formula="y ~ x", data_path=path + ".missing")
    assert bad["isError"] is True
    err = json.loads(bad["content"][0]["text"])
    assert err["error_kind"] == "file_not_found" and len(err["message"]) > 10


def test_client_without_sampling_is_not_made_to_wait(client, csv):
    path, _ = csv
    client.initialize("2025-06-18")  # no sampling capability advertised
    fit = client.tool("regress", formula="y ~ x", data_path=path, as_handle=True)
    rid = fit["structuredContent"]["result_id"]
    started = time.monotonic()
    out = client.tool("interpret_result", result_id=rid)
    assert time.monotonic() - started < 20
    assert out["isError"] is False
    assert out["structuredContent"]["summary"]["fields"]["n_obs"] == 150


def test_lists_are_single_page_whatever_cursor_is_sent(client):
    client.initialize("2025-06-18")
    plain = client.rpc("tools/list")["result"]
    paged = client.rpc("tools/list", {"cursor": "opaque-token"})["result"]
    assert "nextCursor" not in plain and "nextCursor" not in paged
    assert [t["name"] for t in paged["tools"]] == [t["name"] for t in plain["tools"]]
    assert len(plain["tools"]) >= 20
    for method, key in (("resources/list", "resources"), ("prompts/list", "prompts")):
        res = client.rpc(method, {"cursor": "opaque-token"})["result"]
        assert "nextCursor" not in res and len(res[key]) >= 1


def test_handles_do_not_survive_a_restart_and_the_error_says_so(csv):
    path, _ = csv
    first = _Client()
    try:
        first.initialize("2025-06-18")
        fit = first.tool("regress", formula="y ~ x", data_path=path, as_handle=True)
        rid = fit["structuredContent"]["result_id"]
        data_id = first.tool("load_data", data_path=path)["structuredContent"][
            "data_id"
        ]
    finally:
        assert first.close() == 0
    second = _Client()
    try:
        second.initialize("2025-06-18")
        stale = second.tool("audit_result", result_id=rid)
        sc = stale["structuredContent"]
        assert stale["isError"] is True
        assert sc["error_kind"] == "missing_result_handle"
        assert "restarted" in sc["hint"] and "as_handle" in sc["hint"]
        gone = second.tool("regress", formula="y ~ x", data_id=data_id)
        assert gone["isError"] is True
        assert "load_data" in json.dumps(gone["structuredContent"])
        # The recovery the hint names works.
        again = second.tool("regress", formula="y ~ x", data_path=path, as_handle=True)
        assert again["structuredContent"]["result_id"] != rid
    finally:
        assert second.close() == 0


def test_calls_work_without_an_initialize_handshake(client, csv):
    path, slope = csv
    res = client.tool("regress", formula="y ~ x", data_path=path)
    got = res["structuredContent"]["coefficients"]["x"]["estimate"]
    assert got == pytest.approx(slope, rel=1e-9)


def test_unknown_method_is_an_error_and_the_session_survives(client):
    client.initialize("2025-06-18")
    err = client.rpc("tools/does_not_exist")
    assert err["error"]["code"] == -32601
    assert client.rpc("ping")["result"] == {}
    assert len(client.rpc("tools/list")["result"]["tools"]) >= 20
