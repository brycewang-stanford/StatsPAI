"""End-to-end ``statspai-mcp`` over a real subprocess stdio channel.

Every other MCP test drives ``handle_request`` in-process. This one
launches the server the way a client does (``python -m
statspai.agent.mcp_server``) and speaks JSON-RPC over pipes, which is
the only way to exercise:

* the reader thread — a client's reply to a server-initiated
  ``sampling/createMessage`` must be routed while a ``tools/call`` is
  still running (the 2026-09-28 audit found the old single-threaded
  loop could only time out here);
* the locked stdout sink — progress notifications, sampling requests
  and responses share one channel and must never interleave;
* the profile flag and the ``tools/list`` byte budget on the wire.
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


class _Server:
    def __init__(self, *args: str, env: Optional[Dict[str, str]] = None) -> None:
        full_env = dict(os.environ)
        full_env.setdefault("STATSPAI_SKIP_RUST", "1")
        # Short sampling timeout so a regression fails fast instead of
        # hanging the suite for the default 60 s.
        full_env["STATSPAI_MCP_SAMPLING_TIMEOUT_SECONDS"] = "20"
        if env:
            full_env.update(env)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "statspai.agent.mcp_server", *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=full_env,
        )
        self._next_id = 1

    def send(self, msg: Dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: Optional[Dict[str, Any]] = None) -> int:
        rid = self._next_id
        self._next_id += 1
        self.send(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        )
        return rid

    def read(self, timeout: float = 60.0) -> Dict[str, Any]:
        """Read one JSON-RPC line from the server (blocking, bounded)."""
        assert self.proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if line == "":
                if self.proc.poll() is not None:
                    err = self.proc.stderr.read() if self.proc.stderr else ""
                    raise RuntimeError(f"server exited early: {err[-2000:]}")
                time.sleep(0.01)
                continue
            line = line.strip()
            if line:
                return json.loads(line)
        raise TimeoutError("no line from server")

    def read_until(self, predicate, timeout: float = 60.0) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("server did not produce the expected message")
            msg = self.read(timeout=remaining)
            if predicate(msg):
                return msg

    def close(self) -> str:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()
        return self.proc.stderr.read() if self.proc.stderr else ""


@pytest.fixture
def server():
    srv = _Server("--profile", "core")
    try:
        yield srv
    finally:
        srv.close()


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": 1 + 2 * x + rng.normal(size=n)})
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)
    return str(path)


def _initialize(server: _Server, *, sampling: bool) -> Dict[str, Any]:
    caps: Dict[str, Any] = {"sampling": {}} if sampling else {}
    rid = server.request(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": caps,
            "clientInfo": {"name": "pytest", "version": "0"},
        },
    )
    msg = server.read_until(lambda m: m.get("id") == rid)
    assert "result" in msg, msg
    server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    return msg["result"]


def test_handshake_profile_and_tools_list_budget(server):
    _initialize(server, sampling=False)
    rid = server.request("tools/list")
    msg = server.read_until(lambda m: m.get("id") == rid)
    tools = msg["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"search_functions", "describe_function", "call_function", "did"} <= names
    # ``core`` on the wire stays well inside a context window.
    assert len(json.dumps(msg)) < 100_000
    assert len(tools) < 40


def test_sampling_round_trip_during_tools_call(server, csv_path):
    """``interpret_result`` asks the client's LLM and gets the answer back
    while the tools/call is still in flight."""
    _initialize(server, sampling=True)

    fit_id = server.request(
        "tools/call",
        {
            "name": "regress",
            "arguments": {"formula": "y ~ x", "data_path": csv_path, "as_handle": True},
        },
    )
    fit = server.read_until(lambda m: m.get("id") == fit_id)
    assert fit["result"]["isError"] is False, fit
    fitted = fit["result"]["structuredContent"]
    # The slope is 2 by construction (y = 1 + 2x + noise, n = 200).
    assert fitted["coefficients"]["x"]["estimate"] == pytest.approx(2.0, abs=0.3)
    result_id = fitted["result_id"]

    call_id = server.request(
        "tools/call",
        {
            "name": "interpret_result",
            "arguments": {"result_id": result_id, "question": "Is the slope real?"},
        },
    )
    # The server must now *send us* a sampling/createMessage request
    # before it can answer the tools/call.
    sampling_req = server.read_until(
        lambda m: m.get("method") == "sampling/createMessage", timeout=60
    )
    assert sampling_req["params"]["messages"]
    server.send(
        {
            "jsonrpc": "2.0",
            "id": sampling_req["id"],
            "result": {
                "role": "assistant",
                "model": "pytest-fake",
                "stopReason": "endTurn",
                "content": {
                    "type": "text",
                    "text": "CANNED INTERPRETATION FROM THE CLIENT MODEL",
                },
            },
        }
    )
    answer = server.read_until(lambda m: m.get("id") == call_id, timeout=60)
    payload = answer["result"]["structuredContent"]
    assert payload.get("sampling_error") is None, payload
    assert payload["backend"] != "deterministic", payload
    assert "CANNED INTERPRETATION" in payload["interpretation"]


def test_stdout_is_pure_jsonrpc(server, csv_path):
    """Every stdout line the server emits parses as JSON — estimator
    prints and warnings go to stderr."""
    _initialize(server, sampling=False)
    rid = server.request(
        "tools/call",
        {
            "name": "call_function",
            "arguments": {
                "function": "regress",
                "arguments": {"formula": "y ~ x"},
                "data_path": csv_path,
            },
        },
    )
    msg = server.read_until(lambda m: m.get("id") == rid)
    assert msg["result"]["isError"] is False
    payload = msg["result"]["structuredContent"]
    assert payload["called_via"] == "call_function"
    assert payload["coefficients"]["x"]["estimate"] == pytest.approx(2.0, abs=0.3)
    assert payload["n_obs"] == 200
    stderr = server.close()
    # A traceback on stderr would mean the server crashed somewhere.
    assert "Traceback" not in stderr, stderr[-2000:]


def test_help_flag_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "statspai.agent.mcp_server", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "--profile" in proc.stdout
