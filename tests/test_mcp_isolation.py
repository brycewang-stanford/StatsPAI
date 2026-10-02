"""Killable workers, late commits and the queue deadline (review item M3).

Three properties, each measured rather than argued:

* with ``STATSPAI_MCP_ISOLATION=process`` a call that never returns is
  killed at its deadline -- no orphaned thread, no leftover process, the
  server keeps answering, and it stays that way over repeated timeouts;
* under the default thread runner, a computation that outlives its
  timeout cannot commit a handle afterwards;
* a call that waited in the queue past ``STATSPAI_MCP_MAX_QUEUE_SECONDS``
  is refused when its turn comes instead of being run.

The hanging call is a regression on a FIFO nobody writes to: the read
blocks in C, which is the case a cooperative checkpoint cannot reach.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from statspai.agent import _process_worker, _runner, mcp_server
from statspai.agent._result_cache import ResultCache

pytestmark = pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="needs a FIFO to build a call that hangs"
)


class _Server:
    def __init__(self, **env: str) -> None:
        full = dict(os.environ)
        full.setdefault("STATSPAI_SKIP_RUST", "1")
        full.update(env)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "statspai.agent.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=full,
            close_fds=False,  # posix_spawn; see test_mcp_stdio_subprocess
        )
        self._next = 1
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._seen: Dict[int, Dict[str, Any]] = {}

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> int:
        rid = self._next
        self._next += 1
        assert self.proc.stdin is not None
        self.proc.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
            )
            + "\n"
        )
        self.proc.stdin.flush()
        return rid

    def call(self, name: str, **arguments: Any) -> int:
        return self.send("tools/call", {"name": name, "arguments": arguments})

    def wait(self, rid: int, timeout: float = 90.0) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while rid not in self._seen:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                kids = subprocess.run(
                    ["ps", "-o", "pid,ppid,stat,etime,command", "-ax"],
                    capture_output=True,
                    text=True,
                    close_fds=False,
                ).stdout
                mine = [ln for ln in kids.splitlines() if "mcp_server" in ln]
                raise AssertionError(
                    f"no response to request {rid}; server procs: {mine}"
                )
            try:
                line = self._lines.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            assert line is not None, "server exited"
            msg = json.loads(line)
            if "id" in msg:
                self._seen[msg["id"]] = msg
        return self._seen[rid]

    def close(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise


def _children(pid: int) -> list:
    out = subprocess.run(
        ["pgrep", "-P", str(pid)], capture_output=True, text=True, close_fds=False
    )
    return [int(x) for x in out.stdout.split()]


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(size=200)})
    frame["y"] = 1 + 2 * frame["x"] + rng.normal(size=200)
    good = tmp_path / "ok.csv"
    frame.to_csv(good, index=False)
    hang = tmp_path / "hang.csv"
    os.mkfifo(hang)
    return str(good), str(hang)


# ---------------------------------------------------------------------------
# Eligibility: only self-contained calls leave the process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,arguments,expected",
    [
        ("regress", {"formula": "y ~ x", "data_path": "/d.csv"}, True),
        ("regress", {"formula": "y ~ x", "data_records": [{"y": 1, "x": 2}]}, True),
        (
            "regress",
            {"formula": "y ~ x", "data_path": "/d.csv", "as_handle": True},
            False,
        ),
        ("regress", {"formula": "y ~ x", "data_id": "d_1"}, False),
        ("estat", {"result_id": "r_1", "test": "firststage"}, False),
        ("audit_result", {"result_id": "r_1"}, False),
        ("load_data", {"data_path": "/d.csv"}, False),
        ("transform_data", {"data_path": "/d.csv", "operations": []}, False),
        ("pipeline_did", {"data_path": "/d.csv"}, False),
        ("honest_did_from_result", {}, False),
        ("bibtex", {"keys": ["x"]}, True),
    ],
)
def test_eligibility(name, arguments, expected):
    assert _process_worker.eligible(name, arguments) is expected


def test_default_mode_is_the_thread_runner(monkeypatch):
    monkeypatch.delenv(_process_worker.ISOLATION_ENV, raising=False)
    assert _process_worker.isolation_mode() == "thread"
    monkeypatch.setenv(_process_worker.ISOLATION_ENV, "PROCESS")
    assert _process_worker.isolation_mode() == "process"
    monkeypatch.setenv(_process_worker.ISOLATION_ENV, "nonsense")
    assert _process_worker.isolation_mode() == "thread"


def test_annotate_drops_handles_that_died_with_the_child():
    payload = {
        "estimate": 1.0,
        "result_id": "r_dead",
        "result_uri": "statspai://result/r_dead",
        "next_calls": [
            {"tool": "audit_result", "arguments": {"result_id": "r_dead"}},
            {"tool": "bibtex", "arguments": {"keys": ["a"]}},
        ],
    }
    result = {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structuredContent": payload,
        "isError": False,
    }
    out = _process_worker.annotate(result)["structuredContent"]
    assert "result_id" not in out and "result_uri" not in out
    assert out["isolation"]["dropped_handles"] == ["result_id", "result_uri"]
    assert [c["tool"] for c in out["next_calls"]] == ["bibtex"]
    assert json.loads(result["content"][0]["text"]) == out


# ---------------------------------------------------------------------------
# Process mode over the wire
# ---------------------------------------------------------------------------


def test_isolated_call_returns_the_same_numbers_as_the_thread_runner(data):
    good, _ = data
    answers = {}
    for mode in ("thread", "process"):
        srv = _Server(STATSPAI_MCP_ISOLATION=mode)
        try:
            msg = srv.wait(srv.call("regress", formula="y ~ x", data_path=good))
            answers[mode] = msg["result"]["structuredContent"]
        finally:
            assert srv.close() == 0
    thread, proc = answers["thread"], answers["process"]
    assert proc["isolation"] == {"mode": "process"}
    assert "isolation" not in thread
    assert proc["coefficients"] == thread["coefficients"]
    assert proc["n_obs"] == thread["n_obs"] == 200
    assert proc["replay"] == thread["replay"]


def test_hung_call_is_killed_and_nothing_is_left_running(data):
    """Five timeouts in a row: no orphans, no children, pings stay fast."""
    good, hang = data
    srv = _Server(
        STATSPAI_MCP_ISOLATION="process", STATSPAI_MCP_TOOL_TIMEOUT_SECONDS="3"
    )
    try:
        for _ in range(5):
            started = time.monotonic()
            msg = srv.wait(srv.call("regress", formula="y ~ x", data_path=hang))
            elapsed = time.monotonic() - started
            sc = msg["result"]["structuredContent"]
            assert msg["result"]["isError"] is True
            assert sc["error_kind"] == "timeout"
            assert sc["worker_killed"] is True
            assert sc["worker_may_still_be_running"] is False
            assert 3.0 <= elapsed < 12.0
            assert _children(srv.proc.pid) == [], "the killed worker is still alive"
        # No orphan was registered, so the admission limit (4) never trips:
        # the sixth call runs.
        ok = srv.wait(srv.call("regress", formula="y ~ x", data_path=good))
        assert ok["result"]["isError"] is False
        assert ok["result"]["structuredContent"]["n_obs"] == 200
        pinged = time.monotonic()
        assert srv.wait(srv.send("ping"))["result"] == {}
        assert time.monotonic() - pinged < 2.0
    finally:
        assert srv.close() == 0


def test_thread_runner_leaves_an_orphan_on_the_same_hang(data):
    """The contrast that motivates the mode: same call, default runner."""
    _, hang = data
    srv = _Server(STATSPAI_MCP_TOOL_TIMEOUT_SECONDS="2")
    try:
        msg = srv.wait(srv.call("regress", formula="y ~ x", data_path=hang))
        sc = msg["result"]["structuredContent"]
        assert sc["error_kind"] == "timeout"
        assert sc["worker_may_still_be_running"] is True
        assert sc["orphaned_tool_threads"] >= 1
    finally:
        # The blocked read never returns, so the interpreter cannot join
        # it cleanly; release it by opening the FIFO for writing.
        with open(hang, "w"):
            pass
        srv.close()


def test_client_cancel_kills_the_isolated_worker(data):
    _, hang = data
    srv = _Server(STATSPAI_MCP_ISOLATION="process")
    try:
        rid = srv.call("regress", formula="y ~ x", data_path=hang)
        deadline = time.monotonic() + 30
        while not _children(srv.proc.pid):
            assert time.monotonic() < deadline, "worker never started"
            time.sleep(0.1)
        assert srv.proc.stdin is not None
        srv.proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": rid},
                }
            )
            + "\n"
        )
        srv.proc.stdin.flush()
        deadline = time.monotonic() + 15
        while _children(srv.proc.pid):
            assert time.monotonic() < deadline, "cancel did not kill the worker"
            time.sleep(0.1)
        assert srv.wait(srv.send("ping"))["result"] == {}
    finally:
        assert srv.close() == 0


def test_stateful_calls_stay_in_process_and_keep_their_handles(data):
    good, _ = data
    srv = _Server(STATSPAI_MCP_ISOLATION="process")
    try:
        fit = srv.wait(
            srv.call("regress", formula="y ~ x", data_path=good, as_handle=True)
        )["result"]["structuredContent"]
        assert "isolation" not in fit
        rid = fit["result_id"]
        brief = srv.wait(srv.call("brief_result", result_id=rid))["result"]
        assert brief["isError"] is False
        assert brief["structuredContent"]["result_id"] == rid
    finally:
        assert srv.close() == 0


# ---------------------------------------------------------------------------
# Thread runner: nothing is committed after the deadline
# ---------------------------------------------------------------------------


def test_cache_refuses_a_commit_from_a_call_that_already_timed_out():
    cache = ResultCache()
    committed = []
    release = threading.Event()

    def _work():
        release.wait(10)  # outlives the 0.2 s deadline below
        committed.append(cache.put({"late": True}, tool="slow"))
        return "unreachable"

    ok, payload = _runner.run_with_progress(_work, timeout=0.2)
    assert ok is False and isinstance(payload, TimeoutError)
    release.set()
    deadline = time.monotonic() + 5
    while _runner.orphaned_threads() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _runner.orphaned_threads() == 0
    assert committed == [], "a timed-out call committed a handle"
    assert len(cache) == 0
    # A live call still commits.
    ok, rid = _runner.run_with_progress(lambda: cache.put({"v": 1}), timeout=5)
    assert ok is True and cache.get(rid) == {"v": 1}


# ---------------------------------------------------------------------------
# Queue deadline
# ---------------------------------------------------------------------------


def test_call_that_waited_past_the_queue_deadline_is_not_run(monkeypatch):
    import io

    ran = []
    gate = threading.Event()

    def _tool(name, arguments, **kwargs):
        ran.append(arguments.get("tag"))
        if arguments.get("tag") == "first":
            gate.wait(10)
        return {"value": arguments.get("tag")}

    monkeypatch.setattr(mcp_server, "execute_tool", _tool)
    monkeypatch.setenv(mcp_server.MAX_QUEUE_SECONDS_ENV, "1")
    inbox: "queue.Queue[Optional[str]]" = queue.Queue()

    def _stdin():
        while True:
            line = inbox.get()
            if line is None:
                return
            yield line

    out = io.StringIO()
    server = threading.Thread(
        target=mcp_server.serve_stdio,
        kwargs={"stdin": _stdin(), "stdout": out, "workers": 1},
        daemon=True,
    )
    server.start()
    for rid, tag in ((1, "first"), (2, "second")):
        inbox.put(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": rid,
                    "method": "tools/call",
                    "params": {"name": "regress", "arguments": {"tag": tag}},
                }
            )
        )
    time.sleep(1.6)  # the second call is now past its 1 s deadline
    gate.set()
    inbox.put(None)
    server.join(timeout=15)
    assert not server.is_alive()
    replies = {m["id"]: m for m in map(json.loads, out.getvalue().splitlines())}
    assert replies[1]["result"]["structuredContent"]["value"] == "first"
    late = replies[2]["result"]
    assert late["isError"] is True
    assert late["structuredContent"]["error_kind"] == "server_busy"
    assert late["structuredContent"]["queued_seconds"] >= 1.0
    assert ran == ["first"], "the expired call was run anyway"
