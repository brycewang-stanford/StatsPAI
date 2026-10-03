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


def _rss_mb(pid: int) -> float:
    """Resident set size of a process, in MiB (``ps`` reports KiB)."""
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        close_fds=False,
    )
    return int(out.stdout.split()[0]) / 1024.0


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
        # a result handle is handed over from the worker (see below)
        (
            "regress",
            {"formula": "y ~ x", "data_path": "/d.csv", "as_handle": True},
            True,
        ),
        ("regress", {"formula": "y ~ x", "data_id": "d_1"}, False),
        # a call that reads a result is isolated: the entry is shipped in
        ("estat", {"result_id": "r_1", "test": "firststage"}, True),
        ("audit_result", {"result_id": "r_1"}, True),
        ("honest_did_from_result", {"result_id": "r_1"}, True),
        # ... except the one that asks the client for a completion
        ("interpret_result", {"result_id": "r_1"}, False),
        ("load_data", {"data_path": "/d.csv"}, False),
        ("transform_data", {"data_path": "/d.csv", "operations": []}, False),
        ("pipeline_did", {"data_path": "/d.csv"}, False),
        ("honest_did_from_result", {"data_id": "d_1"}, False),
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
    """Three timeouts in a row: no orphans, no children, pings stay fast.

    The deadline covers the child's interpreter start as well as the fit
    (a few seconds cold), so it is set well above that: the last call has
    to finish inside it.
    """
    good, hang = data
    srv = _Server(
        STATSPAI_MCP_ISOLATION="process", STATSPAI_MCP_TOOL_TIMEOUT_SECONDS="10"
    )
    try:
        # Warm call first, so the baseline includes whatever the server
        # itself allocates to supervise a child.
        warm = srv.wait(srv.call("regress", formula="y ~ x", data_path=good))
        assert warm["result"]["isError"] is False
        baseline = _rss_mb(srv.proc.pid)
        for _ in range(3):
            started = time.monotonic()
            msg = srv.wait(srv.call("regress", formula="y ~ x", data_path=hang))
            elapsed = time.monotonic() - started
            sc = msg["result"]["structuredContent"]
            assert msg["result"]["isError"] is True
            assert sc["error_kind"] == "timeout"
            assert sc["worker_killed"] is True
            assert sc["worker_may_still_be_running"] is False
            assert 10.0 <= elapsed < 25.0
            assert _children(srv.proc.pid) == [], "the killed worker is still alive"
        # No orphan was registered and no worker survives: the next call
        # runs normally.
        ok = srv.wait(srv.call("regress", formula="y ~ x", data_path=good))
        assert ok["result"]["isError"] is False, ok["result"]["structuredContent"]
        assert ok["result"]["structuredContent"]["n_obs"] == 200
        pinged = time.monotonic()
        assert srv.wait(srv.send("ping"))["result"] == {}
        assert time.monotonic() - pinged < 2.0
        # Memory: the killed workers took theirs with them, so the server's
        # own footprint does not grow. Measured once on macOS: 165 MiB
        # after the warm call, 40 MiB *lower* after three timeouts (the OS
        # reclaimed pages). The bound only has to catch a leak per timeout.
        grown = _rss_mb(srv.proc.pid) - baseline
        assert grown < 25.0, f"server RSS grew {grown:.1f} MiB over three timeouts"
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


def test_calls_that_read_a_result_are_isolated_too(data):
    """Fit in a worker, hand the result over, read it from a second worker."""
    good, _ = data
    srv = _Server(STATSPAI_MCP_ISOLATION="process")
    try:
        fit = srv.wait(
            srv.call("regress", formula="y ~ x", data_path=good, as_handle=True)
        )["result"]["structuredContent"]
        assert fit["isolation"]["adopted_handles"] == ["result_id"]
        rid = fit["result_id"]
        brief = srv.wait(srv.call("brief_result", result_id=rid))["result"]
        assert brief["isError"] is False
        body = brief["structuredContent"]
        assert body["result_id"] == rid
        assert body["isolation"]["mode"] == "process"
        assert "dropped_handles" not in body["isolation"]
        # the server still holds the result: a second reader works
        audit = srv.wait(srv.call("audit_result", result_id=rid))["result"]
        assert audit["isError"] is False
        # data tools stay in the server
        loaded = srv.wait(srv.call("load_data", data_path=good))["result"]
        assert "isolation" not in loaded["structuredContent"]
        assert loaded["structuredContent"]["n_rows"] == 200
        # an unknown handle is answered by the server, with its usual error
        missing = srv.wait(srv.call("brief_result", result_id="r_deadbeef"))["result"]
        assert missing["isError"] is True
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


# ---------------------------------------------------------------------------
# Result handles cross the process boundary
# ---------------------------------------------------------------------------


def test_annotate_keeps_a_handle_the_parent_adopted():
    payload = {
        "estimate": 1.0,
        "result_id": "r_live",
        "result_uri": "statspai://result/r_live",
        "data_id": "d_child",
        "next_calls": [{"tool": "audit_result", "arguments": {"result_id": "r_live"}}],
    }
    result = {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structuredContent": payload,
        "isError": False,
    }
    out = _process_worker.annotate(result, adopted=["r_live"])["structuredContent"]
    assert out["result_id"] == "r_live" and "result_uri" in out
    assert "data_id" not in out
    assert out["isolation"]["adopted_handles"] == ["result_id"]
    assert out["isolation"]["dropped_handles"] == ["data_id"]


def test_handle_from_an_isolated_fit_works_in_the_parent(data):
    """Fit in a killable worker, then use the handle in the server itself."""
    good, _ = data
    answers = {}
    for mode in ("thread", "process"):
        srv = _Server(STATSPAI_MCP_ISOLATION=mode)
        try:
            fit = srv.wait(
                srv.call("regress", formula="y ~ x", data_path=good, as_handle=True)
            )["result"]["structuredContent"]
            rid = fit["result_id"]
            follow = srv.wait(srv.call("audit_result", result_id=rid))["result"]
            answers[mode] = (fit, follow)
        finally:
            assert srv.close() == 0
    (fit_t, follow_t), (fit_p, follow_p) = answers["thread"], answers["process"]
    assert fit_p["isolation"] == {"mode": "process", "adopted_handles": ["result_id"]}
    assert fit_p["coefficients"] == fit_t["coefficients"]
    assert fit_p["n_obs"] == fit_t["n_obs"] == 200
    assert follow_p["isError"] is False and follow_t["isError"] is False
    # the follow-up reads the adopted result (itself from a second worker)
    keys_p = sorted(k for k in follow_p["structuredContent"] if k != "isolation")
    assert keys_p == sorted(follow_t["structuredContent"])


def test_cache_adopts_under_the_child_s_id_and_refuses_a_collision():
    from statspai.agent._result_cache import CacheEntry, ResultCache

    cache = ResultCache(max_size=4)
    entry = CacheEntry(obj={"b": 1.5}, tool="regress", arguments={"formula": "y ~ x"})
    assert cache.adopt("r_abc12345", entry) is True
    assert cache.get("r_abc12345") == {"b": 1.5}
    assert cache.get_entry("r_abc12345").tool == "regress"
    assert cache.adopt("r_abc12345", CacheEntry(obj=2, tool="x")) is False
    assert cache.get("r_abc12345") == {"b": 1.5}
    assert len(cache) == 1


def test_export_and_import_round_trip_and_report_what_cannot_be_pickled(
    tmp_path, monkeypatch
):
    from statspai.agent._result_cache import RESULT_CACHE, CacheEntry

    RESULT_CACHE.clear()
    good = RESULT_CACHE.put({"estimate": 0.25}, tool="regress")
    bad = RESULT_CACHE.put(lambda: 1, tool="regress")  # not picklable
    key = "ab" * 32
    monkeypatch.setenv(_process_worker.HANDOFF_ENV, str(tmp_path))
    monkeypatch.setenv(_process_worker.HANDOFF_KEY_ENV, key)
    _process_worker.export_handles()
    assert (tmp_path / f"{good}.pkl").exists()
    assert not (tmp_path / f"{bad}.pkl").exists()

    # A file signed with another key is refused before it is unpickled.
    RESULT_CACHE.clear()
    adopted, failed = _process_worker.import_handles(str(tmp_path), "cd" * 32)
    assert adopted == [] and "signature" in failed[good]
    assert len(RESULT_CACHE) == 0

    adopted, failed = _process_worker.import_handles(str(tmp_path), key)
    assert adopted == [good]
    assert bad in failed and "pickle" in failed[bad].lower()
    assert RESULT_CACHE.get(good) == {"estimate": 0.25}
    assert isinstance(RESULT_CACHE.get_entry(good), CacheEntry)
    RESULT_CACHE.clear()


def test_export_is_a_no_op_outside_a_worker(monkeypatch):
    monkeypatch.delenv(_process_worker.HANDOFF_ENV, raising=False)
    monkeypatch.delenv(_process_worker.HANDOFF_KEY_ENV, raising=False)
    _process_worker.export_handles()  # must not raise or write anywhere


def test_inbound_hand_off_round_trip(tmp_path, monkeypatch):
    """Parent writes the entry a call reads; the worker adopts it and does not
    send it back."""
    from statspai.agent._result_cache import RESULT_CACHE

    RESULT_CACHE.clear()
    key = "ef" * 32
    rid = RESULT_CACHE.put({"estimate": 1.25}, tool="regress")
    assert _process_worker.export_inbound(str(tmp_path), key, [rid]) is True
    assert _process_worker.export_inbound(str(tmp_path), key, ["r_missing0"]) is False
    unpicklable = RESULT_CACHE.put(lambda: 1, tool="regress")
    assert _process_worker.export_inbound(str(tmp_path), key, [unpicklable]) is False

    # the worker's side
    RESULT_CACHE.clear()
    _process_worker._INBOUND.clear()
    monkeypatch.setenv(_process_worker.HANDOFF_ENV, str(tmp_path))
    monkeypatch.setenv(_process_worker.HANDOFF_KEY_ENV, key)
    assert _process_worker.import_inbound() == [rid]
    assert RESULT_CACHE.get(rid) == {"estimate": 1.25}
    new = RESULT_CACHE.put({"estimate": 2.5}, tool="audit_result")
    _process_worker.export_handles()
    assert (tmp_path / f"{new}.pkl").exists()
    assert not (tmp_path / f"{rid}.pkl").exists()  # inbound, not sent back
    RESULT_CACHE.clear()
    _process_worker._INBOUND.clear()


def test_inbound_file_with_a_wrong_signature_is_refused(tmp_path, monkeypatch):
    from statspai.agent._result_cache import RESULT_CACHE

    RESULT_CACHE.clear()
    rid = RESULT_CACHE.put({"estimate": 1.25}, tool="regress")
    assert _process_worker.export_inbound(str(tmp_path), "ab" * 32, [rid])
    RESULT_CACHE.clear()
    monkeypatch.setenv(_process_worker.HANDOFF_ENV, str(tmp_path))
    monkeypatch.setenv(_process_worker.HANDOFF_KEY_ENV, "cd" * 32)
    with pytest.raises(ValueError, match="signature"):
        _process_worker.import_inbound()
    assert len(RESULT_CACHE) == 0
