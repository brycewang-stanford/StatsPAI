"""Concurrency and cancellation of the MCP stdio loop.

``serve_stdio`` hands ``tools/call`` to a worker pool, so the main loop
keeps answering ``ping`` / ``tools/list`` while a tool runs, and honours
``notifications/cancelled`` (the tool stops at its next progress
checkpoint and no response is sent for the cancelled request).

The server runs in-process on a thread, fed through a queue-backed stdin
and a queue-backed stdout, with ``execute_tool`` monkeypatched to a slow
fake tool that reports progress — a real estimator would make the test
slow and non-deterministic.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, Iterator, Optional

import pytest

from statspai.agent import _runner, mcp_server


class _Pipe:
    """stdout stand-in: every ``write`` of a full line lands on a queue."""

    def __init__(self) -> None:
        self.lines: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.lines.put(json.loads(line))
        return len(text)

    def flush(self) -> None:
        return None


class _Harness:
    def __init__(self, workers: Optional[int] = None) -> None:
        self.inbox: "queue.Queue[Optional[str]]" = queue.Queue()
        self.out = _Pipe()

        def _stdin() -> Iterator[str]:
            while True:
                item = self.inbox.get()
                if item is None:
                    return
                yield item + "\n"

        self.thread = threading.Thread(
            target=mcp_server.serve_stdio,
            kwargs={"stdin": _stdin(), "stdout": self.out, "workers": workers},
            daemon=True,
        )
        self.thread.start()

    def send(self, msg: Dict[str, Any]) -> None:
        self.inbox.put(json.dumps(msg))

    def read_until(self, pred, timeout: float = 10.0) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                msg = self.out.lines.get(timeout=0.05)
            except queue.Empty:
                continue
            if pred(msg):
                return msg
        raise TimeoutError("expected message not received")

    def drain(self, seconds: float) -> list:
        got = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                got.append(self.out.lines.get(timeout=0.05))
            except queue.Empty:
                pass
        return got

    def close(self) -> None:
        self.inbox.put(None)
        self.thread.join(timeout=15)
        assert not self.thread.is_alive(), "serve_stdio did not exit"


@pytest.fixture
def slow_tool(monkeypatch):
    state = {"started": threading.Event(), "cancelled": threading.Event(), "steps": 0}

    def _tool(
        name, arguments, *, data=None, detail="agent", result_id=None, as_handle=False
    ):
        state["started"].set()
        try:
            for i in range(400):  # ~20 s if never cancelled
                _runner.progress(i, total=400)
                state["steps"] += 1
                time.sleep(0.05)
        except _runner.ToolCancelled:
            state["cancelled"].set()
            raise
        return {"value": "finished"}

    monkeypatch.setattr(mcp_server, "execute_tool", _tool)
    monkeypatch.setenv(_runner.TOOL_TIMEOUT_ENV, "60")
    return state


def _call(rid, token=None):
    params: Dict[str, Any] = {"name": "regress", "arguments": {"formula": "y ~ x"}}
    if token is not None:
        params["_meta"] = {"progressToken": token}
    return {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": params}


def test_ping_and_tools_list_answered_while_tool_runs_then_cancel(slow_tool):
    h = _Harness()
    try:
        h.send(_call(1, token="tok"))
        assert slow_tool["started"].wait(10)
        h.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        pong = h.read_until(lambda m: m.get("id") == 2)
        assert pong["result"] == {}
        h.send({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        listed = h.read_until(lambda m: m.get("id") == 3)
        assert listed["result"]["tools"]
        assert not slow_tool["cancelled"].is_set()

        h.send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 1, "reason": "user abort"},
            }
        )
        assert slow_tool["cancelled"].wait(5), "tool did not stop at a checkpoint"
        # Per the MCP spec no response is sent for a cancelled request.
        later = h.drain(0.5)
        assert not [m for m in later if m.get("id") == 1], later
        assert slow_tool["steps"] < 400

        # The pool is free again: a follow-up request is answered.
        h.send({"jsonrpc": "2.0", "id": 4, "method": "ping"})
        assert h.read_until(lambda m: m.get("id") == 4)["result"] == {}
    finally:
        h.close()


def test_cancel_while_queued_behind_another_call(slow_tool):
    h = _Harness(workers=1)
    try:
        h.send(_call(10))
        assert slow_tool["started"].wait(10)
        h.send(_call(11))  # queued behind 10 on the single worker
        h.send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 11},
            }
        )
        h.send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 10},
            }
        )
        assert slow_tool["cancelled"].wait(5)
        msgs = h.drain(0.8)
        assert not [m for m in msgs if m.get("id") in (10, 11)], msgs
    finally:
        h.close()


def test_uncancelled_call_still_answers(monkeypatch):
    def _tool(
        name, arguments, *, data=None, detail="agent", result_id=None, as_handle=False
    ):
        _runner.progress(0.5, total=1.0)
        return {"value": 42}

    monkeypatch.setattr(mcp_server, "execute_tool", _tool)
    h = _Harness()
    try:
        h.send(_call("abc", token="t"))
        res = h.read_until(lambda m: m.get("id") == "abc")
        assert res["result"]["structuredContent"] == {"value": 42}
    finally:
        h.close()


def test_cancel_of_unknown_request_is_ignored():
    h = _Harness()
    try:
        h.send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 99},
            }
        )
        h.send({"jsonrpc": "2.0", "id": 5, "method": "ping"})
        assert h.read_until(lambda m: m.get("id") == 5)["result"] == {}
    finally:
        h.close()


def test_check_cancelled_is_noop_outside_calls():
    _runner.check_cancelled()
    _runner.progress(1.0)  # no channel, no cancel event: no-op


def test_runner_reports_cancel_and_orphan():
    ev = threading.Event()
    release = threading.Event()

    def _work():
        release.wait(5)  # no checkpoints: cannot be stopped
        return 1

    def _cancel_soon():
        time.sleep(0.1)
        ev.set()

    threading.Thread(target=_cancel_soon, daemon=True).start()
    ok, payload = _runner.run_with_progress(_work, cancel_event=ev)
    assert ok is False and isinstance(payload, _runner.ToolCancelled)
    assert _runner.orphaned_threads() >= 1
    release.set()
    deadline = time.monotonic() + 5
    while _runner.orphaned_threads() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _runner.orphaned_threads() == 0


# ---------------------------------------------------------------------------
# Admission limits (2026-10-02 review, M3)
# ---------------------------------------------------------------------------


def _is_response(rid):
    return lambda m: m.get("id") == rid and ("result" in m or "error" in m)


def test_queue_limit_answers_server_busy_without_queueing(slow_tool, monkeypatch):
    monkeypatch.setenv(mcp_server.MAX_QUEUED_CALLS_ENV, "1")
    srv = _Harness(workers=1)
    try:
        srv.send(_call(1, token="p1"))
        assert slow_tool["started"].wait(5)
        srv.send(_call(2))  # waits behind call 1
        srv.send(_call(3))  # over the limit
        busy = srv.read_until(_is_response(3))
        sc = busy["result"]["structuredContent"]
        assert busy["result"]["isError"] is True
        assert sc["error_kind"] == "server_busy"
        assert sc["calls_in_flight"] == 2 and sc["retryable"] is True
        # Liveness is unaffected, and the refused call never ran.
        srv.send({"jsonrpc": "2.0", "id": 9, "method": "ping"})
        assert srv.read_until(_is_response(9))["result"] == {}
        for rid in (2, 1):
            srv.send(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": rid},
                }
            )
        assert slow_tool["cancelled"].wait(5)
    finally:
        srv.close()


def test_duplicate_in_flight_id_is_refused_and_keeps_the_cancel_handle(slow_tool):
    srv = _Harness(workers=1)
    try:
        srv.send(_call(1, token="p1"))
        assert slow_tool["started"].wait(5)
        srv.send(_call(1))
        dup = srv.read_until(lambda m: m.get("id") == 1 and "error" in m)
        assert dup["error"]["code"] == -32600
        # The original call still owns id 1: cancelling it stops the tool.
        srv.send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": 1},
            }
        )
        assert slow_tool["cancelled"].wait(5)
    finally:
        srv.close()


def test_orphan_limit_refuses_new_calls(monkeypatch):
    calls = []

    def _tool(
        name, arguments, *, data=None, detail="agent", result_id=None, as_handle=False
    ):
        calls.append(name)
        return {"value": 1}

    monkeypatch.setattr(mcp_server, "execute_tool", _tool)
    monkeypatch.setenv(mcp_server.MAX_ORPHANED_CALLS_ENV, "2")
    monkeypatch.setattr(_runner, "orphaned_threads", lambda: 2)
    srv = _Harness(workers=1)
    try:
        srv.send(_call(1))
        sc = srv.read_until(_is_response(1))["result"]["structuredContent"]
        assert sc["error_kind"] == "server_busy"
        assert sc["orphaned_tool_threads"] == 2
        assert calls == []
        monkeypatch.setattr(_runner, "orphaned_threads", lambda: 1)
        srv.send(_call(2))
        ok = srv.read_until(_is_response(2))["result"]
        assert ok["isError"] is False and calls == ["regress"]
    finally:
        srv.close()


def test_oversized_request_line_is_refused_unparsed(monkeypatch):
    monkeypatch.setenv(mcp_server.MAX_REQUEST_BYTES_ENV, "2000")
    srv = _Harness(workers=1)
    try:
        big = _call(5)
        big["params"]["arguments"]["data"] = "x" * 5000
        srv.send(big)
        err = srv.read_until(lambda m: "error" in m)
        assert err["id"] is None and err["error"]["code"] == -32600
        assert "STATSPAI_MCP_MAX_REQUEST_BYTES" in err["error"]["message"]
        srv.send({"jsonrpc": "2.0", "id": 6, "method": "ping"})
        assert srv.read_until(_is_response(6))["result"] == {}
    finally:
        srv.close()
