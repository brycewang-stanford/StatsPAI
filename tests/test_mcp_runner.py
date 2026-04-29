"""Tests for Module 4 — concurrent runner + timeout + progress notifications.

Covers:
* ``run_with_progress`` returns the work() value when no timeout fires.
* Tool can call ``progress(...)`` and the drain callback receives it.
* ``progress(...)`` is a no-op when no channel is registered.
* Timeout fires and returns a ``TimeoutError`` envelope.
* Worker exception is surfaced verbatim.
* ``serve_stdio`` registers/unregisters the progress sink so MCP RPC
  in-flight progress is written to the same stdout stream.
"""
from __future__ import annotations

import io
import json
import time

import pytest

from statspai.agent._runner import (
    progress,
    run_with_progress,
    tool_timeout,
    TOOL_TIMEOUT_ENV,
)
from statspai.agent import mcp_handle_request, mcp_serve_stdio


# ----------------------------------------------------------------------
# run_with_progress core
# ----------------------------------------------------------------------

class TestRunWithProgress:
    def test_returns_work_value(self):
        ok, val = run_with_progress(lambda: 42)
        assert ok is True
        assert val == 42

    def test_surfaces_exception(self):
        def boom():
            raise ValueError("nope")
        ok, val = run_with_progress(boom)
        assert ok is False
        assert isinstance(val, ValueError)
        assert str(val) == "nope"

    def test_progress_callback_receives_events(self):
        events = []

        def work():
            for i in range(3):
                progress(i, total=3, message=f"step {i}")
            return "done"

        ok, val = run_with_progress(
            work, progress_token="tkn-1",
            drain=lambda p: events.append(p),
        )
        assert ok is True
        assert val == "done"
        # All 3 events captured + their payload shape
        assert len(events) == 3
        assert events[0]["progressToken"] == "tkn-1"
        assert events[0]["progress"] == 0
        assert events[0]["total"] == 3
        assert events[2]["message"] == "step 2"

    def test_progress_no_op_when_unset(self):
        # Calling progress() outside a runner ⇒ no exception, just silent
        progress(0.5, total=1.0, message="loose call")  # must not raise

    def test_timeout_fires_and_returns_timeout_error(self):
        def slow():
            time.sleep(2.0)
            return "should not see this"

        ok, val = run_with_progress(slow, timeout=0.1)
        assert ok is False
        assert isinstance(val, TimeoutError)
        assert "timeout" in str(val).lower()


# ----------------------------------------------------------------------
# tool_timeout env var
# ----------------------------------------------------------------------

class TestToolTimeout:
    def test_default_is_a_number(self, monkeypatch):
        monkeypatch.delenv(TOOL_TIMEOUT_ENV, raising=False)
        v = tool_timeout()
        assert v is not None and v > 0

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv(TOOL_TIMEOUT_ENV, "0")
        assert tool_timeout() is None

    def test_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(TOOL_TIMEOUT_ENV, "abc")
        v = tool_timeout()
        assert v is not None and v > 0


# ----------------------------------------------------------------------
# MCP RPC: progress notifications round-trip via serve_stdio
# ----------------------------------------------------------------------

class TestServeStdioProgressSink:
    def test_progress_appears_on_stdout(self, monkeypatch):
        """End-to-end: when a tool emits progress, the stdio loop
        writes a `notifications/progress` JSON-RPC message to stdout
        BEFORE the final tools/call result.

        We register a fake tool by monkey-patching ``execute_tool``
        itself so we can assert the wire format without spinning up
        a slow real estimator.
        """
        from statspai.agent import _runner

        def _fake_tool(name, arguments, *, data=None, detail="agent",
                        result_id=None, as_handle=False):
            # Emit a couple of progress events
            _runner.progress(0.25, total=1.0, message="warmup")
            _runner.progress(0.75, total=1.0, message="finalising")
            return {"value": 99}

        from statspai.agent import mcp_server
        monkeypatch.setattr(mcp_server, "execute_tool", _fake_tool)

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "regress",
                "arguments": {"formula": "y ~ x"},
                "_meta": {"progressToken": "tkn-xyz"},
            },
        }
        stdin = io.StringIO(json.dumps(request) + "\n")
        stdout = io.StringIO()
        mcp_serve_stdio(stdin=stdin, stdout=stdout)

        out = stdout.getvalue().strip().splitlines()
        # Expect: 2 progress notifications + 1 final result
        assert len(out) == 3, out
        prog_msgs = [json.loads(line) for line in out[:-1]]
        for m in prog_msgs:
            assert m["jsonrpc"] == "2.0"
            assert m["method"] == "notifications/progress"
            assert m["params"]["progressToken"] == "tkn-xyz"
        result_msg = json.loads(out[-1])
        assert result_msg["id"] == 1
        body = json.loads(result_msg["result"]["content"][0]["text"])
        assert body == {"value": 99}

    def test_no_meta_token_means_no_progress_emitted(self, monkeypatch):
        from statspai.agent import _runner

        called = {"n": 0}

        def _fake_tool(name, arguments, *, data=None, detail="agent",
                        result_id=None, as_handle=False):
            _runner.progress(0.5, total=1.0)
            called["n"] += 1
            return {"value": 1}

        from statspai.agent import mcp_server
        monkeypatch.setattr(mcp_server, "execute_tool", _fake_tool)

        request = {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "regress", "arguments": {"formula": "y ~ x"}},
        }
        stdin = io.StringIO(json.dumps(request) + "\n")
        stdout = io.StringIO()
        mcp_serve_stdio(stdin=stdin, stdout=stdout)
        out = stdout.getvalue().strip().splitlines()
        # Single line: just the response, no progress notifications.
        assert len(out) == 1
        assert called["n"] == 1


# ----------------------------------------------------------------------
# Tool timeout RPC integration
# ----------------------------------------------------------------------

class TestRpcTimeout:
    def test_slow_tool_times_out_with_clean_error(self, monkeypatch):
        # Force a 0.05 s timeout for this test.
        monkeypatch.setenv(TOOL_TIMEOUT_ENV, "0.05")

        def _slow_tool(name, arguments, *, data=None, detail="agent",
                        result_id=None, as_handle=False):
            time.sleep(1.0)
            return {"value": "never"}

        from statspai.agent import mcp_server
        monkeypatch.setattr(mcp_server, "execute_tool", _slow_tool)

        msg = json.loads(mcp_handle_request(json.dumps({
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {"name": "regress",
                        "arguments": {"formula": "y ~ x"}},
        })))
        # Timeout triggers a -32000 generic server error with the
        # readable message including the env-var name.
        assert "error" in msg
        assert "timeout" in msg["error"]["message"].lower()
        assert TOOL_TIMEOUT_ENV in msg["error"]["message"]
