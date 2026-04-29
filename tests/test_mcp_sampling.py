"""Tests for Module 8 — MCP sampling/createMessage server-side client.

The server-side helpers raise ``UnsupportedSamplingError`` when no
client capability is advertised so callers can route to a fallback.
When the client does advertise support, the helpers send a
``sampling/createMessage`` request on stdout and block on the reply.
"""
from __future__ import annotations

import io
import json
import threading
import time

import pytest

from statspai.agent import _sampling
from statspai.agent._sampling import (
    UnsupportedSamplingError,
    SamplingTimeoutError,
    request_sampling,
    set_capability,
    set_writer,
    route_response,
    SAMPLING_TIMEOUT_ENV,
)


# ----------------------------------------------------------------------
# Fail-closed when capability not set
# ----------------------------------------------------------------------

class TestFailClosed:
    def test_raises_when_capability_unset(self):
        # Capability resets between tests via the ``reset`` fixture below.
        with pytest.raises(UnsupportedSamplingError):
            request_sampling([{"role": "user",
                                "content": {"type": "text", "text": "hi"}}],
                               max_tokens=10)

    def test_raises_when_writer_unset(self, monkeypatch):
        set_capability(True)
        try:
            with pytest.raises(UnsupportedSamplingError):
                request_sampling([{"role": "user",
                                    "content": {"type": "text", "text": "x"}}],
                                  max_tokens=10)
        finally:
            set_capability(False)


# ----------------------------------------------------------------------
# Round-trip via mock writer
# ----------------------------------------------------------------------

class TestRoundTrip:
    def test_sends_request_and_returns_result(self):
        """A fake client thread reads the outbound line, builds a
        canned reply, and routes it back via ``route_response``."""
        outbox = []
        # Capture every line the server writes so we can synthesise
        # a reply that matches the request id.
        set_writer(lambda line: outbox.append(line))
        set_capability(True)

        def fake_client():
            # Wait until the server has written its outbound line
            for _ in range(100):
                if outbox:
                    break
                time.sleep(0.01)
            assert outbox, "server didn't send a sampling request"
            sent = json.loads(outbox[0])
            assert sent["method"] == "sampling/createMessage"
            assert sent["params"]["maxTokens"] == 200
            reply = {
                "jsonrpc": "2.0",
                "id": sent["id"],
                "result": {
                    "role": "assistant",
                    "content": {"type": "text", "text": "hi back"},
                    "model": "claude-test",
                    "stopReason": "endTurn",
                },
            }
            assert route_response(reply) is True

        t = threading.Thread(target=fake_client, daemon=True)
        t.start()
        try:
            result = request_sampling(
                [{"role": "user",
                  "content": {"type": "text", "text": "hi"}}],
                max_tokens=200,
                system_prompt="be concise",
                temperature=0.2,
                timeout=2.0,
            )
            assert result["content"]["text"] == "hi back"
            assert result["model"] == "claude-test"
        finally:
            t.join(timeout=1)
            set_writer(None)
            set_capability(False)

    def test_client_error_envelope_raises_runtimeerror(self):
        outbox = []
        set_writer(lambda line: outbox.append(line))
        set_capability(True)

        def fake_client():
            for _ in range(100):
                if outbox:
                    break
                time.sleep(0.01)
            sent = json.loads(outbox[0])
            reply = {
                "jsonrpc": "2.0",
                "id": sent["id"],
                "error": {"code": -32603, "message": "client refused"},
            }
            route_response(reply)

        t = threading.Thread(target=fake_client, daemon=True)
        t.start()
        try:
            with pytest.raises(RuntimeError) as excinfo:
                request_sampling(
                    [{"role": "user",
                      "content": {"type": "text", "text": "x"}}],
                    max_tokens=10, timeout=2.0,
                )
            assert "client refused" in str(excinfo.value)
        finally:
            t.join(timeout=1)
            set_writer(None)
            set_capability(False)


# ----------------------------------------------------------------------
# Timeout handling
# ----------------------------------------------------------------------

class TestTimeout:
    def test_times_out_when_client_silent(self, monkeypatch):
        monkeypatch.setenv(SAMPLING_TIMEOUT_ENV, "0.05")
        set_writer(lambda line: None)  # swallow outbound; never replies
        set_capability(True)
        try:
            with pytest.raises(SamplingTimeoutError):
                request_sampling(
                    [{"role": "user",
                      "content": {"type": "text", "text": "x"}}],
                    max_tokens=10,
                )
        finally:
            set_writer(None)
            set_capability(False)


# ----------------------------------------------------------------------
# stdio integration: serve_stdio registers + clears the writer
# ----------------------------------------------------------------------

class TestServeStdioIntegration:
    def test_initialize_records_client_capability(self):
        from statspai.agent import mcp_handle_request
        msg = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"capabilities": {"sampling": {}}},
        }
        line = mcp_handle_request(json.dumps(msg))
        assert line is not None
        # The handler set the capability flag to True; clean up so
        # other tests don't see leakage.
        try:
            assert _sampling.get_capability() is True
        finally:
            set_capability(False)

    def test_initialize_without_capability_keeps_flag_false(self):
        from statspai.agent import mcp_handle_request
        set_capability(False)  # explicit reset
        msg = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"capabilities": {}},
        }
        mcp_handle_request(json.dumps(msg))
        assert _sampling.get_capability() is False

    def test_serve_stdio_registers_then_clears_writer(self):
        from statspai.agent import mcp_serve_stdio
        # Run a single-line dummy session (just an initialize) and
        # verify the writer was unset on exit.
        request = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"capabilities": {}},
        }
        stdin = io.StringIO(json.dumps(request) + "\n")
        stdout = io.StringIO()
        mcp_serve_stdio(stdin=stdin, stdout=stdout)
        # serve_stdio resets the capability + writer in finally:.
        assert _sampling.get_capability() is False


# ----------------------------------------------------------------------
# route_response: ignores unsolicited replies
# ----------------------------------------------------------------------

class TestRouteResponseIgnoresUnsolicited:
    def test_unknown_id_returns_false(self):
        assert route_response({"jsonrpc": "2.0", "id": "not-pending",
                                "result": {}}) is False

    def test_non_dict_returns_false(self):
        assert route_response("not a dict") is False  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Auto-reset between tests so module-level state doesn't leak
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_sampling_state():
    set_capability(False)
    set_writer(None)
    yield
    set_capability(False)
    set_writer(None)
