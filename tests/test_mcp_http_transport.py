"""Streamable HTTP transport (``statspai-mcp --transport http``).

Each test talks to a real server on a free loopback port through
``http.client``; nothing is mocked.
"""

from __future__ import annotations

import http.client
import json
import threading
import time

import pytest

from statspai.agent import _http_transport as ht
from statspai.agent import _runner
from statspai.agent import mcp_server as m

CSV = "y,x\n" + "".join(f"{1 + 2 * i + (i % 3)},{i}\n" for i in range(30))
REGRESS = {"formula": "y ~ x", "data_csv": CSV, "detail": "minimal"}


class Client:
    def __init__(self, server, token=None):
        self.port = server.server_address[1]
        self.token = token
        self.session = None

    def request(self, method, body=None, headers=None, path="/mcp"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if self.session:
            h["Mcp-Session-Id"] = self.session
        h.update(headers or {})
        payload = None
        if body is not None:
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        try:
            conn.request(method, path, body=payload, headers=h)
            resp = conn.getresponse()
            return resp.status, dict(resp.getheaders()), resp.read()
        finally:
            conn.close()

    def rpc(self, method, params=None, request_id=1, headers=None):
        status, hdrs, raw = self.request(
            "POST",
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            },
            headers=headers,
        )
        return status, hdrs, (json.loads(raw) if raw else None)

    def initialize(self):
        status, hdrs, msg = self.rpc("initialize", {"protocolVersion": "2025-06-18"})
        assert status == 200, msg
        self.session = hdrs["Mcp-Session-Id"]
        return msg

    def call(self, name, arguments, **kw):
        return self.rpc("tools/call", {"name": name, "arguments": arguments}, **kw)


def _serve(**kwargs):
    server = ht.make_http_server("127.0.0.1", 0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def server():
    srv, thread = _serve()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=10)


def test_handshake_issues_a_session_and_the_same_surface_as_stdio(server):
    c = Client(server)
    init = c.initialize()["result"]
    assert init["protocolVersion"] == "2025-06-18"
    assert init["serverInfo"]["name"] == m.SERVER_NAME
    assert len(c.session) >= 24

    status, _, body = c.request(
        "POST", {"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    assert (status, body) == (202, b"")

    status, _, listed = c.rpc("tools/list")
    assert status == 200
    over_stdio = json.loads(
        m.handle_request('{"jsonrpc":"2.0","id":1,"method":"tools/list"}')
    )
    assert listed == over_stdio

    status, _, pong = c.rpc("ping", request_id="abc")
    assert (status, pong) == (200, {"jsonrpc": "2.0", "id": "abc", "result": {}})


def test_tool_call_returns_the_numbers_handle_request_returns(server):
    c = Client(server)
    c.initialize()
    status, hdrs, msg = c.call("regress", REGRESS)
    assert status == 200 and hdrs["Content-Type"] == "application/json"
    direct = json.loads(
        m.handle_request(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "regress", "arguments": REGRESS},
                }
            )
        )
    )
    assert msg["result"]["structuredContent"] == direct["result"]["structuredContent"]
    assert msg["result"]["structuredContent"]["n_obs"] == 30


def test_handles_chain_across_requests(server):
    c = Client(server)
    c.initialize()
    _, _, fit = c.call("regress", dict(REGRESS, as_handle=True))
    rid = fit["result"]["structuredContent"]["result_id"]
    _, _, brief = c.call("brief_result", {"result_id": rid}, request_id=2)
    assert not brief["result"].get("isError"), brief
    over_http = brief["result"]["structuredContent"]
    assert over_http["result_id"] == rid
    # The follow-up read the fit made two requests earlier: same summary
    # as asking the server in-process, and it is the 30-row regression.
    direct = json.loads(
        m.handle_request(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "brief_result", "arguments": {"result_id": rid}},
                }
            )
        )
    )["result"]["structuredContent"]
    assert over_http == direct
    assert "N=30" in over_http["brief"]
    assert fit["result"]["structuredContent"]["n_obs"] == 30


def _events(raw):
    out = []
    for block in raw.decode("utf-8").split("\n\n"):
        lines = [x for x in block.splitlines() if x]
        if not lines:
            continue
        assert lines[0] == "event: message", block
        assert lines[1].startswith("data: "), block
        out.append(json.loads(lines[1][len("data: ") :]))
    return out


def test_progress_token_and_sse_accept_stream_progress_then_the_result(
    server, monkeypatch
):
    def slow_tool(name, arguments, **kw):
        for i in range(3):
            _runner.progress(i + 1, total=3, message=f"step {i + 1}")
        return {"estimate": 1.0}

    monkeypatch.setattr(m, "execute_tool", slow_tool)
    c = Client(server)
    c.initialize()
    body = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "bibtex",
            "arguments": {"keys": []},
            "_meta": {"progressToken": "tok"},
        },
    }
    status, hdrs, raw = c.request(
        "POST", body, headers={"Accept": "application/json, text/event-stream"}
    )
    assert status == 200 and hdrs["Content-Type"] == "text/event-stream"
    events = _events(raw)
    progress = [e for e in events if e.get("method") == "notifications/progress"]
    assert [e["params"]["progress"] for e in progress] == [1, 2, 3]
    assert all(e["params"]["progressToken"] == "tok" for e in progress)
    assert events[-1]["id"] == 7
    assert events[-1]["result"]["structuredContent"]["estimate"] == 1.0

    # Same call without the SSE Accept: one JSON body, no progress.
    status, hdrs, raw = c.request("POST", body)
    assert hdrs["Content-Type"] == "application/json"
    assert json.loads(raw)["result"]["structuredContent"]["estimate"] == 1.0


def test_cancel_notification_stops_the_running_call(server, monkeypatch):
    started = threading.Event()

    def hanging_tool(name, arguments, **kw):
        started.set()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _runner.check_cancelled()
            time.sleep(0.02)
        return {"estimate": 0.0}

    monkeypatch.setattr(m, "execute_tool", hanging_tool)
    c = Client(server)
    c.initialize()
    answer = {}

    def _call():
        answer["msg"] = c.call("bibtex", {"keys": []}, request_id=41)

    t = threading.Thread(target=_call)
    t0 = time.monotonic()
    t.start()
    assert started.wait(20)
    status, _, _ = c.request(
        "POST",
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 41},
        },
    )
    assert status == 202
    t.join(20)
    assert not t.is_alive() and time.monotonic() - t0 < 20
    result = answer["msg"][2]["result"]
    assert (
        result["isError"] and result["structuredContent"]["error_kind"] == "cancelled"
    )
    assert server.inflight == {}


def test_a_full_queue_answers_server_busy_at_once(monkeypatch):
    monkeypatch.setenv(m.MAX_QUEUED_CALLS_ENV, "1")
    srv, thread = _serve(workers=1)
    release = threading.Event()
    running = threading.Event()

    def blocking_tool(name, arguments, **kw):
        running.set()
        while not release.is_set():
            _runner.check_cancelled()
            time.sleep(0.02)
        return {"estimate": 2.0}

    monkeypatch.setattr(m, "execute_tool", blocking_tool)
    try:
        c = Client(srv)
        c.initialize()
        answers = {}

        def _call(i):
            answers[i] = c.call("bibtex", {"keys": []}, request_id=i)[2]

        threads = [threading.Thread(target=_call, args=(i,)) for i in (1, 2)]
        threads[0].start()
        assert running.wait(20)
        threads[1].start()  # waits for the one slot
        deadline = time.monotonic() + 20
        while len(srv.inflight) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        status, _, third = c.call("bibtex", {"keys": []}, request_id=3)
        assert status == 200
        busy = third["result"]["structuredContent"]
        assert busy["error_kind"] == "server_busy" and busy["retryable"] is True
        release.set()
        for t in threads:
            t.join(20)
        assert all(
            answers[i]["result"]["structuredContent"]["estimate"] == 2.0 for i in (1, 2)
        )
    finally:
        release.set()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)


def test_sessions_end_and_unknown_ones_are_404(server):
    c = Client(server)
    c.initialize()
    sid = c.session
    assert c.rpc("ping")[0] == 200
    assert c.request("DELETE")[0] == 204
    assert c.rpc("ping")[0] == 404
    assert c.request("DELETE")[0] == 404
    # No session header at all is accepted (stateless clients).
    c.session = None
    assert c.rpc("ping")[0] == 200
    # A stale id may still re-initialise.
    c.session = sid
    assert c.rpc("initialize", {"protocolVersion": "2025-06-18"})[0] == 200


def test_requests_the_endpoint_does_not_serve(server):
    c = Client(server)
    assert c.request("GET")[0] == 405
    assert (
        c.request("POST", {"jsonrpc": "2.0", "id": 1, "method": "ping"}, path="/x")[0]
        == 404
    )
    status, _, raw = c.request("POST", b"{not json")
    assert status == 400 and json.loads(raw)["error"]["code"] == -32700
    status, _, raw = c.request("POST", [{"jsonrpc": "2.0", "id": 1, "method": "ping"}])
    assert status == 400 and "batches" in json.loads(raw)["error"]["message"]
    status, _, _ = c.rpc("ping", headers={"MCP-Protocol-Version": "1999-01-01"})
    assert status == 400
    assert c.rpc("ping", headers={"MCP-Protocol-Version": "2025-03-26"})[0] == 200
    status, _, msg = c.rpc("no/such/method")
    assert status == 200 and msg["error"]["code"] == -32601


def test_origin_and_host_are_checked_on_a_loopback_bind(server):
    c = Client(server)
    assert c.rpc("ping", headers={"Origin": "http://evil.example"})[0] == 403
    assert c.rpc("ping", headers={"Origin": "http://localhost.evil.example"})[0] == 403
    assert c.rpc("ping", headers={"Host": "evil.example"})[0] == 403
    status, hdrs, _ = c.rpc("ping", headers={"Origin": "http://localhost:5173"})
    assert status == 200
    assert hdrs["Access-Control-Allow-Origin"] == "http://localhost:5173"
    status, hdrs, _ = c.request("OPTIONS", headers={"Origin": "http://127.0.0.1:6274"})
    assert status == 204 and "Mcp-Session-Id" in hdrs["Access-Control-Allow-Headers"]
    assert c.request("OPTIONS", headers={"Origin": "http://evil.example"})[0] == 403


def test_listed_origins_are_allowed():
    srv, thread = _serve(allowed_origins=["https://app.example.org/"])
    try:
        c = Client(srv)
        assert c.rpc("ping", headers={"Origin": "https://app.example.org"})[0] == 200
        assert c.rpc("ping", headers={"Origin": "https://other.example.org"})[0] == 403
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)


def test_bearer_token_is_required_when_configured():
    srv, thread = _serve(token="s3cret-token")
    try:
        status, hdrs, _ = Client(srv).rpc("ping")
        assert status == 401 and hdrs["WWW-Authenticate"] == "Bearer"
        assert Client(srv, token="wrong").rpc("ping")[0] == 401
        assert Client(srv, token="s3cret-token-and-more").rpc("ping")[0] == 401
        assert Client(srv, token="s3cret-token").rpc("ping")[0] == 200
        # Authentication comes before anything is parsed or run.
        assert Client(srv).request("POST", b"{not json")[0] == 401
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)


def test_a_public_bind_without_a_token_is_refused(monkeypatch):
    monkeypatch.delenv(ht.HTTP_TOKEN_ENV, raising=False)
    with pytest.raises(ValueError, match=ht.HTTP_TOKEN_ENV):
        ht.make_http_server("0.0.0.0", 0)
    monkeypatch.setenv(ht.HTTP_TOKEN_ENV, "from-env")
    srv = ht.make_http_server("0.0.0.0", 0)
    try:
        assert srv.token == "from-env" and not srv.loopback
    finally:
        srv.server_close()


def test_oversized_bodies_are_refused_unread(monkeypatch):
    monkeypatch.setenv(m.MAX_REQUEST_BYTES_ENV, "2000")
    srv, thread = _serve()
    try:
        c = Client(srv)
        big = dict(REGRESS, data_csv="y,x\n" + "1,1\n" * 2000)
        status, _, msg = c.call("regress", big)
        assert status == 413 and "data_id" in msg["error"]["message"]
        assert c.rpc("ping")[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=10)


def test_sampling_is_off_under_http(server):
    from statspai.agent import _sampling

    c = Client(server)
    status, _, _ = c.rpc(
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {"sampling": {}}},
    )
    assert status == 200
    assert _sampling._CAPABILITY is False


def test_the_source_revision_is_read_before_any_request_thread(monkeypatch):
    """A fit on a request thread must not be the first to run ``git``.

    ``source_revision`` forks ``git`` once and caches the answer; forked
    from a tool thread after numerical code ran, the child can crash on
    macOS and every result then records no revision. The server reads it
    while it is still single-threaded.
    """
    from statspai import _build_info

    _build_info.source_revision.cache_clear()
    calls = []
    real = _build_info._git

    def recording_git(root, *args):
        calls.append(threading.current_thread() is threading.main_thread())
        return real(root, *args)

    monkeypatch.setattr(_build_info, "_git", recording_git)
    srv = ht.make_http_server("127.0.0.1", 0)
    srv.server_close()
    try:
        expected = _build_info.source_revision()
        if _build_info._checkout_root(_build_info._PKG_DIR) is not None:
            assert calls and all(calls), calls
        assert expected == _build_info.source_revision()
    finally:
        _build_info.source_revision.cache_clear()
