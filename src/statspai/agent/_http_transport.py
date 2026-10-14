"""Streamable HTTP transport for the StatsPAI MCP server.

``statspai-mcp --transport http`` serves the same JSON-RPC surface as the
stdio loop on one endpoint (``/mcp`` by default), following the MCP
"Streamable HTTP" transport (protocol revisions ``2025-03-26`` and
``2025-06-18``). Standard library only.

What a client sees
------------------
* ``POST /mcp`` with one JSON-RPC message.

  - A request is answered with ``application/json``. A ``tools/call`` that
    carries ``_meta.progressToken`` from a client that accepts
    ``text/event-stream`` is answered with an SSE stream instead: progress
    notifications as they happen, then the response, then the stream
    closes.
  - A notification or a reply is answered ``202 Accepted`` with no body.
    ``notifications/cancelled`` stops the named in-flight call of the
    same session.
  - JSON-RPC batches (arrays) are refused: the current revision removed
    them.

* ``initialize`` returns an ``Mcp-Session-Id`` header. Later requests may
  send it back; an id the server does not know is ``404`` (the client
  re-initialises), no id at all is accepted. ``DELETE /mcp`` ends a
  session.
* ``GET /mcp`` is ``405``: the server never opens a stream of its own, so
  there is no server-to-client request channel and ``sampling`` is off
  under this transport (``interpret_result`` falls back as it does for a
  client without the capability).

Security
--------
* Binds ``127.0.0.1`` unless told otherwise. A non-loopback bind is
  refused without ``STATSPAI_MCP_HTTP_TOKEN``; with the token set every
  request needs ``Authorization: Bearer <token>``.
* ``Origin`` is validated on every request (the DNS-rebinding defence the
  spec requires): absent, loopback, or listed in
  ``STATSPAI_MCP_HTTP_ALLOWED_ORIGINS`` (comma-separated). On a loopback
  bind the ``Host`` header must be a loopback name too.
* No TLS here. Put the server behind a reverse proxy that terminates TLS
  before exposing it beyond the machine.
* Handles are process-wide, not per session: any authenticated client
  that knows a ``data_id`` / ``result_id`` can use it. Ids are random, but
  one server is one trust domain. ``data_path`` reads files on the
  *server*; set ``STATSPAI_MCP_DATA_ROOTS`` on any shared deployment.

Concurrency follows the stdio loop: ``tools/call`` requests share a pool
of ``STATSPAI_MCP_WORKERS`` slots (default 1) with the same queue,
orphan and wait limits, and everything else is answered at once.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

HTTP_TOKEN_ENV = "STATSPAI_MCP_HTTP_TOKEN"
HTTP_ALLOWED_ORIGINS_ENV = "STATSPAI_MCP_HTTP_ALLOWED_ORIGINS"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8765
DEFAULT_HTTP_PATH = "/mcp"

#: Sessions remembered at once; the oldest is forgotten beyond this.
MAX_SESSIONS = 1024

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback_host(host: str) -> bool:
    return host.strip("[]").lower() in _LOOPBACK_HOSTS


def _host_only(value: str) -> str:
    """``host`` of a ``Host`` header or origin authority, port stripped."""
    value = value.strip()
    if value.startswith("["):  # [::1]:8765
        return value[1:].split("]", 1)[0].lower()
    return value.rsplit(":", 1)[0].lower() if ":" in value else value.lower()


class _SseSink:
    """Writes JSON-RPC lines as SSE ``message`` events on one response."""

    def __init__(self, wfile: Any, on_broken: Any) -> None:
        self._wfile = wfile
        self._lock = threading.Lock()
        self._on_broken = on_broken
        self.broken = False

    def write(self, text: str) -> int:
        for line in text.splitlines():
            if line.strip():
                self.write_line(line)
        return len(text)

    def write_line(self, line: str) -> None:
        with self._lock:
            if self.broken:
                return
            try:
                self._wfile.write(b"event: message\ndata: ")
                self._wfile.write(line.encode("utf-8"))
                self._wfile.write(b"\n\n")
                self._wfile.flush()
            except (OSError, ValueError):
                # The client went away: stop the computation it asked for.
                self.broken = True
                self._on_broken()

    def flush(self) -> None:
        return None


class McpHttpServer(ThreadingHTTPServer):
    """One MCP endpoint; state shared by its request threads."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: Tuple[str, int],
        *,
        path: str = DEFAULT_HTTP_PATH,
        token: Optional[str] = None,
        allowed_origins: Optional[List[str]] = None,
        workers: Optional[int] = None,
    ) -> None:
        from . import mcp_server as _srv

        host = address[0]
        if not is_loopback_host(host) and not token:
            raise ValueError(
                f"refusing to serve MCP on {host!r} without a bearer token: "
                f"set {HTTP_TOKEN_ENV} (clients then send "
                "'Authorization: Bearer <token>'), or bind 127.0.0.1."
            )
        self.mcp_path = "/" + path.strip("/")
        self.token = token or None
        self.loopback = is_loopback_host(host)
        self.allowed_origins = {
            o.strip().rstrip("/").lower() for o in (allowed_origins or []) if o.strip()
        }
        self.n_workers = workers if workers else _srv._worker_count()
        self.max_queued = _srv._env_limit(
            _srv.MAX_QUEUED_CALLS_ENV, _srv.DEFAULT_MAX_QUEUED_CALLS
        )
        self.max_orphans = _srv._env_limit(
            _srv.MAX_ORPHANED_CALLS_ENV, _srv.DEFAULT_MAX_ORPHANED_CALLS
        )
        self.max_wait = _srv._env_limit(
            _srv.MAX_QUEUE_SECONDS_ENV, _srv.DEFAULT_MAX_QUEUE_SECONDS
        )
        self.max_request = _srv._env_limit(
            _srv.MAX_REQUEST_BYTES_ENV, _srv.DEFAULT_MAX_REQUEST_BYTES
        )
        self.slots = threading.Semaphore(self.n_workers)
        self.inflight: Dict[Tuple[str, str], threading.Event] = {}
        self.state_lock = threading.Lock()
        #: Serialises everything that is not a ``tools/call`` (the stdio
        #: loop answers those on one thread).
        self.inline_lock = threading.Lock()
        self.sessions: "OrderedDict[str, float]" = OrderedDict()
        # Build the tool list here, on the thread that starts the server.
        # Left to the first ``tools/list`` it would import the whole
        # catalogue on a request thread: the first client waits for it,
        # and on macOS native libraries first loaded off the main thread
        # leave the process unable to fork safely afterwards.
        _srv._tools_list_result_json()
        _srv._prepare_for_threads()
        super().__init__(address, _McpHandler)

    def server_bind(self) -> None:
        # ``HTTPServer.server_bind`` resolves the host with
        # ``socket.getfqdn``: a DNS lookup at start-up that can take
        # seconds and, on macOS, leaves the process unable to fork
        # safely afterwards. The name is not used for anything here.
        import socketserver

        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)

    # -- sessions ------------------------------------------------------ #

    def new_session(self) -> str:
        sid = secrets.token_urlsafe(24)
        with self.state_lock:
            self.sessions[sid] = time.time()
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
        return sid

    def has_session(self, sid: str) -> bool:
        with self.state_lock:
            return sid in self.sessions

    def end_session(self, sid: str) -> bool:
        with self.state_lock:
            existed = self.sessions.pop(sid, None) is not None
            events = [ev for (s, _), ev in self.inflight.items() if s == sid]
        for ev in events:
            ev.set()
        return existed

    def origin_allowed(self, origin: str) -> bool:
        norm = origin.strip().rstrip("/").lower()
        if norm in self.allowed_origins:
            return True
        parts = urlsplit(norm)
        return parts.scheme in ("http", "https") and (parts.hostname or "") in (
            _LOOPBACK_HOSTS
        )


class _McpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "statspai-mcp"
    sys_version = ""
    server: McpHttpServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        from .mcp_server import _debug_enabled

        if _debug_enabled():
            sys.stderr.write("statspai-mcp http: " + (format % args) + "\n")

    # -- small response helpers ---------------------------------------- #

    def _cors(self) -> List[Tuple[str, str]]:
        origin = self.headers.get("Origin")
        if not origin:
            return []
        return [
            ("Access-Control-Allow-Origin", origin),
            ("Access-Control-Expose-Headers", "Mcp-Session-Id"),
            ("Vary", "Origin"),
        ]

    def _send(
        self,
        status: int,
        body: bytes = b"",
        *,
        content_type: str = "application/json",
        headers: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.send_response(status)
        if body:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or []) + self._cors():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_rpc_error(self, status: int, code: int, message: str) -> None:
        from .mcp_server import _jsonrpc_error

        self._send(status, _jsonrpc_error(None, code, message).encode("utf-8"))

    def _gate(self) -> bool:
        """Path, authentication, Origin and Host checks; ``False`` = answered."""
        srv = self.server
        if urlsplit(self.path).path.rstrip("/") != srv.mcp_path.rstrip("/"):
            self._send(404)
            return False
        if srv.token is not None:
            sent = self.headers.get("Authorization", "")
            scheme, _, value = sent.partition(" ")
            ok = scheme.lower() == "bearer" and hmac.compare_digest(
                value.strip().encode("utf-8"), srv.token.encode("utf-8")
            )
            if not ok:
                self._send(401, headers=[("WWW-Authenticate", "Bearer")])
                return False
        origin = self.headers.get("Origin")
        if origin and not srv.origin_allowed(origin):
            self._send_rpc_error(403, -32600, f"Origin {origin!r} is not allowed.")
            return False
        if srv.loopback:
            host = self.headers.get("Host", "")
            if host and _host_only(host) not in _LOOPBACK_HOSTS:
                self._send_rpc_error(
                    403, -32600, f"Host {host!r} is not a loopback name."
                )
                return False
        return True

    # -- methods -------------------------------------------------------- #

    def do_OPTIONS(self) -> None:  # noqa: N802 - CORS preflight
        srv = self.server
        origin = self.headers.get("Origin")
        if urlsplit(self.path).path.rstrip("/") != srv.mcp_path.rstrip("/"):
            self._send(404)
            return
        if origin and not srv.origin_allowed(origin):
            self._send(403)
            return
        self._send(
            204,
            headers=[
                ("Access-Control-Allow-Methods", "POST, DELETE, OPTIONS"),
                (
                    "Access-Control-Allow-Headers",
                    "Content-Type, Accept, Authorization, Mcp-Session-Id, "
                    "MCP-Protocol-Version",
                ),
                ("Access-Control-Max-Age", "600"),
            ],
        )

    def do_GET(self) -> None:  # noqa: N802
        if self._gate():
            self._send(405, headers=[("Allow", "POST, DELETE, OPTIONS")])

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._gate():
            return
        sid = self.headers.get("Mcp-Session-Id")
        if not sid or not self.server.end_session(sid):
            self._send(404)
            return
        self._send(204)

    def do_POST(self) -> None:  # noqa: N802
        from . import _sampling
        from .mcp_server import SUPPORTED_PROTOCOL_VERSIONS, handle_request

        srv = self.server
        if not self._gate():
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_rpc_error(411, -32600, "Content-Length is required.")
            return
        if length < 0 or (srv.max_request is not None and length > srv.max_request):
            self.close_connection = True
            self._send_rpc_error(
                413,
                -32600,
                f"Request body of {length} bytes exceeds the server limit; "
                "pass data by data_path or a data_id handle instead of inline.",
            )
            return
        raw = self.rfile.read(length)
        version = self.headers.get("MCP-Protocol-Version")
        if version and version not in SUPPORTED_PROTOCOL_VERSIONS:
            self._send_rpc_error(
                400, -32600, f"Unsupported MCP-Protocol-Version: {version!r}."
            )
            return
        try:
            line = raw.decode("utf-8")
            msg = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_rpc_error(400, -32700, f"Parse error: {exc}")
            return
        if not isinstance(msg, dict):
            self._send_rpc_error(
                400,
                -32600,
                "Invalid Request: send one JSON-RPC message per POST "
                "(batches are not supported).",
            )
            return

        sid = self.headers.get("Mcp-Session-Id") or ""
        method = msg.get("method")
        if sid and method != "initialize" and not srv.has_session(sid):
            self._send_rpc_error(404, -32600, "Unknown or ended session.")
            return

        if (
            not isinstance(method, str)
            or "id" not in msg
            or method.startswith("notifications/")
        ):
            if method == "notifications/cancelled":
                self._cancel(sid, msg.get("params"))
            self._send(202)
            return

        if method == "tools/call":
            self._tools_call(sid, msg, line)
            return

        extra: List[Tuple[str, str]] = []
        with srv.inline_lock:
            response = handle_request(line)
            if method == "initialize":
                # No server-to-client request channel under this transport.
                _sampling.set_capability(False)
        if method == "initialize" and "result" in json.loads(response or "{}"):
            extra.append(("Mcp-Session-Id", srv.new_session()))
        self._send(200, (response or "").encode("utf-8"), headers=extra)

    # -- tools/call ----------------------------------------------------- #

    def _cancel(self, sid: str, params: Any) -> None:
        from .mcp_server import _request_key

        if not isinstance(params, dict) or "requestId" not in params:
            return
        with self.server.state_lock:
            ev = self.server.inflight.get((sid, _request_key(params["requestId"])))
        if ev is not None:
            ev.set()

    def _wants_stream(self, msg: Dict[str, Any]) -> bool:
        params = msg.get("params")
        meta = params.get("_meta") if isinstance(params, dict) else None
        has_token = isinstance(meta, dict) and meta.get("progressToken") is not None
        return has_token and "text/event-stream" in self.headers.get("Accept", "")

    def _tools_call(self, sid: str, msg: Dict[str, Any], line: str) -> None:
        from . import _runner
        from . import mcp_server as _srv

        srv = self.server
        request_id = msg["id"]
        key = (sid, _srv._request_key(request_id))
        cancel = threading.Event()
        with srv.state_lock:
            duplicate = key in srv.inflight
            busy = (
                None
                if duplicate
                else _srv._admission_refusal(
                    len(srv.inflight), srv.n_workers, srv.max_queued, srv.max_orphans
                )
            )
            if not duplicate and busy is None:
                srv.inflight[key] = cancel
        if duplicate:
            self._send(
                400,
                _srv._jsonrpc_error(
                    request_id,
                    -32600,
                    "Invalid Request: a tools/call with this id is still in flight.",
                ).encode("utf-8"),
            )
            return
        if busy is not None:
            self._send(200, _srv._jsonrpc_result(request_id, busy).encode("utf-8"))
            return

        sink: Optional[_SseSink] = None
        try:
            queued_at = time.monotonic()
            while not srv.slots.acquire(timeout=0.2):
                waited = time.monotonic() - queued_at
                if cancel.is_set():
                    self._send(
                        200,
                        _srv._jsonrpc_result(
                            request_id,
                            _srv._tool_error_result(
                                "cancelled", "The client cancelled this tools/call."
                            ),
                        ).encode("utf-8"),
                    )
                    return
                if srv.max_wait is not None and waited > srv.max_wait:
                    self._send(
                        200,
                        _srv._jsonrpc_result(
                            request_id,
                            _srv._tool_error_result(
                                "server_busy",
                                f"This call waited {waited:.0f}s in the queue, "
                                f"past the {srv.max_wait}s limit; it was not "
                                "started.",
                                hint=(
                                    "Retry now that earlier calls have finished. "
                                    f"{_srv.MAX_QUEUE_SECONDS_ENV} sets the limit."
                                ),
                                queued_seconds=round(waited, 1),
                                retryable=True,
                            ),
                        ).encode("utf-8"),
                    )
                    return
            try:
                if self._wants_stream(msg):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    for k, v in self._cors():
                        self.send_header(k, v)
                    self.end_headers()
                    self.close_connection = True
                    sink = _SseSink(self.wfile, cancel.set)
                    _srv._bind_progress_sink(sink)
                _runner.set_cancel_event(cancel)
                try:
                    response = _srv.handle_request(line) or ""
                finally:
                    _runner.set_cancel_event(None)
                    _srv._bind_progress_sink(None)
            finally:
                srv.slots.release()
            if sink is not None:
                sink.write_line(response)
            else:
                self._send(200, response.encode("utf-8"))
        finally:
            with srv.state_lock:
                srv.inflight.pop(key, None)


def make_http_server(
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    *,
    path: str = DEFAULT_HTTP_PATH,
    token: Optional[str] = None,
    allowed_origins: Optional[List[str]] = None,
    workers: Optional[int] = None,
) -> McpHttpServer:
    """Build (but do not start) the HTTP server; ``port=0`` picks a free one.

    ``token`` / ``allowed_origins`` default to ``STATSPAI_MCP_HTTP_TOKEN``
    / ``STATSPAI_MCP_HTTP_ALLOWED_ORIGINS``. Raises ``ValueError`` for a
    non-loopback ``host`` without a token.
    """
    if token is None:
        token = os.environ.get(HTTP_TOKEN_ENV) or None
    if allowed_origins is None:
        allowed_origins = os.environ.get(HTTP_ALLOWED_ORIGINS_ENV, "").split(",")
    return McpHttpServer(
        (host, port),
        path=path,
        token=token,
        allowed_origins=allowed_origins,
        workers=workers,
    )


def serve_http(
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    *,
    path: str = DEFAULT_HTTP_PATH,
    token: Optional[str] = None,
    allowed_origins: Optional[List[str]] = None,
    workers: Optional[int] = None,
) -> None:
    """Serve MCP over Streamable HTTP until interrupted."""
    from . import _sampling
    from ._data_loader import DATA_ROOTS_ENV

    server = make_http_server(
        host,
        port,
        path=path,
        token=token,
        allowed_origins=allowed_origins,
        workers=workers,
    )
    _sampling.set_writer(None)
    _sampling.set_capability(False)
    bound_host, bound_port = (str(x) for x in server.server_address[:2])
    print(
        f"statspai-mcp: serving MCP on http://{bound_host}:{bound_port}"
        f"{server.mcp_path} ("
        + ("bearer token required" if server.token else "no authentication")
        + ")",
        file=sys.stderr,
    )
    if not server.loopback and not os.environ.get(DATA_ROOTS_ENV):
        print(
            f"statspai-mcp: warning: {DATA_ROOTS_ENV} is not set, so any "
            "authenticated client can read every file this process can.",
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
