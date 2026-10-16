"""Per-request ("modern") MCP protocol revisions alongside the handshake ones.

MCP ``2026-07-28`` dropped the ``initialize`` handshake. Every request now
names its protocol version and the client's capabilities in ``params._meta``
(``io.modelcontextprotocol/protocolVersion`` and
``io.modelcontextprotocol/clientCapabilities``), servers MUST answer
``server/discover``, every result carries ``resultType``, and list / read
results carry the caching hints ``ttlMs`` and ``cacheScope``. A client that
speaks only that revision cannot use a server that knows only the
handshake: it probes with ``server/discover``, gets ``Method not found`` and
gives up.

This module makes the server *dual-era* in the sense of the specification
("Versioning and Compatibility: Backward Compatibility"):

* a request whose ``_meta`` carries a protocol version, or a
  ``server/discover`` request, is served statelessly under the per-request
  rules below;
* an ``initialize`` request selects the handshake revisions
  (``2025-11-25`` and earlier), handled as before by ``mcp_server``.

Nothing here changes what a handshake client sees. The per-request path
reuses the same method handlers; it only validates the metadata first and
decorates the result afterwards.

Requirements implemented, with the clause they come from:

* ``server/discover`` returns ``supportedVersions`` / ``capabilities`` /
  ``instructions`` and the server identity in ``_meta`` (server/discover).
* An unsupported version is ``-32022`` with ``data.supported`` and
  ``data.requested``; a request missing a required ``_meta`` field is
  ``-32602`` (Versioning; Overview, per-request protocol fields).
* Every result carries ``resultType: "complete"`` and
  ``_meta["io.modelcontextprotocol/serverInfo"]`` (Overview, ResultType and
  per-response protocol fields).
* ``server/discover``, ``tools/list``, ``prompts/list``, ``resources/list``,
  ``resources/templates/list`` and ``resources/read`` carry ``ttlMs`` and
  ``cacheScope`` (Caching).
* Resource-not-found is ``-32602`` (``-32002`` is reserved for the older
  revisions), and codes in the legacy ``-32000``..``-32019`` band are not
  emitted (Overview, error codes).
* ``ping`` and ``logging/setLevel`` do not exist in this revision and are
  ``-32601``. The server sends no ``notifications/message`` at all, so the
  per-request log level needs no handling.
* No server-to-client request (``sampling/createMessage``) is sent while a
  per-request call is being served: :func:`in_modern_request` makes the
  sampling helpers fall back, as they do for a client without the
  capability.
* ``subscriptions/listen`` is acknowledged with the subset of notification
  types the server honours. The tool, prompt and resource lists never
  change while a process runs, so that subset is empty and the stream only
  stays open until the client cancels it or the server shuts down
  (Subscriptions).
"""

from __future__ import annotations

import base64
import contextlib
import json
import threading
from typing import Any, Dict, Iterator, Optional, Tuple

#: Per-request protocol revisions this server implements, newest first.
MODERN_PROTOCOL_VERSIONS: Tuple[str, ...] = ("2026-07-28",)

META_PROTOCOL_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"
META_SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"

#: Error codes the 2026-07-28 revision defines (its reserved band).
HEADER_MISMATCH = -32020
MISSING_REQUIRED_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022

#: Methods of the per-request revision this server answers.
MODERN_METHODS = frozenset(
    {
        "server/discover",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/templates/list",
        "resources/read",
        "prompts/list",
        "prompts/get",
        "completion/complete",
        "subscriptions/listen",
    }
)

#: Methods whose results carry ``ttlMs`` / ``cacheScope``.
CACHEABLE_METHODS = frozenset(
    {
        "server/discover",
        "tools/list",
        "prompts/list",
        "resources/list",
        "resources/templates/list",
        "resources/read",
    }
)

#: Methods whose request names a target in ``params.name`` / ``params.uri``
#: (mirrored into the ``Mcp-Name`` header on Streamable HTTP).
NAMED_METHODS = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri"}

#: Freshness of the catalogue results. The tool, prompt and resource lists
#: are fixed for the life of the process, so an hour is safe; a client still
#: re-fetches after a server restart because the cache key includes the
#: server it talks to.
CATALOGUE_TTL_MS = 3_600_000

#: URIs whose contents are per-user data (live handles): private, and
#: immediately stale because a handle can be evicted from the cache.
_PRIVATE_URI_PREFIXES = ("statspai://data/", "statspai://result/", "statspai://handles")

_STATE = threading.local()


# --------------------------------------------------------------------- #
#  Classification and validation
# --------------------------------------------------------------------- #


def request_meta(params: Any) -> Optional[Dict[str, Any]]:
    """``params._meta`` when it is an object, else ``None``."""
    if not isinstance(params, dict):
        return None
    meta = params.get("_meta")
    return meta if isinstance(meta, dict) else None


def is_modern_request(method: Any, params: Any) -> bool:
    """Does this request select the per-request revision?

    ``server/discover`` exists only there, and any request that names a
    protocol version in ``_meta`` has opted in. An ``initialize`` request
    always selects the handshake, whatever its ``_meta`` says.
    """
    if method == "initialize":
        return False
    if method == "server/discover":
        return True
    meta = request_meta(params)
    return meta is not None and META_PROTOCOL_VERSION in meta


def all_supported_versions() -> Tuple[str, ...]:
    """Per-request revisions first, then the handshake revisions."""
    from .mcp_server import SUPPORTED_PROTOCOL_VERSIONS

    return MODERN_PROTOCOL_VERSIONS + tuple(SUPPORTED_PROTOCOL_VERSIONS)


def validate_request(method: Any, params: Any) -> Optional[Tuple[int, str, Any]]:
    """``(code, message, data)`` when the request must be refused, else ``None``."""
    meta = request_meta(params)
    version = meta.get(META_PROTOCOL_VERSION) if meta is not None else None
    if not isinstance(version, str) or not version:
        return (
            -32602,
            f"Invalid params: `_meta` must carry {META_PROTOCOL_VERSION!r} "
            f"(one of {list(MODERN_PROTOCOL_VERSIONS)}), or open the "
            "session with `initialize` for a handshake revision.",
            None,
        )
    if version not in MODERN_PROTOCOL_VERSIONS:
        hint = ""
        if version in all_supported_versions():
            hint = f" {version} is served after an `initialize` handshake."
        return (
            UNSUPPORTED_PROTOCOL_VERSION,
            f"Unsupported protocol version{hint}",
            {"supported": list(all_supported_versions()), "requested": version},
        )
    assert meta is not None
    if not isinstance(meta.get(META_CLIENT_CAPABILITIES), dict):
        return (
            -32602,
            f"Invalid params: `_meta` must carry {META_CLIENT_CAPABILITIES!r} "
            "(an object; {} declares no optional capabilities).",
            None,
        )
    if method not in MODERN_METHODS:
        return (-32601, f"Method not found: {method!r}", None)
    return None


def map_error_code(code: int) -> int:
    """An error code the per-request revision allows in place of ``code``.

    ``-32002`` (resource not found) became ``-32602``; the
    ``-32000``..``-32019`` band is grandfathered for old implementations and
    new ones should not use it, so a generic server failure is reported as
    ``-32603`` (internal error).
    """
    if code == -32002:
        return -32602
    if -32019 <= code <= -32000:
        return -32603
    return code


# --------------------------------------------------------------------- #
#  Request context
# --------------------------------------------------------------------- #


@contextlib.contextmanager
def modern_request() -> Iterator[None]:
    """Mark the current thread as serving a per-request call."""
    previous = getattr(_STATE, "modern", False)
    _STATE.modern = True
    try:
        yield
    finally:
        _STATE.modern = previous


def in_modern_request() -> bool:
    """Is the current thread serving a per-request (2026-07-28) call?"""
    return bool(getattr(_STATE, "modern", False))


# --------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------- #


def server_identity(full: bool = False) -> Dict[str, Any]:
    """The ``Implementation`` object for this server.

    Results carry the short form (name and version) in ``_meta``;
    ``server/discover`` and ``initialize`` carry the full one.
    """
    from .mcp_server import SERVER_NAME, SERVER_VERSION

    ident: Dict[str, Any] = {"name": SERVER_NAME, "version": SERVER_VERSION}
    if full:
        ident.update(
            {
                "title": "StatsPAI",
                "description": (
                    "Causal inference and econometrics for agents: DiD, IV, "
                    "RD, synthetic control, DML, matching, panel and "
                    "time-series estimators with structured results, and "
                    "parity tests against Stata and R where a reference "
                    "implementation exists."
                ),
                "websiteUrl": "https://github.com/brycewang-stanford/StatsPAI",
            }
        )
    return ident


def server_capabilities() -> Dict[str, Any]:
    """Capabilities advertised to per-request clients."""
    return {
        "tools": {"listChanged": False},
        "resources": {"subscribe": False, "listChanged": False},
        "prompts": {"listChanged": False},
        "completions": {},
    }


def cache_hints(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """``ttlMs`` / ``cacheScope`` for a cacheable result."""
    if method == "resources/read":
        uri = str(params.get("uri", ""))
        if uri.startswith(_PRIVATE_URI_PREFIXES):
            return {"ttlMs": 0, "cacheScope": "private"}
    return {"ttlMs": CATALOGUE_TTL_MS, "cacheScope": "public"}


def decorate_result(
    method: str, params: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    """Add the per-request fields to a result produced by a method handler."""
    out = dict(result)
    out["resultType"] = "complete"
    meta = dict(out.get("_meta") or {})
    meta.setdefault(META_SERVER_INFO, server_identity())
    out["_meta"] = meta
    if method in CACHEABLE_METHODS:
        out.update(cache_hints(method, params))
    return out


def decorate_result_json(method: str, params: Dict[str, Any], result_json: str) -> str:
    """:func:`decorate_result` for a result already encoded as a JSON object.

    ``tools/list`` is pre-encoded once per process (70 KB); splicing the
    extra members in avoids decoding and re-encoding it on every request.
    """
    extra: Dict[str, Any] = {
        "resultType": "complete",
        "_meta": {META_SERVER_INFO: server_identity()},
    }
    if method in CACHEABLE_METHODS:
        extra.update(cache_hints(method, params))
    body = result_json.rstrip()
    if not body.endswith("}"):
        raise ValueError("expected an encoded JSON object")
    inner = json.dumps(extra, separators=(",", ":"))[1:]
    if body == "{}":
        return "{" + inner
    return body[:-1] + "," + inner


def discover_result() -> Dict[str, Any]:
    """The ``server/discover`` result (undecorated)."""
    from .mcp_server import _SESSION_INSTRUCTIONS

    return {
        "supportedVersions": list(all_supported_versions()),
        "capabilities": server_capabilities(),
        "instructions": _SESSION_INSTRUCTIONS,
        "_meta": {META_SERVER_INFO: server_identity(full=True)},
    }


# --------------------------------------------------------------------- #
#  subscriptions/listen
# --------------------------------------------------------------------- #

#: Notification types this server can send on a listen stream. The lists
#: are static and no resource is ever updated in place, so none.
HONOURED_SUBSCRIPTIONS: Dict[str, Any] = {}


def listen_acknowledgement(request_id: Any) -> Dict[str, Any]:
    """``notifications/subscriptions/acknowledged`` for a listen request."""
    return {
        "jsonrpc": "2.0",
        "method": "notifications/subscriptions/acknowledged",
        "params": {
            "notifications": dict(HONOURED_SUBSCRIPTIONS),
            "_meta": {META_SUBSCRIPTION_ID: request_id},
        },
    }


def listen_teardown_result(request_id: Any) -> Dict[str, Any]:
    """The result that closes a listen stream when the server shuts down."""
    return {
        "resultType": "complete",
        "_meta": {
            META_SUBSCRIPTION_ID: request_id,
            META_SERVER_INFO: server_identity(),
        },
    }


# --------------------------------------------------------------------- #
#  Streamable HTTP header checks
# --------------------------------------------------------------------- #


def decode_header_value(value: str) -> str:
    """Undo the ``=?base64?...?=`` sentinel encoding of a header value."""
    if value.startswith("=?base64?") and value.endswith("?=") and len(value) >= 11:
        try:
            return base64.b64decode(value[9:-2], validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            raise ValueError("malformed base64 header value") from None
    return value


def header_mismatch(headers: Any, msg: Dict[str, Any]) -> Optional[str]:
    """Why the mirrored HTTP headers disagree with the body, or ``None``.

    ``headers`` is anything with a case-insensitive ``get`` (the request's
    ``email.message.Message``). Checks ``MCP-Protocol-Version`` against
    ``_meta``, ``Mcp-Method`` against ``method`` and, for the methods that
    name a target, ``Mcp-Name`` against ``params.name`` / ``params.uri``.
    """
    method = msg.get("method")
    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
    meta = request_meta(params) or {}
    sent_version = headers.get("MCP-Protocol-Version")
    body_version = meta.get(META_PROTOCOL_VERSION)
    if sent_version is None:
        return "the MCP-Protocol-Version header is missing"
    if body_version is not None and sent_version != body_version:
        return (
            f"MCP-Protocol-Version header value {sent_version!r} does not "
            f"match body value {body_version!r}"
        )
    sent_method = headers.get("Mcp-Method")
    if sent_method is None:
        return "the Mcp-Method header is missing"
    if sent_method != method:
        return (
            f"Mcp-Method header value {sent_method!r} does not match body "
            f"value {method!r}"
        )
    field = NAMED_METHODS.get(str(method))
    if field is not None:
        raw = headers.get("Mcp-Name")
        if raw is None:
            return "the Mcp-Name header is missing"
        try:
            sent_name = decode_header_value(raw)
        except ValueError as exc:
            return f"the Mcp-Name header is invalid: {exc}"
        if sent_name != params.get(field):
            return (
                f"Mcp-Name header value {sent_name!r} does not match body "
                f"value {params.get(field)!r}"
            )
    return None


def http_status_for_error(code: int) -> int:
    """HTTP status for a per-request JSON-RPC error on Streamable HTTP."""
    if code == -32601:
        return 404
    if code in (
        HEADER_MISMATCH,
        MISSING_REQUIRED_CLIENT_CAPABILITY,
        UNSUPPORTED_PROTOCOL_VERSION,
        -32602,
        -32600,
        -32700,
    ):
        return 400
    return 200


__all__ = [
    "MODERN_PROTOCOL_VERSIONS",
    "UNSUPPORTED_PROTOCOL_VERSION",
    "HEADER_MISMATCH",
    "MISSING_REQUIRED_CLIENT_CAPABILITY",
    "is_modern_request",
    "validate_request",
    "decorate_result",
    "discover_result",
    "in_modern_request",
]
