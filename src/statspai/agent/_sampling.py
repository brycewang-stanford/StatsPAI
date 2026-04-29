"""MCP server-initiated LLM sampling — opt-in, capability-gated.

Per MCP 2024-11-05, ``sampling/createMessage`` is a *server-to-client*
JSON-RPC request: the server asks the connected client to call its
LLM and return the completion. This module wraps that protocol so
StatsPAI's LLM-driven helpers (``llm_dag_propose``, ``llm_evalue``,
``llm_sensitivity``, …) can reuse the client's already-authenticated
LLM session instead of forcing each user to configure their own API
key.

Design
------

* **Capability-gated**: nothing happens until the client advertises
  ``capabilities.sampling`` during ``initialize``. Until then, calls
  to :func:`request_sampling` fall through to the
  ``UnsupportedSamplingError`` so the caller can route to the
  user-API-key fallback.
* **Stateful**: the active stdio loop registers a writer callback,
  records pending request IDs, and matches replies. We don't try to
  multiplex multiple in-flight sampling requests yet — the existing
  threading runner serialises tool calls, so one-at-a-time is
  sufficient.
* **Bounded wait**: every sampling request has a hard timeout
  (env: ``STATSPAI_MCP_SAMPLING_TIMEOUT_SECONDS``, default 60s) so
  the server can't hang forever waiting on a buggy client.

Public surface
--------------

* :func:`set_capability` / :func:`get_capability` — toggle the
  client-advertised capability flag (called by ``_handle_initialize``).
* :func:`request_sampling(messages, ...)` — server-to-client request;
  blocks until the response or the timeout. Raises
  :class:`UnsupportedSamplingError` when no capability is set.
* :func:`route_response(message)` — called by the stdio reader when
  it spots a JSON-RPC reply whose ``id`` matches a pending sampling
  request.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional


SAMPLING_TIMEOUT_ENV = "STATSPAI_MCP_SAMPLING_TIMEOUT_SECONDS"
_DEFAULT_SAMPLING_TIMEOUT = 60.0


class UnsupportedSamplingError(RuntimeError):
    """Raised when sampling is requested but the client did not
    advertise the capability. Callers should catch this and fall
    back to a user-supplied API-key path."""


class SamplingTimeoutError(TimeoutError):
    """Raised when the client takes too long to reply to a sampling
    request. The configured limit is read from
    :data:`SAMPLING_TIMEOUT_ENV` (default 60 s)."""


def _sampling_timeout() -> float:
    raw = os.environ.get(SAMPLING_TIMEOUT_ENV)
    if raw is None:
        return _DEFAULT_SAMPLING_TIMEOUT
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_SAMPLING_TIMEOUT
    return v if v > 0 else _DEFAULT_SAMPLING_TIMEOUT


# ---------------------------------------------------------------------------
# Capability + writer registration (set by mcp_server during the loop)
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()
_CAPABILITY: bool = False
_WRITER: Optional[Callable[[str], None]] = None
_PENDING: Dict[Any, "_PendingRequest"] = {}
_REQUEST_ID_COUNTER = itertools.count(1)


class _PendingRequest:
    """In-flight sampling request awaiting a JSON-RPC reply."""

    __slots__ = ("event", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Any = None
        self.error: Optional[Dict[str, Any]] = None


def set_capability(advertised: bool) -> None:
    """Record whether the client advertised ``capabilities.sampling``.

    Called by :func:`mcp_server._handle_initialize` once the client's
    handshake is parsed. Defaults to ``False`` (no sampling) until set.
    """
    global _CAPABILITY
    with _LOCK:
        _CAPABILITY = bool(advertised)


def get_capability() -> bool:
    with _LOCK:
        return _CAPABILITY


def set_writer(writer: Optional[Callable[[str], None]]) -> None:
    """Register / clear the JSON-RPC line writer.

    The stdio loop calls this with its ``stdout.write + flush``
    closure so :func:`request_sampling` can post requests on the
    same channel as the regular response stream. Pass ``None`` to
    clear at shutdown.
    """
    global _WRITER
    with _LOCK:
        _WRITER = writer


# ---------------------------------------------------------------------------
# Server-side sampling request
# ---------------------------------------------------------------------------

def request_sampling(
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 1024,
    system_prompt: str = "",
    model_preferences: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Send a ``sampling/createMessage`` request to the client.

    Parameters
    ----------
    messages : list of dict
        MCP-style messages. Each entry: ``{"role": "user"|"assistant",
        "content": {"type": "text", "text": ...}}``.
    max_tokens : int
        Max tokens to generate. Required by the MCP spec.
    system_prompt : str, optional
        System-level instructions; passed via the ``systemPrompt``
        field per spec.
    model_preferences : dict, optional
        Model selection hints; pass-through per spec.
    temperature : float, optional
    stop_sequences : list of str, optional
    timeout : float, optional
        Override :data:`SAMPLING_TIMEOUT_ENV`.

    Returns
    -------
    dict
        The result block from the client. Shape per spec:
        ``{"role": "assistant", "content": {"type": "text", "text": ...},
        "model": "...", "stopReason": "..."}``.

    Raises
    ------
    UnsupportedSamplingError
        Client never advertised the capability.
    SamplingTimeoutError
        Client did not respond within the timeout.
    RuntimeError
        Client returned a JSON-RPC error envelope.
    """
    if not get_capability():
        raise UnsupportedSamplingError(
            "client did not advertise capabilities.sampling; "
            "fall back to a user-supplied API key path")
    with _LOCK:
        writer = _WRITER
    if writer is None:
        raise UnsupportedSamplingError(
            "no MCP stdio writer registered; sampling unavailable "
            "outside an active server loop")

    request_id = f"sp-sampling-{next(_REQUEST_ID_COUNTER)}"
    params: Dict[str, Any] = {
        "messages": messages,
        "maxTokens": int(max_tokens),
    }
    if system_prompt:
        params["systemPrompt"] = system_prompt
    if model_preferences:
        params["modelPreferences"] = model_preferences
    if temperature is not None:
        params["temperature"] = float(temperature)
    if stop_sequences:
        params["stopSequences"] = list(stop_sequences)

    pending = _PendingRequest()
    with _LOCK:
        _PENDING[request_id] = pending
    try:
        writer(json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "sampling/createMessage",
            "params": params,
        }))
        wait_for = timeout if timeout is not None else _sampling_timeout()
        if not pending.event.wait(timeout=wait_for):
            raise SamplingTimeoutError(
                f"client did not reply to sampling request {request_id} "
                f"within {wait_for:.1f}s "
                f"(env: {SAMPLING_TIMEOUT_ENV})"
            )
        if pending.error is not None:
            raise RuntimeError(
                f"client returned error for sampling request "
                f"{request_id}: {pending.error}")
        return pending.result or {}
    finally:
        with _LOCK:
            _PENDING.pop(request_id, None)


def route_response(message: Dict[str, Any]) -> bool:
    """Hook called by the stdio reader when a JSON-RPC reply arrives.

    Returns ``True`` if the message matched a pending sampling
    request and was consumed; ``False`` otherwise (caller should
    treat the message as an unsolicited / out-of-band reply).
    """
    if not isinstance(message, dict):
        return False
    msg_id = message.get("id")
    with _LOCK:
        pending = _PENDING.get(msg_id) if msg_id is not None else None
    if pending is None:
        return False
    if "error" in message:
        pending.error = message["error"]
    else:
        pending.result = message.get("result", {})
    pending.event.set()
    return True


__all__ = [
    "SAMPLING_TIMEOUT_ENV",
    "UnsupportedSamplingError",
    "SamplingTimeoutError",
    "set_capability",
    "get_capability",
    "set_writer",
    "request_sampling",
    "route_response",
]
