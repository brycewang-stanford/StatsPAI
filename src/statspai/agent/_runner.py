"""Concurrent tool-call runner with timeout + progress notifications.

Why this exists
---------------

The v1.x ``serve_stdio`` is a strictly serial loop:

    for raw in stdin:
        response = handle_request(raw)
        write(response)

That has three operational problems:

1. **Long tools block the channel.** A 10-minute ``spec_curve`` ties up
   the server; the client can't even ``initialize`` a sibling MCP
   session until it ends.
2. **No progress signal.** MCP 2024-11-05 supports
   ``notifications/progress`` so a server can keep the agent's UI
   informed during slow work; the v1.10 server has no plumbing.
3. **No timeout.** A misconfigured BCF / SuperLearner can pin the
   process indefinitely.

This module wraps each ``tools/call`` in a worker thread, drains a
shared progress queue from the main loop, and enforces a global
timeout. Tools opt in to progress reporting via a ``progress=``
keyword (passed through ``tools/call``'s ``_meta.progressToken``);
tools that don't accept it remain unaffected.

Why threading, not asyncio
--------------------------

asyncio on stdin is fragile cross-platform (Windows lacks
``connect_read_pipe`` for pipes; ``anyio`` works but is a new dep).
A small ``threading.Thread`` + ``queue.Queue`` keeps the public
surface unchanged and avoids the asyncio Windows pitfall, at the cost
of giving up structured cancellation (we use ``threading.Event`` +
heartbeat checks instead).

Public surface
--------------

* :func:`run_tools_call_with_progress` — wrap an MCP tools/call
  request_id, params dict, and emit-callback into a runner that
  returns the JSON-RPC result string and posts progress notifications
  along the way.
* :data:`PROGRESS_TOKEN` — thread-local key tools read to discover
  whether to emit progress.
* :data:`TOOL_TIMEOUT_ENV` — env var read by :func:`tool_timeout`.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple


#: Env var: hard timeout (seconds) per ``tools/call``. ``0`` ⇒ disabled.
TOOL_TIMEOUT_ENV = "STATSPAI_MCP_TOOL_TIMEOUT_SECONDS"
_DEFAULT_TIMEOUT_SECONDS = 600  # 10 min — generous for BCF / spec_curve


def tool_timeout() -> Optional[float]:
    """Read the configured tool-call timeout. ``None`` ⇒ no timeout."""
    raw = os.environ.get(TOOL_TIMEOUT_ENV)
    if raw is None:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_SECONDS
    return v if v > 0 else None


# ---------------------------------------------------------------------------
# Per-thread progress channel
# ---------------------------------------------------------------------------

_THREAD_LOCAL = threading.local()


def _set_progress_channel(token: Any, q: queue.Queue) -> None:
    _THREAD_LOCAL.progress_token = token
    _THREAD_LOCAL.progress_queue = q


def _clear_progress_channel() -> None:
    if hasattr(_THREAD_LOCAL, "progress_token"):
        del _THREAD_LOCAL.progress_token
    if hasattr(_THREAD_LOCAL, "progress_queue"):
        del _THREAD_LOCAL.progress_queue


def progress(value: float, total: Optional[float] = None,
             *, message: str = "") -> None:
    """Tool-side helper: emit a ``notifications/progress``.

    Idempotent and safe to call from any tool: when no channel is
    registered (e.g. agent-side direct ``execute_tool`` call), this
    is a no-op. Tools that wrap their own loops can call this
    cheaply; the main loop drains the queue and serialises the
    notification on stdout.
    """
    token = getattr(_THREAD_LOCAL, "progress_token", None)
    q = getattr(_THREAD_LOCAL, "progress_queue", None)
    if token is None or q is None:
        return
    payload = {"progressToken": token,
               "progress": float(value)}
    if total is not None:
        payload["total"] = float(total)
    if message:
        payload["message"] = str(message)
    try:
        q.put_nowait(("progress", payload))
    except queue.Full:  # pragma: no cover — bounded queue safety
        pass


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def run_with_progress(
    work: Callable[[], Any],
    *,
    progress_token: Optional[Any] = None,
    timeout: Optional[float] = None,
    drain: Callable[[Dict[str, Any]], None] = None,
    poll_interval: float = 0.05,
) -> Tuple[bool, Any]:
    """Execute ``work()`` in a worker thread, draining progress events.

    Parameters
    ----------
    work : callable
        Zero-arg function to run. Returns whatever the caller wants;
        we surface the return verbatim or the exception in the second
        element of the tuple.
    progress_token : optional
        When non-None, the worker thread can call
        :func:`progress` to push notifications. When None, those calls
        are no-ops and ``drain`` is never invoked.
    timeout : float, optional
        Wall-clock seconds to wait. ``None`` ⇒ wait indefinitely.
    drain : callable, optional
        Receives each progress payload (a dict) as it arrives. The
        caller serialises it as a JSON-RPC notification and writes
        to stdout.
    poll_interval : float
        How often to check the worker / drain the queue.

    Returns
    -------
    (ok, result_or_exc)
        ``ok=True``: ``result_or_exc`` is the return value.
        ``ok=False``: ``result_or_exc`` is a ``TimeoutError`` (when
        the timeout fired) or the exception raised by ``work``.
    """
    q: queue.Queue = queue.Queue(maxsize=256)
    result: Dict[str, Any] = {}

    def _runner():
        try:
            if progress_token is not None:
                _set_progress_channel(progress_token, q)
            result["value"] = work()
            result["ok"] = True
        except BaseException as exc:  # noqa: BLE001 — preserve everything
            result["ok"] = False
            result["exc"] = exc
        finally:
            _clear_progress_channel()
            q.put_nowait(("done", None))

    t = threading.Thread(target=_runner, name="statspai-mcp-tool",
                          daemon=True)
    t.start()
    deadline = time.monotonic() + timeout if timeout else None

    while True:
        if deadline is not None and time.monotonic() > deadline:
            # Hard timeout — the worker keeps running (Python threads
            # are not cooperatively cancellable) but the response is
            # surfaced now. Tools that do heavy work without yielding
            # back to Python won't honour this — that's the cost of
            # not ripping out the synchronous tool API.
            return False, TimeoutError(
                f"tool exceeded {timeout:.0f}s timeout "
                f"(env: {TOOL_TIMEOUT_ENV})"
            )

        try:
            kind, payload = q.get(timeout=poll_interval)
        except queue.Empty:
            if not t.is_alive() and "ok" in result:
                # Race: thread finished without putting "done"
                # (shouldn't happen given the finally:, but be defensive).
                break
            continue
        if kind == "progress":
            if drain is not None:
                drain(payload)
            continue
        if kind == "done":
            break

    if "ok" not in result:
        # Thread is still running but the queue said done — defensive.
        return False, RuntimeError("worker terminated without result")
    if result["ok"]:
        return True, result["value"]
    return False, result["exc"]


__all__ = [
    "tool_timeout",
    "progress",
    "run_with_progress",
    "TOOL_TIMEOUT_ENV",
]
