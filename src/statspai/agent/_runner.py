"""Tool-call runner: timeout, progress notifications, cooperative cancel.

What this module does
---------------------

:func:`run_with_progress` executes one ``tools/call`` body on a worker
thread and, while it runs, drains the tool's progress events into
``notifications/progress`` messages, enforces the wall-clock timeout
(``STATSPAI_MCP_TOOL_TIMEOUT_SECONDS``) and watches a cancel
:class:`threading.Event` set by the stdio loop when the client sends
``notifications/cancelled``.

What it does *not* do
---------------------

Python threads cannot be killed. On timeout or cancel the caller gets
its answer immediately, but the worker thread keeps running until the
tool returns or next reaches a cancellation checkpoint:

* A tool reaches a checkpoint whenever it calls :func:`progress` or
  :func:`check_cancelled`; both raise :class:`ToolCancelled` once the
  request was cancelled (or timed out), so a tool that reports progress
  stops at its next report.
* A tool that never reports (most estimators today) runs to completion
  in the background. Such threads are counted by
  :func:`orphaned_threads` and the count is reported in the timeout
  error so an operator can see that CPU is still being spent.

Concurrency between *requests* (answering ``ping`` / ``tools/list``
while a tool runs, bounded worker pool) lives in
:func:`statspai.agent.mcp_server.serve_stdio`; this module only runs a
single call.

Why threading, not asyncio
--------------------------

asyncio on stdin is fragile cross-platform (Windows lacks
``connect_read_pipe`` for pipes; ``anyio`` works but is a new dep).
A small ``threading.Thread`` + ``queue.Queue`` keeps the public
surface unchanged.

Public surface
--------------

* :func:`run_with_progress` — run one call with progress / timeout /
  cancel.
* :func:`progress` / :func:`check_cancelled` — tool-side helpers.
* :class:`ToolCancelled` — raised at a checkpoint after cancel/timeout.
* :func:`tool_timeout` / :data:`TOOL_TIMEOUT_ENV` — timeout config.
* :func:`orphaned_threads` — tool threads still running after their
  request was answered (timeout / cancel).
"""

from __future__ import annotations

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
# Cancellation
# ---------------------------------------------------------------------------


class ToolCancelled(BaseException):
    """Raised inside a tool thread at a checkpoint after cancel / timeout.

    Derives from :class:`BaseException` (like ``KeyboardInterrupt`` and
    ``asyncio.CancelledError``) so estimator code that guards its own
    work with ``except Exception`` cannot swallow the cancellation.
    """


_THREAD_LOCAL = threading.local()

#: Tool threads that were still running when their request was answered
#: (timeout or cancel). Pruned lazily by :func:`orphaned_threads`.
_ORPHANS: "set[threading.Thread]" = set()
_ORPHANS_LOCK = threading.Lock()


def _register_orphan(t: threading.Thread) -> None:
    if t.is_alive():
        with _ORPHANS_LOCK:
            _ORPHANS.add(t)


def orphaned_threads() -> int:
    """Number of tool threads still running after their call was answered."""
    with _ORPHANS_LOCK:
        for t in [t for t in _ORPHANS if not t.is_alive()]:
            _ORPHANS.discard(t)
        return len(_ORPHANS)


def set_cancel_event(event: Optional[threading.Event]) -> None:
    """Bind ``event`` as the cancel signal of the *current* thread.

    The stdio loop binds one event per in-flight ``tools/call`` on the
    pool thread that serves it; :func:`run_with_progress` picks it up via
    :func:`current_cancel_event` and re-binds it on the tool thread.
    """
    if event is None:
        if hasattr(_THREAD_LOCAL, "cancel_event"):
            del _THREAD_LOCAL.cancel_event
    else:
        _THREAD_LOCAL.cancel_event = event


def current_cancel_event() -> Optional[threading.Event]:
    """Cancel event bound to the current thread, if any."""
    return getattr(_THREAD_LOCAL, "cancel_event", None)


def check_cancelled() -> None:
    """Tool-side checkpoint: raise :class:`ToolCancelled` if cancelled.

    A no-op outside a cancellable MCP call, so library code can call it
    unconditionally inside long loops.
    """
    ev = getattr(_THREAD_LOCAL, "cancel_event", None)
    if ev is not None and ev.is_set():
        raise ToolCancelled("tool call cancelled by the client (or timed out)")


# ---------------------------------------------------------------------------
# Per-thread progress channel
# ---------------------------------------------------------------------------


def _set_progress_channel(token: Any, q: "queue.Queue[Any]") -> None:
    _THREAD_LOCAL.progress_token = token
    _THREAD_LOCAL.progress_queue = q


def _clear_progress_channel() -> None:
    if hasattr(_THREAD_LOCAL, "progress_token"):
        del _THREAD_LOCAL.progress_token
    if hasattr(_THREAD_LOCAL, "progress_queue"):
        del _THREAD_LOCAL.progress_queue


def progress(value: float, total: Optional[float] = None, *, message: str = "") -> None:
    """Tool-side helper: emit a ``notifications/progress``.

    Also a cancellation checkpoint: raises :class:`ToolCancelled` when
    the client cancelled the request (``notifications/cancelled``) or
    the call already timed out, so a tool that reports progress stops
    promptly. Outside an MCP call (no channel registered, e.g. a direct
    ``execute_tool`` call) this is a no-op.
    """
    check_cancelled()
    token = getattr(_THREAD_LOCAL, "progress_token", None)
    q = getattr(_THREAD_LOCAL, "progress_queue", None)
    if token is None or q is None:
        return
    payload = {"progressToken": token, "progress": float(value)}
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
    drain: Optional[Callable[[Dict[str, Any]], None]] = None,
    poll_interval: float = 0.05,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[bool, Any]:
    """Execute ``work()`` in a worker thread, draining progress events.

    Parameters
    ----------
    work : callable
        Zero-arg function to run. Returns whatever the caller wants;
        we surface the return verbatim or the exception in the second
        element of the tuple.
    progress_token : optional
        When non-None, the worker thread can call :func:`progress` to
        push notifications. When None, those calls only act as
        cancellation checkpoints and ``drain`` is never invoked.
    timeout : float, optional
        Wall-clock seconds to wait. ``None`` ⇒ wait indefinitely. On
        timeout the cancel event is set too, so the tool stops at its
        next checkpoint; a tool without checkpoints keeps running in the
        background (see :func:`orphaned_threads`).
    drain : callable, optional
        Receives each progress payload (a dict) as it arrives.
    poll_interval : float
        How often to check the worker / drain the queue.
    cancel_event : threading.Event, optional
        Set by the caller to cancel the call. Defaults to the event
        bound to the calling thread (:func:`current_cancel_event`).

    Returns
    -------
    (ok, result_or_exc)
        ``ok=True``: ``result_or_exc`` is the return value.
        ``ok=False``: ``result_or_exc`` is a ``TimeoutError`` (timeout),
        a :class:`ToolCancelled` (client cancel) or the exception raised
        by ``work``.
    """
    if cancel_event is None:
        cancel_event = current_cancel_event()
    if progress_token is None and not timeout and cancel_event is None:
        # Nothing to supervise: run inline (cheapest, and keeps
        # thread-affine libraries happy for in-process callers).
        try:
            return True, work()
        except BaseException as exc:  # noqa: BLE001 — preserve everything
            return False, exc

    # ``stop`` is what the tool thread's checkpoints watch. It is set on
    # client cancel *and* on timeout; the caller's ``cancel_event`` is
    # only ever read here, so a timeout never masquerades as a client
    # cancel (whose response the stdio loop suppresses).
    stop = threading.Event()
    q: "queue.Queue[Any]" = queue.Queue(maxsize=256)
    result: Dict[str, Any] = {}

    def _put_done() -> None:
        try:
            q.put_nowait(("done", None))
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(("done", None))

    def _runner() -> None:
        try:
            set_cancel_event(stop)
            _set_progress_channel(progress_token, q)
            result["value"] = work()
            result["ok"] = True
        except BaseException as exc:  # noqa: BLE001 — preserve everything
            result["ok"] = False
            result["exc"] = exc
        finally:
            _clear_progress_channel()
            set_cancel_event(None)
            _put_done()

    t = threading.Thread(target=_runner, name="statspai-mcp-tool", daemon=True)
    t.start()
    deadline = time.monotonic() + timeout if timeout else None

    while True:
        if deadline is not None and time.monotonic() > deadline:
            # Hard timeout. The response is surfaced now; the event makes
            # the tool stop at its next checkpoint, and a tool without
            # checkpoints is tracked as an orphan (threads cannot be
            # killed).
            stop.set()
            _register_orphan(t)
            return False, TimeoutError(
                f"tool exceeded {timeout:.0f}s timeout (env: {TOOL_TIMEOUT_ENV})"
            )
        if cancel_event is not None and cancel_event.is_set() and "ok" not in result:
            stop.set()
            _register_orphan(t)
            return False, ToolCancelled("tool call cancelled by the client")

        try:
            kind, payload = q.get(timeout=poll_interval)
        except queue.Empty:
            if not t.is_alive() and "ok" in result:
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
    "check_cancelled",
    "current_cancel_event",
    "set_cancel_event",
    "orphaned_threads",
    "run_with_progress",
    "ToolCancelled",
    "TOOL_TIMEOUT_ENV",
]
