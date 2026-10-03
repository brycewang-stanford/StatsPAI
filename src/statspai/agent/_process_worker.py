"""Process isolation for ``tools/call`` (opt-in).

A Python thread cannot be killed. Under the default thread runner a call
that exceeds its timeout is answered with an error while its computation
keeps running until it finishes; the stdio loop only bounds how many such
orphans may accumulate (``STATSPAI_MCP_MAX_ORPHANED_CALLS``).

With ``STATSPAI_MCP_ISOLATION=process`` a call that touches no
server-side state is run in a child ``statspai-mcp`` process instead. On
timeout or client cancel the child is killed: the CPU and memory are
released at once, nothing is left running, and warning filters,
matplotlib state and any other process-global side effect of the
estimator die with it.

What is isolated
----------------
Every call except those that need this server's own state. A call is
**not** isolated, and runs on the thread runner as before, when it

* reads a data handle (``data_id``): frames are not shipped to a worker,
* is a tool whose result is a data handle or a multi-step composite
  (``load_data``, ``transform_data``, ``describe_data``, ``detect_design``,
  ``preflight``, the ``pipeline_*`` composites), or
* talks back to the client while it runs (``interpret_result``, which may
  ask the client for a completion).

Result handles
--------------
A call that asks for a handle (``as_handle``) is isolated too. The child
pickles every result it cached into a directory only the two processes
use (created by the parent, mode 0700, removed afterwards), and the
parent adopts those entries under the same ids, so the ``result_id`` in
the response works for follow-up calls. The pickle travels between two
processes of the same installation started by this server; nothing read
from a client is unpickled. Each file is signed (HMAC-SHA256) with a
one-time key the parent generates for that call and passes to the worker
in its environment, and the parent unpickles only bytes whose signature
verifies.

A result that cannot be pickled, or that arrives after the call was
cancelled, is not adopted: its handle is removed from the response and
named under ``isolation.dropped_handles``. Data handles are never handed
over.

The same channel runs the other way for a call that *reads* a result
(``result_id``): the parent signs and writes the cached entry under
``in/``, the worker adopts it before running the tool, and the parent
keeps its own copy. If the entry is missing or cannot be pickled the call
stays on the thread runner, where the usual error or the usual answer is
produced.

Cost
----
Each isolated call pays the import of ``statspai`` in a fresh interpreter
(a few seconds). That is the price of a killable worker without a
serialisation protocol for fitted results, and the reason the mode is
opt-in. Server-initiated sampling is not available inside an isolated
call (the tools that use it take a ``result_id`` and are never isolated).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Env var: ``thread`` (default) or ``process``.
ISOLATION_ENV = "STATSPAI_MCP_ISOLATION"

#: Env var the parent sets on a worker: where to leave pickled results.
HANDOFF_ENV = "STATSPAI_MCP_HANDOFF_DIR"
#: Env var carrying the one-time key that signs them (hex).
HANDOFF_KEY_ENV = "STATSPAI_MCP_HANDOFF_KEY"

#: Arguments that bind a call to this server process.
_SESSION_ARGUMENTS = ("data_id",)

#: Tools that stay in the server: data handles, composites, and the one
#: that sends a request back to the client while it runs.
_SESSION_TOOLS = frozenset(
    {
        "load_data",
        "transform_data",
        "describe_data",
        "detect_design",
        "preflight",
        "interpret_result",
    }
)
_SESSION_PREFIXES = ("pipeline_",)

_HANDLE_KEYS = ("result_id", "result_uri", "data_id", "data_uri")


def isolation_mode() -> str:
    """``"process"`` when opted in, else ``"thread"``."""
    raw = (os.environ.get(ISOLATION_ENV) or "thread").strip().lower()
    return "process" if raw == "process" else "thread"


def eligible(name: str, arguments: Dict[str, Any]) -> bool:
    """Can this call run in a child process without losing anything?"""
    if name in _SESSION_TOOLS:
        return False
    if name.startswith(_SESSION_PREFIXES):
        return False
    return not any(arguments.get(k) not in (None, "") for k in _SESSION_ARGUMENTS)


def _sign(key: bytes, blob: bytes) -> bytes:
    import hashlib
    import hmac

    return hmac.new(key, blob, hashlib.sha256).digest()


def _child_env(
    handoff_dir: Optional[str] = None, handoff_key: Optional[str] = None
) -> Dict[str, str]:
    env = dict(os.environ)
    env[ISOLATION_ENV] = "thread"  # the child runs the call itself
    # The parent owns the deadline; the child must not race it.
    env["STATSPAI_MCP_TOOL_TIMEOUT_SECONDS"] = "0"
    if handoff_dir and handoff_key:
        env[HANDOFF_ENV] = handoff_dir
        env[HANDOFF_KEY_ENV] = handoff_key
    else:
        env.pop(HANDOFF_ENV, None)
        env.pop(HANDOFF_KEY_ENV, None)
    return env


def _write_signed(path: str, key: bytes, obj: Any) -> None:
    import pickle  # nosec B403

    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path + ".tmp", "wb") as fh:
        fh.write(_sign(key, blob) + blob)
    os.replace(path + ".tmp", path)


def _read_signed(path: str, key: bytes) -> Any:
    """Unpickle a file written by :func:`_write_signed`; ``ValueError`` if
    the signature does not verify."""
    import hmac
    import pickle  # nosec B403

    with open(path, "rb") as fh:
        raw = fh.read()
    tag, blob = raw[:32], raw[32:]
    if not hmac.compare_digest(tag, _sign(key, blob)):
        raise ValueError("signature mismatch")
    return pickle.loads(blob)  # nosec B301


def export_inbound(handoff_dir: str, handoff_key: str, rids: List[str]) -> bool:
    """Parent side: leave the results a call reads for the worker.

    Returns ``False`` (writing nothing more) as soon as one of them is not
    in the cache or cannot be pickled; the caller then keeps the call on
    the thread runner.
    """
    import pickle  # nosec B403

    from ._result_cache import RESULT_CACHE

    target = os.path.join(handoff_dir, "in")
    os.makedirs(target, mode=0o700, exist_ok=True)
    key = bytes.fromhex(handoff_key)
    for rid in rids:
        entry = RESULT_CACHE.get_entry(rid)
        if entry is None:
            return False
        try:
            _write_signed(os.path.join(target, f"{rid}.pkl"), key, entry)
        except (pickle.PicklingError, TypeError, AttributeError, OSError):
            return False
    return True


def import_inbound() -> List[str]:
    """Worker side: adopt the results the parent left under ``in/``.

    A no-op outside a worker. Returns the adopted ids, which
    :func:`export_handles` then leaves out of what it sends back.
    """
    target = os.environ.get(HANDOFF_ENV)
    key_hex = os.environ.get(HANDOFF_KEY_ENV)
    if not target or not key_hex:
        return []
    source = os.path.join(target, "in")
    if not os.path.isdir(source):
        return []
    from ._result_cache import RESULT_CACHE

    key = bytes.fromhex(key_hex)
    adopted: List[str] = []
    for fname in sorted(os.listdir(source)):
        if not fname.endswith(".pkl"):
            continue
        rid = fname[: -len(".pkl")]
        entry = _read_signed(os.path.join(source, fname), key)
        if RESULT_CACHE.adopt(rid, entry):
            adopted.append(rid)
    _INBOUND.update(adopted)
    return adopted


#: Ids this worker received from its parent (never sent back).
_INBOUND: set = set()


def export_handles() -> None:
    """Child side: pickle the cached results for the parent to adopt.

    A no-op unless this process was started as an isolated worker. Each
    entry goes to ``<dir>/<result_id>.pkl``; one that cannot be pickled
    is named in ``<dir>/_failed.json`` instead, with the reason.
    """
    target = os.environ.get(HANDOFF_ENV)
    key_hex = os.environ.get(HANDOFF_KEY_ENV)
    if not target or not key_hex or not os.path.isdir(target):
        return
    import pickle  # nosec B403

    key = bytes.fromhex(key_hex)

    from ._result_cache import RESULT_CACHE

    failed: Dict[str, str] = {}
    for rid, entry in RESULT_CACHE.snapshot().items():
        if rid in _INBOUND:
            continue  # the parent already holds it
        path = os.path.join(target, f"{rid}.pkl")
        try:
            blob = pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)
            with open(path + ".tmp", "wb") as fh:
                fh.write(_sign(key, blob) + blob)
            os.replace(path + ".tmp", path)
        except (pickle.PicklingError, TypeError, AttributeError, OSError) as exc:
            failed[rid] = f"{type(exc).__name__}: {exc}"[:300]
            for leftover in (path + ".tmp", path):
                if os.path.exists(leftover):
                    os.remove(leftover)
    if failed:
        with open(os.path.join(target, "_failed.json"), "w", encoding="utf-8") as fh:
            json.dump(failed, fh)


def import_handles(
    handoff_dir: str, handoff_key: str
) -> Tuple[List[str], Dict[str, str]]:
    """Parent side: adopt what the child exported.

    Returns ``(adopted ids, {id: reason} for the ones that were not)``.
    A file whose signature does not verify under ``handoff_key`` is not
    unpickled.
    """
    import hmac
    import pickle  # nosec B403

    key = bytes.fromhex(handoff_key)

    from ._result_cache import RESULT_CACHE
    from ._runner import ToolCancelled

    adopted: List[str] = []
    failed: Dict[str, str] = {}
    note = os.path.join(handoff_dir, "_failed.json")
    if os.path.exists(note):
        with open(note, encoding="utf-8") as fh:
            failed.update(json.load(fh))
    for fname in sorted(os.listdir(handoff_dir)):
        if not fname.endswith(".pkl"):
            continue
        rid = fname[: -len(".pkl")]
        try:
            with open(os.path.join(handoff_dir, fname), "rb") as fh:
                raw = fh.read()
            tag, blob = raw[:32], raw[32:]
            if not hmac.compare_digest(tag, _sign(key, blob)):
                failed[rid] = "signature mismatch: the file was not adopted"
                continue
            # Authenticated above: written by the worker this server started.
            entry = pickle.loads(blob)  # nosec B301
            if RESULT_CACHE.adopt(rid, entry):
                adopted.append(rid)
            else:
                failed[rid] = "the id is already in use in this server"
        except ToolCancelled:
            failed[rid] = "the call was cancelled before the handle was adopted"
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
            failed[rid] = f"{type(exc).__name__}: {exc}"[:300]
    return adopted, failed


def run_isolated(
    name: str,
    arguments: Dict[str, Any],
    meta: Optional[Dict[str, Any]],
    *,
    timeout: Optional[float],
    cancel_event: Optional[threading.Event],
    forward: Optional[Callable[[str], None]] = None,
    poll_interval: float = 0.05,
    handoff_dir: Optional[str] = None,
    handoff_key: Optional[str] = None,
) -> Tuple[str, Any]:
    """Run one ``tools/call`` in a child server and supervise it.

    Returns ``(status, payload)``:

    * ``("result", <tools/call result dict>)`` -- the child answered;
    * ``("rpc_error", <JSON-RPC error object>)`` -- a protocol error;
    * ``("timeout", pid)`` / ``("cancelled", pid)`` -- the child was killed;
    * ``("crashed", {"returncode", "stderr"})`` -- it exited without an
      answer.

    ``forward`` receives every notification line the child writes
    (progress), verbatim.
    """
    params: Dict[str, Any] = {"name": name, "arguments": arguments}
    if meta:
        params["_meta"] = meta
    request = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "statspai.agent.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_child_env(handoff_dir, handoff_key),
        # posix_spawn rather than fork: a fork taken after numerical code
        # ran in this process can crash in the child before exec (macOS).
        close_fds=False,
    )
    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    stderr_tail: List[str] = []

    def _pump_stdout() -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    def _pump_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_tail.append(line)
            del stderr_tail[:-40]

    for target in (_pump_stdout, _pump_stderr):
        threading.Thread(target=target, daemon=True).start()

    try:
        assert proc.stdin is not None
        proc.stdin.write(request + "\n")
        proc.stdin.close()
    except OSError:
        pass  # the child died at start-up; reported as a crash below

    deadline = time.monotonic() + timeout if timeout else None

    def _kill() -> None:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass

    while True:
        if deadline is not None and time.monotonic() > deadline:
            _kill()
            return "timeout", proc.pid
        if cancel_event is not None and cancel_event.is_set():
            _kill()
            return "cancelled", proc.pid
        try:
            line = lines.get(timeout=poll_interval)
        except queue.Empty:
            continue
        if line is None:
            proc.wait()
            return "crashed", {
                "returncode": proc.returncode,
                "stderr": "".join(stderr_tail)[-2000:],
            }
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        if msg.get("id") == 1 and ("result" in msg or "error" in msg):
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill()
            if "error" in msg:
                return "rpc_error", msg["error"]
            return "result", msg["result"]
        if "method" in msg and "id" not in msg and forward is not None:
            forward(line)


def annotate(
    result: Dict[str, Any],
    adopted: Optional[List[str]] = None,
    failed: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Mark a child's result as isolated; drop handles the parent does not hold.

    ``adopted`` are the result ids the parent took over from the child: a
    ``result_id`` among them stays in the response.
    """
    payload = result.get("structuredContent")
    if not isinstance(payload, dict):
        return result
    kept = set(adopted or ())
    child_rid = str(payload.get("result_id") or "")
    result_alive = child_rid in kept
    dropped = [
        k
        for k in _HANDLE_KEYS
        if k in payload and not (result_alive and k in ("result_id", "result_uri"))
    ]
    for key in dropped:
        payload.pop(key, None)
    # follow-up calls that would need a handle the response no longer has
    dead_arguments = tuple(k for k in ("result_id", "data_id") if k in dropped)
    calls = payload.get("next_calls")
    if dropped and isinstance(calls, list):
        payload["next_calls"] = [
            c
            for c in calls
            if not (
                isinstance(c, dict)
                and isinstance(c.get("arguments"), dict)
                and any(k in c["arguments"] for k in dead_arguments)
            )
        ]
    note: Dict[str, Any] = {"mode": "process"}
    if result_alive:
        note["adopted_handles"] = ["result_id"]
    if dropped:
        note["dropped_handles"] = dropped
        reason = (failed or {}).get(child_rid)
        note["hint"] = (
            "the result could not be handed over from the worker"
            if "result_id" in dropped
            else "data handles are not handed over from an isolated worker"
        )
        if reason:
            note["reason"] = reason
    payload["isolation"] = note
    content = result.get("content")
    if isinstance(content, list) and content and content[0].get("type") == "text":
        content[0]["text"] = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return result


__all__ = [
    "HANDOFF_ENV",
    "HANDOFF_KEY_ENV",
    "ISOLATION_ENV",
    "annotate",
    "eligible",
    "export_handles",
    "import_handles",
    "isolation_mode",
    "run_isolated",
]
