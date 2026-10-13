"""Optional on-disk copy of the MCP result and data handles.

Handles (``r_...`` / ``d_...``) live in two in-process LRU caches, so a
server restart, a client reconnect that respawns the server, or plain LRU
pressure used to end an analysis chain: every later ``result_id`` call
answered ``missing_result_handle`` and the agent had to re-fit from the
top. With ``STATSPAI_MCP_STATE_DIR`` set, each cached entry is also
written under that directory and read back on a memory miss, so a handle
keeps resolving across restarts and after LRU eviction.

Off by default: with the variable unset nothing here runs and the caches
behave exactly as before.

Layout
------
``<dir>/hmac.key``             32 random bytes, mode 0600.
``<dir>/<kind>/<id>.pkl``      the cached object (pickle, HMAC-signed).
``<dir>/<kind>/<id>.meta``     tool, arguments, created_at, replay and
                               the StatsPAI version (pickle, signed).

The object and its metadata are separate files because the metadata is
rewritten after the fit (data provenance, replay string) and the object
can be large.

Trust
-----
The files are pickles. Whoever can write to the directory can run code in
the server on the next read, exactly as if they could edit the installed
package, so the directory must belong to the server's user and must not
be writable by group or others; a directory that fails that check is
refused with a warning and persistence stays off. The signature guards
against truncated, foreign or stale files, not against someone who can
also read the key.

Bounds
------
``STATSPAI_MCP_STATE_MAX_BYTES`` (default 4 GiB, ``0`` = unlimited) caps
each kind; the least recently used files are deleted first. An entry
written by a different StatsPAI version is discarded on read instead of
unpickled into classes whose layout may have changed. An object that
cannot be pickled stays memory-only and is counted in :func:`stats`.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
import threading
import warnings
from typing import Any, Dict, Optional, Tuple

STATE_DIR_ENV = "STATSPAI_MCP_STATE_DIR"
STATE_MAX_BYTES_ENV = "STATSPAI_MCP_STATE_MAX_BYTES"
DEFAULT_STATE_MAX_BYTES = 4 * 1024 * 1024 * 1024

_ID_RE = re.compile(r"^[a-z]_[0-9a-f]{4,64}$")
_KEY_FILE = "hmac.key"

_lock = threading.RLock()
_refused: "set[str]" = set()
_counters: Dict[str, int] = {
    "saved": 0,
    "restored": 0,
    "unpicklable": 0,
    "discarded": 0,
}


def _package_version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except (ImportError, AttributeError):  # pragma: no cover
        return "unknown"


def _max_bytes() -> Optional[int]:
    raw = os.environ.get(STATE_MAX_BYTES_ENV)
    if raw is None:
        return DEFAULT_STATE_MAX_BYTES
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_STATE_MAX_BYTES
    return v if v > 0 else None


def _refuse(root: str, why: str) -> None:
    if root in _refused:
        return
    _refused.add(root)
    warnings.warn(
        f"{STATE_DIR_ENV}={root!r} is not used: {why}. Handles stay " "in memory only.",
        RuntimeWarning,
        stacklevel=3,
    )


def _safe_dir(path: str) -> Optional[str]:
    """``None`` when ``path`` is ours alone, else the reason it is not."""
    try:
        st = os.stat(path)
    except OSError as exc:
        return f"cannot stat it ({exc})"
    if not stat.S_ISDIR(st.st_mode):
        return "it is not a directory"
    if os.name == "posix":
        if st.st_uid != os.getuid():
            return "it belongs to another user"
        if st.st_mode & 0o022:
            return "it is writable by group or others (chmod 700 it)"
    return None


def state_root() -> Optional[str]:
    """The validated state directory, or ``None`` when persistence is off."""
    raw = os.environ.get(STATE_DIR_ENV)
    if not raw:
        return None
    root = os.path.abspath(os.path.expanduser(raw))
    if root in _refused:
        return None
    with _lock:
        try:
            os.makedirs(root, mode=0o700, exist_ok=True)
        except OSError as exc:
            _refuse(root, f"cannot create it ({exc})")
            return None
        why = _safe_dir(root)
        if why is not None:
            _refuse(root, why)
            return None
    return root


def enabled() -> bool:
    return state_root() is not None


def _key(root: str) -> Optional[bytes]:
    path = os.path.join(root, _KEY_FILE)
    with _lock:
        try:
            with open(path, "rb") as fh:
                key = fh.read()
            if len(key) >= 32:
                return key
        except FileNotFoundError:
            pass
        except OSError as exc:
            _refuse(root, f"cannot read its key ({exc})")
            return None
        key = secrets.token_bytes(32)
        try:
            fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(key)
            os.replace(path + ".tmp", path)
        except OSError as exc:
            _refuse(root, f"cannot write its key ({exc})")
            return None
        return key


def _kind_dir(root: str, kind: str) -> Optional[str]:
    path = os.path.join(root, kind)
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except OSError:
        return None
    return path


def _paths(kind: str, rid: str) -> Optional[Tuple[str, str, bytes]]:
    """``(object path, metadata path, key)`` or ``None`` when unavailable."""
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        return None
    root = state_root()
    if root is None:
        return None
    key = _key(root)
    folder = _kind_dir(root, kind) if key is not None else None
    if key is None or folder is None:
        return None
    return os.path.join(folder, rid + ".pkl"), os.path.join(folder, rid + ".meta"), key


def _meta_of(entry: Any) -> Dict[str, Any]:
    return {
        "tool": entry.tool,
        "arguments": entry.arguments,
        "created_at": entry.created_at,
        "replay": entry.replay,
        "version": _package_version(),
    }


def _write(path: str, key: bytes, obj: Any) -> None:
    from ._process_worker import _write_signed

    _write_signed(path, key, obj)
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - non-POSIX
        pass


def save(kind: str, rid: str, entry: Any) -> bool:
    """Write ``entry`` (object + metadata). ``False`` when it was not kept."""
    import pickle  # nosec B403

    found = _paths(kind, rid)
    if found is None:
        return False
    obj_path, meta_path, key = found
    try:
        _write(obj_path, key, entry.obj)
        _write(meta_path, key, _meta_of(entry))
    except (pickle.PicklingError, TypeError, AttributeError, RecursionError):
        # Closures, open handles, C objects without a reduce: the handle
        # works for the life of this process and is simply not persisted.
        _discard_files(obj_path, meta_path)
        with _lock:
            _counters["unpicklable"] += 1
        return False
    except OSError as exc:
        _discard_files(obj_path, meta_path)
        _refuse(os.path.dirname(os.path.dirname(obj_path)), f"write failed ({exc})")
        return False
    with _lock:
        _counters["saved"] += 1
    _prune(kind, keep=rid)
    return True


def save_meta(kind: str, rid: str, entry: Any) -> bool:
    """Rewrite only the metadata of an entry already on disk."""
    import pickle  # nosec B403

    found = _paths(kind, rid)
    if found is None:
        return False
    obj_path, meta_path, key = found
    if not os.path.exists(obj_path):
        return False
    try:
        _write(meta_path, key, _meta_of(entry))
    except (pickle.PicklingError, TypeError, AttributeError, RecursionError, OSError):
        # Metadata that cannot be written would leave an object whose
        # provenance is out of date; drop the pair instead.
        _discard_files(obj_path, meta_path)
        return False
    return True


def load(kind: str, rid: str) -> Optional[Dict[str, Any]]:
    """Read an entry back: ``{obj, tool, arguments, created_at, replay}``.

    ``None`` when there is no usable copy. A copy that fails its
    signature, was written by another StatsPAI version, or does not
    unpickle is deleted.
    """
    import pickle  # nosec B403

    from ._process_worker import _read_signed

    found = _paths(kind, rid)
    if found is None:
        return None
    obj_path, meta_path, key = found
    if not (os.path.exists(obj_path) and os.path.exists(meta_path)):
        return None
    try:
        meta = _read_signed(meta_path, key)
        if not isinstance(meta, dict) or meta.get("version") != _package_version():
            raise ValueError("written by another StatsPAI version")
        obj = _read_signed(obj_path, key)
    except (
        OSError,
        ValueError,
        EOFError,
        AttributeError,
        ImportError,
        IndexError,
        pickle.UnpicklingError,
    ):
        _discard_files(obj_path, meta_path)
        with _lock:
            _counters["discarded"] += 1
        return None
    try:
        os.utime(obj_path, None)  # mark as recently used for pruning
    except OSError:
        pass
    with _lock:
        _counters["restored"] += 1
    return {
        "obj": obj,
        "tool": meta.get("tool") or "",
        "arguments": dict(meta.get("arguments") or {}),
        "created_at": meta.get("created_at"),
        "replay": meta.get("replay") or "",
    }


def _discard_files(*paths: str) -> None:
    for p in paths:
        for q in (p, p + ".tmp"):
            try:
                os.remove(q)
            except OSError:
                pass


def delete(kind: str, rid: str) -> None:
    found = _paths(kind, rid)
    if found is not None:
        _discard_files(found[0], found[1])


def clear(kind: str) -> None:
    root = state_root()
    folder = os.path.join(root, kind) if root else None
    if not folder or not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        if name.endswith((".pkl", ".meta", ".tmp")):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass


def _prune(kind: str, keep: str) -> None:
    """Delete least-recently-used entries until ``kind`` fits its budget."""
    budget = _max_bytes()
    root = state_root()
    if budget is None or root is None:
        return
    folder = os.path.join(root, kind)
    with _lock:
        try:
            rows = []
            for name in os.listdir(folder):
                if not name.endswith(".pkl"):
                    continue
                st = os.stat(os.path.join(folder, name))
                rows.append((st.st_mtime, st.st_size, name[:-4]))
        except OSError:
            return
        total = sum(size for _, size, _ in rows)
        for _, size, rid in sorted(rows):
            if total <= budget:
                break
            if rid == keep:
                continue  # the newest entry is always kept
            _discard_files(
                os.path.join(folder, rid + ".pkl"), os.path.join(folder, rid + ".meta")
            )
            total -= size


def stats() -> Dict[str, Any]:
    """Observability snapshot (``enabled``, directory, counters)."""
    root = state_root()
    with _lock:
        out: Dict[str, Any] = {"enabled": root is not None, **_counters}
    if root is not None:
        out["directory"] = root
        out["max_bytes"] = _max_bytes()
    return out
