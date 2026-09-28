"""In-process LRU cache for fitted StatsPAI results.

Provides server-side state for the MCP layer so an agent can chain
``did → audit → sensitivity → honest_did`` without re-running the
estimator on every step.

Design
------

* **Bounded** — defaults to 32 results; bumped via
  ``STATSPAI_MCP_RESULT_CACHE_SIZE``. LRU eviction.
* **Optionally time-bounded** — when ``STATSPAI_MCP_RESULT_CACHE_TTL``
  (seconds) is set, entries older than the TTL (measured from creation,
  i.e. fit time) are treated as absent and swept. Default is *no TTL*,
  preserving the original behaviour exactly. A long-running server can
  set a TTL so a handle never silently resolves to a stale fit after the
  underlying data may have changed.
* **Process-local** — handles do not survive a server restart. Agents
  treat the absence of a result as a recoverable error and re-fit.
* **Reason-aware misses** — when a handle is gone, the cache remembers
  *why* (``ttl`` / ``lru`` / ``explicit``) in a small bounded ledger so
  the MCP layer can tell an agent "expired, re-fit" vs. "unknown id"
  instead of an undifferentiated miss.
* **Type-erased** — we cache *any* fitted object, including
  ``CausalResult``, ``EconometricResults``, ``IdentificationReport``,
  workflow result objects, and pandas DataFrames returned by helper
  tools. The cache is the agent's working memory; downstream tools
  introspect what they got.
* **Tagged** — entries record the tool name + arguments that produced
  them, enabling rich resource representations (``statspai://result/<id>``
  returns the full provenance).

Thread-safety
-------------

The cache uses a re-entrant lock around all mutations. The stdio server
answers ``tools/call`` on a worker pool (one worker by default,
``STATSPAI_MCP_WORKERS`` to raise it) while the main loop keeps serving
``resources/read``, so the cache is accessed from several threads.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_DEFAULT_CACHE_SIZE = 32

#: How many evicted/expired handles to remember (id → reason) so a miss
#: can be explained. Bounded to keep the ledger from growing unboundedly
#: on a long-running server.
_EVICTION_LEDGER_SIZE = 256

#: Miss-reason tags recorded in the eviction ledger.
EVICT_TTL = "ttl"
EVICT_LRU = "lru"
EVICT_EXPLICIT = "explicit"
#: Evicted to keep the cache under its total-bytes budget (data cache).
EVICT_BYTES = "bytes"


def _metadata_value(value: Any) -> Any:
    """Return a compact JSON-friendly value for cache metadata."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(x, (str, int, float, bool)) or x is None for x in value):
            return list(value)
        return f"<{type(value).__name__}, len={len(value)}>"
    if isinstance(value, dict):
        clean: Dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str):
                clean[k] = _metadata_value(v)
        return clean
    return f"<{type(value).__name__}>"


def _cache_size() -> int:
    raw = os.environ.get("STATSPAI_MCP_RESULT_CACHE_SIZE")
    if raw is None:
        return _DEFAULT_CACHE_SIZE
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_CACHE_SIZE


def _cache_ttl() -> Optional[float]:
    """Read the optional TTL (seconds) from the environment.

    Returns ``None`` (no expiry) when unset or unparseable / non-positive,
    so a malformed env var can never silently shrink retention to zero.
    """
    raw = os.environ.get("STATSPAI_MCP_RESULT_CACHE_TTL")
    if raw is None:
        return None
    try:
        ttl = float(raw)
    except (TypeError, ValueError):
        return None
    return ttl if ttl > 0 else None


@dataclass
class CacheEntry:
    """One slot in the result cache."""

    obj: Any
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    #: Python call that reproduces the fit (``sp.<fn>(data=data, ...)``);
    #: empty when the producing tool has no single-call equivalent.
    replay: str = ""
    #: Approximate in-memory size, tracked only by byte-budgeted caches.
    nbytes: int = 0

    def to_metadata(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of this entry's provenance."""
        # Strip DataFrame / array / non-scalar arguments so the metadata
        # fits in an MCP resource without dragging the original dataset
        # back through the wire.
        clean: Dict[str, Any] = {
            k: _metadata_value(v) for k, v in self.arguments.items()
        }
        meta: Dict[str, Any] = {
            "tool": self.tool,
            "arguments": clean,
            "created_at": self.created_at,
            "result_class": type(self.obj).__name__,
        }
        if self.replay:
            meta["replay"] = self.replay
        return meta


class ResultCache:
    """LRU cache mapping ``result_id → CacheEntry``.

    ``_prefix`` is the handle prefix (``r_`` for results; the data cache
    subclass uses ``d_`` so the two kinds of handle are never confused).

    Optionally TTL-bounded (``ttl_seconds``): entries older than the TTL
    (from creation) are treated as absent and swept lazily on access and
    eagerly on insert. ``ttl_seconds=None`` (the default) disables expiry.
    """

    _prefix = "r_"

    def __init__(
        self,
        max_size: Optional[int] = None,
        *,
        ttl_seconds: Optional[float] = None,
        max_bytes: Optional[int] = None,
    ) -> None:
        self._max_size = max_size or _cache_size()
        self._ttl = ttl_seconds if ttl_seconds is not None else _cache_ttl()
        #: Total-bytes budget (``None`` ⇒ count-bounded only). The newest
        #: entry is always kept, even when it alone exceeds the budget.
        self._max_bytes = max_bytes if max_bytes else None
        self._total_bytes = 0
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        #: id → reason for handles that left the cache (bounded ledger).
        self._evicted: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.RLock()

    # -- internal helpers (assume the lock is held) -------------------- #

    def _entry_bytes(self, obj: Any) -> int:
        """Size estimate used by the bytes budget; subclasses override."""
        return 0

    def _drop_locked(self, rid: str, reason: str) -> None:
        entry = self._store.pop(rid, None)
        if entry is not None:
            self._total_bytes -= entry.nbytes
        self._record_eviction(rid, reason)

    def _record_eviction(self, rid: str, reason: str) -> None:
        self._evicted[rid] = reason
        self._evicted.move_to_end(rid)
        while len(self._evicted) > _EVICTION_LEDGER_SIZE:
            self._evicted.popitem(last=False)

    def _is_expired(self, entry: CacheEntry) -> bool:
        return self._ttl is not None and (time.time() - entry.created_at) > self._ttl

    def _purge_expired_locked(self) -> int:
        if self._ttl is None:
            return 0
        now = time.time()
        stale = [
            rid for rid, e in self._store.items() if (now - e.created_at) > self._ttl
        ]
        for rid in stale:
            self._drop_locked(rid, EVICT_TTL)
        return len(stale)

    # -- public API ---------------------------------------------------- #

    def put(
        self,
        obj: Any,
        *,
        tool: str = "",
        arguments: Optional[Dict[str, Any]] = None,
        replay: str = "",
    ) -> str:
        """Cache ``obj`` and return its newly-minted handle."""
        rid = self._prefix + secrets.token_hex(4)
        nbytes = self._entry_bytes(obj) if self._max_bytes else 0
        with self._lock:
            self._purge_expired_locked()
            self._store[rid] = CacheEntry(
                obj=obj,
                tool=tool,
                arguments=dict(arguments or {}),
                replay=replay or "",
                nbytes=nbytes,
            )
            self._total_bytes += nbytes
            self._store.move_to_end(rid)
            # A reused id (vanishingly unlikely) is no longer evicted.
            self._evicted.pop(rid, None)
            while len(self._store) > self._max_size:
                old_rid = next(iter(self._store))
                self._drop_locked(old_rid, EVICT_LRU)
            if self._max_bytes is not None:
                while self._total_bytes > self._max_bytes and len(self._store) > 1:
                    old_rid = next(iter(self._store))
                    self._drop_locked(old_rid, EVICT_BYTES)
        return rid

    def set_replay(self, rid: str, replay: str) -> bool:
        """Replace the replay string of an existing entry."""
        with self._lock:
            entry = self._store.get(rid)
            if entry is None:
                return False
            entry.replay = replay
            return True

    def annotate(self, rid: str, arguments: Dict[str, Any]) -> bool:
        """Merge additional provenance arguments into an existing entry."""
        with self._lock:
            entry = self._store.get(rid)
            if entry is None or self._is_expired(entry):
                if entry is not None:
                    self._drop_locked(rid, EVICT_TTL)
                return False
            entry.arguments.update(arguments)
            self._store.move_to_end(rid)
            return True

    def get(self, rid: str) -> Optional[Any]:
        """Return the cached object for ``rid``, or ``None`` if absent/expired."""
        entry = self.get_entry(rid)
        return entry.obj if entry is not None else None

    def get_entry(self, rid: str) -> Optional[CacheEntry]:
        """Return the full entry (object + provenance) or ``None``.

        An expired entry is swept and recorded as a TTL miss before
        returning ``None``.
        """
        with self._lock:
            entry = self._store.get(rid)
            if entry is None:
                return None
            if self._is_expired(entry):
                self._drop_locked(rid, EVICT_TTL)
                return None
            self._store.move_to_end(rid)
            return entry

    def evict(self, rid: str) -> bool:
        """Explicitly drop a handle. Returns ``True`` if it was present."""
        with self._lock:
            if rid in self._store:
                self._drop_locked(rid, EVICT_EXPLICIT)
                return True
            return False

    def purge_expired(self) -> int:
        """Sweep all expired entries now. Returns the number removed."""
        with self._lock:
            return self._purge_expired_locked()

    def miss_reason(self, rid: str) -> str:
        """Explain why ``rid`` is not resolvable: ``ttl`` / ``lru`` /
        ``bytes`` / ``explicit`` (from the eviction ledger) or ``unknown`` (never
        seen, or evicted long enough ago to fall off the ledger)."""
        with self._lock:
            if rid in self._store and not self._is_expired(self._store[rid]):
                return "present"
            return self._evicted.get(rid, "unknown")

    def stats(self) -> Dict[str, Any]:
        """Observability snapshot for diagnostics / health endpoints."""
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "evicted_tracked": len(self._evicted),
                "total_bytes": self._total_bytes,
                "max_bytes": self._max_bytes,
            }

    def keys(self) -> list:
        """Return current (non-expired) handles, oldest-first."""
        with self._lock:
            self._purge_expired_locked()
            return list(self._store.keys())

    def __contains__(self, rid: str) -> bool:
        with self._lock:
            entry = self._store.get(rid)
            if entry is None:
                return False
            if self._is_expired(entry):
                self._drop_locked(rid, EVICT_TTL)
                return False
            return True

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._evicted.clear()
            self._total_bytes = 0


#: Module-level singleton — shared across the agent + MCP layers so a
#: result cached during a curated ``execute_tool`` call is visible to
#: a later ``audit_result`` request that arrives via the MCP server.
RESULT_CACHE = ResultCache()


_RESULT_MISS_HINTS = {
    "ttl": (
        "result_id expired (server result-cache TTL); the underlying fit may "
        "be stale — re-fit the estimator with as_handle=true for a fresh handle."
    ),
    "lru": (
        "result_id evicted (the result cache keeps only the most recent fits); "
        "re-fit the estimator with as_handle=true to obtain a fresh handle."
    ),
    "explicit": (
        "result_id was explicitly dropped; re-fit the estimator with " "as_handle=true."
    ),
    "unknown": (
        "unknown result_id (never cached, evicted long ago, or the server "
        "restarted); re-fit the estimator with as_handle=true."
    ),
}


def missing_result_error(rid: str) -> Dict[str, Any]:
    """Structured error for an unresolvable ``result_id``.

    The twin of :func:`statspai.agent._data_cache.missing_handle_error`:
    ``error_kind="missing_result_handle"`` plus the cache's
    ``miss_reason`` so an agent can tell "expired, re-fit" from "never
    existed" without parsing prose.
    """
    reason = RESULT_CACHE.miss_reason(rid)
    return {
        "error": f"result_id {rid!r} not found in cache",
        "error_kind": "missing_result_handle",
        "miss_reason": reason,
        "hint": _RESULT_MISS_HINTS.get(reason, _RESULT_MISS_HINTS["unknown"]),
        "available_result_ids": RESULT_CACHE.keys(),
    }


__all__ = [
    "RESULT_CACHE",
    "ResultCache",
    "CacheEntry",
    "EVICT_BYTES",
    "missing_result_error",
]
