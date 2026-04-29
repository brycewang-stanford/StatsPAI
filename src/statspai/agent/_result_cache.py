"""In-process LRU cache for fitted StatsPAI results.

Provides server-side state for the MCP layer so an agent can chain
``did → audit → sensitivity → honest_did`` without re-running the
estimator on every step.

Design
------

* **Bounded** — defaults to 32 results; bumped via
  ``STATSPAI_MCP_RESULT_CACHE_SIZE``. LRU eviction.
* **Process-local** — handles do not survive a server restart. Agents
  treat the absence of a result as a recoverable error and re-fit.
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

The cache uses a re-entrant lock around all mutations. The MCP server
loop is currently single-threaded, but tests exercise concurrent
access.
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


def _cache_size() -> int:
    raw = os.environ.get("STATSPAI_MCP_RESULT_CACHE_SIZE")
    if raw is None:
        return _DEFAULT_CACHE_SIZE
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_CACHE_SIZE


@dataclass
class CacheEntry:
    """One slot in the result cache."""

    obj: Any
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_metadata(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of this entry's provenance."""
        # Strip DataFrame / array / non-scalar arguments so the metadata
        # fits in an MCP resource without dragging the original dataset
        # back through the wire.
        clean: Dict[str, Any] = {}
        for k, v in self.arguments.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                if all(isinstance(x, (str, int, float, bool)) or x is None
                       for x in v):
                    clean[k] = list(v)
                else:
                    clean[k] = f"<{type(v).__name__}, len={len(v)}>"
            else:
                clean[k] = f"<{type(v).__name__}>"
        return {
            "tool": self.tool,
            "arguments": clean,
            "created_at": self.created_at,
            "result_class": type(self.obj).__name__,
        }


class ResultCache:
    """LRU cache mapping ``result_id → CacheEntry``."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        self._max_size = max_size or _cache_size()
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = threading.RLock()

    def put(self, obj: Any, *, tool: str = "",
            arguments: Optional[Dict[str, Any]] = None) -> str:
        """Cache ``obj`` and return its newly-minted handle."""
        rid = "r_" + secrets.token_hex(4)
        with self._lock:
            self._store[rid] = CacheEntry(
                obj=obj, tool=tool,
                arguments=dict(arguments or {}),
            )
            self._store.move_to_end(rid)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)
        return rid

    def get(self, rid: str) -> Optional[Any]:
        """Return the cached object for ``rid``, or ``None``."""
        with self._lock:
            entry = self._store.get(rid)
            if entry is None:
                return None
            self._store.move_to_end(rid)
            return entry.obj

    def get_entry(self, rid: str) -> Optional[CacheEntry]:
        """Return the full entry (object + provenance) or ``None``."""
        with self._lock:
            entry = self._store.get(rid)
            if entry is None:
                return None
            self._store.move_to_end(rid)
            return entry

    def keys(self) -> list:
        """Return current handles, oldest-first."""
        with self._lock:
            return list(self._store.keys())

    def __contains__(self, rid: str) -> bool:
        with self._lock:
            return rid in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


#: Module-level singleton — shared across the agent + MCP layers so a
#: result cached during a curated ``execute_tool`` call is visible to
#: a later ``audit_result`` request that arrives via the MCP server.
RESULT_CACHE = ResultCache()


__all__ = ["RESULT_CACHE", "ResultCache", "CacheEntry"]
