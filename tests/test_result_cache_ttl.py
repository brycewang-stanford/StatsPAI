"""TTL + reason-aware invalidation for the MCP result cache.

The cache is the agent's server-side working memory for chained workflows
(``did → audit → sensitivity``). These tests lock the new behaviour:

* TTL is **opt-in** — the default cache never expires (backward compatible).
* When a TTL is set, an aged handle resolves to a miss and is swept.
* A miss is *explained* (``ttl`` / ``lru`` / ``explicit`` / ``unknown``) so the
  MCP layer can give an agent a precise re-fit hint.

We age entries by editing ``created_at`` directly rather than sleeping, so the
suite stays fast and deterministic.
"""

import time

import pytest

from statspai.agent._result_cache import (
    ResultCache,
    EVICT_TTL,
    EVICT_LRU,
    EVICT_EXPLICIT,
)


def test_default_cache_has_no_ttl_and_never_expires():
    """Backward compatibility: no TTL unless explicitly configured."""
    c = ResultCache(max_size=8)
    assert c.stats()["ttl_seconds"] is None
    rid = c.put({"v": 1}, tool="did")
    # Even a very old entry stays resolvable when TTL is off.
    c._store[rid].created_at = time.time() - 10_000
    assert c.get(rid) == {"v": 1}


def test_ttl_expires_entry_and_records_reason():
    c = ResultCache(max_size=8, ttl_seconds=100.0)
    rid = c.put({"v": 1}, tool="did")
    assert c.get(rid) == {"v": 1}  # fresh
    # Age it past the TTL.
    c._store[rid].created_at = time.time() - 200.0
    assert c.get(rid) is None  # swept on access
    assert rid not in c  # contains agrees
    assert c.miss_reason(rid) == EVICT_TTL  # explained


def test_put_eagerly_purges_expired():
    c = ResultCache(max_size=8, ttl_seconds=100.0)
    rid_old = c.put({"v": "old"}, tool="did")
    c._store[rid_old].created_at = time.time() - 200.0
    # A later insert sweeps the stale one even though nobody read it.
    c.put({"v": "new"}, tool="iv")
    assert rid_old not in c.keys()
    assert c.miss_reason(rid_old) == EVICT_TTL


def test_lru_eviction_records_reason():
    c = ResultCache(max_size=2)
    a = c.put({"v": "a"}, tool="did")
    b = c.put({"v": "b"}, tool="iv")
    c.put({"v": "c"}, tool="rd")  # evicts the oldest (a)
    assert c.get(a) is None
    assert c.miss_reason(a) == EVICT_LRU
    assert c.get(b) is not None  # b still present


def test_explicit_eviction():
    c = ResultCache(max_size=4)
    rid = c.put({"v": 1}, tool="did")
    assert c.evict(rid) is True
    assert c.evict(rid) is False  # already gone
    assert c.get(rid) is None
    assert c.miss_reason(rid) == EVICT_EXPLICIT


def test_unknown_miss_reason():
    c = ResultCache(max_size=4)
    assert c.miss_reason("r_deadbeef") == "unknown"


def test_present_reason_for_live_handle():
    c = ResultCache(max_size=4)
    rid = c.put({"v": 1}, tool="did")
    assert c.miss_reason(rid) == "present"


def test_purge_expired_returns_count():
    c = ResultCache(max_size=8, ttl_seconds=50.0)
    rids = [c.put({"i": i}, tool="did") for i in range(3)]
    for rid in rids:
        c._store[rid].created_at = time.time() - 100.0
    assert c.purge_expired() == 3
    assert len(c) == 0


def test_eviction_ledger_is_bounded():
    from statspai.agent._result_cache import _EVICTION_LEDGER_SIZE

    c = ResultCache(max_size=1)
    # Force many LRU evictions; the ledger must not grow without bound.
    for _ in range(_EVICTION_LEDGER_SIZE + 50):
        c.put({}, tool="did")
    assert c.stats()["evicted_tracked"] <= _EVICTION_LEDGER_SIZE


def test_env_ttl_is_read(monkeypatch):
    monkeypatch.setenv("STATSPAI_MCP_RESULT_CACHE_TTL", "42")
    c = ResultCache(max_size=4)
    assert c.stats()["ttl_seconds"] == 42.0


def test_malformed_env_ttl_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("STATSPAI_MCP_RESULT_CACHE_TTL", "not-a-number")
    c = ResultCache(max_size=4)
    assert c.stats()["ttl_seconds"] is None


def test_nonpositive_env_ttl_is_ignored(monkeypatch):
    monkeypatch.setenv("STATSPAI_MCP_RESULT_CACHE_TTL", "0")
    assert ResultCache(max_size=4).stats()["ttl_seconds"] is None
    monkeypatch.setenv("STATSPAI_MCP_RESULT_CACHE_TTL", "-5")
    assert ResultCache(max_size=4).stats()["ttl_seconds"] is None
