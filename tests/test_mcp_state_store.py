"""Handles on disk: ``STATSPAI_MCP_STATE_DIR``.

A fresh :class:`ResultCache` / :class:`DataCache` stands in for a
restarted server: it shares nothing with the instance that cached the
entry except the directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from statspai.agent import _state_store
from statspai.agent._data_cache import DataCache
from statspai.agent._result_cache import ResultCache

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    root = tmp_path / "state"
    monkeypatch.setenv(_state_store.STATE_DIR_ENV, str(root))
    monkeypatch.delenv(_state_store.STATE_MAX_BYTES_ENV, raising=False)
    _state_store._refused.clear()
    yield root
    _state_store._refused.clear()


def _files(root, kind="results"):
    folder = root / kind
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def test_off_by_default_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(_state_store.STATE_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    cache = ResultCache(max_size=2)
    rid = cache.put({"a": 1}, tool="t")
    assert cache.get(rid) == {"a": 1}
    assert not _state_store.enabled()
    assert list(tmp_path.iterdir()) == []
    assert ResultCache().get(rid) is None


def test_result_survives_a_restart_with_its_provenance(state_dir):
    first = ResultCache()
    rid = first.put({"estimate": 1.5}, tool="did", arguments={"y": "wage"})
    first.annotate(rid, {"_mcp_data_provenance": {"sha256": "abc"}})
    first.set_replay(rid, "sp.did(data=data, y='wage')")
    assert _files(state_dir) == [rid + ".meta", rid + ".pkl"]

    second = ResultCache()
    assert rid in second
    assert second.miss_reason(rid) == "present"
    entry = second.get_entry(rid)
    assert entry.obj == {"estimate": 1.5}
    assert entry.tool == "did"
    assert entry.arguments == {
        "y": "wage",
        "_mcp_data_provenance": {"sha256": "abc"},
    }
    assert entry.replay == "sp.did(data=data, y='wage')"
    assert entry.created_at == first.get_entry(rid).created_at


def test_frame_and_lineage_survive_a_restart(state_dir):
    df = pd.DataFrame({"x": np.arange(5.0), "g": list("aabbc")})
    first = DataCache()
    did = first.put(df, tool="transform_data", arguments={"parent_id": "d_00000000"})
    second = DataCache()
    pd.testing.assert_frame_equal(second.get(did), df)
    assert second.get_entry(did).arguments == {"parent_id": "d_00000000"}
    assert _files(state_dir, "data") == [did + ".meta", did + ".pkl"]
    assert _files(state_dir, "results") == []


def test_lru_eviction_frees_memory_but_keeps_the_handle(state_dir):
    cache = ResultCache(max_size=2)
    ids = [cache.put({"i": i}, tool="t") for i in range(4)]
    assert cache.stats()["size"] == 2
    # The evicted handle is read back, and the cache stays at its bound.
    assert cache.get(ids[0]) == {"i": 0}
    assert cache.stats()["size"] == 2
    for i, rid in enumerate(ids):
        assert cache.get(rid) == {"i": i}


def test_explicit_drop_and_clear_remove_the_disk_copy(state_dir):
    cache = ResultCache()
    a, b = cache.put({"i": 1}, tool="t"), cache.put({"i": 2}, tool="t")
    assert cache.evict(a)
    assert cache.get(a) is None and cache.miss_reason(a) == "explicit"
    assert ResultCache().get(a) is None
    assert ResultCache().get(b) == {"i": 2}
    cache.clear()
    assert _files(state_dir) == []
    assert ResultCache().get(b) is None


def test_ttl_is_measured_from_the_fit_across_restarts(state_dir):
    first = ResultCache()
    rid = first.put({"i": 1}, tool="t")
    first.get_entry(rid).created_at -= 3600
    first._persist_meta(rid, first.get_entry(rid))

    assert ResultCache(ttl_seconds=7200).get(rid) == {"i": 1}
    expired = ResultCache(ttl_seconds=60)
    assert expired.get(rid) is None
    assert expired.miss_reason(rid) == "ttl"
    assert _files(state_dir) == []


def test_a_tampered_file_is_discarded_not_unpickled(state_dir):
    rid = ResultCache().put({"i": 1}, tool="t")
    path = state_dir / "results" / (rid + ".pkl")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    fresh = ResultCache()
    assert fresh.get(rid) is None
    assert fresh.miss_reason(rid) == "unknown"
    assert _files(state_dir) == []


def test_an_entry_from_another_version_is_discarded(state_dir, monkeypatch):
    rid = ResultCache().put({"i": 1}, tool="t")
    monkeypatch.setattr(_state_store, "_package_version", lambda: "0.0.0-other")
    assert ResultCache().get(rid) is None
    assert _files(state_dir) == []


def test_an_unpicklable_result_stays_in_memory_only(state_dir):
    cache = ResultCache()
    before = _state_store.stats()["unpicklable"]
    rid = cache.put({"f": lambda x: x}, tool="t")
    assert cache.get(rid)["f"](3) == 3
    assert _files(state_dir) == []
    assert _state_store.stats()["unpicklable"] == before + 1
    assert ResultCache().get(rid) is None


def test_disk_budget_drops_the_least_recently_used(state_dir, monkeypatch):
    cache = ResultCache()
    blob = "x" * 4000
    monkeypatch.setenv(_state_store.STATE_MAX_BYTES_ENV, "10000")
    ids = []
    for i in range(4):
        ids.append(cache.put({"i": i, "blob": blob}, tool="t"))
        path = state_dir / "results" / (ids[-1] + ".pkl")
        os.utime(path, (1_000_000 + i, 1_000_000 + i))
    kept = {name[:-4] for name in _files(state_dir) if name.endswith(".pkl")}
    assert ids[-1] in kept and ids[0] not in kept
    assert (
        sum((state_dir / "results" / (r + ".pkl")).stat().st_size for r in kept)
        <= 10000
    )
    # No metadata file is left behind without its object.
    assert {n[:-5] for n in _files(state_dir) if n.endswith(".meta")} == kept


@posix_only
def test_a_directory_others_can_write_is_refused(state_dir):
    state_dir.mkdir()
    os.chmod(state_dir, 0o777)
    with pytest.warns(RuntimeWarning, match="writable by group or others"):
        rid = ResultCache().put({"i": 1}, tool="t")
    assert not _state_store.enabled()
    assert list(state_dir.iterdir()) == []
    assert ResultCache().get(rid) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warned once, not on every call
        ResultCache().put({"i": 2}, tool="t")


@posix_only
def test_files_are_private_to_the_server_user(state_dir):
    rid = ResultCache().put({"i": 1}, tool="t")
    assert (state_dir.stat().st_mode & 0o777) == 0o700
    for path in (
        state_dir / "hmac.key",
        state_dir / "results" / (rid + ".pkl"),
        state_dir / "results" / (rid + ".meta"),
    ):
        assert (path.stat().st_mode & 0o077) == 0, path


def test_ids_that_are_not_handles_never_touch_the_filesystem(state_dir):
    cache = ResultCache()
    cache.put({"i": 1}, tool="t")
    for bad in ("../hmac", "r_../../x", "r_ZZZZ", "", "r_" + "a" * 80):
        assert cache.get(bad) is None
        assert _state_store.load("results", bad) is None


_CHAIN = r"""
import json, sys
from statspai.agent import mcp_server as m

def call(name, **arguments):
    line = m.handle_request(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }))
    return json.loads(line)["result"]

if sys.argv[1] == "fit":
    rows = [{"y": 1 + 2 * i + (i % 5) - 2, "x": i} for i in range(60)]
    d = call("load_data", data_records=rows)["structuredContent"]["data_id"]
    r = call("regress", data_id=d, formula="y ~ x", as_handle=True)
    out = {"d": d, "r": r["structuredContent"]["result_id"]}
else:
    ids = json.loads(sys.argv[2])
    brief = call("brief_result", result_id=ids["r"])
    refit = call("regress", data_id=ids["d"], formula="y ~ x", detail="standard")
    out = {
        "brief_error": bool(brief.get("isError")),
        "brief": brief["structuredContent"],
        "refit_error": bool(refit.get("isError")),
        "refit": refit["structuredContent"],
    }
print("RESULT " + json.dumps(out, default=str))
"""


def _run(args, env):
    proc = subprocess.run(
        [sys.executable, "-c", _CHAIN, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=180,
        # posix_spawn, not fork: a fork taken after numerical code ran in
        # this process can crash in the child before exec (macOS).
        close_fds=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def test_an_analysis_chain_continues_in_a_new_server_process(tmp_path):
    """Fit in one process, audit in another: the real restart."""
    env = dict(os.environ)
    env[_state_store.STATE_DIR_ENV] = str(tmp_path / "state")
    ids = _run(["fit"], env)
    after = _run(["use", json.dumps(ids)], env)
    assert not after["brief_error"], after["brief"]
    assert after["brief"]["result_id"] == ids["r"]
    assert not after["refit_error"], after["refit"]
    assert after["refit"]["n_obs"] == 60
    assert after["refit"]["data_provenance"]["data_id"] == ids["d"]

    env.pop(_state_store.STATE_DIR_ENV)
    lost = _run(["use", json.dumps(ids)], env)
    assert lost["brief_error"] and lost["refit_error"]
    assert lost["brief"]["error_kind"] == "missing_result_handle"
    assert "STATSPAI_MCP_STATE_DIR" in lost["brief"]["hint"]
