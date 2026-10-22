"""Fetch-on-demand datasets: pinned hash, user cache, loud offline failure.

No test here touches the network: ``_fetch._download`` is replaced by a
function that serves bytes from memory (or fails like an offline socket).
The gallery tests in ``tests/external_parity/test_did_paper_gallery.py``
exercise the real files.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import statspai as sp
from statspai.datasets import _did_gallery, _fetch

PAYLOAD = b"id,year,y\n1,2000,1.5\n1,2001,2.5\n"
SPEC = _fetch.RemoteFile(
    filename="toy.csv",
    url="https://example.invalid/toy.csv",
    sha256=hashlib.sha256(PAYLOAD).hexdigest(),
    size=len(PAYLOAD),
    licence="test",
    origin="test",
)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv(_fetch.ENV_DATA_DIR, str(tmp_path / "cache"))
    return tmp_path / "cache"


def _serve(payload, calls):
    def fake(spec, timeout):
        calls.append(spec.url)
        return payload, None

    return fake


def test_cache_dir_follows_the_environment_variable(cache):
    assert _fetch.cache_dir() == cache


def test_cache_dir_default_is_under_the_user_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(_fetch.ENV_DATA_DIR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    if not sys.platform.startswith("win"):
        assert _fetch.cache_dir() == tmp_path / "xdg" / "statspai" / "datasets"


def test_first_call_downloads_then_the_cache_serves(cache, monkeypatch):
    calls = []
    monkeypatch.setattr(_fetch, "_download", _serve(PAYLOAD, calls))
    path = _fetch.fetch(SPEC)
    assert path == cache / "toy.csv"
    assert path.read_bytes() == PAYLOAD
    assert _fetch.is_cached(SPEC)
    _fetch.fetch(SPEC)
    assert len(calls) == 1
    # No temporary files are left beside the cached copy.
    assert sorted(p.name for p in cache.iterdir()) == ["toy.csv"]


def test_a_mismatched_download_is_refused_and_not_saved(cache, monkeypatch):
    monkeypatch.setattr(_fetch, "_download", _serve(PAYLOAD + b"x", []))
    with pytest.raises(sp.DataInsufficient) as info:
        _fetch.fetch(SPEC)
    assert info.value.diagnostics["reason"] == "checksum_mismatch"
    assert not (cache / "toy.csv").exists()


def test_a_corrupted_cached_copy_is_replaced(cache, monkeypatch):
    cache.mkdir(parents=True)
    (cache / "toy.csv").write_bytes(b"tampered")
    assert not _fetch.is_cached(SPEC)
    calls = []
    monkeypatch.setattr(_fetch, "_download", _serve(PAYLOAD, calls))
    assert _fetch.fetch(SPEC).read_bytes() == PAYLOAD
    assert calls == [SPEC.url]


def test_offline_raises_with_url_hash_and_target_path(cache, monkeypatch):
    monkeypatch.setattr(
        _fetch, "_download", lambda spec, timeout: (b"", "URLError: offline")
    )
    with pytest.raises(sp.DataInsufficient) as info:
        _fetch.fetch(SPEC)
    err = info.value
    assert err.diagnostics["reason"] == "download_failed"
    for needle in (SPEC.url, SPEC.sha256, str(cache / "toy.csv")):
        assert needle in err.recovery_hint


def test_download_false_never_touches_the_network(cache, monkeypatch):
    def boom(spec, timeout):
        raise AssertionError("network used")

    monkeypatch.setattr(_fetch, "_download", boom)
    with pytest.raises(sp.DataInsufficient) as info:
        _fetch.fetch(SPEC, download=False)
    assert info.value.diagnostics["reason"] == "not_cached"


def test_oversized_responses_are_cut_off(monkeypatch):
    """The real downloader stops reading past twice the pinned size."""

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return b"x" * n

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: Resp())
    data, err = _fetch._download(SPEC, timeout=1)
    assert data == b"" and "exceeds" in err


def test_loader_reads_the_cached_file(cache, monkeypatch):
    """A gallery loader goes through the same pinned cache path."""
    spec = _did_gallery.BACON_DIVORCE
    cache.mkdir(parents=True)
    with pytest.raises(sp.DataInsufficient):
        sp.datasets.bacon_divorce(download=False)
    assert not any(cache.iterdir())
    assert spec.url.startswith("http://pped.org/")


def test_every_remote_file_is_pinned():
    for name, spec in _did_gallery.REMOTE.items():
        assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0, name
        assert spec.size > 0 and spec.url.startswith(("http://", "https://")), name
        assert spec.licence and spec.origin, name


def test_importing_datasets_downloads_nothing(tmp_path):
    code = (
        "import socket\n"
        "def blocked(*a, **k):\n"
        "    raise OSError('network used at import')\n"
        "socket.socket = blocked\n"
        "socket.create_connection = blocked\n"
        "import statspai as sp\n"
        "sp.list_datasets()\n"
        "print('ok')\n"
    )
    src = str(Path(sp.__file__).resolve().parent.parent)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        # the parent environment, not an empty one: Python on Windows does
        # not start without SYSTEMROOT. The blocked socket is the guard.
        env={
            **{k: v for k, v in os.environ.items() if k != "STATSPAI_FETCH_DATASETS"},
            "PYTHONPATH": src,
            "STATSPAI_DATA_DIR": str(tmp_path),
            "HOME": str(tmp_path),
        },
        timeout=300,
    )
    assert out.stdout.strip().splitlines()[-1] == "ok", out.stderr[-2000:]
    assert not any(tmp_path.glob("*.csv")) and not any(tmp_path.glob("*.dta"))


def test_mpdta_real_variant_is_reachable_and_flagged(cache, monkeypatch):
    with pytest.raises(sp.DataInsufficient):
        sp.datasets.mpdta(simulated=False, download=False)
    with pytest.raises(sp.DataInsufficient):
        sp.load_dataset("mpdta", simulated=False, download=False)


def test_jel_sample_is_validated():
    with pytest.raises(sp.MethodIncompatibility):
        sp.datasets.jel_medicaid(sample="2xT", download=False)
