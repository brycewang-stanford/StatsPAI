"""Fetch-on-demand for public datasets that StatsPAI does not ship.

Some replication datasets are too large for the wheel, or their licence
does not clearly allow redistribution. Their loaders download the file
from its canonical public URL the first time they are called, verify a
pinned SHA-256, and keep it in a user cache:

* the cache directory is ``$STATSPAI_DATA_DIR`` when set, otherwise
  ``$XDG_CACHE_HOME/statspai/datasets`` (``~/.cache/statspai/datasets``;
  ``%LOCALAPPDATA%\\statspai\\datasets`` on Windows);
* nothing runs at import time; only a loader call downloads;
* a file whose hash differs from the pinned one is never used (a cached
  copy is re-downloaded, a fresh download is discarded);
* offline, the loader raises :class:`~statspai.exceptions.DataInsufficient`
  naming the URL, the expected SHA-256 and the path to save the file at,
  so the data can be placed there by hand.

``STATSPAI_FETCH_DATASETS=1`` is read only by the test suite: tests that
need a fetched dataset skip unless it is cached or this variable allows
the download.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

ENV_DATA_DIR = "STATSPAI_DATA_DIR"
ENV_FETCH = "STATSPAI_FETCH_DATASETS"

#: A download larger than this many times the pinned size is cut off.
_SIZE_SLACK = 2


@dataclass(frozen=True)
class RemoteFile:
    """One pinned public file: where it lives and what its bytes must hash to."""

    filename: str
    url: str
    sha256: str
    size: int
    licence: str
    origin: str


def cache_dir() -> Path:
    """Directory that holds fetched datasets (not created here)."""
    env = os.environ.get(ENV_DATA_DIR)
    if env:
        return Path(env).expanduser()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "statspai" / "datasets"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base_path = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base_path / "statspai" / "datasets"


def cache_path(spec: RemoteFile) -> Path:
    return cache_dir() / spec.filename


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_cached(spec: RemoteFile) -> bool:
    """True when a copy with the pinned SHA-256 is in the cache."""
    path = cache_path(spec)
    try:
        return path.is_file() and _sha256(path) == spec.sha256
    except OSError:
        return False


def fetch_allowed_in_tests() -> bool:
    """``STATSPAI_FETCH_DATASETS`` is set to a true value (tests only)."""
    return os.environ.get(ENV_FETCH, "").strip().lower() in {"1", "true", "yes", "on"}


def _manual_hint(spec: RemoteFile, path: Path) -> str:
    return (
        f"Download {spec.url} and save it as {path} (SHA-256 {spec.sha256}); "
        f"set {ENV_DATA_DIR} to use another cache directory."
    )


def _download(spec: RemoteFile, timeout: float) -> Tuple[bytes, Optional[str]]:
    """The file's bytes, or an error string. Never raises."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        spec.url, headers={"User-Agent": "statspai-datasets"}
    )
    limit = spec.size * _SIZE_SLACK + 1
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # nosec B310
            data = resp.read(limit)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return b"", f"{type(exc).__name__}: {exc}"
    if len(data) >= limit:
        return b"", f"response exceeds {limit - 1} bytes"
    return data, None


def fetch(spec: RemoteFile, *, download: bool = True, timeout: float = 60.0) -> Path:
    """Local path of ``spec``'s verified bytes, downloading them if needed.

    Raises
    ------
    DataInsufficient
        The file is not cached and ``download`` is False, the download
        failed (offline, server error), or the downloaded bytes do not
        hash to the pinned SHA-256.
    """
    from ..exceptions import DataInsufficient

    path = cache_path(spec)
    if path.is_file():
        try:
            if _sha256(path) == spec.sha256:
                return path
        except OSError:
            pass
    diagnostics = {
        "dataset_file": spec.filename,
        "url": spec.url,
        "sha256": spec.sha256,
        "cache_path": str(path),
    }
    if not download:
        raise DataInsufficient(
            f"{spec.filename} is not in the dataset cache ({path}) and "
            "download=False.",
            recovery_hint="Call again with download=True (needs network). "
            + _manual_hint(spec, path),
            diagnostics={**diagnostics, "reason": "not_cached"},
        )
    data, error = _download(spec, timeout)
    if error:
        raise DataInsufficient(
            f"Could not download {spec.filename} from {spec.url}: {error}.",
            recovery_hint="Check the network connection and retry. "
            + _manual_hint(spec, path),
            diagnostics={**diagnostics, "reason": "download_failed", "error": error},
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec.sha256:
        raise DataInsufficient(
            f"The file downloaded from {spec.url} does not match the pinned "
            f"SHA-256 (got {digest}); it was not saved.",
            recovery_hint="The upstream file changed or the download was "
            "altered. Report it to StatsPAI; do not use an unverified copy.",
            diagnostics={**diagnostics, "reason": "checksum_mismatch", "got": digest},
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".part-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
