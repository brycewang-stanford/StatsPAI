"""Which StatsPAI code produced a result: version, git revision, install.

``__version__`` alone does not identify the code.  Between releases many
commits -- including correctness fixes that change numbers -- share one
version string, and an editable install keeps the package metadata of
the day it was installed (``pip`` can report 1.11.4 for a 1.32.0 tree).
A result is traceable only with the git revision of the source it ran
from, which this module resolves once per process and without touching
the import path (``sp.version_info`` and the provenance records call it
lazily).
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["source_revision", "version_info"]

_PKG_DIR = Path(__file__).resolve().parent


def _checkout_root(start: Path, max_up: int = 4) -> Optional[Path]:
    """The enclosing git checkout (``.git`` dir, or file for a worktree)."""
    for cand in [start, *list(start.parents)[:max_up]]:
        if (cand / ".git").exists():
            return cand
    return None


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


@lru_cache(maxsize=1)
def source_revision() -> Optional[str]:
    """Short git revision of the imported StatsPAI source, if it has one.

    ``"eed77f6c"`` for a clean checkout, ``"eed77f6c+dirty"`` when the
    package sources differ from that commit, ``None`` for an installed
    wheel (no enclosing checkout) or when ``git`` is unavailable.

    Examples
    --------
    >>> from statspai._build_info import source_revision
    >>> rev = source_revision()
    >>> rev is None or isinstance(rev, str)
    True
    """
    root = _checkout_root(_PKG_DIR)
    if root is None:
        return None
    sha = _git(root, "rev-parse", "--short=8", "HEAD")
    if not sha:
        return None
    status = _git(
        root, "status", "--porcelain", "--untracked-files=no", "--", str(_PKG_DIR)
    )
    return f"{sha}+dirty" if status else sha


def _metadata_version() -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return version("statspai")
    except PackageNotFoundError:
        return None


def version_info() -> Dict[str, Any]:
    """Identify the StatsPAI code in use: version, git revision, install.

    Record this next to any replication result.  ``version`` is
    ``sp.__version__``; ``revision`` the git commit of the imported source
    (``None`` for an installed wheel), with ``+dirty`` when the package
    sources have uncommitted changes; ``installed_metadata`` the version
    ``pip`` reports.  ``consistent`` is False when that metadata
    disagrees with ``version`` -- a stale editable install, or a
    ``PYTHONPATH`` pointing at another tree -- and ``warning`` then says
    so.  Every result's provenance (``sp.get_provenance(result)``) carries
    the same ``statspai_revision``.

    Returns
    -------
    dict
        Keys ``version``, ``revision``, ``source``, ``installed_metadata``,
        ``consistent``, ``python`` and, when inconsistent, ``warning``.

    Examples
    --------
    >>> import statspai as sp
    >>> info = sp.version_info()
    >>> info["version"] == sp.__version__
    True
    >>> sorted(k for k in info if k != "warning")
    ['consistent', 'installed_metadata', 'python', 'revision', 'source', 'version']
    """
    from . import __version__

    meta = _metadata_version()
    info: Dict[str, Any] = {
        "version": __version__,
        "revision": source_revision(),
        "source": str(_PKG_DIR),
        "installed_metadata": meta,
        "consistent": meta is None or meta == __version__,
        "python": sys.version.split()[0],
    }
    if not info["consistent"]:
        info["warning"] = (
            f"pip metadata reports statspai {meta}, but the imported code "
            f"({_PKG_DIR}) is {__version__}: the editable install is stale "
            "or PYTHONPATH points at another tree. Cite `revision`, and "
            "refresh the install with `pip install -e <StatsPAI checkout>`."
        )
    return info
