"""Loader for the optional Rust kernels (the ``statspai_hdfe`` extension).

Every module that can use the Rust kernels imports them from here, so the
availability decision is made once per process and the same everywhere:

* ``statspai_hdfe`` is not installed -> NumPy / Numba paths, silently.
* ``STATSPAI_SKIP_RUST`` is set to a true value (``1``, ``true``, ``yes``)
  -> the extension is ignored even if installed.
* it is installed but older than :data:`MIN_API_VERSION` -> ignored, with
  one ``RuntimeWarning``: its entry points have different contracts.

``rust`` is the module or ``None``. :func:`info` reports the decision,
which :func:`statspai.fast.backend_info` exposes.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

#: Interface level of the extension this StatsPAI was written against.
MIN_API_VERSION = 2

INSTALL_HINT = 'pip install "statspai[rust]"'

_TRUE = {"1", "true", "yes", "on"}

rust: Any = None
_reason: Optional[str] = None
_found_version: Optional[str] = None
_found_api: Optional[int] = None

if os.environ.get("STATSPAI_SKIP_RUST", "").strip().lower() in _TRUE:
    _reason = "disabled by STATSPAI_SKIP_RUST"
else:
    try:
        import statspai_hdfe as _mod  # type: ignore[import-not-found]
    except ImportError:
        _reason = "statspai_hdfe is not installed"
    else:
        _found_version = getattr(_mod, "__version__", None)
        _found_api = int(getattr(_mod, "API_VERSION", 1))
        if _found_api < MIN_API_VERSION:
            _reason = (
                f"statspai_hdfe {_found_version} implements kernel API "
                f"{_found_api}; this StatsPAI needs API >= {MIN_API_VERSION}"
            )
            warnings.warn(
                f"{_reason}. The Rust kernels are disabled and the NumPy / "
                f"Numba paths are used instead; upgrade with {INSTALL_HINT}.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            rust = _mod

HAS_RUST: bool = rust is not None
VERSION: Optional[str] = _found_version if HAS_RUST else None


def get_num_threads() -> Optional[int]:
    """Threads the Rust kernels run on, or ``None`` without them."""
    if rust is None:
        return None
    return int(rust.get_num_threads())


def set_num_threads(n: int) -> None:
    """Size the Rust kernels' thread pool (``0`` = one per logical CPU)."""
    from .exceptions import MethodIncompatibility, MissingDependencyError

    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise MethodIncompatibility(
            f"set_num_threads: n must be a non-negative integer, got {n!r}",
            recovery_hint="Pass 0 for one thread per logical CPU, or a positive count.",
        )
    if rust is None:
        err = MissingDependencyError.for_package(
            "statspai_hdfe",
            extra="rust",
            pip_name="statspai-hdfe",
            purpose="the Rust kernels' thread pool",
        )
        err.diagnostics["reason"] = _reason
        raise err
    rust.set_num_threads(int(n))


def info() -> Dict[str, Any]:
    """What the loader decided, as a JSON-ready dict."""
    out: Dict[str, Any] = {
        "available": HAS_RUST,
        "version": _found_version,
        "api_version": _found_api,
        "min_api_version": MIN_API_VERSION,
        "threads": get_num_threads(),
        "reason": _reason,
        "location": getattr(rust, "__file__", None),
    }
    if not HAS_RUST:
        out["install"] = INSTALL_HINT
    return out
