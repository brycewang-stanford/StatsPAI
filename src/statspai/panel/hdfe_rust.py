"""Rust-kernel bridge for HDFE (branch-only; main has no Rust code).

This module is imported at runtime by the HDFE absorber only when
``feat/rust-hdfe`` wheels are available. On ``main`` the import falls
through to :mod:`statspai.panel._hdfe_kernels`, which is the Numba
path. Users never see an `ImportError` because the absorber wraps
every call to :func:`group_demean` in a try/except that falls back to
Numba (and from Numba to NumPy).

Contract:

1. ``group_demean(codes, y, sums, counts)`` must produce bit-equal
   output to the NumPy reference within ``1e-10`` absolute tolerance.
2. The caller owns all buffers — this bridge never reallocates.
3. The PyO3 extension never spawns threads without the caller
   explicitly opting in (we hold back Rayon until Phase 3).
"""
from __future__ import annotations

import numpy as np

try:
    # The compiled PyO3 extension lives at the top level as a sibling
    # of the ``statspai`` package. This mirrors the numpy/pandas model
    # where a binary extension and a pure-Python package sit
    # side-by-side in ``site-packages``.
    from statspai_hdfe import group_demean as _rust_group_demean  # type: ignore

    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    _rust_group_demean = None  # type: ignore


def group_demean_rust(
    codes: np.ndarray,
    y: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Demean ``y`` by group in place via the Rust kernel.

    Parameters
    ----------
    codes : np.ndarray[int64]
        Group code per observation in ``[0, counts.size)``.
    y : np.ndarray[float64]
        Contiguous 1-D outcome vector; mutated in place.
    sums : np.ndarray[float64]
        Scratch buffer of length ``counts.size``. Contents discarded;
        the caller should pre-allocate (avoids alloc on hot path).
    counts : np.ndarray[int64]
        Observations per group. Must all be positive.

    Raises
    ------
    RuntimeError
        If the compiled extension is unavailable. Callers should
        ``try/except`` and fall back to the Numba or NumPy kernel.
    """
    if not HAS_RUST:
        raise RuntimeError(
            "statspai_hdfe compiled extension is not installed; "
            "fall back to the Numba kernel."
        )

    # Enforce the ABI the Rust side expects. The cost is negligible on
    # already-compliant arrays and saves an obscure segfault otherwise.
    codes_c = np.ascontiguousarray(codes, dtype=np.int64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    sums_c = np.ascontiguousarray(sums, dtype=np.float64)
    counts_c = np.ascontiguousarray(counts, dtype=np.int64)

    _rust_group_demean(codes_c, y_c, sums_c, counts_c)

    # Propagate mutations back to the caller's buffer if the contiguity
    # coerce above made a copy.
    if y_c is not y:
        y[:] = y_c


__all__ = ['HAS_RUST', 'group_demean_rust']
