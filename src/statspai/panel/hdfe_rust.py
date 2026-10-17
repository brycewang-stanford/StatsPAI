"""Legacy single-FE bridge to the Rust ``group_demean`` kernel.

Nothing in the package calls it; it is kept for code written against
the early Rust spike. Multi-way demeaning goes through
:func:`statspai.fast.demean`. ``statspai._rust_backend`` decides whether
the extension is usable.

Contract:

1. ``group_demean(codes, y, sums, counts)`` matches the NumPy group
   demean to rounding.
2. The caller owns all buffers — this bridge never reallocates.
"""

from __future__ import annotations

import numpy as np

from .._rust_backend import rust as _rust

HAS_RUST = _rust is not None
_rust_group_demean = _rust.group_demean if _rust is not None else None


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
    if not HAS_RUST:  # pragma: no cover
        raise RuntimeError(  # pragma: no cover
            "statspai_hdfe compiled extension is not installed; "
            "fall back to the Numba kernel."
        )

    # Enforce the ABI the Rust side expects. The cost is negligible on
    # already-compliant arrays and saves an obscure segfault otherwise.
    codes_c = np.ascontiguousarray(codes, dtype=np.int64)  # pragma: no cover
    y_c = np.ascontiguousarray(y, dtype=np.float64)  # pragma: no cover
    sums_c = np.ascontiguousarray(sums, dtype=np.float64)  # pragma: no cover
    counts_c = np.ascontiguousarray(counts, dtype=np.int64)  # pragma: no cover

    _rust_group_demean(codes_c, y_c, sums_c, counts_c)  # pragma: no cover

    # Propagate mutations back to the caller's buffer if the contiguity
    # coerce above made a copy.
    if y_c is not y:  # pragma: no cover
        y[:] = y_c  # pragma: no cover


__all__ = ["HAS_RUST", "group_demean_rust"]
