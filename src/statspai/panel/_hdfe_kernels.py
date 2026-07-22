"""
Numba-accelerated kernels for the HDFE absorber.

The hot path in alternating projections is a triple pass over a column:

    1. bincount-style sum over groups: ``s[g] += x[i]``
    2. division: ``m[g] = s[g] / c[g]``
    3. subtraction: ``x[i] -= m[codes[i]]``

NumPy + ``np.bincount`` does the first step in C but then allocates two
temporary arrays for steps 2–3. Numba fuses the three passes into a
cache-friendly loop with optional SIMD. On a 3-way FE panel with
200 000 observations we see ~3× speed-up over the pure-NumPy fallback.

This module is optional: if Numba is not installed, :func:`sweep` /
:func:`sweep_weighted` transparently fall back to the NumPy path so the
public API is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

_DecoratedFn = TypeVar("_DecoratedFn", bound=Callable[..., Any])

try:  # pragma: no cover — import-time branch
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(
        *_args: Any,
        **_kwargs: Any,
    ) -> Callable[[_DecoratedFn], _DecoratedFn]:  # type: ignore[misc]
        def deco(fn: _DecoratedFn) -> _DecoratedFn:
            return fn

        return deco


_NUMBA_CACHE = _HAS_NUMBA and Path(__file__).exists()


# ======================================================================
# Unweighted sweep
# ======================================================================


@njit(cache=_NUMBA_CACHE, fastmath=True)  # type: ignore[misc]
def _sweep_numba(
    col: np.ndarray,
    codes: np.ndarray,
    counts: np.ndarray,
) -> None:
    G = counts.shape[0]  # pragma: no cover
    sums = np.zeros(G, dtype=np.float64)  # pragma: no cover
    n = col.shape[0]  # pragma: no cover
    # Pass 1: accumulate sums per group
    for i in range(n):  # pragma: no cover
        sums[codes[i]] += col[i]  # pragma: no cover
    # Pass 2: convert to means in place
    for g in range(G):  # pragma: no cover
        if counts[g] > 0.0:  # pragma: no cover
            sums[g] = sums[g] / counts[g]  # pragma: no cover
    # Pass 3: subtract group mean from each observation
    for i in range(n):  # pragma: no cover
        col[i] -= sums[codes[i]]  # pragma: no cover


def _sweep_numpy(
    col: np.ndarray,
    codes: np.ndarray,
    counts: np.ndarray,
) -> None:
    sums = np.bincount(  # pragma: no cover
        codes,
        weights=col,
        minlength=counts.size,
    )
    col -= (sums / counts)[codes]


def sweep(col: np.ndarray, codes: np.ndarray, counts: np.ndarray) -> None:
    """In-place group-mean demean of ``col`` by integer ``codes``.

    Uses the Numba kernel when available, otherwise the NumPy path.
    ``col`` must be contiguous ``float64``; ``codes`` must be ``int64``
    in ``[0, counts.size)``; ``counts`` is the group size array.
    """
    if _HAS_NUMBA:
        _sweep_numba(col, codes, counts)
    else:  # pragma: no cover
        _sweep_numpy(col, codes, counts)


# ======================================================================
# Weighted sweep
# ======================================================================


@njit(cache=_NUMBA_CACHE, fastmath=True)  # type: ignore[misc]
def _sweep_weighted_numba(
    col: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    wsum: np.ndarray,
) -> None:
    G = wsum.shape[0]  # pragma: no cover
    sums = np.zeros(G, dtype=np.float64)  # pragma: no cover
    n = col.shape[0]  # pragma: no cover
    for i in range(n):  # pragma: no cover
        sums[codes[i]] += col[i] * weights[i]  # pragma: no cover
    for g in range(G):  # pragma: no cover
        if wsum[g] > 0.0:  # pragma: no cover
            sums[g] = sums[g] / wsum[g]  # pragma: no cover
    for i in range(n):  # pragma: no cover
        col[i] -= sums[codes[i]]  # pragma: no cover


def _sweep_weighted_numpy(
    col: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    wsum: np.ndarray,
) -> None:
    sums = np.bincount(  # pragma: no cover
        codes,
        weights=col * weights,
        minlength=wsum.size,
    )
    col -= (sums / wsum)[codes]


def sweep_weighted(
    col: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    wsum: np.ndarray,
) -> None:
    """Weighted in-place group-mean demean of ``col``.

    Each observation ``i`` contributes ``col[i] * weights[i]`` to the
    numerator; the denominator is the precomputed per-group weight sum
    ``wsum``. Uses Numba if available.
    """
    if _HAS_NUMBA:
        _sweep_weighted_numba(col, weights, codes, wsum)
    else:  # pragma: no cover
        _sweep_weighted_numpy(col, weights, codes, wsum)


# ======================================================================
# Varying-slope sweep
# ======================================================================
#
# A varying-slope term ``i.g#c.x`` (Stata) / ``g[[x]]`` (fixest) absorbs the
# columns ``{x · 1[g = j]}_j``: one slope per level of ``g``, and *no*
# intercepts. The projection onto the orthogonal complement therefore
# residualizes ``col`` against ``x`` **within** each level of ``g``:
#
#     col_i -= x_i · (Σ_{g} x·col) / (Σ_{g} x²)
#
# The intercept-bearing variant ``i.g##c.x`` / ``g[x]`` additionally absorbs
# the level dummies ``1[g = j]``, i.e. it residualizes against ``[1, x]``
# within each level:
#
#     col_i -= α_g + β_g · x_i
#
# Doing the ``[1, x]`` fit jointly (rather than alternating an ordinary
# demean with a slope sweep) makes the term an exact one-pass projection,
# which keeps the alternating-projection loop over *distinct* absorbed
# dimensions well-conditioned.
#
# Degenerate levels — a single observation, or zero within-level variance
# in ``x`` — make the normal equation singular. The caller precomputes
# ``inv_denom`` with a 0.0 sentinel for those levels (see
# ``hdfe._slope_stats``), which sets ``β_g = 0`` without ever dividing by
# ~0. In the slope-only case that leaves the observation untouched (the
# absorbed column is identically zero, so it projects out nothing); in the
# intercept case it degrades to a plain group demean, which is the correct
# projection once the collinear slope column is dropped.


@njit(cache=_NUMBA_CACHE, fastmath=True)  # type: ignore[misc]
def _sweep_slope_numba(
    col: np.ndarray,
    x: np.ndarray,
    codes: np.ndarray,
    gsum: np.ndarray,
    xsum: np.ndarray,
    inv_denom: np.ndarray,
    with_intercept: bool,
) -> None:
    G = inv_denom.shape[0]  # pragma: no cover
    n = col.shape[0]  # pragma: no cover
    sc = np.zeros(G, dtype=np.float64)  # pragma: no cover
    scx = np.zeros(G, dtype=np.float64)  # pragma: no cover
    for i in range(n):  # pragma: no cover
        g = codes[i]  # pragma: no cover
        sc[g] += col[i]  # pragma: no cover
        scx[g] += x[i] * col[i]  # pragma: no cover
    alpha = np.zeros(G, dtype=np.float64)  # pragma: no cover
    beta = np.zeros(G, dtype=np.float64)  # pragma: no cover
    for g in range(G):  # pragma: no cover
        if with_intercept:  # pragma: no cover
            # Centered cross-product: Σxy − ΣxΣy/Σ1 within the level.
            cross = scx[g] - xsum[g] * sc[g] / gsum[g]  # pragma: no cover
            beta[g] = cross * inv_denom[g]  # pragma: no cover
            alpha[g] = (sc[g] - beta[g] * xsum[g]) / gsum[g]  # pragma: no cover
        else:  # pragma: no cover
            beta[g] = scx[g] * inv_denom[g]  # pragma: no cover
    for i in range(n):  # pragma: no cover
        g = codes[i]  # pragma: no cover
        col[i] -= alpha[g] + beta[g] * x[i]  # pragma: no cover


def _sweep_slope_numpy(
    col: np.ndarray,
    x: np.ndarray,
    codes: np.ndarray,
    gsum: np.ndarray,
    xsum: np.ndarray,
    inv_denom: np.ndarray,
    with_intercept: bool,
) -> None:
    G = inv_denom.size
    sc = np.bincount(codes, weights=col, minlength=G)
    scx = np.bincount(codes, weights=col * x, minlength=G)
    if with_intercept:
        beta = (scx - xsum * sc / gsum) * inv_denom
        alpha = (sc - beta * xsum) / gsum
        col -= alpha[codes] + beta[codes] * x
    else:
        beta = scx * inv_denom
        col -= beta[codes] * x


def sweep_slope(
    col: np.ndarray,
    x: np.ndarray,
    codes: np.ndarray,
    gsum: np.ndarray,
    xsum: np.ndarray,
    inv_denom: np.ndarray,
    with_intercept: bool = False,
) -> None:
    """In-place varying-slope residualization of ``col`` within ``codes``.

    Parameters
    ----------
    col : ndarray (n,)
        Column to residualize, contiguous float64. Modified in place.
    x : ndarray (n,)
        The continuous variable whose slope varies by group.
    codes : ndarray (n,) int64
        Dense group codes in ``[0, G)``.
    gsum : ndarray (G,)
        Per-group observation count (unweighted path). Only read when
        ``with_intercept`` is True.
    xsum : ndarray (G,)
        Per-group ``Σ x``. Only read when ``with_intercept`` is True.
    inv_denom : ndarray (G,)
        Reciprocal of the per-group normal-equation denominator, with
        ``0.0`` marking degenerate (rank-deficient) groups. Built by
        :func:`statspai.panel.hdfe._slope_stats`.
    with_intercept : bool, default False
        False → absorb slopes only (Stata ``i.g#c.x``). True → absorb
        level intercepts *and* slopes (Stata ``i.g##c.x``).
    """
    if _HAS_NUMBA:
        _sweep_slope_numba(col, x, codes, gsum, xsum, inv_denom, with_intercept)
    else:  # pragma: no cover
        _sweep_slope_numpy(col, x, codes, gsum, xsum, inv_denom, with_intercept)


@njit(cache=_NUMBA_CACHE, fastmath=True)  # type: ignore[misc]
def _sweep_slope_weighted_numba(
    col: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    gsum: np.ndarray,
    xsum: np.ndarray,
    inv_denom: np.ndarray,
    with_intercept: bool,
) -> None:
    G = inv_denom.shape[0]  # pragma: no cover
    n = col.shape[0]  # pragma: no cover
    sc = np.zeros(G, dtype=np.float64)  # pragma: no cover
    scx = np.zeros(G, dtype=np.float64)  # pragma: no cover
    for i in range(n):  # pragma: no cover
        g = codes[i]  # pragma: no cover
        wc = weights[i] * col[i]  # pragma: no cover
        sc[g] += wc  # pragma: no cover
        scx[g] += x[i] * wc  # pragma: no cover
    alpha = np.zeros(G, dtype=np.float64)  # pragma: no cover
    beta = np.zeros(G, dtype=np.float64)  # pragma: no cover
    for g in range(G):  # pragma: no cover
        if with_intercept:  # pragma: no cover
            cross = scx[g] - xsum[g] * sc[g] / gsum[g]  # pragma: no cover
            beta[g] = cross * inv_denom[g]  # pragma: no cover
            alpha[g] = (sc[g] - beta[g] * xsum[g]) / gsum[g]  # pragma: no cover
        else:  # pragma: no cover
            beta[g] = scx[g] * inv_denom[g]  # pragma: no cover
    for i in range(n):  # pragma: no cover
        g = codes[i]  # pragma: no cover
        col[i] -= alpha[g] + beta[g] * x[i]  # pragma: no cover


def _sweep_slope_weighted_numpy(
    col: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    gsum: np.ndarray,
    xsum: np.ndarray,
    inv_denom: np.ndarray,
    with_intercept: bool,
) -> None:
    G = inv_denom.size
    wc = col * weights
    sc = np.bincount(codes, weights=wc, minlength=G)
    scx = np.bincount(codes, weights=wc * x, minlength=G)
    if with_intercept:
        beta = (scx - xsum * sc / gsum) * inv_denom
        alpha = (sc - beta * xsum) / gsum
        col -= alpha[codes] + beta[codes] * x
    else:
        beta = scx * inv_denom
        col -= beta[codes] * x


def sweep_slope_weighted(
    col: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    gsum: np.ndarray,
    xsum: np.ndarray,
    inv_denom: np.ndarray,
    with_intercept: bool = False,
) -> None:
    """Weighted in-place varying-slope residualization of ``col``.

    Identical to :func:`sweep_slope` except every group accumulation is
    weighted: ``gsum`` is ``Σ w``, ``xsum`` is ``Σ w·x``, and the
    denominator behind ``inv_denom`` is the weighted ``Σ w·x²`` (centered
    by ``Σ w·x`` when ``with_intercept``).
    """
    if _HAS_NUMBA:
        _sweep_slope_weighted_numba(
            col, x, weights, codes, gsum, xsum, inv_denom, with_intercept
        )
    else:  # pragma: no cover
        _sweep_slope_weighted_numpy(
            col, x, weights, codes, gsum, xsum, inv_denom, with_intercept
        )


__all__ = [
    "sweep",
    "sweep_weighted",
    "sweep_slope",
    "sweep_slope_weighted",
    "_HAS_NUMBA",
]
