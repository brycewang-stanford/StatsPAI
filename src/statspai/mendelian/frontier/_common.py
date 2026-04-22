"""Shared helpers for the v1.6 MR Frontier estimators.

Keep tiny — single-purpose utilities only.  If any helper grows
beyond ~30 lines, move it into the method file that uses it.
"""

from __future__ import annotations

import numpy as np


def as_float_arrays(*arrays):
    """Cast a bundle of inputs to float ndarrays and verify equal length.

    Raises
    ------
    ValueError
        On length mismatch, or when fewer than 3 SNPs are supplied
        (the MR frontier family assumes over-identified IV).
    """
    out = [np.asarray(a, dtype=float) for a in arrays]
    n = len(out[0])
    for a in out[1:]:
        if len(a) != n:
            raise ValueError(
                f"length mismatch: {n} vs {len(a)} in mr.frontier inputs"
            )
    if n < 3:
        raise ValueError(
            f"MR frontier methods require >= 3 SNPs; got {n}"
        )
    return out


def harmonize_signs(bx: np.ndarray, by: np.ndarray):
    """Flip signs so all exposure betas are positive (MR harmonic convention)."""
    flip = bx < 0
    bx = bx.copy()
    by = by.copy()
    bx[flip] = -bx[flip]
    by[flip] = -by[flip]
    return bx, by


def mean_f_statistic(bx: np.ndarray, sx: np.ndarray) -> float:
    """Per-SNP F = (bx/sx)^2; return arithmetic mean (Staiger-Stock 1997)."""
    return float(np.mean((bx / sx) ** 2))
