"""Shared helpers for the v1.6 MR Frontier estimators.

Keep tiny — single-purpose utilities only.  If any helper grows
beyond ~30 lines, move it into the method file that uses it.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import MethodIncompatibility


def as_float_arrays(*arrays: object) -> list[np.ndarray]:
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
            raise ValueError(f"length mismatch: {n} vs {len(a)} in mr.frontier inputs")
    if n < 3:
        raise ValueError(f"MR frontier methods require >= 3 SNPs; got {n}")
    return out


def require_valid_summary(
    bx: np.ndarray,
    by: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    *,
    function: str,
    positive_sx: bool,
    nonzero_bx: bool = False,
) -> None:
    """Refuse non-finite associations and non-positive standard errors.

    ``positive_sx`` requires ``se_exposure > 0`` (methods that divide by
    it); otherwise it must only be non-negative.
    """
    for name, v in (
        ("beta_exposure", bx),
        ("beta_outcome", by),
        ("se_exposure", sx),
        ("se_outcome", sy),
    ):
        if not np.all(np.isfinite(v)):
            raise MethodIncompatibility(
                f"{function}: {name} contains missing or non-finite values.",
                recovery_hint="Drop the variants with missing summary statistics.",
                diagnostics={"argument": name, "n_bad": int((~np.isfinite(v)).sum())},
            )
    if np.any(sy <= 0):
        raise MethodIncompatibility(
            f"{function}: se_outcome must be strictly positive.",
            recovery_hint="Drop or correct variants with a non-positive SE.",
            diagnostics={"n_bad": int((sy <= 0).sum())},
        )
    bad_sx = (sx <= 0) if positive_sx else (sx < 0)
    if np.any(bad_sx):
        rule = "strictly positive" if positive_sx else "non-negative"
        raise MethodIncompatibility(
            f"{function}: se_exposure must be {rule}.",
            recovery_hint="Drop or correct variants with an invalid exposure SE.",
            diagnostics={"n_bad": int(bad_sx.sum())},
        )
    if nonzero_bx and np.any(bx == 0):
        raise MethodIncompatibility(
            f"{function}: beta_exposure is zero for some variants, so their "
            "Wald ratios are undefined.",
            recovery_hint="Drop variants with no association with the exposure.",
            diagnostics={"n_zero": int((bx == 0).sum())},
        )


def harmonize_signs(
    bx: np.ndarray,
    by: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip signs so all exposure betas are positive."""
    flip = bx < 0
    bx = bx.copy()
    by = by.copy()
    bx[flip] = -bx[flip]
    by[flip] = -by[flip]
    return bx, by


def mean_f_statistic(bx: np.ndarray, sx: np.ndarray) -> float:
    """Per-SNP F = (bx/sx)^2; return arithmetic mean (Staiger-Stock 1997)."""
    return float(np.mean((bx / sx) ** 2))
