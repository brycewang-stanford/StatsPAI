"""Extrapolants of SIMEX: curves in ``lambda`` read off at ``lambda = -1``.

Quadratic and linear least squares, the rational ``a + b / (c + lambda)``
(R ``fitting.method = "nonlinear"``) and the log-linear curve of R
``mcsimex``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy import optimize


def _poly_at_minus_one(grid: np.ndarray, values: np.ndarray, degree: int) -> np.ndarray:
    """Least-squares polynomial in lambda, evaluated at lambda = -1 (columns)."""
    V = np.vander(grid, degree + 1, increasing=True)
    coef = np.linalg.lstsq(V, values, rcond=None)[0]
    out: np.ndarray = np.vander(np.array([-1.0]), degree + 1, increasing=True) @ coef
    return out.ravel()


def _rational_at_minus_one(grid: np.ndarray, path: np.ndarray) -> float:
    """Fit ``a + b / (c + lambda)`` to one coefficient path; value at -1.

    Starting values interpolate the quadratic fit at ``0``, ``max / 2`` and
    ``max``; the curve is then refined by nonlinear least squares. Returns
    NaN when the fit has its pole at or beyond ``-1`` (no finite limit).
    """
    return _rational_fit(grid, path)[0]


def _rational_fit(
    grid: np.ndarray, path: np.ndarray
) -> Tuple[float, Tuple[float, float, float]]:
    """``_rational_at_minus_one`` with the fitted ``(a, b, c)``."""
    V = np.vander(grid, 3, increasing=True)
    q = np.linalg.lstsq(V, path, rcond=None)[0]
    h = float(grid.max()) / 2.0
    p0, p1, p2 = (float(q @ np.array([1.0, t, t * t])) for t in (0.0, h, 2 * h))
    scale = max(abs(p0), abs(p1), abs(p2), 1e-12)
    if abs(p1 - p2) <= 1e-12 * scale or abs(p0 - p1) <= 1e-12 * scale:
        # flat path: no error effect to undo
        return float(path.mean()), (float("nan"), float("nan"), float("nan"))
    ratio = (p0 - p1) / (p1 - p2)
    c0 = 2.0 * h / (ratio - 1.0) if abs(ratio - 1.0) > 1e-12 else 1e6
    if not np.isfinite(c0) or c0 <= 1.0:
        c0 = max(2.0, 4.0 * h)
    b0 = (p0 - p1) * c0 * (c0 + h) / h
    a0 = p0 - b0 / c0

    def resid(theta: np.ndarray) -> np.ndarray:
        a, b, c = theta
        out: np.ndarray = a + b / (c + grid) - path
        return out

    fit = optimize.least_squares(
        resid,
        np.array([a0, b0, c0]),
        bounds=([-np.inf, -np.inf, 1.0 + 1e-8], [np.inf, np.inf, np.inf]),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=5000,
    )
    a, b, c = (float(v) for v in fit.x)
    if c - 1.0 <= 1e-6:
        return float("nan"), (a, b, c)
    return a + b / (c - 1.0), (a, b, c)


def _loglinear_fit(grid: np.ndarray, path: np.ndarray) -> Tuple[float, float, float]:
    """``log(path + shift) = a + b lambda`` by least squares (R ``mcsimex``).

    ``shift = |min| + 1`` when the path has a non-positive value, else 0.
    Returns ``(a, b, shift)``; the value at -1 is ``exp(a - b) - shift``.
    """
    low = float(path.min())
    shift = (abs(low) + 1.0) if low <= 0 else 0.0
    V = np.vander(grid, 2, increasing=True)
    a, b = np.linalg.lstsq(V, np.log(path + shift), rcond=None)[0]
    return float(a), float(b), shift


def _extrapolate(
    grid: np.ndarray, values: np.ndarray, method: str
) -> Tuple[np.ndarray, List[int]]:
    """Extrapolate every column of ``values`` (rows = grid) to lambda = -1.

    A column whose rational fit has no finite limit at -1 falls back to the
    quadratic; the indices of those columns are returned.
    """
    if method == "linear":
        return _poly_at_minus_one(grid, values, 1), []
    if method == "loglinear":
        fits = [_loglinear_fit(grid, values[:, j]) for j in range(values.shape[1])]
        return np.array([np.exp(a - b) - sh for a, b, sh in fits]), []
    quad = _poly_at_minus_one(grid, values, 2)
    if method == "quadratic":
        return quad, []
    out = np.array(
        [_rational_at_minus_one(grid, values[:, j]) for j in range(values.shape[1])]
    )
    failed = [j for j in range(out.size) if not np.isfinite(out[j])]
    out[failed] = quad[failed]
    return out, failed


def _curves(
    grid: np.ndarray, path: np.ndarray, method: str, fallback: Sequence[int]
) -> List[Tuple[str, Tuple[float, ...]]]:
    """Extrapolant and fitted parameters of each coefficient path."""
    out: List[Tuple[str, Tuple[float, ...]]] = []
    for j in range(path.shape[1]):
        params = _rational_fit(grid, path[:, j])[1] if method == "nonlinear" else ()
        if method == "nonlinear" and j not in fallback and np.all(np.isfinite(params)):
            out.append(("nonlinear", params))
        elif method == "loglinear":
            out.append(("loglinear", _loglinear_fit(grid, path[:, j])))
        else:
            out.append(("quadratic" if method == "nonlinear" else method, ()))
    return out
