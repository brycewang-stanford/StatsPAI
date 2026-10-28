"""Randomization inference for ``sp.recentered_iv`` (private helpers)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def _ri_pvalue(a: float, c: float, a_s: np.ndarray, c_s: np.ndarray, b: float) -> float:
    """Two-sided RI p-value of ``beta = b`` (the realised draw counted once)."""
    t_obs = abs(a - b * c)
    t_sim = np.abs(a_s - b * c_s)
    tol = 1e-12 * max(t_obs, float(np.max(t_sim, initial=0.0)), 1e-300)
    exceed = int(np.sum(t_sim >= t_obs - tol))
    return (1.0 + exceed) / (len(a_s) + 1.0)


def _ri_confidence_set(
    a: float,
    c: float,
    a_s: np.ndarray,
    c_s: np.ndarray,
    alpha: float,
) -> List[Tuple[float, float]]:
    """Exact inversion of the RI test over ``b``.

    Draw ``s`` counts against ``b`` when ``|a_s - b c_s| >= |a - b c|``,
    i.e. when the quadratic ``f_s(b) = A b^2 + B b + C >= 0``. The count
    is piecewise constant between the roots of the ``f_s``, so a sweep
    over the sorted roots gives the accepted set exactly.
    """
    S = len(a_s)
    A = c_s**2 - c**2
    B = -2.0 * (a_s * c_s - a * c)
    C0 = a_s**2 - a**2
    scale_a = np.maximum(c_s**2 + c**2, 1e-300)
    scale_b = np.maximum(np.abs(a_s * c_s) + abs(a * c), 1e-300)
    quad = np.abs(A) > 1e-12 * scale_a
    lin = (~quad) & (np.abs(B) > 1e-12 * scale_b)

    count0 = 0
    roots: List[float] = []
    deltas: List[int] = []
    for s in range(S):
        if quad[s]:
            disc = B[s] ** 2 - 4.0 * A[s] * C0[s]
            if A[s] > 0:
                count0 += 1
            if disc <= 0:
                continue
            sq = np.sqrt(disc)
            r1, r2 = sorted(((-B[s] - sq) / (2 * A[s]), (-B[s] + sq) / (2 * A[s])))
            sign = -1 if A[s] > 0 else 1
            roots += [r1, r2]
            deltas += [sign, -sign]
        elif lin[s]:
            if B[s] < 0:
                count0 += 1
            roots.append(-C0[s] / B[s])
            deltas.append(1 if B[s] > 0 else -1)
        elif C0[s] >= 0:
            count0 += 1

    def accepted(count: int) -> bool:
        return (1.0 + count) / (S + 1.0) > alpha

    if not roots:
        return [(-np.inf, np.inf)] if accepted(count0) else []
    order = np.argsort(roots)
    r_sorted = np.asarray(roots)[order]
    d_sorted = np.asarray(deltas)[order]
    counts = count0 + np.concatenate([[0], np.cumsum(d_sorted)])
    # Segment k runs from r_sorted[k-1] to r_sorted[k] (open ends at +-inf).
    edges = np.concatenate([[-np.inf], r_sorted, [np.inf]])
    intervals: List[Tuple[float, float]] = []
    for k, cnt in enumerate(counts):
        if not accepted(int(cnt)):
            continue
        lo, hi = float(edges[k]), float(edges[k + 1])
        if intervals and intervals[-1][1] >= lo:
            intervals[-1] = (intervals[-1][0], hi)
        else:
            intervals.append((lo, hi))
    return intervals


def _balance_tests(
    zt: np.ndarray,
    zt_s: np.ndarray,
    R: np.ndarray,
    names: Sequence[str],
    w: np.ndarray,
) -> pd.DataFrame:
    """RI falsification tests that ``z - mu`` is orthogonal to each ``r``."""
    rows = []
    S = zt_s.shape[1]
    for j, name in enumerate(names):
        r = R[:, j]
        t = float(np.sum(w * zt * r) / np.sum(w))
        t_s = (w * r) @ zt_s / np.sum(w)
        tol = 1e-12 * max(abs(t), float(np.max(np.abs(t_s), initial=0.0)), 1e-300)
        p = (1.0 + np.sum(np.abs(t_s) >= abs(t) - tol)) / (S + 1.0)
        rows.append({"variable": name, "statistic": t, "ri_pvalue": float(p)})
    if R.shape[1] > 1:
        sw = np.sqrt(w)
        Rw = R * sw[:, None]
        proj = Rw @ np.linalg.pinv(Rw.T @ Rw) @ Rw.T

        def quad(v: np.ndarray) -> float:
            vw = v * sw
            return float(vw @ proj @ vw)

        t = quad(zt)
        t_s = np.array([quad(zt_s[:, s]) for s in range(S)])
        tol = 1e-12 * max(t, float(np.max(t_s, initial=0.0)), 1e-300)
        p = (1.0 + np.sum(t_s >= t - tol)) / (S + 1.0)
        rows.append({"variable": "joint", "statistic": t, "ri_pvalue": float(p)})
    return pd.DataFrame(rows)
