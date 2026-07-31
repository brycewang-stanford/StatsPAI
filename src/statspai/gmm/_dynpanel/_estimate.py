"""GMM solving primitives for dynamic panel models.

The estimator minimises ``(Δy - Wβ)' Z A Z' (Δy - Wβ)`` for a weight matrix
``A``.  Two choices of ``A`` matter:

* **one-step**: ``A₁ = (Σ_i Z_i' H Z_i)^{-1}``, where ``H`` encodes the MA(1)
  structure the first difference induces in i.i.d. errors.  This is the
  efficient weight *under homoskedasticity* and is what Stata's ``xtabond``
  reports by default.
* **two-step**: ``A₂ = (Σ_i Z_i' ê₁ ê₁' Z_i)^{-1}`` from the one-step
  residuals — efficient under arbitrary heteroskedasticity, at the cost of
  a finite-sample bias in the SEs that the Windmeijer (2005) correction
  addresses (see ``_inference``).
"""

from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple

import numpy as np

from ._moments import Design, first_difference_H, system_H

__all__ = [
    "safe_inv",
    "unit_H_blocks",
    "onestep_weight",
    "moment_covariance",
    "gmm_solve",
    "level_sigma2",
]


def safe_inv(M: np.ndarray, what: str, stacklevel: int = 3) -> np.ndarray:
    """Inverse that warns loudly instead of quietly returning garbage.

    Falls back to the Moore-Penrose pseudo-inverse so the computation can
    finish, but emits a warning naming the matrix — a singular weight matrix
    almost always means collinear regressors or an over-saturated instrument
    set, and both are user-actionable.
    """
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"{what} is singular (rank-deficient); falling back to the "
            f"pseudo-inverse. Results may be unreliable — check for collinear "
            f"regressors or an over-saturated instrument set.",
            stacklevel=stacklevel,
        )
        return np.linalg.pinv(M)


def unit_H_blocks(design: Design) -> List[np.ndarray]:
    """The a-priori error covariance ``H`` for each unit's stacked rows.

    Factored out of :func:`onestep_weight` so that the difference-in-Hansen
    machinery can re-form the one-step weight over an instrument *subset*
    without rebuilding the design.
    """
    system = design.has_level_equation
    operator = design.transform_operator
    blocks = []
    for rows in design.unit_rows:
        periods = design.row_period[rows]
        eqs = design.row_eq[rows]
        if operator is not None:
            blocks.append(_transform_H(operator, periods, eqs))
        elif system:
            blocks.append(system_H(periods, eqs))
        else:
            blocks.append(first_difference_H(periods))
    return blocks


def onestep_weight(design: Design) -> np.ndarray:
    """``A₁ = (Σ_i Z_i' H_i Z_i)^{-1}``.

    ``H`` is the MA(1) first-difference structure for a pure Arellano-Bond
    fit, and the stacked ``[[MM', M], [M', I]]`` block of Roodman's
    ``h(3)`` once a level equation is present.  Both are gap-aware: the
    off-diagonal links are keyed on the actual period distance, so an
    interior gap breaks them instead of pretending non-adjacent rows share
    an error term.
    """
    m = design.n_instruments
    ZHZ = np.zeros((m, m))
    for rows, H in zip(design.unit_rows, unit_H_blocks(design)):
        Zi = design.Z[rows]
        ZHZ += Zi.T @ H @ Zi
    return safe_inv(ZHZ, "GMM weight matrix Z'HZ")


def _transform_H(M: np.ndarray, periods: np.ndarray, eqs: np.ndarray) -> np.ndarray:
    """``H = [[M M', M], [M', I]]`` from a balanced-grid transform operator.

    ``M`` maps the balanced level grid onto the transformed rows, its row
    ``a`` being stored period ``a + 1``.  Restricting it to the rows and
    level periods a unit actually has gives that unit's ``H`` — the
    "all blocks the same" convention ``xtabond2`` documents.  Deriving
    ``H`` from ``M`` rather than hard-coding a band keeps every transform
    on one definition.
    """
    is_diff = eqs == 0
    n = periods.size
    H = np.zeros((n, n))
    d_idx = np.flatnonzero(is_diff)
    l_idx = np.flatnonzero(~is_diff)
    rows = periods[d_idx] - 1
    if d_idx.size:
        Md = M[rows]
        H[np.ix_(d_idx, d_idx)] = Md @ Md.T
    if l_idx.size:
        H[np.ix_(l_idx, l_idx)] = np.eye(l_idx.size)
        if d_idx.size:
            cross = M[np.ix_(rows, periods[l_idx])]
            H[np.ix_(d_idx, l_idx)] = cross
            H[np.ix_(l_idx, d_idx)] = cross.T
    return H


def moment_covariance(
    Z: np.ndarray, resid: np.ndarray, unit_rows: Sequence[np.ndarray]
) -> np.ndarray:
    """``Ω = Σ_i Z_i' ê_i ê_i' Z_i`` — the clustered-by-unit moment meat."""
    m = Z.shape[1]
    Omega = np.zeros((m, m))
    for rows in unit_rows:
        g = Z[rows].T @ resid[rows]
        Omega += np.outer(g, g)
    return Omega


def gmm_solve(
    W: np.ndarray, Z: np.ndarray, dy: np.ndarray, weight: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form linear GMM.

    Returns ``(beta, Minv, WZ)`` where ``Minv = (W'Z A Z'W)^{-1}`` doubles
    as the conventional (efficient-weight) variance factor.
    """
    WZ = W.T @ Z
    M = WZ @ weight @ WZ.T
    Minv = safe_inv(M, "moment matrix W'ZAZ'W")
    beta = Minv @ (WZ @ weight @ (Z.T @ dy))
    return beta, Minv, WZ


def level_sigma2(resid: np.ndarray, design: Design, k: int) -> float:
    """σ̂² of the idiosyncratic error, from the **transformed** rows only.

    A first-differenced error has variance ``2σ²`` under i.i.d. levels, so
    Stata's ``xtabond`` reports ``σ̂² = ê*'ê* / (2 (N* − k))`` where ``N*``
    counts transformed rows.  Level-equation residuals still contain the
    fixed effect ``α_i`` and therefore carry no information about ``σ²``;
    ``xtabond2`` excludes them for the same reason (it uses the same
    transformed-only sum, with ``2 N*`` rather than ``2 (N* − k)`` in the
    denominator — a finite-sample convention, so its Sargan statistic sits a
    factor ``N*/(N* − k)`` above the ``xtabond`` one).

    This is the scale used by the classical one-step VCE and by the Sargan
    statistic.
    """
    rows = design.row_eq == 0
    r = resid[rows]
    n_star = int(rows.sum())
    return float(r @ r) / (2.0 * max(n_star - k, 1))
