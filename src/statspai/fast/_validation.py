"""Private validation helpers shared by native fast estimators."""

from __future__ import annotations

import operator
from typing import Any, Optional, Sequence

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability


def positive_int(value: Any, *, name: str, context: str) -> int:
    """Return ``value`` as an int after requiring ``value >= 1``."""
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be a positive integer"
        ) from exc
    if isinstance(value, bool) or parsed < 1:
        raise MethodIncompatibility(f"{context}: {name} must be a positive integer")
    return int(parsed)


def nonnegative_finite_float(value: Any, *, name: str, context: str) -> float:
    """Return ``value`` as float after requiring finite ``value >= 0``."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and non-negative"
        ) from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and non-negative"
        )
    return parsed


def open_unit_float(value: Any, *, name: str, context: str) -> float:
    """Return ``value`` as float after requiring finite ``0 < value < 1``."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and in the open interval (0, 1)"
        ) from exc
    if not np.isfinite(parsed) or not (0.0 < parsed < 1.0):
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and in the open interval (0, 1)"
        )
    return parsed


def nonempty_sample(n_obs: int, *, context: str) -> None:
    """Reject empty estimator inputs before linear algebra is attempted."""
    if int(n_obs) < 1:
        raise DataInsufficient(f"{context}: data must contain at least one row")


def positive_weight_mass(weights: np.ndarray, *, context: str) -> None:
    """Reject samples whose validated weights have no positive mass."""
    if not (np.asarray(weights, dtype=np.float64) > 0.0).any():
        raise DataInsufficient(f"{context}: weights contain no positive mass")


#: A regressor whose residualised weighted sum of squares is at most this
#: fraction of its centred raw one is treated as absorbed by the FEs.
ABSORBED_REL = 1e-20
#: Smallest eigenvalue of the unit-diagonal Gram matrix of the
#: residualised regressors below which they are treated as collinear
#: (one regressor explains another with R^2 > 1 - COLLINEAR_EIG).
COLLINEAR_EIG = 1e-12
#: An absorbed regressor keeps less than this fraction of its centred sum
#: of squares; the noise test only applies below it.
NOISE_KEPT = 1e-6


def require_identified_regressors(
    X_raw: np.ndarray,
    X_dem: np.ndarray,
    names: Sequence[str],
    *,
    context: str,
    weights: Optional[np.ndarray] = None,
    tol_abs: float = 0.0,
    tol_rel: float = 0.0,
) -> None:
    """Raise when a residualised regressor carries no identifying variation.

    Two failures are checked after the FE residualisation:

    * **absorbed**: a regressor that lies in the span of the FEs leaves
      only the residualisation's convergence noise. It is flagged when its
      weighted sum of squares is at most :data:`ABSORBED_REL` of the
      centred raw one, or when its weighted root mean square is at most
      ten times the residualisation's stopping threshold for that column,
      ``tol_abs + tol_rel * max|x|``, *and* the residualisation removed all but
      :data:`NOISE_KEPT` of its variation (a regressor that is merely
      small in scale keeps its variation and is not flagged).
    * **collinear**: the regressors that remain are linearly dependent:
      the smallest eigenvalue of their unit-diagonal Gram matrix is at
      most :data:`COLLINEAR_EIG`. When one residualised column is
      constant (an intercept, in a model without FEs) the others are
      centred first, so a regressor with a large mean and little spread
      is not mistaken for a copy of the intercept.

    Checking this explicitly matters because inverting the normal
    equations only fails on an *exactly* singular matrix, and neither the
    NumPy nor the Rust residualisation returns exact zeros in general: an
    absorbed column comes back as rounding noise and the inverse is then
    finite garbage.
    """
    X_raw = np.asarray(X_raw, dtype=np.float64)
    X_dem = np.asarray(X_dem, dtype=np.float64)
    if X_dem.ndim != 2 or X_dem.shape[1] == 0 or X_dem.shape[0] == 0:
        return
    n, p = X_dem.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    wsum = float(w.sum())
    if wsum <= 0.0:
        return
    names = list(names)

    def wsq(A: np.ndarray) -> np.ndarray:
        # Column-wise weighted sums of squares, without an (n, p) temporary.
        if weights is None:
            return np.asarray(np.einsum("ij,ij->j", A, A))
        return np.asarray(np.einsum("i,ij,ij->j", w, A, A))

    ss_dem = wsq(X_dem)
    # The centred raw sum of squares is at most the uncentred one, so only
    # columns that lost almost everything need the exact (centred) value.
    cand = ~np.isfinite(ss_dem) | (ss_dem <= NOISE_KEPT * wsq(X_raw))
    absorbed = ~np.isfinite(ss_dem)
    for j in np.flatnonzero(cand & np.isfinite(ss_dem)):
        col = X_raw[:, j]
        mean = float(w @ col) / wsum
        ss_raw = float(w @ (col - mean) ** 2)
        noise = 10.0 * (tol_abs + tol_rel * float(np.abs(col).max()))
        absorbed[j] = ss_dem[j] <= ABSORBED_REL * ss_raw or (
            ss_dem[j] <= wsum * noise**2 and ss_dem[j] <= NOISE_KEPT * ss_raw
        )
    if absorbed.any():
        bad = [names[j] for j in np.flatnonzero(absorbed)]
        raise NumericalInstability(
            f"{context}: regressor(s) {bad} are perfectly collinear with the "
            "absorbed fixed effects (no variation left after residualisation).",
            recovery_hint=(
                "Drop these regressors: the fixed effects already absorb them "
                "(e.g. a unit-level or time-level variable next to unit or "
                "time effects)."
            ),
            diagnostics={"absorbed": bad},
        )
    if p < 2:
        return
    # A constant residualised column is an intercept (a model without FEs).
    constant = (X_dem == X_dem[0]).all(axis=0)
    if constant.sum() > 1:
        bad = [names[j] for j in np.flatnonzero(constant)]
        raise NumericalInstability(
            f"{context}: regressors {bad} are perfectly collinear after the "
            "fixed-effect residualisation (all constant).",
            recovery_hint="Keep at most one constant regressor (the intercept).",
            diagnostics={"collinear": bad},
        )
    cols = np.arange(p)
    Z = X_dem
    if constant.any():
        # Partial out the intercept: centre the other columns.
        cols = np.flatnonzero(~constant)
        if cols.size < 2:
            return
        Z = X_dem[:, cols]
        Z = Z - (w @ Z) / wsum
    gram = Z.T @ Z if weights is None else Z.T @ (w[:, None] * Z)
    scale = 1.0 / np.sqrt(np.diag(gram))
    corr = gram * scale[:, None] * scale[None, :]
    lam, vec = np.linalg.eigh(corr)
    if lam[0] <= COLLINEAR_EIG:
        v = np.abs(vec[:, 0])
        bad = [names[cols[j]] for j in np.flatnonzero(v >= 0.3 * v.max())]
        raise NumericalInstability(
            f"{context}: regressors {bad} are perfectly collinear after the "
            "fixed-effect residualisation (smallest eigenvalue of their "
            f"correlation matrix {lam[0]:.2e}).",
            recovery_hint="Drop one of the collinear regressors.",
            diagnostics={"collinear": bad, "min_eigenvalue": float(lam[0])},
        )
