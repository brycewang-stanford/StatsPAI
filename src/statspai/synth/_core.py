"""
Shared low-level primitives for the synth (synthetic control) module.

Centralizes building blocks that were previously duplicated across
augsynth / cluster / scm / conformal / multi_outcome / scpi and others.
A single canonical definition prevents numerical drift and makes it
cheap to fix bugs, add kernels, or tweak optimizer settings.

Public surface (module-internal, underscore-prefixed to stay private
to statspai.synth):

    solve_simplex_weights(y, X, penalization=0.0, w0=None)
        -> w, the SCM simplex-constrained ridge-penalized weights.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import optimize


def solve_simplex_weights(
    y: np.ndarray,
    X: np.ndarray,
    penalization: float = 0.0,
    w0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve the SCM simplex-constrained (optionally ridge-penalized) problem

        min_w ||y - X @ w||^2 + penalization * ||w||^2
        s.t.  w_j >= 0,  sum(w) = 1.

    Parameters
    ----------
    y : (T,) array
        Target vector (e.g. treated unit's pre-treatment outcomes).
    X : (T, J) array
        Donor matrix, columns = donors, rows = time (or stacked features).
        Callers with a (J, T) layout should pass ``X.T``.
    penalization : float, default 0.0
        Ridge penalty on the weights. 0.0 recovers the classical
        simplex LS problem.
    w0 : (J,) array or None
        Initial guess for the optimizer. ``None`` uses a uniform start
        ``(1/J, ..., 1/J)``.

    Returns
    -------
    w : (J,) array
        Optimal weights. Non-negative and summing to 1 up to the SLSQP
        tolerance ``ftol=1e-12``.

    Notes
    -----
    Uses SLSQP with an analytic gradient, which is ~3-5x faster than the
    finite-difference variant and produces identical optima on this
    convex QP.
    """
    J = X.shape[1]
    if J == 0:
        raise ValueError("No donors supplied (X has zero columns).")
    if J == 1:
        return np.array([1.0])

    def objective(w: np.ndarray) -> float:
        r = y - X @ w
        loss = float(r @ r)
        if penalization > 0:
            loss += float(penalization * (w @ w))
        return loss

    def jac(w: np.ndarray) -> np.ndarray:
        r = y - X @ w
        g = -2.0 * X.T @ r
        if penalization > 0:
            g = g + 2.0 * penalization * w
        return g

    if w0 is None:
        w0 = np.ones(J) / J

    result = optimize.minimize(
        objective, w0, jac=jac, method="SLSQP",
        bounds=[(0.0, 1.0)] * J,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x
