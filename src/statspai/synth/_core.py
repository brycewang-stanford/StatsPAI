"""
Shared low-level primitives for the synth (synthetic control) module.

Two canonical solvers live here:

* ``solve_simplex_weights(y, X, penalization=0.0, w0=None)`` — plain
  simplex-constrained least-squares (ridge-penalized optional). Used by
  augsynth / cluster / conformal / scpi / multi_outcome to solve the
  inner W problem when the predictor is just pre-treatment outcomes.

* ``solve_synth_weights_adh(X1, X0, Z1, Z0, ...)`` — the full
  Abadie-Diamond-Hainmueller (2010) **nested V-W optimization**:

      outer: V* = argmin_V (Z1 - Z0 W(V))' (Z1 - Z0 W(V))
      inner: W(V) = argmin_w (X1 - X0 w)' V (X1 - X0 w)
             s.t.  w_j >= 0, sum(w) = 1

  Used by ``SyntheticControl`` in ``scm.py`` for canonical SCM with
  covariate matching.  Supports predictor standardization, multi-start
  initialisation (equal / regression / random Dirichlet), and a
  configurable outer optimizer (L-BFGS-B or Nelder-Mead).

``standardize_predictors(X1, X0)`` is an ADH-compliant preprocessing
step that rescales each row of ``[X1 | X0]`` to unit range, as
recommended in Abadie, Diamond & Hainmueller (2010, §4).

References
----------
Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic control
methods for comparative case studies. *JASA* 105(490), 493-505.

Kaul, A., Klößner, S., Pfeifer, G. & Schieler, M. (2015). Synthetic
control methods: Never use all pre-intervention outcomes together with
covariates.  *MPRA Working Paper*.

Abadie, A. (2021). Using synthetic controls: feasibility, data
requirements, and methodological aspects.  *Journal of Economic
Literature* 59(2), 391-425.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import optimize


# ---------------------------------------------------------------------------
# Basic simplex solver (inner W problem, reused across module)
# ---------------------------------------------------------------------------

def solve_simplex_weights(
    y: np.ndarray,
    X: np.ndarray,
    penalization: float = 0.0,
    w0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve

        min_w ||y - X @ w||^2 + penalization * ||w||^2
        s.t.  w_j >= 0,  sum(w) = 1.

    Parameters
    ----------
    y : (T,) array
        Target vector (e.g. treated unit's pre-treatment outcomes, or
        ``sqrt(V) * X1`` in the ADH inner problem).
    X : (T, J) array
        Donor design matrix.  Callers holding ``(J, T)`` layouts should
        pass ``X.T``.
    penalization : float, default 0.0
        Ridge penalty on ``w``.
    w0 : (J,) array or None
        Initial guess. ``None`` uses the uniform simplex point.

    Returns
    -------
    w : (J,) array
        Non-negative weights summing to 1 (up to ``ftol=1e-12``).
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


# ---------------------------------------------------------------------------
# Predictor standardisation (ADH 2010 §4)
# ---------------------------------------------------------------------------

def standardize_predictors(
    X1: np.ndarray,
    X0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rescale predictor rows to unit range using the combined treated +
    donor support (ADH 2010, §4).

    This is important so that ``V`` — a diagonal weight matrix on
    predictors — can be interpreted on a common scale and the outer
    optimization is not dominated by the predictor with the largest
    raw magnitude.

    Parameters
    ----------
    X1 : (K,) array
        Treated-unit predictor vector.
    X0 : (K, J) array
        Donor predictor matrix.

    Returns
    -------
    X1s : (K,) array
        Standardised treated vector.
    X0s : (K, J) array
        Standardised donor matrix.
    scale : (K,) array
        Row-wise scale factor (max - min across treated + donors).  Rows
        with zero range get ``scale = 1`` and are left untouched.
    """
    X1 = np.asarray(X1, dtype=np.float64).ravel()
    X0 = np.asarray(X0, dtype=np.float64)
    if X0.shape[0] != X1.shape[0]:
        raise ValueError(
            f"X1 has {X1.shape[0]} predictors but X0 has {X0.shape[0]}."
        )

    combined = np.column_stack([X1[:, None], X0])
    lo = combined.min(axis=1)
    hi = combined.max(axis=1)
    rng = hi - lo
    scale = np.where(rng > 1e-12, rng, 1.0)
    return X1 / scale, X0 / scale[:, None], scale


# ---------------------------------------------------------------------------
# ADH (2010) nested V-W solver
# ---------------------------------------------------------------------------

def _inner_w_given_v(
    V_diag: np.ndarray,
    X1: np.ndarray,
    X0: np.ndarray,
    penalization: float = 0.0,
    w0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Inner problem: W(V) = argmin_w (X1 - X0 w)' diag(V) (X1 - X0 w).

    Reformulated as unweighted simplex LS with
    ``y = sqrt(V) * X1``, ``X = sqrt(V)[:, None] * X0``.
    """
    sqrtV = np.sqrt(np.maximum(V_diag, 0.0))
    y = sqrtV * X1
    X = sqrtV[:, None] * X0
    return solve_simplex_weights(y, X, penalization=penalization, w0=w0)


def _v_from_params(v_params: np.ndarray, K: int) -> np.ndarray:
    """
    Map unconstrained ``v_params`` (length K) to a non-negative diagonal
    ``V`` with ``tr(V) = K``.

    Uses a softmax-like reparameterisation: ``V_k = K * exp(v_k) / sum_k
    exp(v_k)``.  This keeps the outer optimisation unconstrained while
    preserving the ADH scale convention ``tr(V) = K``.
    """
    # Numerical stability: subtract max
    vp = v_params - v_params.max()
    ev = np.exp(vp)
    s = ev.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones(K)
    return K * ev / s


def _regression_v_init(
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
) -> np.ndarray:
    """
    Regression-based V initialisation (R ``Synth`` default).

    Fit OLS of stacked pre-outcomes (treated + donors) on predictors and
    use squared coefficients (normalised to ``tr(V) = K``) as V.
    """
    K, J = X0.shape
    # Stacked design: rows = units (1 treated + J donors), cols = K predictors
    X_stack = np.column_stack([X1[:, None], X0]).T  # (J+1, K)
    # Response: unit-level mean of pre-treatment outcome
    y_stack = np.concatenate([[Z1.mean()], Z0.mean(axis=0)])
    try:
        beta, *_ = np.linalg.lstsq(X_stack, y_stack, rcond=None)
        v = beta ** 2
        if v.sum() <= 1e-12 or not np.all(np.isfinite(v)):
            return np.ones(K)
        return K * v / v.sum()
    except np.linalg.LinAlgError:
        return np.ones(K)


def solve_synth_weights_adh(
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    *,
    standardize: bool = True,
    v_inits: Tuple[str, ...] = ("equal", "regression"),
    n_random_starts: int = 4,
    optimizer: str = "Nelder-Mead",
    max_iter: int = 500,
    ftol: float = 1e-10,
    penalization: float = 0.0,
    random_state: Optional[int] = 42,
) -> Dict[str, object]:
    """
    Canonical Abadie-Diamond-Hainmueller (2010) SCM weights via nested
    V-W optimization.

    Outer loop minimises pre-treatment MSPE of the outcome over a
    diagonal predictor-weight matrix ``V``; inner loop solves the
    V-weighted simplex QP for ``W``.

    Parameters
    ----------
    X1 : (K,) array
        Treated-unit predictor vector.
    X0 : (K, J) array
        Donor predictor matrix.
    Z1 : (T0,) array
        Treated-unit pre-treatment outcome vector (used by the outer
        MSPE loss only).
    Z0 : (T0, J) array
        Donor pre-treatment outcome matrix (outer loss).
    standardize : bool, default True
        Apply ``standardize_predictors`` before optimization.  Strongly
        recommended when predictors differ in magnitude (e.g. prices in
        cents vs log income in units).
    v_inits : tuple of str, default ('equal', 'regression')
        Deterministic starting points to try.  Each is optimized
        independently and the best (lowest outer loss) kept.
    n_random_starts : int, default 4
        Additional random Dirichlet starts.
    optimizer : str, default 'Nelder-Mead'
        scipy ``minimize`` method used for the outer loop. ``'L-BFGS-B'``
        is faster but more prone to getting stuck in flat regions;
        Nelder-Mead is derivative-free and more robust on this
        non-convex objective.
    max_iter : int, default 500
        Max iterations for the outer loop.
    ftol : float, default 1e-10
        Outer-loop tolerance.
    penalization : float, default 0.0
        Ridge penalty passed to the inner W solver.  0 recovers the
        classical ADH problem.
    random_state : int or None, default 42
        Seed for random Dirichlet starts.

    Returns
    -------
    dict with keys
        w : (J,) array   — optimal donor weights
        v : (K,) array   — optimal predictor weights (tr(V) = K scale)
        loss : float     — outer-loop MSPE at the optimum
        inner_loss : float — inner V-weighted predictor mismatch
        scale : (K,) array — predictor standardization scale (1s if
                             ``standardize=False``)
        n_starts : int   — total number of starts attempted
        converged : bool — True if the best start converged
    """
    X1 = np.asarray(X1, dtype=np.float64).ravel()
    X0 = np.asarray(X0, dtype=np.float64)
    Z1 = np.asarray(Z1, dtype=np.float64).ravel()
    Z0 = np.asarray(Z0, dtype=np.float64)
    K, J = X0.shape
    if Z0.shape[1] != J:
        raise ValueError(
            f"Z0 has {Z0.shape[1]} donor columns but X0 has {J}."
        )
    if Z0.shape[0] != Z1.shape[0]:
        raise ValueError(
            f"Z0 has {Z0.shape[0]} pre-periods but Z1 has {Z1.shape[0]}."
        )

    if standardize:
        X1_s, X0_s, scale = standardize_predictors(X1, X0)
    else:
        X1_s, X0_s = X1, X0
        scale = np.ones(K)

    def outer_loss(v_params: np.ndarray) -> float:
        V = _v_from_params(v_params, K)
        w = _inner_w_given_v(V, X1_s, X0_s, penalization=penalization)
        r = Z1 - Z0 @ w
        return float(r @ r)

    # Build starting-point list
    starts = []
    if "equal" in v_inits:
        starts.append(np.zeros(K))  # softmax(zeros) = uniform
    if "regression" in v_inits:
        v_reg = _regression_v_init(X1_s, X0_s, Z1, Z0)
        # Invert softmax: log(V) up to additive constant
        starts.append(np.log(np.maximum(v_reg, 1e-8)))

    if n_random_starts > 0:
        rng = np.random.default_rng(random_state)
        for _ in range(n_random_starts):
            dirichlet = rng.dirichlet(np.ones(K))
            starts.append(np.log(np.maximum(dirichlet * K, 1e-8)))

    best = None
    for v0 in starts:
        try:
            res = optimize.minimize(
                outer_loss, v0, method=optimizer,
                options={"maxiter": max_iter, "xatol": ftol, "fatol": ftol}
                if optimizer == "Nelder-Mead"
                else {"maxiter": max_iter, "ftol": ftol},
            )
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    if best is None:
        # Fallback: equal V, just solve inner once
        V = np.ones(K)
        w = _inner_w_given_v(V, X1_s, X0_s, penalization=penalization)
        r = Z1 - Z0 @ w
        return {
            "w": w,
            "v": V,
            "loss": float(r @ r),
            "inner_loss": float(np.sum(V * (X1_s - X0_s @ w) ** 2)),
            "scale": scale,
            "n_starts": 0,
            "converged": False,
        }

    V_opt = _v_from_params(best.x, K)
    w_opt = _inner_w_given_v(V_opt, X1_s, X0_s, penalization=penalization)
    r = Z1 - Z0 @ w_opt
    inner_r = X1_s - X0_s @ w_opt

    return {
        "w": w_opt,
        "v": V_opt,
        "loss": float(r @ r),
        "inner_loss": float(np.sum(V_opt * inner_r ** 2)),
        "scale": scale,
        "n_starts": len(starts),
        "converged": bool(best.success),
    }
