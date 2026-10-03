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
methods for comparative case studies. *JASA* 105(490), 493-505. [@abadie2010synthetic]

Kaul, A., Klößner, S., Pfeifer, G. & Schieler, M. (2022). Standard
synthetic control methods: the case of using all preintervention
outcomes together with covariates.  *JBES* 40(3), 1362-1376.
[@kaul2022standard]

Abadie, A. (2021). Using synthetic controls: feasibility, data
requirements, and methodological aspects.  *Journal of Economic
Literature* 59(2), 391-425. [@abadie2021synthetic]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import optimize

from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility

# ---------------------------------------------------------------------------
# Basic simplex solver (inner W problem, reused across module)
# ---------------------------------------------------------------------------


def _eq_bounded_lsq(C: np.ndarray, t: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """``min ||C w - t||^2  s.t.  sum(w) = 1, lb <= w <= ub`` (exact).

    The problem R ``pracma::lsqlincon`` hands to ``quadprog::solve.QP``
    inside ``DiSCos:::DiSCo_weights_reg`` (``lb = 0, ub = 1`` is the SCM
    simplex); ``lb`` may be ``-inf``. Shared by ``discos`` and
    :func:`solve_simplex_weights`. Primal
    active-set method: on each working set the equality-constrained least
    squares problem is solved in the null space of the adding-up
    constraint by ``lstsq`` on ``C`` itself (``C'C`` is never formed, so the
    conditioning is not squared); a blocking bound is added on a step that
    leaves the box, and the bound with the most negative multiplier is
    released at a stationary point.
    """
    n, J = C.shape
    # quadprog needs C'C positive definite (R errors otherwise). A rank-
    # deficient problem has a non-unique minimiser; a ridge of 1e-10 of the
    # mean squared column norm then selects one deterministically (the
    # aggregate-panel fallback, where J donors can exceed the number of
    # distinct quantile values).
    if np.linalg.matrix_rank(C) < J:
        delta = 1e-10 * float(np.mean(np.sum(C**2, axis=0)))
        C = np.vstack([C, np.sqrt(delta) * np.eye(J)])
        t = np.concatenate([t, np.zeros(J)])
    scale = max(1.0, float(np.abs(C.T @ t).max()), float(np.sum(C**2, axis=0).max()))
    tol = 1e-12 * scale
    feas = 1e-13
    w = np.full(J, 1.0 / J)
    at_lb = np.zeros(J, dtype=bool)
    at_ub = np.zeros(J, dtype=bool)
    last_released = -1
    for _ in range(50 * J + 50):
        free = ~(at_lb | at_ub)
        F = np.flatnonzero(free)
        fixed = np.flatnonzero(~free)
        fixed_val = np.where(at_lb, lb, ub)[fixed]
        nF = F.size
        r = t - C[:, fixed] @ fixed_val
        s_F = 1.0 - float(fixed_val.sum())
        base = np.full(nF, s_F / nF)
        if nF > 1:
            Q = np.linalg.qr(np.column_stack([np.ones(nF), np.eye(nF)[:, : nF - 1]]))[0]
            Z = Q[:, 1:]
            CF = C[:, F]
            v = np.linalg.lstsq(CF @ Z, r - CF @ base, rcond=None)[0]
            wF = base + Z @ v
        else:
            wF = base
        cur = w[F]
        lo_v = wF < lb - feas
        hi_v = wF > ub + feas
        if lo_v.any() or hi_v.any():
            step_dir = wF - cur
            ratios = np.full(nF, np.inf)
            ratios[lo_v] = (lb - cur[lo_v]) / step_dir[lo_v]
            ratios[hi_v] = (ub - cur[hi_v]) / step_dir[hi_v]
            k = int(np.argmin(ratios))
            if F[k] == last_released and ratios[k] <= 0.0:
                # releasing that bound gives no descent: it is optimal
                return np.clip(w, lb, ub)
            w[F] = cur + max(ratios[k], 0.0) * step_dir
            if lo_v[k]:
                w[F[k]] = lb
                at_lb[F[k]] = True
            else:
                w[F[k]] = ub
                at_ub[F[k]] = True
            last_released = -1
            continue
        w[F] = np.clip(wF, lb, ub)
        grad = C.T @ (C @ w - t)
        mu = -float(np.mean(grad[F]))
        red = grad + mu  # >= 0 at a lower bound, <= 0 at an upper bound
        bad = np.flatnonzero((at_lb & (red < -tol)) | (at_ub & (red > tol)))
        if bad.size == 0:
            return w
        k = int(bad[np.argmax(np.abs(red[bad]))])
        at_lb[k] = False
        at_ub[k] = False
        last_released = k
    raise ConvergenceFailure("bounded least squares (active set) did not converge")


def _simplex_lsq_optimum(
    y: np.ndarray, X: np.ndarray
) -> Optional[Tuple[np.ndarray, bool]]:
    """A minimiser of ``||y - X w||^2`` on the simplex for ``X`` with fewer
    rows than columns, and whether it is the only one.

    Active-set method that builds the support up from the single best donor
    (Lawson-Hanson with the adding-up constraint): add the donor with the
    most negative reduced gradient, solve the equality-constrained least
    squares on the support through its KKT system, step back to the
    boundary and drop donors when the solution leaves the simplex. The
    support never exceeds ``rows + 1`` donors, so each step is a small
    linear solve.

    Returns ``(w, unique)``. ``w`` satisfies the optimality conditions (no
    donor has a negative reduced gradient). ``unique`` is True when the
    donors whose reduced gradient is zero are affinely independent, so that
    the optimal face is a single point. ``None`` when the iteration does not
    finish or the optimality conditions cannot be verified.
    """
    n, J = X.shape
    scale = max(1.0, 2.0 * float(np.abs(X.T @ y).max()))
    tol = 1e-10 * scale
    support = [int(np.argmin(((X - y[:, None]) ** 2).sum(axis=0)))]
    w = np.zeros(J)
    w[support[0]] = 1.0

    def solve_on(cols: list[int]) -> Optional[np.ndarray]:
        k = len(cols)
        if k == 1:
            return np.array([1.0])
        Xp = X[:, cols]
        kkt = np.zeros((k + 1, k + 1))
        kkt[:k, :k] = 2.0 * Xp.T @ Xp
        kkt[:k, k] = 1.0
        kkt[k, :k] = 1.0
        try:
            sol = np.linalg.solve(kkt, np.append(2.0 * Xp.T @ y, 1.0))
        except np.linalg.LinAlgError:
            return None
        return np.asarray(sol[:k])

    for _ in range(20 * (n + 2)):
        grad = -2.0 * X.T @ (y - X @ w)
        reduced = grad - float(grad[support] @ w[support])
        reduced[support] = 0.0
        entering = int(np.argmin(reduced))
        if reduced[entering] >= -tol:
            break
        support.append(entering)
        while True:
            if len(support) > n + 1:
                return None
            target = solve_on(support)
            if target is None:
                return None
            if (target > 1e-14).all():
                w[:] = 0.0
                w[support] = target
                break
            current = w[support]
            leaving = target <= 1e-14
            step = float(
                np.min(current[leaving] / (current[leaving] - target[leaving]))
            )
            moved = current + step * (target - current)
            keep = moved > 1e-13
            w[:] = 0.0
            support = [c for c, k in zip(support, keep) if k]
            if not support:
                return None
            w[support] = moved[keep]
    else:
        return None

    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    grad = -2.0 * X.T @ (y - X @ w)
    reduced = grad - float(grad[support] @ w[support])
    if (reduced < -1e-9 * scale).any():
        return None
    tight = np.flatnonzero(reduced <= 1e-7 * scale)
    unique = tight.size <= n + 1
    if unique:
        face = np.vstack([X[:, tight], np.ones((1, tight.size))])
        unique = bool(np.linalg.matrix_rank(face) == tight.size)
    return w, unique


def _unique_simplex_lsq(y: np.ndarray, X: np.ndarray) -> Optional[np.ndarray]:
    """The minimiser of ``||y - X w||^2`` on the simplex if it is certified
    unique, else ``None`` (see :func:`_simplex_lsq_optimum`)."""
    found = _simplex_lsq_optimum(y, X)
    if found is None or not found[1]:
        return None
    return found[0]


def _least_norm_weights(
    A: np.ndarray, b: np.ndarray, w0: np.ndarray
) -> Optional[np.ndarray]:
    """``min ||w||^2`` subject to ``A w = b``, ``w >= 0``, from a feasible ``w0``.

    Primal active-set method. On the free set the least-norm solution of
    ``A_F w_F = b`` is what ``lstsq`` returns; a step that would make a
    weight negative stops at the boundary and fixes that weight at zero, and
    at a stationary point the fixed weight with the largest positive
    multiplier is released. ``||w||`` falls at every step, so the method
    ends at the unique minimiser. ``None`` if ``w0`` is not feasible or the
    iteration cap is reached.
    """
    J = A.shape[1]
    w = np.clip(np.asarray(w0, dtype=np.float64), 0.0, None).copy()
    free = w > 1e-14
    if not free.any():
        return None
    b_scale = max(1.0, float(np.abs(b).max()))
    a_scale = max(1.0, float(np.abs(A).max()))
    last_released = -1
    visited = set()
    for _ in range(20 * J + 50):
        F = np.flatnonzero(free)
        # A degenerate vertex (the feasible set is a single point, or
        # several weights sit exactly at zero) can send the method round a
        # loop of free sets without lowering the norm. The point it is
        # circling is feasible and no step from it is possible.
        pattern = free.tobytes()
        if pattern in visited:
            return w
        visited.add(pattern)
        AF = A[:, F]
        target = np.linalg.lstsq(AF, b, rcond=None)[0]
        if np.abs(AF @ target - b).max() > 1e-8 * b_scale:
            return None
        current = w[F]
        negative = target < -1e-14
        if negative.any():
            ratios = current[negative] / (current[negative] - target[negative])
            k = int(np.flatnonzero(negative)[int(np.argmin(ratios))])
            step = float(ratios.min())
            if F[k] == last_released and step <= 0.0:
                return w
            w[F] = current + step * (target - current)
            w[F[k]] = 0.0
            free[F[k]] = False
            last_released = -1
            continue
        w[:] = 0.0
        w[F] = np.clip(target, 0.0, None)
        multipliers = np.linalg.lstsq(AF.T, w[F], rcond=None)[0]
        cost = A.T @ multipliers
        cost[free] = -np.inf
        release = int(np.argmax(cost))
        if cost[release] <= 1e-11 * a_scale:
            return w
        free[release] = True
        last_released = release
    return None


def _exact_simplex_lsq(y: np.ndarray, X: np.ndarray) -> Optional[np.ndarray]:
    """Exact simplex least squares for ``X`` with fewer rows than columns.

    If the minimiser is unique, that point. If it is not (a treated unit
    inside the donors' hull, or a predictor the outer search has weighted to
    zero), every minimiser has the same fitted values ``X w``, and the one
    returned is the minimum-norm one, the most evenly spread weights that
    reproduce the optimal fit:

        min ||w||^2  s.t.  X w = X w*,  sum(w) = 1,  w >= 0.

    That is a property of the problem, not of the path an optimiser took,
    so it does not depend on the starting point, the solver or the
    platform. ``None`` when either stage cannot be completed; the caller
    then falls back to SLSQP.
    """
    found = _simplex_lsq_optimum(y, X)
    if found is None:
        return None
    w_opt, unique = found
    if unique:
        return w_opt
    J = X.shape[1]
    A = np.vstack([X, np.ones((1, J))])
    b = np.append(X @ w_opt, 1.0)
    w = _least_norm_weights(A, b, w_opt)
    if w is None:
        return None
    total = float(w.sum())
    if total <= 0:
        return None
    w = w / total
    resid_opt = y - X @ w_opt
    resid = y - X @ w
    tolerance = 1e-12 * max(1.0, float(y @ y))
    if float(resid @ resid) > float(resid_opt @ resid_opt) + tolerance:
        return None
    return np.asarray(w)


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
        return np.asarray(g)

    # Strictly convex problem (donor matrix of full column rank, or a
    # ridge penalty): the minimiser is unique, so solve it exactly with the
    # primal active-set method instead of stopping at SLSQP's tolerance
    # (which left weights ~1e-7 and gaps ~1e-6 away from the optimum).
    # Rank-deficient problems (fewer rows than donors, the usual Prop. 99
    # shape) can have a set of minimisers. A unique minimiser is returned
    # exactly; among several the minimum-norm one is. SLSQP is the fallback
    # if the exact method does not finish.
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    X_arr = np.asarray(X, dtype=np.float64)
    if penalization > 0:
        X_aug = np.vstack([X_arr, np.sqrt(penalization) * np.eye(J)])
        y_aug = np.concatenate([y_arr, np.zeros(J)])
    else:
        X_aug, y_aug = X_arr, y_arr
    if np.all(np.isfinite(X_aug)) and np.all(np.isfinite(y_aug)):
        if np.linalg.matrix_rank(X_aug) == J:
            try:
                return _eq_bounded_lsq(X_aug, y_aug, 0.0, 1.0)
            except RuntimeError as exc:  # pragma: no cover - fall back
                warnings.warn(
                    f"Exact simplex least squares failed ({exc}); using SLSQP.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        elif penalization == 0:
            # Rank deficient. Outside the donors' hull the projection has
            # one representation and it is returned exactly; otherwise the
            # minimum-norm minimiser is.
            w_exact = _exact_simplex_lsq(y_arr, X_arr)
            if w_exact is not None:
                return w_exact

    if w0 is None:
        w0 = np.ones(J) / J

    # The adding-up constraint is linear, so its Jacobian is exactly a row
    # of ones. Supplying it spares SLSQP a finite-difference pass per
    # iteration (J extra evaluations) without changing the solution.
    sum_jac = np.ones((1, J))
    result = optimize.minimize(
        objective,
        w0,
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * J,
        constraints={
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0,
            "jac": lambda w: sum_jac,
        },
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    # SLSQP enforces the bounds/equality only up to its own tolerance, so
    # ``result.x`` can carry sub-tolerance violations (tiny negative weights
    # or a mass slightly off 1.0).  The synthetic-control contract — and
    # every reference implementation (R ``Synth``, ``gsynth``) — is a clean
    # simplex point, so project the solver output back onto it: clip the
    # negative noise to zero and renormalise.  This only moves weights by
    # the solver's sub-tolerance noise and keeps the non-negativity
    # invariant that downstream code (and reference parity) relies on.
    w = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
    total = float(w.sum())
    if total > 0.0:
        w = w / total
    return w


def polish_simplex_weights(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Refine an approximate simplex least-squares solution to the exact one.

    With fewer rows than donors the problem ``min ||y - X w||^2`` on the
    simplex is solved by SLSQP, which stops about 1e-7 from the optimum.
    This re-solves it exactly on the donors that carry weight and accepts
    the result only if it satisfies the optimality conditions of the full
    problem (no excluded donor would lower the loss); otherwise ``w`` is
    returned unchanged.
    """
    w = np.asarray(w, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    support = np.flatnonzero(w > 1e-8)
    if support.size < 2 or support.size > X.shape[0]:
        return w
    Xs = X[:, support]
    if np.linalg.matrix_rank(Xs) < support.size:
        return w
    try:
        ws = _eq_bounded_lsq(Xs, y, 0.0, 1.0)
    except RuntimeError:  # pragma: no cover - keep the approximate solution
        return w
    out = np.zeros_like(w)
    out[support] = ws
    grad = -2.0 * X.T @ (y - X @ out)
    inside = support[ws > 1e-12]
    if inside.size == 0:  # pragma: no cover - defensive
        return w
    lam = float(np.mean(grad[inside]))
    scale = max(1.0, float(np.abs(grad).max()))
    if np.any(grad - lam < -1e-8 * scale):
        return w
    return out


# ---------------------------------------------------------------------------
# Placebo (permutation) inference
# ---------------------------------------------------------------------------


def placebo_rank_pvalue(treated_stat: float, placebo_stats: Any) -> float:
    """
    Permutation p-value from in-space placebos, following the ranking
    convention of Abadie, Diamond & Hainmueller (2010)
    [@abadie2010synthetic].

    The treated unit is ranked *together with* its placebos, so with
    ``J`` usable placebos

        p = (1 + #{j : placebo_j >= treated}) / (J + 1),

    i.e. the treated unit's rank divided by the number of units in the
    permutation distribution. Ties count against the treated unit. The
    smallest attainable value is ``1/(J+1)``; a treated unit ranked
    third of 39 gets ``3/39``.

    Parameters
    ----------
    treated_stat : float
        Test statistic of the treated unit (e.g. post/pre RMSPE ratio or
        ``|ATT|``). ``+inf`` (perfect pre-fit) is allowed.
    placebo_stats : array-like
        The same statistic for each placebo unit, treated unit excluded.
        NaN placebos are dropped from both numerator and denominator.

    Returns
    -------
    float
        The p-value, or NaN when no usable placebo is available.
    """
    stats_arr = np.asarray(placebo_stats, dtype=np.float64).ravel()
    stats_arr = stats_arr[~np.isnan(stats_arr)]
    if stats_arr.size == 0 or np.isnan(treated_stat):
        return float("nan")
    n_at_least = int(np.sum(stats_arr >= treated_stat))
    return float((1 + n_at_least) / (stats_arr.size + 1))


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
        raise MethodIncompatibility(
            f"X1 has {X1.shape[0]} predictors but X0 has {X0.shape[0]}.",
            recovery_hint=(
                "Pass a treated predictor vector and donor matrix with the "
                "same predictor rows."
            ),
            diagnostics={
                "x1_predictors": int(X1.shape[0]),
                "x0_predictors": int(X0.shape[0]),
            },
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
    return np.asarray(K * ev / s)


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
        v = beta**2
        if v.sum() <= 1e-12 or not np.all(np.isfinite(v)):
            return np.ones(K)
        return np.asarray(K * v / v.sum())
    except np.linalg.LinAlgError:
        return np.ones(K)


def _swept_columns(A: np.ndarray, tol: float = 1e-10) -> list[int]:
    """Columns a pivoted sweep of the cross-product matrix ``A`` keeps.

    At each step the unswept column with the largest current diagonal
    element is swept; the sweep stops when that element is no larger than
    ``tol`` times its original value. The columns never swept are linearly
    dependent on the ones that were (Mata's ``invsym``).
    """
    A = np.array(A, dtype=np.float64)
    original = np.diag(A).copy()
    rest = list(range(A.shape[0]))
    kept: list[int] = []
    while rest:
        k = max(rest, key=lambda i: A[i, i])
        if A[k, k] <= tol * original[k]:
            break
        rest.remove(k)
        pivot = A[k, k]
        column, row = A[:, k].copy(), A[k, :].copy()
        A -= np.outer(column, row) / pivot
        A[k, :] = row / pivot
        A[:, k] = -column / pivot
        A[k, k] = 1.0 / pivot
        kept.append(k)
    return sorted(kept)


def regression_based_v(
    X1: np.ndarray,
    X0: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Regression-based predictor weights, the default of Stata ``synth``.

    Every pre-treatment outcome period is regressed, across units (treated
    and donors together, with a constant), on the predictors scaled to unit
    standard deviation. With ``B`` the ``K x T0`` matrix of slopes, the
    weight of predictor ``k`` is the ``k``-th diagonal element of ``B B'``:
    how much the predictor moves the outcome path. No search is involved,
    so the result is deterministic and takes a least-squares fit.

    Returns ``(V, scale)`` with ``V`` normalised to ``tr(V) = K`` (Stata
    reports ``V / K``, which sums to one) and ``scale`` the standard
    deviations the predictors were divided by. Checked against
    ``e(V_matrix)`` of Stata ``synth`` on the Proposition 99 data: all
    eight printed digits.

    With no more units than coefficients the regression is not identified.
    Stata's ``synth`` then inverts the cross-product matrix by sweeping the
    column with the largest remaining diagonal element first and gives a
    zero coefficient to the columns that cannot be swept; a predictor left
    out that way gets weight zero. The same rule is applied here
    (``_swept_columns``) and reproduces ``e(V_matrix)`` in that case too.
    """
    K, J = X0.shape
    X_all = np.column_stack([X1[:, None], X0])  # (K, J + 1)
    scale = X_all.std(axis=1, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)
    design = np.column_stack([np.ones(J + 1), (X_all / scale[:, None]).T])
    outcomes = np.column_stack([Z1[:, None], Z0]).T  # (J + 1, T0)
    if J + 1 > K + 1:
        coef = np.linalg.lstsq(design, outcomes, rcond=None)[0][1:]  # (K, T0)
    else:
        # constant last, as synth orders the columns
        ordered = np.column_stack([design[:, 1:], design[:, 0]])
        keep = _swept_columns(ordered.T @ ordered)
        full = np.zeros((K + 1, outcomes.shape[1]))
        full[keep] = np.linalg.lstsq(ordered[:, keep], outcomes, rcond=None)[0]
        coef = full[:K]
    v = np.einsum("kt,kt->k", coef, coef)
    if not np.all(np.isfinite(v)) or v.sum() <= 0:
        raise DataInsufficient(
            "synth(v_method='regression'): no predictor explains the "
            "pre-treatment outcomes, so regression-based V is undefined.",
            recovery_hint="Use v_method='nested' or 'equal'.",
            diagnostics={"n_predictors": int(K)},
        )
    return np.asarray(K * v / v.sum()), scale


def _hull_feasibility(
    X1_s: np.ndarray,
    X0_s: np.ndarray,
) -> Tuple[Optional[bool], Optional[np.ndarray]]:
    """
    Exact check whether the treated predictors lie in the donors' convex hull.

    Solves the linear feasibility problem ``{w >= 0, 1'w = 1, X0 w = X1}``.
    Inside the hull every V with full support attains a zero predictor
    discrepancy and the inner problem has a whole polytope of solutions, so
    the inner problem needs a selection rule (the minimum-norm weights).

    Returns ``(True, feasible_w)``, ``(False, None)``, or ``(None, None)``
    with a ``RuntimeWarning`` when the LP does not finish.
    """
    K, J = X0_s.shape
    A = np.vstack([X0_s, np.ones((1, J))])
    b = np.append(X1_s, 1.0)
    lp = optimize.linprog(
        np.zeros(J), A_eq=A, b_eq=b, bounds=[(0.0, None)] * J, method="highs"
    )
    if lp.status == 2:  # infeasible: outside the hull
        return False, None
    if lp.status != 0:
        warnings.warn(
            f"Convex-hull check for the nested SCM fit did not finish "
            f"(linprog status {lp.status}: {lp.message}).",
            RuntimeWarning,
            stacklevel=3,
        )
        return None, None
    return True, np.clip(np.asarray(lp.x, dtype=np.float64), 0.0, None)


def _exact_balance_weights(
    X1_s: np.ndarray,
    X0_s: np.ndarray,
    Z1: np.ndarray,
    Z0: np.ndarray,
    w_feasible: np.ndarray,
) -> Dict[str, Any]:
    """
    Exact-covariate-balance weights with the best pre-treatment outcome fit.

    For a treated unit inside the donors' predictor hull, solves

        min_w ||Z1 - Z0 w||^2  s.t.  w >= 0, 1'w = 1, X0 w = X1,

    a convex QP, with ``trust-constr`` (SLSQP mis-solves these degenerate
    equality-constrained problems, reporting success at points far from the
    optimum) and certifies it by the Frank-Wolfe gap
    ``g'w - min_{u in P} g'u``, an upper bound on ``loss(w) - loss*``.

    This is *not* the ADH nested optimum made unique: the outer V search may
    set some V entries to (near) zero, which drops those predictors from the
    balance constraint and can buy a better outcome fit.
    """
    K, J = X0_s.shape
    A = np.vstack([X0_s, np.ones((1, J))])
    b = np.append(X1_s, 1.0)
    bounds = [(0.0, None)] * J
    H = 2.0 * Z0.T @ Z0

    def objective(w: np.ndarray) -> float:
        r = Z1 - Z0 @ w
        return float(r @ r)

    def gradient(w: np.ndarray) -> np.ndarray:
        return np.asarray(-2.0 * Z0.T @ (Z1 - Z0 @ w))

    with warnings.catch_warnings():
        # trust-constr reports quasi-Newton update skips on this quadratic
        # objective; they are informational, the certificate below is not.
        warnings.filterwarnings("ignore", message="delta_grad == 0.0")
        res = optimize.minimize(
            objective,
            w_feasible,
            jac=gradient,
            hess=lambda w: H,
            method="trust-constr",
            bounds=optimize.Bounds(0.0, 1.0),
            constraints=optimize.LinearConstraint(A, b, b),
            options={"maxiter": 5000, "gtol": 1e-12, "xtol": 1e-14},
        )
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, None)
    w = w / w.sum()
    loss = objective(w)

    # Polish: trust-constr stops at a small but finite gap (weights off by
    # ~1e-4 when the optimum has zero loss). Re-solve the equality-constrained
    # least squares exactly on the active support via its KKT system; keep
    # the result only if it stays feasible and does not raise the loss.
    support = np.flatnonzero(w > 1e-9)
    if support.size:
        Q = H[np.ix_(support, support)]
        A_s = A[:, support]
        m = A_s.shape[0]
        kkt = np.block([[Q, A_s.T], [A_s, np.zeros((m, m))]])
        rhs = np.concatenate([2.0 * Z0[:, support].T @ Z1, b])
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        w_s = sol[: support.size]
        if np.all(w_s >= -1e-12):
            w_pol = np.zeros(J)
            w_pol[support] = np.clip(w_s, 0.0, None)
            loss_pol = objective(w_pol)
            viol_pol = float(np.abs(A @ w_pol - b).max())
            if viol_pol <= 1e-10 and loss_pol <= loss + 1e-12 * max(loss, 1.0):
                w, loss = w_pol, loss_pol

    g = gradient(w)
    fw = optimize.linprog(g, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    gap = float(g @ w - fw.fun) if fw.status == 0 else float("inf")
    return {
        "w": w,
        "loss": loss,
        "fw_gap": gap,
        "max_constraint_violation": float(np.abs(A @ w - b).max()),
    }


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
    perfect_fit: str = "legacy",
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
    perfect_fit : {'legacy', 'exact_balance'}, default 'legacy'
        Rule when ``X1`` lies in the convex hull of the columns of ``X0``.
        ``'legacy'`` runs the V search regardless (the ADH estimator; for
        each V the inner weights are the minimum-norm minimiser, see
        ``_exact_simplex_lsq``). ``'exact_balance'`` skips the V
        search and returns the weights that balance every predictor exactly
        with the smallest outer loss (see ``_exact_balance_weights``) — a
        different estimator, not a tie-break. Only applies when
        ``penalization == 0``.

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
                           (``'exact_balance'``: the QP certificate passed)
        in_predictor_hull : bool or None — X1 lies in the donors' hull
                           (None if the LP check did not finish)
        v_identified : False under ``'exact_balance'``, else None
    """
    if perfect_fit not in ("legacy", "exact_balance"):
        raise MethodIncompatibility(
            "perfect_fit must be 'legacy' or 'exact_balance'.",
            diagnostics={"perfect_fit": repr(perfect_fit)},
        )
    X1 = np.asarray(X1, dtype=np.float64).ravel()
    X0 = np.asarray(X0, dtype=np.float64)
    Z1 = np.asarray(Z1, dtype=np.float64).ravel()
    Z0 = np.asarray(Z0, dtype=np.float64)
    K, J = X0.shape
    if Z0.shape[1] != J:
        raise MethodIncompatibility(
            f"Z0 has {Z0.shape[1]} donor columns but X0 has {J}.",
            recovery_hint="Use the same donor ordering and donor count in X0 and Z0.",
            diagnostics={"z0_donors": int(Z0.shape[1]), "x0_donors": int(J)},
        )
    if Z0.shape[0] != Z1.shape[0]:
        raise MethodIncompatibility(
            f"Z0 has {Z0.shape[0]} pre-periods but Z1 has {Z1.shape[0]}.",
            recovery_hint=(
                "Align treated and donor pre-treatment outcome matrices to "
                "the same periods."
            ),
            diagnostics={
                "z0_pre_periods": int(Z0.shape[0]),
                "z1_pre_periods": int(Z1.shape[0]),
            },
        )

    if standardize:
        X1_s, X0_s, scale = standardize_predictors(X1, X0)
    else:
        X1_s, X0_s = X1, X0
        scale = np.ones(K)

    # Hull membership is invariant to the positive row scaling above.
    in_hull, w_feasible = _hull_feasibility(X1_s, X0_s)

    if perfect_fit == "exact_balance" and penalization == 0.0 and in_hull:
        pf = _exact_balance_weights(X1_s, X0_s, Z1, Z0, w_feasible)
        # 1e-8 relative (absolute below loss 1): the polished QP typically
        # certifies at 1e-10 or better; a looser bound would pass unpolished
        # zero-loss solutions whose weights are off by ~1e-4.
        certified = bool(pf["fw_gap"] <= 1e-8 * max(pf["loss"], 1.0))
        if not certified:
            warnings.warn(
                "Exact-balance SCM weights were not certified optimal "
                f"(Frank-Wolfe gap {pf['fw_gap']:.3g} at loss "
                f"{pf['loss']:.6g}); converged=False.",
                RuntimeWarning,
                stacklevel=3,
            )
        V_id = np.ones(K)
        w_pf = pf["w"]
        inner_r_pf = X1_s - X0_s @ w_pf
        inner_loss_pf = float(np.sum(V_id * inner_r_pf**2))
        return {
            "w": w_pf,
            "v": V_id,
            "loss": pf["loss"],
            "inner_loss": inner_loss_pf,
            "scale": scale,
            "n_starts": 0,
            "converged": certified,
            "best_start": "exact_balance",
            "start_diagnostics": [
                {
                    "start": "exact_balance",
                    "success": certified,
                    "loss": pf["loss"],
                    "inner_loss": inner_loss_pf,
                    "weights": w_pf,
                    "v": V_id,
                }
            ],
            "in_predictor_hull": True,
            "v_identified": False,
            "exact_balance_gap": pf["fw_gap"],
        }

    # W(V) must be a function of V alone. When the inner minimiser is not
    # unique (K predictors < J donors and the treated unit inside the hull,
    # or a V entry at zero) ``solve_simplex_weights`` returns the
    # minimum-norm one. A history-dependent choice, such as warm-starting
    # from the previous evaluation's W, makes the outer loss stop being a
    # function of V, and Nelder-Mead then drifts into worse basins (observed
    # on the Basque special-predictor specification).
    def outer_loss(v_params: np.ndarray) -> float:
        V = _v_from_params(v_params, K)
        w = _inner_w_given_v(V, X1_s, X0_s, penalization=penalization)
        r = Z1 - Z0 @ w
        return float(r @ r)

    # Build starting-point list
    starts: list[tuple[str, np.ndarray]] = []
    if "equal" in v_inits:
        starts.append(("equal", np.zeros(K)))  # softmax(zeros) = uniform
    if "regression" in v_inits:
        v_reg = _regression_v_init(X1_s, X0_s, Z1, Z0)
        # Invert softmax: log(V) up to additive constant
        starts.append(("regression", np.log(np.maximum(v_reg, 1e-8))))

    if n_random_starts > 0:
        rng = np.random.default_rng(random_state)
        for idx in range(n_random_starts):
            dirichlet = rng.dirichlet(np.ones(K))
            starts.append(
                (f"dirichlet_{idx + 1}", np.log(np.maximum(dirichlet * K, 1e-8)))
            )

    best = None
    best_start = None
    start_diagnostics: list[dict[str, Any]] = []
    for start_name, v0 in starts:
        try:
            res = optimize.minimize(
                outer_loss,
                v0,
                method=optimizer,
                options=(
                    {"maxiter": max_iter, "xatol": ftol, "fatol": ftol}
                    if optimizer == "Nelder-Mead"
                    else {"maxiter": max_iter, "ftol": ftol}
                ),
            )
            V_start = _v_from_params(res.x, K)
            w_start = _inner_w_given_v(V_start, X1_s, X0_s, penalization=penalization)
            inner_r_start = X1_s - X0_s @ w_start
            start_diagnostics.append(
                {
                    "start": start_name,
                    "success": bool(res.success),
                    "loss": float(res.fun),
                    "inner_loss": float(np.sum(V_start * inner_r_start**2)),
                    "weights": w_start,
                    "v": V_start,
                }
            )
            if best is None or res.fun < best.fun:
                best = res
                best_start = start_name
        except Exception as exc:
            start_diagnostics.append(
                {
                    "start": start_name,
                    "success": False,
                    "loss": np.inf,
                    "inner_loss": np.inf,
                    "weights": np.full(J, np.nan),
                    "v": np.full(K, np.nan),
                    "error": str(exc),
                }
            )
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
            "best_start": None,
            "start_diagnostics": start_diagnostics,
            "in_predictor_hull": in_hull,
            "v_identified": None,
        }

    V_opt = _v_from_params(best.x, K)
    w_opt = _inner_w_given_v(V_opt, X1_s, X0_s, penalization=penalization)
    r = Z1 - Z0 @ w_opt
    inner_r = X1_s - X0_s @ w_opt

    return {
        "w": w_opt,
        "v": V_opt,
        "loss": float(r @ r),
        "inner_loss": float(np.sum(V_opt * inner_r**2)),
        "scale": scale,
        "n_starts": len(starts),
        "converged": bool(best.success),
        "best_start": best_start,
        "start_diagnostics": start_diagnostics,
        "in_predictor_hull": in_hull,
        "v_identified": None,
    }
