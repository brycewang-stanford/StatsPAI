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

from ..exceptions import MethodIncompatibility

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
        return np.asarray(g)

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


def _hull_feasibility(
    X1_s: np.ndarray,
    X0_s: np.ndarray,
) -> Tuple[Optional[bool], Optional[np.ndarray]]:
    """
    Exact check whether the treated predictors lie in the donors' convex hull.

    Solves the linear feasibility problem ``{w >= 0, 1'w = 1, X0 w = X1}``.
    Inside the hull every V with full support attains a zero predictor
    discrepancy and the inner problem has a whole polytope of solutions, so
    the nested V-W weights depend on the optimiser's path.

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
        ``'legacy'`` runs the V search regardless (the ADH estimator; its
        weights are then path-dependent). ``'exact_balance'`` skips the V
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

    # The inner solve deliberately starts from the uniform simplex point on
    # every evaluation. Warm-starting from the previous evaluation's W is
    # ~2x faster, but when W(V) is non-unique (K predictors < J donors) it
    # selects a history-dependent point on the optimal face, so the outer
    # loss stops being a function of V and Nelder-Mead drifts into worse
    # basins (observed on the Basque special-predictor specification).
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
