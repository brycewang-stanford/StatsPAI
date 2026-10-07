"""
Optimized regression discontinuity: minimax linear weights under a
curvature bound.

A sharp regression discontinuity estimate is a weighted sum of outcomes,
``tau_hat = sum_i gamma_i Y_i``. Local linear regression is one way to
pick the weights. This module picks them directly, to minimise the
worst-case mean squared error (or confidence interval length) over all
conditional mean functions whose second derivative is bounded by ``M``
on each side of the cutoff.

Worst-case bias
---------------
The weights must sum to one on the treated side and to minus one on the
control side, and be orthogonal to the running variable on each side;
otherwise a level or a slope makes the bias unbounded. Given those four
constraints, a Taylor expansion with integral remainder gives the bias
under mean functions ``mu_w`` as an integral of ``mu_w''`` against

    G_w(t) = sum over side w of gamma_i (|x_i - c| - t)_+ ,

so the largest bias over ``|mu_w''| <= M`` is
``M * (int |G_1| + int |G_0|)``. ``G_w`` is piecewise linear between the
observed values of the running variable, and the integral is computed
exactly. This is the bias reported and used for the interval, for
whatever weights come out of the optimisation.

Optimisation
------------
By minimax duality the optimal weights are proportional to the values
of a least favourable function: among pairs ``(f_1, f_0)`` with a unit
jump at the cutoff and ``|f_w''| <= kappa``, the one with the smallest
``sum_i f(x_i)^2 / sigma^2``. Writing ``f`` through its level, slope and
a piecewise constant second derivative makes this a least-squares
problem with box constraints. The ratio ``kappa`` of curvature to jump
traces the bias-variance frontier and is chosen by a one-dimensional
search on the exact criterion.

The running variable may be discrete: nothing here needs a density at
the cutoff.

References
----------
[@imbens2019optimized], [@armstrong2018optimal]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._rdhonest import cv_bias, honest_bandwidth, m_rule_of_thumb, sigma_nn

# --------------------------------------------------------------------
# Exact worst-case bias of arbitrary weights
# --------------------------------------------------------------------


def _side_integral(u: np.ndarray, g: np.ndarray) -> float:
    """``int_0^inf |G(t)| dt`` with ``G(t) = sum_i g_i (u_i - t)_+``.

    ``u`` holds distances from the cutoff (non-negative). ``G`` is
    linear between consecutive distinct values of ``u`` (and between 0
    and the smallest), so the integral of its absolute value is exact.
    """
    if u.size == 0:
        return 0.0
    order = np.argsort(u, kind="stable")
    u, g = u[order], g[order]
    knots = np.unique(np.concatenate([[0.0], u]))
    # Suffix sums over points strictly beyond each knot.
    cs0 = np.concatenate([[0.0], np.cumsum(g)])
    cs1 = np.concatenate([[0.0], np.cumsum(g * u)])
    first_beyond = np.searchsorted(u, knots, side="right")
    s0 = cs0[-1] - cs0[first_beyond]
    s1 = cs1[-1] - cs1[first_beyond]
    G = s1 - knots * s0
    a, b = G[:-1], G[1:]
    width = np.diff(knots)
    same = a * b >= 0
    area = np.where(
        same,
        0.5 * np.abs(a + b),
        0.5 * (a * a + b * b) / np.maximum(np.abs(a) + np.abs(b), 1e-300),
    )
    return float(np.sum(area * width))


def worst_case_bias(xc: np.ndarray, gamma: np.ndarray, M: float) -> float:
    """Largest bias of ``sum gamma_i Y_i`` over ``|mu_w''| <= M``.

    ``xc`` is the running variable minus the cutoff; units with
    ``xc >= 0`` are treated. The weights must satisfy the four moment
    constraints (sum to +1 and -1, orthogonal to ``xc`` on each side);
    the function returns ``inf`` when they do not, because the bias is
    then unbounded.
    """
    xc = np.asarray(xc, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    right = xc >= 0
    scale = max(float(np.abs(xc).max()), 1e-300)
    checks = (
        abs(gamma[right].sum() - 1.0),
        abs(gamma[~right].sum() + 1.0),
        abs(gamma[right] @ xc[right]) / scale,
        abs(gamma[~right] @ xc[~right]) / scale,
    )
    if max(checks) > 1e-8:
        return float("inf")
    total = _side_integral(xc[right], gamma[right])
    total += _side_integral(-xc[~right], -gamma[~right])
    return float(M * total)


def _nn_deviation(x: np.ndarray, y: np.ndarray, J: int = 3) -> np.ndarray:
    """Signed deviations whose squares are the nearest-neighbour variances.

    ``x`` must be sorted. Each outcome is compared with the mean of its
    ``J`` nearest neighbours (all ties at the ``J``-th distance), scaled
    by ``sqrt(J / (J + 1))``, as in :func:`sigma_nn`. Keeping the sign
    lets the variance of a linear combination of two outcomes be formed
    from their deviations.
    """
    n = len(x)
    out = np.zeros(n)
    if n < 2:
        return out
    J = min(J, n - 1)
    for k in range(n):
        lo = max(k - J, 0)
        cand = np.concatenate([x[lo:k], x[k + 1 : min(k + J + 1, n)]])
        d = np.sort(np.abs(cand - x[k]))[J - 1]
        ind = np.abs(x - x[k]) <= d
        ind[k] = False
        jk = float(ind.sum())
        if jk > 0:
            out[k] = np.sqrt(jk / (jk + 1.0)) * (y[k] - y[ind].mean())
    return out


# --------------------------------------------------------------------
# Least favourable function on one side
# --------------------------------------------------------------------


def _side_design(u: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    """Design of ``f(u) = a + b u + sum_k g_k psi_k(u)`` at support ``u``.

    ``psi_k`` is the double integral of the indicator of cell ``k``, so
    that ``g_k`` is the second derivative of ``f`` on that cell. Cells
    subdivide the gaps between consecutive support points.
    """
    knots = np.unique(np.concatenate([[0.0], u]))
    if knots.size < 2:
        knots = np.array([0.0, 1.0])
    per_gap = max(1, int(np.ceil(n_cells / (knots.size - 1))))
    edges = np.unique(
        np.concatenate(
            [
                np.linspace(knots[k], knots[k + 1], per_gap + 1)
                for k in range(knots.size - 1)
            ]
        )
    )
    lo, hi = edges[:-1], edges[1:]
    d_lo = np.clip(u[:, None] - lo[None, :], 0.0, None)
    d_hi = np.clip(u[:, None] - hi[None, :], 0.0, None)
    psi = 0.5 * (d_lo**2 - d_hi**2)
    return np.column_stack([np.ones_like(u), u, psi]), edges


def _solve_least_favourable(
    designs: Tuple[np.ndarray, np.ndarray],
    weights: Tuple[np.ndarray, np.ndarray],
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Least favourable function with a unit jump and curvature ``kappa``.

    Minimises ``sum_j w_j f(x_j)^2`` over pairs ``(f_1, f_0)`` with
    ``f_1(c) - f_0(c) = 1`` and ``|f_w''| <= kappa``. A function with
    jump ``v`` and curvature bound ``M`` is ``v`` times such a pair with
    ``kappa = M / v``, so varying ``kappa`` traces the bias-variance
    frontier. With the jump substituted out this is a least-squares
    problem with box constraints, solved by an active-set method.

    Returns the values of ``f_1`` and ``f_0`` at the support points.
    """
    D1, D0 = designs
    w1, w0 = weights
    j1, j0 = D1.shape[0], D0.shape[0]
    k1, k0 = D1.shape[1] - 2, D0.shape[1] - 2
    # Unknowns: f_1(c), slope and scaled curvatures on the treated side,
    # then slope and scaled curvatures on the control side, whose level
    # is f_1(c) - 1.
    A = np.zeros((j1 + j0, 3 + k1 + k0))
    A[:j1, 0] = 1.0
    A[:j1, 1] = D1[:, 1]
    A[:j1, 2 : 2 + k1] = D1[:, 2:] * kappa
    A[j1:, 0] = 1.0
    A[j1:, 2 + k1] = D0[:, 1]
    A[j1:, 3 + k1 :] = D0[:, 2:] * kappa
    target = np.concatenate([np.zeros(j1), np.ones(j0)])
    root = np.sqrt(np.concatenate([w1, w0]))
    lower = np.full(A.shape[1], -1.0)
    upper = np.full(A.shape[1], 1.0)
    for col in (0, 1, 2 + k1):
        lower[col], upper[col] = -np.inf, np.inf
    res = optimize.lsq_linear(
        A * root[:, None],
        target * root,
        bounds=(lower, upper),
        method="bvls",
        tol=1e-13,
        max_iter=20 * A.shape[1],
    )
    fitted = A @ res.x - target
    return fitted[:j1], fitted[j1:], bool(res.status > 0)


def _project(xc: np.ndarray, n: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Impose the four moment constraints exactly on bin-level weights.

    The optimiser satisfies them to its tolerance; the smallest
    (count-weighted) correction that is affine in the running variable
    on the units already carrying weight makes them hold to rounding.
    """
    out = gamma.copy()
    for mask, target in ((xc >= 0, 1.0), (xc < 0, -1.0)):
        act = mask & (gamma != 0)
        if act.sum() < 2:
            raise DataInsufficient(
                "fewer than two distinct values of the running variable "
                "carry weight on one side of the cutoff"
            )
        X = np.column_stack([np.ones(act.sum()), xc[act]])
        nn = n[act]
        # Residual of the constraints: sum n*gamma = target, sum n*gamma*x = 0.
        r = np.array([target, 0.0]) - X.T @ (nn * gamma[act])
        coef = np.linalg.solve(X.T @ (X * nn[:, None]), r)
        out[act] = gamma[act] + X @ coef
    return out


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def rd_optimized(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0.0,
    M: Optional[float] = None,
    *,
    fuzzy: Optional[str] = None,
    M_fuzzy: Optional[float] = None,
    criterion: str = "mse",
    h: Optional[float] = None,
    sigma2: Optional[float] = None,
    num_bins: int = 120,
    n_cells: int = 120,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Sharp regression discontinuity with minimax-optimal linear weights.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x : str
        Running variable; units with ``x >= c`` are treated. It may be
        continuous or discrete.
    c : float, default 0.0
        Cutoff.
    M : float, optional
        Bound on the absolute second derivative of the conditional mean
        on each side of the cutoff. If omitted, the Armstrong-Kolesar
        rule of thumb is used (a global quartic fit on each side), as in
        :func:`rd_honest`. The interval is only as credible as this
        bound: report it and vary it.
    fuzzy : str, optional
        Column with the treatment actually received, for a fuzzy design
        in which crossing the cutoff changes the probability of
        treatment. The estimate is then the ratio of the jump in the
        outcome to the jump in this column.
    M_fuzzy : float, optional
        Bound on the second derivative of the conditional mean of the
        ``fuzzy`` column. Defaults to the same rule of thumb as ``M``.
    criterion : {"mse", "flci"}, default "mse"
        ``"mse"`` minimises the worst-case mean squared error;
        ``"flci"`` the length of the confidence interval.
    h : float, optional
        Use only observations with ``|x - c| <= h``. By default the
        window is 2.5 times the optimal local linear bandwidth, which
        is wide enough for the optimal weights to reach zero inside it.
    sigma2 : float, optional
        Outcome variance used to *choose* the weights (the trade-off
        depends on ``M**2 / sigma2``). By default it is the average of
        nearest-neighbour variance estimates on each side of the
        cutoff. The standard error always uses the unit-level
        estimates.
    num_bins : int, default 120
        If a side has more distinct values of ``x`` than this, they are
        grouped into that many equal-width bins and units in a bin share
        a weight. The reported bias and standard error are computed from
        the unit-level data in either case.
    n_cells : int, default 120
        Approximate number of pieces, per side, of the piecewise
        constant second derivative of the least favourable function.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``model_info`` holds the bound ``M``, the worst-case bias
        (``max_bias``), the unit-level weights (``weights``, aligned
        with ``model_info['index']``), the largest distance from the
        cutoff that carries weight (``effective_bandwidth``) and the
        same quantities for local linear regression with a triangular
        kernel at its optimal bandwidth (``local_linear``).

    Notes
    -----
    The interval is ``estimate +/- cv * se`` where ``cv`` is the
    ``1 - alpha`` quantile of ``|N(max_bias / se, 1)|``, as in
    :func:`rd_honest`. It has the stated coverage for every conditional
    mean function with curvature at most ``M``, whether the running
    variable is continuous or not. The standard error uses
    nearest-neighbour variance estimates.

    With ``fuzzy=`` the weights are still those chosen for the outcome.
    The interval collects every ratio ``t`` for which the weighted sum of
    ``Y - t * D`` is within its own bias-aware critical value of zero,
    where the worst-case bias uses the bound ``M + |t| * M_fuzzy``. This
    is an Anderson-Rubin construction: it stays valid when the jump in
    treatment is small, in which case it can be unbounded (reported as
    infinite endpoints). ``se`` is the delta-method standard error of the
    ratio and is given for reference only.

    Compared with :func:`rd_honest`, which uses local linear weights
    under the same bound, the gain in interval length is a few percent
    for a continuous running variable and can be larger for a discrete
    one. The two functions share ``M`` and should tell the same story.

    References
    ----------
    [@imbens2019optimized], [@armstrong2018optimal], [@noack2024biasaware]

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 800)
    >>> yv = 0.5 * (x >= 0) + np.sin(x) + rng.normal(scale=0.3, size=800)
    >>> r = sp.rd_optimized(pd.DataFrame({"y": yv, "x": x}), "y", "x", M=1.0)
    >>> bool(r.ci[0] < 0.5 < r.ci[1])
    True
    """
    if criterion not in ("mse", "flci"):
        raise MethodIncompatibility("criterion must be 'mse' or 'flci'")
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must be in (0, 1)")
    cols = [y, x] + ([fuzzy] if fuzzy is not None else [])
    for col in cols:
        if col not in data.columns:
            raise MethodIncompatibility(f"column {col!r} not found in data")
    df = data[cols].dropna()
    order = np.argsort(df[x].to_numpy(dtype=float), kind="stable")
    index = df.index.to_numpy()[order]
    xv = df[x].to_numpy(dtype=float)[order]
    yv = df[y].to_numpy(dtype=float)[order]
    dv = df[fuzzy].to_numpy(dtype=float)[order] if fuzzy is not None else None
    xc_all = xv - c
    if (xc_all >= 0).sum() < 5 or (xc_all < 0).sum() < 5:
        raise DataInsufficient("each side of the cutoff needs at least five units")
    if (
        len(np.unique(xc_all[xc_all >= 0])) < 3
        or len(np.unique(xc_all[xc_all < 0])) < 3
    ):
        raise DataInsufficient(
            "each side of the cutoff needs at least three distinct values of "
            "the running variable"
        )

    m_estimated = M is None
    if M is None:
        try:
            M = m_rule_of_thumb(xv, yv, c)
        except ValueError as exc:
            raise DataInsufficient(str(exc)) from exc
    M = float(M)
    if not np.isfinite(M) or M <= 0:
        raise MethodIncompatibility("M must be a positive number")

    # Local linear regression at its optimal bandwidth: the comparison
    # and the scale for the window and the multiplier search.
    h_ll = float(
        honest_bandwidth(
            xv,
            yv,
            c,
            M,
            opt_criterion="FLCI" if criterion == "flci" else "MSE",
            alpha=alpha,
        )
    )
    reach = float(np.abs(xc_all).max())
    window = float(h) if h is not None else min(2.5 * h_ll, reach)
    if window <= 0:
        raise MethodIncompatibility("h must be positive")
    keep = np.abs(xc_all) <= window
    xc, yy = xc_all[keep], yv[keep]
    right = xc >= 0
    if len(np.unique(xc[right])) < 3 or len(np.unique(xc[~right])) < 3:
        raise DataInsufficient(
            "each side of the cutoff needs at least three distinct values "
            "of the running variable inside the window"
        )

    # Unit-level variances (sorted within side, as sigma_nn requires).
    sig2 = np.empty_like(yy)
    for mask in (right, ~right):
        sig2[mask] = sigma_nn(xc[mask], yy[mask])
    if sigma2 is not None:
        if not np.isfinite(sigma2) or sigma2 <= 0:
            raise MethodIncompatibility("sigma2 must be a positive number")
        var_side = (float(sigma2), float(sigma2))
    else:
        var_side = (float(sig2[right].mean()), float(sig2[~right].mean()))
    if min(var_side) <= 0:
        raise DataInsufficient("the outcome has no variation near the cutoff")

    # Support points (bins) per side.
    def bins(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.abs(xc[mask])
        vals = np.unique(u)
        if vals.size <= num_bins:
            code = np.searchsorted(vals, u)
        else:
            edges = np.linspace(0.0, u.max(), num_bins + 1)
            code = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, num_bins - 1)
            code = np.unique(code, return_inverse=True)[1]
        count = np.bincount(code).astype(float)
        mean_u = np.bincount(code, weights=u) / count
        return code, count, mean_u

    code1, n1, u1 = bins(right)
    code0, n0, u0 = bins(~right)
    D1, _ = _side_design(u1, n_cells)
    D0, _ = _side_design(u0, n_cells)
    w1, w0 = n1 / var_side[0], n0 / var_side[1]

    xc_bin = np.concatenate([u1, -u0])
    n_bin = np.concatenate([n1, n0])

    def evaluate(kappa: float) -> Dict[str, Any]:
        f1, f0, ok = _solve_least_favourable((D1, D0), (w1, w0), kappa)
        # Weights are proportional to f / sigma^2 and sum to one on the
        # treated side; the first-order conditions give the rest.
        total = float(w1 @ f1)
        if not np.isfinite(total) or total <= 0:
            return {"value": np.inf, "ok": False}
        g_bin = np.concatenate([f1 / var_side[0], f0 / var_side[1]]) / total
        # Weights the solver left at numerical zero stay zero.
        tiny = 1e-9 * np.abs(g_bin).max()
        g_bin = np.where(np.abs(g_bin) > tiny, g_bin, 0.0)
        g_bin = _project(xc_bin, n_bin, g_bin)
        gamma = np.empty_like(xc)
        gamma[right] = g_bin[: u1.size][code1]
        gamma[~right] = g_bin[u1.size :][code0]
        bias = worst_case_bias(xc, gamma, M)
        var_h = float(
            np.sum(gamma[right] ** 2) * var_side[0]
            + np.sum(gamma[~right] ** 2) * var_side[1]
        )
        sd_h = np.sqrt(var_h)
        if criterion == "mse":
            value = bias**2 + var_h
        else:
            value = cv_bias(bias / sd_h, alpha) * sd_h
        return {"gamma": gamma, "bias": bias, "value": value, "ok": ok}

    # A unit jump with curvature kappa has a bandwidth of about
    # kappa ** -0.5, which gives the scale of the search.
    kappa0 = 1.0 / h_ll**2
    grid = kappa0 * 10.0 ** np.linspace(-1.0, 1.0, 9)
    fits = [evaluate(float(kap)) for kap in grid]
    values = np.array([f["value"] for f in fits])
    k = int(np.argmin(values))
    lo = float(grid[max(k - 1, 0)])
    hi = float(grid[min(k + 1, grid.size - 1)])
    best = fits[k]
    best_kappa = float(grid[k])
    cache: Dict[float, Dict[str, Any]] = {}

    def objective(log_kappa: float) -> float:
        fit = evaluate(float(np.exp(log_kappa)))
        cache[log_kappa] = fit
        return float(fit["value"])

    res = optimize.minimize_scalar(
        objective,
        bounds=(np.log(lo), np.log(hi)),
        method="bounded",
        options={"xatol": 5e-3, "maxiter": 25},
    )
    if res.fun < best["value"]:
        best = cache[float(res.x)]
        best_kappa = float(np.exp(res.x))
    if not np.isfinite(best["value"]):
        raise DataInsufficient(
            "no admissible weights were found; widen the window h or check "
            "the running variable"
        )
    if k in (0, grid.size - 1):
        warnings.warn(
            "the search over the bias-variance trade-off ended at the edge "
            "of its range; the weights are valid but may not be optimal",
            ConvergenceWarning,
            stacklevel=2,
        )
    if not best["ok"]:
        warnings.warn(
            "the least favourable function did not converge; the weights "
            "are valid but may not be optimal",
            ConvergenceWarning,
            stacklevel=2,
        )

    gamma = best["gamma"]
    max_bias = float(best["bias"])
    estimate = float(gamma @ yy)
    se = float(np.sqrt(np.sum(gamma**2 * sig2)))
    crit = cv_bias(max_bias / se, alpha)
    ci = (estimate - crit * se, estimate + crit * se)
    # Two-sided p-value at the least favourable bias, as in rd_honest.
    t = abs(estimate) / se
    b = max_bias / se
    pvalue = float(stats.norm.sf(t - b) + stats.norm.cdf(-t - b))
    carrying = np.abs(xc[gamma != 0])
    eff_bw = float(carrying.max()) if carrying.size else 0.0
    if h is None and eff_bw >= window * (1 - 1e-9) and window < reach:
        warnings.warn(
            "the optimal weights reach the edge of the default window; "
            "pass a larger h",
            ConvergenceWarning,
            stacklevel=2,
        )

    # Local linear regression with a triangular kernel, same M and data.
    from ._rdhonest import honest_weights

    w_ll = honest_weights(xc, 0.0, h_ll, "triangular")
    ll_bias = worst_case_bias(xc, w_ll, M)
    ll_se = float(np.sqrt(np.sum(w_ll**2 * sig2)))
    ll_half = cv_bias(ll_bias / ll_se, alpha) * ll_se

    weights_full = np.zeros(xv.shape[0])
    weights_full[keep] = gamma
    fuzzy_info: Dict[str, Any] = {}
    if dv is not None:
        dd = dv[keep]
        if M_fuzzy is None:
            try:
                M_fuzzy = m_rule_of_thumb(xv, dv, c)
            except ValueError as exc:
                raise DataInsufficient(str(exc)) from exc
        M_fuzzy = float(M_fuzzy)
        if not np.isfinite(M_fuzzy) or M_fuzzy < 0:
            raise MethodIncompatibility("M_fuzzy must be a non-negative number")
        # Signed nearest-neighbour deviations, so that the variance of
        # Y - t * D is available for every t.
        dev_y, dev_d = np.empty_like(yy), np.empty_like(yy)
        for mask in (right, ~right):
            dev_y[mask] = _nn_deviation(xc[mask], yy[mask])
            dev_d[mask] = _nn_deviation(xc[mask], dd[mask])
        reduced = estimate  # jump in the outcome
        first = float(gamma @ dd)  # jump in treatment received
        unit_bias = max_bias / M  # integral of |G| on both sides

        def gap(t: float) -> float:
            """Distance of the statistic at ratio ``t`` from its critical value."""
            se_t = float(np.sqrt(np.sum(gamma**2 * (dev_y - t * dev_d) ** 2)))
            bias_t = (M + abs(t) * M_fuzzy) * unit_bias
            return abs(reduced - t * first) - cv_bias(bias_t / se_t, alpha) * se_t

        if first == 0:
            raise DataInsufficient("the treatment received does not jump at the cutoff")
        ratio = reduced / first
        se_ratio = float(
            np.sqrt(np.sum(gamma**2 * (dev_y - ratio * dev_d) ** 2)) / abs(first)
        )
        step = max(se_ratio, 1e-8 * max(1.0, abs(ratio)))

        def endpoint(direction: float) -> float:
            lo, width = ratio, step
            for _ in range(60):
                hi = lo + direction * width
                if gap(hi) > 0:
                    return float(optimize.brentq(gap, lo, hi, xtol=1e-10 * step))
                lo, width = hi, 2.0 * width
            return direction * np.inf

        ci = (endpoint(-1.0), endpoint(1.0))
        se_first = float(np.sqrt(np.sum(gamma**2 * dev_d**2)))
        fuzzy_info = {
            "fuzzy": fuzzy,
            "M_fuzzy": M_fuzzy,
            "reduced_form": reduced,
            "reduced_form_se": se,
            "first_stage": first,
            "first_stage_se": se_first,
            "first_stage_max_bias": M_fuzzy * unit_bias,
        }
        estimate, se = float(ratio), se_ratio
        # The test of a zero ratio is the test of a zero reduced form.
        t0 = abs(reduced) / fuzzy_info["reduced_form_se"]
        b0 = max_bias / fuzzy_info["reduced_form_se"]
        pvalue = float(stats.norm.sf(t0 - b0) + stats.norm.cdf(-t0 - b0))

    model_info: Dict[str, Any] = {
        "M": M,
        "M_estimated": bool(m_estimated),
        "criterion": criterion,
        "max_bias": max_bias,
        "bias_to_se": max_bias / se,
        "critical_value": float(crit),
        "half_length": float(crit * se),
        "effective_bandwidth": eff_bw,
        "window": window,
        "sigma2_for_weights": var_side,
        "curvature_ratio": best_kappa,
        "n_window": int(keep.sum()),
        "n_weighted": int(np.sum(gamma != 0)),
        "n_treated": int(np.sum(gamma[right] != 0)),
        "n_control": int(np.sum(gamma[~right] != 0)),
        "weights": weights_full,
        "index": index,
        "local_linear": {
            "bandwidth": h_ll,
            "estimate": float(w_ll @ yy),
            "max_bias": float(ll_bias),
            "se": ll_se,
            "half_length": float(ll_half),
        },
        **fuzzy_info,
    }
    return CausalResult(
        method="Optimized regression discontinuity (minimax linear)",
        estimand=(
            "RD effect at the cutoff"
            if fuzzy is None
            else "Fuzzy RD effect at the cutoff (ratio of jumps)"
        ),
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=(float(ci[0]), float(ci[1])),
        alpha=float(alpha),
        n_obs=int(keep.sum()),
        model_info=model_info,
    )


__all__ = ["rd_optimized", "worst_case_bias"]
