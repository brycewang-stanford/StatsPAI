"""
Bias-aware confidence sets for fuzzy RD (Noack & Rothe 2024).

``rd_bias_aware_fuzzy`` reports a bias-aware confidence set for the fuzzy RD
parameter ``tau = Delta_Y / Delta_D``, the ratio of the jumps in the outcome
and treatment means at the cutoff, both estimated by the same local linear
weights. Following ``noack2024biasaware`` the set is obtained by inverting
Anderson-Rubin type tests: ``t`` is in the set when

    |Delta_Y_hat - t Delta_D_hat| <= cv_{1-alpha}(b(t) / s(t)) s(t),

where ``s(t)`` is the standard error of ``Delta_Y_hat - t Delta_D_hat`` and
``b(t) = (M_Y + |t| M_D) B_1`` its worst-case bias over
``|g_Y''| <= M_Y, |g_D''| <= M_D`` (``B_1`` the bias of the weights per unit
curvature). The set is valid however weak the first stage is; it may be
unbounded or a union of two rays, which is the information a weak first stage
carries (``kaliski2025power`` discuss the power asymmetry of the usual
t-ratio interval).

The ingredients (jumps, nearest-neighbour or cluster variance, worst-case
bias, rule-of-thumb bounds, default bandwidth) and the linearised
Wald-ratio interval of ``armstrong2020simple`` are also returned under
``model_info['bias_aware']['rdhonest']``; these follow the documentation of R
``RDHonest`` (vignette, fuzzy RD section). Clean-room implementation: the
reference (GPL-3) was used only as a black box, see
``docs/dev/cleanroom/rd_honest.md``.

References
----------
noack2024biasaware, kaliski2025power, armstrong2020simple
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core._vcov_spec import accepts_vce
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._rdhonest import (
    _KERNELS,
    _ik_bandwidth,
    _kern,
    _ll_residuals,
    _min_bandwidth,
    _optimal_h,
    cv_bias,
    honest_bias,
    honest_weights,
    m_rule_of_thumb,
    nn_deviations,
)

__all__ = ["rd_bias_aware_fuzzy"]


@records_call("rd_bias_aware_fuzzy")
@accepts_vce(default=None)
def rd_bias_aware_fuzzy(
    data: pd.DataFrame,
    y: str,
    x: str,
    fuzzy: str,
    c: float = 0.0,
    M_y: Optional[float] = None,
    M_d: Optional[float] = None,
    h: Optional[float] = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
    cluster: Optional[str] = None,
    n_grid: int = 401,
) -> CausalResult:
    """Bias-aware confidence interval for fuzzy RD (Noack & Rothe 2024).

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        Outcome.
    x : str
        Running variable; units with ``x >= c`` are assigned to treatment.
    fuzzy : str
        Treatment received.
    c : float, default 0
        Cutoff.
    M_y, M_d : float, optional
        Bounds on the second derivatives of the outcome and treatment means
        on either side. When omitted, the rule of thumb of Armstrong and
        Kolesar (global quartic on each side) is applied to each.
    h : float, optional
        Bandwidth. When omitted, the bandwidth minimising the worst-case MSE
        of ``Delta_Y_hat - tau_h Delta_D_hat`` (``tau_h`` the Wald ratio at
        that bandwidth), with worst-case bias ``M_y B_1(h)`` (an initial
        effect of zero) and a preliminary covariance of ``(Y, D)`` that is
        homoskedastic on each side.
    kernel : {'triangular', 'uniform', 'epanechnikov'}
    alpha : float, default 0.05
    cluster : str, optional
        Cluster identifier. Standard errors become cluster-robust sums of the
        weighted local linear residuals, and the bandwidth search adds a
        common within-cluster covariance.
    n_grid : int, default 401
        Points of the diagnostic rejection grid returned in
        ``model_info['bias_aware']['rejection_grid']``. The endpoints of the
        confidence set are solved exactly, not read off the grid.

    Returns
    -------
    CausalResult
        ``estimate`` is the Wald ratio, ``se`` its delta-method standard
        error, ``ci`` the Anderson-Rubin bias-aware set (with infinite ends
        when it is unbounded). ``model_info['bias_aware']`` carries the
        jumps, the 2x2 variance of ``(Delta_Y_hat, Delta_D_hat)``, the bias
        per unit curvature, the first-stage F statistic, whether the set is
        a single interval, and the linearised interval under ``'rdhonest'``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 1500)
    >>> d = (rng.uniform(size=1500) < 0.2 + 0.5 * (x >= 0)).astype(float)
    >>> yy = 1 + 2 * d + 0.5 * x + rng.normal(0, 0.7, 1500)
    >>> df = pd.DataFrame({"y": yy, "d": d, "x": x})
    >>> r = sp.rd_bias_aware_fuzzy(df, y="y", x="x", fuzzy="d", M_y=2, M_d=0.5)

    References
    ----------
    noack2024biasaware, armstrong2020simple
    """
    if kernel not in _KERNELS:
        raise MethodIncompatibility(
            f"kernel must be one of {_KERNELS}, got {kernel!r}",
            recovery_hint="Use kernel='triangular' (the default).",
        )
    check_columns(
        data,
        {"y": y, "x": x, "fuzzy": fuzzy, "cluster": cluster or None},
        function="sp.rd_bias_aware_fuzzy",
    )
    cols = [y, x, fuzzy] + ([cluster] if cluster else [])
    df = data[cols].dropna()
    xv = df[x].to_numpy(dtype=float)
    Y = np.column_stack([df[y].to_numpy(dtype=float), df[fuzzy].to_numpy(dtype=float)])
    cl = df[cluster].to_numpy() if cluster else None
    n_obs = int(xv.size)

    def _rot(col: int, label: str) -> float:
        try:
            return m_rule_of_thumb(xv, Y[:, col], c)
        except ValueError as exc:
            raise DataInsufficient(
                str(exc),
                recovery_hint=f"Supply {label}.",
            ) from exc

    My_est, Md_est = M_y is None, M_d is None
    My = _rot(0, "M_y") if My_est else float(M_y)
    Md = _rot(1, "M_d") if Md_est else float(M_d)
    if My < 0 or Md < 0:
        raise MethodIncompatibility(
            "M_y and M_d must be non-negative",
            recovery_hint="Pass curvature bounds >= 0.",
        )

    if h is None:
        try:
            h = _frd_bandwidth(xv, Y, c, My, Md, kernel, alpha, cl=cl)
        except ValueError as exc:
            raise DataInsufficient(
                str(exc), recovery_hint="Supply h, or check the data near the cutoff."
            ) from exc
    h = float(h)
    try:
        comp = _frd_components(xv, Y, c, h, kernel, cl)
    except ValueError as exc:
        raise DataInsufficient(
            str(exc), recovery_hint="Widen h; each side needs two support points."
        ) from exc

    dY, dD, V, unit = comp["delta_y"], comp["delta_d"], comp["V"], comp["unit_bias"]
    scale_d = max(float(np.max(np.abs(Y[:, 1]))), 1e-300)
    if abs(dD) <= 1e-10 * scale_d:
        raise DataInsufficient(
            "the treatment received does not jump at the cutoff",
            recovery_hint="Check the fuzzy= column; a sharp design needs "
            "sp.rd_honest.",
        )

    # Linearised Wald-ratio interval (Armstrong-Kolesar; RDHonest's output)
    tau = dY / dD
    se_tau = _se_comb(V, tau) / abs(dD)
    bias_tau = (My + abs(tau) * Md) * unit / abs(dD)
    b = bias_tau / se_tau if se_tau > 0 else 0.0
    cv = cv_bias(b, alpha)
    tstat = abs(tau) / se_tau
    rdh = {
        "estimate": float(tau),
        "std.error": float(se_tau),
        "maximum.bias": float(bias_tau),
        "cv": float(cv),
        "conf.low": float(tau - cv * se_tau),
        "conf.high": float(tau + cv * se_tau),
        "p.value": float(stats.norm.sf(tstat - b) + stats.norm.cdf(-tstat - b)),
    }

    # Anderson-Rubin bias-aware set
    pieces, ar_grid = _ar_set(dY, dD, V, My, Md, unit, alpha, tau, se_tau, n_grid)
    lo = pieces[0][0] if pieces else np.nan
    hi = pieces[-1][1] if pieces else np.nan
    z = float(stats.norm.ppf(1 - alpha / 2))
    naive = (tau - z * se_tau, tau + z * se_tau)
    F = float(dD**2 / V[1, 1]) if V[1, 1] > 0 else np.inf

    summary = (
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Bias-Aware Fuzzy RD (Noack & Rothe 2024 ECTA)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  τ̂ (Wald ratio):           {tau:.4f}\n"
        f"  Δ̂_Y (numerator jump):     {dY:.4f}\n"
        f"  Δ̂_D (denominator jump):   {dD:.4f}\n"
        f"  First-stage F:            {F:.2f}\n"
        "\n"
        f"  Naive {int(round((1 - alpha) * 100))}% CI:       [{naive[0]:.4f}, {naive[1]:.4f}]\n"
        f"  Bias-aware {int(round((1 - alpha) * 100))}% CI:  [{lo:.4f}, {hi:.4f}]"
        + ("  (not a single interval)" if len(pieces) > 1 else "")
        + "\n\n"
        f"  M_y:                      {My:.4g} ({'rule of thumb' if My_est else 'supplied'})\n"
        f"  M_d:                      {Md:.4g} ({'rule of thumb' if Md_est else 'supplied'})\n"
        f"  Bandwidth h:              {h:.4f}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    info: Dict[str, Any] = {
        "naive_ci": naive,
        "bias_aware_ci": (float(lo), float(hi)),
        "bias_aware_pieces": [(float(a), float(b_)) for a, b_ in pieces],
        "non_convex_ci": len(pieces) > 1,
        "M_y": float(My),
        "M_d": float(Md),
        "M_y_estimated": bool(My_est),
        "M_d_estimated": bool(Md_est),
        "delta_y": float(dY),
        "delta_d": float(dD),
        "first_stage_F": F,
        "bandwidth": h,
        "kernel": kernel,
        "bias_per_unit_M": float(unit),
        "variance_2x2": [
            float(V[0, 0]),
            float(V[0, 1]),
            float(V[1, 0]),
            float(V[1, 1]),
        ],
        "n_effective": int(comp["n_eff"]),
        "rdhonest": rdh,
        "rejection_grid": ar_grid,
    }
    return CausalResult(
        method="Bias-aware fuzzy RD (Noack-Rothe 2024)",
        estimand="LATE at cutoff",
        estimate=float(tau),
        se=float(se_tau),
        pvalue=float(2 * stats.norm.sf(tstat)),
        ci=(float(lo), float(hi)),
        alpha=alpha,
        n_obs=n_obs,
        model_info={"bias_aware": info, "summary_str": summary},
    )


def _frd_components(
    x: np.ndarray,
    Y: np.ndarray,
    c: float,
    h: float,
    kernel: str,
    cl: Optional[np.ndarray] = None,
    J: int = 3,
) -> Dict[str, Any]:
    """Jumps of both columns, the 2x2 variance of the jump estimator and the
    worst-case bias per unit curvature (Holder class) at bandwidth ``h``.

    Without clusters the variance is ``sum_i w_i^2 Sigma_i`` with ``Sigma_i``
    the nearest-neighbour covariance of ``(Y_i, D_i)`` computed within the
    window on each side; with clusters it is
    ``sum_g (sum_{i in g} w_i e_i)(sum_{i in g} w_i e_i)'`` with ``e`` the
    local linear residuals of both columns.
    """
    w = honest_weights(x, c, h, kernel)
    jumps = w @ Y
    inwin = _kern((x - c) / h, kernel) > 0
    if cl is None:
        dev = np.zeros_like(Y)
        right = x >= c
        for side in (right, ~right):
            m = side & inwin
            dev[m] = nn_deviations(x[m], Y[m], J)
        wd = w[:, None] * dev
        V = wd.T @ wd
    else:
        E = np.column_stack(
            [_ll_residuals(x, Y[:, j], c, h, kernel)[0] for j in range(2)]
        )
        _, inv = np.unique(cl, return_inverse=True)
        S = np.column_stack([np.bincount(inv, weights=w * E[:, j]) for j in range(2)])
        V = S.T @ S
    return {
        "delta_y": float(jumps[0]),
        "delta_d": float(jumps[1]),
        "V": V,
        "unit_bias": honest_bias(w, x, c, 1.0, "H"),
        "n_eff": int(np.count_nonzero(w)),
    }


def _prelim_cov(
    x: np.ndarray, Y: np.ndarray, c: float, cl: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Preliminary covariance of ``(Y, D)`` for the bandwidth search.

    Residuals of triangular-kernel local linear fits of both columns at the
    Imbens-Kalyanaraman bandwidth of the outcome (floored so each side has
    three support points); their cross-products are averaged within each
    side. With clusters, also the common within-cluster covariance matrix of
    the residuals (Moulton structure).
    """
    h0 = max(_ik_bandwidth(x, Y[:, 0], c), _min_bandwidth(x, c))
    E = np.empty_like(Y)
    for j in range(Y.shape[1]):
        E[:, j], inwin = _ll_residuals(x, Y[:, j], c, h0, "triangular")
    right = x >= c
    S = []
    for side in (~right, right):
        e = E[side & inwin]
        S.append(e.T @ e / e.shape[0])
    rho = None
    if cl is not None:
        e = E[inwin]
        _, inv = np.unique(np.asarray(cl)[inwin], return_inverse=True)
        sums = np.column_stack(
            [np.bincount(inv, weights=e[:, j]) for j in range(e.shape[1])]
        )
        n_g = np.bincount(inv)
        pairs = float(np.sum(n_g * (n_g - 1)))
        rho = (sums.T @ sums - e.T @ e) / pairs if pairs > 0 else np.zeros((2, 2))
    return S[0], S[1], rho


def _frd_bandwidth(
    x: np.ndarray,
    Y: np.ndarray,
    c: float,
    M_y: float,
    M_d: float,
    kernel: str,
    alpha: float,
    T0: float = 0.0,
    cl: Optional[np.ndarray] = None,
) -> float:
    """MSE-optimal bandwidth for fuzzy RD.

    For each candidate ``h``: worst-case bias ``(M_y + |T0| M_d) B_1(h)``
    and variance ``sum_i w_i^2 a' Sigma_side(i) a`` with
    ``a = (1, -tau_h)``, ``tau_h`` the Wald ratio at ``h`` and ``Sigma`` the
    preliminary covariance (plus the within-cluster term). Searched as in the
    sharp case (Brent over the range of the running variable; the support
    grid for the uniform kernel).
    """
    Sm, Sp, rho = _prelim_cov(x, Y, c, cl)
    right = x >= c
    inv = None if cl is None else np.unique(np.asarray(cl), return_inverse=True)[1]

    def objective(h: float) -> float:
        try:
            w = honest_weights(x, c, h, kernel)
        except ValueError:
            return np.inf
        jumps = w @ Y
        if jumps[1] == 0:
            return np.inf
        a = np.array([1.0, -jumps[0] / jumps[1]])
        s2 = np.where(right, a @ Sp @ a, a @ Sm @ a)
        var = float(np.sum(w * w * s2))
        if rho is not None:
            sw = np.bincount(inv, weights=w)
            var += float(a @ rho @ a) * float(np.sum(sw * sw) - np.sum(w * w))
        bias = (M_y + abs(T0) * M_d) * honest_bias(w, x, c, 1.0, "H")
        return bias * bias + max(var, 0.0)

    h_lo = _min_bandwidth(x, c, 2)
    h_hi = float(np.max(np.abs(x - c)))
    if kernel == "uniform":
        grid = np.unique(np.abs(x - c))
        grid = grid[grid >= h_lo]
        vals = np.array([objective(h) for h in grid])
        return float(grid[int(np.nanargmin(vals))])
    return _optimal_h(objective, h_lo, h_hi)


def _se_comb(V: np.ndarray, t: float) -> float:
    """Standard error of ``Delta_Y_hat - t Delta_D_hat``."""
    return float(np.sqrt(max(V[0, 0] - 2.0 * t * V[0, 1] + t * t * V[1, 1], 0.0)))


def _ar_set(
    dY: float,
    dD: float,
    V: np.ndarray,
    My: float,
    Md: float,
    unit: float,
    alpha: float,
    tau: float,
    se_tau: float,
    n_grid: int,
) -> Tuple[List[Tuple[float, float]], Tuple[List[float], List[bool]]]:
    """The set ``{t : |dY - t dD| <= cv(b(t)/s(t)) s(t)}`` as a list of
    intervals (ends may be infinite), and a diagnostic grid around ``tau``.

    ``g(t) = |dY - t dD| - cv(b(t)/s(t)) s(t)`` is continuous; the set is
    ``g <= 0``. Its ends are the roots of ``g``, bracketed on a grid and
    solved by Brent's method. Far from ``tau`` both terms grow linearly in
    ``|t|``, so the set is unbounded in a direction exactly when
    ``|dD| <= cv(Md B_1 / sqrt(V_DD)) sqrt(V_DD)``.
    """

    def g(t: float) -> float:
        s = _se_comb(V, t)
        b = (My + abs(t) * Md) * unit
        if s <= 0:
            return abs(dY - t * dD) - b
        return abs(dY - t * dD) - cv_bias(b / s, alpha) * s

    sdd = float(np.sqrt(max(V[1, 1], 0.0)))
    unbounded = sdd > 0 and abs(dD) <= cv_bias(Md * unit / sdd, alpha) * sdd
    width = max(5.0, 20.0 * se_tau)
    span = width
    # widen until the set is bracketed (or known to be unbounded)
    for _ in range(60):
        if unbounded or (g(tau - span) > 0 and g(tau + span) > 0):
            break
        span *= 2.0
    ts = tau + np.linspace(-span, span, max(int(n_grid), 3) * 4 + 1)
    gs = np.array([g(t) for t in ts])
    inside = gs <= 0
    pieces: List[Tuple[float, float]] = []
    start: Optional[float] = -np.inf if (unbounded and inside[0]) else None
    for i in range(1, ts.size):
        if inside[i] and not inside[i - 1]:
            start = optimize.brentq(g, ts[i - 1], ts[i], xtol=1e-14, rtol=1e-15)
        elif inside[i - 1] and not inside[i]:
            end = optimize.brentq(g, ts[i - 1], ts[i], xtol=1e-14, rtol=1e-15)
            pieces.append((start if start is not None else ts[i - 1], end))
            start = None
    if start is not None or (inside[-1] and unbounded):
        pieces.append((start if start is not None else -np.inf, np.inf))
    grid = tau + np.linspace(-width, width, max(int(n_grid), 2))
    accepted = [bool(g(t) <= 0) for t in grid]
    return pieces, ([float(t) for t in grid], accepted)
