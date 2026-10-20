"""Bias-aware ("honest") inference for sharp RD with local linear regression.

Clean-room implementation, written from the method papers
(``armstrong2018optimal``, ``armstrong2020simple``, ``kolesar2018inference``,
``imbens2012optimal``) and from the documentation of the R package
``RDHonest`` (its help pages and vignette). ``RDHonest`` (GPL-3) was used
only as a black box: its outputs on the fixtures under
``tests/reference_parity/_fixtures`` are the acceptance target. The
specification this file implements is ``docs/dev/cleanroom/rd_honest.md``.

The estimator
-------------
Local linear regression on each side of the cutoff with kernel weights
``K((x - c) / h)``. The jump estimate is a linear functional of the
outcomes, ``tau_hat = sum_i w_i Y_i``, where ``w`` holds the intercept
weights of the right-hand fit and minus those of the left-hand fit.

Worst-case bias over ``|f''| <= M`` (``M`` is the bound on the second
derivative on either side):

* Holder class (``sclass='H'``, the default): the least favourable function
  is the piecewise quadratic ``M d^2 (1{d < 0} - 1{d >= 0}) / 2`` with
  ``d = x - c`` (vignette, eq. for ``B_{M,h}``), so the bound is
  ``(M / 2) |sum_i w_i d_i^2 sign(d_i)|``.
* Taylor class (``sclass='T'``): ``(M / 2) sum_i |w_i| d_i^2``.

Standard error ``sqrt(sum_i w_i^2 sigma_i^2)`` with ``sigma_i^2`` the
nearest-neighbour variance (Abadie-Imbens), or, with clusters, the
cluster-robust form ``sqrt(sum_g (sum_{i in g} w_i e_i)^2)`` with ``e`` the
local linear residuals. The interval is ``tau_hat +/- cv_{1-alpha}(B/se) se``
where ``cv_{1-alpha}(t)`` is the ``1 - alpha`` quantile of ``|N(t, 1)|``.

Bandwidth and smoothness choices
--------------------------------
* ``M`` (when not supplied): the rule of thumb of ``armstrong2020simple``,
  the largest ``|f''|`` of a global quartic fitted separately on each side,
  over the range of the data on that side.
* ``h`` (when not supplied): minimises worst-case MSE, the FLCI half-length
  or the ``beta`` quantile of the one-sided excess length, computed under a
  preliminary variance that is homoskedastic on each side of the cutoff,
  estimated from residuals of a local linear fit at the Imbens-Kalyanaraman
  bandwidth (vignette). With clusters the preliminary covariance adds a
  common within-cluster covariance (Moulton structure, vignette).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import optimize, stats

__all__ = [
    "nn_deviations",
    "cv_bias",
    "honest_weights",
    "honest_bias",
    "sigma_nn",
    "honest_fit",
    "honest_bandwidth",
    "m_rule_of_thumb",
    "honest_rd",
]

_KERNELS = ("triangular", "uniform", "epanechnikov")


# --------------------------------------------------------------------------
# Kernels and the linear-estimator weights
# --------------------------------------------------------------------------


def _kern(u: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel ``K(u)`` up to a constant (constants cancel in the weights).

    Supports are closed: ``|u| <= 1``. For the triangular and Epanechnikov
    kernels the boundary point carries zero weight anyway; for the uniform
    kernel an observation exactly at distance ``h`` is inside the window.
    """
    a = np.abs(np.asarray(u, dtype=float))
    if kernel == "triangular":
        return np.clip(1.0 - a, 0.0, None)
    if kernel == "uniform":
        return (a <= 1.0).astype(float)
    if kernel == "epanechnikov":
        return np.clip(1.0 - a * a, 0.0, None)
    raise ValueError(f"kernel must be one of {_KERNELS}, got {kernel!r}")


def _side_intercept_weights(d: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Weights ``a`` with ``a @ y`` = intercept of the ``k``-weighted fit of
    ``y`` on ``(1, d)``. Only rows with ``k > 0`` enter; the rest get 0."""
    a = np.zeros_like(d)
    m = k > 0
    dm, km = d[m], k[m]
    s0, s1, s2 = km.sum(), (km * dm).sum(), (km * dm * dm).sum()
    det = s0 * s2 - s1 * s1
    if not np.isfinite(det) or det <= 1e-12 * max(s0 * s2, 1e-300):
        raise ValueError(
            "fewer than two distinct values of the running variable carry "
            "kernel weight on one side of the cutoff; increase the bandwidth"
        )
    a[m] = km * (s2 - s1 * dm) / det
    return a


def honest_weights(
    x: np.ndarray, c: float, h: float, kernel: str = "triangular"
) -> np.ndarray:
    """Weights ``w`` with ``tau_hat = sum_i w_i Y_i`` for local linear RD.

    Units with ``x >= c`` are treated. The right-hand intercept weights enter
    with a plus sign and the left-hand ones with a minus sign, so ``w`` sums
    to zero and is orthogonal to ``x`` and to ``x 1{x >= c}``.

    Raises
    ------
    ValueError
        If either side has fewer than two distinct values with positive
        kernel weight (the local line is not identified).
    """
    d = np.asarray(x, dtype=float) - c
    k = _kern(d / h, kernel)
    right = d >= 0
    w = np.zeros_like(d)
    w[right] = _side_intercept_weights(d[right], k[right])
    w[~right] = -_side_intercept_weights(d[~right], k[~right])
    return w


def honest_bias(
    w: np.ndarray, x: np.ndarray, c: float, M: float, sclass: str = "H"
) -> float:
    """Worst-case bias of ``sum_i w_i Y_i`` over ``|f''| <= M`` on each side.

    ``'H'`` (Holder): ``(M/2) |sum_i w_i d_i^2 sign(d_i)|``, attained by the
    piecewise quadratic of the RDHonest vignette (the bias is linear in
    ``f``, and local linear weights do not change the sign of the relevant
    integral, ``armstrong2020simple``). ``'T'`` (Taylor):
    ``(M/2) sum_i |w_i| d_i^2``, no cancellation allowed.
    """
    d = np.asarray(x, dtype=float) - c
    w = np.asarray(w, dtype=float)
    if sclass == "H":
        return float(0.5 * M * abs(np.sum(w * d * d * np.sign(d))))
    if sclass == "T":
        return float(0.5 * M * np.sum(np.abs(w) * d * d))
    raise ValueError(f"sclass must be 'H' or 'T', got {sclass!r}")


# --------------------------------------------------------------------------
# Critical value
# --------------------------------------------------------------------------


def cv_bias(t: float, alpha: float = 0.05) -> float:
    """``1 - alpha`` quantile of ``|N(t, 1)|``.

    The critical value of a bias-aware interval for an estimator whose
    worst-case bias is ``t`` standard errors (``armstrong2020simple``). It is
    the root ``c`` of ``Phi(c - t) - Phi(-c - t) = 1 - alpha``; at ``t = 0`` it
    is ``z_{1 - alpha/2}`` and it approaches ``t + z_{1-alpha}`` as ``t``
    grows.

    Examples
    --------
    >>> from statspai.rd._rdhonest import cv_bias
    >>> round(cv_bias(0.0), 6)
    1.959964
    """
    t = abs(float(t))
    alpha = float(alpha)

    def tail(cv: float) -> float:
        # P(|N(t,1)| > cv) - alpha, written with upper tails for accuracy
        return stats.norm.sf(cv - t) + stats.norm.cdf(-cv - t) - alpha

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    # tail(0) = 1 - alpha > 0 and tail decreases to -alpha, so [0, hi] brackets
    hi = t + abs(stats.norm.isf(alpha / 2.0)) + 1.0
    while tail(hi) > 0:  # pragma: no cover - the bracket above always works
        hi *= 2.0
    return float(optimize.brentq(tail, 0.0, hi, xtol=1e-15, rtol=1e-15, maxiter=500))


# --------------------------------------------------------------------------
# Nearest-neighbour variance
# --------------------------------------------------------------------------


def nn_deviations(x: np.ndarray, Y: np.ndarray, J: int = 3) -> np.ndarray:
    """Scaled nearest-neighbour deviations on one side of the cutoff.

    Row ``i`` is ``sqrt(m/(m+1)) (Y_i - Ybar_i)`` where ``Ybar_i`` averages the
    outcomes of the ``m`` nearest neighbours of unit ``i`` in ``x`` (Abadie and
    Imbens's matching estimator of the conditional variance). ``m = J``
    unless neighbours are tied at the ``J``-th distance: all tied units are
    used and ``m`` counts them, which keeps
    ``E[(Y_i - Ybar_i)(Y_i - Ybar_i)'] = (1 + 1/m) Sigma`` and matters with a
    discrete running variable. The outer product of a row with itself
    estimates ``Var(Y_i | x_i)``; ``Y`` may have several columns.
    """
    x = np.asarray(x, dtype=float)
    Y = np.asarray(Y, dtype=float)
    vec = Y.ndim == 1
    Y2 = Y[:, None] if vec else Y
    n = x.size
    out = np.zeros_like(Y2)
    if n < 2:
        return out[:, 0] if vec else out
    J = int(min(J, n - 1))
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], Y2[order]
    # groups of equal x: neighbours at distance zero
    u, start, cnt = np.unique(xs, return_index=True, return_counts=True)
    gsum = np.add.reduceat(ys, start, axis=0)
    G = u.size
    nb_sum = np.empty_like(gsum)
    nb_cnt = np.empty(G, dtype=np.int64)
    for g in range(G):
        tot_s, tot_n = gsum[g].copy(), int(cnt[g])
        lo, hi = g - 1, g + 1
        # expand outward until at least J others are in, keeping ties whole
        while tot_n - 1 < J:
            dl = u[g] - u[lo] if lo >= 0 else np.inf
            dr = u[hi] - u[g] if hi < G else np.inf
            dist = min(dl, dr)
            if dl == dist:
                tot_s += gsum[lo]
                tot_n += int(cnt[lo])
                lo -= 1
            if dr == dist:
                tot_s += gsum[hi]
                tot_n += int(cnt[hi])
                hi += 1
        nb_sum[g] = tot_s
        nb_cnt[g] = tot_n
    grp = np.repeat(np.arange(G), cnt)
    m = (nb_cnt[grp] - 1).astype(float)
    dev = ys - (nb_sum[grp] - ys) / m[:, None]
    out[order] = np.sqrt(m / (m + 1.0))[:, None] * dev
    return out[:, 0] if vec else out


def sigma_nn(x: np.ndarray, y: np.ndarray, J: int = 3) -> np.ndarray:
    """Nearest-neighbour estimates of ``Var(Y_i | x_i)`` on one side:
    ``m/(m+1) (Y_i - Ybar_i)^2``, see :func:`nn_deviations` (ties at the
    ``J``-th distance are all used). Call it separately for each side of the
    cutoff; ``x`` need not be sorted.
    """
    return nn_deviations(x, y, J) ** 2


def _nn_variance_by_side(
    x: np.ndarray, y: np.ndarray, c: float, J: int, inwin: np.ndarray
) -> np.ndarray:
    """Nearest-neighbour variances computed within the estimation window
    (units with positive kernel weight), separately on each side; 0 for
    units outside the window, which carry no weight."""
    right = x >= c
    s2 = np.zeros_like(np.asarray(y, dtype=float))
    for side in (right, ~right):
        m = side & inwin
        s2[m] = sigma_nn(x[m], y[m], J)
    return s2


# --------------------------------------------------------------------------
# Local linear residuals and preliminary variance
# --------------------------------------------------------------------------


def _ll_residuals(
    x: np.ndarray, y: np.ndarray, c: float, h: float, kernel: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Residuals of the kernel-weighted linear fit on each side, and the
    in-window mask (positive kernel weight). Residuals outside are 0."""
    d = np.asarray(x, dtype=float) - c
    y = np.asarray(y, dtype=float)
    k = _kern(d / h, kernel)
    inwin = k > 0
    e = np.zeros_like(d)
    for side in (d >= 0, d < 0):
        m = side & inwin
        X = np.column_stack([np.ones(m.sum()), d[m]])
        sw = np.sqrt(k[m])
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y[m] * sw, rcond=None)
        e[m] = y[m] - X @ beta
    return e, inwin


def _min_bandwidth(x: np.ndarray, c: float, n_min: int = 3) -> float:
    """Smallest bandwidth with at least ``n_min`` distinct support points
    on each side inside the closed window ``|x - c| <= h``."""
    d = x - c
    d = np.asarray(d, dtype=float)
    out = 0.0
    for side in (np.unique(d[d >= 0]), np.unique(-d[d < 0])):
        if side.size < n_min:
            raise ValueError(
                "fewer than three distinct values of the running variable on "
                "one side of the cutoff; honest local linear inference is not "
                "available"
            )
        out = max(out, float(np.sort(side)[n_min - 1]))
    return out


def _edge_kernel_constant() -> float:
    """``C_K`` of ``imbens2012optimal`` for the triangular (edge) kernel,
    computed from the kernel moments (the paper rounds it to 3.4375).

    For boundary local linear regression with kernel ``K`` on ``[0, 1]``,
    with ``nu_j = int u^j K`` and ``pi_j = int u^j K^2``:
    ``B = (nu_2^2 - nu_1 nu_3) / (nu_0 nu_2 - nu_1^2)`` (bias constant) and
    ``V = int (nu_2 - nu_1 u)^2 K^2 / (nu_0 nu_2 - nu_1^2)^2`` (variance
    constant), and ``C_K = (V / B^2)^(1/5)``. For ``K(u) = 1 - u`` the moments
    are rational: nu = (1/2, 1/6, 1/12, 1/20), pi = (1/3, 1/12, 1/30).
    """
    nu0, nu1, nu2, nu3 = 1 / 2, 1 / 6, 1 / 12, 1 / 20
    pi0, pi1, pi2 = 1 / 3, 1 / 12, 1 / 30
    den = nu0 * nu2 - nu1**2
    B = (nu2**2 - nu1 * nu3) / den
    V = (nu2**2 * pi0 - 2 * nu1 * nu2 * pi1 + nu1**2 * pi2) / den**2
    return (V / B**2) ** 0.2


def _ik_bandwidth(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """Imbens-Kalyanaraman bandwidth for the jump (``imbens2012optimal``).

    Pilot only: it fixes the window of the preliminary local linear fit that
    estimates the conditional variance used to choose ``h``. It is not
    clamped to the range of the data. Steps, with ``d = x - c``:

    1. ``h1 = 1.84 s_x N^(-1/5)`` (``s_x`` the sample standard deviation);
       density ``f = (N_-(h1) + N_+(h1)) / (2 N h1)``; side sample variances
       (divisor ``n - 1``) of ``Y`` within ``max(h1, h_min)``, where
       ``h_min`` is the smallest window with three support points on each
       side (binds only with a discrete running variable).
    2. ``m3 = 6 g4`` from a global cubic in ``d`` with a jump indicator;
       ``h2_side = 7200^(1/7) (s2_side / (f m3^2))^(1/7) N_side^(-1/7)``
       (the paper rounds ``7200^(1/7)`` to 3.56); curvatures ``m2_side``
       from an unweighted quadratic on ``[0, h2]`` (resp. ``[-h2, 0)``).
    3. ``r_side = 2160 s2_side / (n2_side h2_side^4)`` and
       ``h = C_K ((s2_- + s2_+) / (f ((m2_+ - m2_-)^2 + r_+ + r_-)))^(1/5)
       N^(-1/5)`` with the exact edge-kernel constant ``C_K``.

    The two unrounded constants and the regularisation ``2160`` were
    identified by matching ``RDHonest``'s selected bandwidths on simulated
    designs (black box); see ``docs/dev/cleanroom/rd_honest.md``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - c
    N = d.size
    right = d >= 0
    Np, Nm = int(right.sum()), int((~right).sum())
    h1 = 1.84 * np.std(x, ddof=1) * N ** (-0.2)
    f0 = (np.sum(right & (d <= h1)) + np.sum((~right) & (d >= -h1))) / (2.0 * N * h1)
    # the side variances use the Silverman window, widened if needed so
    # that each side has three support points (a discrete running variable)
    hv = max(h1, _min_bandwidth(x, c))
    wr = right & (d <= hv)
    wl = (~right) & (d >= -hv)
    s2p = float(np.var(y[wr], ddof=1))
    s2m = float(np.var(y[wl], ddof=1))
    X = np.column_stack([np.ones(N), right.astype(float), d, d**2, d**3])
    gam, *_ = np.linalg.lstsq(X, y, rcond=None)
    m3sq = (6.0 * gam[4]) ** 2
    k2 = 7200.0 ** (1.0 / 7.0)
    h2p = k2 * (s2p / (f0 * m3sq)) ** (1 / 7) * Np ** (-1 / 7)
    h2m = k2 * (s2m / (f0 * m3sq)) ** (1 / 7) * Nm ** (-1 / 7)

    def curv(mask: np.ndarray) -> Tuple[float, int]:
        dd = d[mask]
        T = np.column_stack([np.ones(dd.size), dd, dd * dd])
        lam, *_ = np.linalg.lstsq(T, y[mask], rcond=None)
        return 2.0 * lam[2], int(mask.sum())

    m2p, n2p = curv(right & (d <= h2p))
    m2m, n2m = curv((~right) & (d >= -h2m))
    rp = 2160.0 * s2p / (n2p * h2p**4)
    rm = 2160.0 * s2m / (n2m * h2m**4)
    return float(
        _edge_kernel_constant()
        * ((s2p + s2m) / (f0 * ((m2p - m2m) ** 2 + rp + rm))) ** 0.2
        * N ** (-0.2)
    )


def _moulton_rho(e: np.ndarray, cl: np.ndarray) -> float:
    """Common within-cluster covariance of residuals ``e`` (Moulton):
    the average of ``e_i e_j`` over ordered pairs ``i != j`` in the same
    cluster."""
    _, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=e)
    ss = np.bincount(inv, weights=e * e)
    n = np.bincount(inv)
    pairs = float(np.sum(n * (n - 1)))
    if pairs <= 0:
        return 0.0
    return float(np.sum(s * s - ss) / pairs)


def _prelim_variance(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    kernel: str,
    cluster: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Preliminary variance for bandwidth choice.

    Residuals of a triangular-kernel local linear fit at the
    Imbens-Kalyanaraman bandwidth (floored so each side has three support
    points) are squared and averaged within each side; every unit on a side
    gets that side's value. The pilot fit is triangular whatever ``kernel``
    the estimator uses (``kernel`` is accepted for API symmetry only). With
    clusters, also returns the common within-cluster covariance (Moulton).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    h0 = max(_ik_bandwidth(x, y, c), _min_bandwidth(x, c))
    e, inwin = _ll_residuals(x, y, c, h0, "triangular")
    right = x >= c
    s2 = np.empty_like(e)
    for side in (right, ~right):
        m = side & inwin
        s2[side] = np.mean(e[m] ** 2)
    rho = 0.0
    if cluster is not None:
        rho = _moulton_rho(e[inwin], np.asarray(cluster)[inwin])
    return s2, rho


# --------------------------------------------------------------------------
# One fit, bandwidth choice, rule of thumb for M
# --------------------------------------------------------------------------


def _cluster_sd2(
    w: np.ndarray, s2: np.ndarray, rho: float, cl: Optional[np.ndarray]
) -> float:
    v = float(np.sum(w * w * s2))
    if cl is not None and rho != 0.0:
        _, inv = np.unique(cl, return_inverse=True)
        sw = np.bincount(inv, weights=w)
        v += rho * float(np.sum(sw * sw) - np.sum(w * w))
    return v


def _eff_obs(x: np.ndarray, c: float, h: float, w: np.ndarray) -> float:
    """Effective observations (vignette): units in the window times
    ``sum k_uniform^2 / sum w^2``."""
    wu = honest_weights(x, c, h, "uniform")
    inwin = wu != 0
    return float(inwin.sum() * np.sum(wu * wu) / np.sum(w * w))


def honest_fit(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    h: float,
    M: float,
    kernel: str = "triangular",
    alpha: float = 0.05,
    sclass: str = "H",
    J: int = 3,
    sigma2: Optional[np.ndarray] = None,
    cluster: Optional[np.ndarray] = None,
    rho: float = 0.0,
) -> Dict[str, float]:
    """One honest RD fit at a given bandwidth ``h`` and bound ``M``.

    The standard error uses ``sigma2`` when supplied, otherwise the
    nearest-neighbour variances; with ``cluster`` it is the cluster-robust
    sum over clusters of the weighted local linear residuals. Returns the
    estimate, standard error, worst-case bias, critical value, interval,
    one-sided limits, effective observations, maximal leverage and p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = honest_weights(x, c, h, kernel)
    est = float(w @ y)
    if cluster is not None:
        e, _ = _ll_residuals(x, y, c, h, kernel)
        _, inv = np.unique(np.asarray(cluster), return_inverse=True)
        sg = np.bincount(inv, weights=w * e)
        se = float(np.sqrt(np.sum(sg * sg)))
    else:
        if sigma2 is None:
            inwin = _kern((x - c) / h, kernel) > 0
            s2 = _nn_variance_by_side(x, y, c, J, inwin)
        else:
            s2 = sigma2
        se = float(np.sqrt(np.sum(w * w * s2)))
    bias = honest_bias(w, x, c, M, sclass)
    t = bias / se if se > 0 else 0.0
    cv = cv_bias(t, alpha)
    z1 = stats.norm.isf(alpha)
    tstat = abs(est) / se if se > 0 else np.inf
    return {
        "estimate": est,
        "se": se,
        "bias": bias,
        "cv": cv,
        "ci_lower": est - cv * se,
        "ci_upper": est + cv * se,
        "ci_lower_onesided": est - bias - z1 * se,
        "ci_upper_onesided": est + bias + z1 * se,
        "bandwidth": float(h),
        "M": float(M),
        "eff_obs": _eff_obs(x, c, h, w),
        "leverage": float(np.max(w * w) / np.sum(w * w)),
        "p_value": float(stats.norm.sf(tstat - t) + stats.norm.cdf(-tstat - t)),
    }


def _criterion_value(
    bias: float, sd: float, opt_criterion: str, alpha: float, beta: float
) -> float:
    if opt_criterion == "MSE":
        return bias * bias + sd * sd
    if opt_criterion == "FLCI":
        return cv_bias(bias / sd, alpha) * sd
    if opt_criterion == "OCI":
        # beta quantile of the worst-case excess length of the one-sided CI
        return 2.0 * bias + (stats.norm.isf(alpha) + stats.norm.ppf(beta)) * sd
    raise ValueError(
        f"opt_criterion must be 'MSE', 'FLCI' or 'OCI', got {opt_criterion!r}"
    )


def _optimal_h(objective, h_lo: float, h_hi: float) -> float:
    """Minimise ``objective`` over ``(h_lo, h_hi)`` with Brent's bounded
    golden-section/parabolic search (Forsythe, Malcolm and Moler's ``fmin``).

    The criteria can have several local minima of almost equal height; a
    single bracketed Brent search over the whole range is the convention the
    acceptance fixtures encode, so no grid pre-search is done.
    """
    if h_hi <= h_lo:
        return float(h_lo)
    res = optimize.minimize_scalar(
        objective,
        bounds=(h_lo, h_hi),
        method="bounded",
        options={"xatol": 1e-10 * h_hi, "maxiter": 2000},
    )
    return float(res.x)


def honest_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    M: float,
    kernel: str = "triangular",
    opt_criterion: str = "MSE",
    alpha: float = 0.05,
    beta: float = 0.8,
    sclass: str = "H",
    cluster: Optional[np.ndarray] = None,
) -> float:
    """Bandwidth minimising the chosen criterion under a preliminary
    variance that is homoskedastic on each side of the cutoff.

    ``opt_criterion``: ``'MSE'`` (worst-case mean squared error),
    ``'FLCI'`` (half-length of the fixed-length CI) or ``'OCI'`` (``beta``
    quantile of the worst-case excess length of a one-sided CI).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    s2, rho = _prelim_variance(x, y, c, kernel, cluster)
    cl = None if cluster is None else np.asarray(cluster)

    def objective(h: float) -> float:
        try:
            w = honest_weights(x, c, h, kernel)
        except ValueError:
            return np.inf
        sd = np.sqrt(_cluster_sd2(w, s2, rho, cl))
        return _criterion_value(
            honest_bias(w, x, c, M, sclass), sd, opt_criterion, alpha, beta
        )

    # Search range: from the second distinct distance to the cutoff (the
    # smallest bandwidth at which a line can be fitted on both sides) to the
    # largest distance.
    h_lo = _min_bandwidth(x, c, 2)
    h_hi = float(np.max(np.abs(x - c)))
    if kernel == "uniform":
        # The criterion is a step function of h that changes only where h
        # crosses a data point, so it is minimised over those points.
        grid = np.unique(np.abs(x - c))
        grid = grid[grid >= h_lo]
        if sclass == "H" and cl is None:
            vals = _uniform_holder_path(x - c, s2, grid, M, opt_criterion, alpha, beta)
        else:
            vals = np.array([objective(h) for h in grid])
        return float(grid[int(np.nanargmin(vals))])
    return _optimal_h(objective, h_lo, h_hi)


def _uniform_holder_path(
    d: np.ndarray,
    s2: np.ndarray,
    grid: np.ndarray,
    M: float,
    opt_criterion: str,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Criterion of the uniform-kernel estimator at each bandwidth in
    ``grid`` (Holder class, no clusters), from cumulative sums.

    On one side the intercept weights of an unweighted line through the
    window are ``a_i = (S2 - S1 d_i) / (S0 S2 - S1^2)`` with ``Sk = sum d^k``.
    Hence ``sum a_i d_i^2 = (S2^2 - S1 S3) / det`` and
    ``sum a_i^2 s2_i = (S2^2 T0 - 2 S1 S2 T1 + S1^2 T2) / det^2`` with
    ``Tk = sum d^k s2``; both are prefix sums over units sorted by ``|d|``.
    """
    bias_sum = np.zeros(grid.size)
    var = np.zeros(grid.size)
    ok = np.ones(grid.size, dtype=bool)
    for side in (d >= 0, d < 0):
        ds, vs = d[side], s2[side]
        order = np.argsort(np.abs(ds), kind="mergesort")
        ds, vs = ds[order], vs[order]
        cs = [np.concatenate([[0.0], np.cumsum(ds**k)]) for k in range(4)]
        ct = [np.concatenate([[0.0], np.cumsum(ds**k * vs)]) for k in range(3)]
        idx = np.searchsorted(np.abs(ds), grid, side="right")
        S0, S1, S2, S3 = (c_[idx] for c_ in cs)
        T0, T1, T2 = (c_[idx] for c_ in ct)
        det = S0 * S2 - S1 * S1
        good = det > 1e-12 * np.maximum(S0 * S2, 1e-300)
        ok &= good
        det = np.where(good, det, 1.0)
        bias_sum += (S2 * S2 - S1 * S3) / det
        var += (S2 * S2 * T0 - 2 * S1 * S2 * T1 + S1 * S1 * T2) / det**2
    out = np.full(grid.size, np.inf)
    for i in np.flatnonzero(ok):
        out[i] = _criterion_value(
            0.5 * M * abs(bias_sum[i]),
            float(np.sqrt(var[i])),
            opt_criterion,
            alpha,
            beta,
        )
    return out


def m_rule_of_thumb(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """Rule-of-thumb smoothness bound (``armstrong2020simple``).

    Fits a global quartic in ``x - c`` separately on each side of the
    cutoff and returns the largest absolute second derivative of the two
    fits over the range of the data on its side.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - c
    best = 0.0
    for side in (d >= 0, d < 0):
        ds = d[side]
        if np.unique(ds).size < 5:
            raise ValueError(
                "the rule of thumb for M fits a quartic on each side of the "
                "cutoff and needs at least five distinct values per side; "
                "supply M"
            )
        X = np.column_stack([ds**p for p in range(5)])
        b, *_ = np.linalg.lstsq(X, y[side], rcond=None)
        # f''(d) = 2 b2 + 6 b3 d + 12 b4 d^2: check the ends and the vertex
        pts = [ds.min(), ds.max()]
        if b[4] != 0:
            v = -b[3] / (4.0 * b[4])
            if ds.min() < v < ds.max():
                pts.append(v)
        f2 = [abs(2 * b[2] + 6 * b[3] * p + 12 * b[4] * p * p) for p in pts]
        best = max(best, max(f2))
    return float(best)


def honest_rd(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 0.0,
    M: Optional[float] = None,
    h: Optional[float] = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
    opt_criterion: str = "MSE",
    sclass: str = "H",
    J: int = 3,
    cluster: Optional[np.ndarray] = None,
    beta: float = 0.8,
) -> Tuple[Dict[str, float], bool]:
    """Full honest RD: rule-of-thumb ``M`` and optimal ``h`` when not given.

    Returns ``(fit, M_estimated)`` with ``fit`` as in :func:`honest_fit`.
    """
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}, got {kernel!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    M_estimated = M is None
    if M is None:
        M = m_rule_of_thumb(x, y, c)
    if h is None:
        h = honest_bandwidth(
            x, y, c, M, kernel, opt_criterion.upper(), alpha, beta, sclass, cluster
        )
    fit = honest_fit(x, y, c, h, M, kernel, alpha, sclass, J, cluster=cluster)
    return fit, M_estimated
