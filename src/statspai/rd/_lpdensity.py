"""Local-polynomial density estimators of Cattaneo, Jansson and Ma.

Shared numerical core of :func:`sp.rddensity` (manipulation test) and
:func:`sp.rdplotdensity` (density curves on a grid). The estimator fits a
local polynomial to the empirical distribution function and reads the
density off the slope coefficient (cattaneo2020simple); the manipulation
test compares the two one-sided boundary estimates at the cutoff
(cattaneo2018manipulation).

Clean-room rewrite: this module was written from the papers above, the
published help pages, and black-box runs of the reference implementations
(R ``rddensity`` 2.6 and R ``lpdensity`` 2.5, both GPL-2). No reference
source code was read. The finite-sample conventions that the papers leave
open (which empirical CDF, which variance sum, the pilot bandwidths) were
determined by probing the reference outputs; they are recorded in
``docs/dev/cleanroom/rddensity.md`` and ``docs/dev/cleanroom/lpdensity.md``.

References
----------
cattaneo2020simple
cattaneo2018manipulation
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..exceptions import MethodIncompatibility

__all__ = [
    "lpdensity_numbers",
    "boundary_fit",
    "boundary_bias_constant",
    "normal_reference_bandwidth",
    "kth_distance",
]

_KERNELS = ("triangular", "uniform", "epanechnikov")


# --------------------------------------------------------------------- #
#  Kernels and empirical CDF
# --------------------------------------------------------------------- #


def kernel_weights(u: np.ndarray, kernel: str = "triangular") -> np.ndarray:
    """Kernel ``K(u)`` on ``[-1, 1]`` (zero outside)."""
    a = np.abs(u)
    inside = a <= 1.0
    if kernel == "triangular":
        k = 1.0 - a
    elif kernel == "uniform":
        k = np.full_like(a, 0.5)
    elif kernel == "epanechnikov":
        k = 0.75 * (1.0 - a * a)
    else:  # pragma: no cover - callers validate first
        raise MethodIncompatibility(
            f"Unknown kernel {kernel!r}.",
            recovery_hint="Use 'triangular', 'uniform' or 'epanechnikov'.",
        )
    return np.where(inside, k, 0.0)


def _cdf_counts(x_sorted: np.ndarray, mass_points: bool) -> np.ndarray:
    """Number of observations at or below each sorted observation.

    With ``mass_points`` tied values share the count of their whole tie
    group (the empirical CDF evaluated at the value); without it each
    observation gets its position in the sorted sample, so ties are
    broken by order.
    """
    if mass_points:
        return np.searchsorted(x_sorted, x_sorted, side="right").astype(float)
    return np.arange(1, len(x_sorted) + 1, dtype=float)


def _suffix_sums(
    rows: np.ndarray, order_keys: np.ndarray, query_keys: np.ndarray
) -> np.ndarray:
    """``sum_{j: key_j >= q} rows_j`` for every query key ``q``.

    ``order_keys`` must be sorted ascending and aligned with ``rows``.
    """
    total = np.vstack([np.cumsum(rows[::-1], axis=0)[::-1], np.zeros(rows.shape[1])])
    start = np.searchsorted(order_keys, query_keys, side="left")
    return total[start]


def _solve(A: np.ndarray, B: np.ndarray) -> Optional[np.ndarray]:
    try:
        out = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None
    return out if np.all(np.isfinite(out)) else None


# --------------------------------------------------------------------- #
#  Grid estimator (lpdensity)
# --------------------------------------------------------------------- #


def _grid_point(
    x: np.ndarray,
    keys: np.ndarray,
    cdf: np.ndarray,
    g: float,
    h: float,
    order: int,
    v: int,
    kernel: str,
) -> Tuple[float, float]:
    """Density-type estimate ``F^(v)(g)`` and its standard error.

    ``x`` sorted sample, ``keys`` its comparison keys (values with mass
    points, positions without), ``cdf`` the empirical CDF at each point.
    The variance is the sandwich ``A^-1 (sum_i psi_i psi_i') A^-1`` with
    ``psi_i = (1/n) sum_{j in window} r_j K_j (1[x_i <= x_j] - F(x_j))``
    summed over the whole sample, the plug-in form of the theorem on
    variance estimation in cattaneo2020simple.
    """
    n = len(x)
    u = (x - g) / h
    win = np.abs(u) <= 1.0
    uw = u[win]
    kw = kernel_weights(uw, kernel) / h
    if np.unique(uw[kw > 0]).size < order + 1:  # not identified
        return np.nan, np.nan
    R = np.vander(uw, order + 1, increasing=True)
    RK = R * kw[:, None]
    A = RK.T @ R
    beta = _solve(A, RK.T @ cdf[win])
    if beta is None:
        return np.nan, np.nan
    scale = factorial(v) / h**v
    est = scale * beta[v]
    # psi_i over the full sample: suffix sums of r_j K_j over the window
    keys_w = keys[win]
    suffix = _suffix_sums(RK, keys_w, keys) / n
    centre = (RK * cdf[win][:, None]).sum(axis=0) / n
    psi = suffix - centre
    lever = _solve(A, psi.T)  # A^-1 psi_i for every i
    if lever is None:
        return float(est), np.nan
    se = scale * np.sqrt(np.sum(lever[v] ** 2))
    return float(est), float(se)


def lpdensity_numbers(
    data: np.ndarray,
    grid: np.ndarray,
    bw: Any,
    p: int = 2,
    q: Optional[int] = None,
    v: int = 1,
    kernel: str = "triangular",
    scale: float = 1.0,
    mass_points: bool = True,
) -> Dict[str, np.ndarray]:
    """Local-polynomial density estimates on a grid (cattaneo2020simple).

    Fits a polynomial of order ``p`` (and ``q`` for the bias-corrected
    band) to the empirical CDF around each grid point and returns the
    ``v``-th derivative of the fit (``v=1``: the density).

    Parameters
    ----------
    data : array-like
        Sample (non-finite values are dropped).
    grid : array-like
        Evaluation points.
    bw : float or array-like
        One bandwidth, or one per grid point.
    p : int, default 2
        Polynomial order of the point estimate ``f_p``.
    q : int, optional
        Order of the bias-corrected estimate ``f_q``; default ``p + 1``.
        ``q <= p`` leaves ``f_q`` / ``se_q`` as NaN.
    v : int, default 1
        Derivative of the CDF (0: CDF, 1: density, ...), ``v <= p``.
    kernel : {'triangular', 'uniform', 'epanechnikov'}
    scale : float, default 1.0
        Multiplies estimates and standard errors (e.g. the share of the
        full sample that ``data`` represents).
    mass_points : bool, default True
        Treat tied values as one mass point of the empirical CDF; without
        it ties are broken by sort order.

    Returns
    -------
    dict
        ``grid``, ``bw``, ``nh`` (observations with ``|x - g| <= bw``),
        ``f_p``, ``f_q``, ``se_p``, ``se_q``. Points whose window holds
        fewer distinct values with positive kernel weight than parameters
        are NaN (the fit is not identified).

    References
    ----------
    cattaneo2020simple
    """
    x = np.sort(np.asarray(data, dtype=float).ravel(), kind="stable")
    x = x[np.isfinite(x)]
    grid = np.atleast_1d(np.asarray(grid, dtype=float)).ravel()
    if q is None:
        q = p + 1
    if not isinstance(p, (int, np.integer)) or p < 0:
        raise MethodIncompatibility(
            f"p must be a non-negative integer; got {p!r}.",
            recovery_hint="Use p=2 (the default local quadratic fit).",
        )
    if not 0 <= v <= p:
        raise MethodIncompatibility(
            f"v must be between 0 and p ({p}); got {v!r}.",
            recovery_hint="Use v=1 for the density with p >= 1.",
        )
    if kernel not in _KERNELS:
        raise MethodIncompatibility(
            f"Unknown kernel {kernel!r}.",
            recovery_hint="Use 'triangular', 'uniform' or 'epanechnikov'.",
        )
    bws = np.asarray(bw, dtype=float).ravel()
    if bws.size == 1:
        bws = np.full(grid.shape, float(bws[0]))
    if bws.shape != grid.shape:
        raise MethodIncompatibility(
            "bw must be a scalar or have one value per grid point.",
            recovery_hint="Pass a single bandwidth or len(grid) bandwidths.",
            diagnostics={"len_bw": int(bws.size), "len_grid": int(grid.size)},
        )
    if not np.all(np.isfinite(bws)) or np.any(bws <= 0):
        raise MethodIncompatibility(
            "bw must be positive and finite.",
            recovery_hint="Pass positive bandwidths.",
        )

    n = len(x)
    cdf = _cdf_counts(x, mass_points) / max(n, 1)
    keys = x if mass_points else np.arange(n, dtype=float)
    out = {k: np.full(grid.shape, np.nan) for k in ("f_p", "f_q", "se_p", "se_q")}
    nh = np.zeros(grid.shape, dtype=int)
    for i, (g, h) in enumerate(zip(grid, bws)):
        nh[i] = int(np.sum(np.abs(x - g) <= h))
        if n == 0:
            continue
        out["f_p"][i], out["se_p"][i] = _grid_point(x, keys, cdf, g, h, p, v, kernel)
        if q > p:
            out["f_q"][i], out["se_q"][i] = _grid_point(
                x, keys, cdf, g, h, q, v, kernel
            )
    for k in out:
        out[k] = out[k] * scale
    return {"grid": grid, "bw": bws, "nh": nh, **out}


# --------------------------------------------------------------------- #
#  One-sided boundary fit at the cutoff (rddensity)
# --------------------------------------------------------------------- #


def boundary_fit(
    x_sorted: np.ndarray,
    counts: np.ndarray,
    keys: np.ndarray,
    c: float,
    h: float,
    side: str,
    order: int,
    v: int = 1,
    with_se: bool = True,
) -> Tuple[float, float, int]:
    """One-sided estimate of ``F^(v)`` at the cutoff and its jackknife SE.

    The regressand is the leave-one-out empirical CDF of the full sample,
    ``counts / (n - 1)`` with ``counts`` the number of *other*
    observations at or below each point. Only the observations of
    ``side`` within ``h`` of ``c`` enter the fit (triangular kernel).

    The jackknife variance (cattaneo2018manipulation, section 2.4) is
    ``A^-1 (sum_i psi_i psi_i') A^-1 / (n - 1)^2`` with the sum over the
    window observations and ``psi_i = sum_{j in window, j != i} r_j K_j
    1[x_i <= x_j]``: the change in the normal-equation right-hand side
    when observation ``i`` is deleted from the empirical CDF. (The form
    was pinned by black-box probing; see the clean-room spec.)

    Returns ``(estimate, se, n_window)``; NaN when the window holds fewer
    distinct values with positive kernel weight than parameters.
    """
    n = len(x_sorted)
    on_side = (x_sorted < c) if side == "left" else (x_sorted >= c)
    u = (x_sorted - c) / h
    win = on_side & (np.abs(u) <= 1.0)
    n_win = int(win.sum())
    uw = u[win]
    kw = kernel_weights(uw, "triangular") / h
    if np.unique(uw[kw > 0]).size < order + 1 or n < 2:  # not identified
        return np.nan, np.nan, n_win
    R = np.vander(uw, order + 1, increasing=True)
    RK = R * kw[:, None]
    A = RK.T @ R
    beta = _solve(A, RK.T @ counts[win])
    if beta is None:
        return np.nan, np.nan, n_win
    scale = factorial(v) / h**v / (n - 1)
    est = float(scale * beta[v])
    if not with_se:
        return est, np.nan, n_win
    kw_keys = keys[win]
    psi = _suffix_sums(RK, kw_keys, kw_keys) - RK  # exclude j == i
    lever = _solve(A, psi.T)
    if lever is None:
        return est, np.nan, n_win
    se = float(scale * np.sqrt(np.sum(lever[v] ** 2)))
    return est, se, n_win


# --------------------------------------------------------------------- #
#  Kernel constants and pilot bandwidths
# --------------------------------------------------------------------- #


def _exact_solve(M, rhs):
    """Solve ``M y = rhs`` in exact rational arithmetic (small systems)."""
    size = len(M)
    aug = [list(M[i]) + [rhs[i]] for i in range(size)]
    for col in range(size):
        piv = next(r for r in range(col, size) if aug[r][col] != 0)
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(size):
            if r != col and aug[r][col] != 0:
                f = aug[r][col] / aug[col][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [aug[i][size] / aug[i][i] for i in range(size)]


def boundary_bias_constant(p: int, side: str) -> float:
    """``e_1' S^-1 c`` for the triangular kernel on one side of a boundary.

    ``S = int r(u) r(u)' K(u) du`` and ``c = int u^(p+1) r(u) K(u) du``
    over ``[0, 1]`` (right) or ``[-1, 0]`` (left): the leading-bias
    constant of the order-``p`` density estimator at a boundary point
    (cattaneo2020simple, Theorem 1). Computed exactly.
    """
    sign = 1 if side == "right" else -1

    def moment(k: int) -> Fraction:  # int_side u^k (1 - |u|) du
        return Fraction(sign**k, (k + 1) * (k + 2))

    S = [[moment(a + b) for b in range(p + 1)] for a in range(p + 1)]
    cvec = [moment(a + p + 1) for a in range(p + 1)]
    return float(_exact_solve(S, cvec)[1])


def _uniform_boundary_constants(P: int, v: int) -> Tuple[float, float]:
    """Variance and bias constants of the order-``P`` estimator of
    ``F^(v)`` at a boundary point with the uniform kernel on ``[0, 1]``.

    ``V = (v!)^2 [S^-1 G S^-1]_vv`` and ``B = v! [S^-1 c]_v / (P+1)!``
    with ``S_ab = 1/(a+b+1)``, ``c_a = 1/(a+P+2)`` and
    ``G_ab = int int min(s, t) s^a t^b`` (the kernel's 1/2 cancels).
    Computed exactly (``S`` is a Hilbert matrix).
    """
    m = P + 1
    S = [[Fraction(1, a + b + 1) for b in range(m)] for a in range(m)]
    cvec = [Fraction(1, a + P + 2) for a in range(m)]
    G = [
        [
            Fraction(1, (a + 1) * (b + 2))
            - Fraction(1, (a + 1) * (a + 2) * (a + b + 3))
            for b in range(m)
        ]
        for a in range(m)
    ]
    unit = [Fraction(int(i == v)) for i in range(m)]
    w = _exact_solve(S, unit)  # S^-1 e_v (S symmetric)
    var = sum(w[a] * G[a][b] * w[b] for a in range(m) for b in range(m))
    bias = _exact_solve(S, cvec)[v]
    V = factorial(v) ** 2 * var
    B = factorial(v) * bias / factorial(P + 1)
    return float(V), float(B)


def _normal_derivative(k: int, z: float, sd: float) -> float:
    """``k``-th derivative of the N(mean, sd^2) density at ``mean + z sd``.

    Uses the probabilists' Hermite recursion ``He_{k+1} = z He_k - k He_{k-1}``.
    """
    he_prev, he = 1.0, z
    if k == 0:
        he = 1.0
    else:
        for j in range(1, k):
            he_prev, he = he, z * he - j * he_prev
    dens = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return float((-1) ** k * he * dens / sd ** (k + 1))


def normal_reference_bandwidth(x: np.ndarray, c: float, P: int, v: int) -> float:
    """Normal-reference pilot bandwidth for estimating ``F^(v)(c)``.

    MSE-optimal bandwidth of the order-``P`` boundary estimator when the
    data are N(mean, sd^2) (sample mean, ``ddof=1`` sd), with the uniform
    kernel's constants:

        h^(2P+1) = (2v - 1) V f(c) / (2 (P + 1 - v) B^2 F^(P+1)(c)^2 n).

    Returns ``inf`` when the reference derivative vanishes at ``c``.
    """
    n = len(x)
    sd = float(np.std(x, ddof=1))
    z = (c - float(np.mean(x))) / sd
    V, B = _uniform_boundary_constants(P, v)
    f0 = _normal_derivative(0, z, sd)
    fP = _normal_derivative(P, z, sd)  # = F^(P+1)(c)
    denom = 2.0 * (P + 1 - v) * B * B * fP * fP * n
    if denom == 0:
        return float("inf")
    return float(((2 * v - 1) * V * f0 / denom) ** (1.0 / (2 * P + 1)))


def kth_distance(dist_sorted: np.ndarray, k: int) -> float:
    """``k``-th smallest distance (1-based); the largest if fewer exist."""
    if dist_sorted.size == 0:
        return np.nan
    return float(dist_sorted[min(k, dist_sorted.size) - 1])
