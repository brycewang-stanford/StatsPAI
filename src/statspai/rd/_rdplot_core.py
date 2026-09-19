"""Numerical core of :func:`statspai.rdplot`, ported from R ``rdrobust::rdplot``.

Everything here reproduces R ``rdrobust`` 4.0.0's ``rdplot`` line for line:
the number of bins (``J.fun`` and the mimicking-variance rule for the eight
``binselect`` choices, with the mass-point switch to the polynomial-
regression variants), the bin edges (``seq`` / type-7 ``quantile``), bin
membership (``findInterval(..., rightmost.closed = TRUE)``), the per-bin
means / standard errors / t-based intervals, and the kernel-weighted
global polynomial on each side (with R's covariate adjustment when
``covs`` is given). Plotting lives in ``rdrobust.rdplot``; this module only
computes the numbers R returns in ``vars_bins``, ``vars_poly``, ``coef``,
``J``, ``J_IMSE`` and ``J_MV``.

One deliberate difference: R indexes the left-side ``rdplot_min_bin`` /
``rdplot_max_bin`` by ``rev(-bin)``, which is the right bin only when no
left-side bin is empty. Here each non-empty bin gets its own edges.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats

_BINSELECT = ("es", "espr", "esmv", "esmvpr", "qs", "qspr", "qsmv", "qsmvpr")


def _r_seq(start: float, stop: float, by: float) -> np.ndarray:
    """R ``seq(from, to, by)`` for ``by > 0``, including its end capping."""
    n = int((stop - start) / by + 1e-10)
    return np.minimum(start + np.arange(n + 1) * by, stop)


def _kweight(x: np.ndarray, c: float, h: float, kernel: str) -> np.ndarray:
    """R ``rdrobust_kweight``: kernel weights scaled by 1/h."""
    u = (x - c) / h
    inside = np.abs(u) <= 1
    if kernel in ("epanechnikov", "epa"):
        return 0.75 * (1 - u**2) * inside / h
    if kernel in ("uniform", "uni"):
        return 0.5 * inside / h
    return (1 - np.abs(u)) * inside / h


def _xx_inv(X: np.ndarray) -> Optional[np.ndarray]:
    """R ``qrXXinv``: ``chol2inv(chol(X'X))``; ``None`` if not positive definite."""
    G = X.T @ X
    try:
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        return None
    Linv = np.linalg.inv(L)
    return Linv.T @ Linv


def _r_quantile7(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """R ``quantile(x, probs)`` (type 7), including its rounding path.

    NumPy's ``'linear'`` method is the same estimator but interpolates with
    a different floating-point expression; when ``1 + (n-1) p`` lands a hair
    away from an integer, the two edges differ in the last bit and an
    observation sitting on the edge changes bins.
    """
    xs = np.sort(x)
    n = len(xs)
    index = 1 + max(n - 1, 0) * np.asarray(probs, dtype=float)
    lo = np.floor(index).astype(int)
    hi = np.ceil(index).astype(int)
    qs = xs[lo - 1].copy()
    xhi = xs[hi - 1]
    i = (index > lo) & (xhi != qs)
    h = (index - lo)[i]
    qs[i] = (1 - h) * qs[i] + h * xhi[i]
    return qs


def _find_interval(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """R ``findInterval(x, edges, rightmost.closed = TRUE)``."""
    idx = np.searchsorted(edges, x, side="right")
    idx[x == edges[-1]] = len(edges) - 1
    return idx


def rdplot_numbers(
    y: np.ndarray,
    x: np.ndarray,
    c: float = 0.0,
    p: int = 4,
    nbins: Optional[Any] = None,
    binselect: str = "esmv",
    scale: Optional[Any] = None,
    kernel: str = "uni",
    h: Optional[Any] = None,
    weights: Optional[np.ndarray] = None,
    covs: Optional[np.ndarray] = None,
    ci: float = 95.0,
    masspoints: str = "adjust",
) -> Dict[str, Any]:
    """Return R ``rdplot``'s numerical output (see module docstring)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if covs is not None:
        covs = np.asarray(covs, dtype=float)
        if covs.ndim == 1:
            covs = covs.reshape(-1, 1)
        ok &= np.all(np.isfinite(covs), axis=1)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        ok &= np.isfinite(weights) & (weights >= 0)
    x, y = x[ok], y[ok]
    if covs is not None:
        covs = covs[ok]
    if weights is not None:
        weights = weights[ok]

    if binselect not in _BINSELECT:
        raise ValueError(f"binselect must be one of {_BINSELECT}; got {binselect!r}")
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if c <= x_min or c >= x_max:
        raise ValueError("c should be set within the range of x")
    if len(x) < 20:
        raise ValueError("Not enough observations to perform bin calculations")

    ind_l = x < c
    ind_r = ~ind_l
    x_l, x_r = x[ind_l], x[ind_r]
    y_l, y_r = y[ind_l], y[ind_r]
    range_l, range_r = c - x_min, x_max - c
    n_l, n_r = len(x_l), len(x_r)
    n = n_l + n_r

    if scale is None:
        scale_l = scale_r = 1.0
    elif np.ndim(scale) == 0:
        scale_l = scale_r = float(scale)
    else:
        scale_l, scale_r = float(scale[0]), float(scale[1])
    if h is None:
        h_l, h_r = range_l, range_r
    elif np.ndim(h) == 0:
        h_l = h_r = float(h)
    else:
        h_l, h_r = float(h[0]), float(h[1])

    mass_adjusted = False
    if masspoints in ("check", "adjust"):
        mass_l = 1 - len(np.unique(x_l)) / n_l
        mass_r = 1 - len(np.unique(x_r)) / n_r
        if (mass_l >= 0.2 or mass_r >= 0.2) and masspoints == "adjust":
            mass_adjusted = True
            binselect = {
                "es": "espr",
                "esmv": "esmvpr",
                "qs": "qspr",
                "qsmv": "qsmvpr",
            }.get(binselect, binselect)

    # ---- kernel-weighted global polynomial on each side --------------------
    W_l = _kweight(x_l, c, h_l, kernel)
    W_r = _kweight(x_r, c, h_r, kernel)
    n_h_l, n_h_r = int(np.sum(W_l > 0)), int(np.sum(W_r > 0))
    if weights is not None:
        W_l = weights[ind_l] * W_l
        W_r = weights[ind_r] * W_r
    R_l = np.vander(x_l - c, p + 1, increasing=True)
    R_r = np.vander(x_r - c, p + 1, increasing=True)
    iG_l = _xx_inv(np.sqrt(W_l)[:, None] * R_l)
    iG_r = _xx_inv(np.sqrt(W_r)[:, None] * R_r)
    if iG_l is None:
        iG_l = np.linalg.pinv(R_l.T @ (W_l[:, None] * R_l))
    if iG_r is None:
        iG_r = np.linalg.pinv(R_r.T @ (W_r[:, None] * R_r))
    gamma_p = None
    if covs is None:
        g_l = iG_l @ (R_l * W_l[:, None]).T @ y_l
        g_r = iG_r @ (R_r * W_r[:, None]).T @ y_r
    else:
        z_l, z_r = covs[ind_l], covs[ind_r]
        D_l = np.column_stack([y_l, z_l])
        D_r = np.column_stack([y_r, z_r])
        U_l = (R_l * W_l[:, None]).T @ D_l
        U_r = (R_r * W_r[:, None]).T @ D_r
        B_l = iG_l @ U_l
        B_r = iG_r @ U_r
        ZWD_l = (z_l * W_l[:, None]).T @ D_l
        ZWD_r = (z_r * W_r[:, None]).T @ D_r
        UiGU_l = U_l[:, 1:].T @ iG_l @ U_l
        UiGU_r = U_r[:, 1:].T @ iG_r @ U_r
        ZWZ = (ZWD_l[:, 1:] - UiGU_l[:, 1:]) + (ZWD_r[:, 1:] - UiGU_r[:, 1:])
        ZWY = (ZWD_l[:, 0] - UiGU_l[:, 0]) + (ZWD_r[:, 0] - UiGU_r[:, 0])
        gamma_p = np.linalg.pinv(ZWZ, rcond=1e-20, hermitian=True) @ ZWY
        s_Y = np.concatenate([[1.0], -gamma_p])
        g_l = B_l @ s_Y
        g_r = B_r @ s_Y

    nplot = 500
    x_plot_l = np.linspace(c - h_l, c, nplot)
    x_plot_r = np.linspace(c, c + h_r, nplot)
    y_hat_l = np.vander(x_plot_l - c, p + 1, increasing=True) @ g_l
    y_hat_r = np.vander(x_plot_r - c, p + 1, increasing=True) @ g_r
    if covs is not None:
        gz = float(np.mean(covs, axis=0) @ gamma_p)
        y_hat_l = y_hat_l + gz
        y_hat_r = y_hat_r + gz

    # ---- bin selection: global polynomial of order k on raw x ---------------
    for k in (4, 3, 2):
        rk_l = np.vander(x_l, k + 1, increasing=True)
        rk_r = np.vander(x_r, k + 1, increasing=True)
        iGk_l, iGk_r = _xx_inv(rk_l), _xx_inv(rk_r)
        if iGk_l is not None and iGk_r is not None:
            break
    if iGk_l is None or iGk_r is None:  # R falls back to ginv at k = 2
        iGk_l = np.linalg.pinv(rk_l.T @ rk_l) if iGk_l is None else iGk_l
        iGk_r = np.linalg.pinv(rk_r.T @ rk_r) if iGk_r is None else iGk_r
    gk1_l, gk2_l = iGk_l @ rk_l.T @ y_l, iGk_l @ rk_l.T @ (y_l**2)
    gk1_r, gk2_r = iGk_r @ rk_r.T @ y_r, iGk_r @ rk_r.T @ (y_r**2)

    def _deriv_design(v: np.ndarray) -> np.ndarray:
        return np.column_stack([j * v ** (j - 1) for j in range(1, k + 1)])

    o_l, o_r = np.argsort(x_l, kind="mergesort"), np.argsort(x_r, kind="mergesort")
    xi_l, yi_l, xi_r, yi_r = x_l[o_l], y_l[o_l], x_r[o_r], y_r[o_r]
    dxi_l, dyi_l = np.diff(xi_l), np.diff(yi_l)
    dxi_r, dyi_r = np.diff(xi_r), np.diff(yi_r)
    xb_l = (xi_l[1:] + xi_l[:-1]) / 2
    xb_r = (xi_r[1:] + xi_r[:-1]) / 2
    rki_l = np.vander(xb_l, k + 1, increasing=True)
    rki_r = np.vander(xb_r, k + 1, increasing=True)

    mu0i_l, mu0i_r = rki_l @ gk1_l, rki_r @ gk1_r
    mu2i_l, mu2i_r = rki_l @ gk2_l, rki_r @ gk2_r
    mu0_l, mu0_r = rk_l @ gk1_l, rk_r @ gk1_r
    mu2_l, mu2_r = rk_l @ gk2_l, rk_r @ gk2_r
    mu1_l = _deriv_design(x_l) @ gk1_l[1:]
    mu1_r = _deriv_design(x_r) @ gk1_r[1:]
    mu1i_l = _deriv_design(xb_l) @ gk1_l[1:]
    mu1i_r = _deriv_design(xb_r) @ gk1_r[1:]

    var_y_l = float(np.var(y_l, ddof=1))
    var_y_r = float(np.var(y_r, ddof=1))
    s2b_l = mu2i_l - mu0i_l**2
    s2b_r = mu2i_r - mu0i_r**2
    s2b_l[s2b_l < 0] = var_y_l
    s2b_r[s2b_r < 0] = var_y_r
    s2_l = mu2_l - mu0_l**2
    s2_r = mu2_r - mu0_r**2
    s2_l[s2_l < 0] = var_y_l
    s2_r[s2_r < 0] = var_y_r

    def J_fun(B: np.ndarray, V: np.ndarray) -> np.ndarray:
        return np.ceil(((2 * B / V) * n) ** (1 / 3))

    B_es = np.array(
        [
            ((c - x_min) ** 2 / (12 * n)) * np.sum(mu1_l**2),
            ((x_max - c) ** 2 / (12 * n)) * np.sum(mu1_r**2),
        ]
    )
    V_es_hat = np.array(
        [
            (0.5 / (c - x_min)) * np.sum(dxi_l * dyi_l**2),
            (0.5 / (x_max - c)) * np.sum(dxi_r * dyi_r**2),
        ]
    )
    V_es_chk = np.array(
        [
            (1 / (c - x_min)) * np.sum(dxi_l * s2b_l),
            (1 / (x_max - c)) * np.sum(dxi_r * s2b_r),
        ]
    )
    B_qs = np.array(
        [
            (n_l**2 / (24 * n)) * np.sum(dxi_l**2 * mu1i_l**2),
            (n_r**2 / (24 * n)) * np.sum(dxi_r**2 * mu1i_r**2),
        ]
    )
    V_qs_hat = np.array(
        [
            (1 / (2 * n_l)) * np.sum(dyi_l**2),
            (1 / (2 * n_r)) * np.sum(dyi_r**2),
        ]
    )
    V_qs_chk = np.array([np.sum(s2_l) / n_l, np.sum(s2_r) / n_r])
    var_y = np.array([var_y_l, var_y_r])
    mv = n / np.log(n) ** 2

    J = {
        "es": (J_fun(B_es, V_es_hat), np.ceil(var_y / V_es_hat * mv)),
        "espr": (J_fun(B_es, V_es_chk), np.ceil(var_y / V_es_chk * mv)),
        "qs": (J_fun(B_qs, V_qs_hat), np.ceil(var_y / V_qs_hat * mv)),
        "qspr": (J_fun(B_qs, V_qs_chk), np.ceil(var_y / V_qs_chk * mv)),
    }
    base = binselect.replace("mv", "")
    J_IMSE, J_MV = J[base]
    J_orig = J_MV if "mv" in binselect else J_IMSE
    meth = binselect[:2]

    if nbins is not None:
        # R keeps the spacing of `binselect` (its label says "manually
        # evenly spaced" regardless).
        if np.ndim(nbins) == 0:
            J_l = J_r = int(nbins)
        else:
            J_l, J_r = int(nbins[0]), int(nbins[1])
    else:
        J_l = int(scale_l * J_orig[0])
        J_r = int(scale_r * J_orig[1])
    if var_y_l == 0:
        J_l = 1
    if var_y_r == 0:
        J_r = 1

    if meth == "es":
        jumps_l = _r_seq(x_min, c, range_l / J_l)
        jumps_r = _r_seq(c, x_max, range_r / J_r)
    else:
        jumps_l = _r_quantile7(x_l, _r_seq(0.0, 1.0, 1.0 / J_l))
        jumps_r = _r_quantile7(x_r, _r_seq(0.0, 1.0, 1.0 / J_r))

    bin_l = _find_interval(x_l, jumps_l) - J_l - 1
    bin_r = _find_interval(x_r, jumps_r)

    def _per_bin(bins: np.ndarray, xs: np.ndarray, ys: np.ndarray):
        ub = np.unique(bins)
        mx = np.array([xs[bins == b].mean() for b in ub])
        my = np.array([ys[bins == b].mean() for b in ub])
        nb = np.array([int(np.sum(bins == b)) for b in ub])
        sd = np.array(
            [ys[bins == b].std(ddof=1) if np.sum(bins == b) > 1 else 0.0 for b in ub]
        )
        return ub, mx, my, nb, sd

    ub_l, mx_l, my_l, nb_l, sd_l = _per_bin(bin_l, x_l, y_l)
    ub_r, mx_r, my_r, nb_r, sd_r = _per_bin(bin_r, x_r, y_r)

    if covs is not None:
        # R: lm(y ~ z + factor(bin)) per side, then bin means of the fit.
        def _cov_means(ys, zs, bins, ub):
            F = (bins[:, None] == ub[None, :]).astype(float)
            Xd = np.column_stack([F, zs])
            beta = np.linalg.lstsq(Xd, ys, rcond=None)[0]
            fit = Xd @ beta
            return np.array([fit[bins == b].mean() for b in ub])

        my_l = _cov_means(y_l, covs[ind_l], bin_l, ub_l)
        my_r = _cov_means(y_r, covs[ind_r], bin_r, ub_r)

    def _edges(jumps: np.ndarray, J: int):
        lo = np.full(J, np.nan)
        hi = np.full(J, np.nan)
        m = min(J, len(jumps) - 1)
        lo[:m] = jumps[:m]
        hi[:m] = jumps[1 : m + 1]
        return lo, hi

    lo_l, hi_l = _edges(jumps_l, J_l)
    lo_r, hi_r = _edges(jumps_r, J_r)
    il = ub_l + J_l  # 0-based index of each non-empty left bin
    ir = ub_r - 1
    mean_bin = np.concatenate([(lo_l[il] + hi_l[il]) / 2, (lo_r[ir] + hi_r[ir]) / 2])
    N = np.concatenate([nb_l, nb_r])
    sd = np.concatenate([sd_l, sd_r])
    se = sd / np.sqrt(N)
    quant = -stats.t.ppf((1 - ci / 100) / 2, np.maximum(N - 1, 1))
    mean_y = np.concatenate([my_l, my_r])

    return {
        "coef": np.column_stack([g_l, g_r]),
        "coef_covs": gamma_p,
        "vars_bins": {
            "rdplot_mean_bin": mean_bin,
            "rdplot_mean_x": np.concatenate([mx_l, mx_r]),
            "rdplot_mean_y": mean_y,
            "rdplot_min_bin": np.concatenate([lo_l[il], lo_r[ir]]),
            "rdplot_max_bin": np.concatenate([hi_l[il], hi_r[ir]]),
            "rdplot_se_y": se,
            "rdplot_N": N,
            "rdplot_ci_l": mean_y - quant * se,
            "rdplot_ci_r": mean_y + quant * se,
        },
        "vars_poly": {
            "rdplot_x": np.concatenate([x_plot_l, x_plot_r]),
            "rdplot_y": np.concatenate([y_hat_l, y_hat_r]),
        },
        "J": (J_l, J_r),
        "J_IMSE": (int(J_IMSE[0]), int(J_IMSE[1])),
        "J_MV": (int(J_MV[0]), int(J_MV[1])),
        "binselect": binselect,
        "mass_points_adjusted": mass_adjusted,
        "p": p,
        "c": c,
        "h": (h_l, h_r),
        "N": (n_l, n_r),
        "N_h": (n_h_l, n_h_r),
    }


# --------------------------------------------------------------------------- #
#  rdplotdensity: R lpdensity::lpdensity_fn (Cweights = Pweights = 1)
# --------------------------------------------------------------------------- #
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
    """R ``lpdensity(data, grid, bw, p, q, v, kernel, scale, massPoints)``.

    Local polynomial fit of order ``p`` (and ``q`` for the bias-corrected
    column) to the empirical CDF of ``data`` at each grid point, with the
    influence-function standard errors ``lpdensity_fn`` computes. Returns
    the columns of R's ``$Estimate``: ``grid, bw, nh, f_p, f_q, se_p, se_q``
    (densities and SEs multiplied by ``scale``, as R does).
    """
    from math import factorial

    data = np.sort(np.asarray(data, dtype=float), kind="mergesort")
    grid = np.asarray(grid, dtype=float)
    q = p + 1 if q is None else int(q)
    n, ng = len(data), len(grid)
    bw = np.full(ng, float(bw)) if np.ndim(bw) == 0 else np.asarray(bw, float)

    # lpdensityUnique: last index of each run of equal values.
    last = np.ones(n, dtype=bool)
    last[:-1] = data[1:] != data[:-1]
    idx_last = np.flatnonzero(last)
    freq = np.diff(np.concatenate([[-1], idx_last]))
    if mass_points:
        Fn = np.repeat((np.arange(1, n + 1) / n)[idx_last], freq)
    else:
        Fn = np.arange(1, n + 1) / n

    def _fit(order: int, j: int):
        u = (data - grid[j]) / bw[j]
        inside = np.abs(u) <= 1
        X = np.vander(u, order + 1, increasing=True)
        if kernel == "triangular":
            K = (1 - np.abs(u)) / bw[j] * inside
        elif kernel == "uniform":
            K = 0.5 / bw[j] * inside
        else:
            K = 0.75 * (1 - u**2) / bw[j] * inside
        XK = X * K[:, None]
        if mass_points:
            # One row per distinct value, weighted by its frequency
            # (R's PweightsUnique).
            Xu, XKu, Fu, ins = (
                X[idx_last],
                XK[idx_last] * freq[:, None],
                Fn[idx_last],
                inside[idx_last],
            )
            try:
                inv = np.linalg.inv(Xu[ins].T @ XKu[ins] / n)
            except np.linalg.LinAlgError:
                return np.nan, np.full(n, np.nan)
            est = factorial(v) * (inv @ XKu[ins].T @ Fu[ins])[v] / bw[j] ** v / n
        else:
            try:
                inv = np.linalg.inv(X[inside].T @ XK[inside] / n)
            except np.linalg.LinAlgError:
                return np.nan, np.full(n, np.nan)
            est = factorial(v) * (inv @ XK[inside].T @ Fn[inside])[v] / bw[j] ** v / n
        # Influence function (lpdensity_fn's G matrix)
        if mass_points:
            F_XK = Fu @ XKu / n
            rev = np.cumsum(XKu[::-1], axis=0)[::-1] / n
            G = np.repeat(rev, freq, axis=0) - F_XK
        else:
            F_XK = Fn @ XK / n
            G = np.cumsum(XK[::-1], axis=0)[::-1] / n - F_XK
        iff = (inv @ G.T)[v] * factorial(v) / np.sqrt(n * bw[j] ** (2 * v))
        return est, iff

    f_p = np.full(ng, np.nan)
    f_q = np.full(ng, np.nan)
    iff_p = np.full((n, ng), np.nan)
    iff_q = np.full((n, ng), np.nan)
    nh = np.zeros(ng, dtype=int)
    for j in range(ng):
        nh[j] = int(np.sum(np.abs(data - grid[j]) <= bw[j]))
        f_p[j], iff_p[:, j] = _fit(p, j)
        if q > p:
            f_q[j], iff_q[:, j] = _fit(q, j)
    se_p = np.sqrt(np.abs(np.sum(iff_p**2, axis=0) / n))
    se_q = np.sqrt(np.abs(np.sum(iff_q**2, axis=0) / n))
    return {
        "grid": grid,
        "bw": bw,
        "nh": nh,
        "f_p": f_p * scale,
        "f_q": f_q * scale,
        "se_p": se_p * scale,
        "se_q": se_q * scale,
    }
