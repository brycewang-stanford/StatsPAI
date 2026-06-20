"""
Shared utilities for decomposition analysis.

Provides weighted OLS, logit, bootstrap helpers, and cluster-robust
variance estimation used across multiple decomposition methods.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Callable, cast
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import (
    ConvergenceFailure,
    DataInsufficient,
    MethodIncompatibility,
)

# ════════════════════════════════════════════════════════════════════════
# Weighted OLS
# ════════════════════════════════════════════════════════════════════════


def add_constant(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones."""
    n = X.shape[0]
    return np.column_stack([np.ones(n), X])


def wls(
    y: np.ndarray,
    X: np.ndarray,
    w: Optional[np.ndarray] = None,
    robust: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted least squares with optional HC1 robust variance.

    Parameters
    ----------
    y : (n,) array
    X : (n, k) array — must include constant if desired
    w : (n,) array or None — observation weights (default: 1)
    robust : bool — if True, HC1 sandwich; else σ^2 (X'WX)^{-1}

    Returns
    -------
    beta, vcov, resid
    """
    n, k = X.shape
    if w is None:
        w = np.ones(n)
    w = np.asarray(w, dtype=float)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    # QR for stability
    Q, R = np.linalg.qr(Xw, mode="reduced")
    beta = np.linalg.solve(R, Q.T @ yw)
    resid = y - X @ beta
    XtWX_inv = np.linalg.inv(R.T @ R)

    if robust:
        # HC1: (X'WX)^{-1} X' diag(w * e^2) X (X'WX)^{-1} * n/(n-k)
        e2 = w * resid**2
        meat = (X * e2[:, None]).T @ X
        vcov = XtWX_inv @ meat @ XtWX_inv * (n / max(n - k, 1))
    else:
        sigma2 = float((w * resid**2).sum() / max(n - k, 1))
        vcov = sigma2 * XtWX_inv

    return beta, vcov, resid


def cluster_vcov(
    X: np.ndarray,
    resid: np.ndarray,
    clusters: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Cluster-robust variance (CR1)."""
    n, k = X.shape
    if w is None:
        w = np.ones(n)
    Xw = X * np.sqrt(w)[:, None]
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    g_arr = np.asarray(clusters)
    g_unique = np.unique(g_arr)
    G = len(g_unique)
    meat = np.zeros((k, k))
    for g in g_unique:
        idx = np.where(g_arr == g)[0]
        u_g = (X[idx] * (w[idx] * resid[idx])[:, None]).sum(axis=0)
        meat += np.outer(u_g, u_g)
    factor = G / max(G - 1, 1) * (n - 1) / max(n - k, 1)
    return cast(np.ndarray, XtX_inv @ meat @ XtX_inv * factor)


# ════════════════════════════════════════════════════════════════════════
# Logit (Newton-Raphson)
# ════════════════════════════════════════════════════════════════════════


def logit_fit(
    y: np.ndarray,
    X: np.ndarray,
    w: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    warn_on_nonconvergence: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Logit MLE via Newton-Raphson / IRLS.

    Parameters
    ----------
    y : (n,) binary {0, 1}
    X : (n, k) design matrix (with constant)
    w : (n,) weights or None
    max_iter : int
    tol : float
    warn_on_nonconvergence : bool
        Emit a RuntimeWarning if the NR loop exits without convergence.

    Returns
    -------
    beta : (k,) estimates
    vcov : (k, k) model-based covariance

    Notes
    -----
    On near-separated data NR can diverge to large β with collapsing
    probabilities. The clip at ±30 keeps η finite; if convergence is
    not achieved within max_iter the caller gets warned and should
    consider ridge-penalised logit or entropy balancing instead.
    """
    n, k = X.shape
    if w is None:
        w = np.ones(n)
    w = np.asarray(w, dtype=float)
    beta = np.zeros(k)
    converged = False
    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W_diag = w * p * (1 - p)
        grad = X.T @ (w * (y - p))
        H = -(X * W_diag[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:  # pragma: no cover
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    if not converged and warn_on_nonconvergence:
        warnings.warn(
            f"logit_fit did not converge within {max_iter} iterations "
            "(possible separation or near-separation). Results may be "
            "unreliable; consider reducing dimensionality or trimming "
            "extreme propensity scores.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Covariance
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    W_diag = w * p * (1 - p)
    info = (X * W_diag[:, None]).T @ X
    try:
        vcov = np.linalg.inv(info)
    except np.linalg.LinAlgError:  # pragma: no cover
        vcov = np.linalg.pinv(info)
    return beta, vcov


def logit_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Predicted probabilities from logit coefficients."""
    eta = np.clip(X @ beta, -30, 30)
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-eta)))


# ════════════════════════════════════════════════════════════════════════
# Bootstrap helpers
# ════════════════════════════════════════════════════════════════════════


def bootstrap_stat(
    stat_fn: Callable[[np.ndarray], Union[float, np.ndarray]],
    n: int,
    n_boot: int = 499,
    rng: Optional[np.random.Generator] = None,
    strata: Optional[np.ndarray] = None,
    clusters: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generic non-parametric bootstrap.

    Parameters
    ----------
    stat_fn : function(idx: np.ndarray) -> scalar or 1-d array
        Called with resampled row indices.
    n : int — total sample size
    n_boot : int — number of bootstrap replications
    rng : np.random.Generator or None
    strata : (n,) or None — stratum id for stratified bootstrap
    clusters : (n,) or None — cluster id for block bootstrap

    Returns
    -------
    (n_boot, d) array of bootstrap replications (d=1 for scalar stat)
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    results: list[Union[float, np.ndarray]] = []
    n_failed = 0
    for _ in range(n_boot):
        if clusters is not None:
            g_arr = np.asarray(clusters)
            g_unique = np.unique(g_arr)
            g_sample = rng.choice(g_unique, size=len(g_unique), replace=True)
            idx = np.concatenate([np.where(g_arr == g)[0] for g in g_sample])
        elif strata is not None:
            s_arr = np.asarray(strata)
            s_unique = np.unique(s_arr)
            idx_parts = []
            for s in s_unique:
                s_idx = np.where(s_arr == s)[0]
                idx_parts.append(rng.choice(s_idx, size=len(s_idx), replace=True))
            idx = np.concatenate(idx_parts)
        else:
            idx = rng.integers(0, n, size=n)
        try:
            results.append(stat_fn(idx))
        except Exception:  # noqa: BLE001
            n_failed += 1
            continue
    if not results:
        raise ConvergenceFailure("All bootstrap replications failed.")
    # Warn if more than 5% of replications failed silently
    if n_failed > 0.05 * n_boot:
        warnings.warn(
            f"{n_failed}/{n_boot} bootstrap replications failed "
            f"({100 * n_failed / n_boot:.1f}%). "
            "SE estimates are based on the successful subset. Check "
            "for degenerate resamples or numerical issues in stat_fn.",
            RuntimeWarning,
            stacklevel=2,
        )
    arr = np.asarray(results, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def wild_bootstrap_stat(
    stat_fn: Callable[[np.ndarray], Union[float, np.ndarray]],
    resid: np.ndarray,
    fitted: np.ndarray,
    n_boot: int = 499,
    rng: Optional[np.random.Generator] = None,
    weights: str = "rademacher",
    clusters: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Wild (multiplier) bootstrap for residual-based decomposition statistics.

    Generates pseudo-outcomes ``y* = fitted + v_i * resid_i`` with i.i.d.
    multipliers ``v_i ∈ {-1, 1}`` (Rademacher) or two-point Mammen weights,
    keeping ``X`` fixed. ``stat_fn`` is called with the bootstrap pseudo-y;
    it must accept a 1-d numpy array of length ``n``.

    For cluster-robust wild bootstrap (Cameron-Gelbach-Miller 2008), pass a
    ``clusters`` vector — multipliers then share within cluster.

    Parameters
    ----------
    stat_fn : callable(y_star) -> scalar or 1-d array
    resid : (n,) baseline residuals
    fitted : (n,) baseline fitted values
    n_boot : int
    rng : np.random.Generator or None
    weights : {"rademacher", "mammen"}
    clusters : (n,) cluster id or None

    Returns
    -------
    (n_boot, d) bootstrap replications.
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    resid = np.asarray(resid, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    n = len(resid)
    out: list[Union[float, np.ndarray]] = []
    n_failed = 0

    if clusters is not None:
        c_arr = np.asarray(clusters)
        c_unique, c_idx = np.unique(c_arr, return_inverse=True)
        n_g = len(c_unique)
    else:
        c_idx = None
        n_g = n

    for _ in range(n_boot):
        if weights == "rademacher":
            v_g = rng.choice([-1.0, 1.0], size=n_g)
        elif weights == "mammen":
            # Mammen (1993) two-point distribution
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            p = phi / np.sqrt(5.0)
            v_g = np.where(rng.random(n_g) < p, -(phi - 1.0), phi)
        else:
            raise MethodIncompatibility(
                f"unknown weights {weights!r}",
                recovery_hint="Use weights='rademacher' or weights='mammen'.",
                diagnostics={"weights": weights},
            )
        v = v_g if c_idx is None else v_g[c_idx]
        y_star = fitted + v * resid
        try:
            out.append(stat_fn(y_star))
        except Exception:  # noqa: BLE001
            n_failed += 1
            continue
    if not out:
        raise ConvergenceFailure("All wild-bootstrap replications failed.")
    if n_failed > 0.05 * n_boot:
        warnings.warn(
            f"{n_failed}/{n_boot} wild-bootstrap replications failed "
            f"({100 * n_failed / n_boot:.1f}%). SE estimates use the "
            "successful subset.",
            RuntimeWarning,
            stacklevel=2,
        )
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def bootstrap_ci(
    boot: np.ndarray,
    point: np.ndarray,
    alpha: float = 0.05,
    method: str = "percentile",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap CIs and SEs.

    Parameters
    ----------
    boot : (n_boot, d) replications
    point : (d,) point estimate
    alpha : float — two-sided significance
    method : {'percentile', 'basic', 'normal'}

    Returns
    -------
    se : (d,) bootstrap std
    lo : (d,) lower bound
    hi : (d,) upper bound
    """
    boot = np.atleast_2d(boot)
    if boot.shape[1] == 1 and boot.shape[0] > 1 and point.size > 1:
        boot = boot.T
    se = boot.std(axis=0, ddof=1)
    if method == "percentile":
        lo = np.quantile(boot, alpha / 2, axis=0)
        hi = np.quantile(boot, 1 - alpha / 2, axis=0)
    elif method == "basic":
        q_lo = np.quantile(boot, alpha / 2, axis=0)
        q_hi = np.quantile(boot, 1 - alpha / 2, axis=0)
        lo = 2 * point - q_hi
        hi = 2 * point - q_lo
    elif method == "normal":
        z = stats.norm.ppf(1 - alpha / 2)
        lo = point - z * se
        hi = point + z * se
    else:
        raise MethodIncompatibility(
            f"unknown method {method!r}",
            recovery_hint=(
                "Use method='percentile', method='basic', or method='normal'."
            ),
            diagnostics={"method": method},
        )
    return se, lo, hi


# ════════════════════════════════════════════════════════════════════════
# Weighted quantile / density / CDF
# ════════════════════════════════════════════════════════════════════════


def weighted_quantile(
    y: np.ndarray, q: Union[float, np.ndarray], w: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """
    Weighted quantile via empirical CDF inversion.
    """
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y)
    w = np.asarray(w, dtype=float)
    order = np.argsort(y)
    y_s = y[order]
    w_s = w[order]
    cum = np.cumsum(w_s) / w_s.sum()
    q_arr = np.atleast_1d(q)
    out = np.interp(q_arr, cum, y_s)
    if np.isscalar(q):
        return float(out[0])
    return cast(np.ndarray, out)


def weighted_quantile_hmisc(
    y: np.ndarray, q: Union[float, np.ndarray], w: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """Hmisc/dineq-compatible weighted quantile.

    This ports ``Hmisc::wtd.quantile(type="quantile")``. With unit
    weights it is R's default type-7 quantile; with weights it collapses
    duplicate values, uses ``1 + (sum(w) - 1) * p`` as the fractional
    order, and right-continuous constant interpolation on the weighted
    order statistic.
    """
    y = np.asarray(y, dtype=float).ravel()
    if w is None:
        w = np.ones_like(y)
    w = np.asarray(w, dtype=float).ravel()
    if y.shape[0] != w.shape[0]:
        raise MethodIncompatibility(
            "y and w must have the same length",
            recovery_hint="Pass one positive weight per outcome observation.",
            diagnostics={"n_y": y.shape[0], "n_w": w.shape[0]},
        )

    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        raise DataInsufficient("at least one positive finite weight is required")
    y = y[mask]
    w = w[mask]

    if np.any((np.asarray(q) < 0) | (np.asarray(q) > 1)):
        raise MethodIncompatibility(
            "quantile probabilities must be between 0 and 1",
            recovery_hint="Pass quantile probabilities in the closed unit interval.",
            diagnostics={"q": np.asarray(q).tolist()},
        )

    order = np.argsort(y, kind="mergesort")
    y_s = y[order]
    w_s = w[order]
    uniq, inverse = np.unique(y_s, return_inverse=True)
    w_collapsed = np.bincount(inverse, weights=w_s)
    cum = np.cumsum(w_collapsed)
    n = float(w_collapsed.sum())

    q_arr = np.atleast_1d(q).astype(float)
    pos = 1.0 + (n - 1.0) * q_arr
    low = np.maximum(np.floor(pos), 1.0)
    high = np.minimum(low + 1.0, n)
    frac = np.mod(pos, 1.0)

    xout = np.concatenate([low, high])
    idx = np.searchsorted(cum, xout, side="left")
    idx = np.clip(idx, 0, len(uniq) - 1)
    vals = uniq[idx]
    k = len(q_arr)
    out = (1.0 - frac) * vals[:k] + frac * vals[k:]
    if np.isscalar(q):
        return float(out[0])
    return cast(np.ndarray, out)


def _bw_nrd0(y: np.ndarray) -> float:
    """Port of R ``stats::bw.nrd0`` for Gaussian kernel density."""
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return 1.0
    hi = float(np.std(y, ddof=1))
    q75, q25 = np.quantile(y, [0.75, 0.25])
    iqr = float(q75 - q25)
    lo = min(hi, iqr / 1.34)
    if not np.isfinite(lo) or lo == 0:
        lo = hi
    if not np.isfinite(lo) or lo == 0:
        lo = abs(float(y[0]))
    if not np.isfinite(lo) or lo == 0:
        lo = 1.0
    return float(0.9 * lo * len(y) ** (-0.2))


def kde_at_dineq(y: np.ndarray, point: float, w: Optional[np.ndarray] = None) -> float:
    """dineq-compatible Gaussian density at ``point``.

    ``dineq::rif`` calls R ``density(..., bw="nrd0", kernel="gaussian")`` with
    normalised weights. R's ``density.default`` does not evaluate the Gaussian
    kernel sum directly: it bins observations on a 512-point grid, performs an
    FFT convolution, then interpolates back to the requested point. Mirroring
    that path removes the small but visible RIF-regression drift from using the
    closed-form kernel average.
    """
    y = np.asarray(y, dtype=float).ravel()
    if w is None:
        w = np.ones_like(y)
    w = np.asarray(w, dtype=float).ravel()
    if y.shape[0] != w.shape[0]:
        raise MethodIncompatibility(
            "y and w must have the same length",
            recovery_hint="Pass one positive weight per outcome observation.",
            diagnostics={"n_y": y.shape[0], "n_w": w.shape[0]},
        )
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        raise DataInsufficient("at least one positive finite weight is required")
    y_pos = y[mask]
    w_pos = w[mask]
    h = max(_bw_nrd0(y_pos), 1e-12)
    w_norm = w_pos / w_pos.sum()

    # Port of stats::density.default(..., from=point, to=point, n=1,
    # kernel="gaussian") plus C_BinDist from R's stats package.
    n_grid = 512
    lo = point - 4.0 * h
    up = point + 4.0 * h
    xdelta = (up - lo) / (n_grid - 1)
    bins = np.zeros(2 * n_grid, dtype=float)
    ixmin = 0
    ixmax = n_grid - 2
    int_min = np.iinfo(np.int32).min
    int_max = np.iinfo(np.int32).max
    for xi, wi in zip(y_pos, w_norm):
        xpos = (xi - lo) / xdelta
        if xpos > int_max or xpos < int_min:
            continue
        ix = int(np.floor(xpos))
        fx = xpos - ix
        if ixmin <= ix <= ixmax:
            bins[ix] += (1.0 - fx) * wi
            bins[ix + 1] += fx * wi
        elif ix == -1:
            bins[0] += fx * wi
        elif ix == ixmax + 1:
            bins[ix] += (1.0 - fx) * wi

    kords = np.linspace(
        0.0,
        ((2 * n_grid - 1) / (n_grid - 1)) * (up - lo),
        2 * n_grid,
    )
    kords[n_grid + 1 :] = -kords[n_grid - 1 : 0 : -1]
    kernel = np.exp(-0.5 * (kords / h) ** 2) / (h * np.sqrt(2 * np.pi))
    conv = np.fft.ifft(np.fft.fft(bins) * np.conj(np.fft.fft(kernel))).real
    density_grid = np.maximum(0.0, conv[:n_grid])
    x_grid = np.linspace(lo, up, n_grid)
    return float(np.interp(point, x_grid, density_grid))


def weighted_ecdf(
    y_eval: np.ndarray, y_sample: np.ndarray, w: Optional[np.ndarray] = None
) -> np.ndarray:
    """Weighted ECDF evaluated at y_eval."""
    y_sample = np.asarray(y_sample, dtype=float)
    y_eval = np.atleast_1d(y_eval).astype(float)
    if w is None:
        w = np.ones_like(y_sample)
    w = np.asarray(w, dtype=float)
    order = np.argsort(y_sample)
    ys = y_sample[order]
    ws = w[order]
    cum = np.cumsum(ws) / ws.sum()
    idx = np.searchsorted(ys, y_eval, side="right") - 1
    idx = np.clip(idx, -1, len(ys) - 1)
    out = np.where(idx < 0, 0.0, cum[np.clip(idx, 0, len(ys) - 1)])
    return out


def kde_at(y: np.ndarray, point: float, w: Optional[np.ndarray] = None) -> float:
    """Gaussian kernel density at a single point (weighted)."""
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y)
    w = np.asarray(w, dtype=float)
    n_eff = (w.sum() ** 2) / (w**2).sum()
    sigma = float(np.sqrt(np.cov(y, aweights=w)))
    if sigma <= 0 or not np.isfinite(sigma):
        y_std = float(y.std())
        sigma = y_std if y_std > 0 else 1.0
    h = 1.06 * float(sigma) * n_eff ** (-0.2)
    h = max(h, 1e-6)
    kern = np.exp(-0.5 * ((y - point) / h) ** 2) / (h * np.sqrt(2 * np.pi))
    return float(np.average(kern, weights=w))


# ════════════════════════════════════════════════════════════════════════
# Significance formatting
# ════════════════════════════════════════════════════════════════════════


def weighted_gini(y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Gini coefficient (Lerman-Yitzhaki 1989)."""
    order = np.argsort(y)
    y_s = y[order]
    w_s = w[order]
    W = w_s.sum()
    if W <= 0:
        return float("nan")
    cum_w = np.cumsum(w_s)
    F = (cum_w - 0.5 * w_s) / W
    mu = float(np.average(y_s, weights=w_s))
    if mu <= 0:
        return float("nan")
    return float(2.0 * np.cov(y_s, F, aweights=w_s)[0, 1] / mu)


def statistic_value(
    y: np.ndarray,
    w: np.ndarray,
    stat: str,
    tau: float = 0.5,
) -> float:
    """
    Evaluate a weighted distributional statistic.

    Supported: ``mean``, ``variance``, ``std``, ``quantile`` (with tau),
    ``iqr``, ``gini``, ``log_var``, ``theil_t``, ``theil_l``, ``atkinson``.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if stat == "mean":
        return float(np.average(y, weights=w))
    if stat == "variance":
        return float(np.cov(y, aweights=w))
    if stat == "std":
        return float(np.sqrt(np.cov(y, aweights=w)))
    if stat == "quantile":
        return float(weighted_quantile(y, tau, w=w))
    if stat == "iqr":
        return float(weighted_quantile(y, 0.75, w=w) - weighted_quantile(y, 0.25, w=w))
    if stat == "gini":
        return weighted_gini(y, w)
    if stat == "log_var":
        return float(np.cov(np.log(np.clip(y, 1e-12, None)), aweights=w))
    if stat == "theil_t":
        yp = np.clip(y, 1e-12, None)
        mu = float(np.average(yp, weights=w))
        if mu <= 0:
            return float("nan")  # pragma: no cover
        return float(np.average((yp / mu) * np.log(yp / mu), weights=w))
    if stat == "theil_l":
        yp = np.clip(y, 1e-12, None)
        mu = float(np.average(yp, weights=w))
        if mu <= 0:
            return float("nan")  # pragma: no cover
        return float(np.log(mu) - np.average(np.log(yp), weights=w))
    if stat == "atkinson":
        yp = np.clip(y, 1e-12, None)
        mu = float(np.average(yp, weights=w))
        if mu <= 0:
            return float("nan")  # pragma: no cover
        return float(1.0 - np.exp(np.average(np.log(yp), weights=w)) / mu)
    raise MethodIncompatibility(
        f"unknown statistic {stat!r}",
        recovery_hint=(
            "Use a supported statistic such as 'mean', 'quantile', 'gini', "
            "'theil_t', 'theil_l', or 'atkinson'."
        ),
        diagnostics={"stat": stat},
    )


def analytical_ci(
    point: Union[float, np.ndarray],
    se: Union[float, np.ndarray],
    alpha: float = 0.05,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Two-sided normal confidence interval ``point ± z * se``."""
    z = float(stats.norm.ppf(1 - alpha / 2))
    return point - z * se, point + z * se


def sig_stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    if pval < 0.1:
        return "+"
    return ""


# ════════════════════════════════════════════════════════════════════════
# Influence functions (RIF kernel)
# ════════════════════════════════════════════════════════════════════════


def influence_function(
    y: np.ndarray,
    stat: str,
    tau: float = 0.5,
    w: Optional[np.ndarray] = None,
    quantile_convention: str = "statspai",
) -> np.ndarray:
    """
    Per-observation influence function (RIF kernel) of a distributional
    statistic evaluated at the (optionally weighted) empirical distribution
    of ``y``.

    Canonical implementation used by the FFL two-step decomposition
    (``ffl._rif_for_sample``) and — for the overlapping statistics
    ``variance`` and ``gini`` — by ``rif.rif_values``. Supported stats:

        ``quantile`` (with ``tau``), ``mean``, ``variance``, ``std``,
        ``log_var``, ``iqr``, ``gini``, ``theil_t``, ``theil_l``,
        ``atkinson`` (ε = 1).

    When ``w`` is ``None`` the function uses unit weights and thus
    coincides with the classical (unweighted) influence-function formulas.

    Notes
    -----
    For the quantile statistic the density at the quantile is estimated
    via a manual Silverman-of-thumb Gaussian kernel
    (``h = 1.06 · σ · n_eff^{-0.2}``). This matches FFL (2018) and is
    numerically close to — but not identical with —
    ``scipy.stats.gaussian_kde(bw_method='silverman')``; the relative
    bandwidth difference is about 0.5%.
    """
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y)
    w = np.asarray(w, dtype=float)

    if stat == "quantile":
        if quantile_convention == "statspai":
            q = float(weighted_quantile(y, tau, w=w))
            n_eff = (w.sum() ** 2) / (w**2).sum()
            sigma = np.sqrt(max(float(np.cov(y, aweights=w)), 1e-12))
            h = max(1.06 * sigma * n_eff ** (-0.2), 1e-6)
            kern = np.exp(-0.5 * ((y - q) / h) ** 2) / (h * np.sqrt(2 * np.pi))
            f_q = max(float(np.average(kern, weights=w)), 1e-12)
            return q + (tau - (y <= q).astype(float)) / f_q
        if quantile_convention == "dineq":
            q = float(weighted_quantile_hmisc(y, tau, w=w))
            f_q = max(kde_at_dineq(y, q, w=w), 1e-12)
            return q + (tau - (y < q).astype(float)) / f_q
        raise MethodIncompatibility(
            "quantile_convention must be 'statspai' or 'dineq'",
            recovery_hint=(
                "Use quantile_convention='statspai' or " "quantile_convention='dineq'."
            ),
            diagnostics={"quantile_convention": quantile_convention},
        )
    if stat == "mean":
        return cast(np.ndarray, y.copy())
    if stat == "variance":
        mu = float(np.average(y, weights=w))
        return cast(np.ndarray, (y - mu) ** 2)
    if stat == "std":
        mu = float(np.average(y, weights=w))
        s2 = float(np.average((y - mu) ** 2, weights=w))
        s = np.sqrt(max(s2, 1e-12))
        return cast(np.ndarray, s + ((y - mu) ** 2 - s2) / (2 * s))
    if stat == "log_var":
        ly = np.log(np.clip(y, 1e-12, None))
        mu = float(np.average(ly, weights=w))
        return cast(np.ndarray, (ly - mu) ** 2)
    if stat == "iqr":
        return cast(
            np.ndarray,
            influence_function(
                y,
                "quantile",
                tau=0.75,
                w=w,
                quantile_convention=quantile_convention,
            )
            - influence_function(
                y,
                "quantile",
                tau=0.25,
                w=w,
                quantile_convention=quantile_convention,
            ),
        )
    if stat == "gini":
        order = np.argsort(y)
        y_s = y[order]
        w_s = w[order]
        W = w_s.sum()
        mu = float(np.average(y_s, weights=w_s))
        if mu <= 0:
            return np.full_like(y, np.nan)
        F = (np.cumsum(w_s) - 0.5 * w_s) / W
        GL = np.cumsum(w_s * y_s) / W
        G = 2.0 * float(np.cov(y_s, F, aweights=w_s)[0, 1]) / mu
        rif_sorted = 1.0 + (2.0 / mu) * (y_s * F - GL) - ((G + 1.0) / mu) * y_s
        rif_orig = np.empty_like(rif_sorted)
        rif_orig[order] = rif_sorted
        return cast(np.ndarray, rif_orig)
    if stat == "theil_t":
        yp = np.clip(y, 1e-12, None)
        mu = float(np.average(yp, weights=w))
        if mu <= 0:
            return np.full_like(y, np.nan)  # pragma: no cover
        s = yp / mu
        T = float(np.average(s * np.log(s), weights=w))
        return cast(np.ndarray, s * np.log(s) - (s - 1.0) * (T + 1.0))
    if stat == "theil_l":
        yp = np.clip(y, 1e-12, None)
        mu = float(np.average(yp, weights=w))
        if mu <= 0:
            return np.full_like(y, np.nan)  # pragma: no cover
        return cast(np.ndarray, (yp / mu - 1.0) - (np.log(yp) - np.log(mu)))
    if stat == "atkinson":
        yp = np.clip(y, 1e-12, None)
        mu = float(np.average(yp, weights=w))
        if mu <= 0:
            return np.full_like(y, np.nan)  # pragma: no cover
        mean_log = float(np.average(np.log(yp), weights=w))
        geo_mean = np.exp(mean_log)
        A1 = 1.0 - geo_mean / mu
        return cast(
            np.ndarray,
            A1 + (geo_mean / mu) * ((yp - mu) / mu - (np.log(yp) - mean_log)),
        )
    raise MethodIncompatibility(
        f"unknown statistic {stat!r}",
        recovery_hint=(
            "Use a supported statistic such as 'mean', 'quantile', 'gini', "
            "'theil_t', 'theil_l', or 'atkinson'."
        ),
        diagnostics={"stat": stat},
    )


# ════════════════════════════════════════════════════════════════════════
# DataFrame / formula parsing
# ════════════════════════════════════════════════════════════════════════


def parse_formula(formula: str) -> Tuple[str, list[str]]:
    """'y ~ x1 + x2 + x3' -> ('y', ['x1', 'x2', 'x3'])."""
    if "~" not in formula:
        raise MethodIncompatibility(
            "formula must contain '~'",
            recovery_hint="Use a formula such as 'y ~ x1 + x2'.",
            diagnostics={"formula": formula},
        )
    dep, rhs = [s.strip() for s in formula.split("~", 1)]
    indep = [v.strip() for v in rhs.split("+") if v.strip() and v.strip() != "1"]
    return dep, indep


def prepare_frame(
    data: pd.DataFrame,
    cols: Sequence[str],
    weights: Optional[Union[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Select columns, drop NA, extract weights vector."""
    data = data.copy()
    cols = list(cols)
    if weights is not None and isinstance(weights, str):
        use_cols = list(dict.fromkeys(cols + [weights]))
    else:
        use_cols = list(dict.fromkeys(cols))
    missing = [col for col in use_cols if col not in data.columns]
    if missing:
        raise MethodIncompatibility(
            "prepare_frame columns are missing from data.",
            recovery_hint="Check requested columns and weights against data.columns.",
            diagnostics={
                "missing_columns": missing,
                "available_columns": list(data.columns),
            },
        )
    df = data[use_cols].dropna()
    if weights is None:
        w_out = np.ones(len(df), dtype=float)
    elif isinstance(weights, str):
        w_out = df[weights].to_numpy(dtype=float)
        df = df.drop(columns=[weights])
    else:
        w_out = np.asarray(weights, dtype=float).ravel()
        if len(w_out) != len(df):
            # Likely the user passed original-length weights; align by index
            raise MethodIncompatibility(
                "weights array length does not match data.",
                recovery_hint=(
                    "Pass weights already aligned to the cleaned analysis "
                    "frame or use a weight column name."
                ),
                diagnostics={"n_weights": len(w_out), "n_rows": len(df)},
            )
    return df, w_out
