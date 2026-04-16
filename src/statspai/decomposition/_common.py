"""
Shared utilities for decomposition analysis.

Provides weighted OLS, logit, bootstrap helpers, and cluster-robust
variance estimation used across multiple decomposition methods.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Callable
import warnings

import numpy as np
import pandas as pd
from scipy import stats


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
    Q, R = np.linalg.qr(Xw, mode='reduced')
    beta = np.linalg.solve(R, Q.T @ yw)
    resid = y - X @ beta
    XtWX_inv = np.linalg.inv(R.T @ R)

    if robust:
        # HC1: (X'WX)^{-1} X' diag(w * e^2) X (X'WX)^{-1} * n/(n-k)
        e2 = (w * resid ** 2)
        meat = (X * e2[:, None]).T @ X
        vcov = XtWX_inv @ meat @ XtWX_inv * (n / max(n - k, 1))
    else:
        sigma2 = float((w * resid ** 2).sum() / max(n - k, 1))
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
    return XtX_inv @ meat @ XtX_inv * factor


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
        except np.linalg.LinAlgError:
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
            RuntimeWarning, stacklevel=2,
        )
    # Covariance
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    W_diag = w * p * (1 - p)
    info = (X * W_diag[:, None]).T @ X
    try:
        vcov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        vcov = np.linalg.pinv(info)
    return beta, vcov


def logit_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Predicted probabilities from logit coefficients."""
    eta = np.clip(X @ beta, -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


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
        raise RuntimeError("All bootstrap replications failed.")
    # Warn if more than 5% of replications failed silently
    if n_failed > 0.05 * n_boot:
        warnings.warn(
            f"{n_failed}/{n_boot} bootstrap replications failed "
            f"({100 * n_failed / n_boot:.1f}%). "
            "SE estimates are based on the successful subset. Check "
            "for degenerate resamples or numerical issues in stat_fn.",
            RuntimeWarning, stacklevel=2,
        )
    arr = np.asarray(results, dtype=float)
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
        raise ValueError(f"unknown method {method!r}")
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
    return out


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
    n_eff = (w.sum() ** 2) / (w ** 2).sum()
    sigma = np.sqrt(np.cov(y, aweights=w))
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = y.std() if y.std() > 0 else 1.0
    h = 1.06 * float(sigma) * n_eff ** (-0.2)
    h = max(h, 1e-6)
    kern = np.exp(-0.5 * ((y - point) / h) ** 2) / (h * np.sqrt(2 * np.pi))
    return float(np.average(kern, weights=w))


# ════════════════════════════════════════════════════════════════════════
# Significance formatting
# ════════════════════════════════════════════════════════════════════════

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
# DataFrame / formula parsing
# ════════════════════════════════════════════════════════════════════════

def parse_formula(formula: str) -> Tuple[str, list[str]]:
    """'y ~ x1 + x2 + x3' -> ('y', ['x1', 'x2', 'x3'])."""
    if "~" not in formula:
        raise ValueError("formula must contain '~'")
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
    df = data[use_cols].dropna()
    if weights is None:
        w = np.ones(len(df))
    elif isinstance(weights, str):
        w = df[weights].to_numpy(dtype=float)
        df = df.drop(columns=[weights])
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(df):
            # Likely the user passed original-length weights; align by index
            raise ValueError("weights array length does not match data.")
    return df, w
