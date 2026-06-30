"""Wild bootstrap for IV / 2SLS — the WRE bootstrap of Davidson-MacKinnon (2010).

The OLS wild cluster bootstrap (``wild_cluster_boot``) is WRONG for IV: it would
treat the 2SLS structural design as if it were an OLS regression and ignore the
two-stage / instrument structure. The correct procedure is the **wild restricted
efficient (WRE)** bootstrap, which imposes the null, resamples the *structural*
and *reduced-form* residuals with the same wild weight (preserving their
correlation), regenerates the endogenous regressor and the outcome, and refits
2SLS on each bootstrap sample. This is what Stata's ``boottest`` runs after
``ivreg2`` / ``ivregress``; this implementation is pinned to it numerically
(see ``tests/reference_parity/test_iv_wild_boottest_parity.py``).

Reads the 2SLS structure stored on the ivreg result under ``data_info['iv']``.
Single endogenous regressor (the common case) is supported; the tested
coefficient must be that endogenous regressor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility
from .jackknife import _draw_weights


def _iv_cluster_vcov(
    AX: np.ndarray,
    resid: np.ndarray,
    bread: np.ndarray,
    cl_codes: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """IV cluster-robust sandwich using projected regressors ``AX = P_W X``.

    Matches ``iv._cluster_cov``: meat over clusters with the CRV1 correction
    ``(G/(G-1)) * ((n-1)/(n-k))``.
    """
    n, k = AX.shape
    meat = np.zeros((k, k))
    for cid in range(n_clusters):
        idx = cl_codes == cid
        moments_c = (AX[idx] * resid[idx, None]).sum(axis=0)
        meat += np.outer(moments_c, moments_c)
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    return correction * bread @ meat @ bread


def iv_wild_bootstrap(
    result: Any,
    data: pd.DataFrame,
    cluster: str,
    variable: str,
    n_boot: int = 999,
    weight_type: str = "rademacher",
    seed: Optional[int] = None,
    alpha: float = 0.05,
    beta0: float = 0.0,
    efficient: bool = True,
) -> Dict[str, Any]:
    """WRE wild cluster bootstrap for a single (endogenous) 2SLS coefficient.

    Parameters
    ----------
    result : EconometricResults
        A fitted ``sp.ivreg`` result (2SLS); must carry ``data_info['iv']``.
    data : pd.DataFrame
        The estimation data (row-aligned to the fitted sample).
    cluster : str
        Cluster variable name in ``data``.
    variable : str
        Coefficient to test (``H0: beta = beta0``).
    n_boot, weight_type, seed, alpha, beta0
        Bootstrap controls (see ``wild_cluster_boot``).
    efficient : bool, default True
        If True, use the DM2010 *restricted efficient* reduced form (regress the
        endogenous regressor on the instruments **and** the restricted
        structural residual, exploiting the u-v correlation). If False, use the
        plain restricted reduced form (projection on instruments only).
    """
    iv = getattr(result, "data_info", {}).get("iv")
    if iv is None:
        raise MethodIncompatibility(
            "iv_wild_bootstrap requires an sp.ivreg (2SLS) result carrying the "
            "stored 2SLS structure (data_info['iv'])."
        )
    if iv["n_endog"] != 1:
        raise MethodIncompatibility(
            "iv_wild_bootstrap currently supports a single endogenous regressor."
        )

    names = list(iv["var_names"])
    if variable not in names:
        raise ValueError(f"Variable '{variable}' not found. Available: {names}")
    test_idx = names.index(variable)
    endog_idx = iv["n_exog"]  # single endogenous column sits after the exog block
    if test_idx != endog_idx:
        raise MethodIncompatibility(
            "iv_wild_bootstrap tests the endogenous coefficient "
            f"('{names[endog_idx]}'); got '{variable}'. For exogenous "
            "coefficients use the OLS wild bootstrap."
        )

    y = np.asarray(iv["y"], dtype=float)
    X = np.asarray(iv["X"], dtype=float)
    W = np.asarray(iv["W"], dtype=float)
    n, k = X.shape

    cl_series = data[cluster]
    if cl_series.isna().any() or len(cl_series) != n:
        raise MethodIncompatibility(
            "iv_wild_bootstrap needs the cluster column row-aligned to the "
            "fitted sample with no missing values."
        )
    cl_codes = pd.Categorical(cl_series).codes
    n_clusters = int(cl_codes.max()) + 1

    rng = np.random.default_rng(seed)

    # --- projection / 2SLS bread (instruments are fixed across bootstrap) ---
    WtW_inv = np.linalg.inv(W.T @ W)
    P_W = W @ WtW_inv @ W.T  # A = P_W for kappa = 1 (2SLS)

    def _fit(Xmat: np.ndarray, yvec: np.ndarray):
        AX = P_W @ Xmat
        bread = np.linalg.inv(Xmat.T @ AX)
        beta = bread @ (AX.T @ yvec)
        resid = yvec - Xmat @ beta
        V = _iv_cluster_vcov(AX, resid, bread, cl_codes, n_clusters)
        return beta, V

    beta_hat, V_hat = _fit(X, y)
    se = float(np.sqrt(max(V_hat[test_idx, test_idx], 1e-20)))
    t_obs = (beta_hat[test_idx] - beta0) / se if se > 0 else 0.0

    # --- restricted estimation under H0: beta_endog = beta0 ---
    d = X[:, test_idx]
    other = [j for j in range(k) if j != test_idx]
    Xo = X[:, other]
    g = np.linalg.lstsq(Xo, y - beta0 * d, rcond=None)[0]
    u_tilde = (y - beta0 * d) - Xo @ g

    # --- restricted reduced form for the endogenous regressor ---
    if efficient:
        Wd = np.column_stack([W, u_tilde])
        coef = np.linalg.lstsq(Wd, d, rcond=None)[0]
        d_base = W @ coef[: W.shape[1]]  # the instrument-explained part of d
    else:
        d_base = P_W @ d
    d_resid = d - d_base  # carries the wild-resampled reduced-form variation

    # --- bootstrap ---
    t_boot = np.empty(n_boot)
    for b in range(n_boot):
        w_g = _draw_weights(n_clusters, weight_type, rng)
        eps = w_g[cl_codes]
        u_star = eps * u_tilde
        d_star = d_base + eps * d_resid  # = d_base + eps*(d - d_base)
        y_star = beta0 * d_star + Xo @ g + u_star
        X_star = X.copy()
        X_star[:, test_idx] = d_star
        beta_s, V_s = _fit(X_star, y_star)
        se_s = float(np.sqrt(max(V_s[test_idx, test_idx], 1e-20)))
        t_boot[b] = (beta_s[test_idx] - beta0) / se_s if se_s > 0 else 0.0

    p_boot = float(np.mean(np.abs(t_boot) >= np.abs(t_obs)))
    t_lo = float(np.percentile(t_boot, 100 * alpha / 2))
    t_hi = float(np.percentile(t_boot, 100 * (1 - alpha / 2)))
    ci = (beta_hat[test_idx] - t_hi * se, beta_hat[test_idx] - t_lo * se)

    return {
        "beta_hat": float(beta_hat[test_idx]),
        "se_cluster": se,
        "t_stat": float(t_obs),
        "p_boot": p_boot,
        "ci_boot": ci,
        "t_distribution": t_boot,
        "n_clusters": n_clusters,
        "n_obs": n,
        "n_boot": n_boot,
        "weight_type": weight_type,
        "method": "WRE" if efficient else "WUR",
    }
