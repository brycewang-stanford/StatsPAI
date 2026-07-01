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
Supports one or more endogenous regressors; the tested coefficient must be one
of them (the other endogenous regressors are re-estimated by 2SLS under the
null). Validated against ``boottest`` for both the single- and two-endogenous
cases.
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


def iv_twoway_vcov(
    result: Any,
    data: pd.DataFrame,
    cluster1: str,
    cluster2: str,
) -> Dict[str, Any]:
    """Two-way cluster-robust covariance for 2SLS (Cameron-Gelbach-Miller 2011).

    Inclusion-exclusion on the IV cluster sandwich (projected regressors
    ``AX = P_W X``): ``V = V(c1) + V(c2) - V(c1 ∩ c2)``. The finite-sample
    factor ``(G_min/(G_min-1)) * ((n-1)/(n-k))`` with ``G_min = min(G1, G2)``
    matches Stata ``ivreg2 ..., cluster(c1 c2) small`` and is consistent with
    ``sp.ivreg``'s one-way cluster convention.

    Returns ``{"vcov", "std_errors", "var_names", "n_clusters1/2/12"}``.
    """
    iv = getattr(result, "data_info", {}).get("iv")
    if iv is None:
        raise MethodIncompatibility(
            "iv_twoway_vcov requires an sp.ivreg (2SLS) result (data_info['iv'])."
        )
    y = np.asarray(iv["y"], dtype=float)
    X = np.asarray(iv["X"], dtype=float)
    W = np.asarray(iv["W"], dtype=float)
    names = list(iv["var_names"])
    n, k = X.shape

    for c in (cluster1, cluster2):
        if c not in data.columns or data[c].isna().any() or len(data[c]) != n:
            raise MethodIncompatibility(
                f"Cluster column {c!r} must be present, complete, and row-aligned "
                "to the fitted sample."
            )
    c1 = pd.Categorical(data[cluster1]).codes
    c2 = pd.Categorical(data[cluster2]).codes
    c12 = pd.factorize(pd.Series(list(zip(c1.tolist(), c2.tolist()))))[0]

    P_W = W @ np.linalg.solve(W.T @ W, W.T)
    AX = P_W @ X
    bread = np.linalg.inv(X.T @ AX)
    beta = bread @ (AX.T @ y)
    resid = y - X @ beta

    def _meat(codes: np.ndarray) -> np.ndarray:
        m = np.zeros((k, k))
        for cid in range(int(codes.max()) + 1):
            idx = codes == cid
            mom = (AX[idx] * resid[idx, None]).sum(axis=0)
            m += np.outer(mom, mom)
        return m

    g1 = int(c1.max()) + 1
    g2 = int(c2.max()) + 1
    g12 = int(c12.max()) + 1
    meat = _meat(c1) + _meat(c2) - _meat(c12)
    g_min = min(g1, g2)
    corr = (g_min / (g_min - 1)) * ((n - 1) / (n - k))
    V = corr * bread @ meat @ bread
    se = pd.Series(np.sqrt(np.maximum(np.diag(V), 0)), index=names)
    return {
        "vcov": V,
        "std_errors": se,
        "var_names": names,
        "n_clusters1": g1,
        "n_clusters2": g2,
        "n_clusters12": g12,
    }


def iv_cr_vcov(
    result: Any,
    data: pd.DataFrame,
    cluster: str,
    kind: str = "CR2",
) -> Dict[str, Any]:
    """Bias-reduced cluster-robust covariance for 2SLS (clubSandwich CR2 / CR3).

    Applies the Pustejovsky-Tipton (2018) cluster adjustment on the projected
    2SLS regressors ``Xhat = P_W X`` (with structural residuals ``e = y - Xb``):

        A_g = (I_g - H_g)^{-p},   H_g = Xhat_g (Xhat'X)^{-1} Xhat_g'
        score_g = Xhat_g' A_g e_g,   V = bread (Σ score_g score_g') bread

    with ``p = 1/2`` for **CR2** (Bell-McCaffrey) and ``p = 1`` for **CR3** (the
    jackknife-type adjustment). Both match R ``clubSandwich::vcovCR(ivreg_model,
    type=...)`` to machine precision (see the parity test).
    """
    kind_u = kind.upper()
    power = {"CR2": 0.5, "CR3": 1.0}.get(kind_u)
    if power is None:
        raise MethodIncompatibility("iv_cr_vcov kind must be 'CR2' or 'CR3'.")

    iv = getattr(result, "data_info", {}).get("iv")
    if iv is None:
        raise MethodIncompatibility(
            "iv_cr_vcov requires an sp.ivreg (2SLS) result (data_info['iv'])."
        )
    y = np.asarray(iv["y"], dtype=float)
    X = np.asarray(iv["X"], dtype=float)
    W = np.asarray(iv["W"], dtype=float)
    names = list(iv["var_names"])
    n, k = X.shape

    cl_series = data[cluster]
    if cl_series.isna().any() or len(cl_series) != n:
        raise MethodIncompatibility(
            f"Cluster column {cluster!r} must be complete and row-aligned to the "
            "fitted sample."
        )
    cl_codes = pd.Categorical(cl_series).codes

    x_hat = W @ np.linalg.solve(W.T @ W, W.T) @ X  # P_W X
    bread = np.linalg.inv(x_hat.T @ X)  # (Xhat'X)^-1 = (Xhat'Xhat)^-1
    beta = bread @ (x_hat.T @ y)
    resid = y - X @ beta

    meat = np.zeros((k, k))
    for cid in range(int(cl_codes.max()) + 1):
        idx = cl_codes == cid
        xh_g = x_hat[idx]
        e_g = resid[idx]
        n_g = int(idx.sum())
        i_h = np.eye(n_g) - xh_g @ bread @ xh_g.T
        evals, evecs = np.linalg.eigh(i_h)
        evals = np.maximum(evals, 1e-12)
        a_g = evecs @ np.diag(evals ** (-power)) @ evecs.T
        score = xh_g.T @ (a_g @ e_g)
        meat += np.outer(score, score)

    vcov = bread @ meat @ bread
    se = pd.Series(np.sqrt(np.maximum(np.diag(vcov), 0)), index=names)
    return {"vcov": vcov, "std_errors": se, "var_names": names, "kind": kind_u}


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
    names = list(iv["var_names"])
    if variable not in names:
        raise ValueError(f"Variable '{variable}' not found. Available: {names}")
    test_idx = names.index(variable)
    n_exog = iv["n_exog"]
    endog_cols = list(range(n_exog, n_exog + iv["n_endog"]))
    if test_idx not in endog_cols:
        raise MethodIncompatibility(
            "iv_wild_bootstrap tests an *endogenous* coefficient; got "
            f"'{variable}'. Endogenous regressors: "
            f"{[names[j] for j in endog_cols]}. For exogenous coefficients use "
            "the OLS wild bootstrap."
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
    P_W = W @ np.linalg.solve(W.T @ W, W.T)  # A = P_W for kappa = 1 (2SLS)

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

    # --- restricted estimation under H0: beta_test = beta0 ---
    # Fix the tested coefficient and re-estimate the rest by 2SLS (the *other*
    # endogenous regressors stay endogenous), giving the restricted structural
    # residual u_tilde. With a single endogenous regressor `Xo` is all-exogenous
    # and this 2SLS reduces to OLS.
    y_tilde = y - beta0 * X[:, test_idx]
    other = [j for j in range(k) if j != test_idx]
    Xo = X[:, other]
    AXo = P_W @ Xo
    g = np.linalg.solve(Xo.T @ AXo, AXo.T @ y_tilde)
    u_tilde = y_tilde - Xo @ g

    # --- restricted reduced form for every endogenous regressor ---
    # Efficient: regress each endogenous variable on the instruments AND the
    # restricted structural residual (exploiting the u-v correlation). The
    # instrument-explained part is the bootstrap base; the remainder is
    # wild-resampled with the *same* weight as u_tilde.
    n_w = W.shape[1]
    rf_base: Dict[int, np.ndarray] = {}
    rf_resid: Dict[int, np.ndarray] = {}
    wd = np.column_stack([W, u_tilde]) if efficient else W
    for e in endog_cols:
        if efficient:
            coef = np.linalg.lstsq(wd, X[:, e], rcond=None)[0]
            d_base = W @ coef[:n_w]
        else:
            d_base = P_W @ X[:, e]
        rf_base[e] = d_base
        rf_resid[e] = X[:, e] - d_base

    # --- bootstrap ---
    t_boot = np.empty(n_boot)
    for b in range(n_boot):
        w_g = _draw_weights(n_clusters, weight_type, rng)
        eps = w_g[cl_codes]
        u_star = eps * u_tilde
        x_star = X.copy()
        for e in endog_cols:
            x_star[:, e] = rf_base[e] + eps * rf_resid[e]
        y_star = beta0 * x_star[:, test_idx] + x_star[:, other] @ g + u_star
        beta_s, V_s = _fit(x_star, y_star)
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
