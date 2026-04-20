"""
Subcluster wild bootstrap for few-treated-clusters settings.

When a treatment is assigned at the cluster level and the number of
treated clusters is small (<= 5), the ordinary wild cluster bootstrap
under-rejects dramatically (MacKinnon-Webb 2018). The subcluster wild
bootstrap breaks each cluster into smaller sub-clusters (e.g., individual
observations or finer groupings) before bootstrapping — this expands the
effective number of resamples and restores correct size.

The **subcluster WCR** (restricted) variant imposes the null hypothesis
on residuals before bootstrapping, following the WCR recommendation from
Cameron, Gelbach & Miller (2008). Signs are flipped at the *subcluster*
level rather than cluster level.

References
----------
MacKinnon, J.G. and Webb, M.D. (2018). "The Wild Bootstrap for Few
(Treated) Clusters." Econometrics Journal, 21(2), 114-135.

Roodman, D. et al. (2019). "Fast and Wild: Bootstrap Inference in Stata
Using boottest." Stata Journal, 19(1), 4-60.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


def subcluster_wild_bootstrap(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    cluster: str,
    subcluster: Optional[str] = None,
    test_var: Optional[str] = None,
    h0: float = 0.0,
    n_boot: int = 999,
    weight_type: str = "webb",
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Subcluster wild cluster bootstrap for few-treated-clusters.

    When treatment varies within cluster (or when you want to re-expand
    the randomization at a finer grain), sign-flips happen at the
    sub-cluster level. SEs are still computed clustered at the coarse
    ``cluster`` level.

    Parameters
    ----------
    data : DataFrame
    y : str
    x : list of str
    cluster : str
        Primary cluster column (for SE computation).
    subcluster : str or None
        Finer grouping at which sign-flips occur. If ``None``, every
        observation is its own subcluster (pure Rademacher at obs level).
    test_var : str or None
        Parameter to test; default last element of ``x``.
    h0 : float
        Null value.
    n_boot : int
        Bootstrap replications.
    weight_type : {'rademacher', 'webb', 'mammen'}
        Distribution of sign flips. ``'webb'`` (6-point) recommended
        when treatment has <= 5 treated clusters.
    seed : int, optional
    alpha : float

    Returns
    -------
    dict with ``p_boot``, ``ci_boot``, ``beta_hat``, ``t_stat``,
    ``se_cluster``, ``n_sub``, ``recommendation``.
    """
    if test_var is None:
        test_var = x[-1]
    if test_var not in x:
        raise ValueError(f"test_var '{test_var}' not in x")

    rng = np.random.default_rng(seed)
    cols = [y] + list(x) + [cluster] + ([subcluster] if subcluster else [])
    df = data[cols].dropna().copy()
    Y = df[y].values.astype(float)
    X = np.column_stack([np.ones(len(df)), df[x].values.astype(float)])
    var_names = ["_const"] + list(x)
    j_test = var_names.index(test_var)

    cl = df[cluster].values
    unique_cl, cl_idx = np.unique(cl, return_inverse=True)
    G = len(unique_cl)
    if subcluster is not None:
        sc = df[subcluster].values
        sc_labels, sc_idx = np.unique(sc, return_inverse=True)
    else:
        sc_labels = np.arange(len(df))
        sc_idx = sc_labels
    S = len(sc_labels)
    n, k = X.shape

    # Unrestricted OLS
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ X.T @ Y
    resid = Y - X @ beta_hat
    beta_test = beta_hat[j_test]

    # Cluster-robust SE (Liang-Zeger CR1)
    correction = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    meat = np.zeros((k, k))
    for g in range(G):
        m = cl_idx == g
        u_g = X[m].T @ resid[m]
        meat += np.outer(u_g, u_g)
    V_cl = correction * XtX_inv @ meat @ XtX_inv
    se_cl = float(np.sqrt(V_cl[j_test, j_test]))
    t_stat = (beta_test - h0) / se_cl if se_cl > 0 else 0.0

    # Restricted residuals (impose H0: β_test = h0)
    Y_tilde = Y - h0 * X[:, j_test]
    other = [i for i in range(k) if i != j_test]
    X_o = X[:, other]
    b_o = np.linalg.lstsq(X_o, Y_tilde, rcond=None)[0]
    beta_r = np.zeros(k)
    for ii, jj in enumerate(other):
        beta_r[jj] = b_o[ii]
    beta_r[j_test] = h0
    resid_r = Y - X @ beta_r

    # Bootstrap
    t_boot = np.empty(n_boot)
    for b in range(n_boot):
        w = _draw_weights(S, weight_type, rng)
        w_obs = w[sc_idx]
        Y_star = X @ beta_r + w_obs * resid_r
        beta_b = XtX_inv @ X.T @ Y_star
        resid_b = Y_star - X @ beta_b
        meat_b = np.zeros((k, k))
        for g in range(G):
            m = cl_idx == g
            u_b = X[m].T @ resid_b[m]
            meat_b += np.outer(u_b, u_b)
        V_b = correction * XtX_inv @ meat_b @ XtX_inv
        se_b = np.sqrt(max(V_b[j_test, j_test], 1e-20))
        t_boot[b] = (beta_b[j_test] - h0) / se_b

    p_boot = float(np.mean(np.abs(t_boot) >= np.abs(t_stat)))
    t_lo = np.percentile(t_boot, 100 * alpha / 2)
    t_hi = np.percentile(t_boot, 100 * (1 - alpha / 2))
    ci = (beta_test - t_hi * se_cl, beta_test - t_lo * se_cl)

    rec = (
        f"Subcluster WCR bootstrap on S={S} sub-clusters within G={G} clusters. "
        f"Weights='{weight_type}'. Use when treated clusters < 5."
    )
    return {
        "beta_hat": float(beta_test),
        "se_cluster": se_cl,
        "t_stat": float(t_stat),
        "p_boot": p_boot,
        "ci_boot": ci,
        "n_clusters": G,
        "n_subclusters": S,
        "n_boot": n_boot,
        "weight_type": weight_type,
        "recommendation": rec,
    }


def wild_cluster_ci_inv(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    cluster: str,
    test_var: Optional[str] = None,
    n_boot: int = 999,
    weight_type: str = "webb",
    alpha: float = 0.05,
    grid_size: int = 41,
    grid_span: float = 6.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Confidence interval via bootstrap p-value inversion.

    Runs the wild cluster bootstrap repeatedly across a grid of null
    values and finds the boundary where the two-sided bootstrap p-value
    equals ``alpha``. This yields a WCR-inverted CI with better
    small-cluster coverage than the percentile-t CI.

    The grid is centered on the OLS point estimate with half-width
    ``grid_span * se_cluster`` and ``grid_size`` evenly-spaced points.
    Linear interpolation refines the boundary.

    Parameters
    ----------
    See :func:`subcluster_wild_bootstrap`.
    grid_size : int
        Number of null-value grid points to evaluate (odd preferred).
    grid_span : float
        Half-width of the search grid in units of cluster-robust SE.

    Returns
    -------
    dict with ``ci``, ``p_grid`` (grid_size,), ``h0_grid``, ``beta_hat``,
    ``se_cluster``.
    """
    # Use the lightweight wild_cluster_bootstrap for point + SE, then grid.
    from .wild_bootstrap import wild_cluster_bootstrap

    base = wild_cluster_bootstrap(
        data, y, x, cluster, test_var=test_var, h0=0.0,
        n_boot=n_boot, weight_type=weight_type, seed=seed, alpha=alpha,
    )
    beta_hat = base["beta_hat"]
    se_cl = base["se_cluster"]
    if grid_size % 2 == 0:
        grid_size += 1

    grid = beta_hat + np.linspace(-grid_span, grid_span, grid_size) * se_cl
    p_grid = np.empty(grid_size)
    for i, h0 in enumerate(grid):
        res = wild_cluster_bootstrap(
            data, y, x, cluster, test_var=test_var, h0=float(h0),
            n_boot=n_boot, weight_type=weight_type, seed=seed, alpha=alpha,
        )
        p_grid[i] = res["p_boot"]

    # Boundary: where p crosses alpha
    def _cross(h_arr: np.ndarray, p_arr: np.ndarray, level: float) -> Optional[float]:
        sign = np.sign(p_arr - level)
        idx = np.where(np.diff(sign) != 0)[0]
        if idx.size == 0:
            return None
        # Linear interpolation on the first crossing
        i = int(idx[0])
        x0, x1 = h_arr[i], h_arr[i + 1]
        y0, y1 = p_arr[i] - level, p_arr[i + 1] - level
        if y1 == y0:
            return float((x0 + x1) / 2)
        return float(x0 - y0 * (x1 - x0) / (y1 - y0))

    mid = grid_size // 2
    lo_candidate = _cross(grid[: mid + 1][::-1], p_grid[: mid + 1][::-1], alpha)
    hi_candidate = _cross(grid[mid:], p_grid[mid:], alpha)
    ci = (
        lo_candidate if lo_candidate is not None else float(grid[0]),
        hi_candidate if hi_candidate is not None else float(grid[-1]),
    )

    return {
        "beta_hat": beta_hat,
        "se_cluster": se_cl,
        "ci": ci,
        "h0_grid": grid,
        "p_grid": p_grid,
        "alpha": alpha,
        "method": "WCR p-value inversion",
    }


def _draw_weights(S: int, weight_type: str, rng: np.random.Generator) -> np.ndarray:
    """Draw S i.i.d. bootstrap weights."""
    if weight_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=S)
    if weight_type == "webb":
        vals = np.array([
            -np.sqrt(1.5), -np.sqrt(1.0), -np.sqrt(0.5),
            np.sqrt(0.5), np.sqrt(1.0), np.sqrt(1.5),
        ])
        return rng.choice(vals, size=S)
    if weight_type == "mammen":
        p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        vals = np.array([-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2])
        return rng.choice(vals, size=S, p=[p, 1 - p])
    raise ValueError(f"Unknown weight_type: {weight_type}")


__all__ = ["subcluster_wild_bootstrap", "wild_cluster_ci_inv"]
