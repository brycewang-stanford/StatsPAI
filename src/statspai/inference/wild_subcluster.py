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
(Treated) Clusters." Econometrics Journal, 21(2), 114-135. [@mackinnon2018wild]

Roodman, D. et al. (2019). "Fast and Wild: Bootstrap Inference in Stata
Using boottest." Stata Journal, 19(1), 4-60. [@roodman2019fast]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


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
    dict with ``p_boot`` (symmetric, ``#{|t*| > |t|} / B`` as in
    ``boottest``), ``ci_boot``, ``beta_hat``, ``t_stat``, ``se_cluster``,
    ``n_subclusters``, ``n_boot`` (``2**S`` when Rademacher weights and
    ``2**S <= n_boot`` trigger full enumeration), ``enumerated`` and
    ``recommendation``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> cl = rng.integers(0, 10, size=n)
    >>> treat = (cl < 3).astype(float)
    >>> x1 = rng.normal(size=n)
    >>> y = 1.0 + 0.4 * treat + 0.5 * x1 + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "treat": treat, "x1": x1, "cl": cl})
    >>> out = sp.subcluster_wild_bootstrap(
    ...     df, y="y", x=["treat", "x1"], cluster="cl",
    ...     test_var="treat", n_boot=199, weight_type="webb", seed=0,
    ... )
    >>> 0.0 <= out["p_boot"] <= 1.0
    True
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
    if G < 2:
        raise ValueError("subcluster wild bootstrap requires at least two clusters")
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
    cluster_scores = np.zeros((G, k))
    np.add.at(cluster_scores, cl_idx, X * resid[:, None])
    meat = cluster_scores.T @ cluster_scores
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

    # Bootstrap: signs flipped per sub-cluster, SE clustered at ``cluster``.
    # Rademacher with 2**S <= n_boot enumerates the full grid, as boottest's
    # bootcluster() option does.
    from .wild_bootstrap import (
        _symmetric_boot_pvalue,
        _wcr_t_stats,
        _wild_weight_matrix,
    )

    W, enumerated = _wild_weight_matrix(S, n_boot, weight_type, rng)
    t_boot = _wcr_t_stats(
        X,
        XtX_inv,
        X @ beta_r,
        resid_r,
        np.asarray(sc_idx, dtype=np.int64),
        cl_idx,
        G,
        j_test,
        h0,
        correction,
        W,
    )

    p_boot = _symmetric_boot_pvalue(t_boot, t_stat, W)
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
        "n_boot": int(W.shape[0]),
        "n_boot_requested": n_boot,
        "enumerated": bool(enumerated),
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
    ``grid_span * se_cluster`` and ``grid_size`` evenly-spaced points. It
    only brackets each endpoint: the bootstrap p-value is a step function
    of the null value, and the bracket is then bisected to machine
    precision to locate the jump where p falls below ``alpha``. With
    Rademacher weights and ``2**G <= n_boot`` the bootstrap grid is
    enumerated, so the interval is exact and reproduces Stata ``boottest``.

    Shares the data / model arguments of :func:`subcluster_wild_bootstrap`;
    the grid-specific parameters are documented below.

    Parameters
    ----------
    grid_size : int
        Number of null-value grid points to evaluate (odd preferred).
    grid_span : float
        Half-width of the search grid in units of cluster-robust SE.

    Returns
    -------
    dict with ``ci``, ``p_grid`` (grid_size,), ``h0_grid``, ``beta_hat``,
    ``se_cluster``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> cl = rng.integers(0, 10, size=n)
    >>> treat = (cl < 3).astype(float)
    >>> x1 = rng.normal(size=n)
    >>> y = 1.0 + 0.4 * treat + 0.5 * x1 + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "treat": treat, "x1": x1, "cl": cl})
    >>> out = sp.wild_cluster_ci_inv(
    ...     df, y="y", x=["treat", "x1"], cluster="cl",
    ...     test_var="treat", n_boot=199, grid_size=21, seed=0,
    ... )
    >>> out["p_grid"].shape
    (21,)
    >>> len(out["ci"])
    2
    """
    # Use the lightweight wild_cluster_bootstrap for point + SE, then grid.
    from .wild_bootstrap import wild_cluster_bootstrap

    base = wild_cluster_bootstrap(
        data,
        y,
        x,
        cluster,
        test_var=test_var,
        h0=0.0,
        n_boot=n_boot,
        weight_type=weight_type,
        seed=seed,
        alpha=alpha,
    )
    beta_hat = base["beta_hat"]
    se_cl = base["se_cluster"]
    if grid_size % 2 == 0:
        grid_size += 1

    def _p_at(h0: float) -> float:
        return float(
            wild_cluster_bootstrap(
                data,
                y,
                x,
                cluster,
                test_var=test_var,
                h0=float(h0),
                n_boot=n_boot,
                weight_type=weight_type,
                seed=seed,
                alpha=alpha,
            )["p_boot"]
        )

    grid = beta_hat + np.linspace(-grid_span, grid_span, grid_size) * se_cl
    p_grid = np.array([_p_at(h0) for h0 in grid])

    # The bootstrap p-value is a step function of h0 (it only changes when
    # |t*_b(h0)| crosses |t(h0)|), so the interval endpoint is the location of
    # a jump, not a point where p equals alpha. Bracket the first crossing
    # outward from the estimate on the grid, then bisect the bracket down to
    # machine precision -- the search Stata ``boottest`` / R
    # ``fwildclusterboot`` perform. (Linear interpolation of p between grid
    # points, used before, placed the endpoint anywhere inside a bracket that
    # is ``2 * grid_span / (grid_size - 1)`` cluster SEs wide.)
    def _endpoint(h_arr: np.ndarray, p_arr: np.ndarray) -> Optional[float]:
        inside = p_arr >= alpha
        if not inside[0]:
            return None
        out_idx = np.where(~inside)[0]
        if out_idx.size == 0:
            return None
        j = int(out_idx[0])
        h_in, h_out = float(h_arr[j - 1]), float(h_arr[j])
        for _ in range(200):
            mid = 0.5 * (h_in + h_out)
            if mid in (h_in, h_out):
                break
            if _p_at(mid) >= alpha:
                h_in = mid
            else:
                h_out = mid
        return 0.5 * (h_in + h_out)

    mid = grid_size // 2
    lo_candidate = _endpoint(grid[: mid + 1][::-1], p_grid[: mid + 1][::-1])
    hi_candidate = _endpoint(grid[mid:], p_grid[mid:])
    if lo_candidate is None or hi_candidate is None:
        import warnings

        warnings.warn(
            "wild_cluster_ci_inv: the bootstrap p-value does not fall below "
            f"alpha={alpha} inside the search grid on "
            f"{'both sides' if lo_candidate is None and hi_candidate is None else ('the lower side' if lo_candidate is None else 'the upper side')}"
            "; the reported bound is the grid edge, not an inverted "
            "endpoint. Increase grid_span.",
            UserWarning,
            stacklevel=2,
        )
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
        vals = np.array(
            [
                -np.sqrt(1.5),
                -np.sqrt(1.0),
                -np.sqrt(0.5),
                np.sqrt(0.5),
                np.sqrt(1.0),
                np.sqrt(1.5),
            ]
        )
        return rng.choice(vals, size=S)
    if weight_type == "mammen":
        p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        vals = np.array([-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2])
        return rng.choice(vals, size=S, p=[p, 1 - p])
    raise ValueError(f"Unknown weight_type: {weight_type}")


__all__ = ["subcluster_wild_bootstrap", "wild_cluster_ci_inv"]
