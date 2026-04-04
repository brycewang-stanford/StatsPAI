"""
Wild Cluster Bootstrap for cluster-robust inference.

When the number of clusters is small (< ~50), conventional cluster-robust
standard errors are unreliable. The wild cluster bootstrap provides valid
inference by resampling cluster-level residuals with random sign flips.

This implementation follows the WCR (Wild Cluster Restricted) bootstrap,
which imposes the null hypothesis for better finite-sample performance.

References
----------
Cameron, A.C., Gelbach, J.B. and Miller, D.L. (2008).
"Bootstrap-Based Improvements for Inference with Clustered Errors."
*Review of Economics and Statistics*, 90(3), 414-427.

Webb, M.D. (2014).
"Reworking Wild Bootstrap Based Inference for Clustered Errors."
*Queen's Economics Department Working Paper* No. 1315.

Roodman, D., Nielsen, M.Ø., MacKinnon, J.G. and Webb, M.D. (2019).
"Fast and Wild: Bootstrap Inference in Stata Using boottest."
*Stata Journal*, 19(1), 4-60.

MacKinnon, J.G. and Webb, M.D. (2018).
"The Wild Bootstrap for Few (Treated) Clusters."
*The Econometrics Journal*, 21(2), 114-135.
"""

from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats


def wild_cluster_bootstrap(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    cluster: str,
    test_var: Optional[str] = None,
    h0: float = 0,
    n_boot: int = 999,
    weight_type: str = 'rademacher',
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Wild Cluster Bootstrap p-value and confidence interval.

    Provides valid inference when the number of clusters is small,
    where conventional cluster-robust standard errors are unreliable.

    Implements the WCR (Wild Cluster Restricted) bootstrap from
    Cameron, Gelbach & Miller (2008), with Rademacher weights
    (default) or Webb (2014) 6-point weights for very few clusters.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    x : list of str
        All regressors (including the variable being tested).
    cluster : str
        Cluster variable name.
    test_var : str, optional
        Variable to test. Default: last variable in ``x``.
    h0 : float, default 0
        Null hypothesis value for the test variable coefficient.
    n_boot : int, default 999
        Number of bootstrap replications. Use odd number for exact
        p-value computation.
    weight_type : str, default 'rademacher'
        Bootstrap weight distribution:
        - ``'rademacher'``: ±1 with equal probability. Standard choice.
        - ``'webb'``: 6-point distribution from Webb (2014).
          Recommended when G < 12 clusters.
        - ``'mammen'``: Mammen (1993) 2-point distribution.
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for the confidence interval.

    Returns
    -------
    dict
        Keys:
        - ``beta_hat``: OLS point estimate of test_var
        - ``se_cluster``: conventional cluster-robust SE
        - ``t_stat``: t-statistic under H0
        - ``p_boot``: wild cluster bootstrap p-value (two-sided)
        - ``ci_boot``: bootstrap percentile-t confidence interval
        - ``n_clusters``: number of clusters
        - ``n_boot``: number of replications used
        - ``weight_type``: weight distribution used
        - ``recommendation``: human-readable guidance

    Examples
    --------
    >>> result = wild_cluster_bootstrap(
    ...     df, y='wage', x=['education', 'experience'],
    ...     cluster='state', test_var='education')
    >>> print(f"Bootstrap p = {result['p_boot']:.4f}")
    >>> print(f"Bootstrap CI = {result['ci_boot']}")

    Notes
    -----
    The WCR bootstrap imposes the null H0: β_test = h0 when generating
    bootstrap samples. This yields better size control than the
    unrestricted wild bootstrap, especially with few clusters.

    For G < 12 clusters, Webb (2014) weights are recommended over
    Rademacher. The function will auto-warn if Rademacher is used
    with very few clusters.

    See Cameron, Gelbach & Miller (2008, *REStat*) Section III for
    the full algorithm description.
    """
    import warnings

    if test_var is None:
        test_var = x[-1]
    if test_var not in x:
        raise ValueError(f"test_var '{test_var}' must be in x")

    rng = np.random.default_rng(seed)

    # --- Prepare data ---
    cols = [y] + x + [cluster]
    df = data[cols].dropna()
    Y = df[y].values.astype(float)
    X_df = df[x]
    X = np.column_stack([np.ones(len(df)), X_df.values.astype(float)])
    var_names = ['_const'] + list(x)
    test_idx = var_names.index(test_var)
    cl = df[cluster].values
    unique_cl = np.unique(cl)
    G = len(unique_cl)
    n = len(Y)
    k = X.shape[1]

    if G < 6:
        warnings.warn(
            f"Only {G} clusters. Wild cluster bootstrap results may be "
            f"unreliable with fewer than 6 clusters. Consider Webb (2014) "
            f"weights (weight_type='webb').",
            UserWarning,
        )

    # --- Unrestricted OLS ---
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ Y
    resid = Y - X @ beta_hat
    beta_test = beta_hat[test_idx]

    # --- Cluster-robust SE (CR1) ---
    meat = np.zeros((k, k))
    for g in unique_cl:
        idx = cl == g
        score_g = X[idx].T @ resid[idx]
        meat += np.outer(score_g, score_g)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    vcov_cl = correction * XtX_inv @ meat @ XtX_inv
    se_cl = float(np.sqrt(vcov_cl[test_idx, test_idx]))
    t_stat = (beta_test - h0) / se_cl if se_cl > 0 else 0

    # --- Restricted OLS (impose H0: ��_test = h0) ---
    # Regress Y - h0*X_test on all OTHER regressors (excluding test_var)
    Y_tilde = Y - h0 * X[:, test_idx]
    other_cols = [j for j in range(k) if j != test_idx]
    X_other = X[:, other_cols]
    beta_other = np.linalg.lstsq(X_other, Y_tilde, rcond=None)[0]
    # Reconstruct full beta_r with the constraint β_test = h0
    beta_r = np.zeros(k)
    for i, j in enumerate(other_cols):
        beta_r[j] = beta_other[i]
    beta_r[test_idx] = h0
    resid_r = Y - X @ beta_r

    # --- Bootstrap ---
    t_boot = np.zeros(n_boot)

    for b in range(n_boot):
        # Draw cluster-level weights
        w = _draw_weights(G, weight_type, rng)

        # Map cluster weights to observations
        Y_star = np.zeros(n)
        for g_idx, g_val in enumerate(unique_cl):
            obs_idx = cl == g_val
            # WCR: Y* = X β_r + w_g * ε̂_r
            Y_star[obs_idx] = (X[obs_idx] @ beta_r
                               + w[g_idx] * resid_r[obs_idx])

        # OLS on bootstrap sample
        beta_b = XtX_inv @ X.T @ Y_star
        resid_b = Y_star - X @ beta_b

        # Cluster-robust SE for bootstrap sample
        meat_b = np.zeros((k, k))
        for g_idx, g_val in enumerate(unique_cl):
            idx = cl == g_val
            score_g = X[idx].T @ resid_b[idx]
            meat_b += np.outer(score_g, score_g)
        vcov_b = correction * XtX_inv @ meat_b @ XtX_inv
        se_b = np.sqrt(max(vcov_b[test_idx, test_idx], 1e-20))

        t_boot[b] = (beta_b[test_idx] - h0) / se_b

    # --- Bootstrap p-value (two-sided) ---
    p_boot = float(np.mean(np.abs(t_boot) >= np.abs(t_stat)))

    # --- Percentile-t confidence interval ---
    t_lower = np.percentile(t_boot, 100 * alpha / 2)
    t_upper = np.percentile(t_boot, 100 * (1 - alpha / 2))
    ci_boot = (
        beta_test - t_upper * se_cl,
        beta_test - t_lower * se_cl,
    )

    # --- Recommendation ---
    if G < 12 and weight_type == 'rademacher':
        rec = (f"Warning: {G} clusters with Rademacher weights. "
               f"Consider weight_type='webb' per Webb (2014).")
    elif G < 20:
        rec = (f"{G} clusters — wild cluster bootstrap is appropriate. "
               f"Conventional cluster SEs may over-reject.")
    else:
        rec = (f"{G} clusters — both conventional cluster SE and bootstrap "
               f"should be reliable.")

    return {
        'beta_hat': float(beta_test),
        'se_cluster': se_cl,
        't_stat': float(t_stat),
        'p_boot': p_boot,
        'p_cluster': float(2 * (1 - stats.t.cdf(abs(t_stat), G - 1))),
        'ci_boot': ci_boot,
        'n_clusters': G,
        'n_obs': n,
        'n_boot': n_boot,
        'weight_type': weight_type,
        'recommendation': rec,
    }


# ======================================================================
# Weight distributions
# ======================================================================

from typing import List


def _draw_weights(
    G: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw G bootstrap weights from the specified distribution."""
    if weight_type == 'rademacher':
        # ±1 with equal probability (Cameron et al. 2008 default)
        return rng.choice([-1.0, 1.0], size=G)

    elif weight_type == 'webb':
        # Webb (2014) 6-point distribution
        # Better for very few clusters (G < 12)
        # Values: ±sqrt(3/2), ±sqrt(2/2), ±sqrt(1/2)
        # each with probability 1/6
        vals = np.array([
            -np.sqrt(1.5), -np.sqrt(1.0), -np.sqrt(0.5),
             np.sqrt(0.5),  np.sqrt(1.0),  np.sqrt(1.5),
        ])
        return rng.choice(vals, size=G)

    elif weight_type == 'mammen':
        # Mammen (1993) 2-point distribution
        # w = -(sqrt(5)-1)/2 with prob (sqrt(5)+1)/(2*sqrt(5))
        #     (sqrt(5)+1)/2  with prob (sqrt(5)-1)/(2*sqrt(5))
        p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        vals = np.array([-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2])
        return rng.choice(vals, size=G, p=[p, 1 - p])

    raise ValueError(
        f"weight_type must be 'rademacher', 'webb', or 'mammen', "
        f"got '{weight_type}'"
    )


# ======================================================================
# Citation
# ======================================================================

from ..core.results import CausalResult

CausalResult._CITATIONS['wild_cluster_bootstrap'] = (
    "@article{cameron2008bootstrap,\n"
    "  title={Bootstrap-Based Improvements for Inference with "
    "Clustered Errors},\n"
    "  author={Cameron, A. Colin and Gelbach, Jonah B. and Miller, "
    "Douglas L.},\n"
    "  journal={Review of Economics and Statistics},\n"
    "  volume={90},\n"
    "  number={3},\n"
    "  pages={414--427},\n"
    "  year={2008},\n"
    "  publisher={MIT Press}\n"
    "}"
)
