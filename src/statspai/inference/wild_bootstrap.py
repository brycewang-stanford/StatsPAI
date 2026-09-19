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
*Review of Economics and Statistics*, 90(3), 414-427. [@cameron2008bootstrap]

Webb, M.D. (2014).
"Reworking Wild Bootstrap Based Inference for Clustered Errors."
*Queen's Economics Department Working Paper* No. 1315.

Roodman, D., Nielsen, M.Ø., MacKinnon, J.G. and Webb, M.D. (2019).
"Fast and Wild: Bootstrap Inference in Stata Using boottest."
*Stata Journal*, 19(1), 4-60. [@roodman2019fast]

MacKinnon, J.G. and Webb, M.D. (2018).
"The Wild Bootstrap for Few (Treated) Clusters."
*The Econometrics Journal*, 21(2), 114-135. [@mackinnon2018wild]
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple

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
    weight_type: str = "rademacher",
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
        Number of bootstrap replications. With Rademacher weights and
        ``2**G <= n_boot`` the full grid of ``2**G`` sign vectors is
        enumerated instead (the rule Stata ``boottest`` and R
        ``fwildclusterboot`` apply), so the p-value is exact and ``n_boot``
        in the result reports ``2**G``.
    weight_type : str, default 'rademacher'
        Bootstrap weight distribution:
        - ``'rademacher'``: ±1 with equal probability. Standard choice.
        - ``'webb'``: 6-point distribution from Webb (2014).
          Recommended when G < 12 clusters.
        - ``'mammen'``: Mammen (1993) 2-point distribution.
    seed : int, optional
        Random seed for reproducibility (unused when the grid is enumerated).
    alpha : float, default 0.05
        Significance level for the confidence interval.

    Returns
    -------
    dict
        Keys:
        - ``beta_hat``: OLS point estimate of test_var
        - ``se_cluster``: conventional cluster-robust SE
        - ``t_stat``: t-statistic under H0
        - ``p_boot``: symmetric two-sided bootstrap p-value,
          ``#{|t*| > |t|} / B`` (strict inequality, as in ``boottest``)
        - ``ci_boot``: bootstrap percentile-t confidence interval (for the
          test-inversion interval ``boottest`` reports use
          :func:`sp.wild_cluster_ci_inv`)
        - ``n_clusters``: number of clusters
        - ``n_boot``: number of replications used (``2**G`` if enumerated)
        - ``n_boot_requested``: the ``n_boot`` argument
        - ``enumerated``: whether the full Rademacher grid was used
        - ``weight_type``: weight distribution used
        - ``recommendation``: human-readable guidance

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> education = rng.integers(8, 18, n)
    >>> experience = rng.integers(0, 30, n)
    >>> wage = (2.0 + 0.4 * education + 0.1 * experience
    ...         + rng.normal(0, 2, n))
    >>> df = pd.DataFrame({'wage': wage, 'education': education,
    ...                    'experience': experience,
    ...                    'state': rng.integers(0, 12, n)})
    >>> result = sp.wild_cluster_bootstrap(
    ...     df, y='wage', x=['education', 'experience'],
    ...     cluster='state', test_var='education', n_boot=199, seed=0)
    >>> print(f"Bootstrap p = {result['p_boot']:.4f}")  # doctest: +SKIP
    >>> bool(result['ci_boot'][0] < result['ci_boot'][1])
    True

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
    var_names = ["_const"] + list(x)
    test_idx = var_names.index(test_var)
    cl = df[cluster].values
    unique_cl, cl_inverse = np.unique(cl, return_inverse=True)
    G = len(unique_cl)
    n = len(Y)
    k = X.shape[1]

    if G < 2:
        raise ValueError("wild cluster bootstrap requires at least two clusters")

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
    cluster_scores = np.zeros((G, k))
    np.add.at(cluster_scores, cl_inverse, X * resid[:, None])
    meat = cluster_scores.T @ cluster_scores
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
    # WCR: Y* = X beta_r + w_g * e_r, w drawn per cluster. With Rademacher
    # weights and 2**G <= n_boot the full sign grid is enumerated (boottest /
    # fwildclusterboot rule), which makes the bootstrap distribution exact.
    W, enumerated = _wild_weight_matrix(G, n_boot, weight_type, rng)
    t_boot = _wcr_t_stats(
        X,
        XtX_inv,
        X @ beta_r,
        resid_r,
        cl_inverse,
        cl_inverse,
        G,
        test_idx,
        h0,
        correction,
        W,
    )

    # --- Bootstrap p-value (two-sided, symmetric) ---
    p_boot = _symmetric_boot_pvalue(t_boot, t_stat, W)

    # --- Percentile-t confidence interval ---
    t_lower = np.percentile(t_boot, 100 * alpha / 2)
    t_upper = np.percentile(t_boot, 100 * (1 - alpha / 2))
    ci_boot = (
        beta_test - t_upper * se_cl,
        beta_test - t_lower * se_cl,
    )

    # --- Recommendation ---
    if G < 12 and weight_type == "rademacher":
        rec = (
            f"Warning: {G} clusters with Rademacher weights. "
            f"Consider weight_type='webb' per Webb (2014)."
        )
    elif G < 20:
        rec = (
            f"{G} clusters — wild cluster bootstrap is appropriate. "
            f"Conventional cluster SEs may over-reject."
        )
    else:
        rec = (
            f"{G} clusters — both conventional cluster SE and bootstrap "
            f"should be reliable."
        )

    return {
        "beta_hat": float(beta_test),
        "se_cluster": se_cl,
        "t_stat": float(t_stat),
        "p_boot": p_boot,
        "p_cluster": float(2 * stats.t.sf(abs(t_stat), G - 1)),
        "ci_boot": ci_boot,
        "n_clusters": G,
        "n_obs": n,
        "n_boot": int(W.shape[0]),
        "n_boot_requested": n_boot,
        "enumerated": bool(enumerated),
        "weight_type": weight_type,
        "recommendation": rec,
    }


# ======================================================================
# Shared WCR engine (also used by wild_cluster_boot and the subcluster
# bootstrap)
# ======================================================================

# Largest number of bootstrap units for which the Rademacher grid is
# enumerated (2**20 ~ 1e6 draws). Above this the request can never satisfy
# 2**G <= n_boot in practice anyway.
_MAX_ENUMERATE_UNITS = 20


def _wild_weight_matrix(
    n_units: int,
    n_boot: int,
    weight_type: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, bool]:
    """Bootstrap weights, shape ``(B, n_units)``, and whether they are enumerated.

    Follows Stata ``boottest`` and R ``fwildclusterboot``: when the weights are
    Rademacher and ``2**n_units <= n_boot`` every one of the ``2**n_units``
    sign vectors is used exactly once (so ``B = 2**n_units``), which makes the
    bootstrap distribution -- and hence the p-value -- exact and identical
    across implementations. Otherwise ``n_boot`` draws are sampled with
    ``rng``, one weight vector per draw, in the same order as before.
    """
    if (
        weight_type == "rademacher"
        and n_units <= _MAX_ENUMERATE_UNITS
        and 2**n_units <= n_boot
    ):
        grid = np.array(
            list(itertools.product((1.0, -1.0), repeat=n_units)), dtype=float
        )
        return grid, True
    W = np.empty((n_boot, n_units))
    for b in range(n_boot):
        W[b] = _draw_weights(n_units, weight_type, rng)
    return W, False


def _wcr_t_stats(
    X: np.ndarray,
    XtX_inv: np.ndarray,
    fitted_r: np.ndarray,
    resid_r: np.ndarray,
    boot_idx: np.ndarray,
    err_idx: np.ndarray,
    n_err: int,
    test_idx: int,
    h0: float,
    correction: float,
    W: np.ndarray,
    chunk: int = 2048,
) -> np.ndarray:
    """Bootstrap t statistics of the restricted wild (cluster) bootstrap.

    ``Y*_b = fitted_r + W[b, boot_idx] * resid_r``; each draw is refitted by
    OLS and studentised by the CR1 cluster-robust SE computed over the
    *error* clusters ``err_idx`` (which may be coarser than the bootstrap
    clusters ``boot_idx``, as in the subcluster bootstrap). Vectorised over
    draws; numerically the same computation as a per-draw loop.
    """
    n = X.shape[0]
    a = XtX_inv[test_idx]  # row of (X'X)^-1 for the tested coefficient
    xa = X @ a  # beta_b[test] = xa' Y*
    onehot = np.zeros((n, n_err))
    onehot[np.arange(n), err_idx] = 1.0
    P = X @ XtX_inv  # (n, k); beta_b = Y* @ P
    out = np.empty(W.shape[0])
    for start in range(0, W.shape[0], chunk):
        Wc = W[start : start + chunk]
        Y_star = fitted_r[None, :] + Wc[:, boot_idx] * resid_r[None, :]
        beta_b = Y_star @ P
        resid_b = Y_star - beta_b @ X.T
        scores = (resid_b * xa[None, :]) @ onehot  # (B, n_err)
        var_b = correction * np.sum(scores**2, axis=1)
        se_b = np.sqrt(np.maximum(var_b, 1e-20))
        out[start : start + chunk] = (beta_b[:, test_idx] - h0) / se_b
    return out


def _symmetric_boot_pvalue(
    t_boot: np.ndarray, t_obs: float, W: Optional[np.ndarray] = None
) -> float:
    """Symmetric two-sided bootstrap p-value, ``#{|t*| > |t|} / B``.

    Strict inequality, as in Stata ``boottest`` and R ``fwildclusterboot``
    (``mean(abs(t) < abs(t_boot))``). Ties matter: under full Rademacher
    enumeration the identity draw ``w = 1`` and its negation ``w = -1``
    reproduce ``|t|`` exactly, so counting ties (``>=``) inflates p by
    ``2 / 2**G``. In floating point the refit of those two draws reproduces
    ``|t|`` only up to round-off, so they are identified from ``W`` (every
    weight equal to +1, or every weight equal to -1) and never counted,
    instead of through a tolerance on ``t`` -- a tolerance would move the
    location of every jump of p as a function of the null value, which is
    what the test-inversion interval of :func:`wild_cluster_ci_inv` solves
    for.
    """
    exceed = np.abs(t_boot) > abs(float(t_obs))
    if W is not None:
        tie = np.all(W == 1.0, axis=1) | np.all(W == -1.0, axis=1)
        exceed &= ~tie
    return float(np.mean(exceed))


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
    if weight_type == "rademacher":
        # ±1 with equal probability (Cameron et al. 2008 default)
        return rng.choice([-1.0, 1.0], size=G)

    elif weight_type == "webb":
        # Webb (2014) 6-point distribution
        # Better for very few clusters (G < 12)
        # Values: ±sqrt(3/2), ±sqrt(2/2), ±sqrt(1/2)
        # each with probability 1/6
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
        return rng.choice(vals, size=G)

    elif weight_type == "mammen":
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

CausalResult._CITATIONS["wild_cluster_bootstrap"] = (
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
