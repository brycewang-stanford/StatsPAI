"""
Inverse Probability Weighting (IPW) estimator for ATE / ATT / ATC.

Standalone IPW without outcome-model augmentation (for augmented/doubly-robust,
see :func:`statspai.inference.aipw`).

Implements the Horvitz-Thompson estimator with logistic propensity scores
and optional trimming, normalized weights, and bootstrapped standard errors.

References
----------
Horvitz, D.G. and Thompson, D.J. (1952).
"A Generalization of Sampling Without Replacement From a Finite Universe."
*Journal of the American Statistical Association*, 47(260), 663-685. [@horvitz1952generalization]

Hirano, K., Imbens, G.W. and Ridder, G. (2003).
"Efficient Estimation of Average Treatment Effects Using the Estimated
Propensity Score."
*Econometrica*, 71(4), 1161-1189. [@hirano2003efficient]

Crump, R.K., Hotz, V.J., Imbens, G.W. and Mitnik, O.A. (2009).
"Dealing with Limited Overlap in Estimation of Average Treatment Effects."
*Biometrika*, 96(1), 187-199. [@crump2009dealing]
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


def ipw(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: str = "ATE",
    trim: float = 0.0,
    normalize: bool = True,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Inverse Probability Weighting estimator for treatment effects.

    Estimates ATE, ATT, or ATC by weighting observations by the inverse
    of their propensity to receive the treatment they actually received.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment indicator (0/1).
    covariates : list of str
        Variables for the propensity score model (logistic regression).
    estimand : str, default 'ATE'
        'ATE' (average treatment effect), 'ATT' (on treated),
        or 'ATC' (on controls).
    trim : float, default 0.0
        Trim propensity scores to [trim, 1 - trim]. Common choices: 0.01, 0.05, 0.1.
        Crump et al. (2009) recommend dropping units with p outside [0.1, 0.9].
    normalize : bool, default True
        If True, use Hajek (normalised) weights. Generally recommended
        for finite-sample stability.
    n_bootstrap : int, default 500
        Number of bootstrap iterations for standard error estimation.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    CausalResult
        With `.estimate`, `.se`, `.ci`, `.pvalue`, and propensity score
        diagnostics in `.model_info`.

    Examples
    --------
    >>> result = sp.ipw(df, y='wage', treat='training',
    ...                 covariates=['age', 'education', 'experience'])
    >>> print(result.summary())

    >>> # ATT with trimming
    >>> result = sp.ipw(df, y='wage', treat='training',
    ...                 covariates=['age', 'education'],
    ...                 estimand='ATT', trim=0.05)
    """
    estimand = estimand.upper()
    if estimand not in ("ATE", "ATT", "ATC"):
        raise ValueError(f"estimand must be 'ATE', 'ATT', or 'ATC', got '{estimand}'")

    rng = np.random.RandomState(seed)

    # --- Prepare data ---
    df = data[[y, treat] + covariates].dropna().copy()
    Y = df[y].values.astype(np.float64)
    T = df[treat].values.astype(np.float64)
    X = df[covariates].values.astype(np.float64)
    n = len(Y)

    if not set(np.unique(T)).issubset({0, 1}):
        raise ValueError(f"Treatment variable '{treat}' must be binary (0/1)")

    # --- Estimate propensity scores ---
    pscore = _estimate_propensity(X, T)

    # --- Trim ---
    if trim > 0:
        pscore = np.clip(pscore, trim, 1 - trim)

    # --- Compute weights ---
    weights_1, weights_0 = _compute_weights(T, pscore, estimand, normalize)

    # --- Point estimate ---
    estimate = float(np.sum(weights_1 * Y) - np.sum(weights_0 * Y))

    # --- Bootstrap SE ---
    boot_estimates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        Y_b, T_b, X_b = Y[idx], T[idx], X[idx]
        ps_b = _estimate_propensity(X_b, T_b)
        if trim > 0:
            ps_b = np.clip(ps_b, trim, 1 - trim)
        w1, w0 = _compute_weights(T_b, ps_b, estimand, normalize)
        boot_estimates[b] = np.sum(w1 * Y_b) - np.sum(w0 * Y_b)

    se = float(np.std(boot_estimates, ddof=1))
    t_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - t_crit * se, estimate + t_crit * se)
    pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(estimate / se)))) if se > 0 else 1.0

    # --- Diagnostics ---
    n_treated = int(T.sum())
    n_control = int(n - n_treated)

    model_info = {
        "model_type": "IPW",
        "estimand": estimand,
        "n_treated": n_treated,
        "n_control": n_control,
        "pscore_mean_treated": float(pscore[T == 1].mean()),
        "pscore_mean_control": float(pscore[T == 0].mean()),
        "pscore_min": float(pscore.min()),
        "pscore_max": float(pscore.max()),
        "trim": trim,
        "normalized": normalize,
        "n_bootstrap": n_bootstrap,
    }

    return CausalResult(
        method=f"IPW ({estimand})",
        estimand=estimand,
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
    )


# ====================================================================== #
#  Internal helpers
# ====================================================================== #

def _estimate_propensity(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Logistic regression propensity score."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        max_iter=1000, solver="lbfgs", C=1e6,  # large C ≈ no regularization
    )
    model.fit(X, T)
    ps = model.predict_proba(X)[:, 1]
    # Safety clip to avoid division by zero
    return np.clip(ps, 1e-8, 1 - 1e-8)


def _compute_weights(
    T: np.ndarray,
    pscore: np.ndarray,
    estimand: str,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute IPW weights for treated (w1) and control (w0) groups.

    Returns (weights_1, weights_0) such that:
        estimate = sum(w1 * Y) - sum(w0 * Y)
    """
    n = len(T)

    if estimand == "ATE":
        # Horvitz-Thompson: w1 = T/p, w0 = (1-T)/(1-p)
        w1 = T / pscore
        w0 = (1 - T) / (1 - pscore)
    elif estimand == "ATT":
        # ATT: treated get weight 1, controls get weight p/(1-p)
        w1 = T.copy()
        w0 = (1 - T) * pscore / (1 - pscore)
    elif estimand == "ATC":
        # ATC: controls get weight 1, treated get weight (1-p)/p
        w1 = T * (1 - pscore) / pscore
        w0 = (1 - T).copy()

    if normalize:
        s1 = w1.sum()
        s0 = w0.sum()
        if s1 > 0:
            w1 = w1 / s1
        if s0 > 0:
            w0 = w0 / s0
    else:
        w1 = w1 / n
        w0 = w0 / n

    return w1, w0
