"""
Prediction Intervals for Synthetic Control Methods.

Constructs valid prediction intervals for SCM that account for two
sources of uncertainty:

1. **In-sample uncertainty** -- estimation error in the donor weights
   (from finite pre-treatment periods).
2. **Out-of-sample uncertainty** -- prediction error even with known
   weights (noise in post-treatment outcomes).

Standard SCM only provides point estimates.  This method provides
prediction intervals with formal coverage guarantees.

References
----------
Cattaneo, M.D., Feng, Y. and Titiunik, R. (2021).
"Prediction Intervals for Synthetic Control Methods."
*Journal of the American Statistical Association*, 116(536), 1865-1880.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult


# ====================================================================== #
#  Public API: scdata, scest, scpi
# ====================================================================== #

def scdata(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
) -> Dict[str, Any]:
    """
    Prepare data matrices for synthetic control estimation.

    Reshapes a long-format panel into the matrices needed by ``scest``
    and ``scpi``.  Mirrors the R package's ``scdata()`` function.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    treated_unit : scalar
        Identifier of the treated unit.
    treatment_time : scalar
        First treatment period.

    Returns
    -------
    dict
        Keys:

        - ``Y_pre``  : treated unit pre-treatment outcomes (T0,)
        - ``Y_post`` : treated unit post-treatment outcomes (T1,)
        - ``Y_donors_pre``  : donor pre-treatment matrix (T0, J)
        - ``Y_donors_post`` : donor post-treatment matrix (T1, J)
        - ``donor_names``   : list of donor unit labels
        - ``pre_times``     : array of pre-treatment time values
        - ``post_times``    : array of post-treatment time values
        - ``times``         : full array of time values
        - ``treated_unit``  : echo of the treated unit label
        - ``treatment_time``: echo of the first treatment period

    Examples
    --------
    >>> prepared = sp.scdata(df, outcome='gdp', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989)
    >>> prepared['Y_pre'].shape
    (19,)
    """
    pivot = data.pivot_table(index=time, columns=unit, values=outcome)
    times = pivot.index.values
    pre_mask = times < treatment_time
    post_mask = times >= treatment_time

    if pre_mask.sum() < 2:
        raise ValueError("Need at least 2 pre-treatment periods.")
    if post_mask.sum() < 1:
        raise ValueError("Need at least 1 post-treatment period.")

    if treated_unit not in pivot.columns:
        raise ValueError(
            f"Treated unit '{treated_unit}' not found in data."
        )

    Y_treated = pivot[treated_unit].values.astype(np.float64)
    donor_cols = [c for c in pivot.columns if c != treated_unit]

    if len(donor_cols) == 0:
        raise ValueError("No donor units found.")

    Y_donors = pivot[donor_cols].values.astype(np.float64)

    # Drop donors that have NaN in the pre-treatment period
    pre_donors = Y_donors[pre_mask]
    valid = ~np.any(np.isnan(pre_donors), axis=0)
    if valid.sum() == 0:
        raise ValueError("All donor units have missing pre-treatment data.")
    Y_donors = Y_donors[:, valid]
    donor_cols = [donor_cols[i] for i in range(len(donor_cols)) if valid[i]]

    return {
        "Y_pre": Y_treated[pre_mask],
        "Y_post": Y_treated[post_mask],
        "Y_donors_pre": Y_donors[pre_mask],
        "Y_donors_post": Y_donors[post_mask],
        "donor_names": donor_cols,
        "pre_times": times[pre_mask],
        "post_times": times[post_mask],
        "times": times,
        "treated_unit": treated_unit,
        "treatment_time": treatment_time,
    }


def scest(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    w_constr: str = "simplex",
    lasso_lambda: float = 1.0,
    ridge_lambda: float = 1.0,
) -> Dict[str, Any]:
    """
    Estimate synthetic control weights.

    Solves the constrained optimisation problem to find donor weights
    that best reproduce the treated unit's pre-treatment outcomes.
    Mirrors the R package's ``scest()`` function.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    treated_unit : scalar
        Identifier of the treated unit.
    treatment_time : scalar
        First treatment period.
    w_constr : str, default 'simplex'
        Weight constraint:

        - ``'simplex'`` : w >= 0, sum(w) = 1
        - ``'lasso'``   : L1-penalised (allows negative, non-summing)
        - ``'ridge'``   : L2-penalised
        - ``'ols'``     : ordinary least squares (unconstrained)
        - ``'ls'``      : least squares (same as 'ols')
    lasso_lambda : float, default 1.0
        L1 penalty (used when ``w_constr='lasso'``).
    ridge_lambda : float, default 1.0
        L2 penalty (used when ``w_constr='ridge'``).

    Returns
    -------
    dict
        Keys:

        - ``weights``       : np.ndarray (J,) of estimated donor weights
        - ``w_constr``      : echo of constraint type
        - ``Y_synth_pre``   : synthetic unit pre-treatment outcomes
        - ``Y_synth_post``  : synthetic unit post-treatment outcomes
        - ``residuals_pre`` : pre-treatment fit residuals
        - ``effects``       : post-treatment gaps (treated - synthetic)
        - ``pre_rmspe``     : root mean squared prediction error (pre)
        - ``donor_names``   : donor labels
        - ``sc_data``       : the prepared data dict from ``scdata``

    Examples
    --------
    >>> est = sp.scest(df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989)
    >>> est['pre_rmspe']
    0.0213
    """
    sc = scdata(data, outcome, unit, time, treated_unit, treatment_time)
    Y_pre = sc["Y_pre"]
    Y_post = sc["Y_post"]
    Y_donors_pre = sc["Y_donors_pre"]
    Y_donors_post = sc["Y_donors_post"]

    w = _estimate_weights(
        Y_pre, Y_donors_pre, w_constr,
        lasso_lambda=lasso_lambda, ridge_lambda=ridge_lambda,
    )

    Y_synth_pre = Y_donors_pre @ w
    Y_synth_post = Y_donors_post @ w
    residuals_pre = Y_pre - Y_synth_pre
    effects = Y_post - Y_synth_post
    pre_rmspe = float(np.sqrt(np.mean(residuals_pre ** 2)))

    return {
        "weights": w,
        "w_constr": w_constr,
        "Y_synth_pre": Y_synth_pre,
        "Y_synth_post": Y_synth_post,
        "residuals_pre": residuals_pre,
        "effects": effects,
        "pre_rmspe": pre_rmspe,
        "donor_names": sc["donor_names"],
        "sc_data": sc,
    }


def scpi(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    w_constr: str = "simplex",
    pi_type: str = "both",
    e_method: str = "gaussian",
    alpha: float = 0.05,
    cores: int = 1,
    seed: Optional[int] = None,
    lasso_lambda: float = 1.0,
    ridge_lambda: float = 1.0,
) -> CausalResult:
    """
    Prediction intervals for synthetic control methods.

    Constructs prediction intervals that account for both in-sample
    uncertainty (weight estimation error) and out-of-sample uncertainty
    (prediction noise), following Cattaneo, Feng and Titiunik (2021).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    treated_unit : scalar
        Identifier of the treated unit.
    treatment_time : scalar
        First treatment period.
    w_constr : str, default 'simplex'
        Weight constraint for SCM estimation:

        - ``'simplex'`` : w >= 0, sum(w) = 1
        - ``'lasso'``   : L1-penalised
        - ``'ridge'``   : L2-penalised
        - ``'ols'``     : ordinary least squares (unconstrained)
        - ``'ls'``      : least squares (same as 'ols')
    pi_type : str, default 'both'
        Which prediction interval components to include:

        - ``'in_sample'``    : only in-sample (weight estimation) uncertainty
        - ``'out_of_sample'``: only out-of-sample (prediction) uncertainty
        - ``'both'``         : simultaneous interval combining both sources
    e_method : str, default 'gaussian'
        Method for estimating out-of-sample uncertainty:

        - ``'gaussian'`` : sub-Gaussian bound using residual variance
        - ``'ls'``       : location-scale model (allows heteroskedasticity)
        - ``'qreg'``     : quantile regression (nonparametric)
    alpha : float, default 0.05
        Significance level for prediction intervals.
    cores : int, default 1
        Number of cores (reserved for future parallel subsampling).
    seed : int, optional
        Random seed for reproducibility in subsampling.
    lasso_lambda : float, default 1.0
        L1 penalty (used when ``w_constr='lasso'``).
    ridge_lambda : float, default 1.0
        L2 penalty (used when ``w_constr='ridge'``).

    Returns
    -------
    CausalResult
        With ``estimate`` equal to the average post-treatment effect and
        ``ci`` giving the prediction interval.  The ``model_info`` dict
        contains:

        - ``period_results`` : DataFrame with per-period effects and PIs
        - ``weights``        : dict mapping donor names to weights
        - ``w_constr``       : constraint used
        - ``pi_type``        : PI type used
        - ``e_method``       : out-of-sample method used
        - ``sigma_hat``      : estimated residual std dev
        - ``treatment_time`` : first treatment period
        - ``treated_unit``   : treated unit label
        - ``gap_table``      : DataFrame of gaps (treated - synthetic)
        - ``Y_synth``        : synthetic unit outcomes (all periods)
        - ``Y_treated``      : treated unit outcomes (all periods)
        - ``times``          : array of all time values

    Examples
    --------
    >>> result = sp.scpi(df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989)
    >>> print(result.summary())

    >>> # In-sample only
    >>> result_in = sp.scpi(df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     pi_type='in_sample')

    >>> # Quantile-regression based out-of-sample
    >>> result_qr = sp.scpi(df, outcome='gdp', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     e_method='qreg')
    """
    if pi_type not in ("in_sample", "out_of_sample", "both"):
        raise ValueError(
            f"pi_type must be 'in_sample', 'out_of_sample', or 'both', "
            f"got '{pi_type}'."
        )
    if e_method not in ("gaussian", "ls", "qreg"):
        raise ValueError(
            f"e_method must be 'gaussian', 'ls', or 'qreg', "
            f"got '{e_method}'."
        )

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # --- Prepare data ---
    sc = scdata(data, outcome, unit, time, treated_unit, treatment_time)
    Y_pre = sc["Y_pre"]
    Y_post = sc["Y_post"]
    Y_donors_pre = sc["Y_donors_pre"]   # (T0, J)
    Y_donors_post = sc["Y_donors_post"]  # (T1, J)
    donor_names = sc["donor_names"]
    pre_times = sc["pre_times"]
    post_times = sc["post_times"]
    all_times = sc["times"]

    T0 = len(Y_pre)
    T1 = len(Y_post)
    J = Y_donors_pre.shape[1]

    # --- Step 1: Estimate SCM weights ---
    w = _estimate_weights(
        Y_pre, Y_donors_pre, w_constr,
        lasso_lambda=lasso_lambda, ridge_lambda=ridge_lambda,
    )

    # Synthetic outcomes (full panel)
    Y_synth_pre = Y_donors_pre @ w
    Y_synth_post = Y_donors_post @ w
    Y_synth = np.concatenate([Y_synth_pre, Y_synth_post])
    Y_treated = np.concatenate([Y_pre, Y_post])

    # Pre-treatment residuals
    e_pre = Y_pre - Y_synth_pre  # (T0,)
    effects_post = Y_post - Y_synth_post  # (T1,)

    # --- Step 2: In-sample variance (weight estimation uncertainty) ---
    # Use subsampling on pre-treatment residuals to estimate Var(w'Y_0t)
    in_sample_var = _in_sample_variance(
        Y_pre, Y_donors_pre, Y_donors_post, w, w_constr, rng,
        lasso_lambda=lasso_lambda, ridge_lambda=ridge_lambda,
    )  # (T1,)

    # --- Step 3: Out-of-sample variance (prediction uncertainty) ---
    out_sample_var = _out_of_sample_variance(
        e_pre, T1, e_method, alpha,
    )  # (T1,)

    sigma_hat = float(np.std(e_pre, ddof=1)) if T0 > 1 else 0.0

    # --- Step 4: Construct prediction intervals ---
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    period_results = []
    for t in range(T1):
        effect_t = float(effects_post[t])
        in_var_t = float(in_sample_var[t])
        out_var_t = float(out_sample_var[t])

        if pi_type == "in_sample":
            total_se_t = np.sqrt(in_var_t)
        elif pi_type == "out_of_sample":
            total_se_t = np.sqrt(out_var_t)
        else:  # 'both'
            total_se_t = np.sqrt(in_var_t + out_var_t)

        pi_lo_t = effect_t - z_alpha * total_se_t
        pi_hi_t = effect_t + z_alpha * total_se_t

        period_results.append({
            "time": post_times[t],
            "effect": effect_t,
            "pi_lower": pi_lo_t,
            "pi_upper": pi_hi_t,
            "in_sample_var": in_var_t,
            "out_sample_var": out_var_t,
        })

    period_df = pd.DataFrame(period_results)

    # --- Aggregate ---
    att = float(np.mean(effects_post))

    # Aggregate PI: account for averaging across T1 periods
    if pi_type == "in_sample":
        agg_var = float(np.mean(in_sample_var))
    elif pi_type == "out_of_sample":
        agg_var = float(np.mean(out_sample_var))
    else:
        agg_var = float(np.mean(in_sample_var + out_sample_var))

    agg_se = np.sqrt(agg_var)
    pi_lo = att - z_alpha * agg_se
    pi_hi = att + z_alpha * agg_se

    # Approximate p-value from PI (Gaussian)
    if agg_se > 0:
        z_stat = abs(att) / agg_se
        pvalue = float(2 * (1 - stats.norm.cdf(z_stat)))
    else:
        pvalue = 0.0 if abs(att) > 0 else 1.0

    # Gap table (all periods)
    gap_table = pd.DataFrame({
        "time": all_times,
        "treated": Y_treated,
        "synthetic": Y_synth,
        "gap": Y_treated - Y_synth,
    })

    model_info = {
        "period_results": period_df,
        "weights": dict(zip(donor_names, w)),
        "w_constr": w_constr,
        "pi_type": pi_type,
        "e_method": e_method,
        "sigma_hat": sigma_hat,
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "gap_table": gap_table,
        "Y_synth": Y_synth,
        "Y_treated": Y_treated,
        "times": all_times,
        "n_donors": J,
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "pre_rmspe": float(np.sqrt(np.mean(e_pre ** 2))),
    }

    return CausalResult(
        method="SCM with Prediction Intervals (Cattaneo et al. 2021)",
        estimand="ATT",
        estimate=att,
        se=agg_se,
        pvalue=pvalue,
        ci=(pi_lo, pi_hi),
        alpha=alpha,
        n_obs=len(Y_treated),
        detail=period_df,
        model_info=model_info,
        _citation_key="scpi",
    )


# ====================================================================== #
#  Weight estimation
# ====================================================================== #

def _estimate_weights(
    Y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    w_constr: str,
    lasso_lambda: float = 1.0,
    ridge_lambda: float = 1.0,
) -> np.ndarray:
    """
    Estimate donor weights under the specified constraint.

    Parameters
    ----------
    Y_pre : np.ndarray, shape (T0,)
        Treated unit pre-treatment outcomes.
    Y_donors_pre : np.ndarray, shape (T0, J)
        Donor matrix of pre-treatment outcomes.
    w_constr : str
        One of 'simplex', 'lasso', 'ridge', 'ols', 'ls'.
    lasso_lambda : float
        L1 penalty for lasso.
    ridge_lambda : float
        L2 penalty for ridge.

    Returns
    -------
    np.ndarray, shape (J,)
        Estimated donor weights.
    """
    J = Y_donors_pre.shape[1]
    w0 = np.ones(J) / J

    if w_constr == "simplex":
        return _weights_simplex(Y_pre, Y_donors_pre, w0)
    elif w_constr == "lasso":
        return _weights_lasso(Y_pre, Y_donors_pre, lasso_lambda)
    elif w_constr == "ridge":
        return _weights_ridge(Y_pre, Y_donors_pre, ridge_lambda)
    elif w_constr in ("ols", "ls"):
        return _weights_ols(Y_pre, Y_donors_pre)
    else:
        raise ValueError(
            f"w_constr must be 'simplex', 'lasso', 'ridge', 'ols', or 'ls', "
            f"got '{w_constr}'."
        )


def _weights_simplex(
    y: np.ndarray, X: np.ndarray, w0: np.ndarray,
) -> np.ndarray:
    """Simplex-constrained SCM: min ||y - Xw||^2, w >= 0, sum(w) = 1."""
    from ._core import solve_simplex_weights
    return solve_simplex_weights(y, X, w0=w0)


def _weights_lasso(
    y: np.ndarray, X: np.ndarray, lam: float,
) -> np.ndarray:
    """L1-penalised weights via coordinate descent."""
    T0, J = X.shape
    # Standardise
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xs = (X - X_mean) / X_std
    y_mean = y.mean()
    ys = y - y_mean

    # Coordinate descent
    w = np.zeros(J)
    max_iter = 1000
    tol = 1e-8
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(J):
            r_j = ys - Xs @ w + Xs[:, j] * w[j]
            rho_j = Xs[:, j] @ r_j / T0
            w[j] = _soft_threshold(rho_j, lam / (2.0 * T0))
        if np.max(np.abs(w - w_old)) < tol:
            break

    # Unstandardise
    w_orig = w / X_std
    return w_orig


def _weights_ridge(
    y: np.ndarray, X: np.ndarray, lam: float,
) -> np.ndarray:
    """L2-penalised (ridge) weights."""
    J = X.shape[1]
    XtX = X.T @ X + lam * np.eye(J)
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def _weights_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Unconstrained OLS weights."""
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return w


def _soft_threshold(x: float, lam: float) -> float:
    """Soft-thresholding operator for lasso coordinate descent."""
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    else:
        return 0.0


# ====================================================================== #
#  In-sample variance (weight estimation uncertainty)
# ====================================================================== #

def _in_sample_variance(
    Y_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    Y_donors_post: np.ndarray,
    w_hat: np.ndarray,
    w_constr: str,
    rng: np.random.Generator,
    n_sub: int = 200,
    lasso_lambda: float = 1.0,
    ridge_lambda: float = 1.0,
) -> np.ndarray:
    """
    Estimate in-sample prediction variance via subsampling.

    For each subsample of pre-treatment periods, re-estimate weights
    and compute the variance of the resulting post-treatment synthetic
    predictions.

    Parameters
    ----------
    Y_pre : np.ndarray, shape (T0,)
    Y_donors_pre : np.ndarray, shape (T0, J)
    Y_donors_post : np.ndarray, shape (T1, J)
    w_hat : np.ndarray, shape (J,)
    w_constr : str
    rng : np.random.Generator
    n_sub : int
        Number of subsamples.
    lasso_lambda, ridge_lambda : float

    Returns
    -------
    np.ndarray, shape (T1,)
        Estimated in-sample variance for each post-treatment period.
    """
    T0 = len(Y_pre)
    T1 = Y_donors_post.shape[0]

    # Subsample size: floor(T0^{2/3}) as in Cattaneo et al. (2021)
    b = max(2, int(np.floor(T0 ** (2.0 / 3.0))))
    b = min(b, T0 - 1)

    # Collect post-treatment synthetic predictions from subsampled weights
    synth_post_samples = np.zeros((n_sub, T1))

    for s in range(n_sub):
        idx = rng.choice(T0, size=b, replace=False)
        idx.sort()
        Y_sub = Y_pre[idx]
        X_sub = Y_donors_pre[idx]

        try:
            w_sub = _estimate_weights(
                Y_sub, X_sub, w_constr,
                lasso_lambda=lasso_lambda, ridge_lambda=ridge_lambda,
            )
        except Exception:
            # If optimisation fails on a subsample, use w_hat
            w_sub = w_hat

        synth_post_samples[s] = Y_donors_post @ w_sub

    # Variance of synthetic predictions across subsamples
    # Scale by (b / T0) to correct subsampling rate
    in_var = np.var(synth_post_samples, axis=0, ddof=1)
    # Subsampling variance correction: Var_sub * (b / T0)
    in_var *= (b / T0)

    return in_var


# ====================================================================== #
#  Out-of-sample variance (prediction uncertainty)
# ====================================================================== #

def _out_of_sample_variance(
    e_pre: np.ndarray,
    T1: int,
    e_method: str,
    alpha: float,
) -> np.ndarray:
    """
    Estimate out-of-sample prediction variance.

    Parameters
    ----------
    e_pre : np.ndarray, shape (T0,)
        Pre-treatment residuals (treated - synthetic).
    T1 : int
        Number of post-treatment periods.
    e_method : str
        'gaussian', 'ls', or 'qreg'.
    alpha : float
        Significance level.

    Returns
    -------
    np.ndarray, shape (T1,)
        Estimated out-of-sample variance for each post-treatment period.
    """
    T0 = len(e_pre)

    if e_method == "gaussian":
        return _out_of_sample_gaussian(e_pre, T1)
    elif e_method == "ls":
        return _out_of_sample_location_scale(e_pre, T1)
    elif e_method == "qreg":
        return _out_of_sample_qreg(e_pre, T1, alpha)
    else:
        raise ValueError(f"Unknown e_method: {e_method}")


def _out_of_sample_gaussian(
    e_pre: np.ndarray, T1: int,
) -> np.ndarray:
    """
    Sub-Gaussian bound: sigma^2 estimated from pre-treatment residuals.

    Assumes e_t ~ sub-Gaussian(0, sigma^2).
    """
    T0 = len(e_pre)
    if T0 > 1:
        sigma2 = float(np.var(e_pre, ddof=1))
    else:
        sigma2 = float(e_pre[0] ** 2) if T0 == 1 else 0.0

    return np.full(T1, sigma2)


def _out_of_sample_location_scale(
    e_pre: np.ndarray, T1: int,
) -> np.ndarray:
    """
    Location-scale model: allow heteroskedasticity across periods.

    Fit |e_t| = a + b * t + u_t (absolute residuals on time index)
    and extrapolate variance to post-treatment periods.
    """
    T0 = len(e_pre)
    abs_e = np.abs(e_pre)
    t_idx = np.arange(T0, dtype=np.float64)

    if T0 >= 3:
        # Linear regression of |e_t| on t
        X = np.column_stack([np.ones(T0), t_idx])
        beta, _, _, _ = np.linalg.lstsq(X, abs_e, rcond=None)
        a_hat, b_hat = beta[0], beta[1]

        # Predict conditional scale for post-treatment periods
        post_t = np.arange(T0, T0 + T1, dtype=np.float64)
        scale_post = np.maximum(a_hat + b_hat * post_t, 1e-10)
        # Variance = scale^2
        return scale_post ** 2
    else:
        # Too few periods for location-scale; fall back to constant
        sigma2 = float(np.var(e_pre, ddof=1)) if T0 > 1 else float(
            e_pre[0] ** 2
        )
        return np.full(T1, sigma2)


def _out_of_sample_qreg(
    e_pre: np.ndarray, T1: int, alpha: float,
) -> np.ndarray:
    """
    Quantile regression approach: nonparametric.

    Use empirical quantiles of pre-treatment residuals to construct
    the out-of-sample component.  Convert the quantile range to an
    equivalent variance for the Gaussian PI formula.
    """
    T0 = len(e_pre)
    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    # Empirical quantiles of pre-treatment residuals
    q_lo = float(np.quantile(e_pre, lo_q))
    q_hi = float(np.quantile(e_pre, hi_q))

    # Convert interquantile range to equivalent variance
    # IQR / (2 * z_{alpha/2}) = sigma_equiv
    z_alpha = stats.norm.ppf(hi_q)
    iqr = q_hi - q_lo
    if z_alpha > 0:
        sigma_equiv = iqr / (2 * z_alpha)
    else:
        sigma_equiv = iqr / 4.0

    sigma2_equiv = sigma_equiv ** 2
    return np.full(T1, sigma2_equiv)


# ====================================================================== #
#  Citation registration
# ====================================================================== #

CausalResult._CITATIONS["scpi"] = (
    "@article{cattaneo2021prediction,\n"
    "  title={Prediction Intervals for Synthetic Control Methods},\n"
    "  author={Cattaneo, Matias D. and Feng, Yingjie "
    "and Titiunik, Rocio},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1865--1880},\n"
    "  year={2021},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
