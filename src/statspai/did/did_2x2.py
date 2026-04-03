"""
Classic 2x2 Difference-in-Differences estimator.

Estimates ATT using the standard two-period, two-group DID design
via OLS:  Y = β₀ + β₁·D + β₂·T + β₃·D×T + X'γ + ε

The coefficient β₃ on the interaction term is the ATT.

References
----------
Angrist, J.D. and Pischke, J.-S. (2009).
*Mostly Harmless Econometrics: An Empiricist's Companion*.
Princeton University Press.
"""

from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def did_2x2(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    covariates: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    robust: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Classic 2×2 Difference-in-Differences estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    treat : str
        Binary treatment group indicator (0/1).
    time : str
        Binary time period indicator (0 = pre, 1 = post).
    covariates : list of str, optional
        Additional control variables.
    cluster : str, optional
        Cluster variable for cluster-robust standard errors.
    robust : bool, default True
        Use HC1 heteroskedasticity-robust standard errors.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        Results with ATT estimate, standard errors, and diagnostics.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> d = rng.integers(0, 2, n)
    >>> t = rng.integers(0, 2, n)
    >>> y_val = 1 + 2*d + 3*t + 5*d*t + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({'y': y_val, 'd': d, 't': t})
    >>> result = did_2x2(df, y='y', treat='d', time='t')
    >>> abs(result.estimate - 5.0) < 1.0
    True
    """
    df = data.copy()

    # Validate binary variables
    treat_vals = sorted(df[treat].dropna().unique())
    time_vals = sorted(df[time].dropna().unique())
    if len(treat_vals) != 2:
        raise ValueError(
            f"Treatment variable '{treat}' must have exactly 2 values, "
            f"got {len(treat_vals)}: {treat_vals}"
        )
    if len(time_vals) != 2:
        raise ValueError(
            f"Time variable '{time}' must have exactly 2 values, "
            f"got {len(time_vals)}: {time_vals}"
        )

    # Ensure 0/1 coding
    d = (df[treat] == treat_vals[1]).astype(float).values
    t = (df[time] == time_vals[1]).astype(float).values
    dt = d * t  # interaction = DID coefficient
    y_arr = df[y].values.astype(float)

    # Drop rows with NaN in outcome
    valid = np.isfinite(y_arr)
    if covariates:
        for cov in covariates:
            valid &= np.isfinite(df[cov].values.astype(float))
    d, t, dt, y_arr = d[valid], t[valid], dt[valid], y_arr[valid]

    # Build design matrix: [1, D, T, D×T, covariates...]
    X_parts = [np.ones(len(y_arr)), d, t, dt]
    X_names = ['const', treat, time, f'{treat}x{time}']

    if covariates:
        for cov in covariates:
            X_parts.append(df.loc[valid, cov].values.astype(float)
                           if isinstance(valid, np.ndarray)
                           else df[cov].values.astype(float))
            X_names.append(cov)

    X = np.column_stack(X_parts)
    n, k = X.shape

    # OLS: β = (X'X)⁻¹ X'y
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)

    beta = XtX_inv @ X.T @ y_arr
    resid = y_arr - X @ beta

    # Variance-covariance matrix
    if cluster is not None:
        cl = df.loc[valid, cluster].values if isinstance(valid, np.ndarray) else df[cluster].values
        unique_cl = np.unique(cl)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for c_val in unique_cl:
            idx = cl == c_val
            score = (X[idx] * resid[idx, np.newaxis]).sum(axis=0)
            meat += np.outer(score, score)
        correction = (n_cl / (n_cl - 1)) * ((n - 1) / (n - k))
        vcov = correction * XtX_inv @ meat @ XtX_inv
    elif robust:
        weights = (n / (n - k)) * resid ** 2
        meat = X.T @ np.diag(weights) @ X
        vcov = XtX_inv @ meat @ XtX_inv
    else:
        sigma2 = np.sum(resid ** 2) / (n - k)
        vcov = sigma2 * XtX_inv

    se = np.sqrt(np.diag(vcov))

    # DID coefficient is the interaction term
    did_idx = X_names.index(f'{treat}x{time}')
    att = float(beta[did_idx])
    att_se = float(se[did_idx])
    t_stat = att / att_se if att_se > 0 else np.nan
    df_resid = n - k
    pvalue = float(2 * (1 - stats.t.cdf(abs(t_stat), df_resid)))
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    ci = (att - t_crit * att_se, att + t_crit * att_se)

    # Full coefficient table for detail
    t_stats_all = beta / se
    pvals_all = 2 * (1 - stats.t.cdf(np.abs(t_stats_all), df_resid))
    detail = pd.DataFrame({
        'variable': X_names,
        'coefficient': beta,
        'se': se,
        'tstat': t_stats_all,
        'pvalue': pvals_all,
    })

    # R-squared
    tss = np.sum((y_arr - np.mean(y_arr)) ** 2)
    rss = np.sum(resid ** 2)
    r_squared = 1 - rss / tss

    model_info = {
        'r_squared': round(r_squared, 6),
        'n_treated': int(d.sum()),
        'n_control': int((1 - d).sum()),
        'n_pre': int((1 - t).sum()),
        'n_post': int(t.sum()),
        'robust_se': robust,
        'cluster': cluster,
    }

    return CausalResult(
        method='Difference-in-Differences (2x2)',
        estimand='ATT',
        estimate=att,
        se=att_se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='did_2x2',
    )
