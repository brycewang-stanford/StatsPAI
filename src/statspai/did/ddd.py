"""
Triple Differences (DDD) estimator.

Extends the standard 2×2 DID by adding a third difference — a within-unit
subgroup that should *not* be affected by treatment — to net out additional
confounders beyond what parallel trends can handle.

Model
-----
Y_{itsg} = α + β₁·D_s + β₂·T_t + β₃·G_g
         + β₄·D×T + β₅·D×G + β₆·T×G
         + δ·D×T×G + X'γ + ε

where:
    D_s = treatment group indicator
    T_t = post-treatment period indicator
    G_g = affected subgroup indicator
    δ   = DDD estimate (the triple interaction)

References
----------
Gruber, J. (1994). "The Incidence of Mandated Maternity Benefits."
*American Economic Review*, 84(3), 622-641.

Olden, A. and Møen, J. (2022).
"The Triple Difference Estimator."
*The Econometrics Journal*, 25(3), 531-553. [@olden2022triple]
"""

from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def ddd(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    subgroup: str,
    covariates: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    robust: bool = True,
    alpha: float = 0.05,
    weights: Optional[str] = None,
) -> CausalResult:
    """
    Triple Differences (DDD) estimator.

    Uses a within-treatment-group subgroup that is unaffected by treatment
    to eliminate confounders beyond what standard DID parallel trends
    can handle.

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
    subgroup : str
        Binary affected-subgroup indicator (1 = affected by treatment,
        0 = unaffected within-unit comparison group).
        E.g. low-wage workers (affected) vs high-wage workers (unaffected)
        in a minimum wage study.
    covariates : list of str, optional
        Additional control variables.
    cluster : str, optional
        Cluster variable for cluster-robust standard errors.
    robust : bool, default True
        Use HC1 heteroskedasticity-robust standard errors.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    weights : str, optional
        Column name for analytical weights (e.g. population weights).
        Equivalent to Stata's ``[aweight=...]``.

    Returns
    -------
    CausalResult
        Results with DDD estimate (triple interaction coefficient),
        standard errors, and full coefficient table.

    Examples
    --------
    Minimum wage example — treatment: NJ vs PA; time: pre vs post;
    subgroup: low-wage (affected) vs high-wage (unaffected):

    >>> result = ddd(df, y='employment', treat='nj', time='post',
    ...             subgroup='low_wage')
    >>> print(result.summary())

    With clustering by state:

    >>> result = ddd(df, y='employment', treat='nj', time='post',
    ...             subgroup='low_wage', cluster='state')
    """
    df = data.copy()

    # Validate binary variables
    for col, label in [(treat, 'Treatment'), (time, 'Time'), (subgroup, 'Subgroup')]:
        vals = sorted(df[col].dropna().unique())
        if len(vals) != 2:
            raise ValueError(
                f"{label} variable '{col}' must have exactly 2 values, "
                f"got {len(vals)}: {vals}"
            )

    # Ensure 0/1 coding
    treat_vals = sorted(df[treat].dropna().unique())
    time_vals = sorted(df[time].dropna().unique())
    sub_vals = sorted(df[subgroup].dropna().unique())

    d = (df[treat] == treat_vals[1]).astype(float).values
    t = (df[time] == time_vals[1]).astype(float).values
    g = (df[subgroup] == sub_vals[1]).astype(float).values
    y_arr = df[y].values.astype(float)

    # Two-way interactions
    dt = d * t
    dg = d * g
    tg = t * g

    # Triple interaction = DDD coefficient
    dtg = d * t * g

    # Drop rows with NaN (and invalid weights if provided)
    valid = np.isfinite(y_arr)
    if covariates:
        for cov in covariates:
            valid &= np.isfinite(df[cov].values.astype(float))
    if weights is not None:
        valid &= np.isfinite(df[weights].values.astype(float))
        valid &= df[weights].values.astype(float) > 0

    d, t, g = d[valid], t[valid], g[valid]
    dt, dg, tg, dtg = dt[valid], dg[valid], tg[valid], dtg[valid]
    y_arr = y_arr[valid]

    # Build design matrix
    X_parts = [np.ones(len(y_arr)), d, t, g, dt, dg, tg, dtg]
    X_names = [
        'const', treat, time, subgroup,
        f'{treat}x{time}', f'{treat}x{subgroup}',
        f'{time}x{subgroup}', f'{treat}x{time}x{subgroup}',
    ]

    if covariates:
        for cov in covariates:
            X_parts.append(df.loc[valid, cov].values.astype(float)
                           if isinstance(valid, np.ndarray)
                           else df[cov].values.astype(float))
            X_names.append(cov)

    X = np.column_stack(X_parts)
    n, k = X.shape

    # --- Analytical weights (WLS) ---
    if weights is not None:
        w_raw = df.loc[valid, weights].values.astype(float) if isinstance(valid, np.ndarray) else df[weights].values.astype(float)
        if np.any(w_raw < 0):
            raise ValueError(f"Weights column '{weights}' contains negative values.")
        w = w_raw * (n / w_raw.sum())
        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, np.newaxis]
        yw = y_arr * sqrt_w
    else:
        w = None
        Xw = X
        yw = y_arr

    # OLS on (possibly weighted) data
    try:
        XtX_inv = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)

    beta = XtX_inv @ Xw.T @ yw
    resid = y_arr - X @ beta

    # Variance-covariance
    if cluster is not None:
        cl = df.loc[valid, cluster].values if isinstance(valid, np.ndarray) else df[cluster].values
        unique_cl = np.unique(cl)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for c_val in unique_cl:
            idx = cl == c_val
            if w is not None:
                score = (Xw[idx] * (sqrt_w[idx] * resid[idx])[:, np.newaxis]).sum(axis=0)
            else:
                score = (X[idx] * resid[idx, np.newaxis]).sum(axis=0)
            meat += np.outer(score, score)
        correction = (n_cl / (n_cl - 1)) * ((n - 1) / (n - k))
        vcov = correction * XtX_inv @ meat @ XtX_inv
    elif robust:
        if w is not None:
            hc1_weights = (n / (n - k)) * (w * resid ** 2)
        else:
            hc1_weights = (n / (n - k)) * resid ** 2
        meat = X.T @ np.diag(hc1_weights) @ X
        vcov = XtX_inv @ meat @ XtX_inv
    else:
        if w is not None:
            sigma2 = np.sum(w * resid ** 2) / (n - k)
        else:
            sigma2 = np.sum(resid ** 2) / (n - k)
        vcov = sigma2 * XtX_inv

    se = np.sqrt(np.diag(vcov))

    # DDD coefficient is the triple interaction
    ddd_idx = X_names.index(f'{treat}x{time}x{subgroup}')
    estimate = float(beta[ddd_idx])
    est_se = float(se[ddd_idx])
    t_stat = estimate / est_se if est_se > 0 else np.nan
    df_resid = n - k
    pvalue = float(2 * (1 - stats.t.cdf(abs(t_stat), df_resid)))
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    ci = (estimate - t_crit * est_se, estimate + t_crit * est_se)

    # Also extract the DID coefficient (for comparison)
    did_idx = X_names.index(f'{treat}x{time}')
    did_estimate = float(beta[did_idx])

    # Full coefficient table
    t_stats_all = beta / se
    pvals_all = 2 * (1 - stats.t.cdf(np.abs(t_stats_all), df_resid))
    detail = pd.DataFrame({
        'variable': X_names,
        'coefficient': beta,
        'se': se,
        'tstat': t_stats_all,
        'pvalue': pvals_all,
    })

    # R-squared (weighted if applicable)
    if w is not None:
        y_wmean = np.sum(w * y_arr) / np.sum(w)
        tss = np.sum(w * (y_arr - y_wmean) ** 2)
        rss = np.sum(w * resid ** 2)
    else:
        tss = np.sum((y_arr - np.mean(y_arr)) ** 2)
        rss = np.sum(resid ** 2)
    r_squared = 1 - rss / tss if tss > 0 else 0.0

    model_info = {
        'r_squared': round(r_squared, 6),
        'n_obs': n,
        'n_treated': int(d.sum()),
        'n_control': int((1 - d).sum()),
        'n_subgroup': int(g.sum()),
        'n_comparison': int((1 - g).sum()),
        'did_estimate': round(did_estimate, 6),
        'robust_se': robust,
        'cluster': cluster,
        'weights': weights,
    }

    _result = CausalResult(
        method='Triple Differences (DDD)',
        estimand='ATT',
        estimate=estimate,
        se=est_se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='ddd',
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.did.ddd",
            params={
                "y": y, "treat": treat, "time": time,
                "subgroup": subgroup,
                "covariates": list(covariates) if covariates else None,
                "cluster": cluster, "robust": robust,
                "alpha": alpha, "weights": weights,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
