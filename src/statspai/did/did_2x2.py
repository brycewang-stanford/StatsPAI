"""
Classic 2x2 Difference-in-Differences estimator.

Estimates ATT using the standard two-period, two-group DID design
via OLS:  Y = β₀ + β₁·D + β₂·T + β₃·D×T + X'γ + ε

The coefficient β₃ on the interaction term is the ATT.

References
----------
Angrist, J.D. and Pischke, J.-S. (2009).
*Mostly Harmless Econometrics: An Empiricist's Companion*.
Princeton University Press. [@angrist2009mostly]
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
    weights: Optional[str] = None,
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
    weights : str, optional
        Column name for analytical weights (e.g. population weights).
        Observations are weighted proportionally — equivalent to Stata's
        ``[aweight=...]`` or R's ``weights=`` in ``lm()``.

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
    from statspai.exceptions import MethodIncompatibility

    treat_vals = sorted(df[treat].dropna().unique())
    time_vals = sorted(df[time].dropna().unique())
    if len(treat_vals) != 2:
        raise MethodIncompatibility(
            f"Treatment variable '{treat}' must have exactly 2 values, "
            f"got {len(treat_vals)}: {treat_vals}",
            recovery_hint=(
                "For staggered adoption (multi-period treat), use "
                "sp.callaway_santanna or sp.sun_abraham. "
                "For multi-valued treatment, use sp.multi_treatment."
            ),
            diagnostics={"treat": treat, "n_unique_values": len(treat_vals)},
            alternative_functions=[
                "sp.callaway_santanna", "sp.sun_abraham", "sp.did_multiplegt",
            ],
        )
    if len(time_vals) != 2:
        raise MethodIncompatibility(
            f"Time variable '{time}' must have exactly 2 values, "
            f"got {len(time_vals)}: {time_vals}",
            recovery_hint=(
                "For multi-period panels, use sp.did(method='cs') "
                "(Callaway-Sant'Anna) or sp.event_study."
            ),
            diagnostics={"time": time, "n_unique_values": len(time_vals)},
            alternative_functions=[
                "sp.callaway_santanna", "sp.event_study", "sp.sun_abraham",
            ],
        )

    # Ensure 0/1 coding
    d = (df[treat] == treat_vals[1]).astype(float).values
    t = (df[time] == time_vals[1]).astype(float).values
    dt = d * t  # interaction = DID coefficient
    y_arr = df[y].values.astype(float)

    # Drop rows with NaN in outcome (and weights if provided)
    valid = np.isfinite(y_arr)
    if covariates:
        for cov in covariates:
            valid &= np.isfinite(df[cov].values.astype(float))
    if weights is not None:
        valid &= np.isfinite(df[weights].values.astype(float))
        valid &= df[weights].values.astype(float) > 0
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

    # --- Analytical weights (WLS) ---
    if weights is not None:
        w_raw = df.loc[valid, weights].values.astype(float) if isinstance(valid, np.ndarray) else df[weights].values.astype(float)
        if np.any(w_raw < 0):
            raise ValueError(f"Weights column '{weights}' contains negative values.")
        # Normalize so weights sum to n (aweight convention)
        w = w_raw * (n / w_raw.sum())
        sqrt_w = np.sqrt(w)
        # WLS: transform X and y by sqrt(w)
        Xw = X * sqrt_w[:, np.newaxis]
        yw = y_arr * sqrt_w
    else:
        w = None
        Xw = X
        yw = y_arr

    # OLS on (possibly weighted) data: β = (Xw'Xw)⁻¹ Xw'yw
    try:
        XtX_inv = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)

    beta = XtX_inv @ Xw.T @ yw
    resid = y_arr - X @ beta  # residuals in original scale

    # Variance-covariance matrix
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
        'n_treated': int(d.sum()),
        'n_control': int((1 - d).sum()),
        'n_pre': int((1 - t).sum()),
        'n_post': int(t.sum()),
        'robust_se': robust,
        'cluster': cluster,
        'weights': weights,
    }

    _result = CausalResult(
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
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.did.did_2x2",
            params={
                "y": y, "treat": treat, "time": time,
                "covariates": covariates, "cluster": cluster,
                "robust": robust, "alpha": alpha, "weights": weights,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
