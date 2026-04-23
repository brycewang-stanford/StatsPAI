"""
Quantile Regression.

Estimates conditional quantiles of the outcome distribution, allowing
analysis of heterogeneous effects across the distribution (not just
the mean). More robust to outliers than OLS.

References
----------
Koenker, R. and Bassett, G. (1978).
"Regression Quantiles."
*Econometrica*, 46(1), 33-50. [@koenker1978regression]

Koenker, R. (2005).
*Quantile Regression*. Cambridge University Press.

Chernozhukov, V. and Hansen, C. (2005).
"An IV Model of Quantile Treatment Effects."
*Econometrica*, 73(1), 245-261. [@chernozhukov2005model]
"""

from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

from ..core.results import CausalResult


def qreg(
    data: pd.DataFrame,
    formula: Optional[str] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    quantile: float = 0.5,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Quantile regression at a single quantile.

    Equivalent to Stata's ``qreg y x, quantile(0.5)``.

    Parameters
    ----------
    data : pd.DataFrame
    formula : str, optional
        Formula like ``"y ~ x1 + x2"`` (patsy-style).
    y : str, optional
        Outcome variable (alternative to formula).
    x : list of str, optional
        Regressors (alternative to formula).
    quantile : float, default 0.5
        Quantile to estimate (0 < q < 1). 0.5 = median.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        Coefficients at the specified quantile.

    Examples
    --------
    >>> # Median regression
    >>> result = sp.qreg(df, y='wage', x=['education', 'experience'],
    ...                  quantile=0.5)

    >>> # 90th percentile
    >>> result = sp.qreg(df, y='wage', x=['education', 'experience'],
    ...                  quantile=0.9)

    Notes
    -----
    Quantile regression minimizes:

    .. math::
        \\min_\\beta \\sum_i \\rho_\\tau(Y_i - X_i'\\beta)

    where ρ_τ(u) = u(τ - 1(u < 0)) is the check function.

    Standard errors are computed using the Powell (1991) sandwich
    estimator with a kernel density estimate of f(0|X).

    See Koenker & Bassett (1978, *Econometrica*).
    """
    if not (0 < quantile < 1):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    # Parse inputs
    if formula is not None:
        y_name, x_names = _parse_formula(formula)
    elif y is not None and x is not None:
        y_name, x_names = y, x
    else:
        raise ValueError("Provide either formula or (y, x)")

    df = data[[y_name] + x_names].dropna()
    Y = df[y_name].values.astype(float)
    X = np.column_stack([np.ones(len(df))] +
                        [df[v].values.astype(float) for v in x_names])
    n, k = X.shape
    var_names = ['const'] + x_names

    # Solve quantile regression via linear programming
    beta = _qreg_fit(Y, X, quantile)
    resid = Y - X @ beta

    # Standard errors (Powell sandwich estimator)
    se = _qreg_se(Y, X, beta, resid, quantile)

    z_stats = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    detail = pd.DataFrame({
        'variable': var_names,
        'coefficient': beta,
        'se': se,
        'z': z_stats,
        'pvalue': pvals,
    })

    # Main estimate: first regressor (after constant)
    main_coef = float(beta[1])
    main_se = float(se[1])
    main_p = float(pvals[1])
    ci = (main_coef - z_crit * main_se, main_coef + z_crit * main_se)

    model_info = {
        'quantile': quantile,
        'pseudo_r2': _pseudo_r2(Y, resid, quantile),
        'n_obs': n,
    }

    return CausalResult(
        method=f'Quantile Regression (tau={quantile})',
        estimand=f'Q({quantile}) {x_names[0]}',
        estimate=main_coef,
        se=main_se,
        pvalue=main_p,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='qreg',
    )


def sqreg(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    quantiles: Optional[List[float]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Simultaneous quantile regression at multiple quantiles.

    Equivalent to Stata's ``sqreg y x, quantiles(10 25 50 75 90)``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    x : list of str
    quantiles : list of float, optional
        Default: [0.1, 0.25, 0.5, 0.75, 0.9].
    alpha : float, default 0.05

    Returns
    -------
    pd.DataFrame
        Rows: variables. Columns: quantiles with coefficients and SEs.

    Examples
    --------
    >>> table = sp.sqreg(df, y='wage', x=['education', 'experience'])
    >>> print(table)
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    results = {}
    for q in quantiles:
        r = qreg(data, y=y, x=x, quantile=q, alpha=alpha)
        for _, row in r.detail.iterrows():
            var = row['variable']
            if var not in results:
                results[var] = {'variable': var}
            results[var][f'Q({q})'] = round(row['coefficient'], 4)
            results[var][f'SE({q})'] = round(row['se'], 4)

    return pd.DataFrame(list(results.values()))


# ======================================================================
# Internal
# ======================================================================

def _qreg_fit(Y, X, tau):
    """Solve quantile regression via linear programming (interior point)."""
    n, k = X.shape

    # Reformulate as LP:
    # min tau * 1'u + (1-tau) * 1'v
    # s.t. X β + u - v = Y, u >= 0, v >= 0
    # where u = max(residual, 0) and v = max(-residual, 0)

    c = np.concatenate([np.zeros(k),
                        tau * np.ones(n),
                        (1 - tau) * np.ones(n)])

    # Equality: X β + I u - I v = Y
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    b_eq = Y

    # Bounds: β unbounded, u >= 0, v >= 0
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)

    try:
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                         method='highs', options={'maxiter': 5000})
        if result.success:
            return result.x[:k]
    except Exception:
        pass

    # Fallback: iteratively reweighted least squares
    return _qreg_irls(Y, X, tau)


def _qreg_irls(Y, X, tau, max_iter=50):
    """IRLS fallback for quantile regression."""
    n, k = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]

    for _ in range(max_iter):
        resid = Y - X @ beta
        w = np.where(resid >= 0, tau, 1 - tau)
        w = w / (np.abs(resid) + 1e-6)
        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ Y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    return beta


def _qreg_se(Y, X, beta, resid, tau):
    """Powell (1991) kernel sandwich SE for quantile regression."""
    n, k = X.shape

    # Bandwidth (Silverman rule)
    h = 1.06 * np.std(resid) * n ** (-1 / 5)
    h = max(h, 1e-6)

    # Kernel density of residuals at 0
    f0 = np.mean(stats.norm.pdf(resid / h)) / h
    f0 = max(f0, 1e-6)

    # Sandwich: V = tau(1-tau) / (n f0²) * (X'X)^{-1}
    XtX_inv = np.linalg.pinv(X.T @ X)
    vcov = tau * (1 - tau) / (n * f0 ** 2) * XtX_inv

    return np.sqrt(np.maximum(np.diag(vcov), 1e-20))


def _pseudo_r2(Y, resid, tau):
    """Koenker-Machado (1999) pseudo R² for quantile regression."""
    rho = lambda u: u * (tau - (u < 0))
    obj_full = np.sum(rho(resid))
    obj_null = np.sum(rho(Y - np.quantile(Y, tau)))
    return 1 - obj_full / obj_null if obj_null > 0 else 0


def _parse_formula(formula):
    """Parse 'y ~ x1 + x2' into (y, [x1, x2])."""
    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError(f"Invalid formula: {formula}")
    y = parts[0].strip()
    x = [v.strip() for v in parts[1].split('+') if v.strip()]
    return y, x


# Citation
CausalResult._CITATIONS['qreg'] = (
    "@article{koenker1978regression,\n"
    "  title={Regression Quantiles},\n"
    "  author={Koenker, Roger and Bassett, Gilbert},\n"
    "  journal={Econometrica},\n"
    "  volume={46},\n"
    "  number={1},\n"
    "  pages={33--50},\n"
    "  year={1978},\n"
    "  publisher={Wiley}\n"
    "}"
)
