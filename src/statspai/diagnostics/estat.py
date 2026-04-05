"""
Unified post-estimation diagnostics -- Stata's ``estat`` command suite for Python.

This is the single-entry-point diagnostic dispatcher that operates on fitted
``EconometricResults`` objects.  Every test is implemented with numpy/scipy
only (no statsmodels dependency).

Usage
-----
>>> result = sp.regress("y ~ x1 + x2", data=df)
>>> sp.estat(result, "hettest")       # Breusch-Pagan
>>> sp.estat(result, "white")         # White's general test
>>> sp.estat(result, "reset")         # Ramsey RESET
>>> sp.estat(result, "bgodfrey")      # Breusch-Godfrey serial correlation
>>> sp.estat(result, "dwatson")       # Durbin-Watson
>>> sp.estat(result, "vif")           # Variance Inflation Factors
>>> sp.estat(result, "ic")            # AIC / BIC / HQIC
>>> sp.estat(result, "linktest")      # Specification link test
>>> sp.estat(result, "normality")     # Jarque-Bera + Shapiro-Wilk
>>> sp.estat(result, "leverage")      # Cook's D, DFBETAS
>>> sp.estat(result, "endogenous")    # Durbin-Wu-Hausman (IV)
>>> sp.estat(result, "overid")        # Sargan / Hansen J (IV)
>>> sp.estat(result, "firststage")    # First-stage F (IV)
>>> sp.estat(result, "all")           # Run all applicable tests

References
----------
Breusch, T.S. and Pagan, A.R. (1979). *Econometrica*, 47(5), 1287--1294.
White, H. (1980). *Econometrica*, 48(4), 817--838.
Ramsey, J.B. (1969). *JRSS-B*, 31(2), 350--371.
Breusch, L.G. (1978). *Australian Economic Papers*, 17, 334--355.
Godfrey, L.G. (1978). *Econometrica*, 46(6), 1293--1301.
Jarque, C.M. and Bera, A.K. (1987). *International Statistical Review*, 55, 163--172.
Cook, R.D. (1977). *Technometrics*, 19(1), 15--18.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ======================================================================
#  Line characters for pretty-printing
# ======================================================================
_HEAVY = "\u2501"          # ━
_LINE_WIDTH = 65


# ======================================================================
#  Public dispatcher
# ======================================================================

def estat(
    result,
    test: str = "all",
    *,
    print_results: bool = True,
    lags: int = 1,
    powers: int = 3,
    alpha: float = 0.05,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Unified post-estimation diagnostics dispatcher.

    Parameters
    ----------
    result : EconometricResults
        A fitted result object with ``params``, ``data_info``, etc.
    test : str
        Name of the diagnostic test.  One of ``'hettest'``, ``'white'``,
        ``'reset'``, ``'ovtest'``, ``'bgodfrey'``, ``'dwatson'``, ``'vif'``,
        ``'ic'``, ``'linktest'``, ``'normality'``, ``'leverage'``,
        ``'endogenous'``, ``'overid'``, ``'firststage'``, ``'all'``.
    print_results : bool, default True
        If True, print a formatted table to stdout.
    lags : int, default 1
        Number of lags for the Breusch-Godfrey test.
    powers : int, default 3
        Highest power of y-hat for the RESET test.
    alpha : float, default 0.05
        Significance level for interpretation strings.

    Returns
    -------
    dict or list of dict
        Test result(s).  Each dict has keys ``'test'``, ``'statistic'``
        (or equivalent), ``'pvalue'`` (when applicable), and
        ``'interpretation'``.
    """
    test = test.strip().lower()

    # Alias
    if test == "ovtest":
        test = "reset"

    _dispatch = {
        "hettest":    lambda: _estat_hettest(result, alpha=alpha),
        "white":      lambda: _estat_white(result, alpha=alpha),
        "reset":      lambda: _estat_reset(result, powers=powers, alpha=alpha),
        "bgodfrey":   lambda: _estat_bgodfrey(result, lags=lags, alpha=alpha),
        "dwatson":    lambda: _estat_dwatson(result, alpha=alpha),
        "vif":        lambda: _estat_vif(result, alpha=alpha),
        "ic":         lambda: _estat_ic(result),
        "linktest":   lambda: _estat_linktest(result, alpha=alpha),
        "normality":  lambda: _estat_normality(result, alpha=alpha),
        "leverage":   lambda: _estat_leverage(result, alpha=alpha),
        "endogenous": lambda: _estat_endogenous(result, alpha=alpha),
        "overid":     lambda: _estat_overid(result, alpha=alpha),
        "firststage": lambda: _estat_firststage(result, alpha=alpha),
    }

    if test == "all":
        return _estat_all(result, print_results=print_results,
                          lags=lags, powers=powers, alpha=alpha)

    if test not in _dispatch:
        available = ", ".join(sorted(_dispatch.keys()) + ["all"])
        raise ValueError(
            f"Unknown estat test '{test}'. Available: {available}"
        )

    out = _dispatch[test]()

    if print_results:
        _print_result(out)

    return out


# ======================================================================
#  Helpers: extract arrays from result
# ======================================================================

def _get_residuals(result) -> np.ndarray:
    r = result.data_info.get("residuals")
    if r is None:
        raise ValueError(
            "Residuals not stored in result.data_info['residuals']. "
            "Re-estimate with store_residuals=True or pass residuals manually."
        )
    return np.asarray(r, dtype=float)


def _get_fitted(result) -> np.ndarray:
    yhat = result.data_info.get("fitted_values")
    if yhat is None:
        raise ValueError(
            "Fitted values not stored in result.data_info['fitted_values']."
        )
    return np.asarray(yhat, dtype=float)


def _get_X(result) -> np.ndarray:
    X = result.data_info.get("X")
    if X is None:
        raise ValueError(
            "Design matrix not stored in result.data_info['X']."
        )
    return np.asarray(X, dtype=float)


def _get_y(result) -> np.ndarray:
    y = result.data_info.get("y")
    if y is None:
        raise ValueError(
            "Response vector not stored in result.data_info['y']."
        )
    return np.asarray(y, dtype=float)


def _get_nobs(result) -> int:
    n = result.data_info.get("nobs")
    if n is None:
        n = len(_get_residuals(result))
    return int(n)


def _ols_fit(X: np.ndarray, y: np.ndarray):
    """Fit OLS via normal equations.  Returns (beta, residuals, yhat)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat
    return beta, resid, yhat


def _r_squared(y: np.ndarray, resid: np.ndarray) -> float:
    tss = np.sum((y - y.mean()) ** 2)
    rss = np.sum(resid ** 2)
    return 1.0 - rss / tss if tss > 0 else 0.0


# ======================================================================
#  Individual test implementations
# ======================================================================

# ------------------------------------------------------------------
#  Breusch-Pagan heteroskedasticity test
# ------------------------------------------------------------------

def _estat_hettest(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Breusch-Pagan / Cook-Weisberg test for heteroskedasticity."""
    resid = _get_residuals(result)
    X = _get_X(result)
    n, k = X.shape

    e2 = resid ** 2
    _, aux_resid, _ = _ols_fit(X, e2)
    tss_e2 = np.sum((e2 - e2.mean()) ** 2)
    rss_e2 = np.sum(aux_resid ** 2)
    r2 = 1.0 - rss_e2 / tss_e2 if tss_e2 > 0 else 0.0

    lm = n * r2
    df = k - 1  # regressors excl. constant
    pval = float(1.0 - sp_stats.chi2.cdf(lm, df))

    reject = pval < alpha
    interp = (
        f"REJECT H0 at {alpha:.0%}: evidence of heteroskedasticity. "
        "Consider robust standard errors."
        if reject else
        f"Cannot reject H0 at {alpha:.0%}: no evidence of heteroskedasticity."
    )

    return {
        "test": "Breusch-Pagan test for heteroskedasticity",
        "H0": "Constant variance (homoskedasticity)",
        "H1": "Variance depends on regressors",
        "statistic": float(lm),
        "statistic_label": f"chi2({df})",
        "df": df,
        "pvalue": pval,
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  White's general heteroskedasticity test
# ------------------------------------------------------------------

def _estat_white(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """White's test: regress e^2 on X, X^2, and cross-products."""
    resid = _get_residuals(result)
    X = _get_X(result)
    n, k = X.shape

    e2 = resid ** 2

    # Build auxiliary regressors: original X, squares, cross-products
    # Skip constant column (assume col 0 is constant if all-ones)
    cols = list(range(k))
    const_col = None
    for j in range(k):
        if np.allclose(X[:, j], 1.0):
            const_col = j
            break

    var_cols = [j for j in cols if j != const_col]

    aux_parts = [np.ones((n, 1))]  # constant
    # Original regressors (excl. constant)
    for j in var_cols:
        aux_parts.append(X[:, j : j + 1])
    # Squared terms
    for j in var_cols:
        aux_parts.append((X[:, j] ** 2).reshape(-1, 1))
    # Cross-products
    for j1, j2 in combinations(var_cols, 2):
        aux_parts.append((X[:, j1] * X[:, j2]).reshape(-1, 1))

    X_aux = np.column_stack(aux_parts)
    k_aux = X_aux.shape[1]

    _, aux_resid, _ = _ols_fit(X_aux, e2)
    tss_e2 = np.sum((e2 - e2.mean()) ** 2)
    rss_e2 = np.sum(aux_resid ** 2)
    r2 = 1.0 - rss_e2 / tss_e2 if tss_e2 > 0 else 0.0

    lm = n * r2
    df = k_aux - 1
    pval = float(1.0 - sp_stats.chi2.cdf(lm, df))

    reject = pval < alpha
    interp = (
        f"REJECT H0 at {alpha:.0%}: evidence of heteroskedasticity "
        "(general form). Consider robust or HC standard errors."
        if reject else
        f"Cannot reject H0 at {alpha:.0%}: no evidence of heteroskedasticity."
    )

    return {
        "test": "White's test for heteroskedasticity",
        "H0": "Homoskedasticity",
        "H1": "Unrestricted heteroskedasticity",
        "statistic": float(lm),
        "statistic_label": f"chi2({df})",
        "df": df,
        "pvalue": pval,
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  Ramsey RESET
# ------------------------------------------------------------------

def _estat_reset(result, *, powers: int = 3,
                 alpha: float = 0.05) -> Dict[str, Any]:
    """Ramsey RESET: add yhat^2, yhat^3, ... to detect misspecification."""
    y = _get_y(result)
    X = _get_X(result)
    resid = _get_residuals(result)
    yhat = _get_fitted(result)
    n, k = X.shape

    # Augmented design: original X + yhat^2 ... yhat^powers
    aug_cols = [X]
    for p in range(2, powers + 1):
        aug_cols.append((yhat ** p).reshape(-1, 1))
    X_aug = np.column_stack(aug_cols)
    k_aug = X_aug.shape[1]

    _, resid_aug, _ = _ols_fit(X_aug, y)

    rss_r = np.sum(resid ** 2)
    rss_u = np.sum(resid_aug ** 2)
    df1 = k_aug - k
    df2 = n - k_aug

    if rss_u > 0 and df1 > 0 and df2 > 0:
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        pval = float(1.0 - sp_stats.f.cdf(f_stat, df1, df2))
    else:
        f_stat = 0.0
        pval = 1.0

    reject = pval < alpha
    interp = (
        f"REJECT H0 at {alpha:.0%}: functional form may be misspecified. "
        "Consider adding nonlinear terms or transformations."
        if reject else
        f"Cannot reject H0 at {alpha:.0%}: no evidence of misspecification."
    )

    return {
        "test": "Ramsey RESET test",
        "H0": "Model has no omitted nonlinearities",
        "H1": "Nonlinear terms of fitted values are significant",
        "statistic": float(f_stat),
        "statistic_label": f"F({df1}, {df2})",
        "df1": df1,
        "df2": df2,
        "pvalue": pval,
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  Breusch-Godfrey serial correlation
# ------------------------------------------------------------------

def _estat_bgodfrey(result, *, lags: int = 1,
                    alpha: float = 0.05) -> Dict[str, Any]:
    """Breusch-Godfrey LM test for serial correlation up to *lags* lags."""
    resid = _get_residuals(result)
    X = _get_X(result)
    n, k = X.shape

    # Auxiliary regression: e_t on X, e_{t-1}, ..., e_{t-lags}
    # We lose the first `lags` observations
    e_trimmed = resid[lags:]
    X_trimmed = X[lags:]

    lag_cols = []
    for lag in range(1, lags + 1):
        lag_cols.append(resid[lags - lag : n - lag].reshape(-1, 1))

    X_aux = np.column_stack([X_trimmed] + lag_cols)
    n_aux = X_aux.shape[0]

    _, aux_resid, _ = _ols_fit(X_aux, e_trimmed)
    tss = np.sum((e_trimmed - e_trimmed.mean()) ** 2)
    rss = np.sum(aux_resid ** 2)
    r2 = 1.0 - rss / tss if tss > 0 else 0.0

    lm = n_aux * r2
    df = lags
    pval = float(1.0 - sp_stats.chi2.cdf(lm, df))

    reject = pval < alpha
    lag_label = "lag" if lags == 1 else "lags"
    interp = (
        f"REJECT H0 at {alpha:.0%}: evidence of serial correlation "
        f"up to {lags} {lag_label}. Consider Newey-West SEs."
        if reject else
        f"Cannot reject H0 at {alpha:.0%}: no evidence of serial correlation "
        f"up to {lags} {lag_label}."
    )

    return {
        "test": f"Breusch-Godfrey LM test ({lags} {lag_label})",
        "H0": "No serial correlation",
        "H1": f"Serial correlation up to order {lags}",
        "statistic": float(lm),
        "statistic_label": f"chi2({df})",
        "df": df,
        "pvalue": pval,
        "lags": lags,
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  Durbin-Watson
# ------------------------------------------------------------------

def _estat_dwatson(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Durbin-Watson statistic for first-order autocorrelation."""
    resid = _get_residuals(result)

    diff = np.diff(resid)
    dw = float(np.sum(diff ** 2) / np.sum(resid ** 2))

    # Heuristic interpretation
    if dw < 1.5:
        interp = (
            f"d = {dw:.4f} < 1.5: possible positive autocorrelation. "
            "Consider Newey-West SEs or AR models."
        )
    elif dw > 2.5:
        interp = (
            f"d = {dw:.4f} > 2.5: possible negative autocorrelation. "
            "Investigate further."
        )
    else:
        interp = (
            f"d = {dw:.4f} is near 2: no strong evidence of "
            "first-order autocorrelation."
        )

    return {
        "test": "Durbin-Watson test",
        "H0": "No first-order autocorrelation",
        "H1": "First-order autocorrelation present",
        "statistic": dw,
        "statistic_label": "d",
        "range": "[0, 4]; d = 2 means no autocorrelation",
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  Variance Inflation Factors
# ------------------------------------------------------------------

def _estat_vif(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Compute VIF for each regressor (excluding constant)."""
    X = _get_X(result)
    n, k = X.shape

    # Identify constant column
    const_col = None
    for j in range(k):
        if np.allclose(X[:, j], 1.0):
            const_col = j
            break

    var_cols = [j for j in range(k) if j != const_col]

    # Variable names from params index
    var_names = list(result.params.index)

    rows = []
    for j in var_cols:
        y_j = X[:, j]
        other_cols = [c for c in var_cols if c != j]
        X_other = np.column_stack(
            [np.ones(n)] + [X[:, c] for c in other_cols]
        )
        _, resid_j, _ = _ols_fit(X_other, y_j)
        tss_j = np.sum((y_j - y_j.mean()) ** 2)
        rss_j = np.sum(resid_j ** 2)
        r2_j = 1.0 - rss_j / tss_j if tss_j > 0 else 0.0
        vif_j = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else np.inf

        name = var_names[j] if j < len(var_names) else f"x{j}"
        rows.append({
            "variable": name,
            "VIF": round(vif_j, 2),
            "1/VIF": round(1.0 / vif_j, 4) if np.isfinite(vif_j) else 0.0,
        })

    vif_df = pd.DataFrame(rows)
    max_vif = vif_df["VIF"].max() if len(vif_df) > 0 else 0.0

    if max_vif > 10:
        interp = (
            f"Max VIF = {max_vif:.2f}: serious multicollinearity. "
            "Consider dropping or combining variables."
        )
    elif max_vif > 5:
        interp = (
            f"Max VIF = {max_vif:.2f}: moderate multicollinearity. "
            "Monitor but may be acceptable."
        )
    else:
        interp = f"Max VIF = {max_vif:.2f}: no multicollinearity concern."

    return {
        "test": "Variance Inflation Factors",
        "vif_table": vif_df,
        "mean_vif": float(vif_df["VIF"].mean()) if len(vif_df) > 0 else 0.0,
        "max_vif": float(max_vif),
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  Information criteria
# ------------------------------------------------------------------

def _estat_ic(result) -> Dict[str, Any]:
    """AIC, BIC, and HQIC."""
    resid = _get_residuals(result)
    n = _get_nobs(result)
    X = _get_X(result)
    k = X.shape[1]

    rss = float(np.sum(resid ** 2))
    ll_term = n * np.log(rss / n) if rss > 0 else 0.0

    aic = ll_term + 2.0 * k
    bic = ll_term + k * np.log(n)
    hqic = ll_term + 2.0 * k * np.log(np.log(n)) if n > 1 else ll_term

    return {
        "test": "Information Criteria",
        "AIC": float(aic),
        "BIC": float(bic),
        "HQIC": float(hqic),
        "n": n,
        "k": k,
        "interpretation": (
            "Lower values indicate better fit-complexity trade-off. "
            "BIC penalises complexity more than AIC."
        ),
    }


# ------------------------------------------------------------------
#  Link test (specification)
# ------------------------------------------------------------------

def _estat_linktest(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Specification link test.

    Re-estimate y on yhat and yhat^2.  If yhat^2 is significant the
    model is misspecified (wrong functional form or omitted variables).
    """
    y = _get_y(result)
    yhat = _get_fitted(result)
    n = len(y)

    X_link = np.column_stack([np.ones(n), yhat, yhat ** 2])
    beta_link, resid_link, _ = _ols_fit(X_link, y)

    # t-test on yhat^2 coefficient (index 2)
    rss = np.sum(resid_link ** 2)
    mse = rss / (n - 3)
    XtX_inv = np.linalg.inv(X_link.T @ X_link)
    se_hatsq = np.sqrt(mse * XtX_inv[2, 2])
    t_hatsq = beta_link[2] / se_hatsq if se_hatsq > 0 else 0.0
    pval = float(2.0 * (1.0 - sp_stats.t.cdf(abs(t_hatsq), n - 3)))

    reject = pval < alpha
    interp = (
        f"REJECT H0 at {alpha:.0%}: yhat^2 is significant (t = {t_hatsq:.4f}). "
        "Model may be misspecified."
        if reject else
        f"Cannot reject H0 at {alpha:.0%}: yhat^2 is not significant. "
        "No evidence of link misspecification."
    )

    return {
        "test": "Specification link test",
        "H0": "Model is correctly specified (yhat^2 coefficient = 0)",
        "H1": "Model is misspecified",
        "statistic": float(t_hatsq),
        "statistic_label": f"t({n - 3})",
        "coef_hatsq": float(beta_link[2]),
        "se_hatsq": float(se_hatsq),
        "pvalue": pval,
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  Normality of residuals
# ------------------------------------------------------------------

def _estat_normality(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Jarque-Bera and Shapiro-Wilk tests on residuals."""
    resid = _get_residuals(result)
    n = len(resid)

    # Jarque-Bera
    mu = resid.mean()
    sigma = resid.std(ddof=0)
    if sigma > 0:
        z = (resid - mu) / sigma
        skew = float(np.mean(z ** 3))
        kurt_excess = float(np.mean(z ** 4) - 3.0)
    else:
        skew = 0.0
        kurt_excess = 0.0

    jb = (n / 6.0) * (skew ** 2 + (kurt_excess ** 2) / 4.0)
    jb_pval = float(1.0 - sp_stats.chi2.cdf(jb, 2))

    # Shapiro-Wilk (scipy limit: n <= 5000)
    if n <= 5000:
        sw_stat, sw_pval = sp_stats.shapiro(resid)
        sw_stat = float(sw_stat)
        sw_pval = float(sw_pval)
    else:
        sw_stat = None
        sw_pval = None

    reject_jb = jb_pval < alpha
    interp_parts = []
    if reject_jb:
        interp_parts.append(
            f"Jarque-Bera REJECTS normality at {alpha:.0%} "
            f"(skewness = {skew:.4f}, excess kurtosis = {kurt_excess:.4f})."
        )
    else:
        interp_parts.append(
            f"Jarque-Bera cannot reject normality at {alpha:.0%}."
        )

    if sw_pval is not None:
        if sw_pval < alpha:
            interp_parts.append(
                f"Shapiro-Wilk REJECTS normality at {alpha:.0%}."
            )
        else:
            interp_parts.append(
                f"Shapiro-Wilk cannot reject normality at {alpha:.0%}."
            )
    else:
        interp_parts.append(
            "Shapiro-Wilk skipped (n > 5000)."
        )

    out: Dict[str, Any] = {
        "test": "Normality of residuals",
        "H0": "Residuals are normally distributed",
        "H1": "Residuals are not normally distributed",
        "jarque_bera": float(jb),
        "jb_pvalue": jb_pval,
        "skewness": skew,
        "excess_kurtosis": kurt_excess,
        "interpretation": " ".join(interp_parts),
    }

    if sw_stat is not None:
        out["shapiro_wilk"] = sw_stat
        out["sw_pvalue"] = sw_pval

    return out


# ------------------------------------------------------------------
#  Leverage / influence diagnostics
# ------------------------------------------------------------------

def _estat_leverage(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Cook's distance, DFBETAS, and leverage diagnostics."""
    resid = _get_residuals(result)
    X = _get_X(result)
    y = _get_y(result)
    n, k = X.shape

    # Hat matrix diagonal
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    H = X @ XtX_inv @ X.T
    h = np.diag(H)

    mse = np.sum(resid ** 2) / (n - k)

    # Cook's distance
    cooks_d = (resid ** 2 / (k * mse)) * (h / (1.0 - h) ** 2)
    threshold = 4.0 / n

    influential_idx = np.where(cooks_d > threshold)[0]

    # DFBETAS: change in each beta when obs i removed
    # beta_full already known; compute leave-one-out efficiently
    # DFBETAS_ij = e_i / (1-h_ii) * (X'X)^{-1} x_i' / se(beta_j)_loo
    # Simplified: DFBETAS_ij = (e_i * (X'X)^{-1} x_i_j) / (sqrt(MSE_i) * (1-h_ii))
    # where MSE_i is the leave-one-out MSE
    # For efficiency, use the approximation:
    #   MSE_{(i)} = (n-k)*MSE - e_i^2/(1-h_ii)) / (n-k-1)
    e_loo = resid / (1.0 - h)  # studentized deletion residuals (numerator)
    mse_loo = np.maximum(
        ((n - k) * mse - resid ** 2 / (1.0 - h)) / (n - k - 1),
        1e-16,
    )

    # DFBETAS matrix: n x k
    # DFBETAS[i, j] = e_i / ((1-h_ii) * sqrt(MSE_{(i)})) * (X'X)^{-1} X[i,:]_j
    leverage_factor = resid / (1.0 - h)  # n-vector
    dfbetas = np.zeros((n, k))
    for j in range(k):
        w_j = XtX_inv @ X.T  # k x n
        # Each obs contribution
        dfbetas[:, j] = leverage_factor * w_j[j, :] / np.sqrt(mse_loo)

    # Threshold: |DFBETAS| > 2/sqrt(n)
    dfbetas_thresh = 2.0 / np.sqrt(n)
    dfbetas_flags = np.any(np.abs(dfbetas) > dfbetas_thresh, axis=1)
    dfbetas_flagged_idx = np.where(dfbetas_flags)[0]

    n_influential = len(influential_idx)
    interp_parts = []
    if n_influential > 0:
        pct = 100.0 * n_influential / n
        interp_parts.append(
            f"{n_influential} observation(s) ({pct:.1f}%) have Cook's D > {threshold:.4f} (= 4/n)."
        )
    else:
        interp_parts.append(
            "No observations exceed the Cook's D threshold (4/n)."
        )
    if len(dfbetas_flagged_idx) > 0:
        interp_parts.append(
            f"{len(dfbetas_flagged_idx)} observation(s) have |DFBETAS| > "
            f"{dfbetas_thresh:.4f} (= 2/sqrt(n))."
        )

    return {
        "test": "Leverage and influence diagnostics",
        "cooks_d": cooks_d,
        "cooks_d_threshold": threshold,
        "influential_obs": influential_idx.tolist(),
        "n_influential": n_influential,
        "leverage": h,
        "dfbetas": dfbetas,
        "dfbetas_threshold": dfbetas_thresh,
        "dfbetas_flagged_obs": dfbetas_flagged_idx.tolist(),
        "interpretation": " ".join(interp_parts),
    }


# ------------------------------------------------------------------
#  IV-specific: endogeneity (Durbin-Wu-Hausman)
# ------------------------------------------------------------------

def _estat_endogenous(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Durbin-Wu-Hausman endogeneity test (for IV results)."""
    mi = result.model_info

    # Look for pre-computed DWH statistic
    dwh = mi.get("wu_hausman") or mi.get("dwh_statistic")
    dwh_pval = mi.get("wu_hausman_pvalue") or mi.get("dwh_pvalue")

    if dwh is None:
        return {
            "test": "Durbin-Wu-Hausman endogeneity test",
            "error": (
                "DWH statistic not found in model_info. "
                "This test requires an IV/2SLS estimation result."
            ),
            "interpretation": "Not applicable: model was not estimated via IV.",
        }

    reject = dwh_pval < alpha if dwh_pval is not None else None
    if reject is True:
        interp = (
            f"REJECT H0 at {alpha:.0%}: regressors are endogenous. "
            "IV estimation is warranted."
        )
    elif reject is False:
        interp = (
            f"Cannot reject H0 at {alpha:.0%}: no evidence of endogeneity. "
            "OLS may be consistent and more efficient."
        )
    else:
        interp = "p-value not available."

    return {
        "test": "Durbin-Wu-Hausman endogeneity test",
        "H0": "Regressors are exogenous",
        "H1": "Regressors are endogenous",
        "statistic": float(dwh),
        "statistic_label": "DWH",
        "pvalue": float(dwh_pval) if dwh_pval is not None else None,
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  IV-specific: over-identification (Sargan / Hansen J)
# ------------------------------------------------------------------

def _estat_overid(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """Sargan/Hansen J test for over-identifying restrictions."""
    mi = result.model_info

    sargan = mi.get("sargan_stat") or mi.get("hansen_j")
    sargan_pval = mi.get("sargan_pvalue") or mi.get("hansen_j_pvalue")
    sargan_df = mi.get("sargan_df") or mi.get("overid_df")

    if sargan is None:
        return {
            "test": "Sargan/Hansen J over-identification test",
            "error": (
                "Over-identification statistic not found in model_info. "
                "This test requires an IV estimation with more instruments "
                "than endogenous regressors."
            ),
            "interpretation": "Not applicable or model is exactly identified.",
        }

    reject = sargan_pval < alpha if sargan_pval is not None else None
    if reject is True:
        interp = (
            f"REJECT H0 at {alpha:.0%}: instruments may not all be valid. "
            "Re-examine instrument exogeneity."
        )
    elif reject is False:
        interp = (
            f"Cannot reject H0 at {alpha:.0%}: over-identifying restrictions "
            "are satisfied. Instruments appear valid."
        )
    else:
        interp = "p-value not available."

    label = f"chi2({sargan_df})" if sargan_df else "J"

    return {
        "test": "Sargan/Hansen J over-identification test",
        "H0": "All instruments are valid (exclusion restrictions hold)",
        "H1": "At least one instrument is invalid",
        "statistic": float(sargan),
        "statistic_label": label,
        "df": int(sargan_df) if sargan_df is not None else None,
        "pvalue": float(sargan_pval) if sargan_pval is not None else None,
        "interpretation": interp,
    }


# ------------------------------------------------------------------
#  IV-specific: first-stage F
# ------------------------------------------------------------------

def _estat_firststage(result, *, alpha: float = 0.05) -> Dict[str, Any]:
    """First-stage F-statistic for weak instrument detection."""
    mi = result.model_info

    f_stat = mi.get("first_stage_f") or mi.get("first_stage_F")
    f_pval = mi.get("first_stage_f_pvalue")

    if f_stat is None:
        return {
            "test": "First-stage F-statistic",
            "error": (
                "First-stage F not found in model_info. "
                "This test requires an IV estimation."
            ),
            "interpretation": "Not applicable: model was not estimated via IV.",
        }

    # Stock-Yogo critical values (common thresholds)
    weak = float(f_stat) < 10.0
    if weak:
        interp = (
            f"F = {float(f_stat):.2f} < 10: instruments are weak "
            "(Stock-Yogo rule of thumb). Consider LIML, Fuller, or "
            "Anderson-Rubin confidence sets."
        )
    else:
        interp = (
            f"F = {float(f_stat):.2f} >= 10: instruments are not weak "
            "(Stock-Yogo rule of thumb)."
        )

    out: Dict[str, Any] = {
        "test": "First-stage F-statistic (weak instruments)",
        "H0": "Instruments are weak (excluded instruments have no explanatory power)",
        "statistic": float(f_stat),
        "statistic_label": "F",
        "stock_yogo_threshold": 10.0,
        "interpretation": interp,
    }
    if f_pval is not None:
        out["pvalue"] = float(f_pval)

    return out


# ======================================================================
#  "all" -- run every applicable test
# ======================================================================

def _estat_all(
    result,
    *,
    print_results: bool = True,
    lags: int = 1,
    powers: int = 3,
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Run all applicable tests and return results as a list."""
    outputs: List[Dict[str, Any]] = []

    # Tests that require residuals + X (standard OLS diagnostics)
    has_resid = result.data_info.get("residuals") is not None
    has_X = result.data_info.get("X") is not None
    has_y = result.data_info.get("y") is not None
    has_fitted = result.data_info.get("fitted_values") is not None

    if has_resid and has_X:
        outputs.append(_estat_hettest(result, alpha=alpha))
        outputs.append(_estat_white(result, alpha=alpha))
        outputs.append(_estat_bgodfrey(result, lags=lags, alpha=alpha))
        outputs.append(_estat_dwatson(result, alpha=alpha))
        outputs.append(_estat_vif(result, alpha=alpha))
        outputs.append(_estat_ic(result))
        outputs.append(_estat_normality(result, alpha=alpha))

    if has_y and has_X and has_resid and has_fitted:
        outputs.append(_estat_reset(result, powers=powers, alpha=alpha))
        outputs.append(_estat_linktest(result, alpha=alpha))
        outputs.append(_estat_leverage(result, alpha=alpha))

    # IV-specific tests (always attempt; they handle missing info gracefully)
    mi = result.model_info
    is_iv = mi.get("model_type", "").lower() in ("iv", "2sls", "gmm", "liml")
    if is_iv:
        outputs.append(_estat_endogenous(result, alpha=alpha))
        outputs.append(_estat_overid(result, alpha=alpha))
        outputs.append(_estat_firststage(result, alpha=alpha))

    if not outputs:
        outputs.append({
            "test": "estat: no applicable tests",
            "interpretation": (
                "Could not run any tests. Ensure the result object stores "
                "residuals, fitted values, design matrix (X), and response (y) "
                "in data_info."
            ),
        })

    if print_results:
        _print_all(outputs)

    return outputs


# ======================================================================
#  Pretty-printing
# ======================================================================

def _fmt_line(width: int = _LINE_WIDTH) -> str:
    return _HEAVY * width


def _print_result(out: Dict[str, Any]) -> None:
    """Print a single test result in Stata-style formatted output."""
    w = _LINE_WIDTH
    line = _fmt_line(w)

    print(line)
    print(f"  {out.get('test', 'Test')}")
    print(line)

    # Hypotheses
    if "H0" in out:
        print(f"  H0: {out['H0']}")
    if "H1" in out:
        print(f"  H1: {out['H1']}")
    if "H0" in out or "H1" in out:
        print()

    # Error message (for IV tests when not applicable)
    if "error" in out:
        print(f"  {out['error']}")
        print(line)
        return

    # VIF table
    if "vif_table" in out:
        vif_df = out["vif_table"]
        print(vif_df.to_string(index=False))
        print(f"\n  Mean VIF = {out.get('mean_vif', 0):.2f}")
        print()

    # Information criteria
    elif "AIC" in out:
        print(f"  {'AIC':<10} = {out['AIC']:>12.4f}")
        print(f"  {'BIC':<10} = {out['BIC']:>12.4f}")
        print(f"  {'HQIC':<10} = {out['HQIC']:>12.4f}")
        print(f"  {'n':<10} = {out['n']:>12d}")
        print(f"  {'k':<10} = {out['k']:>12d}")
        print()

    # Normality (two tests)
    elif "jarque_bera" in out:
        print(f"  {'Jarque-Bera':<16} = {out['jarque_bera']:>12.4f}")
        print(f"  {'  p-value':<16} = {out['jb_pvalue']:>12.4f}")
        print(f"  {'  skewness':<16} = {out['skewness']:>12.4f}")
        print(f"  {'  excess kurt.':<16} = {out['excess_kurtosis']:>12.4f}")
        if "shapiro_wilk" in out:
            print()
            print(f"  {'Shapiro-Wilk':<16} = {out['shapiro_wilk']:>12.4f}")
            print(f"  {'  p-value':<16} = {out['sw_pvalue']:>12.4f}")
        print()

    # Leverage / influence
    elif "cooks_d" in out:
        cd = out["cooks_d"]
        n_inf = out["n_influential"]
        thresh = out["cooks_d_threshold"]
        print(f"  Cook's D threshold (4/n) = {thresh:.4f}")
        print(f"  Observations exceeding threshold: {n_inf}")
        if n_inf > 0 and n_inf <= 20:
            print(f"  Influential obs indices: {out['influential_obs']}")
        elif n_inf > 20:
            print(f"  (Showing first 20): {out['influential_obs'][:20]}")
        print(f"\n  DFBETAS threshold (2/sqrt(n)) = {out['dfbetas_threshold']:.4f}")
        print(f"  Observations with large DFBETAS: {len(out['dfbetas_flagged_obs'])}")
        print()

    # Standard single-statistic test
    elif "statistic" in out:
        label = out.get("statistic_label", "stat")
        print(f"  {label:<10} = {out['statistic']:>12.4f}")
        if "pvalue" in out and out["pvalue"] is not None:
            print(f"  {'p-value':<10} = {out['pvalue']:>12.4f}")
        if "range" in out:
            print(f"  Range: {out['range']}")
        print()

    # Interpretation arrow
    interp = out.get("interpretation", "")
    if interp:
        # Wrap long lines
        _print_wrapped(f"  -> {interp}", width=w)

    print(line)


def _print_all(outputs: List[Dict[str, Any]]) -> None:
    """Print the comprehensive estat report."""
    w = _LINE_WIDTH
    line = _fmt_line(w)

    print()
    print(line)
    print("  COMPREHENSIVE POST-ESTIMATION DIAGNOSTICS")
    print(line)
    print()

    for out in outputs:
        _print_result(out)
        print()

    print(line)
    print("  End of diagnostics")
    print(line)


def _print_wrapped(text: str, width: int = _LINE_WIDTH) -> None:
    """Simple word-wrap for interpretation strings."""
    words = text.split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 > width:
            print(current_line)
            current_line = "    " + word  # indent continuation
        else:
            current_line = current_line + " " + word if current_line else word
    if current_line:
        print(current_line)
