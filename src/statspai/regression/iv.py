"""
Instrumental Variables estimation: unified multi-method module.

Methods
-------
- **2SLS** (Two-Stage Least Squares) — the default workhorse.
- **LIML** (Limited Information Maximum Likelihood) — better under weak
  instruments; approximately median-unbiased for over-identified models.
- **Fuller** — finite-sample corrected LIML (Fuller 1977).
- **GMM** — Efficient two-step GMM with optimal weighting matrix;
  efficient under heteroskedasticity when over-identified.
- **JIVE** — Jackknife IV estimator; reduces many-instrument bias
  (Angrist, Imbens & Krueger 1999).

All methods share the same formula interface and produce the same
``EconometricResults`` object with integrated diagnostics (first-stage F,
Sargan/Hansen J, Durbin-Wu-Hausman, Anderson-Rubin).

References
----------
- Wooldridge (2010). *Econometric Analysis of Cross Section and Panel Data*.
- Stock & Yogo (2005). Testing for Weak Instruments.
- Fuller, W. A. (1977). Some Properties of a Modification of the
  Limited Information Estimator. *Econometrica*, 45(4), 939-953.
- Hansen, L. P. (1982). Large Sample Properties of GMM Estimators.
  *Econometrica*, 50(4), 1029-1054.
- Angrist, Imbens & Krueger (1999). Jackknife Instrumental Variables
  Estimation. *Journal of Applied Econometrics*, 14(1), 57-67.
"""

from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
import warnings

from ..core.base import BaseModel, BaseEstimator
from ..core.results import EconometricResults
from ..core.utils import parse_formula


# ====================================================================== #
#  K-class estimator (unifies 2SLS, LIML, Fuller, and user-specified k)
# ====================================================================== #

def _k_class_fit(
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
    kappa: float,
    robust: str = 'nonrobust',
    cluster: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    K-class IV estimator.

    When kappa = 1 this is 2SLS; when kappa equals the LIML eigenvalue
    this is LIML; when kappa = k_liml - a/(n-K) this is Fuller(a).
    """
    n = len(y)
    k2 = X_endog.shape[1]
    m = Z.shape[1]

    if m < k2:
        raise ValueError(
            f"Under-identified: {m} instruments for {k2} endogenous "
            f"variables. Need at least {k2} instruments."
        )

    # Full instrument matrix: [X_exog, Z]
    W = np.column_stack([X_exog, Z])
    k1 = X_exog.shape[1]

    # --- First stage (for diagnostics & projections) ---
    WtW_inv = np.linalg.inv(W.T @ W)
    P_W = W @ WtW_inv @ W.T

    X_endog_hat = P_W @ X_endog
    first_stage_results = _first_stage_diagnostics(
        X_exog, X_endog, W, n, m,
    )

    # --- K-class second stage ---
    # beta_k = (X'(I - kappa*M_W)X)^{-1} X'(I - kappa*M_W)y
    # where M_W = I - P_W
    X_actual = np.column_stack([X_exog, X_endog])
    k = X_actual.shape[1]

    M_W = np.eye(n) - P_W
    A = np.eye(n) - kappa * M_W  # = (1-kappa)*I + kappa*P_W

    XAX = X_actual.T @ A @ X_actual
    XAy = X_actual.T @ A @ y

    try:
        XAX_inv = np.linalg.inv(XAX)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Singular matrix in k-class estimation. Check for collinearity."
        )

    params = XAX_inv @ XAy

    # Residuals always use actual endogenous regressors
    fitted_values = X_actual @ params
    residuals = y - fitted_values

    # --- Standard errors ---
    # Use 2SLS-style sandwich: bread = (X_hat'X_hat)^{-1}, meat varies
    # For k-class, the "bread" is XAX_inv
    if cluster is not None:
        var_cov = _cluster_cov(X_actual, A, residuals, XAX_inv, cluster)
    elif robust != 'nonrobust':
        var_cov = _robust_cov(X_actual, A, residuals, XAX_inv, robust, n, k)
    else:
        sigma2 = np.sum(residuals ** 2) / (n - k)
        var_cov = sigma2 * XAX_inv

    std_errors = np.sqrt(np.maximum(np.diag(var_cov), 0))

    # --- Model diagnostics ---
    y_bar = np.mean(y)
    tss = np.sum((y - y_bar) ** 2)
    rss = np.sum(residuals ** 2)
    r_squared = 1 - rss / tss

    # Sargan/Hansen overidentification test (if over-identified)
    sargan = _sargan_test(residuals, W, m, k2) if m > k2 else None

    # Durbin-Wu-Hausman endogeneity test
    hausman = _hausman_test(y, X_exog, X_endog, W)

    return {
        'params': params,
        'std_errors': std_errors,
        'var_cov': var_cov,
        'fitted_values': fitted_values,
        'residuals': residuals,
        'r_squared': r_squared,
        'nobs': n,
        'df_model': k - 1,
        'df_resid': n - k,
        'rss': rss,
        'tss': tss,
        'first_stage': first_stage_results,
        'sargan': sargan,
        'hausman': hausman,
        'n_instruments': m,
        'n_endogenous': k2,
        'kappa': float(kappa),
    }


# ====================================================================== #
#  LIML eigenvalue computation
# ====================================================================== #

def _liml_kappa(
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
) -> float:
    """
    Compute the LIML kappa — the smallest eigenvalue of (W0'M_exog W0)^{-1}(W0'M_Z W0),
    where W0 = [y, X_endog] and the projections are off exogenous variables.

    This is the Anderson (1951) / Anderson-Rubin LIML formulation.
    """
    n = len(y)
    W_full = np.column_stack([X_exog, Z])  # all instruments

    # Projection matrices
    P_exog = X_exog @ np.linalg.inv(X_exog.T @ X_exog) @ X_exog.T
    P_full = W_full @ np.linalg.inv(W_full.T @ W_full) @ W_full.T

    M_exog = np.eye(n) - P_exog
    M_full = np.eye(n) - P_full

    # W0 = [y, X_endog]
    W0 = np.column_stack([y, X_endog])

    # Matrices for generalized eigenvalue problem
    # A = W0' M_full W0  (residuals from full model)
    # B = W0' M_exog W0  (residuals from exog-only model)
    A = W0.T @ M_full @ W0
    B = W0.T @ M_exog @ W0

    try:
        # kappa_LIML = min eigenvalue of inv(A) @ B
        # where A = W0' M_full W0, B = W0' M_exog W0
        eigvals = np.linalg.eigvalsh(np.linalg.inv(A) @ B)
        kappa = float(np.min(eigvals))
    except np.linalg.LinAlgError:
        warnings.warn("LIML eigenvalue computation failed, falling back to 2SLS (kappa=1)")
        kappa = 1.0

    return kappa


# ====================================================================== #
#  GMM estimator
# ====================================================================== #

def _gmm_fit(
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
    robust: str = 'nonrobust',
    cluster: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Efficient two-step GMM estimator for IV.

    Step 1: 2SLS to get initial residuals.
    Step 2: Re-estimate with optimal weighting matrix S^{-1}.

    Under homoskedasticity this equals 2SLS. Under heteroskedasticity
    and over-identification, this is more efficient.
    """
    n = len(y)
    k2 = X_endog.shape[1]
    m = Z.shape[1]

    if m < k2:
        raise ValueError(
            f"Under-identified: {m} instruments for {k2} endogenous "
            f"variables. Need at least {k2} instruments."
        )

    W = np.column_stack([X_exog, Z])
    X_actual = np.column_stack([X_exog, X_endog])
    k = X_actual.shape[1]

    # Step 1: 2SLS for initial residuals
    WtW_inv = np.linalg.inv(W.T @ W)
    P_W = W @ WtW_inv @ W.T
    X_hat = np.column_stack([X_exog, P_W @ X_endog])
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    beta_init = XhXh_inv @ X_hat.T @ y
    resid_init = y - X_actual @ beta_init

    # Step 2: Optimal weighting matrix
    # S = (1/n) sum_i (Z_i * e_i)(Z_i * e_i)' for heteroskedastic case
    if cluster is not None:
        # Cluster-robust weighting matrix
        clusters = cluster.unique()
        S = np.zeros((W.shape[1], W.shape[1]))
        for cid in clusters:
            idx = cluster == cid
            moments_c = (W[idx] * resid_init[idx, np.newaxis]).sum(axis=0)
            S += np.outer(moments_c, moments_c)
        S /= n
    elif robust != 'nonrobust':
        # Heteroskedasticity-robust weighting matrix
        S = (W * resid_init[:, np.newaxis]).T @ (W * resid_init[:, np.newaxis]) / n
    else:
        # Homoskedastic weighting matrix
        sigma2 = np.sum(resid_init ** 2) / n
        S = sigma2 * (W.T @ W) / n

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        warnings.warn("Optimal weighting matrix singular, using 2SLS weighting")
        S_inv = WtW_inv * n

    # GMM estimator: beta = (X'W S^{-1} W'X)^{-1} X'W S^{-1} W'y
    XW = X_actual.T @ W
    bread = np.linalg.inv(XW @ S_inv @ XW.T)
    params = bread @ XW @ S_inv @ W.T @ y

    fitted_values = X_actual @ params
    residuals = y - fitted_values

    # GMM variance: full sandwich
    # V = (1/n) * (Q_xw S^{-1} Q_wx)^{-1} Q_xw S^{-1} Omega S^{-1} Q_wx (Q_xw S^{-1} Q_wx)^{-1}
    # where Q_xw = X'W/n, Omega = E[W'ee'W]/n
    # Re-estimate Omega with final residuals
    We = W * residuals[:, np.newaxis]
    Omega = We.T @ We / n

    Q_xw = XW / n
    Q_xw_Sinv = Q_xw @ S_inv
    bread_n = np.linalg.inv(Q_xw_Sinv @ Q_xw.T)
    meat_n = Q_xw_Sinv @ Omega @ S_inv @ Q_xw.T
    var_cov = (bread_n @ meat_n @ bread_n) / n

    std_errors = np.sqrt(np.maximum(np.diag(var_cov), 0))

    # Diagnostics
    y_bar = np.mean(y)
    tss = np.sum((y - y_bar) ** 2)
    rss = np.sum(residuals ** 2)
    r_squared = 1 - rss / tss

    first_stage_results = _first_stage_diagnostics(X_exog, X_endog, W, n, m)

    # Hansen J test (GMM overidentification)
    if m > k2:
        g_bar = W.T @ residuals / n
        j_stat = float(n * g_bar @ S_inv @ g_bar)
        j_df = m - k2
        j_pvalue = float(1 - stats.chi2.cdf(j_stat, j_df))
        hansen_j = {'statistic': j_stat, 'pvalue': j_pvalue, 'df': j_df}
    else:
        hansen_j = None

    hausman = _hausman_test(y, X_exog, X_endog, W)

    return {
        'params': params,
        'std_errors': std_errors,
        'var_cov': var_cov,
        'fitted_values': fitted_values,
        'residuals': residuals,
        'r_squared': r_squared,
        'nobs': n,
        'df_model': k - 1,
        'df_resid': n - k,
        'rss': rss,
        'tss': tss,
        'first_stage': first_stage_results,
        'sargan': hansen_j,  # Hansen J generalises Sargan
        'hausman': hausman,
        'n_instruments': m,
        'n_endogenous': k2,
        'kappa': None,
    }


# ====================================================================== #
#  JIVE estimator
# ====================================================================== #

def _jive_fit(
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
    robust: str = 'nonrobust',
    cluster: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Jackknife IV Estimator (JIVE1).

    For each observation i, the first-stage fitted value uses
    leave-one-out: X_hat_i = P_{W,-i} X_i. This removes the
    own-observation bias that plagues 2SLS with many instruments.

    Reference: Angrist, Imbens & Krueger (1999).
    """
    n = len(y)
    k2 = X_endog.shape[1]
    m = Z.shape[1]

    if m < k2:
        raise ValueError(
            f"Under-identified: {m} instruments for {k2} endogenous "
            f"variables. Need at least {k2} instruments."
        )

    W = np.column_stack([X_exog, Z])
    X_actual = np.column_stack([X_exog, X_endog])
    k = X_actual.shape[1]

    # Full projection matrix
    WtW_inv = np.linalg.inv(W.T @ W)
    P_W = W @ WtW_inv @ W.T
    h = np.diag(P_W)  # leverage values

    # JIVE1: X_hat_i = (P_W X_endog)_i / (1 - h_ii) - h_ii/(1-h_ii) * X_endog_i
    # Equivalently: X_hat_jive_i = (P_W X_endog_i - h_ii X_endog_i) / (1 - h_ii)
    X_endog_hat_full = P_W @ X_endog
    X_endog_jive = np.empty_like(X_endog)
    for j in range(k2):
        X_endog_jive[:, j] = (
            (X_endog_hat_full[:, j] - h * X_endog[:, j]) / (1 - h)
        )

    # Second stage with JIVE fitted values
    X_hat_jive = np.column_stack([X_exog, X_endog_jive])
    XhXh_inv = np.linalg.inv(X_hat_jive.T @ X_hat_jive)
    params = XhXh_inv @ X_hat_jive.T @ y

    fitted_values = X_actual @ params
    residuals = y - fitted_values

    # Standard errors (HC1-style with JIVE bread)
    if cluster is not None:
        var_cov = _cluster_cov(X_hat_jive, np.eye(n), residuals, XhXh_inv, cluster)
    elif robust != 'nonrobust':
        var_cov = _robust_cov(X_hat_jive, np.eye(n), residuals, XhXh_inv, robust, n, k)
    else:
        sigma2 = np.sum(residuals ** 2) / (n - k)
        var_cov = sigma2 * XhXh_inv

    std_errors = np.sqrt(np.maximum(np.diag(var_cov), 0))

    # Diagnostics
    y_bar = np.mean(y)
    tss = np.sum((y - y_bar) ** 2)
    rss = np.sum(residuals ** 2)

    first_stage_results = _first_stage_diagnostics(X_exog, X_endog, W, n, m)
    sargan = _sargan_test(residuals, W, m, k2) if m > k2 else None
    hausman = _hausman_test(y, X_exog, X_endog, W)

    return {
        'params': params,
        'std_errors': std_errors,
        'var_cov': var_cov,
        'fitted_values': fitted_values,
        'residuals': residuals,
        'r_squared': 1 - rss / tss,
        'nobs': n,
        'df_model': k - 1,
        'df_resid': n - k,
        'rss': rss,
        'tss': tss,
        'first_stage': first_stage_results,
        'sargan': sargan,
        'hausman': hausman,
        'n_instruments': m,
        'n_endogenous': k2,
        'kappa': None,
    }


# ====================================================================== #
#  Shared diagnostic helpers
# ====================================================================== #

def _first_stage_diagnostics(
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    W: np.ndarray,
    n: int,
    m: int,
) -> List[Dict[str, float]]:
    """First-stage F-statistic and partial R² for each endogenous variable."""
    k2 = X_endog.shape[1]
    WtW_inv = np.linalg.inv(W.T @ W)
    XeXe_inv = np.linalg.inv(X_exog.T @ X_exog)

    results = []
    for j in range(k2):
        gamma_j = WtW_inv @ W.T @ X_endog[:, j]
        resid_full = X_endog[:, j] - W @ gamma_j

        gamma_r = XeXe_inv @ X_exog.T @ X_endog[:, j]
        resid_restricted = X_endog[:, j] - X_exog @ gamma_r

        rss_full = resid_full @ resid_full
        rss_restricted = resid_restricted @ resid_restricted
        df_num = m
        df_denom = n - W.shape[1]

        if rss_full > 0 and df_denom > 0:
            f_stat = ((rss_restricted - rss_full) / df_num) / (rss_full / df_denom)
            f_pvalue = 1 - stats.f.cdf(f_stat, df_num, df_denom)
        else:
            f_stat = f_pvalue = np.nan

        results.append({
            'f_statistic': f_stat,
            'f_pvalue': f_pvalue,
            'partial_r_squared': 1 - rss_full / rss_restricted if rss_restricted > 0 else np.nan,
        })

    return results


def _robust_cov(
    X_hat: np.ndarray,
    A: np.ndarray,
    residuals: np.ndarray,
    bread: np.ndarray,
    robust_type: str,
    n: int,
    k: int,
) -> np.ndarray:
    """Heteroskedasticity-robust covariance (sandwich)."""
    if robust_type == 'hc0':
        weights = residuals ** 2
    elif robust_type == 'hc1':
        weights = (n / (n - k)) * residuals ** 2
    elif robust_type in ('hc2', 'hc3'):
        h = np.diag(X_hat @ bread @ X_hat.T)
        h = np.clip(h, 0, 1 - 1e-8)
        if robust_type == 'hc2':
            weights = residuals ** 2 / (1 - h)
        else:
            weights = residuals ** 2 / (1 - h) ** 2
    else:
        raise ValueError(f"Unknown robust type: {robust_type}")

    meat = X_hat.T @ np.diag(weights) @ X_hat
    return bread @ meat @ bread


def _cluster_cov(
    X_hat: np.ndarray,
    A: np.ndarray,
    residuals: np.ndarray,
    bread: np.ndarray,
    cluster: pd.Series,
) -> np.ndarray:
    """Clustered standard errors."""
    n, k = X_hat.shape
    clusters = cluster.unique()
    n_clusters = len(clusters)

    meat = np.zeros((k, k))
    for cid in clusters:
        idx = cluster == cid
        Xh_c = X_hat[idx]
        resid_c = residuals[idx]
        moments_c = (Xh_c * resid_c[:, np.newaxis]).sum(axis=0)
        meat += np.outer(moments_c, moments_c)

    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    return correction * bread @ meat @ bread


def _sargan_test(
    residuals: np.ndarray,
    W: np.ndarray,
    n_excluded: int,
    n_endog: int,
) -> Dict[str, float]:
    """Sargan test for overidentifying restrictions."""
    n = len(residuals)
    WtW_inv = np.linalg.inv(W.T @ W)
    P_W = W @ WtW_inv @ W.T

    stat = (residuals @ P_W @ residuals) / (residuals @ residuals / n)
    df = n_excluded - n_endog
    pvalue = 1 - stats.chi2.cdf(stat, df) if df > 0 else np.nan

    return {'statistic': stat, 'pvalue': pvalue, 'df': df}


def _hausman_test(
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    W: np.ndarray,
) -> Dict[str, float]:
    """Durbin-Wu-Hausman endogeneity test (regression-based)."""
    n = len(y)
    k2 = X_endog.shape[1]

    WtW_inv = np.linalg.inv(W.T @ W)
    v_hat = np.empty_like(X_endog)
    for j in range(k2):
        gamma_j = WtW_inv @ W.T @ X_endog[:, j]
        v_hat[:, j] = X_endog[:, j] - W @ gamma_j

    X_aug = np.column_stack([X_exog, X_endog, v_hat])
    X_orig = np.column_stack([X_exog, X_endog])

    try:
        XaXa_inv = np.linalg.inv(X_aug.T @ X_aug)
        beta_aug = XaXa_inv @ X_aug.T @ y
        resid_aug = y - X_aug @ beta_aug
        rss_aug = resid_aug @ resid_aug

        XoXo_inv = np.linalg.inv(X_orig.T @ X_orig)
        beta_orig = XoXo_inv @ X_orig.T @ y
        resid_orig = y - X_orig @ beta_orig
        rss_orig = resid_orig @ resid_orig

        df_num = k2
        df_denom = n - X_aug.shape[1]

        if rss_aug > 0 and df_denom > 0:
            f_stat = ((rss_orig - rss_aug) / df_num) / (rss_aug / df_denom)
            f_pvalue = 1 - stats.f.cdf(f_stat, df_num, df_denom)
        else:
            f_stat = f_pvalue = np.nan
    except np.linalg.LinAlgError:
        f_stat = f_pvalue = np.nan

    return {'statistic': f_stat, 'pvalue': f_pvalue, 'df': k2}


# ====================================================================== #
#  Legacy IVEstimator (kept for backward compat)
# ====================================================================== #

class IVEstimator(BaseEstimator):
    """
    Two-Stage Least Squares (2SLS) estimator.

    Legacy class. Prefer using the ``iv()`` function directly.
    """

    def estimate(
        self,
        y: np.ndarray,
        X_exog: np.ndarray,
        X_endog: np.ndarray,
        Z: np.ndarray,
        robust: str = 'nonrobust',
        cluster: Optional[pd.Series] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return _k_class_fit(y, X_exog, X_endog, Z, kappa=1.0,
                            robust=robust, cluster=cluster)


# ====================================================================== #
#  Method name → label mapping
# ====================================================================== #

_METHOD_LABELS = {
    '2sls': 'IV-2SLS',
    'liml': 'IV-LIML',
    'fuller': 'IV-Fuller',
    'gmm': 'IV-GMM (2-step)',
    'jive': 'IV-JIVE',
}

_METHOD_DESCRIPTIONS = {
    '2sls': 'Two-Stage Least Squares',
    'liml': 'Limited Information Maximum Likelihood',
    'fuller': 'Fuller Modified LIML',
    'gmm': 'Efficient Two-Step GMM',
    'jive': 'Jackknife Instrumental Variables',
}


# ====================================================================== #
#  IVRegression model class
# ====================================================================== #

class IVRegression(BaseModel):
    """
    Instrumental Variables regression model.

    Supports multiple estimation methods via ``method`` parameter:
    '2sls', 'liml', 'fuller', 'gmm', 'jive'.

    Parameters
    ----------
    formula : str, optional
        Formula with IV syntax: ``"y ~ (endog ~ z1 + z2) + exog1 + exog2"``
    data : pd.DataFrame, optional
    method : str, default '2sls'
        Estimation method.
    fuller_alpha : float, default 1.0
        Fuller constant (only used when method='fuller'). ``alpha=1``
        gives the bias-corrected Fuller estimator; ``alpha=4`` minimises
        MSE under normal errors.
    y, X_exog, X_endog, Z, var_names : array-like, optional
        Alternative to formula interface.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        method: str = '2sls',
        fuller_alpha: float = 1.0,
        y: Optional[np.ndarray] = None,
        X_exog: Optional[np.ndarray] = None,
        X_endog: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        var_names: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.formula = formula
        self.data = data
        self.method = method.lower()
        self.fuller_alpha = fuller_alpha
        self.y = y
        self.X_exog = X_exog
        self.X_endog = X_endog
        self.Z = Z
        self.var_names = var_names

        if self.method not in ('2sls', 'liml', 'fuller', 'gmm', 'jive'):
            raise ValueError(
                f"Unknown IV method '{method}'. "
                f"Choose from: 2sls, liml, fuller, gmm, jive"
            )

    def _prepare_from_formula(self):
        """Parse formula and build matrices from data."""
        parsed = parse_formula(self.formula)

        if not parsed['endogenous'] or not parsed['instruments']:
            raise ValueError(
                "IV formula must specify endogenous variables and instruments. "
                "Use syntax: \"y ~ (endog ~ z1 + z2) + exog\""
            )

        self.dependent_var = parsed['dependent']
        exog_names = parsed['exogenous']
        endog_names = parsed['endogenous']
        instrument_names = parsed['instruments']

        all_vars = [self.dependent_var] + exog_names + endog_names + instrument_names
        missing = [v for v in all_vars if v not in self.data.columns]
        if missing:
            raise ValueError(f"Variables not found in data: {missing}")

        extra_cols = [c for c in self.data.columns if c not in all_vars]
        clean = self.data[all_vars + extra_cols].dropna(subset=all_vars)

        self.y = clean[self.dependent_var].values

        if parsed['has_constant']:
            const = np.ones((len(clean), 1))
            if exog_names:
                self.X_exog = np.column_stack([const, clean[exog_names].values])
            else:
                self.X_exog = const
            self._exog_names = ['Intercept'] + exog_names
        else:
            self.X_exog = clean[exog_names].values
            self._exog_names = exog_names

        self.X_endog = clean[endog_names].values
        self.Z = clean[instrument_names].values

        self._endog_names = endog_names
        self._instrument_names = instrument_names
        self._clean_data = clean

    def fit(
        self,
        robust: str = 'nonrobust',
        cluster: Optional[str] = None,
        **kwargs,
    ) -> EconometricResults:
        """
        Fit the IV model.

        Parameters
        ----------
        robust : str, default 'nonrobust'
            Standard-error type ('nonrobust', 'hc0', 'hc1', 'hc2', 'hc3').
        cluster : str, optional
            Variable name for clustering.

        Returns
        -------
        EconometricResults
        """
        if self.formula is not None and self.data is not None:
            self._prepare_from_formula()
        elif not (self.y is not None and self.X_exog is not None
                  and self.X_endog is not None and self.Z is not None):
            raise ValueError(
                "Provide either (formula, data) or (y, X_exog, X_endog, Z)"
            )
        else:
            self._exog_names = (
                self.var_names.get('exog', [f'exog{i}' for i in range(self.X_exog.shape[1])])
                if self.var_names else [f'exog{i}' for i in range(self.X_exog.shape[1])]
            )
            self._endog_names = (
                self.var_names.get('endog', [f'endog{i}' for i in range(self.X_endog.shape[1])])
                if self.var_names else [f'endog{i}' for i in range(self.X_endog.shape[1])]
            )
            self._instrument_names = (
                self.var_names.get('instruments', [f'z{i}' for i in range(self.Z.shape[1])])
                if self.var_names else [f'z{i}' for i in range(self.Z.shape[1])]
            )
            self.dependent_var = (
                self.var_names.get('dependent', 'y')
                if self.var_names else 'y'
            )

        # Cluster variable
        cluster_var = None
        if cluster and self.data is not None:
            if hasattr(self, '_clean_data'):
                cluster_var = self._clean_data[cluster]
            else:
                cluster_var = self.data[cluster]

        # --- Dispatch to estimation method ---
        method = self.method

        if method in ('2sls', 'liml', 'fuller'):
            if method == '2sls':
                kappa = 1.0
            elif method == 'liml':
                kappa = _liml_kappa(self.y, self.X_exog, self.X_endog, self.Z)
            else:  # fuller
                kappa_liml = _liml_kappa(self.y, self.X_exog, self.X_endog, self.Z)
                n = len(self.y)
                K = self.X_exog.shape[1] + self.Z.shape[1]
                kappa = kappa_liml - self.fuller_alpha / (n - K)

            results = _k_class_fit(
                self.y, self.X_exog, self.X_endog, self.Z,
                kappa=kappa, robust=robust, cluster=cluster_var,
            )

        elif method == 'gmm':
            results = _gmm_fit(
                self.y, self.X_exog, self.X_endog, self.Z,
                robust=robust, cluster=cluster_var,
            )

        elif method == 'jive':
            results = _jive_fit(
                self.y, self.X_exog, self.X_endog, self.Z,
                robust=robust, cluster=cluster_var,
            )

        # Build results object
        all_names = self._exog_names + self._endog_names
        params = pd.Series(results['params'], index=all_names)
        std_errors = pd.Series(results['std_errors'], index=all_names)

        method_label = _METHOD_LABELS.get(method, method.upper())
        method_desc = _METHOD_DESCRIPTIONS.get(method, method)

        model_info = {
            'model_type': method_label,
            'method': method_desc,
            'robust': robust,
            'cluster': cluster,
        }
        if results.get('kappa') is not None:
            model_info['kappa'] = results['kappa']

        data_info = {
            'nobs': results['nobs'],
            'df_model': results['df_model'],
            'df_resid': results['df_resid'],
            'dependent_var': self.dependent_var,
            'fitted_values': results['fitted_values'],
            'residuals': results['residuals'],
        }

        # Build diagnostics dict
        diagnostics = {
            'R-squared': results['r_squared'],
            'N instruments': results['n_instruments'],
            'N endogenous': results['n_endogenous'],
        }

        for j, fs in enumerate(results['first_stage']):
            endog_name = self._endog_names[j]
            diagnostics[f'First-stage F ({endog_name})'] = fs['f_statistic']
            diagnostics[f'First-stage F p-value ({endog_name})'] = fs['f_pvalue']
            diagnostics[f'Partial R² ({endog_name})'] = fs['partial_r_squared']

        # Weak instrument warning
        for j, fs in enumerate(results['first_stage']):
            if fs['f_statistic'] < 10:
                endog_name = self._endog_names[j]
                warnings.warn(
                    f"Weak instrument warning: First-stage F-statistic for "
                    f"'{endog_name}' is {fs['f_statistic']:.2f} (< 10). "
                    f"See Stock & Yogo (2005). Consider method='liml' or 'fuller'.",
                    UserWarning,
                    stacklevel=2,
                )

        if results['sargan'] is not None:
            test_name = 'Hansen J' if method == 'gmm' else 'Sargan'
            diagnostics[f'{test_name} statistic'] = results['sargan']['statistic']
            diagnostics[f'{test_name} p-value'] = results['sargan']['pvalue']
            diagnostics[f'{test_name} df'] = results['sargan']['df']

        if results['hausman'] is not None:
            diagnostics['Hausman F-stat'] = results['hausman']['statistic']
            diagnostics['Hausman p-value'] = results['hausman']['pvalue']

        # Store for programmatic access
        self._first_stage = results['first_stage']
        self._sargan = results['sargan']
        self._hausman = results['hausman']
        self._instruments = self._instrument_names
        self._raw_results = results

        self._results = EconometricResults(
            params=params,
            std_errors=std_errors,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )

        self.is_fitted = True
        return self._results

    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate predictions from fitted IV model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if data is None:
            return self._results.fitted_values()
        raise NotImplementedError("Out-of-sample prediction not yet implemented")

    @property
    def first_stage(self) -> List[Dict[str, float]]:
        """First-stage diagnostics for each endogenous variable."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self._first_stage

    @property
    def sargan_test(self) -> Optional[Dict[str, float]]:
        """Sargan/Hansen J overidentification test results."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self._sargan

    @property
    def hausman_test(self) -> Dict[str, float]:
        """Durbin-Wu-Hausman endogeneity test results."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self._hausman


# ====================================================================== #
#  Unified public API: iv()
# ====================================================================== #

def iv(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    method: str = '2sls',
    robust: str = 'nonrobust',
    cluster: Optional[str] = None,
    fuller_alpha: float = 1.0,
    **kwargs,
) -> EconometricResults:
    """
    Unified instrumental variables estimation.

    Supports multiple methods through the ``method`` parameter:

    - ``'2sls'`` — Two-Stage Least Squares (default).
    - ``'liml'`` — Limited Information Maximum Likelihood. Better finite-
      sample properties under weak instruments; approximately median-unbiased.
    - ``'fuller'`` — Fuller (1977) modified LIML with finite-sample bias
      correction. ``fuller_alpha=1`` removes first-order bias; ``fuller_alpha=4``
      minimises MSE under normality.
    - ``'gmm'``  — Efficient two-step GMM. More efficient than 2SLS under
      heteroskedasticity when over-identified.
    - ``'jive'`` — Jackknife IV (Angrist, Imbens & Krueger 1999). Reduces
      many-instrument bias by using leave-one-out fitted values.

    For DeepIV (neural network IV) use ``sp.deepiv()``.
    For Bartik shift-share IV use ``sp.bartik()``.

    Parameters
    ----------
    formula : str
        IV formula: ``"y ~ (endog ~ z1 + z2) + exog1 + exog2"``

        - Variables in parentheses before ``~``: endogenous regressors
        - Variables in parentheses after ``~``: excluded instruments
        - Variables outside parentheses: exogenous controls
    data : pd.DataFrame
        Data containing all variables.
    method : str, default '2sls'
        Estimation method: '2sls', 'liml', 'fuller', 'gmm', 'jive'.
    robust : str, default 'nonrobust'
        Standard-error type ('nonrobust', 'hc0', 'hc1', 'hc2', 'hc3').
    cluster : str, optional
        Variable name for clustered standard errors.
    fuller_alpha : float, default 1.0
        Fuller modification constant (only used when ``method='fuller'``).

    Returns
    -------
    EconometricResults
        Fitted model results with integrated IV diagnostics:

        - First-stage F-statistics and partial R²
        - Sargan/Hansen J overidentification test (when over-identified)
        - Durbin-Wu-Hausman endogeneity test
        - Weak instrument warnings

    Examples
    --------
    >>> # Standard 2SLS
    >>> result = sp.iv("wage ~ (education ~ parent_edu + distance) + experience",
    ...               data=df)
    >>> print(result.summary())

    >>> # LIML (better with weak instruments)
    >>> result = sp.iv("wage ~ (education ~ parent_edu + distance) + experience",
    ...               data=df, method='liml')

    >>> # Fuller with bias correction
    >>> result = sp.iv("wage ~ (education ~ parent_edu) + experience",
    ...               data=df, method='fuller', fuller_alpha=1)

    >>> # Efficient GMM with robust SEs
    >>> result = sp.iv("wage ~ (education ~ parent_edu + distance) + experience",
    ...               data=df, method='gmm', robust='hc1')

    >>> # JIVE (many instruments)
    >>> result = sp.iv("wage ~ (education ~ z1 + z2 + z3 + z4 + z5) + experience",
    ...               data=df, method='jive')

    Notes
    -----
    **Which method to choose?**

    - Start with ``'2sls'``. If first-stage F < 10, switch to ``'liml'``
      or ``'fuller'``.
    - If you have many instruments (m >> k₂) and worry about bias, use
      ``'jive'`` or ``'liml'``.
    - If over-identified and you suspect heteroskedasticity, use ``'gmm'``
      for efficiency.
    - For nonparametric / ML-based IV, see ``sp.deepiv()``.

    **Diagnostics included automatically:**

    - First-stage F < 10 triggers a weak-instrument warning.
    - Sargan test (2SLS/LIML/Fuller/JIVE) or Hansen J (GMM) for
      overidentification.
    - Durbin-Wu-Hausman test for endogeneity.

    References
    ----------
    - Wooldridge (2010), Ch. 5-8.
    - Stock & Yogo (2005), for weak-instrument critical values.
    - Fuller (1977), for the finite-sample correction.
    - Hansen (1982), for GMM.
    - Angrist, Imbens & Krueger (1999), for JIVE.
    """
    model = IVRegression(
        formula=formula, data=data, method=method,
        fuller_alpha=fuller_alpha,
    )
    return model.fit(robust=robust, cluster=cluster, **kwargs)


# ====================================================================== #
#  Legacy alias (backward compatibility)
# ====================================================================== #

def ivreg(
    formula: str,
    data: pd.DataFrame,
    robust: str = 'nonrobust',
    cluster: Optional[str] = None,
    **kwargs,
) -> EconometricResults:
    """
    Instrumental variables regression (2SLS).

    .. deprecated::
        Use ``sp.iv(formula, data, method='2sls')`` instead.
        ``ivreg`` is kept for backward compatibility.

    Parameters
    ----------
    formula : str
        IV formula: ``"y ~ (endog ~ z1 + z2) + exog1 + exog2"``
    data : pd.DataFrame
    robust : str, default 'nonrobust'
    cluster : str, optional

    Returns
    -------
    EconometricResults
    """
    return iv(formula=formula, data=data, method='2sls',
              robust=robust, cluster=cluster, **kwargs)
