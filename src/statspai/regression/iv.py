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
        from statspai.exceptions import MethodIncompatibility
        raise MethodIncompatibility(
            f"Under-identified: {m} instruments for {k2} endogenous "
            f"variables. Need at least {k2} instruments.",
            recovery_hint=(
                "Add more instruments (order condition: m ≥ k2), or drop "
                "one endogenous variable. For partial identification use "
                "sp.bounds."
            ),
            diagnostics={"n_instruments": m, "n_endogenous": k2},
            alternative_functions=["sp.bounds"],
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
    except np.linalg.LinAlgError as exc:
        from statspai.exceptions import NumericalInstability
        raise NumericalInstability(
            "Singular matrix in k-class estimation. Check for collinearity.",
            recovery_hint=(
                "Run sp.vif() to locate collinear regressors; drop redundant "
                "ones. For weak-IV-robust inference without full rank in the "
                "second stage, use sp.anderson_rubin_ci."
            ),
            diagnostics={"stage": "k_class_second_stage"},
            alternative_functions=["sp.vif", "sp.anderson_rubin_ci"],
        ) from exc

    params = XAX_inv @ XAy

    # Residuals always use actual endogenous regressors
    fitted_values = X_actual @ params
    residuals = y - fitted_values

    # --- Standard errors ---
    # The k-class first-order condition X' A (y - X β) = 0 implies the
    # influence function β̂ - β = (X'AX)^{-1} (AX)' u, so the sandwich
    # meat must use the PROJECTED regressors AX, not the raw X. For
    # κ = 1 (2SLS) this is AX = P_W X = X̂; for LIML/Fuller it is the
    # k-class transformed regressor. Using raw X here is the classic
    # mistake that inflates 2SLS cluster/robust SEs by a factor that
    # depends on first-stage fit. This implementation matches
    # Cameron–Miller (2015), Stata ivregress, and linearmodels.
    AX = A @ X_actual
    if cluster is not None:
        var_cov = _cluster_cov(AX, A, residuals, XAX_inv, cluster)
    elif robust != 'nonrobust':
        var_cov = _robust_cov(AX, A, residuals, XAX_inv, robust, n, k)
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
    P_exog = X_exog @ np.linalg.solve(X_exog.T @ X_exog, X_exog.T)
    P_full = W_full @ np.linalg.solve(W_full.T @ W_full, W_full.T)

    M_exog = np.eye(n) - P_exog
    M_full = np.eye(n) - P_full

    # W0 = [y, X_endog]
    W0 = np.column_stack([y, X_endog])

    # Matrices for generalized eigenvalue problem
    # A = W0' M_full W0  (residuals from full model)
    # B = W0' M_exog W0  (residuals from exog-only model)
    A = W0.T @ M_full @ W0
    B = W0.T @ M_exog @ W0

    # kappa_LIML solves the generalized symmetric eigenvalue problem
    #     B v = kappa A v
    # with A = W0' M_full W0, B = W0' M_exog W0 (both symmetric PSD).
    # Because B >= A in the Loewner order (extra residualisation shrinks
    # SSR), all eigenvalues are >= 1, and kappa_LIML is the *smallest*.
    # NOTE: the previous implementation used ``np.linalg.eigvalsh`` on the
    # non-symmetric product ``inv(A) @ B`` which silently returned garbage
    # (often negative or complex real parts) — always bug, flipping
    # LIML into a biased direction. Fixed by using the proper generalized
    # eigendecomposition via ``scipy.linalg.eigh(B, A)``.
    try:
        from scipy.linalg import eigh as _sp_eigh
        eigvals = _sp_eigh(B, A, eigvals_only=True)
        kappa = float(np.min(eigvals))
        if not np.isfinite(kappa) or kappa < 1 - 1e-8:
            # Numerical pathology — fall back to 2SLS rather than produce a
            # demonstrably wrong kappa.
            warnings.warn(
                f"LIML kappa computation returned {kappa}; falling back to 2SLS.",
                RuntimeWarning, stacklevel=2,
            )
            kappa = 1.0
    except Exception:
        warnings.warn(
            "LIML generalized eigenvalue solve failed; falling back to 2SLS.",
            RuntimeWarning, stacklevel=2,
        )
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
        from statspai.exceptions import MethodIncompatibility
        raise MethodIncompatibility(
            f"Under-identified: {m} instruments for {k2} endogenous "
            f"variables. Need at least {k2} instruments.",
            recovery_hint=(
                "Add more instruments (order condition: m ≥ k2), or drop "
                "one endogenous variable. For partial identification use "
                "sp.bounds."
            ),
            diagnostics={"n_instruments": m, "n_endogenous": k2},
            alternative_functions=["sp.bounds"],
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
        from statspai.exceptions import MethodIncompatibility
        raise MethodIncompatibility(
            f"Under-identified: {m} instruments for {k2} endogenous "
            f"variables. Need at least {k2} instruments.",
            recovery_hint=(
                "Add more instruments (order condition: m ≥ k2), or drop "
                "one endogenous variable. For partial identification use "
                "sp.bounds."
            ),
            diagnostics={"n_instruments": m, "n_endogenous": k2},
            alternative_functions=["sp.bounds"],
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


def _normalize_robust(robust) -> str:
    """Canonicalise the SE-type vocabulary for the IV estimators.

    Accepts (case-insensitively) ``'nonrobust'`` / ``'hc0'`` / ``'hc1'`` /
    ``'hc2'`` / ``'hc3'`` plus the ergonomic aliases ``True`` / ``'robust'`` /
    ``'white'`` so callers can mirror Stata (``robust`` ≡ HC1) and the
    ``sp.regress`` spelling (uppercase HCk). Raises a clear ``ValueError`` for
    anything else instead of failing deep inside the sandwich kernel.
    """
    if robust is None or robust is False:
        return 'nonrobust'
    if robust is True:
        return 'hc1'  # Stata `robust` ≡ HC1
    if isinstance(robust, str):
        key = robust.strip().lower()
        aliases = {'robust': 'hc1', 'white': 'hc0'}
        key = aliases.get(key, key)
        if key in ('nonrobust', 'hc0', 'hc1', 'hc2', 'hc3'):
            return key
    raise ValueError(
        f"Unknown robust option: {robust!r}. Use one of 'nonrobust', "
        f"'hc0', 'hc1', 'hc2', 'hc3' (case-insensitive), True, or 'robust'."
    )


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

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(2)
    >>> z = rng.normal(size=300)
    >>> u = rng.normal(size=300)
    >>> x = 0.8 * z + u + rng.normal(size=300)
    >>> y = 1.0 + 2.0 * x + u + rng.normal(size=300)
    >>> df = pd.DataFrame({"y": y, "x": x, "z": z})
    >>> model = sp.IVRegression("y ~ (x ~ z)", data=df, method="2sls")
    >>> res = model.fit()
    >>> bool(1.5 < float(res.params["x"]) < 2.5)
    True
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
        robust : str or bool, default 'nonrobust'
            Standard-error type. Accepts 'nonrobust' and 'hc0'–'hc3'
            (case-insensitive), plus the aliases ``True`` / ``'robust'``
            (≡ HC1, matching Stata) and ``'white'`` (≡ HC0). Classical and
            robust SEs match ``ivregress 2sls, small`` / ``..., robust small``
            (the finite-sample t convention).
        cluster : str, optional
            Variable name for clustering.

        Returns
        -------
        EconometricResults
        """
        # Normalise the SE-type vocabulary so the IV path accepts the same
        # spellings as ``sp.regress`` (case-insensitive HC0–HC3) plus the
        # Stata-style ergonomic aliases. Previously a bare ``robust='HC1'``
        # (uppercase) raised "Unknown robust type" — an API inconsistency with
        # OLS, which lower-cases the type at point of use.
        robust = _normalize_robust(robust)

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
        """Generate predictions from the fitted IV model.

        For a structural-form estimator, the natural forecast of ``y`` given
        new data is ``X_exog β_exog + X_endog β_endog`` — i.e. we plug
        observed values of the endogenous variables through the structural
        equation. Instruments are not used at prediction time.

        Parameters
        ----------
        data : pd.DataFrame, optional
            New data at which to predict. Must contain all exogenous and
            endogenous variables referenced by the model's formula. If
            ``None``, returns in-sample fitted values.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if data is None:
            return self._results.fitted_values()
        if self.formula is None:
            raise ValueError(
                "Out-of-sample prediction requires the model to have been fit "
                "with a formula (not raw y, X arrays)."
            )

        parsed = parse_formula(self.formula)
        exog = parsed["exogenous"]
        endog = parsed["endogenous"]
        needed = exog + endog
        missing = [v for v in needed if v not in data.columns]
        if missing:
            raise ValueError(
                f"New data is missing columns referenced by the model: {missing}"
            )

        params = np.asarray(self._results.params)
        names = list(self._results.params.index) if hasattr(
            self._results.params, "index"
        ) else list(self._exog_names) + list(self._endog_names)

        n_new = len(data)
        X_new_cols = []
        for nm in names:
            if nm in {"Intercept", "const"}:
                X_new_cols.append(np.ones(n_new))
            elif nm in data.columns:
                X_new_cols.append(data[nm].to_numpy(dtype=float))
            else:
                raise ValueError(
                    f"Cannot map parameter '{nm}' to a column in the new data"
                )
        X_new = np.column_stack(X_new_cols)
        return X_new @ params

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

# ====================================================================== #
#  Absorb (HDFE) preprocessing for sp.iv(..., absorb=...)
# ====================================================================== #

def _normalise_absorb(absorb: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalise an ``absorb=`` argument to a list of column names.

    Accepts ``None``, ``"firm"``, ``"firm + year"``, or ``["firm", "year"]``.
    """
    if absorb is None:
        return []
    if isinstance(absorb, str):
        return [t.strip() for t in absorb.split('+') if t.strip()]
    return [str(t) for t in absorb]


def _iv_absorb_preprocess(
    formula: str,
    data: pd.DataFrame,
    absorb_terms: List[str],
    cluster_name: Optional[str] = None,
    fe_tol: float = 1e-10,
    fe_maxiter: int = 1_000,
) -> Dict[str, Any]:
    """Demean IV inputs by ``absorb_terms`` via the HDFE Phase 1 kernel.

    Returns a dict with the residualised matrices, var-name dictionary,
    cluster series (post-singleton-mask), and FE diagnostics. The
    intercept is dropped because the absorbed FEs span the constant.

    Same convention as ``sp.fast.feols``: ``fe_dof = sum(G_k - 1)``.
    """
    # Lazy import — keeps regression/iv.py free of fast/* dependencies
    # at module import time.
    from ..fast.demean import demean as _demean

    parsed = parse_formula(formula)
    if not parsed['endogenous'] or not parsed['instruments']:
        raise ValueError(
            "IV formula must specify endogenous variables and instruments. "
            "Use syntax: \"y ~ (endog ~ z1 + z2) + exog\""
        )

    dependent = parsed['dependent']
    exog_names = parsed['exogenous']
    endog_names = parsed['endogenous']
    instrument_names = parsed['instruments']

    needed = [dependent] + exog_names + endog_names + instrument_names
    needed += list(absorb_terms)
    if cluster_name is not None:
        needed.append(cluster_name)
    missing = [v for v in needed if v not in data.columns]
    if missing:
        raise ValueError(f"Variables not found in data: {missing}")
    missing_absorb = [c for c in absorb_terms if c not in data.columns]
    if missing_absorb:
        raise ValueError(
            f"absorb columns not found in data: {missing_absorb}"
        )

    clean = data[needed].dropna(subset=needed)
    n_obs = len(clean)
    if n_obs == 0:
        raise ValueError("No rows remain after dropping NaNs")

    y = clean[dependent].to_numpy(dtype=np.float64)
    X_exog = (
        clean[exog_names].to_numpy(dtype=np.float64)
        if exog_names else np.empty((n_obs, 0), dtype=np.float64)
    )
    if X_exog.ndim == 1:
        X_exog = X_exog.reshape(-1, 1)
    X_endog = clean[endog_names].to_numpy(dtype=np.float64)
    if X_endog.ndim == 1:
        X_endog = X_endog.reshape(-1, 1)
    Z = clean[instrument_names].to_numpy(dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n_exog = X_exog.shape[1]
    n_endog = X_endog.shape[1]
    n_z = Z.shape[1]

    # Stack everything that needs to be residualised into one matrix so
    # the AP loop runs once.
    stacked = np.column_stack([y, X_exog, X_endog, Z])
    fe_df = clean[absorb_terms]
    stacked_dem, info = _demean(
        stacked, fe_df,
        drop_singletons=True,
        tol=1e-12, max_iter=fe_maxiter, tol_abs=fe_tol,
    )

    keep_mask = info.keep_mask
    n_kept = int(info.n_kept)
    n_dropped = int(info.n_dropped)
    fe_card = list(info.n_fe)
    fe_dof = sum(int(g) - 1 for g in fe_card)

    # Slice out columns from the stacked residualised matrix.
    y_dem = stacked_dem[:, 0]
    col = 1
    X_exog_dem = stacked_dem[:, col:col + n_exog]
    col += n_exog
    X_endog_dem = stacked_dem[:, col:col + n_endog]
    col += n_endog
    Z_dem = stacked_dem[:, col:col + n_z]

    # Subset cluster column to kept rows so downstream ``_cluster_cov``
    # sees aligned data.
    if cluster_name is not None:
        cluster_kept = clean[cluster_name].iloc[keep_mask].reset_index(drop=True)
    else:
        cluster_kept = None

    # Do not include an intercept — the absorbed FE block already spans
    # the constant. ``var_names`` mirrors the keys IVRegression uses
    # when invoked via the matrix interface.
    var_names = {
        'dependent': dependent,
        'exog': list(exog_names),
        'endog': list(endog_names),
        'instruments': list(instrument_names),
    }

    return {
        'y': y_dem,
        'X_exog': X_exog_dem,
        'X_endog': X_endog_dem,
        'Z': Z_dem,
        'cluster_series': cluster_kept,
        'var_names': var_names,
        'n_obs': int(n_obs),
        'n_kept': n_kept,
        'n_dropped': n_dropped,
        'fe_dof': int(fe_dof),
        'fe_cardinality': fe_card,
        'absorb_terms': list(absorb_terms),
    }


def _iv_absorb_run(
    formula: str,
    data: pd.DataFrame,
    absorb_terms: List[str],
    method: str,
    robust: str,
    cluster: Optional[str],
    **kwargs: Any,
):
    """Internal helper: run 2SLS with HDFE absorption.

    Returns ``(result, model, pre)`` where ``pre`` is the dict from
    :func:`_iv_absorb_preprocess`. The dispatcher uses ``model`` to
    attach Kleibergen-Paap / Sanderson-Windmeijer / effective-F
    diagnostics in residualised space.

    Restricted to ``method='2sls'`` for now — LIML/Fuller/GMM/JIVE need
    their kappa / weighting reformulated in residualised space (Phase 3b).
    """
    if method != '2sls':
        raise NotImplementedError(
            f"absorb= is currently only wired for method='2sls'; got "
            f"method={method!r}. LIML/Fuller/GMM/JIVE need a kappa / "
            "weighting reformulation in residualised space — track in "
            "Phase 3b."
        )

    pre = _iv_absorb_preprocess(
        formula=formula, data=data,
        absorb_terms=absorb_terms,
        cluster_name=cluster,
    )
    if pre['cluster_series'] is not None:
        cluster_df = pd.DataFrame({cluster: pre['cluster_series']})
    else:
        cluster_df = None

    model = IVRegression(
        method='2sls',
        y=pre['y'],
        X_exog=pre['X_exog'],
        X_endog=pre['X_endog'],
        Z=pre['Z'],
        var_names=pre['var_names'],
    )
    # Inject cluster_df so fit()'s ``cluster_var = self.data[cluster]``
    # branch finds the kept-rows cluster series.
    model.data = cluster_df
    result = model.fit(robust=robust, cluster=cluster, **kwargs)

    k_total = pre['X_exog'].shape[1] + pre['X_endog'].shape[1]
    _scale_vcov_for_fe_dof(
        result, fe_dof=pre['fe_dof'],
        n_kept=pre['n_kept'], k=k_total,
    )

    if hasattr(result, 'model_info') and isinstance(result.model_info, dict):
        result.model_info['absorb'] = list(absorb_terms)
        result.model_info['fe_cardinality'] = list(pre['fe_cardinality'])
        result.model_info['fe_dof'] = int(pre['fe_dof'])
        result.model_info['n_dropped_singletons'] = int(pre['n_dropped'])

    return result, model, pre


def _scale_vcov_for_fe_dof(
    result: EconometricResults,
    fe_dof: int,
    n_kept: int,
    k: int,
) -> None:
    """Charge ``fe_dof`` against the residual DOF on a fitted IV result.

    Multiplies the variance matrix by ``(n - k) / (n - k - fe_dof)`` —
    correct for nonrobust, HC1, and CR1 because all three small-sample
    factors contain ``1 / (n - k)`` in exactly that position. Updates
    std_errors, t-stats, p-values, and ``df_resid`` to match.
    """
    df_resid_old = max(n_kept - k, 1)
    df_resid_new = max(n_kept - k - fe_dof, 1)
    if fe_dof <= 0 or df_resid_new == df_resid_old:
        return
    factor = df_resid_old / df_resid_new
    sqrt_factor = float(np.sqrt(factor))
    # ``EconometricResults`` stores SE as a Series and exposes the raw
    # var_cov via ``_var_cov`` (private). We touch both so any consumer
    # downstream sees a consistent view.
    if hasattr(result, '_var_cov') and result._var_cov is not None:
        result._var_cov = result._var_cov * factor
    if hasattr(result, 'std_errors') and result.std_errors is not None:
        result.std_errors = result.std_errors * sqrt_factor
    if hasattr(result, 'data_info') and isinstance(result.data_info, dict):
        result.data_info['df_resid'] = int(df_resid_new)


def iv(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    method: str = '2sls',
    robust: str = 'nonrobust',
    cluster: Optional[str] = None,
    fuller_alpha: float = 1.0,
    absorb: Optional[Union[str, List[str]]] = None,
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
    absorb : str or list of str, optional
        Column name(s) of high-dimensional fixed effects to **partial out**
        before fitting (e.g. ``absorb="firm"`` or
        ``absorb=["firm", "year"]``). Routes ``y``, exogenous controls,
        endogenous regressors, and instruments through
        :func:`sp.fast.demean` (Rust HDFE backend) and drops singletons,
        then runs 2SLS in residualised space. The intercept is dropped
        because the absorbed FEs span the constant. The residual DOF is
        adjusted by ``sum(G_k - 1)``, mirroring
        :func:`sp.fast.feols(absorb=...)`. Currently only wired for
        ``method='2sls'``; LIML / Fuller / GMM / JIVE raise
        ``NotImplementedError`` (Phase 3b).

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
    absorb_terms = _normalise_absorb(absorb)
    if absorb_terms:
        if formula is None or data is None:
            raise ValueError(
                "absorb= requires (formula, data) — matrix mode is not "
                "supported. Build the formula and pass the DataFrame."
            )
        _result, _model, _pre = _iv_absorb_run(
            formula=formula, data=data,
            absorb_terms=absorb_terms,
            method=method, robust=robust, cluster=cluster,
            **kwargs,
        )
    else:
        model = IVRegression(
            formula=formula, data=data, method=method,
            fuller_alpha=fuller_alpha,
        )
        _result = model.fit(robust=robust, cluster=cluster, **kwargs)
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.iv",
            params={
                "formula": formula,
                "method": method,
                "robust": robust,
                "cluster": cluster,
                "fuller_alpha": fuller_alpha,
                "absorb": list(absorb_terms) if absorb_terms else None,
                **{k: v for k, v in kwargs.items()
                   if k in ("weights", "se_type", "vcov")},
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover — provenance must never break fit
        pass
    return _result


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

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> z = rng.normal(size=n)
    >>> u = rng.normal(size=n)
    >>> x = 0.8 * z + u + rng.normal(size=n)        # endogenous regressor
    >>> y = 1.5 * x + 2.0 * u + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'x': x, 'z': z})
    >>> result = sp.ivreg("y ~ (x ~ z)", data=df)
    >>> bool(abs(result.params['x'] - 1.5) < 0.2)  # 2SLS recovers the true effect
    True

    >>> # Preferred modern entry point:
    >>> result = sp.iv("y ~ (x ~ z)", data=df, method='2sls')
    """
    kwargs.setdefault('method', '2sls')
    return iv(formula=formula, data=data,
              robust=robust, cluster=cluster, **kwargs)
