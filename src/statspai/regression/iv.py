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

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.base import BaseEstimator, BaseModel
from ..core.results import EconometricResults
from ..core.utils import parse_formula
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise MethodIncompatibility(
            f"`{name}` must be a string.",
            diagnostics={name: repr(value)},
        )
    return value


def _require_dataframe(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise MethodIncompatibility(
            f"`{name}` must be a pandas DataFrame.",
            diagnostics={name: type(value).__name__},
        )
    return value


def _not_fitted_error(accessor: str) -> MethodIncompatibility:
    return MethodIncompatibility(
        "Model must be fitted first.",
        recovery_hint="Call fit() before accessing fitted IV diagnostics.",
        diagnostics={"accessor": accessor, "is_fitted": False},
    )


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


# ====================================================================== #
#  K-class estimator (unifies 2SLS, LIML, Fuller, and user-specified k)
# ====================================================================== #


def _k_class_fit(
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
    kappa: float,
    robust: str = "nonrobust",
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
                "sp.iv_bounds (Nevo-Rosen bounds from the instruments you "
                "have) or sp.manski_bounds (assumption-free worst case)."
            ),
            diagnostics={"n_instruments": m, "n_endogenous": k2},
            alternative_functions=["sp.iv_bounds", "sp.manski_bounds"],
        )

    # Full instrument matrix: [X_exog, Z]
    W = np.column_stack([X_exog, Z])

    # --- First stage (for diagnostics & projections) ---
    WtW_inv = np.linalg.inv(W.T @ W)
    P_W = W @ WtW_inv @ W.T

    first_stage_results = _first_stage_diagnostics(
        X_exog,
        X_endog,
        W,
        n,
        m,
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
    elif robust != "nonrobust":
        var_cov = _robust_cov(AX, A, residuals, XAX_inv, robust, n, k)
    else:
        sigma2 = np.sum(residuals**2) / (n - k)
        var_cov = sigma2 * XAX_inv

    std_errors = np.sqrt(np.maximum(np.diag(var_cov), 0))

    # --- Model diagnostics ---
    y_bar = np.mean(y)
    tss = np.sum((y - y_bar) ** 2)
    rss = np.sum(residuals**2)
    r_squared = 1 - rss / tss

    # Over-identification test. Sargan under i.i.d. errors; Hansen's J
    # (the ivreg2 convention) once the vcov is robust or clustered, where
    # Sargan is no longer valid.
    sargan = _sargan_test(residuals, W, m, k2) if m > k2 else None
    hansen = _hansen_j(y, X_actual, W, residuals, robust, cluster)

    # Durbin-Wu-Hausman endogeneity test
    hausman = _hausman_test(y, X_exog, X_endog, W)

    return {
        "params": params,
        "std_errors": std_errors,
        "var_cov": var_cov,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "r_squared": r_squared,
        "nobs": n,
        "df_model": k - 1,
        "df_resid": n - k,
        "rss": rss,
        "tss": tss,
        "first_stage": first_stage_results,
        "sargan": sargan,
        "hansen": hansen,
        "hausman": hausman,
        "n_instruments": m,
        "n_endogenous": k2,
        "kappa": float(kappa),
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
                RuntimeWarning,
                stacklevel=2,
            )
            kappa = 1.0
    except Exception:
        warnings.warn(
            "LIML generalized eigenvalue solve failed; falling back to 2SLS.",
            RuntimeWarning,
            stacklevel=2,
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
    robust: str = "nonrobust",
    cluster: Optional[pd.Series] = None,
    gmm_vcov: str = "sandwich",
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
                "sp.iv_bounds (Nevo-Rosen bounds from the instruments you "
                "have) or sp.manski_bounds (assumption-free worst case)."
            ),
            diagnostics={"n_instruments": m, "n_endogenous": k2},
            alternative_functions=["sp.iv_bounds", "sp.manski_bounds"],
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
        # Cluster-robust weighting matrix. With multiway clustering the
        # inclusion-exclusion matrix is not guaranteed PSD, so it cannot
        # serve as a GMM weight; ``ivreg2`` uses the first clustering
        # dimension for W and the multiway estimator only for the VCE.
        # We follow that convention.
        _frame = _as_cluster_frame(cluster)
        S, _ = _cluster_meat(W, resid_init, _cluster_codes(_frame, (0,)))
        S /= n
    elif robust != "nonrobust":
        # Heteroskedasticity-robust weighting matrix
        S = (W * resid_init[:, np.newaxis]).T @ (W * resid_init[:, np.newaxis]) / n
    else:
        # Homoskedastic weighting matrix
        sigma2 = np.sum(resid_init**2) / n
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

    # --- GMM variance -------------------------------------------------
    # ``gmm_vcov="efficient"`` reports the textbook efficient-GMM variance
    #     V = q * (X'W S^{-1} W'X)^{-1},
    # which is what ``ivreg2``/``ivreghdfe`` print. ``"sandwich"`` (the
    # StatsPAI default) keeps the full sandwich with Omega re-estimated at
    # the final residuals, which stays valid if the weight matrix is not
    # the efficient one. Both carry the same finite-sample factor q as the
    # k-class path: G/(G-1) * (n-1)/(n-k) when clustered, n/(n-k) for HC1,
    # 1 otherwise.
    if cluster is not None:
        _frame = _as_cluster_frame(cluster)
        _codes = _cluster_codes(_frame, (0,))
        _g = int(_codes.max()) + 1
        ssc = (_g / (_g - 1)) * ((n - 1) / max(n - k, 1))
    elif robust != "nonrobust":
        ssc = n / max(n - k, 1)
    else:
        ssc = 1.0

    if gmm_vcov == "efficient":
        # bread == (1/n) (X'W Sigma^{-1} W'X)^{-1} because S == Sigma / n.
        var_cov = ssc * n * bread
    else:
        # Omega must match the clustering structure of the weight matrix;
        # using the heteroskedasticity form under cluster= understates the
        # variance whenever moments are correlated within cluster.
        if cluster is not None:
            Omega, _ = _cluster_meat(W, residuals, _codes)
            Omega = Omega / n
        else:
            We = W * residuals[:, np.newaxis]
            Omega = We.T @ We / n

        Q_xw = XW / n
        Q_xw_Sinv = Q_xw @ S_inv
        bread_n = np.linalg.inv(Q_xw_Sinv @ Q_xw.T)
        meat_n = Q_xw_Sinv @ Omega @ S_inv @ Q_xw.T
        var_cov = ssc * (bread_n @ meat_n @ bread_n) / n

    std_errors = np.sqrt(np.maximum(np.diag(var_cov), 0))

    # Diagnostics
    y_bar = np.mean(y)
    tss = np.sum((y - y_bar) ** 2)
    rss = np.sum(residuals**2)
    r_squared = 1 - rss / tss

    first_stage_results = _first_stage_diagnostics(X_exog, X_endog, W, n, m)

    # Hansen J test (GMM overidentification)
    if m > k2:
        g_bar = W.T @ residuals / n
        j_stat = float(n * g_bar @ S_inv @ g_bar)
        j_df = m - k2
        j_pvalue = float(1 - stats.chi2.cdf(j_stat, j_df))
        hansen_j = {"statistic": j_stat, "pvalue": j_pvalue, "df": j_df}
    else:
        hansen_j = None

    hausman = _hausman_test(y, X_exog, X_endog, W)

    return {
        "params": params,
        "std_errors": std_errors,
        "var_cov": var_cov,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "r_squared": r_squared,
        "nobs": n,
        "df_model": k - 1,
        "df_resid": n - k,
        "rss": rss,
        "tss": tss,
        "first_stage": first_stage_results,
        "sargan": hansen_j,  # Hansen J generalises Sargan
        "hausman": hausman,
        "n_instruments": m,
        "n_endogenous": k2,
        "kappa": None,
    }


# ====================================================================== #
#  JIVE estimator
# ====================================================================== #


def _jive_fit(
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
    robust: str = "nonrobust",
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
                "sp.iv_bounds (Nevo-Rosen bounds from the instruments you "
                "have) or sp.manski_bounds (assumption-free worst case)."
            ),
            diagnostics={"n_instruments": m, "n_endogenous": k2},
            alternative_functions=["sp.iv_bounds", "sp.manski_bounds"],
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
    # Float buffer regardless of the input dtype: an integer-typed
    # endogenous regressor (0/1 treatment, counts) would otherwise
    # truncate the leave-one-out fitted values on assignment. Same
    # failure mode fixed in _hausman_test.
    X_endog_jive = np.empty_like(X_endog, dtype=np.float64)
    for j in range(k2):
        X_endog_jive[:, j] = (X_endog_hat_full[:, j] - h * X_endog[:, j]) / (1 - h)

    # Second stage with JIVE fitted values
    X_hat_jive = np.column_stack([X_exog, X_endog_jive])
    XhXh_inv = np.linalg.inv(X_hat_jive.T @ X_hat_jive)
    params = XhXh_inv @ X_hat_jive.T @ y

    fitted_values = X_actual @ params
    residuals = y - fitted_values

    # Standard errors (HC1-style with JIVE bread)
    if cluster is not None:
        var_cov = _cluster_cov(X_hat_jive, np.eye(n), residuals, XhXh_inv, cluster)
    elif robust != "nonrobust":
        var_cov = _robust_cov(X_hat_jive, np.eye(n), residuals, XhXh_inv, robust, n, k)
    else:
        sigma2 = np.sum(residuals**2) / (n - k)
        var_cov = sigma2 * XhXh_inv

    std_errors = np.sqrt(np.maximum(np.diag(var_cov), 0))

    # Diagnostics
    y_bar = np.mean(y)
    tss = np.sum((y - y_bar) ** 2)
    rss = np.sum(residuals**2)

    first_stage_results = _first_stage_diagnostics(X_exog, X_endog, W, n, m)
    sargan = _sargan_test(residuals, W, m, k2) if m > k2 else None
    hausman = _hausman_test(y, X_exog, X_endog, W)

    return {
        "params": params,
        "std_errors": std_errors,
        "var_cov": var_cov,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "r_squared": 1 - rss / tss,
        "nobs": n,
        "df_model": k - 1,
        "df_resid": n - k,
        "rss": rss,
        "tss": tss,
        "first_stage": first_stage_results,
        "sargan": sargan,
        "hausman": hausman,
        "n_instruments": m,
        "n_endogenous": k2,
        "kappa": None,
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

        results.append(
            {
                "f_statistic": f_stat,
                "f_pvalue": f_pvalue,
                "partial_r_squared": (
                    1 - rss_full / rss_restricted if rss_restricted > 0 else np.nan
                ),
            }
        )

    return results


def _normalize_robust(robust: Any) -> str:
    """Canonicalise the SE-type vocabulary for the IV estimators.

    Accepts (case-insensitively) ``'nonrobust'`` / ``'hc0'`` / ``'hc1'`` /
    ``'hc2'`` / ``'hc3'`` plus the ergonomic aliases ``True`` / ``'robust'`` /
    ``'white'`` so callers can mirror Stata (``robust`` ≡ HC1) and the
    ``sp.regress`` spelling (uppercase HCk). Raises a structured taxonomy
    error for anything else instead of failing deep inside the sandwich kernel.
    """
    if robust is None or robust is False:
        return "nonrobust"
    if robust is True:
        return "hc1"  # Stata `robust` ≡ HC1
    if isinstance(robust, str):
        key = robust.strip().lower()
        aliases = {"robust": "hc1", "white": "hc0"}
        key = aliases.get(key, key)
        if key in ("nonrobust", "hc0", "hc1", "hc2", "hc3"):
            return key
    raise MethodIncompatibility(
        f"Unknown robust option: {robust!r}. Use one of 'nonrobust', "
        f"'hc0', 'hc1', 'hc2', 'hc3' (case-insensitive), True, or 'robust'.",
        recovery_hint=(
            "Use robust='nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', "
            "True, 'robust', or 'white'."
        ),
        diagnostics={"robust": repr(robust)},
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
    if robust_type == "hc0":
        weights = residuals**2
    elif robust_type == "hc1":
        weights = (n / (n - k)) * residuals**2
    elif robust_type in ("hc2", "hc3"):
        h = np.diag(X_hat @ bread @ X_hat.T)
        h = np.clip(h, 0, 1 - 1e-8)
        if robust_type == "hc2":
            weights = residuals**2 / (1 - h)
        else:
            weights = residuals**2 / (1 - h) ** 2
    else:
        raise ValueError(f"Unknown robust type: {robust_type}")

    meat = X_hat.T @ np.diag(weights) @ X_hat
    return _as_float_array(bread @ meat @ bread)


def _normalise_cluster(cluster: Any) -> Optional[Tuple[str, ...]]:
    """Return cluster column names when ``cluster`` names columns.

    Accepts ``"county"``, ``"county + ym"`` (Stata/fixest spelling) and
    ``["county", "ym"]``. Returns ``None`` for array-likes, which are
    handled as data rather than as column references.
    """
    if cluster is None:
        return None
    if isinstance(cluster, str):
        return tuple(t.strip() for t in cluster.split("+") if t.strip())
    if isinstance(cluster, (list, tuple)) and all(isinstance(c, str) for c in cluster):
        return tuple(cluster)
    return None


def _as_cluster_frame(cluster: Any) -> pd.DataFrame:
    """Normalise any cluster specification to a ``(n, d)`` DataFrame.

    Accepts a Series, a DataFrame (one column per clustering dimension),
    a 1-D array, or a 2-D array. Multiway clustering is signalled by
    ``d > 1``.
    """
    if isinstance(cluster, pd.DataFrame):
        return cluster.reset_index(drop=True)
    if isinstance(cluster, pd.Series):
        return cluster.to_frame().reset_index(drop=True)
    arr = np.asarray(cluster)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=[f"cluster{i}" for i in range(arr.shape[1])])


def _cluster_codes(frame: pd.DataFrame, cols: Tuple[int, ...]) -> np.ndarray:
    """Integer group codes for the intersection of ``cols`` in ``frame``."""
    sub = frame.iloc[:, list(cols)]
    if len(cols) == 1:
        return pd.factorize(sub.iloc[:, 0], sort=False)[0]
    keys = pd.MultiIndex.from_frame(sub.astype(object))
    return pd.factorize(keys, sort=False)[0]


def _cluster_meat(
    X_hat: np.ndarray,
    residuals: np.ndarray,
    codes: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Sum of outer products of within-cluster score sums.

    Vectorised: scores are summed per cluster with ``np.add.at`` rather
    than looping over cluster ids, which matters for county-month panels
    with thousands of clusters.
    """
    scores = X_hat * residuals[:, np.newaxis]
    n_clusters = int(codes.max()) + 1 if codes.size else 0
    sums = np.zeros((n_clusters, X_hat.shape[1]), dtype=float)
    np.add.at(sums, codes, scores)
    return sums.T @ sums, n_clusters


def _psd_project(V: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone (Cameron-Gelbach-Miller).

    The multiway inclusion-exclusion variance is not guaranteed PSD in
    finite samples; the standard fix (and what ``ivreg2``/``reghdfe`` do)
    is to zero out negative eigenvalues.
    """
    V = 0.5 * (V + V.T)
    evals, evecs = np.linalg.eigh(V)
    if np.all(evals >= 0):
        return V
    evals = np.maximum(evals, 0.0)
    return evecs @ np.diag(evals) @ evecs.T


def _hansen_j(
    y: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    resid: np.ndarray,
    robust: str,
    cluster: Any,
) -> Optional[Dict[str, float]]:
    """Hansen's J for a k-class fit under heteroskedasticity/clustering.

    The Sargan statistic assumes i.i.d. errors and is not valid once the
    vcov is robust or clustered; ``ivreg2`` therefore switches its
    over-identification test to Hansen's J, evaluated at the efficient
    two-step GMM estimate whose weight matrix is built from the k-class
    residuals. This reproduces that number.

    Returns ``None`` when the model is exactly identified.
    """
    n, k = X.shape
    m = W.shape[1]
    df = m - k
    if df <= 0:
        return None

    if cluster is not None:
        frame = _as_cluster_frame(cluster)
        from itertools import combinations

        d = frame.shape[1]
        S = np.zeros((m, m))
        for size in range(1, d + 1):
            sign = 1.0 if size % 2 == 1 else -1.0
            for cols in combinations(range(d), size):
                meat, _ = _cluster_meat(W, resid, _cluster_codes(frame, cols))
                S += sign * meat
        if d > 1:
            S = _psd_project(S)
    elif robust != "nonrobust":
        Wu = W * resid[:, np.newaxis]
        S = Wu.T @ Wu
    else:
        return None  # caller keeps the Sargan statistic

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None

    # Efficient two-step GMM estimate under this S, then J at that point.
    XW = X.T @ W
    try:
        beta_gmm = np.linalg.solve(XW @ S_inv @ XW.T, XW @ S_inv @ (W.T @ y))
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None
    g = W.T @ (y - X @ beta_gmm)
    stat = float(g @ S_inv @ g)
    return {
        "statistic": stat,
        "pvalue": float(1 - stats.chi2.cdf(stat, df)),
        "df": int(df),
    }


def _cluster_cov(
    X_hat: np.ndarray,
    A: np.ndarray,
    residuals: np.ndarray,
    bread: np.ndarray,
    cluster: Any,
) -> np.ndarray:
    r"""Cluster-robust (one-way or multiway) IV variance.

    One-way reproduces Stata ``ivregress 2sls, cluster()``. With ``d > 1``
    clustering dimensions this is the Cameron, Gelbach & Miller (2011)
    inclusion-exclusion estimator

    .. math:: V = \sum_{\emptyset \ne S \subseteq \{1..d\}}
              (-1)^{|S|+1} V_S,

    with a single ``ivreg2``-style finite-sample factor
    ``G_min/(G_min-1) * (n-1)/(n-k)`` and a PSD projection at the end.

    Absorbed-FE degrees of freedom are *not* handled here: the absorb
    path rescales the whole variance once via ``_scale_vcov_for_fe_dof``.
    """
    from itertools import combinations

    n, k = X_hat.shape
    frame = _as_cluster_frame(cluster)
    d = frame.shape[1]

    V = np.zeros((k, k))
    g_min: Optional[int] = None
    for size in range(1, d + 1):
        sign = 1.0 if size % 2 == 1 else -1.0
        for cols in combinations(range(d), size):
            codes = _cluster_codes(frame, cols)
            meat, n_clusters = _cluster_meat(X_hat, residuals, codes)
            if n_clusters <= 1:
                raise DataInsufficient(
                    "Clustered IV inference needs at least two clusters in "
                    f"every dimension; dimension {cols} has {n_clusters}.",
                    recovery_hint=(
                        "Drop the degenerate clustering dimension, or use "
                        "sp.wild_cluster_boot / sp.cr2_se for few-cluster "
                        "inference."
                    ),
                    diagnostics={"dimension": cols, "n_clusters": n_clusters},
                )
            if size == 1:
                g_min = n_clusters if g_min is None else min(g_min, n_clusters)
            V += sign * (bread @ meat @ bread)

    # ivreg2 convention: ONE finite-sample factor, built from the smallest
    # cluster count, applied to the assembled matrix. (fixest instead
    # corrects each inclusion-exclusion component separately; the two agree
    # to about three digits.)
    assert g_min is not None  # d >= 1 always makes a one-way pass
    V = (g_min / (g_min - 1)) * ((n - 1) / max(n - k, 1)) * V
    if d > 1:
        V = _psd_project(V)
    return _as_float_array(V)


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

    return {"statistic": stat, "pvalue": pvalue, "df": df}


def _hausman_test(
    y: np.ndarray,
    X_exog: np.ndarray,
    X_endog: np.ndarray,
    W: np.ndarray,
) -> Dict[str, float]:
    """Durbin-Wu-Hausman endogeneity test (regression-based)."""
    n = len(y)
    k2 = X_endog.shape[1]

    # ``np.empty_like(X_endog)`` inherits the *input* dtype. With an
    # integer-typed endogenous regressor — a 0/1 treatment or a count,
    # i.e. the common case — the first-stage residuals below were
    # truncated toward zero on assignment: identically 0 for a binary
    # treatment (singular augmented design -> NaN statistic) and
    # silently biased for integer counts. Always accumulate in float.
    X_endog = np.asarray(X_endog, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X_exog = np.asarray(X_exog, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    WtW_inv = np.linalg.inv(W.T @ W)
    v_hat = np.empty_like(X_endog, dtype=np.float64)
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

    return {"statistic": f_stat, "pvalue": f_pvalue, "df": k2}


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
        X: np.ndarray,
        X_endog: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        robust: Any = "nonrobust",
        cluster: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if X_endog is None:
            X_endog = kwargs.get("X_endog")
        if Z is None:
            Z = kwargs.get("Z")
        if X_endog is None or Z is None:
            raise MethodIncompatibility(
                "IVEstimator.estimate requires X_endog and Z.",
                diagnostics={
                    "has_X_endog": X_endog is not None,
                    "has_Z": Z is not None,
                },
            )
        return _k_class_fit(
            y,
            X,
            X_endog,
            Z,
            kappa=1.0,
            robust=_normalize_robust(robust),
            cluster=cluster,
        )


# ====================================================================== #
#  Method name → label mapping
# ====================================================================== #

_METHOD_LABELS = {
    "2sls": "IV-2SLS",
    "liml": "IV-LIML",
    "fuller": "IV-Fuller",
    "gmm": "IV-GMM (2-step)",
    "jive": "IV-JIVE",
}

_METHOD_DESCRIPTIONS = {
    "2sls": "Two-Stage Least Squares",
    "liml": "Limited Information Maximum Likelihood",
    "fuller": "Fuller Modified LIML",
    "gmm": "Efficient Two-Step GMM",
    "jive": "Jackknife Instrumental Variables",
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

    References
    ----------
    Angrist, J. D., Imbens, G. W. and Rubin, D. B. (1996). Identification
    of Causal Effects Using Instrumental Variables. *Journal of the
    American Statistical Association*. doi:10.1080/01621459.1996.10476902
    [@angrist1996identification]

    Angrist, J. D. and Pischke, J.-S. (2009). *Mostly Harmless
    Econometrics: An Empiricist's Companion*. Princeton University Press.
    [@angrist2009mostly]

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
        method: str = "2sls",
        fuller_alpha: float = 1.0,
        y: Optional[np.ndarray] = None,
        X_exog: Optional[np.ndarray] = None,
        X_endog: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        var_names: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._results: Optional[EconometricResults] = None
        self._exog_names: List[str] = []
        self._endog_names: List[str] = []
        self._instrument_names: List[str] = []
        self._first_stage: List[Dict[str, float]] = []
        self._sargan: Optional[Dict[str, float]] = None
        self._hausman: Dict[str, float] = {}
        self._instruments: List[str] = []
        self._raw_results: Dict[str, Any] = {}
        if formula is not None:
            formula = _require_string(formula, "formula")
        if data is not None:
            data = _require_dataframe(data, "data")
        method = _require_string(method, "method")
        self.formula = formula
        self.data = data
        self.method = method.lower()
        self.fuller_alpha = fuller_alpha
        self.y = y
        self.X_exog = X_exog
        self.X_endog = X_endog
        self.Z = Z
        self.var_names = var_names

        if self.method not in ("2sls", "liml", "fuller", "gmm", "jive"):
            raise MethodIncompatibility(
                f"Unknown IV method '{method}'. "
                f"Choose from: 2sls, liml, fuller, gmm, jive",
                recovery_hint=(
                    "Use method='2sls', 'liml', 'fuller', 'gmm', or 'jive'."
                ),
                diagnostics={
                    "method": method,
                    "valid": ["2sls", "liml", "fuller", "gmm", "jive"],
                },
            )

    def _prepare_from_formula(self) -> None:
        """Parse formula and build matrices from data."""
        if self.formula is None or self.data is None:
            raise MethodIncompatibility(
                "Formula preparation requires both formula and data.",
                diagnostics={
                    "has_formula": self.formula is not None,
                    "has_data": self.data is not None,
                },
            )
        data = self.data
        parsed = parse_formula(self.formula)

        if not parsed["endogenous"] or not parsed["instruments"]:
            raise MethodIncompatibility(
                "IV formula must specify endogenous variables and instruments. "
                'Use syntax: "y ~ (endog ~ z1 + z2) + exog"',
                recovery_hint=("Write the endogenous block as (endog ~ instrument)."),
                diagnostics={"formula": self.formula},
            )

        self.dependent_var = parsed["dependent"]
        exog_names = parsed["exogenous"]
        endog_names = parsed["endogenous"]
        instrument_names = parsed["instruments"]

        all_vars = [self.dependent_var] + exog_names + endog_names + instrument_names
        missing = [v for v in all_vars if v not in data.columns]
        if missing:
            raise MethodIncompatibility(
                f"Variables not found in data: {missing}",
                recovery_hint=(
                    "Add the missing outcome, exogenous, endogenous, or "
                    "instrument columns to the DataFrame."
                ),
                diagnostics={"missing_columns": missing},
            )

        extra_cols = [c for c in data.columns if c not in all_vars]
        clean = data[all_vars + extra_cols].dropna(subset=all_vars)
        if len(clean) == 0:
            raise DataInsufficient(
                "No rows remain after dropping NaNs.",
                recovery_hint=(
                    "Provide at least one complete row for the IV formula " "variables."
                ),
                diagnostics={"required_columns": all_vars},
            )

        self.y = clean[self.dependent_var].values

        if parsed["has_constant"]:
            const = np.ones((len(clean), 1))
            if exog_names:
                self.X_exog = np.column_stack([const, clean[exog_names].values])
            else:
                self.X_exog = const
            self._exog_names = ["Intercept"] + exog_names
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
        robust: Any = "nonrobust",
        cluster: Optional[str] = None,
        **kwargs: Any,
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
        cluster : str, list of str, Series or DataFrame, optional
            Clustering dimension(s). More than one selects multiway
            (Cameron-Gelbach-Miller) clustering.

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
        elif not (
            self.y is not None
            and self.X_exog is not None
            and self.X_endog is not None
            and self.Z is not None
        ):
            raise MethodIncompatibility(
                "Provide either (formula, data) or (y, X_exog, X_endog, Z).",
                recovery_hint=(
                    "Pass a formula with a DataFrame, or pass all four raw " "arrays."
                ),
                diagnostics={
                    "has_formula": self.formula is not None,
                    "has_data": self.data is not None,
                    "has_y": self.y is not None,
                    "has_X_exog": self.X_exog is not None,
                    "has_X_endog": self.X_endog is not None,
                    "has_Z": self.Z is not None,
                },
            )
        else:
            y_arr = self.y
            X_exog_arr = self.X_exog
            X_endog_arr = self.X_endog
            Z_arr = self.Z
            if (
                y_arr is None
                or X_exog_arr is None
                or X_endog_arr is None
                or Z_arr is None
            ):
                raise MethodIncompatibility(
                    "Raw IV design arrays are incomplete.",
                    diagnostics={
                        "has_y": y_arr is not None,
                        "has_X_exog": X_exog_arr is not None,
                        "has_X_endog": X_endog_arr is not None,
                        "has_Z": Z_arr is not None,
                    },
                )
            self._exog_names = (
                self.var_names.get(
                    "exog",
                    [f"exog{i}" for i in range(X_exog_arr.shape[1])],
                )
                if self.var_names
                else [f"exog{i}" for i in range(X_exog_arr.shape[1])]
            )
            self._endog_names = (
                self.var_names.get(
                    "endog",
                    [f"endog{i}" for i in range(X_endog_arr.shape[1])],
                )
                if self.var_names
                else [f"endog{i}" for i in range(X_endog_arr.shape[1])]
            )
            self._instrument_names = (
                self.var_names.get(
                    "instruments",
                    [f"z{i}" for i in range(Z_arr.shape[1])],
                )
                if self.var_names
                else [f"z{i}" for i in range(Z_arr.shape[1])]
            )
            self.dependent_var = (
                self.var_names.get("dependent", "y") if self.var_names else "y"
            )

        y_fit = self.y
        X_exog_fit = self.X_exog
        X_endog_fit = self.X_endog
        Z_fit = self.Z
        if y_fit is None or X_exog_fit is None or X_endog_fit is None or Z_fit is None:
            raise MethodIncompatibility(
                "IV design arrays are unavailable after preparation.",
                diagnostics={
                    "has_y": y_fit is not None,
                    "has_X_exog": X_exog_fit is not None,
                    "has_X_endog": X_endog_fit is not None,
                    "has_Z": Z_fit is not None,
                },
            )

        # Cluster variable. Accept either a column-name string (looked up in
        # the model data) or an array-like / pandas Series aligned with the
        # estimation sample. The ``cluster is not None`` guard replaces a bare
        # ``if cluster`` truth test that raised "truth value of a Series is
        # ambiguous" when a Series was passed -- even though the public ``iv``
        # signature documents ``cluster`` as a ``pd.Series``.
        cluster_var = None
        if cluster is not None:
            cluster_spec = _normalise_cluster(cluster)
            if cluster_spec is not None and all(
                isinstance(c, str) for c in cluster_spec
            ):
                src = getattr(self, "_clean_data", None)
                if src is None:
                    src = self.data
                cols = list(getattr(src, "columns", []))
                missing = [c for c in cluster_spec if c not in cols]
                if src is None or missing:
                    raise MethodIncompatibility(
                        f"Cluster variable not found in data: {missing or cluster!r}",
                        recovery_hint=(
                            "Pass cluster column(s) present in the model data, "
                            "or an array/Series aligned with the sample."
                        ),
                        diagnostics={"cluster": cluster, "missing": missing},
                    )
                cluster_var = src.loc[:, list(cluster_spec)]
            else:
                # Array-like / Series / DataFrame passed directly.
                cluster_var = _as_cluster_frame(cluster)
                if len(cluster_var) != len(y_fit):
                    raise MethodIncompatibility(
                        "Cluster vector length does not match the estimation "
                        "sample.",
                        recovery_hint=(
                            "Pass a cluster column name, or an array/Series "
                            "with exactly one entry per observation."
                        ),
                        diagnostics={
                            "n_cluster": int(len(cluster_var)),
                            "n_obs": int(len(y_fit)),
                        },
                    )

        # --- Dispatch to estimation method ---
        method = self.method

        if method in ("2sls", "liml", "fuller"):
            if method == "2sls":
                kappa = 1.0
            elif method == "liml":
                kappa = _liml_kappa(y_fit, X_exog_fit, X_endog_fit, Z_fit)
            else:  # fuller
                kappa_liml = _liml_kappa(
                    y_fit,
                    X_exog_fit,
                    X_endog_fit,
                    Z_fit,
                )
                n = len(y_fit)
                K = X_exog_fit.shape[1] + Z_fit.shape[1]
                kappa = kappa_liml - self.fuller_alpha / (n - K)

            results = _k_class_fit(
                y_fit,
                X_exog_fit,
                X_endog_fit,
                Z_fit,
                kappa=kappa,
                robust=robust,
                cluster=cluster_var,
            )

        elif method == "gmm":
            results = _gmm_fit(
                y_fit,
                X_exog_fit,
                X_endog_fit,
                Z_fit,
                robust=robust,
                cluster=cluster_var,
                gmm_vcov=str(kwargs.get("gmm_vcov", "sandwich")).lower(),
            )

        elif method == "jive":
            results = _jive_fit(
                y_fit,
                X_exog_fit,
                X_endog_fit,
                Z_fit,
                robust=robust,
                cluster=cluster_var,
            )

        # Build results object
        all_names = self._exog_names + self._endog_names
        params = pd.Series(results["params"], index=all_names)
        std_errors = pd.Series(results["std_errors"], index=all_names)

        method_label = _METHOD_LABELS.get(method, method.upper())
        method_desc = _METHOD_DESCRIPTIONS.get(method, method)

        model_info = {
            "model_type": method_label,
            "method": method_desc,
            "robust": robust,
            "cluster": cluster,
        }
        if cluster_var is not None:
            try:
                _cf = _as_cluster_frame(cluster_var)
                _counts = [int(_cf.iloc[:, j].nunique()) for j in range(_cf.shape[1])]
                model_info["n_clusters"] = _counts[0] if len(_counts) == 1 else _counts
                if _cf.shape[1] > 1:
                    model_info["cluster_dims"] = list(_cf.columns)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        if results.get("kappa") is not None:
            model_info["kappa"] = results["kappa"]

        # Surface the first-stage strength so it is machine-readable downstream
        # (``result.violations()`` / ``sp.audit_result``), not just printed in
        # the human ``diagnostics`` table. ``first_stage_f`` is the binding
        # (weakest) instrument across endogenous regressors — the quantity the
        # Stock-Yogo weak-IV threshold applies to.
        _fs = results.get("first_stage") or []
        _fs_fvals = [
            fs["f_statistic"]
            for fs in _fs
            if fs.get("f_statistic") is not None and np.isfinite(fs["f_statistic"])
        ]
        if _fs_fvals:
            model_info["first_stage_f"] = float(min(_fs_fvals))
        model_info["first_stage"] = [dict(fs) for fs in _fs]

        data_info = {
            "nobs": results["nobs"],
            "df_model": results["df_model"],
            "df_resid": results["df_resid"],
            "dependent_var": self.dependent_var,
            "fitted_values": results["fitted_values"],
            "residuals": results["residuals"],
        }

        # Store the 2SLS structure under a dedicated ``iv`` namespace so the
        # IV-aware wild bootstrap (WRE) can refit the two-stage model without
        # re-parsing the formula. Deliberately NOT under the OLS keys
        # ("X"/"y"/"var_names"): those would make the plain-OLS standalone SE
        # helpers (cr2_se / wild_cluster_boot) operate on the structural design
        # as if it were OLS, which is wrong for IV. The IV path must use the WRE
        # bootstrap, which reads this namespace instead.
        if method in ("2sls", "liml", "fuller"):
            data_info["iv"] = {
                "y": np.asarray(y_fit, dtype=float),
                "X": np.column_stack(
                    [X_exog_fit, X_endog_fit]
                ),  # structural design [exog | endog]
                "W": np.column_stack(
                    [X_exog_fit, Z_fit]
                ),  # full instruments [exog | excluded]
                "exog_names": list(self._exog_names),
                "endog_names": list(self._endog_names),
                "var_names": list(self._exog_names) + list(self._endog_names),
                "n_exog": X_exog_fit.shape[1],
                "n_endog": X_endog_fit.shape[1],
                "kappa": float(results.get("kappa", 1.0)),
            }

        # Build diagnostics dict
        diagnostics = {
            "R-squared": results["r_squared"],
            "N instruments": results["n_instruments"],
            "N endogenous": results["n_endogenous"],
        }

        for j, fs in enumerate(results["first_stage"]):
            endog_name = self._endog_names[j]
            diagnostics[f"First-stage F ({endog_name})"] = fs["f_statistic"]
            diagnostics[f"First-stage F p-value ({endog_name})"] = fs["f_pvalue"]
            diagnostics[f"Partial R² ({endog_name})"] = fs["partial_r_squared"]

        # Weak instrument warning — a typed AssumptionWarning so the fit-time
        # signal carries the same machine-readable recovery payload an agent
        # gets from ``result.violations()`` (recovery_hint / alternatives /
        # diagnostics), instead of a bare string UserWarning. The "Weak
        # instrument" prefix is preserved for callers that grep the message.
        for j, fs in enumerate(results["first_stage"]):
            f_stat = fs["f_statistic"]
            if f_stat is not None and np.isfinite(f_stat) and f_stat < 10:
                endog_name = self._endog_names[j]
                warnings.warn(
                    AssumptionWarning(
                        f"Weak instrument warning: First-stage F-statistic for "
                        f"'{endog_name}' is {f_stat:.2f} (< 10, Stock-Yogo "
                        f"5% bias). 2SLS is biased toward OLS and its t-test "
                        f"over-rejects.",
                        recovery_hint=(
                            "Report sp.anderson_rubin_ci (weak-IV-robust, "
                            "correct coverage at any F) or refit with "
                            "method='liml'/'fuller' (smaller weak-IV bias); "
                            "check sp.effective_f_test for the Olea-Pflueger F."
                        ),
                        diagnostics={
                            "endogenous": endog_name,
                            "first_stage_f": float(f_stat),
                            "threshold": 10.0,
                        },
                        alternative_functions=[
                            "sp.anderson_rubin_ci",
                            "sp.effective_f_test",
                            "sp.iv",
                        ],
                    ),
                    stacklevel=2,
                )

        # ivreg2 convention: report Sargan under i.i.d. errors and Hansen's
        # J once the vcov is robust or clustered (where Sargan is invalid).
        overid = results.get("hansen") or results["sargan"]
        if overid is not None:
            test_name = (
                "Hansen J"
                if (method == "gmm" or results.get("hansen") is not None)
                else "Sargan"
            )
            diagnostics[f"{test_name} statistic"] = overid["statistic"]
            diagnostics[f"{test_name} p-value"] = overid["pvalue"]
            diagnostics[f"{test_name} df"] = overid["df"]

        if results["hausman"] is not None:
            diagnostics["Hausman F-stat"] = results["hausman"]["statistic"]
            diagnostics["Hausman p-value"] = results["hausman"]["pvalue"]

        # Store for programmatic access
        self._first_stage = [dict(fs) for fs in results["first_stage"]]
        self._sargan = dict(overid) if overid is not None else None
        self._hausman = dict(results["hausman"])
        self._instruments = self._instrument_names
        self._raw_results = results

        results_obj = EconometricResults(
            params=params,
            std_errors=std_errors,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )

        self._results = results_obj
        self.is_fitted = True
        return results_obj

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
            raise MethodIncompatibility(
                "Model must be fitted before prediction.",
                recovery_hint="Call fit() before predict().",
                diagnostics={"is_fitted": False},
            )
        if self._results is None:
            raise MethodIncompatibility(
                "Model results are unavailable.",
                recovery_hint="Refit the IV model before prediction.",
                diagnostics={"missing_state": "results"},
            )
        if data is None:
            return _as_float_array(self._results.fitted_values())
        if self.formula is None:
            raise MethodIncompatibility(
                "Out-of-sample prediction requires the model to have been fit "
                "with a formula (not raw y, X arrays).",
                recovery_hint=(
                    "Fit IVRegression with formula=... and data=..., or call "
                    "predict() without new data for in-sample fitted values."
                ),
                diagnostics={"formula": None},
            )
        if not isinstance(data, pd.DataFrame):
            raise MethodIncompatibility(
                "Out-of-sample IV prediction requires a pandas DataFrame.",
                recovery_hint=(
                    "Pass a DataFrame containing the fitted formula's "
                    "exogenous and endogenous variables."
                ),
                diagnostics={"data_type": type(data).__name__},
            )

        parsed = parse_formula(self.formula)
        exog = parsed["exogenous"]
        endog = parsed["endogenous"]
        needed = exog + endog
        missing = [v for v in needed if v not in data.columns]
        if missing:
            raise MethodIncompatibility(
                f"New data is missing columns referenced by the model: {missing}",
                recovery_hint=(
                    "Add the missing formula columns to the prediction data "
                    "or refit the model with the desired specification."
                ),
                diagnostics={"missing_columns": missing},
            )

        params = _as_float_array(self._results.params)
        names = (
            list(self._results.params.index)
            if hasattr(self._results.params, "index")
            else list(self._exog_names) + list(self._endog_names)
        )

        n_new = len(data)
        X_new_cols = []
        for nm in names:
            if nm in {"Intercept", "const"}:
                X_new_cols.append(np.ones(n_new))
            elif nm in data.columns:
                try:
                    X_new_cols.append(data[nm].to_numpy(dtype=float))
                except (TypeError, ValueError) as exc:
                    raise MethodIncompatibility(
                        f"Prediction column '{nm}' must be numeric.",
                        recovery_hint=(
                            "Coerce prediction data columns to numeric values "
                            "before calling predict()."
                        ),
                        diagnostics={"column": nm, "error": str(exc)},
                    ) from exc
            else:
                raise MethodIncompatibility(
                    f"Cannot map parameter '{nm}' to a column in the new data.",
                    recovery_hint=(
                        "Use prediction data compatible with the fitted IV "
                        "formula or refit the model."
                    ),
                    diagnostics={"parameter": nm},
                )
        X_new = np.column_stack(X_new_cols)
        return _as_float_array(X_new @ params)

    @property
    def first_stage(self) -> List[Dict[str, float]]:
        """First-stage diagnostics for each endogenous variable."""
        if not self.is_fitted:
            raise _not_fitted_error("first_stage")
        return self._first_stage

    @property
    def sargan_test(self) -> Optional[Dict[str, float]]:
        """Sargan/Hansen J overidentification test results."""
        if not self.is_fitted:
            raise _not_fitted_error("sargan_test")
        return self._sargan

    @property
    def hausman_test(self) -> Dict[str, float]:
        """Durbin-Wu-Hausman endogeneity test results."""
        if not self.is_fitted:
            raise _not_fitted_error("hausman_test")
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
        return [t.strip() for t in absorb.split("+") if t.strip()]
    return [str(t) for t in absorb]


def _materialise_interacted_fe(
    data: pd.DataFrame, absorb_terms: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Expand ``a^b`` absorb terms into a real interacted-factor column.

    ``fixest`` writes an interacted fixed effect as ``prov^year``; the HDFE
    kernel wants one categorical column per dimension, so the interaction
    is built here rather than left for the caller to precompute. Returns a
    (possibly copied) frame plus the rewritten term list; frames without
    any ``^`` term are passed through untouched.
    """
    if not any("^" in t for t in absorb_terms):
        return data, absorb_terms

    out = data.copy()
    rewritten: List[str] = []
    for term in absorb_terms:
        if "^" not in term:
            rewritten.append(term)
            continue
        parts = [p.strip() for p in term.split("^") if p.strip()]
        missing = [p for p in parts if p not in out.columns]
        if missing:
            raise MethodIncompatibility(
                f"absorb term {term!r} references missing column(s): {missing}",
                recovery_hint=(
                    "Interacted fixed effects use the fixest spelling "
                    '"a^b"; every component must be a column in `data`.'
                ),
                diagnostics={"term": term, "missing_columns": missing},
            )
        name = "__fe_" + "_x_".join(parts)
        codes = out[parts[0]].astype(str)
        for extra in parts[1:]:
            codes = codes + "\x1f" + out[extra].astype(str)
        out[name] = codes
        rewritten.append(name)
    return out, rewritten


def _iv_absorb_preprocess(
    formula: str,
    data: pd.DataFrame,
    absorb_terms: List[str],
    cluster_names: Optional[List[str]] = None,
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

    cluster_names = list(cluster_names or [])

    parsed = parse_formula(formula)
    if not parsed["endogenous"] or not parsed["instruments"]:
        raise MethodIncompatibility(
            "IV formula must specify endogenous variables and instruments. "
            'Use syntax: "y ~ (endog ~ z1 + z2) + exog"',
            recovery_hint="Write the endogenous block as (endog ~ instrument).",
            diagnostics={"formula": formula},
        )

    dependent = parsed["dependent"]
    exog_names = parsed["exogenous"]
    endog_names = parsed["endogenous"]
    instrument_names = parsed["instruments"]

    # ``prov^year`` (the fixest spelling) means the interacted fixed effect.
    # Materialise it as a column so the HDFE kernel sees a single factor.
    data, absorb_terms = _materialise_interacted_fe(data, absorb_terms)

    needed = [dependent] + exog_names + endog_names + instrument_names
    needed += list(absorb_terms)
    needed += list(cluster_names)
    # De-duplicate while preserving order. A column can legitimately appear
    # twice — clustering on an absorbed FE dimension (``absorb="county"`` with
    # ``cluster="county"``) is the single most common panel spec — and
    # ``data[needed]`` would otherwise return duplicated columns, so that
    # ``clean[cluster_name]`` yields a DataFrame instead of a Series.
    seen: set = set()
    needed = [c for c in needed if not (c in seen or seen.add(c))]
    missing = [v for v in needed if v not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"Variables not found in data: {missing}",
            recovery_hint=(
                "Add the missing formula, absorb, or cluster columns to the "
                "DataFrame."
            ),
            diagnostics={"missing_columns": missing},
        )
    missing_absorb = [c for c in absorb_terms if c not in data.columns]
    if missing_absorb:
        raise MethodIncompatibility(
            f"absorb columns not found in data: {missing_absorb}",
            recovery_hint="Pass absorb columns present in the DataFrame.",
            diagnostics={"missing_absorb_columns": missing_absorb},
        )

    clean = data[needed].dropna(subset=needed)
    n_obs = len(clean)
    if n_obs == 0:
        raise DataInsufficient(
            "No rows remain after dropping NaNs.",
            recovery_hint=(
                "Provide complete rows for formula, absorb, and cluster " "columns."
            ),
            diagnostics={"required_columns": needed},
        )

    y = clean[dependent].to_numpy(dtype=np.float64)
    X_exog = (
        clean[exog_names].to_numpy(dtype=np.float64)
        if exog_names
        else np.empty((n_obs, 0), dtype=np.float64)
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
        stacked,
        fe_df,
        drop_singletons=True,
        tol=1e-12,
        max_iter=fe_maxiter,
        tol_abs=fe_tol,
    )

    keep_mask = info.keep_mask
    n_kept = int(info.n_kept)
    n_dropped = int(info.n_dropped)
    fe_card = list(info.n_fe)
    fe_dof = sum(int(g) - 1 for g in fe_card)

    # Slice out columns from the stacked residualised matrix.
    y_dem = stacked_dem[:, 0]
    col = 1
    X_exog_dem = stacked_dem[:, col : col + n_exog]
    col += n_exog
    X_endog_dem = stacked_dem[:, col : col + n_endog]
    col += n_endog
    Z_dem = stacked_dem[:, col : col + n_z]

    # Subset cluster columns to kept rows so downstream ``_cluster_cov``
    # sees aligned data.
    if cluster_names:
        cluster_kept = (
            clean.loc[:, cluster_names].iloc[keep_mask].reset_index(drop=True)
        )
    else:
        cluster_kept = None

    # Do not include an intercept — the absorbed FE block already spans
    # the constant. ``var_names`` mirrors the keys IVRegression uses
    # when invoked via the matrix interface.
    var_names = {
        "dependent": dependent,
        "exog": list(exog_names),
        "endog": list(endog_names),
        "instruments": list(instrument_names),
    }

    return {
        "y": y_dem,
        "X_exog": X_exog_dem,
        "X_endog": X_endog_dem,
        "Z": Z_dem,
        "fe_frame": fe_df.iloc[keep_mask].reset_index(drop=True),
        "cluster_frame": cluster_kept,
        "cluster_names": list(cluster_names),
        "var_names": var_names,
        "n_obs": int(n_obs),
        "n_kept": n_kept,
        "n_dropped": n_dropped,
        "fe_dof": int(fe_dof),
        "fe_cardinality": fe_card,
        "absorb_terms": list(absorb_terms),
    }


def _iv_absorb_run(
    formula: str,
    data: pd.DataFrame,
    absorb_terms: List[str],
    method: str,
    robust: str,
    cluster: Optional[Union[str, List[str]]],
    fuller_alpha: float = 1.0,
    **kwargs: Any,
) -> Tuple[EconometricResults, IVRegression, Dict[str, Any]]:
    """Internal helper: run 2SLS with HDFE absorption.

    Returns ``(result, model, pre)`` where ``pre`` is the dict from
    :func:`_iv_absorb_preprocess`. The dispatcher uses ``model`` to
    attach Kleibergen-Paap / Sanderson-Windmeijer / effective-F
    diagnostics in residualised space.

    All estimators run in residualised space, matching ``ivreghdfe``: the
    FE block is partialled out of ``y``, the controls, the endogenous
    regressors and the instruments first, and the k-class kappa / GMM
    weighting matrix is computed on the residualised data.
    """

    cluster_names = _normalise_cluster(cluster)
    if cluster is not None and cluster_names is None:
        raise MethodIncompatibility(
            "absorb= requires cluster to name column(s) in `data`; an "
            "array-like cluster vector cannot be aligned after singleton "
            "dropping.",
            recovery_hint=(
                "Add the cluster variable as a column and pass its name, "
                'e.g. cluster="county" or cluster=["county", "ym"].'
            ),
            diagnostics={"cluster": type(cluster).__name__},
        )
    pre = _iv_absorb_preprocess(
        formula=formula,
        data=data,
        absorb_terms=absorb_terms,
        cluster_names=list(cluster_names or []),
    )
    cluster_df = pre["cluster_frame"]

    model = IVRegression(
        method=method,
        fuller_alpha=fuller_alpha,
        y=pre["y"],
        X_exog=pre["X_exog"],
        X_endog=pre["X_endog"],
        Z=pre["Z"],
        var_names=pre["var_names"],
    )
    # Inject cluster_df so fit()'s ``cluster_var = self.data[cluster]``
    # branch finds the kept-rows cluster series.
    model.data = cluster_df
    result = model.fit(
        robust=robust,
        cluster=list(cluster_names) if cluster_names else None,
        **kwargs,
    )

    k_total = pre["X_exog"].shape[1] + pre["X_endog"].shape[1]
    # Fixed effects nested within a clustering dimension are redundant for
    # cluster-robust inference and must not be charged against the residual
    # DOF (reghdfe ``dofadjustments(clusters)``, fixest ``fixef.K="nested"``).
    from ..inference._dof import absorbed_dof_charge

    fe_dof_charged, nested_fe = absorbed_dof_charge(
        pre["fe_frame"],
        pre["absorb_terms"],
        pre["fe_cardinality"],
        cluster_df,
    )
    _scale_vcov_for_fe_dof(
        result,
        fe_dof=fe_dof_charged,
        n_kept=pre["n_kept"],
        k=k_total,
    )

    # Context the diagnostics layer needs to re-run column-name-based
    # helpers (Olea-Pflueger effective F) on the same specification: the
    # fitted model only carries residualised matrices plus the cluster
    # frame, so the original data and the variable roles must travel too.
    model._absorb_context = {
        "data": data,
        "absorb": list(absorb_terms),
        "cluster": list(cluster_names or []),
        "endog": list(pre["var_names"]["endog"]),
        "instruments": list(pre["var_names"]["instruments"]),
        "exog": list(pre["var_names"]["exog"]),
    }

    if hasattr(result, "model_info") and isinstance(result.model_info, dict):
        result.model_info["absorb"] = list(absorb_terms)
        result.model_info["fe_cardinality"] = list(pre["fe_cardinality"])
        result.model_info["fe_dof"] = int(pre["fe_dof"])
        result.model_info["fe_dof_charged"] = int(fe_dof_charged)
        result.model_info["fe_nested_in_cluster"] = list(nested_fe)
        result.model_info["n_dropped_singletons"] = int(pre["n_dropped"])

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
    if hasattr(result, "_var_cov") and result._var_cov is not None:
        result._var_cov = result._var_cov * factor
    if hasattr(result, "std_errors") and result.std_errors is not None:
        result.std_errors = result.std_errors * sqrt_factor
    if hasattr(result, "data_info") and isinstance(result.data_info, dict):
        result.data_info["df_resid"] = int(df_resid_new)


def iv(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    method: str = "2sls",
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    fuller_alpha: float = 1.0,
    absorb: Optional[Union[str, List[str]]] = None,
    **kwargs: Any,
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
    cluster : str or list of str, optional
        Variable name(s) for clustered standard errors. Several names —
        ``["county", "ym"]`` or the fixest spelling ``"county + ym"`` —
        select multiway clustering (Cameron, Gelbach & Miller 2011), with
        ``ivreg2``'s single ``G_min`` finite-sample factor and a PSD
        projection.
    fuller_alpha : float, default 1.0
        Fuller modification constant (only used when ``method='fuller'``).
    absorb : str or list of str, optional
        Column name(s) of high-dimensional fixed effects to **partial out**
        before fitting (e.g. ``absorb="firm"`` or
        ``absorb=["firm", "year"]``). Routes ``y``, exogenous controls,
        endogenous regressors, and instruments through
        :func:`sp.fast.demean` (Rust HDFE backend) and drops singletons,
        then runs the requested estimator in residualised space. The
        intercept is dropped because the absorbed FEs span the constant.
        Works for every ``method`` (2SLS / LIML / Fuller / GMM / JIVE),
        matching ``ivreghdfe``. The residual DOF charge follows
        ``reghdfe``: ``sum(G_k - 1)`` over fixed effects *not* nested
        within a clustering dimension, plus one for the absorbed
        constant.
    **kwargs
        Estimator-specific options forwarded to the underlying fitter.
        The one most users reach for is ``gmm_vcov``
        ({'sandwich', 'efficient'}, default ``'sandwich'``) — the variance
        formula for ``method='gmm'``. ``'sandwich'`` re-estimates the moment
        variance at the final residuals and stays valid when the weight
        matrix is not the efficient one; ``'efficient'`` reports the
        textbook efficient-GMM variance ``q (X'W S^-1 W'X)^-1``, which is
        what ``ivreg2 gmm2s`` prints.

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
            raise MethodIncompatibility(
                "absorb= requires (formula, data) — matrix mode is not "
                "supported. Build the formula and pass the DataFrame.",
                recovery_hint=("Use formula/data mode when requesting absorbed IV."),
                diagnostics={
                    "has_formula": formula is not None,
                    "has_data": data is not None,
                },
            )
        _result, _model, _pre = _iv_absorb_run(
            formula=formula,
            data=data,
            absorb_terms=absorb_terms,
            method=method,
            robust=robust,
            cluster=cluster,
            fuller_alpha=fuller_alpha,
            **kwargs,
        )
    else:
        model = IVRegression(
            formula=formula,
            data=data,
            method=method,
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
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("weights", "se_type", "vcov")
                },
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


#: vce sentinels (case-insensitive) selecting the IV wild cluster bootstrap.
_IV_WILD_VCOV = frozenset({"wild", "wildbootstrap", "wild_cluster", "wre", "boottest"})


def ivreg(
    formula: str,
    data: pd.DataFrame,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    *,
    vce: Optional[str] = None,
    vcov: Optional[Any] = None,
    wild_reps: int = 999,
    wild_weight_type: str = "rademacher",
    seed: Optional[int] = None,
    conley_lat: Optional[str] = None,
    conley_lon: Optional[str] = None,
    conley_cutoff: Optional[float] = None,
    **kwargs: Any,
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
    vce : str, optional
        Set ``vce="wild"`` (with ``cluster=``) to run the WRE wild cluster
        bootstrap (Davidson-MacKinnon 2010) on the endogenous coefficient —
        pinned to Stata ``boottest`` after ``ivreg2``. Otherwise ``vce`` is the
        canonical alias for ``robust``.
    wild_reps, wild_weight_type, seed
        Controls for the ``vce="wild"`` path.

    Returns
    -------
    EconometricResults

    References
    ----------
    Angrist, J. D., Imbens, G. W. and Rubin, D. B. (1996). Identification
    of Causal Effects Using Instrumental Variables. *Journal of the
    American Statistical Association*. doi:10.1080/01621459.1996.10476902
    [@angrist1996identification]

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
    # --- vcov= (pyfixest spelling) -> native robust=/cluster=/vce= ---
    # Accepted as a canonical cross-estimator alias; previously it fell
    # through **kwargs and was dropped, silently returning default SEs.
    if vcov is not None:
        from ..core._vcov_spec import normalize_vcov

        _robust, _cluster, _vce = normalize_vcov(
            vcov=vcov,
            robust=robust,
            cluster=cluster,
            vce=vce,
            function="ivreg",
        )
        cluster = _cluster
        if _vce is not None:
            vce = _vce
        else:
            robust = _robust if _robust is not None else "nonrobust"

    # Resolve the canonical `vce` alias; intercept the wild sentinel.
    se_kw = vce if vce is not None else robust
    if isinstance(se_kw, str) and se_kw.lower() in _IV_WILD_VCOV:
        if cluster is None:
            from statspai.exceptions import MethodIncompatibility

            raise MethodIncompatibility(
                "ivreg(vce='wild') requires cluster=... — the wild *cluster* "
                "bootstrap resamples residuals within clusters."
            )
        base = iv(
            formula=formula,
            data=data,
            robust="nonrobust",
            cluster=cluster,
            method="2sls",
            **kwargs,
        )
        from statspai.inference.iv_wild import iv_wild_bootstrap

        endog_name = base.data_info["iv"]["endog_names"][0]
        out = iv_wild_bootstrap(
            base,
            data,
            cluster=cluster,
            variable=endog_name,
            n_boot=wild_reps,
            weight_type=wild_weight_type,
            seed=seed,
        )
        # Attach the WRE inference to the endogenous coefficient; exogenous
        # coefficients retain their 2SLS cluster-robust inference.
        base.std_errors[endog_name] = out["se_cluster"]
        # ``pvalues`` may be a plain ndarray on the IV result — rebuild as a
        # name-indexed Series so the endogenous entry can be overridden while
        # the exogenous 2SLS p-values are preserved.
        old_p = np.asarray(
            getattr(base, "pvalues", np.full(len(base.params), np.nan)), dtype=float
        ).ravel()
        if old_p.shape[0] == len(base.params):
            pvals = pd.Series(old_p, index=base.params.index)
        else:
            pvals = pd.Series(np.nan, index=base.params.index, dtype=float)
        pvals[endog_name] = out["p_boot"]
        base.pvalues = pvals
        base.conf_int_lower = pd.Series(np.nan, index=base.params.index, dtype=float)
        base.conf_int_upper = pd.Series(np.nan, index=base.params.index, dtype=float)
        base.conf_int_lower[endog_name], base.conf_int_upper[endog_name] = out[
            "ci_boot"
        ]
        base.model_info = dict(base.model_info)
        base.model_info["vcov_type"] = (
            "WRE wild cluster bootstrap (Davidson-MacKinnon 2010, "
            f"{wild_reps} reps, {wild_weight_type}; endogenous coefficient)"
        )
        base.model_info["wild_endogenous"] = endog_name
        base.model_info["n_boot"] = wild_reps
        base.model_info["cluster"] = cluster
        return base

    # Bias-reduced cluster SEs: ``vce="CR2"`` (Bell-McCaffrey) / ``vce="CR3"``
    # (jackknife-type), matching R clubSandwich for 2SLS.
    if isinstance(se_kw, str) and se_kw.lower() in ("cr2", "cr3", "jackknife"):
        if cluster is None:
            from statspai.exceptions import MethodIncompatibility

            raise MethodIncompatibility(
                f"ivreg(vce={se_kw!r}) requires cluster=... (a cluster-robust "
                "small-sample correction)."
            )
        kind = "CR3" if se_kw.lower() in ("cr3", "jackknife") else "CR2"
        base = iv(
            formula=formula,
            data=data,
            robust="nonrobust",
            cluster=cluster,
            method="2sls",
            **kwargs,
        )
        from scipy import stats as _stats

        from statspai.inference.iv_wild import iv_cr_vcov

        cr = iv_cr_vcov(base, data, cluster, kind=kind)
        se = cr["std_errors"]
        base.std_errors = se
        z = base.params / se
        base.pvalues = pd.Series(
            2 * (1 - _stats.norm.cdf(np.abs(z))), index=base.params.index
        )
        crit = _stats.norm.ppf(0.975)
        base.conf_int_lower = base.params - crit * se
        base.conf_int_upper = base.params + crit * se
        base.model_info = dict(base.model_info)
        base.model_info[
            "vcov_type"
        ] = f"{kind} cluster-robust (clubSandwich, Pustejovsky-Tipton 2018)"
        base.model_info["cluster"] = cluster
        return base

    # Conley spatial HAC: ``vce="conley"`` with coordinates + a distance cutoff.
    if isinstance(se_kw, str) and se_kw.lower() == "conley":
        if conley_lat is None or conley_lon is None or conley_cutoff is None:
            from statspai.exceptions import MethodIncompatibility

            raise MethodIncompatibility(
                "ivreg(vce='conley') requires conley_lat=, conley_lon=, and "
                "conley_cutoff= (planar distance cutoff in km)."
            )
        base = iv(
            formula=formula,
            data=data,
            robust="nonrobust",
            cluster=None,
            method="2sls",
            **kwargs,
        )
        from scipy import stats as _stats

        from statspai.inference.iv_wild import iv_conley_vcov

        cv = iv_conley_vcov(base, data, conley_lat, conley_lon, conley_cutoff)
        se = cv["std_errors"]
        base.std_errors = se
        z = base.params / se
        base.pvalues = pd.Series(
            2 * (1 - _stats.norm.cdf(np.abs(z))), index=base.params.index
        )
        crit = _stats.norm.ppf(0.975)
        base.conf_int_lower = base.params - crit * se
        base.conf_int_upper = base.params + crit * se
        base.model_info = dict(base.model_info)
        base.model_info[
            "vcov_type"
        ] = f"Conley spatial HAC (uniform, {conley_cutoff} km; acreg-compatible)"
        return base

    # Two-way clustering: ``cluster=["firm", "year"]`` runs 2SLS once (clustered
    # on the first dimension to populate the IV structure) then overrides the
    # variance with the two-way IV sandwich (matches Stata `ivreg2, cluster(a b)
    # small`).
    if isinstance(cluster, (list, tuple)) and len(cluster) == 2:
        c1, c2 = cluster
        base = iv(
            formula=formula,
            data=data,
            robust="nonrobust",
            cluster=c1,
            method="2sls",
            **kwargs,
        )
        from statspai.inference.iv_wild import iv_twoway_vcov

        tw = iv_twoway_vcov(base, data, c1, c2)
        from scipy import stats as _stats

        se = tw["std_errors"]
        base.std_errors = se
        z = base.params / se
        base.pvalues = pd.Series(
            2 * (1 - _stats.norm.cdf(np.abs(z))), index=base.params.index
        )
        crit = _stats.norm.ppf(0.975)
        base.conf_int_lower = base.params - crit * se
        base.conf_int_upper = base.params + crit * se
        base.model_info = dict(base.model_info)
        base.model_info["vcov_type"] = "two-way cluster (CGM 2011)"
        base.model_info["cluster"] = list(cluster)
        base.model_info["n_clusters1"] = tw["n_clusters1"]
        base.model_info["n_clusters2"] = tw["n_clusters2"]
        return base

    kwargs.setdefault("method", "2sls")
    # Reject leftovers rather than forwarding them into iv(**kwargs),
    # where an unrecognised option would vanish and silently return
    # default standard errors.
    from ..core._vcov_spec import reject_unknown_kwargs

    reject_unknown_kwargs(
        kwargs,
        function="ivreg",
        known=("method", "alpha", "weights", "small", "iv_diag"),
    )
    return iv(formula=formula, data=data, robust=se_kw, cluster=cluster, **kwargs)
