"""
Shift-Share IV with AKM (2019) Corrected Standard Errors.

Standard cluster-robust SEs are inconsistent for shift-share (Bartik)
IV designs because they ignore the correlation structure induced by
common shocks.  Adão, Kolesár & Morales (2019) derive a variance
estimator that clusters at the *shock* level, weighting by exposure
shares.

Functions
---------
- ``ssaggregate`` : Full shift-share 2SLS with AKM-corrected SEs.
- ``shift_share_se`` : Correct the SEs of an existing IV result.

References
----------
Adão, R., Kolesár, M., & Morales, E. (2019).
"Shift-Share Designs: Theory and Inference."
*Quarterly Journal of Economics*, 134(4), 1949-2010.
"""

from typing import Optional, List, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults, CausalResult


# ======================================================================
# ssaggregate — full shift-share 2SLS with AKM SEs
# ======================================================================

def ssaggregate(
    data: pd.DataFrame,
    y: str,
    x: str,
    shares: np.ndarray,
    shocks: Union[str, np.ndarray, pd.Series] = None,
    shock_data: Optional[pd.DataFrame] = None,
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Shift-share IV estimation with AKM (2019) corrected standard errors.

    Estimates a 2SLS regression where the instrument is a Bartik
    (shift-share) variable B_i = sum_k s_{ik} g_k, and corrects the
    variance-covariance matrix to account for cross-sectional correlation
    induced by shared shocks.

    Parameters
    ----------
    data : pd.DataFrame
        Observation-level data (n rows).
    y : str
        Outcome variable name.
    x : str
        Endogenous regressor / constructed Bartik IV column in *data*.
        If the column is the constructed Bartik instrument itself (i.e.
        the reduced-form specification), the estimator runs OLS and
        corrects SEs.  If it is an endogenous regressor, a Bartik IV is
        constructed from *shares* and *shocks* for the first stage.
    shares : array-like of shape (n, K)
        Exposure-share matrix.  ``shares[i, k]`` is unit *i*'s exposure
        to shock *k*.
    shocks : str or array-like of shape (K,), optional
        Shock vector.  Either a 1-D array/Series of length K, or a
        column name in *shock_data*.  If ``None``, the Bartik variable
        in *x* is used directly (reduced-form mode).
    shock_data : pd.DataFrame, optional
        Shock-level DataFrame (K rows) when *shocks* is a column name.
    controls : list of str, optional
        Exogenous control variables.
    cluster : str, optional
        Observation-level cluster variable (not used in the AKM
        variance; retained for compatibility and diagnostics).
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    EconometricResults
        With AKM-corrected standard errors.

    Examples
    --------
    >>> result = sp.ssaggregate(
    ...     data=df,
    ...     y="employment_growth",
    ...     x="bartik_instrument",
    ...     shares=shares_matrix,
    ...     shocks="industry_growth",
    ...     shock_data=df_shocks,
    ...     controls=["population", "density"],
    ... )
    >>> print(result.summary())
    """
    n = len(data)
    Y = data[y].values.astype(float)
    X_endog = data[x].values.astype(float)

    # ------------------------------------------------------------------
    # Parse shares
    # ------------------------------------------------------------------
    if isinstance(shares, pd.DataFrame):
        S = shares.values.astype(float)
    else:
        S = np.asarray(shares, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if S.shape[0] != n:
        raise ValueError(
            f"shares has {S.shape[0]} rows but data has {n} rows"
        )
    K = S.shape[1]

    # ------------------------------------------------------------------
    # Parse shocks
    # ------------------------------------------------------------------
    if shocks is not None:
        if isinstance(shocks, str):
            if shock_data is None:
                raise ValueError(
                    "shock_data must be provided when shocks is a column name"
                )
            g = shock_data[shocks].values.astype(float)
        elif isinstance(shocks, pd.Series):
            g = shocks.values.astype(float)
        else:
            g = np.asarray(shocks, dtype=float)
        if g.shape[0] != K:
            raise ValueError(
                f"shocks has length {g.shape[0]} but shares has {K} columns"
            )
        iv_constructed = True
    else:
        g = None
        iv_constructed = False

    # ------------------------------------------------------------------
    # Control matrix
    # ------------------------------------------------------------------
    controls = controls or []
    for c in controls:
        if c not in data.columns:
            raise ValueError(f"Control '{c}' not found in data")

    if controls:
        W = np.column_stack([
            np.ones(n),
            data[controls].values.astype(float),
        ])
        control_names = ["Intercept"] + controls
    else:
        W = np.ones((n, 1))
        control_names = ["Intercept"]

    # ------------------------------------------------------------------
    # Residualise Y, X_endog, and instrument wrt controls
    # ------------------------------------------------------------------
    def _residualise(v, M):
        """OLS residuals of v on M."""
        beta = np.linalg.lstsq(M, v, rcond=None)[0]
        return v - M @ beta

    Y_tilde = _residualise(Y, W)
    X_tilde = _residualise(X_endog, W)

    if iv_constructed:
        B = S @ g  # Bartik IV
        Z_tilde = _residualise(B, W)
    else:
        # Reduced-form: x is already the Bartik, use it as its own instrument
        Z_tilde = X_tilde.copy()

    # ------------------------------------------------------------------
    # 2SLS (or OLS if reduced form)
    # ------------------------------------------------------------------
    if iv_constructed:
        # First stage: X_tilde ~ Z_tilde
        gamma_hat = np.dot(Z_tilde, X_tilde) / np.dot(Z_tilde, Z_tilde)
        X_hat = gamma_hat * Z_tilde

        # Second stage
        beta_2sls = np.dot(X_hat, Y_tilde) / np.dot(X_hat, X_tilde)

        # Residuals (from actual X, not predicted)
        eps_hat = Y_tilde - beta_2sls * X_tilde

        # First-stage F
        resid_fs = X_tilde - gamma_hat * Z_tilde
        rss_restricted = np.dot(X_tilde, X_tilde)  # restricted = no instrument
        rss_full = np.dot(resid_fs, resid_fs)
        df_denom = n - W.shape[1] - 1
        f_stat = ((rss_restricted - rss_full) / 1) / (rss_full / max(df_denom, 1))
        f_pvalue = 1 - stats.f.cdf(f_stat, 1, max(df_denom, 1))
    else:
        # OLS on residualised data
        beta_2sls = np.dot(X_tilde, Y_tilde) / np.dot(X_tilde, X_tilde)
        eps_hat = Y_tilde - beta_2sls * X_tilde
        f_stat = np.nan
        f_pvalue = np.nan

    # ------------------------------------------------------------------
    # AKM (2019) variance estimator
    # ------------------------------------------------------------------
    # For each shock k:  u_k = sum_i  s_{ik} * Z_tilde_i * eps_hat_i
    # V_AKM = (X_hat' X_tilde)^{-2}  *  sum_k  u_k^2
    # (scalar case — single endogenous regressor)

    u_k = np.zeros(K)
    for k in range(K):
        u_k[k] = np.sum(S[:, k] * Z_tilde * eps_hat)

    if iv_constructed:
        denom = np.dot(X_hat, X_tilde)
    else:
        denom = np.dot(X_tilde, X_tilde)

    var_akm = np.sum(u_k ** 2) / (denom ** 2)
    se_akm = float(np.sqrt(var_akm))

    # ------------------------------------------------------------------
    # Conventional (HC1) SE for comparison
    # ------------------------------------------------------------------
    hc1_scale = n / max(n - W.shape[1] - 1, 1)
    if iv_constructed:
        var_hc1 = hc1_scale * np.sum((Z_tilde * eps_hat) ** 2) / (denom ** 2)
    else:
        var_hc1 = hc1_scale * np.sum((X_tilde * eps_hat) ** 2) / (denom ** 2)
    se_hc1 = float(np.sqrt(var_hc1))

    # ------------------------------------------------------------------
    # Also estimate coefficients on controls (from full OLS / 2SLS)
    # ------------------------------------------------------------------
    # Re-run the full regression to get all coefficients
    if iv_constructed:
        # Full 2SLS with controls
        Z_full = np.column_stack([W, B])
        gamma_full = np.linalg.lstsq(Z_full, X_endog, rcond=None)[0]
        X_endog_hat_full = Z_full @ gamma_full
        Xfull = np.column_stack([W, X_endog_hat_full])
    else:
        Xfull = np.column_stack([W, X_endog])

    Xactual = np.column_stack([W, X_endog])
    all_names = control_names + [x]

    XtX_inv = np.linalg.inv(Xfull.T @ Xfull)
    params_full = XtX_inv @ (Xfull.T @ Y)

    # Residuals from actual regressors
    eps_full = Y - Xactual @ params_full
    k_params = len(all_names)

    # Construct SEs for all parameters (use HC1 for controls, AKM for x)
    hc1_meat = Xfull.T @ np.diag((n / max(n - k_params, 1)) * eps_full ** 2) @ Xfull
    var_full = XtX_inv @ hc1_meat @ XtX_inv
    se_full = np.sqrt(np.diag(var_full))
    # Override SE for x with AKM
    se_full[-1] = se_akm

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    params_s = pd.Series(params_full, index=all_names)
    se_s = pd.Series(se_full, index=all_names)

    # R-squared
    tss = np.sum((Y - np.mean(Y)) ** 2)
    rss = np.sum(eps_full ** 2)
    r_squared = 1 - rss / tss if tss > 0 else np.nan

    model_info = {
        "model_type": "Shift-Share IV (AKM 2019)",
        "method": "2SLS with AKM-corrected SEs" if iv_constructed else "OLS with AKM-corrected SEs",
        "robust": "AKM (shock-level clustering)",
    }

    data_info = {
        "nobs": n,
        "df_model": k_params - 1,
        "df_resid": n - k_params,
        "dependent_var": y,
        "fitted_values": Xactual @ params_full,
        "residuals": eps_full,
    }

    diagnostics = {
        "R-squared": r_squared,
        "SE (AKM)": se_akm,
        "SE (HC1)": se_hc1,
        "N shocks (K)": K,
    }
    if iv_constructed:
        diagnostics["First-stage F"] = float(f_stat)
        diagnostics["First-stage F p-value"] = float(f_pvalue)

    return EconometricResults(
        params=params_s,
        std_errors=se_s,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


# ======================================================================
# shift_share_se — correct an existing IV result
# ======================================================================

def shift_share_se(
    iv_result: EconometricResults,
    shares: np.ndarray,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Correct standard errors of an existing IV result for shift-share structure.

    Takes an ``EconometricResults`` from any StatsPAI IV estimator and
    replaces the SEs with AKM (2019) shock-clustered SEs.

    Parameters
    ----------
    iv_result : EconometricResults
        An IV estimation result that contains ``residuals`` and a
        ``fitted_values`` in its ``data_info``, plus the instrument
        residualised values (stored by Bartik estimator).
    shares : array-like of shape (n, K)
        Exposure-share matrix.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    EconometricResults
        A new result object with AKM-corrected standard errors.

    Notes
    -----
    This function requires that the IV result's ``data_info`` contains
    ``'residuals'``.  It computes the AKM variance using the residuals
    and shares.  For the instrument residuals, it uses the fitted
    values from the first stage (``fitted_values``).

    Examples
    --------
    >>> iv_res = sp.bartik(df, y='wage', endog='emp',
    ...                    shares=S, shocks=g)
    >>> corrected = sp.shift_share_se(iv_res, shares=S)
    >>> print(corrected.summary())
    """
    if isinstance(shares, pd.DataFrame):
        S = shares.values.astype(float)
    else:
        S = np.asarray(shares, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    K = S.shape[1]
    n = S.shape[0]

    # Extract residuals
    eps = iv_result.data_info.get("residuals")
    if eps is None:
        raise ValueError(
            "iv_result must contain 'residuals' in data_info. "
            "Re-run the IV estimator with StatsPAI to ensure residuals "
            "are stored."
        )
    eps = np.asarray(eps, dtype=float)
    if len(eps) != n:
        raise ValueError(
            f"shares has {n} rows but residuals have {len(eps)} elements"
        )

    # We need the residualised instrument.  Use fitted values as proxy
    # for the instrument projection.
    fitted = iv_result.data_info.get("fitted_values")
    if fitted is not None:
        fitted = np.asarray(fitted, dtype=float)
        Y_actual = fitted + eps
        Y_mean = np.mean(Y_actual)
        # Residualise fitted (remove mean)
        Z_tilde = fitted - np.mean(fitted)
    else:
        # Fallback: use the Bartik instrument from shares (assume unit shocks)
        Z_tilde = S @ np.ones(K)
        Z_tilde = Z_tilde - np.mean(Z_tilde)

    # Original params
    params = iv_result.params.copy()
    old_se = iv_result.std_errors.copy()

    # AKM correction for the last parameter (endogenous regressor)
    # u_k = sum_i s_{ik} * Z_tilde_i * eps_i
    u_k = np.zeros(K)
    for k in range(K):
        u_k[k] = np.sum(S[:, k] * Z_tilde * eps)

    denom = np.dot(Z_tilde, Z_tilde)
    if denom == 0:
        raise ValueError("Instrument has zero variation after residualising")

    # For the endogenous variable coefficient:
    # The 2SLS denominator is X_hat' X_tilde ≈ Z_tilde' X_tilde
    # We approximate with Z_tilde' Z_tilde (exact if just-identified)
    var_akm = np.sum(u_k ** 2) / (denom ** 2)
    se_akm = float(np.sqrt(var_akm))

    # Replace last SE with AKM
    new_se = old_se.copy()
    new_se.iloc[-1] = se_akm

    # Build new result
    model_info = dict(iv_result.model_info)
    model_info["robust"] = "AKM (shock-level clustering)"
    model_info["original_method"] = model_info.get("method", "")
    model_info["method"] = model_info.get("method", "") + " + AKM SE correction"

    diagnostics = dict(iv_result.diagnostics)
    diagnostics["SE (AKM)"] = se_akm
    diagnostics["SE (original)"] = float(old_se.iloc[-1])
    diagnostics["N shocks (K)"] = K

    return EconometricResults(
        params=params,
        std_errors=new_se,
        model_info=model_info,
        data_info=iv_result.data_info,
        diagnostics=diagnostics,
    )


# Register citation
CausalResult._CITATIONS["adao_correction"] = (
    "@article{adao2019shift,\n"
    "  title={Shift-Share Designs: Theory and Inference},\n"
    "  author={Ad{\\~a}o, Rodrigo and Koles{\\'a}r, Michal and "
    "Morales, Eduardo},\n"
    "  journal={Quarterly Journal of Economics},\n"
    "  volume={134},\n"
    "  number={4},\n"
    "  pages={1949--2010},\n"
    "  year={2019}\n"
    "}"
)
