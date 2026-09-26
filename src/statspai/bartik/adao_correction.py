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
*Quarterly Journal of Economics*, 134(4), 1949-2010. [@ado2019shift]
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult, EconometricResults
from ..exceptions import MethodIncompatibility, NumericalInstability
from ._akm import _akm_fit, _bhj_aggregate, _resid

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
    Shift-share IV with AKM (2019) inference and the BHJ shock-level view.

    Estimates the location-level regression of *y* on *x* (plus
    *controls* and an intercept). With *shocks* given, *x* is
    instrumented by the shift-share (Bartik) variable
    ``B_i = sum_k s_ik g_k`` (just-identified 2SLS); without *shocks*,
    *x* is taken to be the shift-share variable itself and the model is
    fitted by OLS (the reduced form / ``reg_ss`` case).

    The reported SE of *x* is the Adao-Kolesar-Morales exposure-robust
    SE, computed exactly as R ``ShiftShareSE::ivreg_ss`` / ``reg_ss``
    (and Stata ``ivreg_ss`` / ``reg_ss``): the control-residualised
    shift-share variable is regressed on the shares to recover the
    control-adjusted shocks ``hX_k``, and
    ``SE = sqrt(sum_k (hX_k * s_k' e)^2) / RX``. The diagnostics also
    carry that package's other rows (Homoscedastic, EHW, region-cluster
    when *cluster* is given, and the AKM0 null-imposed CI).

    With *shocks* given, the Borusyak-Hull-Jaravel shock-level
    aggregation (what Stata / R ``ssaggregate`` produce) is returned in
    ``result.data_info["shock_data"]`` -- per shock ``s_n`` (exposure
    weight, sums to one) and the exposure-weighted means of the
    control-residualised *y* and *x* -- together with the shock-level IV
    coefficient and its HC0 SE (``ivreg2 y (x = g) [aw = s_n], robust``)
    in the diagnostics. When the shares do not sum to one, the shock-level
    coefficient equals the location-level one only if the sum of shares is
    among the *controls*; a warning is raised otherwise.

    Parameters
    ----------
    data : pd.DataFrame
        Observation-level data (n rows).
    y : str
        Outcome variable name.
    x : str
        Endogenous regressor (IV mode, *shocks* given) or the shift-share
        variable itself (OLS mode, *shocks* omitted).
    shares : array-like of shape (n, K)
        Exposure-share matrix. ``shares[i, k]`` is unit *i*'s exposure
        to shock *k*; rows need not sum to one.
    shocks : str or array-like of shape (K,), optional
        Shock vector, or a column name in *shock_data*.
    shock_data : pd.DataFrame, optional
        Shock-level DataFrame (K rows) when *shocks* is a column name.
    controls : list of str, optional
        Location-level controls (an intercept is always added).
    cluster : str, optional
        Location-level cluster variable (e.g. state) for the
        ``SE (Reg. cluster)`` diagnostic. The reported SE is AKM.
    alpha : float, default 0.05
        Level of the AKM / AKM0 confidence intervals in the diagnostics.

    Returns
    -------
    EconometricResults
        Coefficients of the location-level regression; the SE of *x* is
        AKM, the controls carry HC1 SEs. p-values / CIs are normal-based
        (as in the reference). Diagnostics: ``SE (AKM)``, ``SE (AKM0)``,
        ``SE (EHW)``, ``SE (Homoscedastic)``, ``SE (HC1)``, AKM / AKM0 CIs,
        first-stage F, and (IV mode) ``beta (BHJ shock-level)`` /
        ``SE (BHJ shock-level, HC0)``.

    Notes
    -----
    Convention differences inside the reference itself, reproduced as is:
    in IV mode the Homoscedastic row uses ``RSS / n`` and EHW has no
    small-sample factor (it equals HC0); in OLS mode the Homoscedastic
    row uses ``RSS / (n - p)`` and EHW multiplies by ``n / (n - p)``.
    ``SE (HC1)`` is the HC1 SE of *x* from the full regression.
    AKM0 reports the CI half-width over ``z_{1 - alpha/2}`` as its SE; if
    the CI is unbounded, ``model_info["akm0_ci_type"]`` says so.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, K = 80, 5
    >>> shares = rng.dirichlet(np.ones(K), size=n)  # exposure shares, rows sum to 1
    >>> g = rng.normal(0.0, 1.0, K)                 # industry-level shocks
    >>> bartik = shares @ g                          # shift-share instrument
    >>> emp = 0.7 * bartik + rng.normal(0, 0.5, n)   # endogenous regressor
    >>> wage = 1.5 * emp + rng.normal(0, 0.5, n)     # outcome
    >>> df = pd.DataFrame({"wage": wage, "emp": emp})
    >>> df_shocks = pd.DataFrame({"industry_growth": g})
    >>> result = sp.ssaggregate(
    ...     data=df,
    ...     y="wage",
    ...     x="emp",
    ...     shares=shares,
    ...     shocks="industry_growth",
    ...     shock_data=df_shocks,
    ... )
    >>> bool(result.params.shape[0] >= 1)
    True
    """
    n = len(data)
    Y = data[y].values.astype(float)
    X_endog = data[x].values.astype(float)

    # ------------------------------------------------------------------
    # Parse shares
    # ------------------------------------------------------------------
    shock_ids = None
    if isinstance(shares, pd.DataFrame):
        shock_ids = shares.columns.to_numpy()
        S = shares.values.astype(float)
    else:
        S = np.asarray(shares, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if S.shape[0] != n:
        raise ValueError(f"shares has {S.shape[0]} rows but data has {n} rows")
    K = S.shape[1]
    if shock_ids is None:
        shock_ids = np.arange(K)

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
    # Control matrix (always with an intercept)
    # ------------------------------------------------------------------
    controls = controls or []
    for c in controls:
        if c not in data.columns:
            raise ValueError(f"Control '{c}' not found in data")
    if controls:
        W = np.column_stack([np.ones(n), data[controls].values.astype(float)])
        control_names = ["Intercept"] + controls
    else:
        W = np.ones((n, 1))
        control_names = ["Intercept"]

    region = None
    if cluster is not None:
        if cluster not in data.columns:
            raise MethodIncompatibility(
                f"cluster column '{cluster}' not found in data",
                recovery_hint="Check the column name passed as cluster=.",
            )
        region = data[cluster].to_numpy()

    # ------------------------------------------------------------------
    # AKM inference (ShiftShareSE::ivreg_ss / reg_ss conventions)
    # ------------------------------------------------------------------
    if iv_constructed:
        B = S @ g  # Bartik instrument
        akm = _akm_fit(Y, B, S, W, y2=X_endog, region_cvar=region, alpha=alpha)
    else:
        # Reduced form: x is itself the shift-share variable -> OLS.
        B = X_endog
        akm = _akm_fit(Y, X_endog, S, W, y2=None, region_cvar=region, alpha=alpha)

    # First-stage F (homoskedastic partial F on the residualised instrument)
    if iv_constructed:
        X_tilde = _resid(X_endog, W)
        Z_tilde = _resid(B, W)
        gamma_hat = np.dot(Z_tilde, X_tilde) / np.dot(Z_tilde, Z_tilde)
        resid_fs = X_tilde - gamma_hat * Z_tilde
        rss_restricted = np.dot(X_tilde, X_tilde)
        rss_full = np.dot(resid_fs, resid_fs)
        df_denom = n - W.shape[1] - 1
        if rss_full <= 1e-12 * max(rss_restricted, 1e-300):
            # Instrument predicts the regressor (almost) perfectly: report
            # an infinite F instead of dividing by a ~zero RSS.
            f_stat = np.inf
            f_pvalue = 0.0
        else:
            f_stat = (rss_restricted - rss_full) / (rss_full / max(df_denom, 1))
            f_pvalue = stats.f.sf(f_stat, 1, max(df_denom, 1))

    # ------------------------------------------------------------------
    # Full 2SLS / OLS for all coefficients (HC1 SEs for the controls)
    # ------------------------------------------------------------------
    if iv_constructed:
        Z_full = np.column_stack([W, B])
        gamma_full = np.linalg.lstsq(Z_full, X_endog, rcond=None)[0]
        Xfull = np.column_stack([W, Z_full @ gamma_full])
    else:
        Xfull = np.column_stack([W, X_endog])
    Xactual = np.column_stack([W, X_endog])
    all_names = control_names + [x]

    XtX_inv = np.linalg.inv(Xfull.T @ Xfull)
    params_full = XtX_inv @ (Xfull.T @ Y)
    eps_full = Y - Xactual @ params_full
    k_params = len(all_names)
    hc1_meat = (Xfull * ((n / max(n - k_params, 1)) * eps_full**2)[:, None]).T @ Xfull
    var_full = XtX_inv @ hc1_meat @ XtX_inv
    se_full = np.sqrt(np.diag(var_full))
    se_hc1 = float(se_full[-1])
    se_full[-1] = akm["se"]["AKM"]

    params_s = pd.Series(params_full, index=all_names)
    se_s = pd.Series(se_full, index=all_names)

    tss = np.sum((Y - np.mean(Y)) ** 2)
    rss = np.sum(eps_full**2)
    r_squared = 1 - rss / tss if tss > 0 else np.nan

    model_info = {
        "model_type": "Shift-Share IV (AKM 2019)",
        "method": (
            "2SLS with AKM-corrected SEs"
            if iv_constructed
            else "OLS with AKM-corrected SEs"
        ),
        "robust": "AKM (shock-level clustering)",
        "akm0_ci_type": akm["akm0_ci_type"],
    }
    data_info = {
        "nobs": n,
        "df_model": k_params - 1,
        "df_resid": n - k_params,
        # AKM / EHW inference is asymptotic-normal (as in ShiftShareSE and
        # Stata reg_ss / ivreg_ss): p-values and CIs use z, not t(n - k).
        "inference": "z",
        "dependent_var": y,
        "fitted_values": Xactual @ params_full,
        "residuals": eps_full,
        "_shift_share_inputs": {
            "y": Y,
            "endog": X_endog if iv_constructed else None,
            "shift_share": B,
            "controls": W,
        },
    }
    diagnostics = _akm_diagnostics(akm, alpha)
    diagnostics["SE (HC1)"] = se_hc1
    diagnostics["R-squared"] = r_squared
    diagnostics["N shocks (K)"] = K
    if iv_constructed:
        diagnostics["First-stage F"] = float(f_stat)
        diagnostics["First-stage F p-value"] = float(f_pvalue)
        bhj = _bhj_aggregate(Y, X_endog, S, g, W, shock_ids, y, x)
        data_info["shock_data"] = bhj["shock_data"]
        diagnostics["beta (BHJ shock-level)"] = bhj["beta"]
        diagnostics["SE (BHJ shock-level, HC0)"] = bhj["se_hc0"]

    return EconometricResults(
        params=params_s,
        std_errors=se_s,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _akm_diagnostics(akm: dict, alpha: float) -> dict:
    """Flatten an ``_akm_fit`` result into scalar diagnostics."""
    out = {}
    for row in ("Homoscedastic", "EHW", "Reg. cluster", "AKM", "AKM0"):
        if np.isnan(akm["se"][row]):
            continue
        out[f"SE ({row})"] = akm["se"][row]
        out[f"p-value ({row})"] = float(akm["p"][row])
    lvl = f"{100 * (1 - alpha):g}%"
    out[f"CI lower (AKM, {lvl})"] = float(akm["ci_l"]["AKM"])
    out[f"CI upper (AKM, {lvl})"] = float(akm["ci_r"]["AKM"])
    out[f"CI lower (AKM0, {lvl})"] = float(akm["ci_l"]["AKM0"])
    out[f"CI upper (AKM0, {lvl})"] = float(akm["ci_r"]["AKM0"])
    return out


# ======================================================================
# shift_share_se — AKM inference for an existing shift-share result
# ======================================================================


def shift_share_se(
    iv_result: EconometricResults,
    shares: np.ndarray,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Replace the SE of a shift-share regression with the AKM (2019) SE.

    Takes the result of :func:`sp.bartik` (2SLS with the shift-share
    instrument) or :func:`sp.ssaggregate` and recomputes the inference on
    the last coefficient (the endogenous regressor, or the shift-share
    variable in a reduced form) with the Adao-Kolesar-Morales
    exposure-robust formula, exactly as R ``ShiftShareSE::ivreg_ss`` /
    ``reg_ss`` and Stata ``ivreg_ss`` / ``reg_ss`` compute it.

    Parameters
    ----------
    iv_result : EconometricResults
        Result of ``sp.bartik`` or ``sp.ssaggregate``. These record the
        outcome, endogenous regressor, shift-share instrument and controls
        the AKM formula needs; results from other estimators do not, and
        are rejected.
    shares : array-like of shape (n, K)
        Exposure-share matrix (need not sum to one).
    alpha : float, default 0.05
        Level for the AKM / AKM0 confidence intervals.

    Returns
    -------
    EconometricResults
        Same point estimates; the last SE is the AKM SE and the
        diagnostics carry the Homoscedastic / EHW / AKM / AKM0 rows.
        p-values and CIs are normal-based, as in the reference.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, K = 80, 5
    >>> S = pd.DataFrame(rng.dirichlet(np.ones(K), size=n),
    ...                  columns=[f"ind{k}" for k in range(K)])
    >>> g = pd.Series(rng.normal(0, 1, K), index=S.columns)
    >>> bartik = S.values @ g.values
    >>> emp = 0.7 * bartik + rng.normal(0, 0.5, n)
    >>> wage = 1.5 * emp + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({"wage": wage, "emp": emp})
    >>> iv_res = sp.bartik(df, y='wage', endog='emp', shares=S, shocks=g,
    ...                    leave_one_out=False)
    >>> corrected = sp.shift_share_se(iv_res, shares=S.values)
    >>> bool("SE (AKM)" in corrected.diagnostics)
    True
    """
    if isinstance(shares, pd.DataFrame):
        S = shares.values.astype(float)
    else:
        S = np.asarray(shares, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)

    inputs = iv_result.data_info.get("_shift_share_inputs")
    if inputs is None:
        raise MethodIncompatibility(
            "shift_share_se needs the outcome, endogenous regressor, "
            "shift-share instrument and controls of the fit; only results "
            "of sp.bartik or sp.ssaggregate record them. Fit the model with "
            "sp.bartik(...) or sp.ssaggregate(...) instead.",
            recovery_hint="Refit with sp.bartik(...) or sp.ssaggregate(...).",
        )
    n = len(inputs["y"])
    if S.shape[0] != n:
        raise MethodIncompatibility(
            f"shares has {S.shape[0]} rows but the fit has {n}",
            recovery_hint="Pass one shares row per observation.",
        )

    akm = _akm_fit(
        inputs["y"],
        inputs["shift_share"],
        S,
        inputs["controls"],
        y2=inputs["endog"],
        alpha=alpha,
    )
    params = iv_result.params.copy()
    if not np.isclose(akm["beta"], float(params.iloc[-1]), rtol=1e-8, atol=1e-12):
        raise NumericalInstability(
            "The recorded shift-share inputs do not reproduce the fitted "
            f"coefficient ({akm['beta']!r} vs {float(params.iloc[-1])!r})."
        )
    old_se = iv_result.std_errors.copy()
    new_se = old_se.copy()
    new_se.iloc[-1] = akm["se"]["AKM"]

    model_info = dict(iv_result.model_info)
    model_info["robust"] = "AKM (shock-level clustering)"
    model_info["original_method"] = model_info.get("method", "")
    model_info["method"] = model_info.get("method", "") + " + AKM SE correction"
    model_info["akm0_ci_type"] = akm["akm0_ci_type"]

    data_info = dict(iv_result.data_info)
    data_info["inference"] = "z"
    diagnostics = dict(iv_result.diagnostics)
    diagnostics.update(_akm_diagnostics(akm, alpha))
    diagnostics["SE (original)"] = float(old_se.iloc[-1])
    diagnostics["N shocks (K)"] = S.shape[1]

    model_info["alpha"] = alpha
    return EconometricResults(
        params=params,
        std_errors=new_se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


# Register citation
CausalResult._CITATIONS["adao_correction"] = (
    "@article{ado2019shift,\n"
    "  title={Shift-Share Designs: Theory and Inference},\n"
    "  author={Ad{\\~a}o, Rodrigo and Koles{\\'a}r, Michal and "
    "Morales, Eduardo},\n"
    "  journal={Quarterly Journal of Economics},\n"
    "  volume={134},\n"
    "  number={4},\n"
    "  pages={1949--2010},\n"
    "  year={2019},\n"
    "  doi={10.1093/qje/qjz025}\n"
    "}"
)
