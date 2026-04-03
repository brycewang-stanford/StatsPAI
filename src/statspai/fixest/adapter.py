"""
Adapter to convert pyfixest result objects into StatsPAI EconometricResults.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..core.results import EconometricResults


def _pyfixest_to_econometric_results(
    fit: Any,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
) -> EconometricResults:
    """
    Convert a single pyfixest Feols/Fepois/Feglm fit into an EconometricResults.

    Parameters
    ----------
    fit : pyfixest model object
        A fitted pyfixest model (e.g. from ``feols``, ``fepois``).
    vcov : str or dict, optional
        If provided, update the variance-covariance estimator before
        extracting results (e.g. ``"HC1"``, ``{"CRV1": "firm"}``).

    Returns
    -------
    EconometricResults
    """
    if vcov is not None:
        fit.vcov(vcov)

    # --- coefficients & standard errors ---
    params = fit.coef()
    std_errors = fit.se()

    # Ensure pd.Series with matching index
    if not isinstance(params, pd.Series):
        params = pd.Series(params)
    if not isinstance(std_errors, pd.Series):
        std_errors = pd.Series(std_errors, index=params.index)

    # --- t-stats & p-values ---
    tstat = fit.tstat() if hasattr(fit, "tstat") else params / std_errors
    pval = fit.pvalue() if hasattr(fit, "pvalue") else None
    if not isinstance(tstat, pd.Series):
        tstat = pd.Series(tstat, index=params.index)
    if pval is not None and not isinstance(pval, pd.Series):
        pval = pd.Series(pval, index=params.index)

    # --- nobs / degrees of freedom ---
    nobs = int(fit._N) if hasattr(fit, "_N") else len(params)
    k = len(params)
    df_resid = nobs - k

    # --- R-squared ---
    diagnostics: Dict[str, Any] = {}
    if hasattr(fit, "_r2"):
        diagnostics["R-squared"] = fit._r2
    if hasattr(fit, "_r2_within"):
        diagnostics["R-squared (within)"] = fit._r2_within
    if hasattr(fit, "_rmse"):
        diagnostics["RMSE"] = fit._rmse

    # --- fixed effects info ---
    fe_info = ""
    if hasattr(fit, "_fixef"):
        fe_info = str(fit._fixef) if fit._fixef else "None"

    # --- vcov type ---
    vcov_type = ""
    if hasattr(fit, "_vcov_type"):
        vcov_type = str(fit._vcov_type)

    # --- formula ---
    fml_str = str(fit._fml) if hasattr(fit, "_fml") else ""

    # --- model type ---
    model_type_map = {
        "Feols": "OLS (pyfixest)",
        "Fepois": "Poisson (pyfixest)",
        "Feglm": "GLM (pyfixest)",
    }
    class_name = type(fit).__name__
    model_type = model_type_map.get(class_name, f"{class_name} (pyfixest)")

    # --- estimation method label ---
    method = "High-Dimensional Fixed Effects" if fe_info and fe_info != "None" else "OLS"

    model_info: Dict[str, Any] = {
        "model_type": model_type,
        "method": method,
        "formula": fml_str,
        "vcov_type": vcov_type,
        "fixed_effects": fe_info,
    }

    data_info: Dict[str, Any] = {
        "nobs": nobs,
        "df_resid": df_resid,
        "n_params": k,
    }

    # Try to attach residuals and fitted values
    try:
        resids = fit.resid()
        if resids is not None:
            data_info["residuals"] = np.asarray(resids)
    except Exception:
        pass

    try:
        fv = fit.predict()
        if fv is not None:
            data_info["fitted_values"] = np.asarray(fv)
    except Exception:
        pass

    # Store the original pyfixest object for advanced users
    result = EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )

    # Attach pyfixest-native t-stats and p-values (more accurate)
    if pval is not None:
        result.tvalues = tstat
        result.pvalues = pval

    # Keep reference to original fit for power users
    result._pyfixest_fit = fit

    return result


def _multi_fit_to_results(
    multi_fit: Any,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
) -> List[EconometricResults]:
    """
    Convert a pyfixest FixestMulti object into a list of EconometricResults.

    Parameters
    ----------
    multi_fit : pyfixest FixestMulti
        Result from multiple estimation (e.g. ``feols("Y ~ X | csw0(f1, f2)", ...)``).
    vcov : str or dict, optional
        Override the variance-covariance estimator for all models.

    Returns
    -------
    list of EconometricResults
    """
    results = []

    if vcov is not None:
        multi_fit.vcov(vcov)

    # FixestMulti: iterate by integer index
    all_models = multi_fit.all_fitted_models
    if callable(all_models):
        all_models = all_models()

    n_models = len(all_models)
    for i in range(n_models):
        fit = multi_fit.fetch_model(i, print_fml=False)
        results.append(_pyfixest_to_econometric_results(fit))

    return results
