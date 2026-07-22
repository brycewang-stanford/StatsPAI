"""
Adapter to convert pyfixest result objects into StatsPAI EconometricResults.
"""

import warnings
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
    method = (
        "High-Dimensional Fixed Effects" if fe_info and fe_info != "None" else "OLS"
    )

    model_info: Dict[str, Any] = {
        "model_type": model_type,
        "method": method,
        "formula": fml_str,
        "vcov_type": vcov_type,
        "fixed_effects": fe_info,
    }

    # --- cluster count (for few-clusters diagnostic) ---
    # pyfixest exposes ``_G`` as a list of cluster counts (one entry per
    # clustering dimension) when CRV standard errors are used. The smallest
    # entry drives finite-sample cluster-robust inference, so store the min so
    # ``result.violations()`` can flag few-cluster CRV inference just like the
    # native regress/panel/ivreg paths. Wrapped defensively — ``_G`` is a
    # pyfixest internal that can shift by version.
    try:
        if str(vcov_type).upper().startswith("CRV"):
            g_list = getattr(fit, "_G", None)
            if g_list is not None:
                g_vals = [int(g) for g in np.atleast_1d(g_list)]
                if g_vals:
                    model_info["n_clusters"] = min(g_vals)
    except (AttributeError, TypeError, ValueError):  # pragma: no cover
        pass

    data_info: Dict[str, Any] = {
        "nobs": nobs,
        "df_resid": df_resid,
        "n_params": k,
    }

    # Try to attach residuals and fitted values. A failure here silently
    # disables the standalone SE menu (`cr2_se`, `wild_cluster_boot`, ...)
    # downstream, so it must leave a trace (§7).
    try:
        resids = fit.resid()
        if resids is not None:
            data_info["residuals"] = np.asarray(resids)
    except Exception as exc:
        warnings.warn(
            f"pyfixest fit.resid() raised {type(exc).__name__}: {exc}; "
            "residuals are not attached to the result, so standalone "
            "re-inference helpers (cr2_se, wild_cluster_boot, conley) "
            "will not work on it.",
            RuntimeWarning,
            stacklevel=3,
        )

    try:
        fv = fit.predict()
        if fv is not None:
            data_info["fitted_values"] = np.asarray(fv)
    except Exception as exc:
        warnings.warn(
            f"pyfixest fit.predict() raised {type(exc).__name__}: {exc}; "
            "fitted values are not attached to the result.",
            RuntimeWarning,
            stacklevel=3,
        )

    # Attach the within-transformed (FE-absorbed) design + outcome + coef names
    # so the standalone SE menu (`cr2_se`, `wild_cluster_boot`, `twoway_cluster`,
    # `conley`) can operate on the *partialled-out* model instead of re-parsing
    # the formula and refitting plain OLS (which would mishandle the absorbed
    # fixed effects). pyfixest stores the demeaned design as ``_X`` / ``_Y`` and
    # the (FE-excluded) coefficient names as ``_coefnames``; within-OLS on those
    # reproduces the feols coefficients exactly. This is purely additive — it
    # adds ``data_info`` keys and changes no reported feols numerics. Wrapped
    # defensively because these are pyfixest internals that can shift by version.
    try:
        within_x = getattr(fit, "_X", None)
        within_y = getattr(fit, "_Y", None)
        coefnames = getattr(fit, "_coefnames", None)
        if (
            within_x is not None
            and within_y is not None
            and coefnames is not None
            and not getattr(fit, "_X_is_empty", False)
        ):
            within_x = np.asarray(within_x, dtype=float)
            within_y = np.asarray(within_y, dtype=float).ravel()
            names = [str(c) for c in coefnames]
            if (
                within_x.ndim == 2
                and within_x.shape[0] == within_y.shape[0]
                and within_x.shape[1] == len(names)
            ):
                data_info["X"] = within_x
                data_info["y"] = within_y
                data_info["var_names"] = names

                # NOTE: we deliberately do *not* record `fit._data.index` as
                # the estimation sample's row labels. pyfixest does not
                # guarantee it: for a shifted-integer or string index it comes
                # back as a positional RangeIndex, and for a shuffled integer
                # index those positional labels still *look* like a valid
                # subset of the caller's index while pointing at entirely
                # different rows. Aligning per-row data (Conley coordinates,
                # cluster ids) on that would mis-pair observations silently —
                # worse than refusing. Callers that need row alignment must
                # drop incomplete rows before estimating; see
                # inference/jackknife.py::_align_to_fitted_sample.
    except (AttributeError, TypeError, ValueError, IndexError):
        # pyfixest internals (`_X`/`_Y`/`_coefnames`) can shift by version;
        # if their shape/type differs, skip the optional within-design storage
        # (the standalone SE menu then falls back to its formula re-parse).
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
    setattr(result, "_pyfixest_fit", fit)

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
