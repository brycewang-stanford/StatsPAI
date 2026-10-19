"""``sp.feglm(family='negbin')``: the native FE negative binomial, reported
in the layout of the other native ``sp.feglm`` fits (``_native.fit_feglm``).

``fixest::fenegbin`` lists the dispersion ``theta`` as a coefficient
``.theta``. Here the coefficient table holds the regressors only, so that
``predict`` / ``margins`` / ``test`` read it as for any other fit, and
``theta`` with its standard error is in ``model_info['theta']`` /
``model_info['theta_se']`` (and the diagnostics); the covariance matrix
of the regressors and ``theta`` together is ``data_info['var_cov_theta']``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..core.results import EconometricResults
from ._native import Plan, _attach_fe, _inference, _model_info, _psd_two_way


def fit_fenegbin(p: Plan, fml: str) -> EconometricResults:
    """Fit a planned ``sp.feglm(family='negbin')`` call natively."""
    from ..fast._fenegbin import fenegbin_fit

    res, a = fenegbin_fit(
        p.lhs,
        p.rhs,
        p.fe,
        p.data,
        vcov=p.vcov,
        cluster=p.cluster,
        weights=p.weights,
    )
    names = [
        "Intercept" if c == "(Intercept)" else p.renames.get(c, c)
        for c in res.coef_names[:-1]
    ]
    V_full = np.asarray(res.vcov_matrix, dtype=float)
    N, k = int(res.n_kept), len(names)
    G = int(a["n_clusters"]) if p.vcov == "cr1" else 0
    df_k = a["df_k_cluster"] if G else a["df_k"]
    V_full, n_negative = _psd_two_way(p, V_full)
    params = pd.Series(np.asarray(res.coef_vec[:-1], dtype=float), index=names)
    V = V_full[:k, :k]
    se, t, pv = _inference(params, V, p, N - df_k, normal=True)
    theta_se = float(np.sqrt(max(V_full[k, k], 0.0)))
    nan = float("nan")
    diagnostics: Dict[str, Any] = {
        "R-squared": nan,
        "Adj. R-squared": nan,
        "R-squared (within)": nan,
        "Adj. R-squared (within)": nan,
        "RMSE": nan,
        "Deviance": float(res.deviance),
        "Log-likelihood": float(res.log_likelihood),
        "theta": float(res.theta),
        "theta_se": theta_se,
    }
    model_info = _model_info(p, "Negative binomial", "numpy", fml)
    model_info["family"] = "negbin"
    model_info["link"] = "log"
    model_info["theta"] = float(res.theta)
    model_info["theta_se"] = theta_se
    model_info["converged"] = bool(res.converged)
    model_info["iterations"] = int(res.iterations)
    if res.n_dropped_separation:
        model_info["n_dropped_separation"] = int(res.n_dropped_separation)
        warnings.warn(
            f"sp.feglm(family='negbin'): {res.n_dropped_separation} "
            "observations removed because their fixed-effect group has only "
            "zero outcomes or one row (perfect fit), as fixest does.",
            UserWarning,
            stacklevel=4,
        )
    if G:
        model_info["n_clusters"] = G
    if isinstance(p.cluster, list):
        model_info["multiway_negative_eigenvalues"] = n_negative
    y = np.asarray(a["y"], dtype=float)
    mu = np.asarray(a["mu"], dtype=float)
    data_info: Dict[str, Any] = {
        "nobs": N,
        "df_resid": N - max(k, df_k),
        "n_params": k,
        "var_cov": V,
        "var_cov_theta": V_full,
        "residuals": y - mu,
        "fitted_values": mu,
    }
    if G:
        data_info["df_inference"] = G - 1
    out = EconometricResults(
        params=params,
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    out.tvalues = t
    out.pvalues = pv
    _attach_fe(
        out,
        p,
        a["keep_mask"],
        a["eta"],
        list(res.coef_names[:-1]),
        params.to_numpy(),
    )
    return out


__all__ = ["fit_fenegbin"]
