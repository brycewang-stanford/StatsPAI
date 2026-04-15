"""LeSage-Pace (2009) direct / indirect / total impacts for SAR-family models.

For the reduced form  Y = (I - ρW)^{-1} (X β + ε),
the partial derivative matrix w.r.t. regressor k is

    S_k(ρ, β_k) = (I - ρW)^{-1} β_k              (SAR, SAC)
    S_k(ρ, β_k, θ_k) = (I - ρW)^{-1} (β_k I + θ_k W)   (SDM)

Summary measures:
    direct_k   = (1/n) trace(S_k)
    total_k    = (1/n) sum(S_k)
    indirect_k = total_k - direct_k

Confidence intervals are produced by Monte-Carlo simulation from the
estimator's asymptotic distribution (multivariate normal centred at the
point estimate with covariance diag(se²)).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse


def impacts(result, n_sim: int = 1000, seed: Optional[int] = None) -> pd.DataFrame:
    """Compute direct / indirect / total impacts + simulated SEs.

    Parameters
    ----------
    result : EconometricResults
        Output of ``sp.sar``, ``sp.sdm``, or ``sp.sac``.
    n_sim : int, default 1000
        Number of Monte-Carlo draws for simulated SEs.
    seed : int, optional
        RNG seed.

    Returns
    -------
    DataFrame with one row per non-constant regressor and columns
    ``Direct``, ``SE_Direct``, ``Indirect``, ``SE_Indirect``, ``Total``,
    ``SE_Total``.
    """
    model_type = result.model_info.get("model_type", "").split(" ")[0].upper()
    if model_type not in {"SAR", "SAC", "SDM"}:
        raise ValueError(
            f"impacts() only defined for SAR / SDM / SAC models; got {model_type!r}"
        )

    names = list(result.params.index)
    params = np.asarray(result.params.values, dtype=float)
    se = np.asarray(result.std_errors.values, dtype=float)

    # Locate spatial parameter and β slice (and θ slice for SDM)
    if model_type == "SAR":
        rho = float(result.model_info["spatial_param_value"])
        rho_idx = names.index("rho")
        beta_idx = [i for i, nm in enumerate(names) if nm not in {"const", "rho"}]
        theta_idx: list[int] = []
        covariate_names = [names[i] for i in beta_idx]
    elif model_type == "SAC":
        rho = float(result.model_info["spatial_param_value"])
        rho_idx = names.index("rho")
        beta_idx = [i for i, nm in enumerate(names)
                    if nm not in {"const", "rho", "lambda"}]
        theta_idx = []
        covariate_names = [names[i] for i in beta_idx]
    else:    # SDM
        rho = float(result.model_info["spatial_param_value"])
        rho_idx = names.index("rho")
        lag_cols = [i for i, nm in enumerate(names) if nm.startswith("W_")]
        beta_idx = [i for i, nm in enumerate(names)
                    if nm not in {"const", "rho"} and not nm.startswith("W_")]
        theta_idx = lag_cols
        covariate_names = [names[i] for i in beta_idx]

    # Reconstruct W from result (attached by ml.sar/sem/sdm as result... actually not)
    # The legacy result object does not carry W. Instead we require the caller
    # to pass it explicitly via result.data_info["W"] OR we require that the
    # model was fit with n small enough to keep a dense reference. For now,
    # store W on the result from the model code itself (post-fit hook).
    W_matrix = _fetch_W_from_result(result)
    n = W_matrix.shape[0]
    I = np.eye(n)
    S_inv = np.linalg.inv(I - rho * W_matrix)

    def _point_impacts(params_vec):
        r = float(params_vec[rho_idx])
        if not (-0.999 < r < 0.999):
            return None
        Sinv = np.linalg.inv(I - r * W_matrix)
        direct = np.empty(len(beta_idx))
        total = np.empty(len(beta_idx))
        for m, i in enumerate(beta_idx):
            b = float(params_vec[i])
            if theta_idx:
                t = float(params_vec[theta_idx[m]])
                S_k = Sinv @ (b * I + t * W_matrix)
            else:
                S_k = Sinv * b
            direct[m] = float(np.trace(S_k)) / n
            total[m] = float(S_k.sum()) / n
        return direct, total

    direct_pt, total_pt = _point_impacts(params)
    indirect_pt = total_pt - direct_pt

    # Monte-Carlo for SEs
    rng = np.random.default_rng(seed)
    cov = np.diag(se ** 2)                     # diagonal approximation
    draws = rng.multivariate_normal(params, cov, size=n_sim)
    d_draws, t_draws = [], []
    for d in draws:
        out = _point_impacts(d)
        if out is None:
            continue
        d_draws.append(out[0])
        t_draws.append(out[1])
    d_draws = np.asarray(d_draws)
    t_draws = np.asarray(t_draws)
    i_draws = t_draws - d_draws

    return pd.DataFrame(
        {
            "Direct":   direct_pt,
            "SE_Direct":   d_draws.std(axis=0, ddof=1) if len(d_draws) > 1 else np.nan,
            "Indirect": indirect_pt,
            "SE_Indirect": i_draws.std(axis=0, ddof=1) if len(i_draws) > 1 else np.nan,
            "Total":    total_pt,
            "SE_Total":    t_draws.std(axis=0, ddof=1) if len(t_draws) > 1 else np.nan,
        },
        index=covariate_names,
    )


def _fetch_W_from_result(result) -> np.ndarray:
    """Recover the W matrix from a spatial result's data_info payload.

    Requires that the estimator attached W. New ml.sar/sem/sdm do this via
    data_info["W_sparse"] (CSR). Older calls that lose W will need to pass
    it in another way (future work).
    """
    W = result.data_info.get("W_sparse") if hasattr(result, "data_info") else None
    if W is None:
        raise ValueError(
            "No spatial weights found on the result. Re-fit the model with the "
            "current StatsPAI version — it attaches W to data_info automatically."
        )
    if sparse.issparse(W):
        return W.toarray()
    return np.asarray(W, dtype=float)
