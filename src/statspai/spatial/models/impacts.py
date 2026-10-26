"""LeSage-Pace (2009) direct / indirect / total impacts for SAR-family models.

For the reduced form  Y = (I - ρW)^{-1} (X β + ε),
the partial derivative matrix w.r.t. regressor k is

    S_k(ρ, β_k) = (I - ρW)^{-1} β_k              (SAR, SAC)
    S_k(ρ, β_k, θ_k) = (I - ρW)^{-1} (β_k I + θ_k W)   (SDM)

Summary measures:
    direct_k   = (1/n) trace(S_k)
    total_k    = (1/n) sum(S_k)
    indirect_k = total_k - direct_k

Standard errors are produced by Monte-Carlo simulation from the
estimator's asymptotic distribution: a multivariate normal centred at the
point estimate with the fitted covariance of the coefficients and the
spatial parameter, as ``spatialreg::impacts`` draws. Through 1.40.2 the
draws used ``diag(se^2)``, which drops the covariance between ``beta``
and ``rho`` that the indirect and total effects depend on.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from ...exceptions import NumericalInstability


def impacts(result: Any, n_sim: int = 1000, seed: Optional[int] = None) -> pd.DataFrame:
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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(7)
    >>> n = 120
    >>> coords = rng.uniform(size=(n, 2))
    >>> w = sp.knn_weights(coords, k=6)
    >>> w.transform = "R"
    >>> W = w.sparse.toarray()
    >>> x = rng.standard_normal(n)
    >>> y = np.linalg.solve(np.eye(n) - 0.5 * W,
    ...                     1 + 2 * x + rng.standard_normal(n))
    >>> res = sp.sar(w, pd.DataFrame({"y": y, "x": x}), "y ~ x")
    >>> imp = sp.impacts(res, n_sim=200, seed=0)
    >>> list(imp.columns)
    ['Direct', 'SE_Direct', 'Indirect', 'SE_Indirect', 'Total', 'SE_Total']
    >>> list(imp.index)
    ['x']
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
        rho_idx = names.index("rho")
        beta_idx = [i for i, nm in enumerate(names) if nm not in {"const", "rho"}]
        theta_idx: list[int] = []
        covariate_names = [names[i] for i in beta_idx]
    elif model_type == "SAC":
        rho_idx = names.index("rho")
        beta_idx = [
            i for i, nm in enumerate(names) if nm not in {"const", "rho", "lambda"}
        ]
        theta_idx = []
        covariate_names = [names[i] for i in beta_idx]
    else:  # SDM
        rho_idx = names.index("rho")
        lag_cols = [i for i, nm in enumerate(names) if nm.startswith("W_")]
        beta_idx = [
            i
            for i, nm in enumerate(names)
            if nm not in {"const", "rho"} and not nm.startswith("W_")
        ]
        theta_idx = lag_cols
        covariate_names = [names[i] for i in beta_idx]

    # Reconstruct W from result (attached by ml.sar/sem/sdm as result... actually not)
    # The legacy result object does not carry W. Instead we require the caller
    # to pass it explicitly via result.data_info["W"] OR we require that the
    # model was fit with n small enough to keep a dense reference. For now,
    # store W on the result from the model code itself (post-fit hook).
    W_matrix = _fetch_W_from_result(result)
    n = W_matrix.shape[0]
    I = np.eye(n)  # noqa: E741 (identity matrix)

    def _point_impacts(
        params_vec: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
                S_k = np.asarray(Sinv @ (b * I + t * W_matrix), dtype=float)
            else:
                S_k = np.asarray(Sinv * b, dtype=float)
            direct[m] = float(np.trace(S_k)) / n
            total[m] = float(S_k.sum()) / n
        return direct, total

    point = _point_impacts(params)
    if point is None:
        raise NumericalInstability(
            "Spatial autoregressive parameter is outside impact bounds.",
            recovery_hint=(
                "Refit the spatial model with a stable spatial parameter "
                "before computing impacts."
            ),
            diagnostics={"rho": float(params[rho_idx]), "model_type": model_type},
        )
    direct_pt, total_pt = point
    indirect_pt = total_pt - direct_pt

    # Monte-Carlo for SEs, from the joint covariance of the estimates.
    rng = np.random.default_rng(seed)
    cov = _joint_covariance(result, se)
    draws = rng.multivariate_normal(params, cov, size=n_sim)
    d_draws: List[np.ndarray] = []
    t_draws: List[np.ndarray] = []
    for d in draws:
        out = _point_impacts(d)
        if out is None:
            continue
        d_draws.append(out[0])
        t_draws.append(out[1])
    d_draws_arr = np.asarray(d_draws, dtype=float)
    t_draws_arr = np.asarray(t_draws, dtype=float)
    i_draws_arr = t_draws_arr - d_draws_arr
    n_dropped = int(n_sim - len(d_draws))
    if n_dropped:
        warnings.warn(
            f"impacts(): {n_dropped} of {n_sim} draws put the spatial "
            "parameter outside (-0.999, 0.999) and were discarded, which "
            "truncates the simulated distribution.",
            UserWarning,
            stacklevel=2,
        )

    out = pd.DataFrame(
        {
            "Direct": direct_pt,
            "SE_Direct": (
                d_draws_arr.std(axis=0, ddof=1) if len(d_draws_arr) > 1 else np.nan
            ),
            "Indirect": indirect_pt,
            "SE_Indirect": (
                i_draws_arr.std(axis=0, ddof=1) if len(i_draws_arr) > 1 else np.nan
            ),
            "Total": total_pt,
            "SE_Total": (
                t_draws_arr.std(axis=0, ddof=1) if len(t_draws_arr) > 1 else np.nan
            ),
        },
        index=covariate_names,
    )
    out.attrs["n_draws_discarded"] = n_dropped
    return out


def _fetch_W_from_result(result: Any) -> np.ndarray:
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
        return np.asarray(W.toarray(), dtype=float)
    return np.asarray(W, dtype=float)


def _joint_covariance(result: Any, se: np.ndarray) -> np.ndarray:
    """Covariance of ``result.params`` for the impact draws.

    The full matrix the estimator stored (``sp.sar`` / ``sp.sdm`` /
    ``sp.sac``; checked against the reported SEs by
    ``coefficient_covariance``). Without one -- the stochastic large-``n``
    path of ``sp.sar`` / ``sp.sdm`` -- the diagonal is the only thing left,
    and a warning says the impact SEs ignore the covariances.
    """
    from ...postestimation._covariance import coefficient_covariance

    V, _ = coefficient_covariance(result)
    if V is not None:
        return np.asarray(V, dtype=float)
    warnings.warn(
        "impacts(): the result stores no joint covariance of the "
        "coefficients and the spatial parameter (large-n fit), so the "
        "simulated SEs use the variances alone and ignore their covariances.",
        UserWarning,
        stacklevel=3,
    )
    return np.diag(se**2)
