"""LeSage-Pace (2009) direct / indirect / total impacts for spatial models.

For the reduced form  Y = (I - ρW)^{-1} (X β + WX θ + ε),
the partial derivative matrix w.r.t. regressor k is

    S_k(ρ, β_k) = (I - ρW)^{-1} β_k              (SAR, SAC)
    S_k(ρ, β_k, θ_k) = (I - ρW)^{-1} (β_k I + θ_k W)   (SDM, GNS)
    S_k(β_k, θ_k) = β_k I + θ_k W                 (SLX, SDEM)

Summary measures:
    direct_k   = (1/n) trace(S_k)
    total_k    = (1/n) sum(S_k)
    indirect_k = total_k - direct_k

For the models with a spatial lag of ``y`` the standard errors are
produced by Monte-Carlo simulation from the estimator's asymptotic
distribution: a multivariate normal centred at the point estimate with the
fitted covariance of the coefficients and the spatial parameter, as
``spatialreg::impacts`` draws. Through 1.40.2 the draws used
``diag(se^2)``, which drops the covariance between ``beta`` and ``rho``
that the indirect and total effects depend on.

Without a spatial lag of ``y`` (SLX, SDEM) and with ``W`` that has a zero
diagonal, the measures are linear in the estimates (direct ``beta_k``,
indirect ``theta_k`` times the mean row sum of ``W``), so their standard
errors follow exactly from the coefficient covariance and no simulation is
needed (``spatialreg``'s ``impacts`` for ``lmSLX`` and
``errorsarlm(Durbin = TRUE)`` fits).
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from ...exceptions import MethodIncompatibility, NumericalInstability

_LAG_MODELS = {"SAR", "SAC", "SDM", "GNS"}
_WX_MODELS = {"SLX", "SDEM"}
_DURBIN_MODELS = {"SDM", "GNS", "SLX", "SDEM"}


def impacts(result: Any, n_sim: int = 1000, seed: Optional[int] = None) -> pd.DataFrame:
    """Direct, indirect and total impacts of a spatial regression, with SEs.

    Parameters
    ----------
    result : EconometricResults
        Output of ``sp.sar``, ``sp.sdm``, ``sp.sac``, ``sp.gns``
        (simulated SEs) or ``sp.slx``, ``sp.sdem`` (exact SEs).
    n_sim : int, default 1000
        Number of Monte-Carlo draws for simulated SEs. Not used for SLX /
        SDEM.
    seed : int, optional
        RNG seed. Not used for SLX / SDEM.

    Returns
    -------
    DataFrame with one row per non-constant regressor and columns
    ``Direct``, ``SE_Direct``, ``Indirect``, ``SE_Indirect``, ``Total``,
    ``SE_Total``. ``attrs['se_method']`` is ``'simulation'`` or
    ``'analytic'``; ``attrs['n_draws_discarded']`` counts the draws whose
    spatial parameter fell outside the admissible interval
    ``(1 / omega_min, 1 / omega_max)`` of the eigenvalues of ``W`` (real
    parts), which are dropped as ``spatialreg`` drops them.

    Notes
    -----
    The total impact is ``(1/n) 1' S_k 1`` (LeSage and Pace 2009). For a
    row-standardised ``W`` it equals ``beta_k / (1 - rho)`` in a SAR
    model; for any other ``W`` it does not. ``spatialreg::impacts`` on a
    ``lagsarlm`` fit with weights that are not row-standardised reports
    ``beta_k / (1 - rho)`` all the same (spatialreg 1.4.3, checked on a
    binary ring), so its indirect and total impacts differ from these
    there; its direct impacts agree.

    Through 1.40.2 the admissible interval was fixed at (-0.999, 0.999),
    right only for a row-standardised ``W`` with smallest eigenvalue -1:
    draws past ``1 / omega_max`` of an unstandardised ``W`` were kept and
    inflated the SEs, and valid draws below -0.999 were dropped.

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
    if model_type not in _LAG_MODELS | _WX_MODELS:
        raise MethodIncompatibility(
            "impacts() is defined for SAR / SDM / SAC / GNS / SLX / SDEM "
            f"models; got {model_type!r}.",
            recovery_hint=(
                "A spatial error model (SEM) has no spillovers in the mean: "
                "its coefficients are the marginal effects. Fit sp.sdem or "
                "sp.sdm for spillovers through WX."
            ),
            diagnostics={"model_type": model_type},
        )

    names = list(result.params.index)
    params = np.asarray(result.params.values, dtype=float)
    se = np.asarray(result.std_errors.values, dtype=float)

    # Positions: [const, x_1..x_p, (W_x_1..W_x_p), (rho), (lambda)]. Read by
    # position, not by a "W_" prefix, so a regressor whose own name starts
    # with "W_" is not taken for a spatial lag.
    n_spatial = int("rho" in names) + int("lambda" in names)
    n_reg = len(names) - 1 - n_spatial
    if model_type in _DURBIN_MODELS:
        p = n_reg // 2
        beta_idx = list(range(1, 1 + p))
        theta_idx = list(range(1 + p, 1 + 2 * p))
    else:
        beta_idx = list(range(1, 1 + n_reg))
        theta_idx = []
    covariate_names = [names[i] for i in beta_idx]

    W_matrix = _fetch_W_from_result(result)
    n = W_matrix.shape[0]

    if model_type in _WX_MODELS:
        return _wx_impacts(
            result, params, se, beta_idx, theta_idx, covariate_names, W_matrix
        )

    rho_idx = names.index("rho")
    I = np.eye(n)  # noqa: E741 (identity matrix)
    lo, hi = _admissible_interval(W_matrix)

    def _point_impacts(
        params_vec: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        r = float(params_vec[rho_idx])
        if not (lo < r < hi):
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
            "The spatial autoregressive parameter is outside the interval "
            f"({lo:.6g}, {hi:.6g}) on which I - rho W is invertible.",
            recovery_hint=(
                "Refit the spatial model with a stable spatial parameter "
                "before computing impacts."
            ),
            diagnostics={
                "rho": float(params[rho_idx]),
                "interval": (lo, hi),
                "model_type": model_type,
            },
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
            f"parameter outside ({lo:.6g}, {hi:.6g}), where I - rho W is "
            "singular or its inverse changes sign, and were discarded, "
            "which truncates the simulated distribution.",
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
    out.attrs["se_method"] = "simulation"
    return out


def _admissible_interval(W_matrix: np.ndarray) -> Tuple[float, float]:
    """``(1 / omega_min, 1 / omega_max)`` over the real parts of the
    eigenvalues of ``W`` (open interval). An end with no eigenvalue of that
    sign is infinite."""
    ev = np.real(np.linalg.eigvals(W_matrix))
    lo = 1.0 / float(ev.min()) if ev.min() < 0 else -np.inf
    hi = 1.0 / float(ev.max()) if ev.max() > 0 else np.inf
    return lo, hi


def _wx_impacts(
    result: Any,
    params: np.ndarray,
    se: np.ndarray,
    beta_idx: List[int],
    theta_idx: List[int],
    covariate_names: List[str],
    W_matrix: np.ndarray,
) -> pd.DataFrame:
    """Impacts of SLX / SDEM: ``S_k = beta_k I + theta_k W``.

    Direct ``= beta_k + theta_k tr(W) / n`` and total ``= beta_k + theta_k
    (1' W 1) / n``; both are linear in ``(beta_k, theta_k)``, so the
    standard errors come from the coefficient covariance with no
    simulation. With a zero diagonal and row sums one these are
    ``beta_k``, ``theta_k`` and ``beta_k + theta_k``.
    """
    n = W_matrix.shape[0]
    a_dir = float(np.trace(W_matrix)) / n
    a_tot = float(W_matrix.sum()) / n
    V, _ = _require_joint_covariance(result)
    rows = []
    for m, i in enumerate(beta_idx):
        j = theta_idx[m]
        g_dir = np.zeros(len(params))
        g_tot = np.zeros(len(params))
        g_dir[i], g_dir[j] = 1.0, a_dir
        g_tot[i], g_tot[j] = 1.0, a_tot
        g_ind = g_tot - g_dir
        rows.append(
            {
                "Direct": float(g_dir @ params),
                "SE_Direct": float(np.sqrt(max(g_dir @ V @ g_dir, 0.0))),
                "Indirect": float(g_ind @ params),
                "SE_Indirect": float(np.sqrt(max(g_ind @ V @ g_ind, 0.0))),
                "Total": float(g_tot @ params),
                "SE_Total": float(np.sqrt(max(g_tot @ V @ g_tot, 0.0))),
            }
        )
    out = pd.DataFrame(rows, index=covariate_names)
    out.attrs["n_draws_discarded"] = 0
    out.attrs["se_method"] = "analytic"
    return out


def _require_joint_covariance(result: Any) -> Tuple[np.ndarray, np.ndarray]:
    from ...postestimation._covariance import coefficient_covariance

    V, se = coefficient_covariance(result)
    if V is None:
        raise MethodIncompatibility(
            "impacts(): the SLX / SDEM result stores no coefficient "
            "covariance matching its standard errors, and the indirect and "
            "total SEs need the covariance of beta and theta.",
            recovery_hint="Refit with the current sp.slx / sp.sdem, which "
            "store it in model_info['vcov'].",
        )
    return np.asarray(V, dtype=float), se


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
