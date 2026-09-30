"""Two-stage least squares with absorbed high-dimensional fixed effects.

``sp.hdfe_ols("y ~ exog | fe1 + fe2 | endog ~ inst", ...)`` is Stata's
``ivreghdfe y exog (endog = inst), absorb(fe1 fe2)``: every variable is swept
of the fixed effects once by the same absorber as the OLS path (same
singleton pruning, same absorbed degrees of freedom), then 2SLS runs on the
residuals (Frisch-Waugh-Lovell).

The reported statistics follow ``ivreg2`` as ``ivreghdfe`` calls it (always
with ``small``): with ``sdof`` the absorbed degrees of freedom (the FEs
nested in the cluster excluded, as ``e(sdofminus)``), ``K`` the regressors,
``L`` the instruments (excluded plus included) and ``G`` the clusters,

* 2SLS variance ``(X̂'X̂)^{-1} meat (X̂'X̂)^{-1} q`` with
  ``q = (N-1)/(N-K-sdof) · G/(G-1)`` (cluster), ``N/(N-K-sdof)`` (robust)
  or ``s² = rss/(N-K-sdof)`` (iid); t reference ``G-1`` or ``N-K-sdof``;
* first-stage Wald tests of the excluded instruments on the included ones
  partialled out: ``chi2`` without, ``F`` with the first-stage ``q``
  (``N-L-sdof`` in place of ``N-K-sdof``) -- the Sanderson-Windmeijer /
  Angrist-Pischke F, which for one endogenous regressor is also the
  Kleibergen-Paap rk Wald F (``e(rkf)``);
* Kleibergen-Paap rk LM (``e(idstat)``): the score test of the excluded
  instruments in the first stage, variance under the null; Anderson's
  canonical-correlation LM ``N·R²`` under iid;
* Cragg-Donald Wald F (``e(cdf)``): the homoskedastic first-stage F with
  ``N-L-sdof`` residual df;
* Anderson-Rubin Wald test of ``beta_endog = 0`` (``e(arf)`` / ``e(archi2)``);
* Hansen J at the two-step efficient GMM residuals (Sargan's ``N·R²`` under
  iid) for overidentified models.

The rank statistics (KP, Cragg-Donald) are computed for one endogenous
regressor; with more they are reported as NaN (the per-regressor first-stage
F tests, AR and J are still computed).

Verified against ``ivreghdfe`` 1.1.4 / ``ivreg2`` 4.1.11 in
``tests/reference_parity/test_hdfe_iv_ivreghdfe.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility
from .hdfe import Absorber, _absorbed_total_dof, _cluster_effective_fe_dof


def _meat(A: np.ndarray, e: np.ndarray, groups: Optional[np.ndarray]) -> np.ndarray:
    """``Σ_g (A_g'e_g)(A_g'e_g)'`` (``groups=None``: one row per group)."""
    U = A * e[:, None]
    if groups is None:
        return U.T @ U
    codes = pd.factorize(groups)[0]
    S = np.zeros((codes.max() + 1, A.shape[1]))
    np.add.at(S, codes, U)
    return S.T @ S


def _partial(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if B.shape[1] == 0:
        return A
    return A - B @ np.linalg.lstsq(B, A, rcond=None)[0]


def _wald(b: np.ndarray, V: np.ndarray) -> float:
    return float(b @ np.linalg.solve(V, b))


def hdfe_iv(
    *,
    df: pd.DataFrame,
    lhs: str,
    exog: np.ndarray,
    exog_names: List[str],
    endog: np.ndarray,
    endog_names: List[str],
    inst: np.ndarray,
    inst_names: List[str],
    fe_mat: Optional[np.ndarray],
    slope_specs: list,
    fe_names: List[str],
    cluster_names: List[str],
    vce: Optional[str],
    weights: Optional[np.ndarray],
    alpha: float,
    drop_singletons: bool,
    tol: float,
    maxiter: int,
    formula: str,
    df_inference: Optional[Union[str, float]],
) -> Any:
    """Fit the absorbed 2SLS model; returns a ``FEOLSResult``."""
    from .feols import FEOLSResult, _resolve_df_inference

    if weights is not None:
        raise MethodIncompatibility(
            "hdfe_ols with an IV part does not support weights= yet.",
            recovery_hint="Drop weights=, or use sp.feols (pyfixest) for weighted IV.",
            diagnostics={"weights": True},
        )
    if len(cluster_names) > 1:
        raise MethodIncompatibility(
            "hdfe_ols with an IV part supports one-way clustering only.",
            recovery_hint="Pass a single cluster column.",
            diagnostics={"cluster": cluster_names},
        )
    if vce is not None and vce not in ("robust", "hc1"):
        raise MethodIncompatibility(
            f"hdfe_ols with an IV part supports vce='robust' or cluster=; got {vce!r}.",
            recovery_hint="Use cluster='col', vce='robust', or neither (iid).",
            diagnostics={"vce": vce},
        )
    K1, L1 = endog.shape[1], inst.shape[1]
    if L1 < K1:
        raise MethodIncompatibility(
            f"Underidentified: {L1} excluded instrument(s) for {K1} endogenous "
            "regressor(s).",
            recovery_hint="Add instruments or drop endogenous regressors.",
            diagnostics={"n_endog": K1, "n_instruments": L1},
        )

    y = df[lhs].to_numpy(dtype=np.float64)
    ab = Absorber(
        fe_mat,
        drop_singletons=drop_singletons,
        tol=tol,
        maxiter=maxiter,
        slopes=slope_specs,
        n_obs=len(df),
    )
    k_ex = exog.shape[1]
    stacked = np.column_stack([y, endog, exog, inst])
    M = ab.demean(stacked)
    keep = ab.keep_mask
    yt = M[:, 0]
    D = M[:, 1 : 1 + K1]
    W = M[:, 1 + K1 : 1 + K1 + k_ex]
    Z = M[:, 1 + K1 + k_ex :]
    N = int(ab.n_kept)
    groups = df[cluster_names[0]].to_numpy()[keep] if cluster_names else None

    X = np.column_stack([D, W])
    Zf = np.column_stack([Z, W])
    K, L = X.shape[1], Zf.shape[1]
    if np.linalg.matrix_rank(Zf) < L or np.linalg.matrix_rank(X) < K:
        raise MethodIncompatibility(
            "hdfe_ols IV: regressors or instruments are collinear with each "
            "other or with the absorbed fixed effects.",
            recovery_hint="Drop the redundant variables.",
            diagnostics={"rank_x": int(np.linalg.matrix_rank(X)), "k": K},
        )

    # absorbed dof, as e(sdofminus)
    dof_fe = _absorbed_total_dof(list(ab.n_fe), ab.slope_ops, fe_codes=ab.fe_codes)
    nested: List[bool] = [False] * len(ab.n_fe)
    if groups is not None:
        dof_fe_cl, nested = _cluster_effective_fe_dof(
            ab.fe_codes, list(ab.n_fe), groups, ab.slope_ops
        )
        sdof = dof_fe_cl - 1 if any(nested) else dof_fe_cl
    else:
        sdof = dof_fe
    G = int(pd.Series(groups).nunique()) if groups is not None else None

    # --- 2SLS ------------------------------------------------------------
    ZtZ = Zf.T @ Zf
    Xhat = Zf @ np.linalg.solve(ZtZ, Zf.T @ X)
    beta = np.linalg.solve(Xhat.T @ X, Xhat.T @ yt)
    e = yt - X @ beta
    rss = float(e @ e)
    B = np.linalg.inv(Xhat.T @ Xhat)
    dfr = N - K - sdof
    if groups is not None:
        V = B @ _meat(Xhat, e, groups) @ B * ((N - 1) / dfr * G / (G - 1))
        se_type = "cluster"
    elif vce is not None:
        V = B @ _meat(Xhat, e, None) @ B * (N / dfr)
        se_type = "hc_robust (ivreg2 small)"
    else:
        V = B * (rss / dfr)
        se_type = "iid"
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    names = list(endog_names) + list(exog_names)

    df_default = float(min(G - 1, dfr)) if G is not None else float(dfr)
    df_t = _resolve_df_inference(df_inference, df_default, float(dfr))
    t_crit = stats.t.ppf(1 - alpha / 2, df_t)
    tvals = beta / np.where(se > 0, se, np.nan)
    pvals = 2 * stats.t.sf(np.abs(np.nan_to_num(tvals)), df_t)

    # --- first stage / identification -------------------------------------
    Zt = _partial(Z, W)
    Dt = _partial(D, W)
    ytw = _partial(yt[:, None], W)[:, 0]
    ZtZt_inv = np.linalg.inv(Zt.T @ Zt)
    dfr_fs = N - L - sdof
    if groups is not None:
        q_fs = (N - 1) / dfr_fs * G / (G - 1)
        ref_df2 = G - 1
    else:
        q_fs = N / dfr_fs
        ref_df2 = dfr_fs

    def robust_wald(target: np.ndarray) -> tuple:
        """(chi2 without small, F with small) for Zt-coefficients of target."""
        pi = ZtZt_inv @ (Zt.T @ target)
        u = target - Zt @ pi
        if groups is None and vce is None:
            s2n = float(u @ u) / N
            chi2 = float(pi @ (Zt.T @ Zt) @ pi) / s2n
            F = float(pi @ (Zt.T @ Zt) @ pi) / L1 / (float(u @ u) / dfr_fs)
            return pi, u, chi2, F
        Vn = ZtZt_inv @ _meat(Zt, u, groups) @ ZtZt_inv
        chi2 = _wald(pi, Vn)
        return pi, u, chi2, chi2 / L1 / q_fs

    first_stage: Dict[str, Any] = {}
    for j, nm in enumerate(endog_names):
        pi, u, chi2, F = robust_wald(Dt[:, j])
        first_stage[nm] = {
            "coef": dict(zip(inst_names, pi.tolist())),
            "F": F,
            "F_df": (L1, int(ref_df2)),
            "F_p": float(stats.f.sf(F, L1, ref_df2)),
            "chi2": chi2,
            "partial_r2": float(1 - (u @ u) / (Dt[:, j] @ Dt[:, j])),
        }

    nan = float("nan")
    kp_f = kp_lm = cd_f = nan
    if K1 == 1:
        d1 = Dt[:, 0]
        pi, u, chi2, F = robust_wald(d1)
        cd_f = float(pi @ (Zt.T @ Zt) @ pi) / L1 / (float(u @ u) / dfr_fs)
        if groups is None and vce is None:
            kp_lm = N * first_stage[endog_names[0]]["partial_r2"]  # Anderson LM
        else:
            kp_f = F
            s = Zt.T @ d1
            kp_lm = _wald(s, _meat(Zt, d1, groups))
    lm_df = L1 - K1 + 1

    # Anderson-Rubin: excluded instruments in the reduced form of y
    _, _, ar_chi2, ar_f = robust_wald(ytw)

    # Hansen J / Sargan
    j_stat = j_p = nan
    if L1 > K1:
        if groups is None and vce is None:
            Pe = Zf @ np.linalg.solve(ZtZ, Zf.T @ e)
            j_stat = N * float(e @ Pe) / rss
        else:
            Winv = np.linalg.inv(_meat(Zf, e, groups))
            A = X.T @ Zf @ Winv
            b2 = np.linalg.solve(A @ Zf.T @ X, A @ Zf.T @ yt)
            g2 = Zf.T @ (yt - X @ b2)
            j_stat = float(g2 @ Winv @ g2)
        j_p = float(stats.chi2.sf(j_stat, L1 - K1))

    iv_diag: Dict[str, Any] = {
        "estimator": "2SLS",
        "endogenous": list(endog_names),
        "excluded_instruments": list(inst_names),
        "first_stage": first_stage,
        "kp_rk_wald_F": kp_f,
        "kp_rk_lm": kp_lm,
        "kp_rk_lm_df": lm_df,
        "kp_rk_lm_p": float(stats.chi2.sf(kp_lm, lm_df)) if np.isfinite(kp_lm) else nan,
        "underid_test": (
            "Anderson canon. corr. LM"
            if groups is None and vce is None
            else "Kleibergen-Paap rk LM"
        ),
        "cragg_donald_F": cd_f,
        "anderson_rubin_F": ar_f,
        "anderson_rubin_F_df": (L1, int(ref_df2)),
        "anderson_rubin_F_p": float(stats.f.sf(ar_f, L1, ref_df2)),
        "anderson_rubin_chi2": ar_chi2,
        "anderson_rubin_chi2_p": float(stats.chi2.sf(ar_chi2, L1)),
        "hansen_j": j_stat,
        "hansen_j_df": L1 - K1,
        "hansen_j_p": j_p,
        "overid_test": "Sargan" if groups is None and vce is None else "Hansen J",
        "sdofminus": int(sdof),
    }

    tss_w = float(yt @ yt)  # ivreg2: centred TSS after partialling the FEs
    r2w = 1.0 - rss / tss_w if tss_w > 0 else nan
    y_kept = y[keep]
    tss = float(np.sum((y_kept - y_kept.mean()) ** 2))
    coef_s = pd.Series(beta, index=names, name="coef")
    se_s = pd.Series(se, index=names, name="std_err")
    cluster_info: Dict[str, Any] = {}
    if groups is not None:
        cluster_info = {
            "cluster": cluster_names,
            "n_clusters": [G],
            "dof_fe_cluster": int(sdof),
            "nested_fe": [n for n, is_n in zip(fe_names, nested) if is_n],
        }
    return FEOLSResult(
        params=coef_s,
        std_errors=se_s,
        vcov=V,
        tvalues=pd.Series(tvals, index=names).fillna(0.0),
        pvalues=pd.Series(pvals, index=names),
        conf_int_lower=coef_s - t_crit * se_s,
        conf_int_upper=coef_s + t_crit * se_s,
        residuals=e,
        fitted_within=X @ beta,
        n_obs=N,
        n_singletons_dropped=int(ab.n_dropped),
        n_fe=list(ab.n_fe),
        dof_fe=int(dof_fe),
        df_resid=int(dfr),
        r2_within=r2w,
        se_type=se_type,
        cluster_info=cluster_info,
        formula=formula,
        absorber=ab,
        converged=bool(ab._converged),
        iters=int(ab._iters),
        r2=1.0 - rss / tss if tss > 0 else nan,
        r2_a=nan,
        r2_a_within=1.0 - (1.0 - r2w) * N / dfr,
        rss=rss,
        tss=tss,
        rmse=float(np.sqrt(rss / dfr)),
        df_inference=float(df_t),
        iv_diagnostics=iv_diag,
    )
