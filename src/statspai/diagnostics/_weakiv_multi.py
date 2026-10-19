"""Joint and robust weak-IV tests behind ``sp.anderson_rubin_test`` and
``sp.weakrobust`` when there are several endogenous regressors, or the
weak-IV-robust tests should be heteroskedasticity- or cluster-robust.

* :func:`ar_joint` -- the Anderson-Rubin F test of ``beta = h0`` for a
  vector of endogenous coefficients, in the same three forms as the
  one-regressor ``sp.anderson_rubin_test`` (homoskedastic F, EHW Wald,
  cluster Wald). At ``h0 = 0`` it is ``ivreg2``'s ``e(arf)``.
* :func:`weakiv_panel` -- AR, K, J, K-J, CLR and the conditioning rank
  statistic in Stata ``weakiv``'s LM form (``_weakiv_lm.WeakIVLM``), with
  confidence sets for one endogenous regressor and, with several, subset
  and projection AR sets for single coefficients.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import IdentificationFailure, MethodIncompatibility
from ._weakiv_lm import SubsetAR, WeakIVLM, invert_pvalue, invert_pvalues

_TEST_VCOV = {
    "classic": "iid",
    "iid": "iid",
    "nonrobust": "iid",
    "robust": "robust",
    "hc0": "robust",
    "cluster": "cluster",
}


def _null_vector(h0: Any, p: int, names: Sequence[str]) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(h0, dtype=float)).reshape(-1)
    if arr.size == 1 and p > 1:
        arr = np.repeat(arr, p)
    if arr.size != p:
        raise MethodIncompatibility(
            f"h0 has {arr.size} value(s) for {p} endogenous regressors "
            f"({', '.join(names)}).",
            recovery_hint="Pass one null value per endogenous regressor, in "
            "the order of `endog`, or a single number used for all.",
            diagnostics={"n_h0": int(arr.size), "n_endog": p},
        )
    if not np.all(np.isfinite(arr)):
        raise MethodIncompatibility(
            f"h0 must be finite; got {arr.tolist()}.",
            recovery_hint="Pass finite hypothesised coefficients.",
        )
    return arr


def _tsls(Y: np.ndarray, D: np.ndarray, Z: np.ndarray) -> tuple:
    """2SLS coefficients and homoskedastic standard errors (partialled data)."""
    Dhat = Z @ np.linalg.lstsq(Z, D, rcond=None)[0]
    A = Dhat.T @ D
    try:
        beta = np.linalg.solve(A, Dhat.T @ Y)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return np.full(D.shape[1], np.nan), np.full(D.shape[1], np.nan)
    e = Y - D @ beta
    V = float(e @ e) / len(Y) * np.linalg.inv(Dhat.T @ Dhat)
    return beta, np.sqrt(np.clip(np.diag(V), 0.0, None))


def ar_joint(
    data: pd.DataFrame,
    y: str,
    endog: List[str],
    instruments: List[str],
    exog: Optional[List[str]],
    *,
    h0: Any,
    alpha: float,
    ar_key: str,
    absorb_terms: List[str],
    cluster_names: List[str],
) -> Dict[str, Any]:
    """Joint Anderson-Rubin F test of ``beta = h0`` (several regressors)."""
    from .weak_iv import _cluster_meat_multiway, _partial_out, _prep_matrices_absorbed

    Y, D, Z, W, n, cluster_frame, fe_dof = _prep_matrices_absorbed(
        data, y, list(endog), instruments, exog, absorb_terms, cluster_names
    )
    p = D.shape[1]
    k_z = Z.shape[1]
    k_w = W.shape[1]
    if k_z < p:
        raise IdentificationFailure(
            f"Under-identified: {k_z} excluded instrument(s) for {p} "
            "endogenous regressors.",
            recovery_hint="Add instruments or drop endogenous regressors.",
            diagnostics={"n_instruments": k_z, "n_endog": p},
        )
    b0 = _null_vector(h0, p, endog)
    D_t = _partial_out(D, W)
    Z_t = _partial_out(Z, W)
    Y_t = _partial_out(Y, W)
    ZtZt_inv = np.linalg.inv(Z_t.T @ Z_t)
    Y_adj = Y_t - D_t @ b0
    moment = Z_t.T @ Y_adj
    proj = Z_t @ (ZtZt_inv @ moment)
    e_rf = Y_adj - proj

    df1 = k_z
    df2 = n - k_w - k_z - fe_dof
    if cluster_frame is not None:
        g_min = min(
            int(cluster_frame.iloc[:, j].nunique())
            for j in range(cluster_frame.shape[1])
        )
        df2 = g_min - 1
        omega = _cluster_meat_multiway(Z_t, e_rf, cluster_frame)
        omega = omega * (
            (g_min / max(g_min - 1, 1)) * ((n - 1) / max(n - k_w - k_z - fe_dof, 1))
        )
        stat = float(moment @ np.linalg.solve(omega, moment)) / df1
        label = "cluster"
    elif ar_key != "classic":
        if ar_key in ("HC2", "HC3"):
            q_w = np.linalg.qr(W)[0]
            lev = np.sum(q_w**2, axis=1) + np.einsum("ij,jk,ik->i", Z_t, ZtZt_inv, Z_t)
            w = 1.0 / (1.0 - lev) if ar_key == "HC2" else 1.0 / (1.0 - lev) ** 2
        elif ar_key == "HC1":
            w = np.full(n, n / max(n - k_w - k_z - fe_dof, 1))
        else:
            w = np.ones(n)
        omega = (Z_t * (w * e_rf**2)[:, None]).T @ Z_t
        stat = float(moment @ np.linalg.solve(omega, moment)) / df1
        label = ar_key
    else:
        rss = float(e_rf @ e_rf)
        stat = (float(proj @ Y_adj) / df1) / (rss / df2)
        label = "classic"
    pval = float(stats.f.sf(stat, df1, df2))
    beta, _ = _tsls(Y_t, D_t, Z_t)
    names = list(endog)
    return {
        "ar_stat": stat,
        "ar_df": (df1, int(df2)),
        "ar_pvalue": pval,
        "ar_ci": None,
        "ar_ci_disjoint": None,
        "ar_vcov": label,
        "h0": dict(zip(names, b0.tolist())),
        "endog": names,
        "beta_2sls": dict(zip(names, beta.tolist())),
        "first_stage_F": None,
        "first_stage_f": None,
        "effective_F": None,
        "effective_f": None,
        "tF_critical_value": None,
        "n_instruments": k_z,
        "n_obs": int(n),
        "strength": None,
        "interpretation": (
            f"Joint AR test of {', '.join(f'{nm}={v:g}' for nm, v in zip(names, b0))}: "
            f"F({df1},{int(df2)}) = {stat:.4f}, p = {pval:.4f}. With several "
            "endogenous regressors the AR confidence set is a region in "
            f"R^{p}; for intervals on single coefficients use "
            "sp.weakrobust(..., project=[...]) or subset=[...]. Instrument "
            "strength: sp.kleibergen_paap_rk / sp.sanderson_windmeijer."
        ),
    }


def _intervals_payload(intervals: List[tuple]) -> Dict[str, Any]:
    if not intervals:
        return {"set": [], "ci": (np.nan, np.nan), "empty": True, "disjoint": False}
    return {
        "set": [(float(a), float(b)) for a, b in intervals],
        "ci": (float(intervals[0][0]), float(intervals[-1][1])),
        "empty": False,
        "disjoint": len(intervals) > 1,
    }


def weakiv_panel(
    data: pd.DataFrame,
    y: str,
    endog: List[str],
    instruments: List[str],
    exog: Optional[List[str]],
    *,
    h0: Any,
    alpha: float,
    test_vcov: str,
    cluster_names: List[str],
    subset: List[str],
    project: List[str],
    kwt: float = 0.8,
) -> Dict[str, Any]:
    """AR / K / J / K-J / CLR in ``weakiv``'s LM form, plus sets."""
    from .weak_iv import _partial_out, _prep_matrices_absorbed

    vc = _TEST_VCOV.get(str(test_vcov).lower())
    if vc is None:
        raise MethodIncompatibility(
            f"test_vcov must be 'classic', 'robust' or 'cluster'; got {test_vcov!r}.",
            recovery_hint="Use test_vcov='classic' (i.i.d.), 'robust' "
            "(heteroskedasticity) or 'cluster' with cluster=.",
        )
    if vc == "cluster" and not cluster_names:
        raise MethodIncompatibility(
            "test_vcov='cluster' needs cluster=.",
            recovery_hint="Pass cluster='<column>'.",
        )
    if cluster_names and vc != "cluster":
        raise MethodIncompatibility(
            f"cluster= was given with test_vcov={test_vcov!r}, which does not use it.",
            recovery_hint="Drop test_vcov (cluster= implies 'cluster') or "
            "drop cluster=.",
        )
    if len(cluster_names) > 1:
        raise MethodIncompatibility(
            "the weak-IV-robust K / CLR tests support one-way clustering only.",
            recovery_hint="Pass a single cluster column (sp.anderson_rubin_test "
            "accepts multiway clustering for the AR test).",
            diagnostics={"cluster": cluster_names},
        )
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(
            f"alpha must lie in (0, 1); got {alpha!r}.",
            recovery_hint="Pass e.g. alpha=0.05.",
        )
    Y, D, Z, W, n, cluster_frame, _ = _prep_matrices_absorbed(
        data, y, list(endog), instruments, exog, [], cluster_names
    )
    p = D.shape[1]
    k = Z.shape[1]
    k_w = W.shape[1]
    names = list(endog)
    if k < p:
        raise IdentificationFailure(
            f"Under-identified: {k} excluded instrument(s) for {p} "
            "endogenous regressors.",
            recovery_hint="Add instruments or drop endogenous regressors.",
            diagnostics={"n_instruments": k, "n_endog": p},
        )
    b0 = _null_vector(h0, p, names)
    if p == 1 and (subset or project):
        raise MethodIncompatibility(
            "subset= and project= need two or more endogenous regressors.",
            recovery_hint="Drop subset= / project=; with one regressor the "
            "AR / K / CLR sets are already the intervals for it.",
        )
    Yt = _partial_out(Y, W)
    Dt = _partial_out(D, W)
    Zt = _partial_out(Z, W)
    codes = None
    if cluster_frame is not None:
        codes = pd.factorize(cluster_frame.iloc[:, 0])[0]
        n_g = int(codes.max()) + 1
        if n_g < 2:
            raise MethodIncompatibility(
                "cluster-robust weak-IV tests need at least two clusters; "
                f"found {n_g}.",
                recovery_hint="Cluster on a variable with more than one value.",
                diagnostics={"n_clusters": n_g},
            )
    eng = WeakIVLM(Yt, Dt, Zt, vcov=vc, cluster_codes=codes)
    res = eng.at(b0)
    beta, se = _tsls(Yt, Dt, Zt)

    out: Dict[str, Any] = {
        "convention": "weakiv",
        "test_vcov": {"iid": "classic"}.get(vc, vc),
        "endog": names,
        "h0": dict(zip(names, b0.tolist())),
        "beta_2sls": float(beta[0]) if p == 1 else dict(zip(names, beta.tolist())),
        "n_instruments": k,
        "n_obs": int(n),
        "ar_stat": res["ar"],
        "ar_df": res["ar_df"],
        "ar_pvalue": res["ar_p"],
        "k_stat": res["k"],
        "k_df": res["k_df"],
        "k_pvalue": res["k_p"],
        "j_stat": res["j"],
        "j_df": res["j_df"],
        "j_pvalue": res["j_p"],
        "clr_stat": res["clr"],
        "clr_pvalue": res["clr_p"],
        "rk_stat": res["rk"],
        "rk_df": res["rk_df"],
        "rk_pvalue": res["rk_p"],
        "kj_pvalue": None,
        "kwt": kwt,
    }
    if codes is not None:
        out["n_clusters"] = int(codes.max()) + 1
    if k > p:
        kp_, jp_ = res["k_p"], res["j_p"]
        out["kj_pvalue"] = float(
            min(kp_ / kwt * (1 - (1 - kwt) * kp_), jp_ / (1 - kwt) * (1 - kwt * jp_))
        )

    # Kleibergen-Paap under-identification test of the first stage
    from ..iv.weak_identification import kleibergen_paap_rk

    kp = kleibergen_paap_rk(
        Dt,
        Zt,
        add_const=False,
        n_absorbed=k_w,
        cov_type={"iid": "nonrobust"}.get(vc, vc),
        cluster=codes if vc == "cluster" else None,
    )
    out["kp_rk_lm"] = float(kp.rk_lm)
    out["kp_rk_lm_pvalue"] = float(kp.rk_lm_pvalue)
    out["kp_rk_f"] = float(kp.rk_f)

    if p == 1:
        center = float(beta[0]) if np.isfinite(beta[0]) else 0.0
        scale = float(se[0]) if np.isfinite(se[0]) and se[0] > 0 else 1.0
        sets = invert_pvalues(
            lambda b: eng.at([b], clr_alpha=alpha),
            ["ar_p", "k_p", "clr_p"],
            alpha,
            center,
            scale,
        )
        for test in ("ar", "k", "clr"):
            pay = _intervals_payload(sets[test + "_p"])
            out[f"{test}_set"] = pay["set"]
            out[f"{test}_ci"] = pay["ci"]
            out[f"{test}_is_empty"] = pay["empty"]
            out[f"{test}_is_disjoint"] = pay["disjoint"]
            out[f"{test}_is_unbounded"] = bool(
                pay["set"]
                and (np.isinf(pay["set"][0][0]) or np.isinf(pay["set"][-1][1]))
            )
        return out

    for opt, lst in (("subset", subset), ("project", project)):
        unknown = [v for v in lst if v not in names]
        if unknown:
            raise MethodIncompatibility(
                f"{opt}= names {unknown}, which are not endogenous regressors "
                f"({names}).",
                recovery_hint=f"Pass {opt}= a subset of `endog`.",
            )
        if lst and vc != "iid":
            raise MethodIncompatibility(
                f"{opt}= is available with i.i.d. errors only (test_vcov="
                "'classic'), as in Stata weakiv: the closed form that "
                "concentrates out the other coefficients (LIML) holds only "
                "there.",
                recovery_hint="Use test_vcov='classic' for subset / projection "
                "AR, or test the joint null with the robust tests.",
                diagnostics={"test_vcov": test_vcov},
            )
    L = k + k_w  # weakiv's L: excluded plus included instruments
    for key, lst, kind in (
        ("subset_ar", subset, "subset"),
        ("projection_ar", project, "projection"),
    ):
        block: Dict[str, Any] = {}
        for nm in lst:
            j = names.index(nm)
            others = [i for i in range(p) if i != j]
            sar = SubsetAR(Yt, Dt[:, j], Dt[:, others], Zt, L)
            fn = sar.subset if kind == "subset" else sar.projection
            at = fn(float(b0[j]))
            scale = float(se[j]) if np.isfinite(se[j]) and se[j] > 0 else 1.0
            center = float(beta[j]) if np.isfinite(beta[j]) else 0.0
            ints = invert_pvalue(lambda b, f=fn: f(b)["p"], alpha, center, scale)
            pay = _intervals_payload(ints)
            block[nm] = {
                "h0": float(b0[j]),
                "stat": float(at["stat"]),
                "df": int(at["df"]),
                "pvalue": float(at["p"]),
                "set": pay["set"],
                "ci": pay["ci"],
                "is_empty": pay["empty"],
                "is_disjoint": pay["disjoint"],
            }
        if block:
            out[key] = block
    return out
