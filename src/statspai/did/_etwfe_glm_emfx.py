"""Aggregations of a nonlinear ETWFE fit (``sp.etwfe_emfx`` on
``sp.etwfe(family=...)`` results)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._etwfe_glm_design import normalise_scale


def etwfe_glm_emfx(
    result: CausalResult,
    type: str,
    alpha: float,
    scale: Optional[str] = None,
    include_leads: bool = False,
    by_xvar: bool = False,
) -> CausalResult:
    """Serve the aggregations a nonlinear ``sp.etwfe`` fit already computed.

    ``scale=None`` keeps the scale the fit was reported on.  For
    ``type='simple'`` the estimate and SE are that scale's overall ATT.  For
    the other types ``detail`` holds one row per cohort / event time /
    period with its own delta-method SE; the headline ``estimate`` is the
    unweighted mean of those rows and ``se`` the overall ATT's SE (the rows
    share coefficients, so averaging their SEs would understate).

    ``model_info['event_study']`` and ``model_info['event_study_vcov']`` of
    the returned result are on the served scale, so ``sp.event_study_vcov``
    / ``sp.pretrends_test`` / ``sp.honest_did`` applied to it test the
    scale that was asked for.

    ``by_xvar=True`` (a fit with one categorical ``xvar``) reports the
    aggregation separately for every level of that covariate -- Stata
    ``estat ..., over()``, R ``emfx(by_xvar = TRUE)`` -- with a ``level``
    column in ``detail``.
    """
    mi = result.model_info or {}
    sc = normalise_scale(scale if scale is not None else mi.get("scale", "response"))
    aggs = mi.get("aggregations")
    if not isinstance(aggs, dict) or sc not in aggs:
        raise MethodIncompatibility(
            "This nonlinear etwfe result predates scale-aware aggregation; "
            "re-fit it with the current sp.etwfe.",
            recovery_hint="Call sp.etwfe(..., family=...) again.",
            diagnostics={"scale": sc},
        )
    agg = aggs[sc]
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    if by_xvar:
        return _emfx_by_xvar(result, agg, type, sc, alpha, z_crit)
    simple = agg["simple"]
    se_all = float(simple["se"])
    ev_full = agg.get("event")
    scale_mi: Dict[str, Any] = {}
    if isinstance(ev_full, pd.DataFrame) and not ev_full.empty:
        scale_mi = {
            "event_study": ev_full.drop(
                columns=["ci_lower", "ci_upper"], errors="ignore"
            ),
            "event_study_vcov": agg.get("event_vcov"),
        }

    if type == "simple":
        if sc == mi.get("scale", "response") and alpha == result.alpha:
            return result
        est = float(simple["att"])
        z = est / se_all if se_all > 0 else 0.0
        return CausalResult(
            method=f"{result.method} — emfx[simple, {sc}]",
            estimand=(
                "ATT (average marginal effect, response scale)"
                if sc == "response"
                else "ATT (link scale)"
            ),
            estimate=est,
            se=se_all,
            pvalue=float(2 * stats.norm.sf(abs(z))),
            ci=(est - z_crit * se_all, est + z_crit * se_all),
            alpha=alpha,
            n_obs=result.n_obs,
            detail=None,
            model_info={**mi, **scale_mi, "emfx_type": "simple", "emfx_scale": sc},
            _citation_key="wooldridge2021two",
        )

    label = {"event": "relative_time", "group": "cohort", "calendar": "period"}[type]
    frame = agg[type]
    if isinstance(frame, pd.DataFrame) and type == "event" and not include_leads:
        if "relative_time" in frame.columns:
            frame = frame[frame["relative_time"] >= 0]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataInsufficient(
            f"etwfe_emfx(type={type!r}) has no cells to report.",
            recovery_hint="Check the treated cohorts and event window.",
            diagnostics={"type": type},
        )
    frame = frame.copy()
    frame["ci_lower"] = frame["att"] - z_crit * frame["se"]
    frame["ci_upper"] = frame["att"] + z_crit * frame["se"]
    est = float(np.average(frame["att"].to_numpy(dtype=float)))
    z_stat = est / se_all if se_all > 0 else 0.0
    return CausalResult(
        method=f"{result.method} — emfx[{type}]",
        estimand=result.estimand if sc == mi.get("scale") else f"ATT ({sc} scale)",
        estimate=est,
        se=se_all,
        pvalue=float(2 * stats.norm.sf(abs(z_stat))),
        ci=(est - z_crit * se_all, est + z_crit * se_all),
        alpha=alpha,
        n_obs=result.n_obs,
        detail=frame.reset_index(drop=True),
        model_info={
            **mi,
            **scale_mi,
            "emfx_type": type,
            "emfx_label": label,
            "emfx_scale": sc,
            "emfx_note": (
                "estimate is the unweighted mean of the reported cells; "
                "se is the overall delta-method SE from the fit; each row "
                "carries its own delta-method se"
            ),
        },
        _citation_key="wooldridge2021two",
    )


def _emfx_by_xvar(
    result: CausalResult,
    agg: Dict[str, Any],
    type: str,
    sc: str,
    alpha: float,
    z_crit: float,
) -> CausalResult:
    """``etwfe_emfx(..., by_xvar=True)`` for a nonlinear fit."""
    mi = result.model_info or {}
    by = agg.get("by_xvar")
    if not isinstance(by, dict) or not by:
        raise MethodIncompatibility(
            "by_xvar=True needs a fit with exactly one categorical xvar.",
            recovery_hint="Refit with xvar='<column>' where the column is a "
            "pandas category / object / bool (cast integer codes with "
            ".astype('category')).",
            diagnostics={"xvar": mi.get("xvar")},
        )
    frames = []
    for lab, tabs in by.items():
        if type == "simple":
            s = tabs["simple"]
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "level": lab,
                            "att": s["att"],
                            "se": s["se"],
                            "n_treated": s["n_treated"],
                        }
                    ]
                )
            )
        else:
            f = tabs[type]
            if isinstance(f, pd.DataFrame) and not f.empty:
                f = f.drop(columns=["ci_lower", "ci_upper"], errors="ignore").copy()
                f.insert(0, "level", lab)
                frames.append(f)
    detail = pd.concat(frames, ignore_index=True)
    se = detail["se"].to_numpy(dtype=float)
    att_v = detail["att"].to_numpy(dtype=float)
    z = np.divide(att_v, se, out=np.full_like(att_v, np.nan), where=se > 0)
    detail["pvalue"] = 2 * stats.norm.sf(np.abs(z))
    detail["ci_lower"] = att_v - z_crit * se
    detail["ci_upper"] = att_v + z_crit * se
    simple = agg["simple"]
    est, se_all = float(simple["att"]), float(simple["se"])
    zz = est / se_all if se_all > 0 else 0.0
    return CausalResult(
        method=f"{result.method} — emfx[{type}, by {mi.get('by_xvar_var')}]",
        estimand=f"ATT by {mi.get('by_xvar_var')} ({sc} scale)",
        estimate=est,
        se=se_all,
        pvalue=float(2 * stats.norm.sf(abs(zz))),
        ci=(est - z_crit * se_all, est + z_crit * se_all),
        alpha=alpha,
        n_obs=result.n_obs,
        detail=detail,
        model_info={
            **mi,
            "emfx_type": type,
            "emfx_scale": sc,
            "emfx_by_xvar": mi.get("by_xvar_var"),
            "emfx_note": "estimate / se are the pooled simple ATT; detail "
            "holds one row per covariate level (x event time / cohort / "
            "period for those types) with its own delta-method se",
        },
        _citation_key="wooldridge2021two",
    )
