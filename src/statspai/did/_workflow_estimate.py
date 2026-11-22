"""Steps 4-5 of ``sp.did_workflow``: the estimate and the event-study path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility
from ._workflow_common import AGG_TYPE, ESTIMAND_TEXT, ESTIMATOR_LABEL, facts_of, fmt
from ._workflow_spec import DiDStep

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import DiDWorkflow


# ===================================================================== #
# 4. Estimate
# ===================================================================== #
def headline_from_result(result: Any, estimand: str) -> Dict[str, Any]:
    ci = result.ci
    return {
        "estimate": float(result.estimate),
        "se": float(result.se),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "pvalue": float(result.pvalue),
        "estimand": estimand,
    }


def event_average(es_result: Any, alpha: float) -> Dict[str, Any]:
    """Equal-weight average of the post-treatment event-time effects.

    Uses the joint covariance of the path (``sp.event_study_vcov``), so the
    standard error accounts for the correlation between horizons.
    """
    from .es_inference import event_study_vcov

    ev = event_study_vcov(es_result)
    times = np.asarray(ev.times, dtype=float)
    post = times >= 0
    if not post.any():
        raise MethodIncompatibility(
            "sp.did_workflow: the event-study path has no post-treatment "
            "event time to average over.",
            recovery_hint="Widen event_window so that it includes e >= 0.",
        )
    weights = np.zeros(len(times))
    weights[post] = 1.0 / post.sum()
    estimate = float(weights @ np.asarray(ev.beta, dtype=float))
    variance = float(weights @ np.asarray(ev.vcov, dtype=float) @ weights)
    se = float(np.sqrt(max(variance, 0.0)))
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z = estimate / se if se > 0 else np.nan
    return {
        "estimate": estimate,
        "se": se,
        "ci_lower": estimate - crit * se,
        "ci_upper": estimate + crit * se,
        "pvalue": float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        "estimand": ESTIMAND_TEXT["event"],
    }


def fit_estimator(
    wf: "DiDWorkflow",
    *,
    control_group: str,
    covariates: Optional[List[str]],
    absorb: Optional[List[str]] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Fit the declared estimator; return (fit, event-study result, headline)."""
    inp = wf._inputs
    o = wf._options
    facts = facts_of(wf)
    alpha = o["alpha"]
    window = o["event_window"]
    target = o["target"]
    extra = dict(o["estimator_options"])
    estimator = o["estimator"]

    if estimator == "cs":
        from .aggte import aggte
        from .callaway_santanna import callaway_santanna

        kwargs: Dict[str, Any] = dict(
            y=inp.y,
            g=inp.cohort,
            t=inp.time,
            i=inp.id,
            x=covariates,
            weights=inp.weights,
            control_group=control_group,
            anticipation=o["anticipation"],
            alpha=alpha,
            base_period="universal",
        )
        if inp.cluster and inp.cluster != inp.id:
            kwargs["clustervars"] = inp.cluster
        kwargs.update(extra)
        fit = callaway_santanna(inp.data, **kwargs)
        win: Dict[str, Any] = {}
        if window is not None:
            win = {"min_e": window[0], "max_e": window[1]}
        es = aggte(fit, type="dynamic", alpha=alpha, **win)
        if target == "event":
            head = headline_from_result(es, ESTIMAND_TEXT["event"])
        else:
            agg = aggte(fit, type=AGG_TYPE[target], alpha=alpha)
            head = headline_from_result(agg, ESTIMAND_TEXT[target])
        return fit, es, head

    if estimator == "sa":
        from .sun_abraham import sun_abraham

        kwargs = dict(
            y=inp.y,
            g=inp.cohort,
            t=inp.time,
            i=inp.id,
            covariates=covariates,
            weights=inp.weights,
            cluster=inp.cluster,
            control_group=control_group,
            anticipation=o["anticipation"],
            event_window=tuple(window) if window is not None else None,
            alpha=alpha,
        )
        kwargs.update(extra)
        fit = sun_abraham(inp.data, **kwargs)
        head = (
            headline_from_result(fit, ESTIMAND_TEXT["overall"])
            if target == "overall"
            else event_average(fit, alpha)
        )
        return fit, fit, head

    # Borusyak-Jaravel-Spiess imputation.
    from .did_imputation import did_imputation

    cohorts = facts["cohorts"]
    max_pre = int(cohorts["n_pre_periods"].max())
    max_post = int(cohorts["n_post_periods"].max())
    lo, hi = -(max_pre - 1), max_post - 1
    if window is not None:
        lo, hi = max(lo, int(window[0])), min(hi, int(window[1]))
    horizon = [h for h in range(lo, hi + 1)]
    kwargs = dict(
        y=inp.y,
        group=inp.id,
        time=inp.time,
        first_treat=inp.cohort,
        controls=covariates,
        horizon=horizon,
        cluster=inp.cluster,
        weights=inp.weights,
        alpha=alpha,
    )
    absorbed = list(o.get("absorb") or []) if absorb is None else list(absorb)
    kwargs.update(extra)
    if absorbed:
        kwargs["fe"] = [inp.id, inp.time, *absorbed]
    elif absorb is not None:
        kwargs.pop("fe", None)
    custom_model = any(
        kwargs.get(k) is not None for k in ("fe", "unit_covariates", "time_covariates")
    )
    if custom_model and "pretrend_method" not in kwargs:
        # The Stata-style lead regression is pinned for the unit + period
        # model only; with another Y(0) model the leads are the averages
        # of the untreated residuals at each lead.
        kwargs["pretrend_method"] = "in-sample"
    fit = did_imputation(inp.data, **kwargs)
    head = (
        headline_from_result(fit, ESTIMAND_TEXT["overall"])
        if target == "overall"
        else event_average(fit, alpha)
    )
    return fit, fit, head


def run_estimate(wf: "DiDWorkflow", step: DiDStep) -> None:
    o = wf._options
    inp = wf._inputs
    fit, es, head = fit_estimator(
        wf, control_group=o["control_group"], covariates=inp.covariates
    )
    wf._fit, wf._es_result, wf._headline = fit, es, head
    step.objects.update(fit=fit, event_study_result=es)
    step.evidence = dict(head)
    step.evidence.update(
        estimator=o["estimator"],
        control_group=o["control_group"],
        n_obs=int(getattr(fit, "n_obs", facts_of(wf)["n_obs"])),
    )
    level = int(round(100 * (1 - o["alpha"])))
    step.status = "pass"
    step.finding = (
        f"{ESTIMATOR_LABEL[o['estimator']]}: {head['estimand']} = "
        f"{fmt(head['estimate'])} (SE {fmt(head['se'])}, {level}% CI "
        f"[{fmt(head['ci_lower'])}, {fmt(head['ci_upper'])}], "
        f"p = {fmt(head['pvalue'], 3)})."
    )
    if wf._twfe is not None and np.isfinite(wf._twfe):
        step.evidence["twfe"] = wf._twfe
        step.finding += f" TWFE on the same panel gives {fmt(wf._twfe)}."
    # Overlap, read off the fit (Baker et al. 2026, section 4.4: check
    # strong overlap with the estimated propensity scores). Only a fit
    # that estimated them carries the counts.
    info = getattr(fit, "model_info", None) or {}
    if inp.covariates and "n_pscore_trimmed" in info:
        trimmed = int(info["n_pscore_trimmed"])
        step.evidence["overlap"] = {
            "pscore_trim": info.get("pscore_trim"),
            "n_pscore_trimmed": trimmed,
            "n_pscore_boundary_cells": info.get("n_pscore_boundary_cells"),
        }
        step.finding += (
            f" Overlap: {trimmed} comparison unit-cell(s) had an estimated "
            f"propensity score at or above {info.get('pscore_trim')} and "
            "were trimmed."
            if trimmed
            else " Overlap: no comparison unit had an estimated propensity "
            f"score at or above the trimming bound {info.get('pscore_trim')}."
        )


# ===================================================================== #
# 5. Event study
# ===================================================================== #
def _cohorts_at(facts: Dict[str, Any], times: np.ndarray) -> Optional[np.ndarray]:
    """How many treated cohorts are observed at each event time."""
    if not facts["evenly_spaced"]:
        return None
    ranks = np.asarray(facts["cohort_ranks"], dtype=int)
    last = facts["n_periods"] - 1
    return np.array(
        [int(((ranks + int(e) >= 0) & (ranks + int(e) <= last)).sum()) for e in times]
    )


def run_event_study(wf: "DiDWorkflow", step: DiDStep) -> None:
    from .es_inference import uniform_bands

    o = wf._options
    es = wf._es_result
    table = uniform_bands(es, alpha=o["alpha"], seed=o["seed"])
    attrs = dict(table.attrs)
    table = pd.DataFrame(table).reset_index(drop=True)
    times = table["relative_time"].to_numpy(dtype=float)
    if "n_cohorts" not in table.columns:
        counts = _cohorts_at(facts_of(wf), times)
        if counts is not None:
            table["n_cohorts"] = counts
    wf._path = table
    post = table[table["relative_time"] >= 0]
    pre = table[table["relative_time"] < -o["anticipation"]]
    step.evidence = {
        "path": table,
        "crit_pointwise": attrs.get("crit_pointwise"),
        "crit_uniform": attrs.get("crit_uniform"),
        "joint_covariance": bool(attrs.get("joint", False)),
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
    }
    if post.empty:
        step.status = "fail"
        step.finding = "The path has no post-treatment event time."
        step.action = "Widen event_window to include e >= 0."
        return

    outside = int(
        ((post["cband_lower"] > 0) | (post["cband_upper"] < 0)).sum()
        if "cband_lower" in post
        else 0
    )
    lead = (
        f"{len(post)} post-treatment and {len(pre)} pre-treatment event "
        f"times; the simultaneous {int(round(100 * (1 - o['alpha'])))}% band "
        f"(critical value {fmt(attrs.get('crit_uniform'), 3)} against "
        f"{fmt(attrs.get('crit_pointwise'), 3)} pointwise) excludes zero at "
        f"{outside} of {len(post)} post-treatment event times."
    )
    composition = ""
    if "n_cohorts" in post.columns and post["n_cohorts"].nunique() > 1:
        first, last = post.iloc[0], post.iloc[-1]
        composition = (
            f" The cohorts behind the path change with the horizon: "
            f"{int(first['n_cohorts'])} at e = {int(first['relative_time'])}, "
            f"{int(last['n_cohorts'])} at e = {int(last['relative_time'])}. "
            "Differences along the path mix dynamics with composition."
        )
        step.action = (
            "Report the path over a window every cohort reaches "
            "(event_window=), or sp.aggte(fit, type='dynamic', balance_e=k)."
        )
    step.status = "caution" if composition else "pass"
    step.finding = lead + composition
