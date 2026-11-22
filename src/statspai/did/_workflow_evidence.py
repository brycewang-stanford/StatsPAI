"""Steps 6-7 of ``sp.did_workflow``: pre-trends and honest-DiD bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd

from ._workflow_common import RM_BENCHMARK, fmt, path_of, pct
from ._workflow_spec import DiDStep

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import DiDWorkflow


# ===================================================================== #
# 6. Pre-trends
# ===================================================================== #
def run_pretrends(wf: "DiDWorkflow", step: DiDStep) -> None:
    from .pretrends import pretrends_slope_for_power, pretrends_test

    o = wf._options
    es = wf._es_result
    path = path_of(wf)
    k = o["anticipation"]
    pre = path[path["relative_time"] < -k]
    post = path[path["relative_time"] >= 0]
    if pre.empty:
        step.status = "caution"
        step.finding = (
            "No pre-treatment coefficient is available besides the "
            "reference period, so parallel trends rests entirely on the "
            "argument for the comparison group."
        )
        step.action = "Argue the comparison group from institutional detail."
        return

    window = (float(pre["relative_time"].min()), float(-k - 1)) if k else None
    test = pretrends_test(es, alpha=o["alpha"], window=window)
    step.evidence = {
        "statistic": float(test["statistic"]),
        "df": int(test["df"]),
        "pvalue": float(test["pvalue"]),
        "reject": bool(test["reject"]),
        "pre_periods": list(test.get("pre_periods", [])),
    }
    cell = (getattr(wf._fit, "model_info", None) or {}).get("pretrend_test")
    rank_deficient = False
    if isinstance(cell, dict) and "rank" in cell:
        step.evidence["cell_test"] = {
            key: cell.get(key)
            for key in ("statistic", "df", "rank", "rank_deficient", "pvalue")
        }
        rank_deficient = bool(cell.get("rank_deficient"))

    avg_post = float(post["att"].mean()) if len(post) else np.nan
    bias: Dict[float, float] = {}
    for target_power in (0.5, 0.8):

        def detectable_slope(power: float = target_power) -> Dict[str, Any]:
            found: Dict[str, Any] = pretrends_slope_for_power(
                es, target_power=power, alpha=o["alpha"]
            )
            return found

        found = wf._optional(step, "pretrends_slope_for_power", detectable_slope)
        if found is None:
            continue
        slope = float(found["slope"])
        # The hypothesised violation is linear in event time and zero at
        # the reference period e = -1.
        bias[target_power] = slope * float((post["relative_time"] + 1).mean())
        step.evidence[f"slope_power_{int(target_power * 100)}"] = slope
        step.evidence[f"bias_power_{int(target_power * 100)}"] = bias[target_power]
    step.evidence["average_post_effect"] = avg_post

    lead = (
        f"Joint test of {int(test['df'])} pre-treatment coefficients: "
        f"{test.get('stat_label', 'Wald')} = {fmt(test['statistic'], 3)}, "
        f"p = {fmt(test['pvalue'], 3)}."
    )
    power = ""
    ratio = np.nan
    if 0.8 in bias and np.isfinite(avg_post) and avg_post != 0:
        ratio = abs(bias[0.8]) / abs(avg_post)
        step.evidence["bias_to_effect_ratio"] = float(ratio)
        power = (
            f" A linear trend this test detects only 80% of the time would "
            f"bias the average post-treatment effect by {fmt(abs(bias[0.8]))}, "
            f"{pct(ratio)} of the estimate ({fmt(avg_post)})."
        )
    if test["reject"]:
        step.status = "fail"
        step.finding = (
            lead + " The pre-treatment coefficients reject at "
            f"{fmt(o['alpha'], 2)}: the treated and comparison groups were "
            "already diverging." + power
        )
        step.action = (
            "Do not report the point estimate alone. Lead with the "
            "sensitivity analysis of step 7, and reconsider the comparison "
            "group or conditioning covariates."
        )
    elif np.isfinite(ratio) and ratio >= 1.0:
        step.status = "caution"
        step.finding = (
            lead + " It does not reject, but it could not have: " + power.strip()
        )
        step.action = (
            "Do not cite the pre-trend test as support for parallel "
            "trends; report the sensitivity analysis of step 7 instead."
        )
    else:
        step.status = "caution" if rank_deficient else "pass"
        step.finding = (
            lead + " It does not reject, which is not evidence that "
            "parallel trends holds." + power
        )
        if rank_deficient:
            step.finding += (
                " The cell-level covariance is rank deficient; report the "
                "test with its rank."
            )


# ===================================================================== #
# 7. Sensitivity
# ===================================================================== #
def run_sensitivity(wf: "DiDWorkflow", step: DiDStep) -> None:
    from .honest_did import breakdown_m, honest_did

    o = wf._options
    es = wf._es_result
    path = path_of(wf)
    k = o["anticipation"]
    n_pre = int((path["relative_time"] < -k).sum())
    n_post = int((path["relative_time"] >= 0).sum())
    if n_pre == 0 or n_post == 0:
        step.status = "skipped"
        step.finding = (
            "Honest-DiD bounds need at least one pre-treatment coefficient "
            "to calibrate the violation against; the path has none."
        )
        return

    l_vec = np.ones(n_post) / n_post
    alpha, seed = o["alpha"], o["seed"]
    grid = honest_did(
        es,
        method="relative_magnitude",
        l_vec=l_vec,
        m_grid=[0.0, 0.5, 1.0, 2.0],
        alpha=alpha,
        seed=seed,
    )
    grid = pd.DataFrame(grid).reset_index(drop=True)
    rm = float(
        breakdown_m(
            es, method="relative_magnitude", l_vec=l_vec, alpha=alpha, seed=seed
        )
    )
    rm_impact = wf._optional(
        step,
        "breakdown_m (on impact)",
        lambda: float(
            breakdown_m(es, e=0, method="relative_magnitude", alpha=alpha, seed=seed)
        ),
    )
    sd = wf._optional(
        step,
        "breakdown_m (smoothness)",
        lambda: float(breakdown_m(es, method="smoothness", l_vec=l_vec, alpha=alpha)),
    )
    step.evidence = {
        "target": "equal-weight average of post-treatment event times",
        "rm_grid": grid,
        "breakdown_rm": rm,
        "breakdown_rm_on_impact": rm_impact,
        "breakdown_sd": sd,
        "benchmark_rm": RM_BENCHMARK,
        "seed": seed,
    }
    wf._sensitivity = step.evidence

    exact = grid.iloc[0]
    if not bool(exact["rejects_zero"]):
        step.status = "info"
        step.finding = (
            "Under exact parallel trends the interval for the average "
            f"post-treatment effect is [{fmt(exact['ci_lower'])}, "
            f"{fmt(exact['ci_upper'])}] and already includes zero, so "
            "there is no conclusion for a violation to overturn."
        )
        return
    lead = (
        "The sign of the average post-treatment effect survives "
        f"post-treatment violations of parallel trends up to {fmt(rm, 3)} "
        "times the largest violation between consecutive pre-treatment "
        "periods"
    )
    if sd is not None and np.isfinite(sd):
        lead += f" (smoothness restriction: M* = {fmt(sd, 3)})"
    if rm >= RM_BENCHMARK:
        step.status = "pass"
        step.finding = lead + ". That covers the reference case Mbar = 1."
    else:
        step.status = "caution"
        step.finding = (
            lead + ". A violation as large as the largest one already "
            "seen before treatment (Mbar = 1) is enough to lose the sign."
        )
        step.action = (
            "Report the robust interval next to the estimate: "
            "sp.honest_did(wf.event_study_result, method='relative_magnitude')."
        )
