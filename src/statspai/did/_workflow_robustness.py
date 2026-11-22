"""Steps 8-9 of ``sp.did_workflow``: alternatives and the report."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility
from ._workflow_common import (
    AGG_TYPE,
    ESTIMAND_TEXT,
    ESTIMATOR_LABEL,
    JACKKNIFE_AUTO_MAX_CLUSTERS,
    PT_TEXT,
    facts_of,
    fmt,
    headline_of,
    pct,
)
from ._workflow_spec import DiDStep

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import DiDWorkflow

from ._workflow_estimate import fit_estimator, headline_from_result


# ===================================================================== #
# 8. Robustness
# ===================================================================== #
def _row(
    check: str, variant: str, head: Dict[str, Any], note: str = ""
) -> Dict[str, Any]:
    return {
        "check": check,
        "variant": variant,
        "estimate": head.get("estimate", np.nan),
        "se": head.get("se", np.nan),
        "ci_lower": head.get("ci_lower", np.nan),
        "ci_upper": head.get("ci_upper", np.nan),
        "note": note,
    }


def run_robustness(wf: "DiDWorkflow", step: DiDStep) -> None:
    o = wf._options
    inp = wf._inputs
    facts = facts_of(wf)
    head = headline_of(wf)
    estimator = o["estimator"]
    rows: List[Dict[str, Any]] = [
        _row("headline", f"{estimator}, {o['control_group']}", head, head["estimand"])
    ]
    concerns: List[str] = []

    # (a) the other estimators of the family, on one estimand
    if o.get("absorb"):
        rows.append(
            _row(
                "estimator",
                "not run",
                {},
                "The other estimators take no fixed effects beyond unit and "
                "period, so they would answer under a different assumption.",
            )
        )
        alt = wf._optional(
            step,
            "unit and period effects only",
            lambda: fit_estimator(
                wf,
                control_group=o["control_group"],
                covariates=inp.covariates,
                absorb=[],
            )[2],
        )
        if alt is not None:
            rows.append(
                _row(
                    "fixed effects",
                    "unit and period only",
                    alt,
                    "Parallel trends across the absorbed cells; a gap to the "
                    "headline is what the absorbed effects are doing.",
                )
            )
    elif inp.weights:
        rows.append(
            _row(
                "estimator",
                "not run",
                {},
                "sp.did_summary takes no weights; fit the alternatives "
                "with their own weights= argument.",
            )
        )
    else:
        from .summary import did_summary

        summary = wf._optional(
            step,
            "did_summary",
            lambda: did_summary(
                inp.data,
                y=inp.y,
                time=inp.time,
                first_treat=inp.cohort,
                group=inp.id,
                controls=inp.covariates,
                cluster=inp.cluster,
                alpha=o["alpha"],
            ),
        )
        if summary is not None:
            step.objects["did_summary"] = summary
            for _, r in summary.detail.iterrows():
                rows.append(
                    _row(
                        "estimator",
                        str(r["method"]),
                        {
                            "estimate": r["estimate"],
                            "se": r["se"],
                            "ci_lower": r["ci_low"],
                            "ci_upper": r["ci_high"],
                        },
                        str(r["estimand"]),
                    )
                )

    # (b) the other comparison group
    if estimator == "cs":
        other = (
            "notyettreated" if o["control_group"] == "nevertreated" else "nevertreated"
        )
        feasible = (
            facts["n_cohorts"] > 1 or facts["n_comparison_never"] > 0
            if other == "notyettreated"
            else facts["n_comparison_never"] > 0
        )
        if feasible:
            alt = wf._optional(
                step,
                f"control_group={other!r}",
                lambda: fit_estimator(
                    wf, control_group=other, covariates=inp.covariates
                )[2],
            )
            if alt is not None:
                rows.append(_row("comparison group", other, alt, alt["estimand"]))

    # (c) without the covariates
    if inp.covariates:
        alt = wf._optional(
            step,
            "unconditional parallel trends",
            lambda: fit_estimator(
                wf, control_group=o["control_group"], covariates=None
            )[2],
        )
        if alt is not None:
            rows.append(
                _row(
                    "covariates",
                    "none",
                    alt,
                    "Unconditional parallel trends; a gap to the headline is "
                    "what the covariates are doing.",
                )
            )

    # (d) the standard error under a second construction
    n_clusters = int(wf.step("design").evidence.get("n_clusters", 0))
    run_jackknife = o["jackknife"]
    if run_jackknife is None:
        run_jackknife = estimator == "cs" and n_clusters <= JACKKNIFE_AUTO_MAX_CLUSTERS
    if run_jackknife and estimator != "cs":
        raise MethodIncompatibility(
            "sp.did_workflow: jackknife=True is available with "
            f"estimator='cs', not {estimator!r}.",
            recovery_hint="Leave jackknife=None, or set estimator='cs'.",
        )
    if run_jackknife:
        from .cs_jackknife import cs_jackknife

        cs_kwargs: Dict[str, Any] = dict(
            x=inp.covariates,
            weights=inp.weights,
            control_group=o["control_group"],
            anticipation=o["anticipation"],
            base_period="universal",
        )
        cs_kwargs.update(o["estimator_options"])
        jack = wf._optional(
            step,
            "cs_jackknife",
            lambda: cs_jackknife(
                inp.data,
                y=inp.y,
                g=inp.cohort,
                time=inp.time,
                id=inp.id,
                type=AGG_TYPE[o["target"]],
                cluster=inp.cluster,
                alpha=o["alpha"],
                **cs_kwargs,
            ),
        )
        if jack is not None:
            step.objects["jackknife"] = jack
            jhead = headline_from_result(jack, head["estimand"])
            ratio = jhead["se"] / head["se"] if head["se"] > 0 else np.nan
            rows.append(
                _row(
                    "standard error",
                    "cluster jackknife",
                    jhead,
                    f"Jackknife SE is {fmt(ratio, 3)} times the analytic SE.",
                )
            )
            step.evidence["jackknife_se_ratio"] = float(ratio)
            analytic_excludes = head["ci_lower"] > 0 or head["ci_upper"] < 0
            jack_excludes = jhead["ci_lower"] > 0 or jhead["ci_upper"] < 0
            if analytic_excludes and not jack_excludes:
                concerns.append(
                    "the cluster-jackknife interval includes zero where the "
                    "analytic one does not"
                )

    # (e) the outcome scale
    from .functional_form import functional_form_test

    ff = wf._optional(
        step,
        "functional_form_test",
        lambda: functional_form_test(
            inp.data,
            y=inp.y,
            g=inp.cohort,
            t=inp.time,
            i=inp.id,
            control_group=(
                o["control_group"]
                if o["control_group"] in ("nevertreated", "notyettreated")
                else "nevertreated"
            ),
            x=inp.covariates,
            weights=inp.weights,
            anticipation=o["anticipation"],
            alpha=o["alpha"],
            random_state=o["seed"],
        ),
    )
    if ff is not None:
        step.objects["functional_form"] = ff
        step.evidence["functional_form_pvalue"] = float(ff.pvalue)
        rejects = bool(ff.pvalue < o["alpha"])
        rows.append(
            _row(
                "outcome scale",
                "functional-form test",
                {},
                f"p = {fmt(ff.pvalue, 3)}: "
                + (
                    "parallel trends cannot hold for every monotone "
                    "transformation of the outcome."
                    if rejects
                    else "no evidence that parallel trends depends on the "
                    "scale of the outcome."
                ),
            )
        )
        if rejects:
            concerns.append(
                "parallel trends is sensitive to the scale of the outcome "
                "(levels against logs)"
            )

    table = pd.DataFrame(rows)
    wf._robustness = table
    step.evidence["table"] = table

    comparable = table[
        table["check"].isin(
            ["estimator", "comparison group", "covariates", "fixed effects"]
        )
        & np.isfinite(pd.to_numeric(table["estimate"], errors="coerce"))
    ]
    est = head["estimate"]
    if len(comparable):
        values = comparable["estimate"].astype(float)
        step.evidence["min_estimate"] = float(values.min())
        step.evidence["max_estimate"] = float(values.max())
        flips = comparable[np.sign(values) != np.sign(est)]
        outside = comparable[(values < head["ci_lower"]) | (values > head["ci_upper"])]
        if len(flips):
            concerns.append(
                "the sign changes under "
                + ", ".join(
                    f"{c}: {v}" for c, v in zip(flips["check"], flips["variant"])
                )
            )
        elif len(outside):
            concerns.append(
                "the estimate leaves the headline interval under "
                + ", ".join(
                    f"{c}: {v}" for c, v in zip(outside["check"], outside["variant"])
                )
            )
        # A headline that excludes zero is a conclusion about the sign; an
        # alternative whose own interval includes zero does not share it.
        lo = pd.to_numeric(comparable["ci_lower"], errors="coerce")
        hi = pd.to_numeric(comparable["ci_upper"], errors="coerce")
        headline_excludes_zero = head["ci_lower"] > 0 or head["ci_upper"] < 0
        includes_zero = comparable[
            (lo <= 0) & (hi >= 0) & (np.sign(values) == np.sign(est))
        ]
        step.evidence["n_alternatives_including_zero"] = int(len(includes_zero))
        if headline_excludes_zero and len(includes_zero):
            concerns.append(
                "the interval includes zero under "
                + ", ".join(
                    f"{c}: {v}"
                    for c, v in zip(includes_zero["check"], includes_zero["variant"])
                )
            )
        span = (
            f"Across {len(comparable)} alternative specification(s) the "
            f"estimate ranges from {fmt(values.min())} to {fmt(values.max())} "
            f"(headline {fmt(est)})."
        )
    else:
        span = (
            "No alternative specification was fitted; wf.robustness gives the "
            "reason for each check that was left out."
        )

    if concerns:
        step.status = "caution"
        step.finding = span + " But " + "; ".join(concerns) + "."
        step.action = (
            "Say which choice moves the answer and why the headline is the "
            "one that matches the declared target."
        )
    else:
        step.status = "pass" if len(comparable) else "skipped"
        step.finding = span


# ===================================================================== #
# 9. Report
# ===================================================================== #
def run_report(wf: "DiDWorkflow", step: DiDStep) -> None:
    o = wf._options
    inp = wf._inputs
    facts = wf._facts or {}
    head = wf._headline
    level = int(round(100 * (1 - o["alpha"])))
    k = o["anticipation"]
    items: List[Tuple[str, str]] = []

    items.append(("Target parameter", ESTIMAND_TEXT[o["target"]]))
    cov = ", conditional on " + ", ".join(inp.covariates) if inp.covariates else ""
    items.append(
        ("Identifying assumption", PT_TEXT[o["control_group"]].capitalize() + cov)
    )
    if facts:
        items.append(
            (
                "Sample",
                f"{facts['n_units']} units, {facts['n_periods']} periods, "
                f"{facts['n_cohorts']} treated cohort(s), "
                f"{facts['n_comparison_never']} never-treated unit(s)",
            )
        )
    items.append(
        (
            "Weights",
            (
                f"{inp.weights} (population-weighted effect)"
                if inp.weights
                else "none (each unit counts once)"
            ),
        )
    )
    items.append(("Anticipation", f"{k} period(s)"))
    items.append(
        (
            "Reference period",
            f"e = {-1 - k}, the same for every lead and lag (universal base)",
        )
    )
    window = o["event_window"]
    path = wf._path
    if path is not None and len(path):
        lo, hi = int(path["relative_time"].min()), int(path["relative_time"].max())
        items.append(
            (
                "Event window",
                f"e = {lo} to {hi}"
                + (" (declared)" if window is not None else " (all available)"),
            )
        )
    items.append(("Estimator", ESTIMATOR_LABEL[o["estimator"]]))
    design = wf.step("design")
    if design.evidence:
        items.append(
            (
                "Standard errors",
                f"clustered on {inp.cluster or inp.id} "
                f"({design.evidence.get('n_clusters')} clusters, "
                f"{design.evidence.get('n_treated_clusters')} treated); "
                "pointwise and simultaneous intervals reported",
            )
        )
    if head is not None:
        items.append(
            (
                "Estimate",
                f"{fmt(head['estimate'])} (SE {fmt(head['se'])}; {level}% CI "
                f"[{fmt(head['ci_lower'])}, {fmt(head['ci_upper'])}])",
            )
        )
    pre = wf.step("pretrends")
    if pre.evidence.get("pvalue") is not None:
        text = (
            f"joint p = {fmt(pre.evidence['pvalue'], 3)} on "
            f"{pre.evidence['df']} coefficients"
        )
        if "bias_to_effect_ratio" in pre.evidence:
            text += (
                "; a trend detected with 80% power would bias the estimate by "
                f"{pct(pre.evidence['bias_to_effect_ratio'])}"
            )
        items.append(("Pre-trends", text))
    sens = wf.step("sensitivity")
    if sens.evidence.get("breakdown_rm") is not None:
        items.append(
            (
                "Sensitivity",
                f"breakdown Mbar = {fmt(sens.evidence['breakdown_rm'], 3)} "
                "(relative magnitudes, average post-treatment effect)",
            )
        )
    rob = wf.step("robustness")
    if "min_estimate" in rob.evidence:
        items.append(
            (
                "Robustness",
                f"{fmt(rob.evidence['min_estimate'])} to "
                f"{fmt(rob.evidence['max_estimate'])} across estimators and "
                "comparison groups",
            )
        )
    open_items = [
        f"step {s.number} ({s.key}): {s.status}"
        for s in wf._steps
        if s.status in ("caution", "fail")
    ]
    items.append(("Open caveats", "; ".join(open_items) if open_items else "none"))

    checklist = pd.DataFrame(items, columns=["item", "statement"])
    statement = " ".join(f"{name}: {text}." for name, text in items)
    refs: List[str] = []
    for s in wf._steps:
        if s.status not in ("pending", "skipped") or s.key == "report":
            for key in s.references:
                if key not in refs:
                    refs.append(key)
    step.evidence = {
        "checklist": checklist,
        "statement": statement,
        "citations": refs,
    }
    step.status = "info"
    blocked = f" Estimation was not run because {wf._blocked}." if wf._blocked else ""
    step.finding = (
        f"{len(items)} items a reader needs to tell which number this is; "
        f"{len(open_items)} step(s) left a caveat the write-up owes." + blocked
    )
    step.action = "wf.report() returns the statement; wf.cite() the references."
