"""Steps 1-3 of ``sp.did_workflow``: declare, audit the design, diagnose TWFE."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import numpy as np
import pandas as pd

from ._workflow_common import (
    ESTIMAND_TEXT,
    ESTIMATOR_LABEL,
    EVENT_WINDOW_DEFAULT_SPAN,
    FEW_CLUSTERS,
    facts_of,
    fmt,
    pct,
    pt_text,
)
from ._workflow_spec import DiDStep, design_facts

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import DiDWorkflow


# ===================================================================== #
# 1. Declare
# ===================================================================== #
def run_declare(wf: "DiDWorkflow", step: DiDStep) -> None:
    o = wf._options
    inp = wf._inputs
    explicit = wf._explicit

    def source(name: str) -> str:
        return "declared" if name in explicit else "default"

    cov_text = ", conditional on " + ", ".join(inp.covariates) if inp.covariates else ""
    if o.get("absorb"):
        cov_text += ", within cells of " + ", ".join(o["absorb"])
    window = o["event_window"]
    rows = [
        (
            "target",
            o["target"],
            source("target"),
            ESTIMAND_TEXT[o["target"]],
        ),
        (
            "control_group",
            o["control_group"],
            (
                "implied by estimator"
                if o["control_group_implied"]
                else source("control_group")
            ),
            "Identifying assumption: " + pt_text(o) + cov_text + ".",
        ),
        (
            "anticipation",
            o["anticipation"],
            source("anticipation"),
            f"Units may respond up to {o['anticipation']} period(s) before "
            "treatment starts.",
        ),
        (
            "event_window",
            (
                f"all available, at most {EVENT_WINDOW_DEFAULT_SPAN} on each side"
                if window is None
                else tuple(window)
            ),
            source("event_window"),
            "Event times reported, and averaged over when the target is "
            "'event'. The window is part of the estimand.",
        ),
        (
            "weights",
            inp.weights or "none",
            source("weights"),
            (
                f"Units weighted by {inp.weights!r}: the target is a "
                "population-weighted effect."
                if inp.weights
                else "Each unit counts once: the target is the effect on the "
                "average treated unit, not on the average treated person."
            ),
        ),
        (
            "cluster",
            inp.cluster or inp.id,
            source("cluster"),
            "The level treatment is assigned at; what the standard error "
            "treats as independent.",
        ),
        (
            "estimator",
            o["estimator"],
            (
                "implied by absorb"
                if o.get("estimator_implied")
                else source("estimator")
            ),
            ESTIMATOR_LABEL[o["estimator"]] + ".",
        ),
        (
            "absorb",
            ", ".join(o["absorb"]) if o.get("absorb") else "none",
            source("absorb"),
            (
                "Fixed effects beyond unit and period: trends are compared "
                "within these cells only."
                if o.get("absorb")
                else "Unit and period fixed effects only."
            ),
        ),
        (
            "outcome",
            inp.y,
            "declared",
            "Parallel trends in levels and in logs are different "
            "assumptions; the scale of the outcome is part of the design.",
        ),
    ]
    table = pd.DataFrame(rows, columns=["choice", "value", "source", "meaning"])
    step.evidence = {"choices": table, "cohort_source": inp.cohort_source}

    owed = [
        name for name in ("target", "control_group", "cluster") if name not in explicit
    ]
    if o["control_group_implied"] and "control_group" in owed:
        owed.remove("control_group")
    if owed:
        step.status = "caution"
        step.finding = (
            "Left to defaults: "
            + ", ".join(
                f"{n}={table.set_index('choice').loc[n, 'value']!r}" for n in owed
            )
            + ". A choice made by omission is still a choice, and each of "
            "these moves the number."
        )
        step.action = (
            "State them in the call: sp.did_workflow(..., "
            + ", ".join(
                f"{n}={table.set_index('choice').loc[n, 'value']!r}" for n in owed
            )
            + ")."
        )
    else:
        step.status = "pass"
        step.finding = (
            f"Target: {ESTIMAND_TEXT[o['target']]}. Assumption: "
            f"{pt_text(o)}{cov_text}. Clustered on "
            f"{inp.cluster or inp.id!r}."
        )


# ===================================================================== #
# 2. Design
# ===================================================================== #
def run_design(wf: "DiDWorkflow", step: DiDStep) -> None:
    from .design_audit import did_cluster_diagnostics

    inp = wf._inputs
    o = wf._options
    facts = design_facts(inp, anticipation=o["anticipation"])
    wf._facts = facts

    clusters = did_cluster_diagnostics(
        inp.data,
        unit=inp.id,
        first_treat=inp.cohort,
        cluster=inp.cluster or inp.id,
        warn=False,
    )
    step.objects["cluster_diagnostics"] = clusters
    step.evidence = {
        k: facts[k]
        for k in (
            "n_obs",
            "n_units",
            "n_periods",
            "balanced",
            "n_incomplete_units",
            "n_duplicate_rows",
            "n_never_treated",
            "n_treated_after_sample",
            "n_always_treated",
            "n_treated_units",
            "n_cohorts",
            "cohorts",
        )
    }
    step.evidence.update(
        staggered=facts["n_cohorts"] > 1,
        n_reversal_units=inp.n_reversal_units,
        n_clusters=clusters.n_clusters,
        n_treated_clusters=clusters.n_treated_clusters,
        n_comparison_clusters=clusters.n_control_clusters,
        cluster_grade=clusters.verdict,
    )

    failures: List[str] = []
    flags: List[str] = []
    actions: List[str] = []

    if inp.n_reversal_units:
        failures.append(
            f"treatment switches back off in {inp.n_reversal_units} unit(s) "
            f"(e.g. {', '.join(inp.reversal_examples)}): adoption is not "
            "absorbing, and a first-treatment cohort does not describe it"
        )
        actions.append(
            "use sp.did_multiplegt_dyn or sp.panel_match, which take the "
            "treatment path"
        )
    if facts["n_duplicate_rows"]:
        failures.append(
            f"{facts['n_duplicate_rows']} row(s) repeat a (unit, period) pair"
        )
        actions.append("aggregate to one row per unit and period")
    if facts["n_cohorts"] == 0:
        failures.append(
            "no unit starts treatment inside the sample with a period "
            "observed before it"
        )
        actions.append("check the cohort column and the sample window")
    n_comparison = facts["n_comparison_never"]
    control = o["control_group"]
    if facts["n_cohorts"] >= 1:
        if control == "nevertreated" and n_comparison == 0:
            failures.append("there are no never-treated units to compare against")
            actions.append(
                "control_group='notyettreated'"
                if facts["n_cohorts"] > 1
                else "find a comparison group; with one adoption date and "
                "no untreated units there is no DiD"
            )
        elif control != "nevertreated" and n_comparison == 0:
            if facts["n_cohorts"] < 2:
                failures.append(
                    "every unit is treated at the same date: there is no "
                    "comparison group"
                )
                actions.append("this is not a DiD design; see sp.recommend")
            else:
                flags.append(
                    "no never-treated units: effects are identified only "
                    "until the last cohort is treated"
                )

    if not facts["balanced"] and not facts["n_duplicate_rows"]:
        flags.append(
            f"the panel is unbalanced ({facts['n_incomplete_units']} of "
            f"{facts['n_units']} units miss a period)"
        )
        actions.append(
            "decide how incomplete units enter; with estimator='cs' pass "
            "estimator_options={'allow_unbalanced_panel': True} to keep them"
        )
    if facts["n_always_treated"]:
        flags.append(
            f"{facts['n_always_treated']} unit(s) are already treated in the "
            "first period and carry no pre-period; the estimators drop them"
        )
    if facts["n_treated_after_sample"]:
        flags.append(
            f"{facts['n_treated_after_sample']} unit(s) are first treated "
            "after the sample ends and serve as never-treated"
        )
    if control == "nevertreated" and 0 < clusters.n_control_clusters < FEW_CLUSTERS:
        flags.append(
            f"only {clusters.n_control_clusters} never-treated cluster(s) "
            "carry every counterfactual"
        )
    if 0 < clusters.n_treated_clusters < FEW_CLUSTERS:
        flags.append(
            f"only {clusters.n_treated_clusters} treated cluster(s): "
            "cluster-robust intervals over-reject here"
        )
        actions.append("sp.did_few_treated for placebo-based inference")
    if clusters.verdict in ("below-evidence", "weakest-cell"):
        flags.append(
            f"{clusters.n_clusters} clusters ({clusters.verdict} of the "
            "Ulloa-Perez et al. grid, where every staggered estimator "
            "under-covered)"
        )
    cohorts = facts["cohorts"]
    if len(cohorts) and int(cohorts["n_pre_periods"].max()) - o["anticipation"] < 2:
        flags.append(
            "no cohort has two periods before treatment, so parallel "
            "trends cannot be probed with pre-treatment data"
        )

    window_note = _default_window(wf, step, cohorts)
    balance_note = _covariate_balance(wf, step) if not failures else ""

    head = (
        f"{facts['n_units']} units x {facts['n_periods']} periods; "
        f"{facts['n_cohorts']} treated cohort(s) with "
        f"{facts['n_treated_units']} units, {facts['n_never_treated']} "
        f"never treated; {clusters.n_clusters} clusters on "
        f"{clusters.cluster_column!r}."
    )
    if failures:
        step.status = "fail"
        step.finding = head + " Not estimable as declared: " + "; ".join(failures) + "."
    elif flags:
        step.status = "caution"
        step.finding = head + " Note: " + "; ".join(flags) + "."
    else:
        step.status = "pass"
        step.finding = head
    if window_note and not failures:
        step.finding += " " + window_note
    if balance_note:
        step.finding += " " + balance_note
    step.action = "; ".join(dict.fromkeys(actions))
    if failures:
        wf._blocked = "the design audit failed"


def _covariate_balance(wf: "DiDWorkflow", step: DiDStep) -> str:
    """Balance of the conditioning covariates, in levels and in changes.

    Baker et al. (2026, section 4.1) ask for this table before covariates
    enter: group means and normalised differences of the baseline levels
    and of the changes across the treatment date, with and without the
    weights. It describes the largest cohort against its comparison
    group (balance is a two-group statistic). Their benchmark of 0.25 is
    reported, not graded: the covariates are already conditioned on, and
    the table says what that conditioning has to absorb.
    """
    from .balance import did_balance

    inp = wf._inputs
    o = wf._options
    if not inp.covariates:
        return ""
    covariates = list(inp.covariates)
    control = (
        "nevertreated" if o["control_group"] == "nevertreated" else "notyettreated"
    )
    result = wf._optional(
        step,
        "did_balance",
        lambda: did_balance(
            inp.data,
            covariates,
            g=inp.cohort,
            t=inp.time,
            i=inp.id,
            weights=inp.weights,
            control_group=control,
        ),
    )
    if result is None:
        return ""
    step.objects["balance"] = result
    step.evidence["balance"] = result.table
    step.evidence["balance_flagged"] = list(result.flagged)
    worst = float(result.max_abs_norm_diff)
    if result.flagged:
        return (
            f"Before conditioning, {', '.join(result.flagged)} differ"
            f"{'s' if len(result.flagged) == 1 else ''} between the largest "
            "cohort and its comparison group by more than "
            f"{result.threshold:g} standard deviations in levels or in "
            f"changes (largest {worst:.2f}); that is what conditioning has "
            "to absorb. wf.step('design').evidence['balance'] has the table."
        )
    return (
        "The covariates are balanced between the largest cohort and its "
        f"comparison group (largest normalised difference {worst:.2f}, in "
        "levels or in changes)."
    )


def _default_window(wf: "DiDWorkflow", step: DiDStep, cohorts: Any) -> str:
    """Bound the reported event times when ``event_window`` was not given.

    A long panel offers more event times than a report can use: the joint
    pre-test spreads its power over every lead, and the power calculation
    of step 6 and the sensitivity sets of step 7 become intractable. The
    default keeps the ``EVENT_WINDOW_DEFAULT_SPAN`` event times nearest to
    treatment on each side. A panel with fewer is reported in full, and an
    explicit ``event_window`` is never touched.
    """
    o = wf._options
    if "event_window" in wf._explicit or not len(cohorts):
        return ""
    leads = int(cohorts["n_pre_periods"].max())
    lags = int(cohorts["n_post_periods"].max()) - 1
    span = EVENT_WINDOW_DEFAULT_SPAN
    if leads <= span and lags <= span:
        return ""
    window = (-min(leads, span), min(lags, span))
    o["event_window"] = window
    step.evidence["event_window_default"] = {
        "window": window,
        "available": (-leads, lags),
    }
    return (
        f"Event times run from {-leads} to {lags}; with no event_window "
        f"given, steps 5 to 7 report {window[0]} to {window[1]}. Pass "
        "event_window= to report another range."
    )


# ===================================================================== #
# 3. TWFE diagnosis
# ===================================================================== #
def run_twfe(wf: "DiDWorkflow", step: DiDStep) -> None:
    from .bacon import bacon_decomposition

    inp = wf._inputs
    facts = facts_of(wf)
    if facts["n_cohorts"] < 2:
        step.status = "info"
        step.finding = (
            "One adoption date: the two-way fixed effects coefficient has no "
            "already-treated comparisons to diagnose."
        )
        return

    data = inp.data
    periods = facts["periods"]
    name = "_did_workflow_d"
    while name in data.columns:
        name += "_"
    frame = data[[inp.y, inp.id, inp.time, inp.cohort]].copy()
    g = frame[inp.cohort]
    if pd.api.types.is_numeric_dtype(g):
        gnum = pd.to_numeric(g, errors="coerce")
        treated_unit = gnum.notna() & np.isfinite(gnum) & (gnum > 0)
        frame[name] = (treated_unit & (frame[inp.time] >= gnum)).astype(int)
    else:
        frame[name] = (g.notna() & (frame[inp.time] >= g)).astype(int)
    if inp.weights:
        frame[inp.weights] = data[inp.weights]

    result = wf._optional(
        step,
        "bacon_decomposition",
        lambda: bacon_decomposition(
            frame,
            y=inp.y,
            treat=name,
            time=inp.time,
            id=inp.id,
            weights=inp.weights,
            balance="error" if facts["balanced"] else "drop_units",
        ),
    )
    if result is None:
        step.status = "skipped"
        step.finding = (
            "The Goodman-Bacon decomposition could not be computed on this "
            "panel; see the warning for the reason."
        )
        return
    step.objects["bacon"] = result
    forbidden = float(result["already_treated_control_weight_share"])
    beta = float(result["beta_twfe"])
    wf._twfe = beta
    step.evidence = {
        "beta_twfe": beta,
        "forbidden_weight_share": forbidden,
        "n_comparisons": int(result["n_comparisons"]),
        "by_type": result["summary"],
        "n_periods_used": int(len(periods)),
    }
    lead = (
        f"TWFE coefficient {fmt(beta)}; {pct(forbidden)} of its weight is "
        "on comparisons that use already-treated units as controls"
    )
    if forbidden > 1e-12:
        step.status = "caution"
        step.finding = (
            lead + ". With effects that differ across cohorts or grow over "
            "time those comparisons can carry the wrong sign, so TWFE is "
            "reported here as a diagnostic, not as the estimate."
        )
        step.action = "Report the heterogeneity-robust estimate of step 4."
    else:
        step.status = "pass"
        step.finding = lead + "."
