"""What a recommendation does *not* establish.

``sp.recommend`` picks an estimator from the shape of the data and the
declared design. Choosing a tool is not identifying an effect: parallel
trends, exclusion restrictions, no unmeasured confounding and no
manipulation are claims about the world that no data set can verify. This
module turns the design into a structured brief -- the assumptions that
carry the causal interpretation, the checks that bear on (but do not prove)
them, and the questions an analyst or agent should put to the person who
knows how the data were generated -- so that a recommendation is never read
as a finding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: design -> (untestable assumptions, [(check, function)], questions, claim)
_BRIEFS: Dict[str, Dict[str, Any]] = {
    "rct": {
        "untestable": [
            "Treatment was randomly assigned as designed (no manipulation of "
            "assignment)",
            "No differential attrition or non-compliance beyond what is modelled",
            "SUTVA: no interference between units",
        ],
        "checks": [
            ("covariate balance across arms", "balance_check"),
            ("randomisation inference for the sharp null", "ri_test"),
        ],
        "questions": [
            "Was treatment assigned by a random mechanism the researcher "
            "controlled? At what level (individual, cluster)?",
            "Did everyone assigned receive treatment, and was attrition " "balanced?",
        ],
        "claim": "causal if assignment was random",
    },
    "observational": {
        "untestable": [
            "No unmeasured confounding: treatment is as good as random given "
            "the covariates",
            "Covariates are measured before treatment (no post-treatment " "controls)",
            "Overlap: every covariate profile has treated and untreated units",
        ],
        "checks": [
            ("overlap / propensity-score support", "overlap_plot"),
            ("covariate balance after adjustment", "balance_check"),
            ("sensitivity to an unmeasured confounder", "sensemakr"),
            ("E-value", "evalue"),
            ("coefficient-stability bounds", "oster_bounds"),
        ],
        "questions": [
            "Why did some units get treated and not others? Is every driver "
            "of that choice that also affects the outcome measured?",
            "Which covariates were fixed before treatment?",
            "Is there a design (a policy change over time, a threshold, an "
            "instrument) that would not rely on selection on observables?",
        ],
        "claim": "causal only under selection on observables",
    },
    "did": {
        "untestable": [
            "Parallel trends in the post-treatment periods (pre-period tests "
            "cannot establish it)",
            "No anticipation before adoption",
            "No concurrent shock specific to the treated group",
            "Stable group composition (repeated cross-sections)",
        ],
        "checks": [
            ("pre-trend test (a non-rejection is not evidence)", "pretrends_test"),
            ("power of the pre-trend test", "pretrends_power"),
            ("robustness to bounded violations of parallel trends", "honest_did"),
            ("TWFE weights under staggered adoption", "bacon_decomposition"),
        ],
        "questions": [
            "When exactly did treatment start for each unit, and could units "
            "anticipate it?",
            "Did anything else change for the treated group at the same time?",
            "Why were these units treated -- could their outcomes have been "
            "trending differently anyway?",
        ],
        "claim": "causal under parallel trends",
    },
    "iv": {
        "untestable": [
            "Exclusion: the instrument affects the outcome only through the "
            "treatment",
            "Independence: the instrument is as good as randomly assigned",
            "Monotonicity (for a LATE interpretation)",
        ],
        "checks": [
            ("first-stage strength", "effective_f_test"),
            ("weak-instrument-robust inference", "anderson_rubin_ci"),
            ("instrument validity bounds under partial exclusion", "iv_bounds"),
        ],
        "questions": [
            "By what channel other than the treatment could the instrument "
            "move the outcome?",
            "Who are the compliers, and is their effect the one of interest?",
        ],
        "claim": "causal for compliers under exclusion and independence",
    },
    "rd": {
        "untestable": [
            "Continuity of potential outcomes at the cutoff",
            "No other policy switches at the same threshold",
        ],
        "checks": [
            ("manipulation of the running variable", "rddensity"),
            ("covariate balance at the cutoff", "rdbalance"),
            ("placebo cutoffs", "rdplacebo"),
            ("bandwidth sensitivity", "rdbwsensitivity"),
        ],
        "questions": [
            "Can units influence their running-variable value precisely?",
            "Does anything else change at the same cutoff?",
        ],
        "claim": "causal at the cutoff under continuity",
    },
    "synth": {
        "untestable": [
            "The synthetic unit would have tracked the treated unit without "
            "treatment",
            "No spillovers to donor units; no anticipation",
        ],
        "checks": [
            ("leave-one-out donor sensitivity", "synth_loo"),
            ("in-time placebo", "synth_time_placebo"),
            ("donor pool sensitivity", "synth_donor_sensitivity"),
        ],
        "questions": [
            "Were donor units exposed to the policy or to its spillovers?",
            "Did the treated unit face another shock at adoption?",
        ],
        "claim": "causal if the synthetic control is a valid counterfactual",
    },
    "panel": {
        "untestable": [
            "Strict exogeneity: time-varying unobservables are uncorrelated "
            "with the regressors after fixed effects",
        ],
        "checks": [("fixed vs random effects", "hausman_test")],
        "questions": [
            "What varies over time within units that could drive both the "
            "regressor and the outcome?",
        ],
        "claim": "causal only under strict exogeneity",
    },
}
_BRIEFS["ddd"] = {
    **_BRIEFS["did"],
    "untestable": [
        "Parallel trends in the *difference* between eligible and ineligible " "groups",
        "No shock specific to the eligible subgroup of the treated group",
    ],
    "claim": "causal under the triple-difference parallel-trends assumption",
}
_BRIEFS["bartik"] = {
    **_BRIEFS["iv"],
    "untestable": [
        "Exogenous exposure shares (or exogenous shocks) conditional on " "controls",
        "Shocks affect the outcome only through the endogenous regressor",
    ],
}


def identification_brief(
    design: str,
    declared: bool,
    treatment: Optional[str],
    available: Optional[set] = None,
) -> Dict[str, Any]:
    """Structured statement of what the recommendation leaves unproven.

    ``available`` restricts the named check functions to those that exist.
    """
    if treatment is None or design == "cross-section":
        return {
            "design": design,
            "design_source": "declared" if declared else "detected_from_data_shape",
            "claim": "descriptive_only",
            "untestable_assumptions": [],
            "checks": [],
            "questions": [
                "Which variable is the treatment, and how was it assigned? "
                "Without that, estimates are associations, not effects."
            ],
            "note": "No treatment or design information: report associations only.",
        }
    spec = _BRIEFS.get(design)
    if spec is None:
        return {
            "design": design,
            "design_source": "declared" if declared else "detected_from_data_shape",
            "claim": "not_assessed",
            "untestable_assumptions": [],
            "checks": [],
            "questions": [],
            "note": "No identification brief for this design; see the "
            "recommended function's agent card (sp.agent_card(name)).",
        }
    checks: List[Dict[str, str]] = [
        {"check": what, "function": f"sp.{fn}"}
        for what, fn in spec["checks"]
        if available is None or fn in available
    ]
    questions = list(spec["questions"])
    if not declared:
        questions.insert(
            0,
            f"The design '{design}' was inferred from the data's shape, not "
            "stated. Is that how treatment was actually assigned?",
        )
    return {
        "design": design,
        "design_source": "declared" if declared else "detected_from_data_shape",
        "claim": spec["claim"],
        "untestable_assumptions": list(spec["untestable"]),
        "checks": checks,
        "questions": questions,
        "note": (
            "The recommendation chooses an estimator; it does not establish "
            "these assumptions. Checks can reveal violations but cannot prove "
            "that the assumptions hold."
        ),
    }
