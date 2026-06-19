"""Curated Tier-A agent-native metadata overlay (machine-generated input,
human-verified content).

Generated for the v1.16 agent-native sprint to lift assumptions /
failure_modes / alternatives / typical_n_min coverage across the
certified + validated tier.  Same extend-missing semantics as
``registry._AGENT_CARD_SEED_METADATA`` (curated specs in registry.py
always win); applied by ``registry._apply_agent_card_seeds``.

Every ``alternative`` here was validated to resolve to a real registered
function / sp attribute / MCP tool, and every ``exception`` to a real
exception class, by the same checks as
``tests/test_agent_native_contract.py``.  Do not hand-edit dangling
pointers in; run the contract suite after changes.
"""

from __future__ import annotations

from typing import Any, Dict

EXTRA_AGENT_CARDS: Dict[str, Dict[str, Any]] = {
    "OPEResult": {
        "assumptions": [
            "Sequential ignorability / no unmeasured confounding " "in the logged data",
            "Positivity: logging policy assigns positive "
            "probability to every action the target policy can "
            "take",
            "Logging propensities are correctly specified "
            "(IPS/SNIPS/DR weights are valid)",
        ],
        "pre_conditions": [
            "Produced by an off-policy-evaluation estimator "
            "(ips / snips / doubly_robust / direct_method) — "
            "not constructed directly",
            "Logged (action, reward, propensity) tuples "
            "underlie the fitted estimator",
        ],
        "failure_modes": [
            {
                "symptom": "Accessing a field that the chosen "
                "estimator did not populate",
                "exception": "KeyError",
                "remedy": "Read the estimator's documented "
                "OPEResult fields; use the "
                "policy_learning OPEResult subclass "
                "when you need the estimator "
                "attribute",
                "alternative": "policy_value",
            },
            {
                "symptom": "Huge variance / unstable value "
                "estimate from near-zero logging "
                "propensities",
                "exception": "(none — informational)",
                "remedy": "Switch to a self-normalised or "
                "doubly-robust estimator and inspect "
                "effective sample size",
                "alternative": "aipw",
            },
        ],
        "alternatives": ["policy_value", "aipw", "ipw", "sharp_ope_unobserved"],
        "typical_n_min": 500,
    },
    "acf": {
        "assumptions": [
            "Scalar Hicks-neutral productivity omega evolves as a "
            "first-order Markov process",
            "Free inputs (labor) chosen contemporaneously with "
            "the proxy, so labor elasticity is identified only in "
            "stage 2 (not the stage-1 polynomial)",
            "Free inputs are instrumented with their lags; state "
            "inputs (capital) are predetermined and used at "
            "contemporaneous level",
        ],
        "pre_conditions": [
            "Firm-year panel with log output, free input "
            "(labor), state input (capital), and a materials "
            "proxy",
            "At least two consecutive periods per firm so "
            "lagged labor exists for the stage-2 instruments",
        ],
        "failure_modes": [
            {
                "symptom": "Most firms have only a single "
                "observed year, so no lagged labor "
                "can be formed for the stage-2 "
                "moments",
                "exception": "DataInsufficient",
                "remedy": "Restrict to firms observed in at "
                "least two consecutive periods or "
                "supply a longer panel.",
                "alternative": "wooldridge_prod",
            },
            {
                "symptom": "Stage-2 GMM/instrument matrix is "
                "near-singular because free inputs "
                "barely move over time",
                "exception": "IdentificationFailure",
                "remedy": "Add cross-sectional variation in "
                "inputs or lower polynomial_degree to "
                "avoid collinear instruments.",
                "alternative": "levinsohn_petrin",
            },
        ],
        "alternatives": ["levinsohn_petrin", "wooldridge_prod", "olley_pakes", "gmm"],
        "typical_n_min": 500,
    },
    "ackerberg_caves_frazer": {
        "assumptions": [
            "Scalar unobserved productivity following a "
            "first-order Markov law of motion",
            "Functional-dependence fix: all input coefficients "
            "identified in stage 2, with free inputs instrumented "
            "by their lags and capital at contemporaneous level",
            "Timing: state inputs predetermined, free inputs set "
            "jointly with the proxy",
        ],
        "pre_conditions": [
            "Firm-year panel with log output, free input(s), "
            "state input(s), and an intermediate-input proxy",
            "Two or more consecutive periods per firm so "
            "lagged free inputs are available",
        ],
        "failure_modes": [
            {
                "symptom": "Lagged labor unavailable because "
                "firms appear in non-contiguous "
                "years",
                "exception": "DataInsufficient",
                "remedy": "Keep only firm-spells with "
                "consecutive years before fitting.",
                "alternative": "levinsohn_petrin",
            },
            {
                "symptom": "Stage-2 moment conditions fail to "
                "pin down coefficients (weak lag "
                "instruments)",
                "exception": "IdentificationFailure",
                "remedy": "Reduce "
                "productivity_degree/polynomial_degree "
                "or use a proxy with more independent "
                "variation.",
                "alternative": "wooldridge_prod",
            },
        ],
        "alternatives": ["levinsohn_petrin", "olley_pakes", "wooldridge_prod", "gmm"],
        "typical_n_min": 500,
    },
    "attributable_risk": {
        "assumptions": [
            "The supplied or estimated RR is causal and "
            "unconfounded, so AF_exposed = (RR-1)/RR validly "
            "attributes excess risk to exposure.",
            "Exposure prevalence P_e is measured in the target "
            "population to which the PAF is generalized.",
            "The delta-method CI on log(1-PAF) assumes "
            "large-sample normality, which degrades when PAF is "
            "near 0 or 1.",
        ],
        "pre_conditions": [
            "A risk ratio (RR > 0) and an exposure prevalence "
            "P_e in [0,1], or 2x2 counts from which RR is "
            "derived.",
            "RR and P_e refer to the same population and " "outcome definition.",
        ],
        "failure_modes": [
            {
                "symptom": "RR < 1 (protective exposure) yields "
                "a negative attributable fraction "
                "that is misinterpreted as "
                "preventable burden.",
                "exception": "(none — informational)",
                "remedy": "For protective exposures report a "
                "prevented fraction instead, or "
                "recode so the harmful category is "
                "'exposed'.",
                "alternative": "relative_risk",
            },
            {
                "symptom": "P_e outside [0,1] or RR <= 0 passed " "in.",
                "exception": "ValueError",
                "remedy": "Clamp exposure prevalence to a "
                "proportion and pass a strictly "
                "positive RR.",
                "alternative": "relative_risk",
            },
        ],
        "alternatives": [
            "relative_risk",
            "risk_difference",
            "odds_ratio",
            "mantel_haenszel",
        ],
        "typical_n_min": 100,
    },
    "bootstrap": {
        "assumptions": [
            "Observations (or clusters/blocks) are exchangeable "
            "under the resampling scheme used; cluster= resamples "
            "whole clusters, block= preserves within-block "
            "ordering for serial dependence.",
            "The statistic callable is a smooth, well-defined "
            "functional of the data so its sampling distribution "
            "is approximable by resampling.",
            "BCa intervals further assume an estimable "
            "bias/acceleration; percentile/normal CIs assume "
            "approximate pivotality of the resampled "
            "distribution.",
        ],
        "pre_conditions": [
            "statistic is a deterministic callable returning a "
            "finite scalar on any resampled DataFrame.",
            "Enough independent resampling units (rows, "
            "clusters, or blocks) to populate n_boot "
            "replicates.",
            "Specify cluster= or block= when data are "
            "clustered/serially dependent rather than iid.",
        ],
        "failure_modes": [
            {
                "symptom": "Few clusters yields "
                "anti-conservative CIs / wildly "
                "unstable bootstrap distribution",
                "exception": "DataInsufficient",
                "remedy": "Use a wild cluster bootstrap with "
                "Rademacher weights when the number "
                "of clusters is small (<~40).",
                "alternative": "wild_cluster_bootstrap",
            },
            {
                "symptom": "Statistic raises or returns NaN on "
                "a resample (e.g., a level drops "
                "out), aborting the loop",
                "exception": "RuntimeError",
                "remedy": "Make the statistic robust to missing "
                "categories/empty strata or wrap to "
                "skip degenerate resamples.",
                "alternative": "",
            },
        ],
        "alternatives": ["wild_cluster_bootstrap", "did", "iv"],
        "typical_n_min": 50,
    },
    "bradford_hill": {
        "assumptions": [
            "Viewpoint scores in [0,1] reflect external expert "
            "judgement; the function aggregates but does not "
            "itself verify causal evidence.",
            "Unscored viewpoints (None) are missing-at-random "
            "with respect to the aggregate, since they drop from "
            "both numerator and denominator.",
            "Temporality and the other eight viewpoints are "
            "treated as exchangeable contributions to a mean "
            "score (no formal weighting of necessity).",
        ],
        "pre_conditions": [
            "Each provided viewpoint score is a number in " "[0,1] or None.",
            "At least one viewpoint is scored (non-None) so "
            "the denominator is positive.",
        ],
        "failure_modes": [
            {
                "symptom": "All viewpoints left as None gives "
                "an empty/undefined aggregate score.",
                "exception": "ValueError",
                "remedy": "Score at least one Bradford-Hill "
                "viewpoint before calling.",
                "alternative": "",
            },
            {
                "symptom": "A score outside [0,1] (e.g. a raw "
                "effect size) is passed for a "
                "viewpoint.",
                "exception": "ValueError",
                "remedy": "Normalize each viewpoint to the "
                "0/0.5/1 evidence scale before "
                "passing.",
                "alternative": "",
            },
        ],
        "alternatives": ["relative_risk", "odds_ratio", "attributable_risk"],
        "typical_n_min": 1,
    },
    "breslow_day_test": {
        "assumptions": [
            "Under H0 the true odds ratio is constant across all "
            "K strata (homogeneity), against which the test "
            "contrasts observed vs MH-expected cells.",
            "The chi-square reference distribution with K-1 df is "
            "asymptotic and requires adequately large "
            "stratum-specific cell counts.",
            "Tarone's correction is applied so the statistic is "
            "consistent with the MH common-OR estimator used to "
            "compute expected cells.",
        ],
        "pre_conditions": [
            "A (K,2,2) array of stratum 2x2 tables with K >= 2 " "strata.",
            "Each stratum has non-degenerate margins (both "
            "exposure and outcome levels present).",
        ],
        "failure_modes": [
            {
                "symptom": "Sparse strata with zero or "
                "near-zero cells make expected "
                "counts unstable and inflate the "
                "statistic.",
                "exception": "AssumptionWarning",
                "remedy": "Collapse or drop sparse strata, or "
                "rely on the MH estimator with its "
                "OR-homogeneity caveat noted.",
                "alternative": "mantel_haenszel",
            },
            {
                "symptom": "Only one stratum supplied so there "
                "is nothing to test for homogeneity.",
                "exception": "ValueError",
                "remedy": "Provide at least two strata, or "
                "compute a single-table odds ratio.",
                "alternative": "odds_ratio",
            },
        ],
        "alternatives": ["mantel_haenszel", "odds_ratio"],
        "typical_n_min": 40,
    },
    "bridge": {
        "assumptions": [
            "The two estimation paths being bridged target the "
            "same causal estimand under the chosen bridging "
            "theorem (e.g., DID = synthetic control for "
            "kind='did_sc').",
            "Each path's own identification holds (parallel "
            "trends / SC convex-hull / proximal completeness, "
            "depending on kind), so a non-rejected agreement test "
            "is meaningful.",
            "The doubly-robust combined estimate is consistent if "
            "at least one of the two bridged paths is correctly "
            "specified.",
        ],
        "pre_conditions": [
            "kind is one of the supported bridges and the "
            "matching per-bridge kwargs are supplied.",
            "Panel/data shape matches the chosen bridge (e.g., "
            "did_sc needs unit, time, treated_unit, "
            "treatment_time).",
            "Both paths are estimable on the same sample so "
            "the agreement test is well-defined.",
        ],
        "failure_modes": [
            {
                "symptom": "Agreement test rejects: the two "
                "paths give materially different "
                "estimates",
                "exception": "AssumptionViolation",
                "remedy": "Treat divergence as evidence one "
                "bridging assumption fails and "
                "diagnose each path separately rather "
                "than trusting the DR combination.",
                "alternative": "did",
            },
            {
                "symptom": "Unknown kind or missing " "bridge-specific kwargs",
                "exception": "ValueError",
                "remedy": "Pass a supported kind and the exact "
                "keyword arguments documented for "
                "that bridge.",
                "alternative": "",
            },
        ],
        "alternatives": ["did", "synth", "proximal"],
        "typical_n_min": 100,
    },
    "cevae": {
        "assumptions": [
            "Latent-confounder model: a low-dimensional latent "
            "generates the observed proxies, treatment, and "
            "outcome (Louizos et al. 2017)",
            "Proxy sufficiency — observed covariates carry enough "
            "signal to recover the confounder",
            "Correct VAE specification (treatment/outcome "
            "likelihoods and prior) and adequate optimisation",
        ],
        "pre_conditions": [
            "torch installed (neural extra) for the " "variational autoencoder backend",
            "Individual-level data with treatment, outcome, " "and proxy covariates",
        ],
        "failure_modes": [
            {
                "symptom": "torch not installed when invoking "
                "the neural CEVAE backend",
                "exception": "ImportError",
                "remedy": "Install the neural extra: pip " "install 'StatsPAI[neural]'",
                "alternative": "dragonnet",
            },
            {
                "symptom": "Variational lower bound fails to "
                "converge or posterior collapses, "
                "yielding biased effect estimates",
                "exception": "ConvergenceFailure",
                "remedy": "Increase epochs, lower the learning "
                "rate, or add more proxy covariates; "
                "validate against a regression "
                "baseline",
                "alternative": "aipw",
            },
        ],
        "alternatives": ["dragonnet", "aipw", "ipw", "dl_propensity_score"],
        "typical_n_min": 1000,
    },
    "cite": {
        "assumptions": [
            "The result object exposes either params/std_errors "
            "(econometric) or estimate/se (causal) so a "
            "coefficient cell can be formatted.",
            "If term is given it matches an existing coefficient "
            "name; otherwise the headline estimand or first "
            "params row is used.",
            "star_levels and second_row follow the same star/SE "
            "convention as sp.regtable for cross-report "
            "consistency.",
        ],
        "pre_conditions": [
            "Exactly one fitted StatsPAI result object with "
            "accessible point estimate and standard error"
        ],
        "failure_modes": [
            {
                "symptom": "Requested term is not among the "
                "result's coefficient names",
                "exception": "KeyError",
                "remedy": "Pass a term that appears in "
                "result.params (or omit term to use "
                "the default estimand).",
                "alternative": "regtable",
            },
            {
                "symptom": "Result object lacks both "
                "params/std_errors and estimate/se "
                "attributes",
                "exception": "AttributeError",
                "remedy": "Pass a fitted StatsPAI result rather "
                "than raw data or a dict.",
                "alternative": "regtable",
            },
            {
                "symptom": "Unsupported output or second_row " "keyword passed",
                "exception": "ValueError",
                "remedy": "Use output in "
                "{text,latex,markdown,html} and "
                "second_row in {se,t,p,ci,none}.",
                "alternative": "",
            },
        ],
        "alternatives": ["regtable", "modelsummary", "outreg2"],
        "typical_n_min": 1,
    },
    "cluster_cross_interference": {
        "assumptions": [
            "Partial interference: spillovers operate within "
            "clusters but not across cluster boundaries",
            "Exposure mapping correctly captured by "
            "neighbour_treat_share (user-precomputed share of "
            "treated neighbours)",
            "Cluster-level treatment is binary and randomized " "(cluster RCT)",
            "Linear additivity of own-treatment and " "neighbour-exposure effects",
        ],
        "pre_conditions": [
            "Cluster identifier column plus individual-level " "outcome",
            "Cluster-level binary treatment column",
            "Precomputed neighbour_treat_share column from "
            "spatial/network adjacency",
        ],
        "failure_modes": [
            {
                "symptom": "neighbour_treat_share missing or " "not a valid 0-1 share",
                "exception": "ValueError",
                "remedy": "Precompute the treated-neighbour "
                "share per cluster from your "
                "adjacency matrix before calling.",
                "alternative": "network_exposure",
            },
            {
                "symptom": "Too few clusters for cluster-robust " "inference",
                "exception": "DataInsufficient",
                "remedy": "Increase the number of clusters or "
                "use a design with weaker cluster "
                "requirements.",
                "alternative": "interference",
            },
        ],
        "alternatives": [
            "inward_outward_spillover",
            "network_hte",
            "interference",
            "spillover",
        ],
        "typical_n_min": 30,
    },
    "cohen_kappa": {
        "assumptions": [
            "The two raters classify the same units independently "
            "into a common, fixed set of categories.",
            "Chance agreement is modeled from the product of each "
            "rater's marginal category frequencies "
            "(fixed-marginals assumption).",
            "Linear/quadratic weights presume the category scale "
            "is genuinely ordinal with equally interpretable "
            "distances.",
        ],
        "pre_conditions": [
            "rater_a and rater_b are equal-length sequences of " "category labels.",
            "Both raters use the same category label set; for "
            "weighted kappa the categories are ordered.",
        ],
        "failure_modes": [
            {
                "symptom": "Highly imbalanced marginals (one "
                "category dominant) yield low kappa "
                "despite high raw agreement (the "
                "kappa paradox).",
                "exception": "(none — informational)",
                "remedy": "Report observed agreement alongside "
                "kappa and inspect the confusion "
                "matrix.",
                "alternative": "sensitivity_specificity",
            },
            {
                "symptom": "rater_a and rater_b have different " "lengths.",
                "exception": "ValueError",
                "remedy": "Align the two rating vectors to the "
                "same units before calling.",
                "alternative": "",
            },
        ],
        "alternatives": ["sensitivity_specificity", "roc_curve"],
        "typical_n_min": 30,
    },
    "cohort_anchored_event_study": {
        "assumptions": [
            "Parallel trends hold within each treatment cohort "
            "relative to never-treated units (cohort-anchored, "
            "not pooled TWFE)",
            "No anticipation: outcomes in pre-event periods "
            "unaffected by future treatment",
            "Treatment is an absorbing first-treatment event "
            "(staggered adoption, treat encodes first period, 0 = "
            "never-treated)",
        ],
        "pre_conditions": [
            "Long-format balanced/unbalanced panel with id and " "time columns",
            "treat column gives first-treatment period per "
            "unit (0 = never-treated), with at least one "
            "never-treated cohort",
            "Enough pre/post periods to fill the requested "
            "leads/lags event-time window",
        ],
        "failure_modes": [
            {
                "symptom": "All units treated in same period / "
                "no never-treated cohort to anchor "
                "against",
                "exception": "IdentificationFailure",
                "remedy": "Supply a never-treated comparison "
                "group or switch to a not-yet-treated "
                "control design",
                "alternative": "callaway_santanna",
            },
            {
                "symptom": "Requested leads/lags exceed "
                "available pre/post periods so "
                "event-time cells are empty",
                "exception": "DataInsufficient",
                "remedy": "Shrink the leads/lags window to the "
                "periods actually observed around "
                "each cohort's treatment time",
                "alternative": "event_study",
            },
        ],
        "alternatives": [
            "design_robust_event_study",
            "callaway_santanna",
            "sun_abraham",
            "honest_did",
        ],
        "typical_n_min": 200,
    },
    "collect": {
        "assumptions": [
            "Models added via add_regression share a comparable "
            "coefficient namespace for side-by-side columns.",
            "Output format is inferred from the save() filename "
            "extension (.docx/.xlsx/.tex).",
            "Items are accumulated incrementally (Stata 15 "
            "collect-style) before a single export.",
        ],
        "pre_conditions": [
            "A title/name for the collection; fitted results "
            "or a DataFrame are added subsequently via add_* "
            "methods"
        ],
        "failure_modes": [
            {
                "symptom": "save() called with an unrecognized "
                "or missing file extension",
                "exception": "ValueError",
                "remedy": "Use a filename ending in .docx, "
                ".xlsx, or .tex (or pass an explicit "
                "format).",
                "alternative": "regtable",
            },
            {
                "symptom": "Objects added to the collection are "
                "not fitted result objects",
                "exception": "AttributeError",
                "remedy": "Add StatsPAI result objects to "
                "add_regression and a DataFrame to "
                "add_summary.",
                "alternative": "paper_tables",
            },
            {
                "symptom": "Word/Excel export requested but the "
                "optional writer backend is missing",
                "exception": "ImportError",
                "remedy": "Install the export extra "
                "(python-docx / openpyxl) or export "
                "to LaTeX instead.",
                "alternative": "regtable",
            },
        ],
        "alternatives": ["regtable", "paper_tables", "modelsummary"],
        "typical_n_min": 1,
    },
    "conformal_continuous": {
        "assumptions": [
            "Exchangeability of calibration and test rows "
            "(split-conformal coverage guarantee)",
            "Continuous-treatment ignorability: the outcome "
            "regression captures confounding by covariates",
            "Positivity / overlap across the dose grid for the "
            "queried treatment levels",
        ],
        "pre_conditions": [
            "Training sample with continuous treatment and "
            "outcome y plus covariates",
            "Calibration split (calibration_frac) held out for " "conformal coverage",
            "test_data rows (and optional dose_grid) to " "predict bands on",
        ],
        "failure_modes": [
            {
                "symptom": "Calibration fold too small, so the "
                "empirical (1-alpha) quantile is "
                "undefined or unstable",
                "exception": "DataInsufficient",
                "remedy": "Increase sample size or raise "
                "calibration_frac so the calibration "
                "set spans the dose range",
                "alternative": "conformal",
            },
            {
                "symptom": "Bands overcover or undercover "
                "because the dose-response estimator "
                "is misspecified",
                "exception": "(none — informational)",
                "remedy": "Pass a better sklearn-style "
                "estimator for the outcome regression "
                "or widen the dose_grid spacing",
                "alternative": "conformal",
            },
        ],
        "alternatives": ["conformal", "conformal_fair_ite"],
        "typical_n_min": 400,
    },
    "conformal_fair_ite": {
        "assumptions": [
            "Exchangeability for conformal coverage, applied "
            "within protected-group strata",
            "Counterfactual fairness: protected attribute "
            "excluded from the outcome regression (used only for "
            "stratified calibration)",
            "ITE ignorability / overlap so the treated and "
            "control nuisances are identified",
        ],
        "pre_conditions": [
            "DataFrame with y, treat, predictive covariates, "
            "and a categorical protected column",
            "Calibration set per protected stratum for "
            "group-wise conformal coverage",
            "Optional test_data to emit fair ITE intervals on",
        ],
        "failure_modes": [
            {
                "symptom": "A protected stratum has too few "
                "calibration rows for valid "
                "group-wise quantiles",
                "exception": "DataInsufficient",
                "remedy": "Collapse sparse protected categories "
                "or pool strata before calibration",
                "alternative": "conformal",
            },
            {
                "symptom": "protected column accidentally leaks "
                "into covariates, breaking "
                "counterfactual fairness",
                "exception": "ValueError",
                "remedy": "Remove the protected attribute from "
                "covariates; pass it only via the "
                "protected argument",
                "alternative": "conformal_continuous",
            },
        ],
        "alternatives": ["conformal", "conformal_continuous"],
        "typical_n_min": 500,
    },
    "counterfactual_policy_optimization": {
        "assumptions": [
            "One-step linear-Gaussian SCM holds (noise inversion "
            "identifies counterfactual reward)",
            "Additive Gaussian noise is correctly specified",
            "Fixing state and changing action uniquely determines "
            "the new reward (structural invariance)",
        ],
        "pre_conditions": [
            "One row per trajectory with numeric state, " "action, and reward columns",
            "A target_policy callable mapping state -> " "proposed action",
        ],
        "failure_modes": [
            {
                "symptom": "Non-numeric or missing " "state/action/reward columns",
                "exception": "ValueError",
                "remedy": "Coerce state/action/reward to "
                "numeric and drop NaNs before fitting",
                "alternative": "policy_value",
            },
            {
                "symptom": "Proposed policy extrapolates far "
                "outside the observed action range, "
                "so linear-SCM predictions are "
                "unreliable",
                "exception": "(none — informational)",
                "remedy": "Restrict target_policy to the "
                "observed action support or use a "
                "nonparametric policy learner",
                "alternative": "causal_policy_forest",
            },
        ],
        "alternatives": ["causal_policy_forest", "policy_tree", "policy_value"],
        "typical_n_min": 200,
    },
    "demographic_parity": {
        "assumptions": [
            "Fairness defined as equal positive-prediction rate "
            "across protected groups",
            "Predictions are binary (0/1) classifier outputs",
            "Label-blind criterion: ground-truth outcomes are " "intentionally ignored",
            "Protected attribute may be multi-valued but groups " "are well-populated",
        ],
        "pre_conditions": ["Binary predictions column", "Protected-attribute column"],
        "failure_modes": [
            {
                "symptom": "Predictions not binary 0/1",
                "exception": "ValueError",
                "remedy": "Threshold continuous scores to 0/1 "
                "before computing the parity gap.",
                "alternative": "fairness_audit",
            },
            {
                "symptom": "A protected group has too few rows "
                "for a stable rate estimate",
                "exception": "DataInsufficient",
                "remedy": "Merge sparse groups or collect more " "data per group.",
                "alternative": "fairness_audit",
            },
        ],
        "alternatives": ["equalized_odds", "fairness_audit", "counterfactual_fairness"],
        "typical_n_min": 100,
    },
    "design_robust_event_study": {
        "assumptions": [
            "Parallel trends across cohorts; treatment effects "
            "may be heterogeneous across cohort and time",
            "No anticipation prior to the event time",
            "Implicit TWFE comparison weights are non-negative "
            "(negative-weight contamination is diagnosed, not "
            "assumed away)",
        ],
        "pre_conditions": [
            "Long-format panel with y, treat, time, id (same "
            "conventions as callaway_santanna)",
            "Staggered/variable treatment timing so the "
            "per-(cohort, time) weight diagnostic is "
            "meaningful",
            "Event-time window (leads, lags) contained within "
            "observed pre/post coverage",
        ],
        "failure_modes": [
            {
                "symptom": "model_info weights show large "
                "negative TWFE weights flagging "
                "forbidden comparisons",
                "exception": "AssumptionWarning",
                "remedy": "Drop already-treated controls and "
                "use a heterogeneity-robust staggered "
                "estimator instead of TWFE",
                "alternative": "callaway_santanna",
            },
            {
                "symptom": "Too few treated units per "
                "cohort-time cell to identify "
                "weights or SEs",
                "exception": "DataInsufficient",
                "remedy": "Coarsen the event-time window or "
                "pool cohorts to raise per-cell "
                "counts",
                "alternative": "did_imputation",
            },
        ],
        "alternatives": [
            "cohort_anchored_event_study",
            "sun_abraham",
            "callaway_santanna",
            "did_imputation",
        ],
        "typical_n_min": 200,
    },
    "diagnose_result": {
        "assumptions": [
            "The result object carries a recognizable method_type "
            "so the correct diagnostic battery can be routed.",
            "Each sub-check (e.g. parallel-trends, weak-IV, "
            "overid, balance) is only valid under that method's "
            "own identifying assumptions.",
            "Tests use the supplied alpha as the significance "
            "threshold; p-values are interpreted, not corrected "
            "for multiplicity.",
        ],
        "pre_conditions": [
            "A fitted EconometricResults or CausalResult from " "a StatsPAI estimator",
            "The estimator must expose enough fitted internals "
            "(residuals, first-stage, design info) for its "
            "checks",
        ],
        "failure_modes": [
            {
                "symptom": "Passed a raw DataFrame, dict, or "
                "estimate float instead of a fitted "
                "result object",
                "exception": "TypeError",
                "remedy": "Fit an estimator first and pass the "
                "returned result object, not the "
                "input data.",
                "alternative": "",
            },
            {
                "symptom": "Method type is unrecognized so no "
                "diagnostic battery applies and "
                "'checks' comes back empty",
                "exception": "(none — informational)",
                "remedy": "Run the method-appropriate "
                "standalone diagnostic directly "
                "instead of the router.",
                "alternative": "unified_sensitivity",
            },
        ],
        "alternatives": [
            "unified_sensitivity",
            "sensemakr",
            "oster_bounds",
            "spec_curve",
        ],
        "typical_n_min": 30,
    },
    "did_analysis": {
        "assumptions": [
            "Parallel trends between treated and control (2x2) or "
            "across cohorts (staggered) after any covariate "
            "adjustment",
            "No anticipation: pre-treatment outcomes unaffected " "by future treatment",
            "SUTVA / no interference across units; correct design "
            "auto-detection (2x2 vs staggered)",
        ],
        "pre_conditions": [
            "Panel or repeated cross-section with y, treat, "
            "time; id required for staggered designs",
            "treat is binary 0/1 for 2x2, or first-treatment "
            "period (0 = never-treated) for staggered",
            "event_window within observed periods when " "run_event_study is enabled",
        ],
        "failure_modes": [
            {
                "symptom": "Staggered design detected but no id "
                "column supplied, so cohorts cannot "
                "be formed",
                "exception": "ValueError",
                "remedy": "Pass the unit identifier via id= so "
                "first-treatment cohorts can be "
                "inferred",
                "alternative": "callaway_santanna",
            },
            {
                "symptom": "Event-study pre-trend test rejects "
                "parallel trends in the bundled "
                "report",
                "exception": "AssumptionWarning",
                "remedy": "Inspect the included honest_did "
                "sensitivity output and report bounds "
                "rather than the point ATT",
                "alternative": "honest_did",
            },
            {
                "symptom": "Requested method incompatible with "
                "the detected design (e.g. cs on a "
                "clean 2x2)",
                "exception": "MethodIncompatibility",
                "remedy": "Set method='auto' or pick an "
                "estimator matching the design",
                "alternative": "did",
            },
        ],
        "alternatives": [
            "did",
            "callaway_santanna",
            "sun_abraham",
            "bacon_decomposition",
        ],
        "typical_n_min": 100,
    },
    "direct_standardize": {
        "assumptions": [
            "Stratum-specific rates from the study population are "
            "applied to an external standard population's stratum "
            "weights, so within-stratum rates are the estimands "
            "of interest.",
            "Event counts within each stratum are Poisson, which "
            "the delta-method SE on the weighted rate sum relies "
            "on.",
            "Standard weights and study strata are aligned to the "
            "same stratum definitions and ordering.",
        ],
        "pre_conditions": [
            "events, population, and standard_weights are "
            "equal-length per-stratum vectors.",
            "All populations/person-times are positive and "
            "standard_weights are non-negative (normalized "
            "internally).",
        ],
        "failure_modes": [
            {
                "symptom": "A stratum has zero population "
                "denominator producing a "
                "divide-by-zero rate.",
                "exception": "ValueError",
                "remedy": "Collapse empty strata or ensure "
                "every stratum has positive "
                "person-time/population.",
                "alternative": "indirect_standardize",
            },
            {
                "symptom": "Very small event counts in some "
                "strata make the Poisson-delta SE "
                "unreliable.",
                "exception": "AssumptionWarning",
                "remedy": "Use indirect standardization (SMR) "
                "which is more stable with sparse "
                "study strata.",
                "alternative": "indirect_standardize",
            },
        ],
        "alternatives": [
            "indirect_standardize",
            "incidence_rate_ratio",
            "mantel_haenszel",
        ],
        "typical_n_min": 50,
    },
    "dl_propensity_score": {
        "assumptions": [
            "Unconfoundedness / selection on observables: "
            "treatment is conditionally independent of potential "
            "outcomes given covariates",
            "Overlap (positivity): 0 < e(X) < 1 for all units "
            "(scores are clipped to [0.02, 0.98])",
            "Covariates are pre-treatment and the MLP correctly "
            "approximates e(X) = P(T=1 | X)",
        ],
        "pre_conditions": [
            "Cross-sectional or pooled DataFrame with a binary "
            "treatment column and numeric covariate columns",
            "No missing values in the covariate matrix passed " "to the network",
            "Enough observations per treatment arm to fit a "
            "small MLP without overfitting",
        ],
        "failure_modes": [
            {
                "symptom": "Heavy clipping at 0.02/0.98 "
                "indicates near-deterministic "
                "treatment and positivity violation",
                "exception": "AssumptionWarning",
                "remedy": "Trim or restrict to the region of "
                "common support before using the "
                "scores in a weighted estimator",
                "alternative": "overlap_weighted_did",
            },
            {
                "symptom": "torch absent and sklearn MLP fails "
                "to converge in max_iter with lbfgs",
                "exception": "ConvergenceWarning",
                "remedy": "Increase max_iter, scale covariates, "
                "or shrink hidden_sizes for a simpler "
                "network",
                "alternative": "drdid",
            },
        ],
        "alternatives": ["overlap_weighted_did", "drdid", "did"],
        "typical_n_min": 200,
    },
    "dynotears": {
        "assumptions": [
            "Contemporaneous structure W is acyclic (enforced via "
            "the NOTEARS trace-exponential constraint); lagged "
            "structure A may be dense",
            "Stationary structural VAR with linear, additive "
            "relationships across the chosen lag window",
            "Causal sufficiency for the contemporaneous part — no "
            "unobserved confounders driving simultaneous edges",
            "Rows are evenly time-stamped and ordered; the SVAR "
            "treats observations as a single contiguous series",
        ],
        "pre_conditions": [
            "A time-indexed DataFrame, one row per time stamp, "
            "in chronological order",
            "Numeric variables (defaults to all numeric "
            "columns) and a chosen lag >= 1",
        ],
        "failure_modes": [
            {
                "symptom": "augmented-Lagrangian penalty hits "
                "rho_max without h(W) reaching "
                "h_tol; W not acyclic",
                "exception": "ConvergenceFailure",
                "remedy": "Raise max_iter, loosen h_tol, or "
                "increase lambda_w to shrink spurious "
                "contemporaneous edges.",
                "alternative": "lpcmci",
            },
            {
                "symptom": "too few time stamps relative to "
                "variables x (lag+1) yields an "
                "underdetermined, unstable fit",
                "exception": "DataInsufficient",
                "remedy": "Reduce lag or the number of "
                "variables, or supply a longer "
                "series.",
                "alternative": "lpcmci",
            },
            {
                "symptom": "all coefficients pruned to zero — " "empty graph",
                "exception": "(none — informational)",
                "remedy": "Lower threshold and/or the L1 "
                "penalties lambda_w/lambda_a.",
                "alternative": "pc_algorithm",
            },
        ],
        "alternatives": ["lpcmci", "pc_algorithm", "icp"],
        "typical_n_min": 100,
    },
    "equalized_odds": {
        "assumptions": [
            "Hardt-Price-Srebro criterion: max TPR-gap and " "FPR-gap across groups",
            "Both predictions and ground-truth labels are binary",
            "Each protected group contains both positive and "
            "negative ground-truth cases (so TPR/FPR are defined)",
            "Label quality is reliable since the metric " "conditions on true outcomes",
        ],
        "pre_conditions": [
            "Binary predictions column and binary ground-truth " "labels column",
            "Protected-attribute column",
        ],
        "failure_modes": [
            {
                "symptom": "A group has no positives or no "
                "negatives so TPR/FPR is undefined",
                "exception": "DataInsufficient",
                "remedy": "Ensure every protected group has "
                "both label classes or pool sparse "
                "groups.",
                "alternative": "demographic_parity",
            },
            {
                "symptom": "Labels or predictions not binary",
                "exception": "ValueError",
                "remedy": "Binarize both predictions and labels " "before calling.",
                "alternative": "fairness_audit",
            },
        ],
        "alternatives": [
            "demographic_parity",
            "fairness_audit",
            "counterfactual_fairness",
        ],
        "typical_n_min": 200,
    },
    "esttab": {
        "assumptions": [
            "Deprecated thin facade over sp.regtable; behavior "
            "and rendering follow regtable.",
            "Models come from positional arguments or the eststo "
            "global store; positional args take precedence when "
            "both exist.",
            "Input results share a comparable coefficient "
            "namespace for side-by-side columns.",
        ],
        "pre_conditions": [
            "One or more fitted StatsPAI result objects passed "
            "positionally, or a populated eststo store"
        ],
        "failure_modes": [
            {
                "symptom": "No models passed and the eststo " "store is empty",
                "exception": "ValueError",
                "remedy": "Pass model results positionally or "
                "populate the store with sp.eststo "
                "first.",
                "alternative": "regtable",
            },
            {
                "symptom": "Passed objects lack " "params/std_errors attributes",
                "exception": "AttributeError",
                "remedy": "Pass fitted StatsPAI result objects "
                "rather than raw data.",
                "alternative": "regtable",
            },
            {
                "symptom": "Continued use of the deprecated "
                "facade instead of regtable",
                "exception": "(none — informational)",
                "remedy": "Migrate to sp.regtable(*models, ...) " "for full control.",
                "alternative": "regtable",
            },
        ],
        "alternatives": ["regtable", "modelsummary"],
        "typical_n_min": 1,
    },
    "evalue": {
        "assumptions": [
            "VanderWeele-Ding (2017) E-value: quantifies the "
            "minimum confounder-exposure and confounder-outcome "
            "risk ratios that could explain away an observed "
            "association.",
            "The estimate is expressed (or convertible) to a "
            "risk-ratio scale; OR and HR map to RR via the "
            "rare-outcome approximation (rare=True) or the "
            "Ding-VanderWeele common-outcome conversion "
            "(rare=False, default); MD/SMD/OLS use exp(0.91*d).",
            "The E-value bounds joint confounding but assumes no "
            "other bias (selection, measurement, model "
            "misspecification).",
            "The CI E-value uses the confidence limit nearest the "
            "null (or a user-specified non-null `true` value); if "
            "the interval already contains that value its E-value "
            "is 1. Reproduces the R EValue package to machine "
            "precision.",
        ],
        "pre_conditions": [
            "A point estimate on a supported scale "
            "(RR/OR/HR/MD/SMD/OLS); for an exact risk-difference "
            "E-value call sp.evalue_rd with the 2x2 cell counts",
            "Either an SE or an explicit CI to obtain the CI "
            "E-value (OLS also needs the outcome sd)",
        ],
        "failure_modes": [
            {
                "symptom": "Ratio estimate <= 0 supplied for "
                "measure RR/OR/HR, or risk "
                "difference outside [-1, 1]",
                "exception": "ValueError",
                "remedy": "Pass a positive ratio for RR/OR/HR "
                "or a difference within [-1, 1], "
                "matching the measure argument.",
                "alternative": "",
            },
            {
                "symptom": "Reported E-value near 1.0, i.e. "
                "trivially weak confounding "
                "overturns the result",
                "exception": "(none — informational)",
                "remedy": "Treat the finding as fragile to "
                "unmeasured confounding and "
                "corroborate with a regression-based "
                "sensitivity analysis.",
                "alternative": "sensemakr",
            },
        ],
        "alternatives": ["sensemakr", "oster_bounds", "unified_sensitivity"],
        "typical_n_min": 1,
    },
    "frontier": {
        "assumptions": [
            "Composed error with symmetric noise plus a one-sided " "inefficiency term",
            "One-sided inefficiency from chosen distribution "
            "(half-normal / exponential / truncated-normal)",
            "Production frontier subtracts inefficiency (cost "
            "frontier adds it); inefficiency independent of "
            "noise",
        ],
        "pre_conditions": [
            "Cross-sectional data (one observation per unit)",
            "Continuous output (or cost) outcome plus frontier " "regressors",
        ],
        "failure_modes": [
            {
                "symptom": "Estimated lambda near 0 (no "
                "inefficiency, wrong residual skew)",
                "exception": "AssumptionViolation",
                "remedy": "Check residual skew direction (cost "
                "vs production) or refit with OLS if "
                "no inefficiency",
                "alternative": "regress",
            },
            {
                "symptom": "ML fails to converge with "
                "truncated-normal mean parameters",
                "exception": "ConvergenceWarning",
                "remedy": "Start from half-normal, rescale "
                "variables, or switch optimizer "
                "settings",
                "alternative": "regress",
            },
            {
                "symptom": "emean supplied without " "dist='truncated-normal'",
                "exception": "ValueError",
                "remedy": "Set dist='truncated-normal' to use "
                "inefficiency-mean determinants",
                "alternative": "xtfrontier",
            },
        ],
        "alternatives": ["xtfrontier", "regress", "feols"],
        "typical_n_min": 100,
    },
    "harvest_did": {
        "assumptions": [
            "Parallel trends hold for every valid 2x2 "
            "sub-comparison being harvested and aggregated",
            "No anticipation before each cohort's first treated "
            "period (reference horizon -1 is clean)",
            "Independence across units within each cohort "
            "(unit-level cluster-robust SEs); cross-horizon "
            "covariance ignored",
        ],
        "pre_conditions": [
            "Long-format panel with unit, time, outcome " "columns",
            "Either a binary treat indicator or a precomputed "
            "cohort (first-treatment) column with a "
            "never_value marker",
            "Multiple cohorts and overlapping periods so valid "
            "2x2 comparisons exist across the requested "
            "horizons",
        ],
        "failure_modes": [
            {
                "symptom": "No clean (never-treated or "
                "not-yet-treated) controls so no "
                "valid 2x2 cells can be harvested",
                "exception": "IdentificationFailure",
                "remedy": "Add never-treated units or restrict "
                "horizons to periods with available "
                "clean controls",
                "alternative": "callaway_santanna",
            },
            {
                "symptom": "Precision weighting dominated by "
                "one tiny high-variance comparison "
                "distorts the aggregate",
                "exception": "NumericalInstability",
                "remedy": "Switch weighting to 'equal' or "
                "'n_treated' to down-weight unstable "
                "cells",
                "alternative": "did_imputation",
            },
        ],
        "alternatives": [
            "callaway_santanna",
            "did_imputation",
            "sun_abraham",
            "gardner_did",
        ],
        "typical_n_min": 200,
    },
    "hdfe_ols": {
        "assumptions": [
            "Linear conditional mean after absorbing "
            "high-dimensional fixed effects (Frisch-Waugh-Lovell)",
            "Exogeneity of regressors conditional on the absorbed "
            "fixed-effect dimensions",
            "Clustered SEs require enough clusters for "
            "asymptotics; multiway cluster needs each dimension "
            "well-populated",
        ],
        "pre_conditions": [
            "Continuous outcome",
            "Fixed-effect factor variable(s) in the '| fe1 + "
            "fe2' part of the formula",
        ],
        "failure_modes": [
            {
                "symptom": "Singleton groups absorb their own "
                "observations and bias clustered SEs",
                "exception": "AssumptionWarning",
                "remedy": "Keep drop_singletons=True so "
                "singletons are removed before "
                "estimation",
                "alternative": "regress",
            },
            {
                "symptom": "Alternating-projections absorber " "does not converge",
                "exception": "ConvergenceWarning",
                "remedy": "Raise maxiter / loosen tol, or "
                "reduce the number of FE dimensions",
                "alternative": "feols",
            },
            {
                "symptom": "Too few clusters make wild/cluster " "inference unreliable",
                "exception": "DataInsufficient",
                "remedy": "Use a wild-cluster bootstrap or "
                "cluster at a coarser level",
                "alternative": "feols",
            },
        ],
        "alternatives": ["feols", "regress", "qreg"],
        "typical_n_min": 500,
    },
    "heterogeneity_of_effect": {
        "assumptions": [
            "Each input study estimate is approximately unbiased "
            "for its own population's effect with a "
            "known/consistent standard error",
            "DerSimonian-Laird random-effects model: study "
            "effects are exchangeable draws from a common "
            "distribution with a single between-study variance "
            "tau^2",
            "Within-study sampling errors are independent across "
            "studies and approximately normal",
        ],
        "pre_conditions": [
            "A sequence of point estimates plus a matching "
            "sequence of standard errors (same length, all SEs "
            "> 0)",
            "At least two studies so a between-study variance " "is identifiable",
        ],
        "failure_modes": [
            {
                "symptom": "estimates and ses have different "
                "lengths or contain non-positive / "
                "NaN SEs",
                "exception": "ValueError",
                "remedy": "Pass equal-length sequences with "
                "strictly positive, finite standard "
                "errors for every study",
                "alternative": "synthesise_evidence",
            },
            {
                "symptom": "Only one study supplied, so tau^2 "
                "and Q have no degrees of freedom",
                "exception": "DataInsufficient",
                "remedy": "Supply at least two independent "
                "study estimates before assessing "
                "heterogeneity",
                "alternative": "rwd_rct_concordance",
            },
        ],
        "alternatives": ["synthesise_evidence", "rwd_rct_concordance"],
        "typical_n_min": 3,
    },
    "honest_did": {
        "assumptions": [
            "Parallel trends may be violated by a bounded amount "
            "(relaxes exact PT); identified set indexed by "
            "violation magnitude M",
            "Pre-trend coefficients are informative about "
            "post-treatment bias (smoothness or "
            "relative-magnitude restrictions)",
            "No anticipation, so pre-period event-study "
            "coefficients reflect only PT violations not "
            "treatment",
        ],
        "pre_conditions": [
            "A fitted DID CausalResult containing event-study "
            "estimates in result.model_info",
            "At least one estimated pre-period (lead) "
            "coefficient to bound the violation, plus the "
            "target relative time e",
            "Event-study SEs available at the queried horizon " "e",
        ],
        "failure_modes": [
            {
                "symptom": "Passed result lacks event_study "
                "estimates (point-estimate-only DID)",
                "exception": "KeyError",
                "remedy": "Re-run the upstream estimator with "
                "an event-study/dynamic specification "
                "before calling honest_did",
                "alternative": "callaway_santanna",
            },
            {
                "symptom": "No pre-treatment periods so the "
                "violation magnitude cannot be "
                "anchored",
                "exception": "DataInsufficient",
                "remedy": "Estimate additional leads so "
                "pre-trend slope can bound "
                "post-treatment bias",
                "alternative": "sun_abraham",
            },
            {
                "symptom": "Robust CI still excludes zero only "
                "at M=0 — effect not robust to "
                "plausible PT violations",
                "exception": "(none — informational)",
                "remedy": "Report the breakdown M at which "
                "significance is lost rather than the "
                "point estimate",
                "alternative": "",
            },
        ],
        "alternatives": [
            "callaway_santanna",
            "sun_abraham",
            "event_study",
            "did_imputation",
        ],
        "typical_n_min": 100,
    },
    "icp": {
        "assumptions": [
            "Invariance: the conditional distribution of Y given "
            "its true direct causes is stable across all "
            "environments (Peters et al. 2016)",
            "Environments perturb predictors but do not intervene "
            "directly on Y and do not change the Y|causes "
            "mechanism",
            "Linear-Gaussian mechanism under method='linear' "
            "(F-test on residuals); method='nonlinear' relaxes to "
            "a K-S test on local-linear residuals",
            "Returns the intersection of invariant predictor "
            "subsets — yields a conservative (possibly empty) set "
            "of guaranteed direct parents",
        ],
        "pre_conditions": [
            "Predictor matrix X (n x p) and response y of " "length n",
            "An integer-coded environment label of length n "
            "with at least two distinct environments",
        ],
        "failure_modes": [
            {
                "symptom": "fewer than two environments "
                "supplied — no invariance contrast "
                "possible",
                "exception": "ValueError",
                "remedy": "Provide environment with at least "
                "two distinct integer codes; more "
                "environments sharpen identification.",
                "alternative": "dynotears",
            },
            {
                "symptom": "no subset is invariant across "
                "environments, so the accepted "
                "parent set is empty",
                "exception": "(none — informational)",
                "remedy": "Relax alpha, switch to "
                "method='nonlinear', or check whether "
                "an environment directly perturbs Y.",
                "alternative": "pc_algorithm",
            },
        ],
        "alternatives": ["pc_algorithm", "dynotears", "dag"],
        "typical_n_min": 200,
    },
    "identify": {
        "assumptions": [
            "The input graph is a DAG (acyclicity); bidirected "
            "edges are encoded via latent nodes representing "
            "unobserved common causes",
            "Markovian/semi-Markovian structure consistent with "
            "the Shpitser-Pearl ID algorithm; causal effect "
            "identifiability is decided structurally, not "
            "statistically",
            "Treatment and outcome node sets are vertices "
            "actually present in the supplied DAG",
        ],
        "pre_conditions": [
            "A statspai.dag.DAG instance with named nodes",
            "Treatment set X and outcome set Y given as node "
            "names present in the graph",
        ],
        "failure_modes": [
            {
                "symptom": "treatment or outcome name not a " "node in the DAG",
                "exception": "KeyError",
                "remedy": "Use exact node names from dag.nodes; "
                "build the graph with sp.dag first "
                "and confirm spelling.",
                "alternative": "dag",
            },
            {
                "symptom": "P(Y | do(X)) is not identifiable "
                "from the graph due to an "
                "unblockable bidirected (hedge) path "
                "between X and Y",
                "exception": "IdentificationFailure",
                "remedy": "Add measured covariates to break the "
                "confounding path or assume more "
                "structure, then re-run; consider a "
                "design with an instrument or "
                "invariance assumption.",
                "alternative": "swig",
            },
        ],
        "alternatives": ["dag", "swig", "pc_algorithm", "llm_dag_constrained"],
        "typical_n_min": 1,
    },
    "incidence_rate_ratio": {
        "assumptions": [
            "Events in each arm are Poisson over person-time at "
            "risk with a constant underlying rate (no "
            "time-varying hazard within group).",
            "Person-time is measured in consistent units across "
            "the exposed and unexposed arms.",
            "The exact CI uses the F-distribution/Poisson "
            "relationship, valid for count data without "
            "overdispersion.",
        ],
        "pre_conditions": [
            "Event counts and person-time totals for an "
            "exposed and an unexposed group.",
            "Person-time values are strictly positive and on " "the same time scale.",
        ],
        "failure_modes": [
            {
                "symptom": "Zero events in one arm makes the "
                "Wald log-rate SE undefined.",
                "exception": "ZeroDivisionError",
                "remedy": "Use method='exact' which handles "
                "zero-event arms via the Poisson "
                "exact CI.",
                "alternative": "mantel_haenszel",
            },
            {
                "symptom": "Overdispersed counts "
                "(clustering/recurrent events) make "
                "the Poisson CI too narrow.",
                "exception": "AssumptionWarning",
                "remedy": "Fit a Poisson/negative-binomial rate "
                "model with an offset for "
                "person-time.",
                "alternative": "poisson",
            },
        ],
        "alternatives": [
            "poisson",
            "mantel_haenszel",
            "relative_risk",
            "direct_standardize",
        ],
        "typical_n_min": 20,
    },
    "indirect_standardize": {
        "assumptions": [
            "Reference (standard) stratum-specific rates apply to "
            "the study population to form expected events; SMR = "
            "observed/expected.",
            "Observed events follow a Poisson distribution, "
            "underpinning the exact (Byar/Garwood) CI.",
            "Reference rates are estimated precisely enough to be "
            "treated as fixed (no propagation of reference-rate "
            "uncertainty).",
        ],
        "pre_conditions": [
            "Study population sizes per stratum plus reference "
            "events and reference population per matching "
            "stratum.",
            "Strata in the study and reference vectors "
            "correspond one-to-one with positive reference "
            "populations.",
        ],
        "failure_modes": [
            {
                "symptom": "Expected events near zero (rare "
                "outcome, small study pop) yields an "
                "unstable SMR with a huge CI.",
                "exception": "AssumptionWarning",
                "remedy": "Aggregate strata or use direct "
                "standardization if study stratum "
                "rates are estimable.",
                "alternative": "direct_standardize",
            },
            {
                "symptom": "A reference stratum has zero "
                "population so its reference rate is "
                "undefined.",
                "exception": "ValueError",
                "remedy": "Provide positive reference "
                "denominators for every stratum used.",
                "alternative": "direct_standardize",
            },
        ],
        "alternatives": ["direct_standardize", "incidence_rate_ratio", "poisson"],
        "typical_n_min": 50,
    },
    "ipcw": {
        "assumptions": [
            "Censoring is at random given the modeled covariates "
            "(no unmeasured predictors of dropout), so the "
            "censoring hazard is correctly specified.",
            "Positivity: every covariate stratum retains nonzero "
            "probability of remaining uncensored over follow-up.",
            "Stabilized weights use baseline-only covariates in "
            "the numerator; weights have mean about 1 and finite "
            "variance under correct specification.",
        ],
        "pre_conditions": [
            "Data provide time and event (1=event, 0=censored) "
            "plus censor_covariates predicting censoring.",
            "Long person-time format with (id, t, time, event) "
            "when method='pooled_logistic'.",
            "Adequate number of censoring events to fit the " "chosen nuisance model.",
        ],
        "failure_modes": [
            {
                "symptom": "Extreme/exploding weights from "
                "near-positivity violations inflate "
                "variance",
                "exception": "NumericalInstability",
                "remedy": "Keep stabilize=True and truncate at "
                "e.g. (0.01,0.99); re-examine the "
                "censoring model for offending "
                "covariates.",
                "alternative": "clone_censor_weight",
            },
            {
                "symptom": "Censoring model misspecified, "
                "leaving residual selection bias in "
                "weighted estimates",
                "exception": "AssumptionViolation",
                "remedy": "Enrich censor_covariates or use a "
                "doubly-robust outcome-plus-weights "
                "estimator.",
                "alternative": "aipw",
            },
        ],
        "alternatives": ["clone_censor_weight", "msm", "aipw"],
        "typical_n_min": 200,
    },
    "levinsohn_petrin": {
        "assumptions": [
            "Scalar unobserved productivity, monotone in the "
            "intermediate-input demand so the proxy is invertible "
            "for omega",
            "Intermediate input (materials/energy) is strictly "
            "positive and chosen freely each period",
            "First-order Markov productivity process; capital is "
            "predetermined relative to current shock",
        ],
        "pre_conditions": [
            "Long-form firm-year panel with log output, labor, "
            "capital, and a positive materials proxy",
            "At least two periods per firm for the stage-2 " "lagged-productivity step",
        ],
        "failure_modes": [
            {
                "symptom": "Materials proxy contains "
                "zeros/negatives, breaking the "
                "monotone inversion",
                "exception": "ValueError",
                "remedy": "Drop or impute non-positive "
                "materials and confirm the proxy is "
                "strictly positive.",
                "alternative": "olley_pakes",
            },
            {
                "symptom": "Stage-1 polynomial in (labor, "
                "capital, materials) is collinear, "
                "so the control function is not "
                "invertible",
                "exception": "IdentificationFailure",
                "remedy": "Lower polynomial_degree or ensure "
                "inputs are not perfectly "
                "proportional.",
                "alternative": "ackerberg_caves_frazer",
            },
        ],
        "alternatives": [
            "ackerberg_caves_frazer",
            "olley_pakes",
            "wooldridge_prod",
            "gmm",
        ],
        "typical_n_min": 400,
    },
    "levpet": {
        "assumptions": [
            "Scalar productivity monotone in intermediate-input "
            "demand (invertible proxy)",
            "Strictly positive materials used in every period; "
            "capital predetermined",
            "First-order Markov productivity evolution",
        ],
        "pre_conditions": [
            "Long-form firm-year panel with log output, labor, "
            "capital, and positive materials",
            "Two or more periods per firm for the " "lagged-productivity moment",
        ],
        "failure_modes": [
            {
                "symptom": "Non-positive materials values "
                "prevent the proxy inversion",
                "exception": "ValueError",
                "remedy": "Filter to rows with strictly "
                "positive materials before "
                "estimation.",
                "alternative": "olley_pakes",
            },
            {
                "symptom": "Free-input (labor) coefficient is "
                "unstable due to the OP/LP "
                "functional-dependence problem",
                "exception": "IdentificationFailure",
                "remedy": "Switch to an estimator that "
                "identifies labor in stage 2 with "
                "lagged instruments.",
                "alternative": "ackerberg_caves_frazer",
            },
        ],
        "alternatives": [
            "ackerberg_caves_frazer",
            "olley_pakes",
            "wooldridge_prod",
            "gmm",
        ],
        "typical_n_min": 400,
    },
    "llm_causal_assess": {
        "assumptions": [
            "Correctness is decided by case-insensitive substring "
            "matching of the target answer in the LLM response — "
            "not semantic equivalence",
            "The supplied llm_client is deterministic enough that "
            "scores are reproducible; randomness in sampling "
            "inflates score variance",
            "Level-2 DAG-fragment questions presuppose "
            "ground-truth answers reflect a single agreed-upon "
            "causal structure",
        ],
        "pre_conditions": [
            "A callable llm_client(prompt) -> str",
            "At least one of level1_items / level2_items "
            "DataFrames with question and answer columns",
        ],
        "failure_modes": [
            {
                "symptom": "neither level1_items nor "
                "level2_items supplied, or DataFrame "
                "lacks question/answer columns",
                "exception": "ValueError",
                "remedy": "Pass at least one item set with both "
                "question and answer columns.",
                "alternative": "pairwise_causal_benchmark",
            },
            {
                "symptom": "substring matching marks a "
                "semantically-correct paraphrase as "
                "wrong, deflating accuracy",
                "exception": "(none — informational)",
                "remedy": "Phrase target answers as short "
                "canonical tokens, or pre-normalize "
                "responses before scoring.",
                "alternative": "pairwise_causal_benchmark",
            },
        ],
        "alternatives": [
            "pairwise_causal_benchmark",
            "causal_mas",
            "llm_dag_validate",
            "llm_dag_propose",
        ],
        "typical_n_min": 20,
    },
    "load_preregister": {
        "assumptions": [
            "The file on disk is an intact pre-registration "
            "previously written by preregister() and is the "
            "authoritative pre-analysis plan.",
            "Recorded deviations and metadata are preserved "
            "verbatim on the returned CausalQuestion.",
            "Loading is read-only and does not alter the "
            "registered estimand or analysis specification.",
        ],
        "pre_conditions": [
            "The pre-registration file exists at the given " "path and is readable.",
            "File format/version matches what preregister() " "produced.",
        ],
        "failure_modes": [
            {
                "symptom": "Path does not exist or is " "unreadable",
                "exception": "FileNotFoundError",
                "remedy": "Pass the exact path returned by "
                "preregister() and confirm the file "
                "is present.",
                "alternative": "preregister",
            },
            {
                "symptom": "File is corrupted or written by an "
                "incompatible schema version",
                "exception": "ValueError",
                "remedy": "Re-export the plan with the current "
                "preregister() or restore an "
                "uncorrupted copy.",
                "alternative": "preregister",
            },
        ],
        "alternatives": ["preregister"],
        "typical_n_min": 1,
    },
    "local_projections": {
        "assumptions": [
            "Outcome and shock series are (covariance-)stationary "
            "so horizon-h projections are not spurious",
            "Shock is conditionally exogenous at t given controls "
            "and the auto-added lags (no contemporaneous feedback "
            "from y_t to shock_t)",
            "Newey-West truncation lag is adequate for the "
            "moving-average serial correlation induced by "
            "overlapping horizons",
            "Controls are passed at their time-t values verbatim "
            "— lag them yourself; the estimator does not re-lag "
            "them",
            "For identification='lpirfs_cholesky', the endogenous "
            "variable order defines the recursive structural shock, "
            "matching lpirfs::lp_lin with shock_type=1",
        ],
        "pre_conditions": [
            "Single time-ordered series in a DataFrame with "
            "the outcome and shock columns",
            "For identification='lpirfs_cholesky', endog_order "
            "contains the outcome and shock columns in the intended "
            "Cholesky ordering",
            "Length comfortably exceeds horizons + max lag so "
            "the deepest horizon regression retains enough "
            "usable rows",
        ],
        "failure_modes": [
            {
                "symptom": "Confidence bands explode or flip "
                "sign at long horizons after passing "
                "already-lagged controls together "
                "with auto_lag=True",
                "exception": "(none — informational)",
                "remedy": "Set auto_lag=False for a bare "
                "specification, or drop your manual "
                "lags so collinear duplicate-lag "
                "columns are not formed",
                "alternative": "var",
            },
            {
                "symptom": "The horizon-0 response is zero when the "
                "shock is ordered after the response variable",
                "exception": "(none — Cholesky ordering convention)",
                "remedy": "Use the intended endog_order, or switch "
                "back to identification='direct' if the shock column "
                "is already externally identified",
                "alternative": "var",
            },
            {
                "symptom": "Series too short: the "
                "longest-horizon regression has "
                "fewer observations than regressors "
                "and OLS cannot be solved",
                "exception": "DataInsufficient",
                "remedy": "Reduce horizons or nw_lags, or "
                "supply a longer sample so each "
                "horizon retains degrees of freedom",
                "alternative": "var",
            },
        ],
        "alternatives": ["var", "arima", "iv"],
        "typical_n_min": 80,
    },
    "long_term_from_short": {
        "assumptions": [
            "Sequential surrogacy: each surrogate wave blocks the "
            "treatment's effect on later waves and on the "
            "long-term outcome (dynamic surrogacy)",
            "Comparability across samples: the conditional "
            "outcome mean learned in the observational sample "
            "transports to the experimental sample for every wave",
            "Each wave's surrogate columns are measured in both "
            "the experimental and observational samples",
            "Treatment is sustained/long-term and the short-run "
            "waves capture its full downstream path",
        ],
        "pre_conditions": [
            "Experimental sample with treatment and all K "
            "surrogate waves but no long-term outcome",
            "Observational sample with all K surrogate waves "
            "plus the long-term outcome",
            "surrogates_waves given as an ordered list of " "per-wave column lists",
        ],
        "failure_modes": [
            {
                "symptom": "A wave's surrogate columns are "
                "missing from one of the two samples",
                "exception": "KeyError",
                "remedy": "Ensure every wave column list is "
                "present in both experimental and "
                "observational frames",
                "alternative": "surrogate_index",
            },
            {
                "symptom": "Later waves are post-treatment "
                "colliders, breaking the "
                "iterated-expectation identification",
                "exception": "IdentificationFailure",
                "remedy": "Restrict waves to surrogates that "
                "fully mediate the effect, or "
                "validate with a placebo long-term "
                "outcome",
                "alternative": "proximal_surrogate_index",
            },
            {
                "symptom": "Too few units per treatment arm for "
                "stable bootstrap variance across K "
                "nested regressions",
                "exception": "DataInsufficient",
                "remedy": "Increase sample size or reduce the "
                "number of surrogate waves before "
                "bootstrapping",
                "alternative": "surrogate_index",
            },
        ],
        "alternatives": [
            "surrogate_index",
            "proximal_surrogate_index",
            "identify_transport",
        ],
        "typical_n_min": 500,
    },
    "longitudinal_contrast": {
        "assumptions": [
            "Sequential exchangeability (no unmeasured "
            "time-varying confounding) and positivity hold so "
            "each regime's mean outcome is identified via the "
            "underlying g-method.",
            "The two regimes are well-defined dynamic/static "
            "treatment rules over the same time horizon.",
            "The contrast SE uses an independence-style delta "
            "approximation, assuming negligible covariance "
            "between the two regime estimates.",
        ],
        "pre_conditions": [
            "Both regimes are evaluable on the long-format "
            "longitudinal data (matching covariate keys per "
            "period).",
            "Time-varying confounders and treatment history "
            "are present for the g-method to adjust on.",
            "Sufficient subjects following (or weighted "
            "toward) each regime to estimate both means.",
        ],
        "failure_modes": [
            {
                "symptom": "One regime has near-zero effective "
                "support, giving an unstable or "
                "extreme mean",
                "exception": "DataInsufficient",
                "remedy": "Choose less extreme regimes or check "
                "positivity of the treatment rules "
                "over follow-up.",
                "alternative": "msm",
            },
            {
                "symptom": "Reported contrast SE is too narrow "
                "because regime estimates are "
                "correlated",
                "exception": "AssumptionWarning",
                "remedy": "Use a joint/bootstrap variance over "
                "both regimes instead of the additive "
                "delta-method SE.",
                "alternative": "bootstrap",
            },
        ],
        "alternatives": ["msm", "regime", "bootstrap"],
        "typical_n_min": 200,
    },
    "lpcmci": {
        "assumptions": [
            "Stationary time series — lag structure is constant " "over time",
            "Faithfulness: conditional-independence relations in "
            "the data reflect the underlying time-series graph",
            "Allows latent confounders (no causal sufficiency "
            "required); surviving contemporaneous edges typed as "
            "a proxy for hidden common causes",
            "Default CI test is Gaussian partial correlation, "
            "assuming linear dependence unless a custom ci_test "
            "is supplied",
        ],
        "pre_conditions": [
            "A long-format multivariate DataFrame, one row per "
            "time stamp, in chronological order",
            "A chosen tau_max maximum lag and significance " "level alpha",
        ],
        "failure_modes": [
            {
                "symptom": "series too short for reliable CI "
                "tests at the requested tau_max / "
                "conditioning dimension",
                "exception": "DataInsufficient",
                "remedy": "Lower tau_max and max_cond_dim, "
                "reduce variables, or supply more "
                "time stamps.",
                "alternative": "dynotears",
            },
            {
                "symptom": "spurious or missing edges from "
                "violated linearity in the default "
                "partial-correlation CI test",
                "exception": "(none — informational)",
                "remedy": "Pass a nonparametric ci_test, or "
                "tighten/loosen alpha for the CI "
                "tests.",
                "alternative": "pc_algorithm",
            },
        ],
        "alternatives": ["dynotears", "pc_algorithm", "icp"],
        "typical_n_min": 150,
    },
    "mantel_haenszel": {
        "assumptions": [
            "The stratum-specific association (OR or RR) is "
            "constant across strata, so a single pooled measure "
            "is meaningful (test homogeneity separately).",
            "Strata define confounder levels over which the "
            "measure is conditioned to remove confounding.",
            "Sparse-data validity holds: the MH estimator is "
            "consistent even with small per-stratum counts but "
            "many strata.",
        ],
        "pre_conditions": [
            "A (K,2,2) array of stratum tables laid out "
            "[[a,b],[c,d]] = exposure x outcome.",
            "K >= 1 strata with at least one informative " "(non-zero-margin) table.",
        ],
        "failure_modes": [
            {
                "symptom": "Effect modification (OR/RR varies "
                "by stratum) makes the pooled "
                "estimate misleading.",
                "exception": "AssumptionWarning",
                "remedy": "Run Breslow-Day for homogeneity and "
                "report stratum-specific estimates if "
                "it rejects.",
                "alternative": "breslow_day_test",
            },
            {
                "symptom": "All strata degenerate to zero MH "
                "numerator/denominator terms.",
                "exception": "ValueError",
                "remedy": "Drop empty strata and ensure tables "
                "contain discordant cells.",
                "alternative": "odds_ratio",
            },
        ],
        "alternatives": ["odds_ratio", "relative_risk", "breslow_day_test"],
        "typical_n_min": 40,
    },
    "markup": {
        "assumptions": [
            "The supplied production fit has an identified "
            "elasticity for the flexible input (theta_v)",
            "The flexible input is statically optimized with no "
            "adjustment costs, so the output elasticity equals "
            "the expenditure share scaled by the markup",
            "Optional eta-correction removes the stage-1 i.i.d. "
            "output shock before forming the cost share (De "
            "Loecker-Warzynski 2012, eq. 6)",
        ],
        "pre_conditions": [
            "A fitted ProductionResult whose coef contains the "
            "flexible_input elasticity",
            "Columns with log firm-time revenue and log "
            "flexible-input expenditure aligned to "
            "result.sample",
        ],
        "failure_modes": [
            {
                "symptom": "flexible_input name is absent from "
                "result.coef so no elasticity can be "
                "read",
                "exception": "KeyError",
                "remedy": "Pass a flexible_input that matches a "
                "coefficient in the underlying "
                "production fit (e.g. the proxy 'm').",
                "alternative": "levinsohn_petrin",
            },
            {
                "symptom": "Revenue or input-cost columns are "
                "passed in levels, producing "
                "implausible (negative or huge) "
                "markups",
                "exception": "ValueError",
                "remedy": "Log-transform revenue and input "
                "expenditure before calling markup.",
                "alternative": "",
            },
        ],
        "alternatives": [
            "ackerberg_caves_frazer",
            "levinsohn_petrin",
            "wooldridge_prod",
            "olley_pakes",
        ],
        "typical_n_min": 400,
    },
    "mean_comparison": {
        "assumptions": [
            "group is a binary variable defining exactly two " "comparison groups.",
            "Each variable in variables exists in data and is "
            "numeric (or categorical for the chi2 test).",
            "Output format is one of text/latex/html/markdown; "
            "file format auto-detects from the filename "
            "extension.",
        ],
        "pre_conditions": [
            "A pandas DataFrame plus a list of variable "
            "columns and a binary group column"
        ],
        "failure_modes": [
            {
                "symptom": "group has more or fewer than two " "distinct levels",
                "exception": "ValueError",
                "remedy": "Recode group to exactly two levels " "before calling.",
                "alternative": "balance_table",
            },
            {
                "symptom": "A named variable or the group "
                "column is absent from data",
                "exception": "KeyError",
                "remedy": "Verify every variable name and group "
                "exist as columns in data.",
                "alternative": "balance_table",
            },
            {
                "symptom": "Unsupported test name passed",
                "exception": "ValueError",
                "remedy": "Use test in {ttest, ranksum, chi2}.",
                "alternative": "balance_table",
            },
        ],
        "alternatives": ["balance_table", "regtable", "collect"],
        "typical_n_min": 1,
    },
    "melogit": {
        "assumptions": [
            "Binary outcome with logit link conditional on random " "effects",
            "Cluster-level random intercepts (and slopes) are " "normally distributed",
            "Random effects independent of covariates (no "
            "correlated-effects endogeneity)",
        ],
        "pre_conditions": [
            "Binary (0/1) outcome",
            "Grouping variable for the random effects",
        ],
        "failure_modes": [
            {
                "symptom": "Adaptive quadrature likelihood does " "not converge",
                "exception": "ConvergenceWarning",
                "remedy": "Increase quadrature points or "
                "simplify the random-effects "
                "structure to a single intercept",
                "alternative": "mixed",
            },
            {
                "symptom": "Estimated random-effect variance "
                "near zero (no clustering)",
                "exception": "(none — informational)",
                "remedy": "Drop the random effect and fit " "ordinary logit",
                "alternative": "regress",
            },
            {
                "symptom": "Perfect separation in a sparse " "cluster",
                "exception": "NumericalInstability",
                "remedy": "Collapse sparse categories or add a " "weak penalty/prior",
                "alternative": "regress",
            },
        ],
        "alternatives": ["mixed", "regress"],
        "typical_n_min": 300,
    },
    "mixed": {
        "assumptions": [
            "Continuous outcome, linear in fixed effects",
            "Random effects (intercepts/slopes) normally "
            "distributed with the chosen covariance structure",
            "Residuals normal and homoscedastic conditional on "
            "random effects; random effects independent of "
            "covariates",
        ],
        "pre_conditions": [
            "Continuous outcome",
            "Grouping variable (or nested list of grouping "
            "levels) for random effects",
        ],
        "failure_modes": [
            {
                "symptom": "REML/ML optimizer fails to converge "
                "with rich random-slope covariance",
                "exception": "ConvergenceWarning",
                "remedy": "Switch cov_type to 'diagonal' or "
                "'identity', or drop random slopes",
                "alternative": "regress",
            },
            {
                "symptom": "Singular covariance (boundary " "variance estimate)",
                "exception": "NumericalInstability",
                "remedy": "Simplify the random-effects "
                "covariance or remove the offending "
                "random term",
                "alternative": "feols",
            },
            {
                "symptom": "Binary or count outcome passed to a " "linear model",
                "exception": "AssumptionViolation",
                "remedy": "Use a generalized mixed model for " "the appropriate family",
                "alternative": "melogit",
            },
        ],
        "alternatives": ["regress", "melogit", "feols"],
        "typical_n_min": 200,
    },
    "modelsummary": {
        "assumptions": [
            "Deprecated thin wrapper over sp.regtable exposing "
            "the R modelsummary parameter surface.",
            "Input results share a comparable coefficient "
            "namespace for side-by-side columns.",
            "Parameters map onto regtable per the module " "docstring.",
        ],
        "pre_conditions": ["One or more fitted StatsPAI result objects"],
        "failure_modes": [
            {
                "symptom": "Passed objects are not fitted "
                "results with coefficient attributes",
                "exception": "AttributeError",
                "remedy": "Pass StatsPAI result objects rather "
                "than raw data or DataFrames.",
                "alternative": "regtable",
            },
            {
                "symptom": "No models supplied to the table",
                "exception": "ValueError",
                "remedy": "Pass at least one model result " "positionally.",
                "alternative": "regtable",
            },
            {
                "symptom": "Continued use of the deprecated "
                "wrapper instead of regtable",
                "exception": "(none — informational)",
                "remedy": "Migrate to sp.regtable(*models, ...) " "for full control.",
                "alternative": "regtable",
            },
        ],
        "alternatives": ["regtable", "esttab", "outreg2"],
        "typical_n_min": 1,
    },
    "mr_mediation": {
        "assumptions": [
            "Standard MR instrument validity for the SNPs: "
            "relevance, independence (no confounding of "
            "SNP-outcome), and exclusion restriction acting only "
            "through exposure/mediator.",
            "Two-step network-MR structure (Burgess et al. 2015): "
            "IVW for total and exposure->mediator effects, MVMR "
            "for the direct effect, with a "
            "linear/no-effect-modification mediation model.",
            "Delta-method SE for the indirect effect treats the "
            "step components as combinable, assuming approximate "
            "normality of the IVW/MVMR estimates.",
        ],
        "pre_conditions": [
            "snp_associations has one row per SNP with beta+SE "
            "for exposure, mediator, and outcome.",
            "Enough valid, non-overlapping instruments to "
            "identify the MVMR direct effect.",
            "Exposure and mediator instruments are not " "collinear in the MVMR step.",
        ],
        "failure_modes": [
            {
                "symptom": "Weak or collinear instruments make "
                "the MVMR direct effect "
                "unidentified/unstable",
                "exception": "IdentificationFailure",
                "remedy": "Add stronger independent instruments "
                "or check conditional F-statistics "
                "before running MVMR.",
                "alternative": "mr",
            },
            {
                "symptom": "Horizontal pleiotropy biases IVW " "total/direct effects",
                "exception": "AssumptionViolation",
                "remedy": "Run pleiotropy-robust sensitivity "
                "(e.g., MR-Egger) and compare before "
                "trusting the decomposition.",
                "alternative": "mr_egger",
            },
        ],
        "alternatives": ["mr", "mr_egger", "mr_ivw"],
        "typical_n_min": 10,
    },
    "nbreg": {
        "assumptions": [
            "Count outcome with overdispersion: NB2 variance "
            "mu+alpha*mu^2 or NB1 variance mu*(1+delta), with "
            "alpha>0",
            "Log-linear conditional mean exp(x'beta)",
            "Observations conditionally independent given "
            "covariates (or correctly clustered)",
        ],
        "pre_conditions": [
            "Non-negative integer count outcome",
            "Overdispersion present (else Poisson suffices)",
        ],
        "failure_modes": [
            {
                "symptom": "Estimated alpha collapses toward 0 " "(no overdispersion)",
                "exception": "(none — informational)",
                "remedy": "Refit with Poisson since NB reduces "
                "to Poisson at alpha=0",
                "alternative": "poisson",
            },
            {
                "symptom": "ML fails to converge with " "extreme/separated counts",
                "exception": "ConvergenceWarning",
                "remedy": "Rescale covariates, raise maxiter, "
                "or simplify the linear predictor",
                "alternative": "poisson",
            },
            {
                "symptom": "Excess zeros beyond NB dispersion " "produces poor fit",
                "exception": "AssumptionViolation",
                "remedy": "Use a zero-inflated or hurdle count "
                "model for the structural-zero "
                "process",
                "alternative": "poisson",
            },
        ],
        "alternatives": ["poisson", "xtnbreg", "regress"],
        "typical_n_min": 200,
    },
    "odds_ratio": {
        "assumptions": [
            "The 2x2 table cross-classifies a binary exposure and "
            "binary outcome on independent observations.",
            "The Woolf CI relies on asymptotic normality of "
            "log(OR), adequate when no cell is small.",
            "The exact (Fisher conditional) CI assumes fixed "
            "margins under the non-central hypergeometric model.",
        ],
        "pre_conditions": [
            "Four non-negative 2x2 cell counts (a,b,c,d) or a "
            "2x2 array, laid out exposed/unexposed x "
            "outcome+/-.",
            "Observations are independent (not matched pairs " "or repeated measures).",
        ],
        "failure_modes": [
            {
                "symptom": "A zero cell makes the Woolf log-OR "
                "SE infinite or the OR 0/inf.",
                "exception": "ZeroDivisionError",
                "remedy": "Use method='exact' for a conditional "
                "CI that tolerates zero cells.",
                "alternative": "relative_risk",
            },
            {
                "symptom": "Matched/clustered design analyzed "
                "as unmatched, biasing the OR.",
                "exception": "MethodIncompatibility",
                "remedy": "Use conditional logistic regression "
                "for matched 2x2 data.",
                "alternative": "logit",
            },
        ],
        "alternatives": [
            "relative_risk",
            "risk_difference",
            "mantel_haenszel",
            "logit",
        ],
        "typical_n_min": 20,
    },
    "olley_pakes": {
        "assumptions": [
            "Scalar productivity monotone in the investment "
            "policy, so investment is invertible for omega",
            "Investment proxy strictly positive (zero-investment "
            "firm-years are dropped)",
            "Capital is predetermined; first-order Markov "
            "productivity; optional selection correction for "
            "survival",
        ],
        "pre_conditions": [
            "Long-form firm-year panel (one row per firm-year) "
            "with log output, labor, capital, and a positive "
            "investment proxy",
            "Two consecutive periods per firm for stage 2 "
            "(dropping period t also forfeits t+1 for that "
            "firm)",
        ],
        "failure_modes": [
            {
                "symptom": "Large share of firm-years have zero "
                "investment, so the sample collapses "
                "after the proxy inversion",
                "exception": "DataInsufficient",
                "remedy": "Use a proxy that is positive every "
                "period instead of investment.",
                "alternative": "levinsohn_petrin",
            },
            {
                "symptom": "Labor coefficient poorly identified "
                "due to OP functional dependence in "
                "stage 1",
                "exception": "IdentificationFailure",
                "remedy": "Move free-input identification to "
                "stage 2 with lagged instruments.",
                "alternative": "ackerberg_caves_frazer",
            },
        ],
        "alternatives": [
            "levinsohn_petrin",
            "ackerberg_caves_frazer",
            "wooldridge_prod",
            "gmm",
        ],
        "typical_n_min": 500,
    },
    "opreg": {
        "assumptions": [
            "Scalar productivity monotone in investment demand "
            "(invertible investment policy)",
            "Strictly positive investment in observed periods; "
            "zero-investment rows dropped by default",
            "Predetermined capital and first-order Markov " "productivity",
        ],
        "pre_conditions": [
            "Firm-year panel with log output, labor, capital, "
            "and a positive investment proxy",
            "Consecutive periods per firm so the lag operator "
            "in stage 2 is well defined",
        ],
        "failure_modes": [
            {
                "symptom": "Investment column has non-positive "
                "values that cannot be inverted",
                "exception": "ValueError",
                "remedy": "Set drop_zero_proxy=True or switch "
                "to a proxy positive in every period.",
                "alternative": "levpet",
            },
            {
                "symptom": "Sample too thin after dropping "
                "zero-investment and forfeited t+1 "
                "rows",
                "exception": "DataInsufficient",
                "remedy": "Prefer a materials-based proxy that "
                "retains more firm-years.",
                "alternative": "levinsohn_petrin",
            },
        ],
        "alternatives": [
            "levinsohn_petrin",
            "ackerberg_caves_frazer",
            "wooldridge_prod",
            "gmm",
        ],
        "typical_n_min": 500,
    },
    "orthogonal_to_bias": {
        "assumptions": [
            "Bias removable by residualizing each feature on the "
            "protected attribute (regress feature on A, keep "
            "residual)",
            "Linear feature-attribute relationship captured by "
            "the regression (one-hot A if multi-valued)",
            "Features to residualize are numeric",
            "In-sample orthogonality to A is an acceptable "
            "relaxation of counterfactual fairness (no explicit "
            "SCM required)",
        ],
        "pre_conditions": [
            "Numeric feature columns to residualize",
            "Protected-attribute column (numeric or " "categorical)",
        ],
        "failure_modes": [
            {
                "symptom": "Non-numeric feature passed for " "residualization",
                "exception": "ValueError",
                "remedy": "Encode features numerically before " "residualizing.",
                "alternative": "counterfactual_fairness",
            },
            {
                "symptom": "Residual orthogonality holds "
                "in-sample but bias leaks via "
                "nonlinear dependence on A",
                "exception": "AssumptionWarning",
                "remedy": "Audit the downstream predictor with "
                "a fairness metric rather than "
                "relying on linear residualization "
                "alone.",
                "alternative": "fairness_audit",
            },
        ],
        "alternatives": [
            "counterfactual_fairness",
            "fairness_audit",
            "demographic_parity",
            "equalized_odds",
        ],
        "typical_n_min": 100,
    },
    "outreg2": {
        "assumptions": [
            "Deprecated thin wrapper over sp.regtable; output is "
            "book-tab styled and not byte-identical to the legacy "
            "renderer.",
            "Output format auto-detects from the filename "
            "extension (.xlsx/.docx/.tex) unless overridden via "
            "format=.",
            "show_se=False is no longer supported and triggers a " "UserWarning.",
        ],
        "pre_conditions": [
            "One or more EconometricResults objects plus an " "output filename"
        ],
        "failure_modes": [
            {
                "symptom": "Filename extension is unrecognized " "and format='auto'",
                "exception": "ValueError",
                "remedy": "Use a .xlsx/.docx/.tex filename or "
                "set format= explicitly.",
                "alternative": "regtable",
            },
            {
                "symptom": "Word/Excel export requested but the "
                "writer backend is not installed",
                "exception": "ImportError",
                "remedy": "Install python-docx / openpyxl, or "
                "export to LaTeX instead.",
                "alternative": "regtable",
            },
            {
                "symptom": "Passed objects are not fitted " "regression results",
                "exception": "AttributeError",
                "remedy": "Pass EconometricResults objects " "rather than raw data.",
                "alternative": "regtable",
            },
        ],
        "alternatives": ["regtable", "modelsummary", "esttab"],
        "typical_n_min": 1,
    },
    "overlap_weighted_did": {
        "assumptions": [
            "Conditional parallel trends given covariates X after " "overlap weighting",
            "Overlap (positivity): propensity score e(X) bounded "
            "away from 0 and 1 so overlap weights e(X)(1-e(X)) "
            "are well-behaved",
            "No anticipation; correctly specified propensity "
            "model e(X) = P(treat=1 | X)",
        ],
        "pre_conditions": [
            "Two-period panel with a binary treat indicator "
            "and a binary pre/post time indicator",
            "Pre-treatment covariates supplied for the "
            "propensity score (without them it reduces to "
            "plain 2x2 DID)",
            "Both treatment arms present in pre and post " "periods",
        ],
        "failure_modes": [
            {
                "symptom": "Propensity scores pile up near 0/1 "
                "giving near-zero overlap weights "
                "and unstable ATT",
                "exception": "NumericalInstability",
                "remedy": "Trim extreme-PS units or simplify "
                "the covariate set to restore common "
                "support",
                "alternative": "drdid",
            },
            {
                "symptom": "ps_model='dl' requested but the "
                "deep-learning propensity backend "
                "fails to converge",
                "exception": "ConvergenceWarning",
                "remedy": "Fall back to ps_model='logit' or "
                "'gbm' for a more stable propensity "
                "estimate",
                "alternative": "dl_propensity_score",
            },
            {
                "symptom": "More than two time periods passed "
                "to this 2x2-only estimator",
                "exception": "ValueError",
                "remedy": "Collapse to a single pre/post "
                "contrast or use a "
                "staggered/multi-period estimator",
                "alternative": "callaway_santanna",
            },
        ],
        "alternatives": ["drdid", "did", "callaway_santanna", "dl_propensity_score"],
        "typical_n_min": 100,
    },
    "pairwise_causal_benchmark": {
        "assumptions": [
            "Every pair has a known ground-truth direction (A->B "
            "boolean); the task is binary causal-direction "
            "classification, not edge-presence discovery",
            "The LLM's free-text reply is parsed for a yes/no "
            "signal via the prompt template — parsing fidelity "
            "caps measured accuracy",
            "Pairs are treated as independent; shared-confounder "
            "pairs may have ill-defined ground truth",
        ],
        "pre_conditions": [
            "A ground_truth DataFrame with pair_a_col, "
            "pair_b_col, and a boolean truth_col",
            "A callable llm_client(prompt) -> str",
        ],
        "failure_modes": [
            {
                "symptom": "named pair or truth column missing " "from ground_truth",
                "exception": "KeyError",
                "remedy": "Set pair_a_col/pair_b_col/truth_col "
                "to columns that exist in the "
                "DataFrame.",
                "alternative": "llm_causal_assess",
            },
            {
                "symptom": "near-chance accuracy because "
                "replies do not fit the yes/no "
                "prompt template",
                "exception": "(none — informational)",
                "remedy": "Tighten prompt_template to force a "
                "yes/no token, or post-process "
                "responses before grading.",
                "alternative": "llm_causal_assess",
            },
        ],
        "alternatives": ["llm_causal_assess", "causal_mas", "pc_algorithm", "icp"],
        "typical_n_min": 30,
    },
    "paper_tables": {
        "assumptions": [
            "The main panel holds one fitted result per column "
            "(typically baseline then +controls then +FE then "
            "+cluster).",
            "Optional heterogeneity/robustness/placebo panels "
            "each hold comparable fitted results.",
            "template is one of aer/qje/econometrica/restat, "
            "which fixes star thresholds, SE style, and footer "
            "notes.",
        ],
        "pre_conditions": [
            "A sequence of fitted StatsPAI result objects for " "the main panel"
        ],
        "failure_modes": [
            {
                "symptom": "Unrecognized template name passed",
                "exception": "ValueError",
                "remedy": "Use template in {aer, qje, " "econometrica, restat}.",
                "alternative": "regtable",
            },
            {
                "symptom": "Panel entries are not fitted result "
                "objects with coefficient attributes",
                "exception": "AttributeError",
                "remedy": "Pass sequences of StatsPAI result " "objects per panel.",
                "alternative": "regtable",
            },
            {
                "symptom": "Export filename requested but the "
                "corresponding writer backend is "
                "missing",
                "exception": "ImportError",
                "remedy": "Install the docx/xlsx export extra "
                "or write Markdown/LaTeX instead.",
                "alternative": "collect",
            },
        ],
        "alternatives": ["regtable", "collect", "modelsummary"],
        "typical_n_min": 1,
    },
    "particle_filter": {
        "assumptions": [
            "A bootstrap SIR (sequential importance resampling) "
            "filter is appropriate: the latent causal parameter "
            "follows a (possibly non-Gaussian) random-walk "
            "transition with process_sd and the supplied models.",
            "The observation log-pdf correctly links each "
            "period's estimate and standard_error to the latent "
            "state (e.g., Gaussian measurement with the given "
            "SE).",
            "n_particles is large enough to approximate the "
            "filtering distribution, and systematic resampling at "
            "the ESS threshold curbs weight degeneracy.",
        ],
        "pre_conditions": [
            "estimates and standard_errors are aligned, "
            "finite, equal-length sequences over time.",
            "prior_mean/prior_var/process_sd specify a "
            "coherent state-space prior and dynamics.",
            "n_particles and ess_resample_threshold set " "sensibly.",
        ],
        "failure_modes": [
            {
                "symptom": "Particle weights degenerate (ESS "
                "collapses) so the posterior is "
                "driven by one particle",
                "exception": "NumericalInstability",
                "remedy": "Increase n_particles, raise "
                "process_sd, or lower the ESS "
                "resample threshold to trigger "
                "resampling earlier.",
                "alternative": "bridge",
            },
            {
                "symptom": "estimates and standard_errors "
                "differ in length or contain "
                "non-finite values",
                "exception": "ValueError",
                "remedy": "Align the two sequences and "
                "drop/repair NaN or non-positive SE "
                "entries before filtering.",
                "alternative": "",
            },
        ],
        "alternatives": ["bridge"],
        "typical_n_min": 5,
    },
    "proximal_surrogate_index": {
        "assumptions": [
            "Proximal surrogacy: an unobserved U may confound the "
            "surrogate-outcome link, relaxing conditional "
            "independence to two-stage completeness conditions "
            "(Imbens-Kallus-Mao-Wang 2025)",
            "Valid proxies W: related to the unobserved "
            "confounder U and excluded from the direct effect on "
            "Y (IV-style exclusion), present in the observational "
            "sample only",
            "Bridge-function identification holds; the "
            "linear-Gaussian 2SLS form assumes the bridge is "
            "linear",
            "Comparability of the bridge/structure across the "
            "experimental and observational samples",
        ],
        "pre_conditions": [
            "Experimental sample with treatment and "
            "surrogates; observational sample with surrogates, "
            "proxies W, and the long-term outcome",
            "At least as many proxies as needed to satisfy the "
            "completeness/rank condition for 2SLS",
            "Proxies measured only in the observational " "sample",
        ],
        "failure_modes": [
            {
                "symptom": "Proxies are weak or fail the "
                "rank/completeness condition, so the "
                "2SLS bridge is not identified",
                "exception": "IdentificationFailure",
                "remedy": "Supply proxies strongly related to U "
                "yet excluded from Y, ensuring the "
                "first stage is well-conditioned",
                "alternative": "surrogate_index",
            },
            {
                "symptom": "Linear-Gaussian bridge is "
                "misspecified for a nonlinear "
                "data-generating process",
                "exception": "AssumptionViolation",
                "remedy": "Use surrogate_index model hooks with "
                "a kernel/NN bridge and a custom 2SLS "
                "wrapper",
                "alternative": "surrogate_index",
            },
            {
                "symptom": "Near-collinear proxies make the "
                "2SLS solution numerically unstable",
                "exception": "NumericalInstability",
                "remedy": "Drop redundant proxies or regularise "
                "the first stage to restore "
                "conditioning",
                "alternative": "long_term_from_short",
            },
        ],
        "alternatives": [
            "surrogate_index",
            "long_term_from_short",
            "identify_transport",
        ],
        "typical_n_min": 1000,
    },
    "qreg": {
        "assumptions": [
            "Conditional quantile is linear in covariates",
            "Outcome continuous (or finely discretized) so the "
            "check-function minimization is well-posed",
            "Powell (1991) sandwich SEs require a consistent "
            "kernel density of the conditional density at zero",
        ],
        "pre_conditions": [
            "Continuous outcome variable",
            "Quantile tau strictly in (0,1)",
        ],
        "failure_modes": [
            {
                "symptom": "Sparse data in tail quantiles gives "
                "unstable density estimate and wide "
                "SEs",
                "exception": "NumericalInstability",
                "remedy": "Estimate a more central quantile or "
                "pool more data near the tail",
                "alternative": "regress",
            },
            {
                "symptom": "Crossing/degenerate fit when " "regressors are collinear",
                "exception": "ValueError",
                "remedy": "Drop or combine collinear regressors " "before refitting",
                "alternative": "regress",
            },
        ],
        "alternatives": ["regress", "feols"],
        "typical_n_min": 200,
    },
    "rd_flex": {
        "assumptions": [
            "Continuity-based RD identification at the cutoff: "
            "potential outcomes are continuous in the running "
            "variable except for the treatment jump.",
            "Cross-fit ML residualisation of the outcome (and "
            "treatment, when fuzzy) on covariates only removes "
            "outcome variance and does not bias the cutoff "
            "estimate, requiring honest K-fold cross-fitting.",
            "Covariates predict the outcome well enough to "
            "shorten CIs relative to plain rdrobust; covariates "
            "are pre-determined (not affected by treatment).",
        ],
        "pre_conditions": [
            "data has continuous running variable with "
            "adequate mass on both sides of the cutoff.",
            "Covariates list valid pre-treatment columns (or "
            "is empty/None to fall back to rdrobust).",
            "n_folds>=2 for genuine cross-fitting.",
        ],
        "failure_modes": [
            {
                "symptom": "Sparse data near the cutoff makes "
                "the local fit and learner unstable",
                "exception": "DataInsufficient",
                "remedy": "Widen the bandwidth via bwselect or "
                "collect more mass around the cutoff.",
                "alternative": "rdrobust",
            },
            {
                "symptom": "Covariates are post-treatment, so "
                "residualisation biases the estimate",
                "exception": "AssumptionViolation",
                "remedy": "Restrict covariates to strictly "
                "pre-determined variables and "
                "re-estimate.",
                "alternative": "rdrobust",
            },
        ],
        "alternatives": ["rdrobust", "rd_honest"],
        "typical_n_min": 200,
    },
    "rddensity": {
        "assumptions": [
            "Cattaneo-Jansson-Ma (2020) local-polynomial density "
            "test: under no manipulation the running-variable "
            "density is continuous at the cutoff.",
            "Density is estimated from the empirical CDF via "
            "local polynomial regression with a data-driven CJM "
            "bandwidth and bias-corrected inference (no binning).",
            "Sufficient mass on both sides of the cutoff within "
            "the bandwidth for the polynomial fit of order p.",
            "A rejection signals sorting/manipulation, not "
            "necessarily a violated RD design per se.",
        ],
        "pre_conditions": [
            "Running variable column + cutoff c for the RD " "density test",
            "Adequate observations on each side of the cutoff "
            "within the chosen bandwidth",
        ],
        "failure_modes": [
            {
                "symptom": "Too few points on one side of the "
                "cutoff for the local polynomial "
                "density fit",
                "exception": "DataInsufficient",
                "remedy": "Widen the bandwidth h or lower the "
                "polynomial order p to stabilize the "
                "local density estimate.",
                "alternative": "mccrary_test",
            },
            {
                "symptom": "Significant density discontinuity "
                "at the cutoff indicating "
                "manipulation/sorting",
                "exception": "(none — informational)",
                "remedy": "Inspect the histogram, consider a "
                "donut-hole RD, and report the result "
                "as a manipulation warning.",
                "alternative": "rdrobust",
            },
        ],
        "alternatives": ["mccrary_test", "rdrobust", "diagnose_result"],
        "typical_n_min": 100,
    },
    "regime": {
        "assumptions": [
            "The rule is a deterministic function of observed "
            "history and period, encoding a static or dynamic "
            "treatment strategy.",
            "DSL strings reference only available history columns "
            "and are restricted to safe expressions; the rule "
            "returns a valid treatment value each period.",
            "K matches the intended number of decision points "
            "when rule is a scalar or sequence.",
        ],
        "pre_conditions": [
            "History fields named in the rule exist when the "
            "Regime is later evaluated on data.",
            "rule is one of the supported forms: str DSL, "
            "sequence, callable(h,t), or scalar.",
            "K is set consistently with the longitudinal " "horizon being analyzed.",
        ],
        "failure_modes": [
            {
                "symptom": "DSL string references a missing "
                "covariate or uses unsupported "
                "syntax",
                "exception": "KeyError",
                "remedy": "Reference only existing history "
                "columns and keep the rule to the "
                "documented safe-eval grammar.",
                "alternative": "",
            },
            {
                "symptom": "Sequence length disagrees with the " "number of periods K",
                "exception": "ValueError",
                "remedy": "Make the rule sequence length equal "
                "to K or pass a scalar/callable "
                "instead.",
                "alternative": "",
            },
        ],
        "alternatives": ["longitudinal_contrast", "msm"],
        "typical_n_min": 1,
    },
    "regtable": {
        "assumptions": [
            "Input results share a comparable coefficient "
            "namespace for side-by-side columns.",
            "Each positional argument is a result with "
            "params/std_errors (or duck-typed equivalent); a list "
            "argument becomes its own panel.",
            "se_type and star_levels are applied uniformly across "
            "all columns and panels.",
        ],
        "pre_conditions": [
            "One or more fitted StatsPAI result objects (or "
            "lists thereof for multi-panel tables)"
        ],
        "failure_modes": [
            {
                "symptom": "keep/drop/order references a "
                "variable not present in any model",
                "exception": "KeyError",
                "remedy": "Reference only coefficient names "
                "that appear in the supplied models.",
                "alternative": "modelsummary",
            },
            {
                "symptom": "Passed objects lack " "params/std_errors attributes",
                "exception": "AttributeError",
                "remedy": "Pass fitted result objects rather "
                "than raw data or DataFrames.",
                "alternative": "cite",
            },
            {
                "symptom": "Invalid se_type or no models " "supplied",
                "exception": "ValueError",
                "remedy": "Provide at least one model and set "
                "se_type to one of se/t/p/ci.",
                "alternative": "modelsummary",
            },
        ],
        "alternatives": ["modelsummary", "esttab", "outreg2", "paper_tables"],
        "typical_n_min": 1,
    },
    "relative_risk": {
        "assumptions": [
            "The 2x2 table comes from a cohort or cross-sectional "
            "design where outcome risk (not just odds) is "
            "directly estimable.",
            "Observations are independent within each exposure " "group.",
            "The Katz log-RR CI assumes large-sample normality; "
            "the Haldane 0.5 correction is applied only when a "
            "cell is zero.",
        ],
        "pre_conditions": [
            "2x2 counts with exposed/unexposed rows and "
            "outcome+/- columns from a cohort or "
            "cross-sectional sample.",
            "Denominators (row totals) are positive in both " "exposure groups.",
        ],
        "failure_modes": [
            {
                "symptom": "Applied to case-control data where "
                "baseline risk is not sampled, so RR "
                "is not identified.",
                "exception": "MethodIncompatibility",
                "remedy": "Report the odds ratio for "
                "case-control designs instead of RR.",
                "alternative": "odds_ratio",
            },
            {
                "symptom": "Zero events in a group triggers the "
                "Haldane correction and a wide, "
                "approximate CI.",
                "exception": "AssumptionWarning",
                "remedy": "Report the corrected estimate "
                "cautiously or fit a "
                "log-binomial/Poisson model.",
                "alternative": "poisson",
            },
        ],
        "alternatives": ["odds_ratio", "risk_difference", "incidence_rate_ratio"],
        "typical_n_min": 20,
    },
    "rif_decomposition": {
        "assumptions": [
            "The recentered influence function for the chosen "
            "distributional statistic (quantile, variance, Gini, "
            "etc.) is a valid first-order approximation, so its "
            "expectation recovers the statistic "
            "(Firpo-Fortin-Lemieux 2009).",
            "The aggregate Oaxaca-Blinder split into explained "
            "(endowments) vs unexplained (coefficients) requires "
            "no omitted covariates correlated with group and a "
            "correctly specified RIF regression.",
            "Detailed (per-covariate) decompositions assume "
            "path/normalization invariance and, for the "
            "unexplained part, an ignorable reference-group "
            "choice.",
        ],
        "pre_conditions": [
            "group is a binary 0/1 indicator and reference in " "{0,1}.",
            "Covariates and the target distributional "
            "statistic are well-defined in both groups.",
            "Both groups have enough observations to fit the "
            "RIF regression at the chosen statistic.",
        ],
        "failure_modes": [
            {
                "symptom": "RIF for a tail quantile is noisy "
                "where the density is near zero, "
                "giving unstable shares",
                "exception": "NumericalInstability",
                "remedy": "Avoid extreme quantiles or "
                "smooth/bootstrap the density "
                "estimate underlying the RIF.",
                "alternative": "dfl_decompose",
            },
            {
                "symptom": "Limited covariate overlap between "
                "groups makes the explained "
                "component unreliable (specification "
                "error)",
                "exception": "AssumptionWarning",
                "remedy": "Trim to the common support or use a "
                "reweighting-based decomposition.",
                "alternative": "dfl_decompose",
            },
        ],
        "alternatives": ["oaxaca", "dfl_decompose"],
        "typical_n_min": 100,
    },
    "risk_difference": {
        "assumptions": [
            "The 2x2 table is from a cohort or cross-sectional "
            "design so absolute risks per group are estimable.",
            "Observations are independent within each exposure " "arm.",
            "The Wald CI assumes normality (poor near risk 0 or "
            "1); Newcombe's hybrid-score CI corrects the boundary "
            "overshoot.",
        ],
        "pre_conditions": [
            "2x2 counts with exposed/unexposed rows and " "outcome+/- columns.",
            "Both group denominators are positive.",
        ],
        "failure_modes": [
            {
                "symptom": "Wald CI overshoots below -1 or "
                "above 1 when a proportion is near a "
                "boundary.",
                "exception": "(none — informational)",
                "remedy": "Use method='newcombe' for a "
                "boundary-respecting score interval.",
                "alternative": "relative_risk",
            },
            {
                "symptom": "Very small per-group n yields an "
                "unstable, overly wide difference.",
                "exception": "DataInsufficient",
                "remedy": "Increase sample size or report an "
                "exact interval; consider a ratio "
                "measure.",
                "alternative": "relative_risk",
            },
        ],
        "alternatives": ["relative_risk", "odds_ratio", "attributable_risk"],
        "typical_n_min": 30,
    },
    "roc_curve": {
        "assumptions": [
            "y_true is a binary reference label and scores are "
            "continuous with higher values indicating the "
            "positive class.",
            "Observations are independent, so the Hanley-McNeil "
            "AUC SE (based on rank statistics) is valid.",
            "Class labels are correctly specified (mislabeling "
            "inverts the curve below the diagonal).",
        ],
        "pre_conditions": [
            "y_true in {0,1} and a same-length continuous " "score vector.",
            "Both classes (at least one positive and one " "negative) are present.",
        ],
        "failure_modes": [
            {
                "symptom": "Only one outcome class present so "
                "TPR/FPR and AUC are undefined.",
                "exception": "ValueError",
                "remedy": "Ensure both positive and negative "
                "cases are in y_true before computing "
                "the ROC.",
                "alternative": "sensitivity_specificity",
            },
            {
                "symptom": "Severe class imbalance makes AUC "
                "optimistic about clinical "
                "usefulness.",
                "exception": "AssumptionWarning",
                "remedy": "Complement AUC with precision-recall "
                "or threshold-specific "
                "sensitivity/specificity.",
                "alternative": "sensitivity_specificity",
            },
        ],
        "alternatives": ["sensitivity_specificity", "cohen_kappa"],
        "typical_n_min": 50,
    },
    "rwd_rct_concordance": {
        "assumptions": [
            "The RCT estimate is the unbiased benchmark target; "
            "concordance is judged relative to it",
            "RWD and RCT estimate the same estimand for "
            "comparable (or successfully transported) populations",
            "The RCT standard error is a valid measure of "
            "benchmark uncertainty at level alpha",
        ],
        "pre_conditions": [
            "A scalar RCT estimate with its SE plus a scalar "
            "RWD estimate to compare against it",
            "Both estimates are on the same scale and target " "the same contrast",
        ],
        "failure_modes": [
            {
                "symptom": "rct_se is zero, negative, or NaN so "
                "the concordance interval is "
                "undefined",
                "exception": "ValueError",
                "remedy": "Provide a strictly positive finite "
                "RCT standard error and numeric point "
                "estimates",
                "alternative": "synthesise_evidence",
            },
            {
                "symptom": "RWD and RCT target different "
                "populations, so a real divergence "
                "is reported as discordance",
                "exception": "AssumptionWarning",
                "remedy": "Transport the RWD or RCT estimate to "
                "a common target population before "
                "comparing",
                "alternative": "identify_transport",
            },
        ],
        "alternatives": [
            "synthesise_evidence",
            "heterogeneity_of_effect",
            "identify_transport",
        ],
        "typical_n_min": 1,
    },
    "sensemakr": {
        "assumptions": [
            "Cinelli-Hazlett (2020) omitted-variable sensitivity "
            "in the partial-R-squared framework; the Robustness "
            "Value is the minimum partial R-squared a confounder "
            "needs with both treatment and outcome to nullify the "
            "estimate.",
            "Outcome is modeled by OLS of y on treat plus "
            "observed controls; benchmarking calibrates "
            "confounder strength relative to named observed "
            "controls.",
            "rv_q is the RV to change the point-estimate sign; "
            "rv_qa is the RV to lose significance at alpha.",
            "Bias is attributed to a single (or grouped) "
            "unobserved confounder acting linearly.",
        ],
        "pre_conditions": [
            "A DataFrame with outcome, treatment, and observed " "control columns",
            "At least one control to anchor the benchmark " "comparison",
        ],
        "failure_modes": [
            {
                "symptom": "Benchmark names a control not "
                "present in the controls list or "
                "data",
                "exception": "KeyError",
                "remedy": "Pass benchmark names that are a "
                "subset of the controls actually "
                "included in the regression.",
                "alternative": "",
            },
            {
                "symptom": "Perfectly collinear controls make "
                "the partial-R-squared decomposition "
                "unstable",
                "exception": "NumericalInstability",
                "remedy": "Drop redundant collinear controls "
                "before computing the robustness "
                "value.",
                "alternative": "oster_bounds",
            },
        ],
        "alternatives": ["oster_bounds", "evalue", "unified_sensitivity"],
        "typical_n_min": 50,
    },
    "sensitivity_specificity": {
        "assumptions": [
            "y_true is an error-free reference standard against which "
            "y_pred is evaluated (no reference-test "
            "misclassification).",
            "Sensitivity and specificity are intrinsic test "
            "properties assumed stable across the case/control "
            "spectrum.",
            "Wilson-score CIs assume independent binary " "classifications.",
        ],
        "pre_conditions": [
            "Either equal-length binary y_true/y_pred vectors, "
            "or precomputed tp/fn/fp/tn confusion cells.",
            "Both diseased (positives) and non-diseased "
            "(negatives) units are represented.",
        ],
        "failure_modes": [
            {
                "symptom": "No true positives (or no true "
                "negatives) makes sensitivity (or "
                "specificity) undefined.",
                "exception": "ZeroDivisionError",
                "remedy": "Ensure the reference labels contain "
                "both classes; report Wilson CIs "
                "which handle small counts.",
                "alternative": "roc_curve",
            },
            {
                "symptom": "Imperfect reference standard biases "
                "both metrics "
                "(verification/incorporation bias).",
                "exception": "AssumptionWarning",
                "remedy": "Use a verified reference standard or "
                "latent-class correction and report "
                "the limitation.",
                "alternative": "cohen_kappa",
            },
        ],
        "alternatives": ["roc_curve", "cohen_kappa", "odds_ratio"],
        "typical_n_min": 30,
    },
    "sharp_ope_unobserved": {
        "assumptions": [
            "Marginal sensitivity model: true propensity deviates "
            "from the estimated one by at most a factor Gamma",
            "Positivity under the logging policy for the target " "policy's support",
            "Logged rewards are observed for the chosen action "
            "only (bandit feedback)",
        ],
        "pre_conditions": [
            "Logged (action, reward, logging_prob, "
            "target_prob) tuples, one row per interaction",
            "A sensitivity constant gamma >= 1 chosen a priori "
            "(gamma=1 recovers IPS)",
        ],
        "failure_modes": [
            {
                "symptom": "gamma < 1 supplied so the " "sensitivity region is empty",
                "exception": "ValueError",
                "remedy": "Pass gamma >= 1; use gamma=1 to "
                "recover the point IPS estimate",
                "alternative": "policy_value",
            },
            {
                "symptom": "Bounds collapse to a degenerate or "
                "implausibly wide interval",
                "exception": "(none — informational)",
                "remedy": "Re-estimate logging_prob, trim "
                "extreme weights, or report bounds "
                "across a range of gamma",
                "alternative": "aipw",
            },
        ],
        "alternatives": ["policy_value", "aipw", "ipw"],
        "typical_n_min": 500,
    },
    "shift_share_political": {
        "assumptions": [
            "Bartik/shift-share identification: either exposure "
            "shares are exogenous "
            "(Goldsmith-Pinkham-Sorkin-Swift) or the national "
            "shocks are as-good-as-randomly assigned "
            "(Borusyak-Hull-Jaravel)",
            "Exposure shares sum sensibly across industries per "
            "unit and are pre-determined relative to the "
            "long-difference outcome",
            "First-stage relevance: the constructed Bartik "
            "instrument is correlated with the endogenous "
            "long-difference in the treatment",
            "No unit-level confounder correlated with both shares "
            "(or shocks) and the outcome change net of "
            "covariates",
        ],
        "pre_conditions": [
            "Long-format unit x time panel with first and last "
            "periods usable to form per-unit long-differences",
            "Unit x industry exposure-share matrix plus an "
            "industry-indexed national shock vector with "
            "matching labels",
        ],
        "failure_modes": [
            {
                "symptom": "shares row index or shocks index "
                "does not align with unit IDs / "
                "industry columns, so the instrument "
                "cannot be assembled",
                "exception": "KeyError",
                "remedy": "Reindex shares to the unit IDs and "
                "align the shocks index to the shares "
                "columns before calling",
                "alternative": "bartik",
            },
            {
                "symptom": "Weak instrument: the Bartik shifter "
                "barely moves the endogenous "
                "variable, producing an unstable, "
                "wide IV estimate",
                "exception": "IdentificationFailure",
                "remedy": "Check the first-stage F and "
                "share-balance diagnostic; enable "
                "leave_one_out or aggregate sparse "
                "industries",
                "alternative": "iv",
            },
        ],
        "alternatives": ["bartik", "shift_share_political_panel", "iv"],
        "typical_n_min": 30,
    },
    "shift_share_political_panel": {
        "assumptions": [
            "Shift-share exogeneity holds period-by-period: "
            "either time-varying shares are exogenous (GPSS) or "
            "the period-specific shocks are as-good-as-random "
            "(BHJ)",
            "Two-way fixed effects absorb unit and time "
            "confounders; remaining variation in the period "
            "Bartik instrument identifies the structural effect",
            "Period-specific instrument is relevant in the pooled "
            "2SLS first stage across all periods",
            "Cluster structure (default unit) correctly captures "
            "within-cluster dependence for valid panel SEs",
        ],
        "pre_conditions": [
            "Long-format unit x time panel (multiple periods) "
            "with outcome and endogenous columns",
            "Shares and shocks supplied per period with "
            "consistent unit and industry labels across "
            "periods",
        ],
        "failure_modes": [
            {
                "symptom": "Time-varying shares/shocks supplied "
                "as dicts miss a period present in "
                "the panel, so the period instrument "
                "cannot be built",
                "exception": "KeyError",
                "remedy": "Provide a shares/shocks entry for "
                "every time value in the data, or "
                "pass a time-invariant structure "
                "instead",
                "alternative": "shift_share_political",
            },
            {
                "symptom": "Two-way FE absorbs nearly all "
                "instrument variation, leaving a "
                "weak first stage and unstable "
                "estimate",
                "exception": "IdentificationFailure",
                "remedy": "Switch fe to 'unit' or 'time', "
                "verify the first-stage F, and "
                "confirm the Bartik instrument "
                "retains within-panel variation",
                "alternative": "iv",
            },
        ],
        "alternatives": ["shift_share_political", "bartik", "iv"],
        "typical_n_min": 60,
    },
    "spec_curve": {
        "assumptions": [
            "Simonsohn-Simmons-Nelson (2020) specification-curve: "
            "each combination of controls, SE type, sub-sample, "
            "and outcome transform is a defensible specification "
            "of the same x->y effect.",
            "The set of analytical choices is "
            "researcher-enumerated and is assumed to span the "
            "reasonable garden of forking paths, not just "
            "confirmatory ones.",
            "The curve summarizes effect stability across "
            "specifications; it does not by itself license a "
            "causal interpretation of any single fit.",
            "Clustered SEs require a valid cluster_var aligned " "with the data.",
        ],
        "pre_conditions": [
            "A DataFrame with y, x, and the candidate "
            "control/subset/transform columns",
            "At least one control-set, SE-type, or transform " "axis to vary",
        ],
        "failure_modes": [
            {
                "symptom": "cluster requested in se_types but "
                "no cluster_var supplied, or subset "
                "masks misaligned with the data "
                "index",
                "exception": "ValueError",
                "remedy": "Supply cluster_var when using "
                "clustered SEs and align subset "
                "boolean masks to the data index.",
                "alternative": "",
            },
            {
                "symptom": "Combinatorial explosion of "
                "specifications makes the run slow "
                "or memory-heavy",
                "exception": "(none — informational)",
                "remedy": "Prune the control-set / transform "
                "grid to the specifications you can "
                "actually defend.",
                "alternative": "diagnose_result",
            },
        ],
        "alternatives": ["diagnose_result", "unified_sensitivity", "sensemakr"],
        "typical_n_min": 50,
    },
    "surrogate_index": {
        "assumptions": [
            "Surrogacy: conditional on surrogates and covariates "
            "the treatment has no direct effect on the long-term "
            "outcome (strictly stronger than ignorability)",
            "Comparability / surrogate-stability: the conditional "
            "outcome mean estimated in the observational sample "
            "equals that conditional mean in the experimental "
            "sample",
            "Common support of the surrogate (and covariate) "
            "distribution across the two samples",
            "Treatment is randomised (or ignorable) in the " "experimental sample",
        ],
        "pre_conditions": [
            "Experimental sample with treatment and surrogates "
            "(long-term outcome need not be present)",
            "Observational sample with the same surrogates "
            "plus the long-term outcome",
            "Surrogate (and optional covariate) columns shared " "by both samples",
        ],
        "failure_modes": [
            {
                "symptom": "A surrogate or covariate column is "
                "absent from one of the two input "
                "frames",
                "exception": "KeyError",
                "remedy": "Confirm every surrogate and "
                "covariate name exists in both "
                "experimental and observational "
                "frames",
                "alternative": "long_term_from_short",
            },
            {
                "symptom": "Surrogacy fails (surrogate does not "
                "capture the full treatment->outcome "
                "path), biasing the long-term ATE",
                "exception": "AssumptionViolation",
                "remedy": "Defend surrogacy with a placebo "
                "long-term outcome in a validation "
                "sample or add more surrogates",
                "alternative": "proximal_surrogate_index",
            },
            {
                "symptom": "Observational surrogate support "
                "does not cover the experimental "
                "distribution, so the conditional "
                "mean extrapolates",
                "exception": "DataInsufficient",
                "remedy": "Restrict to the overlapping "
                "surrogate region or enrich the "
                "observational sample",
                "alternative": "identify_transport",
            },
        ],
        "alternatives": [
            "proximal_surrogate_index",
            "long_term_from_short",
            "identify_transport",
        ],
        "typical_n_min": 500,
    },
    "swig": {
        "assumptions": [
            "The base graph is a DAG (acyclicity) on which the "
            "node-splitting intervention operation is "
            "well-defined",
            "Counterfactual consistency / modularity: intervening "
            "on X fixes it to the labeled value, severing only "
            "incoming edges to X's fixed copy",
            "Intervention targets are nodes present in the DAG; "
            "each gets split into a random and a fixed (do) "
            "vertex",
        ],
        "pre_conditions": [
            "A statspai.dag.DAG instance",
            "An intervention mapping var->label or an iterable "
            "of variable names to fix",
        ],
        "failure_modes": [
            {
                "symptom": "intervention names a variable that "
                "is not a node in the DAG",
                "exception": "KeyError",
                "remedy": "Restrict intervention targets to "
                "existing dag.nodes; verify with "
                "sp.dag before constructing the SWIG.",
                "alternative": "dag",
            },
            {
                "symptom": "intervention argument is neither a "
                "dict nor an iterable of names",
                "exception": "TypeError",
                "remedy": "Pass a dict for explicit labels or a "
                "list to default labels to lowercase.",
                "alternative": "identify",
            },
        ],
        "alternatives": ["dag", "identify", "pc_algorithm", "llm_dag_constrained"],
        "typical_n_min": 1,
    },
    "synthesise_evidence": {
        "assumptions": [
            "RCT and RWD estimates are each (conditionally) "
            "unbiased for the target estimand once the "
            "transport_shift is applied",
            "Inverse-variance pooling assumes the two estimates "
            "are independent with correctly specified standard "
            "errors",
            "transport_shift correctly transports the RCT "
            "estimate to the RWD target population, with "
            "transport_shift_se capturing its uncertainty (added "
            "in quadrature)",
        ],
        "pre_conditions": [
            "A scalar RCT estimate+SE and a scalar (already "
            "transport-weighted) RWD estimate+SE",
            "A chosen weight_mode ('inverse_variance' or "
            "'rct_heavy') and, if transporting, a "
            "transport_shift and its SE",
        ],
        "failure_modes": [
            {
                "symptom": "Either supplied SE is non-positive "
                "or weight_mode is not a recognised "
                "option",
                "exception": "ValueError",
                "remedy": "Pass positive finite SEs and "
                "weight_mode in "
                "{'inverse_variance','rct_heavy'}",
                "alternative": "heterogeneity_of_effect",
            },
            {
                "symptom": "RCT and RWD estimates are strongly "
                "discordant, so inverse-variance "
                "pooling masks a real conflict",
                "exception": "AssumptionWarning",
                "remedy": "Run a concordance check first and "
                "only pool when the two sources agree",
                "alternative": "rwd_rct_concordance",
            },
        ],
        "alternatives": [
            "rwd_rct_concordance",
            "heterogeneity_of_effect",
            "identify_transport",
        ],
        "typical_n_min": 1,
    },
    "unified_sensitivity": {
        "assumptions": [
            "Dashboard that dispatches every applicable "
            "sensitivity method (E-value, Cinelli-Hazlett RV, "
            "Oster's delta) to one fitted result.",
            "Oster's delta requires the short- and "
            "long-regression R-squared (r2_treated, "
            "r2_controlled) plus beta_uncontrolled; rho_max is "
            "Oster's R_max bound (default 1.0).",
            "The Sensemakr component needs the raw estimation "
            "data via data/y/treat/controls; result objects do "
            "not carry it, so it is skipped (with a note) when "
            "those are omitted.",
            "Each component inherits its own identifying "
            "assumptions (linear OVB for RV/Oster, RR scale for "
            "E-value); the panel does not reconcile conflicting "
            "ones.",
            "The result must expose estimate, se, and ci for the " "methods to run.",
        ],
        "pre_conditions": [
            "A result with point estimate, SE, and CI " "attributes",
            "r2_treated and r2_controlled for the " "Oster's-delta component",
        ],
        "failure_modes": [
            {
                "symptom": "Result lacks estimate/se/ci, so no "
                "sensitivity component can be "
                "computed",
                "exception": "TypeError",
                "remedy": "Pass a fitted result exposing "
                "estimate, se, and ci rather than a "
                "raw scalar or DataFrame.",
                "alternative": "evalue",
            },
            {
                "symptom": "Oster's delta omitted because "
                "r2_treated / r2_controlled were not "
                "provided",
                "exception": "(none — informational)",
                "remedy": "Pass the short- and long-regression "
                "R-squared values to enable the Oster "
                "component.",
                "alternative": "oster_bounds",
            },
            {
                "symptom": "Sensemakr component skipped with a "
                "note because the raw data were not "
                "supplied",
                "exception": "(none — informational)",
                "remedy": "Pass data=, y=, treat=, controls= to "
                "unified_sensitivity / .sensitivity(), "
                "or call sp.sensemakr(data, y, treat, "
                "controls) directly.",
                "alternative": "sensemakr",
            },
        ],
        "alternatives": ["evalue", "sensemakr", "oster_bounds", "diagnose_result"],
        "typical_n_min": 30,
    },
    "var": {
        "assumptions": [
            "All system variables are jointly "
            "covariance-stationary (no unit roots / cointegration "
            "left unmodeled)",
            "Lag order p is adequate so residuals are white "
            "noise; under-fitting biases IRFs and Granger tests",
            "Reduced-form errors are serially uncorrelated; "
            "structural IRF identification relies on the chosen "
            "ordering/recursive scheme",
            "No omitted variable drives the included series " "jointly",
        ],
        "pre_conditions": [
            "Multivariate (>=2 column) time-ordered DataFrame "
            "of comparable-frequency series",
            "Sample length large relative to k*p+trend "
            "parameters to estimate each equation",
        ],
        "failure_modes": [
            {
                "symptom": "Coefficient covariance is singular "
                "or IRFs diverge because k^2*p "
                "parameters exceed available "
                "observations",
                "exception": "DataInsufficient",
                "remedy": "Reduce lags, drop variables, or "
                "extend the sample so n is much "
                "larger than k*p",
                "alternative": "local_projections",
            },
            {
                "symptom": "Explosive IRFs / non-decaying "
                "responses from a non-stationary "
                "(unit-root or trending) system",
                "exception": "NumericalInstability",
                "remedy": "Difference or detrend the series "
                "first, or use trend='ct', and "
                "confirm stationarity before fitting",
                "alternative": "arima",
            },
        ],
        "alternatives": ["local_projections", "arima", "iv"],
        "typical_n_min": 100,
    },
    "wooldridge_prod": {
        "assumptions": [
            "Scalar productivity monotone in the proxy, "
            "controlled by nonparametric h(m,k); first-order "
            "Markov productivity g(omega_{t-1})",
            "Labor and capital coefficients identified jointly "
            "via a stacked level + productivity-substituted "
            "moment system (one-step GMM with identity weight = "
            "NLS)",
            "Instruments equal the regressors; capital "
            "predetermined, proxy positive",
        ],
        "pre_conditions": [
            "Long firm-year panel with log output, free "
            "input(s), state input(s), and a productivity "
            "proxy",
            "Two or more consecutive periods per firm so "
            "lagged productivity g(omega_{t-1}) is available",
        ],
        "failure_modes": [
            {
                "symptom": "Stacked NLS objective fails to "
                "converge with high-degree h and g "
                "polynomials",
                "exception": "ConvergenceFailure",
                "remedy": "Lower polynomial_degree and "
                "productivity_degree (defaults are 2 "
                "because the joint problem is "
                "high-dimensional).",
                "alternative": "levinsohn_petrin",
            },
            {
                "symptom": "Joint moment system "
                "under-identified due to "
                "insufficient input variation",
                "exception": "IdentificationFailure",
                "remedy": "Add cross-sectional/time variation "
                "or reduce the control-function "
                "degree.",
                "alternative": "ackerberg_caves_frazer",
            },
        ],
        "alternatives": [
            "ackerberg_caves_frazer",
            "levinsohn_petrin",
            "olley_pakes",
            "gmm",
        ],
        "typical_n_min": 500,
    },
    "xtfrontier": {
        "assumptions": [
            "Panel composed error with unit inefficiency: "
            "time-invariant (Pitt-Lee), time-varying decay "
            "(Battese-Coelli 1992), or inefficiency-effects "
            "(BC95)",
            "Inefficiency from half-normal or truncated-normal "
            "(BC95 always truncated-normal)",
            "True FE/RE (Greene 2005) separate persistent unit "
            "heterogeneity from transient inefficiency; TFE needs "
            "adequate T",
        ],
        "pre_conditions": [
            "Long-format panel data",
            "Panel id (and time, required for model='tvd')",
            "emean inefficiency determinants required for " "model='bc95'",
        ],
        "failure_modes": [
            {
                "symptom": "model='tvd' or 'tfe' without a time " "variable / short T",
                "exception": "DataInsufficient",
                "remedy": "Provide time= and ensure adequate T, "
                "or use model='ti' for time-invariant "
                "inefficiency",
                "alternative": "frontier",
            },
            {
                "symptom": "Quadrature for the "
                "true-random-effects model fails to "
                "converge",
                "exception": "ConvergenceWarning",
                "remedy": "Increase quadrature nodes/maxiter or "
                "switch to model='tfe'",
                "alternative": "frontier",
            },
            {
                "symptom": "model='bc95' called without emean " "determinants",
                "exception": "ValueError",
                "remedy": "Supply emean= columns for the " "inefficiency-effects model",
                "alternative": "frontier",
            },
        ],
        "alternatives": ["frontier", "regress", "feols"],
        "typical_n_min": 300,
    },
    "xtnbreg": {
        "assumptions": [
            "Panel count overdispersion (NB-2): model='fe' adds "
            "explicit entity dummies, model='re' is a "
            "random-intercept NB model",
            "Log-linear conditional mean within panel units",
            "RE random intercepts are normally distributed and "
            "independent of covariates",
        ],
        "pre_conditions": [
            "Non-negative integer count outcome",
            "Panel/entity identifier supplied via entity= or " "'| id' in formula",
        ],
        "failure_modes": [
            {
                "symptom": "FE with many singleton units "
                "inflates dummy count and degrades "
                "estimates",
                "exception": "DataInsufficient",
                "remedy": "Require multiple within-unit "
                "observations or switch to the RE "
                "model",
                "alternative": "nbreg",
            },
            {
                "symptom": "RE model quadrature fails to " "converge",
                "exception": "ConvergenceWarning",
                "remedy": "Increase quadrature points/maxiter " "or use model='fe'",
                "alternative": "nbreg",
            },
            {
                "symptom": "Entity identifier missing for fixed " "effects",
                "exception": "ValueError",
                "remedy": "Pass entity= or include '| id' in " "the formula",
                "alternative": "nbreg",
            },
        ],
        "alternatives": ["nbreg", "poisson"],
        "typical_n_min": 300,
    },
    # ------------------------------------------------------------------ #
    # Proximal / negative-control identification family.  These ship on
    # sp.<name> and were added to __all__ in the 2026-06 registry-drift
    # repair; cards below give agents the identifying assumptions and the
    # failure modes that distinguish a clean run from a silent violation.
    # ------------------------------------------------------------------ #
    "double_negative_control": {
        "assumptions": [
            "Two valid proxies of the hidden confounder U: a "
            "treatment-confounding proxy and an outcome-confounding proxy",
            "Negative-control exposure has no causal effect on Y given "
            "(U, X) — exclusion restriction on the NCE arm",
            "Negative-control outcome is not caused by the treatment D " "given (U, X)",
            "Completeness / bridge conditions hold so the just-identified "
            "2SLS de-biases U (Miao et al. 2018; Shi et al. 2020)",
        ],
        "pre_conditions": [
            "data has columns for outcome, treatment, a negative-control "
            "exposure (nce) and a negative-control outcome (nco)",
            "nce and nco share the unmeasured confounder but satisfy the "
            "respective exclusion restrictions",
        ],
        "failure_modes": [
            {
                "symptom": "Weak proxy: NCE barely correlated with NCO "
                "(weak-instrument analogue) — unstable ATE",
                "exception": "NumericalInstability",
                "remedy": "Use stronger proxies or the doubly-robust "
                "regression-based estimator instead of just-identified 2SLS",
                "alternative": "proximal_regression",
            },
            {
                "symptom": "A proxy violates its exclusion restriction "
                "(direct effect on Y or caused by D) — biased ATE",
                "exception": "IdentificationFailure",
                "remedy": "Validate the negative controls separately or fall "
                "back to a sensitivity analysis",
                "alternative": "sensemakr",
            },
        ],
        "alternatives": ["proximal", "proximal_regression", "frontdoor"],
        "typical_n_min": 200,
    },
    "proximal_regression": {
        "assumptions": [
            "A treatment-inducing confounding proxy Z and an "
            "outcome-inducing confounding proxy W are observed",
            "Outcome and treatment confounding bridge functions exist "
            "(proximal-g and proximal-h completeness conditions)",
            "Latent confounding is fully captured by (Z, W, X) — no "
            "residual unmeasured confounding outside the proxy span",
        ],
        "pre_conditions": [
            "data has outcome, binary treatment, z_proxy and w_proxy columns",
            "treatment is binary for the bridge logistic step",
        ],
        "failure_modes": [
            {
                "symptom": "Treatment-bridge logistic regression fails to "
                "converge; propensity collapses to the marginal P(D=1)",
                "exception": "ConvergenceWarning",
                "remedy": "Check detail['propensity_fallback']; the DR "
                "correction is neutralised — improve proxies or covariates",
                "alternative": "double_negative_control",
            },
        ],
        "alternatives": ["proximal", "double_negative_control", "frontdoor"],
        "typical_n_min": 200,
    },
    "negative_control_exposure": {
        "assumptions": [
            "The negative-control exposure has no causal effect on the "
            "outcome by design (Lipsitch et al. 2010)",
            "It shares the same confounding structure as the real exposure",
        ],
        "pre_conditions": [
            "data has the outcome and a negative-control exposure column",
        ],
        "failure_modes": [
            {
                "symptom": "Significant coefficient on the negative-control "
                "exposure signals residual confounding, not an effect",
                "exception": "AssumptionWarning",
                "remedy": "Treat a non-zero coefficient as a confounding "
                "alarm; add covariates or run a sensitivity analysis",
                "alternative": "negative_control_outcome",
            },
        ],
        "alternatives": ["negative_control_outcome", "sensemakr", "evalue"],
        "typical_n_min": 100,
    },
    "negative_control_outcome": {
        "assumptions": [
            "The negative-control outcome is not caused by the treatment "
            "(Lipsitch-style calibration)",
            "It shares confounders with the real outcome",
        ],
        "pre_conditions": [
            "data has a negative-control outcome and a treatment column",
        ],
        "failure_modes": [
            {
                "symptom": "Coefficient on treatment differs significantly "
                "from zero — residual confounding detected",
                "exception": "AssumptionWarning",
                "remedy": "Condition on more covariates or quantify the "
                "implied bias with a sensitivity analysis",
                "alternative": "negative_control_exposure",
            },
        ],
        "alternatives": ["negative_control_exposure", "sensemakr", "evalue"],
        "typical_n_min": 100,
    },
    # ------------------------------------------------------------------ #
    # Off-policy evaluation estimators (policy_learning.ope).
    # ------------------------------------------------------------------ #
    "ips": {
        "assumptions": [
            "Logging (behaviour) policy propensities are known or "
            "correctly estimated",
            "Positivity / common support: the logging policy assigns "
            "positive probability to every action the target policy takes",
            "No unmeasured confounding in the logged data",
        ],
        "pre_conditions": [
            "X (context), A (logged action), R (reward) and logging "
            "propensities are available",
        ],
        "failure_modes": [
            {
                "symptom": "High-variance estimate from extreme importance "
                "weights when the target policy diverges from logging",
                "exception": "NumericalInstability",
                "remedy": "Use self-normalised IPS (snips) or the "
                "doubly-robust estimator to reduce variance",
                "alternative": "snips",
            },
        ],
        "alternatives": ["snips", "doubly_robust", "direct_method"],
        "typical_n_min": 500,
    },
    "snips": {
        "assumptions": [
            "Same identification conditions as IPS (known propensities, "
            "positivity, no unmeasured confounding)",
            "Self-normalisation trades a small bias for large variance "
            "reduction under heavy importance weights",
        ],
        "pre_conditions": [
            "X (context), A (logged action), R (reward) and logging "
            "propensities are available",
        ],
        "failure_modes": [
            {
                "symptom": "Residual bias when effective sample size is tiny "
                "(few logged actions overlap the target policy)",
                "exception": "DataInsufficient",
                "remedy": "Collect more on-support logged data or switch to "
                "the doubly-robust estimator",
                "alternative": "doubly_robust",
            },
        ],
        "alternatives": ["ips", "doubly_robust", "direct_method"],
        "typical_n_min": 500,
    },
    "doubly_robust": {
        "assumptions": [
            "Doubly robust: consistent if EITHER the outcome (Q) model OR "
            "the logging propensity model is correctly specified",
            "Positivity / common support holds",
            "No unmeasured confounding in the logged data",
        ],
        "pre_conditions": [
            "X, A, R, logging propensities and a fitted Q-model (or its "
            "predictions) are available",
        ],
        "failure_modes": [
            {
                "symptom": "Both nuisance models misspecified — DR guarantee "
                "is lost and the estimate is biased",
                "exception": "AssumptionWarning",
                "remedy": "Cross-fit the nuisances or validate the Q-model "
                "and propensity fit separately",
                "alternative": "snips",
            },
        ],
        "alternatives": ["ips", "snips", "direct_method"],
        "typical_n_min": 500,
    },
    "direct_method": {
        "assumptions": [
            "Plug-in outcome regression (Q-model) is correctly specified",
            "No unmeasured confounding in the logged data",
        ],
        "pre_conditions": [
            "X (context), A (logged action), R (reward) are available to "
            "fit the outcome model",
        ],
        "failure_modes": [
            {
                "symptom": "Model misspecification bias — the Q-model "
                "extrapolates outside the logged action support",
                "exception": "AssumptionWarning",
                "remedy": "Prefer the doubly-robust estimator, which is "
                "robust to Q-model misspecification",
                "alternative": "doubly_robust",
            },
        ],
        "alternatives": ["doubly_robust", "ips", "snips"],
        "typical_n_min": 500,
    },
    # ------------------------------------------------------------------ #
    # Longitudinal TMLE and neural dose-response estimators.
    # ------------------------------------------------------------------ #
    "ltmle": {
        "assumptions": [
            "Sequential exchangeability (no unmeasured time-varying "
            "confounding) at every treatment time point",
            "Positivity at each time point conditional on the past",
            "Correctly specified outcome (Q) and treatment (g) models, "
            "though TMLE is doubly robust to one of them",
            "Consistency / SUTVA across the longitudinal regime",
        ],
        "pre_conditions": [
            "Long-format data with time-varying treatment, confounders and "
            "the static regime contrast of interest",
        ],
        "failure_modes": [
            {
                "symptom": "Near-positivity violation: estimated treatment "
                "probabilities approach 0 or 1, inflating variance",
                "exception": "NumericalInstability",
                "remedy": "Bound (truncate) the cumulative weights or use a "
                "marginal structural model with stabilised weights",
                "alternative": "msm",
            },
        ],
        "alternatives": ["tmle", "g_computation", "msm"],
        "typical_n_min": 500,
    },
    "vcnet": {
        "assumptions": [
            "Unconfoundedness given covariates X (no hidden confounding of "
            "the continuous treatment)",
            "Positivity over the dose: every dose has support across X",
            "The dose-response curve is smooth (varying-coefficient prior)",
        ],
        "pre_conditions": [
            "data with a continuous treatment (dose), outcome and covariates",
            "torch is installed (neural extra) — imported lazily",
        ],
        "failure_modes": [
            {
                "symptom": "Sparse support at extreme doses yields an "
                "unreliable dose-response curve there",
                "exception": "DataInsufficient",
                "remedy": "Restrict the reported dose range to the supported "
                "region or compare against scigan",
                "alternative": "scigan",
            },
        ],
        "alternatives": ["scigan", "dose_response"],
        "typical_n_min": 500,
    },
    "scigan": {
        "assumptions": [
            "Unconfoundedness given covariates X for the continuous " "treatment",
            "Positivity over the dose support",
            "The adversarial generator recovers the counterfactual dose "
            "distribution (Bica et al. 2020)",
        ],
        "pre_conditions": [
            "data with a continuous treatment (dose), outcome and covariates",
            "torch is installed (neural extra) — imported lazily",
        ],
        "failure_modes": [
            {
                "symptom": "Unstable adversarial training — dose-response "
                "estimates vary across seeds",
                "exception": "ConvergenceWarning",
                "remedy": "Average across seeds or use the smoother "
                "varying-coefficient estimator",
                "alternative": "vcnet",
            },
        ],
        "alternatives": ["vcnet", "dose_response"],
        "typical_n_min": 500,
    },
    # ------------------------------------------------------------------ #
    # Network-interference orthogonal estimators (interference.orthogonal).
    # ------------------------------------------------------------------ #
    "network_hte": {
        "assumptions": [
            "Partial / network interference: unit i's outcome depends on its "
            "own treatment and a correctly-specified scalar summary of "
            "neighbourhood exposure (e.g. share of neighbours treated)",
            "No unmeasured confounding of the direct and spillover effects "
            "given covariates X",
            "Overlap in both own-treatment and neighbourhood exposure given X",
            "Cross-fit nuisances are consistently estimated (the moment is "
            "Neyman-orthogonal, so first-order insensitive to nuisance error)",
        ],
        "pre_conditions": [
            "data has outcome, unit treatment, a precomputed neighbour-exposure "
            "column and covariates",
            "enough rows for n_folds-fold cross-fitting (>= 10*n_folds)",
        ],
        "failure_modes": [
            {
                "symptom": "Too few complete rows for n_folds cross-fitting",
                "exception": "DataInsufficient",
                "remedy": "Add observations or lower n_folds.",
                "alternative": "spillover",
            },
            {
                "symptom": "Mis-specified exposure mapping biases the spillover "
                "estimate (the scalar summary omits the true interference "
                "structure)",
                "exception": "AssumptionViolation",
                "remedy": "Validate the exposure mapping or use the "
                "design-based Aronow-Samii estimator.",
                "alternative": "network_exposure",
            },
        ],
        "alternatives": ["spillover", "network_exposure", "interference"],
        "typical_n_min": 200,
    },
    "inward_outward_spillover": {
        "assumptions": [
            "Directed-network inward and outward exposure summaries are "
            "correctly constructed (incoming vs outgoing treated-neighbour "
            "shares)",
            "The partially-linear spillover model is correctly specified",
            "No unmeasured confounding of treatment and exposures given X",
        ],
        "pre_conditions": [
            "data has outcome, unit treatment, an inward-exposure and an "
            "outward-exposure column",
        ],
        "failure_modes": [
            {
                "symptom": "Inward and outward exposures are near-collinear, so "
                "the in/out ratio is unstable",
                "exception": "NumericalInstability",
                "remedy": "Check the correlation of the two exposure summaries; "
                "report the components rather than the ratio.",
                "alternative": "network_hte",
            },
        ],
        "alternatives": ["network_hte", "spillover", "interference"],
        "typical_n_min": 200,
    },
}
