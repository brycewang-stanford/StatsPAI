"""Agent-native family-template seed data for the registry.

Pure data: each entry pairs one planning template (assumptions,
pre-conditions, failure modes, alternatives, typical_n_min) with the
estimator entry-points that share it. Consumed by
``statspai.registry._expand_family_seeds``. Kept in its own module so the
long descriptive strings do not inflate registry.py's flake8 baseline.

These are textbook-standard methodological facts (unconfoundedness,
parallel trends, overlap, instrument relevance, ...), NOT bibliographic
citations, so they are outside the project's citation-verification rule.
"""

# flake8: noqa: E501  (descriptive-string data module; long lines are content)
from __future__ import annotations

from typing import Any, Dict, List, Tuple

CAUSAL_FAMILY_SEEDS: List[Tuple[Dict[str, Any], List[str]]] = [
    (
        {
            "pre_conditions": [
                "Panel of one or more treated units plus an untreated donor pool, observed over time.",
                "Pre-treatment window long enough to fit donor weights (rule of thumb: more pre-periods than donors used).",
                "Outcome observed for every unit in every period.",
            ],
            "assumptions": [
                "A convex (or regularized) combination of donor units reproduces the treated unit's pre-treatment outcome path.",
                "No interference: the treatment does not affect the donor units (SUTVA).",
                "No anticipation before the treatment date.",
                "The donor pool is not subject to the same shock as the treated unit.",
            ],
            "failure_modes": [
                {
                    "symptom": "Large pre-treatment RMSPE — the synthetic unit fails to track the treated unit before treatment",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Add donors / predictors, lengthen the pre-period, or use a bias-corrected estimator (sdid, augsynth).",
                    "alternative": "sp.sdid",
                },
                {
                    "symptom": "Placebo / permutation inference shows the estimate is not extreme relative to donors",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Report the placebo distribution honestly; the effect may not be distinguishable from noise.",
                    "alternative": "sp.synth_sensitivity",
                },
            ],
            "alternatives": ["sdid", "augsynth", "gsynth", "callaway_santanna"],
            "typical_n_min": 15,
        },
        [
            "SyntheticControl",
            "sc_estimate",
            "scest",
            "scpi",
            "augsynth",
            "gsynth",
            "mc_synth",
            "robust_synth",
            "demeaned_synth",
            "conformal_synth",
            "multi_outcome_synth",
            "staggered_synth",
            "qqsynth",
            "synth_survival",
            "discos",
            "discos_test",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Panel with treated and control units and a clear treatment date.",
                "Pre-treatment periods available to assess comparability of trends.",
            ],
            "assumptions": [
                "Parallel trends in the absence of treatment, after the synthetic/DiD weighting.",
                "No anticipation and no interference between units (SUTVA).",
                "The control pool's outcome process is stable around the intervention.",
            ],
            "failure_modes": [
                {
                    "symptom": "Weighted pre-treatment trends still diverge between treated and synthetic control",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Inspect the unit/time weights and pre-trend fit; consider event-study DiD with honest bounds.",
                    "alternative": "sp.honest_did",
                },
            ],
            "alternatives": ["synth", "augsynth", "callaway_santanna", "gardner_did"],
            "typical_n_min": 15,
        },
        ["sdid", "synthdid_estimate", "sequential_sdid", "did_estimate"],
    ),
    (
        {
            "pre_conditions": [
                "A continuous running/forcing variable with a known cutoff that (sharply or fuzzily) assigns treatment.",
                "Enough observations in a neighbourhood of the cutoff to fit a local polynomial.",
            ],
            "assumptions": [
                "Conditional expectations of potential outcomes are continuous at the cutoff.",
                "Units cannot precisely manipulate the running variable around the cutoff (no sorting).",
                "For fuzzy designs: monotonicity of treatment take-up at the cutoff.",
            ],
            "failure_modes": [
                {
                    "symptom": "Density of the running variable jumps at the cutoff (manipulation / sorting)",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Run a McCrary / density test (rdplotdensity); if manipulation is present the design is invalid near the cutoff.",
                    "alternative": "sp.rdrandinf",
                },
                {
                    "symptom": "Estimate swings with the bandwidth — results are not robust",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Report a bandwidth-sensitivity curve and use a data-driven MSE-optimal bandwidth.",
                    "alternative": "sp.rdbwselect",
                },
            ],
            "alternatives": ["rdrobust", "rdrandinf", "rdbwselect"],
            "typical_n_min": 500,
        },
        [
            "rd2d",
            "geographic_rd",
            "multi_cutoff_rd",
            "multi_score_rd",
            "boundary_rd",
            "rdhte",
            "rdms",
            "rdmc",
            "rdrandinf",
            "rd_forest",
            "rd_lasso",
            "rkd",
            "rdit",
            "rd_extrapolate",
            "rd_multi_extrapolate",
            "rd_interference",
            "rd_external_validity",
            "rd_bayes_hte",
            "rd_distributional_design",
            "rd_distribution",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Panel or repeated cross-section with a unit (or group) identifier and a time identifier.",
                "At least one never-treated or not-yet-treated comparison group.",
                "Pre-treatment periods to assess parallel trends.",
            ],
            "assumptions": [
                "Conditional parallel trends between treated and comparison groups absent treatment.",
                "No anticipation of treatment before its onset.",
                "Treatment effects may be heterogeneous across cohorts and time (no homogeneity required).",
            ],
            "failure_modes": [
                {
                    "symptom": "Pre-treatment event-study coefficients are jointly non-zero (pre-trend violation)",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Use honest DiD bounds to quantify robustness to trend violations, or condition on covariates.",
                    "alternative": "sp.honest_did",
                },
                {
                    "symptom": "Two-way fixed-effects estimate is contaminated by 'forbidden' comparisons / negative weights",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Use a heterogeneity-robust estimator (Callaway-Sant'Anna, Borusyak et al., Gardner two-stage).",
                    "alternative": "sp.callaway_santanna",
                },
            ],
            "alternatives": ["callaway_santanna", "did", "honest_did"],
            "typical_n_min": 100,
        },
        [
            "gardner_did",
            "did_2stage",
            "borusyak_jaravel_spiess",
            "bjs",
            "etwfe_emfx",
            "twfe_decomposition",
            "breakdown_m",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Pre-treatment covariates measured for treated and control units.",
                "A binary (or low-cardinality) treatment indicator.",
                "Sufficient covariate overlap between treatment arms.",
            ],
            "assumptions": [
                "Unconfoundedness: treatment is as-good-as-random given the measured covariates.",
                "Overlap / common support: every unit has a non-degenerate probability of each treatment.",
                "The covariate set blocks all back-door paths.",
            ],
            "failure_modes": [
                {
                    "symptom": "Poor overlap — extreme propensity scores or few acceptable matches",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Trim or restrict to the common-support region and report the discarded units.",
                    "alternative": "sp.trimming",
                },
                {
                    "symptom": "Covariate imbalance remains after matching/weighting",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Re-specify the balancing model (CBPS, entropy balancing) and re-check standardized mean differences.",
                    "alternative": "sp.ebalance",
                },
            ],
            "alternatives": ["propensity_score", "cbps", "ebalance", "dml"],
            "typical_n_min": 200,
        },
        [
            "cbps",
            "ebalance",
            "genmatch",
            "optimal_match",
            "cardinality_match",
            "sbw",
            "overlap_weights",
            "propensity_score",
            "trimming",
            "psm",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Covariates, a treatment indicator, and an outcome for each unit.",
                "Enough data to fit flexible nuisance models with sample-splitting / cross-fitting.",
            ],
            "assumptions": [
                "Unconfoundedness given the covariates.",
                "Overlap / positivity across the covariate space.",
                "Nuisance functions are estimated consistently; cross-fitting controls overfitting bias.",
            ],
            "failure_modes": [
                {
                    "symptom": "CATE estimates are unstable or extrapolate beyond the covariate support",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Restrict to the overlap region, increase data, or use a doubly-robust learner (DR-/R-learner).",
                    "alternative": "sp.dml",
                },
            ],
            "alternatives": ["dml", "causal_forest", "tmle"],
            "typical_n_min": 500,
        },
        [
            "SLearner",
            "TLearner",
            "XLearner",
            "DRLearner",
            "RLearner",
            "auto_cate",
            "auto_cate_tuned",
            "cluster_cate",
            "focal_cate",
            "predict_cate",
            "cate_eval",
            "compare_metalearners",
            "cate_by_group",
            "gate_test",
            "blp_test",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Covariates, treatment, and outcome with enough data for cross-fitted machine-learning nuisances.",
                "For instrumented variants (PLIV / IIVM): an instrument as well.",
            ],
            "assumptions": [
                "Unconfoundedness (IRM/PLR) or instrument validity (IIVM/PLIV) given the covariates.",
                "Overlap / positivity.",
                "Neyman-orthogonal score plus cross-fitting; nuisance estimators converge fast enough (o(n^-1/4)).",
            ],
            "failure_modes": [
                {
                    "symptom": "Propensity scores near 0/1 — overlap failure inflates variance and bias",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Trim extreme scores, restrict the estimand to the overlap region, or report sensitivity (dml_sensitivity).",
                    "alternative": "sp.dml_sensitivity",
                },
            ],
            "alternatives": ["tmle", "auto_cate", "causal_forest"],
            "typical_n_min": 500,
        },
        [
            "DoubleML",
            "DoubleMLPLR",
            "DoubleMLPLIV",
            "DoubleMLIRM",
            "DoubleMLIIVM",
            "dml_model_averaging",
            "dml_sensitivity",
            "dml_diagnostics",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Covariates, treatment, and outcome; for IV-quantile methods, a valid instrument.",
                "Enough data to estimate the outcome distribution across quantiles.",
            ],
            "assumptions": [
                "Selection-on-observables (unconfoundedness + overlap) or, for IV variants, instrument validity.",
                "For IV-QTE: rank invariance / rank similarity (monotonicity of the structural quantile function).",
            ],
            "failure_modes": [
                {
                    "symptom": "Estimated conditional quantiles cross (non-monotone), or tail quantiles are unstable",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Use rearrangement / monotonization and avoid extreme quantiles where data are sparse.",
                    "alternative": "sp.qte",
                },
            ],
            "alternatives": ["qte", "iv", "dml"],
            "typical_n_min": 500,
        },
        [
            "distributional_te",
            "dist_iv",
            "beyond_average_late",
            "qte_hd_panel",
        ],
    ),
    (
        {
            "pre_conditions": [
                "The data needed for the point-identifying analysis, plus the weakest credible identifying restriction.",
                "For Lee bounds: a binary selection/attrition indicator.",
            ],
            "assumptions": [
                "Only weak (set-identifying) assumptions are imposed; the result is an interval, not a point.",
                "Lee bounds add monotonicity of selection; Oster's delta adds proportional selection on observed vs. unobserved.",
            ],
            "failure_modes": [
                {
                    "symptom": "Bounds are too wide to be informative",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Add a credible auxiliary restriction (monotone treatment response, instrument) to tighten the bounds.",
                    "alternative": "sp.iv_bounds",
                },
            ],
            "alternatives": ["oster_delta", "lee_bounds", "manski_bounds"],
            "typical_n_min": 100,
        },
        [
            "manski_bounds",
            "horowitz_manski",
            "lee_bounds",
            "iv_bounds",
            "oster_delta",
            "selection_bounds",
            "breakdown_frontier",
            "ml_bounds",
            "partial_identification",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Covariates, treatment, and outcome (for survival/longitudinal variants: time-to-event and time-varying covariates).",
                "Enough data to fit a Super Learner / HAL nuisance library.",
            ],
            "assumptions": [
                "Unconfoundedness (sequential exchangeability for longitudinal/LTMLE).",
                "Positivity / overlap of treatment given history.",
                "At least one nuisance (outcome or treatment) is estimated consistently; the targeting step gives double robustness.",
            ],
            "failure_modes": [
                {
                    "symptom": "Near-positivity violations create extreme clever-covariate weights",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Truncate weights, restrict the estimand, or report a positivity diagnostic.",
                    "alternative": "sp.dml",
                },
            ],
            "alternatives": ["dml", "ipw", "g_computation"],
            "typical_n_min": 400,
        },
        [
            "TMLE",
            "hal_tmle",
            "super_learner",
            "ltmle_survival",
            "SuperLearner",
            "HALRegressor",
            "HALClassifier",
        ],
    ),
    (
        {
            "pre_conditions": [
                "A behavioural choice variable (earnings, hours, ...) with a known kink or notch in the budget/choice set.",
                "A visible empirical density of the running variable around the threshold.",
            ],
            "assumptions": [
                "The counterfactual density would be smooth through the threshold absent the policy.",
                "Excess mass at the threshold reflects the behavioural elasticity of interest.",
                "No other discontinuity coincides with the threshold.",
            ],
            "failure_modes": [
                {
                    "symptom": "Round-number heaping or a coincident policy contaminates the bunching mass",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Exclude heaping points, widen the excluded region, and test the counterfactual polynomial order.",
                    "alternative": "sp.rdrobust",
                },
            ],
            "alternatives": ["rdrobust", "rkd"],
            "typical_n_min": 500,
        },
        ["bunching", "notch", "kink_unified", "general_bunching"],
    ),
    (
        {
            "pre_conditions": [
                "Domain context and a bounded list of candidate variables.",
                "A configured, logged LLM provider for reproducibility.",
            ],
            "assumptions": [
                "LLM-proposed graphs / priors are hypotheses to validate, not statistical identification.",
                "Human review or data-driven falsification is required before any causal claim.",
            ],
            "failure_modes": [
                {
                    "symptom": "Proposals are unstable across runs or include hallucinated variables/edges",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Fix the model release and seed, add constraints, and cross-check with constraint-based discovery.",
                    "alternative": "sp.causal_discovery",
                },
            ],
            "alternatives": ["llm_dag_constrained", "causal_discovery", "dag"],
            "typical_n_min": 1,
        },
        [
            "llm_dag_propose",
            "llm_sensitivity_priors",
            "llm_unobserved_confounders",
        ],
    ),
    (
        {
            "pre_conditions": [
                "An instrument plausibly affecting treatment, an endogenous treatment, and an outcome.",
                "A strong first stage (assess instrument strength before interpreting estimates).",
            ],
            "assumptions": [
                "Instrument relevance (non-zero first stage).",
                "Exclusion restriction: the instrument affects the outcome only through the treatment.",
                "Independence/exogeneity of the instrument; for LATE, monotonicity (no defiers).",
            ],
            "failure_modes": [
                {
                    "symptom": "Weak first stage — biased point estimates and unreliable conventional SEs",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Report first-stage F / effective F and use weak-IV-robust inference (Anderson-Rubin).",
                    "alternative": "sp.anderson_rubin_ci",
                },
            ],
            "alternatives": ["iv", "anderson_rubin_ci", "dml"],
            "typical_n_min": 200,
        },
        ["kernel_iv", "continuous_iv_late", "iv_diag", "iv_compare", "deepiv"],
    ),
    (
        {
            "pre_conditions": [
                "Covariates, treatment, and outcome with enough data to grow an honest forest.",
            ],
            "assumptions": [
                "Unconfoundedness given the covariates.",
                "Overlap / positivity.",
                "Honesty: separate subsamples are used to choose splits and to estimate effects.",
            ],
            "failure_modes": [
                {
                    "symptom": "Calibration test rejects — the forest's heterogeneity is not well calibrated",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Increase the sample / number of trees, or fall back to a doubly-robust learner.",
                    "alternative": "sp.dml",
                },
            ],
            "alternatives": ["dml", "auto_cate", "tmle"],
            "typical_n_min": 1000,
        },
        [
            "CausalForest",
            "average_treatment_effect",
            "rate",
            "test_calibration",
            "calibration_test",
            "forest_diagnostics",
            "honest_variance",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Logged trajectories (states, actions, rewards) from a known or estimable behaviour policy.",
            ],
            "assumptions": [
                "Sequential ignorability: no unobserved confounders of actions and outcomes.",
                "Positivity: the behaviour policy explores all evaluated actions.",
                "The environment satisfies the assumed (Markov) dynamics.",
            ],
            "failure_modes": [
                {
                    "symptom": "Poor behaviour-policy coverage — the target policy queries unseen state-action regions",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Use offline-safe / pessimistic methods and report effective sample size of the importance weights.",
                    "alternative": "sp.offline_safe_policy",
                },
            ],
            "alternatives": ["offline_safe_policy", "policy_value"],
            "typical_n_min": 1000,
        },
        ["causal_dqn", "offline_safe_policy", "causal_rl_benchmark"],
    ),
    (
        {
            "pre_conditions": [
                "Treatment-inducing and outcome-inducing proxy variables (negative controls) for the unobserved confounder.",
            ],
            "assumptions": [
                "The proxies are valid negative controls (relevant to the confounder, excluded from the causal channel).",
                "A bridge function exists (completeness conditions hold).",
            ],
            "failure_modes": [
                {
                    "symptom": "Proxies are weak or invalid — the bridge function is poorly identified",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Test proxy relevance, select stronger proxies, or fall back to sensitivity analysis.",
                    "alternative": "sp.select_pci_proxies",
                },
            ],
            "alternatives": ["select_pci_proxies", "dml"],
            "typical_n_min": 500,
        },
        ["bidirectional_pci", "fortified_pci", "pci_mtp", "select_pci_proxies"],
    ),
    (
        {
            "pre_conditions": [
                "Covariates, treatment, and outcome; a propensity model is fit internally to limit regularization-induced confounding.",
            ],
            "assumptions": [
                "Unconfoundedness and overlap.",
                "The BART/forest priors are appropriate for the outcome scale.",
            ],
            "failure_modes": [
                {
                    "symptom": "MCMC diagnostics fail to converge, or estimates are sensitive to the prior",
                    "exception": "statspai.ConvergenceWarning",
                    "remedy": "Increase draws/tuning, re-scale the outcome, and report posterior diagnostics.",
                    "alternative": "sp.dml",
                },
            ],
            "alternatives": ["dml", "auto_cate", "causal_forest"],
            "typical_n_min": 250,
        },
        [
            "bcf",
            "bcf_factor_exposure",
            "bcf_longitudinal",
            "bcf_ordinal",
            "BayesianCausalForest",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Sequentially measured covariates, (time-varying) treatment, and outcome.",
                "Models for the treatment process and the outcome (or weights).",
            ],
            "assumptions": [
                "Sequential exchangeability / no unmeasured confounding at each time point.",
                "Positivity: every treatment level is possible given the past.",
                "Correct specification of the treatment and/or outcome models.",
            ],
            "failure_modes": [
                {
                    "symptom": "Stabilized weights have extreme values (positivity near-violation)",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Truncate weights, simplify the treatment model, or use a doubly-robust estimator (TMLE).",
                    "alternative": "sp.hal_tmle",
                },
            ],
            "alternatives": ["tmle", "g_computation", "ipw"],
            "typical_n_min": 300,
        },
        [
            "g_computation",
            "ipw",
            "front_door",
            "msm",
            "stabilized_weights",
            "g_estimation",
            "mediate_interventional",
            "survivor_average_causal_effect",
        ],
    ),
    (
        {
            "pre_conditions": [
                "Constraint-/score-based discovery needs i.i.d. observational data with enough samples for reliable conditional-independence tests.",
                "Invariance-based discovery (ICP) needs data labelled by environment / intervention.",
            ],
            "assumptions": [
                "Causal Markov condition and faithfulness (PC/GES/FCI).",
                "Causal sufficiency for PC/GES (no latent confounders); FCI relaxes this.",
                "Acyclicity; LiNGAM additionally assumes a linear non-Gaussian model.",
            ],
            "failure_modes": [
                {
                    "symptom": "Unstable skeleton / many undirected edges — faithfulness or sample size is the likely culprit",
                    "exception": "statspai.AssumptionWarning",
                    "remedy": "Increase the sample, relax the CI-test threshold, or switch to FCI if latent confounders are plausible.",
                    "alternative": "sp.fci",
                },
            ],
            "alternatives": ["pc_algorithm", "fci", "ges", "lingam"],
            "typical_n_min": 500,
        },
        [
            "pc_algorithm",
            "fci",
            "ges",
            "lingam",
            "notears",
            "nonlinear_icp",
            "pcmci",
            "NOTEARS",
            "PCAlgorithm",
        ],
    ),
    (
        {
            "pre_conditions": [
                "A treated unit with a pre-period and a set of control series, or a panel with a low-rank structure.",
            ],
            "assumptions": [
                "The relationship between the treated unit and controls is stable absent the intervention (causal_impact).",
                "Matrix-completion: the untreated potential outcomes follow a low-rank factor structure with treatment as the missingness pattern.",
                "No concurrent intervention affects the controls.",
            ],
            "failure_modes": [
                {
                    "symptom": "Pre-period fit is poor or controls are themselves affected by the intervention",
                    "exception": "statspai.AssumptionViolation",
                    "remedy": "Re-select controls, lengthen the pre-period, or use synthetic-control / DiD diagnostics.",
                    "alternative": "sp.synth",
                },
            ],
            "alternatives": ["synth", "sdid", "gsynth"],
            "typical_n_min": 50,
        },
        ["mc_panel", "MCPanel", "policy_value"],
    ),
]
