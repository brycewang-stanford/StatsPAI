"""Curated agent cards by method family (roadmap W3, pass 3).

The docstring harvest and the name-based family inheritance in
:mod:`statspai.registry` left ~270 parity-ledger entry points — the
limited-dependent-variable regressions, survival, clustered inference,
weak-IV tools, RD bandwidth helpers, multiple testing, spatial statistics,
networks, power, decompositions, mediation, selection / attrition bounds,
survey estimation, time series, dynamic panels, post-estimation, ML
causal helpers and the g-formula / transport / target-trial toolkit —
without any statement of what they assume or when not to use them.

This module states it, one card per family, applied to every member with
the same extend-missing semantics as :data:`statspai._agent_cards_extra`
(hand-written ``FunctionSpec`` content always wins; a family card only
fills fields that are still empty). Cards are written for the family's
*shared* contract: the identifying / statistical assumptions every member
relies on, the data shape the agent should verify first, the failures an
agent will actually see and what to do next, and the situations where a
different family is the right call. Member-specific detail stays in the
member's docstring.

Every ``alternative`` must resolve to a registered function (the
``tests/test_agent_native_contract.py`` suite checks this) and every
``exception`` to a real exception class; ``tests/test_family_cards.py``
also checks that every member name is registered, so a rename fails
loudly here instead of silently dropping a card.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

Card = Dict[str, Any]


def _fm(
    symptom: str, exception: str, remedy: str, alternative: str = ""
) -> Dict[str, str]:
    out = {"symptom": symptom, "exception": exception, "remedy": remedy}
    if alternative:
        out["alternative"] = alternative
    return out


FAMILY_CARDS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------ #
    "binary_ordered_multinomial": {
        "members": [
            "logit",
            "probit",
            "cloglog",
            "ologit",
            "oprobit",
            "mlogit",
            "mixlogit",
            "biprobit",
            "clogit",
            "meologit",
        ],
        "assumptions": [
            "The link function (logistic / normal / complementary log-log) and the "
            "index form are correctly specified; coefficients are index weights, "
            "not marginal effects",
            "Observations are independent conditional on covariates, or the "
            "dependence is declared through cluster= / the random-effects "
            "structure",
            "No perfect separation: every outcome category occurs at every "
            "level of each discrete covariate",
            "Ordered models (ologit / oprobit): proportional odds / parallel "
            "regressions — one set of slopes shifts every threshold",
            "Multinomial logit: independence of irrelevant alternatives (IIA); "
            "mixlogit relaxes it through random coefficients",
        ],
        "pre_conditions": [
            "Outcome is coded as integers (0/1, ordered levels, or category ids), "
            "not as floats or strings",
            "Each outcome category has enough cases (rule of thumb: >= 10 events "
            "per estimated coefficient)",
        ],
        "failure_modes": [
            _fm(
                "ConvergenceWarning / coefficients drifting to ±inf",
                "statspai.exceptions.ConvergenceWarning",
                "Quasi-complete separation: drop or coarsen the separating covariate, "
                "or use a penalised (Firth-type) or linear-probability specification",
                "sp.regress",
            ),
            _fm(
                "Coefficients reported but the question is about effects on the "
                "probability scale",
                "(none — interpretation)",
                "Report average marginal effects with sp.margins rather than raw "
                "index coefficients",
                "sp.margins",
            ),
        ],
        "alternatives": ["sp.margins", "sp.regress", "sp.glm"],
        "not_recommended_when": [
            "the goal is a causal treatment effect under unconfoundedness — "
            "use sp.ipw, sp.aipw or sp.match and let the outcome model be a "
            "nuisance",
        ],
        "typical_n_min": 100,
    },
    # ------------------------------------------------------------------ #
    "count_models": {
        "members": [
            "poisson",
            "zip_model",
            "zinb",
            "hurdle",
            "ppmlhdfe",
            "mepoisson",
            "menbreg",
        ],
        "assumptions": [
            "The conditional mean is exp(x'b); Poisson pseudo-ML is consistent for "
            "the mean under this alone, but its default SEs assume equidispersion — "
            "use robust / clustered SEs",
            "Zero-inflated and hurdle models assume a separate process generates "
            "the excess zeros; they are not a fix for over-dispersion by itself",
            "Observations are independent conditional on covariates or the "
            "dependence is declared (cluster= / random effects)",
        ],
        "pre_conditions": [
            "Outcome is a non-negative integer count (or non-negative real for "
            "PPML gravity-type models)",
            "For ppmlhdfe: fixed effects that perfectly predict zeros are "
            "dropped along with their observations — check the reported count",
        ],
        "failure_modes": [
            _fm(
                "Over-dispersion: Poisson deviance / Pearson chi-square far above "
                "degrees of freedom",
                "(none — diagnostic)",
                "Switch to negative binomial (sp.nbreg / sp.zinb) or keep Poisson "
                "with robust SEs; the mean is still consistently estimated",
                "sp.nbreg",
            ),
            _fm(
                "Separation: a fixed effect or covariate perfectly predicts zero "
                "outcomes",
                "statspai.exceptions.ConvergenceWarning",
                "ppmlhdfe drops those observations (Correia-Guimarães-Zylkin); "
                "inspect the dropped-observations note before comparing samples",
                "sp.ppmlhdfe",
            ),
        ],
        "alternatives": ["sp.nbreg", "sp.ppmlhdfe", "sp.glm"],
        "not_recommended_when": [
            "the outcome is a rate with a known exposure — model the count with "
            "an exposure offset rather than the ratio",
        ],
        "typical_n_min": 100,
    },
    # ------------------------------------------------------------------ #
    "fractional_truncated_glm": {
        "members": [
            "fracreg",
            "truncreg",
            "betareg",
            "glm",
            "feglm",
            "meglm",
            "megamma",
        ],
        "assumptions": [
            "The chosen family / link matches the outcome's support (fractional "
            "logit for shares in [0,1], beta for shares in (0,1), gamma for "
            "positive skewed outcomes, truncated normal for truncated samples)",
            "Fractional / quasi-likelihood models are consistent under a correct "
            "conditional mean only; use robust SEs",
            "Truncated regression: truncation is on the outcome at a known "
            "threshold and the error is normal — misspecified normality biases "
            "everything, unlike OLS",
        ],
        "pre_conditions": [
            "Outcome support matches the family (no exact 0/1 for betareg; "
            "values inside the truncation region for truncreg)",
        ],
        "failure_modes": [
            _fm(
                "Outcome values outside the family's support",
                "statspai.exceptions.MethodIncompatibility",
                "Rescale or switch family: exact 0/1 shares → sp.fracreg; "
                "censoring rather than truncation → a tobit / censored model",
                "sp.fracreg",
            ),
        ],
        "alternatives": ["sp.fracreg", "sp.glm", "sp.regress"],
        "not_recommended_when": [
            "the sample is censored (values piled at a limit) rather than "
            "truncated (values absent) — that is a censored-regression problem",
        ],
    },
    # ------------------------------------------------------------------ #
    "survival": {
        "members": [
            "cox",
            "cox_frailty",
            "aft",
            "survreg",
            "kaplan_meier",
            "logrank_test",
            "finegray",
            "cuminc",
            "lcsf",
            "survival_sensitivity",
        ],
        "assumptions": [
            "Non-informative (independent) censoring conditional on covariates",
            "Cox: proportional hazards — covariate effects are constant over time; "
            "check Schoenfeld residuals or stratify",
            "AFT / parametric survival: the baseline distribution (Weibull, "
            "log-normal, ...) is correctly specified",
            "Competing risks (finegray / cuminc): the event of interest and the "
            "competing events are all recorded; the Fine-Gray subdistribution "
            "hazard answers a prediction question, the cause-specific hazard an "
            "etiologic one",
            "Frailty models: shared unobserved heterogeneity within a cluster "
            "follows the assumed (gamma / log-normal) distribution",
        ],
        "pre_conditions": [
            "One duration column (> 0) and one event indicator (1 = event, "
            "0 = censored) per subject; competing risks need an event-type code",
            "Time-varying covariates need a (start, stop] counting-process layout",
        ],
        "failure_modes": [
            _fm(
                "Proportional-hazards test rejects for a covariate",
                "statspai.exceptions.AssumptionWarning",
                "Stratify on it, add a time interaction, or report an AFT model",
                "sp.aft",
            ),
            _fm(
                "Fewer than ~10 events per covariate",
                "(none — diagnostic)",
                "Reduce the model or report Kaplan-Meier / log-rank comparisons "
                "instead of a multivariable fit",
                "sp.kaplan_meier",
            ),
        ],
        "alternatives": ["sp.kaplan_meier", "sp.aft", "sp.finegray"],
        "not_recommended_when": [
            "competing events are common and the estimand is the probability of "
            "the event of interest — Kaplan-Meier treating them as censored "
            "overstates it; use sp.cuminc / sp.finegray",
        ],
        "typical_n_min": 50,
    },
    # ------------------------------------------------------------------ #
    "clustered_inference": {
        "members": [
            "cluster_robust_se",
            "cr2_se",
            "cr3_jackknife_vcov",
            "wild_cluster_boot",
            "wild_cluster_ci_inv",
            "subcluster_wild_bootstrap",
            "twoway_cluster",
            "multiway_cluster_vcov",
            "jackknife_se",
            "shift_share_se",
        ],
        "assumptions": [
            "Errors are independent across clusters; any within-cluster "
            "dependence is unrestricted",
            "Clusters are defined at (or above) the level at which treatment "
            "was assigned",
            "Asymptotics run in the number of clusters: the default CR1 sandwich "
            "over-rejects with few (< ~30-50) or very unbalanced clusters; CR2 / "
            "CR3 / wild cluster bootstrap are the small-G repairs",
            "Shift-share SEs: shocks are independent across sectors; inference "
            "is at the shock level (Adão-Kolesár-Morales / Borusyak-Hull-Jaravel)",
        ],
        "pre_conditions": [
            "A fitted regression result (or its design matrix and residuals) "
            "and a cluster column with no missing values",
            "Two-way clustering needs both dimensions to have many clusters",
        ],
        "failure_modes": [
            _fm(
                "Few clusters (G < 30) or one dominant cluster",
                "statspai.exceptions.AssumptionWarning",
                "Report wild cluster bootstrap p-values / inverted CIs "
                "(sp.wild_cluster_boot, sp.wild_cluster_ci_inv) or CR3 jackknife",
                "sp.wild_cluster_boot",
            ),
            _fm(
                "Very few treated clusters",
                "(none — diagnostic)",
                "Cluster-robust SEs are unreliable; use sp.did_few_treated or "
                "randomization inference",
                "sp.did_few_treated",
            ),
        ],
        "alternatives": ["sp.wild_cluster_boot", "sp.cr2_se", "sp.conley"],
        "not_recommended_when": [
            "dependence is spatial rather than by group — use sp.conley with a "
            "distance cutoff",
        ],
    },
    # ------------------------------------------------------------------ #
    "spatial_hac": {
        "members": ["conley"],
        "assumptions": [
            "Spatial dependence in the errors dies out beyond the chosen "
            "distance cutoff; the kernel weights within the cutoff are a "
            "modelling choice, not estimated",
            "Locations are measured without error and distances are meaningful "
            "(great-circle for lat/lon)",
        ],
        "pre_conditions": [
            "Latitude / longitude (or projected coordinates) for every " "observation",
        ],
        "failure_modes": [
            _fm(
                "Results change materially with the cutoff",
                "(none — diagnostic)",
                "Report a cutoff sensitivity table rather than one number",
                "sp.cluster_robust_se",
            ),
        ],
        "alternatives": ["sp.cluster_robust_se", "sp.twoway_cluster"],
        # No cost_profile here: ``sp.conley`` enumerates only within-cutoff
        # pairs (scipy cKDTree), so the dense O(n^2) warning that belongs to
        # ``feols(vce='conley')`` would steer agents away from the scalable
        # path. The accurate profile comes from the registry's negative-
        # guidance table.
    },
    # ------------------------------------------------------------------ #
    "weak_iv_inference": {
        "members": [
            "anderson_rubin_ci",
            "anderson_rubin_test",
            "conditional_lr_ci",
            "weakrobust",
            "tF_adjustment",
            "tF_critical_value",
            "effective_f_test",
            "jive",
        ],
        "inherits_from": "iv",
        "assumptions": [
            "Instrument exogeneity and the exclusion restriction hold; these "
            "tools repair inference under weak identification, not invalidity",
            "Anderson-Rubin / conditional LR sets are exact under homoskedasticity "
            "and asymptotically valid with robust weights; they can be unbounded "
            "when identification is very weak — that is information, not an error",
            "tF adjustment applies to a single instrument with a first-stage F "
            "reported on the same variance estimator",
        ],
        "pre_conditions": [
            "A fitted IV result (or y, d, z arrays) with the same variance "
            "estimator (homoskedastic / robust / clustered) used throughout",
        ],
        "failure_modes": [
            _fm(
                "Unbounded or empty Anderson-Rubin confidence set",
                "(none — informational)",
                "Report it as is: an unbounded set means the data do not pin the "
                "effect down; an empty set rejects the over-identifying "
                "restrictions",
                "sp.conditional_lr_ci",
            ),
            _fm(
                "Effective F below the Montiel Olea-Pflueger threshold",
                "statspai.exceptions.AssumptionWarning",
                "Do not report 2SLS t-ratios; use sp.anderson_rubin_ci or "
                "sp.tF_adjustment for the reported interval",
                "sp.anderson_rubin_ci",
            ),
        ],
        "alternatives": ["sp.anderson_rubin_ci", "sp.iv", "sp.jive"],
    },
    # ------------------------------------------------------------------ #
    "rd_tools": {
        "members": [
            "rdbwselect",
            "lpbwselect_ce_rot",
            "lpbwselect_imse_dpi",
            "lpbwselect_imse_rot",
            "lpbwselect_mse_dpi",
            "lpbwselect_mse_rot",
            "rd2d_bw",
            "rdbwhte",
            "rdwinselect",
            "rdplot",
            "rdplotdensity",
            "rdpower",
            "rdsampsi",
            "rdsensitivity",
            "rdrbounds",
            "rdhte_lincom",
            "lprobust_at_point",
            "lpoly",
        ],
        "inherits_from": "rdrobust",
        "assumptions": [
            "Bandwidth selectors minimise an asymptotic MSE / coverage-error "
            "criterion under smoothness of the conditional expectation on each "
            "side of the cutoff; the selected h is data-driven, so report "
            "results at h/2 and 2h as well",
            "Local randomization window selection (rdwinselect) assumes "
            "covariate balance identifies the window, not the smoothness above",
        ],
        "pre_conditions": [
            "A continuous running variable with support on both sides of the "
            "cutoff; density plots and bandwidth selection need enough mass "
            "near c",
        ],
        "failure_modes": [
            _fm(
                "Selected bandwidth leaves fewer than ~20 effective observations "
                "on a side",
                "statspai.exceptions.DataInsufficient",
                "Report the effective N; consider local randomization "
                "(sp.rdrandinf) or a wider bandwidth with bias correction",
                "sp.rdrandinf",
            ),
        ],
        "alternatives": ["sp.rdrobust", "sp.rdrandinf", "sp.rddensity"],
    },
    # ------------------------------------------------------------------ #
    "multiple_testing": {
        "members": [
            "holm",
            "bonferroni",
            "benjamini_hochberg",
            "romano_wolf",
            "adjust_pvalues",
        ],
        "assumptions": [
            "Each input p-value is valid on its own (correct size marginally)",
            "Bonferroni / Holm control the family-wise error rate under arbitrary "
            "dependence; Benjamini-Hochberg controls the false discovery rate "
            "under independence or positive regression dependence",
            "Romano-Wolf uses the bootstrap to learn the dependence across "
            "hypotheses and is the least conservative FWER control — the "
            "bootstrap must resample at the level of the actual dependence "
            "(clusters)",
        ],
        "pre_conditions": [
            "The family of hypotheses is fixed before looking at the data; "
            "adding tests after the fact invalidates the control",
        ],
        "failure_modes": [
            _fm(
                "All adjusted p-values become 1 with many hypotheses",
                "(none — informational)",
                "Bonferroni / Holm are very conservative for large families; "
                "report FDR (sp.benjamini_hochberg) or Romano-Wolf",
                "sp.romano_wolf",
            ),
        ],
        "alternatives": ["sp.romano_wolf", "sp.benjamini_hochberg", "sp.holm"],
        "not_recommended_when": [
            "the hypotheses are one pre-registered primary outcome — there is "
            "nothing to adjust",
        ],
    },
    # ------------------------------------------------------------------ #
    "spatial_esda": {
        "members": [
            "moran",
            "moran_local",
            "moran_residuals",
            "geary",
            "getis_ord_g",
            "getis_ord_local",
            "join_counts",
            "lm_tests",
        ],
        "assumptions": [
            "The spatial weights matrix W encodes the true neighbourhood "
            "structure; every statistic is conditional on the chosen W",
            "Permutation inference assumes exchangeability of values across "
            "locations under the null of no spatial association",
            "Global statistics summarise one pattern; local versions (LISA / "
            "Getis-Ord local) need multiple-testing control across locations",
        ],
        "pre_conditions": [
            "A weights object from sp.queen_weights / sp.rook_weights / "
            "sp.knn_weights / sp.distance_band aligned to the data rows",
            "No islands (units with zero neighbours), or an explicit decision on "
            "how they are treated",
        ],
        "failure_modes": [
            _fm(
                "Islands in the weights matrix",
                "statspai.exceptions.MethodIncompatibility",
                "Use a k-nearest-neighbour or distance-band weights matrix so "
                "every unit has neighbours",
                "sp.knn_weights",
            ),
            _fm(
                "Residual Moran's I significant after OLS",
                "statspai.exceptions.AssumptionWarning",
                "Run sp.lm_tests to choose between a spatial lag (sp.sar) and a "
                "spatial error (sp.sem) model",
                "sp.lm_tests",
            ),
        ],
        "alternatives": ["sp.lm_tests", "sp.sar", "sp.sem"],
    },
    # ------------------------------------------------------------------ #
    "spatial_models": {
        "members": [
            "impacts",
            "sac",
            "sar_gmm",
            "sarar_gmm",
            "sem_gmm",
            "slx",
            "spatial_panel",
            "gwr",
            "gwr_bandwidth",
            "mgwr",
        ],
        "inherits_from": "sar",
        "assumptions": [
            "GMM spatial estimators (Kelejian-Prucha) do not need normality but "
            "rely on the instrument set built from W X, W^2 X",
            "GWR / MGWR assume the relationship varies smoothly over space; "
            "the bandwidth controls the bias-variance trade-off and is selected "
            "by cross-validation or AICc",
            "Impacts (direct / indirect / total) are the interpretable "
            "quantities in lag models — coefficients are not marginal effects",
        ],
        "pre_conditions": [
            "A spatial weights matrix aligned to the data and, for panels, a "
            "balanced unit x time layout",
        ],
        "failure_modes": [
            _fm(
                "Spatial autoregressive parameter at the boundary of its "
                "admissible range",
                "statspai.exceptions.ConvergenceWarning",
                "Check W row-standardisation and try a different weights "
                "specification or the GMM estimator",
                "sp.sar_gmm",
            ),
        ],
        "alternatives": ["sp.sar", "sp.sem", "sp.impacts"],
    },
    # ------------------------------------------------------------------ #
    "spatial_weights": {
        "members": [
            "queen_weights",
            "rook_weights",
            "knn_weights",
            "kernel_weights",
            "distance_band",
            "block_weights",
        ],
        "pre_conditions": [
            "Geometries (polygons) for contiguity weights; point coordinates for "
            "k-nearest-neighbour, kernel and distance-band weights",
            "A consistent coordinate reference system: distance-based weights on "
            "raw lat/lon degrees are wrong away from the equator",
        ],
        "assumptions": [
            "The weights are a modelling choice that every downstream statistic "
            "conditions on; report which one was used",
        ],
        "failure_modes": [
            _fm(
                "Islands (units with no neighbours)",
                "(none — informational)",
                "Prefer sp.knn_weights, which guarantees k neighbours, or widen "
                "the distance band",
                "sp.knn_weights",
            ),
        ],
        "alternatives": ["sp.knn_weights", "sp.queen_weights", "sp.distance_band"],
    },
    # ------------------------------------------------------------------ #
    "network_descriptives": {
        "members": [
            "centrality",
            "degree_centrality",
            "betweenness_centrality",
            "closeness_centrality",
            "eigenvector_centrality",
            "katz_centrality",
            "bonacich_power",
            "pagerank",
            "hits",
            "assortativity",
            "reciprocity",
            "transitivity",
            "clustering",
            "community_detection",
            "network_components",
            "network_modularity",
            "network_summary",
            "network_graph",
            "florentine_families",
            "karate_club",
        ],
        "assumptions": [
            "The observed edges are the relevant ties: measurement of the "
            "network (censoring at a boundary, missing edges) drives every "
            "statistic",
            "Descriptive statistics carry no sampling model; compare across "
            "networks only with size / density held fixed",
            "Community detection (modularity maximisation) has a resolution "
            "limit and is not deterministic — fix the seed and report stability",
        ],
        "pre_conditions": [
            "An edge list (source, target[, weight]) or adjacency matrix; "
            "directed vs undirected must match the measure (in/out degree, "
            "PageRank, HITS are directed concepts)",
        ],
        "failure_modes": [
            _fm(
                "Disconnected graph: closeness / eigenvector centrality undefined "
                "or zero for some nodes",
                "(none — informational)",
                "Compute per component (sp.network_components) or use "
                "harmonic closeness / Katz centrality",
                "sp.network_components",
            ),
        ],
        "alternatives": [
            "sp.network_summary",
            "sp.centrality",
            "sp.community_detection",
        ],
    },
    # ------------------------------------------------------------------ #
    "network_regression": {
        "members": ["netlm", "netlogit", "dyadic_regression", "peer_effects"],
        "assumptions": [
            "Dyadic observations are not independent: rows sharing a node are "
            "correlated, so inference uses QAP permutations or dyadic-cluster "
            "(two-way by both members) standard errors",
            "Peer-effects models (linear-in-means) face the reflection problem; "
            "identification needs network structure (intransitive triads) or "
            "exogenous peer characteristics, and the network itself must be "
            "exogenous to the outcome",
        ],
        "pre_conditions": [
            "A node-level data frame and an edge list / adjacency matrix with "
            "matching identifiers",
        ],
        "failure_modes": [
            _fm(
                "Endogenous network formation (homophily on unobservables)",
                "(none — identification)",
                "Peer effects are not identified; treat estimates as descriptive "
                "or use an experimental variation in group composition",
                "sp.dyadic_regression",
            ),
        ],
        "alternatives": ["sp.dyadic_regression", "sp.netlm", "sp.regress"],
    },
    # ------------------------------------------------------------------ #
    "power": {
        "members": [
            "power_ols",
            "power_rct",
            "power_cluster_rct",
            "power_two_proportions",
            "power_case_control",
            "power_logrank",
            "mde",
        ],
        "assumptions": [
            "The effect size, outcome variance (or baseline proportion / event "
            "rate) and, for clustered designs, the intra-cluster correlation are "
            "inputs, not outputs: power is only as credible as those guesses",
            "Tests are two-sided at the stated alpha unless the argument says "
            "otherwise; the formulas are large-sample normal approximations",
        ],
        "pre_conditions": [
            "Either the sample size (to get power / MDE) or the target power (to "
            "get the sample size), never both",
        ],
        "failure_modes": [
            _fm(
                "ICC unknown for a cluster-randomised design",
                "(none — planning)",
                "Take the ICC from a pilot or the literature and report power over "
                "a range (0.01-0.20); the design effect 1 + (m-1)·ICC dominates",
                "sp.power_cluster_rct",
            ),
        ],
        "alternatives": ["sp.mde", "sp.power_rct", "sp.power_cluster_rct"],
        "not_recommended_when": [
            "computing 'post-hoc power' from the observed effect — it is a "
            "one-to-one function of the p-value and adds no information",
        ],
    },
    # ------------------------------------------------------------------ #
    "decomposition_family": {
        "members": [
            "bauer_sinning",
            "fairlie",
            "machado_mata",
            "melly_decompose",
            "cfm_decompose",
            "kitagawa_decompose",
            "gelbach",
            "das_gupta",
            "gap_closing",
            "four_way_decomposition",
            "yu_elwert_decompose",
            "source_decompose",
            "shapley_inequality",
            "subgroup_decompose",
            "inequality_index",
            "rifreg",
        ],
        "inherits_from": "decompose",
        "assumptions": [
            "The 'explained' component depends on the reference coefficient "
            "vector; sequential (path-dependent) decompositions depend on the "
            "ordering unless a Shapley / Gelbach-type invariant is used",
            "Distributional decompositions (Machado-Mata, Melly, CFM, RIF) "
            "assume the conditional distribution is invariant to the "
            "counterfactual change (no general-equilibrium effects)",
            "Gap-closing and four-way (mediation) decompositions need the causal "
            "assumptions of the underlying intervention, not only a regression "
            "fit",
        ],
        "pre_conditions": [
            "A group indicator with both groups well represented, and the same "
            "covariate set fitted in each group",
        ],
        "failure_modes": [
            _fm(
                "Explained share changes sign when the reference group is switched",
                "(none — informational)",
                "Report both references or the pooled / Shapley-invariant "
                "decomposition; the index-number problem is real, not a bug",
                "sp.gelbach",
            ),
        ],
        "alternatives": ["sp.decompose", "sp.oaxaca", "sp.gelbach"],
    },
    # ------------------------------------------------------------------ #
    "mediation": {
        "members": [
            "mediation",
            "mediate_sensitivity",
            "mediation_decompose",
            "frontdoor",
        ],
        "assumptions": [
            "Sequential ignorability: treatment is as good as random given "
            "pre-treatment covariates, and the mediator is as good as random "
            "given treatment and those covariates",
            "No treatment-induced mediator-outcome confounder (an intermediate "
            "confounder breaks the natural indirect effect; use interventional "
            "effects or the front-door adjustment when it is present)",
            "Front-door: the mediator is fully caused by treatment and the "
            "mediator-outcome relation is unconfounded given treatment",
        ],
        "pre_conditions": [
            "Treatment, mediator and outcome measured in temporal order on the "
            "same units",
        ],
        "failure_modes": [
            _fm(
                "Indirect effect sensitive to small mediator-outcome confounding "
                "(rho)",
                "(none — sensitivity)",
                "Report the sp.mediate_sensitivity curve and the rho at which the "
                "effect crosses zero",
                "sp.mediate_sensitivity",
            ),
        ],
        "alternatives": ["sp.mediate_sensitivity", "sp.frontdoor", "sp.gformula_mc"],
    },
    # ------------------------------------------------------------------ #
    "selection_and_bounds": {
        "members": [
            "heckman",
            "etregress",
            "attrition_bounds",
            "attrition_test",
            "rosenbaum_bounds",
            "rosenbaum_gamma",
            "bias_factor",
            "evalue_rr",
            "evalue_from_result",
            "calibrate_confounding_strength",
        ],
        "assumptions": [
            "Heckman / endogenous-treatment models: joint normality of the "
            "selection and outcome errors, and an exclusion restriction — a "
            "variable that shifts selection but not the outcome; without it the "
            "model is identified only by functional form",
            "Rosenbaum bounds and E-values quantify how strong an unobserved "
            "confounder would have to be; they do not test whether one exists",
            "Attrition bounds (Lee / Manski-type) assume monotone selection "
            "(treatment affects attrition in one direction)",
        ],
        "pre_conditions": [
            "A selection or attrition indicator observed for every unit, "
            "including those with missing outcomes",
        ],
        "failure_modes": [
            _fm(
                "No credible exclusion restriction for Heckman selection",
                "(none — identification)",
                "Report bounds (sp.attrition_bounds) or a sensitivity analysis "
                "instead of a point estimate that rests on normality",
                "sp.attrition_bounds",
            ),
            _fm(
                "Rosenbaum Gamma at which significance is lost is close to 1",
                "(none — informational)",
                "The result is fragile to weak hidden bias; say so and look for "
                "design-based evidence (placebos, sp.evalue_rr)",
                "sp.evalue_rr",
            ),
        ],
        "alternatives": ["sp.attrition_bounds", "sp.rosenbaum_bounds", "sp.evalue_rr"],
    },
    # ------------------------------------------------------------------ #
    "survey": {
        "members": ["svymean", "svytotal", "svyglm", "rake"],
        "assumptions": [
            "Design weights are the inverse of the inclusion probabilities "
            "(times non-response adjustments); strata and primary sampling units "
            "are declared so the variance reflects the design",
            "Raking assumes the population margins are correct and the "
            "iterative proportional fitting converges to positive weights",
        ],
        "pre_conditions": [
            "Weight, stratum and PSU columns with no missing values; population "
            "margins for every raking dimension",
        ],
        "failure_modes": [
            _fm(
                "Single PSU in a stratum",
                "statspai.exceptions.MethodIncompatibility",
                "Collapse strata or use the 'certainty' / centered adjustment; "
                "variance is otherwise undefined",
                "sp.svymean",
            ),
            _fm(
                "Extreme raked weights",
                "(none — diagnostic)",
                "Trim the weights or drop a raking margin; report the design " "effect",
                "sp.rake",
            ),
        ],
        "alternatives": ["sp.svymean", "sp.svyglm", "sp.regress"],
    },
    # ------------------------------------------------------------------ #
    "time_series": {
        "members": [
            "arima",
            "garch",
            "bvar",
            "irf",
            "granger_causality",
            "engle_granger",
            "johansen",
            "cusum_test",
            "structural_break",
            "causal_kalman",
            "its",
        ],
        "assumptions": [
            "Stationarity after the chosen differencing (ARIMA, VAR, Granger); "
            "unit-root series need cointegration analysis (Engle-Granger, "
            "Johansen), not levels regressions",
            "Granger causality is predictive precedence under the information "
            "set in the model, not structural causality",
            "Impulse responses depend on the identification scheme (Cholesky "
            "ordering, sign restrictions); interrupted time series assumes no "
            "concurrent shock at the intervention date",
        ],
        "pre_conditions": [
            "A regularly spaced time index without gaps; enough observations per "
            "parameter (rule of thumb: >= 50 for ARIMA, more for VAR / GARCH)",
        ],
        "failure_modes": [
            _fm(
                "Unit root detected in a series used in levels",
                "statspai.exceptions.AssumptionWarning",
                "Difference the series or test for cointegration "
                "(sp.engle_granger / sp.johansen) before modelling",
                "sp.johansen",
            ),
            _fm(
                "GARCH / state-space likelihood fails to converge",
                "statspai.exceptions.ConvergenceWarning",
                "Rescale the series (percent returns), reduce the order, or "
                "supply starting values",
                "sp.arima",
            ),
        ],
        "alternatives": ["sp.arima", "sp.its", "sp.causal_impact"],
        "not_recommended_when": [
            "the question is a treatment effect with a control series available "
            "— sp.causal_impact / sp.synth use the control to absorb common "
            "shocks that a single-series ITS attributes to the intervention",
        ],
    },
    # ------------------------------------------------------------------ #
    "dynamic_panel": {
        "members": ["xtdpdsys", "xtlsdvc", "interactive_fe"],
        "inherits_from": "panel",
        "assumptions": [
            "Sequential exogeneity: errors are uncorrelated with past values of "
            "the regressors (lags are valid instruments)",
            "No second-order serial correlation in the differenced errors "
            "(Arellano-Bond AR(2) test) — otherwise the lagged instruments are "
            "invalid",
            "System GMM additionally assumes the first differences of the "
            "instruments are uncorrelated with the fixed effect (stationarity of "
            "initial conditions)",
            "Interactive fixed effects: the factor structure has a small, known "
            "number of factors",
        ],
        "pre_conditions": [
            "A panel with T >= 3 (T >= 4 for AR(2) testing) and unit / time "
            "identifiers",
        ],
        "failure_modes": [
            _fm(
                "Hansen J-test p-value near 1 with many instruments",
                "statspai.exceptions.AssumptionWarning",
                "Instrument proliferation weakens the test; collapse the "
                "instrument matrix or limit the lag depth",
                "sp.xtlsdvc",
            ),
            _fm(
                "AR(2) test rejects",
                "statspai.exceptions.AssumptionViolation",
                "Use deeper lags as instruments or a bias-corrected LSDV "
                "estimator (sp.xtlsdvc)",
                "sp.xtlsdvc",
            ),
        ],
        "alternatives": ["sp.xtlsdvc", "sp.panel", "sp.interactive_fe"],
        "not_recommended_when": [
            "T is large relative to N — the Nickell bias that motivates GMM is "
            "O(1/T) and within estimation is fine",
        ],
    },
    # ------------------------------------------------------------------ #
    "post_estimation": {
        "members": [
            "test",
            "lincom",
            "contrast",
            "margins",
            "margins_at",
            "pwcompare",
            "lrtest",
            "hausman_test",
            "het_test",
            "reset_test",
            "vif",
            "yatchew_linearity_test",
            "diagnostic_test",
        ],
        "assumptions": [
            "The fitted model whose result is passed in is correctly specified "
            "for the test in question; Wald tests use that fit's variance "
            "estimator (robust / clustered if it was), so inference inherits its "
            "validity",
            "Hausman: the efficient estimator is efficient under the null, and "
            "the difference of covariance matrices is positive definite",
            "Marginal effects are averages over the estimation sample unless "
            "margins_at fixes covariate values",
        ],
        "pre_conditions": [
            "A fitted StatsPAI result object (sp.regress / sp.logit / ...) — these "
            "functions do not take raw data",
        ],
        "failure_modes": [
            _fm(
                "Hausman statistic negative / covariance difference not positive "
                "definite",
                "(none — informational)",
                "Use the regression-based (auxiliary) Hausman test or a robust "
                "version; a negative statistic is evidence for the null",
                "sp.hausman_test",
            ),
            _fm(
                "Heteroskedasticity or RESET rejects",
                "statspai.exceptions.AssumptionWarning",
                "Refit with robust SEs (vce='hc1' / cluster) and consider "
                "polynomial or interaction terms",
                "sp.regress",
            ),
        ],
        "alternatives": ["sp.margins", "sp.test", "sp.regress"],
    },
    # ------------------------------------------------------------------ #
    "regression_extensions": {
        "members": [
            "absorb_ols",
            "ancova",
            "demean",
            "sureg",
            "three_sls",
            "gmm",
            "ivqreg",
            "sqreg",
            "stepwise",
            "lasso_select",
            "kdensity",
        ],
        "inherits_from": "regress",
        "assumptions": [
            "GMM: the moment conditions are valid and the weighting matrix is "
            "consistently estimated; over-identification is testable "
            "(Hansen J), exogeneity is not",
            "SUR / 3SLS gain efficiency only when errors are correlated across "
            "equations and regressors differ; 3SLS additionally needs valid "
            "instruments in every equation",
            "Stepwise / lasso selection: post-selection inference on the selected "
            "model is invalid without sample splitting or debiasing — treat "
            "them as screening tools",
            "Quantile IV (ivqreg): rank invariance / monotonicity of the "
            "structural quantile function",
        ],
        "pre_conditions": [
            "A model formula (or equation list for systems) and a data frame; "
            "high-dimensional fixed effects go through absorb= / sp.feols",
        ],
        "failure_modes": [
            _fm(
                "Hansen J-test rejects over-identifying restrictions",
                "statspai.exceptions.AssumptionViolation",
                "At least one moment condition is invalid; drop the suspect "
                "instruments rather than the test",
                "sp.iv",
            ),
        ],
        "alternatives": ["sp.regress", "sp.feols", "sp.iv"],
        "not_recommended_when": [
            "the object of interest is a treatment effect and covariates were "
            "chosen by stepwise selection — use sp.dml, which handles selection "
            "with valid inference",
        ],
    },
    # ------------------------------------------------------------------ #
    "rlasso_family": {
        "members": [
            "rlasso",
            "rlasso_effect",
            "rlasso_effects",
            "rlassologit",
            "rlassologit_effect",
            "rlassologit_effects",
        ],
        "assumptions": [
            "Approximate sparsity: the outcome and treatment equations are well "
            "approximated by a small number of the candidate controls",
            "The rigorous (theory-driven) penalty assumes the design and error "
            "moments the Belloni-Chernozhukov-Hansen theory requires; "
            "heteroskedasticity is handled by the loadings, dependence is not",
            "Double-selection / partialling-out gives valid inference for the "
            "treatment coefficient, not for the selected controls",
        ],
        "pre_conditions": [
            "Treatment(s) of interest separated from the candidate control set; "
            "controls standardised or the penalty loadings enabled",
        ],
        "failure_modes": [
            _fm(
                "No controls selected in either equation",
                "(none — informational)",
                "That is a valid outcome (the treatment effect is estimated "
                "without controls); check the penalty level if the design is "
                "known to be dense",
                "sp.dml",
            ),
        ],
        "alternatives": ["sp.dml", "sp.rlasso_effect", "sp.regress"],
    },
    # ------------------------------------------------------------------ #
    "did_diagnostics": {
        "members": [
            "pretrends_power",
            "pretrends_slope_for_power",
            "pretrends_equivalence",
            "parallel_trends_robustness",
            "quasi_untreated_test",
            "uniform_bands",
            "event_study_vcov",
            "etwfe",
            "check_absorbing",
            "balance_panel",
            "balance_check",
            "always_treat",
            "never_treat",
            "aggte_from_influence",
        ],
        "inherits_from": "did",
        "assumptions": [
            "Pre-trend tests have low power against economically relevant "
            "violations (Roth 2022): a non-rejection is not evidence for "
            "parallel trends; report the slope the test could detect "
            "(sp.pretrends_power)",
            "Uniform (sup-t) bands and joint tests use the full event-study "
            "covariance — pointwise intervals understate joint uncertainty",
        ],
        "pre_conditions": [
            "A fitted event-study / staggered-DiD result carrying its joint "
            "covariance in model_info['event_study_vcov']",
        ],
        "failure_modes": [
            _fm(
                "Pre-trend test passes but power against a linear trend of the "
                "effect's size is below 50%",
                "statspai.exceptions.AssumptionWarning",
                "Report honest confidence sets (sp.honest_did) rather than "
                "relying on the pre-test",
                "sp.honest_did",
            ),
        ],
        "alternatives": ["sp.honest_did", "sp.pretrends_power", "sp.event_study"],
    },
    # ------------------------------------------------------------------ #
    "ml_causal_helpers": {
        "members": [
            "xlearner",
            "blp",
            "best_linear_projection",
            "calibrate_cate",
            "variable_importance",
            "conformal_cate",
            "conformal_ite",
            "conformal_ite_interval",
            "weighted_conformal_prediction",
            "linear_calibration",
            "policy_tree",
            "policy_weight_ate",
            "policy_weight_marginal",
            "policy_weight_observed_prte",
            "policy_weight_subsidy",
            "pate",
            "auc",
        ],
        "inherits_from": "causal_forest",
        "assumptions": [
            "Unconfoundedness and overlap as for the CATE learner that produced "
            "the scores; calibration, BLP and policy evaluation are only as "
            "valid as the doubly-robust scores fed to them",
            "Conformal intervals need exchangeable calibration data (or valid "
            "weights under covariate shift) and are marginal, not conditional, "
            "guarantees",
            "Policy trees are evaluated honestly only on data not used to grow "
            "them — use the split / cross-fitted variants",
        ],
        "pre_conditions": [
            "Out-of-bag or cross-fitted scores; never evaluate a learner on its "
            "own training predictions",
        ],
        "failure_modes": [
            _fm(
                "Calibration slope near zero",
                "(none — informational)",
                "The learner has no detectable heterogeneity beyond the ATE; "
                "report the ATE and do not target on the CATE",
                "sp.average_treatment_effect",
            ),
        ],
        "alternatives": ["sp.causal_forest", "sp.metalearner", "sp.dml"],
    },
    # ------------------------------------------------------------------ #
    "longitudinal_causal": {
        "members": [
            "gformula_ice_fn",
            "gformula_mc",
            "transport_generalize",
            "transport_weights_fn",
            "target_trial_emulate",
            "target_trial_report",
            "immortal_time_check",
            "geolift",
            "interflex",
        ],
        "assumptions": [
            "Sequential exchangeability (no unmeasured time-varying confounding), "
            "positivity at every time point and consistency — the g-formula "
            "and target-trial emulation identify effects of sustained strategies "
            "only under all three",
            "Transportability: the outcome model or the participation weights "
            "are correctly specified and the effect modifiers that differ "
            "between populations are measured",
            "Matrix completion / geolift: the untreated potential outcomes have "
            "a low-rank structure (plus noise) that the treated block shares",
        ],
        "pre_conditions": [
            "Long (person-period) data with time-varying treatment and covariates "
            "in temporal order; a clear time zero for target-trial emulation",
        ],
        "failure_modes": [
            _fm(
                "Immortal-time bias: follow-up starts before treatment is assigned",
                "statspai.exceptions.AssumptionViolation",
                "Align time zero with eligibility and treatment assignment "
                "(sp.immortal_time_check flags the misalignment)",
                "sp.target_trial_emulate",
            ),
            _fm(
                "Positivity violations at later time points",
                "statspai.exceptions.AssumptionWarning",
                "Truncate the weights or restrict the strategy to regimes that "
                "are observed",
                "sp.gformula_mc",
            ),
        ],
        "alternatives": ["sp.gformula_mc", "sp.target_trial_emulate", "sp.msm"],
    },
    # ------------------------------------------------------------------ #
    "missing_data_and_randomization": {
        "members": [
            "mice",
            "mi_estimate",
            "mi_test",
            "ri_test",
            "fisher_exact",
            "subgroup_analysis",
            "influence_functions",
            "validation_scope",
        ],
        "assumptions": [
            "Multiple imputation: data are missing at random given the variables "
            "in the imputation model, which must include the analysis model's "
            "variables (congeniality)",
            "Randomization / Fisher-exact inference: the sharp null and the actual "
            "assignment mechanism are what is being tested — the permutation "
            "must mirror how treatment was assigned (clusters, strata)",
            "Subgroup analyses are pre-specified; post-hoc subgroups need "
            "multiple-testing control and interaction tests, not per-group "
            "p-values",
        ],
        "pre_conditions": [
            "Imputation: every analysis variable present in the imputation "
            "model; randomization inference: the assignment design is known",
        ],
        "failure_modes": [
            _fm(
                "Many subgroups, one 'significant' at 5%",
                "(none — informational)",
                "Test the interaction and adjust with sp.romano_wolf; a lone "
                "subgroup finding at nominal size is expected by chance",
                "sp.romano_wolf",
            ),
        ],
        "alternatives": ["sp.mice", "sp.ri_test", "sp.romano_wolf"],
    },
    # ------------------------------------------------------------------ #
    "mendelian_and_meta": {
        "members": ["mendelian_randomization", "meta_analysis"],
        "assumptions": [
            "Mendelian randomization: the genetic instruments are associated with "
            "the exposure, independent of confounders, and affect the outcome "
            "only through the exposure (no horizontal pleiotropy); use the "
            "robust estimators in the sp.mr family to probe the third",
            "Meta-analysis: study effects are exchangeable (random effects) or "
            "identical (fixed effect); publication bias is diagnosed, not "
            "assumed away",
        ],
        "pre_conditions": [
            "Summary statistics (estimate, SE) per study or per SNP with "
            "harmonised effect alleles",
        ],
        "failure_modes": [
            _fm(
                "Heterogeneity (Cochran's Q) rejects across instruments / studies",
                "statspai.exceptions.AssumptionWarning",
                "Report random-effects / MR-Egger / weighted-median estimates and "
                "the I^2 statistic",
                "sp.mr",
            ),
        ],
        "alternatives": ["sp.mr", "sp.meta_analysis", "sp.iv"],
    },
    # ------------------------------------------------------------------ #
    "frontier_and_misc": {
        "members": [
            "malmquist",
            "metafrontier",
            "translog_design",
            "zisf",
            "assimilative_causal",
            "evidence_without_injustice",
            "number_needed_to_treat",
            "prevalence_ratio",
            "icc",
            "W",
            "scdata",
            "negd",
        ],
        "assumptions": [
            "Stochastic-frontier and Malmquist tools assume the production "
            "technology and the inefficiency distribution (half-normal, "
            "truncated normal) are correctly specified; efficiency scores are "
            "relative to the sample frontier",
            "Epidemiological summaries (NNT, prevalence ratio) transform an "
            "already-estimated effect; their validity is that estimate's",
        ],
        "pre_conditions": [
            "Inputs / outputs per decision-making unit for frontier tools; a "
            "fitted effect and baseline risk for NNT",
        ],
        "alternatives": ["sp.regress", "sp.decompose"],
    },
}


def expand_family_cards() -> Dict[str, Card]:
    """``{function_name: card}`` for every member of every family."""
    out: Dict[str, Card] = {}
    for family, spec in FAMILY_CARDS.items():
        card = {k: v for k, v in spec.items() if k != "members"}
        card["family"] = family
        for member in spec["members"]:
            out.setdefault(member, card)
    return out


def family_members() -> Dict[str, List[str]]:
    return {k: list(v["members"]) for k, v in FAMILY_CARDS.items()}


# --------------------------------------------------------------------------- #
#  Per-variant corrections to inherited statements
# --------------------------------------------------------------------------- #
#: A family card, or a parent reached through ``inherits_from``, is written
#: for the family. Read on one member it can say something that is not true
#: of that member: the binary logit card listed the multinomial model's IIA
#: among its assumptions, and the Callaway-Sant'Anna card advised "use CS or
#: SA". This table removes such statements from a variant's card and adds
#: the ones the variant needs. ``drop`` entries are matched as prefixes of
#: the statement (an assumption string, or a failure mode's symptom).
#:
#: Entries come from reading the cards of the 30 entry points audited by
#: ``scripts/agent_card_audit.py`` (2026-10-03). The rest of the registry
#: has not been read this way.
_ORDERED = "Ordered models (ologit / oprobit)"
_IIA = "Multinomial logit: independence of irrelevant alternatives"
_USE_CS = "For staggered / heterogeneous effects: use CS or SA"
_TWFE_STAGGERED = "Staggered treatment timing with TWFE method"

VARIANT_OVERRIDES: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    # These estimators ARE the remedy the parent `did` card recommends; on
    # their own cards the advice and the TWFE failure mode do not apply.
    "callaway_santanna": {
        "drop": {
            "assumptions": [_USE_CS, "SUTVA", "No anticipation: outcomes in pre"],
            "failure_modes": [_TWFE_STAGGERED],
        },
        "add": {"assumptions": ["SUTVA: no spillovers between units"]},
    },
    "sun_abraham": {
        "drop": {
            "assumptions": [_USE_CS, "SUTVA", "No anticipation: outcomes in pre"],
            "failure_modes": [_TWFE_STAGGERED],
        },
        "add": {"assumptions": ["SUTVA: no spillovers between units"]},
    },
    "did_imputation": {
        "drop": {
            "assumptions": [_USE_CS, "SUTVA", "No anticipation (no pre-treatment"],
            "failure_modes": [_TWFE_STAGGERED],
        },
        "add": {"assumptions": ["SUTVA: no spillovers between units"]},
    },
    "etwfe": {
        "drop": {"assumptions": [_USE_CS], "failure_modes": [_TWFE_STAGGERED]},
    },
    "gardner_did": {
        "drop": {
            "failure_modes": ["Two-way fixed-effects estimate is contaminated"],
        },
    },
    # Filed in catch-all families none of whose statements is about them;
    # after scoping they would be left with nothing. Utilities in the same
    # position (weights constructors, data preparation, exporters) are
    # left empty on purpose: they have no identifying assumption.
    "interflex": {
        "add": {
            "assumptions": [
                "Treatment is unconfounded given the covariates at every "
                "value of the moderator",
                "The linear estimator assumes the marginal effect is linear "
                "in the moderator; the binning and kernel estimators relax "
                "that and are the check on it",
                "Common support: every treatment level is observed across the "
                "range of the moderator over which effects are reported",
            ],
        },
    },
    "negd": {
        "add": {
            "assumptions": [
                "Absent treatment the two groups would have changed by the "
                "same amount between the pre and post measurements (no "
                "selection-by-time interaction)",
                "No event other than the treatment affected one group and not "
                "the other between the two measurements",
            ],
        },
    },
    "icc": {
        "add": {
            "assumptions": [
                "A random-intercept model: group effects are independent of "
                "the residual and each has constant variance; the ICC is the "
                "group-level share of variance under that model",
            ],
        },
    },
    # The card described the sharp design only, and called the continuity
    # framework "local randomization".
    "rdrobust": {
        "drop": {"assumptions": ["Local randomization only in a neighborhood"]},
        "add": {
            "assumptions": [
                "The estimand is local to c: the effect at the cutoff, not an "
                "average over the support of x; extrapolation away from c is "
                "not identified",
                "fuzzy=: the probability of treatment jumps at c (first stage) "
                "and no unit is pushed out of treatment by crossing it "
                "(monotonicity); the estimand is the effect on compliers at c",
                "deriv=1 (kink): the first derivative of the potential-outcome "
                "regression is continuous at c, and the policy rule has a kink "
                "there",
                "cluster=: observations are independent across clusters; "
                "bandwidths and the variance then follow the clustered formulas",
            ],
        },
    },
    # A point-treatment estimator filed under the longitudinal g-methods card.
    "ipw": {
        "drop": {
            "assumptions": [
                "Sequential exchangeability",
                "Positivity: every treatment level is possible given the past",
                "Correct specification of the treatment and/or outcome models",
            ],
        },
        "add": {
            "assumptions": [
                "The propensity model is correctly specified: IPW has no "
                "outcome model to fall back on (sp.aipw is doubly robust)",
                "Unconfoundedness: no unmeasured confounding of the "
                "point treatment given the covariates",
                "Positivity / overlap: 0 < P(D=1 | X) < 1 on the estimand's " "support",
            ],
        },
    },
}


#: Statements of a family card that say, in their own words, which members
#: they are about ("Cox: ...", "Frailty models: ...", "Romano-Wolf ...").
#: ``expand_family_cards`` copies every statement to every member, so the
#: Kaplan-Meier card listed the Cox model's proportional hazards and the
#: Bonferroni card listed Romano-Wolf's bootstrap. Here each such statement
#: (matched by prefix) is given the members it applies to; on any other
#: member of the family it is dropped. A statement not listed applies to
#: the whole family.
#:
#: Written from one reading of all 30 family cards on 2026-10-03.
#: ``FAILURE_SCOPE`` below does the same for ``failure_modes``.
STATEMENT_SCOPE: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "binary_ordered_multinomial": {
        _ORDERED: ("ologit", "oprobit", "meologit"),
        _IIA: ("mlogit", "mixlogit", "clogit"),
    },
    "count_models": {
        "Zero-inflated and hurdle models": ("zip_model", "zinb", "hurdle"),
    },
    "fractional_truncated_glm": {
        "Fractional / quasi-likelihood models": ("fracreg", "glm", "feglm"),
        "Truncated regression:": ("truncreg",),
    },
    "survival": {
        "Cox: proportional hazards": ("cox", "cox_frailty"),
        "AFT / parametric survival": ("aft", "survreg"),
        "Competing risks (finegray / cuminc)": ("finegray", "cuminc"),
        "Frailty models:": ("cox_frailty",),
    },
    "clustered_inference": {
        "Shift-share SEs:": ("shift_share_se",),
    },
    "weak_iv_inference": {
        "Anderson-Rubin / conditional LR sets": (
            "anderson_rubin_ci",
            "anderson_rubin_test",
            "conditional_lr_ci",
            "weakrobust",
        ),
        "tF adjustment applies": ("tF_adjustment", "tF_critical_value"),
    },
    "rd_tools": {
        "Bandwidth selectors minimise": (
            "rdbwselect",
            "lpbwselect_ce_rot",
            "lpbwselect_imse_dpi",
            "lpbwselect_imse_rot",
            "lpbwselect_mse_dpi",
            "lpbwselect_mse_rot",
            "rd2d_bw",
            "rdbwhte",
        ),
        "Local randomization window selection": ("rdwinselect",),
    },
    "multiple_testing": {
        "Bonferroni / Holm control": (
            "holm",
            "bonferroni",
            "benjamini_hochberg",
            "adjust_pvalues",
        ),
        "Romano-Wolf uses the bootstrap": ("romano_wolf", "adjust_pvalues"),
    },
    "spatial_models": {
        "GMM spatial estimators": ("sar_gmm", "sarar_gmm", "sem_gmm"),
        "GWR / MGWR assume": ("gwr", "gwr_bandwidth", "mgwr"),
    },
    "network_descriptives": {
        "Community detection": ("community_detection", "network_modularity"),
    },
    "network_regression": {
        "Dyadic observations are not independent": (
            "netlm",
            "netlogit",
            "dyadic_regression",
        ),
        "Peer-effects models": ("peer_effects",),
    },
    "decomposition_family": {
        "Distributional decompositions": (
            "machado_mata",
            "melly_decompose",
            "cfm_decompose",
            "rifreg",
        ),
        "Gap-closing and four-way": ("gap_closing", "four_way_decomposition"),
    },
    "mediation": {
        "Front-door:": ("frontdoor",),
    },
    "selection_and_bounds": {
        "Heckman / endogenous-treatment models": ("heckman", "etregress"),
        "Rosenbaum bounds and E-values": (
            "rosenbaum_bounds",
            "rosenbaum_gamma",
            "bias_factor",
            "evalue_rr",
            "evalue_from_result",
            "calibrate_confounding_strength",
        ),
        "Attrition bounds": ("attrition_bounds", "attrition_test"),
    },
    "survey": {
        "Raking assumes": ("rake",),
    },
    "time_series": {
        "Granger causality is predictive": ("granger_causality",),
        "Impulse responses depend": ("irf", "bvar", "its"),
    },
    "dynamic_panel": {
        "Sequential exogeneity": ("xtdpdsys", "xtlsdvc"),
        "No second-order serial correlation": ("xtdpdsys",),
        "System GMM additionally": ("xtdpdsys",),
        "Interactive fixed effects:": ("interactive_fe",),
    },
    "post_estimation": {
        "Hausman:": ("hausman_test",),
        "Marginal effects are averages": ("margins", "margins_at"),
    },
    "regression_extensions": {
        "GMM:": ("gmm",),
        "SUR / 3SLS": ("sureg", "three_sls"),
        "Stepwise / lasso selection": ("stepwise", "lasso_select"),
        "Quantile IV (ivqreg)": ("ivqreg",),
    },
    "ml_causal_helpers": {
        "Conformal intervals need": (
            "conformal_cate",
            "conformal_ite",
            "conformal_ite_interval",
            "weighted_conformal_prediction",
        ),
        "Policy trees are evaluated": (
            "policy_tree",
            "policy_weight_ate",
            "policy_weight_marginal",
            "policy_weight_observed_prte",
            "policy_weight_subsidy",
        ),
    },
    "longitudinal_causal": {
        "Sequential exchangeability (no unmeasured time-varying": (
            "gformula_ice_fn",
            "gformula_mc",
            "target_trial_emulate",
            "target_trial_report",
            "immortal_time_check",
        ),
        "Transportability:": ("transport_generalize", "transport_weights_fn"),
        "Matrix completion / geolift": ("geolift",),
    },
    "missing_data_and_randomization": {
        "Multiple imputation:": ("mice", "mi_estimate", "mi_test"),
        "Randomization / Fisher-exact inference": ("ri_test", "fisher_exact"),
        "Subgroup analyses are pre-specified": ("subgroup_analysis",),
    },
    "mendelian_and_meta": {
        "Mendelian randomization:": ("mendelian_randomization",),
        "Meta-analysis:": ("meta_analysis",),
    },
    "frontier_and_misc": {
        "Stochastic-frontier and Malmquist tools": (
            "malmquist",
            "metafrontier",
            "translog_design",
            "zisf",
        ),
        "Epidemiological summaries": ("number_needed_to_treat", "prevalence_ratio"),
    },
}


#: The same for ``failure_modes``, matched on the symptom. Read on
#: 2026-10-03 after the assumptions.
_DECOMP_TWO_GROUP = (
    "bauer_sinning",
    "fairlie",
    "machado_mata",
    "melly_decompose",
    "cfm_decompose",
    "kitagawa_decompose",
    "gelbach",
    "das_gupta",
    "gap_closing",
    "four_way_decomposition",
    "yu_elwert_decompose",
)

FAILURE_SCOPE: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "count_models": {
        "Over-dispersion: Poisson deviance": (
            "poisson",
            "ppmlhdfe",
            "mepoisson",
            "zip_model",
            "hurdle",
        ),
    },
    "survival": {
        "Proportional-hazards test rejects": ("cox", "cox_frailty"),
        "Fewer than ~10 events per covariate": (
            "cox",
            "cox_frailty",
            "aft",
            "survreg",
            "finegray",
        ),
    },
    "weak_iv_inference": {
        "Unbounded or empty Anderson-Rubin confidence set": (
            "anderson_rubin_ci",
            "anderson_rubin_test",
            "conditional_lr_ci",
            "weakrobust",
        ),
        "Effective F below the Montiel Olea-Pflueger threshold": (
            "effective_f_test",
            "weakrobust",
            "tF_adjustment",
            "tF_critical_value",
        ),
    },
    "rd_tools": {
        "Selected bandwidth leaves fewer than": (
            "rdbwselect",
            "lpbwselect_ce_rot",
            "lpbwselect_imse_dpi",
            "lpbwselect_imse_rot",
            "lpbwselect_mse_dpi",
            "lpbwselect_mse_rot",
            "rd2d_bw",
            "rdbwhte",
        ),
    },
    "multiple_testing": {
        "All adjusted p-values become 1": ("bonferroni", "holm", "adjust_pvalues"),
    },
    "spatial_esda": {
        "Residual Moran's I significant after OLS": ("moran_residuals", "lm_tests"),
    },
    "spatial_models": {
        "Spatial autoregressive parameter at the boundary": (
            "sac",
            "sar_gmm",
            "sarar_gmm",
            "sem_gmm",
            "spatial_panel",
        ),
    },
    "network_descriptives": {
        "Disconnected graph: closeness / eigenvector": (
            "centrality",
            "closeness_centrality",
            "eigenvector_centrality",
            "katz_centrality",
            "bonacich_power",
        ),
    },
    "power": {
        "ICC unknown for a cluster-randomised design": ("power_cluster_rct", "mde"),
    },
    "decomposition_family": {
        "Explained share changes sign when the reference group": _DECOMP_TWO_GROUP,
    },
    "mediation": {
        "Indirect effect sensitive to small mediator-outcome": (
            "mediation",
            "mediate_sensitivity",
            "mediation_decompose",
        ),
    },
    "selection_and_bounds": {
        "No credible exclusion restriction for Heckman": ("heckman", "etregress"),
        "Rosenbaum Gamma at which significance is lost": (
            "rosenbaum_bounds",
            "rosenbaum_gamma",
        ),
    },
    "survey": {
        "Single PSU in a stratum": ("svymean", "svytotal", "svyglm"),
        "Extreme raked weights": ("rake",),
    },
    "time_series": {
        "Unit root detected in a series used in levels": (
            "arima",
            "bvar",
            "irf",
            "granger_causality",
            "its",
        ),
        "GARCH / state-space likelihood fails to converge": (
            "garch",
            "causal_kalman",
            "arima",
        ),
    },
    "dynamic_panel": {
        "Hansen J-test p-value near 1 with many instruments": ("xtdpdsys",),
        "AR(2) test rejects": ("xtdpdsys",),
    },
    "post_estimation": {
        "Hausman statistic negative": ("hausman_test",),
        "Heteroskedasticity or RESET rejects": ("het_test", "reset_test"),
    },
    "regression_extensions": {
        "Hansen J-test rejects over-identifying restrictions": ("gmm", "three_sls"),
    },
    "ml_causal_helpers": {
        "Calibration slope near zero": (
            "calibrate_cate",
            "linear_calibration",
            "blp",
            "best_linear_projection",
        ),
    },
    "longitudinal_causal": {
        "Immortal-time bias": (
            "target_trial_emulate",
            "target_trial_report",
            "immortal_time_check",
        ),
        "Positivity violations at later time points": (
            "gformula_ice_fn",
            "gformula_mc",
            "target_trial_emulate",
        ),
    },
    "missing_data_and_randomization": {
        "Many subgroups, one 'significant' at 5%": ("subgroup_analysis",),
    },
}


#: Failure modes that belong to named members only. ``FAILURE_SCOPE`` left
#: these functions with nothing, because every family statement was about a
#: sibling; each entry here was written for the functions it lists.
MEMBER_FAILURE_MODES: List[Tuple[Tuple[str, ...], Dict[str, str]]] = [
    (
        ("kaplan_meier", "cuminc"),
        {
            "symptom": (
                "Few subjects at risk in the right tail: the last steps of the curve "
                "rest on a handful of observations"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Report the number at risk under the curve and stop the plotted range "
                "where it is small"
            ),
        },
    ),
    (
        ("kaplan_meier",),
        {
            "symptom": (
                "Competing events treated as censoring: 1 - KM overstates the "
                "cumulative incidence of the event of interest"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Estimate the cumulative incidence function with the Aalen-Johansen "
                "estimator"
            ),
            "alternative": "sp.cuminc",
        },
    ),
    (
        ("logrank_test",),
        {
            "symptom": (
                "Survival curves cross: the log-rank test has little power against "
                "non-proportional alternatives"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Show the curves and do not read a non-rejection as equal survival; "
                "compare at fixed horizons"
            ),
            "alternative": "sp.kaplan_meier",
        },
    ),
    (
        ("survival_sensitivity",),
        {
            "symptom": (
                "The bounds include the null for small departures from no unmeasured "
                "confounding"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Report the smallest departure at which the conclusion changes, not "
                "the point estimate alone"
            ),
            "alternative": "sp.evalue_rr",
        },
    ),
    (
        ("lcsf",),
        {
            "symptom": (
                "Latent classes are not separated: posterior class probabilities stay "
                "near one half"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Compare against the single-class frontier by information criteria and "
                "report the posterior probabilities"
            ),
            "alternative": "sp.frontier",
        },
    ),
    (
        ("benjamini_hochberg",),
        {
            "symptom": (
                "FDR control read as family-wise control: at q = 0.05 some of the "
                "rejections are expected to be false, and the guarantee assumes "
                "independent or positively dependent p-values"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": "Use Holm or Romano-Wolf when every rejection has to stand",
            "alternative": "sp.holm",
        },
    ),
    (
        ("romano_wolf",),
        {
            "symptom": (
                "Too few bootstrap replications: adjusted p-values move in steps of 1 "
                "/ (B + 1)"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Raise the number of replications until the adjusted p-values are "
                "stable across seeds"
            ),
            "alternative": "sp.holm",
        },
    ),
    (
        ("degree_centrality", "betweenness_centrality", "pagerank", "hits"),
        {
            "symptom": (
                "Centrality scores compared across networks of different size or "
                "density"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Compare ranks or normalised scores within one network; raw values "
                "scale with the number of nodes and edges"
            ),
            "alternative": "sp.network_summary",
        },
    ),
    (
        ("assortativity", "reciprocity", "transitivity", "clustering"),
        {
            "symptom": (
                "Statistic read as a social process without a benchmark: a random "
                "graph of the same density already has nonzero clustering"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Compare with a random graph that preserves density or the degree "
                "sequence"
            ),
            "alternative": "sp.network_summary",
        },
    ),
    (
        ("community_detection", "network_modularity"),
        {
            "symptom": (
                "Modularity optimisation returns different partitions across runs and "
                "can merge small communities (resolution limit)"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Fix the seed, compare several runs and report Q together with the "
                "number of communities"
            ),
            "alternative": "sp.network_components",
        },
    ),
    (
        ("network_components", "network_summary"),
        {
            "symptom": (
                "Whole-graph averages are dominated by the largest component; isolates "
                "and small components disappear in them"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Report the component sizes and compute path-based statistics per "
                "component"
            ),
        },
    ),
    (
        ("network_graph",),
        {
            "symptom": (
                "Duplicate rows or self-loops in the edge list change every "
                "degree-based statistic"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Deduplicate the edge list and decide how self-loops are treated "
                "before building the graph"
            ),
            "alternative": "sp.network_summary",
        },
    ),
    (
        (
            "power_ols",
            "power_rct",
            "power_two_proportions",
            "power_case_control",
            "power_logrank",
        ),
        {
            "symptom": (
                "Effect size taken from a small pilot or from a published significant "
                "estimate: realised power is well below the nominal value"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Compute power for the smallest effect of interest and report the "
                "minimum detectable effect"
            ),
            "alternative": "sp.mde",
        },
    ),
    (
        ("power_rct", "power_two_proportions"),
        {
            "symptom": (
                "Units are randomised in clusters but power is computed as if they "
                "were independent"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": "Apply the design effect 1 + (m - 1) * ICC",
            "alternative": "sp.power_cluster_rct",
        },
    ),
    (
        ("frontdoor",),
        {
            "symptom": (
                "A direct path from treatment to outcome bypasses the mediator, or the "
                "mediator-outcome relation is confounded: the front-door formula no "
                "longer returns the total effect"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Argue full mediation and mediator exogeneity from the design, or "
                "switch to back-door adjustment or an instrument"
            ),
            "alternative": "sp.mediation",
        },
    ),
    (
        ("attrition_bounds",),
        {
            "symptom": (
                "Bounds are too wide to sign the effect when attrition differs sharply "
                "between arms"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Tighten them with baseline covariates and state the monotonicity "
                "assumption in use"
            ),
            "alternative": "sp.lee_bounds",
        },
    ),
    (
        ("attrition_test",),
        {
            "symptom": (
                "Equal attrition rates read as no attrition bias: the same rate does "
                "not mean the same units leave"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Compare baseline covariates of attriters by arm and report bounds"
            ),
            "alternative": "sp.attrition_bounds",
        },
    ),
    (
        (
            "bias_factor",
            "evalue_rr",
            "evalue_from_result",
            "calibrate_confounding_strength",
        ),
        {
            "symptom": (
                "Sensitivity value reported without a benchmark: a large E-value is "
                "not robustness unless it exceeds what measured confounders achieve"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": "Benchmark against the strongest observed covariate",
            "alternative": "sp.sensemakr",
        },
    ),
    (
        ("engle_granger", "johansen"),
        {
            "symptom": (
                "Series are not all integrated of order one, or the deterministic "
                "terms or lag length are wrong: the cointegration conclusion changes "
                "with the specification"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Test the order of integration of each series first and report the "
                "result across lag and trend choices"
            ),
            "alternative": "sp.unitroot",
        },
    ),
    (
        ("cusum_test", "structural_break"),
        {
            "symptom": (
                "Serially correlated residuals or a break near the sample edge: "
                "stability tests over-reject and the break date is imprecise"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Model the dynamics first, trim the sample ends and report the break "
                "date with its interval"
            ),
            "alternative": "sp.its",
        },
    ),
    (
        ("test", "lincom"),
        {
            "symptom": (
                "Few clusters or a nearly singular restriction covariance: the "
                "chi-square / F reference distribution is unreliable"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": "Use a wild cluster bootstrap for the restriction",
            "alternative": "sp.wild_cluster_bootstrap",
        },
    ),
    (
        ("lrtest",),
        {
            "symptom": (
                "Models are not nested, are fitted on different samples, or use a "
                "robust / cluster variance: the likelihood-ratio statistic has no "
                "chi-square reference"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Fit both models on the same rows and use a Wald test when the "
                "variance is robust"
            ),
            "alternative": "sp.test",
        },
    ),
    (
        ("contrast", "margins", "margins_at", "pwcompare"),
        {
            "symptom": (
                "Margins evaluated at covariate combinations the data do not contain"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Evaluate at observed values or within the observed range and report "
                "average marginal effects"
            ),
        },
    ),
    (
        ("vif",),
        {
            "symptom": (
                "A high VIF on a control read as a problem for the coefficient of "
                "interest"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Collinearity among controls does not bias that coefficient; read the "
                "VIF of the regressor of interest"
            ),
            "alternative": "sp.regress",
        },
    ),
    (
        ("yatchew_linearity_test",),
        {
            "symptom": (
                "Small sample: the test is asymptotic and has low power against smooth "
                "departures from the polynomial"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": "Report it next to a RESET test and a plot of binned residuals",
            "alternative": "sp.reset_test",
        },
    ),
    (
        ("diagnostic_test",),
        {
            "symptom": (
                "Sensitivity and specificity estimated in a case-control or referred "
                "sample do not carry over to the target population (spectrum bias)"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "State the sampling design and compute predictive values at the target "
                "prevalence"
            ),
        },
    ),
    (
        ("transport_generalize", "transport_weights_fn"),
        {
            "symptom": (
                "Effect modifiers in the target population have little overlap with "
                "the source sample: the transport weights are extreme"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Restrict the target to the region of overlap and report the effective "
                "sample size"
            ),
            "alternative": "sp.overlap_weights",
        },
    ),
    (
        ("geolift",),
        {
            "symptom": (
                "Poor pre-period fit of the synthetic control: the lift estimate "
                "carries the fit error"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Inspect the pre-period RMSPE and placebo tests; lengthen the "
                "pre-period or change the donor pool"
            ),
            "alternative": "sp.synth",
        },
    ),
    (
        ("interflex",),
        {
            "symptom": (
                "The moderator has sparse support in part of its range: marginal "
                "effects there are extrapolation"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Use the binning or kernel estimator and show the moderator histogram "
                "under the plot"
            ),
        },
    ),
    (
        ("mice", "mi_estimate", "mi_test"),
        {
            "symptom": (
                "Imputation model omits the outcome or other analysis variables: "
                "pooled associations are biased toward zero"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Include the outcome, every analysis covariate and the interactions in "
                "the imputation model"
            ),
        },
    ),
    (
        ("ri_test", "fisher_exact"),
        {
            "symptom": (
                "Permutations do not follow the actual assignment mechanism (blocks, "
                "clusters): the randomization p-value is not valid"
            ),
            "exception": "(none \u2014 informational)",
            "remedy": (
                "Permute within strata and at the level at which treatment was assigned"
            ),
        },
    ),
]


def _family_of() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for family, spec in FAMILY_CARDS.items():
        for member in spec["members"]:
            out.setdefault(member, family)
    return out


_FAMILY_OF = _family_of()


def _out_of_scope_prefixes(
    name: str, table: Optional[Dict[str, Dict[str, Tuple[str, ...]]]] = None
) -> Tuple[str, ...]:
    """Prefixes of family statements that are about other members."""
    family = _FAMILY_OF.get(name)
    if family is None:
        return ()
    scope = STATEMENT_SCOPE if table is None else table
    return tuple(
        prefix
        for prefix, members in scope.get(family, {}).items()
        if name not in members
    )


def apply_variant_overrides(
    name: str, assumptions: List[str], failure_modes: List[Dict[str, Any]]
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Drop and add statements for one variant; no-op without an entry."""
    scoped = _out_of_scope_prefixes(name)
    if scoped:
        assumptions = [a for a in assumptions if not str(a).startswith(scoped)]
    scoped_f = _out_of_scope_prefixes(name, FAILURE_SCOPE)
    if scoped_f:
        failure_modes = [
            fm
            for fm in failure_modes
            if not str(fm.get("symptom", "")).startswith(scoped_f)
        ]
    own = [dict(fm) for members, fm in MEMBER_FAILURE_MODES if name in members]
    if own:
        seen = {str(fm.get("symptom", "")) for fm in failure_modes}
        failure_modes = list(failure_modes) + [
            fm for fm in own if fm["symptom"] not in seen
        ]
    override = VARIANT_OVERRIDES.get(name)
    if not override:
        return assumptions, failure_modes
    drop = override.get("drop", {})
    add = override.get("add", {})
    drop_a = tuple(drop.get("assumptions", ()))
    kept = [a for a in assumptions if not (drop_a and str(a).startswith(drop_a))]
    for text in add.get("assumptions", ()):
        if text not in kept:
            kept.append(text)
    drop_f = tuple(drop.get("failure_modes", ()))
    modes = [
        fm
        for fm in failure_modes
        if not (drop_f and str(fm.get("symptom", "")).startswith(drop_f))
    ]
    return kept, modes


__all__ = [
    "FAILURE_SCOPE",
    "FAMILY_CARDS",
    "MEMBER_FAILURE_MODES",
    "STATEMENT_SCOPE",
    "VARIANT_OVERRIDES",
    "apply_variant_overrides",
    "expand_family_cards",
    "family_members",
]
