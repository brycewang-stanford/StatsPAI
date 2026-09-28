"""Per-function agent cards, batch 2: the DiD, IV, RD, synthetic-control
and DML families (roadmap W3, pass 3 / batch 2).

The family cards in :mod:`statspai._family_cards` are the floor: they
state what a whole family shares. These cards state what is specific to
one entry point — the failure the agent will actually see from *this*
estimator, the situation in which *this* variant is the wrong pick even
though the call succeeds, and the scaling hazard of *this*
implementation. Written from each function's docstring and reference
paper; every ``alternative`` resolves to a registered function and every
``exception`` to a real class (``tests/test_function_cards.py``).

Applied through ``registry._apply_agent_card_seeds`` with extend-missing
semantics: hand-written ``FunctionSpec`` content stays first, these
entries add to it; where a family card (or family template) and a
per-function card both exist, the per-function card overrides it field by
field and inherits the fields it does not state (the family card is the
floor, not a second voice). Hand-written per-name entries in the registry
keep the last word; a per-function card only appends to those.
"""

# flake8: noqa: E501  (descriptive-string data module; long lines are content)
from __future__ import annotations

from typing import Any, Dict

Card = Dict[str, Any]


def _fm(
    symptom: str, exception: str, remedy: str, alternative: str = ""
) -> Dict[str, str]:
    out = {"symptom": symptom, "exception": exception, "remedy": remedy}
    if alternative:
        out["alternative"] = alternative
    return out


FUNCTION_CARDS: Dict[str, Card] = {
    # ================================================================== #
    #  DiD
    # ================================================================== #
    "did": {
        "cost_profile": (
            "method='cs' with bstrap=True runs biters multiplier-bootstrap draws "
            "over every ATT(g,t) cell; cost grows with cohorts x periods x biters. "
            "Analytic SEs (default) are O(n)."
        ),
    },
    "did_balance": {
        "pre_conditions": [
            "Long panel with cohort g (0 = never treated), period t and unit i; "
            "covariates observed in the base and comparison periods",
        ],
        "failure_modes": [
            _fm(
                "Normalized difference above the threshold on LEVELS but not on "
                "CHANGES",
                "(none — diagnostic)",
                "Levels imbalance alone does not threaten parallel trends; DiD "
                "differences it out. Move to a conditional design only when "
                "CHANGES are imbalanced",
                "sp.callaway_santanna",
            ),
            _fm(
                "Normalized difference above the threshold on CHANGES",
                "statspai.exceptions.AssumptionWarning",
                "The covariate is moving differentially; condition on it with the "
                "doubly-robust estimator (x=[...], estimator='dr')",
                "sp.drdid",
            ),
        ],
        "alternatives": ["sp.callaway_santanna", "sp.drdid", "sp.balance_check"],
        "not_recommended_when": [
            "the covariate is itself affected by treatment — balance on a "
            "post-treatment variable is not evidence about parallel trends",
        ],
    },
    "distributional_did": {
        "pre_conditions": [
            "Continuous (or many-valued) outcome so that bins have mass in every "
            "cohort-period cell",
        ],
        "failure_modes": [
            _fm(
                "A bin indicator has no variation in some cohort-period cell",
                "statspai.exceptions.DataInsufficient",
                "Use fewer bins (n_bins) or supply binpoints that keep every bin "
                "populated on both sides of adoption",
                "sp.callaway_santanna",
            ),
            _fm(
                "Per-bin effects do not sum to approximately zero",
                "(none — diagnostic)",
                "Mass is leaving the binned support (outcome range shifted beyond "
                "the outer bins); widen the outer bins",
                "sp.qdid",
            ),
        ],
        "alternatives": ["sp.qdid", "sp.callaway_santanna", "sp.distributional_te"],
        "not_recommended_when": [
            "the question is a quantile treatment effect rather than where "
            "probability mass moved — use sp.qdid (changes-in-changes)",
        ],
        "cost_profile": "One Callaway-Sant'Anna fit per bin: n_bins x the CS cost.",
    },
    "staggered_rollout": {
        "pre_conditions": [
            "Balanced panel; every unit eventually treated or a last-treated cohort "
            "that can serve as the control at each period",
            "Adoption timing assigned by a lottery, phased launch or wave "
            "randomisation — the assumption is about the mechanism, not the data",
        ],
        "failure_modes": [
            _fm(
                "Estimate differs from sp.callaway_santanna on the same data",
                "(none — informational)",
                "Expected: this is a different estimand with design-based "
                "inference; on non-randomised timing (did::mpdta) the two differ "
                "and that is not a bug",
                "sp.callaway_santanna",
            ),
            _fm(
                "Only one adoption cohort, or a never-treated group and no "
                "randomised timing among the treated",
                "statspai.exceptions.MethodIncompatibility",
                "The design-based comparison needs variation in *timing*; with one "
                "cohort use a parallel-trends estimator",
                "sp.did",
            ),
        ],
        "alternatives": ["sp.staggered_cs", "sp.staggered_sa", "sp.callaway_santanna"],
        "not_recommended_when": [
            "the pre-trend test, honest DiD or Bacon decomposition are being run "
            "on this result — they diagnose parallel trends, which this estimator "
            "does not assume",
        ],
        "cost_profile": (
            "fisher=True permutes adoption dates n_fisher times and re-solves the "
            "estimator each time; cost is n_fisher x the point-estimate cost."
        ),
    },
    "staggered_cs": {
        "pre_conditions": ["Balanced panel with randomised adoption timing"],
        "failure_modes": [
            _fm(
                "se_type='adjusted' smaller than se_type='neyman'",
                "(none — informational)",
                "Both are valid; 'adjusted' uses the pre-period covariance that "
                "random timing identifies and is what R staggered prints. Report "
                "which one you used",
                "sp.staggered_rollout",
            ),
        ],
        "alternatives": [
            "sp.staggered_rollout",
            "sp.staggered_sa",
            "sp.callaway_santanna",
        ],
        "cost_profile": "fisher=True re-solves the estimator n_fisher times.",
    },
    "staggered_sa": {
        "pre_conditions": [
            "Balanced panel with randomised adoption timing and a last-treated "
            "cohort large enough to serve as the only control",
        ],
        "failure_modes": [
            _fm(
                "Wide standard errors relative to sp.staggered_cs",
                "(none — informational)",
                "Only the last-treated cohort is the control here; if it is small "
                "the CS estimand (all not-yet-treated as controls) is more precise",
                "sp.staggered_cs",
            ),
        ],
        "alternatives": ["sp.staggered_cs", "sp.staggered_rollout", "sp.sun_abraham"],
        "cost_profile": "fisher=True re-solves the estimator n_fisher times.",
    },
    "drdid": {
        "not_recommended_when": [
            "the design is staggered — drdid is the 2x2 building block; "
            "sp.callaway_santanna(..., estimator='dr') applies it cell by cell "
            "with the right controls",
            "no covariates are needed — the unconditional 2x2 (sp.did) has the "
            "same estimand without nuisance models",
        ],
        "failure_modes": [
            _fm(
                "Propensity scores near 0 or 1 in the treated group",
                "statspai.exceptions.AssumptionWarning",
                "Conditional parallel trends needs overlap; trim or drop the "
                "covariate that predicts treatment almost perfectly",
                "sp.trimming",
            ),
        ],
        "cost_profile": "n_boot bootstrap replications of a logit + OLS fit; O(n_boot x n).",
    },
    "ddd": {
        "not_recommended_when": [
            "the second difference (eligibility) does not split units within the "
            "treated group — without an ineligible comparison inside each group "
            "this is a DiD with an extra dummy, not a triple difference",
        ],
        "failure_modes": [
            _fm(
                "Empty cell in the group x eligibility x period table",
                "statspai.exceptions.DataInsufficient",
                "All eight cells must be populated; merge periods or eligibility "
                "categories",
                "sp.did",
            ),
        ],
    },
    "gardner_did": {
        "not_recommended_when": [
            "there are no never-treated (or long not-yet-treated) units — stage 1 "
            "fits the two-way FEs on untreated observations only and needs them "
            "in every period",
        ],
        "cost_profile": (
            "vce='bootstrap' re-runs both stages n_boot times; the analytic "
            "Butts-Gardner correction is O(n)."
        ),
        "alternatives": ["sp.did_imputation", "sp.callaway_santanna", "sp.etwfe"],
    },
    "lp_did": {
        "not_recommended_when": [
            "periods are not consecutive integers — long differences Y_{t+h} - "
            "Y_{t-1} are undefined across gaps",
            "treatment switches on and off within units and the estimand is a "
            "single absorbing-adoption effect — use sp.did_multiplegt for "
            "switchers/stayers",
        ],
        "failure_modes": [
            _fm(
                "Horizon-specific n_obs shrinks toward the longest horizons",
                "(none — informational)",
                "Long-horizon estimates rest on the earliest switchers only; cap "
                "horizons at the point where most cohorts are still observed",
                "sp.callaway_santanna",
            ),
        ],
        "alternatives": [
            "sp.callaway_santanna",
            "sp.did_multiplegt",
            "sp.did_imputation",
        ],
    },
    "harvest_did": {
        "not_recommended_when": [
            "a paper-standard estimator is the target — harvest_did is a "
            "robustness comparison against CS / SA, not a replacement for them",
        ],
        "failure_modes": [
            _fm(
                "weighting='precision' is dominated by one small high-variance "
                "comparison",
                "(none — diagnostic)",
                "Switch to weighting='simple' or 'cohort' and report both",
                "sp.callaway_santanna",
            ),
        ],
        "alternatives": [
            "sp.callaway_santanna",
            "sp.sun_abraham",
            "sp.bacon_decomposition",
        ],
        "cost_profile": "Number of harvested 2x2 comparisons grows as cohorts x periods squared.",
    },
    "fect": {
        "not_recommended_when": [
            "the treated units have fewer than min_t0 untreated periods — they are "
            "dropped and the ATT no longer describes the treated group",
            "there is one treated unit and a short pre-period — factor / "
            "matrix-completion models cannot be fitted; use sp.synth",
        ],
        "failure_modes": [
            _fm(
                "Cross-validated number of factors r hits the upper bound",
                "(none — diagnostic)",
                "Raise the grid or accept that the low-rank structure is not "
                "identified; compare method='fe' and 'mc'",
                "sp.gsynth",
            ),
        ],
        "cost_profile": (
            "method='ife' / 'mc' iterate an SVD on the N x T outcome matrix per "
            "candidate rank and per bootstrap draw: O(n_boot x ranks x N T "
            "min(N,T))."
        ),
        "alternatives": ["sp.did_imputation", "sp.gsynth", "sp.mc_panel"],
    },
    "did_multiplegt": {
        "not_recommended_when": [
            "treatment is absorbing (never switches off) and a dynamic event study "
            "is wanted — sp.callaway_santanna or sp.did_multiplegt_dyn are the "
            "estimators for that; DID_M pools consecutive-period cells",
        ],
        "cost_profile": "n_boot cluster-bootstrap replications of the full DID_M rollup.",
    },
    "did_multiplegt_dyn": {
        "not_recommended_when": [
            "treatment never switches off and all units adopt at once — the "
            "intertemporal estimator reduces to a 2x2; use sp.did",
        ],
    },
    "event_study": {
        "cost_profile": "One OLS with window x cohort dummies; O(n x window).",
        "alternatives": ["sp.sun_abraham", "sp.aggte", "sp.did_imputation"],
    },
    "aggte": {
        "not_recommended_when": [
            "type='dynamic' with unbalanced cohorts and no balance_e — the "
            "long-horizon averages mix cohorts of different composition",
        ],
        "failure_modes": [
            _fm(
                "type='simple' ATT changes sign relative to type='group'",
                "(none — informational)",
                "Weights differ (treated-observation vs cohort-size); neither is "
                "wrong — state which estimand the paper reports",
                "sp.callaway_santanna",
            ),
        ],
        "cost_profile": (
            "bstrap=True multiplier bootstrap: n_boot draws over the stacked "
            "influence functions, O(n_boot x n)."
        ),
    },
    "qdid": {
        "not_recommended_when": [
            "the untreated distribution may shift by different amounts at "
            "different ranks — QDiD assumes a constant shift; use method='cic'",
        ],
        "cost_profile": "n_boot bootstrap replications x len(quantiles) quantile fits.",
    },
    "spillover_did": {
        "not_recommended_when": [
            "there are no coordinates or distances — ring definitions need a "
            "metric; without one use sp.did with an explicit exposure column",
        ],
        "cost_profile": "Distance matrix is O(N^2) in the number of units.",
    },
    "continuous_did": {
        "not_recommended_when": [
            "a paper-faithful continuous-treatment ATT(d|g,t) is required — this "
            "is a heuristic family; use sp.cgs_continuous_did",
        ],
        "cost_profile": "method='att_gt' / 'dose_response' bootstrap n_boot fits.",
        "alternatives": ["sp.cgs_continuous_did", "sp.dose_response", "sp.did"],
    },
    "cgs_continuous_did": {
        "not_recommended_when": [
            "the dose has few distinct values — a spline in dose is not "
            "identified; treat the levels as arms with sp.multi_treatment",
        ],
        "failure_modes": [
            _fm(
                "ATT(d) and ACRT(d) point in different directions",
                "(none — informational)",
                "Expected under selection into dose: ACRT needs strong parallel "
                "trends; report ATT(d) unless that is credible",
                "sp.continuous_did",
            ),
        ],
    },
    "honest_did": {
        "cost_profile": (
            "method='C-LF' inverts a conditional test over the m_grid with a "
            "simulation per grid point; the FLCI path is a closed-form "
            "optimisation per M. Cost is linear in len(m_grid)."
        ),
        "alternatives": ["sp.pretrends_power", "sp.sensitivity_rr", "sp.uniform_bands"],
    },
    "sun_abraham": {
        "failure_modes": [
            _fm(
                "share_variance=True and the aggregated SE differs from fixest",
                "(none — informational)",
                "The difference is the documented cohort-share variance term "
                "(Prop. 3); share_variance=False reproduces fixest to 1e-9",
                "sp.callaway_santanna",
            ),
        ],
    },
    "stacked_did": {
        "cost_profile": (
            "The stacked panel has one copy of each clean control per cohort: "
            "memory grows as cohorts x n."
        ),
    },
    "wooldridge_did": {
        "cost_profile": "One saturated OLS with cohorts x periods interaction columns.",
    },
    "etwfe": {
        "cost_profile": "One saturated OLS with cohorts x periods interaction columns.",
    },
    # ================================================================== #
    #  IV
    # ================================================================== #
    "ivreg": {
        "pre_conditions": [
            "Formula in the form 'y ~ (endog ~ z1 + z2) + exog'; no '| fe' term — "
            "high-dimensional fixed effects need sp.feols or an explicit dummy set",
        ],
        "failure_modes": [
            _fm(
                "First-stage F below 10 in result.diagnostics",
                "statspai.exceptions.AssumptionWarning",
                "Report an Anderson-Rubin set or the tF-adjusted interval instead "
                "of the 2SLS t-ratio",
                "sp.anderson_rubin_ci",
            ),
            _fm(
                "Over-identification (Sargan-Hansen) test rejects",
                "statspai.exceptions.AssumptionWarning",
                "At least one instrument is invalid; drop the suspect instruments "
                "rather than the test",
                "sp.iv",
            ),
            _fm(
                "'| fe' in the formula",
                "statspai.exceptions.MethodIncompatibility",
                "ivreg does not absorb fixed effects; pass absorb= or use sp.feols "
                "with an IV formula",
                "sp.feols",
            ),
        ],
        "alternatives": ["sp.iv", "sp.liml", "sp.iv_diag"],
        "not_recommended_when": [
            "many instruments relative to n — 2SLS is biased toward OLS; use "
            "sp.jive or sp.rlasso_iv",
        ],
        "cost_profile": (
            "vce='wild' runs wild_reps wild-cluster bootstrap replications of the "
            "restricted-efficient 2SLS; O(wild_reps x n)."
        ),
    },
    "liml": {
        "pre_conditions": ["At least as many instruments as endogenous regressors"],
        "failure_modes": [
            _fm(
                "LIML estimate has a much larger SE than 2SLS",
                "(none — informational)",
                "LIML has no finite moments under weak identification; the Fuller "
                "modification (fuller=1) restores them",
                "sp.anderson_rubin_ci",
            ),
            _fm(
                "kappa reported as 1.0 (LIML fell back to 2SLS)",
                "statspai.exceptions.ConvergenceWarning",
                "The eigenvalue problem was degenerate (collinear instruments); "
                "check the instrument set",
                "sp.ivreg",
            ),
        ],
        "alternatives": ["sp.ivreg", "sp.anderson_rubin_ci", "sp.jive"],
        "not_recommended_when": [
            "the instruments are strong (effective F far above 10) — 2SLS is "
            "more precise and LIML buys nothing",
        ],
    },
    "lasso_iv": {
        "pre_conditions": [
            "A candidate instrument set; penalty chosen by BIC / AIC / CV"
        ],
        "failure_modes": [
            _fm(
                "No instrument selected",
                "statspai.exceptions.DataInsufficient",
                "Lower the penalty or use the rigorous-penalty selector, which is "
                "theory-driven rather than information-criterion driven",
                "sp.rlasso_iv",
            ),
        ],
        "alternatives": ["sp.rlasso_iv", "sp.jive", "sp.ivreg"],
        "not_recommended_when": [
            "valid post-selection inference is required — this is the "
            "information-criterion selector, not Belloni-Chen-Chernozhukov-"
            "Hansen; use sp.rlasso_iv",
        ],
    },
    "rlasso_iv": {
        "pre_conditions": [
            "Instruments z (p_z may exceed n) and optional high-dimensional "
            "controls x, all numeric",
        ],
        "failure_modes": [
            _fm(
                "No instruments selected by the rigorous penalty",
                "(none — informational)",
                "The first stage is not sparse-strong; the reported estimate is "
                "not identified — report weak-IV-robust sets instead",
                "sp.anderson_rubin_ci",
            ),
        ],
        "alternatives": ["sp.lasso_iv", "sp.jive", "sp.dml"],
        "not_recommended_when": [
            "the instrument set is small (a handful) — selection adds nothing "
            "and the rigorous penalty may drop a valid instrument",
        ],
    },
    "spatial_iv": {
        "pre_conditions": [
            "A row-standardised spatial weights matrix aligned to the data rows",
        ],
        "failure_modes": [
            _fm(
                "Spatial-lag instruments (W X, W^2 X) weakly correlated with W y",
                "statspai.exceptions.AssumptionWarning",
                "Add excluded instruments or use the GMM spatial estimators",
                "sp.sar_gmm",
            ),
        ],
        "alternatives": ["sp.sar_gmm", "sp.sar", "sp.ivreg"],
        "not_recommended_when": [
            "the endogeneity is not spatial (no lagged outcome) — plain sp.ivreg "
            "with Conley standard errors is the right tool",
        ],
    },
    "iv": {
        "not_recommended_when": [
            "the instrument is a policy cutoff on a running variable — that is a "
            "fuzzy RD (sp.rdrobust(..., fuzzy=)), whose local inference differs",
        ],
        "cost_profile": (
            "augmented_diagnostics=True adds bootstrap SEs and AR / CLR grid "
            "inversions; cost is n_boot + grid_size extra fits."
        ),
    },
    "iv_diag": {
        "not_recommended_when": [
            "a quick point estimate is all that is needed — this is the full "
            "reporting bundle (bootstrap SEs, tF, AR / CLR / K sets, LTZ) and "
            "costs many fits",
        ],
        "cost_profile": (
            "n_boot bootstrap fits plus grid_size test inversions per confidence "
            "set; minutes on n ~ 10^5."
        ),
    },
    "bartik": {
        "not_recommended_when": [
            "there are few shocks (industries / sectors) — shock-level asymptotics "
            "need many; report the Herfindahl of the shares and the effective "
            "number of shocks",
        ],
        "cost_profile": "Share matrix is n_units x n_industries; leave_one_out doubles the work.",
    },
    "jive": {
        "not_recommended_when": [
            "the instrument set is small — JIVE corrects many-instrument bias and "
            "is less efficient than 2SLS with one or two strong instruments",
        ],
    },
    "anderson_rubin_ci": {
        "not_recommended_when": [
            "there are several endogenous regressors — the AR set is joint over "
            "them; use the conditional LR or K sets for one coefficient",
        ],
        "cost_profile": "n_grid test evaluations over beta_grid; O(n_grid x n).",
    },
    "dist_iv": {
        "not_recommended_when": [
            "the instrument is not binary — kappa weighting needs a binary Z; "
            "discretise or use sp.continuous_iv_late for the average effect",
        ],
        "cost_profile": "se='bootstrap' re-estimates all quantiles n_boot times.",
    },
    # ================================================================== #
    #  RD
    # ================================================================== #
    "rdrobust": {
        "not_recommended_when": [
            "the running variable has few distinct values (dates, integer scores) "
            "— local polynomial asymptotics fail; use sp.rd_discrete or sp.rdit",
            "treatment probability does not change at the cutoff — there is no "
            "RD; consider sp.bunching or a DiD",
        ],
        "cost_profile": (
            "bwselect runs the pilot bandwidth estimators once; vce='nn' nearest-"
            "neighbour residuals are O(n log n). Cheap at any n."
        ),
    },
    "rddensity": {
        "not_recommended_when": [
            "the running variable is heaped or discrete — a density jump at a "
            "mass point is not manipulation; use a binned test or sp.rd_discrete",
        ],
        "failure_modes": [
            _fm(
                "Rejection with an integer-valued or rounded running variable",
                "(none — informational)",
                "Heaping produces spurious density jumps; check sp.mccrary_test "
                "with a coarser bin_width and inspect the histogram",
                "sp.mccrary_test",
            ),
        ],
    },
    "rd_bias_aware_fuzzy": {
        "pre_conditions": [
            "Fuzzy RD with a binary treatment column and smoothness bounds M_y / "
            "M_d (or their rule-of-thumb defaults)",
        ],
        "failure_modes": [
            _fm(
                "Empty or unbounded confidence set",
                "(none — informational)",
                "The first stage at the cutoff is weak; that is the information "
                "the bias-aware set is designed to convey — report it",
                "sp.rdrobust",
            ),
        ],
        "alternatives": ["sp.rdrobust", "sp.rd_honest", "sp.anderson_rubin_ci"],
        "not_recommended_when": [
            "the design is sharp — use sp.rd_honest for honest inference without "
            "a first stage",
        ],
        "cost_profile": "Test inversion over n_grid candidate effects; O(n_grid x n).",
    },
    "rd_discrete": {
        "pre_conditions": [
            "A running variable with a moderate number of distinct values on each "
            "side of the cutoff (dates, integer scores)",
        ],
        "failure_modes": [
            _fm(
                "Interval much wider than sp.rdrobust on the same data",
                "(none — informational)",
                "Expected: rdrobust's asymptotics assume a continuous score and "
                "understate uncertainty at mass points",
                "sp.rd_honest",
            ),
            _fm(
                "Fewer than three support points on one side",
                "statspai.exceptions.DataInsufficient",
                "The specification error is not identified; pool or use a "
                "parametric fit and say so",
                "sp.rdit",
            ),
        ],
        "alternatives": ["sp.rd_honest", "sp.rdit", "sp.rdrobust"],
    },
    "rdd": {
        "alternatives": ["sp.rdrobust", "sp.rd"],
        "not_recommended_when": [
            "anything beyond the blog-post signature is needed — call sp.rdrobust "
            "directly for bandwidth, kernel, covariates and clustering options",
        ],
    },
    "rdhte": {
        "not_recommended_when": [
            "the moderator z is continuous with many levels and n near the cutoff "
            "is small — the interaction fit is under-powered; use sp.rd_forest or "
            "coarsen z",
        ],
        "cost_profile": "One kernel-weighted least squares per evaluation point.",
    },
    "rdmc": {
        "not_recommended_when": [
            "every unit faces the same cutoff — pooling across cutoffs is then a "
            "single RD; use sp.rdrobust",
        ],
        "failure_modes": [
            _fm(
                "Cutoff-specific effects differ in sign",
                "(none — informational)",
                "The pooled estimate averages heterogeneous local effects; report "
                "the per-cutoff table, not only the pooled number",
                "sp.rdrobust",
            ),
        ],
        "cost_profile": "One sp.rdrobust fit per cutoff.",
    },
    "rd2d": {
        "not_recommended_when": [
            "the boundary can be reduced to a signed distance and the effect is "
            "assumed constant along it — sp.geographic_rd is the simpler design",
        ],
        "cost_profile": "One local fit per evaluation point along the boundary (n_eval).",
    },
    "rdrandinf": {
        "assumptions": [
            "Local randomisation: inside the window (wl, wr) treatment is as good "
            "as randomly assigned, so potential outcomes are unrelated to the "
            "score there (no continuity assumption is needed)",
            "Units cannot precisely manipulate the running variable around the "
            "cutoff (no sorting)",
        ],
        "pre_conditions": [
            "A window (wl, wr) chosen by covariate balance (sp.rdwinselect), not by "
            "the outcome",
        ],
        "failure_modes": [
            _fm(
                "Very few units inside the window",
                "statspai.exceptions.DataInsufficient",
                "Widen the window only if balance still holds; otherwise the "
                "local-randomization design has no power here",
                "sp.rdrobust",
            ),
        ],
        "alternatives": ["sp.rdwinselect", "sp.rdrobust", "sp.rd_honest"],
        "not_recommended_when": [
            "the window is wide enough that outcomes trend with the score inside "
            "it — as-if randomisation is no longer credible; use the "
            "continuity-based estimator",
        ],
        "cost_profile": "n_perms permutations of the treatment vector inside the window.",
    },
    "rdit": {
        "pre_conditions": [
            "A time-indexed outcome with a known policy date; autocorrelation "
            "handled by HAC / clustered errors",
        ],
        "failure_modes": [
            _fm(
                "Seasonality around the cutoff date",
                "statspai.exceptions.AssumptionWarning",
                "Pass seasonality= controls or a donut around the date; a seasonal "
                "cycle can masquerade as a jump",
                "sp.its",
            ),
        ],
        "alternatives": ["sp.its", "sp.rd_discrete", "sp.causal_impact"],
        "not_recommended_when": [
            "a control series exists — sp.causal_impact / sp.synth absorb common "
            "shocks that a single time series attributes to the policy",
        ],
    },
    "rkd": {
        "pre_conditions": [
            "A known kink point where the *slope* of the assignment rule changes",
        ],
        "failure_modes": [
            _fm(
                "Estimated slope change in treatment (first-stage kink) near zero",
                "statspai.exceptions.AssumptionWarning",
                "A fuzzy RKD with no first-stage kink is not identified; check the "
                "policy rule",
                "sp.rdrobust",
            ),
        ],
        "not_recommended_when": [
            "the assignment rule changes in level, not slope — that is an RD "
            "(sp.rdrobust), and the kink estimator will read a jump as a huge "
            "slope change",
        ],
        "cost_profile": "Same order as sp.rdrobust with deriv=1.",
    },
    "rd_honest": {
        "not_recommended_when": [
            "the smoothness bound M cannot be defended — an honest CI is only as "
            "honest as M; report results for a range of M",
        ],
    },
    "mccrary_test": {
        "not_recommended_when": [
            "the running variable is discrete or heaped — the histogram-based "
            "test rejects on mass points; prefer sp.rddensity with care or a "
            "binomial test at the heap",
        ],
    },
    "geographic_rd": {
        "not_recommended_when": [
            "the effect plausibly varies along the boundary — sp.rd2d reports it "
            "pointwise",
        ],
    },
    "multi_cutoff_rd": {"alternatives": ["sp.rdmc", "sp.rdrobust"]},
    "rd_forest": {
        "not_recommended_when": [
            "the sample inside the bandwidth is small (hundreds) — a forest in the "
            "moderators has no signal; use sp.rdhte with one or two moderators",
        ],
        "cost_profile": "Two random forests with n_trees each inside the bandwidth.",
    },
    "rd_extrapolate": {
        "not_recommended_when": [
            "the conditional-independence-given-covariates assumption cannot be "
            "argued — extrapolation away from the cutoff rests entirely on it",
        ],
    },
    # ================================================================== #
    #  Synthetic control
    # ================================================================== #
    "bayes_synth": {
        "pre_conditions": [
            "One treated unit, a donor pool and enough pre-periods to fit the "
            "Dirichlet-weighted path; pymc installed (statspai[bayes])",
        ],
        "failure_modes": [
            _fm(
                "R-hat above 1.01 or divergences reported",
                "statspai.exceptions.ConvergenceWarning",
                "Increase tune / draws, raise target_accept, or use inference="
                "'advi' for a first look; do not report a non-converged posterior",
                "sp.synth",
            ),
        ],
        "alternatives": ["sp.synth", "sp.conformal_synth", "sp.scpi"],
        "not_recommended_when": [
            "frequentist placebo inference is what the reader expects — sp.synth "
            "with inference='placebo' is the standard",
        ],
        "cost_profile": "NUTS over the weight simplex: draws x chains posterior samples; minutes.",
    },
    "synth_loo": {
        "failure_modes": [
            _fm(
                "ATT changes sign when one donor is dropped",
                "(none — diagnostic)",
                "The estimate rests on that donor; report the leave-one-out range "
                "and justify the donor or drop it in the main specification",
                "sp.synth_donor_sensitivity",
            ),
        ],
        "alternatives": [
            "sp.synth_donor_sensitivity",
            "sp.synth_rmspe_filter",
            "sp.synth",
        ],
        "cost_profile": "One SCM refit per donor.",
    },
    "synth_power": {
        "pre_conditions": [
            "The design (donors, pre-period) fixed before looking at post-period outcomes"
        ],
        "failure_modes": [
            _fm(
                "MDE larger than any plausible effect",
                "(none — planning)",
                "The design cannot detect the effect; add pre-periods or donors, "
                "or report the null as uninformative rather than as no effect",
                "sp.synth_mde",
            ),
        ],
        "alternatives": ["sp.synth_mde", "sp.synth"],
        "cost_profile": "n_simulations x len(effect_sizes) placebo-inference SCM fits.",
    },
    "synth_mde": {
        "alternatives": ["sp.synth_power", "sp.synth"],
        "cost_profile": "Same as sp.synth_power.",
    },
    "synth_rmspe_filter": {
        "failure_modes": [
            _fm(
                "p-value changes materially across thresholds",
                "(none — diagnostic)",
                "The placebo pool is sensitive to badly fitted placebos; report "
                "the full threshold table (Abadie et al. 2010 report 2x, 5x, 20x)",
                "sp.synth",
            ),
        ],
        "alternatives": ["sp.synth", "sp.synth_loo"],
        "cost_profile": "One placebo SCM per donor.",
    },
    "synth_time_placebo": {
        "failure_modes": [
            _fm(
                "Large 'effect' at a fake pre-treatment date",
                "statspai.exceptions.AssumptionWarning",
                "The synthetic control does not track the treated unit; the real "
                "estimate is not credible at that pre-period fit",
                "sp.augsynth",
            ),
        ],
        "alternatives": ["sp.synth", "sp.augsynth"],
        "cost_profile": "n_placebo_times SCM refits.",
    },
    "synth_donor_sensitivity": {
        "failure_modes": [
            _fm(
                "Wide spread of ATT across donor subsets",
                "(none — diagnostic)",
                "The counterfactual depends on which donors are available; report "
                "the distribution alongside the point estimate",
                "sp.synth_loo",
            ),
        ],
        "alternatives": ["sp.synth_loo", "sp.synth"],
        "cost_profile": "n_samples SCM refits.",
    },
    "synth_compare": {
        "not_recommended_when": [
            "the method has already been chosen on substantive grounds — running "
            "all 20 variants and picking the best-fitting one is specification "
            "search; use it to report a table, not to select",
        ],
        "alternatives": ["sp.synth_recommend", "sp.synth"],
    },
    "synth_recommend": {
        "not_recommended_when": [
            "the recommendation would be used as the pre-registered choice — it is "
            "a fit-based heuristic (placebo inference off), not a design argument",
        ],
        "alternatives": ["sp.synth_compare", "sp.synth", "sp.route"],
    },
    "augsynth": {
        "cost_profile": "One ridge solve on the donor pre-period matrix; placebo=True adds one fit per donor.",
    },
    "scpi": {
        "not_recommended_when": [
            "several units are treated — this ports the single-treated-unit path; "
            "use sp.sdid or sp.gsynth",
        ],
    },
    "mc_synth": {
        "not_recommended_when": [
            "the pre-period is short relative to the rank being fitted — nuclear-"
            "norm completion cannot separate signal from noise",
        ],
        "cost_profile": "Iterative SVD on the N x T panel per lambda and per CV fold.",
    },
    "demeaned_synth": {
        "not_recommended_when": [
            "covariates must enter the weights — this variant fits on outcomes "
            "only and raises on covariates=",
        ],
    },
    "robust_synth": {
        "not_recommended_when": [
            "negative weights are not interpretable for the application — the "
            "unconstrained fit extrapolates outside the donor hull",
        ],
    },
    "conformal_synth": {
        "cost_profile": (
            "Re-estimates the synthetic control under every null on the grid and "
            "permutes residuals: grid_size x (T + permutations) fits."
        ),
    },
    "multi_outcome_synth": {
        "not_recommended_when": [
            "the outcomes have different scales and standardize=False — one "
            "outcome dominates the shared weights",
        ],
    },
    "matrix_completion": {
        "assumptions": [
            "Untreated potential outcomes follow a low-rank (plus noise) structure "
            "shared by treated and control cells",
            "Treatment assignment is ignorable given the latent factors (no "
            "selection on the idiosyncratic error)",
            "No anticipation and SUTVA",
        ],
        "pre_conditions": [
            "Long panel with a 0/1 treatment column; enough untreated cells per "
            "unit and per period to identify the factors",
        ],
        "failure_modes": [
            _fm(
                "Pre-treatment fit of the completed matrix is poor",
                "statspai.exceptions.AssumptionWarning",
                "The low-rank model does not describe Y(0); compare with "
                "sp.gsynth and sp.did_imputation before reporting",
                "sp.gsynth",
            ),
        ],
        "alternatives": ["sp.mc_panel", "sp.gsynth", "sp.fect"],
        "not_recommended_when": [
            "the panel is nearly fully treated — too few untreated cells to "
            "complete the matrix",
        ],
        "cost_profile": "Iterative soft-impute SVD on the N x T matrix per lambda; n_bootstrap refits for SEs.",
    },
    "mc_panel": {
        "cost_profile": "Iterative soft-impute SVD on the N x T matrix per lambda; n_bootstrap refits for SEs.",
    },
    "sdid": {
        "not_recommended_when": [
            "the parallel-trends estimand with unit weights is wanted — sdid "
            "re-weights both units and periods and its estimand differs from CS",
        ],
        "failure_modes": [
            _fm(
                "se_method='jackknife' refused for a single treated unit",
                "statspai.exceptions.MethodIncompatibility",
                "Use se_method='placebo' (Stata's rule) — the jackknife needs two "
                "treated units per cohort",
                "sp.synth",
            ),
        ],
    },
    # ================================================================== #
    #  DML
    # ================================================================== #
    "dml": {
        "not_recommended_when": [
            "the treatment is assigned repeatedly and moves later covariates — "
            "static DML is biased; use sp.dynamic_dml",
            "unit fixed effects carry the confounding — use sp.dml_panel",
        ],
        "cost_profile": (
            "n_folds x n_rep nuisance fits per model (two learners for PLR / IRM, "
            "three or four for PLIV / IIVM); a gradient-boosting learner on n ~ "
            "10^5 takes minutes per repetition."
        ),
    },
    "dml_panel": {
        "not_recommended_when": [
            "T is short and treatment varies little within unit — the within "
            "transform removes most of the treatment variation",
        ],
        "cost_profile": "n_folds cross-fits on the demeaned panel; folds split units.",
    },
    "dynamic_dml": {
        "not_recommended_when": [
            "treatment is assigned once — the static estimator (sp.dml) is the "
            "right one and the dynamic moment system is over-parameterised",
            "lags=0 is tempting — without treatment history in the state the "
            "estimator is confidently wrong (documented bias of -15% / +37%)",
        ],
        "cost_profile": "periods x n_folds nuisance fits for the outcome and each period's treatment.",
    },
    "model_averaging_dml": {
        "alternatives": ["sp.dml_model_averaging", "sp.dml"],
        "not_recommended_when": [
            "one learner is known to be right for the data — averaging trades "
            "a little efficiency for robustness to learner choice",
        ],
        "cost_profile": "len(candidates) full DML-PLR fits.",
    },
    "dml_model_averaging": {
        "not_recommended_when": [
            "one learner is known to be right for the data — averaging trades "
            "a little efficiency for robustness to learner choice",
        ],
        "cost_profile": "len(candidates) full DML-PLR fits.",
    },
    "dml_diagnostics": {
        "pre_conditions": [
            "A CausalResult from sp.dml with post-fit residuals stored "
            "(store_oof / default model_info)",
        ],
        "failure_modes": [
            _fm(
                "Residuals missing from model_info",
                "statspai.exceptions.MethodIncompatibility",
                "Re-fit with sp.dml (not an external OOF import) so the "
                "orthogonality diagnostics can be computed",
                "sp.dml",
            ),
        ],
        "alternatives": ["sp.dml_sensitivity", "sp.dml"],
    },
    "dml_sensitivity": {
        "not_recommended_when": [
            "the benchmark covariates are weak — the calibrated cf_y / cf_d "
            "then say little about a plausible unobserved confounder",
        ],
    },
    "conformal_debiased_ml": {
        "pre_conditions": [
            "Cross-fitting folds and, for coverage, a test_data frame exchangeable "
            "with the training data",
        ],
        "failure_modes": [
            _fm(
                "Intervals very wide relative to the CATE spread",
                "(none — informational)",
                "Marginal coverage at 1 - alpha over noisy outcomes is wide by "
                "construction; report interval length alongside coverage",
                "sp.conformal_cate",
            ),
        ],
        "alternatives": ["sp.conformal_cate", "sp.dml", "sp.causal_forest"],
        "not_recommended_when": [
            "the guarantee needed is conditional on x — conformal intervals are "
            "marginal",
        ],
    },
}


__all__ = ["FUNCTION_CARDS"]
