"""Per-function agent cards, batch 3: estimators and diagnostics that had none.

The 2026-09-28 registry audit counted 232 callables without an agent card,
including estimators and specification tests an agent reaches for directly
(``kitagawa_test``, ``ppi_ols``, ``bayes_its``, ``cluster_staggered_rollout``
…). Each card below was written from the function's own signature, docstring
and source: an assumption is listed only when the docstring states it or the
code relies on it, a failure mode only when the code raises (or warns) on it,
and an alternative only when it is a registered entry point for the same
question. No card cites literature; method references live in the
docstrings.

Merged into :data:`statspai._function_cards.FUNCTION_CARDS` (field by field,
existing entries first), so the precedence and contract tests in
``tests/test_function_cards.py`` cover these cards too.
"""

# flake8: noqa: E501  (descriptive-string data module; long lines are content)
from __future__ import annotations

from typing import Any, Dict

Card = Dict[str, Any]

_MI = "statspai.exceptions.MethodIncompatibility"
_DI = "statspai.exceptions.DataInsufficient"
_NI = "statspai.exceptions.NumericalInstability"


def _fm(
    symptom: str, exception: str, remedy: str, alternative: str = ""
) -> Dict[str, str]:
    out = {"symptom": symptom, "exception": exception, "remedy": remedy}
    if alternative:
        out["alternative"] = alternative
    return out


ESTIMATOR_CARDS: Dict[str, Card] = {
    # ------------------------------------------------------------------ #
    #  IV / LATE diagnostics
    # ------------------------------------------------------------------ #
    "kitagawa_test": {
        "assumptions": [
            "Tests the joint LATE conditions (instrument independence, exclusion, "
            "monotonicity) through their testable implication: the implied complier "
            "potential-outcome distributions must be proper CDFs",
            "A non-rejection is not evidence that the LATE assumptions hold; the "
            "test only has power against violations that bend the complier CDFs",
        ],
        "pre_conditions": [
            "Binary (0/1) instrument and binary (0/1) treatment columns",
            "A first stage that is not essentially zero",
        ],
        "failure_modes": [
            _fm(
                "Instrument or treatment is not coded 0/1",
                "ValueError",
                "Recode to a binary instrument and a binary treatment; the test is "
                "defined for the binary LATE design only",
            ),
            _fm(
                "First stage is essentially zero",
                "ValueError",
                "The instrument does not move the treatment, so there are no "
                "compliers to test; check the first stage before anything else",
                "sp.iv",
            ),
        ],
        "alternatives": ["sp.iv", "sp.anderson_rubin_ci"],
        "not_recommended_when": [
            "the treatment or instrument is multi-valued or continuous — the test "
            "is defined for the binary instrument / binary treatment design",
        ],
        "cost_profile": (
            "n_boot bootstrap replications (default 1000), each re-evaluating the "
            "CDF conditions on n_grid points."
        ),
    },
    "zero_first_stage": {
        "assumptions": [
            "In the zero-first-stage subsample the instrument does not move the "
            "treatment, so any reduced-form association there is a direct effect "
            "of the instrument on the outcome",
            "That direct effect is the same in the main sample, so it can be netted "
            "out of the main-sample IV estimate",
        ],
        "pre_conditions": [
            "One excluded instrument being tested",
            "A zfs marker (boolean or 0/1) identifying observations where the "
            "instrument is believed inert for the treatment",
            "Both the main and the zero-first-stage subsample populated",
        ],
        "failure_modes": [
            _fm(
                "zfs column missing or its length does not match data",
                _MI,
                "Pass a column name present in data or a boolean mask of len(data)",
            ),
            _fm(
                "Main or zero-first-stage subsample too small",
                _DI,
                "Enlarge the zero-first-stage group; the direct-effect regression "
                "needs observations of its own",
            ),
            _fm(
                "Some bootstrap replications failed",
                "(none — warning)",
                "The corrected estimate's SE rests on the successful draws only; "
                "check the warning count and consider more clusters or n_boot",
            ),
        ],
        "alternatives": ["sp.iv", "sp.anderson_rubin_ci"],
        "not_recommended_when": [
            "there is no credible subsample in which the instrument is inert — the "
            "test then has nothing to identify the direct effect from",
        ],
        "cost_profile": (
            "n_boot (default 999) cluster-bootstrap refits of every component "
            "regression; n_boot=0 skips it and leaves the corrected SE NaN."
        ),
    },
    # ------------------------------------------------------------------ #
    #  Sensitivity
    # ------------------------------------------------------------------ #
    "evalue_rd": {
        "assumptions": [
            "The exposure is coded so that the observed risk difference is "
            "non-negative",
            "The E-value is the minimum confounding strength (bias factor) needed "
            "to move the observed risk difference to the reference value; it does "
            "not estimate the actual confounding",
        ],
        "pre_conditions": [
            "The four 2x2 cell counts (exposed / unexposed x case / non-case), all "
            "non-negative, with at least one unit in each exposure group",
        ],
        "failure_modes": [
            _fm(
                "Observed risk difference is negative",
                _MI,
                "Relabel the exposure so the risk difference is positive",
            ),
            _fm(
                "Reference value true is not below the observed risk difference",
                _MI,
                "Choose true strictly below the observed RD (0 by default)",
            ),
            _fm(
                "An exposure group is empty or a cell count is negative",
                _DI,
                "Check the 2x2 table; the RD is undefined without both groups",
            ),
        ],
        "alternatives": ["sp.evalue", "sp.sensemakr"],
        "not_recommended_when": [
            "the effect is a regression coefficient rather than a 2x2 risk "
            "difference — use sp.evalue on the ratio scale or sp.sensemakr",
        ],
    },
    "copula_sensitivity": {
        "assumptions": [
            "One latent unit-level confounder U, jointly Gaussian with the outcome "
            "through a single correlation rho (Gaussian copula)",
            "The bias of the point estimate is linear in rho, scaled by the "
            "confounder and outcome standard deviations with the treatment scale "
            "normalised to one",
        ],
        "pre_conditions": [
            "A point estimate and its standard error from an OLS / DML-type fit",
        ],
        "failure_modes": [
            _fm(
                "Breakpoint rho* lies outside the grid",
                "(none — diagnostic)",
                "Widen rho_grid; the default sweeps -0.5 to 0.5",
            ),
        ],
        "alternatives": ["sp.sensemakr", "sp.evalue", "sp.dml_sensitivity"],
        "not_recommended_when": [
            "the confounding is not well summarised by one Gaussian latent variable "
            "— the linear bias formula is then only a heuristic; prefer "
            "sp.sensemakr's partial-R2 bounds",
        ],
    },
    # ------------------------------------------------------------------ #
    #  Prediction-powered inference
    # ------------------------------------------------------------------ #
    "ppi_mean": {
        "assumptions": [
            "Labeled rows are a random sample from the same population as the "
            "unlabeled rows, so the labeled y - yhat contrast rectifies the "
            "prediction bias",
            "Valid whatever the prediction quality; predictions only change the "
            "width of the interval",
        ],
        "pre_conditions": [
            "Paired y and yhat on the labeled rows, and yhat on the unlabeled rows",
            "At least 4 labeled and 4 unlabeled rows",
        ],
        "failure_modes": [
            _fm(
                "y and yhat have different lengths",
                _MI,
                "Pass paired rows of the labeled sample",
            ),
            _fm(
                "Too few labeled or unlabeled rows",
                _DI,
                "PPI needs a variance estimate from each sample; add rows",
            ),
            _fm(
                "Variance estimate is zero or non-finite",
                _NI,
                "Check for constant outcomes or predictions",
            ),
        ],
        "alternatives": ["sp.ppi_ols"],
        "not_recommended_when": [
            "the labeled sample was selected on the outcome or the predictions — "
            "the rectifier is then biased and coverage fails",
        ],
    },
    "ppi_ols": {
        "assumptions": [
            "Labeled rows are a random sample from the same population as the "
            "unlabeled rows",
            "The target is the population OLS projection; coefficients are "
            "rectified by the labeled-sample OLS(y) - OLS(yhat) contrast",
        ],
        "pre_conditions": [
            "Labeled X, y, yhat and unlabeled X_unlabeled, yhat_unlabeled with the "
            "same covariate columns",
            "More labeled and unlabeled rows than coefficients",
        ],
        "failure_modes": [
            _fm(
                "X and X_unlabeled columns differ, or row counts do not match",
                _MI,
                "Align the covariate columns and pair y / yhat with X row by row",
            ),
            _fm(
                "Labeled or unlabeled sample smaller than the number of parameters",
                _DI,
                "Add rows or drop covariates",
            ),
            _fm(
                "Variance estimates degenerate",
                _NI,
                "Check for collinear covariates or constant predictions",
            ),
        ],
        "alternatives": ["sp.ppi_mean", "sp.regress"],
        "not_recommended_when": [
            "all outcomes are observed — ordinary sp.regress is the same target "
            "without the prediction step",
        ],
    },
    # ------------------------------------------------------------------ #
    #  Bayesian / time series
    # ------------------------------------------------------------------ #
    "bayes_its": {
        "assumptions": [
            "Absent the intervention the pre-period linear trend would have "
            "continued (segmented-regression counterfactual)",
            "The intervention causes an immediate level change and a slope change "
            "at one known row position; nothing else changes then",
            "Posterior conclusions depend on the stated priors on level change, "
            "slope change, trend and noise",
        ],
        "pre_conditions": [
            "One series sorted in time with the intervention given as an integer "
            "row position strictly inside the series",
            "No NaN / inf in the outcome or time column",
            "PyMC installed (the bayes extra) for inference='nuts'",
        ],
        "failure_modes": [
            _fm(
                "intervention missing or not strictly inside the series",
                "ValueError",
                "Pass the integer row position where the intervention begins",
            ),
            _fm(
                "Outcome or time contains NaN / inf",
                "ValueError",
                "Clean the series first",
            ),
        ],
        "alternatives": ["sp.its", "sp.causal_impact", "sp.synth"],
        "not_recommended_when": [
            "a comparable untreated series exists — a comparison-group design "
            "(sp.synth, sp.causal_impact) does not rely on the trend extrapolation "
            "alone",
        ],
        "cost_profile": (
            "NUTS sampling: draws x chains posterior draws after tune warm-up "
            "(default 2000 x 4); inference='advi' is faster and approximate."
        ),
    },
    "policy_weight_prte": {
        "assumptions": [
            "Returns a stylised rectangle weight around propensity 0.5, not the "
            "textbook policy-relevant treatment effect weight, which depends on the "
            "sample's propensity distribution",
        ],
        "pre_conditions": [
            "shift in (-1, 1) and non-zero",
        ],
        "failure_modes": [
            _fm(
                "shift outside (-1, 1) or equal to zero",
                "ValueError",
                "Pass a non-zero propensity-scale shift inside (-1, 1)",
            ),
        ],
        "alternatives": ["sp.policy_weight_observed_prte", "sp.bayes_mte"],
        "not_recommended_when": [
            "the exact policy-relevant weight for the observed propensity "
            "distribution is needed — use sp.policy_weight_observed_prte",
        ],
    },
    # ------------------------------------------------------------------ #
    #  Conformal
    # ------------------------------------------------------------------ #
    "conformal_counterfactual": {
        "assumptions": [
            "Units are exchangeable within each treatment arm conditional on the "
            "covariates (no unmeasured confounding)",
            "The propensity score is correctly estimated; it is the covariate-shift "
            "weight between each arm and the full population",
            "Coverage is marginal (1 - alpha on average over X), not conditional "
            "on a particular x",
        ],
        "pre_conditions": [
            "Binary 0/1 treatment, and both arms large enough to split into a "
            "training and a calibration part (calib_frac)",
        ],
        "failure_modes": [
            _fm(
                "Intervals very wide where propensities are extreme",
                "(none — diagnostic)",
                "Propensities are clipped to [0.001, 0.999]; heavy weights inflate "
                "the weighted quantile — trim the sample to overlap first",
                "sp.overlap_plot",
            ),
        ],
        "alternatives": ["sp.conformal_cate", "sp.conformal_debiased_ml"],
    },
    "conformal_debiased_ml": {
        "assumptions": [
            "Binary treatment and no unmeasured confounding given the covariates",
            "Coverage is marginal over the test distribution, not conditional on "
            "a particular x",
        ],
    },
    "conformal_density_ite": {
        "assumptions": [
            "Binary treatment and no unmeasured confounding given the covariates",
            "Intervals are highest-density sets built from a kernel estimate of "
            "the counterfactual density, so they can be asymmetric or narrower "
            "than mean-based intervals under skew",
            "Coverage is marginal, calibrated on a held-out half of the data",
        ],
        "pre_conditions": [
            "Binary treatment column; enough calibration residuals in both arms",
        ],
        "failure_modes": [
            _fm(
                "Treatment is not binary",
                "ValueError",
                "Recode the treatment to 0/1",
            ),
            _fm(
                "Fewer than 5 calibration residuals",
                "(none — silent fallback)",
                "The interval falls back to a Gaussian approximation; add data "
                "before relying on the density shape",
            ),
        ],
        "alternatives": ["sp.conformal_cate", "sp.conformal_counterfactual"],
        "not_recommended_when": [
            "the conditional outcome distribution is roughly symmetric — the mean-"
            "based sp.conformal_cate is simpler and equally sharp",
        ],
    },
    "conformal_ite_multidp": {
        "assumptions": [
            "Sequential no unmeasured confounding: each stage's treatment is "
            "as-good-as-random given the history recorded for that stage",
            "Joint coverage across stages comes from a Bonferroni adjustment, so "
            "per-stage intervals are conservative",
        ],
        "pre_conditions": [
            "One row per subject with equal-length lists of stage outcomes, binary "
            "stage treatments and non-empty stage histories",
        ],
        "failure_modes": [
            _fm(
                "Stage lists of different lengths, or an empty history",
                "ValueError",
                "Pass one outcome, one treatment and a non-empty covariate list per "
                "stage",
            ),
        ],
        "alternatives": ["sp.conformal_cate", "sp.q_learning"],
        "not_recommended_when": [
            "there is a single decision point — use sp.conformal_cate",
        ],
    },
    # ------------------------------------------------------------------ #
    #  Distributional synthetic control
    # ------------------------------------------------------------------ #
    "stochastic_dominance": {
        "assumptions": [
            "The counterfactual quantile function from the distributional "
            "synthetic control fit is a valid stand-in for the treated unit's "
            "untreated distribution",
        ],
        "pre_conditions": [
            "A result from sp.discos (or qqsynth) carrying treated and "
            "counterfactual quantiles in model_info",
        ],
        "failure_modes": [
            _fm(
                "order is not 1 or 2",
                "ValueError",
                "Use order=1 (CDF dominance) or order=2 (integrated CDF)",
            ),
        ],
        "alternatives": ["sp.discos"],
        "not_recommended_when": [
            "the input came from a mean-based synthetic control — there are no "
            "counterfactual quantiles to compare",
        ],
    },
    # ------------------------------------------------------------------ #
    #  Cluster RCTs
    # ------------------------------------------------------------------ #
    "cluster_staggered_rollout": {
        "assumptions": [
            "Clusters' adoption times are as-good-as-random and treatment is "
            "absorbing",
            "Never-treated clusters are the comparison group; each cohort's effect "
            "is measured against the period just before its adoption",
            "Event-time effects are the simple mean across cohorts (not weighted "
            "by cohort size)",
        ],
        "pre_conditions": [
            "Long panel with a cluster id, a time column and first_treat (0 or NaN "
            "for never-treated)",
            "At least one never-treated cluster and the pre-adoption period c-1 "
            "observed for each cohort",
        ],
        "failure_modes": [
            _fm(
                "No never-treated clusters",
                "ValueError",
                "The estimator needs a never-treated comparison group; use a "
                "not-yet-treated design instead",
                "sp.callaway_santanna",
            ),
            _fm(
                "No event-time ATT could be formed",
                "ValueError",
                "Check that time and first_treat share a calendar and that c-1 "
                "exists for each cohort",
            ),
        ],
        "alternatives": ["sp.callaway_santanna", "sp.staggered_rollout"],
        "not_recommended_when": [
            "adoption timing was not randomised — use a staggered DiD estimator "
            "and defend parallel trends",
        ],
        "cost_profile": (
            "Cluster bootstrap with 100 resamples, each rebuilding every "
            "cohort x event-time contrast."
        ),
    },
    "cluster_matched_pair": {
        "assumptions": [
            "Clusters were paired on baseline covariates and one cluster of each "
            "pair was randomly assigned to treatment",
            "Each pair contains exactly two clusters, one treated and one control",
        ],
        "pre_conditions": [
            "Individual-level rows with cluster id, cluster-level binary treatment "
            "and pair id",
            "At least two valid pairs",
        ],
        "failure_modes": [
            _fm(
                "Fewer than two valid pairs",
                "ValueError",
                "Check the pair column: each pair needs one treated and one control "
                "cluster",
            ),
        ],
        "alternatives": ["sp.regress", "sp.wild_cluster_bootstrap"],
        "not_recommended_when": [
            "assignment was not pair-randomised — the matched-pair variance "
            "estimator does not apply",
        ],
    },
    # ------------------------------------------------------------------ #
    #  Shift-share
    # ------------------------------------------------------------------ #
    "ssaggregate": {
        "assumptions": [
            "Identification comes from as-good-as-random shocks, with exposure "
            "shares taken as given",
            "The reported SE of x is the exposure-robust (AKM) SE, which treats the "
            "shocks rather than the locations as the source of randomness",
        ],
        "pre_conditions": [
            "An n x K share matrix aligned with the data rows and, for IV mode, a "
            "length-K shock vector",
        ],
        "failure_modes": [
            _fm(
                "shares rows, shock length or controls do not match the data",
                "ValueError",
                "Align shares with the rows of data and shocks with the share "
                "columns",
            ),
            _fm(
                "Shares do not sum to one and their sum is not a control",
                "(none — warning)",
                "Add the sum of shares to controls so the shock-level and "
                "location-level coefficients coincide",
            ),
        ],
        "alternatives": ["sp.bartik", "sp.shift_share_se", "sp.ivreg"],
        "not_recommended_when": [
            "identification rests on exogenous shares rather than shocks — the "
            "shock-level view and AKM SEs answer a different design",
        ],
    },
    # ------------------------------------------------------------------ #
    #  RD diagnostics
    # ------------------------------------------------------------------ #
    "rdbalance": {
        "assumptions": [
            "Pre-treatment covariates should be continuous at the cutoff when the "
            "RD design is valid, so a discontinuity signals sorting or a compound "
            "treatment",
        ],
        "pre_conditions": [
            "Running variable plus pre-treatment covariates (all numeric columns "
            "except x are tested when covs is omitted)",
        ],
        "failure_modes": [
            _fm(
                "Several covariates jump at the cutoff",
                "(none — diagnostic)",
                "Check manipulation with sp.rddensity and interpret the RD estimate "
                "with caution",
                "sp.rddensity",
            ),
        ],
        "alternatives": ["sp.rddensity", "sp.rdplacebo"],
        "not_recommended_when": [
            "covs includes post-treatment variables — they can jump because of "
            "the treatment itself",
        ],
    },
    "rdplacebo": {
        "assumptions": [
            "No treatment effect exists at the placebo cutoffs, so significant "
            "estimates there indicate misspecification or other discontinuities",
        ],
        "pre_conditions": [
            "Enough support on the chosen side(s) of the true cutoff to fit local "
            "polynomials at each placebo cutoff",
        ],
        "failure_modes": [
            _fm(
                "Many placebo cutoffs significant",
                "(none — diagnostic)",
                "The functional form or bandwidth is not capturing the regression "
                "function; revisit p / bandwidth before trusting the main estimate",
                "sp.rdbwsensitivity",
            ),
        ],
        "alternatives": ["sp.rdbwsensitivity", "sp.rdbalance"],
        "not_recommended_when": [
            "placebo cutoffs would straddle the true cutoff's bandwidth — use "
            "side='left' / 'right' so treated and control observations are not "
            "mixed",
        ],
        "cost_profile": "One RD fit per placebo cutoff (n_placebo, default 10).",
    },
    "rdbwsensitivity": {
        "assumptions": [
            "A credible RD estimate should be stable across a range of bandwidths "
            "around the MSE-optimal one",
        ],
        "failure_modes": [
            _fm(
                "Estimate changes sign or significance across the grid",
                "(none — diagnostic)",
                "Report the sensitivity table; prefer bias-aware inference",
                "sp.rd_honest",
            ),
        ],
        "alternatives": ["sp.rdplacebo", "sp.rd_honest"],
        "cost_profile": "One RD fit per bandwidth grid point (n_grid, default 15).",
    },
    # ------------------------------------------------------------------ #
    #  Synthetic DiD / DiD inference
    # ------------------------------------------------------------------ #
    "synthdid_placebo": {
        "assumptions": [
            "Control units are exchangeable with the treated unit, so the "
            "distribution of placebo estimates approximates the null distribution",
        ],
        "pre_conditions": [
            "Same inputs as sp.sdid; enough control units for a placebo "
            "distribution",
        ],
        "alternatives": ["sp.sdid", "sp.synth"],
        "not_recommended_when": [
            "there are only a handful of control units — the placebo distribution "
            "has too few draws to calibrate a test",
        ],
        "cost_profile": "One full sdid / sc / did fit per control unit.",
    },
    "bjs_pretrend_joint": {
        "assumptions": [
            "Pre-treatment imputation coefficients are zero under parallel trends "
            "and no anticipation; the test is a joint Wald test of that null",
            "Clusters are independent, so resampling whole clusters reproduces the "
            "sampling covariance of the pre-period coefficients",
        ],
        "pre_conditions": [
            "A did_imputation result whose event study includes negative "
            "horizons, plus the same data and arguments used to fit it",
        ],
        "failure_modes": [
            _fm(
                "Result has no event-study table or no pre-treatment horizons",
                "ValueError",
                "Re-run sp.did_imputation with a horizon that includes negative "
                "values",
                "sp.did_imputation",
            ),
            _fm(
                "Too few bootstrap replications succeeded",
                "RuntimeError",
                "Increase n_boot or check that resampled panels keep treated and "
                "untreated observations",
            ),
        ],
        "alternatives": ["sp.pretrends_test", "sp.honest_did"],
        "cost_profile": (
            "n_boot full imputation re-fits (default 300); the docstring's worked "
            "figure is about 90 s on a 10,000-row panel with 10 horizons."
        ),
    },
    # ------------------------------------------------------------------ #
    #  Design / power
    # ------------------------------------------------------------------ #
    "randomize": {
        "assumptions": [
            "Assignment probabilities are those passed in prob (equal by default); "
            "complete / stratified / cluster designs fix arm counts at "
            "floor(N x prob), allocating the remainder at random",
        ],
        "pre_conditions": [
            "One row per unit; strata= / cluster= columns when blocking or "
            "cluster-randomising",
        ],
        "failure_modes": [
            _fm(
                "prob does not have n_arms non-negative entries summing to 1",
                _MI,
                "Pass one probability per arm",
            ),
            _fm(
                "method='stratified' without strata= or method='cluster' without "
                "cluster=",
                _MI,
                "Pass the blocking / cluster column",
            ),
        ],
        "alternatives": ["sp.optimal_design", "sp.balance_table"],
    },
    "optimal_design": {
        "assumptions": [
            "Normal-approximation sample-size formula with a known outcome SD; "
            "cluster designs inflate the variance by the design effect "
            "1 + (cluster_size - 1) x icc",
            "Baseline covariates reduce variance by the factor 1 - r2",
        ],
        "pre_conditions": [
            "Either mde= (to solve for n) or n= / n_clusters= (to solve for the "
            "MDE)",
        ],
        "failure_modes": [
            _fm(
                "Neither mde nor n (or n_clusters) given, or prop_treat outside (0, 1)",
                _MI,
                "Pass exactly one of the target MDE or the available sample",
            ),
            _fm(
                "Cost-optimal cluster size requested with icc not in (0, 1)",
                _MI,
                "Give a strictly positive icc below 1",
            ),
        ],
        "alternatives": ["sp.power", "sp.mde"],
    },
    "power": {
        "assumptions": [
            "Power is computed from the design's closed-form (normal-approximation) "
            "variance with standardised effect sizes",
        ],
        "pre_conditions": [
            "design in {'rct', 'did', 'rd', 'iv', 'cluster_rct', 'ols'}",
            "When solving for n: effect_size and power_target",
        ],
        "failure_modes": [
            _fm(
                "Unknown design",
                "ValueError",
                "Use one of the supported designs",
            ),
            _fm(
                "n=None without power_target, or solving for n without effect_size",
                "ValueError",
                "Pass power_target and effect_size to solve for n; use sp.mde to "
                "solve for the effect size",
                "sp.mde",
            ),
        ],
        "alternatives": ["sp.mde", "sp.optimal_design"],
    },
    # ------------------------------------------------------------------ #
    #  Propensity-score balance / network AIPW / targeting / decomposition
    # ------------------------------------------------------------------ #
    "ps_balance": {
        "assumptions": [
            "Balance after weighting is only evidence on the covariates checked; "
            "it says nothing about unmeasured confounders",
        ],
        "pre_conditions": [
            "Binary treatment column and the covariates to check",
        ],
        "alternatives": ["sp.balance_table", "sp.love_plot", "sp.ipw"],
        "not_recommended_when": [
            "the estimate being checked used matching weights — pass them via "
            "weights=; the default inverse-PS weights describe a different "
            "estimator's balance",
        ],
    },
    "gnn_causal": {
        "assumptions": [
            "Interference runs only through the supplied adjacency matrix, and the "
            "GCN-propagated neighbour covariates summarise it",
            "No unmeasured confounding given own and propagated covariates, with "
            "propensities bounded away from 0 and 1 (propensity_bounds)",
        ],
        "pre_conditions": [
            "An n x n adjacency aligned with the rows of data",
            "Binary treatment and finite outcome / covariates",
        ],
        "failure_modes": [
            _fm(
                "Missing columns, non-finite values or invalid hyperparameters",
                _MI,
                "Fix the inputs named in the message",
            ),
            _fm(
                "Fewer than 3 complete rows",
                _DI,
                "Provide more complete observations",
            ),
        ],
        "alternatives": ["sp.aipw", "sp.spillover"],
        "not_recommended_when": [
            "units do not interact — plain sp.aipw is the same estimand without the "
            "network featurisation",
        ],
    },
    "policy_targeting": {
        "assumptions": [
            "Expected gains are sums of predicted effects, so they are only as good "
            "as the CATE estimates they rank",
        ],
        "pre_conditions": [
            "Finite per-unit effect estimates (array, CATE result or fitted forest) "
            "and either budget= or frac=",
        ],
        "failure_modes": [
            _fm(
                "Empty or non-finite CATE array",
                _MI,
                "Pass finite per-unit effect estimates",
            ),
            _fm(
                "Both budget and frac given, or frac outside (0, 1]",
                _MI,
                "Pass exactly one budget constraint",
            ),
        ],
        "alternatives": ["sp.policy_value", "sp.policy_tree"],
        "not_recommended_when": [
            "the gain must be reported as evidence — validate the rule with "
            "doubly-robust scores (sp.policy_value) instead of predicted effects",
        ],
    },
    "disparity_decompose": {
        "assumptions": [
            "No unmeasured confounding of the mediator-outcome relationship given "
            "the covariates, so setting the mediator to a reference level is "
            "identified",
            "The group indicator is binary (1 = disadvantaged) and is not itself "
            "treated as manipulable",
        ],
        "pre_conditions": [
            "Binary group column, a mediator and an outcome",
        ],
        "alternatives": ["sp.decompose", "sp.mediate"],
        "not_recommended_when": [
            "the question is the effect of group membership itself — this "
            "decomposition only apportions the observed disparity through the "
            "mediator",
        ],
    },
}


__all__ = ["ESTIMATOR_CARDS"]
