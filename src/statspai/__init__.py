"""
StatsPAI: The AI-powered Statistics & Econometrics Toolkit for Python

Unified API for causal inference and econometrics:

>>> import statspai as sp
>>>
>>> # OLS regression
>>> result = sp.regress("y ~ x1 + x2", data=df)
>>>
>>> # Difference-in-Differences
>>> result = sp.did(df, y='wage', treat='treated', time='post')
>>>
>>> # Staggered DID (Callaway & Sant'Anna)
>>> result = sp.did(df, y='wage', treat='first_treat',
...                time='year', id='worker_id')
>>>
>>> # Causal Forest
>>> cf = sp.causal_forest("y ~ treatment | x1 + x2", data=df)
>>>
>>> # Publication-quality export
>>> sp.outreg2(result, filename="results.xlsx")
"""

__version__ = "0.9.5"
__author__ = "Biaoyue Wang"
__email__ = "bryce@copaper.ai"

from .core.results import EconometricResults, CausalResult
from .regression.ols import regress
from .regression.iv import iv, ivreg, IVRegression
from .causal.causal_forest import CausalForest, causal_forest
from .causal.forest_inference import (
    calibration_test, test_calibration, rate, honest_variance,
)
from .did import (
    did, did_2x2, ddd, callaway_santanna, sun_abraham,
    bacon_decomposition, honest_did, breakdown_m, event_study,
    did_analysis, DIDAnalysis, did_multiplegt, did_imputation, stacked_did, cic,
    wooldridge_did, etwfe, etwfe_emfx, drdid, twfe_decomposition,
    did_summary, did_summary_to_markdown, did_summary_to_latex, did_report,
    pretrends_test, pretrends_power, sensitivity_rr, SensitivityResult, pretrends_summary,
    parallel_trends_plot, bacon_plot, group_time_plot, did_plot,
    enhanced_event_study_plot, treatment_rollout_plot,
    sensitivity_plot, cohort_event_study_plot, did_summary_plot,
    aggte, cs_report, CSReport, ggdid,
    bjs_pretrend_joint,
)
from .rd import (
    rdrobust, rdplot, rdplotdensity, rdbwselect,
    rdbwsensitivity, rdbalance, rdplacebo, rdsummary,
    rkd, rd_honest, rdit, rdpower, rdsampsi,
    rdrandinf, rdwinselect, rdsensitivity, rdrbounds,
    rdhte, rdbwhte, rdhte_lincom,
    rd_forest, rd_boost, rd_lasso, rd_cate_summary,
    rd_extrapolate, rd_multi_extrapolate, rd_external_validity,
)
from .synth import (
    synth, SyntheticControl, synthplot, sdid, augsynth,
    demeaned_synth, robust_synth, gsynth, staggered_synth, conformal_synth, mc_synth,
    multi_outcome_synth,
    scpi, scest, scdata,
    discos, qqsynth, discos_test, discos_plot, stochastic_dominance,
    synth_loo, synth_time_placebo, synth_donor_sensitivity,
    synth_rmspe_filter, synth_sensitivity, synth_sensitivity_plot,
    synth_power, synth_mde, synth_power_plot,
    synth_compare, synth_recommend, SynthComparison,
    synth_report, synth_report_to_file,
    german_reunification, basque_terrorism, california_tobacco,
    synthdid_estimate, sc_estimate, did_estimate,
    synthdid_placebo, synthdid_plot, synthdid_units_plot, synthdid_rmse_plot,
    california_prop99,
)
from .matching import (
    match, MatchEstimator, ebalance, balanceplot, psplot,
    propensity_score, overlap_plot, trimming, love_plot,
    ps_balance, PSBalanceResult,
    optimal_match, cardinality_match,
    OptimalMatchResult, CardinalityMatchResult,
    overlap_weights, cbps,
)
from .dml import dml, DoubleML
from .deepiv import deepiv, DeepIV
from .panel import (
    panel, panel_compare, balance_panel, PanelResults, PanelCompareResults, PanelRegression,
    Absorber, demean, absorb_ols, hdfe_ols, FEOLSResult,
)
from .causal_impact import causal_impact, CausalImpactEstimator, impactplot
from .mediation import mediate, MediationAnalysis, mediate_sensitivity, mediate_interventional
from .bartik import bartik, BartikIV, ssaggregate, shift_share_se
from .output.outreg2 import OutReg2, outreg2
from .output.modelsummary import modelsummary, coefplot
from .output.sumstats import sumstats, balance_table
from .output.tab import tab
from .output.estimates import eststo, estclear, esttab
from .output.regression_table import regtable, RegtableResult, mean_comparison, MeanComparisonResult
from .output.paper_tables import paper_tables, PaperTables, TEMPLATES as PAPER_TABLE_TEMPLATES
from .postestimation import margins, marginsplot, margins_at, margins_at_plot, contrast, pwcompare, test, lincom
from .diagnostics import oster_bounds, mccrary_test, diagnose, het_test, reset_test, vif, sensemakr, rddensity, hausman_test, anderson_rubin_test, effective_f_test, tF_critical_value, evalue, evalue_from_result, diagnose_result, estat, kitagawa_test, KitagawaResult
from .inference import (
    wild_cluster_bootstrap, aipw, ri_test, ipw, bootstrap, BootstrapResult,
    twoway_cluster, conley, pate, PATEEstimator, fisher_exact, FisherResult,
    jackknife_se, cr2_se, wild_cluster_boot,
    subcluster_wild_bootstrap, wild_cluster_ci_inv,
    multiway_cluster_vcov, cluster_robust_se, cr3_jackknife_vcov,
    g_computation, front_door,
)
from .msm import msm, MarginalStructuralModel, stabilized_weights
from .spatial import (
    sar, sem, sdm, slx, sac, SpatialModel,
    sar_gmm, sem_gmm, sarar_gmm,
    gwr, mgwr, gwr_bandwidth,
    spatial_panel,
    W, queen_weights, rook_weights, knn_weights,
    distance_band, kernel_weights, block_weights,
    moran, moran_local, geary, getis_ord_g, getis_ord_local, join_counts,
    moran_plot, lisa_cluster_map,
    lm_tests, moran_residuals, impacts,
)
from . import spatial
# NOTE: `from . import iv` would be shadowed by the `iv` function imported
# on line 31 from `.regression.iv`, so we load the subpackage explicitly.
import importlib as _importlib
iv = _importlib.import_module(".iv", __name__)
del _importlib
from .plots import binscatter, set_theme, list_themes, use_chinese, interactive, get_code
from .utils import (
    label_var, label_vars, get_label, get_labels, describe, pwcorr, winsor, read_data,
    rowmean, rowtotal, rowmax, rowmin, rowsd, rowcount, rank, outlier_indicator,
    dgp_did, dgp_rd, dgp_rd_kink, dgp_rd_multi, dgp_rd_hte, dgp_rd_2d, dgp_rdit,
    dgp_iv, dgp_rct, dgp_panel, dgp_observational,
    dgp_cluster_rct, dgp_bunching, dgp_synth, dgp_bartik,
)
from .gmm import xtabond
from .metalearners import metalearner, SLearner, TLearner, XLearner, RLearner, DRLearner
from .metalearners import cate_summary, cate_by_group, cate_plot, cate_group_plot, predict_cate, compare_metalearners, gate_test, blp_test
from .metalearners import auto_cate, AutoCATEResult
from .metalearners import auto_cate_tuned
from .bayes import bayes_did, bayes_rd, BayesianCausalResult
from .regression.heckman import heckman
from .regression.quantile import qreg, sqreg
from .regression.tobit import tobit
from .regression.logit_probit import logit, probit, cloglog
from .regression.glm import glm, GLMRegression, GLMEstimator
from .regression.count import poisson, nbreg, ppmlhdfe
from .neural_causal import tarnet, cfrnet, dragonnet, TARNet, CFRNet, DragonNet
from .causal_discovery import notears, NOTEARS, pc_algorithm, PCAlgorithm, lingam, LiNGAMResult, ges, GESResult
from .tmle import tmle, TMLE, super_learner, SuperLearner
from .policy_learning import policy_tree, PolicyTree, policy_value
from .conformal_causal import conformal_cate, ConformalCATE
from .bcf import bcf, BayesianCausalForest
from .bunching import bunching, BunchingEstimator, notch, NotchResult
from .matrix_completion import mc_panel, MCPanel
from .dose_response import dose_response, DoseResponse
from .bounds import lee_bounds, manski_bounds, BoundsResult, horowitz_manski, iv_bounds, oster_delta, selection_bounds, breakdown_frontier
from .interference import spillover, SpilloverEstimator
from .dtr import g_estimation, GEstimation
from .multi_treatment import multi_treatment, MultiTreatment
from .robustness import spec_curve, SpecCurveResult, robustness_report, RobustnessResult, subgroup_analysis, SubgroupResult
from .survey import svydesign, SurveyDesign, svymean, svytotal, svyglm, rake, linear_calibration
from .dag import dag, DAG, dag_example, dag_examples, dag_example_positions, dag_simulate

# === Canonical datasets (consolidated facade) ===
from . import datasets

# === End-to-end workflow orchestrator ===
from .workflow import causal, CausalWorkflow

from .power import power, PowerResult, power_rct, power_did, power_rd, power_iv, power_cluster_rct, power_ols, mde
from .decomposition import (
    oaxaca, gelbach, OaxacaResult, GelbachResult, rifreg, rif_decomposition,
    # Tier C additions
    dfl_decompose, ffl_decompose, machado_mata, melly_decompose, cfm_decompose,
    fairlie, bauer_sinning, yun_nonlinear,
    inequality_index, subgroup_decompose, source_decompose, shapley_inequality,
    kitagawa_decompose, das_gupta,
    gap_closing, mediation_decompose, disparity_decompose,
    decompose, available_methods,
    cps_wage, chilean_households, mincer_wage_panel, disparity_panel,
)
from .selection import stepwise, lasso_select, SelectionResult
from .qte import qdid, qte, QTEResult
from .mht import romano_wolf, RomanoWolfResult, adjust_pvalues, bonferroni, holm, benjamini_hochberg
from .registry import list_functions, describe_function, function_schema, search_functions, all_schemas

# === NEW MODULES (v0.6) ===
# GLM & Discrete Choice
from .regression.glm import glm, GLMEstimator
from .regression.logit_probit import logit, probit, cloglog
from .regression.multinomial import mlogit, ologit, oprobit, clogit
from .regression.mixed_logit import mixlogit
from .regression.iv_quantile import ivqreg
# Count Data
from .regression.count import poisson, nbreg, ppmlhdfe
from .regression.zeroinflated import zip_model, zinb, hurdle
# Advanced IV
from .regression.advanced_iv import liml, jive, lasso_iv
# High-dimensional fixed effects (pyfixest backend)
# These are thin wrappers; actual import of pyfixest is deferred to call time
# via fixest.wrapper._check_pyfixest, so top-level import never fails.
from .fixest import feols, fepois, feglm, etable
# Survival / Duration
from .survival import cox, kaplan_meier, survreg, CoxResult, KMResult, logrank_test, cox_frailty, aft
# Nonparametric
from .nonparametric import lpoly, LPolyResult, kdensity, KDensityResult
# Time Series (for causal inference)
from .timeseries import (
    var, VARResult, granger_causality, irf,
    structural_break, StructuralBreakResult, cusum_test,
    local_projections, LocalProjectionsResult,
    garch, GARCHResult,
    arima, ARIMAResult,
    bvar, BVARResult,
)
# Experimental Design
from .experimental import randomize, RandomizationResult, balance_check, BalanceResult, attrition_test, attrition_bounds, AttritionResult, optimal_design, OptimalDesignResult
# Missing Data / Imputation
from .imputation import mice, MICEResult, mi_estimate
# Mendelian Randomization
from .mendelian import mendelian_randomization, MRResult, mr_egger, mr_ivw, mr_median
# Multi-cutoff / Geographic RD
from .rd import rdmc, rdms, RDMultiResult
# 2D Boundary RD (Cattaneo, Titiunik, Yu 2025)
from .rd import rd2d, rd2d_bw, rd2d_plot
# Continuous Treatment DID
from .did import continuous_did

# === NEW v0.6 Round 2 ===
# Interactive Fixed Effects
from .panel.interactive_fe import interactive_fe
# Panel Unit Root Tests
from .panel.unit_root import panel_unitroot, PanelUnitRootResult
# Cointegration
from .timeseries import engle_granger, johansen, CointegrationResult
# Fractional Response & Beta Regression
from .regression.fracreg import fracreg, betareg
# Sample Selection Models
from .regression.selection import biprobit, etregress
# Distributional Treatment Effects
from .qte import distributional_te, DTEResult
# Structural Estimation (BLP)
from .structural import blp, BLPResult

# === Smart Workflow Engine (unique to StatsPAI) ===
from .smart import (
    recommend, RecommendationResult,
    compare_estimators, ComparisonResult,
    assumption_audit, AssumptionResult,
    sensitivity_dashboard, SensitivityDashboard,
    pub_ready, PubReadyResult,
    replicate, list_replications,
    check_identification, IdentificationReport, DiagnosticFinding,
    IdentificationError,
)

# verify / verify_recommendation / verify_benchmark are loaded lazily via
# __getattr__ at the bottom of this file so that `import statspai` doesn't
# drag in the resampling-stability machinery unless the caller actually
# asks for it. Preserves the "zero overhead when verify=False" guarantee
# in recommend().

# === NEW v0.6 Round 3 ===
# Truncated Regression
from .regression.truncreg import truncreg
# SUR & 3SLS
from .regression.sur import sureg, SURResult, three_sls
# Panel Binary (Logit/Probit FE/RE)
from .panel.panel_binary import panel_logit, panel_probit
# Panel FGLS
from .panel.panel_fgls import panel_fgls
# Interactive Fixed Effects
# (already imported in round 2)
# Mixed Effects / Multilevel
from .multilevel import (
    mixed,
    MixedResult,
    meglm,
    melogit,
    mepoisson,
    menbreg,
    megamma,
    meologit,
    MEGLMResult,
    icc,
    lrtest,
)
# Stochastic Frontier
from .frontier import (
    frontier, xtfrontier, FrontierResult,
    metafrontier, MetafrontierResult,
    malmquist, MalmquistResult, translog_design,
    zisf, lcsf,
    te_summary, te_rank,
)
# General GMM
from .gmm import gmm

__all__ = [
    # Core
    "EconometricResults",
    "CausalResult",
    # Regression
    "regress",
    "iv",
    "ivreg",
    "IVRegression",
    # DID
    "did",
    "did_2x2",
    "ddd",
    "callaway_santanna",
    "sun_abraham",
    "bacon_decomposition",
    "honest_did",
    "breakdown_m",
    "event_study",
    "did_analysis",
    "DIDAnalysis",
    "did_multiplegt",
    "did_imputation",
    "stacked_did",
    "cic",
    "pretrends_test",
    "pretrends_power",
    "sensitivity_rr",
    "SensitivityResult",
    "pretrends_summary",
    "parallel_trends_plot",
    "bacon_plot",
    "group_time_plot",
    "did_plot",
    "enhanced_event_study_plot",
    "treatment_rollout_plot",
    "sensitivity_plot",
    "cohort_event_study_plot",
    "did_summary_plot",
    # Wooldridge / DR-DID / TWFE Decomposition
    "wooldridge_did",
    "etwfe",
    "etwfe_emfx",
    "did_summary",
    "did_summary_to_markdown",
    "did_summary_to_latex",
    "did_report",
    "drdid",
    "twfe_decomposition",
    # RD
    "rdrobust",
    "rdplot",
    "rdplotdensity",
    "rdbwselect",
    "rdbwsensitivity",
    "rdbalance",
    "rdplacebo",
    "rdsummary",
    "rkd",
    "rd_honest",
    "rdit",
    "rdrandinf",
    "rdwinselect",
    "rdsensitivity",
    "rdrbounds",
    "rdhte",
    "rdbwhte",
    "rdhte_lincom",
    "rd_forest",
    "rd_boost",
    "rd_lasso",
    "rd_cate_summary",
    "rd_extrapolate",
    "rd_multi_extrapolate",
    "rd_external_validity",
    # Synthetic Control
    "synth",
    "SyntheticControl",
    "demeaned_synth",
    "robust_synth",
    "gsynth",
    "staggered_synth",
    "conformal_synth",
    "sdid",
    "synthdid_estimate",
    "sc_estimate",
    "did_estimate",
    "synthdid_placebo",
    "synthdid_plot",
    "synthdid_units_plot",
    "synthdid_rmse_plot",
    "california_prop99",
    "augsynth",
    "mc_synth",
    # Matching
    "match",
    "MatchEstimator",
    "ebalance",
    "balanceplot",
    "psplot",
    # PS Diagnostics
    "propensity_score",
    "overlap_plot",
    "trimming",
    "love_plot",
    "ps_balance",
    "PSBalanceResult",
    # Double ML
    "dml",
    "DoubleML",
    # DeepIV
    "deepiv",
    "DeepIV",
    # Panel
    "panel",
    "panel_compare",
    "balance_panel",
    "PanelResults",
    "PanelCompareResults",
    "PanelRegression",
    # Causal Impact
    "causal_impact",
    "CausalImpactEstimator",
    # Causal Forest + GRF inference
    "CausalForest",
    "causal_forest",
    "calibration_test",
    "test_calibration",  # GRF-compatible alias of calibration_test
    "rate",
    "honest_variance",
    # HDFE primitives
    "Absorber",
    "demean",
    "absorb_ols",
    "hdfe_ols",
    "FEOLSResult",
    # Matching extensions
    "overlap_weights",
    "cbps",
    # Inference primitives
    "subcluster_wild_bootstrap",
    "wild_cluster_ci_inv",
    "multiway_cluster_vcov",
    "cluster_robust_se",
    "cr3_jackknife_vcov",
    # Output
    "OutReg2",
    "outreg2",
    "modelsummary",
    "coefplot",
    "sumstats",
    "balance_table",
    "tab",
    "eststo",
    "estclear",
    "esttab",
    "regtable",
    "RegtableResult",
    "mean_comparison",
    "MeanComparisonResult",
    # Plots
    "binscatter",
    "set_theme",
    "list_themes",
    "use_chinese",
    "interactive",
    "get_code",
    # Utils
    "label_var",
    "label_vars",
    "get_label",
    "get_labels",
    "describe",
    "pwcorr",
    "winsor",
    "rowmean",
    "rowtotal",
    "rowmax",
    "rowmin",
    "rowsd",
    "rowcount",
    "rank",
    "outlier_indicator",
    "read_data",
    # Dynamic Panel GMM
    "xtabond",
    "heckman",
    "qreg",
    "sqreg",
    "tobit",
    "logit",
    "probit",
    "cloglog",
    "glm",
    "GLMRegression",
    "GLMEstimator",
    "poisson",
    "nbreg",
    "ppmlhdfe",
    # Post-estimation
    "margins",
    "marginsplot",
    "margins_at",
    "margins_at_plot",
    "contrast",
    "pwcompare",
    "test",
    "lincom",
    # Mediation
    "mediate",
    "MediationAnalysis",
    "mediate_interventional",
    # Bartik IV
    "bartik",
    "BartikIV",
    "ssaggregate",
    "shift_share_se",
    # Diagnostics
    "oster_bounds",
    "mccrary_test",
    "diagnose",
    "het_test",
    "reset_test",
    "vif",
    "sensemakr",
    "rddensity",
    "hausman_test",
    "anderson_rubin_test",
    "effective_f_test",
    "tF_critical_value",
    "evalue",
    "evalue_from_result",
    "diagnose_result",
    "estat",
    "kitagawa_test",
    "KitagawaResult",
    # Inference
    "wild_cluster_bootstrap",
    "aipw",
    "ri_test",
    "ipw",
    "bootstrap",
    "BootstrapResult",
    "twoway_cluster",
    "conley",
    "pate",
    "PATEEstimator",
    "fisher_exact",
    "FisherResult",
    "jackknife_se",
    "cr2_se",
    "wild_cluster_boot",
    # G-methods family (g-computation / front-door)
    "g_computation",
    "front_door",
    # Marginal Structural Models (time-varying treatment)
    "msm",
    "MarginalStructuralModel",
    "stabilized_weights",
    # Spatial Econometrics
    "sar",
    "sem",
    "sdm",
    "SpatialModel",
    # Meta-Learners (HTE)
    "metalearner",
    "SLearner",
    "TLearner",
    "XLearner",
    "RLearner",
    "DRLearner",
    # CATE Diagnostics
    "cate_summary",
    "cate_by_group",
    "cate_plot",
    "cate_group_plot",
    "predict_cate",
    "compare_metalearners",
    "gate_test",
    "blp_test",
    "auto_cate",
    "AutoCATEResult",
    "auto_cate_tuned",
    # Bayesian Causal Models
    "bayes_did",
    "bayes_rd",
    "BayesianCausalResult",
    # Neural Causal Models
    "tarnet",
    "cfrnet",
    "dragonnet",
    "TARNet",
    "CFRNet",
    "DragonNet",
    # Causal Discovery
    "notears",
    "NOTEARS",
    "pc_algorithm",
    "PCAlgorithm",
    # TMLE
    "tmle",
    "TMLE",
    "super_learner",
    "SuperLearner",
    # Policy Learning
    "policy_tree",
    "PolicyTree",
    "policy_value",
    # Conformal Causal Inference
    "conformal_cate",
    "ConformalCATE",
    # Bayesian Causal Forest
    "bcf",
    "BayesianCausalForest",
    # Bunching
    "bunching",
    "BunchingEstimator",
    "notch",
    "NotchResult",
    # Matrix Completion
    "mc_panel",
    "MCPanel",
    # Dose-Response
    "dose_response",
    "DoseResponse",
    # Bounds
    "lee_bounds",
    "manski_bounds",
    "BoundsResult",
    "horowitz_manski",
    "iv_bounds",
    "oster_delta",
    "selection_bounds",
    "breakdown_frontier",
    # Interference
    "spillover",
    "SpilloverEstimator",
    # Dynamic Treatment Regimes
    "g_estimation",
    "GEstimation",
    # Multi-valued Treatment
    "multi_treatment",
    "MultiTreatment",
    # Robustness
    "spec_curve",
    "SpecCurveResult",
    "robustness_report",
    "RobustnessResult",
    "subgroup_analysis",
    "SubgroupResult",
    # Survey Design
    "svydesign",
    "SurveyDesign",
    "svymean",
    "svytotal",
    "svyglm",
    # DAG
    "dag",
    "DAG",
    "dag_example",
    "dag_examples",
    "dag_example_positions",
    "dag_simulate",
    # Power Analysis
    "power",
    "PowerResult",
    "power_rct",
    "power_did",
    "power_rd",
    "power_iv",
    "power_cluster_rct",
    "power_ols",
    "mde",
    # Decomposition
    "oaxaca",
    "gelbach",
    "OaxacaResult",
    "GelbachResult",
    # Variable Selection
    "stepwise",
    "lasso_select",
    "SelectionResult",
    # Quantile Treatment Effects
    "qdid",
    "qte",
    "QTEResult",
    # Multiple Hypothesis Testing
    "romano_wolf",
    "RomanoWolfResult",
    "adjust_pvalues",
    "bonferroni",
    "holm",
    "benjamini_hochberg",
    # AI / Agent Registry
    "list_functions",
    "describe_function",
    "function_schema",
    "search_functions",
    "all_schemas",
    # Data Generating Processes
    "dgp_did",
    "dgp_rd",
    "dgp_iv",
    "dgp_rct",
    "dgp_panel",
    "dgp_observational",
    "dgp_cluster_rct",
    "dgp_bunching",
    "dgp_synth",
    "dgp_bartik",
    # === NEW v0.6 ===
    # GLM & Discrete Choice
    "glm", "GLMEstimator",
    "logit", "probit", "cloglog",
    "mlogit", "ologit", "oprobit", "clogit",
    "mixlogit", "ivqreg",
    # Count Data
    "poisson", "nbreg", "ppmlhdfe",
    "zip_model", "zinb", "hurdle",
    # Advanced IV
    "liml", "jive", "lasso_iv",
    # High-dimensional FE (pyfixest backend, optional)
    "feols", "fepois", "feglm", "etable",
    # Survival
    "cox", "kaplan_meier", "survreg", "CoxResult", "KMResult", "logrank_test",
    # Nonparametric
    "lpoly", "LPolyResult", "kdensity", "KDensityResult",
    # Time Series
    "var", "VARResult", "granger_causality", "irf",
    "structural_break", "StructuralBreakResult", "cusum_test",
    # Experimental Design
    "randomize", "RandomizationResult", "balance_check", "BalanceResult",
    "attrition_test", "attrition_bounds", "AttritionResult",
    "optimal_design", "OptimalDesignResult",
    # Missing Data
    "mice", "MICEResult", "mi_estimate",
    # Mendelian Randomization
    "mendelian_randomization", "MRResult", "mr_egger", "mr_ivw", "mr_median",
    # Multi-Cutoff / Geographic RD
    "rdmc", "rdms", "RDMultiResult",
    # Continuous DID
    "continuous_did",
    # === v0.6 Round 2 ===
    "interactive_fe",
    "panel_unitroot", "PanelUnitRootResult",
    "engle_granger", "johansen", "CointegrationResult",
    "fracreg", "betareg",
    "biprobit", "etregress",
    "distributional_te", "DTEResult",
    # Structural Estimation
    "blp", "BLPResult",
    # === Smart Workflow Engine (unique to StatsPAI) ===
    "recommend", "RecommendationResult",
    "check_identification", "IdentificationReport", "DiagnosticFinding",
    "IdentificationError",
    "compare_estimators", "ComparisonResult",
    "assumption_audit", "AssumptionResult",
    "sensitivity_dashboard", "SensitivityDashboard",
    "pub_ready", "PubReadyResult",
    "replicate", "list_replications",
    "verify", "verify_recommendation", "verify_benchmark",
    # === v0.6 Round 3 ===
    "truncreg",
    "sureg", "SURResult", "three_sls",
    "panel_logit", "panel_probit",
    "panel_fgls",
    "mixed", "MixedResult",
    "meglm", "melogit", "mepoisson", "menbreg", "megamma", "meologit", "MEGLMResult",
    "icc", "lrtest",
    "frontier", "xtfrontier", "FrontierResult",
    "metafrontier", "MetafrontierResult",
    "malmquist", "MalmquistResult", "translog_design",
    "zisf", "lcsf",
    "te_summary", "te_rank",
    "gmm",
    # ---- v0.9.3 __all__ completeness pass ----
    # Items below were imported at the top of the file but previously
    # missing from __all__, breaking `from statspai import *` and some
    # IDE autocompleters. Grouped by subsystem for readability.
    # Causal impact / mediation
    "impactplot", "mediate_sensitivity",
    # Synth suite
    "synthplot",
    "multi_outcome_synth", "scpi", "scest", "scdata",
    "discos", "discos_test", "discos_plot", "qqsynth",
    "stochastic_dominance",
    "synth_loo", "synth_time_placebo", "synth_donor_sensitivity",
    "synth_rmspe_filter", "synth_sensitivity", "synth_sensitivity_plot",
    "synth_power", "synth_mde", "synth_power_plot",
    "synth_compare", "synth_recommend", "SynthComparison",
    "synth_report", "synth_report_to_file",
    "german_reunification", "basque_terrorism", "california_tobacco",
    # Spatial
    "W",
    "moran", "moran_local", "moran_plot", "moran_residuals",
    "geary", "getis_ord_g", "getis_ord_local", "join_counts",
    "lisa_cluster_map", "lm_tests", "impacts",
    "queen_weights", "rook_weights", "knn_weights",
    "distance_band", "kernel_weights", "block_weights",
    "gwr", "mgwr", "gwr_bandwidth",
    "sac", "slx", "sar_gmm", "sarar_gmm", "sem_gmm",
    "spatial_panel",
    # RD
    "rd2d", "rd2d_bw", "rd2d_plot", "rdpower", "rdsampsi",
    "dgp_rd_2d", "dgp_rd_hte", "dgp_rd_kink", "dgp_rd_multi", "dgp_rdit",
    # Decomposition
    "decompose", "dfl_decompose",
    "machado_mata", "melly_decompose",
    "rifreg", "rif_decomposition", "ffl_decompose",
    "fairlie", "yun_nonlinear", "cfm_decompose",
    "bauer_sinning", "das_gupta",
    "gap_closing", "kitagawa_decompose",
    "source_decompose", "subgroup_decompose",
    "disparity_decompose", "disparity_panel",
    "mediation_decompose",
    "inequality_index", "shapley_inequality",
    # Panel / DID extras
    "aggte", "ggdid", "bjs_pretrend_joint", "cs_report", "CSReport",
    "local_projections", "LocalProjectionsResult",
    # Matching / survey / survival
    "cardinality_match", "CardinalityMatchResult",
    "optimal_match", "OptimalMatchResult",
    "linear_calibration", "rake",
    "aft", "cox_frailty",
    # Causal discovery
    "lingam", "LiNGAMResult", "ges", "GESResult",
    # Time series
    "arima", "ARIMAResult",
    "bvar", "BVARResult",
    "garch", "GARCHResult",
    # Datasets
    "cps_wage", "mincer_wage_panel", "chilean_households",
    # Recommendations metadata
    "available_methods",
]


def __getattr__(name):
    """Lazy-load the verify / verify_benchmark entry points.

    Pulling verify.py at import time contradicts the "verify=False ⇒
    zero overhead" promise in recommend(); defer it to first access.
    Cache BOTH aliases (``verify`` and ``verify_recommendation``) at
    the same time so a later access of the un-touched alias doesn't
    re-enter __getattr__.
    """
    if name in {"verify", "verify_recommendation"}:
        from .smart.verify import verify_recommendation as _vr
        globals()["verify"] = _vr
        globals()["verify_recommendation"] = _vr
        return _vr
    if name == "verify_benchmark":
        from .smart.benchmark import verify_benchmark as _vb
        globals()["verify_benchmark"] = _vb
        return _vb
    raise AttributeError(f"module 'statspai' has no attribute {name!r}")
