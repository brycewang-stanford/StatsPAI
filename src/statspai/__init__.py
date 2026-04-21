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

__version__ = "1.4.0"
__author__ = "Biaoyue Wang"
__email__ = "brycew6m@stanford.edu"

from .core.results import EconometricResults, CausalResult
from .regression.ols import regress
from .regression.iv import iv, ivreg, IVRegression
from .causal.causal_forest import CausalForest, causal_forest
from .causal.forest_inference import (
    calibration_test, test_calibration, rate, honest_variance,
)
from .causal.multi_arm_forest import multi_arm_forest, MultiArmForestResult
from .causal.iv_forest import iv_forest, IVForestResult
from .did import (
    did, did_2x2, overlap_weighted_did, dl_propensity_score,
    ddd, callaway_santanna, sun_abraham,
    bacon_decomposition, honest_did, breakdown_m, event_study,
    did_analysis, DIDAnalysis, did_multiplegt, did_imputation, stacked_did, cic,
    gardner_did, did_2stage,
    harvest_did, HarvestDIDResult,
    wooldridge_did, etwfe, etwfe_emfx, drdid, twfe_decomposition,
    did_bcf, cohort_anchored_event_study, design_robust_event_study,
    did_misclassified,
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
    rd_interference, RDInterferenceResult,
    rd_multi_score, MultiScoreRDResult,
    rd_distribution, DistRDResult,
    rd_bayes_hte, BayesRDHTEResult,
    rd_distributional_design, DDDResult,
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
from .synth.sequential_sdid import sequential_sdid, SequentialSDIDResult
from .synth.survival import synth_survival, SyntheticSurvivalResult
from .synth.experimental_design import (
    synth_experimental_design,
    SynthExperimentalDesignResult,
)
from .matching import (
    match, MatchEstimator, ebalance, balanceplot, psplot,
    propensity_score, overlap_plot, trimming, love_plot,
    ps_balance, PSBalanceResult,
    optimal_match, cardinality_match,
    OptimalMatchResult, CardinalityMatchResult,
    overlap_weights, cbps,
    genmatch, GenMatchResult,
    sbw, SBWResult,
)
from .dml import (
    dml, DoubleML, DoubleMLPLR, DoubleMLIRM, DoubleMLPLIV, DoubleMLIIVM,
    dml_model_averaging, model_averaging_dml, DMLAveragingResult,
)
from .deepiv import deepiv, DeepIV
from .panel import (
    panel, panel_compare, balance_panel, PanelResults, PanelCompareResults, PanelRegression,
    Absorber, demean, absorb_ols, hdfe_ols, FEOLSResult,
)
from .causal_impact import causal_impact, CausalImpactEstimator, impactplot
from .mediation import mediate, MediationAnalysis, mediate_sensitivity, mediate_interventional, four_way_decomposition, FourWayResult
from .bartik import (
    bartik, BartikIV, ssaggregate, shift_share_se,
    shift_share_political, ShiftSharePoliticalResult,
    shift_share_political_panel, ShiftSharePoliticalPanelResult,
)
from .output.outreg2 import OutReg2, outreg2
from .output.modelsummary import modelsummary, coefplot
from .output.sumstats import sumstats, balance_table
from .output.tab import tab
from .output.estimates import eststo, estclear, esttab
from .output.regression_table import regtable, RegtableResult, mean_comparison, MeanComparisonResult
from .output.paper_tables import paper_tables, PaperTables, TEMPLATES as PAPER_TABLE_TEMPLATES
from .postestimation import margins, marginsplot, margins_at, margins_at_plot, contrast, pwcompare, test, lincom
from .diagnostics import oster_bounds, mccrary_test, diagnose, het_test, reset_test, vif, sensemakr, rddensity, hausman_test, anderson_rubin_test, effective_f_test, tF_critical_value, evalue, evalue_from_result, diagnose_result, estat, kitagawa_test, KitagawaResult, rosenbaum_bounds, rosenbaum_gamma, RosenbaumResult, weakrobust, WeakRobustResult
from .inference import (
    wild_cluster_bootstrap, aipw, ri_test, ipw, bootstrap, BootstrapResult,
    twoway_cluster, conley, pate, PATEEstimator, fisher_exact, FisherResult,
    jackknife_se, cr2_se, wild_cluster_boot,
    subcluster_wild_bootstrap, wild_cluster_ci_inv,
    multiway_cluster_vcov, cluster_robust_se, cr3_jackknife_vcov,
    g_computation, front_door,
)
from .msm import msm, MarginalStructuralModel, stabilized_weights
from .proximal import (
    proximal, ProximalCausalInference,
    negative_control_outcome, negative_control_exposure,
    double_negative_control, NegativeControlResult,
    proximal_regression, ProximalRegResult,
    fortified_pci, bidirectional_pci, pci_mtp,
    select_pci_proxies, ProxyScoreResult,
)
from .principal_strat import (
    principal_strat, PrincipalStratResult, survivor_average_causal_effect,
)
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
    spatial_did, SpatialDiDResult, spatial_iv, SpatialIVResult,
)
from . import spatial
# NOTE: `from . import iv` would be shadowed by the `iv` function imported
# on line 31 from `.regression.iv`, so we load the subpackage explicitly.
import importlib as _importlib
iv = _importlib.import_module(".iv", __name__)
del _importlib
# Expose Kernel IV / Continuous-LATE at top level for agent discoverability.
from .iv.kernel_iv import kernel_iv, KernelIVResult
from .iv.continuous_late import continuous_iv_late, ContinuousLATEResult
from .plots import binscatter, set_theme, list_themes, use_chinese, interactive, get_code
from .utils import (
    label_var, label_vars, get_label, get_labels, describe, pwcorr, winsor, read_data,
    rowmean, rowtotal, rowmax, rowmin, rowsd, rowcount, rank, outlier_indicator,
    scalar_iv_projection,
    dgp_did, dgp_rd, dgp_rd_kink, dgp_rd_multi, dgp_rd_hte, dgp_rd_2d, dgp_rdit,
    dgp_iv, dgp_rct, dgp_panel, dgp_observational,
    dgp_cluster_rct, dgp_bunching, dgp_synth, dgp_bartik,
)
from .gmm import xtabond
from .metalearners import metalearner, SLearner, TLearner, XLearner, RLearner, DRLearner
from .metalearners import cate_summary, cate_by_group, cate_plot, cate_group_plot, predict_cate, compare_metalearners, gate_test, blp_test
from .metalearners import auto_cate, AutoCATEResult
from .metalearners import auto_cate_tuned
from .metalearners import (
    focal_cate, FunctionalCATEResult,
    cluster_cate, ClusterCATEResult,
)
from .bayes import (
    bayes_did, bayes_rd, bayes_iv, bayes_fuzzy_rd, bayes_hte_iv,
    bayes_mte, bayes_dml, BayesianDMLResult,
    BayesianCausalResult, BayesianDIDResult, BayesianHTEIVResult,
    BayesianIVResult, BayesianMTEResult,
    policy_weight_ate, policy_weight_subsidy,
    policy_weight_prte, policy_weight_marginal,
    policy_weight_observed_prte,
)
from .regression.heckman import heckman
from .regression.quantile import qreg, sqreg
from .regression.tobit import tobit
from .regression.logit_probit import logit, probit, cloglog
from .regression.glm import glm, GLMRegression, GLMEstimator
from .regression.count import poisson, nbreg, ppmlhdfe
from .neural_causal import tarnet, cfrnet, dragonnet, TARNet, CFRNet, DragonNet, gnn_causal, GNNCausalResult
from .neural_causal.cevae import cevae, CEVAE, CEVAEResult
from .causal_discovery import notears, NOTEARS, pc_algorithm, PCAlgorithm, lingam, LiNGAMResult, ges, GESResult, fci, FCIResult, icp, nonlinear_icp, ICPResult, pcmci, PCMCIResult, partial_corr_pvalue, lpcmci, LPCMCIResult, dynotears, DYNOTEARSResult
from .tmle import (
    tmle, TMLE, super_learner, SuperLearner,
    ltmle, LTMLEResult, ltmle_survival, LTMLESurvivalResult,
    hal_tmle, HALRegressor, HALClassifier,
)
from .policy_learning import policy_tree, PolicyTree, policy_value, direct_method, ips, snips, doubly_robust, OPEResult
from .conformal_causal import (
    conformal_cate, ConformalCATE,
    weighted_conformal_prediction,
    conformal_counterfactual, ConformalCounterfactualResult,
    conformal_ite_interval, ConformalITEResult,
    conformal_density_ite, ConformalDensityResult,
    conformal_ite_multidp, MultiDPConformalResult,
    conformal_debiased_ml, DebiasedConformalResult,
    conformal_fair_ite, FairConformalResult,
    conformal_continuous, conformal_interference,
    ContinuousConformalResult, InterferenceConformalResult,
)
from .bcf import (
    bcf, BayesianCausalForest, bcf_longitudinal, BCFLongResult,
    bcf_ordinal, BCFOrdinalResult,
    bcf_factor_exposure, BCFFactorExposureResult,
)
from .bunching import bunching, BunchingEstimator, notch, NotchResult
from .bunching import (
    general_bunching, GeneralBunchingResult,
    kink_unified, KinkUnifiedResult,
)
from .matrix_completion import mc_panel, MCPanel
from .dose_response import dose_response, DoseResponse, vcnet, scigan, VCNetResult
from .bounds import lee_bounds, manski_bounds, BoundsResult, horowitz_manski, iv_bounds, oster_delta, selection_bounds, breakdown_frontier, balke_pearl, BalkePearlResult, ml_bounds, MLBoundsResult
from .interference import spillover, SpilloverEstimator, network_exposure, NetworkExposureResult, peer_effects, PeerEffectsResult, network_hte, inward_outward_spillover, NetworkHTEResult, InwardOutwardResult
from .interference import (
    cluster_matched_pair, MatchedPairResult,
    cluster_cross_interference, CrossClusterRCTResult,
    cluster_staggered_rollout, StaggeredClusterRCTResult,
    dnc_gnn_did, DNCGNNDiDResult,
)
from .dtr import g_estimation, GEstimation, q_learning, QLearningResult, a_learning, ALearningResult, snmm, SNMMResult
from .multi_treatment import multi_treatment, MultiTreatment
from .robustness import spec_curve, SpecCurveResult, robustness_report, RobustnessResult, subgroup_analysis, SubgroupResult, copula_sensitivity, survival_sensitivity, calibrate_confounding_strength, FrontierSensitivityResult
from .survey import svydesign, SurveyDesign, svymean, svytotal, svyglm, rake, linear_calibration
from .dag import (
    dag, DAG, dag_example, dag_examples, dag_example_positions, dag_simulate,
    identify, IdentificationResult,
    rule1 as do_rule1, rule2 as do_rule2, rule3 as do_rule3,
    apply_rules as do_calculus_apply, RuleCheck,
    swig, SWIGGraph, SCM,
    llm_dag, LLMDAGResult,
    llm_causal_assess, pairwise_causal_benchmark,
    LLMCausalAssessResult, PairwiseBenchmarkResult,
)

# === Bridging theorems (DiD≡SC, EWM≡CATE, CB≡IPW, Kink≡RDD,
#     DR-Calib, Surrogate≡PCI) ===
from .bridge import bridge, BridgeResult

# === LLM × Causal (DAG / E-value / sensitivity priors) ===
from . import causal_llm
from .causal_llm import (
    llm_dag_propose, LLMDAGProposal,
    llm_unobserved_confounders, UnobservedConfounderProposal,
    llm_sensitivity_priors, SensitivityPriorProposal,
    causal_mas, CausalMASResult,
)

# === Causal RL (Causal-DQN, benchmarks, offline-safe) ===
from . import causal_rl
from .causal_rl import (
    causal_dqn, CausalDQNResult,
    causal_rl_benchmark, BanditBenchmarkResult,
    offline_safe_policy, OfflineSafeResult,
    causal_bandit, counterfactual_policy_optimization, structural_mdp,
    CausalBanditResult, CFPolicyResult, StructuralMDPResult,
)

# === Long-term effects via surrogate indices ===
from . import surrogate
from .surrogate import (
    surrogate_index, long_term_from_short, proximal_surrogate_index,
    SurrogateResult,
)

# === Assimilative Causal Inference (Nature Communications 2026) ===
from . import assimilation
from .assimilation import (
    assimilative_causal, causal_kalman, AssimilationResult,
)

# === Counterfactual fairness / algorithmic-bias diagnostics ===
from . import fairness
from .fairness import (
    counterfactual_fairness, orthogonal_to_bias,
    demographic_parity, equalized_odds, fairness_audit,
    FairnessResult, FairnessAudit,
    evidence_without_injustice, EvidenceWithoutInjusticeResult,
)

# === Transportability (Pearl-Bareinboim + Dahabreh-Stuart) ===
from . import transport
from .transport import (
    weights as transport_weights_fn,
    generalize as transport_generalize,
    TransportWeightResult,
    identify_transport, TransportIdentificationResult,
    synthesise_evidence, heterogeneity_of_effect, rwd_rct_concordance,
    EvidenceSynthesisResult, HeterogeneityResult, ConcordanceResult,
)

# === Off-Policy Evaluation (contextual bandits) ===
from . import ope
from .ope import OPEResult, sharp_ope_unobserved, causal_policy_forest, SharpOPEResult, CausalPolicyForestResult

# === Parametric g-formula (iterative conditional expectation) ===
from . import gformula
from .gformula import ice as gformula_ice_fn, ICEResult
from .gformula import gformula_mc, MCGFormulaResult

# === Target Trial Emulation (JAMA 2022 framework) ===
from . import target_trial
from . import target_trial as tte   # short alias
from .target_trial import (
    protocol as target_trial_protocol,
    emulate as target_trial_emulate,
    to_paper as target_trial_report,
    target_checklist as target_trial_checklist,
    TARGET_ITEMS,
    clone_censor_weight,
    immortal_time_check,
    TargetTrialProtocol,
    TargetTrialResult,
    CloneCensorWeightResult,
)
# === Inverse probability of censoring weights ===
from . import censoring
from .censoring import ipcw, IPCWResult

# === Epidemiology primitives (OR / RR / MH / standardization / BH) ===
from . import epi
from .epi import (
    odds_ratio, relative_risk, risk_difference, attributable_risk,
    incidence_rate_ratio, number_needed_to_treat, prevalence_ratio,
    mantel_haenszel, breslow_day_test,
    direct_standardize, indirect_standardize,
    bradford_hill,
    diagnostic_test, sensitivity_specificity,
    roc_curve, auc, cohen_kappa,
    DiagnosticTestResult, ROCResult, KappaResult,
)

# === Longitudinal causal inference (What If Layer 4) ===
from . import longitudinal
from .longitudinal import (
    analyze as longitudinal_analyze,
    contrast as longitudinal_contrast,
    regime, always_treat, never_treat,
    LongitudinalResult, Regime,
)

# === Causal-question DSL (estimand-first workflow) ===
from . import question
from .question import (
    causal_question, CausalQuestion,
    IdentificationPlan, EstimationResult,
    preregister, load_preregister,
)

# === Unified sensitivity dashboard ===
from .robustness import (
    unified_sensitivity, SensitivityDashboard,
)

# === Canonical datasets (consolidated facade) ===
from . import datasets

# === End-to-end workflow orchestrator ===
from .workflow import causal, CausalWorkflow

# === LLM agent tool-definition surface ===
from . import agent

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
# Unified help entry point (aggregates registry + docstring + category + search)
from .help import help, HelpResult

# === Article-facing aliases (sp.rdd / sp.frontdoor / sp.xlearner / ...) ===
# Thin wrappers around existing implementations; see _article_aliases.py
from ._article_aliases import (
    rdd,
    frontdoor,
    xlearner,
    conformal_ite,
    psm,
    partial_identification,
    anderson_rubin_ci,
    conditional_lr_ci,
    tF_adjustment,
    evalue_rr,
)
# === Auto-race estimators (CS/SA/BJS DiD + 2SLS/LIML/JIVE IV) ===
from ._auto_estimators import (
    auto_did, AutoDIDResult,
    auto_iv, AutoIVResult,
)

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
from .survival import cox, kaplan_meier, survreg, CoxResult, KMResult, logrank_test, cox_frailty, aft, causal_survival_forest, causal_survival, CausalSurvivalForestResult
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
    its, ITSResult,
)
# Experimental Design
from .experimental import randomize, RandomizationResult, balance_check, BalanceResult, attrition_test, attrition_bounds, AttritionResult, optimal_design, OptimalDesignResult
# Missing Data / Imputation
from .imputation import mice, MICEResult, mi_estimate
# Mendelian Randomization
from . import mendelian
from . import mendelian as mr   # short alias
from .mendelian import (
    mendelian_randomization, MRResult,
    mr_egger, mr_ivw, mr_median,
    mr_heterogeneity, mr_pleiotropy_egger, mr_leave_one_out,
    mr_steiger, mr_presso, mr_radial,
    HeterogeneityResult, PleiotropyResult, LeaveOneOutResult,
    SteigerResult, MRPressoResult, RadialResult,
    mr_mode, mr_f_statistic, mr_funnel_plot, mr_scatter_plot,
    ModeBasedResult, FStatisticResult,
    mr_multivariable, mr_mediation, mr_bma,
    MVMRResult, MediationMRResult, MRBMAResult,
)
# Expose recommend_estimator at top level too
from .dag import recommend_estimator as dag_recommend_estimator
# Multi-cutoff / Geographic RD
from .rd import rdmc, rdms, RDMultiResult
from .rd import (
    multi_cutoff_rd, geographic_rd, boundary_rd, multi_score_rd,
)
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
from .qte import (
    dist_iv, kan_dlate, DistIVResult,
    qte_hd_panel, HDPanelQTEResult,
    beyond_average_late, BeyondAverageResult,
)
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
    "gardner_did",
    "did_2stage",
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
    "DoubleMLPLR",
    "DoubleMLIRM",
    "DoubleMLPLIV",
    "DoubleMLIIVM",
    "dml_model_averaging",
    "model_averaging_dml",
    "DMLAveragingResult",
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
    "sbw",
    "SBWResult",
    "genmatch",
    "GenMatchResult",
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
    "scalar_iv_projection",
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
    "weakrobust",
    "WeakRobustResult",
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
    # Proximal Causal Inference (unobserved confounding via proxies)
    "proximal",
    "ProximalCausalInference",
    # Principal Stratification (post-treatment variable strata)
    "principal_strat",
    "PrincipalStratResult",
    "survivor_average_causal_effect",
    # Spatial Econometrics
    "sar",
    "sem",
    "sdm",
    "SpatialModel",
    "spatial_did",
    "SpatialDiDResult",
    "spatial_iv",
    "SpatialIVResult",
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
    "bayes_iv",
    "bayes_fuzzy_rd",
    "bayes_hte_iv",
    "bayes_mte",
    "BayesianCausalResult",
    "BayesianDIDResult",
    "BayesianHTEIVResult",
    "BayesianIVResult",
    "BayesianMTEResult",
    "policy_weight_ate",
    "policy_weight_subsidy",
    "policy_weight_prte",
    "policy_weight_marginal",
    "policy_weight_observed_prte",
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
    "hal_tmle",
    "HALRegressor",
    "HALClassifier",
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
    "help",
    "HelpResult",
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
    "multi_cutoff_rd", "geographic_rd", "boundary_rd", "multi_score_rd",
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
    "fci", "FCIResult",
    "notears", "NOTEARS",
    "pc_algorithm", "PCAlgorithm",
    # Time series
    "arima", "ARIMAResult",
    "bvar", "BVARResult",
    "garch", "GARCHResult",
    # Datasets
    "cps_wage", "mincer_wage_panel", "chilean_households",
    # IV frontier (v1.1)
    "kernel_iv", "KernelIVResult",
    "continuous_iv_late", "ContinuousLATEResult",
    # Recommendations metadata
    "available_methods",
    # === Article-facing aliases (blog API sugar) ===
    "rdd",
    "frontdoor",
    "xlearner",
    "conformal_ite",
    "psm",
    "partial_identification",
    "anderson_rubin_ci",
    "conditional_lr_ci",
    "tF_adjustment",
    # === Auto-race estimators ===
    "auto_did",
    "AutoDIDResult",
    "auto_iv",
    "AutoIVResult",
    # === Namespace-collision fixes + kwarg-alignment wrappers ===
    # These names shadow earlier bindings (submodules / functions with
    # mismatched kwargs) — see the `# Late-bind shadowing` block below.
    "matrix_completion",
    "causal_discovery",
    "mediation",
    "evalue_rr",
    # === v0.9.16 breadth-expansion API (Sprint 1-6) ===
    "ipcw", "IPCWResult",
    "icp", "nonlinear_icp", "ICPResult",
    "identify", "IdentificationResult",
    "do_rule1", "do_rule2", "do_rule3", "do_calculus_apply", "RuleCheck",
    "swig", "SWIGGraph", "SCM",
    "cevae", "CEVAE", "CEVAEResult",
    "TargetTrialProtocol", "TargetTrialResult", "CloneCensorWeightResult",
    "target_trial_protocol", "target_trial_emulate", "target_trial_report",
    "clone_censor_weight", "immortal_time_check", "tte",
    "TransportWeightResult", "TransportIdentificationResult",
    "transport_generalize", "transport_weights_fn", "identify_transport",
    "OPEResult",
    "gformula_ice_fn", "ICEResult",
    "gformula_mc", "MCGFormulaResult",
    # v0.9.17 additions (epi primitives)
    "epi", "odds_ratio", "relative_risk", "risk_difference",
    "attributable_risk", "incidence_rate_ratio",
    "number_needed_to_treat", "prevalence_ratio",
    "mantel_haenszel", "breslow_day_test",
    "direct_standardize", "indirect_standardize", "bradford_hill",
    # v0.9.17 additions (epi clinical diagnostics)
    "diagnostic_test", "sensitivity_specificity",
    "roc_curve", "auc", "cohen_kappa",
    "DiagnosticTestResult", "ROCResult", "KappaResult",
    # v0.9.17 additions (MR full suite)
    "mr", "mendelian",
    "mr_heterogeneity", "mr_pleiotropy_egger", "mr_leave_one_out",
    "mr_steiger", "mr_presso", "mr_radial",
    "HeterogeneityResult", "PleiotropyResult", "LeaveOneOutResult",
    "SteigerResult", "MRPressoResult", "RadialResult",
    # v0.9.17 additions (MR deepening)
    "mr_mode", "mr_f_statistic", "mr_funnel_plot", "mr_scatter_plot",
    "ModeBasedResult", "FStatisticResult",
    # v0.9.17 additions (longitudinal unified)
    "longitudinal", "longitudinal_analyze", "longitudinal_contrast",
    "regime", "always_treat", "never_treat",
    "LongitudinalResult", "Regime",
    # v0.9.17 additions (causal-question DSL + pre-registration)
    "question", "causal_question", "CausalQuestion",
    "IdentificationPlan", "EstimationResult",
    "preregister", "load_preregister",
    # v0.9.17 additions (unified sensitivity)
    "unified_sensitivity", "SensitivityDashboard",
    # v0.9.17 additions (DAG UX)
    "dag_recommend_estimator",
    # v1.0 — bridging theorems
    "bridge", "BridgeResult",
    # v1.0 — DiD frontiers (scaffolded)
    "did_bcf", "cohort_anchored_event_study",
    "design_robust_event_study", "did_misclassified",
    # v1.0 — conformal frontiers
    "conformal_debiased_ml", "DebiasedConformalResult",
    "conformal_density_ite", "ConformalDensityResult",
    "conformal_fair_ite", "FairConformalResult",
    "conformal_ite_multidp", "MultiDPConformalResult",
    # v1.0 — proximal frontiers
    "fortified_pci", "bidirectional_pci", "pci_mtp",
    "select_pci_proxies", "ProxyScoreResult",
    # v1.0 — QTE / RD frontiers
    "beyond_average_late", "BeyondAverageResult",
    "qte_hd_panel", "HDPanelQTEResult",
    "rd_distribution", "DistRDResult",
    "rd_interference", "RDInterferenceResult",
    "rd_multi_score", "MultiScoreRDResult",
    # v1.0 — time-series causal discovery
    "pcmci", "PCMCIResult", "lpcmci", "LPCMCIResult",
    "dynotears", "DYNOTEARSResult", "partial_corr_pvalue",
    # v1.0 — LTMLE survival + BCF longitudinal
    "ltmle_survival", "LTMLESurvivalResult",
    # v1.0 — sequential SDID
    "sequential_sdid", "SequentialSDIDResult",
    "synth_survival", "SyntheticSurvivalResult",
    # v1.0 — ML bounds
    "ml_bounds",
    # v1.0 — TARGET Statement 2025
    "target_trial_checklist",
    # v1.0 — frontier sensitivity
    "copula_sensitivity", "survival_sensitivity",
    "calibrate_confounding_strength", "FrontierSensitivityResult",
    # === v0.10 / v1.0 frontier additions ===
    # Bridging theorems
    "bridge", "BridgeResult",
    # DiD frontier
    "did_bcf", "cohort_anchored_event_study",
    "design_robust_event_study", "did_misclassified",
    # Conformal frontier
    "conformal_density_ite", "ConformalDensityResult",
    "conformal_ite_multidp", "MultiDPConformalResult",
    "conformal_debiased_ml", "DebiasedConformalResult",
    "conformal_fair_ite", "FairConformalResult",
    # Proximal frontier
    "fortified_pci", "bidirectional_pci", "pci_mtp",
    "select_pci_proxies", "ProxyScoreResult",
    # Distributional / panel QTE
    "dist_iv", "kan_dlate", "DistIVResult",
    "qte_hd_panel", "HDPanelQTEResult",
    "beyond_average_late", "BeyondAverageResult",
    # RDD frontier
    "rd_interference", "RDInterferenceResult",
    "rd_multi_score", "MultiScoreRDResult",
    "rd_distribution", "DistRDResult",
    "rd_bayes_hte", "BayesRDHTEResult",
    "rd_distributional_design", "DDDResult",
    # Causal × LLM
    "llm_dag_propose", "LLMDAGProposal",
    "llm_unobserved_confounders", "UnobservedConfounderProposal",
    "llm_sensitivity_priors", "SensitivityPriorProposal",
    # Causal RL
    "causal_dqn", "CausalDQNResult",
    "causal_rl_benchmark", "BanditBenchmarkResult",
    "offline_safe_policy", "OfflineSafeResult",
    # Cluster RCT × interference
    "cluster_matched_pair", "MatchedPairResult",
    "cluster_cross_interference", "CrossClusterRCTResult",
    "cluster_staggered_rollout", "StaggeredClusterRCTResult",
    "dnc_gnn_did", "DNCGNNDiDResult",
    # Meta-learner frontier
    "focal_cate", "FunctionalCATEResult",
    "cluster_cate", "ClusterCATEResult",
    # Bunching frontier
    "general_bunching", "GeneralBunchingResult",
    "kink_unified", "KinkUnifiedResult",
]


# ---------------------------------------------------------------------
# Late-bind shadowing
# ---------------------------------------------------------------------
# `sp.matrix_completion`, `sp.causal_discovery`, `sp.mediation` are
# bound to submodules earlier in this file; `sp.policy_tree` and
# `sp.dml` are bound to functions with different kwargs than the blog
# post advertises.  Importing the article-facing wrappers HERE (after
# all earlier bindings) is what actually makes `sp.mediation(df, ...)`
# callable, and what normalises policy_tree's `depth=` / dml's
# `model_y=` kwargs.  Same trick the package already uses on L127-129
# to re-bind `sp.iv` to the submodule.
# ---------------------------------------------------------------------
from ._article_aliases import (
    matrix_completion,
    causal_discovery,
    mediation,
    policy_tree,
    dml,
)


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
