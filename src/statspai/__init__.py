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

__version__ = "1.13.0"
__author__ = "Biaoyue Wang"
__email__ = "brycew6m@stanford.edu"

from ._citation import citation
__citation__ = citation("bibtex")

from .core.results import EconometricResults, CausalResult
# Agent-native exception taxonomy (load early so every estimator can raise)
from . import exceptions as exceptions  # noqa: F401
from .exceptions import (
    StatsPAIError,
    AssumptionViolation,
    IdentificationFailure,
    DataInsufficient,
    ConvergenceFailure,
    NumericalInstability,
    MethodIncompatibility,
    StatsPAIWarning,
    ConvergenceWarning,
    AssumptionWarning,
)
from .regression.ols import regress
# NB: ``iv`` is intentionally NOT imported here.  ``sp.iv`` resolves to the
# callable :mod:`statspai.iv` subpackage (loaded via ``from .iv.* import``
# below), which dispatches ``method=``/2sls/liml/fuller/gmm/jive/kernel/...
# Importing the function at the top level would shadow the subpackage and
# break ``sp.iv("y ~ (d ~ z)", data=df)``.
from .regression.iv import ivreg, IVRegression
from .forest.causal_forest import CausalForest, causal_forest
from .forest.forest_inference import (
    calibration_test, test_calibration, rate, honest_variance,
)
from .forest.multi_arm_forest import multi_arm_forest, MultiArmForestResult
from .forest.iv_forest import iv_forest, IVForestResult
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
    # v1.7 long-panel DML
    dml_panel, DMLPanelResult,
)
# Eager: ``deepiv`` is both a function (sp.deepiv(...)) and a subpackage.
# Lazy-loading collides with the subpackage attachment — see the
# "PEP 562 collision" note at the bottom of this file.
from .deepiv import deepiv, DeepIV
from .panel import (
    panel, panel_compare, balance_panel, PanelResults, PanelCompareResults, PanelRegression,
    Absorber, demean, absorb_ols, hdfe_ols, FEOLSResult,
)
# Eager: ``causal_impact`` collides (function + subpackage of same name).
from .causal_impact import causal_impact, CausalImpactEstimator, impactplot
from .mediation import mediate, MediationAnalysis, mediate_sensitivity, mediate_interventional, four_way_decomposition, FourWayResult
# Eager: ``bartik`` collides (function + subpackage of same name).
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
from .output.collection import Collection, CollectionItem, collect
from .output._inline import cite
from .output._journals import (
    JOURNALS as JOURNAL_PRESETS,
    list_templates as list_journal_templates,
    get_template as get_journal_template,
)
from .output._lineage import (
    Provenance,
    attach_provenance,
    get_provenance,
    compute_data_hash,
    format_provenance,
    lineage_summary,
)
from .output._replication_pack import ReplicationPack, replication_pack
from .output._gt import to_gt as gt, is_great_tables_available
from .output._bibliography import (
    csl_url,
    csl_filename,
    list_csl_styles,
    parse_citation_to_bib,
    make_bib_key,
    citations_to_bib_entries,
    write_bib,
)
from .postestimation import margins, margins_table, event_study_table, marginsplot, margins_at, margins_at_plot, contrast, pwcompare, test, lincom
from .diagnostics import oster_bounds, mccrary_test, diagnose, het_test, reset_test, vif, sensemakr, rddensity, hausman_test, anderson_rubin_test, effective_f_test, tF_critical_value, evalue, evalue_from_result, diagnose_result, estat, kitagawa_test, KitagawaResult, rosenbaum_bounds, rosenbaum_gamma, RosenbaumResult, weakrobust, WeakRobustResult
from .inference import (
    wild_cluster_bootstrap, aipw, ri_test, ipw, bootstrap, BootstrapResult,
    twoway_cluster, conley, pate, PATEEstimator, fisher_exact, FisherResult,
    jackknife_se, cr2_se, wild_cluster_boot,
    subcluster_wild_bootstrap, wild_cluster_ci_inv,
    multiway_cluster_vcov, cluster_robust_se, cr3_jackknife_vcov,
    g_computation, front_door,
)
# Eager: ``msm`` collides (function + subpackage of same name).
from .msm import msm, MarginalStructuralModel, stabilized_weights
# Eager: ``proximal`` collides (function + subpackage of same name).
from .proximal import (
    proximal, ProximalCausalInference,
    negative_control_outcome, negative_control_exposure,
    double_negative_control, NegativeControlResult,
    proximal_regression, ProximalRegResult,
    fortified_pci, bidirectional_pci, pci_mtp,
    select_pci_proxies, ProxyScoreResult,
)
# Eager: ``principal_strat`` collides (function + subpackage of same name).
from .principal_strat import (
    principal_strat, PrincipalStratResult, survivor_average_causal_effect,
)
# (lazy) spatial: see _LAZY_SUBMODULES / _LAZY_ATTRS
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
# bayes — lazy-loaded (PyMC pulls heavy deps); see _LAZY_ATTRS below.
from .regression.heckman import heckman
from .regression.quantile import qreg, sqreg
from .regression.tobit import tobit
from .regression.logit_probit import logit, probit, cloglog
from .regression.glm import glm, GLMRegression, GLMEstimator
from .regression.count import poisson, nbreg, ppmlhdfe
# neural_causal — lazy-loaded (torch); see _LAZY_ATTRS below.
from .causal_discovery import notears, NOTEARS, pc_algorithm, PCAlgorithm, lingam, LiNGAMResult, ges, GESResult, fci, FCIResult, icp, nonlinear_icp, ICPResult, pcmci, PCMCIResult, partial_corr_pvalue, lpcmci, LPCMCIResult, dynotears, DYNOTEARSResult
# Eager: ``tmle`` collides (function + subpackage of same name).
from .tmle import (
    tmle, TMLE, super_learner, SuperLearner,
    ltmle, LTMLEResult, ltmle_survival, LTMLESurvivalResult,
    hal_tmle, HALRegressor, HALClassifier,
)
from .policy_learning import policy_tree, PolicyTree, policy_value, direct_method, ips, snips, doubly_robust
# ``OPEResult`` is intentionally *not* eagerly imported from
# ``.policy_learning`` here: the canonical class lives in
# ``statspai.ope.estimators`` and is what ``sp.ope.ips(...)`` returns.
# Letting ``sp.OPEResult`` resolve via the lazy ``_register_lazy("ope",
# "OPEResult", ...)`` table keeps ``isinstance(res, sp.OPEResult)`` true
# for results produced by ``sp.ope.*`` (matching v1.12.2 semantics).
# (lazy) conformal_causal: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Eager: ``bcf`` collides (function + subpackage of same name).
from .bcf import (
    bcf, BayesianCausalForest, bcf_longitudinal, BCFLongResult,
    bcf_ordinal, BCFOrdinalResult,
    bcf_factor_exposure, BCFFactorExposureResult,
)
# Eager: ``bunching`` collides (function + subpackage of same name).
from .bunching import bunching, BunchingEstimator, notch, NotchResult
from .bunching import (
    general_bunching, GeneralBunchingResult,
    kink_unified, KinkUnifiedResult,
)
from .matrix_completion import mc_panel, MCPanel
# Eager: ``dose_response`` collides (function + subpackage of same name).
from .dose_response import dose_response, DoseResponse, vcnet, scigan, VCNetResult
# (lazy) bounds: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Eager: ``interference`` collides (function + subpackage of same name).
from .interference import spillover, SpilloverEstimator, network_exposure, NetworkExposureResult, peer_effects, PeerEffectsResult, network_hte, inward_outward_spillover, NetworkHTEResult, InwardOutwardResult
from .interference import (
    cluster_matched_pair, MatchedPairResult,
    cluster_cross_interference, CrossClusterRCTResult,
    cluster_staggered_rollout, StaggeredClusterRCTResult,
    dnc_gnn_did, DNCGNNDiDResult,
    interference, interference_available_designs,
)
# (lazy) dtr: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Eager: ``multi_treatment`` collides (function + subpackage of same name).
from .multi_treatment import multi_treatment, MultiTreatment
# (lazy) robustness_a: see _LAZY_SUBMODULES / _LAZY_ATTRS
# (lazy) survey: see _LAZY_SUBMODULES / _LAZY_ATTRS
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
# Eager: ``bridge`` collides (function + subpackage of same name).
from .bridge import bridge, BridgeResult

# === LLM × Causal (DAG / E-value / sensitivity priors) ===
# (lazy) causal_llm: see _LAZY_SUBMODULES / _LAZY_ATTRS

# causal_text / causal_rl / surrogate / assimilation / fairness — lazy-loaded;
# the seven heavy/niche subtrees are wired through _LAZY_SUBMODULES /
# _LAZY_ATTRS at the bottom of this file so cold ``import statspai``
# does not pay for them up front.

# === Transportability (Pearl-Bareinboim + Dahabreh-Stuart) ===
# (lazy) transport: see _LAZY_SUBMODULES / _LAZY_ATTRS

# === Off-Policy Evaluation (contextual bandits) ===
# (lazy) ope: see _LAZY_SUBMODULES / _LAZY_ATTRS

# === Parametric g-formula (iterative conditional expectation) ===
# (lazy) gformula: see _LAZY_SUBMODULES / _LAZY_ATTRS

# === Target Trial Emulation (JAMA 2022 framework) ===
# (lazy) target_trial: see _LAZY_SUBMODULES / _LAZY_ATTRS
# === Inverse probability of censoring weights ===
# (lazy) censoring: see _LAZY_SUBMODULES / _LAZY_ATTRS

# === Epidemiology primitives (OR / RR / MH / standardization / BH) ===
# (lazy) epi: see _LAZY_SUBMODULES / _LAZY_ATTRS

# === Longitudinal causal inference (What If Layer 4) ===
# (lazy) longitudinal: see _LAZY_SUBMODULES / _LAZY_ATTRS

# === Causal-question DSL (estimand-first workflow) ===
from . import question
from .question import (
    causal_question, CausalQuestion,
    IdentificationPlan, EstimationResult,
    preregister, load_preregister,
)

# === Unified sensitivity dashboard ===
# (lazy) robustness_b: see _LAZY_SUBMODULES / _LAZY_ATTRS

# === Canonical datasets (consolidated facade) ===
from . import datasets

# === End-to-end workflow orchestrator ===
# After ``import statspai.causal`` (the deprecated forest-shim) Python rebinds
# ``sp.causal`` to that submodule, shadowing this function.  The shim works
# around it by making its module object callable (see ``causal/__init__.py``),
# so ``sp.causal(df, y=, treatment=, ...)`` keeps dispatching to this
# workflow function in either order.
from .workflow import causal, CausalWorkflow, paper, PaperDraft

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
# (lazy) mht: see _LAZY_SUBMODULES / _LAZY_ATTRS
from .registry import (
    list_functions,
    describe_function,
    function_schema,
    search_functions,
    all_schemas,
    agent_card,
    agent_cards,
    FailureMode,
    STABILITY_TIERS,
)
from ._agent_docs import render_agent_block, render_agent_blocks
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
# GLM & Discrete Choice — ``glm``/``logit``/``probit``/``cloglog``/
# ``poisson``/``nbreg``/``ppmlhdfe`` are already imported above in the
# core regression block; we only add what's new here.
from .regression.multinomial import mlogit, ologit, oprobit, clogit
from .regression.mixed_logit import mixlogit
from .regression.iv_quantile import ivqreg
# Count Data
from .regression.zeroinflated import zip_model, zinb, hurdle
# Advanced IV
from .regression.advanced_iv import liml, jive, lasso_iv
# High-dimensional fixed effects (pyfixest backend)
# These are thin wrappers; actual import of pyfixest is deferred to call time
# via fixest.wrapper._check_pyfixest, so top-level import never fails.
from .fixest import feols, fepois, feglm, etable
# Survival / Duration
# (lazy) survival: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Nonparametric
# (lazy) nonparametric: see _LAZY_SUBMODULES / _LAZY_ATTRS
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
# (lazy) experimental: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Missing Data / Imputation
# (lazy) imputation: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Mendelian Randomization
# NOTE: v1.5 replaces the `sp.mr` module alias with a dispatcher
# function `sp.mr(method=..., ...)` mirroring sp.synth / sp.decompose /
# sp.dml.  Use `sp.mendelian` for module-level access.
# (lazy) mendelian: see _LAZY_SUBMODULES / _LAZY_ATTRS
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
from .did.lp_did import lp_did
from .did.ddd_heterogeneous import ddd_heterogeneous
from .did.timevarying_covariates import did_timevarying_covariates
from .did.did_multiplegt_dyn import did_multiplegt_dyn

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
# Structural Estimation (BLP, production functions)
# (lazy) structural_a: see _LAZY_SUBMODULES / _LAZY_ATTRS
# (lazy) structural_b: see _LAZY_SUBMODULES / _LAZY_ATTRS

# === Smart Workflow Engine (unique to StatsPAI) ===
from .smart import (
    recommend, RecommendationResult,
    compare_estimators, ComparisonResult,
    assumption_audit, AssumptionResult,
    audit,
    bib_for,
    brief,
    detect_design,
    examples,
    preflight,
    session,
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
# (lazy) multilevel: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Stochastic Frontier
# Eager: ``frontier`` collides (function + subpackage of same name).
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
    # Agent-native exception taxonomy
    "exceptions",
    "StatsPAIError",
    "AssumptionViolation",
    "IdentificationFailure",
    "DataInsufficient",
    "ConvergenceFailure",
    "NumericalInstability",
    "MethodIncompatibility",
    "StatsPAIWarning",
    "ConvergenceWarning",
    "AssumptionWarning",
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
    # v1.7 long-panel DML
    "dml_panel",
    "DMLPanelResult",
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
    "Collection",
    "CollectionItem",
    "collect",
    "paper_tables",
    "PaperTables",
    "PAPER_TABLE_TEMPLATES",
    "cite",
    "citation",
    "JOURNAL_PRESETS",
    "list_journal_templates",
    "get_journal_template",
    # Lineage / provenance (numerical traceability)
    "Provenance",
    "attach_provenance",
    "get_provenance",
    "compute_data_hash",
    "format_provenance",
    "lineage_summary",
    # Replication pack (journal-ready zip)
    "ReplicationPack",
    "replication_pack",
    # great_tables adapter (publication-grade tables)
    "gt",
    "is_great_tables_available",
    # Bibliography / CSL (Quarto citation pipeline)
    "csl_url",
    "csl_filename",
    "list_csl_styles",
    "parse_citation_to_bib",
    "make_bib_key",
    "citations_to_bib_entries",
    "write_bib",
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
    "margins_table",
    "event_study_table",
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
    "agent_card",
    "agent_cards",
    "FailureMode",
    "STABILITY_TIERS",
    "render_agent_block",
    "render_agent_blocks",
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
    # GLM & Discrete Choice (glm/logit/probit/cloglog already in regression block above)
    "mlogit", "ologit", "oprobit", "clogit",
    "mixlogit", "ivqreg",
    # Count Data (poisson/nbreg/ppmlhdfe already in regression block above)
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
    # LP-DiD (Dube-Girardi-Jordà-Taylor 2023)
    "lp_did",
    # Heterogeneity-robust DDD (Olden-Møen 2022)
    "ddd_heterogeneous",
    # Time-varying covariates DiD (Caetano et al. 2022)
    "did_timevarying_covariates",
    # dCDH (2024) intertemporal event-study DiD (MVP — see RFC)
    "did_multiplegt_dyn",
    # === v0.6 Round 2 ===
    "interactive_fe",
    "panel_unitroot", "PanelUnitRootResult",
    "engle_granger", "johansen", "CointegrationResult",
    "fracreg", "betareg",
    "biprobit", "etregress",
    "distributional_te", "DTEResult",
    # Structural Estimation
    "blp", "BLPResult",
    # Production functions (proxy-variable estimators)
    "prod_fn",
    "olley_pakes", "opreg",
    "levinsohn_petrin", "levpet",
    "ackerberg_caves_frazer", "acf",
    "wooldridge_prod",
    "markup",
    "ProductionResult",
    # === Smart Workflow Engine (unique to StatsPAI) ===
    "recommend", "RecommendationResult",
    "check_identification", "IdentificationReport", "DiagnosticFinding",
    "IdentificationError",
    "compare_estimators", "ComparisonResult",
    "assumption_audit", "AssumptionResult",
    "audit",
    "bib_for",
    "brief",
    "detect_design",
    "examples",
    "preflight",
    "session",
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
    # Causal discovery (notears/NOTEARS/pc_algorithm/PCAlgorithm already listed above)
    "lingam", "LiNGAMResult", "ges", "GESResult",
    "fci", "FCIResult",
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
    # v0.9.17 additions (unified sensitivity; SensitivityDashboard already exported)
    "unified_sensitivity",
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
    # === v0.10 / v1.0 frontier additions (most are already exported above) ===
    # Distributional / panel QTE — only the new dist_iv/kan_dlate/DistIVResult here
    "dist_iv", "kan_dlate", "DistIVResult",
    # RDD frontier — only the new rd_bayes_hte / rd_distributional_design here
    "rd_bayes_hte", "BayesRDHTEResult",
    "rd_distributional_design", "DDDResult",
    # Causal × LLM
    "llm_dag_propose", "LLMDAGProposal",
    "llm_unobserved_confounders", "UnobservedConfounderProposal",
    "llm_sensitivity_priors", "SensitivityPriorProposal",
    "llm_dag_constrained", "llm_dag_validate",
    "LLMConstrainedDAGResult", "DAGValidationResult",
    # Causal × Text (P1-B v1.6 experimental)
    "text_treatment_effect", "TextTreatmentResult",
    "llm_annotator_correct", "LLMAnnotatorResult",
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
    # v1.6 MR frontier: sample-overlap / clusters / profile-LL / cML / RAPS
    "mr_lap", "mr_clust", "grapple", "mr_cml", "mr_raps",
    "MRLapResult", "MRClustResult", "GrappleResult",
    "MRcMLResult", "MRRapsResult",
    # v1.5 unified family dispatchers (mr already exported above as the dispatcher)
    "mr_available_methods",
    "conformal", "conformal_available_kinds",
    "interference", "interference_available_designs",
    # v1.5 registry coverage fixes for previously-exposed-but-unregistered
    # single-family functions (now reachable via sp.describe_function too)
    "network_exposure", "NetworkExposureResult",
    "peer_effects", "PeerEffectsResult",
    "weighted_conformal_prediction",
    "conformal_counterfactual", "ConformalCounterfactualResult",
    "conformal_ite_interval", "ConformalITEResult",
]


def _dedupe_public_exports(names):
    """Preserve the first occurrence of each public export name.

    ``__all__`` is also the seed for parts of the registry/help surface;
    duplicate names create avoidable drift in docs and tooling while
    adding no value at runtime.
    """
    seen = set()
    out = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


__all__ = _dedupe_public_exports(__all__)
del _dedupe_public_exports


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


# ---------------------------------------------------------------------
# Lazy submodule registry (Step 1 — cold-start budget)
# ---------------------------------------------------------------------
# Heavy / niche subtrees that ~99% of sessions never touch but whose eager
# import inflated ``import statspai`` cold start to ~1.8 s.  Listing them
# here defers the actual ``import .X`` until ``sp.<name>`` is first
# accessed; afterwards the result is cached in ``globals()`` so the
# ``__getattr__`` hop only happens once per name.
#
# - ``_LAZY_SUBMODULES`` — public name → dotted submodule path; access
#   returns the module object so ``sp.fairness.demographic_parity``-style
#   namespacing keeps working.  Aliases (e.g. ``sp.tte`` for
#   ``target_trial``) are expressed by mapping two public names to the
#   same path.
# - ``_LAZY_ATTRS`` — public leaf name → (submodule path, source attr).
#   ``source_attr`` may differ from public name to express
#   ``from .X import name as alias`` import aliasing.
#
# ``from statspai import *`` still triggers every lazy import via
# ``__all__`` iteration — intentional; the lazy path only buys back cold
# import time for the dominant ``import statspai as sp`` flow.
#
# **PEP 562 collision rule.**  When a public function and its parent
# submodule share a name (``proximal`` / ``principal_strat`` / ``bartik``
# / ``bridge`` / ``causal_impact`` / ``bcf`` / ``bunching`` / ``deepiv``
# / ``dose_response`` / ``frontier`` / ``interference`` / ``msm`` /
# ``multi_treatment`` / ``tmle``), Python's import machinery insists on
# attaching ``statspai.X = <submodule>`` whenever any code path runs
# ``from statspai.X import Y`` (very common in tests).  ``__getattr__``
# can re-bind ``sp.X`` once, but the next ``from-import`` attaches the
# module again — defeating the rebind.  These 14 names are therefore
# imported eagerly above so the standard ``from .X import X`` rebinding
# happens at module load and survives subsequent ``from statspai.X
# import …`` calls.  Do **not** move them into the lazy table without
# also resolving the collision (e.g. by renaming the function).
# ---------------------------------------------------------------------
_LAZY_SUBMODULES: dict = {
    # name on sp -> dotted submodule path (relative to statspai)
    "bayes":             "bayes",
    "neural_causal":     "neural_causal",
    "causal_text":       "causal_text",
    "causal_rl":         "causal_rl",
    "fairness":          "fairness",
    "assimilation":      "assimilation",
    "surrogate":         "surrogate",
    "bounds":            "bounds",
    "dtr":               "dtr",
    "spatial":           "spatial",
    "conformal_causal":  "conformal_causal",
    "ope":               "ope",
    "censoring":         "censoring",
    "epi":               "epi",
    "longitudinal":      "longitudinal",
    "gformula":          "gformula",
    "target_trial":      "target_trial",
    "tte":               "target_trial",  # short alias
    "transport":         "transport",
    "mendelian":         "mendelian",
    "experimental":      "experimental",
    "imputation":        "imputation",
    "survey":            "survey",
    "survival":          "survival",
    "nonparametric":     "nonparametric",
    "multilevel":        "multilevel",
    "structural":        "structural",
    "causal_llm":        "causal_llm",
    "mht":               "mht",
    "robustness":        "robustness",
}


def _register_lazy(modname, *names):
    """Register leaf names exported from a lazy submodule.

    Each ``names`` item is either a bare string (public name == source
    attr) or a ``(public_name, source_attr)`` tuple to express
    ``from .X import attr as public_name`` aliasing.  All entries land
    in ``_LAZY_ATTRS`` keyed by public name.
    """
    for item in names:
        if isinstance(item, str):
            _LAZY_ATTRS[item] = (modname, item)
        else:
            public, source = item
            _LAZY_ATTRS[public] = (modname, source)


_LAZY_ATTRS: dict = {}

_register_lazy("bayes",
    "bayes_did", "bayes_rd", "bayes_iv", "bayes_fuzzy_rd", "bayes_hte_iv",
    "bayes_mte", "bayes_dml", "BayesianDMLResult",
    "BayesianCausalResult", "BayesianDIDResult", "BayesianHTEIVResult",
    "BayesianIVResult", "BayesianMTEResult",
    "policy_weight_ate", "policy_weight_subsidy",
    "policy_weight_prte", "policy_weight_marginal",
    "policy_weight_observed_prte",
)
_register_lazy("neural_causal",
    "tarnet", "cfrnet", "dragonnet", "TARNet", "CFRNet", "DragonNet",
    "gnn_causal", "GNNCausalResult",
)
_register_lazy("neural_causal.cevae",
    "cevae", "CEVAE", "CEVAEResult",
)
_register_lazy("causal_text",
    "text_treatment_effect", "TextTreatmentResult",
    "llm_annotator_correct", "LLMAnnotatorResult",
)
_register_lazy("causal_rl",
    "causal_dqn", "CausalDQNResult",
    "causal_rl_benchmark", "BanditBenchmarkResult",
    "offline_safe_policy", "OfflineSafeResult",
    "causal_bandit", "counterfactual_policy_optimization", "structural_mdp",
    "CausalBanditResult", "CFPolicyResult", "StructuralMDPResult",
)
_register_lazy("fairness",
    "counterfactual_fairness", "orthogonal_to_bias",
    "demographic_parity", "equalized_odds", "fairness_audit",
    "FairnessResult", "FairnessAudit",
    "evidence_without_injustice", "EvidenceWithoutInjusticeResult",
)
_register_lazy("assimilation",
    "assimilative_causal", "causal_kalman", "AssimilationResult",
)
_register_lazy("surrogate",
    "surrogate_index", "long_term_from_short", "proximal_surrogate_index",
    "SurrogateResult",
)
_register_lazy("bounds",
    "lee_bounds", "manski_bounds", "BoundsResult", "horowitz_manski",
    "iv_bounds", "oster_delta", "selection_bounds", "breakdown_frontier",
    "balke_pearl", "BalkePearlResult", "ml_bounds", "MLBoundsResult",
)
_register_lazy("dtr",
    "g_estimation", "GEstimation", "q_learning", "QLearningResult",
    "a_learning", "ALearningResult", "snmm", "SNMMResult",
)
_register_lazy("spatial",
    "sar", "sem", "sdm", "slx", "sac", "SpatialModel",
    "sar_gmm", "sem_gmm", "sarar_gmm",
    "gwr", "mgwr", "gwr_bandwidth",
    "spatial_panel",
    "W", "queen_weights", "rook_weights", "knn_weights",
    "distance_band", "kernel_weights", "block_weights",
    "moran", "moran_local", "geary", "getis_ord_g", "getis_ord_local", "join_counts",
    "moran_plot", "lisa_cluster_map",
    "lm_tests", "moran_residuals", "impacts",
    "spatial_did", "SpatialDiDResult", "spatial_iv", "SpatialIVResult",
)
_register_lazy("conformal_causal",
    "conformal_cate", "ConformalCATE",
    "weighted_conformal_prediction",
    "conformal_counterfactual", "ConformalCounterfactualResult",
    "conformal_ite_interval", "ConformalITEResult",
    "conformal_density_ite", "ConformalDensityResult",
    "conformal_ite_multidp", "MultiDPConformalResult",
    "conformal_debiased_ml", "DebiasedConformalResult",
    "conformal_fair_ite", "FairConformalResult",
    "conformal_continuous", "conformal_interference",
    "ContinuousConformalResult", "InterferenceConformalResult",
    "conformal", "conformal_available_kinds",
)
_register_lazy("ope",
    "OPEResult", "sharp_ope_unobserved", "causal_policy_forest",
    "SharpOPEResult", "CausalPolicyForestResult",
)
_register_lazy("censoring",
    "ipcw", "IPCWResult",
)
_register_lazy("epi",
    "odds_ratio", "relative_risk", "risk_difference", "attributable_risk",
    "incidence_rate_ratio", "number_needed_to_treat", "prevalence_ratio",
    "mantel_haenszel", "breslow_day_test",
    "direct_standardize", "indirect_standardize",
    "bradford_hill",
    "diagnostic_test", "sensitivity_specificity",
    "roc_curve", "auc", "cohen_kappa",
    "DiagnosticTestResult", "ROCResult", "KappaResult",
)
_register_lazy("longitudinal",
    ("longitudinal_analyze",  "analyze"),
    ("longitudinal_contrast", "contrast"),
    "regime", "always_treat", "never_treat",
    "LongitudinalResult", "Regime",
)
_register_lazy("gformula",
    ("gformula_ice_fn", "ice"),
    "ICEResult",
    "gformula_mc", "MCGFormulaResult",
)
_register_lazy("target_trial",
    ("target_trial_protocol",  "protocol"),
    ("target_trial_emulate",   "emulate"),
    ("target_trial_report",    "to_paper"),
    ("target_trial_checklist", "target_checklist"),
    "TARGET_ITEMS",
    "clone_censor_weight",
    "immortal_time_check",
    "TargetTrialProtocol",
    "TargetTrialResult",
    "CloneCensorWeightResult",
)
_register_lazy("transport",
    ("transport_weights_fn",   "weights"),
    ("transport_generalize",   "generalize"),
    "TransportWeightResult",
    "identify_transport", "TransportIdentificationResult",
    "synthesise_evidence", "heterogeneity_of_effect", "rwd_rct_concordance",
    "EvidenceSynthesisResult", "HeterogeneityResult", "ConcordanceResult",
)
_register_lazy("mendelian",
    "mendelian_randomization", "MRResult",
    "mr_egger", "mr_ivw", "mr_median",
    "mr_heterogeneity", "mr_pleiotropy_egger", "mr_leave_one_out",
    "mr_steiger", "mr_presso", "mr_radial",
    "HeterogeneityResult", "PleiotropyResult", "LeaveOneOutResult",
    "SteigerResult", "MRPressoResult", "RadialResult",
    "mr_mode", "mr_f_statistic", "mr_funnel_plot", "mr_scatter_plot",
    "ModeBasedResult", "FStatisticResult",
    "mr_multivariable", "mr_mediation", "mr_bma",
    "MVMRResult", "MediationMRResult", "MRBMAResult",
    "mr_lap", "mr_clust", "grapple", "mr_cml", "mr_raps",
    "MRLapResult", "MRClustResult", "GrappleResult", "MRcMLResult", "MRRapsResult",
    "mr", "mr_available_methods",
)
_register_lazy("experimental",
    "randomize", "RandomizationResult",
    "balance_check", "BalanceResult",
    "attrition_test", "attrition_bounds", "AttritionResult",
    "optimal_design", "OptimalDesignResult",
)
_register_lazy("imputation",
    "mice", "MICEResult", "mi_estimate",
)
_register_lazy("survey",
    "svydesign", "SurveyDesign", "svymean", "svytotal", "svyglm",
    "rake", "linear_calibration",
)
_register_lazy("survival",
    "cox", "kaplan_meier", "survreg",
    "CoxResult", "KMResult", "logrank_test",
    "cox_frailty", "aft",
    "causal_survival_forest", "causal_survival",
    "CausalSurvivalForestResult",
)
_register_lazy("nonparametric",
    "lpoly", "LPolyResult", "kdensity", "KDensityResult",
)
_register_lazy("multilevel",
    "mixed", "MixedResult",
    "meglm", "melogit", "mepoisson", "menbreg", "megamma", "meologit",
    "MEGLMResult", "icc", "lrtest",
)
_register_lazy("structural",
    "blp", "BLPResult",
    "prod_fn",
    "olley_pakes", "opreg",
    "levinsohn_petrin", "levpet",
    "ackerberg_caves_frazer", "acf",
    "wooldridge_prod",
    "markup",
    "ProductionResult",
)
_register_lazy("causal_llm",
    "llm_dag_propose", "LLMDAGProposal",
    "llm_unobserved_confounders", "UnobservedConfounderProposal",
    "llm_sensitivity_priors", "SensitivityPriorProposal",
    "causal_mas", "CausalMASResult",
    "llm_dag_constrained", "llm_dag_validate",
    "LLMConstrainedDAGResult", "DAGValidationResult",
)
_register_lazy("mht",
    "romano_wolf", "RomanoWolfResult",
    "adjust_pvalues", "bonferroni", "holm", "benjamini_hochberg",
)
_register_lazy("robustness",
    "spec_curve", "SpecCurveResult",
    "robustness_report", "RobustnessResult",
    "subgroup_analysis", "SubgroupResult",
    "copula_sensitivity", "survival_sensitivity",
    "calibrate_confounding_strength", "FrontierSensitivityResult",
    "unified_sensitivity", "SensitivityDashboard",
)


def __getattr__(name):
    """Lazy-load heavy/optional submodules and their leaf exports.

    Three classes of lazy entry handled here:

    1. ``verify`` / ``verify_recommendation`` / ``verify_benchmark`` —
       the resampling-stability path; deferred so ``recommend(verify=False)``
       stays zero-overhead.
    2. ``fast`` — the Rust HDFE backend with NumPy fallback; deferred so
       the Phase-1 extension only loads on first ``sp.fast`` use.
    3. ``_LAZY_SUBMODULES`` / ``_LAZY_ATTRS`` (bayes, neural_causal,
       causal_text, causal_rl, fairness, assimilation, surrogate) —
       Step 1 of the cold-import diet.

    Resolved values are cached in ``globals()`` so subsequent accesses
    bypass ``__getattr__``.
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
    if name == "fast":
        # Submodule — load on demand so the Phase 1 Rust extension import
        # (and its NumPy fallback path) only fires if a user touches
        # ``sp.fast``. ``importlib.import_module`` bypasses our
        # ``__getattr__`` so we don't re-enter this branch.
        import importlib
        _fast_mod = importlib.import_module(".fast", package=__name__)
        globals()["fast"] = _fast_mod
        return _fast_mod
    if name in _LAZY_ATTRS:
        import importlib
        _modpath, _attr = _LAZY_ATTRS[name]
        _mod = importlib.import_module(f".{_modpath}", package=__name__)
        _obj = getattr(_mod, _attr)
        globals()[name] = _obj
        return _obj
    if name in _LAZY_SUBMODULES:
        import importlib
        _modpath = _LAZY_SUBMODULES[name]
        _mod = importlib.import_module(f".{_modpath}", package=__name__)
        globals()[name] = _mod
        return _mod
    raise AttributeError(f"module 'statspai' has no attribute {name!r}")
