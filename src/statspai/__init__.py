"""
StatsPAI: Validation-tiered statistics and econometrics workflows for Python

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

__version__ = "1.20.0"
__author__ = "Biaoyue Wang and Scott Rozelle"
__email__ = "brycew6m@stanford.edu"

from ._citation import citation

__citation__ = citation("bibtex")

# (lazy) spatial: see _LAZY_SUBMODULES / _LAZY_ATTRS
# NOTE: `from . import iv` would be shadowed by the `iv` function imported
# on line 31 from `.regression.iv`, so we load the subpackage explicitly.
import importlib as _importlib

# Agent-native exception taxonomy (load early for registered estimators)
from . import exceptions as exceptions  # noqa: F401

# Eager: ``bartik`` collides (function + subpackage of same name).
from .bartik import (
    BartikIV,
    ShiftSharePoliticalPanelResult,
    ShiftSharePoliticalResult,
    bartik,
    shift_share_political,
    shift_share_political_panel,
    shift_share_se,
    ssaggregate,
)

# Eager: ``causal_impact`` collides (function + subpackage of same name).
from .causal_impact import CausalImpactEstimator, causal_impact, impactplot
from .core.effect_summary import EffectSummary, effect_summary  # noqa: E402
from .core.results import CausalResult, EconometricResults

# Eager: ``deepiv`` is both a function (sp.deepiv(...)) and a subpackage.
# Lazy-loading collides with the subpackage attachment — see the
# "PEP 562 collision" note at the bottom of this file.
from .deepiv import DeepIV, deepiv
from .diagnostics import (
    KitagawaResult,
    RosenbaumResult,
    WeakRobustResult,
    anderson_rubin_test,
    bias_factor,
    diagnose,
    diagnose_result,
    effective_f_test,
    estat,
    evalue,
    evalue_from_result,
    evalue_rd,
    hausman_test,
    het_test,
    kitagawa_test,
    mccrary_test,
    oster_bounds,
    rddensity,
    reset_test,
    rosenbaum_bounds,
    rosenbaum_gamma,
    sensemakr,
    tF_critical_value,
    vif,
    weakrobust,
)

# (lazy) forest: see _LAZY_SUBMODULES / _LAZY_ATTRS below.  Eagerly
# importing ``forest.causal_forest`` etc. pulled ~245 sklearn submodules
# into every ``import statspai`` (~270 ms cumulative on cold cache),
# even for sessions that never touch heterogeneous-effect forests.  The
# ``forest`` name does *not* collide with a top-level function (no
# ``sp.forest`` callable export), so the standard lazy path is safe.
from .did import (
    CSReport,
    DIDAnalysis,
    HarvestDIDResult,
    ParallelTrendsRobustnessResult,
    SensitivityResult,
    aggte,
    bacon_decomposition,
    bacon_plot,
    bjs,
    bjs_pretrend_joint,
    borusyak_jaravel_spiess,
    breakdown_m,
    callaway_santanna,
    cic,
    cohort_anchored_event_study,
    cohort_event_study_plot,
    cs_report,
    ddd,
    design_robust_event_study,
    did,
    did_2stage,
    did_2x2,
    did_analysis,
    did_bcf,
    did_imputation,
    did_misclassified,
    did_multiplegt,
    did_plot,
    did_report,
    did_summary,
    did_summary_plot,
    did_summary_to_latex,
    did_summary_to_markdown,
    dl_propensity_score,
    drdid,
    enhanced_event_study_plot,
    etwfe,
    etwfe_emfx,
    event_study,
    gardner_did,
    ggdid,
    group_time_plot,
    harvest_did,
    honest_did,
    overlap_weighted_did,
    parallel_trends_plot,
    parallel_trends_robustness,
    pretrends_power,
    pretrends_summary,
    pretrends_test,
    sensitivity_plot,
    sensitivity_rr,
    stacked_did,
    sun_abraham,
    treatment_rollout_plot,
    twfe_decomposition,
    wooldridge_did,
)
from .dml import (  # v1.7 long-panel DML; v1.13 DML-OVB sensitivity + diagnostics
    DMLAveragingResult,
    DMLDiagnostics,
    DMLPanelResult,
    DMLSensitivityResult,
    DoubleML,
    DoubleMLIIVM,
    DoubleMLIRM,
    DoubleMLPLIV,
    DoubleMLPLR,
    dml,
    dml_diagnostics,
    dml_model_averaging,
    dml_panel,
    dml_sensitivity,
    model_averaging_dml,
)
from .exceptions import (
    AssumptionViolation,
    AssumptionWarning,
    ConvergenceFailure,
    ConvergenceWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    NumericalInstability,
    StatsPAIError,
    StatsPAIWarning,
)
from .inference import (
    BootstrapResult,
    FisherResult,
    MetaAnalysisResult,
    PATEEstimator,
    aipw,
    bootstrap,
    cluster_robust_se,
    conley,
    cr2_se,
    cr3_jackknife_vcov,
    fisher_exact,
    front_door,
    g_computation,
    ipw,
    jackknife_se,
    meta_analysis,
    multiway_cluster_vcov,
    pate,
    ri_test,
    subcluster_wild_bootstrap,
    twoway_cluster,
    wild_cluster_boot,
    wild_cluster_bootstrap,
    wild_cluster_ci_inv,
)
from .matching import (
    BalanceDiagnosticsResult,
    CardinalityMatchResult,
    GenMatchResult,
    MatchEstimator,
    OptimalMatchResult,
    PSBalanceResult,
    PSMatch2Result,
    SBWResult,
    balance_diagnostics,
    balanceplot,
    cardinality_match,
    cbps,
    ebalance,
    genmatch,
    love_plot,
    match,
    optimal_match,
    overlap_plot,
    overlap_weights,
    propensity_score,
    ps_balance,
    psmatch2,
    psplot,
    sbw,
    trimming,
)
from .mediation import (
    FourWayResult,
    MediationAnalysis,
    four_way_decomposition,
    mediate,
    mediate_interventional,
    mediate_sensitivity,
)

# Eager: ``msm`` collides (function + subpackage of same name).
from .msm import MarginalStructuralModel, msm, stabilized_weights
from .output._bibliography import (
    citations_to_bib_entries,
    csl_filename,
    csl_url,
    list_csl_styles,
    make_bib_key,
    parse_citation_to_bib,
    write_bib,
)
from .output._gt import is_great_tables_available
from .output._gt import to_gt as gt
from .output._inline import cite
from .output._journals import JOURNALS as JOURNAL_PRESETS
from .output._journals import get_template as get_journal_template
from .output._journals import list_templates as list_journal_templates
from .output._lineage import (
    Provenance,
    attach_provenance,
    compute_data_hash,
    format_provenance,
    get_provenance,
    lineage_summary,
)
from .output._replication_pack import ReplicationPack, replication_pack
from .output.collection import Collection, CollectionItem, collect
from .output.estimates import estclear, eststo, esttab
from .output.modelsummary import coefplot, coefplot_tikz, modelsummary
from .output.outreg2 import OutReg2, outreg2
from .output.paper_tables import TEMPLATES as PAPER_TABLE_TEMPLATES
from .output.paper_tables import PaperTables, paper_tables
from .output.regression_table import (
    MeanComparisonResult,
    RegtableResult,
    mean_comparison,
    regtable,
)
from .output.sumstats import balance_table, sumstats
from .output.tab import tab
from .panel import (
    Absorber,
    FEOLSResult,
    PanelCompareResults,
    PanelRegression,
    PanelResults,
    SlopeSpec,
    absorb_ols,
    balance_panel,
    demean,
    hdfe_ols,
    panel,
    panel_compare,
)
from .postestimation import (
    contrast,
    event_study_table,
    lincom,
    margins,
    margins_at,
    margins_at_plot,
    margins_table,
    marginsplot,
    postestimation_contract,
    postestimation_report,
    pwcompare,
    test,
)

# Eager: ``principal_strat`` collides (function + subpackage of same name).
from .principal_strat import (
    PrincipalStratResult,
    principal_strat,
    survivor_average_causal_effect,
)

# Eager: ``proximal`` collides (function + subpackage of same name).
from .proximal import (
    NegativeControlResult,
    ProximalCausalInference,
    ProximalRegResult,
    ProxyScoreResult,
    bidirectional_pci,
    double_negative_control,
    fortified_pci,
    negative_control_exposure,
    negative_control_outcome,
    pci_mtp,
    proximal,
    proximal_regression,
    select_pci_proxies,
)
from .rd import (  # v1.15 polish
    BayesRDHTEResult,
    DDDResult,
    DistRDResult,
    MultiScoreRDResult,
    RDInterferenceResult,
    rd_bayes_hte,
    rd_bias_aware_fuzzy,
    rd_boost,
    rd_cate_summary,
    rd_compare,
    rd_dashboard,
    rd_discrete,
    rd_distribution,
    rd_distributional_design,
    rd_external_validity,
    rd_extrapolate,
    rd_flex,
    rd_forest,
    rd_honest,
    rd_interference,
    rd_lasso,
    rd_multi_extrapolate,
    rd_multi_score,
    rd_robustness_table,
    rdbalance,
    rdbwhte,
    rdbwselect,
    rdbwsensitivity,
    rdhte,
    rdhte_lincom,
    rdit,
    rdplacebo,
    rdplot,
    rdplotdensity,
    rdpower,
    rdrandinf,
    rdrbounds,
    rdrobust,
    rdsampsi,
    rdsensitivity,
    rdsummary,
    rdwinselect,
    rkd,
)

# NB: ``iv`` is intentionally NOT imported here.  ``sp.iv`` resolves to the
# callable :mod:`statspai.iv` subpackage (loaded via ``from .iv.* import``
# below), which dispatches ``method=``/2sls/liml/fuller/gmm/jive/kernel/...
# Importing the function at the top level would shadow the subpackage and
# break ``sp.iv("y ~ (d ~ z)", data=df)``.
from .regression.iv import IVRegression, ivreg
from .regression.ols import regress
from .synth import (
    SynthComparison,
    SyntheticControl,
    augsynth,
    basque_terrorism,
    california_prop99,
    california_tobacco,
    conformal_synth,
    demeaned_synth,
    did_estimate,
    discos,
    discos_plot,
    discos_test,
    german_reunification,
    gsynth,
    mc_synth,
    multi_outcome_synth,
    qqsynth,
    robust_synth,
    sc_estimate,
    scdata,
    scest,
    scpi,
    sdid,
    staggered_synth,
    stochastic_dominance,
    synth,
    synth_compare,
    synth_donor_sensitivity,
    synth_loo,
    synth_mde,
    synth_power,
    synth_power_plot,
    synth_recommend,
    synth_report,
    synth_report_to_file,
    synth_rmspe_filter,
    synth_sensitivity,
    synth_sensitivity_plot,
    synth_time_placebo,
    synth_to_excel,
    synth_to_latex,
    synth_to_markdown,
    synthdid_estimate,
    synthdid_placebo,
    synthdid_plot,
    synthdid_rmse_plot,
    synthdid_units_plot,
    synthplot,
)
from .synth.experimental_design import (
    SynthExperimentalDesignResult,
    synth_experimental_design,
)
from .synth.sequential_sdid import SequentialSDIDResult, sequential_sdid
from .synth.survival import SyntheticSurvivalResult, synth_survival

iv = _importlib.import_module(".iv", __name__)
del _importlib
# === LLM agent tool-definition surface ===
# === Canonical datasets (consolidated facade) ===
# === Causal-question DSL (estimand-first workflow) ===
from . import agent  # noqa: F401 — exposed as ``sp.agent``
from . import datasets  # noqa: F401 — exposed as ``sp.datasets``
from . import question
from ._agent_docs import render_agent_block, render_agent_blocks

# === Article-facing aliases (sp.rdd / sp.frontdoor / sp.xlearner / ...) ===
# Thin wrappers around existing implementations; see _article_aliases.py
from ._article_aliases import (  # noqa: E402,F811
    anderson_rubin_ci,
    conditional_lr_ci,
    conformal_ite,
    evalue_rr,
    frontdoor,
    partial_identification,
    psm,
    rdd,
    tF_adjustment,
    xlearner,
)

# === Auto-race estimators (CS/SA/BJS DiD + 2SLS/LIML/JIVE IV) ===
from ._auto_estimators import AutoDIDResult, AutoIVResult, auto_did, auto_iv

# ``OPEResult`` is intentionally *not* eagerly imported from
# ``.policy_learning`` here: the canonical class lives in
# ``statspai.ope.estimators`` and is what ``sp.ope.ips(...)`` returns.
# Letting ``sp.OPEResult`` resolve via the lazy ``_register_lazy("ope",
# "OPEResult", ...)`` table keeps ``isinstance(res, sp.OPEResult)`` true
# for results produced by ``sp.ope.*`` (matching v1.12.2 semantics).
# (lazy) conformal_causal: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Eager: ``bcf`` collides (function + subpackage of same name).
from .bcf import (
    BayesianCausalForest,
    BCFFactorExposureResult,
    BCFLongResult,
    BCFOrdinalResult,
    bcf,
    bcf_factor_exposure,
    bcf_longitudinal,
    bcf_ordinal,
)

# === Bridging theorems (DiD≡SC, EWM≡CATE, CB≡IPW, Kink≡RDD,
#     DR-Calib, Surrogate≡PCI) ===
# Eager: ``bridge`` collides (function + subpackage of same name).
from .bridge import BridgeResult, bridge

# Eager: ``bunching`` collides (function + subpackage of same name).
from .bunching import (
    BunchingEstimator,
    GeneralBunchingResult,
    KinkUnifiedResult,
    NotchResult,
    bunching,
    general_bunching,
    kink_unified,
    notch,
)

# neural_causal — lazy-loaded (torch); see _LAZY_ATTRS below.
from .causal_discovery import (
    NOTEARS,
    DYNOTEARSResult,
    FCIResult,
    GESResult,
    ICPResult,
    LiNGAMResult,
    LPCMCIResult,
    PCAlgorithm,
    PCMCIResult,
    dynotears,
    fci,
    ges,
    icp,
    lingam,
    lpcmci,
    nonlinear_icp,
    notears,
    partial_corr_pvalue,
    pc_algorithm,
    pcmci,
)

# === Cross-engine validation ===
# sp.cross_validate runs one estimand through independent engines (StatsPAI,
# pyfixest, linearmodels, DoubleML, R::fixest, Stata) and reports whether they
# agree. Heavy backends import lazily inside each adapter, so this eager import
# stays light (numpy / pandas only).
from .crossval import CrossValidationResult, cross_validate

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
# (lazy) robustness_a: see _LAZY_SUBMODULES / _LAZY_ATTRS
# (lazy) survey: see _LAZY_SUBMODULES / _LAZY_ATTRS
from .dag import (
    DAG,
    SCM,
    IdentificationResult,
    LLMCausalAssessResult,
    LLMDAGResult,
    PairwiseBenchmarkResult,
    RuleCheck,
    SWIGGraph,
)
from .dag import apply_rules as do_calculus_apply
from .dag import (
    dag,
    dag_example,
    dag_example_positions,
    dag_examples,
    dag_simulate,
    identify,
    llm_causal_assess,
    llm_dag,
    pairwise_causal_benchmark,
)
from .dag import recommend_estimator as dag_recommend_estimator
from .dag import rule1 as do_rule1
from .dag import rule2 as do_rule2
from .dag import rule3 as do_rule3
from .dag import swig

# Data-source ingestion normalisers, also surfaced at top level so an agent can
# reshape a World Bank / FRED / OECD-SDMX payload with one call before fitting.
from .datasets.ingest import from_fred, from_sdmx, from_worldbank
from .decomposition import (  # Tier C additions
    GelbachResult,
    OaxacaResult,
    YuElwertResult,
    available_methods,
    bauer_sinning,
    cfm_decompose,
    chilean_households,
    cps_wage,
    das_gupta,
    decompose,
    dfl_decompose,
    disparity_decompose,
    disparity_panel,
    fairlie,
    ffl_decompose,
    gap_closing,
    gelbach,
    inequality_index,
    kitagawa_decompose,
    machado_mata,
    mediation_decompose,
    melly_decompose,
    mincer_wage_panel,
    oaxaca,
    rif_decomposition,
    rifreg,
    shapley_inequality,
    source_decompose,
    subgroup_decompose,
    yu_elwert_decompose,
    yun_nonlinear,
)

# Continuous Treatment DID
from .did import continuous_did
from .did.ddd_heterogeneous import ddd_heterogeneous
from .did.did_multiplegt_dyn import did_multiplegt_dyn
from .did.lp_did import lp_did
from .did.timevarying_covariates import did_timevarying_covariates

# Eager: ``dose_response`` collides (function + subpackage of same name).
from .dose_response import DoseResponse, VCNetResult, dose_response, scigan, vcnet

# High-dimensional fixed effects (pyfixest backend)
# These are thin wrappers; actual import of pyfixest is deferred to call time
# via fixest.wrapper._check_pyfixest, so top-level import never fails.
from .fixest import etable, feglm, feols, fepois

# Interactive Fixed Effects
# (already imported in round 2)
# Mixed Effects / Multilevel
# (lazy) multilevel: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Stochastic Frontier
# Eager: ``frontier`` collides (function + subpackage of same name).
from .frontier import (
    FrontierResult,
    MalmquistResult,
    MetafrontierResult,
    frontier,
    lcsf,
    malmquist,
    metafrontier,
    te_rank,
    te_summary,
    translog_design,
    xtfrontier,
    zisf,
)

# General GMM
from .gmm import gmm, xtabond

# Unified help entry point (aggregates registry + docstring + category + search)
from .help import HelpResult, help

# (lazy) bounds: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Eager: ``interference`` collides (function + subpackage of same name).
from .interference import (
    CrossClusterRCTResult,
    DNCGNNDiDResult,
    InwardOutwardResult,
    MatchedPairResult,
    NetworkExposureResult,
    NetworkHTEResult,
    PeerEffectsResult,
    SpilloverEstimator,
    StaggeredClusterRCTResult,
    cluster_cross_interference,
    cluster_matched_pair,
    cluster_staggered_rollout,
    dnc_gnn_did,
    interference,
    interference_available_designs,
    inward_outward_spillover,
    network_exposure,
    network_hte,
    peer_effects,
    spillover,
)
from .iv.continuous_late import ContinuousLATEResult, continuous_iv_late

# Modern IV reporting bundle (post-2022 standard) — top-level for ergonomics.
from .iv.iv_diag import IVDiagResult, iv_compare, iv_diag

# Expose Kernel IV / Continuous-LATE at top level for agent discoverability.
from .iv.kernel_iv import KernelIVResult, kernel_iv
from .matrix_completion import MCPanel, mc_panel
from .metalearners import (
    AutoCATEResult,
    CATEEvalResult,
    ClusterCATEResult,
    DRLearner,
    FunctionalCATEResult,
    RLearner,
    SLearner,
    TLearner,
    XLearner,
    auto_cate,
    auto_cate_tuned,
    blp_test,
    cate_by_group,
    cate_eval,
    cate_group_plot,
    cate_plot,
    cate_summary,
    cluster_cate,
    compare_metalearners,
    focal_cate,
    gate_test,
    metalearner,
    predict_cate,
)

# (lazy) dtr: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Eager: ``multi_treatment`` collides (function + subpackage of same name).
from .multi_treatment import MultiTreatment, multi_treatment

# Social network analysis (sp.network). Eager so the SNA surface is on the
# top-level namespace; the subpackage is reachable as ``sp.network``.
from .network import (
    CentralityResult,
    CommunityResult,
    ComponentsResult,
    DyadicRegressionResult,
    ERGMResult,
    Graph,
    NetworkSummaryResult,
    QAPResult,
    assortativity,
    betweenness_centrality,
    bonacich_power,
    centrality,
    closeness_centrality,
    clustering,
    community_detection,
    degree_centrality,
    dyadic_regression,
    eigenvector_centrality,
    ergm,
    florentine_families,
    hits,
    karate_club,
    katz_centrality,
    netlm,
    netlogit,
    network_components,
    network_graph,
    network_modularity,
    network_plot,
    network_summary,
    pagerank,
    reciprocity,
    transitivity,
)

# === NEW v0.6 Round 2 ===
# Interactive Fixed Effects
from .panel.interactive_fe import interactive_fe

# Panel Binary (Logit/Probit FE/RE)
from .panel.panel_binary import panel_logit, panel_probit

# Panel FGLS
from .panel.panel_fgls import panel_fgls

# Panel Unit Root Tests
from .panel.unit_root import PanelUnitRootResult, panel_unitroot
from .parity import ParityStatus, parity_matrix, parity_status, parity_summary
from .plots import (  # noqa: E402
    binscatter,
    counterfactual_data,
    counterfactual_plot,
    list_themes,
    set_theme,
    use_chinese,
)
from .policy_learning import (
    PolicyTree,
    PolicyTreeResult,
    direct_method,
    doubly_robust,
    ips,
    policy_tree,
    policy_value,
    snips,
)
from .power import (
    PowerResult,
    mde,
    power,
    power_case_control,
    power_cluster_rct,
    power_did,
    power_iv,
    power_logrank,
    power_ols,
    power_rct,
    power_rd,
    power_two_proportions,
)

# Distributional Treatment Effects
from .qte import (
    BeyondAverageResult,
    DistIVResult,
    DTEResult,
    HDPanelQTEResult,
    QTEResult,
    beyond_average_late,
    dist_iv,
    distributional_te,
    kan_dlate,
    qdid,
    qte,
    qte_hd_panel,
)
from .quasi import ancova, negd
from .question import (
    CausalQuestion,
    EstimationResult,
    IdentificationPlan,
    causal_question,
    load_preregister,
    preregister,
)

# 2D Boundary RD (Cattaneo, Titiunik, Yu 2025)
# Multi-cutoff / Geographic RD
from .rd import (
    RDMultiResult,
    boundary_rd,
    geographic_rd,
    multi_cutoff_rd,
    multi_score_rd,
    rd2d,
    rd2d_bw,
    rd2d_plot,
    rdmc,
    rdms,
)

# (lazy) mht: see _LAZY_SUBMODULES / _LAZY_ATTRS
from .registry import (
    STABILITY_TIERS,
    VALIDATION_STATUSES,
    FailureMode,
    agent_card,
    agent_cards,
    agent_schema,
    all_schemas,
    describe_function,
    function_schema,
    list_functions,
    search_functions,
)

# Advanced IV
from .regression.advanced_iv import jive, lasso_iv, liml
from .regression.count import nbreg, poisson, ppmlhdfe, xtnbreg

# Fractional Response & Beta Regression
from .regression.fracreg import betareg, fracreg
from .regression.glm import GLMEstimator, GLMRegression, glm

# bayes — lazy-loaded (PyMC pulls heavy deps); see _LAZY_ATTRS below.
from .regression.heckman import heckman
from .regression.iv_quantile import ivqreg
from .regression.logit_probit import cloglog, logit, probit
from .regression.mixed_logit import mixlogit

# === NEW MODULES (v0.6) ===
# GLM & Discrete Choice — ``glm``/``logit``/``probit``/``cloglog``/
# ``poisson``/``nbreg``/``xtnbreg``/``ppmlhdfe`` are already imported above in the
# core regression block; we only add what's new here.
from .regression.multinomial import clogit, mlogit, ologit, oprobit
from .regression.quantile import qreg, sqreg

# Sample Selection Models
from .regression.selection import biprobit, etregress

# SUR & 3SLS
from .regression.sur import SURResult, sureg, three_sls
from .regression.tobit import tobit

# === NEW v0.6 Round 3 ===
# Truncated Regression
from .regression.truncreg import truncreg

# Count Data
from .regression.zeroinflated import hurdle, zinb, zip_model

# Rigorous (data-driven) Lasso — faithful port of R's hdm package
from .rlasso import (  # noqa: E402
    RlassoClassifier,
    RlassologitClassifier,
    RlassoRegressor,
    rlasso,
    rlasso_effect,
    rlasso_effects,
    rlasso_iv,
    rlassologit,
    rlassologit_effect,
    rlassologit_effects,
)
from .selection import SelectionResult, lasso_select, stepwise

# === Smart Workflow Engine ===
from .smart import (
    AssumptionResult,
    ComparisonResult,
    DiagnosticFinding,
    IdentificationError,
    IdentificationReport,
    IntakeResult,
    PubReadyResult,
    RecommendationResult,
    SensitivityDashboard,
    assumption_audit,
    audit,
    bib_for,
    bibtex,
    brief,
    check_identification,
    compare_estimators,
    design_intake,
    detect_design,
    examples,
    list_replications,
    methods_appendix,
    preflight,
    pub_ready,
    recommend,
    replicate,
    sensitivity_dashboard,
    session,
)

# Cointegration
# Survival / Duration
# (lazy) survival: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Nonparametric
# (lazy) nonparametric: see _LAZY_SUBMODULES / _LAZY_ATTRS
# Time Series (for causal inference)
from .timeseries import (
    ARIMAResult,
    BVARResult,
    CointegrationResult,
    GARCHResult,
    ITSResult,
    LocalProjectionsResult,
    StructuralBreakResult,
    VARResult,
    arima,
    bvar,
    cusum_test,
    engle_granger,
    garch,
    granger_causality,
    irf,
    its,
    johansen,
    local_projections,
    structural_break,
    var,
)

# Eager: ``tmle`` collides (function + subpackage of same name).
from .tmle import (
    TMLE,
    HALClassifier,
    HALRegressor,
    LTMLEResult,
    LTMLESurvivalResult,
    SuperLearner,
    hal_tmle,
    ltmle,
    ltmle_survival,
    super_learner,
    tmle,
)
from .utils import (
    describe,
    dgp_bartik,
    dgp_bunching,
    dgp_cluster_rct,
    dgp_did,
    dgp_iv,
    dgp_observational,
    dgp_panel,
    dgp_rct,
    dgp_rd,
    dgp_rd_2d,
    dgp_rd_hte,
    dgp_rd_kink,
    dgp_rd_multi,
    dgp_rdit,
    dgp_synth,
    get_label,
    get_labels,
    label_var,
    label_vars,
    outlier_indicator,
    pwcorr,
    rank,
    read_data,
    rowcount,
    rowmax,
    rowmean,
    rowmin,
    rowsd,
    rowtotal,
    scalar_iv_projection,
    winsor,
)
from .validation import (
    ReproductionResult,
    ReproductionStep,
    ValidationReport,
    coverage_matrix,
    parity_gap_report,
    reproduce_jss_tables,
    validation_report,
)

# === End-to-end workflow orchestrator ===
# After ``import statspai.causal`` (the deprecated forest-shim) Python rebinds
# ``sp.causal`` to that submodule, shadowing this function.  The shim works
# around it by making its module object callable (see ``causal/__init__.py``),
# so ``sp.causal(df, y=, treatment=, ...)`` keeps dispatching to this
# workflow function in either order.
from .workflow import (  # noqa: F401 — ``causal`` kept for the shadowing dance
    CausalWorkflow,
    PaperDraft,
    causal,
    paper,
)

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


# === Unified sensitivity dashboard ===
# (lazy) robustness_b: see _LAZY_SUBMODULES / _LAZY_ATTRS


# Structural Estimation (BLP, production functions)
# (lazy) structural_a: see _LAZY_SUBMODULES / _LAZY_ATTRS
# (lazy) structural_b: see _LAZY_SUBMODULES / _LAZY_ATTRS


# verify / verify_recommendation / verify_benchmark are loaded lazily via
# __getattr__ at the bottom of this file so that `import statspai` doesn't
# drag in the resampling-stability machinery unless the caller actually
# asks for it. Preserves the "zero overhead when verify=False" guarantee
# in recommend().


__all__ = [
    # Core
    "EconometricResults",
    "CausalResult",
    "EffectSummary",
    "effect_summary",
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
    "bjs",
    "borusyak_jaravel_spiess",
    "stacked_did",
    "gardner_did",
    "did_2stage",
    "cic",
    "pretrends_test",
    "pretrends_power",
    "sensitivity_rr",
    "SensitivityResult",
    "pretrends_summary",
    "parallel_trends_robustness",
    "ParallelTrendsRobustnessResult",
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
    "balance_diagnostics",
    "BalanceDiagnosticsResult",
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
    # v1.13 DML-OVB sensitivity + diagnostics (Chernozhukov-Cinelli-Newey 2022)
    "dml_sensitivity",
    "DMLSensitivityResult",
    "dml_diagnostics",
    "DMLDiagnostics",
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
    "average_treatment_effect",
    "forest_diagnostics",
    # HDFE primitives
    "Absorber",
    "SlopeSpec",
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
    "psmatch2",
    "PSMatch2Result",
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
    "coefplot_tikz",
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
    # Replication pack (audited archive)
    "ReplicationPack",
    "replication_pack",
    # great_tables adapter (manuscript/reporting tables)
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
    "ancova",
    "binscatter",
    "counterfactual_data",
    "counterfactual_plot",
    "geolift",
    "negd",
    "set_theme",
    "list_themes",
    "use_chinese",
    "interactive",
    "get_code",
    # Stata / R migration on-ramps
    "from_stata",
    "from_r",
    "translation_coverage",
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
    "xtnbreg",
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
    "postestimation_contract",
    "postestimation_report",
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
    "evalue_rd",
    "bias_factor",
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
    # Meta-analysis (evidence synthesis)
    "meta_analysis",
    "MetaAnalysisResult",
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
    "bayes_its",
    "bayes_synth",
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
    "neural_effects_frame",
    "neural_summary_frame",
    "neural_training_frame",
    "neural_causal_to_markdown",
    "neural_causal_to_html",
    "neural_causal_to_excel",
    "neural_causal_plot",
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
    "PolicyTreeResult",
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
    "power_two_proportions",
    "power_logrank",
    "power_case_control",
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
    "agent_schema",
    "search_functions",
    "all_schemas",
    "agent_card",
    "agent_cards",
    "FailureMode",
    "STABILITY_TIERS",
    "VALIDATION_STATUSES",
    "render_agent_block",
    "render_agent_blocks",
    "ReproductionResult",
    "ReproductionStep",
    "ValidationReport",
    "coverage_matrix",
    "parity_gap_report",
    "reproduce_jss_tables",
    "validation_report",
    "ParityStatus",
    "parity_matrix",
    "parity_status",
    "parity_summary",
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
    "mlogit",
    "ologit",
    "oprobit",
    "clogit",
    "mixlogit",
    "ivqreg",
    # Count Data (poisson/nbreg/xtnbreg/ppmlhdfe already in regression block above)
    "zip_model",
    "zinb",
    "hurdle",
    # Advanced IV
    "liml",
    "jive",
    "lasso_iv",
    "rlasso",
    "rlasso_effect",
    "rlasso_effects",
    "rlasso_iv",
    "rlassologit",
    "rlassologit_effect",
    "rlassologit_effects",
    "RlassoRegressor",
    "RlassoClassifier",
    "RlassologitClassifier",
    # High-dimensional FE (pyfixest backend, optional)
    "feols",
    "fepois",
    "feglm",
    "etable",
    # Survival
    "cox",
    "kaplan_meier",
    "survreg",
    "CoxResult",
    "KMResult",
    "logrank_test",
    "cuminc",
    "finegray",
    "CumIncResult",
    "FineGrayResult",
    # Nonparametric
    "lpoly",
    "LPolyResult",
    "kdensity",
    "KDensityResult",
    # Time Series
    "var",
    "VARResult",
    "granger_causality",
    "irf",
    "structural_break",
    "StructuralBreakResult",
    "cusum_test",
    # Experimental Design
    "randomize",
    "RandomizationResult",
    "balance_check",
    "BalanceResult",
    "attrition_test",
    "attrition_bounds",
    "AttritionResult",
    "optimal_design",
    "OptimalDesignResult",
    # Missing Data
    "mice",
    "MICEResult",
    "mi_estimate",
    # Mendelian Randomization
    "mendelian_randomization",
    "MRResult",
    "mr_egger",
    "mr_ivw",
    "mr_median",
    # Multi-Cutoff / Geographic RD
    "rdmc",
    "rdms",
    "RDMultiResult",
    "multi_cutoff_rd",
    "geographic_rd",
    "boundary_rd",
    "multi_score_rd",
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
    "panel_unitroot",
    "PanelUnitRootResult",
    "engle_granger",
    "johansen",
    "CointegrationResult",
    "fracreg",
    "betareg",
    "biprobit",
    "etregress",
    "distributional_te",
    "DTEResult",
    # Structural Estimation
    "blp",
    "BLPResult",
    # Production functions (proxy-variable estimators)
    "prod_fn",
    "olley_pakes",
    "opreg",
    "levinsohn_petrin",
    "levpet",
    "ackerberg_caves_frazer",
    "acf",
    "wooldridge_prod",
    "markup",
    "ProductionResult",
    # === Smart Workflow Engine ===
    "recommend",
    "RecommendationResult",
    "design_intake",
    "IntakeResult",
    "check_identification",
    "IdentificationReport",
    "DiagnosticFinding",
    "IdentificationError",
    "compare_estimators",
    "ComparisonResult",
    "cross_validate",
    "CrossValidationResult",
    "from_worldbank",
    "from_fred",
    "from_sdmx",
    "assumption_audit",
    "AssumptionResult",
    "audit",
    "bib_for",
    "methods_appendix",
    "bibtex",
    "brief",
    "detect_design",
    "examples",
    "preflight",
    "session",
    "sensitivity_dashboard",
    "SensitivityDashboard",
    "pub_ready",
    "PubReadyResult",
    "replicate",
    "list_replications",
    "verify",
    "verify_recommendation",
    "verify_benchmark",
    "recommend_benchmark",
    # === v0.6 Round 3 ===
    "truncreg",
    "sureg",
    "SURResult",
    "three_sls",
    "panel_logit",
    "panel_probit",
    "panel_fgls",
    "mixed",
    "MixedResult",
    "meglm",
    "melogit",
    "mepoisson",
    "menbreg",
    "megamma",
    "meologit",
    "MEGLMResult",
    "icc",
    "lrtest",
    "frontier",
    "xtfrontier",
    "FrontierResult",
    "metafrontier",
    "MetafrontierResult",
    "malmquist",
    "MalmquistResult",
    "translog_design",
    "zisf",
    "lcsf",
    "te_summary",
    "te_rank",
    "gmm",
    # ---- v0.9.3 __all__ completeness pass ----
    # Items below were imported at the top of the file but previously
    # missing from __all__, breaking `from statspai import *` and some
    # IDE autocompleters. Grouped by subsystem for readability.
    # Causal impact / mediation
    "impactplot",
    "mediate_sensitivity",
    # Synth suite
    "synthplot",
    "multi_outcome_synth",
    "scpi",
    "scest",
    "scdata",
    "discos",
    "discos_test",
    "discos_plot",
    "qqsynth",
    "stochastic_dominance",
    "synth_loo",
    "synth_time_placebo",
    "synth_donor_sensitivity",
    "synth_rmspe_filter",
    "synth_sensitivity",
    "synth_sensitivity_plot",
    "synth_power",
    "synth_mde",
    "synth_power_plot",
    "synth_compare",
    "synth_recommend",
    "SynthComparison",
    "synth_report",
    "synth_report_to_file",
    "synth_to_latex",
    "synth_to_markdown",
    "synth_to_excel",
    "german_reunification",
    "basque_terrorism",
    "california_tobacco",
    # Spatial
    "W",
    "moran",
    "moran_local",
    "moran_plot",
    "moran_residuals",
    "geary",
    "getis_ord_g",
    "getis_ord_local",
    "join_counts",
    "lisa_cluster_map",
    "lm_tests",
    "impacts",
    "line_length_in_polygon",
    "share_within_buffer",
    "distance_to_feature",
    "queen_weights",
    "rook_weights",
    "knn_weights",
    "distance_band",
    "kernel_weights",
    "block_weights",
    "gwr",
    "mgwr",
    "gwr_bandwidth",
    "sac",
    "slx",
    "sar_gmm",
    "sarar_gmm",
    "sem_gmm",
    "spatial_panel",
    # RD
    "rd2d",
    "rd2d_bw",
    "rd2d_plot",
    "rdpower",
    "rdsampsi",
    "dgp_rd_2d",
    "dgp_rd_hte",
    "dgp_rd_kink",
    "dgp_rd_multi",
    "dgp_rdit",
    # Decomposition
    "decompose",
    "dfl_decompose",
    "machado_mata",
    "melly_decompose",
    "rifreg",
    "rif_decomposition",
    "ffl_decompose",
    "fairlie",
    "yun_nonlinear",
    "cfm_decompose",
    "bauer_sinning",
    "das_gupta",
    "gap_closing",
    "kitagawa_decompose",
    "source_decompose",
    "subgroup_decompose",
    "disparity_decompose",
    "disparity_panel",
    "mediation_decompose",
    "yu_elwert_decompose",
    "YuElwertResult",
    "inequality_index",
    "shapley_inequality",
    # Panel / DID extras
    "aggte",
    "ggdid",
    "bjs_pretrend_joint",
    "cs_report",
    "CSReport",
    "local_projections",
    "LocalProjectionsResult",
    # Matching / survey / survival
    "cardinality_match",
    "CardinalityMatchResult",
    "optimal_match",
    "OptimalMatchResult",
    "linear_calibration",
    "rake",
    "aft",
    "cox_frailty",
    # Causal discovery (notears/NOTEARS/pc_algorithm/PCAlgorithm already listed above)
    "lingam",
    "LiNGAMResult",
    "ges",
    "GESResult",
    "fci",
    "FCIResult",
    # Time series
    "arima",
    "ARIMAResult",
    "bvar",
    "BVARResult",
    "garch",
    "GARCHResult",
    # Datasets
    "cps_wage",
    "mincer_wage_panel",
    "chilean_households",
    # IV frontier (v1.1)
    "kernel_iv",
    "KernelIVResult",
    "continuous_iv_late",
    "ContinuousLATEResult",
    "iv_diag",
    "iv_compare",
    "IVDiagResult",
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
    "ipcw",
    "IPCWResult",
    "icp",
    "nonlinear_icp",
    "ICPResult",
    "identify",
    "IdentificationResult",
    "do_rule1",
    "do_rule2",
    "do_rule3",
    "do_calculus_apply",
    "RuleCheck",
    "swig",
    "SWIGGraph",
    "SCM",
    "cevae",
    "CEVAE",
    "CEVAEResult",
    "TargetTrialProtocol",
    "TargetTrialResult",
    "CloneCensorWeightResult",
    "target_trial_protocol",
    "target_trial_emulate",
    "target_trial_report",
    "clone_censor_weight",
    "immortal_time_check",
    "tte",
    "TransportWeightResult",
    "TransportIdentificationResult",
    "transport_generalize",
    "transport_weights_fn",
    "identify_transport",
    "OPEResult",
    "gformula_ice_fn",
    "ICEResult",
    "gformula_mc",
    "MCGFormulaResult",
    # v0.9.17 additions (epi primitives)
    "epi",
    "odds_ratio",
    "relative_risk",
    "risk_difference",
    "attributable_risk",
    "incidence_rate_ratio",
    "number_needed_to_treat",
    "prevalence_ratio",
    "mantel_haenszel",
    "breslow_day_test",
    "direct_standardize",
    "indirect_standardize",
    "bradford_hill",
    # v0.9.17 additions (epi clinical diagnostics)
    "diagnostic_test",
    "sensitivity_specificity",
    "roc_curve",
    "auc",
    "cohen_kappa",
    "DiagnosticTestResult",
    "ROCResult",
    "KappaResult",
    # v0.9.17 additions (MR full suite)
    "mr",
    "mendelian",
    "mr_heterogeneity",
    "mr_pleiotropy_egger",
    "mr_leave_one_out",
    "mr_steiger",
    "mr_presso",
    "mr_radial",
    "HeterogeneityResult",
    "PleiotropyResult",
    "LeaveOneOutResult",
    "SteigerResult",
    "MRPressoResult",
    "RadialResult",
    # v0.9.17 additions (MR deepening)
    "mr_mode",
    "mr_f_statistic",
    "mr_funnel_plot",
    "mr_scatter_plot",
    "ModeBasedResult",
    "FStatisticResult",
    # v0.9.17 additions (longitudinal unified)
    "longitudinal",
    "longitudinal_analyze",
    "longitudinal_contrast",
    "regime",
    "always_treat",
    "never_treat",
    "LongitudinalResult",
    "Regime",
    # v0.9.17 additions (causal-question DSL + pre-registration)
    "question",
    "causal_question",
    "CausalQuestion",
    "IdentificationPlan",
    "EstimationResult",
    "preregister",
    "load_preregister",
    "paper",
    # v0.9.17 additions (unified sensitivity; SensitivityDashboard already exported)
    "unified_sensitivity",
    # v0.9.17 additions (DAG UX)
    "dag_recommend_estimator",
    # v1.0 — bridging theorems
    "bridge",
    "BridgeResult",
    # v1.0 — DiD frontiers (scaffolded)
    "did_bcf",
    "cohort_anchored_event_study",
    "design_robust_event_study",
    "did_misclassified",
    # v1.0 — conformal frontiers
    "conformal_debiased_ml",
    "DebiasedConformalResult",
    "conformal_density_ite",
    "ConformalDensityResult",
    "conformal_fair_ite",
    "FairConformalResult",
    "conformal_ite_multidp",
    "MultiDPConformalResult",
    # v1.0 — proximal frontiers
    "fortified_pci",
    "bidirectional_pci",
    "pci_mtp",
    "select_pci_proxies",
    "ProxyScoreResult",
    # v1.0 — QTE / RD frontiers
    "beyond_average_late",
    "BeyondAverageResult",
    "qte_hd_panel",
    "HDPanelQTEResult",
    "rd_distribution",
    "DistRDResult",
    "rd_interference",
    "RDInterferenceResult",
    "rd_multi_score",
    "MultiScoreRDResult",
    # v1.0 — time-series causal discovery
    "pcmci",
    "PCMCIResult",
    "lpcmci",
    "LPCMCIResult",
    "dynotears",
    "DYNOTEARSResult",
    "partial_corr_pvalue",
    # v1.0 — LTMLE survival + BCF longitudinal
    "ltmle_survival",
    "LTMLESurvivalResult",
    # v1.0 — sequential SDID
    "sequential_sdid",
    "SequentialSDIDResult",
    "synth_survival",
    "SyntheticSurvivalResult",
    # v1.0 — ML bounds
    "ml_bounds",
    # v1.0 — TARGET Statement 2025
    "target_trial_checklist",
    # v1.0 — frontier sensitivity
    "copula_sensitivity",
    "survival_sensitivity",
    "calibrate_confounding_strength",
    "FrontierSensitivityResult",
    # === v0.10 / v1.0 frontier additions (most are already exported above) ===
    # Distributional / panel QTE — only the new dist_iv/kan_dlate/DistIVResult here
    "dist_iv",
    "kan_dlate",
    "DistIVResult",
    # RDD frontier — only the new rd_bayes_hte / rd_distributional_design here
    "rd_bayes_hte",
    "BayesRDHTEResult",
    "rd_distributional_design",
    "DDDResult",
    # v1.15 RDD polish (recent literature)
    "rd_flex",
    "rd_bias_aware_fuzzy",
    "rd_discrete",
    "rd_dashboard",
    "rd_compare",
    "rd_robustness_table",
    # Causal × LLM
    "llm_dag_propose",
    "LLMDAGProposal",
    "llm_unobserved_confounders",
    "UnobservedConfounderProposal",
    "llm_sensitivity_priors",
    "SensitivityPriorProposal",
    "llm_dag_constrained",
    "llm_dag_validate",
    "LLMConstrainedDAGResult",
    "DAGValidationResult",
    # Causal × Text (P1-B v1.6 experimental)
    "text_treatment_effect",
    "TextTreatmentResult",
    "llm_annotator_correct",
    "LLMAnnotatorResult",
    # Causal RL
    "causal_dqn",
    "CausalDQNResult",
    "causal_rl_benchmark",
    "BanditBenchmarkResult",
    "offline_safe_policy",
    "OfflineSafeResult",
    "structural_mdp",
    # Cluster RCT × interference
    "cluster_matched_pair",
    "MatchedPairResult",
    "cluster_cross_interference",
    "CrossClusterRCTResult",
    "cluster_staggered_rollout",
    "StaggeredClusterRCTResult",
    "dnc_gnn_did",
    "DNCGNNDiDResult",
    # Meta-learner frontier
    "focal_cate",
    "FunctionalCATEResult",
    "cluster_cate",
    "ClusterCATEResult",
    # v1.13 backbone-agnostic CATE evaluation (RATE / AUTOC / Qini)
    "cate_eval",
    "CATEEvalResult",
    # Bunching frontier
    "general_bunching",
    "GeneralBunchingResult",
    "kink_unified",
    "KinkUnifiedResult",
    # v1.6 MR frontier: sample-overlap / clusters / profile-LL / cML / RAPS
    "mr_lap",
    "mr_clust",
    "grapple",
    "mr_cml",
    "mr_raps",
    "MRLapResult",
    "MRClustResult",
    "GrappleResult",
    "MRcMLResult",
    "MRRapsResult",
    # v1.5 unified family dispatchers (mr already exported above as the dispatcher)
    "mr_available_methods",
    "conformal",
    "conformal_available_kinds",
    "interference",
    "interference_available_designs",
    # Social network analysis (sp.network)
    "network_graph",
    "Graph",
    "network_summary",
    "NetworkSummaryResult",
    "transitivity",
    "clustering",
    "reciprocity",
    "assortativity",
    "network_components",
    "ComponentsResult",
    "centrality",
    "CentralityResult",
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "katz_centrality",
    "pagerank",
    "bonacich_power",
    "hits",
    "community_detection",
    "CommunityResult",
    "network_modularity",
    "netlm",
    "netlogit",
    "QAPResult",
    "dyadic_regression",
    "DyadicRegressionResult",
    "ergm",
    "ERGMResult",
    "karate_club",
    "florentine_families",
    "network_plot",
    # v1.5 registry coverage fixes for previously-exposed-but-unregistered
    # single-family functions (now reachable via sp.describe_function too)
    "network_exposure",
    "NetworkExposureResult",
    "peer_effects",
    "PeerEffectsResult",
    "weighted_conformal_prediction",
    "conformal_counterfactual",
    "ConformalCounterfactualResult",
    "conformal_ite_interval",
    "ConformalITEResult",
    # Registry/API surface consistency guards
    "mr_mediation",
    "orthogonal_to_bias",
    "surrogate_index",
    "synthesise_evidence",
    # ---- __all__ / registry drift repair (agent-native discoverability) ----
    # These estimators + their result objects were eagerly imported into the
    # ``statspai`` namespace (so ``sp.xxx`` already worked) but were absent
    # from ``__all__``.  The auto-registration pass in ``registry.py`` walks
    # ``__all__``, so every name below was invisible to ``sp.list_functions``,
    # ``sp.describe_function`` and ``sp.function_schema`` — a direct violation
    # of the agent-native design contract ("help tools must resolve for every
    # public symbol").  All run cleanly and build valid schemas; listing them
    # here closes the drift and makes ``from statspai import *`` complete.
    # BCF extensions
    "bcf_factor_exposure",
    "bcf_longitudinal",
    "bcf_ordinal",
    "BCFFactorExposureResult",
    "BCFLongResult",
    "BCFOrdinalResult",
    # Proximal / negative-control identification
    "double_negative_control",
    "negative_control_exposure",
    "negative_control_outcome",
    "proximal_regression",
    "NegativeControlResult",
    "ProximalRegResult",
    # Off-policy evaluation estimators
    "direct_method",
    "doubly_robust",
    "ips",
    "snips",
    # Mediation
    "four_way_decomposition",
    "FourWayResult",
    # Interrupted time series
    "its",
    "ITSResult",
    # Longitudinal TMLE
    "ltmle",
    "LTMLEResult",
    # DiD-family extras
    "harvest_did",
    "overlap_weighted_did",
    "HarvestDIDResult",
    # Spillover / network heterogeneity
    "inward_outward_spillover",
    "network_hte",
    "InwardOutwardResult",
    "NetworkHTEResult",
    # Shift-share (Bartik) political-economy designs
    "shift_share_political",
    "shift_share_political_panel",
    "ShiftSharePoliticalResult",
    "ShiftSharePoliticalPanelResult",
    # Dose-response (neural)
    "scigan",
    "vcnet",
    "VCNetResult",
    # DAG / LLM-causal helpers
    "llm_dag",
    "llm_causal_assess",
    "LLMDAGResult",
    "LLMCausalAssessResult",
    # Workflow / design helpers
    # NB: ``causal`` is intentionally *not* listed — the bare name collides
    # with the ``causal`` category and the ``statspai.causal`` submodule, so
    # registering it as a function would shadow ``sp.help("causal")``'s
    # category listing.  The workflow stays reachable as ``sp.causal(...)``.
    "dl_propensity_score",
    "synth_experimental_design",
    "pairwise_causal_benchmark",
    "CausalWorkflow",
    "SynthExperimentalDesignResult",
    "PairwiseBenchmarkResult",
    "PaperDraft",
    # Rosenbaum sensitivity bounds
    "rosenbaum_bounds",
    "rosenbaum_gamma",
    "RosenbaumResult",
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


# In-place dedup: keep ``__all__`` bound to the literal list above so static
# analysers (pyflakes/flake8 F401) can still see the export set. Rebinding it to
# a function-call result made pyflakes treat ``__all__`` as dynamic and wrongly
# flag ~747 re-exported names as unused imports.
__all__[:] = _dedupe_public_exports(__all__)
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
from ._article_aliases import (  # noqa: E402,F811
    causal_discovery,
    dml,
    matrix_completion,
    mediation,
    policy_tree,
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
    "bayes": "bayes",
    "neural_causal": "neural_causal",
    "causal_text": "causal_text",
    "causal_rl": "causal_rl",
    "fairness": "fairness",
    "assimilation": "assimilation",
    "surrogate": "surrogate",
    "bounds": "bounds",
    "dtr": "dtr",
    "spatial": "spatial",
    "forest": "forest",
    "conformal_causal": "conformal_causal",
    "ope": "ope",
    "censoring": "censoring",
    "epi": "epi",
    "longitudinal": "longitudinal",
    "gformula": "gformula",
    "target_trial": "target_trial",
    "tte": "target_trial",  # short alias
    "transport": "transport",
    "mendelian": "mendelian",
    "experimental": "experimental",
    "imputation": "imputation",
    "survey": "survey",
    "survival": "survival",
    "nonparametric": "nonparametric",
    "multilevel": "multilevel",
    "structural": "structural",
    "causal_llm": "causal_llm",
    "mht": "mht",
    "robustness": "robustness",
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

# Stata / R migration on-ramps (and their coverage matrix). Lazily exposed so a
# Python user can call ``sp.from_stata(...)`` / ``sp.from_r(...)`` directly,
# mirroring the MCP translator tools. Bound to the leaf modules (not the
# ``agent._translation`` package facade) so resolution is robust to changes in
# the package ``__init__`` re-exports.
_register_lazy("agent._translation._stata", "from_stata")
_register_lazy("agent._translation._r", "from_r")
_register_lazy("agent._translation._coverage", "translation_coverage")
_register_lazy(
    "plots.interactive",
    "interactive",
    "get_code",
)
_register_lazy(
    "geolift",
    "geolift",
)
_register_lazy(
    "bayes",
    "bayes_did",
    "bayes_rd",
    "bayes_its",
    "bayes_synth",
    "bayes_iv",
    "bayes_fuzzy_rd",
    "bayes_hte_iv",
    "bayes_mte",
    "bayes_dml",
    "BayesianDMLResult",
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
)
_register_lazy(
    "neural_causal.models",
    "tarnet",
    "cfrnet",
    "dragonnet",
    "TARNet",
    "CFRNet",
    "DragonNet",
)
_register_lazy(
    "neural_causal.gnn_causal",
    "gnn_causal",
    "GNNCausalResult",
)
_register_lazy(
    "neural_causal.exports",
    "neural_effects_frame",
    "neural_summary_frame",
    "neural_training_frame",
    "neural_causal_to_markdown",
    "neural_causal_to_html",
    "neural_causal_to_excel",
)
_register_lazy(
    "neural_causal.plots",
    "neural_causal_plot",
)
_register_lazy(
    "neural_causal.cevae",
    "cevae",
    "CEVAE",
    "CEVAEResult",
)
_register_lazy(
    "causal_text",
    "text_treatment_effect",
    "TextTreatmentResult",
    "llm_annotator_correct",
    "LLMAnnotatorResult",
)
_register_lazy(
    "causal_rl",
    "causal_dqn",
    "CausalDQNResult",
    "causal_rl_benchmark",
    "BanditBenchmarkResult",
    "offline_safe_policy",
    "OfflineSafeResult",
    "causal_bandit",
    "counterfactual_policy_optimization",
    "structural_mdp",
    "CausalBanditResult",
    "CFPolicyResult",
    "StructuralMDPResult",
)
_register_lazy(
    "fairness",
    "counterfactual_fairness",
    "orthogonal_to_bias",
    "demographic_parity",
    "equalized_odds",
    "fairness_audit",
    "FairnessResult",
    "FairnessAudit",
    "evidence_without_injustice",
    "EvidenceWithoutInjusticeResult",
)
_register_lazy(
    "assimilation",
    "assimilative_causal",
    "causal_kalman",
    "AssimilationResult",
)
_register_lazy(
    "surrogate",
    "surrogate_index",
    "long_term_from_short",
    "proximal_surrogate_index",
    "SurrogateResult",
)
_register_lazy(
    "bounds",
    "lee_bounds",
    "manski_bounds",
    "BoundsResult",
    "horowitz_manski",
    "iv_bounds",
    "oster_delta",
    "selection_bounds",
    "breakdown_frontier",
    "balke_pearl",
    "BalkePearlResult",
    "ml_bounds",
    "MLBoundsResult",
)
_register_lazy(
    "dtr",
    "g_estimation",
    "GEstimation",
    "q_learning",
    "QLearningResult",
    "a_learning",
    "ALearningResult",
    "snmm",
    "SNMMResult",
)
_register_lazy(
    "spatial",
    "sar",
    "sem",
    "sdm",
    "slx",
    "sac",
    "SpatialModel",
    "sar_gmm",
    "sem_gmm",
    "sarar_gmm",
    "gwr",
    "mgwr",
    "gwr_bandwidth",
    "spatial_panel",
    "W",
    "queen_weights",
    "rook_weights",
    "knn_weights",
    "distance_band",
    "kernel_weights",
    "block_weights",
    "moran",
    "moran_local",
    "geary",
    "getis_ord_g",
    "getis_ord_local",
    "join_counts",
    "moran_plot",
    "lisa_cluster_map",
    "lm_tests",
    "moran_residuals",
    "impacts",
    "spatial_did",
    "SpatialDiDResult",
    "spatial_iv",
    "SpatialIVResult",
    "line_length_in_polygon",
    "share_within_buffer",
    "distance_to_feature",
)
_register_lazy(
    "forest.causal_forest",
    "CausalForest",
    "causal_forest",
)
_register_lazy(
    "forest.forest_inference",
    "calibration_test",
    ("test_calibration", "calibration_test"),
    "rate",
    "honest_variance",
    "average_treatment_effect",
    "forest_diagnostics",
)
_register_lazy(
    "forest.multi_arm_forest",
    "multi_arm_forest",
    "MultiArmForestResult",
)
_register_lazy(
    "forest.iv_forest",
    "iv_forest",
    "IVForestResult",
)
_register_lazy(
    "conformal_causal",
    "conformal_cate",
    "ConformalCATE",
    "weighted_conformal_prediction",
    "conformal_counterfactual",
    "ConformalCounterfactualResult",
    "conformal_ite_interval",
    "ConformalITEResult",
    "conformal_density_ite",
    "ConformalDensityResult",
    "conformal_ite_multidp",
    "MultiDPConformalResult",
    "conformal_debiased_ml",
    "DebiasedConformalResult",
    "conformal_fair_ite",
    "FairConformalResult",
    "conformal_continuous",
    "conformal_interference",
    "ContinuousConformalResult",
    "InterferenceConformalResult",
    "conformal",
    "conformal_available_kinds",
)
_register_lazy(
    "ope",
    "OPEResult",
    "sharp_ope_unobserved",
    "causal_policy_forest",
    "SharpOPEResult",
    "CausalPolicyForestResult",
)
_register_lazy(
    "censoring",
    "ipcw",
    "IPCWResult",
)
_register_lazy(
    "epi",
    "odds_ratio",
    "relative_risk",
    "risk_difference",
    "attributable_risk",
    "incidence_rate_ratio",
    "number_needed_to_treat",
    "prevalence_ratio",
    "mantel_haenszel",
    "breslow_day_test",
    "direct_standardize",
    "indirect_standardize",
    "bradford_hill",
    "diagnostic_test",
    "sensitivity_specificity",
    "roc_curve",
    "auc",
    "cohen_kappa",
    "DiagnosticTestResult",
    "ROCResult",
    "KappaResult",
)
_register_lazy(
    "longitudinal",
    ("longitudinal_analyze", "analyze"),
    ("longitudinal_contrast", "contrast"),
    "regime",
    "always_treat",
    "never_treat",
    "LongitudinalResult",
    "Regime",
)
_register_lazy(
    "gformula",
    ("gformula_ice_fn", "ice"),
    "ICEResult",
    "gformula_mc",
    "MCGFormulaResult",
)
_register_lazy(
    "target_trial",
    ("target_trial_protocol", "protocol"),
    ("target_trial_emulate", "emulate"),
    ("target_trial_report", "to_paper"),
    ("target_trial_checklist", "target_checklist"),
    "TARGET_ITEMS",
    "clone_censor_weight",
    "immortal_time_check",
    "TargetTrialProtocol",
    "TargetTrialResult",
    "CloneCensorWeightResult",
)
_register_lazy(
    "transport",
    ("transport_weights_fn", "weights"),
    ("transport_generalize", "generalize"),
    "TransportWeightResult",
    "identify_transport",
    "TransportIdentificationResult",
    "synthesise_evidence",
    "heterogeneity_of_effect",
    "rwd_rct_concordance",
    "EvidenceSynthesisResult",
    "HeterogeneityResult",
    "ConcordanceResult",
)
_register_lazy(
    "mendelian",
    "mendelian_randomization",
    "MRResult",
    "mr_egger",
    "mr_ivw",
    "mr_median",
    "mr_heterogeneity",
    "mr_pleiotropy_egger",
    "mr_leave_one_out",
    "mr_steiger",
    "mr_presso",
    "mr_radial",
    "HeterogeneityResult",
    "PleiotropyResult",
    "LeaveOneOutResult",
    "SteigerResult",
    "MRPressoResult",
    "RadialResult",
    "mr_mode",
    "mr_f_statistic",
    "mr_funnel_plot",
    "mr_scatter_plot",
    "ModeBasedResult",
    "FStatisticResult",
    "mr_multivariable",
    "mr_mediation",
    "mr_bma",
    "MVMRResult",
    "MediationMRResult",
    "MRBMAResult",
    "mr_lap",
    "mr_clust",
    "grapple",
    "mr_cml",
    "mr_raps",
    "MRLapResult",
    "MRClustResult",
    "GrappleResult",
    "MRcMLResult",
    "MRRapsResult",
    "mr",
    "mr_available_methods",
)
_register_lazy(
    "experimental",
    "randomize",
    "RandomizationResult",
    "balance_check",
    "BalanceResult",
    "attrition_test",
    "attrition_bounds",
    "AttritionResult",
    "optimal_design",
    "OptimalDesignResult",
)
_register_lazy(
    "imputation",
    "mice",
    "MICEResult",
    "mi_estimate",
)
_register_lazy(
    "survey",
    "svydesign",
    "SurveyDesign",
    "svymean",
    "svytotal",
    "svyglm",
    "rake",
    "linear_calibration",
)
_register_lazy(
    "survival",
    "cox",
    "kaplan_meier",
    "survreg",
    "CoxResult",
    "KMResult",
    "logrank_test",
    "cox_frailty",
    "aft",
    "causal_survival_forest",
    "causal_survival",
    "CausalSurvivalForestResult",
    "cuminc",
    "finegray",
    "CumIncResult",
    "FineGrayResult",
)
_register_lazy(
    "nonparametric",
    "lpoly",
    "LPolyResult",
    "kdensity",
    "KDensityResult",
)
_register_lazy(
    "multilevel",
    "mixed",
    "MixedResult",
    "meglm",
    "melogit",
    "mepoisson",
    "menbreg",
    "megamma",
    "meologit",
    "MEGLMResult",
    "icc",
    "lrtest",
)
_register_lazy(
    "structural",
    "blp",
    "BLPResult",
    "prod_fn",
    "olley_pakes",
    "opreg",
    "levinsohn_petrin",
    "levpet",
    "ackerberg_caves_frazer",
    "acf",
    "wooldridge_prod",
    "markup",
    "ProductionResult",
)
_register_lazy(
    "causal_llm",
    "llm_dag_propose",
    "LLMDAGProposal",
    "llm_unobserved_confounders",
    "UnobservedConfounderProposal",
    "llm_sensitivity_priors",
    "SensitivityPriorProposal",
    "causal_mas",
    "CausalMASResult",
    "llm_dag_constrained",
    "llm_dag_validate",
    "LLMConstrainedDAGResult",
    "DAGValidationResult",
)
_register_lazy(
    "mht",
    "romano_wolf",
    "RomanoWolfResult",
    "adjust_pvalues",
    "bonferroni",
    "holm",
    "benjamini_hochberg",
)
_register_lazy(
    "robustness",
    "spec_curve",
    "SpecCurveResult",
    "robustness_report",
    "RobustnessResult",
    "subgroup_analysis",
    "SubgroupResult",
    "copula_sensitivity",
    "survival_sensitivity",
    "calibrate_confounding_strength",
    "FrontierSensitivityResult",
    "unified_sensitivity",
    "SensitivityDashboard",
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
    if name == "recommend_benchmark":
        from .smart.recommend_benchmark import recommend_benchmark as _rb

        globals()["recommend_benchmark"] = _rb
        return _rb
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
