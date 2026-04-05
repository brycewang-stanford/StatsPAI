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

__version__ = "0.4.0"
__author__ = "Bryce Wang"
__email__ = "bryce@copaper.ai"

from .core.results import EconometricResults, CausalResult
from .regression.ols import regress
from .regression.iv import iv, ivreg, IVRegression
from .causal.causal_forest import CausalForest, causal_forest
from .did import (
    did, did_2x2, ddd, callaway_santanna, sun_abraham,
    bacon_decomposition, honest_did, breakdown_m, event_study,
    did_analysis, DIDAnalysis,
    parallel_trends_plot, bacon_plot, group_time_plot, did_plot,
    enhanced_event_study_plot, treatment_rollout_plot,
    sensitivity_plot, cohort_event_study_plot,
)
from .rd import rdrobust, rdplot, rdplotdensity, rdbwsensitivity, rdbalance, rdplacebo, rdsummary
from .synth import (
    synth, SyntheticControl, synthplot, sdid, augsynth,
    demeaned_synth, robust_synth, gsynth, staggered_synth, conformal_synth,
    synthdid_estimate, sc_estimate, did_estimate,
    synthdid_placebo, synthdid_plot, synthdid_units_plot, synthdid_rmse_plot,
    california_prop99,
)
from .matching import match, MatchEstimator, ebalance, balanceplot, psplot
from .dml import dml, DoubleML
from .deepiv import deepiv, DeepIV
from .panel import panel, panel_compare, PanelResults, PanelCompareResults, PanelRegression
from .causal_impact import causal_impact, CausalImpactEstimator, impactplot
from .mediation import mediate, MediationAnalysis
from .bartik import bartik, BartikIV
from .output.outreg2 import OutReg2, outreg2
from .output.modelsummary import modelsummary, coefplot
from .output.sumstats import sumstats, balance_table
from .output.tab import tab
from .postestimation import margins, marginsplot, test, lincom
from .diagnostics import oster_bounds, mccrary_test, diagnose, het_test, reset_test, vif, sensemakr, rddensity, hausman_test, anderson_rubin_test, evalue, evalue_from_result, diagnose_result
from .inference import wild_cluster_bootstrap, aipw, ri_test, ipw, bootstrap, BootstrapResult
from .spatial import sar, sem, sdm, SpatialModel
from .plots import binscatter, set_theme, list_themes, use_chinese, interactive, get_code
from .utils import label_var, label_vars, get_label, get_labels, describe, pwcorr, winsor, read_data
from .gmm import xtabond
from .metalearners import metalearner, SLearner, TLearner, XLearner, RLearner, DRLearner
from .metalearners import cate_summary, cate_by_group, cate_plot, cate_group_plot, predict_cate, compare_metalearners, gate_test, blp_test
from .regression.heckman import heckman
from .regression.quantile import qreg, sqreg
from .regression.tobit import tobit
from .neural_causal import tarnet, cfrnet, dragonnet, TARNet, CFRNet, DragonNet
from .causal_discovery import notears, NOTEARS, pc_algorithm, PCAlgorithm
from .tmle import tmle, TMLE, super_learner, SuperLearner
from .policy_learning import policy_tree, PolicyTree, policy_value
from .conformal_causal import conformal_cate, ConformalCATE
from .bcf import bcf, BayesianCausalForest
from .bunching import bunching, BunchingEstimator
from .matrix_completion import mc_panel, MCPanel
from .dose_response import dose_response, DoseResponse
from .bounds import lee_bounds, manski_bounds
from .interference import spillover, SpilloverEstimator
from .dtr import g_estimation, GEstimation
from .multi_treatment import multi_treatment, MultiTreatment
from .robustness import spec_curve, SpecCurveResult, robustness_report, RobustnessResult, subgroup_analysis, SubgroupResult
from .survey import svydesign, SurveyDesign, svymean, svytotal, svyglm
from .dag import dag, DAG, dag_example, dag_examples, dag_example_positions, dag_simulate
from .registry import list_functions, describe_function, function_schema, search_functions, all_schemas

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
    "parallel_trends_plot",
    "bacon_plot",
    "group_time_plot",
    "did_plot",
    "enhanced_event_study_plot",
    "treatment_rollout_plot",
    "sensitivity_plot",
    "cohort_event_study_plot",
    # RD
    "rdrobust",
    "rdplot",
    "rdplotdensity",
    "rdbwsensitivity",
    "rdbalance",
    "rdplacebo",
    "rdsummary",
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
    # Matching
    "match",
    "MatchEstimator",
    "ebalance",
    "balanceplot",
    "psplot",
    # Double ML
    "dml",
    "DoubleML",
    # DeepIV
    "deepiv",
    "DeepIV",
    # Panel
    "panel",
    "panel_compare",
    "PanelResults",
    "PanelCompareResults",
    "PanelRegression",
    # Causal Impact
    "causal_impact",
    "CausalImpactEstimator",
    # Causal Forest
    "CausalForest",
    "causal_forest",
    # Output
    "OutReg2",
    "outreg2",
    "modelsummary",
    "coefplot",
    "sumstats",
    "balance_table",
    "tab",
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
    "read_data",
    # Dynamic Panel GMM
    "xtabond",
    "heckman",
    "qreg",
    "sqreg",
    "tobit",
    # Post-estimation
    "margins",
    "marginsplot",
    "test",
    "lincom",
    # Mediation
    "mediate",
    "MediationAnalysis",
    # Bartik IV
    "bartik",
    "BartikIV",
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
    "evalue",
    "evalue_from_result",
    "diagnose_result",
    # Inference
    "wild_cluster_bootstrap",
    "aipw",
    "ri_test",
    "ipw",
    "bootstrap",
    "BootstrapResult",
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
    # Matrix Completion
    "mc_panel",
    "MCPanel",
    # Dose-Response
    "dose_response",
    "DoseResponse",
    # Bounds
    "lee_bounds",
    "manski_bounds",
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
    # AI / Agent Registry
    "list_functions",
    "describe_function",
    "function_schema",
    "search_functions",
    "all_schemas",
]
