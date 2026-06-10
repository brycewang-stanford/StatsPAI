"""Auto-generated Tier-B enrichment for the function registry.

DO NOT EDIT BY HAND — regenerate via::

    python scripts/gen_baseline_cards.py

Every entry below is mechanically extracted from a docstring or from
the function name + category.  The ``apply()`` helper only fills a
:class:`statspai.registry.FunctionSpec` field if that field is
currently empty, so curated content in ``registry.py`` always wins.

See ``docs/agent_cards_spec.md`` for the tier definitions.
"""
from __future__ import annotations

from typing import Any, Dict


BASELINE_CARDS: Dict[str, Dict[str, Any]] = {
    'ARIMAResult': {
        'tags': ['timeseries'],
    },
    'Absorber': {
        'tags': ['panel'],
    },
    'AssumptionResult': {
        'tags': ['smart'],
    },
    'AssumptionViolation': {
        'tags': ['core'],
    },
    'AssumptionWarning': {
        'tags': ['core'],
    },
    'AttritionResult': {
        'tags': ['experimental'],
    },
    'AutoCATEResult': {
        'tags': ['causal'],
    },
    'AutoDIDResult': {
        'tags': ['smart'],
    },
    'AutoIVResult': {
        'tags': ['smart'],
    },
    'BLPResult': {
        'tags': ['structural'],
    },
    'BVARResult': {
        'tags': ['timeseries'],
    },
    'BalanceDiagnosticsResult': {
        'tags': ['causal'],
    },
    'BalanceResult': {
        'tags': ['experimental'],
    },
    'BanditBenchmarkResult': {
        'tags': ['causal'],
    },
    'BartikIV': {
        'tags': ['bartik'],
    },
    'BayesRDHTEResult': {
        'tags': ['causal'],
    },
    'BayesianCausalForest': {
        'tags': ['causal'],
    },
    'BayesianCausalResult': {
        'tags': ['bayes'],
    },
    'BayesianDIDResult': {
        'tags': ['bayes'],
    },
    'BayesianHTEIVResult': {
        'tags': ['bayes'],
    },
    'BayesianIVResult': {
        'tags': ['bayes'],
    },
    'BayesianMTEResult': {
        'tags': ['bayes'],
    },
    'BeyondAverageResult': {
        'tags': ['causal'],
    },
    'BootstrapResult': {
        'tags': ['inference'],
    },
    'BoundsResult': {
        'tags': ['causal'],
    },
    'BridgeResult': {
        'tags': ['bridge'],
    },
    'BunchingEstimator': {
        'tags': ['causal'],
    },
    'CATEEvalResult': {
        'tags': ['causal'],
    },
    'CEVAE': {
        'tags': ['neural_causal'],
    },
    'CEVAEResult': {
        'tags': ['neural_causal'],
    },
    'CFRNet': {
        'tags': ['neural_causal'],
    },
    'CSReport': {
        'tags': ['causal'],
    },
    'CardinalityMatchResult': {
        'tags': ['causal'],
    },
    'CausalDQNResult': {
        'tags': ['causal'],
    },
    'CausalForest': {
        'tags': ['causal'],
    },
    'CausalImpactEstimator': {
        'tags': ['causal'],
    },
    'CausalQuestion': {
        'tags': ['smart'],
    },
    'CausalResult': {
        'tags': ['core'],
    },
    'CloneCensorWeightResult': {
        'tags': ['target_trial'],
    },
    'ClusterCATEResult': {
        'tags': ['causal'],
    },
    'CointegrationResult': {
        'tags': ['timeseries'],
    },
    'Collection': {
        'tags': ['output'],
    },
    'CollectionItem': {
        'tags': ['output'],
    },
    'ComparisonResult': {
        'tags': ['smart'],
    },
    'ConformalCATE': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'ConformalCounterfactualResult': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'ConformalDensityResult': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'ConformalITEResult': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'ContinuousLATEResult': {
        'tags': ['causal'],
    },
    'ConvergenceFailure': {
        'tags': ['core'],
    },
    'ConvergenceWarning': {
        'tags': ['core'],
    },
    'CoxResult': {
        'tags': ['survival'],
    },
    'CrossClusterRCTResult': {
        'tags': ['interference'],
    },
    'DAG': {
        'tags': ['dag'],
    },
    'DAGValidationResult': {
        'tags': ['causal'],
    },
    'DDDResult': {
        'tags': ['causal'],
    },
    'DIDAnalysis': {
        'tags': ['causal'],
    },
    'DMLAveragingResult': {
        'tags': ['causal'],
    },
    'DMLDiagnostics': {
        'tags': ['causal'],
    },
    'DMLPanelResult': {
        'tags': ['causal'],
    },
    'DMLSensitivityResult': {
        'tags': ['causal'],
    },
    'DNCGNNDiDResult': {
        'tags': ['interference'],
    },
    'DRLearner': {
        'tags': ['causal'],
    },
    'DTEResult': {
        'tags': ['causal'],
    },
    'DYNOTEARSResult': {
        'tags': ['causal'],
    },
    'DataInsufficient': {
        'tags': ['core'],
    },
    'DebiasedConformalResult': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'DeepIV': {
        'tags': ['causal'],
    },
    'DiagnosticFinding': {
        'tags': ['smart'],
    },
    'DiagnosticTestResult': {
        'tags': ['epi'],
    },
    'DistIVResult': {
        'tags': ['causal'],
    },
    'DistRDResult': {
        'tags': ['causal'],
    },
    'DoseResponse': {
        'tags': ['causal'],
    },
    'DoubleML': {
        'tags': ['causal'],
    },
    'DoubleMLIIVM': {
        'tags': ['causal'],
    },
    'DoubleMLIRM': {
        'tags': ['causal'],
    },
    'DoubleMLPLIV': {
        'tags': ['causal'],
    },
    'DoubleMLPLR': {
        'tags': ['causal'],
    },
    'DragonNet': {
        'tags': ['neural_causal'],
    },
    'EconometricResults': {
        'tags': ['core'],
    },
    'EstimationResult': {
        'tags': ['smart'],
    },
    'FCIResult': {
        'tags': ['causal'],
    },
    'FEOLSResult': {
        'tags': ['panel'],
    },
    'FStatisticResult': {
        'tags': ['mendelian'],
    },
    'FailureMode': {
        'tags': ['agent'],
    },
    'FairConformalResult': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'FisherResult': {
        'tags': ['inference'],
    },
    'FrontierResult': {
        'tags': ['frontier'],
    },
    'FrontierSensitivityResult': {
        'tags': ['robustness'],
    },
    'FunctionalCATEResult': {
        'tags': ['causal'],
    },
    'GARCHResult': {
        'tags': ['timeseries'],
    },
    'GESResult': {
        'tags': ['causal'],
    },
    'GEstimation': {
        'tags': ['causal'],
    },
    'GLMEstimator': {
        'tags': ['regression'],
    },
    'GLMRegression': {
        'tags': ['regression'],
    },
    'GelbachResult': {
        'tags': ['decomposition'],
    },
    'GenMatchResult': {
        'tags': ['causal'],
    },
    'GeneralBunchingResult': {
        'tags': ['causal'],
    },
    'GrappleResult': {
        'tags': ['mendelian'],
    },
    'HALClassifier': {
        'tags': ['causal'],
    },
    'HALRegressor': {
        'tags': ['causal'],
    },
    'HDPanelQTEResult': {
        'tags': ['causal'],
    },
    'HelpResult': {
        'tags': ['agent'],
    },
    'HeterogeneityResult': {
        'tags': ['mendelian'],
    },
    'ICEResult': {
        'tags': ['gformula'],
    },
    'ICPResult': {
        'tags': ['causal'],
    },
    'IPCWResult': {
        'tags': ['censoring'],
    },
    'IVDiagResult': {
        'tags': ['causal'],
    },
    'IVRegression': {
        'tags': ['regression', 'iv'],
    },
    'IdentificationError': {
        'tags': ['smart'],
    },
    'IdentificationFailure': {
        'tags': ['core'],
    },
    'IdentificationPlan': {
        'tags': ['smart'],
    },
    'IdentificationReport': {
        'tags': ['smart'],
    },
    'IdentificationResult': {
        'tags': ['dag'],
    },
    'KDensityResult': {
        'tags': ['nonparametric'],
    },
    'KMResult': {
        'tags': ['survival'],
    },
    'KappaResult': {
        'tags': ['epi'],
    },
    'KernelIVResult': {
        'tags': ['causal'],
    },
    'KinkUnifiedResult': {
        'tags': ['causal'],
    },
    'KitagawaResult': {
        'tags': ['diagnostics'],
    },
    'LLMAnnotatorResult': {
        'tags': ['causal'],
    },
    'LLMConstrainedDAGResult': {
        'tags': ['causal'],
    },
    'LLMDAGProposal': {
        'tags': ['causal'],
    },
    'LPCMCIResult': {
        'tags': ['causal'],
    },
    'LPolyResult': {
        'tags': ['nonparametric'],
    },
    'LTMLESurvivalResult': {
        'tags': ['causal', 'tmle'],
    },
    'LeaveOneOutResult': {
        'tags': ['mendelian'],
    },
    'LiNGAMResult': {
        'tags': ['causal'],
    },
    'LocalProjectionsResult': {
        'tags': ['timeseries'],
    },
    'LongitudinalResult': {
        'tags': ['longitudinal'],
    },
    'MCGFormulaResult': {
        'tags': ['gformula'],
    },
    'MCPanel': {
        'tags': ['causal'],
    },
    'MEGLMResult': {
        'tags': ['panel'],
    },
    'MICEResult': {
        'tags': ['missing'],
    },
    'MRClustResult': {
        'tags': ['mendelian'],
    },
    'MRLapResult': {
        'tags': ['mendelian'],
    },
    'MRPressoResult': {
        'tags': ['mendelian'],
    },
    'MRRapsResult': {
        'tags': ['mendelian'],
    },
    'MRResult': {
        'tags': ['mendelian'],
    },
    'MRcMLResult': {
        'tags': ['mendelian'],
    },
    'MalmquistResult': {
        'tags': ['frontier'],
    },
    'MarginalStructuralModel': {
        'tags': ['causal'],
    },
    'MatchEstimator': {
        'tags': ['causal'],
    },
    'MatchedPairResult': {
        'tags': ['interference'],
    },
    'MeanComparisonResult': {
        'tags': ['output'],
    },
    'MediationAnalysis': {
        'tags': ['mediation'],
    },
    'MetafrontierResult': {
        'tags': ['frontier'],
    },
    'MethodIncompatibility': {
        'tags': ['core'],
    },
    'MixedResult': {
        'tags': ['panel'],
    },
    'ModeBasedResult': {
        'tags': ['mendelian'],
    },
    'MultiDPConformalResult': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'MultiScoreRDResult': {
        'tags': ['causal'],
    },
    'MultiTreatment': {
        'tags': ['causal'],
    },
    'NOTEARS': {
        'tags': ['causal'],
    },
    'NetworkExposureResult': {
        'tags': ['interference'],
    },
    'NotchResult': {
        'tags': ['causal'],
    },
    'NumericalInstability': {
        'tags': ['core'],
    },
    'OaxacaResult': {
        'tags': ['decomposition'],
    },
    'OfflineSafeResult': {
        'tags': ['causal'],
    },
    'OptimalDesignResult': {
        'tags': ['experimental'],
    },
    'OptimalMatchResult': {
        'tags': ['causal'],
    },
    'OutReg2': {
        'tags': ['output'],
    },
    'PATEEstimator': {
        'tags': ['inference'],
    },
    'PCAlgorithm': {
        'tags': ['causal'],
    },
    'PCMCIResult': {
        'tags': ['causal'],
    },
    'PSBalanceResult': {
        'tags': ['causal'],
    },
    'PanelCompareResults': {
        'tags': ['panel'],
    },
    'PanelRegression': {
        'tags': ['panel'],
    },
    'PanelResults': {
        'example': 'sp.panel(df, "y ~ x1 + x2", entity=\'id\', time=\'t\')',
        'tags': ['panel'],
    },
    'PanelUnitRootResult': {
        'tags': ['panel'],
    },
    'PaperTables': {
        'tags': ['output'],
    },
    'PeerEffectsResult': {
        'tags': ['interference'],
    },
    'PleiotropyResult': {
        'tags': ['mendelian'],
    },
    'PolicyTree': {
        'tags': ['causal'],
    },
    'PolicyTreeResult': {
        'tags': ['causal'],
    },
    'PowerResult': {
        'tags': ['power'],
    },
    'PrincipalStratResult': {
        'tags': ['causal'],
    },
    'ProductionResult': {
        'tags': ['structural'],
    },
    'Provenance': {
        'tags': ['output'],
    },
    'ProximalCausalInference': {
        'tags': ['causal', 'proximal'],
    },
    'ProxyScoreResult': {
        'tags': ['causal'],
    },
    'PubReadyResult': {
        'tags': ['smart'],
    },
    'QTEResult': {
        'tags': ['causal'],
    },
    'RDInterferenceResult': {
        'tags': ['causal', 'interference'],
    },
    'RDMultiResult': {
        'tags': ['causal'],
    },
    'RLearner': {
        'tags': ['causal'],
    },
    'ROCResult': {
        'tags': ['epi'],
    },
    'RadialResult': {
        'tags': ['mendelian'],
    },
    'RandomizationResult': {
        'tags': ['experimental'],
    },
    'RecommendationResult': {
        'tags': ['smart'],
    },
    'Regime': {
        'tags': ['longitudinal'],
    },
    'RegtableResult': {
        'tags': ['output'],
    },
    'ReplicationPack': {
        'tags': ['output'],
    },
    'ReproductionResult': {
        'tags': ['validation'],
    },
    'ReproductionStep': {
        'tags': ['validation'],
    },
    'RobustnessResult': {
        'tags': ['robustness'],
    },
    'RomanoWolfResult': {
        'tags': ['inference'],
    },
    'RuleCheck': {
        'tags': ['dag'],
    },
    'SBWResult': {
        'tags': ['causal'],
    },
    'SCM': {
        'tags': ['dag'],
    },
    'SLearner': {
        'tags': ['causal', 'metalearners'],
    },
    'SURResult': {
        'tags': ['regression'],
    },
    'SWIGGraph': {
        'tags': ['dag'],
    },
    'SelectionResult': {
        'tags': ['regression'],
    },
    'SensitivityDashboard': {
        'tags': ['smart'],
    },
    'SensitivityPriorProposal': {
        'tags': ['causal'],
    },
    'SensitivityResult': {
        'tags': ['causal'],
    },
    'SequentialSDIDResult': {
        'tags': ['causal'],
    },
    'SpatialDiDResult': {
        'tags': ['spatial'],
    },
    'SpatialIVResult': {
        'tags': ['spatial'],
    },
    'SpatialModel': {
        'tags': ['spatial'],
    },
    'SpecCurveResult': {
        'tags': ['robustness'],
    },
    'SpilloverEstimator': {
        'tags': ['interference'],
    },
    'StaggeredClusterRCTResult': {
        'tags': ['interference', 'staggered'],
    },
    'StatsPAIError': {
        'tags': ['core'],
    },
    'StatsPAIWarning': {
        'tags': ['core'],
    },
    'SteigerResult': {
        'tags': ['mendelian'],
    },
    'StructuralBreakResult': {
        'tags': ['timeseries'],
    },
    'SubgroupResult': {
        'tags': ['robustness'],
    },
    'SuperLearner': {
        'tags': ['causal'],
    },
    'SurveyDesign': {
        'tags': ['survey'],
    },
    'SynthComparison': {
        'tags': ['causal'],
    },
    'SyntheticControl': {
        'tags': ['causal'],
    },
    'SyntheticSurvivalResult': {
        'tags': ['causal'],
    },
    'TARNet': {
        'tags': ['neural_causal'],
    },
    'TLearner': {
        'tags': ['causal', 'metalearners'],
    },
    'TMLE': {
        'tags': ['causal', 'tmle'],
    },
    'TargetTrialProtocol': {
        'tags': ['target_trial'],
    },
    'TargetTrialResult': {
        'tags': ['target_trial'],
    },
    'TextTreatmentResult': {
        'tags': ['causal'],
    },
    'TransportIdentificationResult': {
        'tags': ['transport'],
    },
    'TransportWeightResult': {
        'tags': ['transport'],
    },
    'UnobservedConfounderProposal': {
        'tags': ['causal'],
    },
    'VARResult': {
        'tags': ['timeseries'],
    },
    'ValidationReport': {
        'tags': ['validation'],
    },
    'W': {
        'tags': ['spatial'],
    },
    'WeakRobustResult': {
        'tags': ['diagnostics'],
    },
    'XLearner': {
        'tags': ['causal', 'metalearners'],
    },
    'YuElwertResult': {
        'tags': ['decomposition'],
    },
    'absorb_ols': {
        'tags': ['panel'],
    },
    'adjust_pvalues': {
        'example': "sp.adjust_pvalues([0.01, 0.04, 0.03, 0.20], method='holm')",
        'tags': ['inference'],
    },
    'aft': {
        'tags': ['survival'],
    },
    'agent_card': {
        'example': "sp.agent_card('did')",
        'tags': ['agent'],
    },
    'agent_cards': {
        'example': "sp.agent_cards(category='causal', stability='stable')",
        'tags': ['agent'],
    },
    'all_schemas': {
        'example': 'sp.all_schemas()',
        'tags': ['agent'],
    },
    'always_treat': {
        'tags': ['longitudinal'],
    },
    'anderson_rubin_ci': {
        'tags': ['causal'],
    },
    'anderson_rubin_test': {
        'example': "sp.anderson_rubin_test(df, y='wage', endog='education', instruments=['parent_edu', 'distance'])",
        'tags': ['diagnostics'],
    },
    'arima': {
        'tags': ['timeseries'],
    },
    'assumption_audit': {
        'example': 'sp.assumption_audit(result)',
        'tags': ['smart'],
    },
    'attach_provenance': {
        'tags': ['output'],
    },
    'attrition_bounds': {
        'tags': ['experimental'],
    },
    'attrition_test': {
        'example': "sp.attrition_test(df, treatment='treated', observed='endline_observed', covariates=['age', 'income', 'education'])",
        'tags': ['experimental'],
    },
    'auc': {
        'tags': ['epi'],
    },
    'audit': {
        'example': 'sp.audit(r)',
        'tags': ['smart'],
    },
    'auto_cate': {
        'example': "sp.auto_cate(df, y='wage', treat='training', covariates=['age', 'edu', 'exp'])",
        'tags': ['causal'],
    },
    'auto_cate_tuned': {
        'tags': ['causal'],
    },
    'auto_did': {
        'tags': ['smart'],
    },
    'auto_iv': {
        'tags': ['smart'],
    },
    'available_methods': {
        'tags': ['decomposition'],
    },
    'average_treatment_effect': {
        'tags': ['causal'],
    },
    'bacon_plot': {
        'tags': ['causal'],
    },
    'balance_check': {
        'example': "sp.balance_check(df, treatment='treated', covariates=['age', 'income', 'education'])",
        'tags': ['experimental'],
    },
    'balance_diagnostics': {
        'tags': ['causal'],
    },
    'balance_panel': {
        'example': "sp.balance_panel(df, entity='id', time='year')",
        'tags': ['panel'],
    },
    'balance_table': {
        'example': "sp.balance_table(df, treat='treated', covariates=['age', 'edu', 'income'], output='balance.docx')",
        'tags': ['output'],
    },
    'balanceplot': {
        'tags': ['causal'],
    },
    'basque_terrorism': {
        'reference': 'abadie2003economic',
        'tags': ['causal'],
    },
    'bauer_sinning': {
        'tags': ['decomposition'],
    },
    'bayes_hte_iv': {
        'tags': ['bayes', 'bayesian'],
    },
    'bcf': {
        'example': "sp.bcf(df, y='outcome', treat='treatment', covariates=['x1', 'x2', 'x3'])",
        'tags': ['causal'],
    },
    'benjamini_hochberg': {
        'tags': ['inference'],
    },
    'betareg': {
        'example': "sp.betareg(df, y='share', x=['price', 'quality'])",
        'tags': ['regression'],
    },
    'bib_for': {
        'example': 'sp.bib_for(r)',
        'tags': ['smart'],
    },
    'binscatter': {
        'example': "sp.binscatter(df, y='wage', x='education')",
        'tags': ['plots'],
    },
    'biprobit': {
        'example': "sp.biprobit(df, y1='employed', y2='married', x1=['age', 'education'], x2=['age', 'children'])",
        'tags': ['regression'],
    },
    'bjs': {
        'tags': ['causal'],
    },
    'bjs_pretrend_joint': {
        'tags': ['causal'],
    },
    'block_weights': {
        'tags': ['spatial'],
    },
    'blp': {
        'tags': ['structural'],
    },
    'blp_test': {
        'reference': 'chernozhukov2018double',
        'tags': ['causal'],
    },
    'bonferroni': {
        'tags': ['inference'],
    },
    'borusyak_jaravel_spiess': {
        'tags': ['causal'],
    },
    'boundary_rd': {
        'tags': ['causal'],
    },
    'breakdown_frontier': {
        'example': 'sp.breakdown_frontier( estimate=0.05, se=0.02, assumption="parallel_trends", max_violation=0.1, )',
        'tags': ['causal'],
    },
    'breakdown_m': {
        'example': 'sp.breakdown_m(r, e=0)',
        'tags': ['causal'],
    },
    'brief': {
        'example': 'sp.brief(r)',
        'tags': ['smart'],
    },
    'bunching': {
        'example': "sp.bunching(df, running_var='income', threshold=50000, dt=0.10)",
        'tags': ['causal'],
    },
    'bvar': {
        'tags': ['timeseries'],
    },
    'calibrate_confounding_strength': {
        'reference': 'baitairian2025calibrating',
        'tags': ['robustness'],
    },
    'calibration_test': {
        'tags': ['causal'],
    },
    'california_prop99': {
        'tags': ['causal'],
    },
    'california_tobacco': {
        'reference': 'abadie2010synthetic',
        'tags': ['causal'],
    },
    'cardinality_match': {
        'tags': ['causal'],
    },
    'cate_by_group': {
        'tags': ['causal'],
    },
    'cate_eval': {
        'example': 'sp.cate_eval(tau_hat, Y, T, X=X).summary()',
        'tags': ['causal'],
    },
    'cate_group_plot': {
        'tags': ['causal'],
    },
    'cate_plot': {
        'tags': ['causal'],
    },
    'cate_summary': {
        'tags': ['causal'],
    },
    'causal_discovery': {
        'tags': ['causal'],
    },
    'causal_rl_benchmark': {
        'tags': ['causal'],
    },
    'cbps': {
        'tags': ['causal'],
    },
    'cfm_decompose': {
        'tags': ['decomposition'],
    },
    'check_identification': {
        'example': "sp.check_identification( df, y='wage', treatment='training', covariates=['age', 'education'], id='worker', time='year', design='did', )",
        'tags': ['smart'],
    },
    'chilean_households': {
        'tags': ['decomposition'],
    },
    'citation': {
        'example': 'sp.citation()  # BibTeX (default)',
        'tags': ['output'],
    },
    'citations_to_bib_entries': {
        'tags': ['output'],
    },
    'clogit': {
        'example': "sp.clogit('chosen ~ price + quality', data=df, group='case_id')",
        'tags': ['regression'],
    },
    'cloglog': {
        'example': 'sp.cloglog("default ~ income + balance", data=df)',
        'tags': ['regression'],
    },
    'cluster_cate': {
        'tags': ['causal'],
    },
    'cluster_matched_pair': {
        'tags': ['interference'],
    },
    'cluster_robust_se': {
        'tags': ['inference'],
    },
    'cluster_staggered_rollout': {
        'tags': ['interference', 'staggered'],
    },
    'coefplot': {
        'tags': ['output'],
    },
    'cohort_event_study_plot': {
        'example': "sp.did(df, y='y', treat='g', time='t', id='i', method='cs')",
        'tags': ['causal', 'event_study'],
    },
    'compare_estimators': {
        'example': "sp.compare_estimators( data=df, y='wage', treatment='training', methods=['ols', 'matching', 'ipw', 'dml'], covariates=['age', 'education'], )",
        'tags': ['smart'],
    },
    'compare_metalearners': {
        'example': "sp.compare_metalearners(df, y='wage', treat='training', covariates=['age', 'edu'])",
        'tags': ['causal'],
    },
    'compute_data_hash': {
        'tags': ['output'],
    },
    'conditional_lr_ci': {
        'tags': ['causal'],
    },
    'conformal_available_kinds': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'conformal_cate': {
        'example': "sp.conformal_cate(df, y='outcome', treat='treatment', covariates=['x1', 'x2'])",
        'tags': ['conformal_causal', 'conformal'],
    },
    'conformal_counterfactual': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'conformal_debiased_ml': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'conformal_density_ite': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'conformal_ite': {
        'tags': ['causal', 'conformal'],
    },
    'conformal_ite_interval': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'conformal_ite_multidp': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'conformal_synth': {
        'example': "sp.conformal_synth(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal', 'conformal'],
    },
    'conley': {
        'example': 'sp.conley(result, data=df, lat="latitude", lon="longitude", dist_cutoff=100)',
        'tags': ['inference'],
    },
    'contrast': {
        'example': 'sp.contrast(result, data=df, variable="education", method="r", reference=0)',
        'tags': ['postestimation'],
    },
    'copula_sensitivity': {
        'reference': 'balgi2025sensitivity',
        'tags': ['robustness'],
    },
    'coverage_matrix': {
        'tags': ['validation'],
    },
    'cox': {
        'example': 'sp.cox(formula="time ~ age + treatment", data=df, event="status")',
        'tags': ['survival'],
    },
    'cox_frailty': {
        'tags': ['survival'],
    },
    'cps_wage': {
        'tags': ['decomposition'],
    },
    'cr2_se': {
        'tags': ['inference'],
    },
    'cr3_jackknife_vcov': {
        'tags': ['inference'],
    },
    'cs_report': {
        'tags': ['causal'],
    },
    'csl_filename': {
        'tags': ['output'],
    },
    'csl_url': {
        'tags': ['output'],
    },
    'cusum_test': {
        'tags': ['timeseries'],
    },
    'dag_example_positions': {
        'tags': ['dag'],
    },
    'dag_examples': {
        'tags': ['dag'],
    },
    'dag_simulate': {
        'example': "sp.dag_simulate('discrimination')",
        'tags': ['dag'],
    },
    'das_gupta': {
        'tags': ['decomposition'],
    },
    'decompose': {
        'tags': ['decomposition'],
    },
    'deepiv': {
        'example': "sp.deepiv( df, y='lwage', treat='educ', instruments=['nearc4'], covariates=['exper', 'expersq'], )",
        'tags': ['causal'],
    },
    'demean': {
        'tags': ['panel'],
    },
    'demeaned_synth': {
        'example': "sp.demeaned_synth(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal'],
    },
    'describe': {
        'example': 'sp.describe(df)',
        'tags': ['utils'],
    },
    'describe_function': {
        'example': "sp.describe_function('did')",
        'tags': ['agent'],
    },
    'detect_design': {
        'example': 'sp.detect_design(df)',
        'tags': ['smart'],
    },
    'dfl_decompose': {
        'tags': ['decomposition'],
    },
    'dgp_bartik': {
        'tags': ['utils'],
    },
    'dgp_bunching': {
        'tags': ['utils'],
    },
    'dgp_cluster_rct': {
        'tags': ['utils'],
    },
    'dgp_did': {
        'tags': ['utils'],
    },
    'dgp_iv': {
        'tags': ['utils'],
    },
    'dgp_observational': {
        'tags': ['utils'],
    },
    'dgp_panel': {
        'tags': ['utils'],
    },
    'dgp_rct': {
        'tags': ['utils'],
    },
    'dgp_rd': {
        'tags': ['utils'],
    },
    'dgp_rd_2d': {
        'tags': ['utils', 'rd'],
    },
    'dgp_rd_hte': {
        'tags': ['utils', 'rd'],
    },
    'dgp_rd_kink': {
        'tags': ['utils', 'rd'],
    },
    'dgp_rd_multi': {
        'tags': ['utils', 'rd'],
    },
    'dgp_rdit': {
        'tags': ['utils'],
    },
    'dgp_synth': {
        'tags': ['utils'],
    },
    'diagnose': {
        'example': "sp.diagnose(df, y='wage', x=['education', 'experience'])",
        'tags': ['diagnostics'],
    },
    'diagnostic_test': {
        'tags': ['epi'],
    },
    'did_2stage': {
        'tags': ['causal', 'did'],
    },
    'did_estimate': {
        'tags': ['causal', 'did'],
    },
    'did_plot': {
        'tags': ['causal', 'did'],
    },
    'did_report': {
        'tags': ['causal', 'did'],
    },
    'did_summary': {
        'example': "sp.did_summary(df, y='y', time='time', first_treat='first_treat', group='unit')",
        'tags': ['causal', 'did'],
    },
    'did_summary_plot': {
        'example': 'sp.did_summary_plot(out)',
        'tags': ['causal', 'did'],
    },
    'did_summary_to_latex': {
        'tags': ['causal', 'did'],
    },
    'did_summary_to_markdown': {
        'tags': ['causal', 'did'],
    },
    'discos': {
        'example': "sp.discos(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal'],
    },
    'discos_plot': {
        'example': "sp.discos_plot(result, type='quantile_effect')",
        'tags': ['causal'],
    },
    'discos_test': {
        'example': "sp.discos_test(result, test='ks')",
        'tags': ['causal'],
    },
    'disparity_decompose': {
        'tags': ['decomposition'],
    },
    'disparity_panel': {
        'tags': ['decomposition'],
    },
    'dist_iv': {
        'tags': ['causal'],
    },
    'distance_band': {
        'tags': ['spatial'],
    },
    'distributional_te': {
        'tags': ['causal'],
    },
    'dml_diagnostics': {
        'tags': ['causal', 'dml'],
    },
    'dml_sensitivity': {
        'tags': ['causal', 'dml'],
    },
    'dnc_gnn_did': {
        'tags': ['interference'],
    },
    'do_calculus_apply': {
        'tags': ['dag'],
    },
    'do_rule1': {
        'tags': ['dag'],
    },
    'do_rule2': {
        'tags': ['dag'],
    },
    'do_rule3': {
        'tags': ['dag'],
    },
    'ebalance': {
        'example': "sp.ebalance(df, y='outcome', treat='treated', covariates=['age', 'income', 'education'])",
        'tags': ['causal'],
    },
    'effective_f_test': {
        'example': "sp.effective_f_test(df, endog='educ', instruments=['qob'])",
        'tags': ['diagnostics'],
    },
    'engle_granger': {
        'tags': ['timeseries'],
    },
    'enhanced_event_study_plot': {
        'example': "sp.did(df, y='y', treat='g', time='t', id='i')",
        'tags': ['causal', 'event_study'],
    },
    'estat': {
        'tags': ['diagnostics'],
    },
    'estclear': {
        'tags': ['output'],
    },
    'eststo': {
        'tags': ['output'],
    },
    'etable': {
        'tags': ['panel'],
    },
    'etregress': {
        'example': "sp.etregress(df, y='wage', x=['experience', 'education'], treatment='union', z=['father_union', 'region'])",
        'tags': ['regression'],
    },
    'etwfe_emfx': {
        'example': "sp.etwfe_emfx(fit, type='event')",
        'tags': ['causal'],
    },
    'evalue': {
        'example': "sp.evalue(estimate=2.5, measure='RR')",
        'tags': ['diagnostics', 'sensitivity'],
    },
    'evalue_from_result': {
        'example': 'sp.evalue_from_result(result)',
        'tags': ['diagnostics', 'sensitivity'],
    },
    'evalue_rd': {
        'example': 'sp.evalue_rd(200, 150, 100, 250)',
        'tags': ['diagnostics', 'sensitivity', 'epi'],
    },
    'bias_factor': {
        'example': 'sp.bias_factor(rr_eu=2.0, rr_ud=2.0)',
        'tags': ['diagnostics', 'sensitivity'],
    },
    'evalue_rr': {
        'tags': ['causal', 'sensitivity'],
    },
    'event_study_table': {
        'example': 'sp.event_study_table(r, include_reference=True)',
        'tags': ['postestimation', 'event_study'],
    },
    'examples': {
        'example': 'sp.examples("did")',
        'tags': ['smart'],
    },
    'fairlie': {
        'tags': ['decomposition'],
    },
    'fci': {
        'tags': ['causal'],
    },
    'feglm': {
        'tags': ['panel'],
    },
    'feols': {
        'tags': ['panel'],
    },
    'fepois': {
        'tags': ['panel'],
    },
    'ffl_decompose': {
        'tags': ['decomposition'],
    },
    'fisher_exact': {
        'example': 'sp.fisher_exact( data=df, y="outcome", treatment="treated", statistic="ate", n_perm=10000, seed=42)',
        'tags': ['inference'],
    },
    'focal_cate': {
        'tags': ['causal'],
    },
    'forest_diagnostics': {
        'tags': ['causal'],
    },
    'format_provenance': {
        'tags': ['output'],
    },
    'fracreg': {
        'example': "sp.fracreg(df, y='participation_rate', x=['income', 'age'])",
        'tags': ['regression'],
    },
    'frontdoor': {
        'tags': ['causal'],
    },
    'frontier': {
        'example': "sp.frontier(df, y='log_y', x=['log_k', 'log_l'])",
        'tags': ['frontier'],
    },
    'function_schema': {
        'example': "sp.function_schema('regress')",
        'tags': ['agent'],
    },
    'g_estimation': {
        'example': "sp.g_estimation( df, y='outcome', treatments=['treatment_stage1', 'treatment_stage2'], covariates_by_stage=[['x1', 'x2'], ['x1', 'x2', 'x3']])",
        'tags': ['causal'],
    },
    'gap_closing': {
        'tags': ['decomposition'],
    },
    'garch': {
        'tags': ['timeseries'],
    },
    'gate_test': {
        'tags': ['causal'],
    },
    'geary': {
        'tags': ['spatial'],
    },
    'gelbach': {
        'example': 'sp.gelbach( data=df, y="wage", base_x=["education"], added_x=["experience", "tenure", "union"], )',
        'tags': ['decomposition'],
    },
    'general_bunching': {
        'tags': ['causal'],
    },
    'genmatch': {
        'tags': ['causal'],
    },
    'geographic_rd': {
        'tags': ['causal'],
    },
    'german_reunification': {
        'reference': 'abadie2015comparative',
        'tags': ['causal'],
    },
    'ges': {
        'tags': ['causal'],
    },
    'get_code': {
        'tags': ['plots'],
    },
    'get_journal_template': {
        'tags': ['output'],
    },
    'get_label': {
        'tags': ['utils'],
    },
    'get_labels': {
        'tags': ['utils'],
    },
    'get_provenance': {
        'tags': ['output'],
    },
    'getis_ord_g': {
        'tags': ['spatial', 'rd'],
    },
    'getis_ord_local': {
        'tags': ['spatial', 'rd'],
    },
    'gformula_mc': {
        'example': 'sp.gformula_mc(..., strategy=dynamic)',
        'tags': ['gformula'],
    },
    'ggdid': {
        'tags': ['causal'],
    },
    'glm': {
        'tags': ['regression'],
    },
    'gmm': {
        'example': 'sp.gmm(moment_fn, theta0=np.zeros(2))',
        'tags': ['panel'],
    },
    'granger_causality': {
        'tags': ['timeseries'],
    },
    'grapple': {
        'example': 'sp.grapple(bx, by, sx, sy)',
    },
    'group_time_plot': {
        'example': "sp.did(df, y='y', treat='g', time='t', id='i', method='cs')",
        'tags': ['causal'],
    },
    'gsynth': {
        'example': "sp.gsynth(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal'],
    },
    'gt': {
        'example': 'sp.gt(rt)',
        'tags': ['output'],
    },
    'gwr': {
        'tags': ['spatial'],
    },
    'gwr_bandwidth': {
        'tags': ['spatial'],
    },
    'hausman_test': {
        'example': "sp.hausman_test(df, y='wage', x=['education', 'experience'], id='worker', time='year')",
        'tags': ['diagnostics'],
    },
    'hdfe_ols': {
        'tags': ['panel'],
    },
    'help': {
        'tags': ['agent'],
    },
    'het_test': {
        'reference': 'breusch1979simple',
        'tags': ['diagnostics'],
    },
    'holm': {
        'tags': ['inference'],
    },
    'honest_did': {
        'example': 'sp.honest_did(r, e=0)',
        'tags': ['causal'],
    },
    'honest_variance': {
        'tags': ['causal'],
    },
    'horowitz_manski': {
        'example': 'sp.horowitz_manski( data=df, y="wage", treatment="trained", covariates=["age", "education"], y_lower=0, y_upper=100, )',
        'tags': ['causal'],
    },
    'hurdle': {
        'example': "sp.hurdle(data=df, y='doctor_visits', x=['age', 'income'], count_model='negbin')",
        'tags': ['regression'],
    },
    'icc': {
        'tags': ['panel'],
    },
    'immortal_time_check': {
        'tags': ['target_trial'],
    },
    'impactplot': {
        'tags': ['causal'],
    },
    'impacts': {
        'tags': ['spatial'],
    },
    'inequality_index': {
        'tags': ['decomposition'],
    },
    'interactive': {
        'example': 'sp.interactive(fig)',
        'tags': ['plots'],
    },
    'interactive_fe': {
        'example': "sp.interactive_fe(df, y='gdp', x=['investment', 'trade'], id='country', time='year', n_factors=2)",
        'tags': ['panel'],
    },
    'interference_available_designs': {
        'tags': ['interference'],
    },
    'irf': {
        'tags': ['timeseries'],
    },
    'is_great_tables_available': {
        'tags': ['output'],
    },
    'iv_bounds': {
        'example': 'sp.iv_bounds( data=df, y="wage", treatment="trained", instrument="lottery", assumption="monotone_iv", )',
        'tags': ['causal'],
    },
    'ivqreg': {
        'example': "sp.ivqreg(df, y='earnings', endog='schooling', instruments='quarter_of_birth', exog=['age', 'race'], tau=0.5, bootstrap=400)",
        'tags': ['regression'],
    },
    'jackknife_se': {
        'example': 'sp.jackknife_se(result, data=df, cluster="state")',
        'tags': ['inference'],
    },
    'jive': {
        'tags': ['regression'],
    },
    'johansen': {
        'example': "sp.johansen(df, variables=['gdp', 'consumption', 'investment'], lags=2)",
        'tags': ['timeseries'],
    },
    'join_counts': {
        'tags': ['spatial'],
    },
    'kan_dlate': {
        'tags': ['causal'],
    },
    'kaplan_meier': {
        'example': 'sp.kaplan_meier(data=df, duration="time", event="status")',
        'tags': ['survival'],
    },
    'kdensity': {
        'example': "sp.kdensity(df, x='income')",
        'tags': ['nonparametric'],
    },
    'kernel_weights': {
        'tags': ['spatial'],
    },
    'kink_unified': {
        'tags': ['causal'],
    },
    'kitagawa_decompose': {
        'tags': ['decomposition'],
    },
    'kitagawa_test': {
        'example': 'sp.kitagawa_test( data=df, y="outcome", treatment="treated", instrument="assigned", n_boot=1000, seed=42, )',
        'tags': ['diagnostics'],
    },
    'knn_weights': {
        'tags': ['spatial'],
    },
    'label_var': {
        'example': "sp.label_var(df, 'wage', 'Monthly wage (CNY)')",
        'tags': ['utils'],
    },
    'label_vars': {
        'example': "sp.label_vars(df, { 'wage': 'Monthly wage (CNY)', 'edu': 'Years of education', 'exp': 'Work experience (years)', })",
        'tags': ['utils'],
    },
    'lasso_iv': {
        'example': "sp.lasso_iv(df, y='lwage', x_endog=['educ'], x_exog=['exper'], z=[f'qob_{i}' for i in range(40)])",
        'tags': ['regression'],
    },
    'lasso_select': {
        'tags': ['regression'],
    },
    'lcsf': {
        'tags': ['frontier'],
    },
    'lee_bounds': {
        'example': "sp.lee_bounds(df, y='wage', treat='training', selection='employed')",
        'tags': ['causal'],
    },
    'liml': {
        'example': "sp.liml(data=df, y='lwage', x_endog=['educ'], x_exog=['exper', 'expersq'], z=['nearc4'])",
        'tags': ['regression'],
    },
    'lincom': {
        'example': 'sp.lincom(result, "x1 + x2")',
        'tags': ['postestimation'],
    },
    'lineage_summary': {
        'tags': ['output'],
    },
    'linear_calibration': {
        'tags': ['survey'],
    },
    'lingam': {
        'tags': ['causal'],
    },
    'lisa_cluster_map': {
        'tags': ['spatial'],
    },
    'list_csl_styles': {
        'tags': ['output'],
    },
    'list_functions': {
        'tags': ['agent'],
    },
    'list_journal_templates': {
        'tags': ['output'],
    },
    'list_replications': {
        'example': 'sp.list_replications()',
        'tags': ['smart'],
    },
    'list_themes': {
        'example': 'sp.list_themes()',
        'tags': ['plots'],
    },
    'llm_dag_propose': {
        'tags': ['causal', 'dag'],
    },
    'llm_sensitivity_priors': {
        'tags': ['causal'],
    },
    'llm_unobserved_confounders': {
        'tags': ['causal'],
    },
    'lm_tests': {
        'tags': ['spatial'],
    },
    'local_projections': {
        'tags': ['timeseries'],
    },
    'logit': {
        'example': 'sp.logit("admit ~ gre + gpa + rank", data=df)',
        'tags': ['regression'],
    },
    'logrank_test': {
        'example': 'sp.logrank_test(data=df, duration="time", event="status", group="treatment")',
        'tags': ['survival'],
    },
    'love_plot': {
        'tags': ['causal'],
    },
    'lpoly': {
        'example': "sp.lpoly(df, y='wage', x='experience')",
        'tags': ['nonparametric'],
    },
    'lrtest': {
        'tags': ['panel'],
    },
    'ltmle_survival': {
        'tags': ['causal', 'tmle'],
    },
    'machado_mata': {
        'tags': ['decomposition'],
    },
    'make_bib_key': {
        'tags': ['output'],
    },
    'malmquist': {
        'tags': ['frontier'],
    },
    'manski_bounds': {
        'example': "sp.manski_bounds(df, y='employed', treat='training', y_lower=0, y_upper=1)",
        'tags': ['causal'],
    },
    'margins': {
        'example': 'sp.margins(result, data=df)',
        'tags': ['postestimation'],
    },
    'margins_at': {
        'example': 'sp.margins_at(result, data=df, at={"experience": [1, 5, 10, 15, 20]})',
        'tags': ['postestimation'],
    },
    'margins_at_plot': {
        'tags': ['postestimation'],
    },
    'margins_table': {
        'example': 'sp.margins_table(m)',
        'tags': ['postestimation'],
    },
    'marginsplot': {
        'tags': ['postestimation'],
    },
    'matrix_completion': {
        'tags': ['causal'],
    },
    'mc_panel': {
        'example': "sp.mc_panel(df, y='gdp', unit='country', time='year', treat='treated')",
        'tags': ['causal'],
    },
    'mc_synth': {
        'example': "sp.mc_synth(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal'],
    },
    'mde': {
        'example': 'sp.mde("did", n=1000, n_periods=10, n_treated_periods=5)',
        'tags': ['power'],
    },
    'mediate_sensitivity': {
        'tags': ['mediation'],
    },
    'mediation': {
        'tags': ['causal'],
    },
    'mediation_decompose': {
        'tags': ['decomposition'],
    },
    'megamma': {
        'tags': ['panel'],
    },
    'meglm': {
        'tags': ['panel'],
    },
    'melly_decompose': {
        'tags': ['decomposition'],
    },
    'melogit': {
        'tags': ['panel'],
    },
    'menbreg': {
        'tags': ['panel'],
    },
    'mendelian_randomization': {
        'example': "sp.mendelian_randomization( data=snp_stats, beta_exposure='beta_x', beta_outcome='beta_y', se_exposure='se_x', se_outcome='se_y', exposure_name='BMI', outcome_name='T2D', )",
        'tags': ['mendelian'],
    },
    'meologit': {
        'tags': ['panel'],
    },
    'mepoisson': {
        'tags': ['panel'],
    },
    'metafrontier': {
        'tags': ['frontier'],
    },
    'mgwr': {
        'tags': ['spatial'],
    },
    'mi_estimate': {
        'example': 'sp.mi_estimate(mice_res, sp.regress, formula="y ~ x1 + x2")',
        'tags': ['missing'],
    },
    'mice': {
        'example': "sp.mice(df, m=5, method='pmm')",
        'tags': ['missing'],
    },
    'mincer_wage_panel': {
        'tags': ['decomposition'],
    },
    'mixed': {
        'tags': ['panel'],
    },
    'mixlogit': {
        'example': "sp.mixlogit( df, y='chosen', alt='alt_id', chid='obs_id', x_fixed=['price'], x_random=['quality', 'time'], panel_id='person_id', n_draws=1000, )",
        'tags': ['regression'],
    },
    'ml_bounds': {
        'example': "sp.ml_bounds(df, y='wage', treat='training', covariates=['age', 'educ', 'exper'])",
        'tags': ['causal'],
    },
    'mlogit': {
        'example': "sp.mlogit('choice ~ price + income', data=df, base=0)",
        'tags': ['regression'],
    },
    'model_averaging_dml': {
        'tags': ['causal'],
    },
    'moran': {
        'tags': ['spatial'],
    },
    'moran_local': {
        'tags': ['spatial'],
    },
    'moran_plot': {
        'tags': ['spatial'],
    },
    'moran_residuals': {
        'tags': ['spatial'],
    },
    'mr_available_methods': {
        'tags': ['mendelian', 'mendelian_randomization'],
    },
    'mr_clust': {
        'example': 'sp.mr_clust(bx, by, sx, sy, K_range=(1, 4))',
    },
    'mr_cml': {
        'example': 'sp.mr_cml(bx, by, sx, sy)',
    },
    'mr_egger': {
        'tags': ['mendelian', 'mendelian_randomization'],
    },
    'mr_funnel_plot': {
        'tags': ['mendelian', 'mendelian_randomization'],
    },
    'mr_ivw': {
        'tags': ['mendelian', 'mendelian_randomization'],
    },
    'mr_lap': {
        'example': 'sp.mr_lap(bx, by, sx, sy, overlap_fraction=0.4, overlap_rho=0.18)',
    },
    'mr_median': {
        'tags': ['mendelian', 'mendelian_randomization'],
    },
    'mr_scatter_plot': {
        'tags': ['mendelian', 'mendelian_randomization'],
    },
    'multi_cutoff_rd': {
        'tags': ['causal'],
    },
    'multi_outcome_synth': {
        'example': "sp.multi_outcome_synth( df, outcomes=['gdp', 'employment', 'wages'], unit='state', time='year', treated_unit='California', treatment_time=1989, )",
        'tags': ['causal'],
    },
    'multi_score_rd': {
        'tags': ['causal'],
    },
    'multiway_cluster_vcov': {
        'tags': ['inference'],
    },
    'neural_causal_plot': {
        'tags': ['neural_causal'],
    },
    'neural_causal_to_excel': {
        'tags': ['neural_causal'],
    },
    'neural_causal_to_html': {
        'tags': ['neural_causal'],
    },
    'neural_causal_to_markdown': {
        'tags': ['neural_causal'],
    },
    'neural_effects_frame': {
        'tags': ['neural_causal'],
    },
    'neural_summary_frame': {
        'tags': ['neural_causal'],
    },
    'neural_training_frame': {
        'tags': ['neural_causal'],
    },
    'never_treat': {
        'tags': ['longitudinal'],
    },
    'nonlinear_icp': {
        'tags': ['causal'],
    },
    'notch': {
        'example': "sp.notch(df, x='income', notch_point=50000, notch_size=0.10, bin_width=500)",
        'tags': ['causal'],
    },
    'notears': {
        'example': "sp.notears(df, variables=['X', 'Z', 'M', 'Y'])",
        'tags': ['causal'],
    },
    'number_needed_to_treat': {
        'tags': ['epi'],
    },
    'oaxaca': {
        'tags': ['decomposition'],
    },
    'offline_safe_policy': {
        'tags': ['causal'],
    },
    'ologit': {
        'example': "sp.ologit('satisfaction ~ income + age', data=df)",
        'tags': ['regression'],
    },
    'oprobit': {
        'example': "sp.oprobit(data=df, y='rating', x=['quality', 'price'])",
        'tags': ['regression'],
    },
    'optimal_design': {
        'example': 'sp.optimal_design(mde=0.2, sigma=1.0, icc=0.05, cluster_size=20)',
        'tags': ['experimental'],
    },
    'optimal_match': {
        'tags': ['causal'],
    },
    'oster_delta': {
        'example': 'sp.oster_delta( data=df, y="wage", x_base=["education"], x_controls=["experience", "tenure"], r_max=1.3, )',
        'tags': ['causal', 'sensitivity'],
    },
    'outlier_indicator': {
        'tags': ['utils'],
    },
    'overlap_plot': {
        'tags': ['causal'],
    },
    'overlap_weights': {
        'tags': ['causal'],
    },
    'panel_fgls': {
        'example': "sp.panel_fgls(df, y='gdp', x=['investment', 'trade'], id='country', time='year', panels='heteroskedastic', corr='ar1')",
        'tags': ['panel'],
    },
    'panel_logit': {
        'tags': ['panel'],
    },
    'panel_probit': {
        'tags': ['panel'],
    },
    'panel_unitroot': {
        'example': "sp.panel_unitroot(df, variable='gdp', id='country', time='year')",
        'tags': ['panel'],
    },
    'parallel_trends_plot': {
        'tags': ['causal'],
    },
    'parity_gap_report': {
        'tags': ['validation'],
    },
    'parse_citation_to_bib': {
        'tags': ['output'],
    },
    'partial_corr_pvalue': {
        'tags': ['causal'],
    },
    'partial_identification': {
        'tags': ['causal'],
    },
    'pate': {
        'example': 'sp.pate( data_experiment=df_rct, data_target=df_pop, y="outcome", treatment="treated", covariates=["age", "edu", "income"], method="aipw", )',
        'tags': ['inference'],
    },
    'pc_algorithm': {
        'example': "sp.pc_algorithm(df, variables=['X', 'Z', 'M', 'Y'])",
        'tags': ['causal'],
    },
    'pcmci': {
        'example': "sp.pcmci(df_ts, variables=['gdp', 'inflation', 'rates'], tau_max=4, pc_alpha=0.01)",
        'tags': ['causal'],
    },
    'peer_effects': {
        'tags': ['interference'],
    },
    'poisson': {
        'example': 'sp.poisson("num_awards ~ math + prog", data=df)',
        'tags': ['regression'],
    },
    'policy_tree': {
        'tags': ['causal', 'policy_learning'],
    },
    'policy_value': {
        'tags': ['causal', 'policy_learning'],
    },
    'policy_weight_ate': {
        'tags': ['bayes', 'policy_learning'],
    },
    'policy_weight_marginal': {
        'tags': ['bayes', 'policy_learning'],
    },
    'policy_weight_observed_prte': {
        'reference': 'carneiro2011estimating',
        'tags': ['bayes', 'policy_learning'],
    },
    'policy_weight_prte': {
        'tags': ['bayes', 'policy_learning'],
    },
    'policy_weight_subsidy': {
        'tags': ['bayes', 'policy_learning'],
    },
    'postestimation_contract': {
        'tags': ['postestimation'],
    },
    'postestimation_report': {
        'tags': ['postestimation'],
    },
    'power': {
        'example': 'sp.power("did", n=1000, effect_size=0.1, n_periods=10, n_treated_periods=5)',
        'tags': ['power'],
    },
    'power_cluster_rct': {
        'tags': ['power'],
    },
    'power_did': {
        'tags': ['power'],
    },
    'power_iv': {
        'tags': ['power'],
    },
    'power_ols': {
        'tags': ['power'],
    },
    'power_rct': {
        'tags': ['power'],
    },
    'power_rd': {
        'tags': ['power'],
    },
    'ppmlhdfe': {
        'example': 'sp.ppmlhdfe("trade ~ dist + contig | origin + dest + year", data=df, cluster="pair_id")',
        'tags': ['regression'],
    },
    'predict_cate': {
        'tags': ['causal'],
    },
    'preflight': {
        'example': "sp.preflight(df, 'did', y='y', treat='treated', time='t')",
        'tags': ['smart'],
    },
    'pretrends_power': {
        'example': 'sp.pretrends_power(result)',
        'reference': 'roth2022pretest',
        'tags': ['causal'],
    },
    'pretrends_summary': {
        'tags': ['causal'],
    },
    'prevalence_ratio': {
        'tags': ['epi'],
    },
    'probit': {
        'example': 'sp.probit("admit ~ gre + gpa + rank", data=df)',
        'tags': ['regression'],
    },
    'propensity_score': {
        'tags': ['causal'],
    },
    'ps_balance': {
        'tags': ['causal'],
    },
    'psm': {
        'tags': ['causal'],
    },
    'psplot': {
        'example': "sp.psplot(df, treat='D', covariates=['x1', 'x2'])",
        'tags': ['causal'],
    },
    'pub_ready': {
        'example': "sp.pub_ready(results=[r1, r2], venue='top5_econ', design='did')",
        'tags': ['smart'],
    },
    'pwcompare': {
        'example': 'sp.pwcompare(result, data=df, variable="group", adjust="bonferroni")',
        'tags': ['postestimation'],
    },
    'pwcorr': {
        'example': "sp.pwcorr(df, vars=['wage', 'education', 'experience'])",
        'tags': ['utils'],
    },
    'qqsynth': {
        'example': "sp.qqsynth(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal'],
    },
    'qte_hd_panel': {
        'tags': ['causal'],
    },
    'queen_weights': {
        'tags': ['spatial'],
    },
    'rake': {
        'tags': ['survey'],
    },
    'randomize': {
        'example': "sp.randomize(df, strata='district', balance_vars=['age', 'income'])",
        'tags': ['experimental'],
    },
    'rank': {
        'tags': ['utils'],
    },
    'rate': {
        'tags': ['causal'],
    },
    'rd2d': {
        'tags': ['causal'],
    },
    'rd2d_bw': {
        'tags': ['causal'],
    },
    'rd2d_plot': {
        'tags': ['causal'],
    },
    'rd_bayes_hte': {
        'tags': ['causal', 'rd', 'bayesian'],
    },
    'rd_boost': {
        'tags': ['causal', 'rd'],
    },
    'rd_cate_summary': {
        'tags': ['causal', 'rd'],
    },
    'rd_compare': {
        'tags': ['causal', 'rd'],
    },
    'rd_dashboard': {
        'tags': ['causal', 'rd'],
    },
    'rd_distribution': {
        'tags': ['causal', 'rd'],
    },
    'rd_distributional_design': {
        'tags': ['causal', 'rd'],
    },
    'rd_external_validity': {
        'tags': ['causal', 'rd'],
    },
    'rd_extrapolate': {
        'reference': 'angrist2015wanna',
        'tags': ['causal', 'rd'],
    },
    'rd_forest': {
        'tags': ['causal', 'rd'],
    },
    'rd_interference': {
        'tags': ['causal', 'rd', 'interference'],
    },
    'rd_lasso': {
        'tags': ['causal', 'rd'],
    },
    'rd_multi_extrapolate': {
        'reference': 'cattaneo2021extrapolating',
        'tags': ['causal', 'rd'],
    },
    'rd_multi_score': {
        'tags': ['causal', 'rd'],
    },
    'rd_robustness_table': {
        'tags': ['causal', 'rd'],
    },
    'rdbalance': {
        'tags': ['causal'],
    },
    'rdbwhte': {
        'tags': ['causal'],
    },
    'rdbwselect': {
        'example': "sp.rdbwselect(df, y='outcome', x='score', c=0)",
        'reference': 'calonico2020optimal',
        'tags': ['causal', 'rd'],
    },
    'rdbwsensitivity': {
        'tags': ['causal'],
    },
    'rdd': {
        'tags': ['causal'],
    },
    'rddensity': {
        'example': "sp.rddensity(df, x='score', c=0)",
        'tags': ['diagnostics'],
    },
    'rdhte': {
        'tags': ['causal'],
    },
    'rdhte_lincom': {
        'tags': ['causal'],
    },
    'rdit': {
        'example': 'sp.rdit(df, y="electricity", time="date", cutoff="2015-01-01", seasonality="month")',
        'tags': ['causal'],
    },
    'rdmc': {
        'example': "sp.rdmc(df, y='score', x='running_var', cutoffs=[50, 70, 90])",
        'tags': ['causal'],
    },
    'rdms': {
        'example': "sp.rdms(df, y='outcome', x1='dist_lat', x2='dist_lon')",
        'tags': ['causal'],
    },
    'rdplacebo': {
        'tags': ['causal'],
    },
    'rdplot': {
        'tags': ['causal', 'rd'],
    },
    'rdplotdensity': {
        'reference': 'cattaneo2020simple',
        'tags': ['causal', 'rd'],
    },
    'rdpower': {
        'tags': ['causal'],
    },
    'rdrandinf': {
        'reference': 'cattaneo2016inference',
        'tags': ['causal'],
    },
    'rdrbounds': {
        'tags': ['causal'],
    },
    'rdsampsi': {
        'tags': ['causal'],
    },
    'rdsensitivity': {
        'tags': ['causal'],
    },
    'rdsummary': {
        'tags': ['causal'],
    },
    'rdwinselect': {
        'tags': ['causal'],
    },
    'read_data': {
        'example': "sp.read_data('survey.dta')",
        'tags': ['utils'],
    },
    'recommend': {
        'example': "sp.recommend(df, y='wage', treatment='training', id='worker', time='year')",
        'tags': ['smart'],
    },
    'render_agent_block': {
        'tags': ['agent'],
    },
    'render_agent_blocks': {
        'tags': ['agent'],
    },
    'replicate': {
        'example': "sp.replicate('card_1995')",
        'tags': ['smart'],
    },
    'reproduce_jss_tables': {
        'tags': ['validation'],
    },
    'reset_test': {
        'reference': 'ramsey1969tests',
        'tags': ['diagnostics'],
    },
    'ri_test': {
        'example': "sp.ri_test(df, y='outcome', treat='treatment', n_perms=5000, seed=42)",
        'tags': ['inference'],
    },
    'rif_decomposition': {
        'tags': ['decomposition'],
    },
    'rifreg': {
        'tags': ['decomposition'],
    },
    'rkd': {
        'tags': ['causal'],
    },
    'robust_synth': {
        'example': "sp.robust_synth(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989, variant='unconstrained')",
        'tags': ['causal'],
    },
    'robustness_report': {
        'example': 'sp.robustness_report( data=df, formula="wage ~ education + experience", x=\'education\', cluster_var=\'region\', extra_controls=[\'female\', \'age\'], winsor_levels=[0.01, 0.05], )',
        'tags': ['robustness'],
    },
    'romano_wolf': {
        'example': 'sp.romano_wolf( data=df, y=["wage", "hours", "employment", "benefits"], x=["treatment"], controls=["age", "education", "experience"], n_boot=1000, seed=42, )',
        'tags': ['inference'],
    },
    'rook_weights': {
        'tags': ['spatial'],
    },
    'rowcount': {
        'tags': ['utils'],
    },
    'rowmax': {
        'tags': ['utils'],
    },
    'rowmean': {
        'tags': ['utils'],
    },
    'rowmin': {
        'tags': ['utils'],
    },
    'rowsd': {
        'tags': ['utils'],
    },
    'rowtotal': {
        'tags': ['utils'],
    },
    'sac': {
        'tags': ['spatial'],
    },
    'sar_gmm': {
        'tags': ['spatial'],
    },
    'sarar_gmm': {
        'tags': ['spatial'],
    },
    'sbw': {
        'example': "sp.sbw(df, treat='D', covariates=['age', 'educ', 'race'], y='wage', delta=0.02)",
        'tags': ['causal'],
    },
    'sc_estimate': {
        'tags': ['causal'],
    },
    'scalar_iv_projection': {
        'example': "sp.scalar_iv_projection( df, treat='schooling', instruments=['quarter_of_birth', 'distance_to_college'], covariates=['age', 'parent_edu'], )",
        'tags': ['utils'],
    },
    'scdata': {
        'example': "sp.scdata(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal'],
    },
    'scest': {
        'example': "sp.scest(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal'],
    },
    'scpi': {
        'example': "sp.scpi(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal'],
    },
    'sdid': {
        'example': "sp.sdid(df, y='packspercapita', unit='state', time='year', treat_unit='California', treat_time=1989)",
        'tags': ['causal'],
    },
    'search_functions': {
        'example': "sp.search_functions('treatment effect')",
        'tags': ['agent'],
    },
    'select_pci_proxies': {
        'tags': ['causal'],
    },
    'selection_bounds': {
        'example': 'sp.selection_bounds( data=df, y="wage", treatment="trained", selection="employed", covariates=["age", "education"], method="conditional", )',
        'tags': ['causal'],
    },
    'sem_gmm': {
        'tags': ['spatial'],
    },
    'sensitivity_dashboard': {
        'example': 'sp.sensitivity_dashboard(result, data=df)',
        'tags': ['smart'],
    },
    'sensitivity_plot': {
        'example': 'sp.honest_did(result, e=0)',
        'tags': ['causal'],
    },
    'session': {
        'example': 'sp.session(seed=42)',
        'tags': ['smart'],
    },
    'set_theme': {
        'example': "sp.set_theme('academic')",
        'tags': ['plots'],
    },
    'shapley_inequality': {
        'tags': ['decomposition'],
    },
    'shift_share_se': {
        'example': 'sp.shift_share_se(iv_res, shares=S)',
        'tags': ['bartik'],
    },
    'slx': {
        'tags': ['spatial'],
    },
    'source_decompose': {
        'tags': ['decomposition'],
    },
    'spatial_did': {
        'tags': ['spatial'],
    },
    'spatial_iv': {
        'tags': ['spatial'],
    },
    'spatial_panel': {
        'tags': ['spatial'],
    },
    'sqreg': {
        'example': "sp.sqreg(df, y='wage', x=['education', 'experience'])",
        'tags': ['regression'],
    },
    'ssaggregate': {
        'example': 'sp.ssaggregate( data=df, y="employment_growth", x="bartik_instrument", shares=shares_matrix, shocks="industry_growth", shock_data=df_shocks, controls=["population", "density"], )',
        'tags': ['bartik'],
    },
    'stabilized_weights': {
        'tags': ['causal'],
    },
    'staggered_synth': {
        'example': "sp.staggered_synth(df, outcome='gdp', unit='state', time='year', treatment='treated')",
        'tags': ['causal', 'staggered'],
    },
    'stepwise': {
        'tags': ['regression'],
    },
    'stochastic_dominance': {
        'example': 'sp.stochastic_dominance(result, order=1)',
        'tags': ['causal'],
    },
    'structural_break': {
        'example': "sp.structural_break(df, y='gdp_growth', x=['inflation'])",
        'tags': ['timeseries'],
    },
    'subcluster_wild_bootstrap': {
        'tags': ['inference'],
    },
    'subgroup_analysis': {
        'example': 'sp.subgroup_analysis( data=df, formula="wage ~ education + experience", x=\'education\', by={\'Gender\': \'female\', \'Region\': \'region\'}, )',
        'tags': ['robustness'],
    },
    'subgroup_decompose': {
        'tags': ['decomposition'],
    },
    'sumstats': {
        'example': "sp.sumstats(df, vars=['wage', 'edu', 'exp'], by='female')",
        'tags': ['output'],
    },
    'super_learner': {
        'tags': ['causal'],
    },
    'sureg': {
        'example': "sp.sureg( equations={ 'demand': ('quantity', ['price', 'income']), 'supply': ('quantity', ['price', 'cost']), }, data=df, )",
        'tags': ['regression'],
    },
    'survival_sensitivity': {
        'reference': 'hu2025nonparametric',
        'tags': ['robustness'],
    },
    'survivor_average_causal_effect': {
        'tags': ['causal'],
    },
    'survreg': {
        'example': 'sp.survreg("time ~ age + treatment", data=df, event="status", dist="weibull")',
        'tags': ['survival'],
    },
    'svyglm': {
        'tags': ['survey'],
    },
    'svymean': {
        'tags': ['survey'],
    },
    'svytotal': {
        'tags': ['survey'],
    },
    'synth_compare': {
        'example': "sp.synth_compare( df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989, )",
        'tags': ['causal', 'synth'],
    },
    'synth_donor_sensitivity': {
        'example': "sp.synth_donor_sensitivity(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989, n_samples=200, seed=42)",
        'tags': ['causal', 'synth'],
    },
    'synth_loo': {
        'example': "sp.synth_loo(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal', 'synth'],
    },
    'synth_mde': {
        'example': "sp.synth_mde( df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989, seed=42, )",
        'tags': ['causal', 'synth'],
    },
    'synth_power': {
        'example': "sp.synth_power( df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989, n_simulations=500, seed=42, )",
        'tags': ['causal', 'synth'],
    },
    'synth_power_plot': {
        'example': 'sp.synth_power_plot(power_df)',
        'tags': ['causal', 'synth'],
    },
    'synth_recommend': {
        'example': "sp.synth_recommend( df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989, )",
        'tags': ['causal', 'synth'],
    },
    'synth_report': {
        'example': "sp.synth_report( df, outcome='cigsale', unit='state', time='year', treated_unit='California', treatment_time=1989, )",
        'tags': ['causal', 'synth'],
    },
    'synth_report_to_file': {
        'example': "sp.synth_report_to_file( df, outcome='cigsale', unit='state', time='year', treated_unit='California', treatment_time=1989, filename='california_scm.md', )",
        'tags': ['causal', 'synth'],
    },
    'synth_rmspe_filter': {
        'example': "sp.synth_rmspe_filter(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal', 'synth'],
    },
    'synth_sensitivity': {
        'example': "sp.synth_sensitivity(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989, n_donor_samples=200, seed=42)",
        'tags': ['causal', 'synth'],
    },
    'synth_sensitivity_plot': {
        'example': 'sp.synth_sensitivity_plot(sens)',
        'tags': ['causal', 'synth'],
    },
    'synth_time_placebo': {
        'example': "sp.synth_time_placebo(df, outcome='gdp', unit='state', time='year', treated_unit='California', treatment_time=1989)",
        'tags': ['causal', 'synth'],
    },
    'synth_to_excel': {
        'tags': ['causal', 'synth'],
    },
    'synth_to_latex': {
        'example': 'sp.synth_to_latex(result, show_weights=True)',
        'tags': ['causal', 'synth'],
    },
    'synth_to_markdown': {
        'tags': ['causal', 'synth'],
    },
    'synthdid_estimate': {
        'tags': ['causal', 'did'],
    },
    'synthdid_placebo': {
        'tags': ['causal', 'did'],
    },
    'synthdid_plot': {
        'tags': ['causal', 'did'],
    },
    'synthdid_rmse_plot': {
        'tags': ['causal', 'did'],
    },
    'synthdid_units_plot': {
        'tags': ['causal', 'did'],
    },
    'synthplot': {
        'example': 'sp.synthplot(result)',
        'tags': ['causal'],
    },
    'tF_adjustment': {
        'tags': ['causal'],
    },
    'tF_critical_value': {
        'example': 'sp.tF_critical_value(10.0)',
        'tags': ['diagnostics'],
    },
    'tab': {
        'example': "sp.tab(df, 'treatment', 'outcome')",
        'tags': ['output'],
    },
    'target_trial_emulate': {
        'tags': ['target_trial'],
    },
    'te_rank': {
        'tags': ['frontier'],
    },
    'te_summary': {
        'tags': ['frontier'],
    },
    'test': {
        'example': 'sp.test(result, "x1 = x2")',
        'tags': ['postestimation'],
    },
    'test_calibration': {
        'tags': ['causal'],
    },
    'three_sls': {
        'example': "sp.three_sls( equations={ 'demand': ('q', ['p', 'income'], []), 'supply': ('q', ['p', 'cost'], []), }, data=df, instruments=['income', 'cost', 'weather'], )",
        'tags': ['regression'],
    },
    'translog_design': {
        'example': 'sp.frontier(df_tl, y="log_y", x=terms)',
        'tags': ['frontier'],
    },
    'transport_generalize': {
        'tags': ['transport'],
    },
    'treatment_rollout_plot': {
        'tags': ['causal'],
    },
    'trimming': {
        'tags': ['causal'],
    },
    'truncreg': {
        'example': "sp.truncreg(df, y='wage', x=['education', 'experience'], ll=0)",
        'tags': ['regression'],
    },
    'twfe_decomposition': {
        'example': "sp.twfe_decomposition(df, y='y', group='unit', time='period', first_treat='first_treat')",
        'tags': ['causal'],
    },
    'twoway_cluster': {
        'example': 'sp.twoway_cluster(result, data=df, cluster1="firm", cluster2="year")',
        'tags': ['inference'],
    },
    'use_chinese': {
        'example': 'sp.use_chinese()',
        'tags': ['plots'],
    },
    'validation_report': {
        'tags': ['validation'],
    },
    'var': {
        'example': "sp.var(df, variables=['gdp', 'inflation', 'interest_rate'], lags=2)",
        'tags': ['timeseries'],
    },
    'verify': {
        'tags': ['smart'],
    },
    'verify_benchmark': {
        'tags': ['smart'],
    },
    'verify_recommendation': {
        'tags': ['smart'],
    },
    'vif': {
        'tags': ['diagnostics'],
    },
    'weakrobust': {
        'example': "sp.weakrobust(df, y='wage', endog='educ', instruments=['nearc2','nearc4'], exog=['age','exper'])",
        'reference': 'anderson1949estimation',
        'tags': ['diagnostics'],
    },
    'weighted_conformal_prediction': {
        'tags': ['conformal_causal', 'conformal'],
    },
    'wild_cluster_boot': {
        'example': 'sp.wild_cluster_boot(result, data=df, cluster="state", variable="x1", n_boot=999)',
        'tags': ['inference'],
    },
    'wild_cluster_ci_inv': {
        'tags': ['inference'],
    },
    'winsor': {
        'example': "sp.winsor(df, vars=['wage', 'income'], cuts=(1, 99))",
        'tags': ['utils'],
    },
    'write_bib': {
        'tags': ['output'],
    },
    'xlearner': {
        'tags': ['causal', 'metalearners'],
    },
    'xtfrontier': {
        'tags': ['frontier'],
    },
    'yu_elwert_decompose': {
        'tags': ['decomposition'],
    },
    'yun_nonlinear': {
        'tags': ['decomposition'],
    },
    'zinb': {
        'example': "sp.zinb(data=df, y='doctor_visits', x=['age', 'income'], inflate=['age', 'chronic'])",
        'tags': ['regression'],
    },
    'zip_model': {
        'example': "sp.zip_model(data=df, y='doctor_visits', x=['age', 'income'], inflate=['age', 'chronic'])",
        'tags': ['regression'],
    },
    'zisf': {
        'tags': ['frontier'],
    },
}


def apply(registry: Dict[str, Any]) -> None:
    """Fill empty Tier-B fields on registered specs.

    ``registry`` is the live ``_REGISTRY`` dict from
    :mod:`statspai.registry`.  We mutate :class:`FunctionSpec`
    instances in place but only when the target field is empty —
    curated specs are never overwritten.

    Idempotent: running twice has no further effect because once a
    field is populated, the empty check stops re-application.
    """
    for name, enrich in BASELINE_CARDS.items():
        spec = registry.get(name)
        if spec is None:
            continue
        ex = enrich.get('example')
        if ex and not spec.example:
            spec.example = ex
        ref = enrich.get('reference')
        if ref and not spec.reference:
            spec.reference = ref
        tags = enrich.get('tags')
        if tags and not spec.tags:
            spec.tags = list(tags)
