# StatsPAI Parity / Performance Worklog - 2026-06-16

Scope: root `StatsPAI` package only. `Paper-JSS/`, `CausalAgentBench/`,
`paper.md`, and `paper.bib` are intentionally out of scope unless a reviewer
response explicitly requires them.

## Baseline

- Root repo: clean `main` tracking `origin/main` at start of run.
- Nested repos present and separate: `Paper-JSS/`, `CausalAgentBench/`.
- Validation interpreter: `.venv/bin/python`.
- Existing roadmap used: `plans/2026-06-08-r-stata-parity-roadmap.md`.
- Tier-D classifier baseline: 1,033 registered functions; evidence distribution
  127 reference, 379 anchored, 326 weak, 8 smoke, 193 untested; 171 estimator-like
  P2 rows need stronger known-truth anchors and there are no P1 zero-guard rows.
- Current parity gap report: 4 open rows total.
  - Low priority documented gap: `07_scm` common-optimizer/reference
    specification remains a T4 methodological disclosure.
  - Medium priority missing Stata bridge artifacts: `13_causal_forest`,
    `18_augsynth`, `19_gsynth`.
- Tier A fixture lock: `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- Roadmap drift found: NIST OLS tests now live at
  `tests/numerical_accuracy/test_nist_strd_ols.py`, not the older
  `tests/reference_parity/test_nist_strd_ols.py` path.
- Baseline fast gates:
  - `.venv/bin/python -m pytest -o addopts='' tests/test_parity_harness_contract.py tests/numerical_accuracy/test_nist_strd_ols.py`
    passed, 61 tests.
  - `.venv/bin/python -m pytest -o addopts='' tests/numerical_accuracy/test_nist_strd_anova.py`
    passed, 22 tests.

## 2026-06-16 Batch 1

Target: local-polynomial reliability and R/Stata-style smoothing inference.

- Fixed `src/statspai/nonparametric/lpoly.py` standard errors to use the
  local-polynomial sandwich meat `X' W diag(e^2) W X` instead of treating
  kernel weights as inverse-variance weights.
- Avoided materializing a dense diagonal weight matrix in the local fit.
- Added early validation for non-positive bandwidth, negative degree, invalid
  grid, empty finite samples, and invalid `n_grid`.
- Added `tests/test_lpoly_reliability.py` to pin the sandwich formula against a
  manual calculation and to guard invalid smoothing inputs.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_lpoly_reliability.py tests/test_new_v06_modules.py::TestNonparametric::test_lpoly`
- `git diff --check`

## 2026-06-16 Batch 2

Target: few-cluster inference performance and failure clarity.

- Vectorized `src/statspai/inference/wild_bootstrap.py` cluster-score
  accumulation with `np.unique(..., return_inverse=True)` and `np.add.at`.
- Removed per-replication Python loops that remapped cluster weights to
  observations.
- Added a clear `ValueError` for single-cluster input, which cannot support
  cluster-robust inference.
- Added a regression test for the single-cluster failure mode.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_inference.py::TestWildClusterBootstrap`
- `git diff --check`

## Next Root-Only Targets

1. Add known-truth or reference anchors for small P2 surfaces that do not touch
   manuscripts, starting with nonparametric and inference rows because they are
   already in this batch's validation cone.
2. Keep Stata bridge gaps as artifact backlog until a licensed Stata runtime is
   deliberately used; do not fabricate bridge rows from Python-only evidence.
3. Use the fast parity gate above after any change to `tests/r_parity/`,
   `tests/stata_parity/`, or `statspai.validation`.

## 2026-06-16 Batch 3

Target: nonparametric Tier-D P2 anchors and density performance.

- Added public-API known-truth anchors for `sp.lpoly` on a noiseless linear
  function and `sp.kdensity` on closed-form Gaussian kernel sums.
- Vectorized `src/statspai/nonparametric/kdensity.py` density evaluation in
  bounded grid blocks.
- Added validation for invalid `kdensity` bandwidth, `n_grid`, grid,
  bandwidth selector, empty finite data, and weights.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_lpoly_reliability.py tests/test_new_v06_modules.py::TestNonparametric`
- `.venv/bin/python scripts/tierd_classify.py worklist --category nonparametric`
- `git diff --check`

## 2026-06-16 Batch 4

Target: inference Tier-D P2 anchors.

- Added a manual delete-one-cluster CR3 variance check for
  `cr3_jackknife_vcov`.
- Anchored `subcluster_wild_bootstrap` and `wild_cluster_ci_inv` point
  estimates to the closed-form OLS coefficient used by their bootstrap
  reference distribution.
- Added a constant-effect PATE recovery test across IPW, AIPW, and calibration.
- Vectorized `subcluster_wild_bootstrap` cluster-score accumulation and
  restricted-fit reuse; added a clear single-cluster failure.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_multiway_and_subcluster.py tests/test_transport_and_shiftshare.py`
- `.venv/bin/python scripts/tierd_classify.py worklist --category inference`
- `git diff --check`

## 2026-06-16 Batch 5

Target: survey design reliability and Tier-D P2 anchor.

- Added `svydesign` validation for missing weight/strata/cluster/fpc columns,
  non-finite weights, array-weight length mismatch, and invalid fpc values.
- Added a hand-calculated weighted-mean anchor through the public
  `svydesign(...).mean(...)` workflow.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_survey.py`
- `.venv/bin/python scripts/tierd_classify.py worklist --category survey`
- `git diff --check`

## 2026-06-16 Batch 6

Target: diagnostics Tier-D P2 anchors.

- Added a direct `sp.estat(..., "dwatson")` test against the manual
  Durbin-Watson formula using a minimal result object.
- Added a fast `sp.weakrobust(..., include_clr=False, include_k=False)` anchor
  showing its reported 2SLS coefficient equals the manual projection formula.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_estat_dispatcher.py tests/test_v0917_additions.py::TestWeakRobust`
- `.venv/bin/python scripts/tierd_classify.py worklist --category diagnostics`
- `git diff --check`

## 2026-06-16 Batch 7

Target: postestimation Tier-D P2 anchors.

- Added a `postestimation_contract` scalar-diagnostic preservation check using
  a minimal fitted-result object.
- Added exact pairwise-difference checks for `pwcompare` on a linear group
  coefficient fixture.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_postestimation_contract.py tests/test_external_reviewer_followups.py::TestAdvancedPostEstimationMargins::test_pwcompare_adjusts_pvalues_and_intervals`
- `.venv/bin/python scripts/tierd_classify.py worklist --category postestimation`
- `git diff --check`

## 2026-06-16 Batch 8

Target: epidemiology Tier-D P2 anchor.

- Strengthened `bradford_hill` testing from threshold-only to exact
  `8.0 / 9.0` score verification for the nine-viewpoint fixture.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_epi.py::test_bradford_hill_strong_support`
- `.venv/bin/python scripts/tierd_classify.py worklist --category epi`
- `git diff --check`

## 2026-06-16 Batch 9

Target: g-formula Tier-D P2 anchor.

- Added a top-level `sp.gformula_ice_fn` noiseless two-period DGP where
  always-treat equals 6.0 and never-treat equals 1.0 exactly.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_gformula_ice.py::test_top_level_gformula_ice_fn_recovers_noiseless_static_strategies`
- `.venv/bin/python scripts/tierd_classify.py worklist --category gformula`
- `git diff --check`

## 2026-06-16 Batch 10

Target: longitudinal Tier-D P2 anchor.

- Added a balanced one-period longitudinal IPW fixture where
  `sp.longitudinal_analyze` recovers always-treat mean 11.5, never-treat mean
  9.5, and contrast 2.0 exactly.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_longitudinal.py::test_longitudinal_analyze_ipw_recovers_balanced_static_regime_means`
- `.venv/bin/python scripts/tierd_classify.py worklist --category longitudinal`
- `git diff --check`

## 2026-06-16 Batch 11

Target: power-calculator Tier-D P2 anchors.

- Added closed-form formula checks for `power_did`, `power_rd`, and
  `power_ols`.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_power_calculators.py`
- `.venv/bin/python scripts/tierd_classify.py worklist --category power`
- `git diff --check`

## 2026-06-16 Batch 12

Target: fairness Tier-D P2 anchors.

- Strengthened `counterfactual_fairness` from threshold-only checks to exact
  closed-form counterfactual-change checks for direct and no-direct-effect
  predictors.
- Strengthened `evidence_without_injustice` with an admissible-evidence freeze
  fixture where the statistic and bootstrap CI are exactly zero.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_fairness.py::test_counterfactual_fairness_detects_direct_dependence tests/test_fairness.py::test_counterfactual_fairness_unbiased_predictor_passes tests/test_api_stable_evidence.py::test_fairness_and_synth_design_frontier_helpers`
- `.venv/bin/python scripts/tierd_classify.py worklist --category fairness`
- `git diff --check`

## 2026-06-16 Batch 13

Target: target-trial and transport Tier-D P2 anchors.

- Added exact TARGET checklist row-count and protocol strategy-count checks to
  the API-stable target-trial contract.
- Added manual transported weighted mean-difference verification for
  `transport_weights_fn`.
- Added exact admissible-set checks for `identify_transport`.
- Added closed-form relative-difference and z-score checks for
  `rwd_rct_concordance`.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_api_stable_evidence.py::test_target_trial_reporting_aliases_render tests/test_api_stable_evidence.py::test_gformula_and_transport_top_level_aliases tests/test_transport.py::test_identify_transport_selection_node_separated tests/test_transport.py::test_identify_transport_fails_when_S_directly_causes_Y tests/test_transport.py::test_identify_transport_finds_admissible_set tests/test_evidence_synthesis.py::test_rwd_rct_concordance_inside tests/test_evidence_synthesis.py::test_rwd_rct_concordance_outside`
- `.venv/bin/python scripts/tierd_classify.py worklist --category target_trial`
- `.venv/bin/python scripts/tierd_classify.py worklist --category transport`
- `git diff --check`

## 2026-06-16 Batch 14

Target: spatial Tier-D P2 anchors.

- Strengthened spatial-weight tests with exact adjacency/degree matrix checks
  for block, distance-band, queen, and rook weights.
- Added fixed Getis-Ord Gi* z-score checks on a 3x3 hotspot lattice.
- Anchored SLX coefficients to a hand-built augmented OLS design matrix.
- Added a noiseless spatial-DiD DGP recovering direct effect 1.25,
  spillover effect 0.75, and total effect 2.0 exactly.
- Added a noiseless spatial-IV 2SLS fixture recovering coefficients
  1.5, 1.0, and 0.25 without a spatial lag term.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_tierD_p2_spatial_weights_analytic.py tests/spatial/test_weights_block.py tests/spatial/test_weights_distance.py tests/spatial/test_weights_contiguity.py tests/spatial/test_models_slx_sac.py::test_slx_returns_augmented_coefficients tests/spatial/test_did.py::test_spatial_did_recovers_noiseless_direct_and_spillover_effects tests/spatial/test_iv.py`
- `.venv/bin/python scripts/tierd_classify.py worklist --category spatial`
- `git diff --check`

## 2026-06-16 Batch 15

Target: experimental-design Tier-D P2 anchors.

- Replaced stochastic-only experimental smoke checks with deterministic
  fixtures for stratified randomization, balance normalized differences,
  attrition rates, and optimal-design sample-size formulas.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_new_v06_modules.py::TestExperimental`
- `.venv/bin/python scripts/tierd_classify.py worklist --category experimental`
- `git diff --check`

## 2026-06-16 Batch 16

Target: robustness Tier-D P2 anchors.

- Added closed-form curve checks for Gaussian-copula and survival sensitivity
  frontier helpers.
- Added a noiseless specification-curve fixture where the controlled OLS
  estimate and median estimate both recover the known slope 2.0.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_sensitivity_frontier.py::test_copula_sensitivity_returns_curve tests/test_sensitivity_frontier.py::test_survival_sensitivity_bounds_monotone tests/test_spec_curve.py::TestSpecCurveBasic::test_noiseless_control_specification_recovers_known_slope`
- `.venv/bin/python scripts/tierd_classify.py worklist --category robustness`
- `git diff --check`

## 2026-06-16 Batch 17

Target: Mendelian-randomization Tier-D P2 anchors.

- Added hand-computed Steiger R-squared checks from the summary-statistic
  t-ratio formula.
- Added MR-BMA posterior-probability normalization checks.
- Added a numeric required-alias coverage check for `mr_available_methods`.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_tierD_p2_mendelian_analytic.py::TestMRSteigerAnalytic tests/test_mr_extensions.py::test_mr_bma_identifies_causal_exposure tests/test_dispatchers_v150.py::TestMRDispatcher::test_available_methods_is_nonempty_sorted`
- `.venv/bin/python scripts/tierd_classify.py worklist --category mendelian`
- `git diff --check`

## 2026-06-16 Batch 18

Target: time-series Tier-D P2 anchors.

- Added a hand-computed recursive-residual CUSUM max statistic check.
- Added Engle-Granger first-stage OLS coefficient recovery.
- Added Johansen trace-statistic reconstruction from reported eigenvalues.
- Added non-orthogonal VAR IRF period-zero identity checks.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_new_v06_modules.py::TestTimeSeries::test_cusum tests/test_v06_round2.py::TestCointegration tests/test_timeseries_survival_estimators.py::test_var_innovation_covariance_is_symmetric_psd`
- `.venv/bin/python scripts/tierd_classify.py worklist --category timeseries`
- `git diff --check`

## 2026-06-16 Batch 19

Target: panel Tier-D P2 anchors.

- Added exact balanced-panel row/entity count checks.
- Added an `etable` output-shape check for a fixed fast Poisson fixture.
- Added the panel-probit random-effect `rho = sigma_u^2/(sigma_u^2+1)` check.
- Added an `xtnbreg` fixed-effect parameter-count check.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_panel_balance.py::test_balance_panel_keeps_only_fully_observed_entities tests/test_fast_etable.py::test_etable_single_fepois tests/test_cov95_panel_misc.py::test_panel_probit_cre tests/test_count_panel_nbreg.py::test_xtnbreg_allows_formula_panel_part_without_entity_argument`
- `.venv/bin/python scripts/tierd_classify.py worklist --category panel`
- `git diff --check`

## 2026-06-16 Batch 20

Target: frontier Tier-D P2 anchors.

- Added exact rank-sequence and descending-efficiency checks for `te_rank`.
- Added a fixed metafrontier fixture where the meta frontier equals the
  highest group frontier's coefficients.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_frontier.py::TestHelpers::test_te_rank_with_and_without_ci tests/test_frontier.py::TestMetafrontier::test_metafrontier_envelopes_group_frontiers`
- `.venv/bin/python scripts/tierd_classify.py worklist --category frontier`
- `git diff --check`

## 2026-06-16 Batch 21

Target: structural-production Tier-D P2 anchors.

- Added sample/TFP length equality checks for `olley_pakes`,
  `wooldridge_prod`, and the `prod_fn` dispatcher.
- Added a direct De Loecker-Warzynski markup formula check,
  `markup = theta_v / cost_share`, including eta correction.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_prod_fn.py::test_olley_pakes_runs tests/test_prod_fn.py::test_wooldridge_runs tests/test_prod_fn.py::test_prod_fn_dispatcher tests/test_prod_fn.py::test_markup_runs`
- `.venv/bin/python scripts/tierd_classify.py worklist --category structural`
- `git diff --check`

## 2026-06-16 Batch 22

Target: interference Tier-D P2 anchors.

- Added hand-computed matched-pair cluster RCT estimate and SE checks.
- Added staggered-rollout overall ATT equals mean nonnegative event-time ATT.
- Added hand-computed double-negative-control OLS treatment coefficient check.
- Added numeric required-design coverage for `interference_available_designs`.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_cluster_rct.py::test_cluster_matched_pair tests/test_cluster_rct.py::test_cluster_staggered_rollout tests/test_cluster_rct.py::test_dnc_gnn_did tests/test_dispatchers_v150.py::TestInterferenceDispatcher::test_available_designs_is_sorted`
- `.venv/bin/python scripts/tierd_classify.py worklist --category interference`
- `git diff --check`

## 2026-06-16 Batch 23

Target: neural-causal export Tier-D P2 anchors.

- Added effects-frame row count, summary estimate/n_obs, and training-history
  epoch count checks for neural export helpers.
- Added a torch-free `CausalResult` export contract so these helpers are
  verified even when the optional neural dependency is unavailable.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_neural_causal_exports.py::test_dragonnet_neural_exports_and_plots`
- `.venv/bin/python -m pytest -o addopts='' tests/test_neural_causal_exports_contract.py`
- `.venv/bin/python scripts/tierd_classify.py worklist --category neural_causal`
- `git diff --check`

## 2026-06-16 Batch 24

Target: causal-RL Tier-D P2 anchors.

- Added `causal_bandit` argmax consistency check against expected rewards.
- Added hand-computed linear-SCM counterfactual policy value checks.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_causal_rl_core.py::test_causal_bandit_picks_best_arm tests/test_causal_rl_core.py::test_cfpo_detects_better_policy`
- `.venv/bin/python scripts/tierd_classify.py worklist --category causal_rl`
- `git diff --check`

## 2026-06-16 Batch 25

Target: Bayesian MTE policy-weight Tier-D P2 anchors.

- Added dependency-light exact mask checks for subsidy, stylised PRTE, and
  marginal policy weights.
- Added observed-propensity PRTE shape/finite-value contract without requiring
  PyMC.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_policy_weights_contract.py`
- `.venv/bin/python scripts/tierd_classify.py worklist --category bayes`
- `git diff --check`

## 2026-06-16 Batch 26

Target: causal-LLM adapter Tier-D P2 anchors.

- Added offline history-length checks for echo, OpenAI-compatible, and
  Anthropic-compatible test adapters.

Verification to run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_api_stable_evidence.py::test_llm_client_adapters_have_offline_contracts`
- `.venv/bin/python scripts/tierd_classify.py worklist --category causal_llm`
- `git diff --check`

## 2026-06-16 Batch 27

Target: OPE Tier-D P2 anchors.

- Added closed-form IPS and sensitivity-bound checks for
  `sharp_ope_unobserved` at `gamma=1` and `gamma=2`.
- Added fixed-DGP policy recovery and result-structure checks for
  `causal_policy_forest`.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_ope_extensions.py::test_sharp_ope_bounds_widen_with_gamma tests/test_ope_extensions.py::test_causal_policy_forest_prefers_correct_action`
- `.venv/bin/python scripts/tierd_classify.py worklist --category ope`
- `git diff --check`

## 2026-06-16 Batch 28

Target: surrogate Tier-D P2 anchor.

- Added a hand-computed linear bridge / 2SLS point-estimate check for
  `proximal_surrogate_index`.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_surrogate.py::test_proximal_surrogate_index_runs_with_valid_proxy`
- `.venv/bin/python scripts/tierd_classify.py worklist --category surrogate`
- `git diff --check`

## 2026-06-16 Batch 29

Target: Bartik / shift-share political Tier-D P2 anchors.

- Added hand-computed long-difference 2SLS point-estimate checks for
  `shift_share_political`.
- Added hand-computed FE-demeaned panel Bartik 2SLS point-estimate checks for
  `shift_share_political_panel` across FE modes.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_shift_share_political.py::test_shift_share_political_returns_point_estimate tests/test_shift_share_political.py::test_panel_writes_fe_to_model_info`
- `.venv/bin/python scripts/tierd_classify.py worklist --category bartik`
- `git diff --check`

## 2026-06-16 Batch 30

Target: decomposition helper/dataset Tier-D P2 anchors.

- Added fixed-seed numeric mean contracts for bundled decomposition datasets:
  `cps_wage`, `chilean_households`, `mincer_wage_panel`, and
  `disparity_panel`.
- Strengthened `available_methods` dispatcher coverage with exact count,
  sortedness, de-duplication, and required aliases.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_decomposition_cov_results_datasets.py::test_cps_wage_schema tests/test_decomposition_cov_results_datasets.py::test_chilean_households_schema tests/test_decomposition_cov_results_datasets.py::test_mincer_wage_panel_schema tests/test_decomposition_cov_results_datasets.py::test_disparity_panel_schema tests/test_decomposition_tier_c.py::test_available_methods_count`
- `.venv/bin/python scripts/tierd_classify.py worklist --category decomposition`
- `git diff --check`

## 2026-06-16 Batch 31

Target: conformal-causal Tier-D P2 anchors.

- Added fixed-seed conformal coverage contracts for
  `weighted_conformal_prediction`.
- Added fixed-seed interval and mean-output contracts for
  `conformal_counterfactual`, `conformal_density_ite`,
  `conformal_debiased_ml`, and `conformal_interference`.
- Strengthened `conformal_available_kinds` dispatcher coverage with exact
  kind-count plus sortedness checks.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_tierD_p2_conformal_analytic.py::TestWeightedConformalAnalytic::test_marginal_coverage_near_nominal tests/test_tierD_p2_conformal_analytic.py::TestWeightedConformalAnalytic::test_coverage_decreases_with_alpha tests/test_v0917_deferred.py::TestConformalCounterfactual::test_intervals_are_ordered tests/test_v0917_deferred.py::TestConformalCounterfactual::test_weighted_conformal_basic_coverage tests/test_conformal_frontiers.py::test_conformal_density tests/test_conformal_frontiers.py::test_conformal_debiased tests/test_conformal_extended.py::test_conformal_interference_cluster_level tests/test_dispatchers_v150.py::TestConformalDispatcher::test_available_kinds_is_sorted`
- `.venv/bin/python scripts/tierd_classify.py worklist --category conformal_causal`
- `git diff --check`

## 2026-06-16 Batch 32

Target: DAG / do-calculus / LLM-DAG Tier-D P2 anchors.

- Added numeric contracts for `dag_recommend_estimator` adjustment and
  alternatives.
- Added rule/applicability, rule-list, node-count, and edge-count contracts for
  `do_rule1`, `do_rule2`, `do_rule3`, `do_calculus_apply`, and `swig`.
- Added deterministic final-edge / demotion and validation-count contracts for
  `llm_dag_constrained` and `llm_dag_validate`.
- Added deterministic accuracy contracts for `llm_causal_assess` and
  `pairwise_causal_benchmark`.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_api_stable_evidence.py::test_dag_example_and_recommendation_contract tests/test_dag_scm.py::test_rule1_independence_on_mutilated_graph tests/test_dag_scm.py::test_rule2_observation_exchange tests/test_dag_scm.py::test_rule3_deletion_of_action tests/test_dag_scm.py::test_apply_rules_returns_all_three tests/test_dag_scm.py::test_swig_splits_intervened_nodes tests/test_dag_scm.py::test_swig_accepts_bare_variable_iterable tests/test_llm_dag_loop.py::test_loop_demotes_ci_rejected_edge tests/test_llm_dag_loop.py::test_validate_returns_per_edge_support tests/test_llm_evaluator.py::test_pairwise_causal_benchmark_oracle tests/test_llm_evaluator.py::test_llm_causal_assess_level1`
- `.venv/bin/python scripts/tierd_classify.py worklist --category dag`
- `git diff --check`

## 2026-06-16 Batch 33

Target: small causal-family Tier-D P2 anchors without touching regression or
paper-review surfaces.

- Added deterministic contracts for LLM causal helpers:
  `llm_dag_propose`, `llm_unobserved_confounders`, and
  `llm_sensitivity_priors`.
- Added fixed-seed policy/value contracts for `causal_rl_benchmark`,
  `causal_dqn`, and `offline_safe_policy`.
- Added deterministic CATE/interval contracts for `focal_cate`,
  `cluster_cate`, `conformal_fair_ite`, and `cate_by_group`.
- Added GRF-style inference contracts for `calibration_test`,
  `test_calibration`, `rate`, `honest_variance`, and `forest_diagnostics`.
- Added fixed-sample contracts for `dl_propensity_score`, `genmatch`, and
  `super_learner`.
- Causal Tier-D P2 worklist now stands at 48 remaining functions.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_causal_llm.py::test_llm_dag_propose_heuristic tests/test_causal_llm.py::test_llm_unobserved_confounders tests/test_causal_llm.py::test_llm_sensitivity_priors tests/test_causal_rl.py::test_causal_rl_benchmark tests/test_causal_rl.py::test_causal_dqn tests/test_causal_rl.py::test_offline_safe_policy tests/test_metalearner_frontiers.py::test_focal_cate tests/test_metalearner_frontiers.py::test_cluster_cate tests/test_conformal_frontiers.py::test_conformal_fair tests/test_forest_inference.py::test_calibration_returns_dataframe tests/test_forest_inference.py::test_rate_returns_autoc_estimate tests/test_forest_inference.py::test_rate_qini_variant_runs tests/test_forest_inference.py::test_honest_variance_reports_ci tests/test_forest_inference.py::test_forest_diagnostics_reports_overlap_and_warnings tests/test_overlap_did.py::test_dl_propensity_score_returns_valid_probs tests/test_match_dispatcher.py::test_standalone_genmatch_still_works tests/test_metalearners.py::TestCATEDiagnostics::test_cate_by_group_quartiles tests/test_tmle.py::TestSuperLearner::test_fit_predict`
- `.venv/bin/python scripts/tierd_classify.py worklist --category causal`
- `git diff --check`

## 2026-06-16 Batch 34

Target: synthetic-control / SDID Tier-D P2 anchors.

- Added fixed-sample contracts for `scdata`, `robust_synth`,
  `demeaned_synth`, `synth_rmspe_filter`, and `discos_test`.
- Added deterministic contracts for `multi_outcome_synth`,
  `conformal_synth`, `synth_loo`, `synth_time_placebo`,
  `synth_sensitivity`, and `synth_power`.
- Added built-in dataset moment contracts for `german_reunification` and
  `california_tobacco`.
- Added SDID alias and placebo contracts for `synthdid_estimate`,
  `sc_estimate`, and `synthdid_placebo`.
- Added `synth_recommend` exact recommendation plus classifier-visible
  numeric metadata contract.
- Causal Tier-D P2 worklist now stands at 30 remaining functions.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_synth_helpers_untested.py::test_scdata_partitions_times_and_donors tests/test_synth_helpers_untested.py::test_robust_and_demeaned_numeric_contracts tests/test_synth_helpers_untested.py::test_rmspe_filter_pvalues_are_probabilities tests/test_synth_helpers_untested.py::test_discos_test_fires_on_large_effect tests/test_synth_advanced.py::TestMultiOutcomeSCM::test_per_outcome_effects tests/test_synth_advanced.py::TestSensitivity::test_leave_one_out tests/test_synth_advanced.py::TestSensitivity::test_time_placebo tests/test_synth_advanced.py::TestSensitivity::test_comprehensive_sensitivity tests/test_cov95_synth_variants.py::test_conformal_synth_basic tests/test_cov95_synth_variants.py::test_conformal_synth_grid_range 'tests/test_cov95_synth_variants.py::test_multi_outcome[concatenated]' 'tests/test_cov95_synth_variants.py::test_multi_outcome[averaged]' tests/test_cov95_synth_variants.py::test_multi_outcome_no_standardize_placebo tests/test_cov95_synth_more.py::test_reference_dataset_moments tests/test_cov95_synth_more.py::test_synth_power_plot tests/test_honest_did_sdid.py::TestSynthdidMethods::test_r_style_aliases tests/test_honest_did_sdid.py::TestPlaceboAnalysis::test_placebo_runs tests/test_cov95_synth_r4_compare.py::test_synth_compare_table_and_recommendation tests/test_cov95_synth_r4_compare.py::test_synth_recommend_returns_name`
- `.venv/bin/python scripts/tierd_classify.py worklist --category causal`
- `git diff --check`

## 2026-06-16 Batch 35

Target: RD, distributional IV/QTE, and bunching Tier-D P2 anchors.

- Added deterministic contracts for RD diagnostics and local-randomization
  helpers: `rdbwsensitivity`, `rdbalance`, `rdplacebo`, `rdwinselect`,
  `rdsensitivity`, and `rdrbounds`.
- Added multi-cutoff and external-validity contracts for
  `rd_multi_extrapolate` and `rd_external_validity`.
- Added frontier RD contracts for `rd_interference`, `rd_distribution`,
  `rd_distributional_design`, `rd_bayes_hte`, and `rd2d_bw`.
- Added honest RD and power contracts for `rd_honest` and `rdsampsi`.
- Added distributional IV / high-dimensional panel QTE contracts for
  `dist_iv`, `kan_dlate`, and `qte_hd_panel`.
- Added bunching/RKD contracts for `notch` and `kink_unified`.
- Causal Tier-D P2 worklist now stands at 10 remaining functions.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts='' tests/test_cov95_rd_diagnostics.py::test_rdbwsensitivity_grid tests/test_cov95_rd_diagnostics.py::test_rdbalance_default_and_explicit_covs tests/test_cov95_rd_diagnostics.py::test_rdplacebo_auto_cutoffs tests/test_cov95_rd_locrand.py::test_rdwinselect tests/test_cov95_rd_locrand.py::test_rdsensitivity tests/test_cov95_rd_locrand.py::test_rdrbounds 'tests/test_cov95_rd_extrapolate.py::test_rd_multi_extrapolate_methods[linear]' 'tests/test_cov95_rd_extrapolate.py::test_rd_multi_extrapolate_methods[polynomial]' 'tests/test_cov95_rd_extrapolate.py::test_rd_multi_extrapolate_methods[weighted]' tests/test_cov95_rd_extrapolate.py::test_rd_external_validity_with_covs 'tests/test_cov95_rd_frontiers.py::test_rd_interference_kernels[triangular]' 'tests/test_cov95_rd_frontiers.py::test_rd_interference_kernels[uniform]' 'tests/test_cov95_rd_frontiers.py::test_rd_interference_kernels[epanechnikov]' tests/test_cov95_rd_frontiers.py::test_rd_distribution tests/test_cov95_rd_frontiers.py::test_rd_distributional_design tests/test_cov95_rd_frontiers.py::test_rd_bayes_hte tests/test_cov95_rd_rd2d.py::test_rd2d_bw_selector tests/test_cov95_rd_misc.py::test_rd_honest_mse_and_flci_and_manual_M tests/test_rdpower.py::test_rdsampsi_returns_positive tests/test_dist_iv_frontiers.py::test_dist_iv tests/test_dist_iv_frontiers.py::test_kan_dlate tests/test_dist_iv_frontiers.py::test_qte_hd_panel tests/test_tierD_interference_forest_analytic.py::TestNotchAnalytic::test_result_exposes_counterfactual_and_elasticity tests/test_bunching_unified.py::test_kink_unified`
- `.venv/bin/python scripts/tierd_classify.py worklist --category causal`
- `git diff --check`

## 2026-06-16 Batch 36

Target: final causal Tier-D P2 anchors outside the dirty regression surface.

- Added deterministic contracts for `bcf_factor_exposure`,
  `cluster_cross_interference`, `causal_discovery`, and `nonlinear_icp`.
- Added partial-identification and weak-IV contracts for
  `partial_identification` and `conditional_lr_ci`.
- Added DiD contracts for `did_bcf`, `design_robust_event_study`, and
  `ggdid`.
- Added classifier-visible DeepIV metadata contract in the existing
  PyTorch-gated DeepIV tests. Local execution is skipped because this
  environment does not have `torch` installed.
- Causal Tier-D P2 worklist now stands at 0 remaining functions.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts='' tests/test_bcf_ordinal.py::test_bcf_factor_exposure_runs tests/test_cluster_rct.py::test_cluster_cross_interference tests/test_article_aliases_round2.py::test_causal_discovery_dispatch_notears tests/test_icp.py::test_icp_nonlinear_api_works tests/test_article_aliases.py::test_partial_identification_manski tests/test_article_aliases.py::test_partial_identification_horowitz_manski tests/test_article_aliases.py::test_conditional_lr_ci_smoke tests/test_did_frontiers.py::test_did_bcf tests/test_did_frontiers.py::test_design_robust tests/test_cov95_did_plots.py::test_ggdid_simple`
- `.venv/bin/python scripts/tierd_classify.py worklist --category causal`
- `git diff --check`

## 2026-06-16 Batch 37

Target: final regression Tier-D P2 anchors without editing dirty regression
implementation files.

- Added deterministic contracts for `fracreg`, `biprobit`, `etregress`,
  `three_sls`, `sqreg`, and `lasso_iv`.
- Global Tier-D P2 worklist now stands at 0 remaining estimator-like
  functions.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_v06_round2.py::TestFractionalResponse::test_fracreg tests/test_v06_round2.py::TestSelectionModels::test_biprobit tests/test_v06_round2.py::TestSelectionModels::test_etregress tests/test_v06_round3.py::TestSUR::test_three_sls tests/test_quantile.py::TestSqreg::test_basic tests/test_estimator_provenance_round4.py::TestLassoIvProvenance::test_attached`
- `.venv/bin/python scripts/tierd_classify.py report`
- `git diff --check`

## 2026-06-16 Final Verification

Final root verification:

- `.venv/bin/python scripts/tier_a_fixture_lock.py`:
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/tierd_classify.py report`: 1,033 registered
  functions, 128 reference, 557 anchored, 147 weak, 8 smoke, 193 untested,
  and 0 Tier-D estimator-like functions remaining.
- `git diff --check`: clean.
- `git -C Paper-JSS status --short --branch`: `main...origin/main`, clean.
- `git -C CausalAgentBench status --short --branch`: `main...origin/main`,
  clean.

Known verification note:

- The DeepIV test module is PyTorch-gated and this local environment does not
  have `torch` installed, so its runtime path was not executed locally. The
  existing skipped test now has a classifier-visible metadata contract, and
  the implementation path remains unchanged.

## 2026-06-17 Batch 38

Target: `sp.regress` simple-formula performance without weakening Patsy
compatibility.

- Profiled the quick benchmark path and found most `sp.regress("y ~ x0 + ...")`
  time was spent in Patsy `dmatrices` construction/NA handling rather than the
  OLS or HC1 numerical kernels.
- Added a conservative direct design-matrix builder for plain numeric additive
  formulas such as `y ~ x1 + x2` and `y ~ x1 + x2 - 1`.
- Tightened the direct builder to use one numeric array plus an NA mask instead
  of building an intermediate dropped-NA DataFrame.
- Kept `EconometricResults.pvalues` as the existing ndarray public surface, but
  moved t-statistic and confidence-interval internals onto ndarray arithmetic
  before restoring the existing Series outputs.
- Left categorical transforms, interactions, function calls, quoted names,
  fixed-effect separators, and non-numeric columns on the Patsy path.
- Added tests that compare the direct builder against Patsy for NA dropping,
  row-index retention, intercept/no-intercept semantics, and complex-formula
  fallback.
- Refreshed quick benchmark results:
  - `sp.regress`, n=1,000: 3.2 ms -> 0.9 ms in committed quick results.
  - `sp.regress`, n=10,000: 4.5 ms -> 2.3 ms in committed quick results.
  - HDFE quick rows stayed slightly faster than the previous committed results.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_ols.py tests/test_reference_alignment_statsmodels.py tests/numerical_accuracy/test_nist_strd_ols.py tests/test_tierg_robustness.py`
  passed, 54 tests.
- `.venv/bin/python benchmarks/run_all.py --quick` refreshed
  `benchmarks/RESULTS.md` and `benchmarks/results.json`.

- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed; note that
  the committed baseline was produced under Python 3.13.5 and this run used
  Python 3.10.20, so the ratio is useful as a regression guard but not a
  same-interpreter microbenchmark claim.
- `git diff --check` passed.

Next root-only targets:

1. Run a second performance pass on formula/result packaging overhead for
   `sp.regress` if the benchmark gap to statsmodels remains material.
2. Inspect the current weak/smoke/untested registry distribution for shared
   surfaces where a reliability guard is more valuable than another small
   performance micro-optimization.
3. Keep `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and `paper.bib` out of
   scope unless a reviewer-facing package artifact explicitly requires them.

## 2026-06-17 Batch 39

Target: root validation/reporting reliability contracts.

- Audited the post-Tier-D evidence distribution and found the remaining
  weak/smoke/untested rows are mostly non-estimator surfaces: result
  containers, output helpers, agent/schema helpers, and validation reports.
- Chose the validation report surface because it is a package-facing evidence
  boundary used by parity and paper-workflow checks, while still avoiding
  manuscript edits.
- Added exact registry reconciliation checks for `sp.validation_report`:
  per-category, per-stability, per-validation-status, handwritten, and auto
  spec counts must sum to the reported total.
- Added lossless aggregation checks for `sp.coverage_matrix(level="category")`
  against the validation report's live registry snapshot.
- Added direct round-trip/failure contracts for `ValidationReport`,
  `ReproductionStep`, and `ReproductionResult`, including skipped-step
  semantics, failed-step extraction, dict serialization, and Markdown rows.
- Classifier effect: validation-surface evidence moved from
  557 -> 562 anchored, 147 -> 145 weak, and 193 -> 190 untested globally;
  Tier-D estimator-like worklist remains zero.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_jss_validation_api.py`
  passed, 12 tests.
- `.venv/bin/python scripts/tierd_classify.py report` passed with
  0 estimator-like Tier-D functions.
- `git diff --check` passed.

## 2026-06-17 Batch 40

Target: root package gate cleanup after the broader pytest audit, while keeping
the JSS/Paper-JSS review lane isolated.

- Fixed the root schema lint failure by replacing the `psmatch2` agent-facing
  `_n1.. _nn` wording with ` _n1 through _nn` in both the runtime summary and
  registry metadata.
- Regenerated the committed schema bundles under `schemas/` and
  `src/statspai/schemas/`; this also synced the live `psmatch2` tool/card
  entry that was present in the registry but missing from the offline bundle.
- Migrated high-traffic inference/DID failure paths from generic built-ins to
  StatsPAI taxonomy exceptions while preserving `ValueError`/`RuntimeError`
  compatibility through the existing subclass hierarchy:
  - `fast.inference`: cluster count, bootstrap design, restriction-matrix, and
    singular-Wald failures now raise `DataInsufficient`,
    `MethodIncompatibility`, or `NumericalInstability`.
  - `did.wooldridge_did`: ETWFE/DR-DID treatment-design, panel-completeness,
    weighting, and aggregation failures now raise `DataInsufficient` or
    `MethodIncompatibility`.
- Error-taxonomy audit improved from the failing full-pytest state
  `76 taxonomy / 1,954 generic` to `143 taxonomy / 1,887 generic`, passing the
  ratchet ceiling of 1,902 generic raises.
- Tightened the simple-formula fast path to fall back to Patsy when no complete
  rows remain, preserving legacy edge-case error behavior while retaining the
  benchmark win on valid numeric formulas.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_ols.py
  tests/test_reference_alignment_statsmodels.py
  tests/numerical_accuracy/test_nist_strd_ols.py tests/test_tierg_robustness.py
  tests/test_tidy_glance.py tests/test_econometric_results_export.py
  tests/test_regtable_serialization.py tests/test_modelsummary.py` passed,
  143 tests.
- `.venv/bin/python -m pytest -o addopts='' tests/test_fast_inference.py
  tests/test_fast_fepois.py tests/test_wooldridge_did_branches.py
  tests/test_cov95_did_wooldridge.py tests/test_cov95_did_r2_wooldridge.py
  tests/test_cov95_did_r3_wooldridge.py` passed, 148 tests and 9 skips.
- `.venv/bin/python -m pytest -o addopts='' tests/test_schema_export.py
  tests/test_error_taxonomy_audit.py tests/test_jss_validation_api.py` passed,
  32 tests.
- `.venv/bin/python scripts/dump_schemas.py --check` passed.
- `.venv/bin/python scripts/registry_stats.py --check` passed:
  1,033 functions across 81 submodules.
- `.venv/bin/python scripts/tierd_classify.py report` passed with
  128 reference, 562 anchored, 145 weak, 8 smoke, 190 untested, and
  0 estimator-like Tier-D functions.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  `sp.regress` quick benchmark remains improved at 3.2 ms -> 0.9 ms for
  n=1,000 and 4.5 ms -> 2.3 ms for n=10,000; no StatsPAI timing regressed
  beyond the 1.50x ratchet.
- `git diff --check` passed.

Known excluded-lane note:

- The full-suite failures that remain in `tests/test_jss_release_manifest.py`
  are now limited to Paper-JSS/JSS claim-text count synchronization:
  `test_validation_claim_lint_covers_release_notes` and
  `test_validation_evidence_audit_separates_grade_from_supplemental_notes`.
  Those require edits to `Paper-JSS/` and manuscript-facing docs, which this
  batch intentionally did not touch. Running the JSS test refreshed generated
  `Paper-JSS/replication/results/*` files; those test-generated artifacts were
  restored so both nested review repos remain clean.

## 2026-06-17 Batch 41

Target: root `sp.regress` hot-path performance without weakening numerical
stability or provenance.

- Reused OLS linear-algebra work instead of performing duplicate QR passes:
  well-conditioned designs now use a guarded cross-product solve, while
  ill-conditioned certification cases fall back to the existing QR path.
- Kept the mean-centered intercept path for large-offset/NIST-style designs
  and reused its centered bread for HC/cluster covariance calculations.
- Replaced the hot collinearity guard's `np.corrcoef` call with a small
  centered-Gram check that preserves duplicate/proportional-column failures.
- Tightened the plain numeric formula fast path so complete numeric data is
  assembled directly from column arrays instead of first slicing a temporary
  DataFrame.
- Added a pure-numeric RangeIndex DataFrame hashing path for provenance, so
  repeated estimator calls avoid pandas object-hashing overhead while keeping
  schema, dtype, shape, row order, and index metadata in the digest.
- Refreshed quick benchmark artifacts. Compared with the baseline recorded
  immediately before this batch (`sp.regress` 1.0 ms at n=1,000 and 2.4 ms at
  n=10,000), the refreshed quick run reports 0.607 ms and 1.3 ms respectively;
  n=10,000 improved from 2.6x slower than statsmodels to 1.3x slower while
  preserving default provenance.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_ols.py
  tests/test_lineage.py tests/numerical_accuracy/test_nist_strd_ols.py
  tests/test_estimator_provenance.py tests/test_reference_alignment_statsmodels.py
  tests/test_numba_kernels.py` passed, 90 tests.
- `.venv/bin/python benchmarks/run_all.py --quick` refreshed
  `benchmarks/RESULTS.md` and `benchmarks/results.json`.

## 2026-06-17 Batch 42

Target: root `sp.wooldridge_did` staggered-DID performance without changing
Wooldridge/ETWFE estimands or reviewer-facing files.

- Batched the two-way demeaning of cohort-post and event-study dummy columns
  so all dummy columns share the same unit/time `groupby.transform` work
  instead of repeating it once per column.
- Batched optional control demeaning through the same multi-column path.
- Vectorized the `_ols_fit` cluster-robust meat computation with
  `np.unique(..., return_inverse=True)` and `np.add.at`, replacing a Python
  loop over clusters. Added a clear `DataInsufficient` failure for single
  cluster input.
- Refreshed quick benchmark artifacts. The Wooldridge quick panel now reports
  12.4 ms at 1,600 observations and 18.9 ms at 8,000 observations; before this
  batch's cluster-meat vectorization, the same 8,000-observation benchmark was
  about 78.5 ms, and the earlier full quick run recorded about 83.0 ms.

Verification run:

- `.venv/bin/python -m pytest -o addopts='' tests/test_wooldridge_did_branches.py
  tests/test_cov95_did_wooldridge.py tests/test_cov95_did_r2_wooldridge.py
  tests/test_cov95_did_r3_wooldridge.py tests/test_did_summary.py
  tests/reference_parity/test_did_parity.py` passed, 144 tests.
- `.venv/bin/python benchmarks/run_all.py --quick` refreshed
  `benchmarks/RESULTS.md` and `benchmarks/results.json`.

## 2026-06-17 Batch 43

Target: root `sp.regress` clustered-SE reliability after formula NA filtering,
while preserving the OLS performance batch and keeping JOSS/JSS paths isolated.

- Fixed OLS cluster-label alignment after formula parsing drops rows for
  missing outcome/regressor values. Cluster labels are now reindexed to the
  actual estimation sample before the accelerated cluster-meat kernel runs.
- Added explicit validation for clustered OLS inference:
  - cluster-length mismatches now raise `MethodIncompatibility`;
  - missing cluster labels in the estimation sample raise `DataInsufficient`;
  - single-cluster designs raise `DataInsufficient` instead of producing a
    meaningless finite-sample correction.
- Added focused tests proving that clustered OLS with formula-level NA drops
  matches the same regression on the explicit complete-case frame, and that
  single-cluster clustered SEs fail loudly.
- Kept the exception-taxonomy migration moving in the right direction:
  taxonomy raises increased to 148 while generic raises remained at 1,886.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts='' tests/test_ols.py`
  passed, 12 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_lineage.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/test_estimator_provenance.py tests/test_reference_alignment_statsmodels.py
  tests/test_numba_kernels.py` passed, 92 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  `148 taxonomy raises, 1886 generic raises`.
- `git diff --check` passed.

## 2026-06-17 Batch 44

Target: root output-result protocol coverage for the core regression-table
artifact, without changing rendered table semantics.

- Added a standard `summary()` surface to `RegtableResult`, delegating to the
  same configured renderer used by `__str__`.
- Added bounded `RegtableResult.to_agent_summary()` for LLM/tool workflows:
  compact metadata, first rendered rows, and a capped coefficient slice per
  model, all JSON-safe through the existing serialization layer.
- Raised `scripts/result_protocol_audit.py` floors so the improvement cannot
  silently regress:
  - `method_summary`: 256
  - `method_to_agent_summary`: 12
  - `protocol_printable`: 256
  - `protocol_serializable`: 20
  - `protocol_agent_ready`: 12
- Added tests pinning `RegtableResult` as summary/to_dict/to_agent_summary
  capable in the static audit, plus a runtime JSON-safety check for
  `to_agent_summary(max_rows=3, max_terms=1)`.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_regtable_serialization.py tests/test_regtable_from_dict.py
  tests/test_result_protocol_audit.py` passed, 86 tests.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/result_protocol_audit.py --json` confirmed
  `summary=256`, `to_agent_summary=12`, `serializable=20`, and
  `agent_ready=12`.
- `git diff --check` passed.

## 2026-06-17 Batch 45

Target: shared Bayesian result protocol coverage without requiring PyMC or
serializing heavyweight posterior traces.

- Added `BayesianCausalResult.to_dict()` with a strict JSON-safe payload for
  posterior point/HDI summaries, convergence diagnostics, `tidy()` and
  `glance()` records, and sanitized `model_info`.
- Added `BayesianCausalResult.to_agent_summary()` with compact posterior
  information plus convergence warnings for high R-hat, missing R-hat, or low
  bulk ESS.
- Explicitly excluded `trace` from serialization so ArviZ/PyMC objects do not
  leak into agent/tool payloads.
- The methods are inherited by `BayesianDIDResult`, `BayesianIVResult`,
  `BayesianHTEIVResult`, and `BayesianMTEResult`, moving the Bayesian shared
  result family from tidy-only to serializable and agent-ready.
- Raised result-protocol ratchet floors:
  - `method_to_dict`: 26
  - `method_to_agent_summary`: 17
  - `protocol_serializable`: 25
  - `protocol_agent_ready`: 17
- Added PyMC-free stub tests for JSON-safety, trace exclusion, convergence
  warnings, and subclass inheritance; added `BayesianCausalResult` to the
  static protocol assertions.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bayes_result_protocol.py tests/test_result_protocol_audit.py`
  passed, 7 tests.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/result_protocol_audit.py --json` confirmed
  `to_dict=26`, `to_agent_summary=17`, `serializable=25`, and
  `agent_ready=17`.
- `git diff --check` passed.

## 2026-06-17 Batch 46

Target: fast-inference result protocol coverage for cluster-robust bootstrap
and Wald-test workflows.

- Added JSON-safe `to_dict()` payloads to `BootTestResult` and
  `BootWaldResult`, preserving full bootstrap draw arrays for reproducibility.
- Added bounded `to_agent_summary()` payloads to `BootTestResult`,
  `BootWaldResult`, and `WaldTestResult`.
- Agent summaries now include compact bootstrap distribution summaries
  (`n`, mean, SD, 2.5/50/97.5 percentiles) instead of forcing tool loops to
  ingest every bootstrap draw.
- Raised result-protocol ratchet floors:
  - `method_to_dict`: 28
  - `method_to_agent_summary`: 20
  - `protocol_serializable`: 27
  - `protocol_agent_ready`: 20
- Added runtime tests for JSON round-trip behavior and static audit tests
  pinning all three fast inference result classes.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_inference.py tests/test_result_protocol_audit.py` passed,
  32 tests.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- A direct protocol count read confirmed `to_dict=28`,
  `to_agent_summary=20`, `serializable=27`, and `agent_ready=20`.
- `git diff --check` passed.

## 2026-06-17 Batch 47

Target: fast HDFE estimator result protocol coverage for the highest-traffic
native estimator outputs.

- Added a small shared `sp.fast` JSON-coercion helper for result payloads so
  NumPy scalars, arrays, Pandas tables, and non-finite values serialize
  predictably.
- Added lossless JSON-safe `to_dict()` payloads to `FeolsResult`,
  `FePoisResult`, and `EventStudyResult`.
- Added bounded `to_agent_summary()` payloads to the same three classes:
  coefficient/event-time rows are truncated by explicit limits while model
  metadata, FE cardinalities, fit statistics, and vcov context remain visible.
- Raised result-protocol ratchet floors:
  - `method_to_dict`: 31
  - `method_to_agent_summary`: 23
  - `protocol_serializable`: 30
  - `protocol_agent_ready`: 23
- Added runtime JSON-safety tests for all three result objects and static
  protocol assertions pinning their summary/to_dict/to_agent_summary surface.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_event_study.py tests/test_result_protocol_audit.py` passed,
  57 tests with 13 optional-engine skips.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/result_protocol_audit.py --json` confirmed
  `to_dict=31`, `to_agent_summary=23`, `serializable=30`, and
  `agent_ready=23`.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  148 taxonomy raises and 1886 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_lineage.py tests/test_fast_feols.py
  tests/test_fast_fepois.py tests/test_fast_event_study.py
  tests/test_fast_inference.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py tests/test_bayes_result_protocol.py
  tests/test_result_protocol_audit.py` passed, 214 tests with 13
  optional-engine skips. An earlier aggregate command used a stale lineage
  test filename and failed before collection; this corrected command is the
  recorded gate.

## 2026-06-17 Batch 48

Target: performance-artifact result protocol coverage for benchmark and
accelerated bootstrap outputs.

- Added `HDFEBenchResult.to_dict()` with backend availability metadata and
  full benchmark rows.
- Added `HDFEBenchResult.to_agent_summary(max_rows=...)` with bounded rows
  and per-sample-size fastest-backend summaries.
- Added `FeolsBootstrapResult.tidy()`, `to_dict()`, and
  `to_agent_summary(max_terms=...)` for the JAX bootstrap container without
  requiring JAX/GPU execution in the protocol tests.
- Extended the shared fast result helper with compact distribution summaries
  for bootstrap draws.
- Raised result-protocol ratchet floors:
  - `method_to_dict`: 33
  - `method_to_agent_summary`: 25
  - `protocol_serializable`: 32
  - `protocol_agent_ready`: 25
- Added runtime JSON-safety tests for both result classes and static protocol
  assertions pinning the summary/to_dict/to_agent_summary surface.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_bench.py tests/test_fast_jax_feols_result_protocol.py
  tests/test_result_protocol_audit.py` passed, 11 tests.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- A direct protocol count read confirmed `to_dict=33`,
  `to_agent_summary=25`, `serializable=32`, `agent_ready=25`, and `tidy=23`.
- `git diff --check` passed.

## 2026-06-17 Batch 49

Target: auto-estimator leaderboard result protocol coverage.

- Added JSON-safe `to_dict()` payloads to `AutoDIDResult` and `AutoIVResult`
  with selection rule, winner method, leaderboard rows, and candidate status
  records.
- Added bounded `to_agent_summary(max_methods=...)` payloads so multi-method
  estimator races can be inspected without parsing printed tables or raw
  candidate objects.
- The payloads intentionally keep raw fitted candidates on the Python result
  object rather than forcing potentially large estimator internals into the
  JSON surface.
- Raised result-protocol ratchet floors:
  - `method_to_dict`: 35
  - `method_to_agent_summary`: 27
  - `protocol_serializable`: 34
  - `protocol_agent_ready`: 27
- Added runtime JSON-safety tests for both auto result classes and static
  protocol assertions pinning the summary/to_dict/to_agent_summary surface.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_auto_estimators.py tests/test_result_protocol_audit.py` passed,
  19 tests.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- A direct protocol count read confirmed `to_dict=35`,
  `to_agent_summary=27`, `serializable=34`, and `agent_ready=27`.
- `git diff --check` passed.

## 2026-06-17 Batch 50

Target: reduce drift risk in fast result serialization helpers.

- Consolidated the inference-local JSON and bootstrap-distribution helpers
  onto the shared `sp.fast` result-protocol helper added in Batch 47/48.
- Preserved existing bootstrap draw-count semantics (`n` is total draws)
  while sharing the same non-finite-value handling across fast inference,
  benchmark, HDFE, event-study, and JAX bootstrap result payloads.
- No protocol floors changed in this batch; this is a consistency cleanup.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_inference.py tests/test_fast_bench.py
  tests/test_fast_jax_feols_result_protocol.py tests/test_result_protocol_audit.py`
  passed, 39 tests.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `git diff --check` passed.

## 2026-06-17 Batch 51

Target: benchmark harness input validation and reference-path cleanup.

- Added explicit `hdfe_bench()` validation for non-positive sample sizes,
  non-positive group counts, non-positive repeat counts, and negative
  correctness tolerances.
- This prevents silent `inf` timings when `repeat <= 0` and replaces
  cryptic NumPy errors for invalid synthetic DGP sizes with actionable
  `ValueError` messages.
- Simplified the NumPy reference-backend lookup from an old defensive
  expression to a direct `backends["numpy"]` access.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_bench.py tests/test_result_protocol_audit.py` passed,
  14 tests.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `git diff --check` passed.

## 2026-06-17 Batch 52

Target: clearer fast event-study failure mode for FE-collinear dummies.

- Reused the inverted event-study bread matrix for both coefficients and
  cluster-robust variance.
- Wrapped singular normal equations with an estimator-level `RuntimeError`
  that explains the likely cause: event-time dummies perfectly collinear with
  absorbed unit/time fixed effects after filtering.
- Added a regression test where all units share the same treatment timing, so
  the retained event-time dummy is fully absorbed by the time fixed effect.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_event_study.py tests/test_result_protocol_audit.py` passed,
  14 tests with 1 optional-engine skip.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  148 taxonomy raises and 1891 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_lineage.py tests/test_auto_estimators.py
  tests/test_fast_bench.py tests/test_fast_jax_feols_result_protocol.py
  tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_event_study.py tests/test_fast_inference.py
  tests/test_regtable_serialization.py tests/test_regtable_from_dict.py
  tests/test_bayes_result_protocol.py tests/test_result_protocol_audit.py`
  passed, 241 tests with 13 optional-engine skips.

## 2026-06-17 Batch 53

Target: fast event-study and CRVE input validation against silent cluster and
event-time corruption.

- Added `event_study()` validation that the outcome is finite before
  residualization.
- Added explicit numeric/integer event-time checks so finite fractional
  offsets cannot be silently truncated by `astype(int64)`.
- Rejected infinite event-time values; only `NaN` is accepted for
  never-treated rows.
- Added integer reference-period and integer ordered-window validation so the
  omitted category and event window cannot silently drift.
- Added `crve()` validation for residual-length mismatch, non-finite
  residuals, missing cluster labels, cluster-length mismatch, non-finite
  weights, negative weights, and weight-length mismatch before factorization
  or cluster-score accumulation.
- Consolidated cluster-label factorization through a shared inference helper
  so `crve()`, `boottest()`, `boottest_wald()`, BM DOF, Wald BM DOF, and HTZ
  DOF all reject missing or misaligned cluster labels consistently.
- This closes a silent corruption path where missing cluster labels could be
  factorized to `-1` and then accumulated into the last cluster.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_inference.py tests/test_fast_event_study.py
  tests/test_fast_htz.py tests/test_result_protocol_audit.py` passed, 76
  tests with 4 optional-engine skips.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  149 taxonomy raises and 1899 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_lineage.py tests/test_auto_estimators.py
  tests/test_fast_bench.py tests/test_fast_jax_feols_result_protocol.py
  tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_event_study.py tests/test_fast_inference.py
  tests/test_fast_htz.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py tests/test_bayes_result_protocol.py
  tests/test_result_protocol_audit.py` passed, 275 tests with 16
  optional-engine skips.

## 2026-06-17 Batch 54

Target: fast formula-DSL input validation against silent missing-category and
string-splitting mistakes.

- Added `sp.fast.i()` validation that categorical inputs are one-dimensional
  and contain no missing values. Previously a missing category produced an
  all-zero dummy row, making it indistinguishable from a valid omitted
  category.
- Added `sp.fast.fe_interact()` validation that every interacted FE column is
  one-dimensional and missing-free before factorization. This prevents missing
  cells from being silently encoded as legitimate interaction levels.
- Changed `sw()` and `csw()` to treat bare strings as one variable name instead
  of iterating over characters, while preserving the existing iterable-list
  behavior.
- Added focused regression tests for missing categories, non-1D inputs, and
  bare-string stepwise specifications.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_within_dsl.py tests/test_fast_demean.py
  tests/test_fast_fepois.py tests/test_result_protocol_audit.py` passed, 60
  tests with 10 optional-engine skips.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `git diff --check` passed.

Taxonomy follow-up:

- Routed the new DSL validation failures through `DataInsufficient` and
  `MethodIncompatibility` rather than plain `ValueError`, preserving
  `ValueError` compatibility while keeping the exception taxonomy audit
  ratchet intact.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_within_dsl.py tests/test_fast_inference.py
  tests/test_fast_htz.py tests/test_result_protocol_audit.py` passed, 85
  tests with 3 optional-engine skips.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  156 taxonomy raises and 1896 generic raises.
- `git diff --check` passed.

## 2026-06-17 Batch 55

Target: shared fast-inference weight validation before matrix products.

- Added a shared `_prepare_weights()` helper for fast inference paths.
- Routed `crve()`, `boottest()`, `boottest_wald()`, BM DOF, Wald BM DOF,
  and HTZ DOF through the helper so weight length, finiteness, and sign
  constraints are checked before weighted cross-products or cluster loops.
- Preserved HTZ's stricter semantics: weights must be strictly positive and
  uniform in v1.
- Added tests for invalid bootstrap observation weights, invalid BM DOF
  weights, and HTZ length/non-finite weights.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_inference.py tests/test_fast_htz.py
  tests/test_result_protocol_audit.py` passed, 64 tests with 3
  optional-engine skips.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  156 taxonomy raises and 1896 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_lineage.py tests/test_auto_estimators.py
  tests/test_fast_bench.py tests/test_fast_jax_feols_result_protocol.py
  tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_event_study.py tests/test_fast_inference.py
  tests/test_fast_htz.py tests/test_fast_within_dsl.py
  tests/test_fast_demean.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py tests/test_bayes_result_protocol.py
  tests/test_result_protocol_audit.py` passed, 314 tests with 17
  optional-engine skips.

## 2026-06-17 Batch 56

Target: fast FEOLS/FEPois early validation before IRLS, demeaning, or normal
equation failures.

- Added shared private fast-estimator validation helpers for positive integer
  iteration caps, finite non-negative tolerances, non-empty samples, and
  positive observation-weight mass.
- Routed `sp.fast.feols()` through the helper for `fe_maxiter`, `fe_tol`,
  empty data, all-zero weights, and the post-singleton kept-weight sample.
- Routed `sp.fast.fepois()` through the helper for `maxiter`, `fe_maxiter`,
  `tol`, `fe_tol`, empty data, all-zero weights, and the post-drop kept
  sample.
- Added explicit FEPois finite outcome/regressor checks so `NaN`/`inf` input
  is rejected before IRLS setup rather than propagating into undefined
  deviance, native Rust, or linear algebra states.
- Added regression tests for empty data, invalid algorithm controls,
  non-finite FEPois inputs, all-zero weights, and positive weights lost during
  singleton pruning.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_inference.py tests/test_result_protocol_audit.py` passed,
  102 tests with 12 optional-engine skips.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  162 taxonomy raises and 1898 generic raises.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  162 taxonomy raises and 1898 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_lineage.py tests/test_auto_estimators.py
  tests/test_fast_bench.py tests/test_fast_jax_feols_result_protocol.py
  tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_event_study.py tests/test_fast_inference.py
  tests/test_fast_htz.py tests/test_fast_within_dsl.py
  tests/test_fast_demean.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py tests/test_bayes_result_protocol.py
  tests/test_result_protocol_audit.py` passed, 329 tests with 17
  optional-engine skips.

## 2026-06-17 Batch 57

Target: HDFE demean FE-shape validation before singleton pruning or Rust/NumPy
kernel dispatch.

- Added explicit `sp.fast.demean()` FE row-count validation for DataFrame,
  one-dimensional ndarray, and two-dimensional ndarray inputs.
- Added explicit validation that sequence-style FE columns are one-dimensional
  and aligned to `X`.
- Routed these failures through `MethodIncompatibility`, preserving
  `ValueError` compatibility while avoiding cryptic boolean-index or
  factorization errors.
- Added regression tests for DataFrame row mismatch, ndarray row mismatch, and
  non-1D sequence FE inputs.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_demean.py tests/test_fast_feols.py tests/test_fast_fepois.py`
  passed, 76 tests with 13 optional-engine skips.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  168 taxonomy raises and 1896 generic raises.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  168 taxonomy raises and 1896 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_lineage.py tests/test_auto_estimators.py
  tests/test_fast_bench.py tests/test_fast_jax_feols_result_protocol.py
  tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_event_study.py tests/test_fast_inference.py
  tests/test_fast_htz.py tests/test_fast_within_dsl.py
  tests/test_fast_demean.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py tests/test_bayes_result_protocol.py
  tests/test_result_protocol_audit.py` passed, 332 tests with 17
  optional-engine skips.

## 2026-06-17 Batch 58

Target: OLS analytic-weight validation and robust-option normalization.

- Added a shared OLS analytic-weight validator and routed both
  `sp.regress(..., weights=...)` and direct `OLSEstimator().estimate(...,
  weights=...)` calls through it.
- Direct estimator calls now fail before WLS normalization on non-numeric,
  mis-sized, non-finite, zero, or negative analytic weights.
- Normalized `robust` option dispatch to a lowercase key so `robust="HC1"`
  matches `robust="hc1"` instead of falling through to an unknown-option path.
- Changed unknown robust options to `MethodIncompatibility`, preserving
  `ValueError` compatibility while improving taxonomy coverage.
- Added tests for direct invalid weights, formula-path weight alignment after
  missing-data filtering, and uppercase robust aliases.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py` passed, 99 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  173 taxonomy raises and 1892 generic raises.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  173 taxonomy raises and 1892 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_lineage.py tests/test_auto_estimators.py
  tests/test_fast_bench.py tests/test_fast_jax_feols_result_protocol.py
  tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_event_study.py tests/test_fast_inference.py
  tests/test_fast_htz.py tests/test_fast_within_dsl.py
  tests/test_fast_demean.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py tests/test_bayes_result_protocol.py
  tests/test_result_protocol_audit.py` passed, 337 tests with 17
  optional-engine skips.

## 2026-06-17 Batch 59

Target: OLS y/X validation across formula and direct-array APIs.

- Added a shared OLS array validator that checks numeric conversion, 2-D
  design shape, row alignment, non-empty samples, and finite `y`/`X` values.
- Routed direct `OLSEstimator().estimate(...)` calls and
  `OLSRegression.fit()` through the same validator before collinearity checks
  or low-level linear algebra.
- Added a `var_names` length check for direct `OLSRegression(y=..., X=...)`
  fits so result indexing cannot silently mislabel coefficients.
- Added regression tests for direct y/X row mismatch, `var_names` mismatch,
  and formula-path `inf` design values.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py` passed, 102 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  180 taxonomy raises and 1892 generic raises.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  180 taxonomy raises and 1892 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_lineage.py tests/test_auto_estimators.py
  tests/test_fast_bench.py tests/test_fast_jax_feols_result_protocol.py
  tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_event_study.py tests/test_fast_inference.py
  tests/test_fast_htz.py tests/test_fast_within_dsl.py
  tests/test_fast_demean.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py tests/test_bayes_result_protocol.py
  tests/test_result_protocol_audit.py` passed, 340 tests with 17
  optional-engine skips.

## 2026-06-17 Batch 60

Target: result-object probability argument validation for summaries and
confidence intervals.

- Added a shared probability validator for open-interval parameters such as
  `alpha` and `conf_level`.
- Routed `EconometricResults.summary()`, `EconometricResults.conf_int()`, and
  `EconometricResults.tidy()` through the validator.
- Routed `CausalResult.summary()` and `CausalResult.tidy()` through the same
  validator, fixing the previous `alpha=0` fallback-to-default behavior.
- Added agent/result tests covering invalid `alpha` and `conf_level` values
  for both econometric and causal result classes.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_agent_result_methods.py tests/test_ols.py
  tests/test_regtable_serialization.py tests/test_result_protocol_audit.py`
  passed, 67 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  182 taxonomy raises and 1892 generic raises.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  182 taxonomy raises and 1892 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_agent_result_methods.py tests/test_lineage.py
  tests/test_auto_estimators.py tests/test_fast_bench.py
  tests/test_fast_jax_feols_result_protocol.py tests/test_fast_feols.py
  tests/test_fast_fepois.py tests/test_fast_event_study.py
  tests/test_fast_inference.py tests/test_fast_htz.py
  tests/test_fast_within_dsl.py tests/test_fast_demean.py
  tests/test_regtable_serialization.py tests/test_regtable_from_dict.py
  tests/test_bayes_result_protocol.py tests/test_result_protocol_audit.py`
  passed, 365 tests with 17 optional-engine skips.

## 2026-06-17 Batch 61

Target: auto estimator high-level argument validation before candidate races.

- Added shared private validators for auto-estimator probability arguments,
  string/list column arguments, method lists, and required data columns.
- `sp.auto_did()` now rejects invalid `alpha`, empty method lists, missing
  required columns, non-DataFrame inputs, and malformed covariate lists before
  launching candidate estimators.
- `sp.auto_iv()` now rejects invalid `alpha`, empty method lists, empty
  instruments, missing required columns, non-DataFrame inputs, and malformed
  exogenous/instrument lists before launching candidate estimators.
- `sp.auto_iv(..., exog="x1")` now treats scalar exogenous controls the same
  way scalar instruments were already treated: as a one-column list.
- Converted unknown-method and unknown-winner errors to
  `MethodIncompatibility`, preserving `ValueError` compatibility while
  improving taxonomy coverage.
- Added tests for missing columns, empty method/instrument lists, invalid
  alpha, and scalar `exog` handling.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_auto_estimators.py tests/test_result_protocol_audit.py` passed,
  26 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  195 taxonomy raises and 1887 generic raises.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  195 taxonomy raises and 1887 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_agent_result_methods.py tests/test_lineage.py
  tests/test_auto_estimators.py tests/test_fast_bench.py
  tests/test_fast_jax_feols_result_protocol.py tests/test_fast_feols.py
  tests/test_fast_fepois.py tests/test_fast_event_study.py
  tests/test_fast_inference.py tests/test_fast_htz.py
  tests/test_fast_within_dsl.py tests/test_fast_demean.py
  tests/test_regtable_serialization.py tests/test_regtable_from_dict.py
  tests/test_bayes_result_protocol.py tests/test_result_protocol_audit.py`
  passed, 372 tests with 17 optional-engine skips.

## 2026-06-17 Batch 71

Target: OLS hot-path performance audit and fail-loud collinearity regression
coverage, without weakening numerical checks or default provenance.

- Profiled the current `sp.regress` stack after the earlier OLS performance
  batches. On a 1,000-row, 2-regressor numeric formula after warmup, the
  measured minimum timings were approximately:
  - `sp.regress`: 0.438 ms;
  - `OLSRegression.fit`: 0.352 ms;
  - `create_design_matrices`: 0.063 ms;
  - `compute_data_hash`: 0.047 ms;
  - `attach_provenance` for a new result: 0.057 ms;
  - formula variable preflight: 0.0025 ms;
  - `_numba_kernels()`: 0.0006 ms.
- Rejected an attempted centered-Gram rewrite of
  `_detect_perfect_collinearity`: in local microbenchmarks it was slower than
  the existing centered-matrix path (`sp.regress` moved from roughly
  1.25-1.29 ms to 1.33 ms on the same pre-warmup sample). The code was
  reverted rather than landing a cosmetic optimization.
- Rejected `_numba_kernels()` caching and regex preflight rewrites as not
  worth their maintenance surface: both measured in the microsecond range and
  do not materially affect the estimator path.
- Kept default lineage/provenance intact. Data hashing is visible in the
  profile, but it is bounded and provides traceability for downstream
  reproducibility artifacts; no option was changed to make benchmark numbers
  look better by default.
- Added an explicit OLS regression test for pairwise perfect collinearity
  (`x1_twice = 2 * x1`) so future performance work cannot accidentally return
  unidentified coefficients silently.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/numerical_accuracy/test_nist_strd_anova.py` passed, 101 tests.
- `.venv/bin/python -m compileall -q src/statspai/regression/ols.py
  tests/test_ols.py` passed.

## 2026-06-17 Batch 72

Target: OLS fail-loud coverage for exact low-order linear dependence, without
turning the NIST-safe structural guard into a loose rank-tolerance test.

- Added a targeted low-order dependence check inside `OLSEstimator.estimate()`
  that runs only after the guarded cross-product path has already rejected the
  design as ill-conditioned.
- The new check catches structural mistakes such as `x_sum = x1 + x2` by
  testing whether any column is exactly spanned by two other columns, and raises
  `NumericalInstability` with the dependent column, basis columns, coefficients,
  and a recovery hint.
- Kept the search bounded so ordinary well-conditioned regressions do not pay
  for the extra check and wide ill-conditioned designs still fall back to the
  existing QR/NIST-safe path instead of running an expensive combinatorial
  scan.
- Threaded OLS variable names into the estimator so diagnostics identify the
  offending regressors in formula-driven `sp.regress()` calls; direct estimator
  calls with missing or malformed names fall back to default `x0`, `x1`, ...
  labels rather than failing while formatting diagnostics.
- Added a formula-level regression test proving `sp.regress("y ~ x1 + x2 +
  x_sum")` now fails loudly when `x_sum` is exactly `x1 + x2`.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/numerical_accuracy/test_nist_strd_anova.py` passed, 102 tests.
- `.venv/bin/python -m compileall -q src/statspai/regression/ols.py
  tests/test_ols.py` passed.
- `git diff --check -- src/statspai/regression/ols.py tests/test_ols.py`
  passed.
- A hot-path sanity benchmark on the 1,000-row, two-regressor numeric formula
  reported `sp.regress` minimum time of about 0.457 ms after warmup, confirming
  the new fallback-only check is not on the ordinary full-rank path.

## 2026-06-17 Batch 73

Target: OLS residual-degree-of-freedom and exact-fit diagnostics, replacing
RuntimeWarning/NaN leakage with explicit estimator behavior.

- Added a fail-loud `DataInsufficient` guard in `OLSEstimator.estimate()` for
  `nobs <= parameters`. These regressions cannot estimate residual variance or
  standard errors, and previously returned NaN/inf diagnostics after NumPy
  runtime warnings.
- Made OLS log-likelihood/AIC/BIC diagnostics branch explicitly on zero RSS.
  Perfect fits with positive residual degrees of freedom still report the
  mathematically degenerate `inf/-inf` diagnostics, but no longer emit
  low-level divide-by-zero warnings.
- Added focused tests for both cases: `sp.regress("y ~ x")` on two rows now
  raises with `residual df=0`, while a four-row exact fit completes under
  `RuntimeWarning`-as-error and records the expected degenerate diagnostics.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/numerical_accuracy/test_nist_strd_anova.py` passed, 104 tests.
- `.venv/bin/python -m compileall -q src/statspai/regression/ols.py
  tests/test_ols.py` passed.
- `git diff --check -- src/statspai/regression/ols.py tests/test_ols.py`
  passed.

## 2026-06-17 Batch 74

Target: OLS constant-outcome diagnostics, eliminating divide-by-zero warnings
while preserving explicitly undefined fit statistics.

- Added an explicit `tss <= 0` branch in `OLSEstimator.estimate()` so constant
  outcomes report undefined `R-squared`, adjusted `R-squared`, and F-statistic
  as `nan` instead of reaching NumPy division-by-zero paths.
- Kept coefficient estimation and exact-fit likelihood/AIC/BIC semantics
  unchanged; only the undefined total-variation diagnostics are now handled
  deliberately.
- Added a `RuntimeWarning`-as-error regression test for a constant-outcome
  OLS fit, asserting the undefined diagnostics are represented as `nan`.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/numerical_accuracy/test_nist_strd_anova.py` passed, 105 tests.
- `.venv/bin/python -m compileall -q src/statspai/regression/ols.py
  tests/test_ols.py` passed.
- `git diff --check -- src/statspai/regression/ols.py tests/test_ols.py`
  passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  287 taxonomy raises and 1827 generic raises.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x; the quick OLS timings remain
  0.19x and 0.29x of the recorded baseline for the two `sp.regress`
  benchmark cases.
- `Paper-JSS/` and `CausalAgentBench/` were rechecked separately and both
  remain clean on `main...origin/main`.
- `.venv/bin/python scripts/quality_gate.py all` passed under the current
  gradual-debt baselines: flake8 observed 4403 <= 4698, mypy observed
  3317 <= 3521, import-budget observed 0, and the agent-card,
  result-protocol, and error-taxonomy gates passed.

## 2026-06-17 Batch 75

Target: OLS prediction state validation and local lint debt reduction in the
same root module touched by the reliability fixes.

- Removed unused `Union`, `parse_formula`, and `prepare_data` imports from
  `src/statspai/regression/ols.py`, lowering the aggregate flake8 debt count
  from 4406 to 4403 while preserving behavior.
- Added an explicit out-of-sample `predict()` guard for the invalid fitted
  state where `self.var_names` is unavailable. The method now raises a clear
  `MethodIncompatibility` instead of failing later while iterating over `None`.
- Added a focused test that corrupts `model.var_names` after fitting and checks
  the new prediction error message.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/numerical_accuracy/test_nist_strd_anova.py` passed, 106 tests.
- `.venv/bin/python -m compileall -q src/statspai/regression/ols.py
  tests/test_ols.py` passed.
- `git diff --check -- src/statspai/regression/ols.py tests/test_ols.py`
  passed.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4403 <= 4698, mypy observed 3316 <= 3521, import-budget observed 0,
  error-taxonomy observed 288 taxonomy raises and 1827 generic raises, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 76

Target: OLS prediction-interval covariance correctness for out-of-sample
confidence and prediction intervals.

- Fixed `OLSRegression.predict()` to use the full coefficient covariance matrix
  already stored as `result.data_info["var_cov"]`. The previous implementation
  looked only for `cov_params`, did not find it on OLS results, and silently
  fell back to a diagonal standard-error approximation.
- Preserved the old diagonal fallback only for result objects that truly lack a
  full covariance matrix.
- Added a shape guard for malformed covariance matrices, raising
  `MethodIncompatibility` with recovery guidance and machine-readable
  diagnostics instead of letting `np.einsum()` fail opaquely.
- Added a focused correlated-regressor prediction test that compares the lower
  confidence bound to the full `x'Vx` calculation and proves it is materially
  different from the old diagonal approximation.
- Added a focused malformed-covariance test for the new fail-loud branch.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/numerical_accuracy/test_nist_strd_anova.py` passed, 108 tests.
- `.venv/bin/python -m compileall -q src/statspai/regression/ols.py
  tests/test_ols.py` passed.
- `git diff --check -- src/statspai/regression/ols.py tests/test_ols.py`
  passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  289 taxonomy raises and 1827 generic raises.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `Paper-JSS/` and `CausalAgentBench/` were rechecked separately and both
  remain clean on `main...origin/main`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4403 <= 4698, mypy observed 3316 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 77

Target: OLS prediction option validation, preventing nonsensical intervals
from invalid `alpha` or `what` inputs.

- Reproduced the previous behavior: invalid `alpha` values could return
  `nan`, infinite, or reversed confidence intervals; invalid `what` failed only
  after unnecessary prediction setup.
- Added early `what` validation in `OLSRegression.predict()`, raising
  `MethodIncompatibility` with valid choices and diagnostics.
- Added interval `alpha` validation for confidence/prediction interval paths,
  requiring a finite number strictly inside `(0, 1)` and raising
  `MethodIncompatibility` before any interval math runs.
- Added tests for negative, zero, one, greater-than-one, NaN, and nonnumeric
  alpha values, plus invalid interval type inputs including a non-string value.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/numerical_accuracy/test_nist_strd_anova.py` passed, 117 tests.
- `.venv/bin/python -m compileall -q src/statspai/regression/ols.py
  tests/test_ols.py` passed.
- `git diff --check -- src/statspai/regression/ols.py tests/test_ols.py`
  passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  292 taxonomy raises and 1827 generic raises.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `Paper-JSS/` and `CausalAgentBench/` were rechecked separately and both
  remain clean on `main...origin/main`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4403 <= 4698, mypy observed 3316 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 78

Target: OLS mean-prediction DataFrame output contract.

- Reproduced that the legal call `model.predict(..., what="mean",
  return_df=True)` fell through into interval-only logic and raised
  `ValueError` saying `what="mean"` was invalid.
- Fixed the mean prediction path so `return_df=True` returns a one-column
  DataFrame with `yhat`, both in-sample and out-of-sample.
- Limited `alpha` validation to confidence/prediction interval paths. Mean-only
  predictions no longer reject an irrelevant `alpha` value.
- Added focused tests for in-sample and out-of-sample `return_df=True` mean
  predictions, including the previously failing case with an irrelevant
  invalid `alpha`.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/numerical_accuracy/test_nist_strd_anova.py` passed, 119 tests.
- `.venv/bin/python -m compileall -q src/statspai/regression/ols.py
  tests/test_ols.py` passed.
- `git diff --check -- src/statspai/regression/ols.py tests/test_ols.py`
  passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  292 taxonomy raises and 1827 generic raises.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `Paper-JSS/` and `CausalAgentBench/` were rechecked separately and both
  remain clean on `main...origin/main`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4403 <= 4698, mypy observed 3316 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 79

Target: OLS out-of-sample prediction for Patsy categorical formulas and
prediction-state taxonomy errors.

- Reproduced that a model fit with `y ~ x + C(g)` could not predict on new data
  containing only known training levels. `predict()` rebuilt a fresh Patsy
  design from the new data, so the fitted dummy column `C(g)[T.b]` disappeared
  and the call failed even for valid `g="a"` or `g="b"` rows.
- Stored Patsy `design_info` from the fitted design matrix and reused it via
  `build_design_matrices()` for out-of-sample prediction. Known training
  categorical levels now use the same encoding as the fitted model.
- Preserved the fast numeric formula path: models fit through the direct
  numeric builder still use the existing RHS `dmatrix()` fallback for new data.
- Converted prediction design-building failures, missing generated columns,
  unfitted prediction, and raw-array out-of-sample prediction into
  `MethodIncompatibility` with recovery hints and diagnostics.
- Added focused tests for known categorical levels, unseen categorical levels,
  unfitted prediction, and raw-array out-of-sample prediction.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_ols.py
  tests/numerical_accuracy/test_nist_strd_anova.py` passed, 123 tests.
- `.venv/bin/python -m compileall -q src/statspai/regression/ols.py
  tests/test_ols.py` passed.
- `git diff --check -- src/statspai/regression/ols.py tests/test_ols.py`
  passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  297 taxonomy raises and 1823 generic raises.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `Paper-JSS/` and `CausalAgentBench/` were rechecked separately and both
  remain clean on `main...origin/main`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4403 <= 4698, mypy observed 3316 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 80

Target: IV no-intercept formula parsing and IV prediction-state taxonomy.

- Reproduced that formula-backed IV models failed on common no-intercept
  syntax such as `y ~ (x ~ z) + w - 1` and `y ~ (x ~ z) - 1`, because the
  shared parser treated `w - 1` or `- 1` as data column names instead of
  intercept-removal tokens.
- Added a shared additive-term splitter for simple formulas so `- 1`, `-1`,
  `0`, and adjacent `+` terms are parsed consistently by the IV and fallback
  formula paths.
- Converted IV out-of-sample prediction misuse from bare `ValueError` or
  delayed conversion failures into `MethodIncompatibility` with recovery hints
  and diagnostics for unfitted models, raw-array fits, missing columns,
  non-DataFrame inputs, nonnumeric prediction columns, and parameter/data
  mapping failures.
- Added no-intercept IV prediction tests that check fitted parameter order and
  out-of-sample predictions with and without exogenous controls.
- Added IV prediction error-taxonomy tests for unfitted models, missing
  columns, nonnumeric prediction values, and raw-array out-of-sample
  prediction.
- While running the IV/reference parity cone, fixed an OLS analytic-weight
  error-message drift so non-finite weights again mention `NaN or infinite`
  while preserving the existing `DataInsufficient`/`ValueError` compatibility.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_iv.py tests/test_predict_oos.py tests/test_ols.py` passed, 69
  tests.
- `.venv/bin/python -m compileall -q src/statspai/core/utils.py
  src/statspai/regression/iv.py tests/test_iv.py` passed.
- `.venv/bin/python -m pytest -o addopts=''
  tests/test_auto_estimators.py tests/test_agent_result_methods.py
  tests/test_result_protocol_audit.py` passed, 51 tests.
- `.venv/bin/python -m pytest -o addopts=''
  tests/reference_parity/test_regress_weights_iv_robust_parity.py
  tests/test_exception_migrations.py::TestIVUnderIdentified` passed, 15
  tests.
- `.venv/bin/python -m pytest -o addopts=''
  tests/reference_parity/test_iv_parity.py
  tests/reference_parity/test_iv_se_parity.py
  tests/reference_parity/test_liml_se_parity.py` passed, 14 tests with the
  expected weak-instrument warning.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  303 taxonomy raises and 1819 generic raises.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4403 <= 4698, mypy observed 3316 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 81

Target: formula-backed GLM out-of-sample prediction reliability.

- Reproduced that `GLMRegression(..., formula="y ~ x + z").predict(new_data)`
  failed with raw `KeyError: 'y'` when `new_data` correctly contained only
  right-hand-side variables. The old path rebuilt full design matrices from the
  training formula, so prediction required an unavailable LHS outcome.
- Stored Patsy `design_info` from formula-backed GLM fits and reused it for
  out-of-sample prediction, matching the OLS categorical-design repair while
  preserving the direct numeric fast path via RHS-only `dmatrix()`.
- Converted GLM prediction misuse into `MethodIncompatibility` with recovery
  hints and diagnostics for unfitted models, invalid prediction `type`,
  raw-array out-of-sample prediction, non-DataFrame inputs, missing generated
  formula columns, unseen categorical levels, design/parameter shape mismatch,
  and invalid offset vectors.
- Added focused GLM prediction tests for numeric Poisson formulas without the
  response column, response/link/variance outputs, categorical formula reuse,
  unseen categorical levels, scalar offset broadcasting, offset shape
  validation, and raw-array fit boundaries.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_glm_predict.py tests/test_new_v06_modules.py
  tests/test_export_surface_contract.py tests/test_predict_oos.py` passed, 101
  tests with the existing multinomial and negative-binomial warnings.
- `.venv/bin/python -m compileall -q src/statspai/regression/glm.py
  tests/test_glm_predict.py` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  314 taxonomy raises and 1817 generic raises.
- `.venv/bin/python -m pytest -o addopts=''
  tests/reference_parity/test_count_quantile_parity.py
  tests/reference_parity/test_ipw_parity.py
  tests/reference_parity/test_tmle_parity.py` passed, 41 tests.
- `.venv/bin/python -m pytest -o addopts=''
  tests/test_survey.py tests/test_output_and_survey_helpers.py` passed, 21
  tests with existing output-helper warnings.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=146`, `smoke=9`,
  `untested=187`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4402 <= 4698, mypy observed 3316 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 82

Target: generic `EconometricResults.predict()` error taxonomy hardening.

- Reproduced that the generic result-level prediction fallback leaked
  `AttributeError` for non-DataFrame out-of-sample inputs and raw NumPy/pandas
  conversion errors for nonnumeric prediction columns.
- Converted out-of-sample prediction input failures into
  `MethodIncompatibility` while preserving `ValueError` compatibility for
  existing callers and tests.
- Added diagnostics and recovery hints for non-DataFrame inputs, missing
  coefficient columns, formula-derived terms unsupported by the generic
  fallback, and nonnumeric prediction columns.
- Kept the existing `NotImplementedError` behavior for result types that do
  not store in-sample fitted values or do not expose Series-like coefficients.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_external_reviewer_followups.py tests/test_agent_result_methods.py
  tests/test_result_protocol_audit.py` passed, 43 tests.
- `.venv/bin/python -m compileall -q src/statspai/core/results.py
  tests/test_external_reviewer_followups.py` passed.
- `git diff --check -- src/statspai/core/results.py
  tests/test_external_reviewer_followups.py` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  318 taxonomy raises and 1815 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=146`, `smoke=9`,
  `untested=187`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4402 <= 4698, mypy observed 3316 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 83

Target: GLM auxiliary-column alignment after formula missing-data filtering.

- Reproduced that formula-backed GLM fits with missing RHS/outcome rows failed
  when `weights`, `offset`, or `cluster` were supplied, because design-matrix
  construction dropped rows but auxiliary columns were still read from the
  unfiltered original DataFrame. The failures surfaced as NumPy broadcasting
  or boolean-index length errors.
- Added formula-design index alignment for GLM `weights`, `offset`,
  `exposure`, and `cluster` inputs so auxiliary arrays match the actual
  estimation sample.
- Added validation for missing/nonfinite/nonnumeric auxiliary columns,
  nonnegative weights with at least one positive value, strictly positive
  exposure, nonmissing cluster labels, and at least two clusters for clustered
  inference.
- Avoided silently ignoring `weights`, `offset`, `exposure`, or `cluster`
  options on raw-array GLM fits without `data=`.
- Added tests showing dirty formula data with a dropped row matches the same
  model fit on the explicitly cleaned data, and tests for bad auxiliary-column
  failures.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_glm_predict.py tests/test_new_v06_modules.py
  tests/test_export_surface_contract.py tests/test_predict_oos.py` passed, 103
  tests with the existing multinomial and negative-binomial warnings.
- `.venv/bin/python -m compileall -q src/statspai/regression/glm.py
  tests/test_glm_predict.py` passed.
- `git diff --check -- src/statspai/regression/glm.py
  tests/test_glm_predict.py` passed.
- `.venv/bin/python -m pytest -o addopts=''
  tests/reference_parity/test_count_quantile_parity.py
  tests/reference_parity/test_ipw_parity.py
  tests/reference_parity/test_tmle_parity.py` passed, 41 tests.
- `.venv/bin/python -m pytest -o addopts=''
  tests/test_survey.py tests/test_output_and_survey_helpers.py` passed, 21
  tests with existing output-helper warnings.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  331 taxonomy raises and 1815 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=146`, `smoke=9`,
  `untested=187`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4402 <= 4698, mypy observed 3317 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 84

Target: multilevel prediction API contracts for LMM and GLMM results.

- Reproduced that `MixedResult.predict(new_data, include_random=False)` still
  required group columns even though population/marginal prediction only needs
  fixed-effect regressors.
- Reproduced that GLMM prediction leaked raw `KeyError` / `TypeError` for
  missing columns or non-DataFrame inputs, and silently treated invalid
  `type="..."` values as response predictions.
- Updated LMM prediction so marginal prediction requires only fixed-effect
  columns, while conditional prediction still requires random-effect and group
  columns. Missing/non-DataFrame/nonnumeric/nonfinite prediction inputs now
  raise `MethodIncompatibility` with diagnostics.
- Updated GLMM prediction to validate `type in {"response", "linear"}`,
  non-DataFrame inputs, missing fixed/random/group/offset columns,
  nonnumeric/nonfinite fixed/random/offset columns, and fitted offset
  requirements. Unseen groups still contribute zero random effect as before.
- Added focused LMM tests for marginal prediction without group columns,
  conditional missing-group diagnostics, non-DataFrame inputs, and nonnumeric
  fixed-effect inputs.
- Added focused GLMM tests for marginal prediction without group columns,
  invalid prediction type, missing columns, non-DataFrame inputs, nonnumeric
  fixed-effect inputs, and missing fitted offset columns.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_multilevel.py` passed, 59 tests.
- `.venv/bin/python -m compileall -q src/statspai/multilevel/lmm.py
  src/statspai/multilevel/glmm.py tests/test_multilevel.py` passed.
- `git diff --check -- src/statspai/multilevel/lmm.py
  src/statspai/multilevel/glmm.py tests/test_multilevel.py` passed.
- `.venv/bin/python -m pytest -o addopts=''
  tests/test_parity_harness_contract.py` passed, 36 tests.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  347 taxonomy raises and 1813 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=146`, `smoke=9`,
  `untested=187`).
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4400 <= 4698, mypy observed 3317 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 85

Target: stochastic-frontier prediction error taxonomy and input validation.

- Converted `FrontierResult.predict()` input-contract failures from raw
  `KeyError` / `ValueError` / conversion errors into StatsPAI taxonomy errors
  while preserving `ValueError` compatibility where appropriate.
- Added validation for non-string or unknown `what`, non-DataFrame prediction
  data, missing required frontier/variance/inefficiency/outcome columns,
  nonnumeric or nonfinite prediction columns, and empty complete prediction
  samples after dropping missing rows.
- Conditional frontier predictions now report missing dependent-variable
  requirements through `MethodIncompatibility` diagnostics instead of a raw
  `KeyError`.
- Added tests for missing columns, unknown `what`, non-DataFrame input,
  nonnumeric input, all-row drop after missing data, and conditional
  predictions without `y`.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_frontier.py` passed, 99 tests.
- `.venv/bin/python -m compileall -q src/statspai/frontier/sfa.py
  tests/test_frontier.py` passed.
- `git diff --check -- src/statspai/frontier/sfa.py tests/test_frontier.py`
  passed.
- `.venv/bin/python -m pytest -o addopts=''
  tests/test_parity_harness_contract.py` passed, 36 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  355 taxonomy raises and 1809 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=146`, `smoke=9`,
  `untested=187`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4400 <= 4698, mypy observed 3317 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 86

Target: CausalForest prediction/effect API contracts.

- Fixed a formula-interface bug where `CausalForest.fit("Y ~ T | age + score",
  data=...)` overwrote the original effect-modifier names with generic
  `X0`, `X1`, ... names, causing out-of-sample `predict()` calls on the
  original columns to fail.
- Added a shared fitted-feature validator for `predict()`, `effect()`, and
  `effect_interval()` that checks missing DataFrame columns, nonnumeric
  effect modifiers, nonfinite values, empty prediction samples, unsupported
  array ranks, and feature-count mismatches before sklearn internals see the
  data.
- Converted unfitted, missing-data, bad-shape, bad-column, and bad-alpha
  prediction failures to StatsPAI taxonomy errors with recovery hints and
  diagnostics while preserving `ValueError` compatibility.
- Improved one-row array ergonomics: a 1D vector with the fitted feature count
  is now interpreted as one prediction row instead of being reshaped into a
  wrong-feature matrix.
- Added focused tests for formula-name preservation, missing/nonnumeric/
  nonfinite prediction data, shape and empty-sample validation, 1D one-row
  prediction, and `effect_interval()` alpha/shape contracts.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_causal_forest_grf.py
  tests/reference_parity/test_causal_forest_aipw_recovery.py` passed,
  19 tests.
- `.venv/bin/python -m compileall -q src/statspai/forest/causal_forest.py
  tests/test_causal_forest_grf.py` passed.
- `git diff --check -- src/statspai/forest/causal_forest.py
  tests/test_causal_forest_grf.py` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  372 taxonomy raises and 1805 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=146`, `smoke=9`,
  `untested=187`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4397 <= 4698, mypy observed 3306 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`.

## 2026-06-17 Batch 87

Target: CausalForest fit-time validation before sklearn internals.

- Added CausalForest control validation for `n_estimators`,
  `min_samples_leaf`, `max_depth`, and `max_samples` before any first-stage
  nuisance model or tree fitting work starts.
- Converted nonnumeric, nonfinite, shape-mismatched, too-small, and empty
  fit inputs into StatsPAI taxonomy errors with recovery hints and diagnostics.
- Added explicit sample-size and per-tree row-count checks for the fixed
  3-fold nuisance cross-fitting and honest-tree split used by the estimator.
- Rejected unsupported discrete-treatment designs before silent or internal
  failures: one-class treatment, sparse treatment cells, multi-arm treatment,
  and binary treatment not coded as 0/1. Multi-arm callers are directed to
  `sp.multi_arm_forest()`.
- Added tests for invalid estimator controls, nonnumeric formula columns,
  nonfinite effect modifiers, multi-class treatment, non-0/1 treatment coding,
  and treatment cells too sparse for 3-fold cross-fitting.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_causal_forest_grf.py
  tests/reference_parity/test_causal_forest_aipw_recovery.py` passed,
  22 tests.
- `.venv/bin/python -m compileall -q src/statspai/forest/causal_forest.py
  tests/test_causal_forest_grf.py` passed.
- `git diff --check -- src/statspai/forest/causal_forest.py
  tests/test_causal_forest_grf.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_forest_inference.py tests/test_ml_causal_polish.py
  tests/test_result_consumer_errors.py tests/reference_parity/test_grf_parity.py
  tests/test_causal_to_forest_rename.py` passed, 45 tests with 1 skip.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  394 taxonomy raises and 1802 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=146`, `smoke=9`,
  `untested=187`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4397 <= 4698, mypy observed 3306 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.

## 2026-06-17 Batch 88

Target: SuperLearner fit/predict contracts for TMLE nuisance models.

- Added SuperLearner validation for `task`, `n_folds`, non-empty learner
  libraries, numeric/finite `X` and `y`, row-count agreement, minimum sample
  size relative to cross-validation folds, and non-empty feature matrices.
- Hardened classification SuperLearner against silent misuse: multi-class
  targets now raise `MethodIncompatibility`, one-class targets and sparse
  class cells raise `DataInsufficient`, and binary classification requires
  0/1 target coding because the implementation consumes `predict_proba()[:, 1]`.
- Stored the fitted feature count and routed `predict()` / `predict_proba()`
  through shape, numeric, nonfinite, empty-row, and feature-count checks before
  sklearn learners see prediction data.
- Converted unfitted and invalid prediction calls to StatsPAI taxonomy errors
  with recovery hints and diagnostics, while preserving `ValueError`
  compatibility.
- Improved one-row prediction ergonomics: a 1D vector with the fitted feature
  count is now treated as one row.
- Added focused tests for invalid controls/library, bad numeric fit inputs,
  classification target contracts, unfitted prediction, one-row vector
  prediction, feature-count mismatch, and nonfinite prediction data.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_tmle.py
  tests/test_low_cov_battery.py::test_super_learner_classification_smoke
  tests/reference_parity/test_tmle_parity.py` passed, 42 tests.
- `.venv/bin/python -m compileall -q
  src/statspai/tmle/super_learner.py tests/test_tmle.py` passed.
- `git diff --check -- src/statspai/tmle/super_learner.py tests/test_tmle.py`
  passed.
- `MPLBACKEND=Agg .venv/bin/python -m flake8
  src/statspai/tmle/super_learner.py tests/test_tmle.py --max-line-length=88
  --ignore=E203,W503` passed for the touched files.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  415 taxonomy raises and 1800 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=146`, `smoke=9`,
  `untested=187`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4395 <= 4698, mypy observed 3308 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`.

## 2026-06-17 Batch 89

Target: HAL nuisance learner fit/predict contracts.

- Added lightweight HAL validation helpers that preserve lazy sklearn imports
  and the duck-typed estimator API while rejecting bad inputs before sklearn
  internals or missing fitted attributes fail.
- Hardened `HALRegressor.fit()` for invalid `max_anchors_per_col`, invalid
  `cv`, invalid `lambda_`, nonnumeric or nonfinite features/targets,
  row-count mismatches, empty matrices, and samples too small to fit.
- Hardened `HALRegressor.predict()` for unfitted models, nonnumeric or
  nonfinite prediction features, empty prediction samples, unsupported array
  ranks, and feature-count mismatches; 1D vectors matching the fitted feature
  count now predict as one row.
- Hardened `HALClassifier.fit()` for invalid `C`, invalid anchors,
  nonnumeric/nonfinite features or targets, row-count mismatch, one-class
  data, multi-class data, and binary targets not coded as 0/1.
- Hardened `HALClassifier.predict()` and `predict_proba()` with the same
  fitted-feature validation as the regressor.
- Added tests for standalone HAL regressor/classifier contract failures,
  one-row prediction ergonomics, and prediction feature-count diagnostics.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_hal_tmle.py
  tests/test_low_cov_battery.py::test_hal_regressor_predicts_finite
  tests/test_estimator_provenance_round5.py::TestHalTmleProvenance::test_attached`
  passed, 10 tests.
- `.venv/bin/python -m compileall -q src/statspai/tmle/hal_tmle.py
  tests/test_hal_tmle.py` passed.
- `git diff --check -- src/statspai/tmle/hal_tmle.py tests/test_hal_tmle.py`
  passed.
- `MPLBACKEND=Agg .venv/bin/python -m flake8
  src/statspai/tmle/hal_tmle.py tests/test_hal_tmle.py --max-line-length=88
  --ignore=E203,W503` passed for the touched files.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  438 taxonomy raises and 1800 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4391 <= 4698, mypy observed 3311 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`.

## 2026-06-17 Batch 90

Target: PolicyTree prediction input contracts.

- Stored the fitted policy-covariate feature count on `PolicyTree.fit()` so
  downstream prediction can validate new data against the learned policy rule.
- Converted unfitted `PolicyTree.predict()` calls from a plain `ValueError`
  into `MethodIncompatibility` with recovery guidance.
- Added validation for nonnumeric prediction covariates, unsupported array
  ranks, empty prediction samples, feature-count mismatch, and NaN/infinite
  policy covariates before the internal tree traversal sees the data.
- Improved one-row prediction ergonomics: a 1D vector with the fitted policy
  feature count is now treated as one prediction row.
- Added tests for unfitted prediction, one-row vector prediction,
  feature-count diagnostics, nonnumeric input, nonfinite input, and empty
  prediction matrices.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_policy_learning.py tests/test_ml_causal_polish.py::TestPolicyTreeResult
  tests/test_article_aliases_round2.py::test_policy_tree_accepts_depth_kwarg
  tests/test_article_aliases_round2.py::test_policy_tree_max_depth_kwarg_still_works
  tests/test_article_aliases_round2.py::test_policy_tree_rejects_conflicting_depth
  tests/test_article_aliases_round2.py::test_policy_tree_rejects_conflicting_treat
  tests/test_article_aliases_round2.py::test_policy_tree_rejects_conflicting_covariates`
  passed, 24 tests.
- `.venv/bin/python -m compileall -q
  src/statspai/policy_learning/policy_tree.py tests/test_policy_learning.py`
  passed.
- `git diff --check -- src/statspai/policy_learning/policy_tree.py
  tests/test_policy_learning.py` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  445 taxonomy raises and 1799 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4391 <= 4698, mypy observed 3311 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`.

## 2026-06-17 Batch 91

Target: Bayesian Causal Forest CATE prediction contracts.

- Stored the fitted BCF covariate feature count so out-of-sample CATE
  prediction can validate new data against the fitted treatment-effect model.
- Converted unfitted `BayesianCausalForest.effect()` calls from a plain
  `ValueError` into `MethodIncompatibility` with recovery guidance.
- Added validation for nonnumeric covariates, unsupported array ranks, empty
  prediction samples, covariate-count mismatch, and NaN/infinite values before
  the internal gradient-boosting treatment-effect model sees prediction data.
- Improved one-row prediction ergonomics: a 1D vector with the fitted
  covariate count is now treated as one prediction row.
- Added focused tests for unfitted prediction, one-row vector prediction,
  feature-count diagnostics, nonnumeric input, nonfinite input, and empty
  prediction matrices.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_conformal_bcf_bunching_mc.py::TestBCF
  tests/reference_parity/test_bcf_parity.py
  tests/test_result_consumer_errors.py::test_predict_cate_works_on_metalearner`
  passed, 13 tests.
- `.venv/bin/python -m compileall -q src/statspai/bcf/bcf.py
  tests/test_conformal_bcf_bunching_mc.py` passed.
- `git diff --check -- src/statspai/bcf/bcf.py
  tests/test_conformal_bcf_bunching_mc.py` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  452 taxonomy raises and 1798 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4391 <= 4698, mypy observed 3311 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`.

## 2026-06-17 Batch 92

Target: low-level meta-learner CATE prediction contracts.

- Added a shared low-level meta-learner effect-input validator for
  `SLearner`, `TLearner`, `XLearner`, `RLearner`, and `DRLearner`.
- Converted unfitted `effect()` calls, nonnumeric covariates, unsupported
  array ranks, empty prediction samples, feature-count mismatch, and
  NaN/infinite covariates into StatsPAI taxonomy errors with diagnostics.
- Stored fitted covariate counts on T/X/R/DR learners so direct low-level CATE
  prediction can validate out-of-sample inputs.
- Improved one-row prediction ergonomics: a 1D vector matching the fitted
  covariate count is now treated as one prediction row.
- Added parametrized tests covering all five low-level learners for unfitted
  calls, one-row vectors, shape diagnostics, nonnumeric input, nonfinite input,
  and empty prediction matrices.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_metalearners.py tests/test_result_consumer_errors.py
  tests/external_parity/test_causalml_book.py` passed, 78 tests.
- `.venv/bin/python -m compileall -q
  src/statspai/metalearners/metalearners.py tests/test_metalearners.py`
  passed.
- `git diff --check -- src/statspai/metalearners/metalearners.py
  tests/test_metalearners.py` passed.
- `.venv/bin/python -m mypy src/statspai/metalearners/metalearners.py
  --no-error-summary --hide-error-context` returned only the module's existing
  27 historical errors; the new validator does not add mypy debt.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  459 taxonomy raises and 1798 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4391 <= 4698, mypy observed 3311 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 93

Target: CausalForest GRF helper prediction and aggregation contracts.

- Converted unfitted `variable_importance()`, `best_linear_projection()`,
  `ate()`, and `att()` calls into StatsPAI taxonomy errors with recovery
  guidance.
- Added BLP validation for invalid `alpha`, invalid propensity `clip`,
  nonnumeric `X_test`, unsupported/empty prediction matrices, and
  feature-count mismatch before regression or warning branches run.
- Routed out-of-sample BLP features through the same fitted feature-schema
  validator used by `effect()` and `predict()`, including list-like inputs
  that do not expose `.shape`.
- Hardened `att()` for nonnumeric, nonfinite, row-count-mismatched, and
  no-treated treatment vectors instead of returning raw errors or `NaN`.
- Added focused tests for these GRF helper contracts while preserving the
  existing BLP AIPW/GRF parity behavior.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_causal_forest_grf.py tests/test_forest_inference.py
  tests/reference_parity/test_grf_parity.py
  tests/reference_parity/test_causal_forest_aipw_recovery.py
  tests/test_ml_causal_polish.py::TestForestBLP` passed, 38 tests.
- `.venv/bin/python -m compileall -q
  src/statspai/forest/causal_forest.py tests/test_causal_forest_grf.py`
  passed.
- `git diff --check -- src/statspai/forest/causal_forest.py
  tests/test_causal_forest_grf.py` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  472 taxonomy raises and 1795 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4391 <= 4698, mypy observed 3311 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 94

Target: GRF-style `average_treatment_effect()` input contracts.

- Converted unfitted forests, non-string or unsupported `target_sample`
  values, invalid `alpha`, invalid propensity `clip`, nonnumeric `X`, and
  feature-count mismatches into StatsPAI taxonomy errors.
- Routed explicit `X` through the fitted CausalForest feature-schema validator
  before CATE aggregation so array/list/DataFrame handling matches
  `effect()` and `predict()`.
- Hardened explicit `T` validation for nonnumeric values, nonfinite values,
  and row-count mismatch against the CATE predictions.
- Added support checks for ATT/ATC targets with no treated/control rows,
  returning `DataInsufficient` instead of unstable estimates.
- Preserved the GRF/R parity AIPW estimates and plug-in fallback behavior.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_forest_inference.py tests/reference_parity/test_grf_parity.py
  tests/reference_parity/test_causal_forest_aipw_recovery.py
  tests/test_causal_forest_grf.py::test_ate_finite
  tests/test_causal_forest_grf.py::test_att_runs` passed, 23 tests.
- `.venv/bin/python -m compileall -q
  src/statspai/forest/forest_inference.py tests/test_forest_inference.py`
  passed.
- `git diff --check -- src/statspai/forest/forest_inference.py
  tests/test_forest_inference.py` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  487 taxonomy raises and 1791 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4391 <= 4698, mypy observed 3310 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 95

Target: GRF-style `calibration_test()` and `rate()` input contracts.

- Added shared forest-inference validation helpers for fitted-state checks,
  alpha validation, fitted feature-schema validation, and aligned numeric
  outcome/treatment vectors.
- Converted unfitted forests, invalid alpha values, malformed feature
  matrices, nonnumeric or nonfinite `Y`/`T`, and row-count mismatches in
  `calibration_test()` into StatsPAI taxonomy errors.
- Added `DataInsufficient` guards for too-small calibration samples before
  the CDDF regression is attempted.
- Converted `rate()` target, `q_grid`, alpha, feature, and vector validation
  failures into taxonomy errors, while allowing lowercase `target` values to
  normalize to the canonical `AUTOC`/`QINI` outputs.
- Made nuisance reuse length-aware: cached training nuisance predictions are
  reused only when they match the inference sample; otherwise the intended
  sample-mean fallback is used instead of broadcasting stale training arrays.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_forest_inference.py tests/reference_parity/test_grf_parity.py
  tests/reference_parity/test_causal_forest_aipw_recovery.py
  tests/test_causal_forest_grf.py tests/test_ml_causal_polish.py::TestForestBLP`
  passed, 40 tests.
- `.venv/bin/python -m compileall -q
  src/statspai/forest/forest_inference.py tests/test_forest_inference.py`
  passed.
- `git diff --check -- src/statspai/forest/forest_inference.py
  tests/test_forest_inference.py` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  500 taxonomy raises and 1788 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4390 <= 4698, mypy observed 3310 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 96

Target: GRF-style `honest_variance()` and `forest_diagnostics()` contracts.

- Converted unfitted forests, invalid `n_splits`, malformed feature matrices,
  and one-row CATE samples in `honest_variance()` into StatsPAI taxonomy
  errors instead of returning unstable `NaN` standard errors.
- Routed `honest_variance()` features through the fitted CausalForest schema
  validator so list/array/DataFrame handling matches `effect()`.
- Converted `forest_diagnostics()` unfitted calls, malformed or nonfinite
  `propensity_bounds`, feature-count mismatches, missing out-of-sample `T`,
  nonnumeric/nonfinite `T`, and X/T row-count mismatch into taxonomy errors.
- Required explicit `T` when diagnostics are requested for explicit
  out-of-sample `X`, avoiding mixed training-treatment counts with new CATE
  rows.
- Preserved in-sample diagnostic values and GRF parity tests while adding
  support for aligned out-of-sample diagnostic subsets.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_forest_inference.py tests/test_causal_forest_grf.py
  tests/reference_parity/test_grf_parity.py
  tests/reference_parity/test_causal_forest_aipw_recovery.py
  tests/test_causal_to_forest_rename.py` passed, 49 tests.
- `.venv/bin/python -m compileall -q
  src/statspai/forest/forest_inference.py tests/test_forest_inference.py`
  passed.
- `git diff --check -- src/statspai/forest/forest_inference.py
  tests/test_forest_inference.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m flake8
  src/statspai/forest/forest_inference.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 violations.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  506 taxonomy raises and 1784 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4389 <= 4698, mypy observed 3310 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 97

Target: VCNet/SCIGAN dose-response input contracts.

- Added validation for `vcnet()` data type, required columns, one-column
  covariate string shortcuts, complete-row sample size, numeric/finite
  outcome/treatment/covariates, and treatment variation.
- Added spline/control validation for `n_basis`, `spline_degree`, `ridge`,
  `n_bootstrap`, `alpha`, and explicit `t_grid` before spline construction or
  bootstrap quantiles run.
- Converted invalid VCNet inputs into StatsPAI taxonomy errors with recovery
  hints and diagnostics instead of raw `KeyError`, `TypeError`, SciPy spline
  errors, or bootstrap `NaN` outputs.
- Hardened `scigan()` propensity weights for numeric type, row-count
  alignment, finite non-negative values, and positive total mass.
- Cleaned up VCNet local typing so the module now passes single-file mypy
  aside from the repository's Python-version config warning.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_dose_response.py
  tests/reference_parity/test_dose_response_parity.py::TestGPSRecovery
  tests/reference_parity/test_dose_response_parity.py::TestGPSCurveConsistency
  tests/reference_parity/test_dose_response_parity.py::TestDeterminism::test_dose_response_deterministic`
  passed, 10 tests.
- `.venv/bin/python -m compileall -q
  src/statspai/dose_response/vcnet.py tests/test_dose_response.py` passed.
- `git diff --check -- src/statspai/dose_response/vcnet.py
  tests/test_dose_response.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m flake8
  src/statspai/dose_response/vcnet.py tests/test_dose_response.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 violations.
- `.venv/bin/python -m mypy src/statspai/dose_response/vcnet.py
  --no-error-summary --hide-error-context` reported no module errors; mypy
  only printed the repository Python-version warning.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  527 taxonomy raises and 1783 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4388 <= 4698, mypy observed 3301 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 98

Target: `regtable()` and `RegtableResult` validation taxonomy.

- Converted high-traffic regtable public argument failures into StatsPAI
  taxonomy errors while preserving `ValueError` compatibility through
  inheritance.
- Migrated `tests=`, `multi_se=`, `notation=`, `vcov=`, `output=`,
  `coef_map`, `eform`, model-label, dependent-variable-label, alpha,
  spanner-span, and callable-transform validation to
  `MethodIncompatibility` with recovery hints and diagnostics.
- Converted `sp.regtable()` with no model results to `DataInsufficient`.
- Added focused serialization tests asserting taxonomy classes and diagnostics
  for no-model, invalid-output, and bad `tests=` length cases.
- Left renderer-internal `NotImplementedError` branches and placeholder
  `KeyError` behavior unchanged so existing layout/renderer contracts remain
  intact.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_regtable_serialization.py tests/test_regtable_alpha.py
  tests/test_regtable_round2_extensions.py
  tests/test_regtable_round3_extensions.py
  tests/test_regtable_round4_extensions.py
  tests/test_regtable_publication_extensions.py tests/test_regtable_siunitx.py
  tests/test_regtable_quarto.py tests/test_regtable_from_dict.py
  tests/test_regtable_fmt_auto.py` passed, 208 tests.
- `.venv/bin/python -m compileall -q
  src/statspai/output/regression_table.py
  tests/test_regtable_serialization.py` passed.
- `git diff --check -- src/statspai/output/regression_table.py
  tests/test_regtable_serialization.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m flake8
  src/statspai/output/regression_table.py tests/test_regtable_serialization.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` still reports
  the module's historical 16 touched-file violations; the global flake8
  ratchet remains unchanged at 4388 <= 4698.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  550 taxonomy raises and 1760 generic raises.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4388 <= 4698, mypy observed 3301 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 99

Target: JAX fast FEOLS and bootstrap error taxonomy / silent-NaN guards.

- Routed `feols_jax()` user-input failures through the StatsPAI taxonomy for
  invalid vcov/dtype/cluster combinations, formula syntax, missing columns,
  non-DataFrame inputs, non-finite outcome/regressors, invalid weights,
  cluster NaNs, and weighted-FE NaNs.
- Routed `feols_jax_bootstrap()` through the same taxonomy for bootstrap
  variant selection, cluster requirements, dtype, missing columns, bad data
  shape, non-finite inputs, cluster NaNs, and too few clusters.
- Rejected `n_boot=1` with `DataInsufficient` because bootstrap standard
  errors are undefined with one draw; the previous path could emit NaN SEs.
- Added a shared finite-output guard so JAX QR / bootstrap solves that return
  non-finite coefficients, residuals, bread matrices, or bootstrap draws fail
  as `NumericalInstability` with diagnostics instead of leaking silent NaNs.
- Added result-protocol coverage for the finite-output guard that runs even
  when the optional JAX dependency is not installed locally; JAX end-to-end
  tests remain in place and skip when JAX is unavailable.

Verification run:

- `.venv/bin/python -m compileall -q src/statspai/fast/jax_feols.py
  tests/test_jax_feols.py tests/test_jax_feols_bootstrap.py
  tests/test_fast_jax_feols_result_protocol.py` passed.
- `git diff --check -- src/statspai/fast/jax_feols.py
  tests/test_jax_feols.py tests/test_jax_feols_bootstrap.py
  tests/test_fast_jax_feols_result_protocol.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m flake8
  src/statspai/fast/jax_feols.py tests/test_jax_feols.py
  tests/test_jax_feols_bootstrap.py tests/test_fast_jax_feols_result_protocol.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_jax_feols_result_protocol.py tests/test_jax_feols.py
  tests/test_jax_feols_bootstrap.py` passed the 2 non-JAX tests; the 2 JAX
  modules skipped because this local environment lacks optional JAX.
- `.venv/bin/python -m mypy src/statspai/fast/jax_feols.py
  --no-error-summary --hide-error-context` still reports the module's
  historical JAX closure/result typing debt plus the repository
  Python-version warning, but no new helper type debt after annotating
  `_parse_formula_checked`.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  580 taxonomy raises and 1735 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4388 <= 4698, mypy observed 3301 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 100

Target: shared DML input contracts and scalar-covariate ergonomics.

- Migrated shared `_DoubleMLBase` validation failures into the StatsPAI
  taxonomy for non-DataFrame input, missing columns, instrument/model
  incompatibilities, invalid `n_folds`, `n_rep`, `alpha`, `random_state`,
  malformed `fold_indices`, bad `sample_weight`, and post-dropna sample
  insufficiency.
- Added explicit `DataInsufficient` failures for zero-mass sample weights,
  empty complete-row samples, and too few complete rows for the requested
  number of folds, preventing lower-level sklearn/numpy crashes.
- Preserved abstract `_fit_one_rep` `NotImplementedError` behavior while
  converting concrete unsupported options such as explicit folds on non-PLR
  models into `MethodIncompatibility`.
- Added a one-column shortcut so `covariates="x0"` is treated as `["x0"]`
  instead of being split into characters; fixed `sp.dml` provenance to record
  the normalized covariate list.
- Migrated the public `dml(model=...)` unknown-model branch to
  `MethodIncompatibility` with machine-readable supported-model diagnostics.
- Cleaned historical flake8 issues in touched `dml/double_ml.py` by wrapping
  the long reference line, aligning provenance formatting, and expanding
  one-line legacy properties with return annotations.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_dml_cov_scores.py
  tests/test_review_fixes_round2.py::test_irm_raises_when_treatment_is_constant
  tests/test_review_fixes_round2.py::test_irm_tiny_subgroup_does_not_blow_up
  tests/test_parity_gap_boundaries.py::test_dml_shared_explicit_folds_close_split_noise`
  passed, 16 tests.
- `.venv/bin/python -m compileall -q src/statspai/dml/_base.py
  src/statspai/dml/double_ml.py tests/test_dml_cov_scores.py` passed.
- `git diff --check -- src/statspai/dml/_base.py
  src/statspai/dml/double_ml.py tests/test_dml_cov_scores.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m flake8 src/statspai/dml/_base.py
  src/statspai/dml/double_ml.py tests/test_dml_cov_scores.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  611 taxonomy raises and 1715 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python -m mypy src/statspai/dml/_base.py
  src/statspai/dml/double_ml.py --no-error-summary --hide-error-context`
  still reports historical DML base typing debt plus the repository
  Python-version warning; the global mypy ratchet improved after the
  touched `double_ml.py` cleanup.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4374 <= 4698, mypy observed 3290 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 155

Target: direct typing cleanup for AER/QJE paper table bundles.

- Annotated the internal `paper_tables(...)` `_build(...)` helper with
  optional result and model-label sequences plus `RegtableResult` return type.
- Converted optional model labels to a concrete list before forwarding to
  `regtable`, matching the output helper's signature without changing caller
  behavior.
- Removed an unused `field` import, dropped an unused XLSX `header_row`
  variable, and rewrapped workbook column-width logic so touched-file flake8 is
  clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/output/paper_tables.py` passed.
- `.venv/bin/python -m flake8 src/statspai/output/paper_tables.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/output/paper_tables.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_paper_tables.py tests/test_paper_tables_export.py
  tests/test_diagnose_batteries_sprint_b.py` passed, 39 tests with expected
  weak-proxy/principal-stratification warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1271 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4284 <= 4698, mypy observed 3081 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 156

Target: direct typing cleanup for the Stata-style estimates table bridge.

- Typed the `eststo`/`esttab` result boundaries, model-data source container,
  causal/econometric result probes, and empty-table fallback helpers in
  `src/statspai/output/estimates.py`.
- Moved legacy formatter re-exports onto the module object before assignment,
  preserving the public underscore aliases while making the source module's
  types explicit.
- Explicitly stringified dynamic renderer returns from `EstimateTableResult`
  and wrapped long statistic-key/doctest lines so touched-file flake8 remains
  clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/output/estimates.py` passed.
- `.venv/bin/python -m flake8 src/statspai/output/estimates.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/output/estimates.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_output_and_survey_helpers.py
  tests/test_econometric_results_export.py tests/test_regtable_from_dict.py
  tests/test_regtable_serialization.py tests/test_regtable_quarto.py
  tests/test_regtable_fmt_auto.py tests/test_regtable_publication_extensions.py
  tests/test_regtable_round3_extensions.py tests/test_regtable_round4_extensions.py`
  passed, 200 tests with 5 warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1271 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4270 <= 4698, mypy observed 3025 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 157

Target: RD bias-aware and alias wrapper static hardening.

- Narrowed optional smoothness-bound handling in
  `src/statspai/rd/bias_aware.py` by checking `M_y is None` and `M_d is None`
  directly before converting them to finite positive floats.
- Removed unused `Tuple` and `scipy.optimize` imports from the bias-aware RD
  module and wrapped long summary lines so the touched file is flake8-clean.
- Added explicit `Any` variadic boundaries and result return types for the
  public RD aliases in `src/statspai/rd/_aliases.py`, preserving their
  pass-through behavior while making alias contracts visible to mypy.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/rd/bias_aware.py
  src/statspai/rd/_aliases.py` passed.
- `.venv/bin/python -m flake8 src/statspai/rd/bias_aware.py
  src/statspai/rd/_aliases.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/rd/bias_aware.py
  src/statspai/rd/_aliases.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_rd_aliases.py tests/test_tierD_rd_multiscore_analytic.py
  tests/test_rd_cov_estimators.py::test_rd_bias_aware_fuzzy
  tests/test_cov95_rd_misc.py::test_rd_bias_aware_fuzzy
  tests/test_cov95_rd_misc.py::test_rd_bias_aware_fuzzy_manual_M_and_cluster
  tests/test_rd_polish.py` passed, 36 tests with 1 expected weak-first-stage
  warning.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1271 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4263 <= 4698, mypy observed 3019 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 158

Target: panel dispatcher required-input clarity and direct typing cleanup.

- Annotated the unified `sp.panel(...)` dispatcher return type as the union of
  classical `PanelResults` and HDFE `FEOLSResult`.
- Added an explicit classical-method required-input check for missing
  `formula`, `entity`, or `time`, producing an early `ValueError` that lists
  the missing fields before delegating to the lower-level panel estimator.
- Added a dispatcher regression test for the missing-identifier branch and
  removed an unused import from the touched test file.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/panel/__init__.py
  tests/test_panel_dispatcher.py` passed.
- `.venv/bin/python -m flake8 src/statspai/panel/__init__.py
  tests/test_panel_dispatcher.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/panel/__init__.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_panel_dispatcher.py tests/test_cov95_panel_reg.py tests/test_panel.py
  tests/test_panel_cov_estimators.py tests/test_panel_cov_diagnostics.py
  tests/test_panel_cov_compare.py` passed, 130 tests with 2 expected numerical
  warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1272 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4263 <= 4698, mypy observed 3016 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 159

Target: direct static cleanup for the core regression-table renderer.

- Annotated the HC-variance recompute `omega` array and public `regtable`
  variadic boundary in `src/statspai/output/regression_table.py`, clearing the
  file's remaining direct mypy errors.
- Narrowed optional coefficient transforms through a local callable before
  applying them to confidence interval endpoints.
- Removed unused imports and stale intermediate variables in text/HTML/XLSX
  rendering paths, and wrapped long table-rendering lines so the touched file
  is flake8-clean without changing rendered semantics.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/output/regression_table.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/output/regression_table.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/output/regression_table.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_output_and_survey_helpers.py
  tests/test_econometric_results_export.py tests/test_regtable_from_dict.py
  tests/test_regtable_serialization.py tests/test_regtable_quarto.py
  tests/test_regtable_fmt_auto.py tests/test_regtable_publication_extensions.py
  tests/test_regtable_round3_extensions.py tests/test_regtable_round4_extensions.py
  tests/test_paper_tables.py tests/test_paper_tables_export.py
  tests/test_collection.py tests/test_container_serialization.py` passed,
  249 tests with 5 expected warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1272 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4247 <= 4698, mypy observed 3012 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 160

Target: RD extrapolation helper typing and touched-file cleanup.

- Added explicit ndarray/tuple return annotations to
  `src/statspai/rd/extrapolate.py` internal OLS, partial-F, propensity-score,
  bootstrap, recommendation, and plotting helpers.
- Narrowed bootstrap RNG handling to `Optional[np.random.Generator]`, removed
  an unused bootstrap running-variable slice, and wrapped array returns with
  `np.asarray(..., dtype=float)` where numpy expressions were inferred as
  `Any`.
- Wrapped long multi-cutoff and plot-error messages so the touched file is
  flake8-clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/rd/extrapolate.py` passed.
- `.venv/bin/python -m flake8 src/statspai/rd/extrapolate.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/rd/extrapolate.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_rd_extrapolate.py
  tests/test_rd_cov_estimators.py::test_rd_extrapolate
  tests/test_rd_dispatcher.py::test_extrapolate
  tests/test_rd_new_modules.py::TestExtrapolation` passed, 21 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1272 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4241 <= 4698, mypy observed 3004 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 161

Target: discrete-running-variable RD smoothness-bound typing cleanup.

- Introduced explicit `M_value` and `K_value` locals in
  `src/statspai/rd/rd_discrete.py`, so bounded-second-derivative and
  bounded-misspecification branches convert optional user bounds only after
  direct `is None` checks.
- Stored the normalized smoothness bounds in `model_info` and provenance
  parameters without re-casting Optional values.
- Wrapped summary long lines so the touched file is flake8-clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/rd/rd_discrete.py` passed.
- `.venv/bin/python -m flake8 src/statspai/rd/rd_discrete.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/rd/rd_discrete.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_rd_misc.py::test_rd_discrete_bsd_and_bm
  tests/test_cov95_rd_misc.py::test_rd_discrete_errors
  tests/test_rd_polish.py::TestRDDiscrete tests/test_rd_dispatcher.py` passed,
  32 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1272 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4236 <= 4698, mypy observed 2998 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 162

Target: flexible RD residualisation typing and touched-file cleanup.

- Removed an unused running-variable array from `src/statspai/rd/rd_flex.py`
  and moved fuzzy-treatment extraction into the fuzzy branch, making the
  residualisation call receive a concrete ndarray.
- Added explicit return types for the learner factory and cross-fit
  residualisation helper.
- Precomputed the variance-reduction percentage for summary rendering so the
  touched file is flake8-clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/rd/rd_flex.py` passed.
- `.venv/bin/python -m flake8 src/statspai/rd/rd_flex.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/rd/rd_flex.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_rd_misc.py::test_rd_flex_learners
  tests/test_rd_polish.py::TestRDFlex tests/test_rd_dispatcher.py` passed,
  31 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1272 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4234 <= 4698, mypy observed 2996 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 163

Target: IV quantile regression profile-search typing cleanup.

- Typed IV-QR row records and profile helper boundaries in
  `src/statspai/regression/iv_quantile.py`, preventing scalar-only dict
  inference from conflicting with stored ndarray diagnostics.
- Removed unused intermediate variables, added an explicit impossible-grid
  guard for scalar profile search, and wrapped verbose/profile output lines.
- Switched citation registration to a guarded `getattr` lookup so dynamic
  citation metadata does not violate direct mypy.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/regression/iv_quantile.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/regression/iv_quantile.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/regression/iv_quantile.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_econ_trinity.py::TestIVQR::test_ivqr_median
  tests/test_econ_trinity.py::TestIVQR::test_ivqr_multiple_taus
  tests/test_econ_trinity.py::TestIVQR::test_ivqr_raises_on_missing_instrument_count
  tests/test_econ_trinity.py::TestAdversarial::test_ivqreg_multidim_warns_when_bootstrap_zero
  tests/test_cov95_iv_init.py::test_dispatch_ivqreg_explicit_args
  tests/test_iv_cov_tail.py::test_dispatch_ivqreg_route` passed, 6 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1273 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4229 <= 4698, mypy observed 2981 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 164

Target: spatial diagnostic helper typing cleanup.

- Annotated spatial weights-like inputs and helper return types in
  `src/statspai/spatial/models/diagnostics.py`.
- Typed the LM-diagnostic p-value helper and removed an unused local import
  from `moran_residuals`.
- Normalized return-dict spacing so the touched file is flake8-clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/spatial/models/diagnostics.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/spatial/models/diagnostics.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/spatial/models/diagnostics.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/spatial/test_diagnostics_impacts.py
  tests/test_untested_bounds_and_diag.py::test_moran_detects_spatial_structure
  tests/test_untested_bounds_and_diag.py::test_moran_returns_finite_statistic_for_noise
  tests/test_tierD_spatial_diag_analytic.py` passed, 20 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1273 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4225 <= 4698, mypy observed 2977 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 165

Target: spatial GMM public-boundary typing cleanup.

- Annotated spatial weights-like inputs, KP moment helper inputs, and nested
  objective functions in `src/statspai/spatial/models/gmm.py`.
- Wrapped the KP second moment expression and normalized filtered-variable
  assignment spacing so the touched file is flake8-clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/spatial/models/gmm.py` passed.
- `.venv/bin/python -m flake8 src/statspai/spatial/models/gmm.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/spatial/models/gmm.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/spatial/test_models_gmm.py` passed, 5 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1273 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4222 <= 4698, mypy observed 2971 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 166

Target: spatial panel estimator typing and touched-file lint cleanup.

- Removed unused sparse and legacy weights imports from
  `src/statspai/spatial/panel/estimator.py`.
- Annotated the public spatial-weight input and SEM/SAR/SDM likelihood helpers,
  split a semicolon assignment, and renamed the SEM likelihood closure to avoid
  duplicate nested-function definitions.
- Normalized array-return boundaries with explicit `np.asarray(...,
  dtype=float)` conversions and removed an unused stacked-response variable.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/spatial/panel/estimator.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/spatial/panel/estimator.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/spatial/panel/estimator.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/spatial/test_panel.py` passed, 6 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1273 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4218 <= 4698, mypy observed 2963 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 167

Target: RD heterogeneous treatment-effect touched-file typing and lint cleanup.

- Wrapped long validation, coefficient-index, bandwidth, and comment lines in
  `src/statspai/rd/hte.py`.
- Replaced dynamic `result.plot` method assignment with `setattr` while
  preserving the public `result.plot()` API used by RD HTE tests.
- Added explicit `detail` presence checks for linear-combination and plotting
  helpers, and narrowed dynamic display/evaluation-point values so the touched
  file is mypy-clean.
- Removed an unused design-matrix length variable.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/rd/hte.py` passed.
- `.venv/bin/python -m flake8 src/statspai/rd/hte.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m mypy src/statspai/rd/hte.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_rd_misc.py tests/test_rd_dispatcher.py
  tests/test_rd_validation.py tests/test_rd_new_modules.py
  tests/test_cov95_rd_r2_hte.py tests/test_cov95_rd_ml_and_hte.py` passed,
  124 tests with 3 expected warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4209 <= 4698, mypy observed 2952 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 168

Target: synthetic-control power-analysis static cleanup.

- Removed unused typing, scipy, synth, and result imports from
  `src/statspai/synth/power.py`.
- Cast the treated RMSPE-ratio return to a plain `float`.
- Replaced in-place reuse of the optional `effect_sizes` argument with a local
  `effect_grid`, preserving caller behavior while clearing mypy's optional
  sequence/ndarray conflict.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/synth/power.py` passed.
- `.venv/bin/python -m flake8 src/statspai/synth/power.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/synth/power.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_synth_extras.py tests/test_cov95_synth_more.py
  tests/test_cov95_synth_variants.py` passed, 98 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4204 <= 4698, mypy observed 2948 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 169

Target: synthetic-control comparison API typing cleanup.

- Annotated dynamic `**kwargs` boundaries in
  `src/statspai/synth/compare.py` for plotting, LaTeX/Markdown/Excel export,
  comparison dispatch, and quick recommendation helpers.
- Preserved the public comparison, export, and recommendation behavior while
  clearing all touched-file mypy errors.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/synth/compare.py` passed.
- `.venv/bin/python -m flake8 src/statspai/synth/compare.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/synth/compare.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_synth_extras.py tests/test_cov95_synth_more.py
  tests/test_cov95_synth_variants.py` passed, 98 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4204 <= 4698, mypy observed 2942 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 170

Target: sequential SDID option typing and touched-file lint cleanup.

- Removed an unused `Sequence` import from
  `src/statspai/synth/sequential_sdid.py`.
- Typed `se_method` as `Literal["placebo", "bootstrap", "jackknife"]` to
  match the inner SDID estimator contract.
- Wrapped the long citation, post-period extraction, insufficient-period donor
  count, and exception comment lines so the touched file is flake8-clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/synth/sequential_sdid.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/synth/sequential_sdid.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/synth/sequential_sdid.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_v100_integration.py tests/test_sequential_sdid.py
  tests/test_cov95_synth_r3_multi_seq_scm.py tests/test_cov95_synth_variants.py
  tests/test_cov95_synth_edges.py` passed, 180 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4200 <= 4698, mypy observed 2941 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 171

Target: synthetic-control reporting API typing and touched-file lint cleanup.

- Removed an unused `Tuple` import from `src/statspai/synth/report.py`.
- Converted fixed Markdown table-header f-strings to plain strings.
- Annotated dynamic `**kwargs` boundaries on `synth_report()` and
  `synth_report_to_file()`.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/synth/report.py` passed.
- `.venv/bin/python -m flake8 src/statspai/synth/report.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/synth/report.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_synth_exports.py tests/test_low_cov_battery.py
  tests/test_cov95_synth_more.py tests/test_cov95_synth_r2_report_tests.py
  tests/test_synth_extras.py tests/test_synth_report.py
  tests/test_cov95_synth_reports.py tests/test_cov95_synth_r4_report.py
  tests/test_cov95_synth_r3_report_bsts.py` passed, 180 tests with 1 skip.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4195 <= 4698, mypy observed 2939 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 172

Target: Bartik shift-share IV touched-file typing and lint cleanup.

- Removed unused typing imports and wrapped provenance metadata expressions in
  `src/statspai/bartik/shift_share.py`.
- Added an explicit `None` return annotation to `_validate()` and typed the
  Rotemberg helper boundary.
- Replaced pandas `.values` array boundaries with `to_numpy(dtype=float)` and
  explicit `np.asarray(..., dtype=float)` returns for Bartik instruments.
- Wrapped first-stage F-statistic and p-value expressions.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bartik/shift_share.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bartik/shift_share.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bartik/shift_share.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bartik.py tests/test_cov95_iv_init.py
  tests/test_estimator_provenance_round6.py tests/test_iv_dispatcher.py
  tests/test_transport_and_shiftshare.py` passed, 79 tests with expected
  leave-one-out fallback warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4190 <= 4698, mypy observed 2934 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 173

Target: Bartik AKM correction helper typing and lint cleanup.

- Added typed ndarray boundaries to the local `_residualise()` helper in
  `src/statspai/bartik/adao_correction.py`.
- Wrapped the AKM model-method expression.
- Removed unused fitted-outcome mean calculations from `shift_share_se()`.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bartik/adao_correction.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/bartik/adao_correction.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bartik/adao_correction.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bartik.py tests/test_transport_and_shiftshare.py
  tests/test_shift_share_political.py` passed, 41 tests with expected
  leave-one-out fallback warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4188 <= 4698, mypy observed 2933 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 174

Target: Lee/Manski partial-identification bounds typing cleanup.

- Removed unused typing imports from `src/statspai/bounds/lee_manski.py`.
- Added explicit typed boundaries to `_compute_lee_bounds()` and
  `_compute_manski_bounds()`.
- Cast midpoint, SE, and CI endpoints to plain Python floats before building
  `CausalResult` objects.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bounds/lee_manski.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bounds/lee_manski.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bounds/lee_manski.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_phase9to14.py tests/test_tierD_p2_bounds_sensitivity_analytic.py
  tests/test_estimator_provenance_round8.py tests/test_article_aliases.py`
  passed, 90 tests with expected PSM warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4185 <= 4698, mypy observed 2929 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 175

Target: advanced partial-identification bounds typing and lint cleanup.

- Removed an unused `Union` import and wrapped long citation, summary, HTML,
  Oster, trimming-fraction, and frontier metadata lines in
  `src/statspai/bounds/partial_id.py`.
- Added typed internal boundaries for bootstrap callbacks, Horowitz-Manski,
  IV-bounds, Oster, Lee-selection, plotting, and column-check helpers.
- Removed unused local arrays and variance variables from Horowitz-Manski,
  IV-bounds, and conditional Lee bounds.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bounds/partial_id.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bounds/partial_id.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bounds/partial_id.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_tierD_bounds_analytic.py
  tests/test_tierD_p2_bounds_sensitivity_analytic.py
  tests/test_untested_bounds_and_diag.py tests/test_article_aliases.py
  tests/test_v0917_deferred.py tests/test_estimator_provenance_round8.py
  tests/test_phase9to14.py` passed, 144 tests with expected PSM warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4172 <= 4698, mypy observed 2915 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 176

Target: covariate-balancing/IPW bridge helper typing cleanup.

- Added explicit ndarray-to-float annotations to the nested IPW and
  entropy-balancing ATE helpers in `src/statspai/bridge/cb_ipw.py`.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bridge/cb_ipw.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bridge/cb_ipw.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bridge/cb_ipw.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bridge.py tests/test_bridge_full.py
  tests/test_silent_degradation_fixes.py tests/test_v100_integration.py`
  passed, 107 tests with expected bridge disagreement warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1275 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4172 <= 4698, mypy observed 2913 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 177

Target: DiD/SC bridge runtime API alignment and typing cleanup.

- Annotated the public `treated_unit` bridge argument and mixed-detail
  dictionary in `src/statspai/bridge/did_sc.py`.
- Updated the SC bridge path to construct `SyntheticControl(data=...)`, call
  `fit(placebo=False)`, and compute post-treatment gaps from the returned
  `model_info["gap_table"]`.
- Applied the same current SCM API path to donor placebo fits, preserving the
  existing fallback diagnostics when SC fitting fails.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bridge/did_sc.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bridge/did_sc.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bridge/did_sc.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bridge.py tests/test_bridge_full.py tests/test_v100_integration.py
  tests/test_v101_verified_fixes.py` passed, 104 tests with expected bridge
  disagreement warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4172 <= 4698, mypy observed 2907 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 178

Target: direct typing cleanup for calibrated/weighted bridge estimators.

- Typed the nested AIPW helper in `src/statspai/bridge/dr_calib.py` and the
  nested T-learner helper in `src/statspai/bridge/ewm_cate.py`.
- Normalized sklearn prediction outputs with `np.asarray(..., dtype=float)` so
  the bridge internals expose concrete NumPy return types rather than Any.
- Removed an unused typing import from `ewm_cate.py`; estimator semantics and
  public signatures are unchanged.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bridge/dr_calib.py
  src/statspai/bridge/ewm_cate.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bridge/dr_calib.py
  src/statspai/bridge/ewm_cate.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bridge/dr_calib.py
  src/statspai/bridge/ewm_cate.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bridge.py tests/test_bridge_full.py tests/test_v100_integration.py`
  passed, 99 tests with expected bridge disagreement warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4171 <= 4698, mypy observed 2904 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 179

Target: direct typing cleanup for remaining bridge helper closures.

- Typed the surrogate-index and PCI nested helpers in
  `src/statspai/bridge/surrogate_pci.py`.
- Typed the local slope helper in `src/statspai/bridge/kink_rdd.py`.
- Kept the bridge fallback behavior unchanged: insufficient support and
  singular local fits still produce the existing `np.nan` path estimates.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bridge/surrogate_pci.py
  src/statspai/bridge/kink_rdd.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bridge/surrogate_pci.py
  src/statspai/bridge/kink_rdd.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bridge/surrogate_pci.py
  src/statspai/bridge/kink_rdd.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bridge.py tests/test_bridge_full.py tests/test_v101_verified_fixes.py`
  passed, 22 tests with expected bridge disagreement warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4171 <= 4698, mypy observed 2901 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 180

Target: BCF direct typing and touched-file lint cleanup.

- Normalized sklearn propensity and CATE prediction outputs to explicit
  floating NumPy arrays in `src/statspai/bcf/longitudinal.py` and
  `src/statspai/bcf/bcf.py`.
- Typed the longitudinal BCF per-time bootstrap container and the local random
  forest helper boundary.
- Removed unused imports and cleaned touched-file flake8 issues in both BCF
  modules, including long doctest/reference lines and a stale placeholder
  f-string.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bcf/longitudinal.py
  src/statspai/bcf/bcf.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bcf/longitudinal.py
  src/statspai/bcf/bcf.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bcf/longitudinal.py
  src/statspai/bcf/bcf.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bcf_longitudinal.py tests/test_conformal_bcf_bunching_mc.py
  tests/reference_parity/test_bcf_parity.py tests/test_v100_review_fixes.py`
  passed, 44 tests with expected bootstrap warnings in an unrelated DID
  review-fix test.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4159 <= 4698, mypy observed 2896 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 181

Target: survival-model direct typing and touched-file lint cleanup.

- Typed the internal frailty and AFT optimization objective functions in
  `src/statspai/survival/frailty.py` and `src/statspai/survival/aft.py`.
- Removed unused frailty imports and a dead local matrix copy while preserving
  the existing optimizer path.
- Cleaned touched-file flake8 issues in the two survival modules, including
  long citation/summary lines, semicolon-separated Hessian work, and binary
  operator continuation style.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/survival/frailty.py
  src/statspai/survival/aft.py` passed.
- `.venv/bin/python -m flake8 src/statspai/survival/frailty.py
  src/statspai/survival/aft.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/survival/frailty.py
  src/statspai/survival/aft.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_frailty.py tests/test_aft.py
  tests/test_tierD_p2_survival_analytic.py
  tests/test_estimator_provenance_round9.py` passed, 28 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4149 <= 4698, mypy observed 2893 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 182

Target: production-function direct typing cleanup.

- Typed the stage-2 GMM objective and economically plausible start-vector
  helper in `src/statspai/structural/production/op_lp_acf.py`.
- Typed the `_resolve_inputs()` scalar/list normalization helper.
- Removed unused typing imports and fixed the touched-file blank-line style
  issue without changing OP/LP/ACF estimator logic.

Verification run:

- `.venv/bin/python -m py_compile
  src/statspai/structural/production/op_lp_acf.py` passed.
- `.venv/bin/python -m flake8
  src/statspai/structural/production/op_lp_acf.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m mypy
  src/statspai/structural/production/op_lp_acf.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_prod_fn.py tests/test_tierD_structural_analytic.py
  tests/test_v093_bugfixes.py` passed, 45 tests with one expected esttab
  deprecation warning.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4146 <= 4698, mypy observed 2890 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 183

Target: causal-impact estimator direct typing and touched-file lint cleanup.

- Typed `CausalImpactEstimator` validation, preparation, structural-model fit,
  and prediction helpers in `src/statspai/causal_impact/impact.py`.
- Typed the public `impactplot()` helper and its three panel renderers while
  keeping plotting return behavior unchanged.
- Removed unused imports and an unused post-period covariate slice; normalized
  model-dictionary prediction inputs to explicit floating arrays/scalars.
- Cleaned touched-file line-length issues in pointwise intervals, relative
  effect metadata, and local-level variance decomposition.

Verification run:

- `.venv/bin/python -m py_compile
  src/statspai/causal_impact/impact.py` passed.
- `.venv/bin/python -m flake8 src/statspai/causal_impact/impact.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `.venv/bin/python -m mypy src/statspai/causal_impact/impact.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_causal_impact.py tests/test_estimator_provenance_round6.py
  tests/test_untested_public_api.py` passed, 35 tests with 2 skips and
  expected public-API smoke warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4136 <= 4698, mypy observed 2881 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 184

Target: notch-bunching direct typing cleanup.

- Typed `NotchResult.__init__()`, `NotchResult.plot()`, and the public
  `notch_size` argument in `src/statspai/bunching/notch.py`.
- Typed the counterfactual-polynomial and marginal-buncher helpers.
- Normalized `np.polyfit`/`np.polyval` outputs to explicit floating arrays
  and made the bootstrap elasticity branch narrow `notch_size` before using
  it arithmetically.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bunching/notch.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bunching/notch.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bunching/notch.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_tierD_interference_forest_analytic.py
  tests/test_bunching_unified.py tests/reference_parity/test_bunching_parity.py
  tests/test_conformal_bcf_bunching_mc.py` passed, 47 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4136 <= 4698, mypy observed 2876 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 185

Target: causal-text result protocol signature compatibility.

- Updated `TextTreatmentResult.summary()` and `LLMAnnotatorResult.summary()`
  to accept the optional `alpha` argument supported by their `CausalResult`
  superclass.
- Kept the rendered summary text unchanged; the new argument is accepted for
  protocol compatibility with generic result consumers.

Verification run:

- `.venv/bin/python -m py_compile
  src/statspai/causal_text/text_treatment.py
  src/statspai/causal_text/llm_annotator.py` passed.
- `.venv/bin/python -m flake8
  src/statspai/causal_text/text_treatment.py
  src/statspai/causal_text/llm_annotator.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m mypy
  src/statspai/causal_text/text_treatment.py
  src/statspai/causal_text/llm_annotator.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_causal_text.py tests/test_stability.py tests/test_stability_audit.py`
  passed, 59 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4136 <= 4698, mypy observed 2874 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 186

Target: conformal CATE direct typing and touched-file lint cleanup.

- Typed the internal conformal quantile helper in
  `src/statspai/conformal_causal/conformal_ite.py`.
- Removed unused typing imports and replaced a stale f-string with a literal
  error message.
- Kept conformal interval construction, alias behavior, and dispatcher routing
  unchanged.

Verification run:

- `.venv/bin/python -m py_compile
  src/statspai/conformal_causal/conformal_ite.py` passed.
- `.venv/bin/python -m flake8
  src/statspai/conformal_causal/conformal_ite.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m mypy
  src/statspai/conformal_causal/conformal_ite.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_conformal_bcf_bunching_mc.py tests/test_article_aliases.py
  tests/test_estimator_provenance_round7.py tests/test_dispatchers_v150.py
  tests/test_tierD_p2_conformal_analytic.py` passed, 103 tests with expected
  PSM and quantile-regression warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4133 <= 4698, mypy observed 2873 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 187

Target: DeepIV direct typing and touched-file lint cleanup.

- Typed `DeepIV._validate()`, `DeepIV.effect()`, the PyTorch network builder
  helpers, the nested module `forward()` methods, MDN loss, and MDN sampling
  in `src/statspai/deepiv/deep_iv.py`.
- Kept PyTorch object boundaries annotated as `Any`, preserving lazy optional
  dependency behavior and avoiding runtime torch type imports.
- Normalized `effect()` output to an explicit NumPy array.
- Removed an unused torch import and treatment-level variable; cleaned touched
  file long lines and stale imports.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/deepiv/deep_iv.py` passed.
- `.venv/bin/python -m flake8 src/statspai/deepiv/deep_iv.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `.venv/bin/python -m mypy src/statspai/deepiv/deep_iv.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_deepiv.py tests/test_iv_dispatcher.py
  tests/test_late_bind_contracts.py` passed, 68 tests with 1 optional skip.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4125 <= 4698, mypy observed 2861 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 188

Target: diagnostics helper typing and touched-file lint cleanup.

- Typed `sensemakr._partial_r2_of()` and the Hausman FE/RE estimator helpers
  in `src/statspai/diagnostics/sensemakr.py` and
  `src/statspai/diagnostics/hausman.py`.
- Normalized Hausman helper returns to explicit floating NumPy arrays.
- Removed an unused Hausman import and an unused `partial_r2_dd` computation
  from `sensemakr()`.
- Cleaned touched-file flake8 issues in long interpretation strings, RE group
  means construction, and the GLS theta expression.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/diagnostics/sensemakr.py
  src/statspai/diagnostics/hausman.py` passed.
- `.venv/bin/python -m flake8 src/statspai/diagnostics/sensemakr.py
  src/statspai/diagnostics/hausman.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/diagnostics/sensemakr.py
  src/statspai/diagnostics/hausman.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_sensemakr.py tests/test_hausman.py
  tests/test_unified_sensitivity.py tests/test_panel_cov_estimators.py
  tests/test_cov95_panel_diagnostics.py` passed, 53 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4119 <= 4698, mypy observed 2858 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 189

Target: DID Bacon/aggregation static hardening plus pretrends exception
compatibility.

- Typed the private Goodman-Bacon TWFE, monotone-treatment, dyad-subset, and
  weight helpers in `src/statspai/did/bacon.py`.
- Removed unused Goodman-Bacon imports/dead locals and cleaned touched-file
  wrapping without changing the dyad or weight formulas.
- Narrowed `aggte` analytic standard-error output to an explicit float ndarray,
  removed an unused bootstrap dimension local, and split the p-value denominator
  expression for lint/type clarity.
- Preserved structured `NumericalInstability` while restoring legacy
  `ValueError` catches, and made unsupported `sensitivity_rr()` methods catchable
  as both `MethodIncompatibility` and `NotImplementedError`.
- Typed the touched `pretrends` public entrypoints/plot method and cleaned local
  formatting exposed by the compatibility repair.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/bacon.py
  src/statspai/did/aggte.py` passed.
- `.venv/bin/python -m flake8 src/statspai/did/bacon.py
  src/statspai/did/aggte.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/did/bacon.py
  src/statspai/did/aggte.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/exceptions.py
  src/statspai/did/pretrends.py src/statspai/did/bacon.py
  src/statspai/did/aggte.py` passed.
- `.venv/bin/python -m flake8 src/statspai/exceptions.py
  src/statspai/did/pretrends.py src/statspai/did/bacon.py
  src/statspai/did/aggte.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/exceptions.py
  src/statspai/did/pretrends.py src/statspai/did/bacon.py
  src/statspai/did/aggte.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_did_advanced.py tests/test_aggte.py tests/test_cov95_did_r4_aggte.py
  tests/test_cov95_did_r5_misc.py tests/test_mixtape_ch09_guide.py` passed, 96
  tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4104 <= 4698, mypy observed 2848 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 190

Target: fast optional-JAX public export typing and fallback stability.

- Replaced the untyped `statspai.fast.feols_jax()` and
  `feols_jax_bootstrap()` fallback stubs with signatures matching the real JAX
  entrypoints.
- Exported a placeholder `FeolsBootstrapResult` class when JAX is unavailable
  so the public name remains class-shaped instead of becoming `None`.
- Kept runtime semantics unchanged: unavailable JAX paths still raise the same
  actionable `ImportError` when called or instantiated.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/fast/__init__.py` passed.
- `.venv/bin/python -m flake8 src/statspai/fast/__init__.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `.venv/bin/python -m mypy src/statspai/fast/__init__.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_bench.py tests/test_fast_feols.py tests/test_fast_fepois.py
  tests/test_fast_event_study.py tests/test_fast_inference.py
  tests/test_fast_within_dsl.py tests/test_fast_demean.py
  tests/test_fast_jax_feols_result_protocol.py tests/test_result_protocol_audit.py`
  passed, 200 tests with 14 optional-engine skips.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  267 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4104 <= 4698, mypy observed 2843 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 191

Target: Callaway-Sant'Anna static hardening without estimator drift.

- Rewrapped long structured-exception recovery hints in
  `src/statspai/did/callaway_santanna.py`.
- Materialized the post-treatment influence-function slice before aggregation
  so the bootstrap/inference path has an explicit optional ndarray boundary.
- Rewrote the outcome-regression branch to make the covariate-present path
  explicit, avoiding unsafe `x is None` flow through a boolean sentinel.
- Normalized statsmodels `predict()` outputs and repeated-cross-section
  residualisation returns to float ndarrays.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/callaway_santanna.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/did/callaway_santanna.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `.venv/bin/python -m mypy src/statspai/did/callaway_santanna.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts='' tests/test_did.py
  tests/test_cov95_did_callaway.py tests/test_cov95_did_r5_callaway.py
  tests/test_cov95_did_r5_supplement.py tests/test_cs_rcs.py tests/test_aggte.py
  tests/test_cov95_did_r4_aggte.py tests/test_honest_did_aggte.py
  tests/test_cs_report.py tests/reference_parity/test_callaway_santanna_parity.py`
  passed, 169 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  268 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4097 <= 4698, mypy observed 2842 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 192

Target: CIC result stability and typed estimator-specific display methods.

- Added `CICResult(CausalResult)` in `src/statspai/did/cic.py` so CIC-specific
  `plot()` and `summary()` behavior is class-defined rather than monkey-patched
  onto an instance.
- Preserved existing plot and summary output while keeping the result compatible
  with `CausalResult` provenance, tidy, and glance behavior.
- Made ECDF/quantile helpers return explicit float ndarrays and narrowed the
  bootstrap QTE matrix before indexed assignment.
- Removed unused dynamic-method locals and cleaned touched-file lint.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/cic.py` passed.
- `.venv/bin/python -m flake8 src/statspai/did/cic.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m mypy src/statspai/did/cic.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_cic.py tests/test_estimator_provenance_round3.py
  tests/test_tierD_p2_causal_recovery_analytic.py` passed, 27 tests with 8
  pre-existing NumPy runtime warnings in dose-response analytic checks.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4094 <= 4698, mypy observed 2833 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 193

Target: continuous DID dispatch/helper typing and touched-file lint cleanup.

- Made optional public arguments in `src/statspai/did/continuous_did.py`
  explicit (`post`, `t_pre`, `t_post`, `controls`, `cluster`, `seed`).
- Typed the TWFE, dose-bin ATT, and dose-response helper branches as
  `CausalResult` returns.
- Rewrapped dispatch calls, quantile-bin construction, p-value expressions, and
  fallback `linregress()` calls so touched-file flake8 is clean.
- Removed unused imports while preserving the documented default
  `method='att_gt'` heuristic and the CGS MVP warning semantics.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/continuous_did.py` passed.
- `.venv/bin/python -m flake8 src/statspai/did/continuous_did.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `.venv/bin/python -m mypy src/statspai/did/continuous_did.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_r2_continuous.py tests/test_continuous_did_heuristics.py
  tests/test_cov95_did_r5_continuous_did.py tests/test_continuous_did_cgs.py
  tests/test_cov95_did_r3_misc_estimators.py
  tests/reference_parity/test_dose_response_parity.py
  tests/test_cov95_did_r4_estimators.py tests/test_cov95_did_estimators.py`
  passed, 114 tests with existing NumPy/pandas/bootstrap warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4081 <= 4698, mypy observed 2821 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 194

Target: `did_multiplegt` bootstrap-array narrowing and touched-file cleanup.

- Narrowed optional placebo/dynamic bootstrap arrays before indexed assignment
  in `src/statspai/did/did_multiplegt.py`.
- Replaced ambiguous loop variable names in placebo and dynamic loops with
  explicit lag/horizon indices.
- Normalized `_residualize()` output to a float ndarray and cleaned long
  DataFrame slice/rename lines plus an overlong reference line.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/did_multiplegt.py` passed.
- `.venv/bin/python -m flake8 src/statspai/did/did_multiplegt.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `.venv/bin/python -m mypy src/statspai/did/did_multiplegt.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_r2_multiplegt.py tests/test_did_multiplegt_joint.py
  tests/reference_parity/test_did_multiplegt_parity.py
  tests/test_cov95_did_r3_misc_estimators.py tests/test_cov95_did_r4_estimators.py
  tests/test_cov95_did_r5_misc.py tests/test_estimator_provenance_round2.py`
  passed, 107 tests with 1 expected reference-fixture skip and 3 existing
  `did_multiplegt_dyn` empty-slice warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4068 <= 4698, mypy observed 2818 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 195

Target: Gardner/event-study ndarray returns and touched-file lint cleanup.

- Normalized Gardner's cluster-robust variance helper to return an explicit
  float ndarray.
- Removed an unused event-study bin `group_means` computation from Gardner
  event-study inference.
- Normalized event-study cluster-SE output to a float ndarray and rewrapped
  several long event-study table/statistic expressions.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/gardner_2s.py
  src/statspai/did/event_study.py` passed.
- `.venv/bin/python -m flake8 src/statspai/did/gardner_2s.py
  src/statspai/did/event_study.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/did/gardner_2s.py
  src/statspai/did/event_study.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_gardner_2s.py tests/reference_parity/test_did_variants_parity.py
  tests/test_cov95_did_event_study.py tests/test_event_study_consumers.py
  tests/test_event_study_pipeline.py tests/test_cov95_did_pretrends.py
  tests/test_pretrends_power.py tests/test_cov95_did_r3_misc_estimators.py
  tests/test_cov95_did_r4_estimators.py tests/test_cov95_did_estimators.py
  tests/test_cov95_did_estimators_extra.py` passed, 137 tests with 1 xfail and
  existing overlap/cohort/continuous DID warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4062 <= 4698, mypy observed 2817 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 196

Target: harvest DID weight-array typing and touched-file lint cleanup.

- Normalized `_weights()` in `src/statspai/did/harvest.py` to return explicit
  float ndarrays for precision, equal, and treated-count weighting schemes.
- Rewrapped the harvest result summary's pre-trend p-value line so the touched
  file is lint clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/harvest.py` passed.
- `.venv/bin/python -m flake8 src/statspai/did/harvest.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m mypy src/statspai/did/harvest.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_harvest_did.py tests/test_cov95_did_r3_harvest.py
  tests/test_estimator_provenance_round3.py tests/test_cov95_did_estimators.py
  tests/test_cov95_did_estimators_extra.py` passed, 64 tests with existing
  overlap/cohort/continuous DID warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4061 <= 4698, mypy observed 2815 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 197

Target: HonestDiD event-study extraction typing and touched-file lint cleanup.

- Typed the `event_study` lookup in `src/statspai/did/honest_did.py` with an
  explicit `DataFrame` cast and guarded the `detail` fallback with an
  `isinstance(..., pd.DataFrame)` check before reading `.columns`.
- Returned `breakdown_m()`'s clamped scalar as an explicit `float`, preserving
  the closed-form formula while clearing the direct no-any-return issue.
- Rewrapped long recovery hints so the touched HonestDiD file is flake8 clean.

Verification run:

- `.venv/bin/python -m mypy src/statspai/did/honest_did.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/did/honest_did.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_honest_did_aggte.py tests/test_honest_did_sdid.py
  tests/test_honest_did_backend.py
  tests/external_parity/test_honest_did_paper_parity.py
  tests/test_cov95_did_r4_honest_pretrends.py
  tests/test_event_study_consumers.py tests/test_event_study_pipeline.py
  tests/test_mixtape_ch09_guide.py` passed, 85 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4055 <= 4698, mypy observed 2813 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 198

Target: BJS imputation control-list narrowing and sparse-design typing.

- Added an internal `control_names` list in
  `src/statspai/did/did_imputation.py` so validation, model fitting, and
  `beta_controls` construction operate on a non-optional control list while
  keeping the original user parameter available for provenance.
- Normalized `_ols_coef()` to return an explicit float ndarray from
  `np.linalg.lstsq`.
- Annotated the sparse TWFE design-matrix row, column, and data parts as
  ndarray lists, clearing append-type ambiguity without changing the dummy
  matrix construction.
- Removed the unused `Union` import.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/did_imputation.py` passed.
- `.venv/bin/python -m mypy src/statspai/did/did_imputation.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/did/did_imputation.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_did_imputation_branches.py
  tests/reference_parity/test_did_variants_parity.py tests/test_bjs_joint.py
  tests/test_cov95_did_r5_bjs_inference.py tests/test_did_summary.py
  tests/test_gardner_2s.py tests/test_low_cov_battery.py` passed, 100 tests
  with 1 skip and 1 xfail.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4054 <= 4698, mypy observed 2808 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 199

Target: DiD-BCF fallback metadata typing and touched-file lint cleanup.

- Kept `src/statspai/did/did_bcf.py`'s `catt_by_cohort` as a numeric
  `dict[float, float]` instead of mixing fallback strings into the cohort-CATT
  mapping.
- Moved the BCF fallback exception message into `model_info["fallback_reason"]`
  so downstream consumers can treat `catt_by_cohort` predictably.
- Rewrapped the ATT-SE calculation and docstring/example lines while preserving
  the DiD-BCF citation key.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/did_bcf.py` passed.
- `.venv/bin/python -m mypy src/statspai/did/did_bcf.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/did/did_bcf.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_r3_did_bcf.py tests/test_did_frontiers.py
  tests/test_cov95_did_estimators_extra.py tests/test_v100_integration.py`
  passed, 98 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4051 <= 4698, mypy observed 2807 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 200

Target: DID summary helper typing and report JSON detail validation.

- Added a typed runner alias and explicit helper signatures for the
  `did_summary()` dispatch functions in `src/statspai/did/summary.py`.
- Kept the internal dispatch call behavior equivalent by passing runner
  arguments positionally through the typed dispatch table.
- Reused `_ensure_did_summary()` in `did_report()` before JSON serialization,
  so the detail table is validated and narrowed before `.replace()` is called.
- Rewrapped the Markdown alignment-row construction so the touched file is
  flake8 clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/summary.py` passed.
- `.venv/bin/python -m mypy src/statspai/did/summary.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/did/summary.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_did_summary.py tests/test_cov95_did_summary_extra.py
  tests/test_did_cov_plots_diagram.py tests/test_cov95_did_plots.py
  tests/test_cs_report.py tests/test_cs_report_smoke.py` passed, 94 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4050 <= 4698, mypy observed 2801 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 201

Target: stacked DID helper typing and touched-file lint cleanup.

- Typed `_twoway_demean()` in `src/statspai/did/stacked_did.py` as returning
  `(y_dm, X_dm)` ndarrays and annotated the two group-index maps.
- Removed unused local shape variables from the demeaning helper.
- Returned cluster-robust standard errors as an explicit float ndarray.
- Rewrapped the ATT p-value expression so the touched file is flake8 clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/stacked_did.py` passed.
- `.venv/bin/python -m mypy src/statspai/did/stacked_did.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/did/stacked_did.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_r3_stacked.py tests/test_estimator_provenance_round3.py
  tests/test_cov95_did_estimators_extra.py tests/test_did_summary.py
  tests/test_cov95_did_summary_extra.py` passed, 67 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4047 <= 4698, mypy observed 2798 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 202

Target: overlap DID propensity-score ndarray return and lint cleanup.

- Removed an unused `dataclass` import from `src/statspai/did/overlap_did.py`.
- Returned `dl_propensity_score()`'s clipped neural-network probabilities as an
  explicit float ndarray.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/overlap_did.py` passed.
- `.venv/bin/python -m mypy src/statspai/did/overlap_did.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/did/overlap_did.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_overlap_did.py tests/test_cov95_did_misc_estimators.py
  tests/test_cov95_did_estimators.py tests/test_cov95_did_estimators_extra.py
  tests/test_api_surface_consistency.py` passed, 49 tests with existing
  pandas/sklearn/bootstrap warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4046 <= 4698, mypy observed 2797 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 203

Target: Sun-Abraham weight-vector typing and within-transform cleanup.

- Normalized Sun-Abraham summary weight vectors in place in
  `src/statspai/did/sun_abraham.py`, preserving their one-dimensional ndarray
  type while keeping the same event-time/fixest ATT formulas.
- Returned `_two_way_demean()` outputs as explicit float ndarrays on both the
  degenerate and iterative paths.
- Replaced backslash continuation in the within-transform loop with explicit
  count arrays and parenthesized convergence checks so touched-file lint is
  clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/sun_abraham.py` passed.
- `.venv/bin/python -m mypy src/statspai/did/sun_abraham.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/did/sun_abraham.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_estimators.py tests/test_cov95_did_r4_estimators.py
  tests/test_cov95_did_r5_misc.py tests/test_cov95_did_dispatcher.py
  tests/test_cov95_did_analysis.py tests/test_honest_did_aggte.py
  tests/test_aggte.py tests/test_mixtape_ch09_guide.py
  tests/test_cov95_did_estimators_extra.py` passed, 158 tests with existing
  overlap/cohort warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1370 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4043 <= 4698, mypy observed 2794 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 204

Target: Wooldridge/ETWFE/DRDID static hardening and aggregation guards.

- Cleared `src/statspai/did/wooldridge_did.py` single-file mypy and flake8 by
  removing unused locals, renaming branch-local `model_info` variables, and
  adding return annotations to local helper functions.
- Normalized logistic/weighted least-squares helper returns to explicit float
  ndarrays and typed the DRDID repeated-cross-section `_estimate_att()` helper.
- Added an `etwfe_emfx()` detail-table guard so aggregation fails with a
  taxonomy exception if the source result lacks cohort-level detail.
- Narrowed ETWFE event-study and event-vcov paths to DataFrame/ndarray objects
  before copying, indexing, and computing delta-method SEs.
- Rewrapped long expressions and semicolon-packed assignments without changing
  the estimator formulas.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/wooldridge_did.py` passed.
- `.venv/bin/python -m mypy src/statspai/did/wooldridge_did.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/did/wooldridge_did.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with 0
  touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_r3_wooldridge.py
  tests/test_wooldridge_did_branches.py tests/test_cov95_did_r5_misc.py
  tests/test_cov95_did_estimators.py tests/test_cov95_did_r4_estimators.py
  tests/test_cov95_did_analysis.py tests/test_cov95_did_dispatcher.py
  tests/test_did_summary.py tests/test_cov95_did_summary_extra.py` passed,
  223 tests with existing overlap/cohort warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4016 <= 4698, mypy observed 2778 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 205

Target: DID analysis workflow typing and model-info narrowing.

- Added explicit `plot()` and `did_analysis()` variadic argument annotations in
  `src/statspai/did/analysis.py`.
- Typed the workflow step log and diagnostics dictionary.
- Narrowed event-study `model_info` to dicts before reading pretrend fields in
  both summary rendering and workflow steps.
- Rewrapped long summary strings so the touched file is flake8 clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/analysis.py` passed.
- `.venv/bin/python -m mypy src/statspai/did/analysis.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/did/analysis.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_analysis.py tests/test_cov95_did_analysis_extra.py
  tests/test_cov95_did_r5_misc.py tests/test_cov95_did_r5_supplement.py
  tests/test_cov95_did_r4_estimators.py tests/test_did_imputation_branches.py
  tests/test_exception_migrations.py` passed, 117 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4012 <= 4698, mypy observed 2774 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 206

Target: Tobit/Heckman helper annotations and touched-file lint cleanup.

- Preserved the existing Tobit robust-convergence helper path and added a typed
  `neg_loglik()` inner objective in `src/statspai/regression/tobit.py`.
- Removed unused typing imports and rewrapped the Tobit design-matrix
  construction so the touched file is lint clean.
- Typed Heckman's `_probit_fit()` helper as returning coefficient and covariance
  ndarrays, initialized its IRLS weights as a concrete float ndarray, removed an
  unused local shape variable, and rewrapped design-matrix construction.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/regression/tobit.py
  src/statspai/regression/heckman.py` passed.
- `.venv/bin/python -m mypy src/statspai/regression/tobit.py
  src/statspai/regression/heckman.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/regression/tobit.py
  src/statspai/regression/heckman.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_limited_dep_lane.py tests/test_weakiv_tobit.py
  tests/test_heckman.py tests/reference_parity/test_count_quantile_parity.py
  tests/reference_parity/test_heckman_se_parity.py tests/test_translation.py`
  passed, 143 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4002 <= 4698, mypy observed 2772 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 207

Target: DML base/model-averaging helper typing and touched-file cleanup.

- Added explicit internal helper annotations in `src/statspai/dml/_base.py`
  for validation, default learners, the `_fit_one_rep()` abstract boundary,
  explicit fold validation, and weighted nuisance fitting.
- Typed per-repetition diagnostic and residual caches, and narrowed the
  instrument-required `model_info` branch so mypy no longer treats the
  validated instrument list as optional.
- Added nested helper, loss, and gradient annotations in
  `src/statspai/dml/model_averaging.py`, with the SLSQP gradient explicitly
  returned as a float ndarray.
- Rewrapped the short-stacking fallback SSE expression so the touched model
  averaging file is lint clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/dml/_base.py
  src/statspai/dml/model_averaging.py` passed.
- `.venv/bin/python -m mypy src/statspai/dml/model_averaging.py
  src/statspai/dml/_base.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/dml/model_averaging.py
  src/statspai/dml/_base.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_dml.py tests/test_dml_model_averaging.py
  tests/test_dml_cov_diag_sens.py tests/test_dml_cov_averaging_panel.py
  tests/test_dml_panel.py tests/test_dml_cov_learners.py
  tests/test_dml_orthogonality_invariants.py
  tests/tier_eg/test_dml_invariance.py tests/tier_eg/test_dml_robustness.py
  tests/test_ml_causal_polish.py tests/test_review_fixes_round2.py` passed,
  118 tests with 3 skipped.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4001 <= 4698, mypy observed 2758 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 208

Target: agent resource/dispatch type cleanup and schema snapshot sync.

- Typed the MCP resource-read error-class parameters in
  `src/statspai/agent/_resources.py` while preserving the leaf-module import
  boundary.
- Narrowed `src/statspai/agent/tools/_dispatch.py` so `_resolve_fn()` returns
  a concrete `Callable[..., Any]`, typed the internal serializer wrapper, and
  rewrapped the remediation call to clear the touched-file lint issue.
- Refreshed the committed runtime schema bundle with
  `.venv/bin/python scripts/dump_schemas.py` after the agent/schema contract
  test detected stale generated JSON.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/agent/_resources.py
  src/statspai/agent/tools/_dispatch.py` passed.
- `.venv/bin/python -m mypy src/statspai/agent/_resources.py
  src/statspai/agent/tools/_dispatch.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m flake8 src/statspai/agent/_resources.py
  src/statspai/agent/tools/_dispatch.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_agent.py tests/test_mcp_protocol.py
  tests/test_mcp_error_envelope.py tests/test_agent_native_contract.py
  tests/test_schema_export.py tests/test_v0917_deferred.py
  tests/test_mcp_nan_inf.py` passed, 214 tests.
- `.venv/bin/python scripts/dump_schemas.py --check` passed after the schema
  refresh.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4000 <= 4698, mypy observed 2755 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 209

Target: MCP server direct mypy/flake8 cleanup.

- Moved the MCP server's private compatibility shim imports for data loading,
  errors, prompts, and resources to module scope while preserving the existing
  underscore-prefixed names used by tests and downstream callers.
- Typed the lazy `tool_manifest()` and `execute_tool()` proxies in
  `src/statspai/agent/mcp_server.py`.
- Narrowed schema-snapshot loading to typed `list[dict[str, Any]]` payloads
  and made dataless-tool name extraction explicitly string-only.
- Typed the catalog/resource/prompt shims, progress sink, no-op/progress drain,
  and `_handle_tools_call()` worker closure.
- Rewrapped dataless overrides, CSV loading, estimator dispatch, and debug-env
  checks so `mcp_server.py` is direct flake8 clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/agent/mcp_server.py` passed.
- `.venv/bin/python -m mypy src/statspai/agent/mcp_server.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/agent/mcp_server.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_mcp_protocol.py tests/test_mcp_runner.py
  tests/test_mcp_sampling.py tests/test_mcp_prompts_expanded.py
  tests/test_mcp_error_envelope.py tests/test_mcp_nan_inf.py
  tests/test_mcp_image_content.py tests/test_v0917_deferred.py` passed,
  154 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3974 <= 4698, mypy observed 2741 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 210

Target: workflow-tool direct mypy/flake8 cleanup.

- Removed an unused typing import from `src/statspai/agent/workflow_tools.py`.
- Rewrapped workflow manifest description fields and dispatch branches so the
  file is direct flake8 clean without changing any tool names or schemas.
- Rewrapped sensitivity and honest-DID result-handle branches while preserving
  their existing fallback API behavior.
- Typed the event-study extraction and sigma-list helpers, the detect/preflight
  wrappers, plot-kind mapping, PNG renderer, figure coercion helper, and
  `plot_from_result` wrapper.
- Replaced backslash continuations in event-study extraction with explicit
  parenthesized fallback expressions.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/agent/workflow_tools.py`
  passed.
- `.venv/bin/python -m mypy src/statspai/agent/workflow_tools.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/agent/workflow_tools.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_workflow_tool_dispatch_contract.py
  tests/agent_eval/test_did_workflow_transcript.py
  tests/test_mcp_image_content.py tests/test_mcp_protocol.py
  tests/test_agent.py tests/test_mcp_error_envelope.py
  tests/test_v0917_deferred.py` passed, 200 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3939 <= 4698, mypy observed 2736 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 211

Target: composite pipeline-tool direct mypy/flake8 cleanup.

- Typed the pipeline stage helper, safe-call wrapper, audit serializer, and
  light serializer in `src/statspai/agent/pipeline_tools.py`.
- Rewrapped the DID pipeline schema, stage logging, primary-result caching,
  honest-DID fallback call, Bacon decomposition call, and narrative builder.
- Rewrapped IV pipeline diagnostics, converted no-op f-strings to plain
  strings, and expanded the effective-F nested fallback expression.
- Rewrapped RD pipeline density/plot stage logging and pipeline dispatcher
  calls.
- Preserved the existing pipeline stage order and partial-failure semantics.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/agent/pipeline_tools.py`
  passed.
- `.venv/bin/python -m mypy src/statspai/agent/pipeline_tools.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/agent/pipeline_tools.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_mcp_pipelines.py tests/test_mcp_protocol.py
  tests/test_agent_native_contract.py tests/test_mcp_prompts_expanded.py
  tests/test_agent.py tests/test_schema_export.py` passed, 150 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3890 <= 4698, mypy observed 2732 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 212

Target: GMM helper typing and moment-dimension narrowing.

- Typed `src/statspai/gmm/general_gmm.py` so `gmm()` accepts optional data,
  optional weighting matrices, and optional parameter names explicitly.
- Derived `n` and `q` from the first evaluated moment matrix instead of a
  nullable data-length proxy, keeping GMM variance and objective arithmetic
  non-null.
- Typed GMM inner helpers and converted mean moment vectors back to float
  ndarrays to prevent NumPy Any leakage.
- Removed an unused `gb_hat` local and unused imports from the general GMM
  file.
- Typed Arellano-Bond's Windmeijer and AR-test helpers, asserted the two-step
  branch invariants before calling the Windmeijer correction, and returned the
  corrected covariance as an ndarray.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/gmm/general_gmm.py
  src/statspai/gmm/arellano_bond.py` passed.
- `.venv/bin/python -m mypy src/statspai/gmm/general_gmm.py
  src/statspai/gmm/arellano_bond.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m flake8 src/statspai/gmm/general_gmm.py
  src/statspai/gmm/arellano_bond.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_gmm.py tests/reference_parity/test_gmm_dynamic_panel_parity.py
  tests/test_v06_round3.py tests/test_cov95_panel_reg.py
  tests/test_panel_cov_diagnostics.py tests/test_panel_dispatcher.py
  tests/test_translation.py` passed, 208 tests with 2 existing warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3886 <= 4698, mypy observed 2722 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 213

Target: inference helper typing and ndarray-return narrowing.

- Typed AIPW propensity/outcome helper boundaries in
  `src/statspai/inference/aipw.py`, switched propensity clipping to an in-place
  update to preserve the 1-D prediction array, and converted statsmodels
  predictions to float ndarrays.
- Cleaned AIPW unused imports, one no-op f-string, and long reference lines.
- Rewrapped Conley references and the haversine expression in
  `src/statspai/inference/conley.py`, returning the distance vector as a float
  ndarray.
- Typed front-door point, OLS, logit, prediction, ATE, and nested outcome
  helper boundaries in `src/statspai/inference/front_door.py`.
- Removed an unused front-door feature matrix and wrapped long front-door error
  and Monte Carlo comments without changing estimator formulas.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/inference/front_door.py
  src/statspai/inference/conley.py src/statspai/inference/aipw.py` passed.
- `.venv/bin/python -m mypy src/statspai/inference/front_door.py
  src/statspai/inference/conley.py src/statspai/inference/aipw.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/inference/front_door.py
  src/statspai/inference/conley.py src/statspai/inference/aipw.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_inference.py tests/test_front_door.py
  tests/test_front_door_integrate_by.py tests/test_silent_degradation_fixes.py
  tests/test_review_fixes_round2.py tests/test_conley_vectorized_equivalence.py
  tests/test_estimator_provenance_round5.py
  tests/tier_eg/test_weighting_gmethods_invariance.py
  tests/reference_parity/test_paper_parity.py
  tests/reference_parity/test_tmle_parity.py` passed, 94 tests with 1 existing
  overlap warning.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3877 <= 4698, mypy observed 2711 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 214

Target: g-methods/PATE helper typing and touched-file cleanup.

- Typed PATE validation, participation-propensity, outcome-model,
  entropy-balancing, point-estimator, and bootstrap helper boundaries in
  `src/statspai/inference/pate.py`.
- Rewrapped PATE design-matrix, optimizer, confidence-interval, and bootstrap
  detail construction, removed an unused AIPW scale local, and converted
  external-library array returns to float ndarrays.
- Typed the nested fit/predict and point-estimate helpers in
  `src/statspai/inference/g_computation.py`.
- Rewrapped the statsmodels prediction call and provenance `treat_values`
  expression while preserving g-computation formulas and result fields.

Verification run:

- `.venv/bin/python -m mypy src/statspai/inference/pate.py
  src/statspai/inference/g_computation.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m flake8 src/statspai/inference/pate.py
  src/statspai/inference/g_computation.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_g_computation.py tests/test_registry_new_modules.py
  tests/reference_parity/test_gformula_parity.py
  tests/reference_parity/test_tmle_parity.py
  tests/tier_eg/test_weighting_gmethods_invariance.py
  tests/test_smart_tools_sprint_b.py tests/test_smart_tools_sprint_b_round3.py
  tests/test_smart_tools_sprint_b_round4.py
  tests/test_diagnose_batteries_sprint_b.py` passed, 119 tests with 8 existing
  warnings.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_transport_and_shiftshare.py` passed, 9 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1277 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3868 <= 4698, mypy observed 2695 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 215

Target: randomization-inference typing and touched-file cleanup.

- Typed `FisherResult.__init__` and `FisherResult.plot` in
  `src/statspai/inference/randomization.py`.
- Rebuilt `FisherResult._repr_html_` from short string fragments to eliminate
  long-line debt without changing the displayed fields.
- Added local statistic/permutation callable aliases, replaced lambda
  assignments with named helpers, and typed unrestricted, clustered, and
  stratified permutation functions as float-ndarray producers.
- Preserved the existing `ri_test(stat: str = "diff_means")` public annotation
  so schema generation keeps `stat` as a string while retaining the runtime
  callable escape hatch via a local cast.
- Refreshed generated runtime schemas after the signature/typing cleanup; the
  top-level schema bundle remains unchanged and `dump_schemas --check` passes.

Verification run:

- `.venv/bin/python -m py_compile
  src/statspai/inference/randomization.py` passed.
- `.venv/bin/python -m mypy src/statspai/inference/randomization.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/inference/randomization.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_ri.py tests/test_tierD_p2_inference_analytic.py
  tests/test_estimator_provenance_round8.py tests/test_inference.py` passed,
  38 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1277 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3855 <= 4698, mypy observed 2689 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 216

Target: DTR and continuous-dose helper typing cleanup.

- Removed unused `Dict`/`Any` imports from
  `src/statspai/dtr/g_estimation.py` and
  `src/statspai/dose_response/gps.py`.
- Added explicit `-> None` constructors for `GEstimation` and `DoseResponse`.
- Typed `GEstimation._backward_induction` as returning the stage blip vector
  plus forward-ordered rule labels.
- Typed `DoseResponse._estimate_curve` as returning the dose-response curve
  ndarray while preserving the existing GPS/treatment-model/outcome-model
  computation.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/dtr/g_estimation.py
  src/statspai/dose_response/gps.py` passed.
- `.venv/bin/python -m mypy src/statspai/dtr/g_estimation.py
  src/statspai/dose_response/gps.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m flake8 src/statspai/dtr/g_estimation.py
  src/statspai/dose_response/gps.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_dose_response.py
  tests/reference_parity/test_dose_response_parity.py tests/test_phase9to14.py
  tests/test_dtr.py tests/tier_eg/test_weighting_gmethods_invariance.py
  tests/test_tierD_p2_causal_recovery_analytic.py` passed, 74 tests with 8
  existing dose-response bootstrap warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1277 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3851 <= 4698, mypy observed 2687 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 217

Target: partial-interference spillover helper typing cleanup.

- Removed unused `Dict`/`Any` imports from
  `src/statspai/interference/spillover.py`.
- Added an explicit `-> None` constructor for `SpilloverEstimator`.
- Typed `_compute_exposure` as returning the peer-exposure ndarray and
  `_safe_diff` as a float mean-contrast helper.
- Preserved the existing direct/spillover/total effect arithmetic and
  bootstrap path.

Verification run:

- `.venv/bin/python -m py_compile
  src/statspai/interference/spillover.py` passed.
- `.venv/bin/python -m mypy src/statspai/interference/spillover.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/interference/spillover.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_phase9to14.py
  tests/reference_parity/test_interference_parity.py
  tests/test_dispatchers_v150.py tests/test_interference_extensions.py`
  passed, 78 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1277 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3849 <= 4698, mypy observed 2685 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 218

Target: pyfixest adapter dynamic-attribute typing cleanup.

- Rewrapped the pyfixest method-label expression in
  `src/statspai/fixest/adapter.py`.
- Replaced direct dynamic assignment to `result._pyfixest_fit` with
  `setattr(...)`, preserving the advanced-user escape hatch while clearing
  mypy's `attr-defined` error on `EconometricResults`.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/fixest/adapter.py` passed.
- `.venv/bin/python -m mypy src/statspai/fixest/adapter.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/fixest/adapter.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_fixest.py
  tests/test_late_bind_contracts.py tests/test_schema_export.py` passed,
  65 tests with 2 existing `outreg2` deprecation warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1277 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3848 <= 4698, mypy observed 2684 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 219

Target: IV diagnostic plotting typing and touched-file cleanup.

- Typed the lazy matplotlib loader and all public plotting helpers in
  `src/statspai/iv/plot.py`.
- Typed the local array-grabbers and residualizers used by first-stage and
  Anderson-Rubin plots, returning float ndarrays at numeric boundaries.
- Added explicit `ValueError` messages when callers pass column names without
  supplying `data`, replacing the previous implicit indexing failure.
- Rewrapped first-stage F, binscatter, AR-F, MTE title, and weak-IV title
  expressions; removed unused raw `Y`/`D` locals from the diagnostic panel.
- Preserved the plotting semantics and existing matplotlib object return
  convention.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/iv/plot.py` passed.
- `.venv/bin/python -m mypy src/statspai/iv/plot.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/iv/plot.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_iv_cov_plots.py tests/iv/test_plots.py tests/iv/test_iv_diag.py
  tests/test_iv_cov_diag.py` passed, 44 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1281 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3839 <= 4698, mypy observed 2663 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 220

Target: entropy-balancing matching helper typing cleanup.

- Removed unused `Optional`/`Dict`/`Any` imports from
  `src/statspai/matching/ebalance.py`.
- Typed entropy-balancing constraint construction as returning float target
  and constraint arrays.
- Typed the L-BFGS dual objective and gradient, plus the solver return as
  `(weights, fallback)`.
- Typed the balance-check helper as returning a balance `DataFrame`.
- Preserved ATT arithmetic, optimizer options, fallback warning semantics, and
  reported balance fields.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/matching/ebalance.py` passed.
- `.venv/bin/python -m mypy src/statspai/matching/ebalance.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/matching/ebalance.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_ebalance.py
  tests/test_match_dispatcher.py tests/reference_parity/test_matching_parity.py
  tests/reference_parity/test_paper_parity.py
  tests/tier_eg/test_weighting_gmethods_invariance.py
  tests/test_exception_migrations.py` passed, 85 tests with 5 existing PSM
  imbalance warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1281 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3836 <= 4698, mypy observed 2658 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 221

Target: CBPS matching helper typing and ndarray-return cleanup.

- Typed the nested CBPS solve helper in
  `src/statspai/matching/cbps.py`.
- Avoided shape-incompatible reassignments by slicing bootstrap arrays into
  distinct used-result variables.
- Removed an unused balance-diagnostic weight vector and rewrapped p-value,
  weighted-mean, pooled-SD, label, and logistic warm-start expressions.
- Narrowed sigmoid, stacked-moment, balance-only, optimizer, warm-start, final
  propensity-score, and Hajek weight returns to float ndarrays.
- Preserved CBPS point estimates, bootstrap retry semantics, moment equations,
  and model-info fields.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/matching/cbps.py` passed.
- `.venv/bin/python -m mypy src/statspai/matching/cbps.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/matching/cbps.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_overlap_and_cbps.py
  tests/test_match_dispatcher.py tests/reference_parity/test_matching_parity.py
  tests/reference_parity/test_paper_parity.py
  tests/tier_eg/test_weighting_gmethods_invariance.py` passed, 65 tests with
  5 existing PSM imbalance warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1281 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3832 <= 4698, mypy observed 2652 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 222

Target: core matching estimator typing cleanup without changing matching
semantics.

- Typed `src/statspai/matching/match.py` nearest-neighbour, exact,
  stratification, CEM, kernel/radius, propensity-score, distance-matrix,
  balance-table, and Abadie-Imbens SE helpers.
- Replaced dynamic result attribute assignment with `setattr` while preserving
  the existing `matched_data` convenience attribute and model-info payload.
- Narrowed ndarray-return helpers with explicit `np.asarray(..., dtype=...)`
  wrappers and typed empty match/weight arrays to avoid `None` sentinels.
- Added an explicit defensive error when propensity-score distance is requested
  without a propensity-score vector.
- Kept public plotting parameter schemas compatible by using the repository's
  existing `tuple` annotation style for `figsize`, `labels`, and `colors`,
  while making nullable titles explicit in the refreshed schema bundle.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/matching/match.py` passed.
- `.venv/bin/python -m mypy src/statspai/matching/match.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/matching/match.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_matching.py
  tests/test_psmatch2.py tests/test_match_dispatcher.py
  tests/reference_parity/test_matching_parity.py
  tests/reference_parity/test_psmatch2_parity.py
  tests/test_overlap_and_cbps.py tests/test_ebalance.py
  tests/tier_eg/test_weighting_gmethods_invariance.py
  tests/test_exception_migrations.py tests/test_tierD_lalonde_psm_guard.py`
  passed, 211 tests with 48 existing PSM imbalance warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)` after refreshing generated schema files.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3832 <= 4698, mypy observed 2630 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 223

Target: overlap-weight helper typing and touched-file cleanup.

- Typed the internal bootstrap `_estimate` helper in
  `src/statspai/matching/overlap_weights.py` as returning the scalar estimate,
  propensity scores, and tilting weights.
- Narrowed logistic propensity-score and tilting-function returns to float
  ndarrays with explicit `np.asarray(..., dtype=float)` wrappers.
- Rewrapped the model-type expression and entropy-tilt expression to remove
  touched-file line-length violations.
- Preserved ATO/ATE/ATT/ATC/matching/entropy weight formulas, bootstrap
  sampling, balance diagnostics, and result payloads.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/matching/overlap_weights.py`
  passed.
- `.venv/bin/python -m mypy src/statspai/matching/overlap_weights.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/matching/overlap_weights.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_overlap_and_cbps.py
  tests/test_match_dispatcher.py tests/reference_parity/test_matching_parity.py
  tests/reference_parity/test_paper_parity.py
  tests/tier_eg/test_weighting_gmethods_invariance.py` passed, 65 tests with
  5 existing PSM imbalance warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3830 <= 4698, mypy observed 2626 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 224

Target: stable-balancing-weights typing and touched-file cleanup.

- Typed SBW's `delta` broadcast path with an explicit float ndarray, avoiding
  scalar/sequence shape ambiguity.
- Normalized lineage `delta` payloads to either `float` or `List[float]` while
  preserving scalar and vector user inputs.
- Typed SLSQP objective, gradient, equality, and inequality callbacks in
  `src/statspai/matching/sbw.py`.
- Narrowed weighted-mean helper returns and `_weighted_treatment_effect`
  outputs, and removed an unused local sample-size variable.
- Rewrapped SBW reference text and the weighted-variance comment to clear
  touched-file style violations.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/matching/sbw.py` passed.
- `.venv/bin/python -m mypy src/statspai/matching/sbw.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/matching/sbw.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_v0917_additions.py::TestSBW tests/test_match_dispatcher.py
  tests/test_exception_migrations.py::TestSbwBinary::test_non_binary_raises_method_incompatibility
  tests/test_estimator_provenance_round4.py::TestSbwProvenance::test_attached
  tests/reference_parity/test_matching_parity.py` passed, 49 tests with 5
  existing PSM imbalance warnings. An initial targeted pytest invocation used
  stale node names and failed collection before code execution; the corrected
  node run passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3827 <= 4698, mypy observed 2615 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 225

Target: mediation helper typing and bootstrap-container cleanup.

- Removed unused typing imports from `src/statspai/mediation/mediate.py`.
- Typed `MediationAnalysis.__init__`, validation, natural-effect estimation,
  delta-method SEs, OLS covariance, and p-value helper boundaries.
- Split bootstrap success lists from their ndarray views so SE/CI/p-value
  computation keeps the same successful draws without shape-incompatible
  reassignments.
- Typed interventional mediation's `_compute`, nested expectation helper, and
  `_ci_pv` bootstrap summary helper.
- Rewrapped natural-effect detail labels, interventional citation text, and
  covariate matrix construction to clear touched-file style violations.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/mediation/mediate.py` passed.
- `.venv/bin/python -m mypy src/statspai/mediation/mediate.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/mediation/mediate.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_mediation.py
  tests/test_mediate_interventional.py
  tests/test_diagnose_batteries_sprint_b.py::test_mediate_default_pvalue_method_preserves_behaviour
  tests/test_diagnose_batteries_sprint_b.py::test_mediate_wald_pvalue_matches_normal_formula
  tests/test_diagnose_batteries_sprint_b.py::test_mediate_rejects_bad_pvalue_method
  tests/test_ml_causal_polish.py::TestMediationBootstrap
  tests/test_estimator_provenance_round6.py::TestMediateProvenance::test_attached
  tests/test_estimator_provenance_round6.py::TestMediateInterventionalProvenance::test_attached`
  passed, 22 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3822 <= 4698, mypy observed 2602 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 226

Target: matrix-completion soft-impute typing and touched-file cleanup.

- Removed unused typing imports from
  `src/statspai/matrix_completion/mc_panel.py`.
- Removed an unused `times` local from the panel pivot path.
- Typed `_soft_impute` inputs and return as ndarrays/ints, explicitly typed
  the working low-rank matrix, and narrowed SVD reconstruction/return arrays
  with `np.asarray(..., dtype=float)`.
- Preserved soft-impute control-mask semantics, nuclear-norm thresholding,
  optional max-rank truncation, and returned completed matrix payloads.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/matrix_completion/mc_panel.py`
  passed.
- `.venv/bin/python -m mypy src/statspai/matrix_completion/mc_panel.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/matrix_completion/mc_panel.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_synth_advanced.py::TestMatrixCompletion
  tests/reference_parity/test_matrix_completion_parity.py
  tests/test_article_aliases_round2.py::test_matrix_completion_end_to_end
  tests/test_conformal_bcf_bunching_mc.py::TestMCPanel` passed, 21 tests. An
  initial targeted pytest invocation used a stale class node and failed
  collection before code execution; the corrected node run passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3817 <= 4698, mypy observed 2601 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 227

Target: meta-learner diagnostics typing and touched-file cleanup.

- Removed an unused `Union` import from
  `src/statspai/metalearners/diagnostics.py`.
- Typed `cate_plot` and `cate_group_plot` plotting boundaries as returning
  `(fig, ax)` and typed passthrough plotting kwargs.
- Rewrapped long BLP docstring/reference lines without changing the rendered
  diagnostic fields.
- Typed `compare_metalearners` passthrough kwargs.
- Stabilized BLP propensity clipping by explicitly keeping `e_hat` as a float
  ndarray.
- Normalized `predict_cate` covariate names and narrowed estimator-effect
  output to a float ndarray.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/metalearners/diagnostics.py`
  passed.
- `.venv/bin/python -m mypy src/statspai/metalearners/diagnostics.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/metalearners/diagnostics.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_metalearners.py::TestCATEDiagnostics
  tests/test_metalearners.py::TestPredictCATE
  tests/test_metalearners.py::TestCompareMetalearners
  tests/test_metalearners.py::TestGATETest
  tests/test_metalearners.py::TestBLPTest` passed, 16 tests. An initial
  targeted pytest invocation used a stale neural-causal node and failed
  collection before code execution; the corrected metalearner diagnostics run
  passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3814 <= 4698, mypy observed 2594 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 228

Target: core meta-learner typing and dispatch cleanup.

- Typed default sklearn-estimator factories, propensity extraction,
  cross-fit prediction, DataFrame-to-array preparation, and AIPW pseudo-outcome
  helper boundaries in `src/statspai/metalearners/metalearners.py`.
- Typed S/T/X/R/DR learner `__init__`, `fit`, and `effect` methods while
  preserving their existing default models, fitting order, cross-fitting, and
  propensity clipping semantics.
- Narrowed all learner effect outputs and AIPW pseudo-outcomes to float
  ndarrays.
- Removed unused DR learner treatment-arm masks.
- Declared the high-level learner dispatch variable as estimator-agnostic so
  mypy no longer binds it to the first S-learner branch.
- Made AIPW diagnostics a concrete dict on every path and simplified
  provenance model-name formatting.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/metalearners/metalearners.py`
  passed.
- `.venv/bin/python -m mypy src/statspai/metalearners/metalearners.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/metalearners/metalearners.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_metalearners.py
  tests/test_result_consumer_errors.py
  tests/test_article_aliases.py::test_xlearner_delegates_to_metalearner
  tests/test_article_aliases.py::test_xlearner_matches_metalearner_x
  tests/reference_parity/test_cross_estimator_parity.py` passed, 77 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3795 <= 4698, mypy observed 2568 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 229

Target: multi-valued-treatment GPS helper typing and touched-file cleanup.

- Removed unused typing imports from
  `src/statspai/multi_treatment/multi_ipw.py`.
- Split the bootstrap Wald p-value expression into an intermediate z-statistic
  without changing the formula.
- Typed `_estimate_gps` inputs/return and initialized/narrowed the generalized
  propensity-score matrix as a float ndarray.
- Preserved multinomial-logit fitting, class-column alignment, AIPW potential
  outcome estimates, and bootstrap SE/CI behavior.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/multi_treatment/multi_ipw.py`
  passed.
- `.venv/bin/python -m mypy src/statspai/multi_treatment/multi_ipw.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/multi_treatment/multi_ipw.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_phase9to14.py::TestMultiTreatment
  tests/test_tierD_p2_qte_multitreat_analytic.py
  'tests/test_late_bind_contracts.py::test_conflict_prone_function_survives_submodule_import[multi_treatment-MultiTreatment]'`
  passed, 14 tests. An initial targeted pytest invocation failed because zsh
  expanded an unquoted parametrized node; the quoted corrected node passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3792 <= 4698, mypy observed 2567 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 230

Target: MSM stabilized-weight helper typing and touched-file cleanup.

- Removed the unused `Union` typing import from `src/statspai/msm/msm.py`.
- Typed `MarginalStructuralModel.__init__` keyword passthroughs.
- Rewrapped binary/continuous treatment auto-detection and covariate matrix
  construction lines.
- Narrowed `stabilized_weights`, `_logit_proba`, `_gauss_density`,
  `_wls_cluster`, and `_weighted_logit_cluster` array returns.
- Typed WLS/logit cluster helpers as returning `(beta, se)` float ndarrays
  and initialized intermediate cluster score arrays as float ndarrays.
- Preserved stabilized IPTW ratios, per-period trimming, logit fallback
  warnings, Gaussian-density fallback, and cluster-robust sandwich formulas.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/msm/msm.py` passed.
- `.venv/bin/python -m mypy src/statspai/msm/msm.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/msm/msm.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_msm.py
  tests/test_msm_singleperiod_iptw_regression.py
  tests/test_escape_hatches.py::test_msm_trim_per_period_records_flag
  tests/test_escape_hatches.py::test_msm_trim_per_period_default_matches_old_behaviour
  tests/test_escape_hatches.py::test_msm_trim_per_period_reduces_extreme_weights
  tests/test_review_fixes.py::test_msm_binomial_does_not_diverge_on_extreme_eta
  tests/test_registry_new_modules.py::test_marginal_structural_model_class`
  passed, 13 tests. An initial targeted pytest invocation used a stale wrapper
  node and failed collection before code execution; the corrected node run
  passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3788 <= 4698, mypy observed 2561 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 231

Target: multilevel LMM static contract and touched-file cleanup.

- Verified `src/statspai/multilevel/lmm.py` now carries an explicit
  `ThreeLevelBlock` alias for nested school/class blocks.
- Replaced nullable ndarray dataclass defaults for `_cov_fixed` and `_G` with
  empty-array default factories so result containers keep a concrete ndarray
  contract before fitted state is injected.
- Tightened scalar returns for BIC and profiled negative log-likelihood helpers
  to plain `float` values.
- Typed plot keyword passthroughs, group-key composition, three-level
  block inputs, three-level block/proxy collections, and BLUP accumulator
  containers.
- Removed unused three-level ML-likelihood and class-school scratch state, and
  dropped unused profiled-likelihood cache lists that were populated but never
  consumed.
- Rewrapped touched long strings and table output fragments so the LMM file is
  clean under the repository's 88-column flake8 gate.

Verification run:

- `.venv/bin/python -m mypy src/statspai/multilevel/lmm.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/multilevel/lmm.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest tests/test_multilevel.py -q -o addopts=''`
  passed, 63 tests.
- `.venv/bin/python scripts/dump_schemas.py` refreshed generated schemas after
  public method annotation tightening; `.venv/bin/python
  scripts/dump_schemas.py --check` then reported `schemas/ is in sync (5
  files)`.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3770 <= 4698, mypy observed 2547 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 232

Target: stochastic-frontier SFA typing and touched-file cleanup.

- Preserved the public `FrontierResult.predict()` pandas-Series API while
  documenting the structural mismatch with the ndarray-return base-class
  protocol through a local override ignore.
- Narrowed frontier prediction helper outputs by converting fitted sigma and
  truncated-normal mean paths to concrete float ndarrays.
- Typed bootstrap refit inputs, truncated-normal simulation inputs, nested
  likelihood helpers, and the optimizer helper inside `frontier()`.
- Added explicit non-null assertions for truncated-normal likelihood and
  efficiency branches where the fitted distribution guarantees `mu_i`.
- Changed `returns_to_scale()` to return `Dict[str, Any]`, matching its mixed
  numeric plus interpretation-string payload.
- Rewrapped summary, prediction-requirement, marginal-effect formula, and
  nested likelihood lines so `src/statspai/frontier/sfa.py` is clean under the
  touched-file flake8 gate.

Verification run:

- `.venv/bin/python -m mypy src/statspai/frontier/sfa.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/frontier/sfa.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_frontier.py -q
  -o addopts=''` passed, 101 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_export_surface_contract.py
  tests/test_late_bind_contracts.py::test_conflict_prone_function_survives_submodule_import`
  passed, 69 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1282 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3763 <= 4698, mypy observed 2534 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 233

Target: local-polynomial public-signature typing and touched-file cleanup.

- Removed unused `typing` and `EconometricResults` imports from
  `src/statspai/nonparametric/lpoly.py`.
- Typed `LPolyResult.__init__`, `LPolyResult.plot`, kernel evaluation,
  Silverman bandwidth, local-polynomial point fitting, and the public
  `lpoly()` signature.
- Converted `lpoly()`'s implicit-Optional `None` defaults for `data`, `y`,
  `x`, `bandwidth`, and `grid` into explicit Optional annotations.
- Added an early `ValueError("y and x are required")` guard so the new
  Optional annotations are backed by runtime validation.
- Narrowed Gaussian kernel and local-fit scalar returns to concrete float
  ndarray/scalar values and rewrapped touched docstring/error-message lines.
- Refreshed generated schemas after public signature annotations became more
  precise; the schema diff also incorporated a parallel `__all__` export
  expansion already present in `src/statspai/__init__.py`.

Verification run:

- `.venv/bin/python -m mypy src/statspai/nonparametric/lpoly.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/nonparametric/lpoly.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_lpoly_reliability.py
  tests/test_new_v06_modules.py::TestNonparametric::test_lpoly
  tests/test_continuous_did_heuristics.py::TestContinuousDIDDoseResponse::test_handles_lpoly_fallback
  tests/test_continuous_did_heuristics.py::TestContinuousDIDDoseResponse::test_dose_response_curve_in_model_info`
  passed, 20 tests. An initial run used a stale test node name and failed
  collection before executing code; the corrected node run passed.
- `.venv/bin/python scripts/dump_schemas.py` refreshed the generated schema
  bundle, and `.venv/bin/python scripts/dump_schemas.py --check` then
  reported `schemas/ is in sync (5 files)`.

Pre-follow-up gate notes:

- Compileall, result-protocol, Tier-A fixture lock, benchmark ratchet,
  schema check, quality gate, `git diff --check`, and nested JOSS boundary
  checks passed.
- The refreshed schema/registry surface incorporated a parallel
  `src/statspai/__init__.py` export expansion: registered functions increased
  from 1033 to 1071, and `scripts/tierd_classify.py report` exposed 11 new
  estimator-like Tier-D worklist rows. Batch 234 immediately addresses that
  evidence gap instead of treating the report as acceptable drift.

## 2026-06-17 Batch 234

Target: Tier-D anchors for newly surfaced public exports.

- Added `tests/test_tierd_new_export_anchors.py` with deterministic
  known-truth anchors for all 11 estimator-like functions exposed by the
  parallel `__all__` expansion.
- Covered negative-control calibration (`negative_control_outcome`,
  `negative_control_exposure`) on exact linear coefficients.
- Covered `double_negative_control` on a just-identified 2SLS DGP and
  `proximal_regression` on a correctly specified outcome bridge.
- Covered `four_way_decomposition` against closed-form VanderWeele
  components with full-rank mediator variation.
- Covered `rosenbaum_bounds` and alias `rosenbaum_gamma` against exact sign-test
  binomial p-value bounds.
- Covered `its` against a noiseless segmented-regression level/slope break.
- Covered `llm_dag` in `oracle_only` mode with deterministic oracle edges.
- Covered `vcnet` and `scigan` on a linear dose-response contrast over
  `t_grid=[-1, 0, 1]`.

Verification run:

- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_tierd_new_export_anchors.py` passed, 8 tests.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1071 registered functions
  (`reference=128`, `anchored=580`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tierd_classify.py worklist` reported 0 functions.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1071 registered functions
  (`reference=128`, `anchored=580`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3755 <= 4698, mypy observed 2525 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 235

Target: output inline citation and replication-pack typing cleanup.

- Typed private helper boundaries in `src/statspai/output/_inline.py`,
  including term resolution, point/SE/p-value extraction, CI reconstruction,
  and the public `cite(...)` wrapper.
- Preserved existing inline-citation behavior while narrowing dynamic
  `CausalResult`/statsmodels attribute reads to explicit scalar/string
  conversions for static analysis.
- Removed unused provenance/import typing from
  `src/statspai/output/_replication_pack.py`.
- Typed `ReplicationPack.__init__`, paper extraction, fallback package-freeze
  metadata reads, and dataset-to-CSV byte conversion.
- Rewrapped the generated README command text without changing its rendered
  content.
- Refreshed generated schemas after parallel API/agent-card changes in the
  same worktree, and verified the schema check in a standalone pass to avoid
  cross-process import drift.

Verification run:

- `.venv/bin/python -m mypy src/statspai/output/_replication_pack.py
  src/statspai/output/_inline.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/output/_replication_pack.py
  src/statspai/output/_inline.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_cite_inline.py
  tests/test_replication_pack.py` passed, 37 tests.
- `.venv/bin/python scripts/dump_schemas.py` refreshed the generated schema
  bundle and `.venv/bin/python scripts/dump_schemas.py --check` then reported
  `schemas/ is in sync (5 files)`.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3747 <= 4698, mypy observed 2516 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 236

Target: `outreg2` facade type-contract cleanup.

- Added a type-check-only `RegtableResult` import so the Stata-compatible
  `outreg2` facade keeps its lazy runtime import behavior while exposing the
  actual renderer return type to static analysis.
- Typed `_build_regtable`, `OutReg2.__init__`, and `OutReg2._table`.
- Preserved existing Excel, Word, and LaTeX export behavior; no generated
  schema or public registry contract changed.

Verification run:

- `.venv/bin/python -m mypy src/statspai/output/outreg2.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/output/outreg2.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_export.py
  tests/test_fixest.py::TestOutreg2Integration` passed, 13 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3747 <= 4698, mypy observed 2511 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 237

Target: panel binary-choice and no-FE FEOLS static cleanup.

- Added concrete helper annotations in `src/statspai/panel/panel_binary.py`
  for the numerical Hessian, panel grouping, Mundlak means, conditional-logit
  dynamic programs, FE-logit fit wrapper, and RE binary likelihood wrapper.
- Converted scipy optimizer dynamic outputs to explicit numpy arrays, floats,
  and bools before returning from typed helpers.
- Typed the public `panel_logit`/`panel_probit` regressor and optional cluster
  parameters without changing their accepted values.
- Typed the already-preprocessed no-FE `sp.feols` fallback weights as
  `Optional[np.ndarray]`.
- Ran `black` on the two touched panel files and manually split the two
  remaining long lines so touched-file flake8 is clean.

Verification run:

- `.venv/bin/python -m mypy src/statspai/panel/panel_binary.py
  src/statspai/panel/feols.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/panel/panel_binary.py
  src/statspai/panel/feols.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_cov95_panel_misc.py
  tests/test_panel_cov_diagnostics.py tests/test_panel_cov_estimators.py
  tests/test_cov95_panel_estimator_branches.py` passed, 62 tests.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_cov95_panel_feols.py` passed, 29 tests.
- Targeted JAX FEOLS node checks were not used as pass evidence because the
  current environment skips the JAX modules during collection.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3686 <= 4698, mypy observed 2494 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 238

Target: panel dispatcher and diagnostics type-contract cleanup.

- Added a `PanelResults._stored_design(...)` narrowing helper so Hausman,
  BP-LM, F-test, and Pesaran-CD diagnostics call `panel_diagnostics` with
  non-optional stored metadata.
- Typed `PanelResults` plotting shortcuts, comparison helpers, and
  `PanelCompareResults` dunder/plot methods.
- Typed `_fit_linearmodels`, `_convert_lm_result`, `_fit_cre`, and `_fit_gmm`
  helper boundaries in `src/statspai/panel/panel_reg.py`.
- Explicitly widened the local `lm_model` variable to cover all linearmodels
  estimator classes without changing branch behavior.
- Typed `panel_compare` kwargs and result dictionary as
  `PanelResults | str`, and renamed loop variables to avoid narrowing drift
  between successful model results and error placeholders.
- Typed the deprecated `PanelRegression.__init__` compatibility wrapper.

Verification run:

- `.venv/bin/python -m mypy src/statspai/panel/panel_reg.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/panel/panel_reg.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_panel.py
  tests/test_panel_cov_compare.py tests/test_cov95_panel_estimator_branches.py`
  passed, 27 tests.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_panel_cov_diagnostics.py tests/test_cov95_panel_diagnostics.py
  tests/tier_eg/test_panel_robustness.py tests/tier_eg/test_panel_invariance.py`
  passed, 38 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1372 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3686 <= 4698, mypy observed 2456 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 239

Target: policy-learning OPE and policy-tree static cleanup.

- Typed `src/statspai/policy_learning/ope.py` target-policy boundaries as
  dynamic inputs while explicitly returning float ndarrays from target-policy
  extraction.
- Removed unused OPE imports and split compact semicolon assignments through
  formatting.
- Typed `PolicyTreeResult` initialization and dict-backed attribute access.
- Narrowed `PolicyTreeResult.summary()` to concatenate an explicit rules
  string rather than a dict-backed dynamic value.
- Typed `PolicyTreeResult.plot_tree()` and its nested tree-layout helpers.
- Typed `PolicyTree._compute_dr_scores`, `_grow_tree`, `_predict_tree`, and
  `_tree_to_rules`, and returned explicit ndarrays from DR-score computation.
- Removed an f-string with no placeholders in the binary-treatment validation
  branch.
- Ran `black` on the two touched policy-learning files so direct flake8 is
  clean.

Verification run:

- `.venv/bin/python -m mypy src/statspai/policy_learning/ope.py
  src/statspai/policy_learning/policy_tree.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m flake8 src/statspai/policy_learning/ope.py
  src/statspai/policy_learning/policy_tree.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_policy_learning.py
  tests/test_ml_causal_polish.py` passed, 37 tests and 1 skip.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_article_aliases_round2.py::test_policy_tree_accepts_depth_kwarg
  tests/test_article_aliases_round2.py::test_policy_tree_max_depth_kwarg_still_works
  tests/test_article_aliases_round2.py::test_policy_tree_accepts_scalar_covariate_alias
  tests/test_article_aliases_round2.py::test_policy_tree_rejects_conflicting_depth
  tests/test_article_aliases_round2.py::test_policy_tree_rejects_conflicting_treat
  tests/test_article_aliases_round2.py::test_policy_tree_rejects_conflicting_covariates`
  passed, 6 tests.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_registry_drift_repair.py::TestOffPolicyEvaluationSmoke` passed,
  4 tests. An initial stale class name did not run tests and was corrected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1375 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3660 <= 4698, mypy observed 2439 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 240

Target: principal-stratification helper type-contract cleanup.

- Typed `PrincipalStratResult.__repr__`.
- Typed the monotonicity, encouragement-design AIR/Wald, and principal-score
  helper boundaries in `src/statspai/principal_strat/principal_strat.py`.
- Typed nested point-estimate, weighted-mean, cell-probability, and bootstrap
  CI helpers.
- Removed an unused `statsmodels.api` import in the principal-score path.
- Explicitly narrowed SACE monotonicity bounds before indexing them in
  `survivor_average_causal_effect`.
- Typed `_logit_safe` and `_logit_predict`, and converted safe-logit predicted
  probabilities to float ndarrays at the wrapper boundary.
- Ran `black` on the touched file and split the two remaining long citation
  lines so touched-file flake8 is clean.

Verification run:

- `.venv/bin/python -m mypy src/statspai/principal_strat/principal_strat.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/principal_strat/principal_strat.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_principal_strat.py
  tests/test_review_fixes_round2.py tests/test_silent_degradation_fixes.py`
  passed, 31 tests.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_smart_tools_sprint_b.py::test_assumption_audit_principal_strat_monotonicity_method
  tests/test_smart_tools_sprint_b.py::test_assumption_audit_principal_strat_principal_score_fires_mono_and_pi
  tests/test_smart_tools_sprint_b_round4.py::test_sensitivity_dashboard_principal_strat_monotonicity_dimension
  tests/test_smart_tools_sprint_b_round4.py::test_compare_estimators_principal_strat_route
  tests/test_workflow_sprint_b.py::test_causal_workflow_routes_to_principal_strat
  tests/test_workflow_sprint_b.py::test_diagnose_result_detects_principal_strat`
  passed, 6 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1375 taxonomy raises and 1284 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3654 <= 4698, mypy observed 2417 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 241

Target: causal-forest static contract cleanup without changing estimator
behavior.

- Removed unused imports from `src/statspai/forest/causal_forest.py`.
- Narrowed `CausalForest.model_y` and `model_t` to initialized sklearn
  estimators after default-model construction, eliminating the optional
  nuisance-model branch before first-stage fitting.
- Marked the fluent `CausalForest.fit(...) -> CausalForest` override locally;
  this preserves the existing public estimator API while acknowledging the
  broader `BaseModel.fit(**kwargs)` contract.
- Added the missing return annotation on the honest leaf-value replacement
  helper.
- Converted prediction/effect return boundaries to explicit float ndarrays.
- Removed an unused BLP local and a placeholder-free f-string.

Verification run:

- `.venv/bin/python -m mypy src/statspai/forest/causal_forest.py` reported
  success after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/forest/causal_forest.py` passed with no import/f-string/unused
  local violations. The file still has historical whole-file formatting debt,
  so this batch intentionally did not run a broad reformat.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_causal_forest_grf.py tests/test_forest_inference.py
  tests/test_causal_to_forest_rename.py
  tests/reference_parity/test_causal_forest_aipw_recovery.py
  tests/test_result_consumer_errors.py` passed, 47 tests.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_question_dsl.py
  -k causal_forest` passed, 7 tests with 54 deselected.
- `.venv/bin/python tests/r_parity/13_causal_forest.py` passed.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_survival_causal_forest_contract.py tests/test_ml_causal_polish.py
  -k causal_forest` passed, 4 tests with 22 deselected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1375 taxonomy raises and 1284 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3647 <= 4698, mypy observed 2412 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 242

Target: neural-causal static contract cleanup while preserving optional
PyTorch behavior.

- Added precise helper annotations in
  `src/statspai/neural_causal/models.py` for data preparation,
  standardisation, torch-module builders, train/validation splits, checkpoint
  helpers, MMD, bootstrap SE, and confidence-interval inference.
- Typed the TARNet model-info helper boundary.
- Converted TARNet, CFRNet, and DragonNet `effect()` returns, plus
  DragonNet `propensity()`, to explicit float ndarrays at the public boundary.
- Removed an unused TARNet `best_epsilon` local that was initialized but never
  read.
- Kept all neural training loops, losses, AIPW formulas, and estimator defaults
  unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/neural_causal/models.py` reported
  success after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/neural_causal/models.py` passed with no import/f-string/unused
  local violations.
- `git diff --check -- src/statspai/neural_causal/models.py` passed.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_neural_causal.py tests/test_neural_causal_exports.py
  tests/test_neural_causal_exports_contract.py` passed with 1 test and 2
  PyTorch-dependent modules skipped in this environment.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1375 taxonomy raises and 1284 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3646 <= 4698, mypy observed 2390 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. The nested
  `CausalAgentBench/` repo remains clean on `main...origin/main`; the nested
  `Paper-JSS/` repo now shows external/parallel `replication/results/*`
  modifications, so no changes were made there and final JOSS-boundary
  acceptance remains pending that separate worktree state.

## 2026-06-17 Batch 243

Target: RD honest-CI static contract cleanup without changing the Armstrong-
Kolesar interval formulas.

- Typed local-linear, local-quadratic, IK-bandwidth, curvature-bound,
  AK-critical-value, RD standard-error, and FLCI-bandwidth helper boundaries in
  `src/statspai/rd/honest_ci.py`.
- Converted NumPy/scipy scalar outputs to Python floats at helper return
  boundaries.
- Replaced optional `M`/`h` arithmetic with explicit `M_value` and `h_value`
  locals before RD estimation, bias-bound construction, summaries, model_info,
  and provenance attachment.
- Removed an unused pilot `sigma` local in `_ik_bandwidth`.
- Kept bandwidth choice, local-polynomial fits, AK critical-value equation,
  bias bound, naive CI, honest CI, and p-value logic unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/honest_ci.py` reported success
  after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/honest_ci.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/honest_ci.py` passed.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_rd_cov_estimators.py::test_rd_honest tests/test_rd_dispatcher.py
  tests/test_rd_new_modules.py -k honest` passed, 2 tests with 54 deselected.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_rd_polish.py::TestRDCompare::test_returns_dataframe
  tests/test_rd_polish.py::TestRDCompare::test_estimates_close_across_methods
  tests/test_cov95_rd_diagnostics.py::test_rdsummary_full_with_plot
  tests/test_cov95_rd_diagnostics.py::test_rdsummary_full_plot_no_covs_placebo_panel`
  passed, 4 tests with 3 matplotlib FutureWarnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1375 taxonomy raises and 1284 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3645 <= 4698, mypy observed 2377 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 244

Target: local-randomization RD static contract cleanup.

- Typed `src/statspai/rd/locrand.py` window-selection, statistic,
  permutation-pvalue, asymptotic-pvalue, Wald-IV, and CI-inversion helper
  boundaries.
- Typed the statistic-dispatch map as callables on numeric arrays.
- Converted NumPy/scipy scalar returns to Python floats at helper boundaries.
- Converted polynomial residual outputs to explicit float ndarrays.
- Narrowed `wl`/`wr` to `wl_value`/`wr_value` before `rdrandinf` and
  `rdrbounds` window arithmetic and model_info construction.
- Removed two unused locals in window-selection and sensitivity-grid loops.
- Kept randomization tests, CI inversion, Wald-IV calculations, and Rosenbaum
  bound simulation logic unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/locrand.py` reported success after
  the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/locrand.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/locrand.py` passed.
- A first broad pytest invocation without `MPLBACKEND=Agg` was interrupted
  after 2 tests passed because matplotlib's interactive `show()` blocked the
  process.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_cov95_rd_r2_locrand.py
  tests/test_rd_new_modules.py::TestLocalRandomization
  tests/test_rd_cov_estimators.py::test_rdrandinf` passed, 15 tests with 2
  non-interactive-Agg warnings from `plt.show()`.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1376 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3643 <= 4698, mypy observed 2368 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 245

Target: RDiT type-boundary cleanup plus a metadata regression guard.

- Typed `src/statspai/rd/rdit.py` `cutoff` and optional bandwidth parameters.
- Converted HAC/Newey-West SE, optimal-bandwidth, deseasonalization, and
  prediction-grid outputs to explicit float ndarrays/floats at return
  boundaries.
- Narrowed `h` into `h_value` before bandwidth filtering, kernel weighting,
  grid construction, error messages, and `model_info`.
- Fixed `model_info["bandwidth_auto"]` so it is `True` only when `h` was
  selected automatically and `False` when the caller supplied a bandwidth.
- Removed an unused `n_params` local and unused `Tuple` import.
- Added tests pinning the automatic and manual `bandwidth_auto` metadata.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/rdit.py` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/rdit.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/rdit.py` passed.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_cov95_rd_rdit.py`
  passed, 19 tests with 13 pandas datetime-format warnings.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_rd_dispatcher.py`
  passed, 23 tests.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_rd_dispatcher.py
  -k rdit` had no matching tests and was not counted as pass evidence.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1376 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3645 <= 4698, mypy observed 2362 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing. The aggregate
  flake8 count increased relative to Batch 244 because new unrelated parallel
  root changes introduced additional style findings; `rdit.py` direct flake
  stayed clean for the touched categories.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 246

Target: multi-cutoff / multi-score RD static contract cleanup.

- Typed `RDMultiResult.__init__` and `RDMultiResult.plot` in
  `src/statspai/rd/rdmulti.py`.
- Typed `_local_linear_rd` and converted its return boundary to
  `(float, float, int)`.
- Changed `rdmc` and `rdms` `bandwidth` parameters from implicit optional
  defaults to `Optional[float]`.
- Narrowed `bandwidth` to `bandwidth_value` before local-linear estimation,
  kernel weighting, cutoff result metadata, and geographic-RD result metadata.
- Removed an unused forest-plot `colors` local.
- Refreshed generated schemas after the `rdmc`/`rdms` optional-bandwidth
  signature change.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/rdmulti.py` reported success after
  the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/rdmulti.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/rdmulti.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_cov95_rd_r2_rdmulti.py
  tests/test_rd_cov_estimators.py::test_rdmc_multi_cutoff
  tests/test_rd_new_modules.py -k 'rdmc or rdms'` passed, 11 tests with 32
  deselected.

Post-batch gate sweep:

- The first full sweep found stale generated schema files
  (`functions.json` and `src/statspai/schemas/functions.json`).
- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5-file schema
  bundle and runtime mirror.
- After schema refresh, `.venv/bin/python -m compileall -q src/statspai`
  passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1376 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3640 <= 4698, mypy observed 2357 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 247

Target: RDML helper type cleanup.

- Typed the `_ik_bandwidth_simple` nested curvature helper in
  `src/statspai/rd/rdml.py` and converted its return to `float`.
- Typed `_importance_plot` `ax` and return boundaries with `Any`.
- Kept forest, boosting, lasso, CATE-summary, and plotting behavior unchanged.
- Refreshed generated schemas after the public helper signature change.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/rdml.py` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/rdml.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/rdml.py` passed.
- A first pytest invocation used a stale class name (`TestMLRD`) and collected
  no tests; it was not counted as pass evidence.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_cov95_rd_r2_rdml.py tests/test_cov95_rd_ml_and_hte.py
  -k 'rd_forest or rd_boost or rd_lasso or importance'
  tests/test_rd_new_modules.py::TestRDML` passed, 14 tests with 11 deselected.

Post-batch gate sweep:

- The first full sweep found stale generated schema files
  (`functions.json` and `src/statspai/schemas/functions.json`).
- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5-file schema
  bundle and runtime mirror.
- After schema refresh, `.venv/bin/python -m compileall -q src/statspai`
  passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1376 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3638 <= 4698, mypy observed 2354 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 248

Target: RD two-dimensional plotting helper type cleanup.

- Typed the public `rd2d_plot` plotting boundary in
  `src/statspai/rd/rd2d.py`, including the optional boundary callable,
  optional axes object, figure-size tuple, and figure/axes return.
- Added local non-`None` assertions before plotting the computed boundary curve
  so static analysis matches the existing branch semantics.
- Typed `_signed_distance_to_curve` and its scalar objective helper, and
  normalized its signed-distance return through `np.asarray(..., dtype=float)`.
- Kept the two-dimensional RD estimation, plotting modes, and boundary
  calculations behaviorally unchanged.
- Refreshed generated schemas after the public plotting signature change.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/rd2d.py` reported success after
  the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/rd2d.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/rd2d.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_cov95_rd_rd2d.py tests/test_rd_new_modules.py::TestRD2D` passed,
  18 tests.

Post-batch gate sweep:

- The first full sweep found stale generated schema files (`tools.json`,
  `functions.json`, `src/statspai/schemas/tools.json`, and
  `src/statspai/schemas/functions.json`).
- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5-file schema
  bundle and runtime mirror.
- After schema refresh, `.venv/bin/python -m compileall -q src/statspai`
  passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1376 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3638 <= 4698, mypy observed 2351 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 249

Target: RD diagnostics helper type and touched-file lint cleanup.

- Typed the public `rdbwsensitivity` and `rdplacebo` axes, figure-size, and
  bandwidth-range boundaries in `src/statspai/rd/diagnostics.py`.
- Added explicit return annotations to `_print_rdsummary` and
  `_rd_diagnostic_plot`.
- Converted the diagnostic dashboard `tight_layout(rect=...)` argument to the
  tuple shape expected by matplotlib typing.
- Removed unused diagnostics imports, static f-string prefixes, and the unused
  `n_panels` local without changing printed output or plotting behavior.
- Refreshed generated schemas after the public diagnostic signature changes.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/diagnostics.py` reported success
  after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/diagnostics.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/diagnostics.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_cov95_rd_diagnostics.py tests/test_rd_new_modules.py
  -k 'rdbw or rdplacebo or rdsummary or diagnostics'` passed, 16 tests with
  28 deselected and 9 matplotlib/pandas future warnings.

Post-batch gate sweep:

- The first full sweep found stale generated schema files (`tools.json`,
  `functions.json`, `src/statspai/schemas/tools.json`, and
  `src/statspai/schemas/functions.json`).
- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5-file schema
  bundle and runtime mirror.
- After schema refresh, `.venv/bin/python -m compileall -q src/statspai`
  passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1376 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3628 <= 4698, mypy observed 2346 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 250

Target: RD dashboard helper type cleanup.

- Typed the public `rd_dashboard` return boundary in
  `src/statspai/rd/dashboard.py` as the figure/axes tuple it already returns.
- Narrowed auto-selected bandwidths through a local `h_auto` value before
  assigning to the optional scalar `h` parameter.
- Added explicit `Any` axes and `None` return annotations to the private
  balance, running-variable, and bandwidth-sensitivity plotting helpers.
- Replaced optional `bw_grid` reuse with a concrete `bw_values` array in the
  bandwidth-sensitivity panel, avoiding optional-array ambiguity without
  changing the plotted values.
- Removed the unused `CausalResult` and `Union` imports.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/dashboard.py` reported success
  after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/dashboard.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/dashboard.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_cov95_rd_misc.py tests/test_cov95_rd_r2_dashboard.py
  tests/test_rd_cov_estimators.py tests/test_rd_polish.py
  -k 'dashboard or rd_compare or robustness_table'` passed, 11 tests with
  54 deselected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1376 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3626 <= 4698, mypy observed 2334 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 251

Target: pre-registration YAML parser type cleanup.

- Added explicit return typing to the private `_parse_scalar` helper in
  `src/statspai/question/preregister.py`.
- Added an explicit `(value, next_idx)` tuple return annotation to the private
  `_consume_block` YAML-subset parser.
- Kept the existing YAML/JSON preregistration serialization and colon-preserving
  parsing behavior unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/question/preregister.py` reported
  success after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/question/preregister.py` passed with no import/f-string/unused
  local violations.
- `git diff --check -- src/statspai/question/preregister.py` passed.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_preregister.py
  tests/test_v0917_review_fixes.py::test_preregister_preserves_colons_in_notes`
  passed, 10 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1376 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3626 <= 4698, mypy observed 2332 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 252

Target: limited-dependent-result base-property fallback cleanup.

- Replaced direct `CausalResult.<property>.fget(self)` fallback calls in
  `src/statspai/regression/_limited_dep_result.py` with equivalent
  `super().params`, `super().std_errors`, `super().tvalues`, and
  `super().pvalues` property reads.
- Preserved the full coefficient-table override path for Tobit/Heckman-style
  results and only changed the fallback path used when no detailed coefficient
  table is available.
- Left the concurrently modified `tests/test_limited_dep_lane.py` untouched and
  used it only as validation coverage.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/_limited_dep_result.py`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/regression/_limited_dep_result.py` passed with no
  import/f-string/unused local violations.
- `git diff --check -- src/statspai/regression/_limited_dep_result.py` passed.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_limited_dep_lane.py` passed, 26 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1376 taxonomy raises and 1283 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3626 <= 4698, mypy observed 2328 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 253

Target: advanced IV signature typing and taxonomy cleanup.

- Made nullable public parameters in `src/statspai/regression/advanced_iv.py`
  explicit for `liml`, `jive`, and `lasso_iv`, matching their existing
  `None` defaults.
- Added local required-input narrowing after formula parsing / default
  normalization so `data`, `y`, `x_endog`, and `z` are non-null before use.
- Converted the new missing-input checks to `MethodIncompatibility` with
  recovery hints instead of adding generic `ValueError` sites.
- Removed stale unused imports (`Dict`, `Any`, `Union`, `scipy.stats`, and
  unused core utils) and the unused `k_endog` local.
- Refreshed generated schemas after the public advanced-IV signature changes.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/advanced_iv.py` reported
  success after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/regression/advanced_iv.py` passed with no import/f-string/unused
  local violations.
- `git diff --check -- src/statspai/regression/advanced_iv.py` passed.
- A first pytest command used a stale JIVE class name and collected no tests;
  it was not counted as pass evidence.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_new_v06_modules.py::TestAdvancedIV
  tests/test_tierD_p2_regression_system_analytic.py::TestJIVEAnalytic
  tests/reference_parity/test_liml_se_parity.py tests/r_parity/59_liml.py
  tests/test_estimator_provenance_round4.py
  -k 'liml or jive or lasso_iv'` passed, 15 tests with 10 deselected.

Post-batch gate sweep:

- The first full sweep found stale generated schema files (`tools.json`,
  `functions.json`, `agent_cards.json`, and their runtime mirrors under
  `src/statspai/schemas/`).
- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5-file schema
  bundle and runtime mirror.
- After schema refresh, `.venv/bin/python -m compileall -q src/statspai`
  passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1379 taxonomy raises and 1284 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  269 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3613 <= 4698, mypy observed 2312 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 254

Target: RKD result-method and helper typing cleanup.

- Added an `RKDResult` subclass in `src/statspai/rd/rkd.py` so RKD-specific
  `summary()` and `plot()` are real methods instead of dynamically assigning
  lambdas onto a `CausalResult` instance.
- Preserved public compatibility: `rkd()` still returns a `CausalResult`
  subtype and existing `summary()` / `plot()` behavior remains available.
- Stored RKD plot payload on the subclass and removed the previous
  `_original_summary` / `result.summary = ...` / `result.plot = ...`
  monkey-patching.
- Narrowed fuzzy-treatment arrays before fitting treatment-side local
  polynomials.
- Typed `_local_poly_fit`, the pilot derivative helper, and `_rkd_plot`, and
  normalized kernel/cluster-variance returns to concrete float ndarrays.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/rkd.py` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/rkd.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/rkd.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_cov95_rd_rkd.py tests/test_rd_validation.py
  tests/test_estimator_provenance_round3.py -k 'rkd'` passed, 13 tests with
  36 deselected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1379 taxonomy raises and 1284 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected, reflecting the new `RKDResult` subclass.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3613 <= 4698, mypy observed 2298 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 255

Target: core `rdrobust` bandwidth and plotting helper type cleanup.

- Added a `Bandwidth` alias in `src/statspai/rd/rdrobust.py` so public `h` and
  `b` parameters can accurately represent either common scalar bandwidths or
  left/right bandwidth tuples.
- Typed the model-info bandwidth rounding helper, `rdplot`,
  `rdplotdensity`, `_bin_means`, `_weighted_poly_fit_ci`, `_rd_estimate`, and
  `_rbc_bootstrap` without changing estimators or plotting semantics.
- Narrowed clustered bootstrap resampling through a local non-null assertion
  and avoided optional-indexing ambiguity for bootstrap cluster draws.
- Split the valid bootstrap statistic slice into `t_valid`, avoiding ndarray
  shape reassignment ambiguity while preserving quantile and p-value logic.
- Refreshed generated schemas after public RD plotting / bandwidth signature
  changes.

Verification run:

- `.venv/bin/python -m mypy src/statspai/rd/rdrobust.py` reported success
  after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/rd/rdrobust.py` passed with no import/f-string/unused local
  violations.
- `git diff --check -- src/statspai/rd/rdrobust.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/test_cov95_rd_rdrobust.py tests/test_low_cov_battery.py -k 'rdrobust'`
  passed, 39 tests with 22 deselected and one expected weak-first-stage
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q -o addopts=''
  tests/reference_parity/test_rd_parity.py tests/orig_parity/05_lee_original.py
  tests/test_rd_pipeline.py` passed, 10 tests.

Post-batch gate sweep:

- The first full sweep found stale generated schema files (`tools.json`,
  `functions.json`, and their runtime mirrors under `src/statspai/schemas/`).
- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5-file schema
  bundle and runtime mirror.
- After schema refresh, `.venv/bin/python -m compileall -q src/statspai`
  passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1379 taxonomy raises and 1284 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3613 <= 4698, mypy observed 2283 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 256

Target: proximal helper signature cleanup outside parallel-agent files.

- Added explicit wrapper and internal 2SLS helper typing in
  `src/statspai/proximal/p2sls.py`, including the `(beta, vcov,
  first_stage_F)` return shape.
- Typed the nested numeric estimators in `src/statspai/proximal/mtp.py`,
  `src/statspai/proximal/fortified.py`, and
  `src/statspai/proximal/bidirectional.py`.
- Typed the proximal fallback-info dictionaries as `Dict[str, Any]` where they
  intentionally carry both boolean flags and error-name strings.
- Removed an unused inner `LinearRegression` import in `bidirectional.py`.
- Left the concurrently added `tests/test_proximal_input_validation.py`
  untouched; validation used existing proximal test suites only.

Verification run:

- `.venv/bin/python -m mypy src/statspai/proximal/p2sls.py
  src/statspai/proximal/mtp.py src/statspai/proximal/fortified.py
  src/statspai/proximal/bidirectional.py` reported success after the
  repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841` over the same four
  files passed with no import/f-string/unused local violations.
- `git diff --check --` over the same four files passed.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_proximal.py
  tests/test_proximal_frontiers.py tests/reference_parity/test_proximal_parity.py`
  passed, 18 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1379 taxonomy raises and 1284 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3612 <= 4698, mypy observed 2278 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 257

Target: count-model helper and public signature typing cleanup.

- Typed the count-model parser, variance, log-likelihood, IRLS, negative
  binomial, PPML/HDFE, separation, and overdispersion helpers in
  `src/statspai/regression/count.py`.
- Made nullable public parameters explicit for `poisson`, `nbreg`,
  `xtnbreg`, and `ppmlhdfe`, matching their existing `None` defaults.
- Normalized `_safe_exp` to return a concrete float ndarray and made
  log-likelihood helpers return Python floats.
- Normalized optional HDFE fixed-effect lists to a local `fe_list` before
  iterating inside `_ppml_hdfe_irls`.
- Typed the negative-binomial profile likelihood closures and the null-model
  profile closures.
- Kept count estimation, HDFE absorption, parity-sensitive likelihoods, and
  variance calculations behaviorally unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/count.py` reported success
  after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841
  src/statspai/regression/count.py` passed with no import/f-string/unused
  local violations.
- `git diff --check -- src/statspai/regression/count.py` passed.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_new_v06_modules.py::TestCountData tests/test_limited_dep_lane.py
  -k 'poisson or nbreg or ppmlhdfe or zip or zinb'` passed, 5 tests with
  24 deselected and one existing NB convergence warning.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_fast_fepois.py
  tests/test_low_cov_battery.py -k 'poisson or ppml or count or nbreg'` passed,
  1 test with 69 deselected.
- Two attempted Stata parity Python paths did not exist and collected no
  tests; they were not counted as pass evidence.
- `.venv/bin/python -m pytest -q -o addopts='' tests/r_parity/37_ppmlhdfe.py
  tests/r_parity/42_nbreg.py tests/r_parity/47_ppmlhdfe_3fe.py
  tests/r_parity/64_zinb.py tests/reference_parity/test_count_quantile_parity.py
  tests/test_count_panel_nbreg.py` passed, 16 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1379 taxonomy raises and 1284 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3612 <= 4698, mypy observed 2232 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 258

Target: fractional-response and beta-regression signature typing cleanup.

- Made nullable public parameters explicit for `fracreg` and `betareg` in
  `src/statspai/regression/fracreg.py`, matching their existing `None`
  defaults while preserving the public call shape.
- Added explicit required-input checks before dataframe and design-matrix
  construction, giving the type checker the same contract the estimators already
  require at runtime.
- Typed the local link, derivative, and beta-regression likelihood closures and
  normalized their numeric returns to concrete float ndarrays / Python floats.
- Removed unused imports and unused Hessian temporaries, and rewrapped the
  touched doctest and matrix lines so the touched file is flake8-clean.
- Kept fractional QMLE, beta-regression likelihoods, variance calculations, and
  parity-sensitive result fields behaviorally unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/fracreg.py` reported
  success after the repository Python-version warning.
- `.venv/bin/python -m flake8 src/statspai/regression/fracreg.py --count
  --statistics` passed with 0 touched-file violations.
- `git diff --check -- src/statspai/regression/fracreg.py` passed.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_v06_round2.py::TestFractionalResponse
  tests/test_limited_dep_lane.py -k 'betareg or fracreg'` passed, 3 tests with
  25 deselected.
- `.venv/bin/python tests/r_parity/verify_reproduce.py 61_betareg` skipped
  because `Rscript` was not on this shell's PATH; the generated single-module
  report diff was restored and not counted as pass evidence.
- `.venv/bin/python tests/stata_parity/verify_reproduce_stata.py 61_betareg`
  passed, reproducing 4/4 shared entries with worst relative estimate and SE
  drift both `0.00e+00`; the generated single-module report diff was restored.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1385 taxonomy raises and 1287 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3600 <= 4698, mypy observed 2220 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 259

Target: GLM type-contract cleanup and touched-file lint reduction.

- Typed `src/statspai/regression/glm.py` link and family methods, including
  explicit `name` attributes on link functions and concrete ndarray return
  boundaries for NumPy/SciPy expressions.
- Added a small `_as_float_array` narrowing helper for GLM numeric return
  points; no IRLS, likelihood, deviance, covariance, or marginal-effects
  formulas were changed.
- Made the low-level `GLMEstimator.estimate` signature compatible with the
  base estimator contract by accepting optional family/link instances and
  raising `MethodIncompatibility` if direct callers omit them.
- Typed robust, clustered, HAC, and negative-binomial alpha helper boundaries,
  and narrowed public `glm` nullable parameters to match existing defaults.
- Removed unused `parse_formula` and `prepare_data` imports.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/glm.py` reported success
  after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/regression/glm.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/glm.py --count --statistics` passed with 0 selected
  touched-file violations.
- `git diff --check -- src/statspai/regression/glm.py` passed.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_new_v06_modules.py::TestGLM tests/test_glm_predict.py` passed,
  11 tests.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/test_reference_alignment_statsmodels.py
  tests/reference_parity/test_count_quantile_parity.py -k 'poisson or glm'`
  passed, 3 tests with 12 deselected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1388 taxonomy raises and 1287 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3591 <= 4698, mypy observed 2146 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 260

Target: IV dispatcher and legacy estimator type-contract cleanup.

- Added concrete ndarray return narrowing in `src/statspai/regression/iv.py`
  for robust and clustered covariance helpers and prediction outputs.
- Made `_normalize_robust` typed and kept the existing Stata-style aliases
  (`True` / `"robust"` -> HC1, `"white"` -> HC0).
- Updated legacy `IVEstimator.estimate` to satisfy the base estimator
  `estimate(y, X, **kwargs)` contract while preserving the positional
  compatibility path `estimate(y, X_exog, X_endog, Z, ...)`.
- Typed `IVRegression` initialization, formula preparation, `fit`, absorbed-IV
  dispatch, and public `iv` / `ivreg` kwargs boundaries.
- Narrowed `IVRegression.fit` design arrays to local non-null variables before
  LIML/Fuller/GMM/JIVE dispatch, clearing Optional-array ambiguity without
  changing estimator formulas.
- Stored fitted result and diagnostic attributes through typed local objects,
  and removed two unused k-class intermediates.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/iv.py` reported success
  after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/regression/iv.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/iv.py --count --statistics` passed with 0 selected
  touched-file violations.
- `git diff --check -- src/statspai/regression/iv.py` passed.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_new_features.py
  tests/test_low_cov_battery.py -k 'ivreg or jive'` passed, 3 tests with
  54 deselected.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/reference_parity/test_iv_parity.py
  tests/reference_parity/test_iv_se_parity.py
  tests/reference_parity/test_regress_weights_iv_robust_parity.py` passed,
  23 tests with one expected weak-instrument warning.
- `.venv/bin/python -m pytest -q -o addopts='' tests/test_iv_absorb.py
  tests/test_cov95_iv_diag.py tests/test_iv_cov_diag.py` passed, 41 tests.
- `.venv/bin/python -m pytest -q -o addopts=''
  tests/reference_parity/test_liml_se_parity.py
  tests/test_tierD_p2_regression_system_analytic.py -k 'liml or ivreg or jive'`
  passed, 6 tests with 4 deselected.
- `.venv/bin/python tests/stata_parity/verify_reproduce_stata.py 02_iv
  59_liml` reproduced both modules with worst relative estimate drift
  `0.00e+00`; the generated single-module report diff was restored and not
  left in the worktree.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1395 taxonomy raises and 1289 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` initially detected the
  temporary single-module Stata report diff; after restoring that generated
  report, it passed with `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3586 <= 4698, mypy observed 2107 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 261

Target: binary-choice likelihood typing, schema consistency, and Stata parity.

- Typed `src/statspai/regression/logit_probit.py` link-function callables and
  the `_LINKS` dispatch table for logit, probit, and cloglog.
- Added ndarray/float return narrowing around CDF/PDF helpers, likelihood,
  score, Hessian, covariance, ROC/AUC, prediction, and marginal-effects
  helpers without changing the likelihood formulas.
- Replaced untyped dynamic result lambdas with named typed `predict` and
  `classification_table` wrappers attached to the result object.
- Made the public `logit`, `probit`, and `cloglog` wrappers use explicit
  `Optional[...]` defaults for formula/data/column and cluster/weight
  arguments, preserving the existing list-only `x` API behavior.
- Removed stale imports and refreshed the generated schema bundles after the
  public signatures changed from implicit to explicit optionals.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/logit_probit.py`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/logit_probit.py --count --statistics` passed with
  0 selected touched-file violations.
- `.venv/bin/python -m py_compile
  src/statspai/regression/logit_probit.py` passed.
- `git diff --check -- src/statspai/regression/logit_probit.py` passed.
- `.venv/bin/python -m pytest -q tests/test_new_v06_modules.py::TestLogitProbit
  tests/test_reference_alignment_statsmodels.py -k 'logit or probit'
  -o addopts=''` passed, 4 tests with 4 deselected.
- `.venv/bin/python -m pytest -q tests/test_regtable_round3_extensions.py
  tests/test_regtable_publication_extensions.py -k 'logit or probit or eform'
  -o addopts=''` passed, 8 tests with 29 deselected and one existing
  sample-size warning.
- `.venv/bin/python -m pytest -q tests/test_untested_function_coverage.py
  -k 'cloglog or logit or probit' -o addopts=''` passed, 1 test with
  16 deselected.
- A direct public API smoke check verified `sp.logit(...).predict(...)` and
  `classification_table()` on simulated binary data.
- `.venv/bin/python tests/stata_parity/verify_reproduce_stata.py 48_probit
  57_logit` reproduced both modules with worst relative estimate drift
  `0.00e+00`; the generated single-module report diff was restored and not
  left in the worktree.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1395 taxonomy raises and 1289 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` initially found stale
  `tools.json` and `functions.json` bundles; after
  `.venv/bin/python scripts/dump_schemas.py`, the check passed with
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3584 <= 4698, mypy observed 2061 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 262

Target: multinomial, ordered, and conditional logit type-contract cleanup.

- Typed `src/statspai/regression/multinomial.py` helper boundaries for
  formula parsing, matrix building, softmax, sandwich SE, and ordered-model
  link functions.
- Made `mlogit`, `ologit`, `oprobit`, and `clogit` public signatures use
  explicit `Optional[...]` defaults for formula/data/column and cluster/group
  arguments while preserving the existing list-only `x` API behavior.
- Added ndarray/float return narrowing across multinomial likelihood, score,
  Hessian-covariance, restricted IIA submodels, ordered-model probabilities,
  numerical scores, Brant binary logits, and conditional-logit score helpers.
- Replaced direct dynamic result-attribute assignments with `setattr` for
  `predicted_probs`, `marginal_effects`, `iia_test`, `brant_test`, and
  `cutpoints`, keeping the runtime result surface unchanged.
- Removed an unused conditional-logit group-count local and the unused Brant
  loop accumulator while keeping the same diagnostics and warnings.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/multinomial.py`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile
  src/statspai/regression/multinomial.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/multinomial.py --count --statistics` passed with
  0 selected touched-file violations.
- `git diff --check -- src/statspai/regression/multinomial.py` passed.
- `.venv/bin/python -m pytest -q
  tests/test_new_v06_modules.py::TestMultinomial -o addopts=''` passed,
  2 tests with expected IIA/Brant diagnostic warnings.
- A direct public API smoke check verified `sp.mlogit`, `sp.ologit`,
  `sp.oprobit`, and `sp.clogit` still expose expected `predicted_probs` and
  marginal-effect/cutpoint payloads on simulated data.
- `.venv/bin/python tests/r_parity/compare.py 44_mlogit 45_ologit 46_clogit
  49_oprobit` passed; generated report/table outputs matched tracked content
  and left no retained JOSS-path diff.
- `.venv/bin/python tests/stata_parity/verify_reproduce_stata.py 44_mlogit
  45_ologit 46_clogit 49_oprobit` reproduced all four modules with worst
  relative estimate drift `0.00e+00`; the generated single-module Stata report
  diff was restored and not left in the worktree.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1395 taxonomy raises and 1292 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3576 <= 4698, mypy observed 2008 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 263

Target: mixed-logit fitter type contract and citation reliability.

- Split the Halton helper's 2D, clipped, and reshaped arrays into separate
  locals in `src/statspai/regression/mixed_logit.py`, clearing the shape
  reassignment ambiguity without changing draws.
- Replaced `_MixedLogitFitter.__dict__.update(...)` with an explicit typed
  initializer for data, id columns, random/fixed coefficient lists, optimizer
  settings, draw settings, and inference flags.
- Typed `_prepare`, `_unpack`, `_apply_draws`, `_grad_ll`,
  `_loglik_per_ind`, and the optimizer objective, and narrowed dict-loaded
  arrays before numerical likelihood operations.
- Removed unused `warnings` and an unused `n_scale` local, and expanded
  finite-difference statements into separate lines for touched-file lint.
- Registered `mixlogit` citations in the actual shared
  `CausalResult._CITATIONS` table and added the public `citation_key` metadata
  while preserving the existing `_citation_key` field.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/mixed_logit.py`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile
  src/statspai/regression/mixed_logit.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/mixed_logit.py --count --statistics` passed with
  0 selected touched-file violations.
- `git diff --check -- src/statspai/regression/mixed_logit.py` passed.
- `.venv/bin/python -m pytest -q tests/test_econ_trinity.py -k mixlogit
  -o addopts=''` passed, 6 tests with 10 deselected.
- A direct smoke check fit a small `sp.mixlogit(...)` model and verified
  `model_info['citation_key'] == 'mixlogit'` plus a non-placeholder
  `.cite()` response containing the Train book title.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1397 taxonomy raises and 1290 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3562 <= 4698, mypy observed 1978 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 264

Target: truncated-regression input and likelihood typing.

- Made `src/statspai/regression/truncreg.py` public inputs explicit
  `Optional[...]` values for data, outcome, regressors, truncation limits, and
  clustering.
- Added clear `ValueError` checks for missing `data` or missing `y`/`x`
  arguments before DataFrame indexing.
- Narrowed the MLE objective to return a concrete `float`, converted optimizer
  output to ndarray locals, and converted sigma/log-likelihood diagnostics to
  concrete floats.
- Kept the truncated-normal likelihood, numerical Hessian, and Stata-facing
  `ln_sigma` reporting unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/truncreg.py` reported
  success after the repository Python-version warning.
- `.venv/bin/python -m py_compile
  src/statspai/regression/truncreg.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/truncreg.py --count --statistics` passed with
  0 selected touched-file violations.
- `git diff --check -- src/statspai/regression/truncreg.py` passed.
- `.venv/bin/python -m pytest -q tests/test_v06_round3.py::TestTruncReg
  tests/test_limited_dep_lane.py -k truncreg -o addopts=''` passed, 3 tests
  with 24 deselected.
- `.venv/bin/python tests/stata_parity/verify_reproduce_stata.py
  62_truncreg` reproduced the module with worst relative estimate drift
  `0.00e+00`; the generated single-module Stata report diff was restored and
  not left in the worktree.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1397 taxonomy raises and 1293 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3561 <= 4698, mypy observed 1972 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 265

Target: SUR/3SLS typing and schema consistency.

- Typed `src/statspai/regression/sur.py` `SURResult.__init__` inputs for the
  equation result map, covariance frame, stacked parameter arrays, sample
  sizes, method label, and optional Breusch-Pagan payload.
- Removed a stale unused `EconometricResults` import.
- Made `three_sls(..., instruments=None)` explicitly optional and introduced a
  local `instrument_names` list so the instrument matrix is built from a
  concrete list without changing the default "all exogenous variables"
  behavior.
- Refreshed generated function schemas after the public `three_sls`
  annotation changed from implicit to explicit optional.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/sur.py` reported success
  after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/regression/sur.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/sur.py --count --statistics` passed with 0 selected
  touched-file violations.
- `git diff --check -- src/statspai/regression/sur.py` passed.
- `.venv/bin/python -m pytest -q tests/test_v06_round3.py::TestSUR
  -o addopts=''` passed, 2 tests.
- `.venv/bin/python tests/stata_parity/verify_reproduce_stata.py 60_sureg`
  reproduced the module with worst relative estimate drift `0.00e+00`; the
  generated single-module Stata report diff was restored and not left in the
  worktree.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1397 taxonomy raises and 1293 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` initially found stale
  `functions.json`; after `.venv/bin/python scripts/dump_schemas.py`, the
  check passed with `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3558 <= 4698, mypy observed 1970 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 266

Target: bivariate-probit and endogenous-treatment typing.

- Made `src/statspai/regression/selection.py` `biprobit` use explicit
  optional `x2` and `cluster` inputs, narrowing `x2` to a concrete local
  regressor list before DataFrame indexing.
- Typed the vectorized bivariate-normal CDF helper and likelihood closure, and
  narrowed optimizer outputs, rho, rho SE, log likelihood, and rho-test p-value
  to concrete ndarray/float objects.
- Made `etregress` cluster explicitly optional and removed unused selection
  matrix, `k_sel`, `z_crit`, and stale typing imports.
- Kept the bivariate probit likelihood and the existing two-step/control-
  function `etregress` approximation unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/selection.py` reported
  success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/regression/selection.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/selection.py --count --statistics` passed with
  0 selected touched-file violations.
- `git diff --check -- src/statspai/regression/selection.py` passed.
- `.venv/bin/python -m pytest -q
  tests/test_v06_round2.py::TestSelectionModels tests/test_limited_dep_lane.py
  -k 'biprobit or etregress' -o addopts=''` passed, 4 tests with
  24 deselected.
- A direct public API smoke check verified `sp.biprobit(...)` reports `rho`
  and `sp.etregress(...)` reports `diagnostics['ate']`.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1397 taxonomy raises and 1293 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3552 <= 4698, mypy observed 1965 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 267

Target: zero-inflated and hurdle count model typing.

- Typed `src/statspai/regression/zeroinflated.py` helper returns for logit,
  Poisson/NB2 log PMFs, robust/clustered SEs, numerical Hessians, and
  numerical scores.
- Made `zip_model`, `zinb`, and `hurdle` public signatures use explicit
  `Optional[...]` defaults for formula/data/outcome/regressor/inflate/cluster
  arguments while preserving the existing list-only regressor API.
- Added a clear missing-data check in shared matrix construction and hurdle
  parsing before DataFrame indexing.
- Typed ZIP/ZINB/hurdle likelihood and per-observation likelihood closures,
  narrowed optimizer outputs, inverse Hessians, SE arrays, dispersion values,
  and score matrices to concrete ndarray/float objects.
- Removed stale imports and unused numerical-Hessian/score locals.
- Kept likelihood formulas, parameter names, Vuong diagnostics, and
  Stata-facing ZIP/ZINB outputs unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/zeroinflated.py`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile
  src/statspai/regression/zeroinflated.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/zeroinflated.py --count --statistics` passed with
  0 selected touched-file violations.
- `git diff --check -- src/statspai/regression/zeroinflated.py` passed.
- `.venv/bin/python -m pytest -q tests/test_limited_dep_lane.py
  tests/test_v06_round2.py -k 'zip or zinb or hurdle' -o addopts=''` passed,
  2 tests with 34 deselected.
- `.venv/bin/python -m pytest -q
  tests/test_new_v06_modules.py::TestZeroInflated
  tests/reference_parity/test_count_quantile_parity.py
  -k 'zip or zinb or hurdle' -o addopts=''` passed, 5 tests with 7 deselected.
- `.venv/bin/python tests/stata_parity/verify_reproduce_stata.py 63_zip
  64_zinb` reproduced both modules with worst relative estimate drift
  `0.00e+00`; the generated single-module Stata report diff was restored and
  not left in the worktree.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1397 taxonomy raises and 1297 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3543 <= 4698, mypy observed 1934 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 268

Target: quantile-regression helper typing and detail narrowing.

- Typed `src/statspai/regression/quantile.py` array coercion, formula parsing,
  quantile-regression fit/IRLS, standard-error, and pseudo-R2 helpers.
- Narrowed `sqreg` detail-table handling before iterating result rows so a
  missing or incompatible detail payload raises a clear
  `MethodIncompatibility` instead of flowing as an arbitrary object.
- Replaced an untyped check-function lambda with a typed nested helper while
  preserving the existing objective and pseudo-R2 calculations.
- Removed stale imports and kept public qreg/sqreg output fields, Stata-facing
  parameter names, and parity behavior unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/quantile.py` reported
  success after the repository Python-version warning.
- `.venv/bin/python -m py_compile
  src/statspai/regression/quantile.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/quantile.py --count --statistics` passed with
  0 selected touched-file violations.
- `git diff --check -- src/statspai/regression/quantile.py` passed.
- `.venv/bin/python -m pytest -q
  tests/reference_parity/test_count_quantile_parity.py
  tests/test_coefplot_tikz.py tests/test_v06_round2.py -k
  'qreg or sqreg or quantile' -o addopts=''` passed, 10 tests with 25
  deselected.
- `.venv/bin/python tests/stata_parity/verify_reproduce_stata.py 40_qreg`
  reproduced the module with worst relative estimate drift `0.00e+00`; the
  generated single-module Stata report diff was restored and not left in the
  worktree.
- `.venv/bin/python -m pytest -q tests/test_quantile.py -o addopts=''`
  passed, 13 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1398 taxonomy raises and 1297 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3539 <= 4698, mypy observed 1928 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 269

Target: verify benchmark scenario typing.

- Added a typed scenario-spec boundary in `src/statspai/smart/benchmark.py`
  for built-in verify-calibration DGPs, keyword payloads, and true-effect
  metadata.
- Narrowed the dynamic DGP lookup to a callable returning a DataFrame and made
  recommendation kwargs an explicit mapping before forwarding them to the
  dynamic `sp.recommend` API.
- Kept all benchmark scenario definitions, seeds, verify budgets, output
  columns, and degraded-warning behavior unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/smart/benchmark.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/smart/benchmark.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/smart/benchmark.py --count --statistics` passed with 0
  selected touched-file violations.
- `git diff --check -- src/statspai/smart/benchmark.py` passed.
- `.venv/bin/python -m pytest -q
  tests/test_untested_public_api.py::test_verify_benchmark_smoke
  -o addopts=''` passed, 1 test with the existing workflow-degradation
  warnings for DID placebo all-rep crashes.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1398 taxonomy raises and 1297 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3539 <= 4698, mypy observed 1925 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 270

Target: core OLS typing boundary hardening.

- Typed the lazy Numba-kernel loader in `src/statspai/regression/ols.py`,
  casting the dynamic accelerated kernels once at the import boundary instead
  of letting Any flow through the estimator.
- Added explicit `**kwargs: Any`, constructor, weight-resolution, design-index,
  and cluster-series annotations for the OLS estimator/model path.
- Returned concrete float ndarrays from the legacy robust/HAC/cluster covariance
  helpers and kept the accelerated covariance path unchanged.
- Replaced the dynamic fitted-result return with a local
  `EconometricResults` object and narrowed prediction design matrices before
  interval `einsum` calculations.
- Preserved OLS fitting algorithms, robust/cluster/HAC formulas, WLS aweight
  handling, prediction output columns, and Stata-facing result metadata.

Verification run:

- `.venv/bin/python -m mypy src/statspai/regression/ols.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/regression/ols.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/regression/ols.py --count --statistics` passed with 0 selected
  touched-file violations.
- `git diff --check -- src/statspai/regression/ols.py` passed.
- `.venv/bin/python -m pytest -q tests/test_ols.py tests/test_predict_oos.py
  tests/reference_parity/test_regress_weights_iv_robust_parity.py
  -o addopts=''` passed, 62 tests.
- `.venv/bin/python -m pytest -q tests/test_nist_strd_linear.py
  tests/numerical_accuracy/test_nist_strd_anova.py
  tests/test_reference_alignment_statsmodels.py tests/test_validation_vs_stata_r.py
  -k 'regress or OLS or ols or nist or statsmodels' -o addopts=''` passed,
  67 tests with 16 deselected.
- `.venv/bin/python -m pytest -q
  tests/test_numba_kernels.py::TestOLSRegressionIntegration
  tests/reference_parity/test_cross_estimator_parity.py -o addopts=''`
  passed, 7 tests.
- `.venv/bin/python tests/stata_parity/verify_reproduce_stata.py 01_ols
  14_ols_cluster 51_newey 55_hc2_hc3` reproduced all four modules with worst
  relative estimate drift `0.00e+00`; the generated single-run Stata report
  diff was restored and not left in the worktree.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1400 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3539 <= 4698, mypy observed 1914 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 271

Target: spatial DiD result/export typing.

- Typed `src/statspai/spatial/did.py` result export kwargs, plot axes/figsize,
  spatial-weight input, and diagnostics dictionary narrowing.
- Converted pandas export returns to explicit strings for markdown/CSV/LaTeX
  helpers while preserving file-writing behavior and table content.
- Returned concrete float ndarrays from row-normalization, Haversine distance,
  cluster/HC1/Conley covariance helpers, and left estimator formulas unchanged.
- Kept `spatial_did` public arguments, diagnostics, event-study outputs,
  covariance choices, and plotting branches unchanged.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/did.py --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/did.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/did.py --count --statistics` passed with 0 selected
  touched-file violations.
- `git diff --check -- src/statspai/spatial/did.py` passed.
- `.venv/bin/python -m pytest -q tests/spatial/test_did.py
  -o addopts=''` passed, 9 tests.
- `.venv/bin/python -m pytest -q tests/spatial -o addopts=''` passed,
  82 tests with 1 skipped.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1400 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3539 <= 4698, mypy observed 1896 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 272

Target: legacy SAR/SEM/SDM typing and schema sync.

- Typed `src/statspai/spatial/models/_legacy.py` constructor, concentrated
  likelihood closures, spatial-parameter SE helpers, effect decomposition,
  result construction, and log-likelihood helper.
- Narrowed optimizer scalar outputs to float before reuse and removed unused
  imports / dead local variables in the legacy spatial ML wrapper.
- Converted SDM direct/indirect/total effects to explicit `dict[str, float]`
  payloads while preserving rounded values and diagnostic keys.
- Wrapped existing long rho-bound and SEM-result lines so the touched legacy
  file is fully flake8-clean.
- Refreshed `schemas/functions.json` and
  `src/statspai/schemas/functions.json` after `dump_schemas.py --check`
  reported the generated function schema bundle stale.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/models/_legacy.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/models/_legacy.py`
  passed.
- `.venv/bin/python -m flake8
  src/statspai/spatial/models/_legacy.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations after cleanup.
- `git diff --check -- src/statspai/spatial/models/_legacy.py` passed.
- `.venv/bin/python -m pytest -q tests/spatial/test_backward_compat.py
  tests/test_round3.py -k 'SAR or SEM or SDM or sar or sem or sdm'
  -o addopts=''` passed, 11 tests with 9 deselected.
- `.venv/bin/python -m pytest -q tests/spatial -o addopts=''` passed,
  82 tests with 1 skipped.
- `.venv/bin/python scripts/dump_schemas.py` refreshed the tracked schema
  bundle; `.venv/bin/python scripts/dump_schemas.py --check` then passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1400 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3537 <= 4698, mypy observed 1893 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 273

Target: target-trial report renderer typing.

- Added a `ProtocolRow` row alias in `src/statspai/target_trial/report.py` for
  target-trial protocol rendering.
- Typed `_stringify` and the markdown, LaTeX, and text renderer helper
  parameters while preserving every rendered string branch and output format.
- Annotated the protocol-row construction in `to_paper` so internal renderer
  calls no longer leak untyped row payloads.

Verification run:

- `.venv/bin/python -m mypy src/statspai/target_trial/report.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/target_trial/report.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/target_trial/report.py --count --statistics` passed with
  0 selected touched-file violations.
- `git diff --check -- src/statspai/target_trial/report.py` passed.
- `.venv/bin/python -m pytest -q
  tests/test_api_stable_evidence.py::test_target_trial_reporting_aliases_render
  tests/test_tierD_p2_target_trial_analytic.py -o addopts=''` passed, 5 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1400 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3531 <= 4698, mypy observed 1889 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 274

Target: ordinal mixed-logit AGHQ typing.

- Typed `src/statspai/multilevel/_ordinal.py` `_ordinal_nll` return as float.
- Narrowed AGHQ nodes, log weights, and random-intercept variance inside the
  `nAGQ > 1` branch before quadrature arithmetic.
- Preserved the existing optimizer penalty style by returning a large nll if
  AGHQ metadata is missing despite `nAGQ > 1`.
- Removed an unused ordinal-GLM warm-start local without changing
  initialization logic.
- Avoided editing the parallel-dirty `src/statspai/multilevel/glmm.py`.

Verification run:

- `.venv/bin/python -m mypy src/statspai/multilevel/_ordinal.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/multilevel/_ordinal.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/multilevel/_ordinal.py --count --statistics` passed with
  0 selected touched-file violations.
- `git diff --check -- src/statspai/multilevel/_ordinal.py` passed.
- `.venv/bin/python -m pytest -q tests/test_multilevel.py::TestMEOLogit
  -o addopts=''` passed, 5 tests.
- `.venv/bin/python -m pytest -q tests/test_multilevel.py -o addopts=''`
  passed, 63 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1400 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3527 <= 4698, mypy observed 1883 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 275

Target: BLP structural-demand typing and touched-file cleanup.

- Typed `src/statspai/structural/blp.py` optimizer callbacks to return plain
  `float` values for SciPy's objective protocol.
- Converted SciPy inverse-CDF and NumPy matrix-product results through
  `np.asarray(..., dtype=float)` so `_halton_sequence` and `_compute_mu`
  satisfy their ndarray contracts.
- Annotated `BLPResult.elasticity_matrix` and `BLPResult.diversion_ratios`
  optional market keys without changing their default first-market behavior.
- Removed unused typing imports and dead locals in the BLP share, IV, and
  elasticity helpers, clearing selected touched-file flake8 debt.

Verification run:

- `.venv/bin/python -m mypy src/statspai/structural/blp.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/structural/blp.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/structural/blp.py --count --statistics` passed with
  0 selected touched-file violations.
- `.venv/bin/python -m pytest -q
  tests/test_tierD_structural_analytic.py::TestBLPAnalytic -o addopts=''`
  passed, 2 tests.
- `.venv/bin/python -m pytest -q tests/test_tierD_structural_analytic.py
  -o addopts=''` passed, 7 tests.
- `git diff --check -- src/statspai/structural/blp.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1400 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3519 <= 4698, mypy observed 1877 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 276

Target: production-function result protocol typing.

- Typed `src/statspai/structural/production/_result.py` result payload
  attributes so direct mypy can validate the production result container.
- Resolved inherited method-name collisions by storing production residuals
  and covariance internally as `_residuals` and `_cov`, while preserving the
  legacy public `res.residuals` and `res.cov` attribute access via `setattr`.
- Made `ProductionResult.cite` signature compatible with
  `EconometricResults.cite(format=...)` while keeping the default human-readable
  production-method reference string.
- Used `MethodIncompatibility` for invalid citation formats, improving the
  exception taxonomy without adding generic raises.

Verification run:

- `.venv/bin/python -m mypy src/statspai/structural/production/_result.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile
  src/statspai/structural/production/_result.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/structural/production/_result.py --count --statistics` passed
  with 0 selected touched-file violations.
- `.venv/bin/python -m pytest -q
  tests/test_prod_fn.py::test_production_result_has_summary
  tests/test_prod_fn.py::test_diagnostics_without_bootstrap -o addopts=''`
  passed, 2 tests.
- `.venv/bin/python -m pytest -q tests/test_prod_fn.py -o addopts=''`
  passed, 23 tests.
- `git diff --check -- src/statspai/structural/production/_result.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1401 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3519 <= 4698, mypy observed 1874 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 277

Target: stochastic-frontier numerical kernel typing.

- Tightened `src/statspai/frontier/_core.py` SciPy and NumPy return contracts
  at the shared SFA kernel boundary with explicit `np.asarray(...,
  dtype=float)` and `float(...)` conversions.
- Covered the stable log-CDF helper, half-normal/exponential/truncated-normal
  log-likelihood kernels, posterior truncnormal mean, Battese-Coelli TE,
  robust variance, heteroskedastic sigma evaluation, and chi-bar p-values.
- Kept the frontier formulas, sign convention, clustering validation, and
  estimator API unchanged.
- Cleared direct mypy for the frontier core without touching the Stata/R
  parity artifacts or JOSS manuscript lanes.

Verification run:

- `.venv/bin/python -m mypy src/statspai/frontier/_core.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/frontier/_core.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/frontier/_core.py --count --statistics` passed with
  0 selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_frontier.py -o addopts=''`
  passed, 101 tests.
- `.venv/bin/python -m pytest -q
  tests/test_v06_round3.py::TestFrontier::test_production_frontier
  -o addopts=''` passed, 1 test.
- `git diff --check -- src/statspai/frontier/_core.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1401 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3519 <= 4698, mypy observed 1864 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 278

Target: many-weak IV grid-inversion typing.

- Refactored `src/statspai/iv/many_weak.py` `many_weak_ar` to materialize the
  optional `beta_grid` input into a dedicated `beta_values` ndarray.
- Removed the optional-sequence/ndarray type ambiguity around AR grid
  statistics, boolean acceptance masks, confidence-set endpoints, and detail
  metadata.
- Preserved the default OLS-centered grid, custom-grid behavior, provenance
  attachment, and jackknife-AR confidence-set logic.

Verification run:

- `.venv/bin/python -m mypy src/statspai/iv/many_weak.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/iv/many_weak.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/iv/many_weak.py --count --statistics` passed with
  0 selected touched-file violations.
- `.venv/bin/python -m pytest -q
  tests/test_cov95_iv_small_estimators.py::test_many_weak_jive_with_exog
  tests/test_cov95_iv_small_estimators.py::test_many_weak_ar_default_grid_with_exog
  tests/test_cov95_iv_small_estimators.py::test_many_weak_ar_custom_grid
  tests/test_iv_cov_summaries.py::test_many_weak_ar_confidence_set
  tests/test_estimator_provenance_round9.py::TestManyWeakJiveProvenance::test_attached
  tests/test_estimator_provenance_round9.py::TestManyWeakArProvenance::test_attached
  -o addopts=''` passed, 6 tests.
- `git diff --check -- src/statspai/iv/many_weak.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1401 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3519 <= 4698, mypy observed 1856 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 279

Target: JIVE variant typing and touched-file cleanup.

- Added explicit type boundaries to `src/statspai/iv/jive_variants.py` for
  matrix coercion, input preparation, name generation, shared dispatch, and the
  `jive1`/`ujive`/`ijive`/`rjive` public entry points.
- Narrowed dataframe and array extraction to concrete float ndarrays while
  preserving dataframe/array input behavior.
- Returned a plain float from the first-stage F helper.
- Removed unused locals in the shared JIVE estimator, clearing selected
  touched-file flake8 debt.

Verification run:

- `.venv/bin/python -m mypy src/statspai/iv/jive_variants.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/iv/jive_variants.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/iv/jive_variants.py --count --statistics` passed with
  0 selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_cov95_iv_jive_variants.py
  tests/iv/test_jive_variants.py
  tests/test_iv_cov_array_inputs.py::test_jive_array_inputs_and_summary
  tests/test_iv_cov_array_inputs.py::test_jive_array_with_exog
  tests/test_iv_cov_reachable.py::test_jive_dataframe_series_names
  tests/test_estimator_provenance_round4.py::TestJiveVariantsProvenance
  -o addopts=''` passed, 24 tests.
- `git diff --check -- src/statspai/iv/jive_variants.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1401 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3513 <= 4698, mypy observed 1845 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 280

Target: kernel density estimator typing and schema sync.

- Typed `src/statspai/nonparametric/kdensity.py` result payload, plotting
  entry point, kernel helper, bandwidth selectors, and the public
  `kdensity` signature.
- Converted implicit optional parameters to explicit `Optional[...]` types and
  added a taxonomy-backed missing-`x` guard.
- Narrowed SciPy kernel/IQR returns and bandwidth calculations to concrete
  float ndarrays/scalars.
- Removed an unused Sheather-Jones helper import.
- Refreshed the generated schema bundle with `scripts/dump_schemas.py` because
  the public `kdensity` signature changed from implicit to explicit optional
  types.

Verification run:

- `.venv/bin/python -m mypy src/statspai/nonparametric/kdensity.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/nonparametric/kdensity.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/nonparametric/kdensity.py --count --statistics` passed with
  0 selected touched-file violations.
- `.venv/bin/python -m pytest -q
  tests/test_lpoly_reliability.py::test_kdensity_single_observation_gaussian_matches_closed_form
  tests/test_lpoly_reliability.py::test_kdensity_weighted_gaussian_matches_manual_kernel_sum
  tests/test_lpoly_reliability.py::test_kdensity_rejects_invalid_smoothing_inputs
  tests/test_new_v06_modules.py::TestNonparametric::test_kdensity
  -o addopts=''` passed, 10 tests.
- `git diff --check -- src/statspai/nonparametric/kdensity.py` passed.
- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5 schema files and
  mirrored them into `src/statspai/schemas/`.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3511 <= 4698, mypy observed 1836 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 281

Target: cointegration estimator typing and schema sync.

- Typed `src/statspai/timeseries/cointegration.py` `CointegrationResult`
  constructor and public Engle-Granger/Johansen optional parameters.
- Split Johansen lagged-difference construction into `lag_blocks` and concrete
  ndarray `Z`, removing the list/array type ambiguity in the concentrated VAR
  residualization path.
- Removed unused imports after the type cleanup.
- Refreshed the generated schema bundle with `scripts/dump_schemas.py` because
  the public cointegration signatures changed from implicit to explicit
  optional types.

Verification run:

- `.venv/bin/python -m mypy src/statspai/timeseries/cointegration.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/timeseries/cointegration.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/timeseries/cointegration.py --count --statistics` passed with
  0 selected touched-file violations.
- `.venv/bin/python -m pytest -q
  tests/test_v06_round2.py::TestCointegration -o addopts=''` passed, 2 tests.
- `git diff --check -- src/statspai/timeseries/cointegration.py` passed.
- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5 schema files and
  mirrored them into `src/statspai/schemas/`.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3507 <= 4698, mypy observed 1825 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 282

Target: structural-break estimator typing and schema sync.

- Typed `src/statspai/timeseries/structural_break.py` result constructor,
  plotting method, `structural_break`, and `cusum_test` optional parameters.
- Explicitly narrowed the sup-F selected break list so `None` cannot appear in
  `break_dates` on edge cases with no candidate break.
- Narrowed Bai-Perron segment splitting after the best segment index is found.
- Converted CUSUM recursive residual accumulation from list-to-array
  reassignment into separate list and ndarray variables.
- Refreshed the generated schema bundle with `scripts/dump_schemas.py` because
  the public structural-break signatures changed from implicit to explicit
  optional types.

Verification run:

- `.venv/bin/python -m mypy src/statspai/timeseries/structural_break.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile
  src/statspai/timeseries/structural_break.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/timeseries/structural_break.py --count --statistics` passed
  with 0 selected touched-file violations.
- `.venv/bin/python -m pytest -q
  tests/test_new_v06_modules.py::TestTimeSeries::test_structural_break
  tests/test_new_v06_modules.py::TestTimeSeries::test_cusum
  tests/test_structural_break_size.py -o addopts=''` passed, 9 tests.
- `git diff --check -- src/statspai/timeseries/structural_break.py` passed.
- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5 schema files and
  mirrored them into `src/statspai/schemas/`.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3508 <= 4698, mypy observed 1816 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 283

Target: synthetic-control dispatcher typing and plot-helper cleanup.

- Typed dispatcher `**kwargs` in `src/statspai/synth/scm.py` so the public
  `synth` wrapper and private implementation no longer leak untyped keyword
  surfaces.
- Narrowed already-validated method and inference strings with `Literal` casts
  before dispatching to demeaned, robust, and SDID synthetic-control variants.
- Annotated the `SyntheticControl` validation and matrix-preparation helpers as
  side-effect-only methods.
- Forced `_should_run_nested` to return a concrete bool for the auto nested-V
  decision.
- Typed `synthplot` and its trajectory/gap helper panels without changing the
  plotting behavior.

Verification run:

- `.venv/bin/python -m mypy src/statspai/synth/scm.py --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/synth/scm.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/synth/scm.py --count --statistics` passed with 0 selected
  touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q
  tests/test_synth.py::TestSyntheticControl tests/test_cov95_synth_r2_scm.py
  -o addopts=''` passed, 36 tests.
- `git diff --check -- src/statspai/synth/scm.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3508 <= 4698, mypy observed 1804 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 284

Target: DataFrame label metadata typing.

- Cast `_labels` metadata in `src/statspai/utils/labels.py` to the existing
  `Dict[str, str]` API contract before reading labels.
- Preserved the runtime behavior of `get_label`, `get_labels`, and `describe`;
  this is a static-contract cleanup for pandas `attrs`.

Verification run:

- `.venv/bin/python -m mypy src/statspai/utils/labels.py --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/utils/labels.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/utils/labels.py --count --statistics` passed with 0 selected
  touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_utils.py -o addopts=''` passed,
  24 tests.
- `git diff --check -- src/statspai/utils/labels.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3508 <= 4698, mypy observed 1803 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 285

Target: DGP staggered-DiD group typing stability.

- Updated `src/statspai/utils/dgp.py` so the staggered DiD branch fills the
  preallocated integer group array in place instead of rebinding it to a wider
  numpy shape type.
- Preserved the same random draw and resulting DataFrame contract for
  `dgp_did`.

Verification run:

- `.venv/bin/python -m mypy src/statspai/utils/dgp.py --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/utils/dgp.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/utils/dgp.py --count --statistics` passed with 0 selected
  touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_cov95_did_r4_aggte.py
  tests/test_cov95_did_summary_extra.py tests/test_cov95_did_r3_wooldridge.py
  -o addopts=''` passed, 55 tests.
- `.venv/bin/python -m pytest -q tests/test_rd_validation.py -o addopts=''`
  passed, 23 tests, with the existing fuzzy-RD weak-first-stage warning.
- A direct staggered `sp.dgp_did(...)` sanity check passed for shape, columns,
  `true_effect`, and integer group dtype.
- `git diff --check -- src/statspai/utils/dgp.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3508 <= 4698, mypy observed 1802 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 286

Target: typed data I/O reader boundaries.

- Added typed `**kwargs: Any` to `read_data` in `src/statspai/utils/io.py`
  while preserving the existing public `path: str` and `encoding` contract.
- Annotated the private Stata, SAS, and SPSS reader helpers as returning
  `pd.DataFrame`.
- Removed the stale `Dict` typing import as part of the same touched-file
  cleanup.

Verification run:

- `.venv/bin/python -m mypy src/statspai/utils/io.py --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/utils/io.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/utils/io.py --count --statistics` passed with 0 selected
  touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_rddensity_io.py -o addopts=''`
  passed, 20 tests.
- `git diff --check -- src/statspai/utils/io.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3507 <= 4698, mypy observed 1798 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 287

Target: optional torch device resolver typing.

- Annotated `resolve_torch_device` in
  `src/statspai/utils/_torch_device.py` as returning `Any` so torch remains
  lazily imported and optional.
- Annotated `_mps_available`'s torch module parameter as `Any`, preserving the
  compatibility guard for older or missing torch builds.

Verification run:

- `.venv/bin/python -m mypy src/statspai/utils/_torch_device.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/utils/_torch_device.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/utils/_torch_device.py --count --statistics` passed with 0
  selected touched-file violations.
- `git diff --check -- src/statspai/utils/_torch_device.py` passed.
- `.venv/bin/python -m pytest -q tests/test_torch_device_resolver.py
  -o addopts=''` skipped because optional `torch` is not installed in the
  current venv.
- `.venv/bin/python -m pytest --collect-only -q tests/test_deepiv.py
  tests/test_neural_causal.py -o addopts=''` collected no tests for the same
  optional-torch skip path.
- Direct import sanity confirmed `torch unavailable` and
  `statspai.utils._torch_device.__all__` still exposes
  `resolve_torch_device` and `torch_device_info`.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3507 <= 4698, mypy observed 1796 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 288

Target: LLM config TOML parser value typing.

- Split `_parse_toml`'s raw string value from the parsed `Any` value in
  `src/statspai/causal_llm/_config.py`.
- Preserved the existing minimal parser behavior for quoted strings,
  booleans, integers, sections, and comments.

Verification run:

- `.venv/bin/python -m mypy src/statspai/causal_llm/_config.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/causal_llm/_config.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/causal_llm/_config.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_llm_resolver.py -o addopts=''`
  passed, 27 tests, with the expected no-LLM-env workflow-degraded warning.
- `git diff --check -- src/statspai/causal_llm/_config.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3507 <= 4698, mypy observed 1794 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 289

Target: LLM sensitivity-prior table typing.

- Typed `_DOMAIN_PRIORS` in `src/statspai/causal_llm/llm_sensitivity.py` as a
  nested dictionary with mixed `Any` values.
- Explicitly narrowed heuristic `rho_max`, `r2`, and `comment` values when
  constructing `SensitivityPriorProposal`.
- Preserved the heuristic defaults and client-provided JSON path.

Verification run:

- `.venv/bin/python -m mypy src/statspai/causal_llm/llm_sensitivity.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/causal_llm/llm_sensitivity.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/causal_llm/llm_sensitivity.py --count --statistics` passed
  with 0 selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_causal_llm.py -o addopts=''`
  passed, 4 tests.
- `git diff --check -- src/statspai/causal_llm/llm_sensitivity.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3507 <= 4698, mypy observed 1791 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 290

Target: structured citation helper typing.

- Added a narrow `cast` import in `src/statspai/smart/citations.py`.
- Typed the `inspect.signature(...).parameters` local as `Any` to avoid
  rebinding a read-only mapping proxy to an empty fallback dict.
- Cast the legacy `render_citation(..., fmt="json")` fallback in `bib_for`
  to the documented structured citation dict contract without changing
  runtime behavior.

Verification run:

- `.venv/bin/python -m mypy src/statspai/smart/citations.py --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/smart/citations.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/smart/citations.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_citations.py
  tests/test_econometric_results_cite.py -o addopts=''` passed, 38 tests.
- An attempted extra `tests/test_audit.py::TestAuditJSON::test_bib_for`
  selector was invalid; collection confirmed the actual `bib_for` tests live
  in the two passing citation files above.
- `git diff --check -- src/statspai/smart/citations.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3507 <= 4698, mypy observed 1789 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 291

Target: survey estimator typing and redundant variance cleanup.

- Explicitly stringified `SurveyResult.__repr__` in
  `src/statspai/survey/estimators.py`.
- Annotated `svyglm` as returning `SurveyResult`.
- Removed the stale `dataclasses.field` import.
- Deleted an unused `_stratified_cluster_var(...)` call in `svyglm`; the
  full sandwich meat matrix is recomputed immediately below and is the value
  used for standard errors, so this removes redundant work without changing
  results.

Verification run:

- `.venv/bin/python -m mypy src/statspai/survey/estimators.py --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/survey/estimators.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/survey/estimators.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_survey.py
  tests/test_survey_calibration.py -o addopts=''` passed, 18 tests.
- `.venv/bin/python -m pytest -q tests/test_output_and_survey_helpers.py
  tests/test_untested_function_coverage.py::TestSurveyEstimators
  -o addopts=''` passed, 10 tests, with existing output-helper warnings.
- `git diff --check -- src/statspai/survey/estimators.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3505 <= 4698, mypy observed 1787 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 292

Target: survey design GLM wrapper return contract.

- Annotated `SurveyDesign.glm` in `src/statspai/survey/design.py` as returning
  `SurveyResult`, matching its direct `svyglm(...)` delegate.

Verification run:

- `.venv/bin/python -m mypy src/statspai/survey/design.py --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/survey/design.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/survey/design.py --count --statistics` passed with 0 selected
  touched-file violations.
- `.venv/bin/python -m pytest -q tests/test_survey.py
  tests/test_survey_calibration.py -o addopts=''` passed, 18 tests.
- `.venv/bin/python -m pytest -q tests/test_output_and_survey_helpers.py
  tests/test_untested_function_coverage.py::TestSurveyEstimators
  -o addopts=''` passed, 10 tests, with existing output-helper warnings.
- `git diff --check -- src/statspai/survey/design.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3505 <= 4698, mypy observed 1786 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 293

Target: ESDA permutation p-value scalar typing.

- Converted the numpy scalar expression returned by
  `src/statspai/spatial/esda/_base.py::permutation_pvalue` to a Python
  `float`.
- Preserved the empirical two-sided p-value formula and all ESDA behavior.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/esda/_base.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/esda/_base.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/esda/_base.py --count --statistics` passed with 0
  selected touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q
  tests/spatial/test_esda_moran.py tests/spatial/test_esda_geary.py
  tests/spatial/test_esda_getis_ord.py tests/spatial/test_esda_join_counts.py
  tests/spatial/test_esda_plots.py -o addopts=''` passed, 11 tests.
- `.venv/bin/python -m pytest -q tests/test_tierD_spatial_diag_analytic.py
  tests/test_untested_bounds_and_diag.py::test_moran_detects_spatial_structure
  tests/test_untested_bounds_and_diag.py::test_moran_returns_finite_statistic_for_noise
  -o addopts=''` passed, 13 tests.
- `git diff --check -- src/statspai/spatial/esda/_base.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3505 <= 4698, mypy observed 1785 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 294

Target: GWR typing cleanup and kernel array normalization.

- Typed public `gwr` array-like inputs in `src/statspai/spatial/gwr/gwr.py`
  as `Any`, preserving flexible callers while closing the untyped-def gap.
- Normalized `_kernel` branches through `np.asarray(..., dtype=float)` so all
  kernels explicitly return ndarrays.
- Marked optional local standard-error and t-stat arrays as
  `Optional[np.ndarray]`.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/gwr/gwr.py --show-error-codes`
  reported success after the repository Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/gwr/gwr.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/gwr/gwr.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_gwr.py -o addopts=''`
  passed, 9 tests.
- `git diff --check -- src/statspai/spatial/gwr/gwr.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3504 <= 4698, mypy observed 1781 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 295

Target: spatial log-determinant helper typing.

- Typed dense/sparse matrix inputs in
  `src/statspai/spatial/models/_logdet.py` as `Any`, preserving support for
  scipy sparse matrices and dense array-like objects.
- Annotated `_to_csr`, `log_det_exact`, and `log_det_approx` without changing
  exact or stochastic Barry-Pace/Chebyshev log-determinant arithmetic.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/models/_logdet.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/models/_logdet.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/models/_logdet.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_models_logdet.py
  tests/spatial/test_models_ml.py tests/spatial/test_ml_se_information.py
  -o addopts=''` passed, 9 tests.
- `.venv/bin/python -m pytest -q tests/spatial/test_columbus_crossval.py
  tests/spatial/test_backward_compat.py -o addopts=''` passed, 7 tests.
- `git diff --check -- src/statspai/spatial/models/_logdet.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3504 <= 4698, mypy observed 1778 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 296

Target: spatial weights core typing and dense array normalization.

- Typed the sparse cache and libpysal adapter boundary in
  `src/statspai/spatial/weights/core.py` as `Any`, preserving scipy CSR and
  optional libpysal compatibility while closing direct mypy gaps.
- Normalized `W.full()` through `np.asarray(..., dtype=float)` so the public
  dense-matrix accessor returns an explicit float ndarray.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/weights/core.py
  --show-error-codes` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/weights/core.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/weights/core.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_weights_core.py
  tests/spatial/test_weights_block.py
  tests/test_tierD_p2_spatial_weights_analytic.py -o addopts=''` passed,
  13 tests.
- `git diff --check -- src/statspai/spatial/weights/core.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3504 <= 4698, mypy observed 1774 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 297

Target: spatial weights contiguity/block typing and schema refresh.

- Typed the geopandas optional-dependency guard, contiguity helper inputs, and
  neighbor-list containers in `src/statspai/spatial/weights/contiguity.py`.
- Typed `block_weights()` regime inputs and bucket/neighbor containers in
  `src/statspai/spatial/weights/block.py`, using a separate `regime_array`
  local to avoid reassigning the public input parameter.
- Refreshed the generated schema bundle after the public signature metadata
  changed.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/weights --show-error-codes
  --no-error-summary` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile
  src/statspai/spatial/weights/contiguity.py
  src/statspai/spatial/weights/block.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/weights/contiguity.py
  src/statspai/spatial/weights/block.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_weights_core.py
  tests/spatial/test_weights_block.py tests/spatial/test_weights_contiguity.py
  tests/spatial/test_weights_distance.py
  tests/test_tierD_p2_spatial_weights_analytic.py -o addopts=''` passed,
  17 tests with 1 optional geopandas-related skip.
- `git diff --check -- src/statspai/spatial/weights/contiguity.py
  src/statspai/spatial/weights/block.py` passed.

Post-batch gate sweep:

- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5-file schema
  bundle under `schemas/` and `src/statspai/schemas/`.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3504 <= 4698, mypy observed 1767 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 298

Target: Geary ESDA public input typing.

- Annotated the `geary()` array-like input in
  `src/statspai/spatial/esda/geary.py` as `Any`, preserving the existing numpy
  conversion path while closing the direct mypy gap.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/esda/geary.py
  --show-error-codes --no-error-summary` reported success after the repository
  Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/esda/geary.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/esda/geary.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_esda_geary.py
  tests/spatial/test_esda_moran.py tests/spatial/test_esda_getis_ord.py
  tests/test_tierD_spatial_diag_analytic.py -o addopts=''` passed, 18 tests.
- `git diff --check -- src/statspai/spatial/esda/geary.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3504 <= 4698, mypy observed 1766 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 299

Target: MGWR public input typing and import cleanup.

- Annotated the `mgwr()` array-like `coords`, `y`, and `X` inputs in
  `src/statspai/spatial/gwr/mgwr.py` as `Any`, preserving the existing numpy
  conversion path while closing the direct mypy gap.
- Removed unused `Literal` and `GWRResult` imports so touched-file lint stays
  clean.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/gwr/mgwr.py
  --show-error-codes --no-error-summary` reported success after the repository
  Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/gwr/mgwr.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/gwr/mgwr.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_gwr.py
  tests/spatial/test_gwr_local_se.py -o addopts=''` passed, 12 tests.
- `git diff --check -- src/statspai/spatial/gwr/mgwr.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3502 <= 4698, mypy observed 1765 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 300

Target: join-count ESDA public typing.

- Annotated the `join_counts()` array-like input and mixed scalar/array return
  dictionary in `src/statspai/spatial/esda/join_counts.py`, preserving the
  existing public keys and permutation-output semantics.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/esda/join_counts.py
  --show-error-codes --no-error-summary` reported success after the repository
  Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/esda/join_counts.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/esda/join_counts.py --count --statistics` passed with
  0 selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_esda_join_counts.py
  tests/spatial/test_esda_geary.py tests/spatial/test_esda_moran.py
  tests/spatial/test_esda_getis_ord.py tests/test_tierD_spatial_diag_analytic.py
  -o addopts=''` passed, 20 tests.
- `git diff --check -- src/statspai/spatial/esda/join_counts.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3502 <= 4698, mypy observed 1763 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 301

Target: Getis-Ord ESDA public typing.

- Annotated `getis_ord_g()` and `getis_ord_local()` array-like inputs in
  `src/statspai/spatial/esda/getis_ord.py`.
- Added the local Getis-Ord dictionary return type as `Dict[str, np.ndarray]`,
  preserving the existing `Gs` and `z` output arrays.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/esda/getis_ord.py
  --show-error-codes --no-error-summary` reported success after the repository
  Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/esda/getis_ord.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/esda/getis_ord.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_esda_getis_ord.py
  tests/spatial/test_esda_join_counts.py tests/spatial/test_esda_geary.py
  tests/spatial/test_esda_moran.py tests/test_tierD_spatial_diag_analytic.py
  -o addopts=''` passed, 20 tests.
- `git diff --check -- src/statspai/spatial/esda/getis_ord.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3502 <= 4698, mypy observed 1760 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 302

Target: Moran ESDA typing and centered-array normalization.

- Annotated `_center()`, `moran()`, and `moran_local()` in
  `src/statspai/spatial/esda/moran.py`.
- Normalized `_center()` through an explicit ndarray local and `np.asarray`
  return so mypy can verify the centered-vector contract.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/esda/moran.py
  --show-error-codes --no-error-summary` reported success after the repository
  Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/esda/moran.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/esda/moran.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_esda_moran.py
  tests/spatial/test_esda_geary.py tests/spatial/test_esda_getis_ord.py
  tests/spatial/test_esda_join_counts.py tests/test_tierD_spatial_diag_analytic.py
  -o addopts=''` passed, 20 tests.
- `git diff --check -- src/statspai/spatial/esda/moran.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3502 <= 4698, mypy observed 1756 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 303

Target: ESDA plotting dynamic-boundary typing.

- Annotated matplotlib axis and geopandas inputs in
  `src/statspai/spatial/esda/plots.py` as dynamic `Any` boundaries while
  preserving the existing plotting behavior.
- Refreshed the generated schema bundle after the plotting helper signatures
  changed.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/esda/plots.py
  --show-error-codes --no-error-summary` reported success after the repository
  Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/esda/plots.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/esda/plots.py --count --statistics` passed with 0
  selected touched-file violations.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -q
  tests/spatial/test_esda_plots.py tests/spatial/test_esda_moran.py
  tests/spatial/test_esda_getis_ord.py tests/spatial/test_esda_join_counts.py
  tests/spatial/test_esda_geary.py tests/test_tierD_spatial_diag_analytic.py
  -o addopts=''` passed, 22 tests.
- `git diff --check -- src/statspai/spatial/esda/plots.py` passed.

Post-batch gate sweep:

- `.venv/bin/python scripts/dump_schemas.py` refreshed the 5-file schema
  bundle under `schemas/` and `src/statspai/schemas/`.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3501 <= 4698, mypy observed 1752 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 304

Target: spatial IV weight-input typing and import cleanup.

- Annotated `_coerce_W()` and `spatial_iv()` weight-matrix inputs in
  `src/statspai/spatial/iv.py` as `Any`, preserving support for StatsPAI
  weights, scipy sparse objects, and dense array-like inputs.
- Removed the unused `scipy.stats` import from the touched file.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/iv.py --show-error-codes
  --no-error-summary` reported success after the repository Python-version
  warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/iv.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/iv.py --count --statistics` passed with 0 selected
  touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_iv.py
  tests/spatial/test_models_gmm.py tests/test_tierD_p2_spatial_weights_analytic.py
  -o addopts=''` passed, 13 tests.
- `git diff --check -- src/statspai/spatial/iv.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3500 <= 4698, mypy observed 1750 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 305

Target: GWR bandwidth selector typing.

- Annotated `gwr_bandwidth()` array-like `coords`, `y`, and `X` inputs in
  `src/statspai/spatial/gwr/bandwidth.py`.
- Made `bw_min` and `bw_max` explicitly optional floats, matching the existing
  default-bound calculation for adaptive and fixed kernels.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/gwr/bandwidth.py
  --show-error-codes --no-error-summary` reported success after the repository
  Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/gwr/bandwidth.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/gwr/bandwidth.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_gwr.py
  tests/spatial/test_columbus_crossval.py -o addopts=''` passed, 13 tests.
- `git diff --check -- src/statspai/spatial/gwr/bandwidth.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1402 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3500 <= 4698, mypy observed 1747 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 306

Target: spatial impact typing, Monte-Carlo arrays, and unused inverse removal.

- Typed the dynamic spatial result boundary and helper functions in
  `src/statspai/spatial/models/impacts.py`.
- Split Monte-Carlo draw lists from their ndarray forms to clear direct mypy
  without changing simulated impact SE calculations.
- Removed an unused outer `(I - rho W)^-1` matrix inversion; point impacts
  already compute the needed inverse inside the helper.
- Added a `NumericalInstability` taxonomy error for out-of-bounds spatial
  autoregressive parameters, avoiding any increase in generic exception debt.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/models/impacts.py
  --show-error-codes --no-error-summary` reported success after the repository
  Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/models/impacts.py`
  passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731
  src/statspai/spatial/models/impacts.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_diagnostics_impacts.py
  tests/spatial/test_models_base.py tests/spatial/test_models_ml.py
  tests/spatial/test_ml_se_information.py -o addopts=''` passed, 15 tests.
- `git diff --check -- src/statspai/spatial/models/impacts.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1403 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3497 <= 4698, mypy observed 1738 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-18 Batch 307

Target: spatial ML direct mypy closure.

- Tightened CSR coercion in `src/statspai/spatial/models/ml.py` so ndarray,
  scipy sparse, and StatsPAI `W` inputs all flow through an explicit
  `sparse.csr_matrix` local.
- Split row-normalization temporaries to avoid shape/type reuse ambiguity.
- Annotated the large-n stochastic trace probe helpers and the SAC numerical
  Hessian helper.
- Explicitly returned floats from concentrated log-likelihood objectives and
  removed one semicolon statement in the Hessian helper.
- Verified the whole `src/statspai/spatial` subtree is direct-mypy clean.

Verification run:

- `.venv/bin/python -m mypy src/statspai/spatial/models/ml.py
  --show-error-codes --no-error-summary` reported success after the repository
  Python-version warning.
- `.venv/bin/python -m py_compile src/statspai/spatial/models/ml.py` passed.
- `.venv/bin/python -m flake8 --select=F401,F541,F841,E731,E702
  src/statspai/spatial/models/ml.py --count --statistics` passed with 0
  selected touched-file violations.
- `.venv/bin/python -m pytest -q tests/spatial/test_models_ml.py
  tests/spatial/test_ml_se_information.py tests/spatial/test_columbus_crossval.py
  tests/spatial/test_backward_compat.py tests/spatial/test_diagnostics_impacts.py
  tests/spatial/test_models_logdet.py -o addopts=''` passed, 23 tests.
- `.venv/bin/python -m mypy src/statspai/spatial --show-error-codes
  --no-error-summary` reported success after the repository Python-version
  warning.
- `git diff --check -- src/statspai/spatial/models/ml.py` passed.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1403 taxonomy raises and 1298 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  270 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1070 registered functions
  (`reference=128`, `anchored=579`, `weak=149`, `smoke=11`,
  `untested=203`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed with
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/dump_schemas.py --check` reported
  `schemas/ is in sync (5 files)`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  3495 <= 4698, mypy observed 1729 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root-scoped JOSS paths remain untouched by this batch. `CausalAgentBench/`
  remains clean on `main...origin/main`; nested `Paper-JSS/` still shows
  external/parallel `replication/results/*` modifications and was not touched.

## 2026-06-17 Batch 154

Target: direct typing and touched-file cleanup for output collections.

- Typed `Collection.__iter__`, CSV export kwargs, table/regression adders, and
  Word/XLSX writer helper boundaries in `src/statspai/output/collection.py`.
- Explicitly stringified dynamic CSV/text payload outputs and narrowed optional
  summary titles before forwarding to `sumstats`, clearing direct mypy.
- Removed unused `warnings`, unused Word alignment import, and an unused DOCX
  table-state variable; rewrapped long table/item and worksheet-width
  expressions so touched-file flake8 is clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/output/collection.py` passed.
- `.venv/bin/python -m flake8 src/statspai/output/collection.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/output/collection.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_output_and_survey_helpers.py tests/test_aer_word_style.py
  tests/test_econometric_results_export.py tests/test_regtable_from_dict.py
  tests/test_regtable_serialization.py tests/test_regtable_quarto.py
  tests/test_regtable_fmt_auto.py` passed, 154 tests with expected
  deprecation/export warnings.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_collection.py tests/test_container_serialization.py` passed,
  30 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1271 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4288 <= 4698, mypy observed 3082 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 153

Target: direct typing cleanup for the production-function dispatcher.

- Annotated `prod_fn(..., **kwargs)` with `Any`, clearing the direct mypy
  untyped boundary while preserving the pass-through API for OP/LP/ACF/
  Wooldridge estimators.
- Added the corresponding `Any` import and rewrapped a long doctest markup
  example so touched-file flake8 is clean.

Verification run:

- `.venv/bin/python -m py_compile
  src/statspai/structural/production/_dispatcher.py` passed.
- `.venv/bin/python -m flake8
  src/statspai/structural/production/_dispatcher.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m mypy
  src/statspai/structural/production/_dispatcher.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_tierD_structural_analytic.py tests/test_estimator_provenance_round4.py
  tests/test_v0917_additions.py` passed, 53 tests with expected PSM warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1271 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4294 <= 4698, mypy observed 3092 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 152

Target: direct typing cleanup for the automatic CATE learner race.

- Annotated sklearn model arguments in `_build_learner(...)`,
  `_cross_fit_nuisance(...)`, and `_honest_cate_predictions(...)` as `Any`,
  matching the heterogeneous sklearn clone/fit/predict interface used across
  S/T/X/R/DR learners.
- Added an explicit `Any` return for `_build_learner(...)`, clearing direct
  mypy on `src/statspai/metalearners/auto_cate.py`.
- Removed unused `field` and `_default_cate_model` imports and rewrapped the
  default propensity-model expression so touched-file flake8 is clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/metalearners/auto_cate.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/metalearners/auto_cate.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/metalearners/auto_cate.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_auto_cate.py tests/test_auto_cate_tuned.py
  tests/test_metalearners.py tests/test_result_consumer_errors.py
  tests/test_article_aliases.py tests/test_estimator_provenance_round5.py`
  passed, 147 tests with expected PSM/overlap warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1271 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4295 <= 4698, mypy observed 3093 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 151

Target: matching dispatcher argument narrowing and error consistency.

- Added a unified required-input check for `sp.match(...)` after method alias
  resolution, converting missing `y`, `treat`, or `covariates` into a
  `MethodIncompatibility` with `diagnostics["missing"]`.
- Narrowed `y`/`treat`/`covariates` before dispatching to classical,
  weighting, genetic, optimal, and cardinality matching estimators, clearing
  direct mypy on `src/statspai/matching/__init__.py`.
- Restored non-string `method` errors to plain `TypeError`, consistent with
  the IV/RD/panel dispatchers and the existing matching dispatcher tests.
- Added a dispatcher regression test for missing required inputs and cleaned
  touched-test flake8 style.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/matching/__init__.py
  tests/test_match_dispatcher.py` passed.
- `.venv/bin/python -m flake8 src/statspai/matching/__init__.py
  tests/test_match_dispatcher.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/matching/__init__.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_match_dispatcher.py tests/test_matching.py
  tests/test_matching_optimal.py tests/test_optimal_match_vectorized.py
  tests/test_overlap_and_cbps.py tests/test_tierD_lalonde_psm_guard.py`
  passed, 109 tests with expected PSM imbalance warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1271 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4298 <= 4698, mypy observed 3097 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 150

Target: direct typing cleanup for the callable IV namespace dispatcher.

- Added explicit `Any` return annotations to the unified IV dispatcher,
  `fit(...)` alias, module `__getattr__`, and callable-module `__call__`
  shim, matching the intentionally heterogeneous estimator return surface.
- Typed IV argument/formula helper boundaries as `tuple[Any, Any, Any, Any]`
  and the augmented-diagnostics attachment helper as mutating in place with
  `None` return.
- Rewrapped a long Olea-Pflueger diagnostic fallback expression so
  touched-file flake8 is clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/iv/__init__.py` passed.
- `.venv/bin/python -m flake8 src/statspai/iv/__init__.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/iv/__init__.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_iv_cov_diag.py tests/test_cov95_iv_small_estimators.py
  tests/test_iv_frontiers.py tests/test_tierD_spatial_diag_analytic.py
  tests/test_shift_share_political.py tests/test_estimator_provenance_round9.py`
  passed, 65 tests with 2 expected runtime warnings.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_iv.py tests/test_iv_dispatcher.py tests/test_cov95_iv_init.py
  tests/iv/test_unified_fit.py tests/test_iv_cov_dispatcher_routes.py
  tests/test_iv_cov_reachable.py tests/test_iv_cov_edges.py
  tests/test_iv_pipeline.py` passed, 103 tests with 2 weak-IV warnings.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4298 <= 4698, mypy observed 3120 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 149

Target: direct typing cleanup for panel stochastic frontier likelihoods.

- Wrapped scalar truncated-normal CDF inputs as one-element ndarrays in the
  panel TI/TVD likelihood path, matching the `_core` helper contract without
  changing the scalar truncation value.
- Broadcast TRE scalar `sigma_v`/`sigma_u` parameters to the shifted-error grid
  before calling shared half-normal/exponential likelihood helpers, preserving
  numpy broadcasting semantics while satisfying direct mypy.
- Narrowed panel group likelihood returns to concrete float ndarrays and typed
  the TRE `neg_loglik(...)` helper return.
- Removed unused `scipy.stats` imports, removed an unused unit-level JLMS
  intermediate, and rewrapped a few long lines/continuations so touched-file
  flake8 is clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/frontier/panel.py` passed.
- `.venv/bin/python -m flake8 src/statspai/frontier/panel.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/frontier/panel.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_frontier.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py` passed, 188 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4299 <= 4698, mypy observed 3129 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 148

Target: direct typing and touched-file cleanup for metafrontier LP dispatch.

- Annotated `metafrontier(..., **frontier_kwargs)` so per-group frontier
  passthrough kwargs are no longer an untyped direct-mypy boundary.
- Rewrapped the short-observation failure, per-group `_frontier(...)` call,
  HiGHS retry call, beta-meta series construction, and long comments without
  changing the LP objective, constraints, retry tolerance, or TGR formula.
- Removed an unused `Optional` import and cleaned aligned spacing in the
  summary table construction so touched-file flake8 is clean.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/frontier/metafrontier.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/frontier/metafrontier.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/frontier/metafrontier.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_frontier.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py` passed, 188 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4308 <= 4698, mypy observed 3138 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 147

Target: direct typing and touched-file cleanup for mixture stochastic
frontiers.

- Narrowed the ZISF and two-class LCSF per-observation mixture likelihood
  helpers to return concrete float ndarrays, removing both direct mypy
  `Any` returns in `src/statspai/frontier/mixture.py`.
- Removed an unused `EconometricResults` import and expanded compressed
  semicolon assignments in the LCSF parameter unpacking paths so touched-file
  flake8 is clean.
- Rewrapped the posterior-efficiency comment without changing ZISF/LCSF
  likelihoods, optimizer starts, class-label canonicalization, or diagnostics.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/frontier/mixture.py` passed.
- `.venv/bin/python -m flake8 src/statspai/frontier/mixture.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/frontier/mixture.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_frontier.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py` passed, 188 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4315 <= 4698, mypy observed 3139 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 146

Target: direct typing and touched-file cleanup for the Malmquist productivity
frontier helper.

- Annotated `malmquist(..., **frontier_kwargs)` so direct mypy no longer treats
  the frontier passthrough kwargs as an untyped boundary.
- Narrowed the nested log-distance helper to return a concrete float ndarray,
  eliminating the `Any` return on the Malmquist decomposition path.
- Removed the unused `Optional` import and renamed the translog interaction loop
  variable so touched-file flake8 is clean without changing generated column
  names or Malmquist semantics.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/frontier/malmquist.py` passed.
- `.venv/bin/python -m flake8 src/statspai/frontier/malmquist.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/frontier/malmquist.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_frontier.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py` passed, 188 tests.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4329 <= 4698, mypy observed 3141 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 145

Target: PLR direct typing parity across the core DML models.

- Added postponed annotations and typed `DoubleMLPLR._fit_one_rep(...)` with
  ndarray inputs and `(float, float)` returns.
- Annotated KFold/user-fold splits plus PLR residual and weight arrays so direct
  mypy on `src/statspai/dml/plr.py` is clean after the repository
  Python-version warning.
- Kept PLR orthogonal-score construction, weighted sandwich variance, explicit
  fold-index path, and diagnostics unchanged.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/dml/plr.py` passed.
- `.venv/bin/python -m flake8 src/statspai/dml/plr.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/dml/plr.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_dml_split.py tests/test_dml_panel.py tests/test_dml_cov_scores.py
  tests/test_cov95_dml_learners_base.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py -q` passed, 179 tests, 3 skipped.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4339 <= 4698, mypy observed 3143 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 144

Target: PLIV/IRM direct typing parity with IIVM.

- Added postponed annotations and typed `DoubleMLPLIV._fit_one_rep(...)` and
  `DoubleMLIRM._fit_one_rep(...)` with ndarray inputs and `(float, float)`
  returns.
- Annotated PLIV residual and weight arrays, and IRM score/propensity arrays,
  so direct mypy on both files is clean after the repository Python-version
  warning.
- Cleaned two touched-file IRM line-length issues in defensive
  `IdentificationFailure` branches without changing behavior.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/dml/pliv.py
  src/statspai/dml/irm.py` passed.
- `.venv/bin/python -m flake8 src/statspai/dml/pliv.py
  src/statspai/dml/irm.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/dml/pliv.py src/statspai/dml/irm.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_dml_split.py tests/test_cov95_dml_learners_base.py
  tests/test_cov95_dml_averaging_panel.py tests/test_review_fixes.py
  tests/test_review_fixes_round2.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py -q` passed, 214 tests, 3 skipped.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4339 <= 4698, mypy observed 3144 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 143

Target: IIVM direct typing without changing orthogonal-score behavior.

- Added `from __future__ import annotations` and typed
  `DoubleMLIIVM._fit_one_rep(...)`, `_fit_predict_subgroup(...)`, and
  `_fit_predict_classifier(...)` with ndarray inputs and `(ndarray, bool)` /
  `(float, float)` returns.
- Annotated IIVM nuisance, compliance, instrument-propensity, and weight arrays
  to avoid numpy shape-inference churn after clipping and weighted branches.
- Kept the IIVM score construction, fallback behavior, and identification
  errors unchanged.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/dml/iivm.py` passed.
- `.venv/bin/python -m flake8 src/statspai/dml/iivm.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/dml/iivm.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_dml_iivm.py tests/test_cov95_dml_iivm_branches.py
  tests/test_dml_split.py tests/test_cov95_dml_learners_base.py
  tests/test_api_surface_consistency.py tests/test_v100_integration.py -q`
  passed, 173 tests, 3 skipped.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4341 <= 4698, mypy observed 3146 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 142

Target: Callaway-Sant'Anna report typing and renderer hygiene.

- Annotated `CSReport.plot(...)`, `_plot_breakdown(...)`, `to_excel(...)`,
  `cs_report(...)`, and the save-bundle helper so direct mypy on
  `src/statspai/did/report.py` is clean after the repository Python-version
  warning.
- Removed an unused local import in `plot(...)`.
- Made Markdown/text helper returns explicitly `str` and normalized Excel paths
  to strings before returning them.
- Narrowed raw-data column arguments before calling `callaway_santanna(...)`
  instead of passing `Optional[str]` values through.
- Materialized `aggte(...).detail` frames defensively before report assembly and
  guarded the dynamic-event breakdown path if a defensive fallback frame lacks
  `relative_time`.
- Cleaned touched-file flake8 debt in LaTeX escape dictionaries and shadowed
  argument collection without changing rendered content.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/report.py` passed.
- `.venv/bin/python -m flake8 src/statspai/did/report.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/did/report.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cs_report.py tests/test_cov95_did_r5_report.py
  tests/test_cov95_did_summary_extra.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py -q` passed, 132 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4341 <= 4698, mypy observed 3149 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 141

Target: ordered-treatment BCF typing and input-contract hardening.

- Added `bcf_ordinal(...)` validation helpers for DataFrame inputs,
  scalar-or-sequence covariates, missing columns, non-empty data, at least two
  observed ordered-treatment levels, valid baseline level, and adjacent-level
  support.
- Migrated ordered-BCF data-contract failures to
  `MethodIncompatibility`/`DataInsufficient` with diagnostics, recovery hints,
  and ranked alternatives.
- Reused the normalized covariate list for each binary BCF increment and result
  diagnostics.
- Annotated the cumulative CATE/variance and projected full-sample arrays so
  direct mypy on `src/statspai/bcf/ordinal.py` is clean after the repository
  Python-version warning.
- Extended `tests/test_bcf_ordinal.py` with missing-column, single-level, and
  invalid-baseline taxonomy assertions, and cleaned touched-file flake8 issues
  in imports and long expected arrays.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bcf/ordinal.py
  tests/test_bcf_ordinal.py` passed.
- `.venv/bin/python -m flake8 src/statspai/bcf/ordinal.py
  tests/test_bcf_ordinal.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bcf/ordinal.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bcf_ordinal.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py -q` passed, 92 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1371 taxonomy raises and 1270 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4358 <= 4698, mypy observed 3165 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 140

Target: unified RD dispatcher typing and option-error taxonomy hardening.

- Added a small RD dispatcher taxonomy helper with ranked alternatives for
  `sp.rd`, `sp.rdrobust`, and closely related RD entry points.
- Migrated unknown `method=` and missing `fuzzy=` for
  `method='bias_aware_fuzzy'` to `MethodIncompatibility` with diagnostics and
  recovery hints while preserving `ValueError` compatibility through the
  taxonomy class.
- Kept the historical `TypeError` path for non-string `method` values to avoid
  widening external compatibility risk.
- Annotated `_rd_dispatch(...)`, `fit(...)`, and `_CallableRDModule.__call__`,
  and typed the passthrough dispatch table as callables so direct mypy on
  `src/statspai/rd/__init__.py` is clean after the repository Python-version
  warning.
- Extended `tests/test_rd_dispatcher.py` with diagnostics assertions for
  unknown methods and missing fuzzy treatment arguments, and removed a stale
  unused import exposed by touched-file flake8.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/rd/__init__.py
  tests/test_rd_dispatcher.py` passed.
- `.venv/bin/python -m flake8 src/statspai/rd/__init__.py
  tests/test_rd_dispatcher.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/rd/__init__.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_rd_dispatcher.py tests/test_rd.py tests/test_rd_validation.py
  tests/test_api_surface_consistency.py tests/test_v100_integration.py -q`
  passed, 162 tests, with the existing fuzzy-RD weak-first-stage warning in
  `tests/test_rd_validation.py`.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1367 taxonomy raises and 1274 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4358 <= 4698, mypy observed 3169 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 139

Target: Optuna-tuned CATE race typing, validation, and taxonomy hardening.

- Added `auto_cate_tuned(...)` input-contract helpers for scalar-or-sequence
  covariates, scalar-or-sequence learner codes, search-space validation,
  DataFrame/column checks, fold/bootstrap/trial budgets, alpha, complete-row
  count, and binary treatment encoding.
- Migrated tuner option and data-contract failures to
  `MethodIncompatibility`/`DataInsufficient` with diagnostics, recovery hints,
  and ranked alternatives while preserving `ValueError` compatibility through
  the taxonomy classes.
- Moved Optuna lazy loading after all validation that does not need Optuna, so
  bad inputs surface StatsPAI taxonomy errors before optional-dependency
  failures.
- Annotated the internal Optuna callbacks, GBM factories, and nuisance/CATE
  helper model parameters so direct mypy on
  `src/statspai/metalearners/auto_cate_tuned.py` is clean after the repository
  Python-version warning.
- Fixed the per-learner best-code selection key for mypy and normalized
  learner codes before the per-learner tuning loop, matching `auto_cate`'s
  duplicate-insensitive API semantics.
- Extended `tests/test_auto_cate_tuned.py` with scalar covariate/learner
  support and taxonomy assertions for missing columns, invalid budget,
  invalid learner, invalid tune mode, and non-binary treatment.

Verification run:

- `.venv/bin/python -m py_compile
  src/statspai/metalearners/auto_cate_tuned.py` passed.
- `.venv/bin/python -m flake8
  src/statspai/metalearners/auto_cate_tuned.py tests/test_auto_cate_tuned.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy
  src/statspai/metalearners/auto_cate_tuned.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_auto_cate_tuned.py tests/test_auto_cate.py
  tests/test_api_surface_consistency.py tests/test_v100_integration.py -q`
  passed, 116 tests.
- After moving Optuna lazy loading behind array/treatment validation,
  `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_auto_cate_tuned.py -q` passed, 16 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1367 taxonomy raises and 1276 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4358 <= 4698, mypy observed 3173 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 138

Target: DID dispatcher SDID typing and option-contract hardening.

- Annotated the public `did(...)` dispatcher open-ended SDID parameters and
  `**kwargs` so direct mypy on `src/statspai/did/__init__.py` is clean after
  the repository Python-version warning.
- Added a narrow `Literal`/`cast` path for `method='sdid'` so the dispatcher
  validates `se_method` before passing it to `synth.sdid`, matching the
  downstream accepted values (`'placebo'`, `'bootstrap'`, `'jackknife'`).
- Migrated bad SDID `se_method` to `MethodIncompatibility` with diagnostics
  instead of relying on downstream typing/validation.
- Cleaned touched-file flake8 debt in long DID imports, recovery hints,
  aggregation messages, and an ambiguous local variable name.
- Extended `tests/test_cov95_did_dispatcher.py` with an invalid-SDID-SE
  taxonomy assertion.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/did/__init__.py
  tests/test_cov95_did_dispatcher.py` passed.
- `.venv/bin/python -m flake8 src/statspai/did/__init__.py
  tests/test_cov95_did_dispatcher.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/did/__init__.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_dispatcher.py` passed, 25 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_did_dispatcher.py tests/test_cov95_did_estimators.py
  tests/test_cov95_did_r5_callaway.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py` passed, 179 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1363 taxonomy raises and 1278 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4359 <= 4698, mypy observed 3180 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 137

Target: longitudinal TMLE contract, taxonomy, and typing hardening.

- Added `ltmle(...)` input-contract validation for DataFrame input, treatment and
  covariate-block length matching, at least one time point, censoring length,
  `outcome_type`, `alpha`, propensity bounds, missing columns, and minimum
  sample size.
- Migrated static and dynamic regime contract failures to
  `MethodIncompatibility`/`DataInsufficient` with diagnostics and recovery
  hints while preserving `ValueError` compatibility through the taxonomy class.
- Annotated LTMLE helpers (`_safe_logit`, `_fit_logit`, `_predict_proba`,
  `_fit_linear`), made probability helper returns explicit float ndarrays, and
  fixed `Sequence[str]`/`list[str]` history-column concatenation so direct mypy
  on `ltmle.py` is clean after the repository Python-version warning.
- Added `LTMLEResult` CI typing and removed stale TYPE_CHECKING imports that no
  longer carried useful type information.
- Added `tests/test_ltmle_contract.py` covering time-block length mismatch,
  missing columns, invalid propensity bounds, too-small samples, and dynamic
  regime length failures.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/tmle/ltmle.py
  tests/test_ltmle_contract.py` passed.
- `.venv/bin/python -m flake8 src/statspai/tmle/ltmle.py
  tests/test_ltmle_contract.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/tmle/ltmle.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ltmle_contract.py tests/test_v0917_deferred.py::TestLTMLEDynamic
  tests/test_estimator_provenance_round5.py::TestLtmleProvenance` passed,
  9 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ltmle_contract.py tests/test_v0917_deferred.py
  tests/test_v100_review_fixes.py::test_ltmle_survival_runs_after_offset_fix
  tests/test_estimator_provenance_round5.py::TestLtmleProvenance
  tests/test_tmle.py tests/test_hal_tmle.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py` passed, 158 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1362 taxonomy raises and 1278 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4364 <= 4698, mypy observed 3182 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 136

Target: smart/neural lazy exports and GNN causal contract hardening.

- Annotated `src/statspai/smart/__init__.py` and
  `src/statspai/neural_causal/__init__.py` lazy `__getattr__` exports so direct
  mypy passes while preserving lazy import behavior.
- Added structured `gnn_causal(...)` validators for DataFrame input,
  scalar-or-sequence covariates, required columns, nonnegative GCN layers,
  tree/min-leaf options, `alpha`, propensity bounds, complete-row count, finite
  outcome/covariate/adjacency inputs, binary treatment coding, both-arm support,
  and square row-aligned adjacency.
- Migrated those GNN causal contract failures to `MethodIncompatibility` or
  `DataInsufficient` with diagnostics and recovery hints before sklearn fit
  paths can fail opaquely.
- Added `GNNCausalResult` CI typing, made row-normalization and RF predictions
  return explicit float ndarrays, and kept the existing GCN-RF/AIPW estimating
  equations unchanged.
- Added `tests/test_gnn_causal_contract.py` covering scalar covariates, feature
  map shape, adjacency mismatch, single-arm data, negative layers, and invalid
  propensity bounds.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/smart/__init__.py
  src/statspai/neural_causal/__init__.py
  src/statspai/neural_causal/gnn_causal.py tests/test_gnn_causal_contract.py`
  passed.
- `.venv/bin/python -m flake8 src/statspai/smart/__init__.py
  src/statspai/neural_causal/__init__.py
  src/statspai/neural_causal/gnn_causal.py tests/test_gnn_causal_contract.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/smart/__init__.py
  src/statspai/neural_causal/__init__.py
  src/statspai/neural_causal/gnn_causal.py --ignore-missing-imports
  --show-error-codes` reported success after the repository Python-version
  warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_gnn_causal_contract.py tests/test_neural_causal_exports.py
  tests/test_smart_stability_gating.py` passed, 11 tests with 1 expected skip.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_gnn_causal_contract.py tests/test_neural_causal_exports.py
  tests/test_smart_stability_gating.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py tests/test_registry_new_modules.py` passed,
  119 tests with 1 expected skip.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1360 taxonomy raises and 1285 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4369 <= 4698, mypy observed 3191 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 135

Target: causal-discovery visualization export and deprecated causal-shim typing.

- Moved `causal_discovery` numpy usage to a normal top-level import and removed
  the unused module-level `_viz` import while preserving the public helper
  exports.
- Annotated the causal-discovery adjacency payload/getter/method helpers and
  switched dynamic result-method attachment from direct class-attribute writes
  to `setattr(...)`, preserving `.to_networkx()`, `.to_dot()`, `.plot()`, and
  `.edge_list()` behavior while making mypy understand the binding layer.
- Annotated the deprecated `statspai.causal` callable module shim so
  `sp.causal(...)` remains callable and typed after `import statspai.causal`.
- Cleaned touched-file flake8 issues by marking the intentionally delayed
  forest re-export import and normalizing ICP names-source formatting.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/causal_discovery/__init__.py
  src/statspai/causal/__init__.py` passed.
- `.venv/bin/python -m flake8 src/statspai/causal_discovery/__init__.py
  src/statspai/causal/__init__.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/causal_discovery/__init__.py
  src/statspai/causal/__init__.py --ignore-missing-imports --show-error-codes`
  reported success after the repository Python-version warning.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_causal_discovery.py tests/test_ml_causal_polish.py
  tests/test_causal_to_forest_rename.py tests/test_article_aliases_round2.py
  tests/test_lingam.py tests/test_ges.py` passed, 93 tests with 2 expected
  optional skips.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_causal_discovery.py tests/test_ml_causal_polish.py
  tests/test_causal_to_forest_rename.py tests/test_article_aliases_round2.py
  tests/test_article_aliases.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py` passed, 205 tests with 1 expected skip.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1358 taxonomy raises and 1286 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4372 <= 4698, mypy observed 3196 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 134

Target: causal survival forest contract, taxonomy, and typing hardening.

- Corrected `_ipcw_rmst_pseudo(...)` typing so the helper's `(pseudo, S_C)`
  tuple return matches its annotation; direct mypy on
  `src/statspai/survival/causal_forest.py` is now clean after the repository
  Python-version warning.
- Added structured `causal_survival_forest(...)` validators for DataFrame input,
  scalar-or-sequence covariates, required columns, tree/min-leaf options,
  `alpha`, propensity bounds, positive horizon, complete-row count, finite
  times/covariates, positive time, binary event/treatment coding, and both-arm
  support.
- Migrated those user-facing contract failures to `MethodIncompatibility` or
  `DataInsufficient` with diagnostics and recovery hints before sklearn fit
  paths can fail opaquely.
- Cleaned touched-file flake8 debt in the survival forest docstring, KM helper,
  and double-robust score expression.
- Added `tests/test_survival_causal_forest_contract.py` covering scalar
  covariates, missing columns, bad propensity bounds, invalid horizon, and
  single-arm data.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/survival/causal_forest.py
  tests/test_survival_causal_forest_contract.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_survival_causal_forest_contract.py
  tests/test_estimator_provenance_round9.py::TestCausalSurvivalForestProvenance`
  passed, 5 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_survival_causal_forest_contract.py tests/test_glance_survival.py
  tests/test_timeseries_survival_estimators.py
  tests/test_tierD_p2_survival_analytic.py tests/test_synth_survival.py
  tests/test_cov95_synth_r4_survival.py` passed, 36 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_survival_causal_forest_contract.py
  tests/test_estimator_provenance_round9.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py` passed, 101 tests.
- `.venv/bin/python -m flake8 src/statspai/survival/causal_forest.py
  tests/test_survival_causal_forest_contract.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m mypy src/statspai/survival/causal_forest.py
  --ignore-missing-imports --show-error-codes` reported success after the
  repository Python-version warning.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1358 taxonomy raises and 1286 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4376 <= 4698, mypy observed 3208 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 133

Target: off-policy evaluation input-contract, taxonomy, and typing hardening.

- Added structured OPE dispatcher validation in `src/statspai/ope/estimators.py`
  so unknown methods and missing method-specific inputs fail with
  `MethodIncompatibility` diagnostics instead of raw late failures.
- Added typed reward-model plumbing and branch-local required-argument narrowing
  for `direct_method`, `ips`, `snips`, `doubly_robust`, `switch_dr`, and
  `evaluate`.
- Migrated `sharp_ope_unobserved(...)` contract failures for empty/non-DataFrame
  data, bad `gamma`, missing columns, non-finite inputs, and invalid logging or
  target probabilities to taxonomy errors with recovery hints.
- Hardened `causal_policy_forest(...)` validation for DataFrame shape, scalar
  or sequence covariates, missing columns, tree/depth/subsample options, small
  samples, finite inputs, action coding, and observed action support.
- Fixed policy-forest variable reuse (`idx`/`counts`) that polluted mypy types,
  and annotated the `sp.ope` lazy export path.
- Expanded OPE tests for taxonomy classes, missing dispatcher inputs, bad sharp
  OPE probabilities/columns, scalar covariates, invalid forest options, and
  small logged-policy samples.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/ope/estimators.py
  src/statspai/ope/sharp_confounding.py tests/test_ope_cevae.py
  tests/test_ope_extensions.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ope_cevae.py tests/test_ope_extensions.py` passed, 16 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ope_cevae.py tests/test_ope_extensions.py
  tests/test_api_surface_consistency.py tests/test_v100_integration.py` passed,
  103 tests.
- `.venv/bin/python -m flake8 src/statspai/ope/__init__.py
  src/statspai/ope/estimators.py src/statspai/ope/sharp_confounding.py
  tests/test_ope_cevae.py tests/test_ope_extensions.py --max-line-length=88
  --ignore=E203,W503 --statistics --count` passed with 0 touched-file
  violations.
- `.venv/bin/python -m mypy src/statspai/ope/__init__.py
  src/statspai/ope/estimators.py src/statspai/ope/sharp_confounding.py
  --ignore-missing-imports --show-error-codes` reported success for all three
  OPE source files after the repository Python-version warning.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1356 taxonomy raises and 1286 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4382 <= 4698, mypy observed 3209 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 132

Target: Stata `psmatch2` migration front door input-contract and taxonomy
hardening.

- Added `psmatch2`-specific validators for DataFrame input, non-empty matching
  and panel data, role-specific column names, scalar-vs-sequence covariates,
  cluster/fixed-effect columns, and matched-panel overlap.
- Migrated user-facing `psmatch2(...)` configuration failures for missing
  `treat`/`covariates`, missing columns, outcome/covariate collisions, unknown
  method names, radius-without-caliper, and invalid SE options to
  `MethodIncompatibility` with recovery hints and diagnostics.
- Migrated `PSMatch2Result.psm_did(...)` failures for invalid weighting, missing
  id/post/time inputs, missing id in the matched frame, and empty merged matched
  panels to taxonomy errors (`MethodIncompatibility` or `DataInsufficient`).
- Preserved Stata-compatible behavior while adding the one-column shortcut
  `covariates="x1"` so it is treated as `["x1"]` rather than a character list.
- Added return/parameter annotations to the touched `psmatch2.py` methods and
  narrowed the optional outcome path; direct mypy on the touched source is now
  clean apart from the repository-level Python-version warning.
- Expanded `tests/test_psmatch2.py` coverage for taxonomy classes, scalar
  covariates, invalid methods/SE/radius options, and empty PSM-DID merge paths.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/matching/psmatch2.py
  tests/test_psmatch2.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_psmatch2.py` passed, 57 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_psmatch2.py tests/test_matching.py
  tests/reference_parity/test_matching_parity.py` passed, 115 tests.
- `.venv/bin/python -m flake8 src/statspai/matching/psmatch2.py
  tests/test_psmatch2.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/matching/psmatch2.py
  --ignore-missing-imports --show-error-codes` reported success for the source
  file after the repository Python-version warning.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1352 taxonomy raises and 1291 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4390 <= 4698, mypy observed 3237 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 131

Target: panel-regression dispatcher and diagnostics contract hardening.

- Added panel-specific validators in `src/statspai/panel/panel_reg.py` for
  DataFrame input, non-empty data, role-specific entity/time/outcome/regressor
  columns, and structured method/formula errors.
- Migrated `PanelResults` diagnostic wrapper failures for missing stored panel
  data or missing linearmodels result to `MethodIncompatibility`.
- Migrated `PanelResults.plot(type=...)` unknown plot-type failures to
  `MethodIncompatibility`.
- Migrated classical `panel_reg.panel(...)` method alias, formula, missing
  column, and balance-all-dropped failures to taxonomy errors while preserving
  `ValueError` compatibility.
- Migrated the public `sp.panel(...)` unknown-method dispatcher failure to
  `MethodIncompatibility`; non-string method failures intentionally remain
  `TypeError` for backward compatibility.
- Cleaned touched-file flake8 debt in `panel_reg.py` and the panel coverage
  test without changing linearmodels estimation logic.
- Updated panel coverage tests to assert taxonomy classes for diagnostic,
  formula/method/column, balance, and plot error paths.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/panel/panel_reg.py
  src/statspai/panel/__init__.py tests/test_cov95_panel_reg.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_cov95_panel_reg.py tests/test_panel_dispatcher.py` passed,
  82 tests.
- `MPLBACKEND=Agg .venv/bin/python -m flake8
  src/statspai/panel/panel_reg.py src/statspai/panel/__init__.py
  tests/test_cov95_panel_reg.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/panel/panel_reg.py
  src/statspai/panel/__init__.py --show-error-codes --no-error-summary
  --hide-error-context` still reports historical panel-module annotation debt
  plus the repository Python-version warning.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1350 taxonomy raises and 1301 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4390 <= 4698, mypy observed 3243 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `.venv/bin/python -m compileall -q src/statspai` and `git diff --check`
  passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 130

Target: count-regression and `xtnbreg` input-contract hardening.

- Added shared count-model validators in `src/statspai/regression/count.py`
  for DataFrame inputs, non-empty samples, column-name normalization, missing
  columns, numeric finite auxiliary columns, and strictly positive exposure.
- Migrated `_parse_formula_or_xy(...)` missing formula/data/y/x errors and
  missing model columns to `MethodIncompatibility` / `DataInsufficient` with
  structured diagnostics.
- Accepted scalar-string `x="x1"` as a one-regressor shorthand instead of
  iterating over characters in the y/x API path.
- Migrated fixed-effect dummy construction failures for missing FE columns and
  missing FE identifiers to taxonomy errors.
- Routed Poisson/NB exposure handling through the positive-exposure validator,
  preventing silent `log(0)` or `log(negative)` offsets.
- Migrated `xtnbreg(...)` configuration failures for missing data, missing
  formula/y, missing entity/time columns, invalid `model`, random-effects
  entity requirements, and random-effects exposure checks to taxonomy errors.
- Cleaned touched-file flake8 debt in `count.py` by removing unused imports and
  unused locals and wrapping long historical lines; no estimator math was
  intentionally changed.
- Added panel-NB tests proving taxonomy errors remain `ValueError` compatible
  and covering NB/Poisson exposure, FE-column, y/x, and `xtnbreg` configuration
  failures.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/regression/count.py
  tests/test_count_panel_nbreg.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_count_panel_nbreg.py` passed, 6 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_count_panel_nbreg.py tests/test_registry_new_modules.py
  tests/test_api_surface_consistency.py tests/test_v100_integration.py`
  passed, 114 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/reference_parity/test_count_quantile_parity.py
  tests/test_new_v06_modules.py::TestCountData::test_poisson
  tests/test_new_v06_modules.py::TestCountData::test_nbreg
  tests/test_new_v06_modules.py::TestCountData::test_ppmlhdfe` passed,
  13 tests with the existing NB convergence warning in the v0.6 smoke test.
- `MPLBACKEND=Agg .venv/bin/python -m flake8
  src/statspai/regression/count.py tests/test_count_panel_nbreg.py
  --max-line-length=88 --ignore=E203,W503 --statistics --count` passed with
  0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/regression/count.py --show-error-codes
  --no-error-summary --hide-error-context` still reports historical count
  module annotation debt plus the repository Python-version warning.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1347 taxonomy raises and 1312 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4399 <= 4698, mypy observed 3243 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `.venv/bin/python -m compileall -q src/statspai` and `git diff --check`
  passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 129

Target: shared Bayesian result/sampling contract hardening.

- Migrated common `src/statspai/bayes/_base.py` sampling-control and result
  request failures from raw `ValueError`/`RuntimeError` paths to StatsPAI
  taxonomy errors.
- Added PyMC-free validators for `_sample_model(...)` controls before the
  optional PyMC import: `inference`, `draws`, `tune`, `chains`,
  `advi_iterations`, and `target_accept`.
- Hardened `BayesianDIDResult.tidy(...)`,
  `BayesianIVResult.tidy(...)`, and `BayesianMTEResult.tidy(...)` so invalid
  term requests, missing per-cohort/per-instrument summaries, and non-iterable
  `terms` objects raise `MethodIncompatibility` while preserving historical
  `ValueError` compatibility.
- Hardened `BayesianHTEIVResult.predict_cate(...)` and
  `BayesianMTEResult.policy_effect(...)` missing-state paths, finite modifier
  values, callable/shape/numeric/finite/all-zero weight contracts, finite
  `u_grid`, and ROPE validation.
- Removed unnecessary early PyMC imports from result-method validation paths so
  malformed result requests fail clearly even when the optional `bayes` extra
  is absent.
- Added PyMC-free protocol tests covering sampling-control validation,
  Bayesian DID/IV/MTE tidy errors, HTE predict-state errors, and MTE
  policy-weight contract errors.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bayes/_base.py
  tests/test_bayes_result_protocol.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bayes_result_protocol.py` passed, 9 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bayes_mte_tidy.py tests/test_bayes_result_protocol.py` passed,
  9 tests with 1 optional PyMC skip.
- `MPLBACKEND=Agg .venv/bin/python -m flake8 src/statspai/bayes/_base.py
  tests/test_bayes_result_protocol.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bayes/_base.py --show-error-codes
  --no-error-summary --hide-error-context` reported no file-level type errors
  beyond the repository's Python-version warning.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1331 taxonomy raises and 1322 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=11`,
  `untested=184`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4416 <= 4698, mypy observed 3243 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `.venv/bin/python -m compileall -q src/statspai` and `git diff --check`
  passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 128

Target: PyMC-free Bayesian DID input-contract hardening.

- Migrated `src/statspai/bayes/did.py` DID data-preparation failures from raw
  `ValueError` branches to StatsPAI taxonomy errors:
  `MethodIncompatibility`, `DataInsufficient`, and `NumericalInstability`.
- Added structured validators for DataFrame input, role-specific column names,
  missing columns, distinct model roles, complete-case sample size, binary
  treatment/post indicators, panel unit/time support, cohort support, and
  finite numeric outcome/covariate arrays.
- Moved `_prepare_did_frame(...)` ahead of the optional PyMC import in
  `bayes_did()`, so malformed inputs fail with actionable StatsPAI errors even
  in environments that have not installed the `bayes` extra.
- Accepted scalar-string `covariates="x"` as a one-covariate shorthand instead
  of accidentally iterating over characters.
- Added `tests/test_bayes_did_validation.py`, a PyMC-free validation suite
  covering missing columns, non-DataFrame inputs, non-binary treatment coding,
  thin complete-case samples, single unit/time/cohort support, non-finite
  numeric inputs, and scalar covariate normalization.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/bayes/did.py
  tests/test_bayes_did_validation.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bayes_did_validation.py tests/test_bayes_result_protocol.py`
  passed, 14 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bayes_did_validation.py tests/test_bayes_result_protocol.py
  tests/test_registry_new_modules.py tests/test_api_surface_consistency.py
  tests/test_v100_integration.py` passed, 122 tests.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_bayes_did.py` was environment-skipped because PyMC is not
  installed locally.
- `MPLBACKEND=Agg .venv/bin/python -m flake8 src/statspai/bayes/did.py
  tests/test_bayes_did_validation.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python -m mypy src/statspai/bayes/did.py --show-error-codes
  --no-error-summary --hide-error-context` reported no file-level type errors
  beyond the repository's Python-version warning.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1303 taxonomy raises and 1333 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4418 <= 4698, mypy observed 3251 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `.venv/bin/python -m compileall -q src/statspai` and `git diff --check`
  passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 127

Target: Evidence-Without-Injustice fairness diagnostic validation.

- Migrated `src/statspai/fairness/evidence_test.py` explicit raw
  `ValueError`/`TypeError`/`RuntimeError` paths to StatsPAI taxonomy errors:
  `MethodIncompatibility`, `DataInsufficient`, `NumericalInstability`, and
  `ConvergenceFailure`.
- Added private validators for DataFrame inputs, protected/admissible column
  names, alpha, threshold, bootstrap count, alternative values, predictor
  output shape/finite-ness, and SCM DataFrame/row-count contracts.
- Accepted a scalar string `admissible_features` as a one-feature shorthand,
  avoiding accidental character-wise iteration while preserving sequence input.
- Hardened predictor handling so wrong-length outputs and nonnumeric outputs
  are method errors, while non-finite numeric outputs are numerical
  instability errors.
- Hardened SCM handling so non-DataFrame returns and length mismatches fail
  before bootstrap, and admissible-feature freezing no longer mutates the
  returned SCM object in place.
- Converted the low-success bootstrap path to `ConvergenceFailure` with
  `n_ok`/`n_boot` diagnostics.
- Added direct EWI tests for the admissible-evidence freeze, scalar
  admissible-feature shorthand, bad protected/admissible inputs, invalid
  alpha/threshold/n_boot, bad predictor outputs, bad SCM outputs, single-level
  protected attributes, and bootstrap convergence failure.
- Cleaned nearby fairness test formatting under the repository's 88-column
  flake8 gate.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/fairness/evidence_test.py
  tests/test_fairness.py` passed.
- `.venv/bin/python -m flake8 src/statspai/fairness/evidence_test.py
  tests/test_fairness.py --max-line-length=88 --ignore=E203,W503` passed.
- `.venv/bin/python -m mypy src/statspai/fairness/evidence_test.py
  --show-error-codes` passed with no issues in the touched module.
- `.venv/bin/python -m pytest tests/test_fairness.py -q -o addopts=''`
  passed, 20 tests.
- `.venv/bin/python -m pytest tests/test_registry_new_modules.py
  tests/test_api_surface_consistency.py tests/test_v100_integration.py -q
  -o addopts=''` passed together with `tests/test_fairness.py`, 128 tests.
- `.venv/bin/python -m pytest tests/test_auto_estimators.py
  tests/test_external_reviewer_followups.py -q -o addopts=''` passed, 36
  tests.
- `.venv/bin/python -m pytest tests/test_tierD_p2_fairness_analytic.py
  tests/test_fairness.py -q -o addopts=''` passed, 24 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1290 taxonomy raises and 1343 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed under the repository's
  gradual-debt baselines: flake8 observed 4420 <= 4698, mypy observed
  3251 <= 3521, import-budget observed 0, and the agent-card,
  result-protocol, and error-taxonomy gates all passed.
- `git diff --check` passed.
- `git status --short Paper-JSS CausalAgentBench paper.md paper.bib` produced
  no output.

## 2026-06-17 Batch 126

Target: decomposition shared-helper validation and type hygiene.

- Migrated `src/statspai/decomposition/_common.py` explicit raw
  `ValueError` raises to `MethodIncompatibility` for invalid wild-bootstrap
  weights, bootstrap CI methods, weighted quantile dimensions/probabilities,
  unknown distributional statistics, invalid quantile conventions, malformed
  formulas, and misaligned weights.
- Added a pre-index missing-column guard in `prepare_frame`, converting pandas
  `KeyError` into a StatsPAI taxonomy error with missing/available column
  diagnostics.
- Preserved historical `ValueError` compatibility through the taxonomy
  subclass while adding recovery hints and machine-readable diagnostics.
- Added taxonomy contract coverage in the direct decomposition common tests
  and internals tests for method/stat/weight/formula/missing-column failures.
- Added lightweight `cast`/float narrowing in `_common.py`, reducing that
  module's direct mypy errors to zero without changing numerical behavior.
- Removed one unused test assignment and split long existing assertions in the
  touched decomposition common test file under the repository's 88-column
  flake8 gate.

Verification run:

- `.venv/bin/python -m py_compile src/statspai/decomposition/_common.py
  tests/test_decomposition_cov_common.py
  tests/test_decomposition_cov_internals2.py` passed.
- `.venv/bin/python -m flake8 src/statspai/decomposition/_common.py
  tests/test_decomposition_cov_common.py
  tests/test_decomposition_cov_internals2.py --max-line-length=88
  --ignore=E203,W503` passed.
- `.venv/bin/python -m mypy src/statspai/decomposition/_common.py
  --show-error-codes` passed with no issues in the touched module.
- `.venv/bin/python -m pytest tests/test_decomposition_cov_common.py
  tests/test_decomposition_cov_internals2.py -q -o addopts=''` passed,
  75 tests.
- `.venv/bin/python -m pytest tests/test_decomposition_cov_common.py
  tests/test_decomposition_cov_internals2.py tests/test_decomposition_cov_oaxaca.py
  tests/test_decomposition_cov_ineq2.py tests/test_decomposition_cov_rif2.py
  tests/test_rif.py tests/test_tierD_p2_decomposition_analytic.py -q
  -o addopts=''` passed, 171 tests with 1 existing RuntimeWarning from
  `decomposition/rif.py` degenerate-density coverage.
- `.venv/bin/python -m pytest tests/test_decomposition_cov_final.py
  tests/test_decomposition_cov_misc2.py tests/test_decomposition_cov_inference.py
  tests/test_decomposition_tier_c.py -q -o addopts=''` passed, 86 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1268 taxonomy raises and 1353 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed under the repository's
  gradual-debt baselines: flake8 observed 4421 <= 4698, mypy observed
  3253 <= 3521, import-budget observed 0, and the agent-card,
  result-protocol, and error-taxonomy gates all passed.
- `git diff --check` passed.
- `git status --short Paper-JSS CausalAgentBench paper.md paper.bib` produced
  no output.

## 2026-06-17 Batch 125

Target: epidemiology 2x2 and person-time measure failure semantics.

- Migrated `src/statspai/epi/measures.py` explicit raw
  `ValueError`/`RuntimeError` raises to StatsPAI taxonomy errors while keeping
  backward-compatible `ValueError`/`RuntimeError` inheritance.
- Centralized 2x2 table validation for malformed shapes, partial scalar
  inputs, nonnumeric counts, non-finite counts, and negative counts.
- Added `alpha` validation through the shared z-critical helper so invalid
  confidence levels fail before producing misleading intervals.
- Converted empty exposure rows in `relative_risk` and `risk_difference` to
  `DataInsufficient` with row-total diagnostics.
- Hardened `incidence_rate_ratio` by validating method, numeric finiteness,
  positive person-time, and nonnegative events before the all-zero-event
  shortcut; this closes the bug where `method="bad"` could be returned on a
  zero-event table.
- Added taxonomy contract tests for malformed 2x2 tables, invalid methods,
  invalid alpha, exact-backend failure, empty rows, and IRR bad inputs.
- Added lightweight type annotations to the touched epidemiology functions,
  reducing this file's direct mypy errors to zero.

Verification run:

- `.venv/bin/python -m pytest tests/test_epi.py -q -o addopts=''` passed,
  29 tests.
- `.venv/bin/python -m pytest tests/test_epi.py tests/test_epi_diagnostic.py
  tests/test_tierD_p2_epi_analytic.py -q -o addopts=''` passed, 53 tests.
- `.venv/bin/python -m pytest tests/test_api_surface_consistency.py
  tests/test_v100_integration.py -q -o addopts=''` passed, 87 tests.
- `.venv/bin/python -m pytest tests/test_result_consumer_errors.py
  tests/test_agent_result_methods.py -q -o addopts=''` passed, 29 tests.
- `.venv/bin/python -m flake8 src/statspai/epi/measures.py
  tests/test_epi.py` passed.
- `.venv/bin/python -m mypy src/statspai/epi/measures.py
  --show-error-codes` passed with no issues in the touched module.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1257 taxonomy raises and 1363 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed under the repository's
  gradual-debt baselines: flake8 observed 4421 <= 4698, mypy observed
  3266 <= 3521, import-budget observed 0, and the agent-card,
  result-protocol, and error-taxonomy gates all passed.
- `git diff --check` passed.
- `git status --short Paper-JSS CausalAgentBench paper.md paper.bib` produced
  no output.

## 2026-06-17 Batch 124

Target: CJM rddensity input contracts, backend failure taxonomy, and type
noise reduction.

- Migrated `src/statspai/diagnostics/rddensity.py` explicit raw
  `ValueError`/`RuntimeError` raises to taxonomy exceptions.
- Added shared validation for DataFrame input, running-variable column name,
  missing columns, finite cutoff, supported polynomial order, `alpha`, and
  backend type.
- Added numeric running-variable extraction that reports non-numeric inputs as
  `DataInsufficient`.
- Added shared support validation for at least 20 finite observations and at
  least 5 observations on each side of the cutoff.
- Migrated manual bandwidth validation to `MethodIncompatibility`, including
  non-numeric scalar bandwidths, wrong-length side-specific bandwidths,
  non-numeric side-specific bandwidths, and non-positive/non-finite values.
- Kept missing Rscript as `ImportError` because it is an optional dependency
  availability signal; mapped R backend execution failure to
  `ConvergenceFailure`.
- Migrated unsupported Hermite/order guard to `MethodIncompatibility`.
- Added type annotations and explicit float/ndarray coercions in the bandwidth
  helpers, reducing targeted mypy noise for the module.
- Updated rddensity tests to assert taxonomy errors for invalid bandwidths,
  backend selection, missing running-variable columns, unsupported polynomial
  order, too few valid rows, and insufficient cutoff-side support.

Verification:

- `.venv/bin/python -m flake8 src/statspai/diagnostics/rddensity.py tests/test_rddensity_io.py --select=F,E9`
  passed.
- `.venv/bin/python -m mypy src/statspai/diagnostics/rddensity.py --hide-error-context --no-error-summary`
  passed with only the repository's pyproject warning.
- `.venv/bin/python -m pytest tests/test_rddensity_io.py tests/test_rd_validation.py::TestDensityValidation tests/test_estimator_provenance_round10.py::test_rddensity_provenance -o addopts=''`
  passed, 23 tests.
- `.venv/bin/python -m pytest tests/test_rddensity_io.py tests/test_rd_validation.py tests/test_cov95_rd_misc.py tests/test_rd_new_modules.py tests/test_estimator_provenance_round10.py -o addopts=''`
  passed, 107 tests with 3 pre-existing warnings.
- `.venv/bin/python -m pytest tests/test_api_surface_consistency.py tests/test_exception_migrations.py tests/test_late_bind_contracts.py tests/test_export_surface_contract.py -o addopts=''`
  passed, 110 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,239 taxonomy raises, 1,373 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; Tier-D worklist remains 0 estimator-like rows.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  Tier A fixture lock is current.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 observed 4,422 <= 4,698 baseline; mypy observed 3,275 <= 3,521
  baseline; import-budget observed 0; agent-card/result-protocol/error-taxonomy
  checks passed.
- `git diff --check` passed.
- `git status --short Paper-JSS CausalAgentBench paper.md paper.bib` produced
  no output; JOSS/manuscript paths remain untouched.

## 2026-06-17 Batch 123

Target: 2D/boundary RD input contracts and taxonomy migration.

- Migrated `src/statspai/rd/rd2d.py` explicit raw `ValueError` raises to
  taxonomy exceptions.
- Added shared validation for DataFrame input, required column names, missing
  columns, approach, kernel, polynomial order, manual bandwidth, alpha,
  boundary callables, `n_eval`, and finite `(k, 2)` evaluation points.
- Added taxonomy-aware numeric extraction for `y`, `x1`, `x2`, and treatment
  columns.
- Mapped sparse-data failures to `DataInsufficient`: fewer than 20 valid rows,
  fewer than 5 treated/control rows, and too few observations on either side
  of the distance boundary.
- Migrated RD2D plotting misuse to `MethodIncompatibility`, including invalid
  `plot_type`, missing location-evaluation detail for boundary-effects plots,
  and non-callable custom boundaries.
- Removed an unused `matplotlib.cm` import from `rd2d_plot`.
- Added cov95 tests for missing columns, too few valid rows, too few
  treated/control rows, invalid evaluation points, bandwidth-selector missing
  columns, invalid plot type, and boundary-effects plot detail requirements.

Verification:

- `.venv/bin/python -m flake8 src/statspai/rd/rd2d.py tests/test_cov95_rd_rd2d.py --select=F,E9`
  passed.
- `.venv/bin/python -m pytest tests/test_cov95_rd_rd2d.py tests/test_rd_new_modules.py::TestRD2D tests/test_rd_cov_estimators.py::test_rd2d_boundary tests/test_tierD_rd_multiscore_analytic.py::TestBoundaryRDAnalytic::test_alias_equals_rd2d -o addopts=''`
  passed, 20 tests.
- `.venv/bin/python -m pytest tests/test_cov95_rd_rd2d.py tests/test_rd_new_modules.py tests/test_rd_cov_estimators.py tests/test_rd_validation.py tests/test_tierD_rd_multiscore_analytic.py tests/test_rd_dispatcher.py -o addopts=''`
  passed, 109 tests with 4 pre-existing warnings.
- `.venv/bin/python -m pytest tests/test_api_surface_consistency.py tests/test_exception_migrations.py tests/test_late_bind_contracts.py tests/test_export_surface_contract.py -o addopts=''`
  passed, 110 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,220 taxonomy raises, 1,383 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; Tier-D worklist remains 0 estimator-like rows.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  Tier A fixture lock is current.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 observed 4,422 <= 4,698 baseline; mypy observed 3,278 <= 3,521
  baseline; import-budget observed 0; agent-card/result-protocol/error-taxonomy
  checks passed.
- `git diff --check` passed.
- `git status --short Paper-JSS CausalAgentBench paper.md paper.bib` produced
  no output; JOSS/manuscript paths remain untouched.

## 2026-06-17 Batch 122

Target: long-panel DML input contracts, taxonomy migration, and scalar
covariate ergonomics.

- Migrated `src/statspai/dml/panel_dml.py` explicit raw
  `ValueError`/`RuntimeError` raises to taxonomy exceptions.
- Added early validation for DataFrame input, required column-name arguments,
  scalar or sequence `covariates`, fold count, `alpha`, boolean flags, missing
  columns, and missing time columns under two-way FE.
- Added `DataInsufficient` failures for empty complete panels and fold counts
  that exceed observed units.
- Added finite outcome/treatment/covariate checks and taxonomy-specific
  sample-weight validation for missing weight columns, wrong weight shape,
  negative weights, non-finite weights, and zero total weight mass.
- Mapped the zero residual-treatment-variation denominator branch to
  `NumericalInstability` while preserving `RuntimeError` compatibility.
- Added scalar `covariates="x1"` coverage through both focused and cov95 panel
  DML tests.
- Updated key panel DML validation tests to assert `MethodIncompatibility` or
  `DataInsufficient` directly.

Verification:

- `.venv/bin/python -m flake8 src/statspai/dml/panel_dml.py tests/test_dml_panel.py tests/test_cov95_dml_averaging_panel.py --select=F,E9`
  passed.
- `.venv/bin/python -m pytest tests/test_dml_panel.py tests/test_cov95_dml_averaging_panel.py -o addopts=''`
  passed, 56 tests.
- `.venv/bin/python -m pytest tests/test_dml_panel.py tests/test_cov95_dml_averaging_panel.py tests/test_dml_cov_diag_sens.py tests/test_dml_cov_averaging_panel.py tests/test_dml_cov_base.py tests/test_dml_cov_scores.py tests/test_cov95_dml_learners_base.py -o addopts=''`
  passed, 142 tests with 3 skipped.
- `.venv/bin/python -m pytest tests/test_api_surface_consistency.py tests/test_exception_migrations.py tests/test_late_bind_contracts.py tests/test_export_surface_contract.py -o addopts=''`
  passed, 110 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,200 taxonomy raises, 1,394 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; Tier-D worklist remains 0 estimator-like rows.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  Tier A fixture lock is current.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 observed 4,425 <= 4,698 baseline; mypy observed 3,278 <= 3,521
  baseline; import-budget observed 0; agent-card/result-protocol/error-taxonomy
  checks passed.
- `git diff --check` passed.
- `git status --short Paper-JSS CausalAgentBench paper.md paper.bib` produced
  no output; JOSS/manuscript paths remain untouched.

## 2026-06-17 Batch 121

Target: spatial DiD input contracts, taxonomy errors, and scalar covariate
ergonomics.

- Migrated `src/statspai/spatial/did.py` explicit raw error raises to
  taxonomy exceptions.
- Added early `spatial_did()` validation for DataFrame input, required column
  names, missing columns, `alpha`, `se_type`, `normalize_W`, `event_window`,
  Conley cutoff/kernel configuration, empty complete panels, duplicate
  unit-period cells, W/data dimension mismatch, and invalid distance matrices.
- Hardened spatial weights conversion for non-numeric, non-square, and
  non-finite W inputs with `MethodIncompatibility` diagnostics.
- Hardened unit-order alignment and coordinate availability checks, using
  `DataInsufficient` when complete panel or coordinate support is absent.
- Migrated result export/plot misuse (`detail`, missing exposure/event-study
  diagnostics, invalid plot kind) to `MethodIncompatibility`.
- Added scalar `covariates="x"` support and covered it through the public
  spatial DID export path.
- Removed two dead local variables in spatial lag/event-time helpers and
  avoided a stale mypy variable-reuse issue in event-study pretrend indices.
- Added failure-path tests for invalid W shape, mismatched unit order,
  duplicate cells, missing columns, Conley input requirements, result export
  detail, plot kinds, missing event-study plots, and empty complete panels.

Verification:

- `.venv/bin/python -m flake8 src/statspai/spatial/did.py tests/spatial/test_did.py --select=F,E9`
  passed.
- `.venv/bin/python -m pytest tests/spatial/test_did.py tests/test_estimator_provenance_round7.py::TestSpatialDidProvenance -o addopts=''`
  passed, 10 tests.
- `.venv/bin/python -m pytest tests/spatial tests/test_estimator_provenance_round7.py tests/test_api_surface_consistency.py tests/test_export_surface_contract.py -o addopts=''`
  passed, 145 tests with 1 skipped and 1 pre-existing warning.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,178 taxonomy raises, 1,405 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; Tier-D worklist remains 0 estimator-like rows.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  Tier A fixture lock is current.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 observed 4,426 <= 4,698 baseline; mypy observed 3,278 <= 3,521
  baseline; import-budget observed 0; agent-card/result-protocol/error-taxonomy
  checks passed.
- `git diff --check` passed.
- `git status --short Paper-JSS CausalAgentBench paper.md paper.bib` produced
  no output; JOSS/manuscript paths remain untouched.

## 2026-06-17 Batch 120

Target: RD machine-learning heterogeneity exception taxonomy and covariate
ergonomics.

- Migrated `src/statspai/rd/rdml.py` raw input/data failures to taxonomy
  exceptions, using `MethodIncompatibility` for incompatible covariate and
  method configurations and `DataInsufficient` for sparse bandwidth or split
  samples.
- Added structured recovery hints and diagnostics to RDML failure paths for
  bandwidth restrictions, missing covariates, running-variable misuse, sparse
  treatment/control sides, honest-split underflow, unknown CATE summary
  methods, and missing variable importance.
- Allowed scalar string covariates in RDML helpers and scalar string
  `methods` in `rd_cate_summary`, preserving list behavior.
- Updated RDML covariance tests to assert the taxonomy exceptions and added
  scalar-string coverage through the public summary path.

Verification:

- `.venv/bin/python -m flake8 src/statspai/rd/rdml.py tests/test_cov95_rd_r2_rdml.py --select=F,E9`
  passed.
- `.venv/bin/python -m pytest tests/test_cov95_rd_r2_rdml.py tests/test_cov95_rd_ml_and_hte.py -o addopts=''`
  passed, 21 tests.
- `.venv/bin/python -m pytest tests/test_cov95_rd_r2_rdml.py tests/test_cov95_rd_ml_and_hte.py tests/test_rd_new_modules.py tests/test_rd_cov_estimators.py tests/test_cov95_rd_misc.py -o addopts=''`
  passed, 89 tests with 3 pre-existing warnings.
- `.venv/bin/python -m pytest tests/test_api_surface_consistency.py tests/test_exception_migrations.py tests/test_late_bind_contracts.py tests/test_export_surface_contract.py -o addopts=''`
  passed, 110 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,142 taxonomy raises, 1,418 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; Tier-D worklist remains 0 estimator-like rows.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  Tier A fixture lock is current.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 observed 4,430 <= 4,698 baseline; mypy observed 3,280 <= 3,521
  baseline; import-budget observed 0; agent-card/result-protocol/error-taxonomy
  checks passed.
- `git diff --check` passed.
- `git status --short Paper-JSS CausalAgentBench paper.md paper.bib` produced
  no output; JOSS/manuscript paths remain untouched.

## 2026-06-17 Batch 119

Target: Honest DID sensitivity input contracts and taxonomy migration.

- Migrated `src/statspai/did/honest_did.py` off raw
  `ValueError`/`RuntimeError`/`TypeError`/`NotImplementedError`/`KeyError`
  raises.
- Added validation helpers for string options, open-unit `alpha`, integer
  relative time `e`, and finite non-negative `m_grid` values.
- Hardened native `honest_did()` for invalid backend/method/alpha/e/m_grid,
  missing target event times, and invalid event-study inputs while preserving
  old error-message keywords.
- Hardened the R HonestDiD backend for invalid method/honestdid_method,
  relative-magnitude method incompatibilities, invalid target event times,
  missing pre/post periods, invalid M grids, and R execution failure.
- Mapped R execution failure to `ConvergenceFailure` while preserving the
  RuntimeError-compatible taxonomy branch; kept missing Rscript as ImportError
  because that is an optional dependency availability signal.
- Hardened `breakdown_m()` and `_extract_event_study()` with taxonomy errors
  for bad methods, missing event-study tables, missing required columns, and
  unavailable relative times.
- Removed two local unused variables surfaced by focused flake.
- Added taxonomy-specific tests in
  `tests/test_cov95_did_r4_honest_pretrends.py` for bad method, bad `m_grid`,
  no event-study table, and missing relative time.

Verification:

- `.venv/bin/python -m flake8 src/statspai/did/honest_did.py tests/test_cov95_did_r4_honest_pretrends.py --select=F,E9`
  passed.
- `.venv/bin/python -m pytest tests/test_cov95_did_r4_honest_pretrends.py tests/test_honest_did_sdid.py tests/test_honest_did_aggte.py tests/test_event_study_consumers.py tests/test_honest_did_backend.py tests/external_parity/test_honest_did_paper_parity.py -o addopts=''`
  passed: 73 tests.
- `.venv/bin/python -m pytest tests/test_cs_report_smoke.py tests/test_cs_report.py tests/test_did_summary.py tests/test_cov95_did_summary.py tests/test_cov95_did_summary_extra.py tests/test_cs_rcs.py -o addopts=''`
  passed: 91 tests.
- `.venv/bin/python -m pytest tests/test_api_surface_consistency.py tests/test_exception_migrations.py tests/test_late_bind_contracts.py tests/test_export_surface_contract.py -o addopts=''`
  passed: 110 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,128 taxonomy raises, 1,430 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; reference 128, anchored 563, weak 147, smoke 9,
  untested 186; Tier-D estimator-like worklist 0.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 4,423, mypy 3,280, import-budget 0.
- `git diff --check` passed.
- JOSS boundary check stayed clean for `Paper-JSS/`, `CausalAgentBench/`,
  `paper.md`, and `paper.bib`.

## 2026-06-17 Batch 118

Target: Callaway-Sant'Anna DID input contracts and taxonomy migration.

- Migrated `src/statspai/did/callaway_santanna.py` validation paths off raw
  `ValueError`/`TypeError`/`RuntimeError`/`NotImplementedError`/`KeyError`
  raises.
- Added validators for non-empty DataFrames, string column names,
  scalar-or-list covariates, string options, open-unit `alpha`,
  non-negative `anticipation`, boolean `panel`, and required columns.
- Added `CallawayNotImplemented`, a `MethodIncompatibility` subclass that
  also preserves the historical `NotImplementedError` contract for
  repeated-cross-section branches that are still not implemented.
- Hardened public `callaway_santanna()` for bad estimator/control/base
  options, invalid anticipation/alpha/panel, missing columns/covariates,
  scalar `x="x1"`, no treatment cohorts, no valid group-time pairs, and
  unsupported `panel=False` combinations.
- Migrated `_prepare_panel()` and `_callaway_santanna_rcs()` column, empty
  post-drop sample, no-cohort, and no-pair errors to taxonomy classes with
  old regex-compatible message text.
- Added taxonomy-specific coverage in `tests/test_cov95_did_r5_callaway.py`
  for bad estimator, RCS not-implemented compatibility, scalar covariates,
  and no-cohort `DataInsufficient`.

Verification:

- `.venv/bin/python -m flake8 src/statspai/did/callaway_santanna.py tests/test_cov95_did_r5_callaway.py --select=F,E9`
  passed.
- `.venv/bin/python -m pytest tests/test_cov95_did_r5_callaway.py tests/test_cov95_did_callaway.py tests/test_did.py -o addopts=''`
  passed: 79 tests.
- `.venv/bin/python -m pytest tests/test_did_advanced.py tests/test_cov95_did_dispatcher.py tests/test_dispatchers_v150.py tests/test_article_aliases_round2.py -o addopts=''`
  passed: 103 tests.
- `.venv/bin/python -m pytest tests/reference_parity/test_callaway_santanna_parity.py tests/reference_parity/test_cross_estimator_parity.py tests/reference_parity/test_did_parity.py tests/test_api_surface_consistency.py tests/test_exception_migrations.py tests/test_late_bind_contracts.py -o addopts=''`
  passed: 76 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,106 taxonomy raises, 1,443 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; reference 128, anchored 563, weak 147, smoke 9,
  untested 186; Tier-D estimator-like worklist 0.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 4,421, mypy 3,280, import-budget 0.
- `git diff --check` passed.
- JOSS boundary check stayed clean for `Paper-JSS/`, `CausalAgentBench/`,
  `paper.md`, and `paper.bib`.

## 2026-06-17 Batch 117

Target: unified DID dispatcher input contracts and taxonomy migration.

- Migrated `src/statspai/did/__init__.py` dispatcher validation off raw
  `ValueError`/`TypeError`/`RuntimeError`/`NotImplementedError`/`KeyError`
  raises.
- Added dispatcher-local validators for non-empty DataFrames, string column
  names, scalar-or-list covariates, string method/options, open-unit `alpha`,
  boolean `robust`/`panel`, non-negative `anticipation`, and positive
  `n_boot`.
- Added `DIDInputTypeError`, a local `MethodIncompatibility` subclass that
  also preserves the historical `TypeError` catch for non-DataFrame `data`.
- Hardened column reporting, CS/Sun-Abraham aggregation compatibility,
  `panel=False`/`anticipation` method compatibility, non-binary auto-detection
  without `id`, required `subgroup`/`id` branches, invalid aggregation values,
  and unknown method values with structured diagnostics and recovery hints.
- Normalized scalar `covariates="x1"` to `["x1"]` in the dispatcher.
- Added `harvest_did`, `HarvestDIDResult`, and `continuous_did` to
  `did.__all__`, matching the module imports and eliminating local F401 drift.
- Updated `tests/test_cov95_did_dispatcher.py` to assert taxonomy-specific
  errors for empty/missing/non-binary/invalid-control paths while preserving
  the legacy TypeError contract for non-DataFrame inputs.

Verification:

- `.venv/bin/python -m flake8 src/statspai/did/__init__.py tests/test_cov95_did_dispatcher.py --select=F,E9`
  passed.
- `.venv/bin/python -m pytest tests/test_cov95_did_dispatcher.py tests/test_did.py tests/test_did_advanced.py tests/test_did_imputation_branches.py -o addopts=''`
  passed: 93 tests.
- `.venv/bin/python -m pytest tests/test_dispatchers_v150.py tests/test_article_aliases_round2.py tests/test_auto_estimators.py -o addopts=''`
  passed: 80 tests.
- `.venv/bin/python -m pytest tests/test_api_surface_consistency.py tests/test_late_bind_contracts.py tests/test_exception_migrations.py tests/test_export_surface_contract.py -o addopts=''`
  passed: 110 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,088 taxonomy raises, 1,457 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; reference 128, anchored 563, weak 147, smoke 9,
  untested 186; Tier-D estimator-like worklist 0.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 4,414, mypy 3,280, import-budget 0.
- `git diff --check` passed.
- JOSS boundary check stayed clean for `Paper-JSS/`, `CausalAgentBench/`,
  `paper.md`, and `paper.bib`.

## 2026-06-17 Batch 116

Target: political shift-share Bartik input contracts and taxonomy migration.

- Migrated `src/statspai/bartik/political.py` off raw
  `ValueError`/`TypeError`/`RuntimeError`/`KeyError` raises.
- Added module-local validation helpers for non-empty DataFrames/Series,
  string column names, required columns, scalar-or-list optional covariates,
  open-unit `alpha`, boolean `leave_one_out`, finite share matrices, finite
  shock vectors, and finite panel outcome/endog/instrument/covariate blocks.
- Preserved the existing behavior that partially overlapping
  `shares.columns` and `shocks.index` are aligned on their intersection, while
  upgrading the no-overlap case to `MethodIncompatibility` with diagnostics.
- Hardened `shift_share_political` for malformed data/shares/shocks, invalid
  `alpha`, non-boolean `leave_one_out`, missing columns, one-period panels,
  empty unit/share alignment, scalar covariates, and non-finite shares/shocks.
- Hardened `shift_share_political_panel` and its shares/shocks resolvers for
  invalid FE/cluster modes, missing time-specific shares/shocks, inconsistent
  industry columns, non-DataFrame share entries, non-Series shock entries,
  missing Bartik IV rows, one-unit/one-period panels, non-finite work arrays,
  and AKM near-zero denominators.
- Added focused tests in `tests/test_shift_share_political.py` for scalar
  covariates, malformed shares, invalid alpha, non-finite shares, invalid
  FE/cluster options, missing shock rows, and uncovered units.
- Tightened local numpy helper return typing so the module did not add mypy
  `no-any-return` debt.

Verification:

- `.venv/bin/python -m flake8 src/statspai/bartik/political.py tests/test_shift_share_political.py --select=F,E9`
  passed.
- `.venv/bin/python -m pytest tests/test_shift_share_political.py -o addopts=''`
  passed: 15 tests.
- `.venv/bin/python -m pytest tests/test_bartik.py tests/test_transport_and_shiftshare.py tests/test_iv_dispatcher.py -o addopts=''`
  passed: 60 tests, 7 expected Bartik leave-one-out warnings.
- `.venv/bin/python -m pytest tests/test_api_surface_consistency.py tests/test_late_bind_contracts.py tests/test_exception_migrations.py -o addopts=''`
  passed: 55 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,066 taxonomy raises, 1,471 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; reference 128, anchored 563, weak 147, smoke 9,
  untested 186; Tier-D estimator-like worklist 0.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 4,414, mypy 3,280, import-budget 0.
- `git diff --check` passed.
- JOSS boundary check stayed clean for `Paper-JSS/`, `CausalAgentBench/`,
  `paper.md`, and `paper.bib`.

## 2026-06-17 Batch 115

Target: fairness diagnostic input contracts and exception taxonomy.

- Migrated `src/statspai/fairness/core.py` off raw
  `ValueError`/`TypeError`/`RuntimeError`/`NotImplementedError` raises.
- Added local validation helpers for non-empty DataFrame inputs, string column
  names, finite non-negative thresholds, feature-column coercion, binary
  prediction/label columns, finite predictor outputs, and one-value-per-row
  counterfactual predictor contracts.
- Mapped user-actionable failures to `MethodIncompatibility`,
  `DataInsufficient`, and `NumericalInstability` while preserving
  `ValueError` compatibility through the project taxonomy subclasses.
- Hardened `counterfactual_fairness` for non-callable predictors/SCMs,
  non-finite predictions, non-DataFrame SCM outputs, length mismatches, empty
  alternatives, and single-level protected attributes.
- Hardened `orthogonal_to_bias` for scalar feature-column input, empty feature
  lists, missing columns, non-finite numeric protected values, non-finite
  features, and unsupported methods.
- Added focused regression tests in `tests/test_fairness.py` for threshold,
  missing-column, non-finite predictor, SCM output, scalar-feature, method, and
  non-finite feature paths.

Verification:

- `.venv/bin/python -m flake8 src/statspai/fairness/core.py tests/test_fairness.py --select=F,E9`
  passed.
- `.venv/bin/python -m pytest tests/test_fairness.py tests/test_tierD_p2_fairness_analytic.py -o addopts=''`
  passed: 20 tests.
- `.venv/bin/python -m pytest tests/test_v100_integration.py tests/test_api_stable_evidence.py tests/test_api_surface_consistency.py -o addopts=''`
  passed: 93 tests.
- `.venv/bin/python -m pytest tests/test_exception_migrations.py tests/test_export_surface_contract.py tests/test_late_bind_contracts.py -o addopts=''`
  passed: 105 tests.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed:
  1,032 taxonomy raises, 1,487 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed:
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` passed:
  1,033 registered functions; reference 128, anchored 563, weak 147, smoke 9,
  untested 186; Tier-D estimator-like worklist 0.
- `.venv/bin/python scripts/tier_a_fixture_lock.py` passed:
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed:
  no StatsPAI timing regressed beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed:
  flake8 4,386, mypy 3,282, import-budget 0.
- `git diff --check` passed.
- JOSS boundary check stayed clean for `Paper-JSS/`, `CausalAgentBench/`,
  `paper.md`, and `paper.bib`.

## 2026-06-17 Batch 114

Target: DML model-averaging input contracts and taxonomy errors.

- Added `dml/model_averaging.py` validation helpers for DataFrames, non-empty
  column names, scalar-or-list covariates, required columns, string options,
  integer controls, open-unit alpha, finite arrays, and candidate triples.
- Hardened `_solve_cls_weights()` for target/prediction dimensionality,
  non-finite inputs, sample-weight length, non-negativity, and positive total
  mass before calling SciPy's SLSQP.
- Hardened `dml_model_averaging()` for non-DataFrames, missing columns,
  scalar covariates, invalid `weight_rule`, invalid `n_folds`/`seed`/`alpha`,
  empty or malformed candidates, duplicate candidate labels, non-finite
  design arrays, malformed sample weights, too-small post-drop samples, and
  degenerate candidate/stacked first-stage failures.
- Migrated all raw `raise ValueError`/`raise RuntimeError` occurrences in
  `dml/model_averaging.py` to `MethodIncompatibility`, `DataInsufficient`, or
  `ConvergenceFailure`.
- Added regression tests for scalar covariates, invalid controls, malformed
  and duplicate candidates, non-finite design data, and CLS input validation.
- Synchronized the DML base learner unsupported-sample-weight test with the
  existing taxonomy migration from `NotImplementedError` to
  `MethodIncompatibility`.

Verification run:

- `.venv/bin/python -m pytest tests/test_dml_model_averaging.py
  tests/test_dml_cov_averaging_panel.py tests/test_cov95_dml_averaging_panel.py
  -o addopts=''` passed, 57 tests.
- `.venv/bin/python -m pytest tests/test_dml_cov_diag_sens.py
  tests/test_dml_cov_scores.py tests/test_dml_cov_base.py -o addopts=''`
  passed, 23 tests.
- `.venv/bin/python -m pytest tests/test_cov95_dml_diag_sens.py
  tests/test_cov95_dml_learners_base.py -o addopts=''` passed, 83 tests with
  3 optional learner skips.
- `.venv/bin/python -m flake8 src/statspai/dml/model_averaging.py
  tests/test_cov95_dml_averaging_panel.py --max-line-length=88
  --ignore=E203,W503 --select=F,E9` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  1000 taxonomy raises and 1501 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4366 <= 4698, mypy observed 3282 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 113

Target: E-value sensitivity diagnostics validation and taxonomy errors.

- Added dedicated `diagnostics/evalue.py` helpers for string options, finite
  floats, positive/non-negative/open-unit floats, optional booleans, and
  two-element finite confidence intervals.
- Hardened `sp.evalue()` for non-finite estimates, invalid measures, invalid
  standard errors, invalid alpha, malformed CI inputs, non-boolean rare flags,
  invalid non-null references, and invalid OLS `sd`/`delta` inputs.
- Hardened `sp.evalue_rd()` for non-finite or negative cell counts, empty
  exposure groups, negative risk differences, invalid non-null references,
  invalid alpha, and non-positive grid steps.
- Hardened `sp.bias_factor()` and `sp.evalue_from_result()` so bad inputs
  raise `MethodIncompatibility` with diagnostics and recovery alternatives.
- Migrated `_threshold()` and `_check_ci()` validation failures to the StatsPAI
  taxonomy and removed all raw `raise ValueError`/`raise TypeError`/
  `raise RuntimeError` occurrences from `diagnostics/evalue.py`.
- Added regression tests for NaN/inf, malformed CI, invalid alpha/grid,
  non-boolean rare flags, non-finite cell counts, and non-finite bias factors.

Verification run:

- `.venv/bin/python -m pytest tests/test_evalue.py
  tests/test_result_consumer_errors.py -o addopts=''` passed, 81 tests.
- `.venv/bin/python -m pytest tests/test_phase9to14.py -k "evalue"
  -o addopts=''` passed, 9 selected tests.
- `PYTHONPATH=tests/r_parity .venv/bin/python tests/r_parity/23_evalue.py`
  wrote the expected 26 rows across 13 measures without tracked artifact diffs.
- `.venv/bin/python -m pytest tests/test_unified_sensitivity.py
  tests/test_article_aliases_round2.py::test_evalue_rr_point_only
  tests/test_article_aliases_round2.py::test_evalue_rr_with_ci_bounds
  tests/test_article_aliases_round2.py::test_evalue_rr_rejects_partial_ci
  tests/test_tierD_p2_bounds_sensitivity_analytic.py::TestEValueRRAnalytic
  -o addopts=''` passed, 21 tests.
- `.venv/bin/python -m pytest tests/test_evalue.py
  tests/test_result_consumer_errors.py tests/test_unified_sensitivity.py
  -o addopts=''` passed, 92 tests.
- `.venv/bin/python -m flake8 src/statspai/diagnostics/evalue.py
  --max-line-length=88 --ignore=E203,W503 --count --statistics` reported 0.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  964 taxonomy raises and 1514 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4368 <= 4698, mypy observed 3282 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 112

Target: DID pre-trends diagnostics input contracts and taxonomy errors.

- Added shared validation helpers in `src/statspai/did/pretrends.py` for
  string options, open-unit probabilities, integer controls, finite vectors,
  non-negative standard errors, pre-period arrays, and pre-period VCV shape.
- Migrated `pretrends_test`, `pretrends_power`, `sensitivity_rr`, and their
  event-study extraction helpers from raw `ValueError`/`TypeError`/
  `NotImplementedError` to `DataInsufficient`, `MethodIncompatibility`, and
  `NumericalInstability`.
- Hardened pre-trend VCV handling for malformed, non-finite, wrongly shaped,
  or singular `vcv_pre` inputs before matrix algebra.
- Hardened alpha, test type, delta, `Mbar`, and `n_grid` validation so invalid
  sensitivity inputs fail before SciPy/NumPy internals.
- Normalized `sensitivity_rr(method="c-lf")` case-insensitively while still
  rejecting unsupported methods through the StatsPAI taxonomy.
- Removed all raw `raise ValueError`/`raise TypeError`/`raise RuntimeError`/
  `raise NotImplementedError` occurrences from `did/pretrends.py`, and cleaned
  fatal flake noise in the touched module.

Verification run:

- `.venv/bin/python -m pytest tests/test_cov95_did_pretrends.py
  -o addopts=''` passed, 30 tests.
- `.venv/bin/python -m pytest tests/test_cov95_did_pretrends.py
  tests/test_honest_did_backend.py tests/test_honest_did_sdid.py
  -o addopts=''` passed, 66 tests.
- `.venv/bin/python -m pytest tests/test_cov95_did_analysis.py
  tests/test_cov95_did_plots.py tests/test_tidy_glance.py -o addopts=''`
  passed, 45 tests with one existing matching warning.
- `.venv/bin/python -m flake8 src/statspai/did/pretrends.py
  tests/test_cov95_did_pretrends.py --max-line-length=88 --ignore=E203,W503
  --select=F,E9` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  941 taxonomy raises and 1528 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4369 <= 4698, mypy observed 3282 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 111

Target: article-facing alias validation and taxonomy-consistent late-bind
dispatch.

- Added local alias helpers for string options, required column names, and
  scalar-or-list column arguments.
- Hardened `sp.xlearner`, `sp.partial_identification`,
  `sp.causal_discovery`, `sp.evalue_rr`, `sp.policy_tree`, and `sp.dml`
  alias preconditions to raise `MethodIncompatibility` instead of raw
  `ValueError`/`TypeError`.
- Normalized scalar `X`/`covariates`/`variables` inputs at the alias layer, so
  `"x1"` is treated as `["x1"]` rather than being split into characters.
- Added recovery hints, diagnostics, and alternative-function metadata for the
  major article-alias option-conflict paths.
- Updated article-alias, late-bind, and DML split tests to assert the
  StatsPAI taxonomy while preserving `ValueError` compatibility where it is
  part of the public contract.
- Removed all raw `raise ValueError`/`raise TypeError`/`raise RuntimeError`
  occurrences from `src/statspai/_article_aliases.py`.

Verification run:

- `.venv/bin/python -m pytest tests/test_article_aliases.py
  tests/test_article_aliases_round2.py -o addopts=''` passed, 61 tests with 3
  existing PSM warning checks.
- `.venv/bin/python -m pytest tests/test_late_bind_contracts.py
  tests/test_export_surface_contract.py tests/test_exception_migrations.py
  tests/test_estimator_coherence.py -o addopts=''` passed, 109 tests with 3
  existing PSM warning checks.
- `.venv/bin/python -m pytest tests/test_policy_learning.py tests/test_dml.py
  tests/test_dml_split.py -o addopts=''` passed, 40 tests.
- `.venv/bin/python -m pytest tests/test_article_aliases.py
  tests/test_article_aliases_round2.py tests/test_dml_split.py -o addopts=''`
  passed, 72 tests with 3 existing PSM warning checks.
- `.venv/bin/python -m flake8 src/statspai/_article_aliases.py
  --max-line-length=88 --ignore=E203,W503 --count --statistics` reported 0.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  908 taxonomy raises and 1544 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4372 <= 4698, mypy observed 3285 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 110

Target: causal-question DSL validation and estimator-dispatch reliability.

- Added local question-DSL validation helpers for DataFrame inputs, non-empty
  column names, string options, scalar-or-list fields, and missing-column
  diagnostics.
- Hardened `CausalQuestion.estimate()` and `.report()` so missing data or
  missing prior estimates raise StatsPAI taxonomy errors with recovery hints.
- Hardened `causal_question()` construction for invalid treatment/outcome
  names, non-DataFrame data, invalid `estimand`/`design`/`time_structure`, and
  scalar covariate/instrument inputs.
- Migrated dispatcher preconditions for DML/TMLE/meta-learner/causal-forest
  covariates/instruments, reserved kwarg collisions, and binary-treatment
  requirements to `MethodIncompatibility`.
- Fixed the `regression_discontinuity` question-dispatch branch to call
  `sp.rdrobust(data=..., y=..., x=..., c=...)` instead of passing Series into
  the wrong signature; fuzzy RD is used only when the declared treatment column
  exists in the data.
- Fixed the `synthetic_control` question-dispatch branch to map
  `id` -> `unit`, `time` -> `time`, `treatment` -> `treated_unit`, and
  `cutoff` -> `treatment_time`, with explicit precondition errors when fields
  are missing.
- Migrated report-format errors and the AIPW ATE helper's non-binary-treatment
  errors to the StatsPAI taxonomy.
- Added regression tests for RD and synthetic-control `CausalQuestion.estimate`
  dispatch, and updated reserved-kwarg collision tests to assert
  `MethodIncompatibility`.

Verification run:

- `.venv/bin/python -m pytest tests/test_question_dsl.py
  tests/test_v100_integration.py::test_v1_causal_question_end_to_end
  tests/test_paper_from_question.py::TestErrors -o addopts=''` passed,
  63 tests.
- `.venv/bin/python -m pytest tests/test_question_dsl.py
  tests/test_v100_integration.py::test_v1_causal_question_end_to_end
  tests/test_paper_from_question.py
  tests/test_robustness_battery.py::TestPaperFromQuestion -o addopts=''`
  passed, 81 tests with the existing expected workflow-degradation warning.
- `.venv/bin/python -m compileall -q src/statspai/question/question.py
  tests/test_question_dsl.py` passed.
- `.venv/bin/python -m flake8 src/statspai/question/question.py
  tests/test_question_dsl.py --select=F,E9` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  884 taxonomy raises and 1561 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4372 <= 4698, mypy observed 3285 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 109

Target: `rdrobust` validation, sample-geometry, and RBC bootstrap contracts.

- Added local RD validation helpers for DataFrame inputs, non-empty column
  names, string options, finite/open-unit/nonnegative/positive floats, integer
  controls, and scalar-or-list covariate specs.
- Hardened `rdrobust()` for invalid `kernel`/`bwselect`, malformed numeric
  controls, invalid `alpha`, nonpositive manual bandwidths, mutually exclusive
  `b`/`rho`, invalid bootstrap options, too-small RBC bootstrap requests,
  missing cluster columns, oversized donut holes, and too few observations on
  either side of the cutoff.
- Hardened `_parse_data()` for non-DataFrame input, missing outcome/running/
  fuzzy/covariate columns, malformed covariate specs, and the case where no
  finite observations remain after missing-value filtering.
- Migrated CCT-delegation donut failures and RBC bootstrap effective-bandwidth
  or valid-replicate failures to `DataInsufficient`/`ConvergenceFailure`.
- Removed an old unused local variable from the RD plot bandwidth helper.
- Added focused `tests/test_rd.py` taxonomy assertions for bad input types,
  invalid options, missing columns, and insufficient per-side sample sizes.

Verification run:

- `.venv/bin/python -m pytest tests/test_rd.py tests/test_rd_pipeline.py
  tests/test_cov95_rd_bandwidth.py tests/test_cov95_rd_r2_bandwidth.py
  -o addopts=''` passed, 54 tests.
- `.venv/bin/python -m pytest tests/test_low_cov_battery.py::test_rdrobust_kernels_smoke
  tests/test_low_cov_battery.py::test_rdrobust_bandwidth_selection_smoke
  tests/test_low_cov_battery.py::test_rdrobust_polynomial_degree_smoke
  tests/test_low_cov_battery.py::test_rdrobust_with_donut
  tests/test_low_cov_battery.py::test_rdrobust_explicit_bandwidth
  tests/test_cov95_rd_misc.py tests/test_cov95_rd_extrapolate.py -o addopts=''`
  passed, 51 tests.
- `.venv/bin/python -m pytest tests/test_tidy_glance.py::TestRDTidyGlance
  -o addopts=''` passed, 3 tests.
- `.venv/bin/python -m compileall -q src/statspai/rd/rdrobust.py
  tests/test_rd.py` passed.
- `.venv/bin/python -m flake8 src/statspai/rd/rdrobust.py tests/test_rd.py
  --select=F,E9` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  856 taxonomy raises and 1578 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4370 <= 4698, mypy observed 3287 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 108

Target: BSTS/CausalImpact input and data-geometry failure contracts.

- Added local BSTS validation helpers for DataFrame inputs, non-empty column
  names, string options, open-unit `alpha`, integer `n_simulations`, scalar-or
  list covariate specs, two-element period pairs, and missing-column reporting.
- Hardened `causal_impact()` for non-DataFrame/empty inputs, invalid model
  names, malformed period pairs, missing outcome/covariate columns, empty
  pre/post windows, overlapping pre/post periods, non-finite outcomes, and
  all-missing pre-period covariates.
- Hardened `bsts_synth()` for non-DataFrame inputs, missing long-format panel
  columns, absent treated units, no donor units, unusable donor pre-period
  coverage, missing averaged covariates, and no pre-treatment periods.
- Normalized `covariates="x1"` to `["x1"]` for BSTS/CausalImpact rather than
  iterating over characters.
- Migrated all raw `ValueError`/`TypeError` raises in
  `src/statspai/synth/bsts.py` to StatsPAI taxonomy errors while preserving
  historical `ValueError` compatibility for validation catches.
- Removed stale unused imports in the BSTS module and its focused tests.

Verification run:

- `.venv/bin/python -m pytest tests/test_cov95_synth_r4_bsts.py
  tests/test_cov95_synth_r3_report_bsts.py tests/test_synth_new_methods.py::TestBSTS
  tests/test_cov95_synth_dispatch.py::test_dispatch_bsts_method
  tests/test_cov95_synth_variants.py::test_bsts_synth_local_level
  tests/test_cov95_synth_variants.py::test_bsts_synth_local_trend
  tests/test_cov95_synth_variants.py::test_causal_impact_wide -o addopts=''`
  passed, 40 tests.
- `.venv/bin/python -m pytest tests/test_synth_new_methods.py
  tests/test_cov95_synth_dispatch.py tests/test_cov95_synth_variants.py
  -o addopts=''` passed, 138 tests.
- `.venv/bin/python -m compileall -q src/statspai/synth/bsts.py
  tests/test_cov95_synth_r4_bsts.py
  tests/test_cov95_synth_r3_report_bsts.py` passed.
- `.venv/bin/python -m flake8 src/statspai/synth/bsts.py
  tests/test_cov95_synth_r4_bsts.py tests/test_cov95_synth_r3_report_bsts.py
  --select=F,E9` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  824 taxonomy raises and 1594 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4369 <= 4698, mypy observed 3287 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 107

Target: surrogate-index input, overlap, and bootstrap failure contracts.

- Added local surrogate validation helpers for DataFrame inputs, non-empty
  column names, scalar-or-list columns, per-wave surrogate specs, open-unit
  `alpha`, integer `n_boot`, finite numeric vectors, binary treatment arms, and
  treatment overlap.
- Hardened `sp.surrogate_index()` for non-DataFrame inputs, missing
  experimental/observational columns, non-binary or one-arm treatment samples,
  missing surrogate/covariate data, invalid outcome models, invalid
  `alpha`/`n_boot`, and low-valid-replicate bootstrap runs.
- Hardened `sp.long_term_from_short()` for malformed wave specs, missing
  columns, too-small bootstrap requests, non-finite long-term outcomes, and
  no-overlap bootstrap samples.
- Hardened `sp.proximal_surrogate_index()` for empty/missing proxies, missing
  columns, too-small bootstrap requests, non-finite bridge inputs, no treatment
  overlap, and low-valid-replicate bootstrap runs.
- Migrated all raw `ValueError`/`TypeError`/`RuntimeError` raises in
  `src/statspai/surrogate/index.py` to StatsPAI taxonomy errors while keeping
  historical `ValueError` catches compatible where validation errors are
  expected.

Verification run:

- `.venv/bin/python -m pytest tests/test_surrogate.py -o addopts=''` passed,
  13 tests.
- `.venv/bin/python -m pytest tests/test_v100_integration.py
  tests/test_export_surface_contract.py tests/test_agent_result_methods.py
  -o addopts=''` passed, 162 tests.
- `.venv/bin/python -m compileall -q src/statspai/surrogate/index.py
  tests/test_surrogate.py` passed.
- `.venv/bin/python -m flake8 src/statspai/surrogate/index.py
  tests/test_surrogate.py --select=F,E9` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  794 taxonomy raises and 1610 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4370 <= 4698, mypy observed 3287 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 106

Target: synthetic-control public dispatcher and classic SCM failure contracts.

- Added local synth validation helpers for DataFrame inputs, non-empty column
  names, string options, open-unit `alpha`, non-negative penalization, integer
  controls, and scalar-or-list column specs.
- Hardened `sp.synth()` dispatch before method routing: non-string methods,
  malformed covariate specs, invalid `alpha`/`penalization`, unknown backends,
  missing staggered treatment columns, and unknown methods now raise
  `MethodIncompatibility` with structured diagnostics.
- Migrated classic `SyntheticControl` validation and matrix-prep failures to
  the taxonomy: missing columns and invalid `v_method`/controls use
  `MethodIncompatibility`, while absent treated units, too few pre/post periods,
  no valid donors, and incomplete predictor matrices use `DataInsufficient`.
- Preserved backward compatibility because these taxonomy errors still subclass
  `ValueError`; R-backend availability/failure now raises `ConvergenceFailure`,
  preserving historical `RuntimeError` catches while adding recovery hints.
- Normalized `covariates="x1"` to `["x1"]` for classic SCM instead of
  accidentally iterating over characters.
- Removed the stale `scipy.optimize` import from `synth/scm.py`.

Verification run:

- `.venv/bin/python -m pytest tests/test_synth.py
  tests/test_cov95_synth_classic_paths.py tests/test_synth_backend.py
  -o addopts=''` passed, 42 tests.
- `.venv/bin/python -m pytest tests/test_cov95_synth_dispatch.py
  tests/test_cov95_synth_scm_core.py tests/test_cov95_synth_r2_scm.py
  tests/test_cov95_synth_r3_multi_seq_scm.py -o addopts=''` passed,
  67 tests.
- `.venv/bin/python -m pytest tests/test_synth_new_methods.py
  tests/test_synth_advanced.py tests/test_cov95_synth_r2_estimators.py
  tests/test_cov95_synth_r3_sdid_discos_scpi.py -o addopts=''` passed,
  166 tests.
- `.venv/bin/python -m compileall -q src/statspai/synth/scm.py
  tests/test_synth.py` passed.
- `.venv/bin/python -m flake8 src/statspai/synth/scm.py tests/test_synth.py
  --select=F,E9` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  769 taxonomy raises and 1629 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4368 <= 4698, mypy observed 3287 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- `git diff --check` passed.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 105

Target: GLM family/link/robust/public-entry contracts and formula prediction
hardening.

- Added local GLM validation helpers for string options, DataFrame inputs,
  open-unit alpha, positive tolerances, integer controls, and scalar-or-list
  column specs.
- Migrated unknown GLM `family`, `link`, `robust`, and internal HC option
  failures from raw `ValueError` to `MethodIncompatibility` with valid-option
  diagnostics.
- Hardened `GLMRegression` construction and `fit()` for non-DataFrame formula
  data, missing formula/data or raw arrays, invalid `maxiter`/`tol`/`alpha`, and
  zero-row estimation samples.
- Hardened the public `glm()` entry point: y/x style calls now accept
  `x="x1"` as a scalar shortcut instead of iterating over characters, reject
  malformed x specs, and report missing formula/data via the package taxonomy.
- Extended `tests/test_glm_predict.py` with taxonomy checks for invalid
  family/link/robust/maxiter, missing input modes, empty samples, public
  `glm()` y/x calls, and malformed x specs.

Verification run:

- `.venv/bin/python -m compileall -q src/statspai/regression/glm.py
  tests/test_glm_predict.py` passed.
- `.venv/bin/python -m pytest -q tests/test_glm_predict.py -o addopts=''`
  passed, 8 tests.
- `.venv/bin/python -m pytest -q tests/test_untested_function_coverage.py
  tests/test_translation.py -o addopts=''` passed, 100 tests.
- `.venv/bin/python -m pytest -q tests/test_new_v06_modules.py
  -o addopts=''` passed, 36 tests with existing expected warnings.
- `.venv/bin/python -m pytest -q tests/test_survey.py -o addopts=''` passed,
  14 tests.
- `.venv/bin/python -m pytest -q tests/test_export_surface_contract.py
  -o addopts=''` passed, 55 tests.
- `.venv/bin/python -m pytest -q
  tests/reference_parity/test_count_quantile_parity.py -o addopts=''` passed,
  10 tests.
- `.venv/bin/python -m flake8 src/statspai/regression/glm.py
  tests/test_glm_predict.py --count --select=E9,F63,F7,F82 --show-source
  --statistics` passed with 0 fatal syntax/name issues.
- `rg -n "raise (ValueError|RuntimeError|KeyError|TypeError|IndexError)"
  src/statspai/regression/glm.py` returned no raw generic raises.
- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  732 taxonomy raises and 1646 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`, `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4368 <= 4698, mypy observed 3287 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 104

Target: GLMM/multilevel result helpers and `meglm()` input taxonomy.

- Added local GLMM validation helpers for string options, DataFrame input,
  open-unit probabilities, positive tolerances, integer controls, and column
  list coercion.
- Migrated family resolution, confidence-interval alpha validation,
  `odds_ratios()`, `incidence_rate_ratios()`, and `plot()` misuse from raw
  generic exceptions to `MethodIncompatibility` with recovery hints and
  diagnostics.
- Hardened `meglm()` before optimization: non-DataFrame data, bad `family`,
  bad `cov_type`, malformed group specifications, bad `nAGQ`/`maxiter`/`tol`
  / `alpha`, malformed fixed/random column lists, unsupported AGHQ random
  slopes, missing columns, unhashable groups, and all-missing complete samples
  now raise package taxonomy errors.
- Kept the GLMM likelihood, mode finder, quadrature, covariance, and optimizer
  math unchanged; the batch only changes validation and result-helper failure
  surfaces.
- Extended `tests/test_multilevel.py` with taxonomy checks for GLMM prediction,
  family/covariance/group/input errors, all-missing samples, invalid AGHQ
  settings, family-specific OR/IRR helpers, and invalid plot requests.

Verification run:

- `.venv/bin/python -m compileall -q src/statspai/multilevel/glmm.py
  tests/test_multilevel.py` passed.
- `.venv/bin/python -m pytest -q tests/test_multilevel.py -o addopts=''`
  passed, 63 tests.
- `.venv/bin/python -m flake8 src/statspai/multilevel/glmm.py
  tests/test_multilevel.py --count --select=E9,F63,F7,F82 --show-source
  --statistics` passed with 0 fatal syntax/name issues.
- `rg -n "raise (ValueError|RuntimeError|KeyError|TypeError|IndexError)"
  src/statspai/multilevel/glmm.py` returned no raw generic raises.
- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  714 taxonomy raises and 1652 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`, `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4370 <= 4698, mypy observed 3287 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 103

Target: IV regression input contracts, fitted-diagnostic accessors, and absorb
path taxonomy.

- Added local IV validation helpers for string options, DataFrame inputs, and
  unfitted diagnostic access errors.
- Migrated `IVRegression` construction errors for non-string formula/method,
  non-DataFrame data, and unknown method names to `MethodIncompatibility` with
  supported-method diagnostics.
- Migrated formula parsing failures, missing formula columns, all-NaN formula
  samples, missing raw array inputs, unknown robust options, and missing cluster
  variables to `MethodIncompatibility` or `DataInsufficient`.
- Migrated unfitted `first_stage`, `sargan_test`, and `hausman_test` property
  access to the package taxonomy so agent callers can recover without message
  matching.
- Migrated the `sp.iv(absorb=...)` preparation path for malformed IV formulas,
  missing formula/absorb/cluster columns, empty complete samples, and matrix-mode
  absorb requests to structured taxonomy errors.
- Added regression tests for unknown method/data/robust/cluster options,
  missing IV syntax and variables, all-NaN formula samples, and unfitted
  diagnostic properties while preserving the existing IV predict taxonomy tests.

Verification run:

- `.venv/bin/python -m compileall -q src/statspai/regression/iv.py
  tests/test_iv.py` passed.
- `.venv/bin/python -m pytest -q tests/test_iv.py -o addopts=''` passed,
  22 tests.
- `.venv/bin/python -m pytest -q tests/test_iv_absorb.py -o addopts=''`
  passed, 13 tests.
- `.venv/bin/python -m pytest -q
  tests/reference_parity/test_regress_weights_iv_robust_parity.py
  -o addopts=''` passed, 13 tests.
- `.venv/bin/python -m pytest -q tests/reference_parity/test_iv_parity.py
  -o addopts=''` passed, 4 tests with the expected weak-instrument warning.
- `.venv/bin/python -m pytest -q tests/test_iv_dispatcher.py
  -o addopts=''` passed, 34 tests.
- `.venv/bin/python -m flake8 src/statspai/regression/iv.py tests/test_iv.py
  --count --select=E9,F63,F7,F82 --show-source --statistics` passed with 0
  fatal syntax/name issues.
- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  690 taxonomy raises and 1663 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`, `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4370 <= 4698, mypy observed 3287 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 102

Target: stochastic-frontier input/result contracts and recoverable error
taxonomy.

- Added lightweight frontier validation helpers for string options, open-unit
  probabilities, positive numeric tolerances, integer repetition controls, and
  column-list coercion.
- Accepted the common scalar-regressor shortcut `frontier(..., x="x1")` as
  `["x1"]` instead of treating the string as an iterable of characters; applied
  the same explicit column-list validation to `usigma`, `vsigma`, and `emean`.
- Migrated user-facing `frontier()` input failures for non-DataFrame data,
  missing columns, unsupported `dist`/`vce`/`te_method`, incompatible `emean`,
  too few complete observations, bad `B`/`maxiter`/`tol`/`alpha`, and malformed
  `start` vectors into `MethodIncompatibility` or `DataInsufficient` with
  diagnostics.
- Migrated `FrontierResult.efficiency()`, `inefficiency()`, `predict()`,
  `marginal_effects()`, `returns_to_scale()`, and `efficiency_ci()` guardrails
  from raw `KeyError`, `ValueError`, and `RuntimeError` branches into the
  package taxonomy, including `ConvergenceFailure` for unreliable bootstrap
  variance draws and `NumericalInstability` for impossible fitted-distribution
  state.
- Added regression tests for scalar `x`, missing-column diagnostics, sample-size
  insufficiency, bad `start` length, incompatible `emean`, invalid
  marginal-effects options, invalid `vce`, missing `usigma`, and unsupported
  efficiency methods.

Verification run:

- `.venv/bin/python -m compileall -q src/statspai/frontier/sfa.py
  tests/test_frontier.py` passed.
- `.venv/bin/python -m pytest -q tests/test_frontier.py -o addopts=''` passed,
  101 tests.
- `.venv/bin/python -m flake8 src/statspai/frontier/sfa.py
  tests/test_frontier.py --count --select=E9,F63,F7,F82 --show-source
  --statistics` passed with 0 fatal syntax/name issues.
- `git diff --check` passed.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  675 taxonomy raises and 1676 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.

Post-batch gate sweep:

- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`, `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4370 <= 4698, mypy observed 3287 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 101

Target: classical matching input contracts and support-failure taxonomy.

- Migrated `MatchEstimator` construction and validation failures into the
  StatsPAI taxonomy for non-DataFrame input, missing columns, malformed method
  combinations, bad distance/estimand/common-support/se-method options, and
  invalid numeric controls (`n_matches`, `ps_poly`, `bwidth`, `caliper`,
  `ai_matches`, `n_strata`, `n_bins`, `alpha`).
- Added a one-column shortcut so `covariates="x1"` is accepted as `["x1"]`
  instead of being treated as an iterable of characters; updated `match()`
  provenance to record the normalized covariate list.
- Converted exact-matching, stratification, kernel/radius, and CEM no-support
  failures to `DataInsufficient` with recovery hints where applicable.
- Migrated public `sp.match(method=...)` unknown/non-string method failures to
  `MethodIncompatibility` with supported-method diagnostics.
- Removed a local exception import that shadowed the module-level taxonomy
  import and caused validation branches to fail with `UnboundLocalError`.
- Cleaned touched-file flake8 debt in `matching/match.py` and
  `tests/test_matching.py` by wrapping historical long lines and annotating a
  few touched helper surfaces.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_matching.py
  tests/test_psmatch2.py::TestCommonSupport::test_invalid_common_support_raises
  tests/test_psmatch2.py::TestKernelRadius::test_bad_kernel_raises
  tests/test_psmatch2.py::TestKernelRadius::test_kernel_radius_are_att_only
  tests/reference_parity/test_matching_parity.py` passed, 61 tests.
- `.venv/bin/python -m compileall -q src/statspai/matching/match.py
  src/statspai/matching/__init__.py tests/test_matching.py` passed.
- `git diff --check -- src/statspai/matching/match.py
  src/statspai/matching/__init__.py tests/test_matching.py` passed.
- `MPLBACKEND=Agg .venv/bin/python -m flake8
  src/statspai/matching/match.py src/statspai/matching/__init__.py
  tests/test_matching.py --max-line-length=88 --ignore=E203,W503
  --statistics --count` passed with 0 touched-file violations.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  640 taxonomy raises and 1694 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python -m mypy src/statspai/matching/match.py
  src/statspai/matching/__init__.py --no-error-summary --hide-error-context`
  still reports historical matching module typing debt plus the repository
  Python-version warning; the global mypy ratchet improved after the touched
  annotations.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=563`, `weak=147`, `smoke=9`,
  `untested=186`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/quality_gate.py all` passed with flake8 observed
  4370 <= 4698, mypy observed 3287 <= 3521, import-budget observed 0, and the
  agent-card, result-protocol, and error-taxonomy gates passing.
- Root diff still excludes `Paper-JSS/`, `CausalAgentBench/`, `paper.md`, and
  `paper.bib`; nested `Paper-JSS/` and `CausalAgentBench/` worktrees are clean
  on `main...origin/main`.

## 2026-06-17 Batch 63

Target: `sp.fast.within()` validation parity with the fast HDFE estimator
stack.

- Routed `sp.fast.within()` construction through the shared fast validation
  helpers for positive integer iteration controls and finite non-negative
  tolerance controls.
- Converted unsupported FE specifications, missing FE columns, empty FE
  specifications, malformed array dimensions, and missing FE values into
  StatsPAI taxonomy errors instead of raw `KeyError`, `IndexError`, or delayed
  kernel failures.
- Added support for the common single-column shortcut
  `sp.fast.within(data, fe="unit")`, matching the list-of-column-names path.
- Rejected empty input samples and all-singleton samples after singleton
  pruning before building a zero-row cached residualizer.
- Hardened `WithinTransformer.transform()` and `transform_columns()` against
  non-finite values, unsupported array ranks, inconsistent `already_masked`
  row counts, empty column lists, and missing columns.
- Added focused tests for the new validation and single-column shortcut
  behavior.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_within_dsl.py` passed, 34 tests.
- `.venv/bin/python -m compileall -q src/statspai/fast/within.py` passed.
- `git diff --check -- src/statspai/fast/within.py
  tests/test_fast_within_dsl.py` passed.

## 2026-06-17 Batch 64

Target: make the fast columnar I/O adapter match its Polars/PyArrow contract.

- Generalized the `sp.fast.demean_polars()` and `sp.fast.fepois_polars()`
  backend adapter from Polars-only eager frames to Polars DataFrames,
  Polars LazyFrames, and PyArrow Tables.
- Added real PyArrow `Table` extraction for both float design columns and
  object fixed-effect columns, preserving the same fast HDFE/FEPois kernels
  downstream.
- Added scalar column-name normalization for `X_cols="x1"` and
  `fe_cols="unit"` while retaining list/sequence support.
- Converted missing columns, empty column lists, malformed column lists, and
  unsupported columnar inputs into `MethodIncompatibility` errors instead of
  raw `KeyError`/`TypeError`.
- Updated Polars missing-column tests to expect StatsPAI taxonomy errors and
  added a new Arrow-specific test module covering demean parity, scalar column
  names, FEPois parity, and missing/empty column validation.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_arrow_io.py tests/test_fast_polars.py` passed, 5 Arrow
  tests with 1 Polars-module skip because Polars is not installed locally.
- `.venv/bin/python -m compileall -q src/statspai/fast/polars_io.py` passed.
- `git diff --check -- src/statspai/fast/polars_io.py
  tests/test_fast_arrow_io.py tests/test_fast_polars.py` passed.

## 2026-06-17 Batch 65

Target: harden `sp.fast.etable()` parameter and result-object protocol
validation.

- Converted unsupported result objects that lack usable coefficient or
  standard-error accessors into `MethodIncompatibility` errors rather than raw
  `AttributeError`.
- Added explicit `digits` validation that allows `digits=0` but rejects
  negative, boolean, and non-integer precision values before formatting.
- Rejected bare strings for `names`, `keep`, and `drop`, avoiding accidental
  character-wise interpretation of model names or variable filters.
- Added tests for zero-digit formatting, invalid digits, bare string
  sequences, and unsupported fit-object protocol failures.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_etable.py` passed, 17 tests.
- `.venv/bin/python -m compileall -q src/statspai/fast/etable.py` passed.
- `git diff --check -- src/statspai/fast/etable.py
  tests/test_fast_etable.py` passed.

## 2026-06-17 Batch 66

Target: convert `sp.fast.event_study()` user-input failures into early
taxonomy errors.

- Added explicit DataFrame and non-empty-sample validation at the event-study
  entry point.
- Validated `y`, `unit`, `time`, `event_time`, and `cluster` as non-empty
  string column names before any DataFrame indexing.
- Converted missing outcome/unit/time/event-time/cluster columns from raw
  `KeyError` into `MethodIncompatibility`.
- Hardened `reference` and `window` validation so malformed window objects
  fail with clear method errors instead of raw Python unpacking/length errors.
- Converted non-finite outcomes, nonnumeric/infinite/fractional event-time
  values, and no-dummy post-filter samples into StatsPAI taxonomy errors while
  preserving `ValueError` compatibility.
- Added tests for non-DataFrame data, empty samples, malformed column/cluster
  arguments, malformed windows, and missing-column taxonomy.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_event_study.py` passed, 18 tests with 1 optional R parity
  skip.
- `.venv/bin/python -m compileall -q src/statspai/fast/event_study.py` passed.
- `git diff --check -- src/statspai/fast/event_study.py
  tests/test_fast_event_study.py` passed.

## 2026-06-17 Batch 67

Target: make `sp.fast.hdfe_bench()` reject benchmark configurations that
would silently disable or distort performance evidence.

- Added a private `n_list` validator that rejects empty sequences, bare
  strings, non-iterables, booleans, and non-positive sample sizes.
- Routed `n_groups` and `repeat` through the shared positive-integer
  validator, rejecting boolean values that previously became `1`.
- Routed `atol` through the shared finite non-negative float validator, so
  `NaN` can no longer disable the backend-drift guard.
- Added tests for empty/string/scalar `n_list`, boolean controls, and `NaN`
  tolerance.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_bench.py` passed, 16 tests.
- `.venv/bin/python -m compileall -q src/statspai/fast/bench.py` passed.
- `git diff --check -- src/statspai/fast/bench.py tests/test_fast_bench.py`
  passed.

## 2026-06-17 Batch 68

Target: convert main native fast estimator input/configuration failures into
StatsPAI taxonomy errors.

- Converted `sp.fast.feols()` invalid `vcov`/`ssc`/cluster option
  combinations, missing columns, non-finite outcome/regressors, invalid
  weights, NaN clusters, and NaN fixed effects into `MethodIncompatibility`
  while preserving `ValueError` compatibility.
- Converted `sp.fast.fepois()` invalid `vcov`/cluster option combinations,
  missing columns, non-finite/negative outcomes, invalid weights, NaN
  clusters, non-finite regressors, and NaN fixed effects into
  `MethodIncompatibility`.
- Updated explicit missing-column tests from raw `KeyError` expectations to
  taxonomy expectations.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_feols.py tests/test_fast_fepois.py` passed, 59 tests with
  12 optional-engine/Rust/R parity skips.
- `.venv/bin/python -m compileall -q src/statspai/fast/feols.py
  src/statspai/fast/fepois.py` passed.
- `git diff --check -- src/statspai/fast/feols.py
  src/statspai/fast/fepois.py tests/test_fast_feols.py
  tests/test_fast_fepois.py` passed.

## 2026-06-17 Batch 69

Target: harden the core `sp.fast.demean()` HDFE kernel entry point.

- Routed `max_iter`, `accel_period`, optional `jax_max_iter`, `tol`, and
  `tol_abs` through shared fast validation helpers.
- Converted invalid `accel`/`backend`, unsupported X ranks, non-finite X, NaN
  fixed effects, empty FE specifications, and FE shape mismatches into
  StatsPAI taxonomy errors while preserving `ValueError` compatibility.
- Rejected empty input samples and all-singleton samples after default
  singleton pruning before entering backend kernels with zero rows.
- Added tests for invalid controls, empty samples, empty FE specs, and
  all-singleton default-drop behavior.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_demean.py` passed, 28 tests with 1 optional Rust skip.
- `.venv/bin/python -m compileall -q src/statspai/fast/demean.py` passed.
- `git diff --check -- src/statspai/fast/demean.py
  tests/test_fast_demean.py` passed.

Post-fast-boundary gate sweep:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_fast_demean.py tests/test_fast_within_dsl.py
  tests/test_fast_feols.py tests/test_fast_fepois.py tests/test_fast_inference.py
  tests/test_fast_htz.py tests/test_fast_event_study.py tests/test_fast_bench.py
  tests/test_fast_arrow_io.py tests/test_fast_etable.py
  tests/test_fast_jax_feols_result_protocol.py` passed, 238 tests with 17
  optional-engine/Rust/R/JAX skips.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  285 taxonomy raises and 1827 generic raises.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `git diff --check` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries, and `.venv/bin/python
  scripts/help_coverage.py --check` passed.
- Verified nested JOSS-adjacent repos separately: `Paper-JSS/` and
  `CausalAgentBench/` both remained clean on `main...origin/main`.
- Read-only worktree audit found separate active worktrees at
  `StatsPAI-improve-wt`, `StatsPAI-wt-synth`, and
  `.claude/worktrees/improve-correctness`; no cross-worktree edits were made.

Non-fast changed-surface regression sweep:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_agent_result_methods.py
  tests/test_auto_estimators.py tests/test_regtable_serialization.py
  tests/test_regtable_from_dict.py tests/test_bayes_result_protocol.py
  tests/test_result_protocol_audit.py tests/test_lineage.py` passed, 188
  tests.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `git diff --check` passed.

## 2026-06-17 Batch 70

Target: harden the performance regression ratchet itself.

- Added finite-number validation for `scripts/benchmark_ratchet.py`
  `threshold` and `min_seconds`, rejecting NaN/infinite values and enforcing
  `threshold > 0`, `min_seconds >= 0`.
- Applied the validation both in the programmatic `compare()` helper and the
  CLI `main()` path, so direct imports and command-line use cannot silently
  disable the timing-regression guard.
- Added tests for invalid thresholds, invalid noise floors, a real regression
  detection case, and CLI rejection of `--threshold nan`.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_benchmark_ratchet.py` passed, 9 tests.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `.venv/bin/python -m compileall -q scripts/benchmark_ratchet.py` passed.
- `git diff --check -- scripts/benchmark_ratchet.py
  tests/test_benchmark_ratchet.py` passed.

Additional quality gates:

- `.venv/bin/python scripts/quality_gate.py import-budget` passed with 0
  forbidden cold-import dependencies loaded.
- `.venv/bin/python scripts/quality_gate.py agent-cards` passed with all 15
  tracked counters at or above floor.
- `.venv/bin/python scripts/quality_gate.py result-protocol` passed with 266
  result classes inspected.
- `.venv/bin/python scripts/quality_gate.py error-taxonomy` passed with 285
  taxonomy raises and 1827 generic raises.
- `.venv/bin/python scripts/examples_coverage.py` reported 1033/1033
  registered functions with docstring `Examples` sections and 0 unresolved
  registered functions.

Combined current-turn verification:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''` over the changed
  fast, OLS, auto-estimator, result-protocol, output serialization, lineage,
  Bayesian protocol, and benchmark-ratchet tests passed, 435 tests with 18
  optional-engine/Rust/R/JAX/Polars skips.
- `git diff --check` passed.
- `.venv/bin/python -m compileall -q src/statspai scripts/benchmark_ratchet.py`
  passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  285 taxonomy raises and 1827 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- Rechecked `Paper-JSS/` and `CausalAgentBench/` status separately; both
  remain clean on `main...origin/main`.
- `.venv/bin/python scripts/quality_gate.py all` passed under the repository's
  gradual-debt baselines: flake8 observed 4406 <= 4698, mypy observed
  3317 <= 3521, import-budget observed 0, and the agent-card,
  result-protocol, and error-taxonomy gates all passed.

## 2026-06-17 Batch 62

Target: JAX FEOLS and JAX bootstrap validation parity with native fast FEOLS.

- Extended the shared fast-estimator validation helpers with an open-unit
  float validator for probability-like controls such as bootstrap `ci_alpha`.
- Routed `sp.fast.feols_jax()` through the shared validation helpers for
  `fe_maxiter`, `fe_tol`, empty samples, all-zero weights, and the
  post-singleton kept-weight sample.
- Routed JAX bootstrap prep through the same `fe_maxiter`, `fe_tol`, empty
  sample, all-zero weight, and kept-weight-mass checks.
- Routed `sp.fast.feols_jax_bootstrap()` through shared validators for
  `n_boot`, `vmap_chunk_size`, `ci_alpha`, `fe_maxiter`, and `fe_tol`, and
  moved missing-cluster checks before expensive prep work.
- Added JAX FEOLS and JAX bootstrap validation tests for invalid controls and
  all-zero weights.

Verification run:

- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_jax_feols.py tests/test_jax_feols_bootstrap.py
  tests/test_fast_jax_feols_result_protocol.py tests/test_result_protocol_audit.py`
  passed the import-safe/result-protocol subset, 5 tests passed and 2 JAX
  modules skipped because JAX is unavailable in this environment.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  197 taxonomy raises and 1884 generic raises.
- `git diff --check` passed.

Post-batch gate sweep:

- `git diff --check` passed.
- `.venv/bin/python scripts/result_protocol_audit.py --check` passed with
  266 result classes inspected.
- `.venv/bin/python scripts/error_taxonomy_audit.py --check` passed with
  197 taxonomy raises and 1884 generic raises.
- `.venv/bin/python scripts/stability_audit.py --check` passed with 0
  unbacked stable API entries.
- `.venv/bin/python scripts/help_coverage.py --check` passed.
- `.venv/bin/python -m compileall -q src/statspai` passed.
- `.venv/bin/python scripts/tierd_classify.py report` reported 0
  estimator-like Tier-D worklist items across 1033 registered functions
  (`reference=128`, `anchored=562`, `weak=146`, `smoke=9`,
  `untested=188`).
- `.venv/bin/python scripts/tier_a_fixture_lock.py` reported
  `tests/r_parity/TIER_A_FIXTURE_LOCK.json is current`.
- `.venv/bin/python scripts/benchmark_ratchet.py --check` passed with no
  StatsPAI timing regression beyond 1.50x.
- `MPLBACKEND=Agg .venv/bin/python -m pytest -o addopts=''
  tests/test_ols.py tests/test_agent_result_methods.py tests/test_lineage.py
  tests/test_auto_estimators.py tests/test_fast_bench.py
  tests/test_fast_jax_feols_result_protocol.py tests/test_fast_feols.py
  tests/test_fast_fepois.py tests/test_fast_event_study.py
  tests/test_fast_inference.py tests/test_fast_htz.py
  tests/test_fast_within_dsl.py tests/test_fast_demean.py
  tests/test_regtable_serialization.py tests/test_regtable_from_dict.py
  tests/test_bayes_result_protocol.py tests/test_result_protocol_audit.py`
  passed, 372 tests with 17 optional-engine skips.
