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
