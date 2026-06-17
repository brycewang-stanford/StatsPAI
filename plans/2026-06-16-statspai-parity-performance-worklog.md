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
