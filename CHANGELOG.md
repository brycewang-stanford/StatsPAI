# Changelog

All notable changes to StatsPAI will be documented in this file.

## [1.2.0] — 2026-04-21 — Doc-alignment sprint (v3 reference document)

Closes the remaining gaps between the *Causal-Inference Method Family
万字剖析 v3* (2026-04-20) reference document and the StatsPAI public API.
Most v3 frontier methods were already implemented in v1.0.x but lived in
sub-packages without top-level exposure or curated registry specs. This
release wires them up, adds the eight genuinely missing classical/frontier
methods, and upgrades 14 frontier estimators from auto-generated to
hand-written registry specifications so that LLM agents see proper
parameter docs, examples, references, and tags.

### Added — new estimators

**Staggered DID**

- `sp.gardner_did` / `sp.did_2stage` — Gardner (2021) two-stage DID
  estimator (the Stata `did2s` analogue). Stage-1 fits two-way fixed
  effects on untreated rows; Stage-2 regresses the residualised outcome
  on treatment dummies (overall ATT or event study) with cluster-robust
  SEs. Numerically agrees with `did_imputation` to within ~2% on
  synthetic staggered panels.

**DML**

- `sp.dml_model_averaging` / `sp.model_averaging_dml` — Ahrens, Hansen,
  Kurz, Schaffer & Wiemann (2025, *JAE* 40(3):381-402) model-averaging
  DML-PLR. Fits DML under multiple candidate nuisance learners and
  reports a risk-weighted (or equal/single-best) average θ with a
  cross-score-covariance-adjusted SE. Default candidate roster:
  Lasso / Ridge / RandomForest / GradientBoosting.

**IV**

- `sp.kernel_iv` (top-level alias of `sp.iv.kernel_iv`) — Lob et al.
  (2025, arXiv:2511.21603) kernel IV regression with wild-bootstrap
  uniform confidence band over the structural function `h*(d)`.
- `sp.continuous_iv_late` (top-level alias) — Xie et al. (2025,
  arXiv:2504.03063) LATE on the maximal complier class for continuous
  instruments via quantile-bin Wald estimator. (Also fixed a summary
  formatting bug — see below.)

**TMLE**

- `sp.hal_tmle` + `sp.HALRegressor` / `sp.HALClassifier` — TMLE with
  Highly Adaptive Lasso nuisance learners (Qian & van der Laan, 2025,
  arXiv:2506.17214). Two variants: `"delta"` (plug HAL into standard
  TMLE) and `"projection"` (apply tangent-space shrinkage to the
  targeting epsilon). Recovers ATE within ~3% on n=400 with rich
  nuisance.

**Synthetic Control**

- `sp.synth_survival` — Synthetic Survival Control (Agarwal & Shah,
  2025, arXiv:2511.14133). Donor convex combination on the
  complementary log-log scale matches the treated arm's pre-treatment
  Kaplan-Meier, then projects forward and reports the survival gap
  with a placebo-permutation uniform band. Pre-treat fit RMSE typically
  < 0.01 on synthetic Cox data.

**RDD aliases**

- `sp.multi_cutoff_rd` (alias for `sp.rdmc`), `sp.geographic_rd`
  (alias for `sp.rdms`), `sp.boundary_rd` (alias for `sp.rd2d`),
  `sp.multi_score_rd` (alias for `sp.rd_multi_score`) — user-friendly
  aliases mirroring the v3 document terminology.

### Added — registry / agent surface

- 14 frontier estimators promoted from auto-generated to **hand-written**
  registry specs with curated parameter descriptions, examples, tags,
  and references: `gardner_did`, `dml_model_averaging`, `kernel_iv`,
  `continuous_iv_late`, `hal_tmle`, `synth_survival`, `bridge`,
  `causal_dqn`, `fortified_pci`, `bidirectional_pci`, `pci_mtp`,
  `cluster_cross_interference`, `beyond_average_late`,
  `conformal_fair_ite`. This is what `sp.describe_function(...)` and
  `sp.function_schema(...)` now return for these names.
- Total registered functions: **836 → 860**.
- `__all__` repaired so previously-imported-but-not-exported symbols
  surface in `sp.list_functions()`: `fci` / `FCIResult`, `spatial_did`
  / `SpatialDiDResult`, `spatial_iv`, `notears`, `pc_algorithm`.

### Fixed

- `iv.continuous_late.ContinuousLATEResult.summary` — header line was
  being multiplied 42× by an implicit string-concat × `"=" * 42`
  precedence bug (`"title\n" "=" * 42` parsed as
  `("title\n" + "=") * 42`). Replaced with explicit f-string concatenation.

### Test coverage

35 new test cases across 7 new test modules:
`test_gardner_2s.py` (7), `test_dml_model_averaging.py` (5),
`test_kernel_iv.py` (5), `test_continuous_iv_late.py` (4),
`test_hal_tmle.py` (5), `test_synth_survival.py` (6),
`test_rd_aliases.py` (3). All pass on Python 3.13.

## [1.0.1] - 2026-04-21 — Post-review correctness pass + deferred-item closeout

Bugfix release closing every Critical / High / Medium finding from the
independent code-review-expert pass on the v1.0.0 frontier modules,
plus resolution of the two `# NEEDS_VERIFICATION` items that had been
deferred in v1.0.0.

### Fixed — post-review correctness pass

**Critical (silent wrong numbers)**

- `pcmci.partial_corr_pvalue`: Fisher-z SE now uses the effective
  sample size `sqrt(n - |Z| - 3)` instead of the off-by-one
  `sqrt(df - 1)`. The previous formula systematically missed edges
  in PCMCI by making partial-correlation p-values too large.
- `cohort_anchored_event_study`: the `cluster` argument was silently
  dropped — the bootstrap resampled cohort ATTs instead of the user-
  supplied cluster level. Fixed to resample at the requested cluster
  and re-compute ATT(c, k) per draw.
- `ltmle_survival` targeting step: the TMLE one-step update applied
  `logit(h_hat_regime)` inline instead of using the pre-computed
  `offset` variable, leaving the regime-counterfactual hazard
  untargeted. Rebound `offset_regime = logit(clip(h_hat_regime))`.

**High (wrong formula / silent tautology)**

- `conformal_density_ite`: previously fell back to split-conformal on
  Gaussian-residual quantiles, with the KDE bandwidth computed but
  unused. Now builds a proper KDE of the ITE-residual convolution and
  returns the Hyndman (1996) highest-density region via a shortest-
  window sweep over sorted smoothed samples.
- `bridge.ewm_cate`: Path A and Path B shared the same CATE-plug-in
  DR score, making the agreement test tautological. Path A now uses
  the Kitagawa-Tetenov (2018) pure-IPW welfare score so that the two
  paths have genuinely different failure modes, giving a real bridge.
- `mr_multivariable` conditional F-stat (Sanderson-Windmeijer): the
  partition `ss_full - ss_resid` used raw (uncentred) sum of squares
  and unweighted OLS. Replaced with centred SS over WLS residuals,
  matching the MVMR weighting scheme.
- `bcf_longitudinal.average_ate`: point estimate and CI were computed
  on different sampling distributions (per-time-point mean vs.
  bootstrap quantiles). Headline now uses the bootstrap mean.

**Medium**

- `conformal_fair_ite`: small protected-group fallback no longer
  mixes arms (which destroyed per-group coverage). Falls back to the
  conservative MAX per-group quantile across well-covered groups, or
  a pooled quantile with an explicit warning when all groups are small.
- `causal_rl.structural_mdp`: the `A` / `B` matrix slices were
  numerically verified correct, but shape assertions were added so any
  future refactor that flips the slice semantics fails loudly.
- `causal_llm.llm_dag_propose`: user-provided `domain` and `variables`
  are now sanitized (non-printable and newline characters stripped;
  length capped) before interpolation into the LLM prompt, closing
  the prompt-injection vector.

**Dead-variable cleanup**

- Removed stale `bM`, `fe_cols`, `avg`, `rng` names across
  `mendelian/multivariable.py`, `did/design_robust.py`,
  `bcf/longitudinal.py`, and `qte/hd_panel.py`.

### Changed — deferred-item closeout

- `beyond_average_late`: replaced the ad-hoc quantile-range rescaling
  with an Abadie (2002) κ-weighted complier-CDF construction that
  inverts the CDF difference on the complier subpopulation only. The
  result is a proper complier quantile treatment effect.
- `bridge.surrogate_pci`: path A (surrogate index) and path B (PCI
  bridge) now use genuinely different identifying assumptions — path
  A relies on surrogacy (no direct D→Y path given S), path B relies
  on proxy completeness (D is a valid IV for itself under the bridge
  function). The old OLS-on-(D, S, X) construction for path B is
  replaced with a 2SLS that uses S and X as exogenous controls while
  leaving D as the treatment of interest.

### Tests

- `tests/test_v100_review_fixes.py`: 8 pinning regression tests, each
  corresponding 1:1 to a review finding.
- Full-suite regression: 2 515+ tests passing, zero regressions.

## [1.0.0+] - 2026-04-21 — v3 frontier sweep (12-module / 38-estimator pass)

Round-out pass triggered by the v3 全景图 doc (2026-04-20), filling the
remaining 2025-2026 frontier gaps that Stata / R / EconML / DoWhy /
CausalML still lack. **38 new public estimators** across 12 modules,
all routed through `sp.*` and registered in `sp.list_functions()`.

### Added — v3 frontier estimators

- **DiD frontier** (`sp.did_*`): `did_bcf` (Forests for Differences,
  Wüthrich-Zhu 2025), `cohort_anchored_event_study` (arXiv 2509.01829),
  `design_robust_event_study` (arXiv 2601.18801),
  `did_misclassified` (arXiv 2507.20415).
- **Conformal frontier** (`sp.conformal_*`): `conformal_density_ite`
  (arXiv 2501.14933), `conformal_ite_multidp` (arXiv 2512.08828),
  `conformal_debiased_ml` (arXiv 2604.03772),
  `conformal_fair_ite` (arXiv 2510.08724).
- **Proximal frontier** (`sp.fortified_pci`, `sp.bidirectional_pci`,
  `sp.pci_mtp`, `sp.select_pci_proxies`): doubly-robust, bidirectional,
  modified-treatment-policies, plus a heuristic proxy selector
  (arXiv 2506.13152 / 2507.13965 / 2512.12038 / 2512.24413).
- **Distributional / panel QTE** (`sp.dist_iv`, `sp.kan_dlate`,
  `sp.qte_hd_panel`, `sp.beyond_average_late`): full distributional-
  layer LATE + HD-panel QTE + complier-distribution LATE
  (arXiv 2502.07641 / 2506.12765 / 2504.00785 / 2509.15594).
- **RDD frontier** (`sp.rd_interference`, `sp.rd_multi_score`,
  `sp.rd_distribution`, `sp.rd_bayes_hte`,
  `sp.rd_distributional_design`): five new 2025–2026 supports
  (arXiv 2410.02727 / 2508.15692 / 2504.03992 / 2504.10652 / 2602.19290).
- **`sp.causal_llm`** (NEW namespace): `llm_dag_propose`,
  `llm_unobserved_confounders`, `llm_sensitivity_priors` — all with
  deterministic heuristic backends (no API key required); accept a
  `client` arg for real LLM injection.
- **`sp.causal_rl`** (NEW namespace): `causal_dqn` (Confounding-Robust
  Deep Q, arXiv 2510.21110), `causal_rl_benchmark` (5 benchmarks per
  arXiv 2512.18135), `offline_safe_policy` (arXiv 2510.22027).
- **Cluster RCT × interference** (`sp.cluster_*`, `sp.dnc_gnn_did`):
  matched-pair, cross-cluster, staggered-rollout, DNC+GNN+DiD
  (arXiv 2211.14903 / 2310.18836 / 2502.10939 / 2601.00603).
- **IV frontier** (`sp.iv.kernel_iv`, `sp.iv.continuous_iv_late`,
  `sp.iv.ivdml`): kernel IV uniform CI + continuous-instrument
  maximal-complier LATE + LASSO-efficient instrument × DML
  (arXiv 2511.21603 / 2504.03063 / 2503.03530).
- **Meta-learner frontier** (`sp.focal_cate`, `sp.cluster_cate`):
  functional CATE (FOCaL, arXiv 2602.11118) + K-means cluster CATE
  (arXiv 2409.08773).
- **Bunching unification** (`sp.general_bunching`, `sp.kink_unified`):
  high-order bias correction (Song 2025, arXiv 2411.03625) +
  RDD/RKD/Bunching joint estimator (Lu-Wang-Xie 2025).

### Tests (v3 sweep)

- 55 new smoke tests added under `tests/test_*_frontiers.py`,
  `tests/test_causal_llm.py`, `tests/test_causal_rl.py`,
  `tests/test_cluster_rct.py`, `tests/test_metalearner_frontiers.py`,
  `tests/test_bunching_unified.py`. All pass; no regressions in the
  153 core tests for did / iv / rd / dml / proximal / metalearners.

### Registry (v3 sweep)

- Total registered functions: **794 → 831** (37 new symbols + 1 result
  class auto-discovered).
- All 38 surfaced via `sp.list_functions()`, `sp.help()`,
  `sp.function_schema()`, and the OpenAI-compatible JSON schema export.

## [1.0.0] - 2026-04-21 — Research-frontier capstone: bridging theorems, fairness, surrogates, MVMR, PCMCI, beyond-average QTE

StatsPAI 1.0 is the capstone release that integrates three years of
development into one coherent toolkit. On top of the v0.9.17
three-school completion, v1.0 ships the **2025-2026 research-frontier
modules** that Stata / R have not yet caught up with, wires every
scaffolded subpackage into the top-level `sp.*` namespace, and
upgrades the target-trial reporting layer to the JAMA/BMJ 2025
TARGET Statement.

### Added — v1.0 research-frontier modules

**Bridging theorems (`sp.bridge`)** — dual-path doubly-robust
identification. Each theorem pairs two seemingly different estimators
on the same target parameter: if *either* assumption holds, the
estimate is consistent.

- `bridge(..., kind="did_sc")`       — DiD ≡ Synthetic Control (Shi-Athey 2025)
- `bridge(..., kind="ewm_cate")`     — EWM ≡ CATE → policy (Ferman et al. 2025)
- `bridge(..., kind="cb_ipw")`       — Covariate balancing ≡ IPW × DR (Zhao-Percival 2025)
- `bridge(..., kind="kink_rdd")`     — Kink-bunching ≡ RDD (Lu-Wang-Xie 2025)
- `bridge(..., kind="dr_calib")`     — DR via calibration (Zhang 2025)
- `bridge(..., kind="surrogate_pci")` — Long-term surrogate ≡ PCI (Kallus-Mao 2026)
- `BridgeResult` reports both path estimates, their agreement test,
  and the recommended doubly-robust point estimate.

**Fairness (`sp.fairness`)** — counterfactual fairness as causal
inference, not pure statistics.

- `counterfactual_fairness` — Kusner et al. (2018) Level-2/3
  predictor evaluation on a user-supplied SCM.
- `orthogonal_to_bias` — Marchesin & Zhang (2025) residualization
  pre-processing that removes the component of non-protected features
  correlated with the protected attribute.
- `demographic_parity`, `equalized_odds`, `fairness_audit` —
  statistical fairness metrics + one-shot dashboard.

**Long-term surrogates (`sp.surrogate`)** — extrapolate short-term
experiments to long-term outcomes.

- `surrogate_index` — Athey, Chetty, Imbens, Pollmann & Taubinsky (2019).
- `long_term_from_short` — Ghassami, Yang, Shpitser, Tchetgen Tchetgen
  (2024).
- `proximal_surrogate_index` — Imbens, Kallus, Mao (2026): proximal
  identification when unobserved confounders link surrogate and
  long-term outcome.

**Multivariable MR (`sp.mendelian` extended)**

- `mr_multivariable` — MVMR on multiple correlated exposures.
- `mr_mediation` — causal-pathway decomposition for two-sample MR.
- `mr_bma` — Bayesian Model Averaging for MR with many candidate
  exposures (Yao et al. 2026 roadmap).

**DiD frontiers (`sp.did` extended)**

- `cohort_anchored_event_study` — cohort-robust event-study weights.
- `design_robust_event_study` — design-robust dynamic ATT.
- `did_misclassified` — treatment-misclassification-robust DiD.
- `did_bcf` — Bayesian Causal Forest wrapper for DiD.

**Conformal-inference frontiers (`sp.conformal_causal` extended)**

- `conformal_debiased_ml` — debiased-ML-aligned conformal intervals.
- `conformal_density_ite` — density-valued ITE conformal bounds.
- `conformal_fair_ite` — fairness-constrained ITE conformal.
- `conformal_ite_multidp` — multi-stage differentially-private ITE
  conformal bounds.

**Proximal causal frontiers (`sp.proximal` extended)**

- `bidirectional_pci` — two-sided proxy-based causal inference.
- `fortified_pci` — variance-fortified PCI.
- `pci_mtp` — multiple-testing-corrected PCI.
- `select_pci_proxies` — automated proxy-variable selector.

**Quantile / distributional-IV frontiers (`sp.qte` extended)**

- `beyond_average_late` — beyond-mean LATE for heterogeneous
  quantile treatment effects.
- `qte_hd_panel` — high-dimensional panel QTE.

**RD frontiers (`sp.rd` extended)**

- `rd_distribution` — distribution-valued (functional) RD.
- `rd_multi_score`, `rd_interference` — already shipped.

**Time-series causal discovery (`sp.causal_discovery` extended)**

- `pcmci` / `lpcmci` / `dynotears` — Peter-Clark-MCI family for
  observational + latent-confounder time-series DAG discovery.

**LTMLE survival + BCF longitudinal (`sp.tmle` / `sp.bcf` extended)**

- `ltmle_survival` — LTMLE for survival outcomes with time-varying
  treatments.
- `bcf_longitudinal` — BCF for longitudinal panel settings.

**Target Trial 2025 upgrade (`sp.target_trial` extended)**

- `target_checklist(result)` + `to_paper(..., fmt="target")` — render
  the JAMA/BMJ September-2025 TARGET Statement 21-item reporting
  checklist as a completed table, with `[AUTO]` / `[TODO]` tags for
  items that can be filled from the protocol + result vs. need
  author-supplied narrative.

**Synthetic control frontier**

- `sequential_sdid` — sequential synthetic difference-in-differences.

**ML bounds**

- `ml_bounds` — partial-identification bounds with ML nuisance
  estimation.

### Added — MCP server + bridge layer

- `sp.agent.mcp_server` — Model Context Protocol server scaffold so
  external LLMs (Claude, GPT-4, local models) can call every
  registered StatsPAI function via natural-language tool-calling.

### Changed

- `statspai/__init__.py`: 80+ new names in `__all__`; v1.0 total
  registered functions ≈ 729+.
- Registry now includes rich FunctionSpec entries for the core new
  frontier APIs (bridge, fairness, surrogate, mr_multivariable, etc.).

### Stability & scope

- All 229 tests added in the v0.9.17 + v1.0 window pass.
- Zero regressions in the 2158-test existing suite.
- Three-school completion from v0.9.17 carries forward intact
  (`sp.epi`, `sp.longitudinal`, `sp.question`, unified sensitivity,
  DAG recommender, preregistration).

### Versioning

- This is a major release (breaking-change policy starts here). The
  public API surface is the set of names in `statspai.__all__` as of
  v1.0.0; anything outside that list remains unstable.

## [0.9.17] - 2026-04-21 — Modern-weighting + MC g-formula + weakrobust panel + three-school completion

Two-pronged release. First, a surgical pass targeting four of the most-
requested gaps from the v1.0 gap-analysis: a Stata-style unified
weak-IV-robust diagnostic panel, the Zubizarreta (2015) stable-balancing-
weights estimator, the Robins (1986) Monte-Carlo g-formula (complementing
the existing Bang-Robins ICE), and a truly end-to-end `sp.causal()`
orchestrator. Second, a three-school completion pass mapping the
*Econometrics ↔ Epidemiology ↔ ML* knowledge-map article onto StatsPAI:
epidemiology primitives, MR full suite, longitudinal dispatcher, DAG-to-
estimator recommender, estimand-first DSL, and a unified sensitivity
dashboard attached to every `Result` object.

### Added

- `sp.weakrobust(data, y, endog, instruments, exog)` — one-call
  diagnostic panel that bundles Anderson-Rubin (1949), Moreira (2003)
  Conditional LR, Kleibergen (2002) K score test, Kleibergen-Paap
  (2006) rk LM + Wald F, Olea-Pflueger (2013) effective F, and
  Lee-McCrary-Moreira-Porter (2022) tF critical values. `WeakRobustResult`
  exposes `.summary()`, `.to_frame()`, and dict-style lookup. This is
  the Python analogue of Stata 19's `estat weakrobust`, unifying
  functionality scattered across `ivmodel` (R), `linearmodels`
  (Python), and the Stata user-written `weakiv` / `rivtest` packages.

- `sp.sbw(data, treat, covariates, y=..., estimand='att')` — Stable
  Balancing Weights (Zubizarreta 2015 JASA). Minimises variance (or
  KL) of the weights subject to per-covariate SMD balance tolerances
  solved via SLSQP. Supports ATT / ATC / ATE. Reports an effective
  sample size and before/after balance table. Complements `sp.ebalance`
  (exact balance) and `sp.cbps` (CBPS).

- `sp.gformula_mc(data, treatment_cols, confounder_cols, outcome_col)`
  — Monte-Carlo parametric g-formula (Robins 1986). Fits per-timepoint
  conditional models for confounders (binary logit / Gaussian OLS) and
  simulates counterfactual trajectories under user-supplied static or
  **dynamic** (callable) treatment strategies. Non-parametric bootstrap
  CI. Complements the existing `sp.gformula.ice` (Bang-Robins 2005 ICE).

- **Enhanced `sp.causal()` workflow** — three new stages auto-run
  after `estimate` / `robustness`:
  - `.compare_estimators()` — design-aware multi-estimator panel:
    CS + SA + BJS + Wooldridge for staggered DiD; 2SLS + LIML for IV;
    OLS + EB + CBPS + SBW + DML-PLR for observational.
  - `.sensitivity_panel()` — E-value + Oster δ* + Rosenbaum Γ in one
    DataFrame, matching the modern "sensitivity triad" expected by
    top-5 econ journals.
  - `.cate()` — X-Learner and Causal Forest heterogeneity summary
    (per-unit CATE mean, SD, q10/q50/q90).
  - Report output gains sections 4b / 4c / 4d.
  - Opt-out via `CausalWorkflow.run(full=False)`; `_extract_effect`
    helper unifies `CausalResult` and `EconometricResults` extraction.

### Reviewer-identified fixes (v0.9.17 internal review)

- `SBWResult.__init__` now forwards `model_info` + `_citation_key` to
  the `CausalResult` parent, wiring it into the citation registry.
- `MCGFormulaResult._is_binary` now requires **both** 0 and 1 levels
  present — a degenerate column (all-0 or all-1) no longer triggers
  the logistic Newton-Raphson loop.
- `_extract_effect` in `CausalWorkflow` now returns NaN when the
  treatment column is missing from the fitted params, rather than
  silently surfacing the intercept coefficient.
- SBW docstring clarified: reported SE is conditional-on-weights;
  users who need full parameter-uncertainty propagation should
  bootstrap `sp.sbw` externally.

### Deferred to a separate sprint

The original gap analysis also flagged TMLE dynamic regimes +
censoring, Conformal counterfactual / weighted variants, PCMCI
time-series causal discovery, Partial-ID + ML bounds, and the
Agent-MCP server integration. Each is substantial enough to warrant
its own focused sprint rather than being shipped half-finished here.

### Added — three-school completion (2026-04-21 sub-release)

Driven by a cross-reference audit against the article
"Causal Inference Knowledge Map — Econometrics, Epidemiology, ML",
which pinpointed Layer-4 (*What If* longitudinal), epidemiology
entry-level primitives, Mendelian randomization diagnostic depth,
DAG-to-estimator UX, and estimand-first workflow as the remaining
gaps vs. Stata / R dominance.

**Epidemiology primitives (`sp.epi`) — NEW subpackage**

- `odds_ratio`, `relative_risk`, `risk_difference`,
  `attributable_risk` (Levin PAF), `incidence_rate_ratio` (exact
  Poisson CI via Clopper-Pearson), `number_needed_to_treat`,
  `prevalence_ratio` — Woolf / Fisher-exact / Katz / Wald / Newcombe
  intervals; Haldane-Anscombe correction for zero cells.
- `mantel_haenszel` (OR / RR with Robins-Breslow-Greenland variance),
  `breslow_day_test` (homogeneity of OR with Tarone correction).
- `direct_standardize`, `indirect_standardize` — direct-standardized
  rates + SMR with Garwood exact Poisson CI.
- `bradford_hill` — structured 9-viewpoint causal-assessment rubric
  with prerequisite check (no causality claim without temporality).

**Mendelian randomization full suite (`sp.mr` / `sp.mendelian`)**

- `mr_heterogeneity` — Cochran's Q (IVW) or Rücker's Q' (Egger) + I².
- `mr_pleiotropy_egger` — formal MR-Egger intercept test for
  directional horizontal pleiotropy (Bowden 2015).
- `mr_leave_one_out` — per-SNP drop-one IVW sensitivity.
- `mr_steiger` — Hemani (2017) directionality test using Fisher-z of
  per-trait R² contributions.
- `mr_presso` — Verbanck (2018) global outlier test + per-SNP outlier
  detection + distortion test for raw-vs-corrected comparison.
- `mr_radial` — Bowden (2018) radial reparameterization + Bonferroni-
  thresholded outlier flagging.

**Target trial emulation — publication-ready report**

- `TargetTrialResult.to_paper(fmt=...)` / `sp.target_trial.to_paper` —
  render STROBE-compatible Methods + Results block in Markdown /
  LaTeX / plain-text for direct inclusion in manuscripts.  Table
  structure tracks the JAMA 2022 7-component TTE spec exactly.

**Longitudinal causal dispatcher (`sp.longitudinal`) — NEW subpackage**

- `sp.longitudinal.analyze` — unified entry point that auto-routes
  to IPW (no time-varying confounders) / MSM (dynamic regime with
  time-varying confounders) / parametric g-formula ICE (static
  regime) based on data shape and the supplied regime object.
- `sp.longitudinal.contrast` — plug-in estimator of
  `E[Y(regime_a)] - E[Y(regime_b)]` with delta-method SE.
- `sp.regime`, `sp.always_treat`, `sp.never_treat` — dynamic-treatment-
  regime DSL supporting static sequences, callables, and a safe
  `"if cd4 < 200 then 1 else 0"` string DSL. The string DSL is parsed
  into a whitelisted AST and interpreted by a tiny tree-walker — no
  dynamic code execution is ever invoked, and disallowed constructs
  are rejected at regime-construction time.

**Estimand-first causal-question DSL (`sp.causal_question`) — NEW subpackage**

- `sp.causal_question(treatment=, outcome=, estimand=, design=, ...)`
  declares a causal question up front.  `.identify()` picks an
  estimator + lists the identifying assumptions the user must defend;
  `.estimate()` runs the analysis; `.report()` produces a Markdown
  Methods + Results paragraph.
- Auto-design selects IV when instruments are present, RD when running
  variable + cutoff given, DiD when panel + time, longitudinal when
  repeated measures, else selection-on-observables.
- Dispatches internally to `sp.regress` / `sp.aipw` / `sp.iv` /
  `sp.did` / `sp.rdrobust` / `sp.synth` / `sp.longitudinal.analyze` /
  `sp.event_study`.

**DAG → estimator recommender (`sp.dag.recommend_estimator`)**

- `DAG.recommend_estimator(exposure, outcome)` — inspects the declared
  graph and suggests a StatsPAI estimator with a plain-English
  identification story. Priority order: backdoor adjustment (OLS /
  IPW / matching) → IV (heuristic relevance + exclusion check) →
  frontdoor → not-identifiable (with sensitivity-analysis fallbacks).
- Detects mediators on the causal path automatically.

**Unified sensitivity dashboard (`sp.unified_sensitivity`)**

- `result.sensitivity()` — method added to both `CausalResult` and
  `EconometricResults`. Single call runs E-value (always), Oster δ
  (when R² inputs supplied), Rosenbaum Γ (when a matched structure is
  exposed), Sensemakr (regression models), and a breakdown-frontier
  bias estimate.

### Changed (three-school completion)

- `__init__.py`: 40+ new names exposed at top level including `sp.epi`,
  `sp.longitudinal`, `sp.question`, `sp.tte` / `sp.mr` short aliases.

### Fixed (three-school completion)

- Regime DSL: AST validation moved from evaluate-time to compile-time
  so unsafe expressions are rejected immediately at `sp.regime(...)`
  construction, before any history is supplied.

## [0.9.16] - 2026-04-20 — v1.0 breadth expansion + Bayesian family polish + Rust Phase-2 CI

The largest release since the v1.0 breadth pass. Maps StatsPAI onto
the full Mixtape + What If + Elements of Causal Inference curriculum:
Hernan-Robins target-trial emulation, Pearl-Bareinboim SCM machinery,
modern off-policy / neural-causal estimators, plus three additions
that close long-standing gaps in the Bayesian family, plus a CI
scaffold for the Rust HDFE spike.

### Added (0.9.16) — v1.0 breadth expansion (27+ new modules)

**Target trial emulation & censoring (`sp.target_trial`, `sp.ipcw`)**

- `target_trial_protocol`, `target_trial_emulate`, `clone_censor_weight`,
  `immortal_time_check` — JAMA 2022 7-component TTE framework with
  explicit eligibility / time-zero / per-protocol contrast support.
- `ipcw` — Robins-Finkelstein inverse probability of censoring weights
  (pooled-logistic or Cox hazard) with stabilization + truncation.

**SCM / DAG machinery (`sp.dag` extended)**

- `identify` — Shpitser-Pearl ID algorithm; returns do-free estimand
  when identifiable, witness hedge `(F, F')` otherwise.
- `do_rule1 / do_rule2 / do_rule3`, `do_calculus_apply` — mechanized
  do-calculus with d-separation on mutilated graphs `G_{bar X}`,
  `G_{underline Z}`, and `G_{bar Z(W)}`.
- `swig` — Richardson-Robins Single-World Intervention Graphs via
  node-splitting of intervened variables.
- `SCM` — abduction-action-prediction counterfactual runner with
  rejection sampling fallback for non-Gaussian structural equations.
- `llm_dag` — LLM-backed DAG extraction from free-form descriptions.

**Causal discovery with latents (`sp.causal_discovery`)**

- `fci` — FCI for PAGs with unobserved confounders (Zhang 2008):
  skeleton + v-structures + FCI rules R1-R4.
- `icp`, `nonlinear_icp` — Peters-Bühlmann-Meinshausen invariant
  causal prediction; linear F-test / K-S nonlinear invariance.

**Transportability (`sp.transport`)**

- `transport_weights_fn` / `transport_generalize` — Stuart / Dahabreh
  density-ratio transport with inverse odds of sampling weighting.
- `identify_transport` — Bareinboim-Pearl s-admissibility; enumerates
  adjustment sets on selection diagrams, returns transport formula.

**Off-policy evaluation (`sp.ope`)**

- `ips`, `snips`, `doubly_robust`, `switch_dr`, `direct_method`,
  `evaluate` — Dudik-Langford-Li DR family plus Swaminathan-Joachims
  SNIPS and Wang-Agarwal-Dudík Switch-DR for bandits / RL.

**Deep causal & latent-confounder models (`sp.neural_causal`)**

- `cevae` — Louizos et al. CEVAE with PyTorch path + numpy
  variational fallback so import never fails.

**Longitudinal / G-methods (`sp.gformula`, `sp.tmle`, `sp.dtr`)**

- `gformula_ice_fn` — Bang-Robins iterative conditional expectation
  parametric g-formula; sequential backward regression with recursive
  strategy plug-in. Supports static / scalar / callable strategies.
- `ltmle` — van der Laan-Gruber longitudinal TMLE.
- `q_learning`, `a_learning`, `snmm` — dynamic treatment regime
  estimators.

**Additional estimators across the stack**

- Causal forests: `multi_arm_forest`, `iv_forest`,
  `survival/causal_forest` (Cui-Kosorok 2023).
- Proximal: `negative_controls`, `pci_regression` (Miao-Shi-Tchetgen).
- Interference: `network_exposure` (Aronow-Samii 2017), `peer_effects`.
- Dose-response: `vcnet` + `scigan` (Nie-Brunskill-Wager 2021).
- Matching: `genmatch` (Diamond-Sekhon 2013).
- Sensitivity: `rosenbaum_bounds`.
- Spatial: `spatial_did`, `spatial_iv` (Kelejian-Prucha 1998).
- Time series: `its` (interrupted time series).
- Bounds: `balke_pearl`.
- Mediation: `four_way_decomposition` (VanderWeele 2014).

**Registry / agent surface**

- 11 hand-written `FunctionSpec` entries for the new flagship APIs,
  each with parameter schemas, tags, and canonical references.
- `sp.list_functions()` now reports 664 entries.
- `sp.search_functions("target trial")` / `"invariance"` /
  `"transport"` all resolve correctly.

### Added (0.9.16) — Bayesian family gap-closing

- **`bayes_mte(mte_method='bivariate_normal')`** — full textbook
  Heckman-Vytlacil trivariate-normal model `(U_0, U_1, V) ~ N(0, Σ)`
  with `D = 1{Z'π > V}`. Identifies the structural gap
  `β_D = μ_1 - μ_0` and the two selection covariances `σ_0V, σ_1V`
  via inverse-Mills-ratio corrections in the structural equation, so
  `MTE(v) = β_D + (σ_1V - σ_0V)·v` is closed-form linear on V scale.
  Requires `selection='normal'` and `first_stage='joint'`; `poly_u`
  is overridden to 1 with a `UserWarning` if the user set something
  else. Exposes `b_mte` as a 2-vector Deterministic
  `[β_D, σ_1V - σ_0V]` so every downstream code path
  (`mte_curve`, ATT/ATU integrator, `policy_effect`) works unchanged.
  This is the last missing piece of the Heckman-Vytlacil pipeline
  that `selection='uniform'`/`'normal'` + `mte_method='polynomial'`/
  `'hv_latent'` started.

- **`bayes_did(cohort=...)` + `BayesianDIDResult`** — when the user
  supplies a `cohort` column (typically first-treatment period in a
  staggered design), the scalar `tau` is replaced with a vector
  `tau_cohort` of length `n_cohorts` under the same
  `Normal(prior_ate)` prior. The result carries
  `cohort_summaries: Dict[str, dict]` and `cohort_labels`; the
  top-level pooled ATT is the treated-size-weighted mean of the
  per-cohort τ posteriors. `result.tidy(terms='per_cohort')`
  returns one row per cohort with `term='cohort:<label>'`; explicit
  `terms=['att', 'cohort:2019', ...]` selection is supported for
  modelsummary / gt pipelines. **Back-compat:** calling without
  `cohort=...` returns a `BayesianDIDResult` that behaves byte-
  identically to the v0.9.15 `BayesianCausalResult`.

- **`bayes_iv(per_instrument=True)` + `BayesianIVResult`** — on a
  multi-instrument fit, additionally runs one just-identified
  Bayesian IV sub-fit per `Z_j` and stores per-instrument posteriors
  as `instrument_summaries: Dict[str, dict]`. Surface mirrors the
  DID extension: `tidy(terms='per_instrument')` emits one row per
  `Z` with `term='instrument:<name>'`. The top-level pooled LATE
  remains the joint over-identified fit; per-instrument rows are an
  add-on diagnostic. Sub-fit priors and sampler controls mirror the
  pooled fit, so runtime scales roughly `(K+1)×`.

- **`.github/workflows/build-wheels.yml`** — Rust Phase-2
  cibuildwheel matrix workflow (macOS arm64 + x86_64,
  manylinux_2_17 x86_64 + aarch64, musllinux_1_2 x86_64, Windows
  x86_64) with a `check_rust_present` guard job that makes the
  workflow a no-op when `rust/statspai_hdfe/Cargo.toml` is absent
  (the state on `main`). The workflow activates automatically on
  `feat/rust-hdfe`/`feat/rust-phase2` and on PRs touching
  `rust/**`, so the Rust spike's CI lights up the moment the
  branch is ready — no second PR for CI scaffolding.

### Tests (0.9.16)

- `tests/test_bayes_mte_bivariate_normal.py` — 7 tests covering
  API validation (selection + first_stage gates, poly_u override),
  structural-param presence in posterior, method label contents,
  and slope recovery on a genuine trivariate-normal DGP at n=800.
- `tests/test_bayes_did_cohort.py` — 9 tests covering back-compat
  (no cohort → single-row tidy identical to v0.9.15), cohort fit
  populates summaries, multi-row tidy via `per_cohort` + explicit
  list, unknown-term raises, τ ordering recovered on a two-cohort
  staggered DGP with heterogeneous true ATTs (2.0 vs 0.5), and
  cohort weights recorded in model_info.
- `tests/test_bayes_iv_per_instrument.py` — 8 tests covering
  back-compat, per-instrument summary population, `per_instrument`
  tidy, explicit-list tidy, unknown-term raises, error path when
  asking for `per_instrument` tidy without the sub-fit, and each
  sub-fit's HDI covers the true LATE on a two-Z DGP.

### Not in this release

- Round-trip testing of the cibuildwheel matrix on real runner
  hardware — this must happen on `feat/rust-hdfe`, where the
  crate lives. The workflow on `main` is inert by design.

## [0.9.15] - 2026-04-20 — Multi-term `tidy(terms=[...])` + ATT/ATU prob_positive

Completes the broom-pipeline integration of v0.9.13's per-population
ATT/ATU uncertainty. Users can now `pd.concat` ATE/ATT/ATU rows
across fits in one call.

### Added (0.9.15)

- **`BayesianMTEResult.tidy(conf_level=None, terms=None)`** override:
  - `terms=None` (default) — unchanged, single ATE row.
  - `terms='ate' | 'att' | 'atu'` — single row of that term.
  - `terms=['ate', 'att', 'atu']` — multi-row DataFrame.
  - Invalid names → clear `ValueError`.

- **Two new result fields**: `att_prob_positive`, `atu_prob_positive`
  (NaN-defaulted for pre-v0.9.15 snapshot compatibility). Populated
  by `_integrated_effect` from per-draw ATT/ATU posteriors.

- **`_integrated_effect` returns 5-tuple** `(mean, sd, hdi_lower,
  hdi_upper, prob_positive)`. Caller unpacks + passes to the result.

### Round-B review found 1 HIGH; Round-C fixed

- **HIGH-1** — label divergence: default `tidy()` emits
  `term='ate (integrated mte)'` (via parent `estimand.lower()`),
  but `tidy(terms='ate')` emitted the short literal `'ate'`. Byte-
  compat broken when a user mixed both call styles inside
  `pd.concat`. **Fixed** — `_row('ate')` now also uses
  `self.estimand.lower()` so both paths produce identical rows.
  ATT / ATU rows keep their short labels (no parent-default
  precedent; short is the natural broom shape for new terms).

- Round C: 0 blockers.

### Tests (0.9.15)

- `tests/test_bayes_mte_tidy.py` (13 tests) — back-compat default
  schema, single-term paths for all three labels, multi-row order
  preservation, concat workflow, invalid-term + mixed-valid
  rejection, NaN prob_positive stub back-compat, prob_positive
  scalars populated on real fits, **default-vs-explicit label
  byte-parity** (Round-C regression).
- Bayesian family suite: 101/101 focused tests green.

### Design spec (0.9.15)

- `docs/superpowers/specs/2026-04-20-v0915-tidy-multiterm.md`

### Non-goals (0.9.15)

- Multi-term `.tidy()` on other Bayesian estimators — DID/RD/IV
  have no ATT/ATU concept; the primary-estimand row is already
  what they emit.
- Full bivariate-normal HV model.
- Rust Phase 2.

---

## [0.9.14] - 2026-04-20 — Summary rendering completes v0.9.13 spec §3.3

Tiny patch release. Completes the "ATT/ATU in `summary()`" promise
from v0.9.13 spec §3.3 that was not actually wired at ship time
(the six uncertainty fields landed but `summary()` never printed
them).

### Added (0.9.14)

- **`BayesianMTEResult.summary()`** override. Extends
  `BayesianCausalResult.summary` with a `Population-integrated effects`
  block:

      ATT: 0.2407 (sd 0.0370, 95% HDI [0.1693, 0.3136])
      ATU: 0.2147 (sd 0.0435, 95% HDI [0.1341, 0.2947])

  Rendered inside the framing `=` ruler for visual coherence.
  Silently skipped when either SD is NaN (empty subpopulation or
  pre-v0.9.13 deserialised result).

### Round-B review: no blockers

Reviewer confirmed:
1. `base.endswith('=' * 70)` is exact — parent `summary()` returns
   `'\n'.join(lines)` with the rule as the final element.
2. Block splicing preserves the closing ruler visually.
3. NaN stub path is safe; fallback branch is defensive.
4. `'ATT:'` / `'ATU:'` are unique substrings — no collision with
   parent output.
5. Pure reader; thread-safe.

### Tests (0.9.14)

- `tests/test_bayes_mte_uncertainty.py` now has:
  - `test_summary_shows_att_atu_uncertainty` — after fit, string
    contains `'ATT:'`, `'ATU:'`, `'sd '`, `'HDI ['`.
  - `test_summary_skips_att_atu_when_nan` — NaN-SD stub → no
    `'ATT:'` / `'ATU:'` in output.
- Full Bayesian suite: 88/88 focused MTE + sibling green in 1:55.

### Non-goals (0.9.14)

- `.tidy()` multi-row variant with ATE/ATT/ATU as separate rows
  — queued for v0.9.15+.
- Full bivariate-normal HV model.
- Rust Phase 2.

---

## [0.9.13] - 2026-04-20 — ArviZ HDI compat shim + ATT/ATU uncertainty

Small-but-load-bearing cleanup release. Closes two items deferred
across the v0.9.10 / v0.9.11 / v0.9.12 code reviews.

### Added (0.9.13)

- **`_az_hdi_compat(samples, hdi_prob)`** in `statspai.bayes._base`
  — calls `az.hdi(samples, hdi_prob=...)` first, falls back to
  `az.hdi(samples, prob=...)` on `TypeError`. Routes **every**
  `az.hdi(...)` call site in the Bayesian sub-package through one
  place so the inevitable arviz ≥ 0.18 kwarg rename is a one-line
  change. Previously identified as time-bomb by v0.9.12 round-C
  review.

- **ATT / ATU uncertainty** on `BayesianMTEResult`:
  - `att_sd`, `att_hdi_lower`, `att_hdi_upper`
  - `atu_sd`, `atu_hdi_lower`, `atu_hdi_upper`

  `_integrated_effect` now returns `(mean, sd, hdi_lower,
  hdi_upper)` instead of `(mean, sd)`. `posterior_sd` on the parent
  result already covers ATE uncertainty — no redundant `ate_sd`.

- **Appended-at-end field order** on `BayesianMTEResult` — all six
  new fields are NaN-defaulted and positioned after the v0.9.12
  schema (`selection`). Serialised results from earlier releases
  deserialise cleanly.

### Round-B code review found no blockers

Reviewer confirmed:
1. `_az_hdi_compat` fallback shape correct for any future arviz
   kwarg rename.
2. Dataclass field order verified via live introspection.
3. No `__hash__` risk on NaN fields; broom-style `.tidy()` /
   `.glance()` intentionally do not surface the new SD/HDI fields
   (opt-in access).
4. Imports clean in `mte.py` + `hte_iv.py`.
5. Empty-population NaN guardrail is defensive-only; unreachable
   from `bayes_mte` because `_logit_propensity` enforces 2-class
   requirement upstream. Test renamed to reflect this honestly.

One MEDIUM item (test-docstring mislabel) fixed inline.

### Incident log

A mass `regex` rewrite from `az.hdi(...)` to `_az_hdi_compat(...)`
accidentally matched the helper's own body, creating a
`_az_hdi_compat → _az_hdi_compat` self-recursion. Caught by running
the Bayesian focused suite (would have been a stack-overflow the
moment any Bayesian estimator shipped). Reverted + re-applied
manually in the same session before tests ever ran outside dev.

### Tests (0.9.13)

- `tests/test_bayes_hdi_compat.py` (4 tests) — forwards on current
  arviz, falls back on monkey-patched future arviz, returns length-2
  array, propagates `TypeError` when both kwargs rejected (no silent
  success).
- `tests/test_bayes_mte_uncertainty.py` (4 tests) — ATT/ATU SD
  populated + > 0, HDI brackets mean, no redundant `ate_sd`, realistic-
  DGP both-finite.
- Bayesian family suite: 145/145 focused MTE + sibling tests green.

### Design spec

- `docs/superpowers/specs/2026-04-20-v0913-hdi-compat-and-att-sd.md`

### Non-goals (0.9.13)

- Full bivariate-normal HV `(U_0, U_1, V) ~ N(0, Σ)` — stays queued.
- Rust Phase 2.
- Expose ATT/ATU HDI on `.tidy()` — today `.tidy()` describes the
  primary estimand (ATE); adding a multi-row variant for ATT/ATU is
  a v0.9.14+ API question.

---

## [0.9.12] - 2026-04-20 — Probit-scale MTE (Heckman selection frame)

Adds the third orthogonal axis to `sp.bayes_mte`: the MTE polynomial
can now be fit on either the uniform scale `U_D ∈ [0, 1]`
(v0.9.11 default) or the probit / V scale
`V = Φ^{-1}(U_D) ∈ ℝ` — the conventional Heckman (1979) / HV 2005
frame. All `(first_stage, mte_method, selection)` combinations fit.

### Added (0.9.12)

- **`sp.bayes_mte(..., selection='uniform' | 'normal')`** — new kwarg.
  - `'uniform'` (default) preserves v0.9.11 behaviour: polynomial
    in `U_D ∈ [0, 1]`.
  - `'normal'` reinterprets the abscissa as `V = Φ^{-1}(U_D)` via
    `pt.sqrt(2) * pt.erfinv(2a-1)` on the tensor side and
    `scipy.stats.norm.ppf` on numpy side. Under strict HV + bivariate-
    normal, `poly_u=1 + selection='normal' + mte_method='hv_latent'`
    exactly recovers the linear Heckman MTE slope.

- **`mte_curve` exposes `v` column** under `selection='normal'`
  (empty otherwise) so users can plot on the scale their model
  was fit on.

- **Shared `PROBIT_CLIP` constant** in `statspai.bayes._base` —
  fit-time, ATT/ATU integrator, and `policy_effect` all read the
  same clip so the three paths stay numerically consistent.

### Empirical recovery on Heckman DGP (true `(b_0, b_1) = (0.5, 1.5)`)

| combo | `b_0` | `b_1` |
|---|---|---|
| plugin × polynomial × V | -0.73 | 0.82 |
| plugin × hv_latent × V | 0.42 | 1.37 ✓ |
| joint × polynomial × V | -0.73 | 0.81 |
| joint × hv_latent × V | 0.46 | 1.40 ✓ |

Same story as earlier releases: `hv_latent` recovers truth;
`polynomial` fits `g(v)` not `MTE(v)` and is biased.

### Round-B review found 2 BLOCKERS + 2 HIGHs; Round-C fixed all

1. **BLOCKER-1**: `_integrated_effect` (ATT/ATU) was raising `U_population`
   to polynomial powers directly, even under `'normal'` where the
   posterior is on V scale. **Fixed** — transforms to
   `Φ^{-1}(U_population)` first.
2. **BLOCKER-2**: `BayesianMTEResult.policy_effect` computed
   `u_pow = [u^k ...]` instead of `[v^k ...]` under `'normal'`,
   silently integrating a V-scale polynomial against u-scale powers.
   **Fixed** — `BayesianMTEResult` now carries a `selection` field,
   and `policy_effect` transforms the grid to V scale when needed.
   Regression test asserts `policy_effect(policy_weight_ate())`
   matches `.ate` to 1e-8 under `'normal'`.
3. **HIGH-1**: `mte_curve` lacked a `v` column — **added**.
4. **Round-C follow-up**: extracted `PROBIT_CLIP = 1e-6` to a shared
   module constant consumed by both `mte.py` and `_base.py` so the
   three-site fit/summary/policy paths cannot drift.

### Tests (0.9.12)

- `tests/test_bayes_mte_selection.py` (NEW, 12 tests) — back-compat,
  method-label, Heckman DGP recovery, all-8-combo orthogonality,
  input validation, `v` column presence/absence, ATT/ATU V-scale
  correctness (Round-C regression), `policy_effect` V-scale
  parity with `.ate` (Round-C regression), uniform-vs-normal
  non-trivial disagreement.
- 78 focused MTE tests green.

### Non-goals (0.9.12)

- Full bivariate-normal error covariance `(U_0, U_1, V) ~ N(0, Σ)`
  with free `ρ_{0V}`, `ρ_{1V}` — convergence-intensive MvNormal
  mixture, queued for 0.9.13+.
- Rust Phase 2 — separate branch.

---

## [0.9.11] - 2026-04-20 — Multi-instrument MTE + true CHV-2011 PRTE weights

Closes two long-standing API gaps plus an empirical math debt.

### Added (0.9.11)

- **`sp.bayes_mte(instrument: str | Sequence[str], ...)`** — MTE
  now accepts multiple instruments, matching `sp.bayes_iv` /
  `sp.bayes_hte_iv`. Scalar calls unchanged.
- **`sp.policy_weight_observed_prte(propensity_sample, shift)`** —
  true Carneiro-Heckman-Vytlacil (2011) PRTE weights from the
  observed propensity distribution via
  `kde.integrate_box_1d(u-Δ, u) / Δ` (CDF difference). Closes the
  v0.9.9 docstring gap where `policy_weight_prte` was flagged
  stylised.

### Round-B review found 2 HIGH + 3 MEDIUM; all fixed

1. **CHV sign bug** — my original `(kde(u) - kde(u-Δ))/Δ` AND the
   reviewer's proposed swap were both wrong (both compute
   derivative of density, not CDF difference). Self-sweep verified
   CHV-2011 Theorem 1 is a CDF difference. Fixed via
   `integrate_box_1d`. Empirical: uniform propensity + Δ=0.2 now
   gives the textbook trapezoid; previously gave a spurious
   boundary spike.
2. **Unconditional `np.clip(w, 0, None)`** silently altered the
   estimand. Dropped — contraction policies now yield signed
   negative weights, matching CHV convention.
3. **`gaussian_kde` thread safety** — forced covariance
   precomputation inside the builder.
4. **`model_info['instrument']` type varied** — dropped the raw
   key; only `instruments` (list) + `n_instruments` remain.
5. **Back-compat test** uses relative-to-posterior-SD tolerance.

### Tests (0.9.11)

- `tests/test_bayes_mte_multi_iv.py` (9 tests).
- `tests/test_bayes_mte_policy.py` (+7 tests).
- 61 focused MTE tests green.

### Code review

- Round B agent: 5 items. Self-sweep caught one HIGH the agent
  got wrong. All 5 fixed.
- Round C agent: zero ship-blockers.

### Design spec (0.9.11)

- `docs/superpowers/specs/2026-04-20-v0911-multi-iv-mte-observed-prte.md`

---

## [0.9.10] - 2026-04-20 — HV-latent MTE (textbook Heckman-Vytlacil via latent U_D)

Closes the semantic debt v0.9.9 flagged but did not pay: the
previous releases fitted a polynomial in the propensity `p_i`
(`g(p)` = LATE-at-propensity), which coincides with the textbook
MTE only under HV-2005 linear-separable + bivariate-normal errors.
v0.9.10 adds an opt-in **fully HV-faithful** model that samples a
latent `U_D_i ~ Uniform(0, 1)` per unit via the truncated-uniform
reparameterisation trick, making the fitted polynomial a genuine
posterior over `tau(u) = E[Y_1 - Y_0 | U_D = u]`.

### Added (0.9.10)

- **`sp.bayes_mte(..., mte_method='polynomial' | 'hv_latent')`** —
  new kwarg, orthogonal to the existing `first_stage` kwarg.
  - `'polynomial'` (default) — v0.9.9 behaviour; polynomial in
    propensity.
  - `'hv_latent'` — textbook HV. For each unit, sample
    `raw_U_i ~ Uniform(0, 1)`, then transform deterministically:

        D_i = 1 ⇒ U_D_i = raw_U_i · p_i            ∈ [0, p_i]
        D_i = 0 ⇒ U_D_i = p_i + raw_U_i·(1 - p_i)  ∈ [p_i, 1]

    The polynomial is then evaluated at `U_D_i` (not `p_i`).
    Structural equation:
    `Y_i = α + β_X' X_i + D_i · τ(U_D_i) + ε_i`.

  Orthogonal to `first_stage`: all four
  `(plugin|joint) × (polynomial|hv_latent)` combinations run.

- **Memory-warning guard** — `hv_latent` registers a shape-(n,)
  latent stored as `(chains, draws, n)` in the posterior. The
  function emits a `UserWarning` when
  `n × draws × chains > 50,000,000` (~400 MB at f64), mentioning
  `draws`, `chains`, and `mte_method='polynomial'` as mitigations.

### HV-augmentation factorisation (documented in docstring)

`bayes_mte` uses the standard Form-2 data-augmentation
factorisation:

    p(Y, D, U_D | p, θ) = p(Y | U_D, D, θ) · p(U_D | D, p) · p(D | p)

where the truncated-uniform transform gives `p(U_D | D, p)` and
`pm.Bernoulli(D | p)` gives the marginal `p(D | p)`. Both are
needed — dropping the Bernoulli in a counter-factual experiment
made `piZ` flip sign (true 0.8 → posterior -1.01) and biased the
MTE polynomial to `[0.81, 1.25]` vs true `[2, -2]`. This test is
documented in the v0.9.10 round-B code review.

### Empirical recovery evidence

Decreasing-MTE DGP with truth `(b_0, b_1) = (2.0, -2.0)`:

| combo | b_0 posterior | b_1 posterior | recovers? |
|---|---|---|---|
| plugin × polynomial    | 1.73 | -0.43 | biased |
| plugin × hv_latent     | 2.03 | -2.13 | ✓ |
| joint  × polynomial    | 1.73 | -0.44 | biased |
| joint  × hv_latent     | 2.05 | -2.16 | ✓ |

The polynomial modes are systematically biased on HV DGPs — the
honesty caveat v0.9.9 added is empirically validated; hv_latent
is the mathematical fix.

### Method label

- `polynomial` → `"Bayesian treatment-effect-at-propensity (...)"`
  (v0.9.9 label retained).
- `hv_latent` → `"Bayesian HV-latent MTE (...)"`.

### Tests (0.9.10)

- **`tests/test_bayes_mte_hv_latent.py`** (10 tests) — API, recovery
  of true `(b_0, b_1) = (2, -2)` on an HV DGP, disagreement with
  polynomial mode on same DGP, orthogonality with
  `first_stage='joint'`, input validation, memory-warning fires
  above threshold (unittest.mock), memory-warning stays silent
  below threshold, `policy_effect` still works on hv_latent
  results.

### Code review (two rounds)

- **Round B** (agent) raised 3 HIGH items:
  1. "Double-counting Bernoulli" — **rejected after math + counter-
     factual**. Form-2 factorisation is correct; dropping Bernoulli
     wildly biased the result. Defended in docstring.
  2. "Marginal U_D not Uniform(0,1)" — **rejected after algebra**.
     `p(U_D|p) = p·U(0,p) + (1-p)·U(p,1) = Uniform(0,1)` holds.
  3. "Memory blow-up" — **accepted**; added `UserWarning`.
- **Round C** (agent) on the round-B resolutions: **no ship-blockers**.
  One cosmetic nit on the integration notation in the docstring
  fixed inline.

### Design spec

- `docs/superpowers/specs/2026-04-20-v0910-hv-latent-mte.md`

### Non-goals (0.9.10)

- Full bivariate-normal error structure on `(U_0, U_1, U_D)` —
  linear-separable only. Natural 0.9.11+ extension.
- Multi-instrument HV MTE.
- GP over `u` (still polynomial of order `poly_u`).
- Rust Phase 2 — branch work.

### Article-surface round-2: namespace fixes + kwarg alignment

Completes the API-cleanup thread started by v0.9.9's first alias pass.
The 2026-04-20 survey post advertises `sp.matrix_completion`,
`sp.causal_discovery`, `sp.mediation`, `sp.evalue_rr`, plus
article-style kwargs on `sp.policy_tree` / `sp.dml` — all of which
either resolved to the *submodule* or rejected the blog-post kwargs
before this round.

#### Added — article-facing aliases

- `sp.matrix_completion(df, y, d, unit, time)` — thin wrapper over
  `sp.mc_panel`, renames `d → treat`. Shadows the former module binding.
- `sp.causal_discovery(df, method='notears'|'pc'|'ges'|'lingam',
  variables=None)` — dispatcher. Handles each backend's native
  signature (notears/pc accept `variables=`; ges/lingam do not, so
  the dispatcher subsets the frame upfront).
- `sp.mediation(df, y, d, m, X)` — article wrapper over `sp.mediate`;
  shadows the former module binding.
- `sp.evalue_rr(rr, rr_lower=None, rr_upper=None)` — risk-ratio
  convenience for the shape documented in the blog post.
- `sp.policy_tree` accepts either `d=/treat=`, `X=/covariates=`,
  and `depth=/max_depth=`. Conflicting values raise `TypeError`.
- `sp.dml` accepts `model_y=` / `model_d=` as aliases for `ml_g` /
  `ml_m`, and the same dual-convention naming.

#### Hardened

- `sp.auto_did` now fails fast with `TypeError` when `g` is a
  non-numeric cohort label (BJS branch silently misbehaves otherwise).
- `AutoDIDResult.__repr__` / `AutoIVResult.__repr__` now return a
  one-line summary (Jupyter list-of-results display); call
  `.summary()` for the full leaderboard.
- `statspai.agent.tools._default_serializer` is now scalar-safe
  (new `_scalar_or_none` helper) — handles Series-valued result
  fields without crashing JSON serialisation.

#### Reverted — deliberate non-goal

- An experimental addition of `.estimate` / `.se` / `.pvalue` / `.ci`
  properties to `EconometricResults` was *reverted* when regression
  testing showed it broke `agent/tools.py` and `causal_workflow.py`
  which use `hasattr(r, 'estimate')` to dispatch between scalar
  `CausalResult` and multi-coef `EconometricResults`. A NOTE in
  `core/results.py` documents why the aliases are intentionally
  absent; use `.tidy()` for cross-estimator code.

#### Tests (article-surface round-2)

`tests/test_article_aliases_round2.py` adds 25 tests covering all of
the above, including the conflict-detection and backend-signature
branches flagged by the round-2 code review.

---

## [0.9.9] - 2026-04-20 — Joint first-stage MTE + policy-relevant weights + honesty pass

Closes v0.9.8's two explicit follow-ons (joint first stage,
policy-relevant weights) and ships a **semantic correction** on the
MTE labelling that survived two rounds of code review.

### Added (0.9.9)

- **`sp.bayes_mte(..., first_stage='plugin' | 'joint')`** — new
  kwarg. `'plugin'` (default) preserves v0.9.8 behaviour: logit MLE
  computes propensity as a fixed constant. `'joint'` puts the
  first-stage logit coefficients inside the PyMC graph
  (`pi_intercept`, `pi_Z`, optional `pi_X`), models
  `D ~ Bernoulli(sigmoid(pi'W))`, and evaluates the effect
  polynomial at the random propensity — so first-stage uncertainty
  propagates into the returned curve. 2-4× slower than plug-in but
  honest about identification noise.

- **`BayesianMTEResult.policy_effect(weight_fn, label, rope=None)`**
  (`src/statspai/bayes/_base.py`) — posterior summary of
  `int w(u) g(u) du / int w(u) du` using trapezoidal integration
  on the fit's `u_grid`. With `policy_weight_ate()` it is now
  **numerically identical** to `.ate` (both trapezoid on the same
  grid) — test asserts `< 1e-8` parity.

- **`sp.policy_weight_*`** — four weight-function builders
  (`src/statspai/bayes/policy_weights.py`):
  - `policy_weight_ate()` — uniform weight = 1.
  - `policy_weight_subsidy(u_lo, u_hi)` — indicator on `[u_lo, u_hi]`.
  - `policy_weight_prte(shift)` — **stylised** rectangle around the
    mean propensity. The docstring leads with "NOT the textbook
    Carneiro-Heckman-Vytlacil 2011 PRTE" and shows a worked
    `scipy.stats.gaussian_kde` snippet users can adapt for the
    true CHV PRTE with their observed propensity sample.
  - `policy_weight_marginal(u_star, bandwidth)` — marginal PRTE at
    a specific propensity via a narrow band.

### Semantic correction (honesty pass)

- **Labelling fix**: v0.9.8's fit was described as the "MTE curve",
  but the structural model fits `g(p) = E[Y|D=1,P=p] - E[Y|D=0,P=p]`
  — the *treatment-effect-at-propensity* function. Under the
  Heckman-Vytlacil (2005) linear-separable + bivariate-normal
  assumption, `g(p) = MTE(p)`; under arbitrary heterogeneity,
  `g(p)` is a LATE summary at propensity `p`, not the textbook
  MTE(u). The module docstring now leads with this caveat and the
  method label reads `"Bayesian treatment-effect-at-propensity"`
  rather than `"Bayesian MTE"`. Function name, result class name,
  and the `mte_curve` field are unchanged for API continuity — the
  "MTE" naming is retained because applied users expect it and
  search for it.

### Performance

- Removed `pm.Deterministic('p', ...)` from joint mode. Under large
  `n`, storing per-unit propensity per draw was
  `O(chains × draws × n)` memory (e.g. 64MB at n=1000, draws=2000,
  chains=4). Post-hoc ATT/ATU propensity is now recomputed from
  the posterior means of `pi_intercept` / `pi_Z` / `pi_X`.

### Tests (0.9.9)

- **`tests/test_bayes_mte_policy.py`** (NEW, 14 tests) — builders'
  input validation (bad bounds rejected, FP-safe grids), joint
  mode runs + agrees with plug-in on well-specified DGPs,
  policy_effect contract, trapezoid parity with `.ate` at 1e-8,
  top-level export of all four weight builders.

### Code review

- Round-A (agent) found **4 items**: B1 (semantic mislabel),
  H1 (normalisation inconsistency), H2 (memory blow-up under
  joint+ADVI), M1 (PRTE-builder naming).
- Round-B (agent) on the fixes confirmed **no remaining blockers**;
  one follow-up (test tolerance too loose after the H1 fix) was
  applied inline before shipping.

### Design spec (0.9.9)

- `docs/superpowers/specs/2026-04-20-v099-mte-joint-policy-weights.md`

### Non-goals (0.9.9)

- Fully H-V-faithful joint model (sampling latent `U_D` per unit) —
  still a future release. Documented as the natural 0.9.10+ extension.
- Multi-instrument MTE with per-instrument PRTE weights.
- Gaussian-process surfaces on `u` (current release is polynomial).
- Rust Phase 2 — branch work.

---

## [0.9.8] - 2026-04-20 — Bayesian Marginal Treatment Effects + Pathfinder / SMC backends

Closes the two explicit next-batch items from v0.9.7's non-goals
list. Ships **the first Bayesian Marginal Treatment Effect
estimator in the Python causal-inference stack** and extends the
sampler dispatch with two new backends.

### Added (0.9.8)

- **`sp.bayes_mte(data, y, treat, instrument, covariates=None, u_grid=..., poly_u=2, ...)`**
  (`src/statspai/bayes/mte.py`) — Heckman-Vytlacil (2005) Marginal
  Treatment Effects via PyMC. Returns a `BayesianMTEResult` with:
  - `.mte_curve` — DataFrame on the user-supplied (or default
    19-point) grid of propensity-to-be-treated values ``U_D``:
    columns ``u, posterior_mean, posterior_sd, hdi_low, hdi_high,
    prob_positive``.
  - `.ate`, `.att`, `.atu` — integrated MTE over the population /
    treated / untreated regions.
  - `.plot_mte()` — quick matplotlib visualisation of the MTE curve
    with an HDI ribbon.

  Uses a plug-in logit first stage (same pragmatic shortcut as
  `bayes_iv`): the Bayesian layer lies over the MTE polynomial
  coefficients only. Asymptotically correct under correctly
  specified first stage; explicit non-goal is full joint
  first-stage-+-MTE posterior (queued for 0.9.9+).

- **`inference='pathfinder'`** — new sampler backend routing to
  PyMC's `pm.fit(method='fullrank_advi')`. Captures pairwise
  covariance between parameters (mean-field ADVI misses this) at
  similar speed. Placeholder for when PyMC's `pmx.fit` stabilises;
  full-rank ADVI is the same spirit.

- **`inference='smc'`** — new sampler backend routing to PyMC's
  `pm.sample_smc`. Sequential Monte Carlo; slower than NUTS on
  unimodal posteriors but robust to multi-modal ones where NUTS
  gets stuck. Unlike ADVI / Pathfinder, SMC returns a multi-chain
  trace so R-hat stays meaningful.

- **`BayesianMTEResult`** — top-level export
  (`sp.BayesianMTEResult`). Inherits `BayesianCausalResult` and
  adds `mte_curve`, `u_grid`, `ate`, `att`, `atu`, `.plot_mte()`.

- **Summary output** now recognises the full sampler menu:
  - NUTS / SMC: R-hat is meaningful; flagged on > 1.01.
  - ADVI / Pathfinder: R-hat is variational and flagged as such
    with a concrete "use NUTS or SMC for calibrated uncertainty"
    caveat.

### Design spec (0.9.8)

- `docs/superpowers/specs/2026-04-20-v098-bayes-mte-samplers.md`

### Tests (0.9.8)

- **`tests/test_bayes_mte.py`** (9 tests) — API surface, flat-MTE
  recovery, monotone-MTE slope recovery, custom `u_grid`, `poly_u=1`
  path, covariate plumbing, top-level export, missing-column and
  non-binary-treat validation.
- **`tests/test_bayes_advi.py`** (+5 tests) — Pathfinder on
  bayes_iv and bayes_did, SMC on bayes_iv and bayes_did, Pathfinder
  summary() caveat.

### Non-goals (0.9.8, explicit)

- Full joint first-stage + MTE posterior (propagating first-stage
  uncertainty into `tau(u)`). Plug-in propensity is the v0.9.8
  choice — correct asymptotically under correctly specified first
  stage; next release can add a joint model.
- Multi-instrument MTE — requires policy-relevant weighting
  (Carneiro-Heckman-Vytlacil 2011) and is out of scope.
- Non-linear MTE surfaces (GP over `u`) — polynomial of order
  `poly_u` is what this release supports.
- Rust Phase 2 — stays on `feat/rust-hdfe` branch.

---

## [0.9.7] - 2026-04-20 — Heterogeneous-effect Bayesian IV + ADVI toggle

Closes two of the three items queued at v0.9.6's "诚实汇报" list.
The third (Bayesian bunching) is **explicitly declined** — see the
"Non-goals" section below.

### Added (0.9.7)

- **`sp.bayes_hte_iv(data, y, treat, instrument, effect_modifiers, ...)`**
  (`src/statspai/bayes/hte_iv.py`) — Bayesian IV with a linear
  CATE-by-covariate model. Returns a `BayesianHTEIVResult` carrying:
  - Average LATE (`tau_0`, at modifier means) with posterior + HDI.
  - `.cate_slopes` DataFrame — one row per effect modifier with
    posterior mean, SD, HDI, and `prob_positive`.
  - `.predict_cate(values: dict) -> dict` — posterior summary of
    the CATE at user-specified modifier values.

  Model:

      D = pi_0 + pi_Z' Z + pi_X' X + v
      tau(M) = tau_0 + tau_hte' (M - M_bar)
      Y = alpha + tau(M) * D + beta_X' X + rho * v_hat + eps

  Control-function formulation keeps NUTS sampling tractable.
  Multiple instruments + multiple modifiers + exogenous controls
  all supported.

- **`inference='nuts' | 'advi'`** parameter on every Bayesian
  estimator — `bayes_did`, `bayes_rd`, `bayes_iv`, `bayes_fuzzy_rd`,
  and the new `bayes_hte_iv`. Under `'advi'` the estimator goes
  through `pm.fit(method='advi')` for a 10-50× speedup at the cost
  of mean-field calibration. `rhat` is reported as `NaN` in ADVI
  mode (no meaning for variational approximations).

  A shared `_sample_model` helper now owns sampling dispatch, so
  future backends (`'smc'`, `'pathfinder'`) plug in trivially.

- **`BayesianHTEIVResult`** — top-level export
  (`sp.BayesianHTEIVResult`). Extends `BayesianCausalResult` with
  `cate_slopes`, `effect_modifiers`, and `predict_cate(...)`.

### Design spec (0.9.7)

- `docs/superpowers/specs/2026-04-20-v097-bayes-hte-iv-advi.md`

### Tests (0.9.7)

- **`tests/test_bayes_hte_iv.py`** (8 tests) — API surface, avg-LATE
  recovery on heterogeneous DGP, slope recovery, null-slope
  coverage on homogeneous DGP, `predict_cate` schema, multi-modifier
  fit, input validation.
- **`tests/test_bayes_advi.py`** (10 tests) — ADVI runs on all five
  Bayesian estimators, posterior means finite,
  `model_info['inference']` reports correctly, invalid inference
  modes raise for every estimator (parametrised over 5 functions).

### Non-goals (0.9.7, explicitly declined)

- **Bayesian bunching** (`sp.bayes_bunching`) — after review we
  decline. Kleven / Saez / Chetty bunching estimators are
  *structural* public-finance models whose identification depends
  on utility / optimisation parameterisations that don't generalise
  across kink types, priors on taste heterogeneity that are
  domain-specific and hard to default well, and model fits only as
  interpretable as the structural model itself. This defeats the
  package's "agent-native one-liner" thesis. The frequentist
  `sp.bunching` stays where it is. We revisit only on a concrete
  user use-case that fits the agent-native workflow.

- MTE / complier-heterogeneity IV — queued for 0.9.8+.
- Extra VI backends beyond ADVI (Pathfinder, SMC) — `_sample_model`
  is now extensible but the backends stay out of this release.
- Rust Phase 2 — on `feat/rust-hdfe` branch until the cibuildwheel
  matrix is green.

---

## [0.9.6] - 2026-04-20 — Bayesian IV + fuzzy RD + per-learner Optuna + Rust branch + g-methods family

This release bundles two independent sprints that landed the same day:

### Sprint A — Bayesian depth + tuning granularity + Rust branch

1. Bayesian 口袋深度 — adds `sp.bayes_iv` and `sp.bayes_fuzzy_rd`.
2. Optuna 粒度 — `sp.auto_cate_tuned` now supports `tune='nuisance'`
   (v0.9.5 behaviour), `tune='per_learner'`, and `tune='both'`.
3. Rust 工作流 — `feat/rust-hdfe` branch opened with Cargo crate
   scaffold; `main` stays maturin-free.

### Sprint B — G-methods family, Proximal, Principal Stratification

Closes a causal-inference-coverage audit against the 2026-04-20 gap
table: ships DML IIVM, g-computation, front-door estimator, MSM,
interventional mediation, plus two new top-level modules
**Proximal Causal Inference** and **Principal Stratification**. After
self-review, a second pass re-polished weight-semantics, bootstrap
diagnostics, MC vectorisation, and did a full DML internal refactor
(four per-model files sharing `_DoubleMLBase`).

### Added

- **`sp.bayes_iv(data, y, treat, instrument, covariates=None, ...)`**
  (`src/statspai/bayes/iv.py`) — Bayesian linear IV via a
  control-function formulation. First-stage OLS residuals enter the
  structural equation as an exogeneity correction, so the posterior
  on the LATE equals 2SLS asymptotically while remaining trivially
  sampleable in PyMC. Accepts a single instrument or a list. The HDI
  widens naturally as the instrument gets weaker (no "F < 10" cliff
  — the posterior prices identification automatically).

- **`sp.bayes_fuzzy_rd(data, y, treat, running, cutoff, ...)`**
  (`src/statspai/bayes/fuzzy_rd.py`) — Bayesian fuzzy RD via joint
  ITT-on-Y and ITT-on-D local polynomials with a deterministic
  ratio for the LATE. Under partial compliance the posterior
  inherits both noise channels (Wald-ratio posterior); under full
  compliance it collapses to the sharp RD result. Non-binary uptake
  is rejected with a clear error. `model_info` reports
  `first_stage_mean` / `first_stage_sd` so users can eyeball
  compliance.

- **`sp.auto_cate_tuned(..., tune='nuisance' | 'per_learner' | 'both')`** —
  new `tune` flag toggles between three regimes:

  - `'nuisance'` (default, v0.9.5 behaviour): shared outcome /
    propensity GBMs tuned against held-out R-loss.
  - `'per_learner'`: each learner's final-stage CATE model is
    tuned independently against held-out R-loss; nuisance stays
    at defaults. `model_info['per_learner_params']` and
    `['per_learner_r_loss']` are populated; the best learner's
    tuned CATE model is fed to `auto_cate` as a hint.
  - `'both'`: tune the nuisance first, then per-learner CATE on
    top of that nuisance.

  Also adds `n_trials_per_learner` (defaults to `max(5, n_trials//3)`)
  and `per_learner_search_space`. Selection-rule text now records
  which tuning regime ran.

- **`feat/rust-hdfe` branch** (pushed, not merged) — Cargo crate
  scaffold plus PyO3 stub for the eventual `group_demean` kernel.
  `main` stays maturin-free so `pip install statspai` is unaffected.

### Design spec

- `docs/superpowers/specs/2026-04-20-v096-bayes-iv-fuzzyrd-perlearner.md`

### Tests

- **`tests/test_bayes_iv.py`** (8 tests) — API, top-level export,
  strong-IV recovery, weak-IV HDI widens, multi-instrument fit,
  covariate plumbing, input validation, tidy/glance shape.
- **`tests/test_bayes_fuzzy_rd.py`** (7 tests) — API, recovery
  under partial compliance, sharp-equivalence under full
  compliance, bandwidth shrinks sample, first-stage diagnostics
  reported, non-binary uptake rejected.
- **`tests/test_auto_cate_tuned.py`** (+5 tests) — invalid
  `tune` mode rejected, `'per_learner'` populates params, no
  nuisance metadata leaks in per_learner mode, `'both'` mode
  covers both channels, selection_rule mentions per-learner tuning.

### Non-goals (deferred)

- Bunching Bayesian estimator (Kleven-style is structural /
  macro-flavoured; poor fit for the agent-native API). Queue for
  0.9.7.
- Heterogeneous-effect Bayesian IV — LATE only in this release.
- VI sampler (ADVI) — NUTS only.
- Rust kernel merged to `main` — stays on `feat/rust-hdfe` until
  the cibuildwheel matrix is green.

### Added (Sprint B)

- **`sp.dml(..., model='iivm', instrument=Z)`**
  (`src/statspai/dml/iivm.py`) — Interactive IV (binary D, binary Z)
  DML estimator for LATE. Uses the efficient-influence-function ratio
  of two doubly-robust scores `(ψ_a, ψ_b)` with Neyman-orthogonal
  cross-fitting; SE via delta-method on the ratio. Weak-instrument
  guard raises `RuntimeError` when `|E[ψ_b]| ≈ 0`. Class form:
  `sp.DoubleMLIIVM`.

- **`sp.DoubleMLPLR / DoubleMLIRM / DoubleMLPLIV / DoubleMLIIVM`**
  (`src/statspai/dml/*.py`) — each DML model family now lives in its
  own file with a shared `_DoubleMLBase` in `dml/_base.py` that
  handles validation, default learners (auto-selecting classifier vs
  regressor per model), cross-fitting, and `CausalResult` construction.
  The legacy `sp.DoubleML(model=...)` façade still works.

- **`sp.g_computation(data, y, treat, covariates, estimand='ATE'|'ATT'|'dose_response', ...)`**
  (`src/statspai/inference/g_computation.py`) — Robins' (1986)
  parametric g-formula / standardisation estimator. Supports binary
  treatment (ATE, ATT) and continuous treatment dose-response grids.
  Default OLS outcome model or any sklearn-compatible learner via
  `ml_Q=`. Nonparametric bootstrap SE with NaN-based failure tracking
  (`model_info['n_boot_failed']`) — replaces silent point-estimate
  fallback that would shrink SE.

- **`sp.front_door(data, y, treat, mediator, covariates=None, mediator_type='auto', integrate_by='marginal'|'conditional', ...)`**
  (`src/statspai/inference/front_door.py`) — Pearl (1995) front-door
  adjustment estimator. Closed-form sums for binary mediator; Monte
  Carlo integration over a Gaussian conditional density for continuous
  mediator. Two identification variants exposed: `integrate_by='marginal'`
  (Pearl 95 aggregate formulation) and `'conditional'` (Fulcher et al.
  2020 generalised front-door). Bootstrap SE with NaN-based failure
  tracking.

- **`sp.msm(data, y, treat, id, time, time_varying, baseline=None, exposure='cumulative'|'current'|'ever', family='gaussian'|'binomial', trim=0.01, ...)`**
  (`src/statspai/msm/`) — Robins-Hernán-Brumback (2000) Marginal
  Structural Models via stabilised IPTW. Handles time-varying
  treatment + time-varying confounders (binary or continuous).
  Weighted pooled regression of outcome on exposure history with
  cluster-robust CR1 sandwich at the unit level.
  `sp.stabilized_weights(...)` is exposed as a standalone helper for
  users who want the weights without fitting the outcome model.

- **`sp.mediate_interventional(data, y, treat, mediator, covariates=None, tv_confounders=None, ...)`**
  (`src/statspai/mediation/mediate.py`) — VanderWeele, Vansteelandt &
  Robins (2014) interventional (in)direct effects. Identified in the
  presence of a treatment-induced mediator-outcome confounder
  (`tv_confounders=[...]`) where natural (in)direct effects are not.
  Fully vectorised MC integration (~100× faster than naïve
  per-observation loop).

- **`sp.proximal(data, y, treat, proxy_z, proxy_w, covariates=None, n_boot=0, ...)`**
  (`src/statspai/proximal/`) — Proximal Causal Inference (Tchetgen
  Tchetgen et al. 2020; Miao, Geng & Tchetgen Tchetgen 2018) via
  linear 2SLS on the outcome bridge function. Handles ATE
  identification with an unobserved confounder when two proxies
  (treatment-side `Z` and outcome-side `W`) are available. Reports a
  first-stage F-stat for the proxy equation and warns when F < 10.
  Optional nonparametric bootstrap SE via `n_boot=`.

- **`sp.principal_strat(data, y, treat, strata, covariates=None, method='monotonicity'|'principal_score', ...)`**
  (`src/statspai/principal_strat/`) — Principal Stratification
  (Frangakis & Rubin 2002). `method='monotonicity'` applies the
  Angrist-Imbens-Rubin compliance decomposition to identify the
  complier PCE (= LATE) and returns Zhang-Rubin (2003) sharp bounds
  for the always-survivor SACE. `method='principal_score'` implements
  Ding & Lu (2017) principal-score weighting to point-identify
  always-taker / complier / never-taker PCEs under principal
  ignorability. Returns a dedicated `PrincipalStratResult` with
  `strata_proportions`, `effects`, `bounds`.

- **`sp.survivor_average_causal_effect(data, y, treat, survival, ...)`**
  — friendly wrapper around `principal_strat(method='monotonicity')`
  for the classical truncation-by-death problem. Reports SACE
  midpoint + endpoint-union confidence interval.

### Changed (Sprint B)

- **MSM binomial outcome family**: `_weighted_logit_cluster` replaced
  the previous `statsmodels.GLM(freq_weights=w)` call (which treats
  weights as integer replication counts) with a hand-rolled IRLS that
  uses probability-weight semantics. Matches Cole & Hernán (2008) and
  Stata's `pweight` convention for IPTW.

- **Bootstrap failure reporting**: `g_computation`,
  `mediate_interventional`, `front_door`, and `proximal` now leave
  failed bootstrap replications as `NaN`, emit a `RuntimeWarning`
  with the failure count and first error message, and record
  `n_boot_failed` / `n_boot_success` / `first_bootstrap_error` in
  `model_info`. If fewer than two replications succeed, a clean
  `RuntimeError` is raised rather than silently under-estimating SE.

- **`mediate_interventional` MC loop**: the previous `O(n × n_mc)`
  Python comprehension is replaced by a closed-form vectorisation
  that exploits OLS linearity of the outcome model in the
  treatment-induced-confounder block (`X_tv`). The outer expectation
  over units collapses to `β_tv · mean(X_tv)`, reducing runtime to
  `O(n_mc + n)` and giving a measured ~100× speed-up on the
  reference configuration (n=800, n_boot=200, n_mc=300 drops from
  ~4 s to ~0.04 s).

- **`sp.dml` internal layout**: the 466-line single-class
  `dml/double_ml.py` is split into five files
  (`_base.py` + `plr.py` + `irm.py` + `pliv.py` + `iivm.py`) each
  owning a single Neyman-orthogonal score and its validation. The
  public `dml()` function and `DoubleML` class are unchanged; new
  per-model classes are now directly importable.

- `sp.front_door` with covariates and continuous mediator gained
  `integrate_by` (see Added).

### Tests (Sprint B)

- **`tests/test_dml_iivm.py`** (5 tests) — LATE recovery on
  one-sided-noncompliance DGP, significance, binary-D/binary-Z
  validation, `model_info` fields.
- **`tests/test_dml_split.py`** (5 tests) — direct-class API equals
  dispatcher, legacy `DoubleML` façade, PLIV rejects multi-instrument
  list.
- **`tests/test_g_computation.py`** (5 tests) — ATE / ATT /
  dose-response curves recovered within tolerance, validation errors.
- **`tests/test_front_door.py`** (4 tests) — continuous-M and
  binary-M ATE recovery on DGP with unobserved confounder, strictly
  closer to truth than naïve OLS.
- **`tests/test_front_door_integrate_by.py`** (3 tests) — marginal
  and conditional variants both recover truth, invalid values rejected.
- **`tests/test_msm.py`** (5 tests) — cumulative-exposure slope
  recovery, stabilised-weight shape / mean, `exposure='ever'`
  requires binary treatment, weight diagnostics exposed.
- **`tests/test_mediate_interventional.py`** (4 tests) — IIE + IDE
  decomposition additivity, total-effect sign, binary-D validation.
- **`tests/test_proximal.py`** (6 tests) — linear-bridge ATE
  recovery, strictly-better-than-OLS, order-condition check,
  covariate compatibility, bootstrap SE path, first-stage F reported.
- **`tests/test_principal_strat.py`** (7 tests) — monotonicity LATE
  + stratum proportions, valid SACE bounds, principal-score method
  with informative X, input validation, SACE helper.

### Notes (Sprint B)

- No new required dependency. All additions use NumPy / pandas /
  scipy / scikit-learn only (statsmodels optional).
- Full new-module suite: 44 new tests pass; the existing 28
  DML + mediation regression tests still pass; full collection
  reports 1960 tests, zero import errors introduced by this sprint.

---

## [0.9.5] - 2026-04-20 — Bayesian causal + Optuna-tuned CATE + Rust spike

This release closes three items from the v0.9.4 post-release
retrospective (Section 8 "认怂" list):

1. **Bayesian causal** — `sp.bayes_did` + `sp.bayes_rd` (PyMC).
2. **ML CATE調参** — `sp.auto_cate_tuned` (Optuna).
3. **Rust HDFE kernel** — spec + benchmark harness shipped;
   actual Rust crate deferred to 1.0 on a dedicated branch (any
   `maturin` change to `pip install` is postponed until a full
   cross-platform wheel matrix is green).

### Added

- **`sp.bayes_did(data, y, treat, post, unit=None, time=None, ...)`**
  (`src/statspai/bayes/did.py`) — Bayesian difference-in-differences
  via PyMC. 2×2 for no panel indices, hierarchical Gaussian random
  effects when `unit` and/or `time` are supplied. NUTS sampler,
  configurable priors, `rope=(lo, hi)` for "practical equivalence"
  posterior probabilities. Returns a `BayesianCausalResult` with
  posterior mean/median/SD, 95 % HDI, `prob_positive`, `rhat`, `ess`,
  and the full ArviZ `InferenceData` on `.trace` for downstream
  plotting.

- **`sp.bayes_rd(data, y, running, cutoff, bandwidth=None, poly=1, ...)`**
  (`src/statspai/bayes/rd.py`) — Bayesian sharp regression
  discontinuity with local polynomial (order ≥ 1) and Normal prior
  on the jump. Bandwidth defaults to `0.5 * std(running)`.

- **`sp.BayesianCausalResult`** — sibling of `CausalResult` with
  broom-style `.tidy()` / `.glance()` / `.summary()` and
  Bayesian-native fields (`hdi_lower`, `hdi_upper`, `prob_positive`,
  `prob_rope`, `rhat`, `ess`). Slots into the same agent-native
  `pd.concat([r.tidy() for r in results])` workflow as the
  frequentist estimators.

- **`sp.auto_cate_tuned(..., n_trials=25, timeout=None, search_space=None)`**
  (`src/statspai/metalearners/auto_cate_tuned.py`) — Optuna's
  `TPESampler` searches over the nuisance GBM hyperparameters
  (outcome and propensity model separately), scoring each trial by
  shared-nuisance held-out R-loss. Best trial's models are handed to
  `sp.auto_cate`; the winner's `model_info['tuned_params']` records
  the chosen HP and `['n_trials']` the search budget. Closes the
  econml "nuisance cross-validation before CATE" ergonomic gap.

- **`sp.fast.hdfe_bench(n_list, n_groups, repeat, seed, atol)`**
  (`src/statspai/fast/bench.py`) — benchmark harness for HDFE
  group-demean kernels. Times NumPy, Numba, and (future) Rust paths
  on the same DGPs and asserts correctness to ≤ 1 × 10⁻¹⁰ vs the
  NumPy reference. Unavailable backends are recorded, not crashed,
  so the same harness runs on CI environments that lack Numba and on
  dev boxes with a future Rust wheel installed.

- **Optional install extras**: `pip install "statspai[bayes]"` pulls
  `pymc >= 5` + `arviz >= 0.15`. `pip install "statspai[tune]"`
  pulls `optuna >= 3`. Core `import statspai` works in either's
  absence; the estimators raise a clean `ImportError` at call time
  with the install recipe.

### Design docs

- `docs/superpowers/specs/2026-04-20-v095-bayes-optuna-rust-spike.md`
  — full spec for this release.
- `docs/superpowers/specs/2026-04-20-v095-rust-hdfe-spike.md` — the
  phased plan for the Rust HDFE port (crate layout, PyO3 FFI
  surface, cibuildwheel matrix, graceful-degradation contract).

### Tests

- **`tests/test_bayes_did.py`** (11 tests) — 2×2 + panel recovery,
  prob_positive calibration, HDI coverage, input validation, ROPE,
  tidy/glance shape.
- **`tests/test_bayes_rd.py`** (9 tests) — sharp recovery, null-effect
  HDI straddles 0, bandwidth shrinks local sample, poly=2 runs,
  validation errors.
- **`tests/test_auto_cate_tuned.py`** (7 tests) — API, `n_trials`
  respected, ATE recovery, custom search space honoured, invalid
  treatment rejected.
- **`tests/test_fast_bench.py`** (5 tests) — harness returns
  `HDFEBenchResult`, dry-run <5 s, Numba/NumPy agree to 1e-10,
  unavailable paths recorded not crashed, summary string.

### Non-goals (explicit)

- **Variational inference** (`pymc.fit` ADVI) — NUTS only for 0.9.5.
- **Bayesian fuzzy RD, IV, bunching** — deferred to 0.9.6+.
- **Rust crate itself** — ships on a dedicated branch with a full
  `cibuildwheel` matrix; adding `maturin` to `pyproject.toml` without
  that matrix would break `pip install` for some users.

---

## [0.9.4] - 2026-04-20 — `sp.auto_cate` + strict identification

This release closes two concrete commitments from the 0.9.3 post-release
retrospective (`社媒文档/4.20-升级说明/StatsPAI-0.9.3之后的一周…`):

1. **Section 5 promise**: *"下一步打算加 `strict_mode=True`"* on
   `sp.check_identification`. Delivered as `strict=True` plus the
   `sp.IdentificationError` exception.
2. **Section 8 gap**: *"ML CATE scheduling isn't as good as econml."*
   Delivered as `sp.auto_cate()` — one-line multi-learner race with
   honest Nie-Wager R-loss scoring and BLP calibration.

### Added

- **`sp.auto_cate(data, y, treat, covariates, learners=('s','t','x','r','dr'))`**
  (`src/statspai/metalearners/auto_cate.py`, +400 LOC) — races the five
  meta-learners on shared cross-fitted nuisances, scores each on
  held-out predictions via the Nie-Wager R-loss, runs the
  Chernozhukov-Demirer-Duflo-Fernández-Val BLP calibration test on
  each, and returns an `AutoCATEResult` with:
  - `.leaderboard` — sorted by R-loss, with ATE, SE, CI, BLP β₁/β₂,
    CATE std/IQR per learner;
  - `.best_learner` / `.best_result` — winner selected by lowest
    held-out Nie-Wager R-loss; BLP β₁/β₂ are reported in the
    leaderboard as diagnostics, not selection gates (β₁ equals the
    ATE in units of Y in this parametrization, so there is no
    natural "β₁ ≈ 1" gate);
  - `.results` — the full fitted `CausalResult` for every learner;
  - `.agreement` — Pearson-ρ matrix of in-sample CATE vectors across
    learners (quick sanity check for model dependence);
  - `.summary()` — a printable leaderboard + agreement table.

  Python's first unified CATE learner race with honest held-out
  scoring. `econml`'s multi-metalearner pipeline is not bundled into
  a single call; `causalml`'s BaseMetaLearner comparison doesn't run
  BLP calibration per learner.

- **`sp.check_identification(..., strict=True)`** raises
  `sp.IdentificationError` when the report's verdict is `'BLOCKERS'`.
  The exception carries the full report on `.report` for post-mortem
  inspection. Default remains `strict=False` (non-breaking).

- **`sp.IdentificationError`** — new exception type, exported at the
  top level.

- **IV first-stage strength check** in `sp.check_identification`
  (`_check_iv_strength`) — computed from a first-stage OLS
  `treatment ~ intercept + covariates + instrument` (covariates
  partialled out before computing the instrument's F, so the
  reported F matches the Staiger-Stock definition when controls are
  present). Flags F < 5 as `blocker`, F < 10 as `warning`
  (Staiger-Stock 1997), F ∈ [10, 30) as `info`. Fires only when
  `instrument` is supplied.

### Tests

- **`tests/test_auto_cate.py`** (13 tests) — API surface, leaderboard
  shape, ATE recovery on constant-effect DGP, all-positive ATE on
  positive DGP, learner subset, invalid learner rejection, selection
  rule string, agreement matrix, `CausalResult` delegation
  (`.tidy()`, `.glance()`), custom model override, summary string,
  top-level `sp.*` availability, heterogeneous-DGP CATE dispersion.
- **`tests/test_check_identification.py`** (+5 tests) — `strict=True`
  raises on blockers, tolerates warnings, default non-strict
  behaviour unchanged, `sp.IdentificationError` top-level export,
  weak-instrument flagged, strong-instrument not flagged.

### Design

- Published spec at
  `docs/superpowers/specs/2026-04-20-v094-auto-cate-strict-id-design.md`.

### Non-goals (deferred to 0.9.5+)

- Optuna hyperparameter search inside `auto_cate` — for now the user
  either accepts the boosted-tree defaults or passes pre-tuned
  estimators via `outcome_model=`/`propensity_model=`/`cate_model=`.
- Bayesian `sp.bayes_did` / `sp.bayes_rd` — announced as a 0.9.5
  preview line.
- Rust HDFE inner kernel — remains Section 8's open item.

---

## [Unreleased] — 0.9.3 post-release bugfixes

Four user-reported bugs surfaced during the 0.9.3 end-to-end smoke test.
All are fixed on `main` without a version bump (pending a later patch release).

### Fixed

- **`sp.use_chinese()` failed on Linux** (`plots/themes.py`) — the auto-detect
  candidate list only covered macOS fonts plus `Noto Sans CJK SC` and
  `WenQuanYi Micro Hei`, so a Linux/Docker host with `fonts-noto-cjk` (which
  ships `Noto Sans CJK JP/TC/KR` by default) or `fonts-wqy-zenhei`
  (`WenQuanYi Zen Hei`) installed got an empty return plus a "no Chinese
  font" warning. Priority lists are now segmented by platform (macOS →
  Windows → Linux → cross-platform Source Han), all four Noto CJK regional
  variants are listed, and a substring fallback (`CJK`, `Han Sans`,
  `Han Serif`, `WenQuanYi`, `Heiti`, `Ming`) picks up custom/renamed builds.
  Warning message now includes the exact `apt install fonts-noto-cjk
  fonts-wqy-zenhei` recipe.

- **`sp.regtable(...)` printed the table twice in REPL/Jupyter**
  (`output/regression_table.py`, `output/estimates.py`) — `regtable()`,
  `mean_comparison()` and `esttab()` each called `print(result)` internally
  and then returned the result, which REPL/Jupyter re-displayed via
  `__repr__`/`_repr_html_`. All three internal prints are removed; display
  now flows through the standard Python display protocol.

  **Behaviour change**: scripts that relied on the auto-print side-effect
  must switch to `print(sp.regtable(...))`. Jupyter and interactive REPLs
  are unaffected.

- **`sp.regtable(..., output="latex")` was silently ignored**
  (`output/regression_table.py`) — the `output=` parameter previously
  controlled only the Word/Excel warning branch; `__str__` always rendered
  text. `RegtableResult` and `MeanComparisonResult` now store `_output` and
  dispatch in `__str__`/`__repr__` through `_render(fmt)` over
  `{text, latex, tex, html, markdown, md}`. Jupyter's `_repr_html_` still
  always returns HTML. Invalid `output=` values now raise `ValueError`
  instead of falling back silently.

- **`sp.did()` `treat=` column semantics were easy to mis-specify**
  (`did/__init__.py`) — for staggered designs the column must hold each
  unit's first-treatment period (never-treated = `0`, **not** `1`), but
  users with a pre-existing 0/1 `treated` column consistently passed it
  straight through and got nonsense estimates. Docstring now carries an
  explicit callout and a verified pandas idiom for constructing
  `first_treat` (`.loc[treated==1].groupby('id')['year'].min()` + `.map` +
  `.fillna(0)`) that broadcasts correctly to pre-treatment rows.

### Added

- Documentation clarifies that `regtable(output=...)` controls `str(result)`
  while `regtable(filename=...)` dispatches on the file extension — they can
  diverge, and users should pass matching values.
- Input validation on `regtable()` / `mean_comparison()` rejects unknown
  `output=` values with a helpful `ValueError` listing valid choices.

### Tests

`tests/test_v093_bugfixes.py` — 15 regression tests covering all four bugs
plus the new validation. Full suite: 1655 passed, 4 skipped, 0 regressions.

---

## [0.9.3] - 2026-04-19 — Stochastic Frontier + Multilevel + GLMM + Econometric Trinity

**Overview.** This release bundles four simultaneous deep overhauls plus an
author-metadata correction:

1. **Stochastic Frontier Analysis** — `sp.frontier` / `sp.xtfrontier` rewritten
   to Stata/R-grade, with a critical correctness bug fix.
2. **Multilevel / Mixed-Effects** — `sp.multilevel` rewritten to lme4/Stata-grade.
3. **GLMM hardening** — AGHQ (`nAGQ>1`) plus three new families (Gamma,
   Negative Binomial, Ordinal Logit) and cross-family AIC comparability.
4. **Econometric Trinity** — three new P0 pillars: DML-PLIV, Mixed Logit, IV-QR.
5. **Author attribution** corrected to `Biaoyue Wang`.

⚠️ **Critical correctness fix** — `sp.frontier` carried a latent Jondrow posterior
sign error in all prior versions (0.9.2 and earlier). Efficiency scores were
systematically biased; the normal-exponential path additionally returned NaN
for unit efficiency. **Re-run any prior frontier analyses.** Detail below.

---

### Stochastic Frontier Analysis Overhaul

Release focus: `statspai.frontier`. The prior implementation was a
270-line single file with one function covering cross-sectional
half-normal / exponential / truncated-normal frontiers, no panel
support, no heteroskedasticity, no inefficiency determinants, and —
critically — a sign error in the Jondrow posterior that silently
produced wrong efficiency scores, plus a wrong ε-coefficient in the
exponential log-likelihood that the old test never exercised. The
module has been rewritten (~1,300 LOC across `_core.py`, `sfa.py`,
`panel.py`, `te_tools.py`) to match or exceed Stata's
`frontier` / `xtfrontier` and R's `frontier` / `sfaR`.

### Correctness fixes

- **Jondrow posterior μ\***: corrected `sign` convention in all three
  distributions — the old code's `μ* = -sign·ε·σ_u²/σ²` has been
  replaced by the derivation-verified `μ* = sign·ε·σ_u²/σ²` (and the
  analogous correction for truncated-normal). Efficiency scores from
  the old implementation were systematically biased; re-run any prior
  analyses.
- **Normal-exponential log-density**: fixed the ε-coefficient and
  Φ argument (the old form was `+ sign·ε/σ_u + log Φ((-sign·ε - σ_v²/σ_u)/σ_v)`;
  correct per Greene 2008 eq. 2.39 is `- sign·ε/σ_u + log Φ(sign·ε/σ_v - σ_v/σ_u)`).
  The old exponential path never produced efficiency scores (returned NaN) —
  now returns correct Battese-Coelli scores.
- **Truncated-normal density**: fixed the `centered` offset in the
  φ factor from `(ε + sign·μ)/σ` to `(ε - sign·μ)/σ`.
- Monte-Carlo density-integration tests (`∫ f(ε) dε = 1`) now guard
  against regressions for all three distributions.

### New cross-sectional `sp.frontier`

- **Heteroskedastic inefficiency** via `usigma=[...]` — parameterises
  `ln σ_u_i = γ_u' [1, w_i]` (Caudill-Ford-Gropper 1995, Hadri 1999).
- **Heteroskedastic noise** via `vsigma=[...]` — parameterises
  `ln σ_v_i = γ_v' [1, r_i]` (Wang 2002).
- **Inefficiency determinants** via `emean=[...]` — the
  Battese-Coelli (1995) / Kumbhakar-Ghosh-McGuckin (1991) model
  `μ_i = δ' [1, z_i]` for `dist='truncated-normal'`.
- **Battese-Coelli (1988) TE**: `result.efficiency(method='bc')` returns
  `E[exp(-u)|ε]` (the Stata default) in addition to the JLMS
  approximation `exp(-E[u|ε])` (`method='jlms'`).
- **LR test for absence of inefficiency**: one-sided mixed χ̄²
  (Kodde-Palm 1986) via `result.lr_test_no_inefficiency()`.
- **Bootstrap CI for unit efficiency**: parametric-bootstrap bounds
  via `result.efficiency_ci(alpha=.05, B=500)`.
- **Residual skewness diagnostic** stored at
  `result.diagnostics['residual_skewness']`.
- Optimiser now has hard bounds on `ln σ` and guards against
  σ → 0 / σ → ∞ excursions that previously caused truncated-normal
  fits to diverge.

### New panel `sp.xtfrontier`

- **Pitt-Lee (1981) time-invariant** (`model='ti'`):
  `u_it = u_i`, half-normal or truncated-normal.  Closed-form group
  log-likelihood derived from the per-unit integration; unit-level
  TE stored at `result.diagnostics['efficiency_bc_unit']`.
- **Battese-Coelli (1992) time-varying decay** (`model='tvd'`):
  `u_it = exp(-η(t - T_i)) · u_i` with η estimated jointly.  The
  obs-level efficiency uses `E[exp(-a_it u_i)|e_i]` under the
  posterior `u_i ~ N⁺(μ*, σ*²)` (MGF form).
- **Battese-Coelli (1995) inefficiency effects** (`model='bc95'`):
  `u_it ~ N⁺(z_it' δ, σ_u²)` independently; returned with unit-mean
  efficiency roll-up.

### Helpers

- `sp.te_summary(result)` — Stata-style descriptive table of TE
  scores (n, mean, sd, quartiles, share > 0.9, share < 0.5).
- `sp.te_rank(result, with_ci=True)` — efficiency ranking with
  optional bootstrap CIs for benchmarking.

### Tests

- **33 new tests** covering: parameter recovery for all three
  cross-sectional distributions, cost vs production sign handling,
  heteroskedastic σ_u / σ_v, BC95 determinants, LR specification tests,
  TE-score bounds and internal consistency, bootstrap CI structure,
  Pitt-Lee / BC92 / BC95 panel recovery, and density-integrates-to-1
  kernel sanity checks.

### Advanced frontier extensions

Three frontier extensions shipped after the initial overhaul (commit `e876937`):

- **`sp.zisf`** — Zero-Inefficiency SFA mixture (Kumbhakar-Parmeter-Tsionas
  2013). Mixture of fully-efficient (`u=0`, pure noise) and standard
  composed-error regimes; mixing probability `p_i` parameterised via logit
  on optional `zprob` covariates. Posterior `P(efficient|ε)` exposed in
  `diagnostics['p_efficient_posterior']`. Recovery test: true efficient
  share 0.30 → estimated 0.286 on `n=2000`.
- **`sp.lcsf`** — 2-class Latent-Class SFA (Orea-Kumbhakar 2004;
  Greene 2005). Two separate frontiers with their own `β_k` and variance
  parameters; class-membership logit on optional `z_class` covariates.
  Direct MLE with perturbed starts to break label symmetry.
- **`xtfrontier(..., model='tfe', bias_correct=True)`** — Dhaene-Jochmans
  (2015) split-panel jackknife for TFE:
  `β_BC = 2·β_full − (β_first_half + β_second_half)/2`. Cuts the `O(1/T)`
  incidental-parameters bias. Guards against degenerate halves by skipping
  σ corrections with an annotation in `model_info`. Verified at `T=30`,
  `N=25`: raw `σ_u=0.374` → BC `σ_u=0.359` (true 0.35).

### Productivity helpers

Shipped in commit `be59260`:

- **`sp.malmquist`** — Färe-Grosskopf-Lindgren-Roos (1994) Malmquist TFP
  index via period-by-period parametric frontier fits. Returns per-
  transition decomposition `M = EC × TC` (efficiency change × technical
  change). Row-wise identity `M == EC·TC` verified to `rtol=1e-8`. Cost
  frontiers supported via reciprocal distance convention. Validated on
  3-period DGP with 5%/year intercept growth: mean TC ≈ 1.07–1.09,
  mean EC ≈ 1.0.
- **`sp.translog_design`** — Cobb-Douglas → Translog design-matrix helper.
  Appends `0.5·log(x_k)²` squares and `log(x_k)·log(x_l)` interactions;
  the `translog_terms` list is stored in `df.attrs` for one-line feed to
  `frontier()` / `xtfrontier()`. Toggleable squares and interactions.

### Migration

- Old: `frontier(df, y='y', x=['x1'])` still works (same required args).
- New keyword-only args: `usigma`, `vsigma`, `emean`, `te_method`,
  `start`.
- Existing efficiency scores should be recomputed — prior values were
  systematically biased by the Jondrow sign error.

---

### Multilevel / Mixed-Effects Overhaul

Release focus: `statspai.multilevel`. The previous implementation was a
400-line single file covering only the two-level linear mixed model
with a diagonal random-effect covariance. It has been rewritten as a
proper sub-package (~2,000 LOC across `_core.py`, `lmm.py`, `glmm.py`,
`diagnostics.py`, `comparison.py`) with feature parity against
`lme4`/Stata `mixed` and additions on top.

### New in `sp.mixed`

- **Unstructured covariance** `G` for random effects is now the
  default (`cov_type='unstructured'`, Cholesky-parameterised so the
  optimiser is unconstrained). `diagonal` and `identity` remain
  available for nested-model comparisons.
- **Three-level nested models** via `group=['school', 'class']` —
  fits school- and class-level random intercepts jointly (verified to
  match `statsmodels.MixedLM(..., re_formula="1", vc_formula={...})`
  to four decimals on the variance components and fixed effects).
- **BLUP posterior standard errors** (`result.ranef(conditional_se=
  True)`) — exposes
  `Var(u|y) = G − GZ'V⁻¹ZG + GZ'V⁻¹X Cov(β̂) X'V⁻¹ZG` for use in
  caterpillar plots.
- **`predict(new_data, include_random=…)`** — population-marginal and
  group-conditional predictions, with zeroed-out BLUPs for unseen
  groups.
- **Nakagawa-Schielzeth marginal & conditional R²** via
  `result.r_squared()`.
- **AIC / BIC, `wald_test()`** for linear restrictions,
  **`to_markdown()` / `to_latex()` / `_repr_html_()` / `cite()`**,
  and `plot(kind='caterpillar' | 'residuals')`.

### New functions

- **`sp.melogit` / `sp.mepoisson` / `sp.meglm`** — Generalised linear
  mixed models (binomial logit, Poisson log, Gaussian identity) fitted
  by Laplace approximation with canonical-link observed information.
  Supports random intercepts and random slopes, `cov_type` as for
  `sp.mixed`, binomial `trials=` and Poisson `offset=`. Results expose
  `odds_ratios()` / `incidence_rate_ratios()` and a `predict(type=
  'response'|'linear')` method.
- **`sp.icc(result)`** — intra-class correlation with a delta-method
  (logit-scale) 95% CI.
- **`sp.lrtest(restricted, full)`** — likelihood-ratio test between
  two nested mixed-model fits with automatic Self-Liang χ̄²
  boundary correction when variance components are being tested.

### Validation

- Linear mixed models: fixed effects and variance components agree
  with `statsmodels.MixedLM` to 4 decimal places on both random-
  intercept and unstructured random-slope specifications
  (`test_multilevel.py::TestRandomSlopeUnstructured::
  test_matches_statsmodels`).
- Three-level nested: variance components identified jointly and match
  the reference implementation to 2 decimal places
  (`TestThreeLevelNested::test_separates_variance_components`).
- GLMM recovery tests on 2,000-observation synthetic panels confirm
  slope and random-intercept variance within expected sampling ranges.

### Behavioural changes

- The default `cov_type` for `sp.mixed` is now `'unstructured'`
  (previously effectively diagonal). Pass `cov_type='diagonal'`
  explicitly for the old behaviour.
- `LR test vs. pooled OLS` now uses the ML-converted likelihood
  (previously a mix of REML and ML that could produce inconsistent
  values when `method='reml'`).

### Post-review hardening (post oracle + code-reviewer audit)

- **[BLOCKER fix]** `MixedResult.predict(data=None)` previously returned
  predictions in group-iteration order rather than the original row
  order. `_GroupBlock` now carries the training row indices and
  `predict()` scatters the output back to the correct positions.
  Regression test: `tests/test_multilevel.py::TestRandomIntercept::
  test_predict_is_row_aligned_with_training_frame`.
- **[BLOCKER fix]** GLMM inner Newton (`_find_mode`) now damps large
  steps and returns a convergence flag. `meglm` aggregates per-cluster
  failures and emits a `RuntimeWarning` when any cluster fails to
  converge — a previously silent failure mode.
- **[HIGH fix]** `MEGLMResult` gains `to_latex()` and `plot()` so it
  matches the unified StatsPAI result contract.
- **[HIGH fix]** `lrtest` now raises `ValueError` on cross-family
  comparisons and on REML fits whose fixed-effect design differs,
  preventing invalid LR statistics. Multi-component boundary
  corrections emit a `RuntimeWarning` explaining the conservative
  upper bound (Stram–Lee 1994 mixture not implemented).
- **[HIGH fix]** `mixed()` / `meglm()` reject non-hashable group
  values with a descriptive `TypeError` instead of producing a
  silently corrupted BLUP dict.
- **[MED fix]** `icc(result, n_boot>0)` raises `NotImplementedError`
  instead of silently returning the delta-method CI. `icc()` warns
  when `n_groups < 30` (delta-method CI unreliable).
- **[MED fix]** Three-level nested fit emits a warning when any
  outer group has only one inner group (class variance then not
  identified), and exposes both school and class ICCs via
  `variance_components['icc(outer)']` / `icc(outer+inner)`.

### GLMM hardening — AGHQ + Gamma / NegBin / Ordinal

Closes the three GLMM gaps flagged in the multilevel self-audit. All
changes are additive (no API breaks); existing `meglm` / `melogit` /
`mepoisson` calls produce numerically identical fits.

**Adaptive Gauss-Hermite quadrature (AGHQ) — `nAGQ` parameter.**
Previously `meglm` only offered the Laplace approximation (`nAGQ=1`),
which is known to underestimate random-effect variances on small
clusters with binary or other non-Gaussian outcomes. The new `nAGQ`
argument selects the number of adaptive quadrature points per scalar
random effect:

```python
sp.melogit(df, "y", ["x"], "g", nAGQ=7)   # matches Stata intpoints(7)
sp.megamma(df, "y", ["x"], "g", nAGQ=15)  # converged-grade quadrature
```

`nAGQ=1` reduces exactly to the Laplace formula (verified to 1e-10).
`nAGQ>1` is restricted to single-scalar random-effect models (no random
slopes), matching the same restriction `lme4::glmer` imposes — full
tensor-product AGHQ over `q>1` random effects is deferred because cost
scales as `nAGQ^q`. AGHQ is wired into all five families
(Gaussian / Binomial / Poisson / Gamma / NegBin) plus `meologit`.

**New families:**

- **`sp.megamma`** — Gamma GLMM with log link and dispersion `φ`
  estimated by ML, packed as `log φ` for unconstrained optimisation.
  IRLS weight uses Fisher information `1/φ` (Fisher scoring) for PSD
  Hessian regardless of fitted means.
- **`sp.menbreg`** — Negative-binomial NB-2 GLMM (`Var = μ + α μ²`)
  with log link, dispersion `α` (alias `family='negbin'` accepted).
  Reduces analytically to Poisson as `α → 0`; verified.
- **`sp.meologit`** — Random-effects ordinal logit (Stata `meologit`,
  R `ordinal::clmm`). K−1 thresholds reparameterised as
  `κ_1, log(κ_2−κ_1), ...` so strict ordering is enforced
  unconditionally. Returns `MEGLMResult` with new `thresholds`
  attribute. Supports `nAGQ>1`.

**Cross-family AIC comparability.** Poisson and Binomial log-
likelihoods now include the full normalisation constants (`-log(y!)`
for Poisson, log-binomial-coefficient for Binomial). Previously these
constants were dropped, which made `mepoisson` vs `menbreg` AIC
comparisons biased by ~Σ log(y!). β and variance estimates are
unchanged; only `log_likelihood` and `aic` / `bic` absolute values
shift — relative comparisons within a family are unaffected.

**Tests (multilevel).** `tests/test_multilevel.py` grows from 35 to 53
tests:

- `TestAGHQ` (7 tests) — nAGQ=1↔Laplace identity, AGHQ improves vs
  Laplace on small clusters, convergence in nAGQ, random-slope rejection.
- `TestMEGamma` (3) — truth recovery, dispersion accounting, summary.
- `TestMENegBin` (3) — truth recovery, IRR availability, alias resolution.
- `TestMEOLogit` (5) — truth recovery, threshold ordering, no intercept,
  summary, K≥3 enforcement.

Backwards compatibility: all 35 prior multilevel tests pass unchanged.

### Synth API-drift fixes (post-0.9.3-initial)

- **`SyntheticControl._solve_weights` signature migration** — three
  stale call sites in `synth/power.py` and `synth/sensitivity.py`
  migrated to the new (Y_treated_pre, Y_donors_pre, X_treated,
  X_donors, run_nested) signature (fixes 8 test failures in
  `tests/test_synth_advanced.py` and `tests/test_synth_extras.py`).
- **Placebo alignment** — `synth/power.py` placebo builder now follows
  `scm.py:888` exactly so LOO ↔ main placebo results stay consistent.
- **numpy 2.x compatibility** — `tests/test_frontier.py` switches
  `np.trapz` → `np.trapezoid` (removed in numpy 2.x).

---

### Econometric Trinity — P0 Pillars (DML-PLIV, Mixed Logit, IV-QR)

Three foundational econometric estimators identified as the highest-ROI gaps
vs. Stata, R, and existing Python packages are now first-class `sp.*` APIs
(~1,170 new LOC, 10 tests in `test_econ_trinity.py`).

- **`sp.dml(model='pliv', instrument=…)` — DML-PLIV (Partially Linear IV).**
  Chernozhukov et al. (2018, §4.2) Neyman-orthogonal score with cross-fitted
  nuisance functions `g(X)=E[Y|X]`, `m(X)=E[D|X]`, `r(X)=E[Z|X]`. Returns the
  LATE with influence-function-based standard errors. Closes the IV gap in
  the existing `DoubleML` (previously only PLM + IRM).
- **`sp.mixlogit` — Mixed Logit.** Random-coefficient multinomial logit via
  simulated maximum likelihood with Halton quasi-random draws. Supports:
  fixed + random coefficients, normal / log-normal / triangular mixing
  distributions, diagonal or full Cholesky covariance, panel (repeated-choice)
  data, OPG-sandwich robust SEs. Benchmarked against Stata `mixlogit` and R
  `mlogit`. Python's first feature-complete implementation.
- **`sp.ivqreg` — IV Quantile Regression.** Chernozhukov-Hansen (2005, 2006,
  2008) instrumental-variable quantile regression via inverse-QR profile.
  Scalar endogenous case uses grid + Brent refinement; multi-dim uses BFGS on
  the `b̂(α)` criterion. Multiple quantiles return a tidy DataFrame; single
  quantile returns `EconometricResults`. Optional pairs-bootstrap SEs.

All three reuse `_qreg_fit`, `CausalResult`, `EconometricResults` for API
consistency with the rest of StatsPAI.

#### Post-self-audit hardening

Self-audit + code-reviewer agent surfaced and fixed 4 BLOCKER + 7 HIGH bugs
in the first-cut implementation (see commit `2aa709b`). Parameter-recovery
tests now pass against controlled DGPs.

---

### Smart Workflow — Posterior Verification

Shipped in commit `be59260`:

- **`sp.verify`** / **`sp.verify_benchmark`** — posterior verification engine
  for `sp.recommend()` outputs. Runs bootstrap stability, placebo pass rate,
  and subsample agreement, aggregated into `verify_score ∈ [0, 100]`.
  Opt-in via `sp.recommend(verify=True)`; zero overhead when disabled.
- Calibration card shows top-method `verify_score` 85–95 on clean DGPs
  (RD lower at ≈ 74 due to local-polynomial bootstrap variance).
- 18/18 smart tests pass.

---

### Meta — Author Attribution

- Author metadata corrected from `Bryce Wang` to `Biaoyue Wang` in:
  `pyproject.toml` (`authors` + `maintainers`), `src/statspai/__init__.py`
  (`__author__`), `README.md` / `README_CN.md` (team line + BibTeX),
  `docs/index.md` (BibTeX), and `mkdocs.yml` (`site_author`).
  JOSS submission (`paper.md`) was already correct.

## [0.9.2] - 2026-04-16

### Decomposition Analysis — Most Comprehensive Decomposition Toolkit in Python

Release focus: `statspai.decomposition`. **18 first-class decomposition methods across 13 modules (~6,200 LOC, 54 tests)** — Python's first (and most complete) implementation of the full decomposition analysis toolkit spanning mean, distributional, inequality, demographic, and causal decomposition. Beats Stata `ddecompose` / `cdeco` / `oaxaca` / `rifhdreg` / `mvdcmp` / `fairlie` and R `Counterfactual` / `ddecompose` / `oaxaca` / `dineq` in scope; occupies the previously empty Python high-ground where only one unmaintained PyPI package existed.

#### What's in `sp.decompose` (18 methods, 30 aliases)

**Mean decomposition**

| Function | Method / Paper |
|---|---|
| `sp.oaxaca(df, ...)` | Blinder-Oaxaca threefold with 5 reference coefficients (Blinder 1973; Oaxaca 1973; Neumark 1988; Cotton 1988; Reimers 1983) |
| `sp.gelbach(df, ...)` | Sequential orthogonal decomposition of omitted-variable bias (Gelbach 2016, *JoLE*) |
| `sp.fairlie(df, ...)` | Nonlinear logit/probit decomposition (Fairlie 1999, 2005) |
| `sp.bauer_sinning(df, ...)` / `sp.yun_nonlinear(df, ...)` | Bauer-Sinning (2008) + Yun (2004, 2005) detailed nonlinear |

**Distributional decomposition**

| Function | Method / Paper |
|---|---|
| `sp.rifreg(df, ...)` / `sp.rif_decomposition(...)` | Recentered Influence Function regression + OB (Firpo, Fortin & Lemieux 2009, *Econometrica*) |
| `sp.ffl_decompose(df, ...)` | Two-step detailed decomposition (Firpo, Fortin & Lemieux 2018, *Econometrics*) |
| `sp.dfl_decompose(df, ...)` | Reweighting counterfactual distributions (DiNardo, Fortin & Lemieux 1996, *Econometrica*) |
| `sp.machado_mata(df, ...)` | Simulation-based quantile regression decomposition (Machado & Mata 2005, *JAE*) |
| `sp.melly_decompose(df, ...)` | Analytical quantile regression decomposition (Melly 2005, *Labour Economics*) |
| `sp.cfm_decompose(df, ...)` | Distribution regression counterfactuals (Chernozhukov, Fernández-Val & Melly 2013, *Econometrica*) |

**Inequality decomposition**

| Function | Method / Paper |
|---|---|
| `sp.subgroup_decompose(df, ...)` | Between/within for Theil T, Theil L, GE(α), Gini (Dagum 1997), Atkinson, CV² (Shorrocks 1984) |
| `sp.shapley_inequality(df, ...)` | Shorrocks-Shapley allocation of inequality to covariates (Shorrocks 2013, *JoEI*) |
| `sp.source_decompose(df, ...)` | Gini source decomposition (Lerman & Yitzhaki 1985, *ReStat*) |

**Demographic standardization**

| Function | Method / Paper |
|---|---|
| `sp.kitagawa_decompose(df, ...)` | Two-factor rate decomposition (Kitagawa 1955, *JASA*) |
| `sp.das_gupta(df_a, df_b, ...)` | Multi-factor symmetric decomposition (Das Gupta 1993) |

**Causal decomposition**

| Function | Method / Paper |
|---|---|
| `sp.gap_closing(df, method=...)` (regression / IPW / AIPW) | Gap-closing estimator (Lundberg 2021, *Sociol. Methods Res.*) |
| `sp.mediation_decompose(df, ...)` | Natural direct/indirect effects (VanderWeele 2014, *Epidemiology*) |
| `sp.disparity_decompose(df, ...)` | Causal disparity decomposition (Jackson & VanderWeele 2018, *Epidemiology*) |

**Unified entry point**

```python
import statspai as sp
result = sp.decompose(method='ffl', data=df, y='log_wage',
                      group='female', x=['education', 'experience'],
                      stat='quantile', tau=0.5)
result.summary(); result.plot(); result.to_latex()
```

30 aliases supported (`'mm'` → `machado_mata`, `'dinardo_fortin_lemieux'` → `dfl`, etc.).

#### Why this matters
- **Stata** has it scattered across 6+ packages (`oaxaca`, `ddecompose`, `cdeco`, `rifhdreg`, `mvdcmp`, `fairlie`) with no unified API.
- **R** has `ddecompose`, `Counterfactual`, `dineq` — three different authors, three different conventions.
- **Python** previously had only one 2018-vintage unmaintained PyPI package (basic Oaxaca).
- **StatsPAI 0.9.2**: one API, one result-class contract (`.summary()` / `.plot()` / `.to_latex()` / `._repr_html_()`), three inference modes (analytical / bootstrap / none), all numpy/scipy/pandas.

#### Quality bar
- 54 tests including cross-method consistency (`test_dfl_ffl_mean_agree`, `test_mm_melly_cfm_aligned_reference`, `test_dfl_mm_reference_convention_opposite`) and numerical identity checks (FFL four-part sum, weighted Gini RIF E_w[RIF]=G).
- Closed-form influence functions for Theil T / Theil L / Atkinson (no O(n²) numerical fallback).
- Weighted O(n log n) Dagum Gini via sorted-ECDF pairwise-MAD identity.
- Logit non-convergence surfaces as RuntimeWarning; bootstrap failure rate >5% warns.

## [0.9.1] - 2026-04-16

### Regression Discontinuity — Most Comprehensive RD Toolkit in Any Language

Release focus: `statspai.rd`. **18+ RD estimators, diagnostics, and inference methods across 14 modules (~10,300 LOC)** — now the most feature-complete RD package in Python, R, or Stata. The full machinery behind Calonico-Cattaneo-Titiunik (CCT), Cattaneo-Jansson-Ma density tests, Armstrong-Kolesar honest CIs, Cattaneo-Titiunik-Vazquez-Bare local randomization, Cattaneo-Titiunik-Yu boundary (2D) RD, and Angrist-Rokkanen external validity — all in one `import statspai as sp`.

#### What's in `sp.rd` (14 modules)

**Core estimation**

| Function | Method / Paper |
|---|---|
| `sp.rdrobust(df, ...)` | Sharp / Fuzzy / Kink RD with bias-corrected robust inference (Calonico, Cattaneo & Titiunik 2014, *Econometrica*; 2020, *Stata Journal*) |
| `sp.rdrobust(..., covs=...)` | Covariate-adjusted local polynomial (Calonico, Cattaneo, Farrell & Titiunik 2019, *ReStat*) |
| `sp.rd2d(df, x1, x2, ...)` | Boundary discontinuity / 2D RD designs (Cattaneo, Titiunik & Yu 2025) |
| `sp.rkd(df, ...)` | Regression Kink Design (Card, Lee, Pei & Weber 2015, *Econometrica*) |
| `sp.rdit(df, time, ...)` | Regression Discontinuity in Time (Hausman & Rapson 2018, *Annual Review*) |
| `sp.rdmc(df, cutoffs=[...])` | Multi-cutoff RD (Cattaneo, Titiunik, Vazquez-Bare & Keele 2016) |
| `sp.rdms(df, scores=[...])` | Multi-score RD (Cattaneo, Idrobo & Titiunik 2024) |

**Bandwidth selection**

| Function | Selector |
|---|---|
| `sp.rdbwselect(df, bwselect='mserd')` | MSE-optimal (Imbens-Kalyanaraman 2012) |
| `sp.rdbwselect(..., bwselect='msetwo')` | Two-bandwidth MSE |
| `sp.rdbwselect(..., bwselect='cerrd'/'cercomb1'/'cercomb2')` | CER-optimal coverage-error-rate (Calonico, Cattaneo & Farrell 2020, *Econometrics Journal*) |

**Inference**

| Function | Method |
|---|---|
| `sp.rd_honest(df, ...)` | Honest CIs with worst-case bias bound (Armstrong & Kolesar 2018, *Econometrica*; 2020, *QE*) |
| `sp.rdrandinf(df, ...)` | Local randomization inference via Fisher exact tests (Cattaneo, Frandsen & Titiunik 2015) |
| `sp.rdwinselect(df, ...)` | Data-driven window selection for local randomization |
| `sp.rdsensitivity(df, ...)` | Sensitivity analysis across windows |
| `sp.rdrbounds(df, ...)` | Rosenbaum sensitivity bounds for hidden selection |

**Heterogeneous treatment effects**

| Function | Method |
|---|---|
| `sp.rdhte(df, covs=[...])` | CATE via fully interacted local linear (Calonico et al. 2025) |
| `sp.rdbwhte(df, ...)` | HTE-optimal bandwidth |
| `sp.rd_forest(df, ...)` | Causal forest + RD |
| `sp.rd_boost(df, ...)` | Gradient boosting + RD |
| `sp.rd_lasso(df, ...)` | LASSO-assisted RD with covariate selection |

**External validity & extrapolation**

| Function | Method |
|---|---|
| `sp.rd_extrapolate(df, ...)` | Away-from-cutoff extrapolation (Angrist & Rokkanen 2015, *JASA*) |
| `sp.rd_multi_extrapolate(df, cutoffs=[...])` | Multi-cutoff extrapolation (Cattaneo, Keele, Titiunik & Vazquez-Bare 2024) |

**Diagnostics & visualization**

| Function | Purpose |
|---|---|
| `sp.rdsummary(df, ...)` | **One-click dashboard** — rdrobust + density test + bandwidth sensitivity + placebo cutoffs + covariate balance |
| `sp.rdplot(df, ...)` | IMSE-optimal binned scatter with pointwise CI bands (Calonico, Cattaneo & Titiunik 2015, *JASA*) |
| `sp.rddensity(df, ...)` | Cattaneo-Jansson-Ma (2020, *JASA*) manipulation test |
| `sp.rdbalance(df, covs=[...])` | Covariate balance tests at cutoff |
| `sp.rdplacebo(df, cutoffs=[...])` | Placebo cutoff tests |

**Power analysis**

| Function | Purpose |
|---|---|
| `sp.rdpower(df, effect_sizes=[...])` | Power curves for RD designs |
| `sp.rdsampsi(df, target_power=0.8)` | Required sample size |

#### Refactor — rd/\_core.py consolidation

A 5-sprint refactor (commit 44f7529) centralized shared low-level primitives that had been duplicated across 9 RD files into a single private module `rd/_core.py` (191 lines):

- `_kernel_fn` — triangular / epanechnikov / uniform / gaussian (previously 4 duplicate definitions)
- `_kernel_constants` / `_kernel_mse_constant` — MSE-optimal bandwidth constants
- `_local_poly_wls` — WLS local polynomial fit with HC1 / cluster-robust variance + optional covariate augmentation
- `_sandwich_variance` — HC1 / cluster sandwich for arbitrary design matrices

**Net effect**: 253 lines of duplicated math consolidated into 191 lines of canonical implementation. 97 RD tests pass with zero regression.

#### Bug fixes (since 0.9.0)

- RDD extrapolation: `_ols_fit` singular matrix fallback (commit 052594a)
- 3 critical + 3 high-priority bugs from comprehensive RD code review (commit 6489270)
- Density test: bug in CJM (2020) implementation + DGP helper fixes + validation tests (commit b66f312)

#### Tests

- **97 RD tests + 1 skipped, 0 failed** across 5 test files.

### Also in 0.9.1

- **`synth/_core.py`** — simplex weight solver consolidated from 6 duplicate implementations (commit a4036a2). Analytic Jacobian now available to all six callers for ~3-5x speedup.
- **`decomposition/_common.py`** — new `influence_function(y, stat, tau, w)` is the canonical 9-stat RIF kernel. `rif.rif_values` public API **expands from 3 to 9 statistics** (commits 0789223, 5569fd0).

---

## [0.9.0] - 2026-04-16

### Synthetic Control — Most Comprehensive SCM Toolkit in Any Language

Release focus: `statspai.synth`. **20 SCM methods + 6 inference strategies + full research workflow (compare / power / sensitivity / one-click reports)**, all behind the unified `sp.synth(method=...)` dispatcher. No competing package in Python, R, or Stata offers this breadth.

#### Seven new SCM estimators

| Method | Reference |
|---|---|
| `bayesian_synth` | Dirichlet-prior MCMC with full posterior credible intervals (Vives & Martinez 2024) |
| `bsts_synth` / `causal_impact` | Bayesian Structural Time Series via Kalman filter/smoother (Brodersen et al. 2015) |
| `penalized_synth` (penscm) | Pairwise discrepancy penalty (Abadie & L'Hour 2021, *JASA*) |
| `fdid` | Forward DID with optimal donor subset selection (Li 2024) |
| `cluster_synth` | K-means / spectral / hierarchical donor clustering (Rho 2024) |
| `sparse_synth` | L1 / constrained-LASSO / joint V+W (Amjad, Shah & Shen 2018, *JMLR*) |
| `kernel_synth` + `kernel_ridge_synth` | RKHS / MMD-based nonlinear matching |

Previous methods — classic, penalized, demeaned, unconstrained, augmented, SDID, gsynth, staggered, MC, discos, multi-outcome, scpi — remain with bug fixes (see below).

#### Research workflow

- `synth_compare(df, ...)` — run every method at once, tabular + graphical comparison
- `synth_recommend(df, ...)` — auto-select best estimator by pre-fit + robustness
- `synth_report(result, format='markdown'|'latex'|'text')` — one-click publication-ready report
- `synth_power(df, effect_sizes=[...])` — first power-analysis tool for SCM designs
- `synth_mde(df, target_power=0.8)` — minimum detectable effect
- `synth_sensitivity(result)` — LOO + time placebos + donor sensitivity + RMSPE filtering
- Three canonical datasets shipped: `california_tobacco()`, `german_reunification()`, `basque_terrorism()`

#### Release-blocker fixes from comprehensive module review

Following a 5-parallel-agent code review (correctness / numerics / API / perf / docs), nine release blockers were fixed:

- **ASCM correction formula** — `augsynth` now follows Ben-Michael, Feller & Rothstein (2021) Eq. 3 per-period ridge bias `(Y1_pre − Y0'γ) @ β(T0, T1)`, replacing the scalar mean-residual placeholder. `_ridge_fit` RHS bug also fixed.
- **Bayesian likelihood scale** — covariate rows are now z-scored to the pooled pre-outcome SD before concatenation, preventing scale mismatch from dominating the Gaussian `σ²` posterior.
- **Bayesian MCMC Jacobian** — missing `log(σ′/σ)` correction for the log-normal random-walk proposal on σ has been added to the MH acceptance ratio.
- **BSTS Kalman filter** — innovation variance floored at `1e-12` (prevents `log(0)` on constant outcome series); RTS smoother `inv → solve + pinv` fallback on near-singular predicted covariance.
- **gsynth factor estimation** — four `np.linalg.inv` calls (loadings + placebo loop) replaced with `np.linalg.lstsq` (robust to rank-deficient `F'F` / `L'L`).
- **Dispatcher `**kwargs` leakage** — `augsynth` gains `**kwargs + placebo=True`; `sp.synth(method='augmented', placebo=False)` no longer raises `TypeError`.
- **Dispatcher `kernel_ridge` placebo bypass** — `placebo=` now forwarded correctly.
- **Cross-method API consistency** — `sdid()` now accepts canonical `outcome / treated_unit / treatment_time` (legacy `y / treat_unit / treat_time` aliases retained for backwards compatibility).
- **Documentation accuracy** — `synth_compare` docstring reflects 20 methods (was 12); `synth()` Returns section enumerates all `CausalResult` fields.

#### Tests & validation

- **144 synth tests passing** (new: 12-method cross-method consistency benchmark verifying every estimator recovers a known ATT within 1.5 units on a clean DGP).
- **Full suite: 1481 passed, 4 skipped, 0 failed** (5m42s).
- New guide: `docs/guides/synth.md` — complete tutorial covering all 20 methods with a method-choice decision table.

#### API migration notes

`sdid(y=, treat_unit=, treat_time=)` still works but `outcome=, treated_unit=, treatment_time=` is preferred for consistency with every other `sp.synth.*` function. A deprecation of the legacy names is planned for v1.0.

### Other Modules

Decomposition and Regression Discontinuity modules received significant upgrades in this release cycle (tier-C decomposition expansion to 18 methods + unified `sp.decompose()`; RD `_core.py` primitive centralization + bug fixes from code review). These will be highlighted in a dedicated follow-up release note.

---

## [0.8.0] - 2026-04-16

### Spatial Econometrics Full-Stack + 10-Domain Breadth Upgrade

**Largest release in StatsPAI history. 60+ new functions across 10 domains.**

#### Spatial Econometrics (NEW — 38 API symbols)

From 3 functions / 419 LOC to **38 functions / 3,178 LOC / 69 tests**. Python's first unified spatial econometrics package.

- **Weights (L1)**: `W` (sparse CSR), `queen_weights`, `rook_weights`, `knn_weights`, `distance_band`, `kernel_weights`, `block_weights`
- **ESDA (L2)**: `moran` (global + local), `geary`, `getis_ord_g`, `getis_ord_local`, `join_counts`, `moran_plot`, `lisa_cluster_map`
- **ML Regression (L3)**: `sar`, `sem`, `sdm`, `slx`, `sac` — sparse-backed, dual log-det path (exact + Barry-Pace), scales to N=100K
- **GMM (L3)**: `sar_gmm`, `sem_gmm`, `sarar_gmm` — Kelejian-Prucha (1998/1999), heteroskedasticity-robust
- **Diagnostics**: `lm_tests` (Anselin 1988 full battery), `moran_residuals`
- **Effects**: `impacts` (LeSage-Pace 2009 direct/indirect/total + simulated SE)
- **GWR (L4)**: `gwr`, `mgwr` (Multiscale GWR), `gwr_bandwidth` (AICc/CV golden-section)
- **Spatial Panel (L5)**: `spatial_panel` (SAR-FE / SEM-FE / SDM-FE, entity + twoways)
- **Cross-validated**: Columbus rtol<1e-7 vs PySAL spreg 1.9.0; Georgia GWR bit-identical vs mgwr 2.2.1; GMM rtol<1e-4 vs spreg GM_*

#### Time Series

- `local_projections` — Jordà (2005) IRF with Newey-West HAC
- `garch` — GARCH(p,q) MLE with multi-step forecast
- `arima` — ARIMA/SARIMAX with auto (p,d,q) AICc grid search
- `bvar` — Bayesian VAR with Minnesota (Litterman) prior

#### Causal Discovery

- `lingam` — DirectLiNGAM (Shimizu 2011), bit-identical vs lingam package
- `ges` — Greedy Equivalence Search (Chickering 2002)

#### Matching

- `optimal_match` — Hungarian 1:1 matching (min total Mahalanobis distance)
- `cardinality_match` — Zubizarreta (2014) LP-based matching with balance constraints

#### Decomposition & Mediation

- `rifreg` — RIF regression (Firpo-Fortin-Lemieux 2009)
- `rif_decomposition` — RIF Oaxaca-Blinder for distributional statistics
- `mediate_sensitivity` — Imai-Keele-Yamamoto (2010) ρ-sensitivity

#### RD & Survey

- `rdpower`, `rdsampsi` — power/sample-size for RD designs
- `rake`, `linear_calibration` — survey calibration (Deville-Särndal 1992)

#### Survival

- `cox_frailty` — Cox with shared gamma frailty (Therneau-Grambsch)
- `aft` — Accelerated Failure Time (exponential/Weibull/lognormal/loglogistic)

#### ML-Causal (GRF)

- `CausalForest.variable_importance()`, `.best_linear_projection()`, `.ate()`, `.att()`
- **Bugfix**: honest leaf values now correctly vary per-leaf

#### Infrastructure

- OLS/IV `predict(data, what='confidence'|'prediction')` with intervals
- Pre-release code review: 3 critical + 2 high-priority bugs fixed

## [0.7.1] - 2026-04-15

DID-focused polish release. Brings the Wooldridge (2021) ETWFE
implementation to full feature parity with the R `etwfe` package,
adds a one-call method-robustness workflow, and closes 12 issues
uncovered by an internal code review round. All 27 new / updated
DID tests pass (`pytest tests/test_did_summary.py`).

### Added — ETWFE full parity with R `etwfe`

- **`sp.etwfe()` explicit API** aligned with R `etwfe` (McDermott 2023)
  naming. Thin alias over `wooldridge_did()` with a full argument-
  mapping table in the docstring.
- **`xvar=` covariate heterogeneity** (single string or list of names).
  Adds per-cohort × post × `(x_j − mean(x_j))` interactions; `detail`
  gains `slope_<x>` / `slope_<x>_se` / `slope_<x>_pvalue` columns.
  Baseline ATT is reported at the sample means of every covariate.
- **`panel=False` repeated cross-section mode** — replaces unit FE
  with cohort + time dummies (R `etwfe(ivar=NULL)` equivalent).
- **`cgroup='nevertreated'`** — per-cohort regressions restricted to
  (cohort g) ∪ (never-treated); cohort-size-weighted aggregation
  (R `etwfe(cgroup='never')` equivalent). Default `'notyet'` preserves
  prior ETWFE behaviour.
- **`sp.etwfe_emfx(result, type=…)`** — R `etwfe::emfx`-equivalent
  four aggregations: `'simple'`, `'group'`, `'event'`, `'calendar'`.
  `include_leads=True` returns full event-time output including pre-
  treatment leads for pre-trend inspection (`rel_time = -1` is the
  reference category).

### Added — one-call DID method-robustness workflow

- **`sp.did_summary()`** — fits five modern staggered-DID estimators
  (CS, SA, BJS, ETWFE, Stacked) to the same data and returns a tidy
  comparison table with per-method (estimate, SE, p, 95 % CI). Mean
  across methods + cross-method SD flag method-sensitivity of results.
- **`include_sensitivity=True`** — attaches the Rambachan-Roth (2023)
  breakdown `M*` to the CS row, giving a three-way robustness readout
  in a single call.
- **`sp.did_summary_plot()`** — forest plot of per-method estimates
  with cross-method mean line; `sort_by='estimate'` supported.
- **`sp.did_summary_to_markdown()` / `_to_latex()`** — publication-
  ready exports (GFM tables / booktabs LaTeX with auto-escaped
  ampersands).
- **`sp.did_report(save_to=dir)`** — one-call bundle that writes
  `did_summary.txt` / `.md` / `.tex` / `.png` / `.json` to a folder.

### Fixed — 12 issues from the internal code review

Blockers (C-severity):

- `etwfe(xvar=…)` now raises a clear `ValueError` when the covariate
  is all-NaN or (near-)constant. Previously returned `n_obs = 0,
  estimate = 0` silently.
- `etwfe(panel=False, cgroup='nevertreated')` now raises a crisp
  `NotImplementedError` instead of silently falling back to
  `'notyet'`.
- `did_summary` now validates column names up front (raises
  `KeyError` listing missing columns) and only catches narrow
  estimator-side exceptions inside the fit loop; user typos in
  `controls=` / `cluster=` surface as proper errors.
- `did_summary` results round-trip cleanly through stdlib
  serialisation (`DIDSummaryResult(CausalResult)` subclass with a
  real `.summary()` method, replacing the prior closure-bound
  instance attribute).

High-severity:

- `etwfe_emfx(type='event'/'calendar')` now computes SEs via the
  delta method on the stored event-study vcov instead of the
  independent-coefficient approximation. `model_info['se_method']`
  advertises which path was used.
- `etwfe_emfx(type='group')` headline `se` / `pvalue` / `ci` are now
  populated (match the underlying fit's overall ATT exactly).
- Validation for `did_summary_plot` / `_to_markdown` / `_to_latex`
  aligned on a single sentinel `model_info['_did_summary_marker']`.
- `_etwfe_never_only` no longer leaves a `_ft_cache` helper column
  on the caller's DataFrame.
- Slope indexing in `_etwfe_with_xvar` is now name-keyed
  (`coef_index` dict); regression test verifies swapping
  `xvar=['x1','x2']` vs `['x2','x1']` produces identical slopes per
  name.
- `etwfe(panel=False)` with rank-deficient designs emits a
  `RuntimeWarning` pointing at concrete remedies (previously fell
  through to `pinv` silently).

### Tests

- New test module `tests/test_did_summary.py` — 27 cases covering
  consistency with direct estimator calls, export formats, forest
  plot rendering, `etwfe_emfx` round-trips, xvar / panel / cgroup
  options, the 12 review fixes, and the `include_leads` mode.

## [0.7.0] - 2026-04-14

Focused release reaching feature parity with the R `did` / `HonestDiD`
packages and the Python `csdid` / `differences` packages for staggered
Difference-in-Differences.  All core algorithms are reimplemented from
the original papers — **no wrappers, no runtime dependencies on upstream
DID packages**.  Full DiD test suite: 47 → 170+ (including three rounds
of post-implementation audit that surfaced and fixed 9 bugs before
release).

### Added — Core estimation

- **`sp.aggte(result, type=...)`** — unified aggregation layer for
  `callaway_santanna()` results.  Four aggregation schemes (`simple`,
  `dynamic`, `group`, `calendar`) backed by a single weighted-
  influence-function engine.  Callaway & Sant'Anna (2021) Section 4.
- **Mammen (1993) multiplier bootstrap** — IQR-rescaled pointwise
  standard errors *and* simultaneous (uniform / sup-t) confidence
  bands over the aggregation dimension.  Matches the uniform-band
  behaviour of the R `did::aggte` function.
- **`balance_e` / `min_e` / `max_e`** — event-study cohort balancing
  and window truncation (CS2021 eq. 3.8).
- **`anticipation=δ`** parameter on `callaway_santanna()` — shifts
  the base period back by δ periods per CS2021 §3.2.
- **Repeated cross-sections** support via `callaway_santanna(panel=False)`
  — unconditional 2×2 cell-mean DID with observation-level influence
  functions (CS2021 eq. 2.4, RCS version).  Optional covariate
  residualisation with `x=[...]` for regression adjustment.  All
  downstream modules (`aggte`, `cs_report`, `ggdid`, `honest_did`)
  work on RCS results with no code changes.
- **dCDH joint inference** (`did_multiplegt`) — `joint_placebo_test`
  (Wald χ² across placebo lags with bootstrap covariance, dCDH 2024
  §3.3) and `avg_cumulative_effect` (mean of dynamic[0..L] with
  SE preserving cross-horizon covariance, dCDH 2024 §3.4).
- **`sp.bjs_pretrend_joint()`** — cluster-bootstrap joint Wald pre-
  trend test for BJS imputation results.  Upgrades the default
  sum-of-z² test (which assumes pre-period independence) to a full
  covariance-aware statistic.

### Added — Reporting & visualisation

- **`sp.cs_report(data, ...)`** — one-call report card.  Runs the
  full pipeline (ATT(g,t) → four aggregations with uniform bands →
  pre-trend Wald → Rambachan–Roth breakdown M\* for every post event
  time) under a single bootstrap seed and pretty-prints the result.
  Returns a structured `CSReport` dataclass.
- **`sp.ggdid(result)`** — plot routine for `aggte()` output,
  mirroring R `did::ggdid`.  Auto-dispatches on aggregation type;
  uniform band overlaid on pointwise CI.
- **`CSReport.plot()`** — one-call 2×2 summary figure: event study
  with uniform band (top-left), θ(g) per-cohort (top-right), θ(t)
  per-calendar-time (bottom-left), Rambachan–Roth breakdown M\*
  bars (bottom-right).
- **`CSReport.to_markdown()`** — GitHub-flavoured Markdown export
  with proper integer-column rendering and a configurable
  `float_format`.
- **`CSReport.to_latex()`** — publication-ready booktabs fragment
  wrapped in a `table` float.  Zero `jinja2` dependency (hand-rolled
  booktabs renderer); auto-escapes LaTeX special characters.
- **`CSReport.to_excel()`** — six-sheet workbook (`Summary`,
  `Dynamic`, `Group`, `Calendar`, `Breakdown`, `Meta`).  Engine
  autoselect (openpyxl → xlsxwriter) with a clear ImportError when
  neither is installed.
- **`cs_report(..., save_to='prefix')`** — one-call dump of the
  full export matrix: writes `<prefix>.{txt,md,tex,xlsx,png}` in
  a single invocation, auto-creating missing parent directories.
  Optional dependencies (openpyxl, matplotlib) are skipped silently
  so a minimal install still produces text + md + tex.
- **`sp.did(..., aggregation='dynamic', n_boot=..., random_state=...)`**
  — the top-level dispatcher now forwards CS-style arguments
  (`aggregation`, `panel`, `anticipation`) and can pipe a CS result
  straight through `aggte()` in a single call.

### Changed

- **`sun_abraham()` inference layer rewritten** — replaces the
  former ad-hoc `√(σ²/(total·T))` approximation with a Liang–Zeger
  cluster-robust sandwich `(X'X)⁻¹ Σ_c X_c' u_c u_c' X_c (X'X)⁻¹`
  (small-sample adjusted), delta-method IW aggregation SEs
  `w' V_β w`, iterative two-way within transformation (correct on
  unbalanced panels), and optional `control_group='lastcohort'` per
  SA 2021 §6.
- **`sp.honest_did()` / `sp.breakdown_m()` made polymorphic** — now
  accept the legacy `callaway_santanna()` / `sun_abraham()` format
  (event study in `model_info`) *and* the new `aggte(type='dynamic')`
  format (event study in `detail` with Mammen uniform bands).  The
  idiomatic pipeline `cs → aggte → honest_did → breakdown_m` now
  runs end-to-end with no manual plumbing.
- **README DiD parity matrix** added, comparing StatsPAI against
  `csdid`, `differences`, and R `did` + `HonestDiD` across 15
  capabilities.

### Fixed (from pre-release audit rounds)

- **Critical — `aggte(type='dynamic').estimate`** previously averaged
  pre- *and* post-treatment event times into the overall ATT,
  polluting the headline number with placebo signal.  Now averages
  only e ≥ 0, matching R `did::aggte`'s print convention.  On a
  typical DGP the bug shifted the reported overall by nearly a
  factor of 2.
- **LaTeX escape non-idempotence** in `CSReport.to_latex()`:
  `\` → `\textbackslash{}` followed by `{` → `\{` mangled the
  just-inserted braces.  Fixed with a single-pass `re.sub`.
- **`cs_report(save_to='~/study/…')`** did not expand `~`; fixed
  via `os.path.expanduser`.
- **`cs_report(sa_result)` / `aggte(sa_result)`** raised cryptic
  `KeyError: 'group'`; both entry points now detect non-CS input
  up-front and raise a clear `ValueError`.
- **`cs_report(pre_fitted_cs, estimator=…)`** silently ignored the
  override; now emits a `UserWarning` listing every shadowed arg.
- **`sp.did(method='2x2', aggregation='dynamic')`** silently ignored
  CS-only arguments; now raises an informative `ValueError`.
- **`bjs_pretrend_joint`** swallowed all exceptions as "bootstrap
  failed"; now narrows to expected failure modes and re-raises
  unexpected errors with context.
- **`matplotlib.use('Agg')`** in `_save_report_bundle` no longer
  switches the backend unconditionally (respects Jupyter sessions).

### References

- Callaway, B. and Sant'Anna, P.H.C. (2021). *J. of Econometrics* 225(2).
- Sun, L. and Abraham, S. (2021). *J. of Econometrics* 225(2).
- Mammen, E. (1993). *Ann. Statist.* 21(1).
- Liang, K.-Y. and Zeger, S.L. (1986). *Biometrika* 73(1).
- de Chaisemartin, C. and D'Haultfoeuille, X. (2020). *AER* 110(9).
- de Chaisemartin, C. and D'Haultfoeuille, X. (2024). *RESt*, forthcoming.
- Rambachan, A. and Roth, J. (2023). *Rev. Econ. Studies* 90(5).
- Borusyak, K., Jaravel, X. and Spiess, J. (2024). *ReStud* 91(6).

## [0.6.2] - 2026-04-12

### Added

- **OLS `predict()`**: `result.predict(newdata=)` for out-of-sample prediction on OLS results
- **`balance_panel()`**: Utility to keep only units observed in every time period (`sp.balance_panel()`)
- **Panel `balance=True`**: Convenience flag in `sp.panel()` to auto-balance before estimation
- **Analytical weights for DID**: `weights=` parameter added to `did()`, `ddd()`, and `event_study()` for population-weighted estimation (Stata `[aweight=...]` equivalent)
- **Matching `ps_poly=`**: Polynomial propensity score specification (`ps_poly=2` adds interactions/squares, following Cunningham 2021 Ch. 5)
- **Synth `rmspe` plot**: Post/pre RMSPE ratio histogram (`synthplot(result, type='rmspe')`) per Abadie et al. (2010)
- **Synth placebo gap plot**: Full spaghetti placebo gap paths with `rmspe_threshold` filter (Abadie et al. 2010, Figure 4)
- **Graddy (2006) replication**: Fulton Fish Market IV example added to `sp.replicate()` (Mixtape Ch. 7)
- **Numerical validation tests**: Cross-validated against Stata/R reference values with humanized error messages

### Fixed

- **`outreg2` format auto-detection**: Correctly infers `.xlsx`/`.csv`/`.tex` from filename extension
- **Synth placebo p-value**: Now uses RMSPE *ratio* (√post/√pre) instead of squared ratio, matching Abadie et al. (2010) convention

### Improved

- **DID/DDD/Event Study**: Weights propagation through WLS with proper normalization and validation
- **Synth placebos**: Store full placebo gap trajectories, per-unit RMSPE ratios, and unit labels for richer post-estimation analysis
- **Matching tests**: Added comprehensive test suite for PSM, Mahalanobis, CEM, and stratification methods

## [0.6.1] - 2026-04-07

### Fixed

- **Interactive Editor — Theme switching**: Themes now fully reset before applying, so switching between themes (e.g. ggplot → academic) correctly updates all visual properties instead of leaking stale settings
- **Interactive Editor — Apply button**: Fixed Apply button being clipped/hidden on the Layout tab due to panel overflow
- **Interactive Editor — Panel layout**: Fixed panel content disappearing when using flex layout for bottom-pinned Apply button
- **Interactive Editor — Style tab**: Fixed Style tab stuck on "Loading" after Theme tab was reordered to first position
- **Interactive Editor — Error visibility**: Widget callback errors now surface in the status bar instead of being silently swallowed

### Improved

- **Interactive Editor — Auto mode**: Clicking Auto now always refreshes the preview, giving immediate visual feedback
- **Interactive Editor — Auto/Manual toggle**: Compact toggle button moved to panel header with sticky positioning
- **Interactive Editor — Apply button**: Separated from Auto toggle and placed at panel bottom-right for better UX
- **Interactive Editor — Theme tab**: Moved to first position for better discoverability
- **Interactive Editor — Color pickers**: Added visual confirmation feedback on all color changes
- **Interactive Editor — Code generation**: Auto-generate reproducible code with text selection support in the editor
- **Smart recommendations**: Enhanced compare and recommend logic
- **Registry**: Expanded module support in the registry

## [0.1.0] - 2024-07-26

### Added
- **Core Regression Framework**
  - OLS (Ordinary Least Squares) regression with formula interface
  - Robust standard errors (HC0, HC1, HC2, HC3)
  - Clustered standard errors
  - Weighted Least Squares (WLS) support

- **Causal Inference Module**
  - Causal Forest implementation inspired by Wager & Athey (2018)
  - Honest estimation for unbiased treatment effect estimation
  - Bootstrap confidence intervals for treatment effects
  - Formula interface: `"outcome ~ treatment | features | controls"`

- **Output Management (outreg2)**
  - Excel export functionality similar to Stata's outreg2
  - Support for multiple regression models in single output
  - Customizable formatting options
  - Professional table layout

- **Unified API Design**
  - Consistent `reg()` function interface
  - Formula parsing: R/Stata-style syntax `"y ~ x1 + x2"`
  - Type hints throughout the codebase
  - Comprehensive documentation

### Technical Details
- Python 3.8+ support
- Dependencies: numpy, scipy, pandas, scikit-learn, openpyxl
- MIT License
- Comprehensive test suite
