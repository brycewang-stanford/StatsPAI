# Changelog

All notable changes to StatsPAI will be documented in this file.

## [Unreleased]

### ⚠️ Correctness

These change DiD point estimates for affected staggered/switching designs. See
`MIGRATION.md` before comparing new output to earlier StatsPAI runs.

- **`sp.callaway_santanna(control_group="nevertreated")` now fails loudly when
  the panel has no never-treated units** instead of silently returning
  `ATT = 0.0`. With every unit eventually treated, each `ATT(g,t)` lost its
  comparison cell and returned `0.0`, which aggregated to a headline ATT of
  `0.0` — a wrong number that reads as "no effect" rather than an error (no
  warning was emitted). The estimator now raises `MethodIncompatibility` with a
  hint to use `control_group="notyettreated"` or add never-treated units. The
  internal group encoding treats `NaN`/`inf` `g` as never-treated (0), so a
  panel with any never-treated (including `NaN`-coded) unit is unaffected, and
  `control_group="notyettreated"` is unaffected. No previously-valid estimate
  moves — only the silent `0.0` degenerate path changes. New guard:
  `tests/tier_eg/test_did_robustness.py::test_cs_no_never_treated_control_documented`.
- **`sp.eigenvector_centrality` on bipartite graphs** now returns the true
  leading eigenvector instead of a spurious near-uniform vector. The score was
  computed by naive power iteration `x <- A x`, which *oscillates* on a
  bipartite graph (its spectrum is symmetric, `lambda_max = -lambda_min`) and
  never converges — a star returned ~`1/sqrt(n)` for every node instead of a
  dominant hub. The leading eigenvector is now recovered by direct
  eigendecomposition (`eigh` for undirected, `eig` for directed), so a star
  hub scores `1/sqrt(2)` and its leaves `1/sqrt(8)`, matching the
  igraph / networkx convention. New guard in
  `tests/reference_parity/test_network_centrality_parity.py`.

- **`sp.ges` (Greedy Equivalence Search)** no longer adds a spurious edge
  between the parents of a collider. The greedy edge search had no acyclicity
  constraint, so on a v-structure `X -> Z <- Y` it would add edges into *and*
  out of `Z`; conditioning on the collider then made the independent parents
  `X`, `Y` look dependent and a false `X -- Y` edge was added (observed on every
  seed of a strong-signal collider). The search now rejects edges that create a
  directed cycle, and the recovered DAG is reported as its CPDAG (v-structures
  stay directed, reversible edges render undirected). Colliders now recover the
  correct `X -> Z <- Y`; chains recover the undirected skeleton. New guard:
  `tests/reference_parity/test_ges_parity.py`.

- **`sp.dist_iv` / `sp.kan_dlate` with a binary instrument** no longer return a
  silent all-`NaN` `late_q`. The Wald quantile-LATE split used a strict
  `Z > median(Z)` rule, which leaves the high group empty whenever the median
  sits on the top mass point of a discrete instrument (a binary `Z` with more
  1s than 0s has `median == 1`, so `Z > 1` is empty) — roughly half of ordinary
  data draws returned NaN at every quantile with no error. The split now falls
  back to `Z >= median` so both instrument groups are non-empty, giving up (a
  single NaN) only when the instrument is genuinely constant. Estimates on the
  previously-working draws (including the documented `seed=42` example) are
  unchanged. New guard: `tests/reference_parity/test_dist_iv_parity.py`.
- **`sp.contrast` / `sp.pwcompare` with `C(var)` categoricals** now return the
  correct treatment contrasts instead of all zeros. The predictive-margin
  engine matched design-matrix terms by raw column name, so a model fit with a
  formula-encoded factor (terms like `C(g)[T.1]`) never fired the dummies when
  a level was set — every contrast, SE, and p-value came back `0`. The margin
  builder now parses `C(var)[T.level]` (and string levels) so the reference /
  adjacent / pairwise contrasts equal the corresponding dummy coefficients
  exactly (verified to ≤1e-15). Models specified with numeric-coded factors
  were already correct and are unchanged. New parity guard:
  `tests/reference_parity/test_contrast_pwcompare_parity.py`.
- **`sp.did_multiplegt` DID_M / dynamic / placebo** now conditions
  switcher-vs-stayer cells on the baseline treatment `d_{t-1}`, matching the
  de Chaisemartin–D'Haultfoeuille estimator and Stata
  `did_multiplegt (old)`. Static DID_M no longer pools already-treated stayers
  with untreated stayers, switch-off cells enter with the treatment-gain sign,
  dynamic effects use robust stayers that keep the baseline treatment through
  the full horizon, and placebo effects use the Stata mirror sign convention.
  New parity guards cover both Stata-pinned static panels and hand-checkable
  dynamic/placebo regressions.
- **`sp.sar` / `sp.sdm` coefficient standard errors** now come from the full
  `(β, ρ, σ²)` information matrix (the leading block of its inverse), matching
  `spatialreg::lagsarlm`. The previous concentrated `σ²(XᵀX)⁻¹` treated `ρ` as
  known and understated the coefficient SEs — most visibly ~2× too small on the
  intercept for a row-standardised `W`. Point estimates and the `ρ`/`λ` SEs are
  substantively unchanged; the bounded `ρ`/`λ` ML optimiser was also tightened
  (`xatol=1e-10`), shifting estimates by ≲1e-5 toward the exact MLE. `sp.sem`
  and `sp.slx` standard errors are unaffected. See `MIGRATION.md`.
- **`sp.etwfe` control-group and headline aggregation** now honors the public
  `cgroup` contract. The default `cgroup="notyet"` headline is the
  treated-observation-weighted simple ATT used by R `etwfe::emfx(type="simple")`
  and Stata `jwdid, estat simple`; `cgroup="nevertreated"` now matches R
  `etwfe(cgroup="never")`. `sp.wooldridge_did` remains the historical saturated
  TWFE helper, and `sp.etwfe_emfx(..., weighting=)` still exposes the historical
  cohort-share aggregation through `weighting="cohort"`.

### Added

- **Analytical parity coverage for 10 previously uncovered estimator
  families** — structural (`olley_pakes`, `levinsohn_petrin`,
  `ackerberg_caves_frazer`, `wooldridge_prod`, `prod_fn`, `markup`, `blp`),
  longitudinal regimes (`longitudinal_analyze`, `longitudinal_contrast`,
  `regime`, `always_treat`, `never_treat`), `msm`, parametric g-formula
  (`gformula_ice_fn`, `gformula_mc`), fairness (`demographic_parity`,
  `equalized_odds`, `fairness_audit`, `orthogonal_to_bias`,
  `counterfactual_fairness`, `evidence_without_injustice`), multiple
  imputation (`mice`, `mi_estimate`), `multi_treatment`, off-policy
  evaluation (`sharp_ope_unobserved`, `causal_policy_forest`), surrogate
  index (`surrogate_index`, `long_term_from_short`,
  `proximal_surrogate_index`), and target-trial emulation
  (`target_trial_protocol`, `target_trial_emulate`, `clone_censor_weight`,
  `immortal_time_check`, `target_trial_checklist`, `target_trial_report`).
  Each family gets deterministic-DGP known-truth recovery tests plus
  machine-precision internal identities under
  `tests/reference_parity/test_<family>_parity.py`; the parity index now
  carries 340 graded records. Known limitation surfaced (not fixed here):
  `proximal_surrogate_index`'s second-stage design `[1, W, S_hat]` is
  exactly rank-deficient (`S_hat` is affine in `W`), so its confounded-ATE
  point estimate depends on the scale of the proxy `W`; the recovery test
  is skipped with a documented reason pending a ⚠️ correctness fix.

  The 10 new test files shipped in PR #39 (`f9cc2932`, merged despite
  the misleading title `feat(parity): analytical-only mice + mi_estimate
  Rubin-rules recovery (#39)`, which only highlights the imputation
  family) are:

  | test file | estimator family covered |
  | --- | --- |
  | `tests/reference_parity/test_structural_parity.py` | structural (`olley_pakes`, `levinsohn_petrin`, `ackerberg_caves_frazer`, `wooldridge_prod`, `prod_fn`, `markup`, `blp`) |
  | `tests/reference_parity/test_longitudinal_parity.py` | longitudinal regimes (`longitudinal_analyze`, `longitudinal_contrast`, `regime`, `always_treat`, `never_treat`) |
  | `tests/reference_parity/test_msm_family_parity.py` | `msm` |
  | `tests/reference_parity/test_gformula_family_parity.py` | parametric g-formula (`gformula_ice_fn`, `gformula_mc`) |
  | `tests/reference_parity/test_fairness_parity.py` | fairness (`demographic_parity`, `equalized_odds`, `fairness_audit`, `orthogonal_to_bias`, `counterfactual_fairness`, `evidence_without_injustice`) |
  | `tests/reference_parity/test_imputation_parity.py` | multiple imputation (`mice`, `mi_estimate`) |
  | `tests/reference_parity/test_multi_treatment_parity.py` | `multi_treatment` |
  | `tests/reference_parity/test_ope_parity.py` | off-policy evaluation (`sharp_ope_unobserved`, `causal_policy_forest`) |
  | `tests/reference_parity/test_surrogate_parity.py` | surrogate index (`surrogate_index`, `long_term_from_short`, `proximal_surrogate_index`) |
  | `tests/reference_parity/test_target_trial_parity.py` | target-trial emulation (`target_trial_protocol`, `target_trial_emulate`, `clone_censor_weight`, `immortal_time_check`, `target_trial_checklist`, `target_trial_report`) |

  See [`docs/dev/parity_pr39.md`](docs/dev/parity_pr39.md) for the
  reviewer-facing rationale (why the title was narrower than the body,
  what to expect on `git log` / JOSS audit).

- **Parity index coverage expansion — 305 estimators now carry a graded
  parity record (129 bit-exact), queryable via `sp.parity_status()` /
  `sp.parity_summary()`. This closes the Tier-D worklist of reference-less
  estimators to zero. Across multiple sessions this pass added closed-form
  / known-truth guards across decomposition (Gelbach, Das-Gupta, Kitagawa,
  Lerman-Yitzhaki source, subgroup-Theil, natural-effects mediation,
  interventional effects, four-way-decomposition, sensitivity-mr_DL,
  partial_corr_pvalue, immigration_quartet, etc.), robustness (`sensemakr`,
  `oster_delta`, `breakdown_frontier`), inference (`bootstrap`, `lrtest`,
  `icc`, `fisher_exact`, `romano_wolf`, `anderson_rubin_test`,
  `effective_f_test`, `tF_critical_value`, `mr_f_statistic`, `mr_steiger`,
  `mr_mode`, `meta_analysis`, `cluster_robust_se`), bounds
  (`manski_bounds`, `lee_bounds`, `selection_bounds`, `horowitz_manski`),
  policy learning (`policy_value`, `policy_tree`), postestimation (`margins_at`,
  `mr` IVW), time series (`structural_break`, `cusum_test`, `engle_granger`),
  causal discovery (`fci`, `ges` CPDAGs), conformal (`conformal_ite`,
  `conformal_fair_ite`), interference (`cluster_cross_interference`),
  transportability (`pate`, `front_door`), QTE (`ivqreg`, `beyond_average_late`,
  `continuous_iv_late`, `dist_iv`), the first network rows
  (`degree_centrality`, `betweenness_centrality`, `clustering`), and the
  causal-forest / synthetic-control aggregates (`rate` AUTOC sign anchor +
  `honest_variance` mean-CATE identity, `geolift` exact convex-combo lift
  recovery, `bcf_factor_exposure` adding-up identities + effect recovery,
  `bayes_synth` Dirichlet-simplex ATT recovery, `calibration_test` /
  `test_calibration` BLP heterogeneity detection). A later pass extended the
  guards across the spatial, time-series, panel, survival, frontier and
  distributional-decomposition families: spatial autocorrelation
  (`moran_local`, `getis_ord_local`, `join_counts` on a segregated field) and
  spatial regression (`slx` = augmented OLS, `sac` / `sarar_gmm` rho recovery,
  `spatial_did` effect + no-spurious-spillover, `spatial_iv`, `spatial_panel`);
  time series (`garch` persistence, `granger_causality` direction, `irf` =
  closed-form `A**h`, `johansen` rank, `panel_unitroot`, `bvar` Minnesota
  prior, `its` level shift); panel (`panel_fgls`, `interactive_fe`,
  `panel_logit`, `panel_probit`); competing risks (`cuminc` CIF closed form +
  Gray's test, `finegray`, `cox_frailty`); efficiency frontiers (`malmquist`
  `M = EC·TC` identity, `metafrontier` envelope + technology-gap ratio); and
  decompositions (`rifreg`-at-mean = OLS, `shapley_inequality` additivity,
  `fairlie` and `ffl_decompose` aggregate identities). A further pass added
  interference (`peer_effects` linear-in-means recovery), transportability
  (`transport_generalize` homogeneous-effect invariance + covariate-shift
  ordering), conformal causal inference (`weighted_conformal_prediction`
  marginal coverage, `conformal_ite_interval` coverage + ATE point,
  `conformal_cate`), and shift-share (`ssaggregate` Borusyak-Hull-Jaravel
  location-level equivalence). Finally, the Bayesian and policy-weight layers:
  `bayes_iv` / `bayes_rd` posterior recovery with `rhat < 1.01` / adequate ESS
  convergence guards (skipped without the `[bayes]` extra), and the
  Mogstad-Santos-Torgovitsky `policy_weight_ate` / `policy_weight_subsidy` /
  `policy_weight_marginal` / `policy_weight_observed_prte` closed-form weight
  identities. Every record
  traces to a committed `tests/reference_parity/` guard; grades are honest
  (bit-exact only for machine-precision identities, analytical-only for
  DGP-recovery / coverage guarantees). See `docs/dev/parity_gap_inventory.md`.
- **Universal SE menu — every regression-family estimator is now 8/8 native,
  externally validated.** The reviewer question *"is wild cluster bootstrap
  usable on any estimator or only `feols`?"* now has a clean answer: `regress`,
  `feols`, `ivreg`, `panel(method="fe")` and `hdfe_ols` each expose the full
  canonical `vce=` menu (classical / HC / cluster / two-way `cluster=[a,b]` /
  CR2–CR3 / jackknife / wild cluster bootstrap / Conley spatial HAC), and
  `fepois` / `feglm` add CR2–CR3/jackknife plus the score wild bootstrap.
  Every cell is pinned to an external reference
  (`tests/reference_parity/REFERENCES.md`); the coverage ratchet
  `scripts/se_menu_matrix.py --check` now holds 64 native cells / 0 unsafe.
  Highlights:
  - `sp.feols` / `sp.panel(method="fe")` / `sp.hdfe_ols`:
    `vce="CR2"/"CR3"/"jackknife"` (Pustejovsky–Tipton 2018 on the within
    design — matches R `clubSandwich::vcovCR(plm, model="within")` to machine
    precision) and `vce="conley"` (Stata `acreg` planar-distance convention).
    `panel` also gains two-way `cluster=["a", "b"]` (CGM 2011).
  - `sp.hdfe_ols(vce="robust")`: HC1 with reghdfe's `N/(N-k-df_a)` factor —
    matches Stata `reghdfe, vce(robust)` exactly.
  - `sp.panel(method="fe", vce="wild", cluster=...)`: WCR wild cluster
    bootstrap on the entity-within design, byte-identical to
    `sp.regress(vce="wild")` on the hand-demeaned data.
  - `sp.fepois` / `sp.feglm` `vce="CR2"/"CR3"/"jackknife"`: the IRLS-weighted
    clubSandwich glm adjustment on the FE-as-dummies design — matches R
    `clubSandwich::vcovCR(glm)` for Poisson and logit. (The weighted
    projection does **not** carry CR2 leverage through FE absorption, so the
    dummy design is required; guarded against high-dimensional FE.)
  - `sp.fepois` / `sp.feglm` `vce="wild"`: the restricted score wild cluster
    bootstrap (Kline–Santos 2012) with Stata `boottest`'s exact
    studentization — reverse-engineered from boottest's enumerated bootstrap
    distribution (cluster-share-centered CRVE + strict exceedance counting)
    and **bit-exact** vs `boottest` in the enumerated regime (`2^G <= reps`),
    verified for Poisson and logit.
  - `sp.ppmlhdfe(cluster=["a", "b"])`: CGM-2011 two-way clustering on the
    FE-residualised PPML design — byte-identical to Stata
    `ppmlhdfe, cluster(a b)`.
  - `sp.ppmlhdfe(vce="wild")`: the boottest-convention score bootstrap
    computed **on the FE-absorbed design at scale** (the weighted-FWL
    reduction of the one-step numerator is exact, verified to 1e-17; runs
    500-level FE in <1s) — byte-identical to `fepois(vce="wild")` on
    low-dimensional FE and hence transitively bit-exact vs Stata `boottest`.
    A beyond-Stata capability: `boottest` cannot run after Stata's
    `ppmlhdfe` at all. `vce="CR2"/"CR3"` added with the same
    reference-matching dummy design + high-dim guard as `fepois`.
  - `sp.fepois` / `sp.feglm` / `sp.ppmlhdfe` `vce="conley"`: GLM Conley
    spatial HAC referenced to R `conleyreg` (the only reference supporting
    GLMs; reproduces its spherical uniform kernel to ~1e-7). OLS menu keeps
    the Stata `acreg` planar convention; the split is documented.
  - `sp.did(method="2x2", vce="wild", cluster=...)`: the canonical
    few-clusters DiD inference (MacKinnon-Webb 2017) — WCR wild cluster
    bootstrap on the interaction regression, validated against Stata
    `boottest` and byte-identical to `sp.regress(vce="wild")` on the same
    design. Staggered methods refuse `vce=` (Callaway-Sant'Anna's multiplier
    bootstrap on influence functions is itself the wild bootstrap for that
    estimator and is exposed via `sp.aggte`). `did` deliberately does NOT
    offer unclustered classical/HC SEs (Bertrand-Duflo-Mullainathan 2004).
  See `docs/guides/grammar.md` for the full matrix and the verified-reference
  map.
- **Cross-language parity — within-transformation pinned to textbook
  mean-within.** New Track A module `68_demean_within` aligns
  `sp.demean(solver="map")` against the textbook entity-mean projection
  (algorithmic, no R package sibling). Worst observed gap is 3.5e-15 (machine
  tier). The function is now graded **bit-exact** in the parity matrix.
- **Cross-language parity — panel-balance filter aligned.** New Track A
  module `69_balance_panel` aligns `sp.balance_panel` against the base R
  `counts == n_periods` row filter on an unbalanced 5×4 panel. All eight
  rows agree to 0.0 (bit-exact). The function is now graded **bit-exact** in
  the parity matrix.
- **Cross-language parity — absorbed-FE panel GLM family added.** New Track A
  module `67_panel_glm` aligns `sp.feglm` (Bernoulli logit) and `sp.fepois`
  against `fixest::feglm` and `fixest::fepois` (single entity fixed effect
  absorbed by both sides). Coefficients agree to ~1e-8 (machine tier); SEs
  differ at ~1e-5 because the two IWLS implementations iterate to slightly
  different working-weight roots — well within the iterative tier. Both
  functions are now graded **bit-exact** in the parity matrix.
- **Cross-language parity — spatial family opened.** Two new Track A modules
  (spatialreg 1.4.3 / spdep 1.4.2, row-standardised 12×12 rook lattice):
  - `65_spatial` aligns `sp.sar` / `sp.sem` / `sp.sdm` against
    `spatialreg::lagsarlm`, `spatialreg::errorsarlm`, and
    `spatialreg::lagsarlm(Durbin=TRUE)` — all three **bit-exact** (worst
    relative error 8.3e-8 on estimates, 2.0e-8 on SEs).
  - `66_spatial_gmm` aligns `sp.sar_gmm` against `spatialreg::stsls(W2X=FALSE)`
    (a closed-form spatial-2SLS projection — coefficients and n−k SEs agree to
    ~1e-15) and `sp.sem_gmm` against `spatialreg::GMerrorsar` (coefficients and
    the spatial-error `lambda` **bit-exact**, worst 4.6e-8; point-only, as the
    two coefficient-SE variance estimators differ by convention).

  Together this moves the `spatial` family from 0 verified estimators to 5.
  `sp.sarar_gmm` is left unverified: its joint GS-lag + GM-error path does not
  match `spatialreg::gstsls` (different moment sequence).
- **`sp.rlasso` — rigorous (data-driven) Lasso, a faithful port of R's
  `hdm`.** A new first-class module (`statspai.rlasso`) that ports
  `hdm::rlasso` / `rlassoEffect` / `rlassoIV` line-for-line, validated to
  agree numerically with `hdm` 0.3.2:
  - `sp.rlasso(X, y, post=, intercept=, penalty=, control=)` — rigorous
    post-Lasso with the data-driven penalty
    `λ₀ = 2c√n·Φ⁻¹(1−γ/2p)` (`c=1.1`, `γ=0.1/log n`), heteroskedastic
    penalty loadings refined by iteration, and hdm's `LassoShooting`
    coordinate descent. Matches `hdm::rlasso` to **machine precision**
    (coefficients, λ₀, loadings, residuals and selected support are
    bit-exact across `post`/`intercept`/`homoscedastic` variants).
  - `sp.rlasso_effect(x, y, d, method=)` / `sp.rlasso_effects(...)` —
    treatment-effect inference after Lasso-selecting controls
    (`"partialling out"` and `"double selection"`); α and SE match
    `hdm::rlassoEffect` to ~1e-14.
  - `sp.rlasso_iv(y, d, z, x, select_Z=, select_X=)` — IV with rigorous
    selection of instruments and/or controls (`hdm::rlassoIV`). On the
    canonical eminent-domain application it reproduces
    `hdm::rlassoIV(..., select.X=FALSE, select.Z=TRUE)` exactly
    (coef 0.2274, SE 0.2466) — **resolving the previously-tracked
    divergence** where the older `iv.bch_post_lasso_iv` was ~17× off
    (0.013). All four selection regimes (Z-only, X-only, both, none)
    match `hdm` to ~1e-6 on a well-conditioned design. Also routable
    through the IV family dispatcher: `sp.iv(method='rlasso', ...)`
    (instrument selection by default; double selection when `exog=`
    controls are passed).
  - `sp.RlassoRegressor` / `sp.RlassoClassifier` — scikit-learn-compatible
    adapters so the rigorous Lasso can serve as a Double-ML nuisance
    learner: `sp.dml(model='plr', ml_g='rlasso', ml_m='rlasso')` now
    works (clone-safe across cross-fitting folds).
  - `sp.rlassologit` — the **logistic** rigorous (post-)Lasso, a faithful
    port of `hdm::rlassologit`. hdm delegates the penalized fit to glmnet's
    binomial lasso at a single data-driven `λ`; StatsPAI reproduces glmnet
    directly (IRLS + weighted coordinate descent, `1/n` deviance,
    population-variance standardization, `pmin` clamp). The selected
    support matches glmnet **exactly**; engine coefficients match R glmnet
    4.1 to ~1e-6; `post=True` coefficients/residuals (unpenalized logistic
    refit on the selected set) match `hdm` to ~1e-9. `sp.RlassologitClassifier`
    is a *genuine* (calibrated) logistic propensity for Double-ML —
    `sp.dml(model='irm', ml_m='rlassologit')` — unlike the linear-probability
    `RlassoClassifier`. The high-dim logistic *effect* (`rlassologitEffect`)
    is intentionally deferred (a separate parity exercise). 5 hdm/glmnet
    parity pins in `test_rlassologit_parity.py`.
  Coverage: `tests/reference_parity/test_rlasso_parity.py` (17 pins vs
  `hdm`, generated by `_generate_rlasso.R` — including `rlasso_effects`
  multi-target and a tight `sp.dml(ml_g='rlasso')` pin against a manual
  R DoubleML-PLR-with-`hdm::rlasso` reference on shared folds, agreeing to
  machine precision) and `tests/test_rlasso.py` (24 behavioural/edge tests,
  incl. the §3 result-contract). References (verified): `belloni2012sparse`
  (Econometrica 80(6), doi `10.3982/ECTA9626`), `belloni2014inference`
  (RES 81(2)), `chernozhukov2016hdm` (R Journal 8(2), doi
  `10.32614/RJ-2016-040`).
  - Implementation note: hdm's IV routines use `MASS::ginv` with tolerance
    `√eps` on the (rank-deficient) control blocks; matching that cutoff —
    rather than numpy `pinv`'s default `rcond=1e-15` — is what restores
    eminent-domain parity. `iv.bch_post_lasso_iv` keeps its original
    numerics (its docstring now flags `sp.rlasso_iv` as the hdm-faithful
    path).

- **`sp.dml` — DoubleML-compatible score & IPW options.** Additive
  options on `sp.dml` / the per-model classes, matching `doubleml-for-py`:
  - `score='IV-type'` for `model='plr'` (moment denominator `Σ v̂·D`
    instead of `Σ v̂²`); default stays `'partialling out'`.
  - `score='ATTE'` for `model='irm'` (effect on the treated); default
    stays `'ATE'`.
  - `normalize_ipw=` (Hájek self-normalized IPW) and
    `trimming_threshold=` (symmetric propensity clip) for `irm` / `iivm`.
  Each new path is pinned against `doubleml-for-py` 0.11.3 in
  `tests/external_parity/test_dml_python_parity.py` (7 pins total) and
  guarded by `tests/test_dml_score_options.py`. **Defaults reproduce the
  previous output bit-for-bit** — `score=None` / `normalize_ipw=False` /
  `trimming_threshold=0.01` is the historical estimator (verified PLR
  partialling-out and IRM ATE both unchanged to machine precision). This
  is a feature addition, **not** a correctness change to any default.
  New options are surfaced in `result.model_info` (`score`,
  `normalize_ipw`, `trimming_threshold`). See the expanded guide
  [`docs/guides/sp_dml_vs_doubleml.md`](docs/guides/sp_dml_vs_doubleml.md),
  which also adds a custom-orthogonal-score extension recipe and a
  DoubleML-alignment roadmap.

### Changed

- **Docs / citations.** Added the verified `hdm` (Chernozhukov, Hansen &
  Spindler, *The R Journal* 8(2), 2016, doi `10.32614/RJ-2016-040`) and
  Chiang–Kato–Ma–Sasaki (*JBES* 40(3), 2022, doi
  `10.1080/07350015.2021.1895815`) references to `paper.bib`; cite `hdm`
  from the post-Lasso IV / RD-lasso modules that implement its methods.
  The DML guide documents the relationship to `hdm` openly. (The original
  hand-rolled `bch_post_lasso_iv` under-selected instruments vs
  `hdm::rlassoIV` on weak-instrument designs; this is now resolved by the
  dedicated `sp.rlasso` port — `sp.rlasso_iv` reproduces `hdm::rlassoIV`
  exactly — and `bch_post_lasso_iv` is deprecated. See the `sp.rlasso`
  entry above.)
- **Quality gate (`scripts/quality_gate.py`).** The `mypy` debt ratchet now
  counts only StatsPAI-authored errors (lines under `src/statspai/`), so errors
  mypy surfaces while following imports into installed third-party packages no
  longer inflate the count in a venv-dependent way — the same environment
  stability the `ignore_missing_imports` config note already targets. Failure
  detection no longer keys off the exit code or a config-warning regex (a
  third-party parse error under `python_version = 3.9`, or a newer mypy's
  deprecation *note* about that floor, drove the gate to a spurious permanent
  failure); a run is now trusted whenever mypy actually produced analysis. With
  the count reflecting only our code — measured at 0 under the pinned mypy 1.x —
  `DEFAULT_MYPY_MAX` is tightened from `1058` to `25`, converting a dormant
  ratchet into a live one. New guards in `tests/test_import_budget.py`.
- **`StatsPAI_full_data_analysis_skill/SKILL.md`.** The frontmatter description,
  keyword list, and triggers now advertise the distributional / gap-decomposition
  family (`sp.oaxaca`, `sp.kitagawa_decompose`, `sp.dfl_decompose`, `sp.gelbach`,
  `sp.fairlie`, `sp.rif_decomposition` via the `sp.decompose` dispatcher), which
  the runtime already ships but the skill never surfaced, so downstream skill
  family-tagging no longer reports zero coverage for it (closes #35). The
  `SkillOpt-style execution gate (task-local card)` subsection (the `best_skill`
  card plus its seven promotion rules) is restored inside the operating-loop
  section so the weekly downstream sync gate stops re-deleting it (closes #34).

### Deprecated

- **`iv.bch_post_lasso_iv`** now emits a `DeprecationWarning`. It is
  StatsPAI's original from-memory BCH-2012 reconstruction and does not
  agree numerically with `hdm` (≈17× off on eminent domain). Use the new,
  parity-tested `sp.rlasso_iv` instead. Behaviour is unchanged during the
  deprecation window; see [`MIGRATION.md`](MIGRATION.md).

### Fixed

- **`sp.spatial_iv` now accepts a native StatsPAI `W` object.** Passing a
  weights object built by `sp.queen_weights` / `sp.rook_weights` /
  `sp.knn_weights` (the natural way to get spatial weights) raised an opaque
  `IndexError: tuple index out of range` deep inside the estimator. `_coerce_W`
  assumed the libpysal convention where `W.full()` returns a `(array, ids)`
  tuple and indexed `W.full()[0]`, but StatsPAI's own `W.full()` returns the
  dense `(n, n)` array directly — so `[0]` sliced out the first *row*, a
  length-`n` vector, and the matrix multiply collapsed. The coercion now
  detects the tuple form and otherwise uses the array as-is, so both StatsPAI
  and libpysal weights work; calls that passed a raw NumPy array (the only
  path that worked before) are numerically unchanged. New guard:
  `tests/reference_parity/test_spatial_models_parity.py::test_spatial_iv_accepts_native_W_object`.
- **`sp.peer_effects` and the `sp.network` graph layer now accept a native
  StatsPAI `W` object.** Both carried the same `W.full()[0]` first-row bug as
  `spatial_iv` (peer_effects raised `AxisError: axis 1 is out of bounds` on the
  row-normalisation; `network._to_dense` silently returned a length-`n` vector
  instead of the `(n, n)` adjacency). Both now detect the libpysal `(array,
  ids)` tuple form and otherwise use the dense array directly. Raw-array /
  scipy-sparse inputs are unaffected. New guard:
  `tests/reference_parity/test_peer_effects_parity.py::test_peer_effects_accepts_native_W_object`.

## [1.20.0] — 2026-06-22

### ⚠️ Correctness

These change *inference output* (standard errors / p-values / CIs), not point
estimates. See `MIGRATION.md` for per-function detail.

- **`sp.cusum_test`** — the recursive-residual CUSUM now compares the path
  against the Brown–Durbin–Evans *linear* boundary `a·[1 + 2 s/(n−k)]`
  (`a = 0.948` at 5%) instead of a constant `1.358` (the sup|Brownian-bridge|
  value, which belongs to the OLS-CUSUM). The old boundary rejected ≈32% of
  stable series at a nominal 5% level (empirical size 0.32 → 0.04).
  `result["critical_value"]` is now the per-recursion boundary array.
- **`sp.lee_bounds`** — the confidence interval is now the genuine Imbens &
  Manski (2004) interval for the partially identified parameter (critical
  value `C_n` interpolating between the one- and two-sided `z`), replacing a
  Horowitz–Manski *set* interval that over-covered. CIs are narrower (correct).
- **`sp.rdrobust` / `sp.rd2d` / RD HTE / `sp.rd_bias_aware_fuzzy`** — the
  heteroskedasticity-robust local-polynomial variance now uses the
  Calonico–Cattaneo–Titiunik (2014) kernel weighting `X'W·diag(e²)·W·X`
  (kernel weight **squared**). The previous meat carried only one power of the
  kernel weight, inflating every HC-robust RD standard error (≈1.4× for a
  uniform kernel vs R `rdrobust` `vce="hc0"`). Point estimates are unchanged;
  SEs now match R. (Cluster-robust RD SEs were already correct.)
- **`sp.callaway_santanna` pre-trend test** — the joint pre-trend Wald p-value
  (`model_info["pretrend_test"]`) now applies a Hotelling-T² finite-sample
  correction, referring `W·(G−k)/(k·(G−1))` to `F(k, G−k)` instead of the
  plug-in `χ²(k)`. The pre-period ATT(g,t) are strongly correlated and the
  covariance is estimated, so the plug-in χ² over-rejected (empirical size
  ≈0.15 at a nominal 5% level for ~60 units → ≈0.07). ATT point estimates and
  SEs are unchanged.
- **`gardner_did(event_study=True)` overall ATT** — the headline ATT is now the
  treated-observation-**weighted** mean of post-period coefficients (the
  `did2s` convention), matching the non-event-study path exactly. Previously an
  *unweighted* mean disagreed with the non-ES ATT under heterogeneous effects /
  unbalanced horizon support (e.g. 1.63 vs 1.75).
- **`sp.hdfe_ols` / `sp.absorb_ols` cluster-robust SEs (native HDFE backend)** —
  the CRV1 finite-sample factor `(N−1)/(N−K) · G/(G−1)` no longer counts, in
  `K`, fixed-effect levels that are **nested within the cluster variable**. In
  the canonical `absorb(unit + time) + cluster(unit)` layout the absorbed
  `unit` FE is fully nested in `cluster(unit)`, so the cluster-robust sandwich
  already captures that within-cluster correlation; charging `unit`'s `(G−1)`
  levels again in `K` inflated every clustered SE. The native backend now omits
  nested-FE levels from the cluster DOF (matching the `reghdfe` / `pyfixest` /
  `sp.feols` convention) while still charging non-nested levels (e.g. `time`).
  Point estimates and non-clustered (`iid` / `hetero`) SEs are unchanged; only
  `cluster=` SEs change — they get *smaller*. The reporter's MRE (#26) now gives
  `sp.hdfe_ols` and `sp.feols` identical SEs (ratio 1.0); the prior inflation
  was ≈5.4% on the synthetic panel and ≈6.3% on a 37,869-row firm-year panel.
  Two new result fields expose the correction: `dof_fe_cluster` (FE dof actually
  charged to CRV1) and `nested_fe` / `nested_fe_in_cluster` (which absorbed
  dimensions were detected as nested in the cluster).

### Known limitations

- **`sp.rdrobust` bias-corrected estimate for `rho != 1` (`b != h`).** The
  "Robust" row currently uses the standalone order-`(p+1)` fit at `b` rather
  than the Calonico–Cattaneo–Titiunik `μ̂_p(h) − bias(b)` construction; the two
  coincide only at the default `rho=1`. Point/SE differ slightly from R
  `rdrobust` when `b != h`. The default (`rho=1`) is exact. A full CCT
  bias-corrected point+variance for `rho != 1` is tracked for an R-parity sprint.

### Added

- **CausalPy-inspired convergence layer (one contract across quasi-experiments).**
  - **`sp.counterfactual_data` / `sp.counterfactual_plot`** — one contract that
    normalises any observed-vs-counterfactual result (causal impact, synthetic
    control, interrupted time series, and the Bayesian time-series designs) into a
    tidy frame and a two-panel observed-vs-counterfactual + pointwise-effect plot
    with uncertainty bands. `sp.its` now stores the observed/fitted/counterfactual
    series so it joins the contract (previously its `detail` was empty).
  - **`sp.ancova` / `sp.negd`** — lightweight pre/post quasi-experiment wrappers
    (covariate-adjusted ANCOVA; non-equivalent group design via ANCOVA or
    change-score) returning the unified `CausalResult` with surfaced assumptions.
  - **`sp.geolift`** — geo-experiment lift measurement: aggregates treated markets
    and builds a synthetic counterfactual from untreated markets (via `sp.synth`),
    reporting ATT + relative lift % and feeding `counterfactual_plot`.
  - **`sp.bayes_its` / `sp.bayes_synth`** — Bayesian interrupted time series
    (segmented; estimand = level change) and Bayesian synthetic control
    (Dirichlet-simplex donor weights; estimand = ATT). Both return posterior
    counterfactual trajectories with credible bands and require the `bayes` extra.
  - **Inference engine switch.** `sp.rdrobust(..., engine='bayes')` and
    `sp.did_2x2(..., engine='bayes')` route the same design to the Bayesian
    backend (`sp.bayes_rd` / `sp.bayes_did`); options with no Bayesian analogue
    raise rather than being silently dropped. `sp.causal_question(..., engine=
    'bayes')` routes regression-discontinuity questions to `sp.bayes_rd`.
    `engine='ols'` (default) is unchanged.
- **`gardner_did(vce='bootstrap')` and `did_imputation(vce='bootstrap')`** —
  a pairs-cluster bootstrap of the *full* estimator for valid overall-ATT
  inference. The analytic defaults cluster only the second-stage / imputation
  residuals and ignore the variance from estimating the fixed effects, so they
  are anti-conservative; a `UserWarning` now steers users to `vce='bootstrap'`.
  Simulated 95% coverage improves from ≈0.78 → ≈0.90 (`gardner_did`) and
  ≈0.87 → ≈0.94 (`did_imputation`). Default behaviour and point estimates are
  unchanged; `n_boot` / `boot_seed` control the bootstrap.
- **`sp.iv(..., cluster=<Series>)`** — passing the cluster as a `pandas.Series`
  (or array) is now accepted, matching the documented signature; previously it
  raised `ValueError: truth value of a Series is ambiguous`. A misaligned
  cluster vector now fails loudly with a clear length-mismatch error.
- **`sp.effect_summary(...)` / `EffectSummary`** — a decision-ready table-plus-
  text summary surface for frequentist `CausalResult` objects and Bayesian
  posterior results, with `CausalResult.effect_summary(...)` and
  `BayesianCausalResult.effect_summary(...)` method hooks.
- **`sp.counterfactual_data(...)` and `sp.counterfactual_plot(...)`** — a shared
  observed-vs-counterfactual contract for causal-impact, synthetic-control, and
  interrupted-time-series results. `sp.its(...)` now stores the fitted
  no-intervention counterfactual series in `ITSResult.detail` so it joins the
  same plotting path.
- **`sp.design_intake(...)` and `statspai.checks`** — additive agent-facing
  contracts for pre-estimator design routing and pluggable diagnostic checks,
  backed by a new `quality_gate.py contract-inventory` drift guard.
- **`sp.rdrobust(..., engine='bayes')`** — sharp-RD calls can now route to the
  existing Bayesian RD backend while the default `engine='ols'` path remains
  byte-identical.
- **`sp.ancova(...)` and `sp.negd(...)`** — named pre/post quasi-experimental
  wrappers around the shared OLS machinery, surfacing ANCOVA, change-score
  NEGD, assumptions, and regression-to-the-mean warnings as `CausalResult`
  objects.
- **`scripts/agent_workflow_spec_audit.py`** — a static audit for agent
  empirical-analysis workflow specs, including a bundled DID example contract
  and focused regression tests.

### Changed

- **`sp.match` (nearest-neighbour) default SE.** The public docstring and a new
  `UserWarning` now flag that the default `se_method='ai'` matched-pair SE is
  anti-conservative under matching with replacement (~0.81 coverage at a
  nominal 95% level) and recommend `se_method='abadie_imbens'`. The default
  *number* is unchanged (JOSS-review stability); the internal `_ai_se`
  docstring no longer mislabels the simple SE as Abadie–Imbens (2006).

### Fixed

- **pandas 3.0 compatibility (formulas with string categoricals).** pandas 3.0
  makes `StringDtype` the default for text columns, which patsy's categorical
  sniffer cannot interpret — so any formula with a string categorical
  (e.g. `y ~ x + C(group)`) raised `TypeError: Cannot interpret
  '<StringDtype>' as a data type` in `sp.regress` / `sp.ols` / `sp.glm` /
  `sp.svyglm` at both `fit` and `predict`. String-extension columns are now
  coerced to `object` at every patsy entry point. No-op on pandas < 3.0;
  point estimates and standard errors are unchanged.
- **Nonlinear (Fairlie/Yun) decomposition probit covariance symmetry.** On a
  singular / collinear design the Newton-fallback covariance from
  `numpy.linalg.pinv` could pick up BLAS-backend-dependent asymmetric float
  noise off the diagonal, so the reported probit covariance was not exactly
  symmetric on rank-deficient designs (and differed across platforms). It is now
  symmetrised by construction. Variances (the diagonal / reported SEs) and point
  estimates are unchanged — a ~1e-15 no-op on well-conditioned fits.
- **Nonlinear (Fairlie/Yun) decomposition probit singular-information
  fallback.** `_probit_fit` now chooses the SVD pseudo-inverse from the
  information-matrix condition number instead of relying on `np.linalg.inv` to
  raise on rank-deficient designs. Some OpenBLAS/LAPACK builds returned an
  indefinite inverse-like matrix with huge negative variances rather than
  raising, which made the singular-design regression fail in the pandas-3 CI
  lane. Well-conditioned fits keep the exact `inv` path, so point estimates and
  regular-design covariance output are unchanged.
- **`sp.fast.feols(..., backend='jax')` rank-deficiency detection.** Newer JAX
  returns a finite least-norm solution for a singular design instead of
  `NaN`/`Inf`, so perfectly collinear regressors silently produced output. The
  JAX path now raises `NumericalInstability` on a rank-deficient design (via the
  bread condition number), matching the native `feols` behaviour.
- **`sp.regress` / `sp.ols` constant-outcome warning.** A zero-variance outcome
  now emits an explicit `UserWarning` (R² is undefined); previously this relied
  on an incidental NumPy divide warning that newer NumPy no longer raises.
- **Agent tool error envelope.** `execute_tool` no longer crashes if a custom
  result's `to_dict()` raises an unexpected exception type; the error-handling
  fallback now degrades gracefully on any exception.

### Internal

- **Test-suite forward-compatibility (no library behaviour change).** The
  default `pytest` run is now clean under pandas 3.0 and across BLAS backends:
  the `error_taxonomy` governance ratchet moved from the default suite to a
  dedicated CI step (so a fresh clone is not gated by a drifting count); a CoW
  read-only test-setup bug was fixed (`Series.to_numpy().copy()`); platform-
  sensitive RD/synthetic-control coverage snapshots now check data-driven
  standard errors / optimiser outputs for validity while keeping point
  estimates pinned (cross-language numerical parity is still guarded by
  `tests/reference_parity/`); and arviz/PyMC/exception-type test assumptions
  were made version-agnostic. A new CI lane runs the full suite under pandas
  3.x on every push/PR.

## [1.19.0] — 2026-06-20

### Added

- **Cross-engine validation (`sp.cross_validate`).** Estimate one model with
  several *independent* engines and report whether they agree — operationalising
  the cross-package-reproducibility discipline (Scott Cunningham: "estimate the
  same model two ways, trust it only when they match") as one call for humans
  and agents.
  - **Engines.** StatsPAI native, **pyfixest**, **linearmodels**, **DoubleML**
    (in-process Python), R's **`fixest`** and **`did`** via `Rscript`, and
    **Stata** (`regress` / `ivregress` / `reghdfe`) via batch `do`.
    `engines="auto"` runs every installed, applicable backend; a named-but-
    missing engine is reported `unavailable` and recorded as a degradation —
    never silently skipped.
  - **Estimands.** `ols`, `feols` (HDFE), `iv` (2SLS), `poisson`, `dml`, and
    **`did`** (Callaway–Sant'Anna: StatsPAI's `callaway_santanna` overall ATT
    vs R's `did::att_gt` + `aggte`, reproducing Scott Cunningham's exact
    cross-package experiment — they agree to ~1e-15 on the canonical `mpdta`).
    Accepts a fixest-style formula, structured args, or a fitted StatsPAI
    result. The `R::did` adapter coerces integer cohort/time/id columns to
    numeric — `att_gt` silently returns a different estimate for integer vs
    numeric group columns, a real cross-package fragility the cross-check
    neutralises.
  - **Honest verdicts.** AGREE / PARTIAL / DISAGREE / INSUFFICIENT, with a
    tolerance regime chosen *and explained* per estimand: closed-form methods
    (OLS/IV/FE) compared at `rtol=1e-6`; randomised methods (DML/forests) on a
    standard-error scale. Verified: StatsPAI reproduces R's `fixest` and
    pyfixest to machine precision (`max_rel ≈ 1e-15`) on OLS/IV/FE.
  - **Result object.** `CrossValidationResult` with `.summary()`, `.plot()`,
    `.to_markdown()`, `.to_latex()`, `.to_dict(detail="agent")` (verdict +
    per-engine table + next steps + version/data `provenance`,
    `engine_status_counts`, and `can_claim_cross_engine_agreement`). Registered
    and MCP-exposed.
- **Data-source ingestion normalisers (`sp.from_worldbank` / `sp.from_fred` /
  `sp.from_sdmx`).** Reshape a payload a data MCP already fetched (World Bank
  Indicators, FRED series, OECD/Eurostat SDMX-JSON) into a tidy long/wide panel
  ready for `sp.detect_design` → `sp.recommend` → `sp.cross_validate`. Pure
  normalisers — no network calls, deterministic and offline-testable. The
  normalized frames carry `df.attrs["provenance"]`, which `sp.cross_validate`
  preserves under `cv.provenance["data"]`.
- **Social network analysis (`sp.network`).** A new numpy/scipy-native SNA
  module aligned with R's `igraph` / `sna` / `statnet` and Stata's
  `nwcommands` — no `networkx` dependency. Covers the full applied stack:
  - **Graph object + factory** — `sp.network_graph(...)` builds from a dense/
    sparse adjacency, edge list, or tidy `DataFrame` (directed/undirected,
    weighted); asymmetric matrices passed as undirected are symmetrised with a
    loud warning, never silently.
  - **Descriptives** — `sp.network_summary`, `sp.transitivity`,
    `sp.clustering`, `sp.reciprocity`, `sp.assortativity` (Newman),
    `sp.network_components`. Verified to match `networkx` to machine precision
    on Zachary's karate club (density, transitivity, average clustering,
    diameter, average path length, degree assortativity).
  - **Centrality** — `sp.centrality(g, kind=...)` dispatcher plus degree,
    closeness (Wasserman-Faust), betweenness (Brandes 2001), eigenvector,
    Katz, PageRank (Brin-Page), Bonacich power, and HITS. Betweenness/
    closeness/eigenvector/Katz match `networkx`; PageRank matches the exact
    Google-matrix stationary distribution (`igraph`-equivalent) to 1e-13.
  - **Community detection** — `sp.community_detection(g, method=...)` with
    Louvain (Blondel 2008), greedy/CNM (Clauset-Newman-Moore 2004), and label
    propagation (Raghavan 2007), plus `sp.network_modularity`. Greedy
    reproduces `networkx`'s CNM partition exactly on the karate club
    (Q = 0.3807, 3 communities); Louvain reaches Q ≈ 0.419.
  - **Network regression** — `sp.netlm` / `sp.netlogit` (QAP / MRQAP with
    Dekker-Krackhardt-Snijders double-semi-partialling) and
    `sp.dyadic_regression` with Aronow-Samii-Assenova (2015) dyadic-cluster-
    robust standard errors (verified against a brute-force O(D²) computation).
  - **Network formation** — `sp.ergm(...)` fits exponential random graph
    models by maximum pseudo-likelihood (MPLE; Strauss-Ikeda 1990). MPLE
    coincides with the exact MLE for dyad-independent terms (edges / nodematch
    / nodecov / absdiff / mutual); dyad-dependent terms (`triangles`) emit a
    loud warning that MPLE is approximate. Full MCMC-MLE and SAOM/RSiena
    dynamics are documented as the roadmap (not silently stubbed).
  - **Data & plots** — `sp.karate_club()` (Zachary 1977) and
    `sp.florentine_families()` (Padgett-Ansell 1993) canonical reference
    networks; `sp.network_plot(...)` node-link drawing with a numpy
    Fruchterman-Reingold layout and lazy `matplotlib` import (no hard
    dependency).

  All eight flagship functions carry curated agent-native registry specs
  (`sp.describe_function` / `sp.function_schema`). 21 new references verified
  via the Crossref API and added to `paper.bib`. New test suite
  `tests/test_network.py` (42 cases: karate/Florentine parity, analytic
  small-graph closed forms, and boundary cases).

### ⚠ Correctness

- **`sp.conformal_synth` average-effect p-value corrected to the moving-block
  test (Chernozhukov, Wüthrich & Zhu 2021).** The per-period conformal p-values
  were already correct, but the *average* post-treatment-effect p-value compared
  a `T1`-averaged statistic (`|mean of the T1 post-period gaps − τ0|`) against
  *single* pre-period residuals. A `T1`-average has roughly `1/√T1` the spread of
  one residual, so this scale mismatch pinned the average p-value at its
  `1/(T0+1)` floor regardless of the data. `_conformal_avg_pvalue` now builds the
  null from all length-`T1` cyclic blocks of the full residual series under H0
  (matched scale); at `T1 = 1` it reduces *exactly* to the per-period p-value.
  Effect: only the average-effect `.pvalue` (and its inverted average CI) change;
  the point estimate, SE, and per-period results are unchanged. Null p-values are
  now well-calibrated (mean ≈ 0.5 under H0 instead of being biased toward the
  floor). Regression anchor `test_cov95_synth_variants.py` updated (`pvalue`
  1/12 → 1/18 on the fixture).
- **`sp.sensitivity_dashboard` no longer fabricates subsample / outlier
  stability (CLAUDE.md §7).** The "80% subsample stability" and
  "outlier sensitivity (trimming)" dimensions previously reported metrics built
  from `baseline_est + N(0, baseline_se)` Gaussian jitter and
  `baseline_est * (1 + N(0, frac_removed))` — i.e. synthetic noise around the
  point estimate, with the actually-sliced subsample never re-estimated. They now
  perform a *genuine* OLS re-fit of the headline coefficient on the stored linear
  design (`data_info['X']`/`'y'`) for each 80% row resample / outcome-trimmed
  subset, gated by a self-consistency check (the full-sample re-fit must
  reproduce the baseline). Results that do not expose a plain linear design
  (weighted / IV / DiD / synthetic-control families) now **skip** these
  dimensions instead of emitting fabricated numbers. The unimplemented MSM
  `trim_sweep` (its loop body was `pass`) is relabelled as the positivity check
  it actually is, with every flag tied to the real max-stabilized-weight signal.
  Also fixes the headline-coefficient selection, which excluded only Stata's
  `_cons` and so analysed the **intercept** for patsy-`Intercept` regressions;
  it now excludes `_cons` / `Intercept` / `const` and analyses the first real
  regressor.
- **`registry.py` `sp.bridge` `kind` parameter metadata fix.** A positional
  `ParamSpec(...)` call placed the description text in the `default` slot and the
  allowed-values list in the `description` slot; the `kind` enum/description/
  default are now correct (keyword arguments), so the generated agent/MCP schema
  for `sp.bridge` describes its `kind` choices accurately.
- **`_agent_cards_extra.py` duplicate agent cards removed.** The
  `network_hte` and `inward_outward_spillover` agent-native cards were each
  defined twice in one dict literal; the earlier (richer) definitions were
  silently shadowed by the later ones. The dead shadowed entries were removed
  (the runtime-active cards are unchanged).

### Added

- **Agent-native result protocol (`.to_dict()` / `.to_latex()` / `.cite()`) on
  64 domain result objects.** The flagship `CausalResult` /
  `EconometricResults` already exposed the full protocol, but the lighter
  result dataclasses only had `.summary()`. A new `ResultProtocolMixin`
  (`statspai._result_serialize`) now gives **64** result classes — across the
  negative-control / proximal / mediation / interference (orthogonal + cluster
  designs) / ITS / Rosenbaum / BCF, plus the Mendelian-randomization,
  meta-learner, multiple-testing, multilevel, QTE, robustness, transport,
  target-trial, OPE, longitudinal-TMLE, g-formula, bootstrap, meta-analysis,
  matching, selection and time-series families — a uniform, JSON-safe `.to_dict()`
  (numpy / pandas / NaN-aware, via the existing `core.results._to_jsonable`), a
  compact booktabs `.to_latex()` table of the scalar fields, and a `.cite()`
  returning the estimator's **verified** paper.bib citation key(s). The
  citation keys are sourced from the modules' existing references and checked
  against `paper.bib` — zero-hallucination (CLAUDE.md §10): a method with no
  single canonical paper honestly returns a placeholder rather than a
  fabricated reference, and a CI test (`tests/test_result_protocol.py`) fails
  if any `_citation_keys` value is absent from `paper.bib`. The serialisation
  logic lives in exactly one place. Pinned by `tests/test_result_protocol.py`
  and `tests/test_result_to_dict.py`.
- **29 estimators made agent-discoverable (`__all__` / registry drift repair).**
  A family of estimators — the proximal / negative-control identification
  methods (`sp.double_negative_control`, `sp.proximal_regression`,
  `sp.negative_control_outcome`, `sp.negative_control_exposure`), the
  off-policy-evaluation estimators (`sp.ips`, `sp.snips`, `sp.doubly_robust`,
  `sp.direct_method`), `sp.four_way_decomposition`, `sp.its`, `sp.ltmle`,
  `sp.harvest_did`, `sp.overlap_weighted_did`, `sp.network_hte`,
  `sp.inward_outward_spillover`, the shift-share political-economy designs,
  the neural dose-response estimators (`sp.vcnet`, `sp.scigan`),
  `sp.rosenbaum_bounds` / `sp.rosenbaum_gamma`, the BCF extensions, and more
  — already shipped on `sp.<name>` but were missing from `__all__`. Because the
  registry auto-pass walks `__all__`, every one of them was invisible to
  `sp.list_functions()`, `sp.describe_function()` and `sp.function_schema()`,
  violating the agent-native contract that the help tools resolve for *every*
  public symbol. They (plus their result classes) are now in `__all__`, so they
  appear in the registry (1033 → 1070 functions), `from statspai import *` is
  complete, and the MCP schema bundle covers them. Eleven of them additionally
  gained hand-written agent-native cards (identifying assumptions, failure
  modes, alternatives). The `__all__`↔registry drift guard baseline shrank from
  44 → 25, locking the gain in. Pinned by `tests/test_registry_drift_repair.py`
  (discoverability + correctness-by-construction on a known-truth DGP).
- **Full coefficient table on limited-dependent-variable models.** `sp.tobit`
  and `sp.heckman` now expose the *entire* coefficient vector through
  `.params` / `.std_errors` / `.tvalues` / `.pvalues` (plus a new
  `.coef_table()` helper), matching the Stata (`tobit`, `heckman`) and R
  (`AER::tobit`, `sampleSelection::heckit`) convention. Previously `.params`
  surfaced only the headline first-regressor coefficient, so an agent calling
  `sp.tobit(...).params` got a one-element Series instead of `const`, every
  regressor, and `sigma`. The headline `.estimate` / `.se` / `.ci` are
  unchanged and `isinstance(result, CausalResult)` still holds, so no consumer
  breaks; as a bonus `sp.etable(tobit_fit)` now renders the full regression
  table. Implemented via a small `LimitedDepResult` subclass — the shared
  `CausalResult` base is untouched. Pinned in
  `tests/test_limited_dep_lane.py`.
- **Standard result accessors on parametric survival models.** `sp.aft`
  (`AFTResult`) and `sp.cox_frailty` (`FrailtyResult`) now expose
  `.params` / `.std_errors` / `.tvalues` / `.pvalues` like every other
  estimator. Previously the coefficient vector was reachable only through
  raw `.beta` / `.se` arrays and `AFTResult.params` was literally `None`, so
  an agent calling the documented accessor got nothing. `sp.aft` additionally
  now reports the scale standard error (`log(sigma)` SE was computed in the
  Hessian and then discarded by a `[:k]` slice). The two independent Weibull
  AFT implementations (`sp.aft` and `sp.survreg`) are cross-validated to agree
  on coefficients (~1e-5) and standard errors (~1e-7). Pinned in
  `tests/test_limited_dep_lane.py`.
- **`sp.garch` now reports parameter standard errors / inference.** The
  GARCH(p,q) fit previously returned only point estimates (`omega`, `alpha`,
  `beta`, `mu`) with no standard errors, so the volatility parameters could
  not be tested or given confidence intervals. SEs now come from the inverse
  observed information (numerical Hessian at the MLE) and are exposed through
  the standard `.params` / `.std_errors` / `.tvalues` / `.pvalues` accessors,
  with a coef/std-err/z/p table in `.summary()`. Monte-Carlo coverage tracks
  the empirical SD of α̂/β̂ to ratio ~0.96. Pinned in
  `tests/test_garch_inference.py`.
- **`sp.bvar` now reports posterior uncertainty (SD + credible intervals).**
  The Bayesian VAR previously returned only the posterior-mean coefficient
  matrix, so credible intervals could not be formed. The marginal posterior
  standard deviation is now available in closed form from the matrix-normal
  posterior `B ~ MN(B_post, (X'X + V⁻¹)⁻¹, Σ)` as
  `sd[i,k] = sqrt([(X'X+V⁻¹)⁻¹]ᵢᵢ · Σ_kk)`, exposed via the new `coef_sd`
  attribute and a `credible_interval(level)` method, with a posterior-SD block
  added to `.summary()`. Validated against statsmodels VAR: in the loose-prior
  limit the posterior SD equals the OLS VAR standard error (to the n vs n-k df
  factor), and the Minnesota prior correctly shrinks it. Pinned in
  `tests/test_bvar_posterior_se.py`.
- **`gwr` (geographically weighted regression) now reports local standard
  errors and t-values.** GWR previously returned only the local coefficient
  matrix `params (n, k)`, so a user could not tell *where* a covariate's
  effect is significant — the core purpose of GWR. Local SEs now follow
  Fotheringham, Brunsdon & Charlton (2002, §2.4),
  `Var(β̂ᵢ) = σ̂² CᵢCᵢ'` with `Cᵢ = (X'WᵢX)⁻¹X'Wᵢ` and
  `σ̂² = RSS / (n − 2 tr(S) + tr(S'S))`, exposed via the new `se` and `tvals`
  attributes. Verified against the global limit: at a very large bandwidth the
  per-location SE collapses to the OLS standard error (to ~0.5%, identical
  across locations). Pinned in `tests/spatial/test_gwr_local_se.py`.
  (Multiscale GWR / `mgwr` local SEs remain a known follow-up — their
  back-fitting variance is materially more involved.)

### Fixed

- **Fail-loud input validation on the proximal / negative-control / mediation
  / network-interference estimators.** `sp.double_negative_control`,
  `sp.proximal_regression`, `sp.negative_control_outcome`,
  `sp.negative_control_exposure`, `sp.four_way_decomposition`,
  `sp.network_hte` and `sp.inward_outward_spillover` previously sliced
  `data[[cols]]` (often without even `dropna`) with no checks, so a typo'd
  column name surfaced as a cryptic `KeyError`/`LinAlgError` and an empty,
  all-NaN, or too-small frame produced a **silent NaN / silent garbage
  estimate** from the downstream linear algebra (the "吞异常返回 NaN"
  anti-pattern — `inward_outward_spillover` happily decomposed 3 data points).
  They now validate up front through a new package-level
  `statspai._input_validation` helper (one shared primitive, not
  re-implemented per estimator): a non-DataFrame raises `sp.StatsPAIError`, a
  missing column raises `DataInsufficient` (both a `StatsPAIError` and a
  `ValueError`, message keeps the conventional "Missing columns" phrasing), and
  too few complete rows to identify the model raises `DataInsufficient`, each
  with an agent-native `recovery_hint`. The happy-path numerics are
  byte-identical (the four-way decomposition still satisfies
  TE = CDE + INT_ref + INT_med + PIE exactly; the network estimators recover
  their direct/spillover/inward/outward coefficients on a known DGP). Also
  fixes a latent `NameError` in `sp.proximal_regression`'s documented
  graceful-degradation path: when the treatment-bridge logistic fit failed
  (e.g. sklearn absent), the fallback branch then referenced an unbound `lr`,
  so the "degrade gracefully" promise itself crashed. The two network
  estimators additionally gained agent-native cards (assumptions / failure
  modes / alternatives). Pinned by `tests/test_input_validation.py`,
  `tests/test_proximal_input_validation.py` and
  `tests/test_interference_orthogonal.py`.
- **Fail-loud input validation on the DiD / time-series / shift-share /
  sensitivity estimators.** A second hardening sweep closes the same
  silent-failure class in five more estimators that an investigation surfaced:
  `sp.its` returned a **silent 0.0 / NaN** segmented-regression on an empty,
  single-row, or all-NaN series (and a bare `KeyError` on a typo'd column);
  `sp.overlap_weighted_did` returned a **silent `estimate=0.0`** on an all-NaN
  outcome (the overlap-weighted cell mean NaN-sums to zero);
  `sp.dl_propensity_score` returned a **silent degenerate `[0.02]`** score on a
  single row and a bare `KeyError` on a typo'd column; `sp.shift_share_political`
  returned a **silent `estimate=nan`** on an all-NaN outcome (its panel sibling
  already guarded this); and `sp.rosenbaum_bounds` / `sp.rosenbaum_gamma` leaked
  a cryptic pandas `KeyError` on a missing column in the DataFrame interface.
  All now raise a `StatsPAIError` subclass (`DataInsufficient` /
  `NumericalInstability`, both also `ValueError`) with an agent-native
  `recovery_hint`, while happy-path numerics are unchanged. `sp.its` also gained
  a min-observations floor (n > parameters) and an intervention-in-range check;
  `sp.dl_propensity_score` validates ≥2 rows spanning both treatment classes
  without dropping rows (the returned score stays row-aligned). Pinned by
  `tests/test_estimator_input_hardening.py`. Separately, the long-standing
  `test_dl_propensity_score_returns_valid_probs` pinned a non-converging
  `MLPClassifier`'s output to `atol=1e-12` (sklearn/BLAS-version fragile); it now
  asserts environment-robust invariants instead. The same sweep also clarifies
  four estimators that already failed loudly but with a *confusing* message on
  an all-NaN outcome: `sp.ltmle` raised a raw sklearn `Input y contains NaN`,
  and `sp.bcf_factor_exposure` / `sp.bcf_longitudinal` / `sp.bcf_ordinal` raised
  `Treatment must be binary (0/1)` (the real cause — an empty frame after the
  inner BCF's `dropna` — was mislabelled). All four now raise an error naming
  the *outcome* column before any model fit. Likewise `sp.synth_experimental_design`
  reported a misleading `k must be in [1, -1]` on an empty / fully-missing /
  single-unit panel; it now raises `DataInsufficient` naming the collapsed
  panel (and an unbalanced panel raises a clear balance hint), via the shared
  `require_columns` guard.
- **⚠️ Correctness — `sp.regress(..., weights=)` was silently ignored.** The
  OLS estimator accepted a `weights=` argument via `**kwargs` and then dropped
  it, returning the *unweighted* OLS fit with no warning. It now fits weighted
  least squares with Stata `aweight` semantics: point estimates, classical /
  HC1-robust / clustered standard errors, and R² match `regress y x [aw=w]`
  to machine precision (verified live against **Stata 18 MP**). Invalid weights
  (non-positive, NaN/inf, wrong length, missing column) now raise `ValueError`
  instead of being ignored. This is the same fail-silently bug class fixed for
  `sp.feols` in 1.18.0, now closed in the `sp.regress` / OLS path. The
  unweighted path is byte-identical. See
  [`MIGRATION.md`](MIGRATION.md#regress-weights).
- **`sp.iv(..., robust=)` rejected standard spellings.** `sp.iv(robust='HC1')`
  (uppercase) raised `ValueError: Unknown robust type: HC1` even though
  `sp.regress` accepts it — the IV path did not normalise the SE-type string.
  The IV estimators now accept case-insensitive `'hc0'`–`'hc3'`, the
  Stata-style aliases `True` / `'robust'` (≡ HC1) and `'white'` (≡ HC0), and
  raise a clear message for anything else. Classical and robust IV standard
  errors match `ivregress 2sls, small` / `ivregress 2sls, robust small` (the
  finite-sample *t* convention StatsPAI uses) to machine precision. Pinned in
  `tests/reference_parity/test_regress_weights_iv_robust_parity.py`.
- **⚠️ Correctness — `sp.biprobit` always reported zero error correlation.**
  The bivariate-probit MLE estimated `rho` via a per-observation
  `scipy.stats.multivariate_normal.cdf` loop whose ~1e-8 precision floor
  corrupted BFGS's finite-difference gradient, so BFGS died at the starting
  point on its first step (status 2) and `rho` stayed pinned at its initial
  value of **0** for *every* dataset — the model silently claimed the two
  equations' errors were uncorrelated no matter how correlated they actually
  were (a researcher would wrongly conclude "no cross-equation selection").
  The bivariate-normal CDF is now a smooth, fully vectorised Drezner–Wesolowsky
  (1990) Gauss–Legendre quadrature (matches SciPy to ~8e-6, ~280× faster), so
  the optimiser recovers `rho`: on simulated data with true ρ≈0.45 it returns
  ρ̂≈0.47 (p<0.001), and on independent errors ρ̂≈0 (p≈0.97). `biprobit` now
  also reports `model_info['converged']`. Marginal coefficients match separate
  univariate probits and reproduce the pre-regression reference values
  (`tests/test_v06_round2.py::TestSelectionModels::test_biprobit`, previously
  failing, now green). Behavioural regression test added in
  `tests/test_limited_dep_lane.py`. Reference: Drezner & Wesolowsky (1990),
  *J. Stat. Comput. Simul.* 35(1–2), 101–107 (verified via Taylor & Francis +
  Semantic Scholar).
- **⚠️ Correctness — `sp.sar` / `sp.sem` spatial-parameter standard errors
  were overstated ~40-50%.** The maximum-likelihood SE for the spatial-lag
  coefficient `rho` (SAR, also `sp.sdm`) and the spatial-error coefficient
  `lambda` (SEM) dropped the `tr(G'G)` term from the information matrix
  (`G = W(I-ρW)^{-1}`) and, for SAR, used a projection of `X` in place of
  `(G Xβ)`. The reported SE came out ~1.5× (SAR) / ~1.4× (SEM) too large, so
  spatial dependence was systematically *under*-detected (t-statistics ~33%
  too small). The information matrix is now the textbook Ord (1975) / Anselin
  (1988) form. Verified two ways: (i) the reported SE now equals the inverse
  numerical Hessian of the exact log-likelihood (`slogdet` Jacobian) to <5%,
  and (ii) Monte-Carlo coverage matches the empirical SD of ρ̂/λ̂ (ratio
  0.89/0.94, vs 1.51/1.39 before). Point estimates are unchanged. Pinned in
  `tests/spatial/test_ml_se_information.py`.
- **Reliability — spurious `converged: False` on the BFGS-based MLEs.**
  `sp.tobit`, `sp.heckman`'s selection probit, `sp.truncreg`, `sp.zip` /
  `sp.zinb`, `sp.betareg` and `sp.biprobit` reported
  `model_info['converged'] = False` at perfectly good optima because SciPy's
  BFGS returns `success=False` with status 2 ("Desired error not necessarily
  achieved due to precision loss") on flat log-likelihoods — a line-search
  artefact, not a real failure (Nelder-Mead reaches the identical objective
  and coefficients to ~1e-13). A single shared helper
  (`statspai.regression._optim_helpers.robust_convergence`) now derives the
  flag from the gradient norm at the optimum (`success` **or** ‖∇‖ < 1e-3 with
  a finite objective) and returns `model_info['gradient_norm']` for
  transparency. It only ever relaxes a false negative — a genuinely
  non-converged run keeps a large gradient and still reports `False`. Point
  estimates and standard errors are byte-identical; only the boolean flag
  changes. Pinned in `tests/test_limited_dep_lane.py`.

## [1.18.0] — 2026-06-15

### Added

- **`sp.psmatch2` Abadie-Imbens (2006) heteroskedasticity-robust SE** (Stata
  `psmatch2 , ai(J)`). `sp.psmatch2(..., ai=J)` (or `se='abadie_imbens'`)
  estimates the within-cell outcome noise `σ²(X)` from each unit's `J`
  nearest *same-arm* neighbours (psmatch2's `_self_y`) and forms
  `seatt = sqrt(Σ_i (J/(J+1))(Y_i−Ȳ_self_i)²·(D_i−(1−D_i)w_i)²)/N1` —
  reproducing Stata's `r(seatt)` **digit for digit** (machine precision, since
  the within-arm match and `_weight` are both discrete). `sp.match` gains
  `se_method='abadie_imbens'` + `ai_matches=J`. Pinned in
  `tests/reference_parity/test_psmatch2_parity.py` against Stata 18 `ai(1)` /
  `ai(2)`. (Note: psmatch2's `llr` local-linear matching is intentionally
  **not** ported — Stata routes its default through the `lpoly` command, whose
  bandwidth/boundary handling is not bit-reproducible; a faithful "align with
  Stata" port is infeasible, so it is omitted rather than shipped misaligned.)
- **`sp.psmatch2` kernel & radius matching + Stata's digit-exact analytic SE.**
  Building on the matched-frame work below, `sp.psmatch2` now accepts
  `method={'neighbor','kernel','radius'}` (kernel uses `kernel=` /
  `bwidth=`; radius is a uniform kernel with bandwidth `caliper`), reproducing
  Stata `psmatch2 , kernel` / `, radius`. Kernel/radius assemble the matched
  frame with per-treated propensity-kernel weights (`_weight_j = Σ_i K_ij/Σ_k
  K_ik`, summing to the on-support treated count) and the matched-control mean
  `_y`. A new `se={'psmatch2','ai'}` selects the standard error: `'psmatch2'`
  (now the `sp.psmatch2` default) reproduces Stata's analytic ATT SE
  `sqrt(var1/N1 + var0·Σw²/N1²)` (var0 over *used* controls) **digit for
  digit** — nearest-neighbour SE and radius ATT+SE match Stata 18 to machine
  precision (≤1e-12); the smooth Epanechnikov kernel ATT matches to ~1e-8,
  bounded only by the independent logit propensity-score estimate (the
  matching algorithm is exact given the same score). `sp.match` gains the
  same `method='kernel'/'radius'`, `kernel`, `bwidth`, `se_method` options;
  its nearest-neighbour SE default is unchanged (`se_method='auto'` → AI),
  so existing results are untouched. Algorithm and SE formula were
  reverse-engineered from `psmatch2.ado` and pinned against Stata 18 in
  `tests/reference_parity/test_psmatch2_parity.py` (with kernel/radius
  fixtures and scalars).
- **`sp.psmatch2` — Stata `psmatch2`-faithful propensity-score matching with
  a full post-matching toolkit (`src/statspai/matching/psmatch2.py`).** Closes
  the gap that `sp.match(method='psm')` produced only an ATT, not the
  per-observation matched-sample variables Stata writes back. `sp.psmatch2`
  returns a `PSMatch2Result` whose `.matched_data` carries the psmatch2
  columns — `_id`, `_treated`, `_pscore`, `_support`, `_weight`,
  `_n1 … _nk`, `_nn`, `_pdif`, `_y` — and exposes the three operations that
  previously required Stata: `.balance()` (post-matching balance, the
  `pstest` analogue: `smd_raw` before vs `_weight`-weighted `smd_weighted`
  after), `.psplot()` (matched-sample propensity-score density with the
  controls reweighted by `_weight`), and `.psm_did()` (frequency-weighted
  PSM-DID — merges `_weight`/`_support` onto a panel by `id`, keeps the
  matched sample, and fits the weighted `y ~ treat*post` DiD via `sp.feols`,
  dropping FE-absorbed main effects automatically). `outcome=` is optional
  (Stata-faithful: the matched frame is produced for PSM-DID even without a
  baseline outcome). Verified **row-for-row against Stata 18 `psmatch2`** —
  ATT exact to 6 digits, and `_pscore` / `_weight` / `_nn` / `_pdif` / `_n1`
  (neighbour identity) / `_y` all match (`tests/reference_parity/
  test_psmatch2_parity.py`, `tests/test_psmatch2.py`, +40 tests). New guide:
  [`docs/guides/psm_did.md`](docs/guides/psm_did.md). Refs verified via
  Crossref (Heckman-Ichimura-Todd 1997, DOI 10.2307/2971733) and the SSC
  archive (Leuven & Sianesi 2003, S432001).
- **`sp.match` / `sp.psm` now attach a `result.matched_data` frame** for the
  nearest-neighbour path (the same psmatch2 columns above), plus a
  `common_support={'none','minmax'}` option (default `'none'`, i.e. the
  historical behaviour — the matched frame is pure additive bookkeeping over
  the assignment already used for the point estimate, so the ATT/SE are
  unchanged). `common_support='minmax'` mirrors Stata's `common`: off-support
  treated are dropped before matching and the ATT is taken over the
  on-support treated.

- **Analytic-anchor reference parity for 10 previously smoke-only headline
  estimator families (`tests/reference_parity/`, +97 tests).** First real
  numerical guarantees for `proximal` / `fortified_pci` / `bidirectional_pci`,
  `principal_strat` / `survivor_average_causal_effect`, `qte` / `qdid`,
  `distributional_te` / `stochastic_dominance`, `interference` / `spillover` /
  `network_exposure`, causal discovery (`pc_algorithm` / `lingam` / `notears`),
  `matrix_completion` / `mc_panel`, `gmm` / `xtabond`, `dose_response` /
  `continuous_did`, and `bunching` / `general_bunching`. Every file is
  hermetic (numpy/scipy only, no R, fixed seeds, ~40s total) and uses
  non-tautological anchors — known-DGP recovery within a documented σ band,
  closed-form collapses (e.g. the QTE location-shift identity to 4e-16),
  naive-bias contrasts (the naive estimator is provably off, the method
  recovers truth), and structure-recovery metrics (skeleton precision =
  recall = 1). Each was adversarially verified to FAIL under a 20% injected
  estimate bias; DGPs and tolerance rationale are logged in
  `tests/reference_parity/REFERENCES.md`.

### Changed

- **R-parity SE tolerances tightened for 9 mid-tier modules** (`42_nbreg`,
  `43_heckman`, `28_frontier`, `44_mlogit`, `41_tobit`, `14_ols_cluster`,
  `24_coxph`, `46_clogit`, `49_oprobit`) to ≈3× the worst observed gap
  across both reference sides (floored at the `1e-6` machine tier) — none
  loosened; `docs/dev/r_parity_tolerances.md` carries the before/after
  table and the harness contract test stays green.
- **Hardened ~555 weak assertions** across 12 core-estimator test suites
  (DiD / RD / DML / synth / panel / matching / proximal / principal-strat /
  GMM / bounds): `is-not-None` / `se>0`-only checks replaced with
  recovery-band pins, sign correctness, CI-bracketing, and closed-form
  structural identities. Surfaced (documented, not regressions):
  `sp.continuous_did` returns NaN SE/CI on a 2-period DGP (self-labeled
  dose-bin heuristic); FE `fitted_values` exclude entity effects by design.

### Fixed

- **⚠️ Correctness — `sp.feols` now honours `weights=` when no fixed effects
  are absorbed.** The intercept-only fallback path (`feols` with regressors but
  no FE terms) accepted the `weights=` argument but silently discarded it and
  returned an *unweighted* OLS fit — coefficients, standard errors, and R² were
  all affected. The fallback now solves the weighted (WLS) normal equations,
  validates the weight vector (finite, non-negative, positive total mass),
  aligns an array-valued `weights=` to the post-`dropna` sample, and threads the
  weights through the clustered sandwich. FE-absorbed paths were already
  weighted and are numerically unchanged; no-`weights=` calls are unchanged.
  Guarded by new cases in `tests/test_panel_cov_feols.py`. See
  [MIGRATION.md](MIGRATION.md#feols-nofe-weights).

- Typo `boostrap` → `bootstrap` in a `structural/production/wooldridge.py`
  comment; added `.codespellrc` (curated econ-vocabulary ignore list) so
  `codespell` runs clean as a dev convenience. `pyproject.toml` advertises
  `Typing :: Typed` (py.typed ships) and points `Documentation` at the
  MkDocs site.

- **Docstring `Examples` coverage campaign — 35.9% → 92.3% of the public
  surface, every example verified runnable.** Added verified `Examples`
  blocks to the registered functions that lacked them (coverage 370/1031
  → 952/1031) and repaired the package's long-standing *illustrative*
  examples on headline functions (`sp.did(df, ...)` with a placeholder
  `df`, bare `did(...)`, `sp.synth.california_prop99()` submodule-access
  bugs) so they actually execute. A new `scripts/check_example_execution.py`
  extracts every example's `>>>` source (dropping `# doctest: +SKIP`
  lines reserved for heavy-optional / external-data blocks) and runs it:
  the package now reports **939 examples run, 0 failing**. Two CI gates in
  `parity-guards.yml` lock both dimensions — `examples_coverage --check`
  (presence ratchet, budget tightened 561 → 79) and
  `check_example_execution --max-failures 0` (runnability ratchet) — so a
  new public function can no longer ship without a *running* example.
- **Reference-parity anchors for the doubly-robust trio — `sp.tmle`,
  `sp.ipw`, `sp.g_computation` (27 anchors, 44 tests,
  `tests/reference_parity/test_{tmle,ipw,gformula}_parity.py`).** The three
  estimators previously had smoke tests only. Anchors: saturated
  closed-form collapses (hand-computed stratified cells; TMLE = AIPW =
  g-formula at the nonparametric MLE), frozen base-R fixtures
  (`stats::glm`/`lm` only — generation scripts + CSVs committed; the TMLE
  logistic fluctuation is hand-coded in base R and psi/SE pinned at 1e-9),
  the TMLE EIF-mean-zero targeting property, cross-estimator combined-SE
  parity, confounded-DGP recovery with an explicit naive-bias contrast, and
  SE-vs-Monte-Carlo-SD sanity bands. Every file passed an adversarial
  verification round including a 2% injected-bias mutation check.
- **Three new user guides + nav repair.** `docs/guides/panel_data.md`
  (FE/RE/Hausman/HDFE/dynamic GMM — previously the largest no-guide gap),
  `docs/guides/sensitivity_analysis.md` (a design→tool decision tree across
  sensemakr / Oster / E-value / Rosenbaum / honest-DiD / weak-IV /
  DML-sensitivity / Manski-Lee bounds), and `docs/guides/mediation.md`
  (ACME/ADE, interventional effects, front-door, Gelbach). Every code block
  executed against the live package before landing; all citations grep-
  verified against `paper.bib`. Four previously orphaned guides (synth,
  migration-from-r, mixtape_ch09_did, agent_native_workflow) are now wired
  into the mkdocs nav, and `mkdocs build --strict` is green.
- **R/Stata parity tolerance registry
  (`docs/dev/r_parity_tolerances.md`, linked next to the JSS dossier).**
  Every per-module tolerance in `tests/r_parity/compare.py` now carries an
  A (mechanistic) / B (empirical, with recomputed observed gaps) / C
  (honestly unjustified) grade, and 13 stale loose SE budgets were
  tightened — none loosened — including the former `11_psm rel_se = 5.0`
  and four `rel_se = 1.0` entries, all re-verified against the committed
  golden artifacts (36/36 harness contract tests).
- **Verified `Examples` blocks for 27 high-frequency public functions**
  (mixed, absorb_ols, demean, aft, local_projections, granger_causality,
  etable, sun_abraham, etwfe, ivreg, jive, rdplot, rd_honest, marginsplot,
  synth, dml, xlearner, propensity_score, front_door, mediation_decompose,
  psm, cbps, overlap_weights, love_plot, balance_diagnostics,
  rif_decomposition, dfl_decompose). Each example executed before landing
  and independently re-run by a verifier agent (27/27 pass). New
  `scripts/examples_coverage.py` audits Examples coverage per category
  (currently 370/1031 registered functions, 35.9%) for future ratcheting.
- **Performance-regression ratchet (`scripts/benchmark_ratchet.py` +
  scheduled `benchmarks.yml` workflow).** The released PyPI wheel and the
  source tree are benchmarked back-to-back on the same runner twice a
  month; any `sp_*` timing >1.5× slower than the release fails the
  scheduled run (PRs are never blocked). A committed
  `benchmarks/baseline.json` enables the same-machine local check.
- **Community/supply-chain files**: `SECURITY.md` (private vulnerability
  reporting policy with scope notes) and `.github/dependabot.yml`
  (weekly pip + GitHub-Actions, monthly cargo update PRs).
- **Track A cross-language parity expansion: 56 → 64 modules
  (`tests/r_parity/` 57–64, all three sides py + R + Stata).** New modules:
  binary logit (`sp.logit`), Poisson ML (`sp.poisson`), LIML k-class
  (`sp.liml` vs `ivmodel::LIML` / `ivregress liml, small`), SUR one-step FGLS
  (`sp.sureg` vs `systemfit` noDfCor / `sureg`), beta regression
  (`sp.betareg` vs `betareg` / Stata `betareg`), truncated regression
  (`sp.truncreg` vs `truncreg` method=NR / `truncreg, ll(0)`), ZIP and ZINB
  (`sp.zip_model` / `sp.zinb` vs `pscl::zeroinfl` / `zip` / `zinb`). Seven of
  the eight land in the machine tier (rel ≤ 1e-6; LIML and SUR at ~1e-15
  including SEs); ZINB registers an honest 1e-5 budget because its
  likelihood is flat near the optimum. Strictness tiers are now
  57 machine / 5 iterative / 1 moderate / 1 methodological-T4, with 61 of 64
  modules carrying a Stata reference. The eight symbols are promoted to the
  `certified` validation tier (registry: 61 certified / 24 validated).
  Documented convention findings: Stata `ivregress liml` needs `small` for
  the shared RSS/(n−k) divisor; `systemfit`'s default `geomean` residual
  covariance corresponds to Stata `sureg, dfk`, not the `sureg` default;
  R `betareg` reports expected-information SEs vs observed-information in
  sp/Stata (≤0.7% documented gap).
- **Parity reproducibility harness hardening.** `verify_reproduce.py` now
  also re-runs the non-CSV modules (10/21/23), so all 64 R golden artifacts
  are gate-verified (64/64 reproduce, 0 drift; Stata leg 61/61);
  `_gen_renv_lock.R` adds the previously missing canonical reference
  packages (EValue, sfaR, ddecompose, dineq, censReg, sampleSelection,
  nnet, clubSandwich, DRDID, sensemakr + the new ivmodel, systemfit,
  betareg, truncreg, pscl), regenerating `renv.lock` at 285 pinned packages.
- **R/Stata reference-parity expansion — panel, count/quantile, and SDID
  estimators now carry frozen cross-package parity (19 new tests,
  `tests/reference_parity/` 124 → 143).** Each ships a deterministic data
  fixture, a `_generate_*.R` script, and a frozen `*_R.json`:
  - `sp.panel` vs R `plm` — within (FE), Swamy-Arora RE, and between, with
    classical and cluster-robust SEs (matches `plm::vcovHC` HC1/group);
    coefficients and classical SEs agree to ~1e-5. Closes the gap where
    `panel`, a core ≥95% estimator, had *no* reference parity.
  - `sp.poisson` / `sp.nbreg` / `sp.qreg` / `sp.tobit` / `sp.zip_model` /
    `sp.zinb` vs `glm` / `MASS::glm.nb` / `quantreg::rq` / `AER::tobit` /
    `pscl::zeroinfl` — exact coefficient parity (Poisson/NB/Tobit also exact
    model SEs and Tobit σ; NB `alpha == 1/theta`).
  - `sp.sdid` vs the authors' `synthdid` R package (Arkhangelsky et al. 2021)
    on `sp.california_prop99()` — point estimate agrees to ~1e-6
    (−17.8985 packs/capita).
- **NIST StRD one-way ANOVA certification for `sp.regress`
  (`tests/numerical_accuracy/test_nist_strd_anova.py`).** All 11 NIST ANOVA
  datasets (SiRstv, SmLs01–09, AtmWtAg) bundled verbatim; a one-way ANOVA is
  OLS of `y ~ C(group)`, so these certify the F-statistic / R² / sum-of-squares
  numerical accuracy. Backed by the new mean-centred OLS fit (see ⚠️
  Correctness below), `sp.regress` reproduces the certified F to machine
  precision through the average-difficulty family (incl. n=18009); the three
  highest-difficulty designs (`SmLs07/08/09`, 9 constant leading digits) reach
  the irreducible IEEE-754 float64 floor (~7e-5) of their data and are checked
  at a documented 1e-3 tolerance. All reference R DOIs verified via Crossref.
- **Tier D analytic special-case test campaign — 24 reference-less estimators
  now carry known-truth tests (77 tests across 9 `tests/test_tierD_*.py`
  files).** Estimators that had *no* numerical-assertion test (only smoke calls
  or none) are now anchored to closed-form identities or known-DGP recovery:
  partial-identification bounds (`horowitz_manski`, `iv_bounds`, `oster_delta`,
  `trimming`), power/MDE (`power`, `mde`, `power_cluster_rct`, `power_iv`),
  `frontdoor`, `ps_balance`, `moran_local`, `effective_f_test`, `stepwise`,
  `feglm`, `mi_estimate`, boundary/multi-score RD aliases (`boundary_rd`,
  `geographic_rd`, `multi_score_rd`), production functions (`levpet`, `opreg`),
  `peer_effects`, `notch`, `model_averaging_dml`, `test_calibration`. A
  read-only classifier `scripts/tierd_classify.py` grades every registered
  function's test evidence (reference / anchored / weak / smoke / untested) and
  emits the worklist; the zero-guard P1 floor went from 25 → 0 (the lone
  remainder, `blp`, fixed in this release — see Fixed). Purely
  additive: no estimator numerics changed.
- **Tier B executable replication notebooks
  (`Paper-JSS/replication/notebooks/*.ipynb`).** Self-contained
  Jupyter notebooks reproduce the headline numbers of Card (1995), ADH (2010)
  Prop 99, LaLonde/DW NSW, Lee (2008) RD, and Graddy (2006) from the bundled
  *real* data; each ends with a drift-guard cell, and
  `tests/test_replication_notebooks.py` executes them headless in CI (fails on
  drift). Generated by `scripts/build_replication_notebooks.py`; new
  `notebooks` extra (`pip install -e ".[notebooks]"`).
- **Agent-native surface hardening (agent-infra work line).** A batch of
  additive metadata / result-object / MCP-runtime improvements that make the
  schema an agent reads (`sp.describe_function`) trustworthy. No estimator
  numerics, signatures, or `paper.md`/`paper.bib` touched (see
  `docs/dev/agent_infra_campaign.md` for the JOSS-isolation contract):
  - **`EconometricResults.cite(format=...)` — citation parity with
    `CausalResult`.** Previously `sp.bib_for(<regression result>)` raised
    because only `CausalResult` carried `.cite()`. The bib key resolves exactly
    from `model_info['citation_key'] → model_type → method` against the shared
    `CausalResult._CITATIONS`; OLS/logit/probit/poisson return a placeholder
    rather than a fuzzy/fabricated match (§10 zero-hallucination). Pure addition
    (`tests/test_econometric_results_cite.py`).
  - **Opt-in TTL + reason-aware misses for the MCP result cache.** New
    `STATSPAI_MCP_RESULT_CACHE_TTL` env var (default unset = no expiry, prior
    behaviour byte-identical); expired handles are swept lazily on access /
    eagerly on insert, and a bounded ledger records why a handle left
    (`ttl`/`lru`/`explicit`) so the "result expired" hint an agent receives is
    tailored to the cause. New `evict()` / `purge_expired()` / `stats()`
    (`tests/test_result_cache_ttl.py`).
  - **Docstring parser now reads type-less NumPy param headers** (`name` then an
    indented description, no `: type`), recovering **65 parameter descriptions
    across 41 functions** (`feols`, `causal_forest`, `match`, …) that
    `describe_function` was silently dropping. No new dependency; a column-0
    barename branch with the false-positive boundary pinned by
    `tests/test_docstring_param_parser.py`.
  - **Three forward CI guards** (test-only, no runtime change): every MCP
    workflow tool has a real dispatch branch
    (`tests/test_workflow_tool_dispatch_contract.py`); the 42-edge
    `FunctionSpec.inherits_from` graph stays free of dangling parents /
    self-references / cycles (`tests/test_inherits_from_integrity.py`); and a
    concrete source annotation never collapses to `Any` in auto-generated specs
    (`tests/test_auto_spec_type_resolution.py`).

### Changed

- **Repositioned StatsPAI as a "library" (not "platform")** across the README,
  README_CN, `pyproject.toml` description, and the GitHub repo description.
  Documentation/positioning wording only — no API, behaviour, or estimator
  numerics changed.

### Fixed

- **Silent numerical fallbacks now fail loudly (CLAUDE.md §3.7 sweep, 15
  sites).** Optimizer/solver failures that previously degraded in silence
  now emit `ConvergenceWarning`/`StatsPAIWarning` and record a diagnostics
  flag, with the fallback documented in the docstring — numerical outputs
  are bit-identical: `sp.ebalance` (failed dual → uniform weights, now
  `model_info['weights_fallback']`), `sp.genmatch` (degenerate KS → p=1.0,
  now counted in `result.detail['ks_test_failures']`), bootstrap-replicate
  failures in `sp.horowitz_manski`/`sp.iv_bounds`/`sp.oster_delta`/
  `sp.selection_bounds` (`model_info['n_boot_failed']`), propensity
  fallbacks in `sp.ope` IPS/SNIPS/DR (`diagnostics['propensity_fallback']`),
  bridge-function fallbacks in proximal `bidirectional`/`fortified`/
  `pci_regression`/`proxy_selector`, LTMLE targeting-step failures
  (`detail['targeting_failures']`), and balance/attrition diagnostics in
  `experimental`. Orchestration-layer stragglers in
  `workflow.causal_workflow`, `smart.benchmark`, and
  `smart.identification` now route through `record_degradation` (bare-
  swallow debt ratchet 8 → 7 in `tests/test_no_silent_degradation.py`).
- **⚠️ Correctness — `sp.unified_sensitivity` dashboard components revived
  and corrected.** (1) The Sensemakr component was 100% dead code: it called
  `sensemakr(result, treatment=...)` against the real signature
  `sensemakr(data, y, treat, controls, ...)`, so it always raised `TypeError`
  that was swallowed into `dash.notes`. The dashboard now accepts optional
  `data=`, `y=`, `treat=`, `controls=` (forwarded by `result.sensitivity()`)
  and runs the real Cinelli-Hazlett RV (`rv_q1` = `rv_q`, plus `rv_qa`);
  without them it is skipped with an actionable note pointing to
  `sp.sensemakr(data, y, treat, controls)`. (2) The Rosenbaum component was
  equally dead: it imported a nonexistent module
  (`diagnostics.rosenbaum_bounds` instead of `diagnostics.rosenbaum`) and
  passed the result object where outcome arrays were expected. It now coerces
  `result.matched_pairs` ((treated, control) arrays, (n, 2) array, or
  DataFrame/dict with treated/control entries) and calls
  `sp.rosenbaum_bounds(treated, control, alternative="two-sided")`. (3) The
  Oster component routed `r2_treated`/`r2_controlled` into
  `oster_bounds(r2_short=, r2_long=)` swapped relative to its documented
  "short and long regression respectively" contract, and reported the *input*
  proportionality `delta` (always 1.0) instead of the breakdown
  `delta_for_zero`; both fixed, so `dash.oster["delta"]` is now the actual
  Oster δ*. Locked by parity tests against direct `sp.sensemakr` /
  `sp.rosenbaum_bounds` / `sp.oster_bounds` calls in
  `tests/test_unified_sensitivity.py`.
- **RIF/UQR parity now mirrors `dineq::rif`'s exact density convention.**
  The `quantile_convention="dineq"` path now ports R `stats::density`'s
  binned Gaussian estimator at the quantile, rather than a direct Gaussian
  kernel average with the same `bw.nrd0` bandwidth. The Stata/Mata bridge for
  `32_rif` implements the same binned interpolation, so Python, R, and Stata
  agree at machine precision; Track A strictness moves to **57 machine-level /
  5 iterative / 1 moderate / 1 methodological** modules after the multinomial
  and ordered logit R references, the LMM REML reference, the AGHQ GLMM
  reference, the cross-sectional SFA default optimizer, the `xtabond`
  `plm::pgmm` reference, and the newer GLM / IV / system / limited-dependent
  variable rows were pinned to
  tight optimizer/reporting tolerances.

- **⚠️ Correctness fix (pandas ≥ 3.0): `sp.horowitz_manski` bounds silently
  collapsed to `0.0`/`0.0` when a covariate stratum mapped to NaN.** The
  internal `_create_strata` helper discretises continuous covariates with
  `pd.qcut`, then cast the bin labels with `.astype(str)`. On a degenerate
  covariate (e.g. a constant column, where `qcut(..., duplicates='drop')`
  yields all-NaN) pandas < 3.0 stringified NaN to `"nan"` — accidentally
  forming one valid stratum — but pandas ≥ 3.0 preserves it as `<NA>`, so every
  per-stratum mask matched nothing and the bounds summed to zero with no error.
  NaN strata are now bucketed into an explicit `-1` sentinel stratum, recovering
  the correct closed-form bounds. The analytic guard
  (`tests/test_tierD_bounds_analytic.py::TestHorowitzManskiAnalytic::test_single_stratum_matches_closed_form`)
  pins the single-stratum case to the closed form on both pandas 2.x and 3.x.
  Numerics are unchanged on pandas < 3.0.

- **⚠️ Functionality fix: `sp.blp` was non-functional on every estimation
  path.** The GMM objective called `_gmm_objective(..., maxiter=1000)` but the
  parameter is named `maxiter_inner`, so every `sp.blp` call raised
  `TypeError: _gmm_objective() got an unexpected keyword argument 'maxiter'`
  before producing any output. Renamed the keyword at both call sites (first-
  and second-stage GMM). A new analytic recovery test
  (`tests/test_tierD_structural_analytic.py::TestBLPAnalytic`) recovers the
  known linear price/characteristic coefficients on a logit DGP with
  endogenous price and valid cost instruments, and guards the keyword
  regression directly. This clears the last Tier D zero-guard remainder.

- **⚠️ Correctness fix: `sp.granger_causality` test statistic was wrong by
  orders of magnitude.** The Wald variance was a placeholder `V = sigma2 * I`
  (the *outcome* residual variance as if it were the *coefficient* covariance),
  ignoring the design-matrix term `(X'X)⁻¹`. The F-statistic was therefore too
  small by a factor of ≈ `T·Var(regressors)`, so the test almost never rejected
  regardless of the true causal structure — on a DGP where `x[t-1]` drives `y`
  with coefficient 0.8, the genuine F is ≈326 (p≈1e-16) but the function
  returned F≈0.36 (p≈0.70, `reject=False`). `VARResult` now carries `(X'X)⁻¹`
  and `granger_causality` forms the proper coefficient covariance
  `σ²_caused·(X'X)⁻¹`, so the F now matches the restricted-vs-unrestricted OLS
  F-test and detects causal direction correctly. Guarded by
  `tests/test_tierD_p2_timeseries_analytic.py`. No prior valid result is
  invalidated (the old statistic was statistically meaningless); no JOSS/JSS
  table uses `granger_causality`. Found by the Tier D campaign (CLAUDE.md §5).

- **⚠️ Correctness fix: d-separation (`statspai.dag`) was wrong on forks and
  colliders.** The moralisation step in `_d_separated` married *siblings*
  (children of a common parent) instead of *co-parents* (parents of a common
  child). As a result conditioning on a common cause failed to block a fork
  (`A ⊥ C | M` on `M→A, M→C` returned `False`) and conditioning on a collider
  failed to open it (`A ⊥ C | K` on `A→K←C` returned `True`) — the two
  non-trivial d-separation cases were both backwards. This propagated to
  everything built on `_d_separated`: `DAG.d_separated`, `adjustment_sets`,
  `backdoor_paths`, `do_rule1/2/3`, `do_calculus_apply`, `swig`, and
  `dag_recommend_estimator`. Moralisation now connects every pair of a node's
  parents; the chain/fork/collider truths and back-door adjustment-set finding
  are correct (e.g. `adjustment_sets("X","Y")` on `W→X, W→Y, X→Y` now returns
  `[{W}]`). Guarded by `tests/test_tierD_p2_dag_dsep_analytic.py`; all 9
  dag-touching test files still pass (none had pinned the broken behaviour).
  No JOSS/JSS table uses these graph routines. Found by the Tier D campaign.

- **⚠️ Correctness fix: `sp.evalue` HR E-values and confidence-interval
  E-values (now full R `EValue` parity).** Two behaviours that can move a
  previously-reported number (#21):
  - **`measure='HR'` was always treated as a rare-outcome ratio**
    (`OR ≈ RR ≈ HR`). It now uses the exact common-outcome conversion
    `(1 − 0.5^√HR)/(1 − 0.5^√(1/HR))` by default (`rare=False`), matching
    `EValue::evalues.HR`. HR-based E-values therefore change for non-rare
    outcomes; pass `rare=True` to recover the old rare approximation. (`OR`
    already converted to RR by default, so OR results are unchanged; `RR` is
    unaffected.)
  - **A confidence-interval E-value is now clamped to exactly 1 when the
    interval contains the null** (or a user-supplied `true` reference). The
    E-value was previously computed from the interval limit regardless, so a
    non-significant result could report a spurious E-value > 1; it now
    correctly reports 1 (no unmeasured confounding is needed to explain a
    result already compatible with the null).
  The keyword `rare_outcome` is renamed `rare`; the old name still works as a
  `DeprecationWarning` alias. Verified at machine precision against R `EValue`
  across every measure (`tests/r_parity/23_evalue.py`, 26 rows, worst relative
  difference 5.8e-14). **JOSS/JSS note:** this is the `23_evalue` row of the JSS
  cross-language parity table
  (`Paper-JSS/manuscript/tables/appendix_b_parity.tex`); the change *increases*
  agreement with the gold-standard R package and the row stays a
  machine-precision **PASS** (CI "Numerical reference parity" + "R closed-form
  parity" jobs green). No JOSS (#10604) figure uses an HR or CI E-value.

- **Agent-UX: `describe_function` advertised stale params / wrong defaults for
  29 hand-written specs (two invariant classes, 18 + 11).** An agent that reads
  the schema and calls `sp.<name>(**kwargs)` verbatim was led into `TypeError`s
  and incorrect defaults. No estimator numerics, signatures, or parity numbers
  changed — metadata only.
  - **18 specs advertised param *names* the function cannot accept** (e.g.
    `metalearner` said `treatment`/`method`; the signature — and the spec's own
    example — use `treat`/`learner`). Fixed to match real signatures;
    `tests/test_registry_signature_contract.py` now locks two invariants over
    all hand-written specs (no phantom params; every required signature param
    documented).
  - **11 specs reported a default *value* the estimator does not use** — e.g.
    `drdid` method `'dr'`→`'imp'` (enum `['dr','or','ipw','reg','stdipw']`→
    `['imp','trad']`), `did_bcf` `n_trees` `200`→`50`, `harvest_did` `reference`
    `'pre'` (str)→`-1` (int), `iv_compare`/`iv_diag` defaults stored as
    string-reprs-of-tuples → real lists. `tests/test_registry_default_contract.py`
    locks that a pinned default neither contradicts the signature nor falls
    outside its own enum. (Also surfaced and corrected a latent duplicate
    `harvest_did` registration.)

- **⚠️ Reproducibility fix: `sp.match(method='nearest')` now resolves exact
  nearest-neighbor ties by source DataFrame index.** The Euclidean/propensity
  nearest-neighbor path previously relied on `argpartition` / incidental row
  order for equal-distance controls (and target order when matching without
  replacement), so exact ties on discrete or binary covariates could move the
  ATT across environments. Equal-distance ties now select lower-index pool
  units first, with lower-index target units used as the without-replacement
  fallback. A shuffled-row regression test guards the policy, and
  `tests/test_tierD_lalonde_psm_guard.py` is tightened from the old
  cross-backend band (width ~$300) to two exact anchors (±0.1): GitHub CI is
  now bitwise-identical across ubuntu/windows/macos (`1967.94`), while
  Accelerate on macOS 26 lands at `1963.43` — a residual ULP-level *near*-tie
  sensitivity in the BLAS-computed distances, not in the tie-break itself.
  Existing results change only when previous data contained exact
  equal-distance ties. See
  [MIGRATION.md](MIGRATION.md#matching-nearest-tie-break).

### Known issues

- **NIST StRD Linear Least Squares certification for the OLS kernel
  (`tests/numerical_accuracy/test_nist_strd_ols.py`).** All 11 NIST Statistical
  Reference Datasets for linear regression (Norris, Pontius, NoInt1/2, Filip,
  Longley, Wampler1–5) are bundled verbatim under
  `tests/numerical_accuracy/_fixtures/nist_strd/` and checked against their
  embedded certified values. Both the low-level `ols_fit`/`OLSEstimator`
  kernel and the certified standard errors are verified, plus a
  QR-beats-normal-equations guard on the ill-conditioned designs. Lives in a
  dedicated `tests/numerical_accuracy/` suite (separate from the R/Stata
  `reference_parity/` fixtures) since it certifies certified-value accuracy
  rather than cross-language parity.

### Changed (⚠️ Correctness)

- **`sp.regress` now fits in mean-centred (Frisch-Waugh-Lovell) coordinates,
  fixing coefficient/F/R² loss under large constant offsets.** When an
  intercept is present, `OLSEstimator.estimate` demeans `y` and the
  non-intercept regressors before the least-squares solve and reconstructs the
  intercept, so a large constant offset in `y` (or a regressor) no longer
  destroys the slope coefficients through catastrophic cancellation. FWL makes
  this **algebraically identical** to the previous fit, so well-conditioned
  designs are unchanged to machine precision (verified: coefficients match
  `numpy.linalg.lstsq` to ~1e-15; the NIST StRD Linear LS certification and all
  R/Stata parity suites are unchanged). The effect is only visible on
  pathological designs: on the NIST StRD ANOVA datasets `SmLs07/08/09` (9
  constant leading digits, e.g. `y ≈ 1.0000000004e12`) the certified-F relative
  error drops from ≈4.8e-4 / 9.9e-3 / 2.3e-2 to the irreducible IEEE-754 float64
  floor (~3–7e-5). Previously these were the lone failing NIST ANOVA cases;
  they now pass at a documented float64-floor tolerance. No user-facing numbers
  on real data change.

- **OLS kernel now solves via QR factorisation instead of the normal
  equations.** `ols_fit` (coefficients) and `OLSEstimator.estimate`
  (covariance) previously solved `(X'X) b = X'y` and formed `inv(X'X)`, which
  *squares* the condition number of the design matrix. On well-conditioned
  data nothing observable changes (results match the old path to ~1e-12; the
  full `reference_parity` + `external_parity` JOSS suites are unaffected). On
  **ill-conditioned designs the new path is dramatically more accurate**: on the
  NIST StRD suite the degree-10 Filippelli polynomial went from **0 correct
  digits to ~7**, Longley from ~7.7 to ~11, and Wampler1 from ~6.1 to ~9.6
  correct digits. Any code that fit OLS on near-collinear or high-degree
  polynomial designs and relied on (silently wrong) coefficients will now get
  materially different — correct — numbers. See
  [MIGRATION.md](MIGRATION.md#ols-qr-kernel).
- **Exact-fit OLS no longer emits a divide-by-zero `RuntimeWarning`.** When a
  regression fits the data exactly (`R² == 1`, e.g. NIST Wampler1/2), the
  F-statistic is now reported as `inf` (matching the NIST-certified
  "Infinity") with `f_pvalue = 0.0`, computed without tripping the warning. The
  reported value is unchanged for every non-exact fit.
- **Auxiliary OLS in `sp.did(method='wooldridge')` and the Romano–Wolf step-down
  (`sp.romano_wolf`) hardened to the same QR solve.** Both carried private
  `_ols_fit` helpers that still formed `inv(X'X)` — Wooldridge for both the
  coefficients and the cluster/HC1 sandwich bread, Romano–Wolf for the bread
  only (its coefficients were already QR). They now reuse the QR factor
  (`(X'X)⁻¹ = R⁻¹R⁻ᵀ`). Output is **bit-identical to ~1e-15 on well-conditioned
  designs** (verified directly, and the Callaway–Sant'Anna R-parity pin plus the
  MHT suites are unchanged); the change only adds headroom on near-collinear
  designs (e.g. saturated group×period dummies). No API change.
- **`sp.regress` now fails loudly on a perfectly collinear design instead of
  returning unidentified garbage.** A rank-deficient design (duplicate or
  proportional regressors, the dummy-variable trap with complementary 0/1
  dummies, or a constant non-intercept regressor) previously returned enormous
  meaningless coefficients (e.g. `~1e14`) with no warning. It now raises
  `NumericalInstability`, naming the offending columns in `.diagnostics`.
  Detection is **structural** (duplicate/proportional columns, zero-variance
  regressors), not conditioning-based — a singular-value/rank tolerance loose
  enough to catch collinearity would also flag legitimately ill-conditioned
  full-rank designs such as NIST Filippelli (`s_min/s_max ~ 6e-16`), so those
  still fit. General exact dependence among 3+ columns that is not a pairwise
  duplicate/constant is intentionally not auto-detected for the same reason.
  See [MIGRATION.md](MIGRATION.md#regress-collinearity-guard).
- **`sp.logit` / `sp.probit` now warn on perfect (quasi-complete) separation.**
  When the outcome is perfectly predicted by the linear index the MLE does not
  exist; Newton–Raphson previously "converged" by its step tolerance to large
  finite coefficients with no signal. It now emits a `ConvergenceWarning`
  (the reported coefficients/SEs are stopping-rule artefacts; the hint suggests
  penalized/Firth logistic regression). Non-separable fits are unaffected.

### Fixed

- **`sp.iv(method='lasso', formula=...)` raised `TypeError` instead of
  fitting.** The IV dispatcher forwarded `formula=` verbatim into
  `lasso_iv`, which takes native `x_endog`/`z`/`x_exog` lists and does not
  accept a `formula` argument, so the Patsy-style formula route
  (`sp.iv(method='lasso', formula="y ~ (d ~ z1 + z2) + x", data=df)`) failed
  with `lasso_iv() got an unexpected keyword argument 'formula'`. The
  dispatcher now parses the formula into `lasso_iv`'s native parameter names
  (and accepts the canonical `endog`/`instruments`/`exog` aliases), so the
  formula path returns the **same** estimates as the explicit
  `x_endog=[...]`, `z=[...]` calling convention — verified bit-for-bit
  (`atol=0`) in `tests/test_iv_cov_tail.py::test_dispatch_lasso_formula_matches_native`.
  No existing numerics move: the native `lasso_iv` path is unchanged; this only
  turns a previously-erroring route into a working one.

- **`decomposition/_kernel_density_at` NumPy 1.25 `DeprecationWarning`.** The
  legacy RIF kernel-density helper called `float()` on the length-1 array
  returned by `scipy.stats.gaussian_kde(...)(point)`, which NumPy 1.25
  deprecates (and a future NumPy will turn into an error). It now indexes the
  single element first (`float(np.asarray(kde(point)).ravel()[0])`). The
  returned value is **identical** — it is the same element `float()` already
  extracted — so no decomposition output changes; the warning is simply gone.

- **`sp.event_study` crashed on string/extension-dtype time columns under
  pandas ≥ 3.0.** The non-numeric-time branch guarded on
  `np.issubdtype(col.dtype, np.number)`, which raises
  `TypeError: Cannot interpret '<StringDtype...>'` when pandas ≥ 3.0 infers a
  string column as `StringDtype` (rather than `object`). Switched both checks
  to `pd.api.types.is_numeric_dtype(col)`, whose truth value is **identical**
  for every numpy numeric dtype — so numeric-time results are byte-for-byte
  unchanged (verified against the full event-study suite and the `did`
  reference-parity set); it only repairs the previously-crashing string-time
  path. Surfaced by the Windows/macOS CI matrix on pandas 3.0 / numpy 2.4.

- **`sp.lpcmci` / `sp.dynotears` would crash on string columns under
  pandas ≥ 3.0.** Their default numeric-variable filter used the same
  `np.issubdtype(col.dtype, np.number)` anti-pattern, which raises on a
  `StringDtype` column. Switched to `pd.api.types.is_numeric_dtype` — identical
  numeric-column selection (verified against the causal-discovery suite), now
  simply skipping string/extension columns instead of crashing. Pre-emptive
  hardening found by a repo-wide scan after the `event_study` fix above.

### Added

- **Registry-example bind guard (`tests/test_registry_examples_bind.py`).**
  A parametrized test now statically parses every registered `example` string
  and binds the keyword arguments of its `sp.<name>(...)` call against the real
  signature, failing if an example references a keyword the function does not
  accept or does not parse. This locks down the agent copy-paste path — an
  agent reads `sp.describe_function(name)` and runs the example verbatim — so
  the registry/example drift fixed below cannot silently return. 373 examples
  bind green.

- **`parity` optional extra — opt-in DoubleML reference pin for `sp.dml`.**
  `pip install -e ".[dev,parity]"` now installs `doubleml-for-py` (the Python
  DoubleML reference of Bach, Chernozhukov, Kurz & Spindler, JMLR 23(53),
  2022), so `tests/external_parity/test_dml_python_parity.py` *runs* instead
  of silently skipping. Under identical scikit-learn learners and folds,
  `sp.dml(model='plr')` reproduces `doubleml-for-py` to machine precision on
  the seed-42 fixture — `|Δ coefficient| = 1.1e-16` and `|Δ standard error| =
  1.4e-17`, i.e. one float64 unit in the last place. `doubleml` remains *not*
  a runtime dependency. The measured numbers, software versions, and the
  divergence discussion are recorded in the source-audit evidence trail
  (`docs/jss_source_audit_dossier.md` plus the parity artifacts under
  `tests/external_parity/`). Verified by installing the
  extra and running both `tests/external_parity/test_dml_python_parity.py` and
  `tests/reference_parity/test_dml_parity.py` (55 DML tests green).

### Fixed

- **⚠️ Correctness: `sp.drdid(method='trad')` returned ~half the true ATT.**
  The traditional doubly-robust DiD branch of `sp.drdid` (Sant'Anna & Zhao
  2020) normalised each of its four cell terms (treated/control × post/pre) by
  the **full sample size** `n` instead of by that cell's weight mass. This
  multiplied every term by the cell's sample share (~0.25 each on a balanced
  2×2), biasing the ATT toward zero by ~50%: on a 2×2 with a true ATT of 2.0
  (raw DiD 1.96) the traditional estimator returned ≈1.04. Each term is now
  normalised by its own weight total, so `method='trad'` reduces exactly to the
  raw 2×2 DiD when no covariates are supplied and recovers the true ATT with
  covariates. The improved (locally efficient) `method='imp'` — the **default**
  — already normalised correctly and is **unchanged**, so no default-path or
  parity/dossier numbers move. `sp.drdid` now also raises `ValueError` on an
  unknown `method` instead of silently treating it as traditional (previously
  e.g. `method='ipw'` ran the traditional branch). See MIGRATION.md.

- **⚠️ Correctness: `sp.multiway_cluster_vcov` undercounted intersection
  clusters, biasing multiway-cluster-robust standard errors.** The
  Cameron-Gelbach-Miller inclusion-exclusion builds an *intersection* cluster
  (the unique combinations of the clustering dimensions); its key was formed by
  joining the dimensions into a single string with a `"\0"` separator, but
  NumPy fixed-width unicode strips the embedded NUL, so e.g. `(1, 23)` and
  `(12, 3)` both collapsed to `"123"`. On a 40×50 crossed-cluster DGP this
  merged 1733 true intersection clusters into 1639, biasing the two-way SE by
  ~0.2% (~0.5% at three-way) versus the canonical estimator. The intersection
  key is now built collision-free via `np.unique(axis=0)` on per-dimension
  integer codes. `sp.multiway_cluster_vcov` now reproduces `sandwich::vcovCL`
  and `sp.twoway_cluster` to machine precision (two-way exact; three-way rel
  ~4e-7), pinned by the new `tests/r_parity` module 56 and a direct
  twoway-vs-multiway regression test. Propagates to `did.harvest` and
  `panel.feols` multiway-clustered SEs. `sp.twoway_cluster` itself was already
  correct (distinct collision-free key) and is unchanged. See MIGRATION.md.

- **`.glance()` crashed (`OverflowError: cannot convert float infinity to
  integer`) on Cox and parametric-survival (`survreg`) results.** Those
  estimators deliberately store `df_resid = inf` to signal a large-sample
  (normal) reference distribution, but `glance()` cast the residual degrees of
  freedom with `int()` unconditionally. The cast now passes non-finite degrees
  of freedom through unchanged; finite results keep an integer `df_resid`
  (no change). A crash-hunt across ~48 fitted results confirmed the rest of
  the §3 unified-result export surface (`summary`/`to_latex`/`to_markdown`/
  `to_word`/`to_excel`/`cite`/`tidy`/`for_agent`/`plot`) is otherwise clean
  across 20+ estimator families. Covered by `tests/test_glance_survival.py`.

- **`sp.event_study` results crashed the library's own exporters, plotters and
  pre-trend tools (canonical-column mismatch).** `sp.event_study` emitted its
  coefficient table under the column name `estimate`, but the rest of the DID
  family — and every downstream consumer — keys on the canonical `att` column
  (`did._core.EVENT_STUDY_COLUMNS`). So the *canonical* event-study estimator
  was incompatible with its own tooling: `.tidy()` (and the `.to_markdown()` /
  `.to_excel()` / `.to_word()` exporters that delegate to it) raised
  `TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'`;
  `.plot()` / `.event_study_plot()` / `sp.enhanced_event_study_plot` raised
  `KeyError: 'att'`; and `sp.honest_did` / `sp.breakdown_m` raised
  `ValueError: missing {'att'}`. The event-study table now carries the
  canonical `att` column (with `estimate` retained as a backward-compatible
  alias), fixing every consumer at the source. No numerical change.

- **`sp.pretrends_test` / `sp.pretrends_summary` crashed (`LinAlgError:
  Singular matrix`) on every standard `sp.event_study` result** — the same
  reference-period defect already fixed in `pretrends_power`: the SE = 0
  omitted period made the diagonal VCV singular. It is now dropped before
  inversion, with a clear `ValueError` on a genuinely collinear pre-period set.

- **`sp.diagnose_result` crashed (`TypeError: bad operand type for abs():
  'str'`) on `sp.synth` results.** The donor-pool check iterated the synthetic
  weights, but `sp.synth` stores them as a `['unit', 'weight']` DataFrame, so
  iteration yielded column-name strings. The weights are now coerced to their
  numeric values regardless of container (DataFrame / Series / dict / array).

- **`sp.pretrends_power` crashed (`LinAlgError: Singular matrix`) on every
  standard `sp.event_study` result.** Roth's (2022) pre-trend power calculation
  inverts the pre-period variance–covariance matrix, but the omitted reference
  period (relative time −1) is reported with a standard error of exactly zero,
  so the diagonal VCV was singular and `np.linalg.inv` raised on the *exact*
  workflow shown in the function's own docstring. The reference period (and any
  other mechanically-normalised, zero-SE period) is now dropped before
  inversion — it is the baseline, not an estimated coefficient — so the joint
  pre-trend test runs on the estimated pre-periods only. A full-rank
  `model_info['vcv_pre']` and a full-length `delta` are aligned to the retained
  periods, and a still-singular VCV now raises a clear `ValueError` (collinear
  pre-periods) instead of an opaque NumPy error. No output changes for any call
  that previously succeeded. Covered by `tests/test_pretrends_power.py`.

- **⚠️ Correctness — `sp.structural_break` sup-F p-value used the wrong null
  distribution.** The Chow/sup-F statistic is a *supremum* of the F statistic
  over candidate break points, so under H0 it follows the Andrews (1993)
  sup-F law — **not** `F(k, n-2k)`. The previous code referred the maximised
  statistic to the ordinary F CDF, which ignored the maximisation and
  massively over-rejected: on pure Gaussian white noise at the 5% level the
  test flagged a spurious structural break in **33–37%** of series (measured,
  n ∈ {100, 200, 400}). The p-value is now computed from the Andrews (1993)
  limiting null — a q-vector Brownian-bridge functional sampled by a
  deterministic, cached simulation on a grid tied to the sample size —
  restoring **nominal size (~0.05)** while retaining power (1.00 / 0.88 to
  detect a one-/half-σ mean shift at n=200). The same correct threshold now
  drives the Bai-Perron sequential `supF(l+1|l)` stopping rule (previously the
  same naive-F over-detection), so `method='bai-perron'` no longer
  over-segments noise. As a side benefit the Bai-Perron result now populates
  `f_stats` / `p_values` (one sup-F statistic and Andrews p-value per detected
  break, chronologically aligned) instead of returning `None`. Reference
  verified via Crossref / Econometric Society / RePEc: Andrews, D.W.K. (1993),
  *Econometrica* 61(4), 821-856, doi:10.2307/2951764. See `MIGRATION.md`.

- **33 registered `example` strings were statically broken (agent-UX).** Six
  failed to parse (stray/`unmatched` parens, a positional-after-keyword
  `...` placeholder, an unclosed call) and 27 passed a keyword the function
  does not accept — a deterministic `TypeError`/`SyntaxError` on the exact
  agent copy-paste path. Root causes were two long-standing parameter-name
  drifts: the Mendelian-randomization family (`mr_egger`/`mr_ivw`/`mr_raps`/
  `mr_presso`) used the short `b_exp`/`b_out`/`se_exp`/`se_out` names instead
  of the implemented `beta_exposure`/`beta_outcome`/`se_exposure`/`se_outcome`,
  and the Bayesian family (`bayes_rd`/`bayes_fuzzy_rd`/`bayes_mte`/`bayes_did`/
  `bayes_iv`) plus `metalearner`/`tmle`/`causal_impact`/`sensemakr`/
  `spec_curve`/`qreg`/`tobit`/`heckman`/`cluster_cross_interference`/
  `causal_dqn`/`pci_mtp`/`bartik`/`ffl_decompose` drifted from their
  signatures. `qreg`/`tobit`/`heckman`/`spec_curve` additionally had wrong
  call *shapes* (formula passed positionally into `data`; flat `controls`
  where a list-of-lists was required) and were rebuilt to runnable form and
  executed to confirm. Fixed across `registry.py` and `_baseline_cards.py`;
  the regenerated `schemas/` bundle stays in sync. Guarded by the new
  bind test above.
- **`sp.heckman` / `sp.tobit` MCP tool schemas advertised parameters the
  functions reject (agent-native path).** The prior `example`-string fix above
  repaired the copy-paste path, but the *other* drifted field — the
  `FunctionSpec.params` that generate each tool's MCP `input_schema` — was left
  pointing at a never-implemented formula API: `heckman` exposed required
  `formula`/`select_formula` and `tobit` exposed `formula`/`lower`/`upper`,
  while the implementations take `data, y, x, select, z` and `data, y, x, ll,
  ul`. An agent calling either tool exactly as the manifest instructed hit a
  deterministic `TypeError: ... got an unexpected keyword argument 'formula'`.
  The `example`-bind guard above does not cover this because it validates the
  `example` field, not `params`. Re-pointed both `params` blocks at the real
  signatures in `registry.py`; the regenerated `schemas/` bundle is back in
  sync and both tools now dispatch (`sp.heckman` recovers the known-truth
  β=2.05 fixture through the MCP dispatch layer). The estimator numerics were
  never affected — Python-direct `sp.heckman(...)` / `sp.tobit(...)` always
  worked; only the MCP schema was wrong.
- **⚠️ Correctness — `sp.stabilized_weights` / `sp.msm` silently dropped all
  IPTW adjustment on single-period panels.** On a single-period (point-
  treatment) panel the within-unit lagged-treatment column is all-zero, which
  made the logistic treatment-model design singular. The internal
  `_logit_proba` helper swallowed the resulting `LinAlgError` and silently
  returned the marginal mean for *both* the numerator and denominator models,
  so every stabilized weight collapsed to exactly `1.0` — turning the marginal
  structural model into an unweighted, **confounded** regression with no
  warning (a CLAUDE.md §7 silent-degradation violation). `_logit_proba` now
  drops zero-variance columns before fitting (the numerically correct move:
  such columns only duplicate the intercept that `add_constant` adds) and,
  if the treatment model still fails to converge (e.g. perfect separation),
  emits a `RuntimeWarning` instead of silently degrading. On the regression
  fixture the fixed path reproduces a textbook stabilized-IPTW computation to
  machine precision (`max|Δw| = 1.8e-15`); the already-correct multi-period
  path is unchanged. Verified by `tests/test_msm_singleperiod_iptw_regression.py`
  (3 new tests) plus the existing `tests/test_msm.py` (8 green total). See
  MIGRATION.md.
- **CI: scikit-learn 1.9 compatibility — `LassoCV(n_alphas=...)` removal.**
  scikit-learn 1.7 deprecated the `n_alphas` argument of the coordinate-descent
  CV estimators (`LassoCV`/`ElasticNetCV`/…) in favour of passing an integer to
  `alphas`, and 1.9 removed it outright — constructing `LassoCV(n_alphas=20)`
  now raises `TypeError: LassoCV.__init__() got an unexpected keyword argument
  'n_alphas'`. The `CI/CD Pipeline` matrix resolves to scikit-learn 1.9.0, so
  `sp.tmle(method='hal')` (`HALRegressor`) and `sp.rd_flex(learner='lasso')`
  both failed at construction (`tests/test_hal_tmle.py`,
  `tests/test_estimator_provenance_round5.py::TestHalTmleProvenance`,
  `tests/test_low_cov_battery.py::test_hal_regressor_predicts_finite`). A new
  version-robust shim `statspai.compat.sklearn.lasso_cv_alphas_kwargs(n)` emits
  `{"alphas": n}` on scikit-learn >= 1.7 and `{"n_alphas": n}` on older
  releases; both call sites now route through it. The number of path alphas
  (20 for HAL, 50 for `rd_flex`) is unchanged — no numerical effect. Verified
  on the local scikit-learn 1.6.1 pin (8 HAL tests + 6 `rd_flex` tests green)
  and by version-logic assertion across 1.6/1.7/1.8/1.9.
- **CI: `schemas/functions.json` no longer drifts by pandas version.** The
  auto-registered schema export stringified parameter annotations via
  `str(typing.Optional[pandas.DataFrame])`, which pandas 3.0 renders as
  `pandas.DataFrame` while pandas < 3.0 renders as the internal
  `pandas.core.frame.DataFrame`. Because pandas 3.0 requires Python >= 3.11,
  the committed bundle matched the CI `ubuntu x 3.10` shard but was flagged
  stale on every 3.11/3.12/3.13 shard, failing `CI/CD Pipeline`
  (`tests/test_schema_export.py::test_committed_schemas_dir_is_in_sync`).
  `_stringify_annotation` now canonicalises pandas internal module paths to
  their public form, so the exported bundle is byte-identical across pandas
  versions. Verified by regenerating under both pandas 2.x and pandas 3.0 and
  confirming identical output. Agent-facing schema metadata only — no
  numerical effect.

- **CI: Windows shards no longer fail on validation-evidence path separators.**
  `_scan_reference_tests` recorded each parity test as a validation-evidence
  note via `str(path.relative_to(root))`, which emits OS-native separators —
  so Windows runners wrote `tests\reference_parity\...` backslash notes. That
  (a) failed `tests/test_jss_validation_api.py::test_certified_validated_symbols_have_attached_evidence_notes`
  because the JSS evidence-grade markers are forward-slash, and (b) drifted
  `agent_cards.json` away from the POSIX-generated committed bundle, failing
  the schema-sync guard on every Windows shard. Notes are now built with
  `Path.as_posix()` and the test files are iterated in a POSIX-keyed sort, so
  the evidence notes — and the exported bundle — are byte-identical on Windows
  and POSIX. Agent-facing metadata only — no numerical effect.

- **CI hardening: swept the same OS-portability class across the codebase.** A
  multi-agent audit surfaced the remaining latent siblings of the two bugs
  above (none yet red, but each a Windows landmine of the identical class):
  `validation._rel` (artifact paths in `sp.validation_report()`) and
  `scripts/stability_audit.py` now build paths with `Path.as_posix()`;
  `tests/test_causal_workflow.py` and `tests/test_paper_tables.py` now read the
  UTF-8 HTML/LaTeX/markdown reports they generate with `encoding="utf-8"`
  (cp1252 default would mojibake on Windows — CLAUDE.md §5). Two POSIX-runnable
  schema-bundle invariants were added (no backslash separators; no internal
  `pandas.core.` paths) so both classes are caught on any shard, not only the
  one whose regeneration diverges.

- **Docs: `docs/reference/dml.md` documented four `sp.dml` patterns that
  raised on copy-paste.** `r.coef` → `r.estimate`; `r.ci(alpha=0.05)` →
  `r.ci` (a `(lower, upper)` tuple — the level is set via
  `sp.dml(..., alpha=)`); `r.influence_function` (never exposed on
  `CausalResult`) → `r.diagnostics` / `r.pvalue`; and an IRM example passing a
  non-existent `trim=` keyword (propensity clipping is automatic at
  `[0.01, 0.99]`, with the clip counts reported in `r.diagnostics`). Every
  documented result attribute and all four `model=` types now run on the
  bundled fixture. Docs only — no API or numerical change.

- **Docs: corrected an unverified claim in
  `docs/guides/sp_dml_vs_doubleml.md`.** The guide asserted that matching
  propensity-trimming thresholds eliminates the small `sp.dml`-vs-
  `doubleml-for-py` IRM gap; the gap (0.0076 absolute, ≈ 0.10 SE) is verified
  *not* to move with trimming nor with `normalize_ipw`, and stems from
  internal AIPW score construction. PLR agreement is now stated as the
  measured machine-precision figure (`1.1e-16` / `1.4e-17`) rather than "four
  decimal places", and an over-broad "PLR / PLIV / IIVM agree to machine
  precision" line was narrowed to the PLR case that is actually pinned.

- **Docs: refreshed live registry-stats drift.** `docs/stats.md` (per-module
  table + the measured source/test LOC rows), `README.md`, and `README_CN.md`
  now match `python scripts/registry_stats.py` (269,043 core LOC / 96,514 test
  LOC; 1,020 functions across 81 submodules). This restores
  `tests/test_jss_release_manifest.py::test_registry_stats_docs_are_live`
  (`scripts/registry_stats.py --check` exits 0). Docs only — no API change.

## [1.16.1] — 2026-06-01

### ⚠️ Correctness fix — `sp.synth()` default restored to canonical classic SCM

- **`sp.synth(...)` now defaults to `method='classic'`** (the Abadie, Diamond
  & Hainmueller 2010 synthetic control) again, matching the documented default
  in the `sp.synth` docstring (`method : str, default 'classic'`), the
  `Synth::synth` → `sp.synth` mapping in the migration-from-R guide, and the
  canonical Prop99 examples in the docs and `examples/synth_prop99.py`. The
  signature default had silently drifted to `method='augmented'` (Augmented
  SCM, Ben-Michael, Feller & Rothstein 2021), whose ridge correction is
  *designed* to allow negative donor weights by extrapolating outside the
  donor convex hull — surprising for the canonical `synth()` entry point and
  inconsistent with the package's own documentation. **A bare `sp.synth(...)`
  call again returns convex, non-negative, sum-to-one donor weights.**
  Augmented SCM remains fully available via `method='augmented'` (or
  `'ascm'`); every non-default method is unchanged. Verified against the full
  synth test surface (170 tests) and the R `Synth` recovery parity test.
  Guarded by
  `tests/test_synth.py::TestSyntheticControl::test_weights_non_negative`.

### ⚠️ Correctness fix — synthetic-control weights projected back onto the simplex

- **`solve_simplex_weights` (the inner W solver shared by `sp.synth` and the
  SCM/sdid/augsynth/gsynth family) now projects the SLSQP solution back onto
  the unit simplex** — clipping sub-tolerance negative weights to zero and
  renormalising to sum 1 — before returning. SLSQP enforces the
  `w_j ≥ 0, Σw = 1` constraints only up to its own tolerance, so the raw
  `result.x` could carry small negative donor weights (observed down to
  `≈ -7.5e-4`), violating the non-negativity invariant that the synthetic
  control estimand and every reference implementation (R `Synth`, `gsynth`)
  rely on. **Donor weights change** by the solver's sub-tolerance noise; the
  projection moves the native output *toward* the reference clean-simplex
  solution, so reference parity is preserved (verified against the synth
  parity suite). Guarded by
  `tests/test_synth.py::TestSyntheticControl::test_weights_non_negative`.

### Fixed — agent schema generation preserves full typing shapes

- **`sp.function_schema` / the registry schema generator now keep
  parametrised typing annotations intact across Python 3.9–3.13.**
  `registry._stringify_annotation` checked `hasattr(ann, "__name__")` before
  inspecting typing generics; on Python 3.10+ aliases such as
  `Optional[Dict[str, Any]]` expose `__name__`, so the helper collapsed them to
  the bare origin name (`Optional`, `Dict`) and dropped the inner element
  types. It now resolves `typing.`-prefixed and `__origin__`-bearing
  annotations first, so the machine-readable parameter shapes agents consume
  stay stable and version-independent. No estimator numbers change.

### Docs

- Reviewer-facing validation docs (`docs/jss_source_audit_dossier.md`,
  `Paper-JSS/README.md`, `README.md`) refreshed: the focused
  reviewer follow-up regression command is documented, the
  `tests/test_joss_reviewer_followups.py` compatibility path is restored for
  the public review thread (delegating to
  `tests/test_external_reviewer_followups.py`), and the activity/measurement
  dates are updated to 2026-06-01. Live `docs/stats.md` counts re-measured
  against the 1.16.1 source tree (source LOC 269,010).

## [1.16.0+source.20260531] — 2026-05-31

### Correctness fix — RD density native path now ports `rddensity` defaults

- **`sp.rddensity(backend="native")` now mirrors the default
  `rddensity::rddensity` unrestricted triangular-kernel path** instead of
  using a dependency-light Silverman-style pilot bandwidth and ECDF-slope
  approximation. The native implementation ports `rdbwdensity` combination
  bandwidths, mass-point ECDF handling, and jackknife CJM local-polynomial
  density inference. On the Lee/RD Senate replica, native StatsPAI now
  matches `rddensity` on the robust p-value at `rel = 3.3e-11` and on the
  Stata reference at `rel = 8.9e-11`; `09_rddensity` moves from a T4
  bandwidth-selector disclosure to a T2 native reference-parity pass.
  Side-specific manual bandwidths remain supported, and `backend="r"` still
  delegates to the R package for users who want direct package execution.
  Guarded by `tests/test_rddensity_io.py` and the Track A parity harness.

### ⚠️ Correctness fix — causal-forest ATE/ATT now doubly-robust (AIPW)

- **`CausalForest.average_treatment_effect` (and the `target_sample`
  `"all"`/`"treated"`/`"control"`/`"overlap"` aggregations) now returns
  the doubly-robust AIPW influence-function mean** — the estimand
  `grf::average_treatment_effect` reports — instead of a plug-in average
  of the regularisation-shrunk CATE predictions. The plug-in average
  overshoots the true effect (≈ 15 % on a clean-overlap DGP) and
  disagreed with `grf` by an order of magnitude on overlap-pathological
  samples. The AIPW score reuses the forest's own cross-fitted nuisances
  `m̂(X)=Ê[Y|X]`, `ê(X)=Ê[T|X]` and the CATE `τ̂(X)`:
  `Γ_i = τ̂ + (T−ê)/(ê(1−ê))·(Y − m̂ − (T−ê)τ̂)`. **Numbers change** for
  any code reading `average_treatment_effect(...)['estimate']`; the SE is
  now the influence-function SE `sd(Γ)/√n`. The plug-in `cf.ate()` /
  `cf.att()` convenience methods are unchanged and retained, but are no
  longer the validation estimand. Users who reported a causal-forest ATE
  from `average_treatment_effect` should re-run.
  - On a clean-overlap DGP (`e(X)∈[0.30,0.70]`, known ATE = 1) the AIPW
    ATE recovers the truth within 0.2 SE and agrees with `grf` at
    `rel = 0.0019` (`z = 0.037` combined SE); the ATT agrees at
    `rel = 0.0028` (`z = 0.05`). For the JSS source snapshot,
    `13_causal_forest` is now a T3 combined-Monte-Carlo-error pass:
    the row is like-for-like AIPW versus `grf` and is graded against
    combined sampling error, not sold as deterministic machine-precision
	    equality. The strictness-tier denominator at this checkpoint was
	    `50 / 4 / 1 / 1 on the 56 R-joined modules` (the current source
	    snapshot has since expanded to `57 / 5 / 1 / 1 on 64` — see the
	    Unreleased Track A entry): the forest row is now
    the only moderate-stochastic T3 row, and the remaining
    methodological/T4 bucket is the documented classical-SCM
    non-uniqueness/reference-disagreement gap.
  - Guards: `tests/reference_parity/test_causal_forest_aipw_recovery.py`
    (recovery against truth, no R needed) and the tightened
    `tests/reference_parity/test_grf_parity.py` (combined-SE parity vs a
    committed `grf` fixture).
- **Synthetic-control solver certified on identified problems and exact
  `Synth` parity exposed when reviewers need it.** Added
  `tests/reference_parity/test_scm_recovery.py` and the cross-language
  module `tests/r_parity/52_scm_unique`: on a DGP whose synthetic-control
  weights are uniquely identified (treated unit exactly a convex
  combination of donors in the pre-period), `sp.synth(method="classic")`
	  recovers the exact weights and gap (pre-RMSE = 0) and agrees with
	  `Synth::synth` at machine-level point precision after fixing the
	  predictor-weight vector and tightening the inner `ipop` QP controls.
	  For the ambiguous Basque-data row, the parity
  harness keeps the native default visible as a documented
  donor-weight-non-uniqueness/reference-disagreement gap. On the same ADH
  special-predictor specification, native StatsPAI tracks Stata `synth`
  at `rel = 4.2e-4` while R `Synth` and Stata differ at `rel = 2.3e-2`;
  the methodological-gap ledger now fails if this guard disappears. Users
  who need exact R numbers can call the optional
  `sp.synth(method="classic", backend="synth")` bridge. The release claim
  is therefore native-solver certification on identified SCM problems plus
  an explicit Basque reference-disagreement limitation, not a hidden
  machine-precision victory on a non-unique original-data example.
- **Augmented SCM native path now ports `augsynth`'s centered
  Ridge+SCM weight convention.** The Basque `18_augsynth` row now
  compares the native Python estimator directly with
  `augsynth::augsynth`, not the optional R bridge: ATT relative error is
  `7.9e-06` and pre-RMSPE relative error is `3.0e-06`.
- **Generalized SCM native path now ports `gsynth`/`fect`'s two-way FE
  factor convention.** The Basque `19_gsynth` row now compares the
  native Python estimator directly with `gsynth::gsynth`, not the
  optional R bridge: ATT and pre-RMSPE match at machine precision, moving
  the row out of the T4 factor-convention bucket. `backend="augsynth"`
  and `backend="gsynth"` remain reference-package migration bridges, not
  native parity comparators.

### Changed — synthetic-DID regularisation aligned to `synthdid` convention

- **`sp.sdid` native unit/time weights now use the
  \citet{arkhangelsky2021synthetic} `synthdid` regularisation and
  Frank-Wolfe weight solver.** The native path now mirrors
  `synthdid:::collapsed.form`, `synthdid:::sc.weight.fw`, the default
  sparsification step, `ζ_ω = (N_tr · T_post)^{1/4} · σ̂`, and
  `ζ_λ = 10^{-6}·σ̂`, where `σ̂` is the standard deviation of the control
  units' pre-treatment first differences. On the California-99 replica,
  the native ATT now matches `synthdid::synthdid_estimate` on identical
  CSV bytes at `rel = 2.6e-15`; the row moves from a T4
  regularisation-zeta gap to a T2 native reference-parity pass. The SE
  still uses StatsPAI's deterministic all-control placebo convention
  while `synthdid_se` uses random placebo replications, so the SE remains
  compared under a 5% tolerance rather than sold as bitwise equality.

### Added — MCP protocol modernization

- **Protocol version negotiation + bump to `2025-06-18`
  (`src/statspai/agent/mcp_server.py`)** — the server now negotiates its
  protocol revision with the client (`SUPPORTED_PROTOCOL_VERSIONS =
  ("2025-06-18", "2025-03-26", "2024-11-05")`): it echoes the client's
  requested revision when supported, else offers the latest. Replaces the
  hard-coded `protocolVersion: "2024-11-05"`. Fully backward-compatible —
  a client negotiating `2024-11-05` ignores the new fields below.
- **Tool annotations (MCP `2025-03-26`)** — every advertised tool now
  carries `annotations: {readOnlyHint: true, openWorldHint: false}`.
  StatsPAI tools read the supplied dataset and compute; they never mutate
  the input file or external state, so a client can auto-approve calls
  without a confirmation prompt. A manifest entry may override either hint.
- **Structured tool output (MCP `2025-06-18`)** — every `tools/call` now
  returns the result object as `structuredContent` alongside the existing
  `text` block, so agents get typed data without re-parsing serialized
  JSON. Error envelopes are structured too (branch on `error_kind`).
  Strict-JSON guarantee (no `NaN`/`Infinity`) holds for the structured
  payload as well. Every tool advertises a compact `outputSchema`; the
  full documented result envelope (`estimate` / `std_error` / `conf_low` /
  `conf_high` / `method` / `n_obs` / `coefficients` / `diagnostics` /
  `violations` / `next_steps` / `next_calls` / `citations` / `result_id` /
  `error` …) is served once via the new `statspai://schema/result`
  resource rather than inlined into all ~480 tools — which keeps the
  `tools/list` payload ~1.2 MB smaller (the schema is byte-identical per
  tool, so inlining it everywhere was 50% duplication).

### Added — JSS reproducibility hardening (Track A R/Stata parity)

- **R reference environment locked + proven reproducible
  (`tests/r_parity/`)** — every committed `results/*_R.json` golden value
  now carries an inline `provenance` block (R version, platform,
  BLAS/LAPACK, and the version of each attached/loaded package), emitted by
  `_common.R::.r_provenance`. The full 245-package dependency closure is
  pinned in `tests/r_parity/renv.lock` (exact versions; true GitHub commit
  SHAs for `augsynth`/`synthdid`), with a human-readable manifest in
  `tests/r_parity/R_ENVIRONMENT.md`. A new `verify_reproduce.py` re-runs
  each R reference on the committed CSV bytes and diffs every statistic
  against the golden value at a 1e-9 reproducibility tolerance: 46 of 47
  data-driven modules reproduce bit-for-bit under R 4.5.2; the report is
  `results/REPRODUCIBILITY_REPORT.md`.
- **`r-parity.yml` CI workflow** — re-runs the closed-form / MLE / matching
  R core (fixest, sandwich, AER, survival, MASS, oaxaca, MatchIt) on every
  push and fails the build on any drift from the committed golden JSON, so
  the cross-language closed-form parity is genuinely refreshed in CI rather
  than only frozen.
- **`sp.validation_report(collect_tests=True)`** — shells out to
  `pytest --collect-only` and returns the authoritative, parametrize-expanded
  parity test counts (124 reference-parity, 52 external-parity, 12 coverage
  Monte Carlo on the current source snapshot); a regression test pins those three to the
  JSS manuscript headline so a parity test added/removed without updating the
  paper fails CI. Default `validation_report()` path is unchanged (fast,
  metadata-only).
- **Strictness-tier breakdown in the Track A parity tables
  (`tests/r_parity/compare.py`)** — each module is classified by its
  registered point-estimate tolerance into machine-level / iterative /
	  moderate / methodological-T4 tiers (50 / 4 / 1 / 1 on the 56
  R-joined modules at this checkpoint; 57 / 5 / 1 / 1 on the 64 R-joined modules
  in the current source snapshot), shown in the Markdown ledger and the LaTeX appendix
  caption so a machine-level point-estimate match is not flattened together with a deliberately loose
  stochastic or documented-convention tolerance.
- **Stata leg brought to the same rigor as R (`tests/stata_parity/`)** —
  `_common.do` now writes an inline `provenance` block (engine version,
  edition, OS) onto every `*_Stata.json`; `verify_reproduce_stata.py` re-runs
  each `.do` on the committed CSV bytes and confirms all 53 Stata modules
  at this checkpoint (61 in the current source snapshot)
  reproduce **bit-for-bit** (worst rel 0) under Stata 18 MP, including the
  iterative-optimiser commands (`set seed 42` + deterministic solvers);
  `_capture_stata_env.do` + `_gen_stata_env.py` pin the engine and the
  verbatim `*!` banner of 17 community ado packages in `STATA_ENVIRONMENT.md`
  (the Stata analogue of `renv.lock`). Reproduction ledger:
  `results/REPRODUCIBILITY_REPORT_STATA.md`.
- **Provenance drift guard
  (`tests/test_parity_harness_contract.py`)** — the normal-CI contract suite
  now fails the build if any committed `*_R.json` (r_version + packages) or
  `*_Stata.json` (stata_version + edition) loses its provenance block.

### Changed — Track A R golden values regenerated under the locked environment

- The current R parity ledger covers 55 rendered R-joined modules under
  R 4.5.2 with the `renv.lock` package set so each is self-describing.
  The material parity-status movement is the added `52_scm_unique`
  counterpart for classical SCM: an identified synthetic-control DGP now
  separates native solver correctness from ambiguous real-data weight
  selection, while the Basque row remains a documented native
  non-uniqueness gap. This is primarily a reference-fixture and evidence
  refresh, not a broad `statspai` estimator-output change.
- The 44 committed `results/*_Stata.json` were likewise refreshed to embed
  the engine `provenance` block; their numbers are **unchanged** (every
  module reproduces at exactly 0 under the locked Stata 18 MP environment).

### Fixed — MCP cold-start bundle drift

- **Stale schema-bundle guard (`.github/workflows/parity-guards.yml`)** —
  the MCP server serves `tools/list` + resources from the committed
  `schemas/` bundle on its cold-start fast path, gated only on a matching
  `statspai_version`. A registry change within the same version (a refactor
  between releases) drifted the bundle silently, so the server served a
  stale tool list. CI now runs `python scripts/dump_schemas.py --check`
  alongside the existing `registry_stats --check`, failing the build until
  the bundle is regenerated. Regenerated the bundle (`schemas/` +
  `src/statspai/schemas/`) to clear the existing drift.

### Added — agent-native sprint

- **Agent-card metadata overlay (`src/statspai/_agent_cards_extra.py`)** —
  89 curated Tier-A cards (assumptions / pre_conditions / failure_modes /
  alternatives / typical_n_min) for certified + validated estimators that
  previously had none, applied via `registry._apply_agent_card_seeds` with
  extend-missing semantics (hand-written `FunctionSpec` content always
  wins). Lifts curated agent-native field coverage roughly threefold. Every
  `alternative` and `exception` is CI-validated to resolve.
- **Relational-integrity contract suite (`tests/test_agent_native_contract.py`)**
  — guards that every agent-card `alternatives` / `failure_modes.alternative`
  resolves to a real function/MCP tool, every `failure_modes.exception` is a
  real class, every advertised MCP tool is executable, every
  `_FOLLOWUP_BY_TOOL` next-call is an advertised tool, and every
  `_CITATIONS_BY_TOOL` bib key exists in `paper.bib`.
- **Machine-readable schema bundle (`scripts/dump_schemas.py`,
  `src/statspai/_schema_export.py`, `schemas/`)** — an import-free, versioned
  bundle (`tools.json` / `functions.json` / `agent_cards.json` /
  `result.schema.json` / `index.json`) so a non-Python client can discover
  the full surface offline. Includes a JSON Schema (draft 2020-12) for the
  agent-facing result payload, contract-tested against real `CausalResult`
  and `EconometricResults` outputs. `--check` gates drift.
- **MCP-sampling LLM client (`statspai.causal_llm.sampling_client`)** —
  `SamplingLLMClient` / `resolve_llm_client()` bridge the MCP server→client
  `sampling/createMessage` round-trip into the `LLMClient` interface, so
  `sp.llm_dag_propose` (and friends) can reuse the connected agent's own
  model with no extra API key, falling back to the deterministic heuristic
  when sampling is unavailable.
- **`interpret_result` MCP tool** — natural-language explanation of a
  fitted result from its cached handle. Wires `resolve_llm_client()` into
  the server's tool-dispatch path: when the client advertised sampling it
  reuses the agent's own model (grounded in the result's own numbers — the
  model is told not to invent estimates), and degrades to a deterministic
  structured brief otherwise. Mid-call sampling failures fall back loudly
  (the error is surfaced in `sampling_error`, never swallowed). Optional
  `question` / `audience` knobs; exposed as a dataless tool so strict-schema
  clients dispatch it without a `data_path`.
- **Auto-tool citation enrichment** — `_enrichment.build_citations` now
  falls back to verified citation tokens in a function's registry
  `reference` field, so hundreds of carded estimators carry citations in
  their MCP output automatically. Only keys that resolve in `paper.bib` are
  ever surfaced (CLAUDE.md §10 red line holds).
- **Agent-workflow regression net (`tests/agent_eval/`)** — an end-to-end
  transcript test (`detect_design → preflight → fit(as_handle) →
  audit_result → sensitivity_from_result`) plus handle-chaining and
  graceful-failure UX contracts.
- **Docs** — [`docs/guides/agent_native_workflow.md`](docs/guides/agent_native_workflow.md),
  an operational playbook for driving StatsPAI as an agent.

### Added — regression-table export

- **Symmetric single-model export surface on `EconometricResults`** —
  `sp.regress` / `sp.ols` / `sp.iv` results now expose `.to_latex()`,
  `.to_html()`, `.to_markdown()`, `.to_excel()`, and `.to_word()`, closing the
  asymmetry where these lived only on `CausalResult` (so `sp.did(...).to_latex()`
  worked but `sp.regress(...).to_latex()` raised `AttributeError`). Each method
  delegates to the canonical `sp.regtable` renderer and forwards every
  `regtable` keyword (`coef_labels` / `keep` / `drop` / `order` / `stats` /
  `se_type` / `stars` / `fmt` / `template` / `notes` …); `to_latex` adds
  `caption=` / `label=`, and the string formats accept an optional `path=`.
- **Agent-native table serialisation (`RegtableResult.to_dict()` /
  `.to_json()`)** — a JSON-safe payload with three layers: metadata, the
  rendered cell grid (the formatted `"2.067***"` / `"(0.074)"` strings), and
  the numeric truth per model (estimate / SE / t / p / CI / stats / depvar).
  NaN/Inf coerce to `null`. `renders=True` (or a format list) optionally embeds
  rendered strings. `RegtableResult.save()` and `regtable(..., filename=...)`
  now recognise the `.json` extension.
- **Docs** —
  [`docs/guides/exporting-regression-tables.md`](docs/guides/exporting-regression-tables.md):
  single- and multi-model export across all six formats, the agent-native
  payload, journal templates, multi-panel / `Collection` containers, and a
  Stata (`esttab` / `estout` / `outreg2`) and R (`modelsummary` / `stargazer` /
  `fixest::etable` / `texreg`) cross-reference table. Every code snippet was
  executed to verify it runs. `migration-from-r.md` corrected:
  `sp.regress` returns `EconometricResults` (not `CausalResult`) and the
  `etable` mapping points at `sp.regtable`.
- **`RegtableResult.from_dict()`** — the inverse of `to_dict()`. The payload
  now carries a `render_spec` block (`fmt` / `alpha` / `panel_sizes` /
  `add_rows` / `keep` / `drop` / `order` / `se_label`) so a serialised table
  reconstructs and re-renders byte-identically across text/LaTeX/HTML/Markdown
  for the common feature set (exotic `multi_se` / `eform` / `column_spanners` /
  `tests` are documented as not surviving the round-trip). Makes the JSON a
  faithful cache, not just a snapshot.
- **Journal-grade LaTeX** — `to_latex(siunitx=True)` decimal-aligns numeric
  columns with `siunitx` v3 `S` columns (coefficients align on the decimal
  point; stars ride along as `\textsuperscript` via `table-space-text-post`;
  SE / text cells wrapped so they are not mis-parsed). `threeparttable=True`
  moves the footnotes into a `tablenotes` block; `siunitx_preamble=True` emits
  the required `\usepackage` hint. Unsupported regimes raise
  `NotImplementedError` rather than emit non-compiling LaTeX. The default
  LaTeX path is byte-identical. Threaded through `EconometricResults.to_latex`.
- **`sp.coefplot_tikz()`** — `pgfplots` / TikZ coefficient forest plot (the
  LaTeX-native counterpart to `sp.coefplot`, whose `(fig, ax)` already gives
  PNG/PDF via `fig.savefig`): one `\addplot` series per model with horizontal
  CI error bars, reversed y-axis, dashed zero line; `coef_labels` / `level` /
  `standalone` options. Auto-registered.
- **Export-surface contract (`tests/test_export_surface_contract.py`)**
  — 55 parametrized checks asserting `sp.regtable(r)` consumes registered
  result-class outputs (`EconometricResults` / `CausalResult` / `PanelResults` /
  `FrontierResult`) and round-trips, so a future non-exportable result fails
  loudly. Guide §7–§8 document coefplots and the table/non-table boundary.
- **`Collection.to_dict()` / `.to_json()` and `PaperTables.to_dict()` /
  `.to_json()`** — agent-native serialisation extended to the multi-table
  containers. Each regression-table item/panel reuses
  `RegtableResult.to_dict()` (and stays `from_dict`-round-trippable);
  DataFrame items round-trip through `DataFrame.to_json` (NaN → null);
  text/heading carry their string. Both `to_json()` emit strict JSON.

### Fixed

- **Two dangling enrichment citations** — `_CITATIONS_BY_TOOL` referenced a
  mistyped `dechaisemartin2020twoway` (corrected to the existing
  `dechaisemartin2020two`) and a `cattaneo2015randomization` key absent from
  `paper.bib` (added, verified via De Gruyter DOI 10.1515/jci-2013-0010 and
  the rdpackages reference). refs verified via aeaweb.org + RePEc
  (de Chaisemartin & D'Haultfœuille 2020) and degruyterbrill.com + rdpackages
  (Cattaneo, Frandsen & Titiunik 2015).
- **Four dangling agent-card alternatives** — `rif_regression`
  (→ `rif_decomposition`), `sp.ope_ipw` (→ `sp.ipw`), and two `fixest_in_r`
  external references (→ `hdfe_ols`) pointed at non-existent functions; an
  agent following them would have hit `AttributeError`.

## [1.16.0] — 2026-05-29

### ⚠️ Correctness fix

- **`sp.qreg` Powell sandwich SE was wrong by a factor of √n** — every
  pre-fix p-value, z-statistic, and confidence interval emitted by
  `sp.qreg` was unusable. The closed-form Koenker (2005, eq. 3.7) iid
  kernel sandwich is `V = τ(1−τ) / f̂(0)² · (X'X)⁻¹`. `_qreg_se` had an
  extra factor of `n` in the denominator (`/ (n * f0**2)`), so the
  reported SE was the correct SE divided by √n — on n = 500 the SE
  was ~20× too small. The fix removes the spurious `n`. After the fix
  the three-way parity at the median tolerance (`tests/r_parity/40_qreg`)
  matches `quantreg::rq` within 1.4–6.8 % and Stata `qreg` within 2.9 %,
  consistent with the documented kernel-vs-Koenker-Bassett SE method
  gap. **Action**: any analysis that previously used `sp.qreg` SE,
  z-statistic, p-value, or CI must be re-run; point estimates are
  unaffected. See [`MIGRATION.md` § sp-qreg-se-fix](MIGRATION.md#sp-qreg-se-fix)
  for the per-call impact and rerun recipe.

- **`sp.xtabond` (Arellano-Bond difference GMM) point estimates AND SEs
  were wrong** — finding #12. The estimator built a *flat, fixed* set of
  lagged-level instrument columns (`gmm_lags=(2,5)`) and then dropped
  every row missing any of them, which on a short panel discards most of
  the sample; it also used `W = (Z'Z)⁻¹` as the one-step weight. The
  correct Arellano-Bond estimator uses a **block-diagonal** GMM
  instrument matrix (every available deeper lag is a period-specific
  moment, missing lags filled with 0, *no* rows dropped) and the
  one-step weight `W = (Σᵢ Zᵢ'H Zᵢ)⁻¹` where `H` carries the MA(1)
  structure of the differenced errors (2 on the diagonal, −1 on the
  first off-diagonals). On the parity DGP the old code gave
  `β_{y₋₁}=0.264 (se 0.224)` vs Stata's `0.391 (se 0.046)` — a 48 %
  estimate gap and an 80 % SE gap. After the rewrite the one-step robust
  estimates match Stata's `xtabond y x, lags(1) vce(robust)` to **machine
  precision** (`tests/r_parity/50_xtabond`, rel ≈ 1e-15 on both β and
  SE). The default `gmm_lags` is now `(2, None)` (all available deeper
  lags, matching Stata's default; pass an explicit max to cap). Two-step
  GMM now applies the Windmeijer (2005) finite-sample SE correction.
  **Action**: re-run any analysis that used `sp.xtabond` — both point
  estimates and SEs change. See
  [`MIGRATION.md` § sp-xtabond-fix](MIGRATION.md#sp-xtabond-fix).

- **`sp.xtabond(method='system')` / `sp.panel(method='system')` now raise
  `NotImplementedError`** instead of returning an unvalidated (and, after
  the difference-GMM rewrite, badly distorted) estimate. Proper
  Blundell-Bond system GMM requires a stacked level equation and its own
  Stata `xtdpdsys` parity reference, which is planned for a future
  release. **Action**: use `method='difference'` (Arellano-Bond), now
  validated to machine precision.

### Added — Parity coverage expansion (2026-05-28 session)

- 15 net-new parity modules (`tests/r_parity/{37–51}_*`) covering
  `sp.ppmlhdfe`, `sp.drdid`, `sp.arima`, `sp.qreg`, `sp.tobit`,
  `sp.nbreg`, `sp.heckman`, `sp.mlogit`, `sp.ologit`, `sp.clogit`,
  `sp.probit`, `sp.oprobit`, `sp.xtabond`, `sp.newey`, and a 3-FE PPML
  variant. The 3-way Track A table
  (`tests/r_parity/results/parity_table_3way.md`) covered 50
  R-joined modules versus 36 previously at that checkpoint, with a Stata
  reference for 43 versus 21. The current source snapshot supersedes that
  checkpoint with 64 R-joined modules, 61 Stata references, and a materialized
  `50_xtabond` R reference through `plm::pgmm`. The expansion surfaced the qreg and newey
  SE fixes above and further P1/P2 findings recorded in
  `tests/r_parity/PARITY_SESSION_2026-05-28.md`.

### Fixed

- Cleaned up external-review follow-ups: removed two uncited duplicate BibTeX
  entries that caused editorialbot DOI suggestions, aligned the AKM
  shift-share citation key / DOI metadata, and refreshed v1.15.6 wording in
  reviewer-facing docs and README release callouts.
- `tools/audit_citations.py` now treats transient HTTP/socket/SSL timeouts as
  unresolved citation lookups instead of leaking Python tracebacks.
- `tests/r_parity/36_mediation.py` referenced `model_info["n_boot"]`,
  but `sp.mediation`'s schema renamed this to `n_boot_requested` /
  `n_boot_successful` / `n_boot_failed`. The parity script crashed
  before producing JSON; pinned it to the new key.

## [1.15.6] — 2026-05-24

### Changed — Co-authorship, software-journal submission readiness

- Added **Scott Rozelle** as co-author across all package metadata:
  `pyproject.toml`, `src/statspai/__init__.py` (`__author__`), `CITATION.cff`,
  `.zenodo.json`, `mkdocs.yml`, the package citation templates in
  `src/statspai/_citation.py` (BibTeX / APA / plain), and the README BibTeX
  snippets (English and Chinese).
- ⚠️ **Downstream-facing rename**: unified the package BibTeX key to
  **`wang2026statspai`** (CLAUDE.md §10 `lastnameYEARkeyword` convention).
  Previous keys emitted or documented in earlier versions
  (`wang_statspai_2026`, `wang_rozelle_statspai_2026`, `statspai2026software`,
  bare `statspai`) are removed in favor of a single canonical key. Downstream
  `.tex` files that cite the previous key need a one-line rename to
  `\cite{wang2026statspai}`. The impact surface is small — only users who
  literally copied the previous BibTeX entry into their own `.bib` are
  affected; users who regenerate via `sp.citation("bibtex")` get the new key
  automatically.
- `sp.citation("bibtex")` now emits the unified key and the updated author
  list. `sp.citation("apa")` and `sp.citation("plain")` already reflected the
  co-author; both surfaces now also carry the 1.15.6 version string.
- `CITATION.cff` `version` / `date-released` bumped to `1.15.6` / `2026-05-24`.

### Added — reviewer-facing documentation

- reviewer guide — install, smoke test, representative offline
  examples, targeted tests, and build check, intended as a short reviewer
  path. All five smoke-test API calls are verified against the current
  registry (`ivreg`, `callaway_santanna` + `aggte`, `rdrobust`, `synth`,
  `describe_function` / `function_schema`).
- validation dossier — project status, registry counts,
  validation tracks (R-parity / Stata-parity / reference-parity / Monte Carlo
  coverage / snapshot tests / citation audits), parity anchors, research-use
  statement (working-paper use; no published peer-reviewed article yet),
  open-core / commercial-downstream disclosure (StatsPAI Inc. + CoPaper.AI),
  and reproducible-check commands.
- Both pages added to the MkDocs navigation.

### Changed — `paper.md` (software-journal manuscript)

- Repo URL casing corrected to canonical `StatsPAI`; added Zenodo archive
  reference (`@wang2026statspai`).
- Research-impact paragraph rephrased to match the actual current state:
  StatsPAI is used in working-paper workflows connected to Stanford REAP; no
  peer-reviewed research article using the package has yet been published.
- AI Usage Disclosure rewritten to spell out exactly what generative AI was
  used for (code generation, refactoring, test scaffolding, documentation
  drafting, manuscript copy-editing), to note that exact model identifiers
  were not retained for all exploratory sessions, and to confirm that
  generative AI will not be used to produce substantive responses to journal
  editors or reviewers.
- Acknowledgements split: explicit **Author Contributions** subsection
  attributing roles to each author, and an open-core / commercial-downstream
  disclosure (StatsPAI Inc. is the legal entity; CoPaper.AI is a commercial
  downstream product that may call the MIT-licensed StatsPAI package; the
  package itself remains permanently open source under MIT).
- `paper.bib` adds the `wang2026statspai` software entry pointing at the
  Zenodo concept DOI.

## [1.15.5] — 2026-05-21

### Added — Agent-card coverage ratchet and baseline enrichment

- Added `scripts/agent_card_coverage.py`, `docs/agent_cards_spec.md`, and
  `tests/test_agent_card_coverage.py` to make raw curated agent-card metadata
  measurable and CI-ratcheted. The committed floor tracks 15 counters across
  Tier-B, Tier-A, Tier-S, per-field coverage, and certified / validated
  evidence counts.
- Added generated `src/statspai/_baseline_cards.py` plus
  `scripts/gen_baseline_cards.py` to fill empty Tier-B fields from docstrings
  without overwriting curated registry entries. The baseline pass lifts tags to
  100% of the then-current v1.15.5 registry and keeps examples / references
  limited to mechanically extracted, auditable content.
- Added `FunctionSpec.inherits_from` and inherited agent-card rendering for
  canonical estimator variants. Child specs keep their own descriptions,
  examples, parameters, references, validation status, and limitations, while
  sharing parent assumptions, preconditions, failure modes, alternatives, and
  `typical_n_min` where appropriate.

### Changed — Registry and documentation refresh

- Expanded validation evidence seeds for tested long-tail estimators so the
  agent registry distinguishes stable APIs from functions with explicit unit,
  regression, parity, or reference-test coverage.
- Refreshed registry count, module statistics, and agent-platform positioning
  for the then-current v1.15.5 public surface.
- Updated DiD and agent-facing docs to mark `continuous_did(method='cgs')` and
  `did_multiplegt_dyn` as experimental MVP paths rather than fully
  paper-parity estimators.

## [1.15.4] — 2026-05-18

### Added — Auto-CJK font fallback on import

- `import statspai as sp` now auto-registers a detected CJK font (PingFang
  SC / Microsoft YaHei / Noto Sans CJK / SimHei / Source Han Sans / …) as
  a per-glyph fallback in matplotlib's `font.family` list. **Chinese text
  in plots renders correctly without calling `sp.use_chinese()`** on any
  system that already has a CJK font installed (default on macOS / Windows /
  most modern Linux desktops).
- Safe-by-design: the user's primary family stays at `font.family[0]`, so
  Latin text is rendered with the original primary font (DejaVu Sans /
  Helvetica / Times New Roman / …) — **no visual change for English-only
  plots**. `axes.unicode_minus` is untouched, so the minus sign on tick
  labels stays as the proper U+2212 from the Latin primary.
- Opt-out: set environment variable `STATSPAI_NO_AUTO_CJK=1` before
  importing statspai. The explicit `sp.use_chinese()` entry point still
  exists for users who want a specific font or who need primary-font
  control (e.g., `sp.use_chinese('serif')`).
- Mechanism: `font.family` becomes `['sans-serif', '<best CJK sans>',
  '<best CJK serif>']` (or whatever family was set). matplotlib 3.6+
  per-glyph fallback walks this list, so CJK glyphs missing from the
  primary fall through to the appended CJK font without affecting Latin
  glyphs. Empirically required vs. appending to `font.sans-serif`, which
  does not trigger fallback in matplotlib 3.10.
- New tests: [`tests/test_auto_cjk_fallback.py`](tests/test_auto_cjk_fallback.py)
  covers the 7-point behavior contract (primary preserved, unicode_minus
  preserved, family-specific lists preserved, Chinese renders without
  warnings, user override wins, idempotent, env var opt-out).

## [1.15.3] — 2026-05-17

### Fixed — PyPI long-description hero banner

- The v1.15.2 PyPI project page rendered the hero banner as a broken
  image. Root cause: [`README.md`](README.md) and
  [`README_CN.md`](README_CN.md) referenced
  `docs/logo/readme-1.png` with a **repo-relative** path, which GitHub
  resolves against the rendered tree but PyPI's long-description
  renderer has no base URL for and therefore leaves as a 404. Both
  READMEs now point at the absolute raw GitHub URL
  `https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/main/docs/logo/readme-1.png`
  so the banner loads on PyPI / TestPyPI / Open Source Insights /
  any other off-GitHub README renderer.
- **No code changes.** All shipped artifacts have the same module
  hashes as v1.15.2 except for the regenerated long-description
  metadata baked into the wheel + sdist.

## [1.15.2] — 2026-05-17

### Headline

Patch release on top of v1.15.1. **No estimator numerical paths change.**
Three independent hardening tracks land together:

1. **Agent-native infrastructure** — `sp.agent.mcp_server` is now strict-
   JSON-clean over the wire (no `NaN` / `Infinity` literals reaching
   Claude Desktop / RFC 8259 parsers), agent schema metadata extraction
   is more complete, and a new `text` extra makes
   `sentence-transformers` an explicit opt-in for the v1.6 `causal_text`
   surface instead of a soft import surprise.
2. **`sp.replicate` dual-track** — Card (1995), Abadie-Diamond-
   Hainmueller (2010), Lalonde (1986) / DW (1999), and Lee (2008)
   replications are promoted from single-track stubs to full
   **classic + modern** recipes that ship with the original public-
   domain CSVs and pinned golden numbers.
3. **Release packaging** — wheel smoke tests now **fail loudly** on
   import error, `py.typed` ships in the wheel for downstream
   `mypy --strict` consumers, the result `_repr_html_` path escapes
   user-controlled cell content (notebook XSS-safety), and the
   `formulaic` dependency that `sp.spatial` parses formulas with is
   declared explicitly instead of relying on `linearmodels`'s
   transitive resolution.

### Added — `text` optional extra

- New `[project.optional-dependencies] text = ["sentence-transformers>=2.2.0"]`
  in [`pyproject.toml`](pyproject.toml). The v1.6 `sp.causal_text` MVP
  used to lazy-import `sentence_transformers` and raise an opaque
  `ImportError` on first call. `pip install statspai[text]` now wires
  the dependency explicitly; the lazy import still triggers a clear
  pointer to the extra when missing.

### Added — `sp.replicate` dual-track guides for four canonical papers

- **Card (1995)** — proximity-to-college IV for returns to schooling.
- **Abadie, Diamond & Hainmueller (2010)** — California Proposition 99
  synthetic-control.
- **Lalonde (1986) / Dehejia-Wahba (1999)** — NSW + PSID-1 (MatchIt
  subset, `n=614`) propensity-score matching.
- **Lee (2008)** — US Senate RD (`n=1390`) bandwidth-selected jump.

  Each entry now ships a **classic track** (the estimator the paper
  used: 2SLS for Card, weighted synthetic-control for Abadie, naive
  OLS / adjusted OLS / 1:1 NN PSM for Lalonde, local-linear CCT RD
  for Lee) **and a modern track** (DML PLR + entropy balancing,
  bias-corrected robust RD, multi-method synth-compare). Real CSVs
  land under [`src/statspai/datasets/data/`](https://github.com/brycewang-stanford/StatsPAI/tree/main/src/statspai/datasets/data/)
  (`card_1995.csv`, `california_prop99.csv`, `lalonde_matchit.csv`,
  `lee_2008_senate.csv`) and `sp.datasets.nsw_lalonde` /
  `sp.datasets.lee_2008_senate` gain a `simulated=False` real-data
  branch with published-paper benchmarks exposed via `df.attrs`. The
  Lalonde classic track now reproduces DW (1999) Table 3-4 within a
  $5 drift tolerance; the Lee CCT track returns Conv `7.414` and
  bias-corrected robust `7.507` (SE `1.741`, `h=17.754`), matching the
  R `rdrobust` reference. All 13 BibTeX keys cited across the four
  entries are verified in [`paper.bib`](paper.bib).

### Fixed — MCP wire format is strict-JSON-clean

- [`sp.agent.mcp_server`](src/statspai/agent/mcp_server.py) used to
  serialise responses with `json.dumps(..., default=_json_default)`,
  which **does not** intercept native Python `float('nan')` /
  `float('inf')` — those become the non-standard literals `NaN` /
  `Infinity` in the JSON output, which RFC 8259 parsers (including
  Claude Desktop's `JSON.parse`) reject with errors like
  `"No number after minus sign"`. Responses now pass through
  `_clean_floats` (recursively replaces `NaN` / `±Infinity` with
  `null` across `dict` / `list` / `tuple` containers) and serialise
  with `allow_nan=False`, so the server can never emit a JSON token a
  strict parser refuses. Covered by 273-line regression suite
  [`tests/test_mcp_nan_inf.py`](tests/test_mcp_nan_inf.py).
- Agent schema metadata extraction (`sp.function_schema`,
  `sp.describe_function`) now surfaces more signature detail for
  registry entries built from auto-introspection.
- Stability-tier audit (`scripts/stability_audit.py`) accounts for
  evidence files more precisely; new
  [`tests/test_agent_schema.py`](tests/test_agent_schema.py) locks
  the schema metadata fields agents rely on.

### Fixed — result HTML escaping (notebook XSS-safety)

- [`CausalResult._repr_html_`](src/statspai/core/results.py) (and the
  surrounding rich-display helpers) now route every user-derived cell
  through `html.escape`. Previously, any string column whose contents
  contained `<` / `>` / `&` / `"` would interpolate raw into the
  rendered HTML, opening a path for notebook XSS when a result was
  displayed in Jupyter / VS Code / nbviewer. New regression test:
  [`tests/test_results_html_escape.py`](tests/test_results_html_escape.py).

### Fixed — release packaging hygiene

- `pyproject.toml` bumps the build requirement to
  `setuptools>=77.0.0` and migrates `license = {text = "MIT"}` to the
  modern PEP 639 `license = "MIT"` + `license-files = ["LICENSE"]`
  pair. Drops the deprecated `License :: OSI Approved :: ...`
  classifier path implicitly.
- [`MANIFEST.in`](MANIFEST.in) now includes `src/statspai/py.typed`
  and the sdist test fixtures so `pip install --no-binary :all:` and
  `mypy --strict` both behave correctly on the published artifacts.
- [`.github/workflows/build-wheels.yml`](.github/workflows/build-wheels.yml)
  and [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml):
  wheel smoke tests now **fail the job** on `ImportError` instead of
  swallowing it as a warning. Releases that silently ship a broken
  wheel are no longer possible from a green CI run.
- New explicit dependency `formulaic>=0.6.0` in `dependencies`
  (`sp.spatial.*` parses Wilkinson formulas through it; relying on
  `linearmodels`'s transitive resolution broke when downstream users
  pinned older `linearmodels`).

### Docs

- Software-journal submission [`paper.md`](paper.md) is rewritten for the Scott
  Rozelle review pass — tighter scope statement, cleaner schema
  description, explicit AI-use disclosure, 12 May 2026 submission
  date. Cited bibliography entries in [`paper.bib`](paper.bib) are
  refreshed to match.
- README / README\_CN add the hero banner image
  ([`docs/logo/readme-1.png`](docs/logo/readme-1.png)).
- Track-C performance comparison table
  ([`tests/perf/results/perf_table.tex`](tests/perf/results/perf_table.tex))
  switches to `\scriptsize` with package-name macros and a
  direction-aware "Winner" column; log-log figure regenerated to match.

### Internal

- Perf-benchmark harness factored — new
  [`tests/perf/_common.py`](tests/perf/_common.py) shared utilities;
  `tests/perf/05_feols_jax_bootstrap_bench.py` rewritten on top.
- Full-suite validation snapshot refreshed in
  [`test_results_full_suite.md`](test_results_full_suite.md).

## [1.15.1] — 2026-05-07

### Headline

Patch release on top of v1.15.0 preparing the public PyPI cut. Existing
estimator defaults are preserved. The only new runtime path is opt-in:
`sp.rdrobust(..., bwselect='cct')` delegates to the official
`rdrobust>=1.3` Python package for bit-equal R `rdrobust::rdrobust`
replications. The default `bwselect='mserd'` remains StatsPAI's
internal MSE-optimal recipe.

The release notes and README now also document the negative-binomial
count-regression implementation that is already exposed through
`sp.nbreg`, `sp.xtnbreg`, and `sp.menbreg`. This is documentation /
release-packaging work plus the `1.15.0 → 1.15.1` version bump; it does
not change the negative-binomial numerical path.

### Docs — negative-binomial regression implementation note

- `sp.nbreg` is a log-link MLE for overdispersed non-negative count
  outcomes. The default `dispersion="mean"` path is NB2,
  `Var(Y|X)=mu + alpha * mu^2`; `dispersion="constant"` switches to
  NB1, `Var(Y|X)=mu * (1 + delta)`.
- The optimizer starts from a Poisson IRLS fit. It then alternates
  NB-weighted IRLS updates for the coefficient vector with scalar
  profile-likelihood optimization for the dispersion parameter on the
  log scale (`alpha` for NB2, `delta` for NB1).
- Inference uses the NB working-weight bread. `robust="robust"` /
  `"hc0"` / `"hc1"` select sandwich SEs, and `cluster=` selects
  cluster-robust SEs. `irr=True` reports incidence-rate ratios by
  exponentiating coefficients and delta-method SEs.
- Offsets and exposure are supported (`offset=` supplies a log offset;
  `exposure=` is logged internally). Diagnostics include log-likelihood,
  AIC/BIC, pseudo-R2, fitted values / residuals, and a one-sided
  likelihood-ratio test of the dispersion path against Poisson using
  the standard 50:50 chi-bar-square mixture.
- Formula fixed effects such as `y ~ x | id` are implemented by
  explicit dummy expansion. That choice is transparent and compatible
  with the MLE path, but it is intended for moderate-cardinality panels;
  high-cardinality HDFE count models should use the Poisson/PPML
  surface (`sp.fepois`, `sp.ppmlhdfe`) unless a full dummy NB fit is
  really intended.
- `sp.xtnbreg(model="fe")` wraps `sp.nbreg` with entity/time fixed
  effects and defaults `cluster=` to the entity id. `model="pooled"`
  strips the panel fixed-effect part. `model="re"` dispatches to
  `sp.menbreg`, the random-intercept NB2 GLMM, and returns the
  multilevel `MEGLMResult`.

### Added — R-parity opt-in for `sp.rdrobust`

- **New `bwselect='cct'`** in [`sp.rdrobust`](src/statspai/rd/rdrobust.py)
  delegates the entire estimation (bandwidth selection + bias-corrected
  inference) to the official `rdrobust>=1.3` Python port (Calonico,
  Cattaneo & Titiunik 2014). This guarantees **bit-equal alignment with
  R `rdrobust::rdrobust`** for users who need exact replication of
  CCT-2014 published numbers — for example the canonical Lee/CCT Senate
  case where R returns `Conv = 7.4141 / Robust = 7.5065 / h = 17.754`.
  The internal `bwselect='mserd'` (default) is **kept unchanged for
  backward compatibility** — it uses StatsPAI's own MSE-optimal recipe
  which can drift from R's `rdbwselect` by up to ~70% on certain
  datasets (documented in
  [`tests/orig_parity/results/parity_table_orig.md`](tests/orig_parity/results/parity_table_orig.md)
  row 52, module `05_lee_original`).
- Install with `pip install statspai[rd-cct]` (adds the official
  `rdrobust>=1.3` dependency).  Calling `bwselect='cct'` without it
  raises a clear `ImportError` pointing to the install command.
- See [MIGRATION.md](MIGRATION.md#sp-rdrobust-bwselect-cct-r-parity-opt-in)
  for guidance on when to switch from `'mserd'` to `'cct'`.

### Tests — `did::aggte` parity lock

- Added [`TestAggteRParity`](tests/external_parity/test_published_replications.py)
  in `tests/external_parity/test_published_replications.py`. Asserts
  `sp.aggte(type='simple')` is bit-equal (≤1e-10) with R `did::aggte`
  recorded in [`tests/orig_parity/results/02_mpdta_original_R.json`](tests/orig_parity/results/02_mpdta_original_R.json),
  and `type='dynamic'` matches R's published vignette output to 1e-3.
  Prevents future refactors from silently drifting away from R.
- Added [`TestCCTDelegationParity`](tests/external_parity/test_published_replications.py)
  and an `ImportError`-guarded test that pin the new `bwselect='cct'`
  delegation to R `rdrobust` Senate-replication numbers (Conv 7.4141,
  Robust 7.5065, h=17.754, 1e-3 tolerance).

### Internal

- Added `[project.optional-dependencies] rd-cct = ["rdrobust>=1.3"]` to
  [`pyproject.toml`](pyproject.toml).
- `sp.datasets.list_datasets()` now returns six columns
  (added `paper_original` column to honestly distinguish the published
  paper number from the simulated-replica's actual estimator output).

## [1.15.0] — 2026-05-05

### Docs — `sp.dml_panel` citation correction

- ⚠️ **Docs-only correction** — `sp.dml_panel` (originally shipped in
  v1.7) was attributed in its docstring, registry entry, README blurb,
  and CHANGELOG release note to *"Semenova & Chernozhukov (2023)
  Econometrics Journal 26(2), Debiased Machine Learning of Conditional
  Average Treatment Effects and Other Causal Functions."* That
  citation is **fabricated**: independent verification via Crossref
  and the Oxford ECTJ issue TOC confirms no Semenova or Chernozhukov
  paper appears anywhere in *Econometrics Journal* 26(2) (May 2023),
  and the cited title in fact belongs to Semenova & Chernozhukov
  **(2021)** *ECTJ* **24(2)** 264-289 (DOI 10.1093/ectj/utaa027) — a
  paper on CATE / debiased ML for causal functions, unrelated to
  long-panel PLR with fixed effects.
- The estimator's actual reference is **Clarke, P. S. & Polselli, A.
  (2025).** *"Double Machine Learning for Static Panel Models with
  Fixed Effects."* *The Econometrics Journal* **29(1)** 69-86, DOI
  [10.1093/ectj/utaf011](https://doi.org/10.1093/ectj/utaf011),
  arXiv:2312.08174. The paper specifies the within-group / first-
  difference transform, block-k-fold cross-fitting that allocates
  each unit's full time series to a single fold, and cluster-robust
  variance at the unit level — point-for-point match with the
  StatsPAI implementation. Companion Stata package: `xtdml`.
- `sp.synth(method='cluster')` method-citations registry: ClusterSC
  second-author surname corrected (was a misattribution; now matches
  the arXiv 2503.21629 author list — Rho, Tang, Bergam, Cummings,
  Misra). `paper.bib` was already correct; the typo only lived in
  `src/statspai/synth/report.py`.
- Updated callsites: [`paper.bib`](paper.bib) (new
  `clarke2025double` entry), [`src/statspai/dml/panel_dml.py`](src/statspai/dml/panel_dml.py)
  (module docstring + within-transform comment), [`src/statspai/dml/__init__.py`](src/statspai/dml/__init__.py)
  (lazy-export tag), [`src/statspai/registry.py`](src/statspai/registry.py)
  (FunctionSpec description + reference field), [`README.md`](README.md)
  (Long-panel Double-ML row), [`src/statspai/synth/report.py`](src/statspai/synth/report.py)
  (ClusterSC author list), and the historical v1.7 entry below
  (annotated, not silently rewritten). No code logic, numerical path,
  API signature, or test changed — pure citation correction.
- Refs verified via Crossref (DOIs `10.1093/ectj/utaf011` and arXiv
  `2503.21629`) and OpenAlex.

### Docs — v1.14 GPU sprint follow-up

- JSS manuscript (`Paper-JSS/manuscript/`, gitignored — local working
  copy) gets a substantive expansion of §5.6 *Performance backbone*
  documenting `sp.fast.feols_jax`, `sp.fast.feols_jax_bootstrap` (with
  the score-formulation derivation for wild and wild-cluster variants
  and the Cameron-Gelbach-Miller 2008 attribution added to
  `jss-bib.bib`), the Rust `cluster_meat` kernel, and `sp.iv(absorb=...)`.
  §6 originally sketched an accelerator-benchmark table for
  `sp.fast.feols_jax_bootstrap`, but the JSS source snapshot now relies
  only on packaged measurements. The active Track C table is generated
  from measured CPU benchmarks under `tests/perf/`, and GPU/JAX timing
  remains an opt-in engineering benchmark until a dedicated accelerator
  run is packaged as evidence.
- Software-journal bullet 5 in `paper.md` simplified from a four-paragraph
  exposition to a single tight paragraph that cross-references the
  JSS companion paper for full architecture detail.
- Note on version numbering: the `pyproject.toml` version moved from
  1.14.0 (GPU sprint cut, commit `a87d788`) to 1.15.0 (RDD polish
  cut) within a single day. Both `[1.14.0]` and `[1.15.0]` entries
  below remain historically correct for the work each release
  contained; no retroactive renaming is needed. **PyPI publishes
  `1.13.1 → 1.15.0` directly — `1.14.0` was an internal cut that was
  never released to PyPI and is recorded here for git / CHANGELOG
  history only.**

### Headline

Five pushes in this cycle. First, an IV-module polish to the post-2022
reporting standard (the `sp.iv.iv_diag` bundle, see below). Second, a
synthetic-control polish pass: supported synthetic-control estimators in
that release now have a publication-oriented table-export pipeline, the trajectory
and gap plots gain prediction-interval / pre-RMSPE ribbon options
following Cattaneo, Feng and Titiunik (2021, *JASA* 116, DOI
10.1080/01621459.2021.1979561) and Cattaneo, Feng, Palomba and Titiunik
(2025, *JSS* 113(1), DOI 10.18637/jss.v113.i01), and the SDID schema
is canonicalised end-to-end so `sp.synth_report(method='sdid', ...)`
produces a full Markdown / text / LaTeX report rather than a row of
N/As. Seven 2022–2025 SCM citations were added to `paper.bib`,
each verified independently via Crossref / arXiv (refs verified
via `crossref` + `arxiv`). Third, a decomposition-module polish — see
the dedicated section below. Fourth, a **ML+causal polish wave (v1.15)**
covering DML / meta-learners / causal forests / causal discovery /
policy learning / mediation — see the dedicated section below. Fifth,
an **RDD module polish (v1.15)** to the 2018–2026 frontier with three
new estimators (`sp.rd_flex`, `sp.rd_bias_aware_fuzzy`, `sp.rd_discrete`),
three reporting helpers (`sp.rd_dashboard`, `sp.rd_compare`,
`sp.rd_robustness_table`), an `sp.rdrobust` polish pass with the CCT-2018
`rho` parameter and discrete-RV / weak-first-stage warnings, and a
`sp.rdplotdensity` upgrade to the Cattaneo-Jansson-Ma (2020) boundary-
adaptive density estimator — see the dedicated section below.

### Added — ML+causal polish (v1.15)

- **DML-OVB sensitivity analysis** (`sp.dml_sensitivity`,
  `DMLSensitivityResult`) implementing the Chernozhukov–Cinelli–
  Newey–Sharma–Syrgkanis (2022) "Long Story Short" framework
  (NBER WP 30302; arXiv:2112.13398). Returns the robustness value
  RV_q (strength of confounder needed to shrink the estimate to
  zero), the significance-loss value RV_{q,α}, scenario bias
  bounds for user-specified (cf_y, cf_d), benchmark-covariate
  comparisons, and a `plot()` rendering bias contours over the
  (cf_d, cf_y) grid à la R `sensemakr`. Refs verified via NBER + arXiv.
- **DML diagnostics bundle** (`sp.dml_diagnostics`, `DMLDiagnostics`)
  bundles overlap (propensity histogram for IRM; |D-residual|
  distribution for PLR), score density (with N(0,σ̂²) overlay and
  Q-Q plot), residual-balance check (corr(X_k, Ỹ) and corr(X_k, D̃)
  for each covariate), and an orthogonality-score test in a single
  2×2 publication-style panel matching DoubleML's defaults
  (Bach–Kurz–Chernozhukov–Spindler–Klaassen 2024, *JSS* 108(3),
  DOI 10.18637/jss.v108.i03).
- **Backbone-agnostic CATE evaluation** (`sp.cate_eval`,
  `CATEEvalResult`) computing Yadlowsky–Fleming–Shah–Brunskill–
  Wager (2025) RATE / AUTOC / Qini with closed-form influence-
  function SEs for *any* CATE array (meta-learner, BCF, conformal-
  CATE, neural-CATE), so the metric is decoupled from the forest
  backbone. JASA 120(549), DOI 10.1080/01621459.2024.2393466
  (arXiv:2111.07966). Verified via Crossref + arXiv.
- ⚠️ **Correctness fix** — `forest.CausalForest.best_linear_projection`
  is rewritten to use the Semenova–Chernozhukov (2021) AIPW
  pseudo-outcome Γ_i with HC1 standard errors. The previous
  implementation regressed the plug-in CATE estimate on X with
  naïve OLS SEs, which was anti-conservative in finite samples.
  *Econometrics Journal* 24(2): 264–289, DOI 10.1093/ectj/utaa027.
  Users who relied on the prior BLP SEs should re-fit and report
  the new HC1 numbers.
- ⚠️ **Correctness fix** — `mediation.mediate` no longer silently
  substitutes the point estimate for failed bootstrap replicates
  (which artificially shrunk SEs). Each failure now triggers up to
  five retry draws; remaining failures are dropped, and a
  `RuntimeWarning` fires if more than 10% of replicates fail. The
  result's `model_info` exposes `n_boot_requested`,
  `n_boot_successful`, `n_boot_failed`, and `boot_failure_rate`
  for audit. SEs estimated under heavy bootstrap failure on prior
  versions should be regenerated.
- **OPE namespace deduplication** — `sp.policy_learning.OPEResult`
  is now an alias for the canonical `sp.ope.estimators.OPEResult`,
  so `isinstance(sp.direct_method(X, A, R, π), sp.OPEResult)` is
  True regardless of which entry point was used. The legacy
  `estimator` / `n_obs` attributes survive as properties on the
  unified class.
- **Causal-discovery graph visualization** — every result class
  (`LiNGAMResult`, `GESResult`, `FCIResult`, `ICPResult`,
  `PCMCIResult`, `LPCMCIResult`, `DYNOTEARSResult`) and the dict-
  shaped returns from `sp.notears` and `sp.pc_algorithm` (now
  promoted to a `DAGDict` thin subclass) expose a unified
  `.to_networkx()` / `.to_dot()` / `.plot()` / `.edge_list()` API.
  Module-level helpers `sp.causal_discovery.{to_networkx, to_dot,
  plot_dag, edge_list, shd}` work standalone on any adjacency
  matrix; `shd()` follows the Tsamardinos–Brown–Aliferis (2006)
  Structural Hamming Distance convention.
- **PolicyTreeResult promotion** — `sp.policy_tree` now returns a
  `PolicyTreeResult` (subclass of `dict` for full back-compat) with
  influence-function SE on the policy value and a 95% CI from the
  AIPW scores, plus a Graphviz-style `plot_tree()`, `summary()`,
  `to_latex()`, `to_excel()`, and `cite()` (Athey & Wager 2021,
  *Econometrica* 89(1)).
- **Mediation sensitivity plot upgrade** — `MediateSensitivityResult.plot()`
  now produces a publication-style ACME(ρ) curve with coloured fill
  for the {ACME>0} / {ACME<0} regions, annotated baseline, and
  explicit ρ-at-zero (the robustness threshold).
- **`CausalResult.to_word`** added alongside the existing `.to_latex`
  / `.to_excel`. Renders a publication-style three-block Word
  document (estimates / detail / notes) using the AER booktab styling
  helpers in `output/_aer_style.py`. Coverage: supported estimators that
  already returns `CausalResult` (DML, TMLE, BCF, mediate,
  conformal_cate, proximal.p2sls, matrix_completion, metalearners,
  hal_tmle, did, rd, synth) now has uniform LaTeX / Excel / Word
  export.
- **DTR + QTE test coverage** — `tests/test_dtr.py` (10 new tests)
  and `tests/test_qte.py` (7 new tests) close two zero-coverage
  modules flagged in the v1.13 audit. Tests verify (i) Q-learning
  exactly recovers the optimal terminal-stage rule under a linear
  blip, (ii) A-learning's terminal contrast aligns with the truth,
  (iii) `qte` and `qdid` recover the constant-shift / parallel-
  trends benchmarks of Firpo (2007) and Athey–Imbens (2006)
  respectively to within 0.30 absolute error at every quantile,
  and (iv) `distributional_te`'s CDFs are monotone with stochastic
  dominance in the right direction.
- **`tests/test_ml_causal_polish.py`** (22 new tests) covers all of
  the above end-to-end (BLP DR-score recovery, mediation bootstrap
  diagnostics, OPE isinstance, DAG viz, `PolicyTreeResult` contract,
  DML sensitivity / diagnostics, `cate_eval` direction, `to_word`
  integration).
- **Citation expansion** — 4 new bib entries added to `paper.bib`,
  each verified independently via NBER / arXiv / journal site:
  `chernozhukov2022long`, `semenova2021debiased`,
  `yadlowsky2025evaluating`, `bach2024doubleml`.

### Added — decomposition module polish (v1.15)

- **Yu–Elwert (2025) nonparametric causal decomposition**
  (`sp.yu_elwert_decompose`, dispatcher aliases `yu_elwert` / `cdgd`)
  — splits an observed group disparity into baseline, prevalence,
  effect, and selection mechanisms. Two estimators: a plug-in version
  that returns *exactly* zero residual by algebraic identity, and a
  doubly-robust `method="efficient"` augmented variant. Cluster-aware
  bootstrap inference, plot helper (`yu_elwert_mechanisms_plot`), and
  a per-component CI table. Aligned conceptually with the R `cdgd`
  package. Reference verified via Crossref:
  doi:10.1214/24-AOAS1990 (*Annals of Applied Statistics*, 19(1)
  821–845).
- **Unified `DecompResultMixin`** — every result class in
  `sp.decomposition` (Oaxaca, Gelbach, RIF, FFL, DFL, Machado–Mata,
  Melly, CFM, Fairlie, Bauer–Sinning, Yun, Kitagawa, Das Gupta,
  Subgroup / Source / Shapley inequality, GapClosing, Mediation,
  Disparity, Yu–Elwert) now exposes the same surface:
  `confint(alpha)`, `cite()` / `cite("bibtex_keys")`, `to_dict()`,
  `to_json()`, `to_excel(path)` (multi-sheet workbook), and
  `to_word(path)` (python-docx report) in addition to the existing
  `summary()` / `plot()` / `to_latex()`.
- **Plot polish** — `sp.decomposition.plots` now ships a unified
  Material-style palette (`DECOMP_PALETTE`) and a despined minimal
  grid via `apply_decomp_style`. New helpers: `forest_plot` (with
  significance shading), `mediation_forest`, and
  `yu_elwert_mechanisms_plot`. Existing plots (`detailed_waterfall`,
  `quantile_process_plot`, `dfl_plot`, `ffl_waterfall`,
  `gap_closing_plot`, `inequality_subgroup_plot`,
  `counterfactual_cdf_plot`, `rif_heatmap`) gain optional
  95% CI whiskers / ribbons whenever the result carries SEs.
- **Wild bootstrap** in `decomposition._common.wild_bootstrap_stat`
  with Rademacher and Mammen multipliers and cluster-aware multiplier
  sharing (Cameron–Gelbach–Miller 2008 style). `analytical_ci(point,
  se, alpha)` helper for two-sided normal intervals.
- **Citation expansion** — 15 new bib entries added to `paper.bib`,
  each verified via Crossref: `blinder1973wage`, `oaxaca1973male`,
  `neumark1988employers`, `cotton1988estimation`, `reimers1983labor`,
  `jann2008blinder`, `gelbach2016covariates`, `kline2011oaxaca`,
  `shorrocks1980class`, `cowell2007income`, `riosavila2020rif`,
  `kroger2021kitagawa`, `oaxaca2025meets`, `yu2025nonparametric`,
  `park2024choosing` (refs verified via Crossref).
- **Docs** — new family guide
  `docs/guides/decomposition_family.md`, wired into MkDocs nav under
  the v1.15 entry. Decomposition section in `paper.md` rewritten with
  inline citation keys.
- **Tests** — `tests/test_decomposition_polish.py` (14 new tests)
  covering the Yu–Elwert algebraic identity, bootstrap inference,
  dispatcher routing, plot smoke test, the unified mixin (cite /
  to_dict / to_excel / confint), and the wild bootstrap helper
  (Rademacher / Mammen / clustered).

### Added — synthetic control polish

- `sp.synth_to_latex(obj, ...)` — booktabs LaTeX for any
  `CausalResult` from `sp.synth(method=...)` or for a
  `SynthComparison` (side-by-side multi-method layout). Optional
  donor-weights panel, configurable significance stars, and a
  `\\hline` fall-back for non-booktabs documents.
- `sp.synth_to_markdown(obj, ...)` — pipe-table Markdown counterpart
  rendering on GitHub, in pandoc, and in static-site generators.
- `sp.synth_to_excel(obj, path, ...)` — multi-sheet workbook with
  Summary / Weights / Diagnostics / per-method Gap sheets. Soft
  dependency on `openpyxl`; raises an actionable `ModuleNotFoundError`
  with install hint when missing.
- `SynthComparison.to_latex()`, `.to_markdown()`, `.to_excel()` —
  thin object-method wrappers over the three exporters above.
- `sp.synthplot(..., pi_band=True)` — overlay the
  prediction-interval / conformal CI ribbon on the synthetic
  counterfactual when the result carries `period_results` with
  `pi_lower` / `pi_upper` columns (`sp.scpi`,
  `sp.conformal_synth`).
- `sp.synthplot(..., pre_band=True)` — overlay a
  $\\pm 1.96 \\times$pre-RMSPE noise envelope on the trajectory and
  gap plots (the convention popularised in Abadie, Diamond and
  Hainmueller 2010, *JASA*).
- Method-aware citations in `sp.synth_report()`: SDID, scpi,
  conformal, gsynth, augmented, matrix-completion, multi-outcome,
  cluster, sparse, fdid, BSTS, and penalised SCM each close with
  their own citation rather than the generic Abadie--Diamond--
  Hainmueller (2010) reference.
- SDID schema canonicalisation in `sp.synth_report()`: the report
  now backfills `n_pre_periods` / `n_post_periods` / `n_donors` /
  `pre_treatment_rmse` / `gap_table` from SDID-style keys
  (`T_pre`, `T_post`, `n_control`, `Y_obs`) so the Markdown / text
  / LaTeX output is uniform across SC variants.
- Seven new verified bib entries in `paper.bib`:
  Abadie & Cattaneo (2021, *JASA* 116(536), 1713–1715,
  DOI 10.1080/01621459.2021.2002600); Abadie & Vives-i-Bastida
  (2022, arXiv:2203.06279); Cattaneo, Feng, Palomba & Titiunik
  (2025, *JSS* 113(1), DOI 10.18637/jss.v113.i01); Liu, Wang & Xu
  (2024, *AJPS* 68(1), 160–176, DOI 10.1111/ajps.12723); Qiu, Shi,
  Miao, Dobriban & Tchetgen Tchetgen (2024, *Biometrics* 80(2),
  ujae055, DOI 10.1093/biomtc/ujae055); Clarke, Pailañir, Athey &
  Imbens (2024, *Stata Journal* 24(4), 557–598, DOI
  10.1177/1536867X241297914); Bottmer, Imbens, Spiess & Warnick
  (2024, *JBES* 42(2), 762–773, DOI 10.1080/07350015.2023.2238788).
- 17 new tests in `tests/test_synth_exports.py` covering single-result
  and comparison LaTeX / Markdown / Excel exports, the new plot
  options, and SDID-canonicalised reports.

### Added — IV polish (v1.15)

- `sp.iv.iv_diag(data, y, endog, instruments, exog, ...)` — modern IV
  reporting bundle. Returns an `IVDiagResult` containing:
  - 2SLS point estimate, analytic + pairs / wild Rademacher bootstrap
    SEs and CIs (cluster-aware) following Davidson--MacKinnon (2010)
    and Young (2022, *EER* 147, 104112);
  - Olea--Pflueger (2013, *JBES* 31(3), 358–369) robust effective F;
  - Lee--McCrary--Moreira--Porter (2022, *AER* 112(10), 3260–3290)
    `tF` adjusted critical value and tF-corrected confidence interval;
  - Anderson--Rubin (1949) / optional Moreira (2003) CLR / optional
    Kleibergen (2002) K weak-IV-robust confidence sets;
  - Kleibergen--Paap (2006, *J. Econometrics* 133, 97–126) rk LM and
    Wald F;
  - Conley--Hansen--Rossi (2012, *ReStat* 94(1), 260–272)
    plausibly-exogenous LTZ sensitivity CI;
  - Blandhol--Bonney--Mogstad--Torgovitsky (NBER WP 29709, latest
    revision Jan 2025) and Słoczyński (2024, arXiv:2011.06695)
    `TSLS-as-LATE` negative-weights caveat — automatically surfaced
    when covariates are present and the endogenous regressor is
    binary 0/1;
  - OLS comparator (informative; not causal) per Young (2022).
- `IVDiagResult` exposes `.summary()`, `.to_frame()`, `.to_dict()`,
  `.to_latex()`, `.to_excel()`, `.to_word()`, and
  `.plot('diagnostic'|'forest'|'weak_iv'|'first_stage')`.
- `sp.iv.iv_compare(formula, data, methods=...)` — run several
  k-class / JIVE estimators side-by-side and return a one-row-per-
  method comparison DataFrame (point, SE, CI, first-stage F).
  Auto-resolves the endogenous coefficient name across heterogeneous
  result classes.
- `sp.iv.plot.plot_iv_forest(table, reference=...)` — forest plot of
  estimates and CIs across methods.
- `sp.iv.plot.plot_iv_forest_from_diag(result)` — forest plot built
  directly from an `IVDiagResult` bundle.
- `sp.iv.plot.plot_weak_iv_ci_overlay(result)` — Wald / tF / AR /
  CLR / K / pairs- and wild-bootstrap / LTZ confidence-set overlay.
- `sp.iv.plot.plot_iv_diagnostics(result)` — 2x2 panel: first-stage
  scatter, AR set, weak-IV-CI overlay, leverage diagnostic
  (Young 2022 spirit).
- `sp.iv_diag`, `sp.iv_compare`, `sp.IVDiagResult` re-exported at top
  level for agent ergonomics; both also wired into `sp.list_functions`
  via the registry.
- New verified `paper.bib` entries (DOI / arXiv ID double-checked
  against Crossref + journal pages, May 2026): `keane2024practical`,
  `young2022consistency`, `lal2024much`, `mikusheva2022inference`,
  `borusyak2023nonrandom`, `borusyak2025practical`,
  `kaido2021decentralization`, `chernozhukov2022automatic`,
  `chernozhukov2022rieszn`, `brinch2017beyond`, `bennett2023minimax`,
  `blandhol2025tsls`, `sloczynski2024should`. Existing entries
  enriched with verified volume / issue / pages: `mikusheva2024weak`,
  `lee2022valid`, `borusyak2022quasi`, `masten2021salvaging`.

### Changed — IV polish (v1.15)

- `docs/guides/choosing_iv_estimator.md` adds §10 (`sp.iv.iv_compare`
  forest comparison), §11 (`sp.iv.iv_diag` modern reporting bundle),
  and a TL;DR pointer to the new bundle.
- `paper.md` gains a self-contained IV bullet under
  *Methodological coverage* documenting the new bundle and recent
  methodology references.

### Notes — IV polish (v1.15)

- `iv_diag` is single-endogenous by design; for multi-endogenous
  specifications continue to use `sp.weakrobust` plus
  `sp.iv.sanderson_windmeijer` per regressor.
- Existing IV functions (`sp.iv`, `sp.weakrobust`,
  `sp.kleibergen_paap_rk`, `sp.anderson_rubin_test`, etc.) are
  unchanged — `iv_diag` is purely additive and does not alter any
  numeric path. The 18 new tests in `tests/iv/test_iv_diag.py` all
  pass; no regressions in the 188 prior IV tests.

### Added — RDD polish (v1.15)

RDD module polish tracking the 2018–2026 literature. Six
additions close the gap between `sp.rd` and the canonical R/Stata
`rdpackages` ecosystem on three fronts — *recent methodology*,
*publication-oriented reporting*, and *automatic diagnostics*:

- **Three new estimators** corresponding to flagship 2018–2025 papers:
  - `sp.rd_flex` — flexible (machine-learning) covariate adjustment
    via cross-fit residualisation following Noack, Olma & Rothe (2025,
    arXiv:2107.07942 v5). Built-in learners `boost`/`forest`/`ridge`/
    `lasso` (any sklearn regressor accepted) with K-fold cross-fitting;
    reduces τ̂ variance whenever covariates predict the outcome and
    remains consistent under free-of-cutoff continuity of η. Reports
    out-of-sample R² and variance reduction relative to plain
    `rdrobust`.
  - `sp.rd_bias_aware_fuzzy` — Anderson–Rubin-style bias-aware CI for
    fuzzy RD following Noack & Rothe (2024, *Econometrica* 92(3),
    687–711, doi:10.3982/ECTA19466). Robust to weak first stages and
    avoids the power asymmetry of conventional fuzzy 2SLS-style CIs
    documented by Kaliski, Keane & Neal (2025, NBER 33972).  Reports
    first-stage F and warns on F < 10.
  - `sp.rd_discrete` — honest CIs for RD with a discrete running
    variable following Kolesár & Rothe (2018, *AER* 108(8),
    2277–2304, doi:10.1257/aer.20160945).  Two smoothness classes:
    bounded second derivative (`bsd`) and bounded misspecification
    (`bm`).  Both have provable finite-sample coverage when standard
    rdrobust asymptotics break down because mass points are sparse.

- **Three new reporting helpers** for the standard CCT–Cattaneo–
  Idrobo–Titiunik best-practice workflow:
  - `sp.rd_dashboard` — single-figure 4-panel diagnostic (RD plot,
    CJM-2020 density, covariate balance, bandwidth sensitivity)
    following the recommendations of Calonico, Cattaneo & Titiunik
    (2015, *JASA* 110(512), doi:10.1080/01621459.2015.1017578) and
    Cattaneo, Idrobo & Titiunik (2024, doi:10.1017/9781009441896).
  - `sp.rd_compare` — side-by-side estimation across an arbitrary
    list of methods (`rdrobust`, `honest`, `randinf`, `flex`, …) on
    the same data; returns a tidy `pd.DataFrame` ready for
    `sp.outreg2` / `sp.modelsummary`.
  - `sp.rd_robustness_table` — sweep over kernel × bandwidth × poly
    × donut, returning paper-facing specifications with
    `to_latex()` / `to_excel()` for one-shot supplemental tables.

- **`sp.rdrobust` polish**:
  - New `rho` parameter — Calonico, Cattaneo & Farrell (2018,
    *JASA* 113(522)) ratio bandwidth `b = h / rho`.
  - Auto warning when running variable has < 30 distinct values,
    pointing to `sp.rd_discrete`.
  - Auto warning on fuzzy first-stage F < 10, pointing to
    `sp.rd_bias_aware_fuzzy` and quoting the Kaliski-Keane-Neal
    (2025) ITT recommendation.  Both warnings can be silenced via
    `warn_mass_points=False` / `warn_weak_first_stage=False`.
  - First-stage F now exposed at
    `result.model_info['first_stage_F']`.

- **`sp.rdplotdensity` upgrade**: replaced the legacy kernel-sum
  density estimate with the boundary-adaptive local-polynomial CDF-
  regression density of Cattaneo, Jansson & Ma (2020, *JASA* 115(531),
  1449–1455).  Same signature; better boundary behaviour.

- **Dispatcher**: `sp.rd(..., method='flex' | 'bias_aware' | 'discrete')`
  with full alias coverage (`rd_flex`, `flexible`, `ml_adjust`,
  `bias_aware_fuzzy`, `noack_rothe`, `rd_discrete`,
  `kolesar_rothe`, `discrete_rv`, …).

### Tests — RDD polish (v1.15)

- New `tests/test_rd_polish.py` with 21 checks: estimator parity
  recovery, dispatcher routing, warning behaviour, dashboard smoke
  tests.
- All 156 RD tests (existing + new) pass on Python 3.13 / macOS.

### Citations — RDD polish (v1.15)

DOI-verified via Crossref / publisher pages 2026-05-05:
`noack2024biasaware`, `noack2025flexible`, `kolesar2018inference`,
`kaliski2025power`, `cattaneo2024extensions`,
`cattaneopalomba2025covariates`, `calonico2015optimal`.

## [1.14.0] — 2026-05-05

### Headline

GPU-acceleration sprint. Three workloads now opt into accelerator
backends without changing their public API: (1) the neural causal
estimators (`sp.deepiv`, `sp.tarnet`, `sp.cfrnet`, `sp.dragonnet`,
`sp.cevae`) route through PyTorch CUDA / MPS via a centralised
device resolver; (2) `sp.fast.feols_jax` runs the full WLS solve on
JAX / XLA; and (3) `sp.fast.feols_jax_bootstrap` lifts a JIT-compiled
single-iteration WLS kernel to a `jax.vmap` batched primitive,
giving a 10–100x speedup over sequential CPU bootstrap on CUDA / TPU
at B ≥ 1000. Four bootstrap variants share the same JAX kernel
infrastructure: pairs (multinomial-weight resampling), cluster
(Cameron–Gelbach–Miller 2008 §III.A), wild (row-level Rademacher),
and wild cluster (Cameron–Gelbach–Miller 2008 §III.B); the wild
variants use the score formulation
`β* = β̂ + (X'WX)⁻¹ X'W (η ⊙ û)` which is mathematically identical
to refitting on `y* = X β̂ + η ⊙ û` but needs one mat-vec per
iteration instead of a full QR. A new `cluster_meat` Rust kernel
in `statspai_hdfe` (PyO3 + Rayon, parallel over clusters) is wired
behind `statspai.core._numba_kernels.cluster_meat` with the existing
numba kernel as automatic fallback. `sp.iv(absorb=...)` is the new
2SLS-with-HDFE entry point: residualises `y`, exogenous controls,
endogenous regressors, and instruments by one or more FE columns
via the Phase-1 Rust demean kernel before fitting, with the residual
DOF adjusted by `Σ(G_k - 1)` to charge the absorbed FE rank against
iid / HC1 / CR1 SEs. A new `docs/guides/gpu_acceleration.md` is the
canonical landing page for the accelerator story; the README and
`paper.md` link to it and explicitly bound the GPU promise (most
estimators are CPU-only by design — DiD / RD / synth / GMM are
bandwidth-bound or small-K convex programs where a tuned CPU
kernel matches GPU performance).

### Added

- `sp.fast.feols_jax` — JAX-backed end-to-end OLS / WLS with HDFE.
  Same formula DSL and `FeolsResult` return type as `sp.fast.feols`;
  the WLS solve and HC1 sandwich run on the default JAX device.
  CR1 cluster sandwich delegates to the existing `crve` (which
  itself dispatches to the new `cluster_meat` Rust kernel when
  built). Default `dtype="float64"` preserves bit-comparable
  numerics; `dtype="float32"` available for the GPU fast path.
- `sp.fast.feols_jax_bootstrap` — vmap'd bootstrap with four
  variants (`pairs`, `cluster`, `wild`, `wild_cluster`).
  `vmap_chunk_size` parameter for memory control on tight devices.
  Same-seed → bit-identical reproducibility via `jax.random` PRNG.
  Returns a `FeolsBootstrapResult` dataclass with `coef`, `se_boot`,
  percentile `ci_lower` / `ci_upper`, and the full `boot_betas`
  table for custom CI methods.
- `sp.iv(absorb=...)` — 2SLS with HDFE residualisation. Accepts
  `"firm + year"` string syntax or `["firm", "year"]` list.
  LIML / Fuller / GMM / JIVE raise `NotImplementedError` (Phase 3b).
- `STATSPAI_TORCH_DEVICE` environment variable (`cpu` / `cuda` /
  `cuda:N` / `mps` / `auto`) routes neural causal estimators
  through the requested device. Default `cpu` preserves existing
  pinned numerics; explicit `cuda` raises if the device is
  unavailable rather than silently falling back. New
  `sp.fast.torch_device_info()` mirrors `sp.fast.jax_device_info()`.
- `statspai_hdfe::cluster_meat` Rust kernel — Rayon parallel over
  clusters with thread-local k×k upper-triangle accumulator and
  elementwise reduction. Bumped the crate version 0.5.0-alpha.1 →
  0.7.0-alpha.1. Activation requires a one-time
  `pip install maturin && cd rust/statspai_hdfe && maturin develop --release`;
  Python falls back to the numba kernel transparently when the
  Rust extension is absent.
- `docs/guides/gpu_acceleration.md` — accelerator landing page
  with activation recipes, a Google Colab quickstart benchmark,
  and an explicit "what is *not* GPU-accelerated and why" table.

### Changed

- `paper.md` adds a fifth bullet to the *Unique features*
  list documenting the accelerator story, and notes the Rust
  HDFE / cluster-meat kernel in the implementation paragraph.
- `README.md` comparison-table accelerator row now links to the
  new GPU guide; the *What StatsPAI is — and is not* bullet
  expands to explicitly mention `feols_jax`,
  `feols_jax_bootstrap`, and the vmap mechanism.

### Internal

- New helper `_jax_prep_inputs` shares formula-parse + FE-residualise
  logic between `feols_jax` and `feols_jax_bootstrap`.
  `feols_jax` itself is unchanged in this release; consolidation
  into a shared call site is a candidate follow-up.
- Rust crate adds `src/cluster.rs` (kernel) and a `cluster_meat`
  PyO3 binding in `src/lib.rs`. 3 cargo unit tests cover small-DGP
  reference parity, k=1 closed form, and empty-input safety.

### Verified

- 10 PyTorch device-resolver tests
  (`tests/test_torch_device_resolver.py`); 51 existing neural
  tests pass without numerical drift on default CPU.
- 9 `cluster_meat` Rust parity tests
  (`tests/test_cluster_meat_rust.py`) — auto-skip when
  `statspai_hdfe` is not built.
- 13 `sp.iv(absorb=)` parity tests vs explicit drop-first
  dummies (`tests/test_iv_absorb.py`); coefficients agree to
  `atol=1e-9`, iid SE to `rtol=1e-3`, cluster SE to `rtol=1e-2`.
- 12 `feols_jax` parity tests vs `feols`
  (`tests/test_jax_feols.py`); iid / hc1 / cr1 / weighted /
  float32 / 6 error-path validations.
- 24 `feols_jax_bootstrap` tests
  (`tests/test_jax_feols_bootstrap.py`); convergence to HC1 SE
  for pairs / wild and CR1 SE for cluster / wild_cluster at
  B=2000 (rtol 10–15%); algebraic identity check that the wild
  score formulation reproduces the literal "refit on pseudo-y"
  bootstrap bit-for-bit on a no-FE DGP (`atol=1e-9`).

## [1.13.1] — 2026-05-05

### Headline

Stability tiers, external-validity dossier, and cold-start surgery in a
single release. Every `FunctionSpec` now carries a `stability` field
plus per-function `limitations`, surfaced through
`sp.describe_function`, `sp.help`, `sp.list_functions(stability=...)`,
the `statspai list` CLI, and the LLM-facing `sp.function_schema`
description; `sp.recommend` / `sp.causal` / `sp.paper` default to
dropping `experimental` / `deprecated` entries unless
`allow_experimental=True` is passed — closing a path where an agent
could silently land on a frontier MVP. Eight high-impact estimators
(`aipw`, `aggte`, `pretrends_test`, `sensitivity_rr`, `mccrary_test`,
`oster_bounds`, `wild_cluster_bootstrap`, `rd_honest`) are upgraded
from auto-registered stubs to hand-written specs with full assumption
/ failure-mode / alternative metadata. A weak-instrument preflight
gate in `sp.preflight(data, "ivreg", formula=...)` raises a structured
`warning` row when the first-stage F falls below the Staiger–Stock
(1997) or Stock–Yogo (2005) thresholds, and `sp.recommend(...
design='iv')` adaptively reorders LIML / AR ahead of 2SLS on weak
first stages. A 36-module R parity harness, 21-module Stata parity
harness, 4-dataset original-paper replay (Card 1995, Callaway–Sant'Anna
`mpdta`, Abadie Basque, LaLonde NSW + PSID-1), Track-C performance
harness (HDFE / CS-DiD / SCM / DML log-log scaling), B=1000
Monte-Carlo coverage run, and a 900-trial CausalAgentBench prompt
suite all ship under `tests/r_parity/`, `tests/stata_parity/`,
`tests/orig_parity/`, `tests/perf/`, and `tests/agent_bench/` with
paired R/Python/Stata drivers, JSON results, and 3-way Markdown +
LaTeX parity tables suitable for direct paper inclusion. A new
`sp.validation_report()` / `sp.coverage_matrix()` /
`sp.reproduce_jss_tables()` meta-API summarises the live registry,
materialises the parity / coverage / agent-bench artifacts as JSON,
and optionally re-runs the harnesses end-to-end so referees can
verify StatsPAI's external-validity claims without leaving Python.
Cold-start surgery in three steps brings sklearn submodules pulled by
`import statspai` from 245 to **0** (`statspai.forest` lazy-loaded —
Step 1B; 18 estimator files import sklearn lazily inside function
bodies — Step 1C; HAL TMLE classes drop sklearn class inheritance —
Step 1D), pinned by a new
`test_sklearn_budget_ceiling_on_bare_import_statspai` contract. The
workflow / paper orchestration layer replaces silent `except: pass`
paths with `WorkflowDegradedWarning` + structured `degradations`
records on the result object, so optional-stage failures surface in
`PaperDraft.to_dict()` and the rendered `Pipeline notes` section
instead of disappearing. `sp.principal_strat(instrument=...)` ships a
proper Angrist-Imbens-Rubin Wald-LATE estimator (the kwarg was
previously stubbed); `sp.hal_tmle(variant='projection')` keeps its
`NotImplementedError` but now points at a written-out RFC
(`docs/rfc/hal_tmle_projection.md`) instead of raising in silence.
Lazy-loading of optional families via `__getattr__` keeps `import
statspai` fast without breaking same-name function/subpackage
collisions (`bartik`, `deepiv`, `proximal`, …) — pinned by a
late-bind / post-import-shadow contract test and a committed
`__init__.pyi` stub generator so IDE / mypy see lazy-loaded names. A
latent Callaway–Sant'Anna REG inference scaling bug — discovered
*because* the parity harness flagged it — is fixed in
`did/callaway_santanna.py`.

### Added

- **Weak-instrument preflight gate** in `sp.preflight(data, "ivreg",
  formula=...)`. The new `first_stage_strength` check parses the
  Wilkinson IV formula, runs the first-stage OLS, and emits a
  `warning` row when the partial F-statistic falls below either the
  Staiger–Stock (1997) rule of thumb (F < 10, "very weak") or the
  Stock–Yogo (2005) 10% maximum-size critical value of 16.38 for
  one endog / one instrument (F < 16.38, "weak"). The warning
  payload includes structured `recovery_hints` pointing at
  `method='liml'`, `inference='ar'`, and `sp.anderson_rubin_ci(...)`
  so an LLM agent can branch on the typed envelope without parsing
  prose. Closes the §5.3 robustness DGP follow-up: Track B's
  `test_iv_weak_instrument_undercoverage` documents that 2SLS+HC1
  under-covers at the 0.88 level on a pi=0.10 first stage; the
  preflight now flags this *before* the user pays for the 2SLS fit.
- **Adaptive IV ranking in `sp.recommend(... design='iv')`.** When
  the live first-stage F is below 10 the recommendation list is
  reordered: LIML moves to the top, an Anderson–Rubin row is
  inserted, and the 2SLS row is annotated with
  `very_weak_iv=True` plus a rationale that explains the
  HC1-coverage failure mode. When F is in [10, 16.38) the order
  stays 2SLS → LIML but the rationales reference the
  Stock–Yogo threshold. The 2SLS row now also carries
  `first_stage_F` as a numeric field so downstream tooling can
  consume it without regex.
- **Preflight / recommend tests.** `tests/test_preflight.py` adds
  `TestIVFirstStageStrength` (6 cases covering strong / weak /
  borderline / non-IV / missing-columns / JSON-safe payloads).
  `tests/test_smart_workflow.py::TestRecommend` adds two cases
  pinning the 2SLS-first ordering on strong instruments and the
  LIML-first / AR-included ordering on weak instruments.

- **R parity harness — 36 paired R/Python modules.** New
  `tests/r_parity/` ships 36 paired scripts (one R, one Python) that
  replay the same DGPs through `fixest` / `did` / `csdid` / `gsynth` /
  `MatchIt` / `DoubleML` / `rdrobust` / `Synth` / `lme4` / `plm` /
  `frontier` / `MR-PRESSO` / `lavaan` / `mediation` / `WeightIt` /
  `cobalt` / `mlogit` / `nlme` / `ordinal` / … and StatsPAI's matching
  estimator. Each module emits `<id>_R.json` and `<id>_py.json`;
  `compare.py` produces a 3-way parity table (`parity_table.md` /
  `.tex` / `parity_table_3way.md` / `.tex`) tightened with a small
  ID-column + longtable layout for direct paper inclusion. Two
  parallel tracks — "orig" (canonical-dataset replays) and "perf"
  (timing under matched DGPs) — share the same compare-tooling.

- **Stata parity harness — 21-module StatsPAI ↔ Stata 3-way
  compare.** New `tests/stata_parity/` ships 21 paired
  `.do`/`.py` scripts covering `reghdfe`, `xtreg`, `csdid`, `did_imputation`,
  `synth`, `synth_runner`, `ivreg2`, `xtivreg`, `rdrobust`,
  `psmatch2`, `teffects`, `xtfrontier`, `mixed` / `melogit` /
  `mepoisson`, `bayes`, `gmm`, `boottest`, plus the Stata→Python
  translator round-trip. Drivers write Stata results to JSON via the
  `stata-mcp` `stata_do` tool; Python drivers run StatsPAI through
  `import statspai as sp`; `compare_stata.py` joins on `(module,
  estimator, statistic)` and emits `parity_table_stata.md` /
  `parity_table_stata.tex` plus the 3-way StatsPAI ↔ R ↔ Stata table.

- **Canonical-dataset original-paper replays.** New
  `tests/orig_parity/` adds 4 module pairs that replay each paper's
  headline number bit-equal to the published value: Card (1995)
  returns-to-schooling on `wooldridge::card`; Callaway–Sant'Anna
  (2021) staggered DiD on `did::mpdta` (the package's vendored
  minimum-wage panel); Abadie–Diamond–Hainmueller Basque on
  `Synth::basque`; LaLonde (1986) NSW on `MatchIt::lalonde` plus a
  4b sub-module on `causalsens::lalonde.psid` (true Dehejia–Wahba
  NSW + PSID-1, 2675 obs) where `sp.regress` + `sp.psm` recover the
  published −15,205 to relative tolerance 1.5e-05. Drivers + bundled
  data + JSON results + `parity_table_orig.md` are all committed.

- **Track-C performance harness — log-log timing.** New
  `tests/perf/` adds matched R/Python timing runs for HDFE
  (`fixest::feols` vs `sp.feols`), CS-DiD (`did::att_gt` vs
  `sp.callaway_santanna`), SCM (`Synth::synth` vs `sp.synth`), and
  DML (`DoubleML::DoubleMLPLR` vs `sp.dml`) at log-spaced N. Drivers
  plus `compare_perf.py` produce `perf_table.md` / `.tex` and a
  `track_c_loglog.{pdf,png}` log-log scaling figure.

- **Coverage Monte Carlo at B=1000.** New
  `tests/coverage_monte_carlo/run_b1000.py` measures 95% CI coverage
  for OLS (0.952), 2×2 DiD (0.955), and strong-Z IV (0.962) — all
  inside the 99% Wilson band [0.935, 0.967] around nominal 0.95. The
  full slow `pytest tests/coverage_monte_carlo/ -m slow` sweep at
  B=1000 also passes 8/8 (753.51 s). Frozen run lives at
  `results_b1000/coverage_b1000.json`; previous fast-track results
  remain at the default B=200.

- **CausalAgentBench scaffolding (mock-mode shipped, API run gated).**
  New `tests/agent_bench/` ships a 50-prompt × L1/L2/L3 difficulty
  × 6 cells × 3 reps = 900-trial agent bench with a deterministic
  mock-LLM runner (`runners/mock_llm.py`), a frozen OSF
  pre-registration protocol (`prompts/_protocol.md`), and a grader
  (`runners/grader.py`) emitting an H1–H5 directional results table.
  Mock dry-run completes in <1 s and produces
  `results/headline.md` + `results/scores.csv` + `results/trials.jsonl`;
  the production `--api` flag is one switch away once the OSF
  pre-registration and API budget clear.

- **`sp.validation_report` / `sp.coverage_matrix` /
  `sp.reproduce_jss_tables` — JSS-grade validation meta-API.** New
  `src/statspai/validation.py` (863 LOC, 58-test battery in
  `tests/test_jss_validation_api.py`) exposes three top-level
  functions for the paper-submission audit trail:
  `sp.validation_report()` summarises the live `sp.registry` plus
  the materialised parity / coverage / agent-bench artifacts as a
  structured `ValidationReport` (`registry.total_functions`,
  `evidence.r_parity_modules`, `evidence.stata_parity_modules`,
  `evidence.coverage_b1000`, `evidence.agent_bench_trials`, …) with a
  one-paragraph `.summary()` and full JSON `.to_dict()`;
  `sp.coverage_matrix()` enumerates every reference-implementation
  parity claim with its expected tolerance, observed gap, and the
  driver script that produced the JSON; `sp.reproduce_jss_tables()`
  returns a `ReproductionResult` enumerating the exact `Rscript`,
  `python`, `pytest`, and `xelatex` commands needed to regenerate
  every table — and, when called with `dry_run=False`, executes them
  in dependency order with timing + return codes recorded per
  step. The default mode is metadata-only (no R / Stata / LaTeX
  required), so `import statspai as sp; print(sp.validation_report().summary())`
  works on a stock `pip install statspai==1.13.1` install.

- **8 high-impact estimators upgraded from auto-registered to
  hand-written FunctionSpec.** `aipw`, `aggte`, `pretrends_test`,
  `sensitivity_rr`, `mccrary_test`, `oster_bounds`,
  `wild_cluster_bootstrap`, and `rd_honest` now ship with full
  agent-native metadata: 2–4 assumptions per spec, 2 failure modes
  with recovery hints, ranked alternatives, typical_n_min, vetted
  references with paper.bib bib keys, and full enum-validated
  ParamSpecs. Previously these were auto-registered with only the
  first docstring line and inferred parameter types — agents calling
  `sp.describe_function('aipw')` could not see the doubly-robust
  guarantee, the propensity overlap requirement, or the alternatives
  to fall back on. Hand-written count moves from 203 to 211; auto-
  registered drops from 768 to 760. (Step H of v1.13 stability
  roadmap.)

- **`sp.principal_strat(instrument=...)` — encouragement-design AIR /
  Wald LATE.** The previously-stubbed `instrument=` parameter now
  routes to a proper estimator (Angrist-Imbens-Rubin 1996 §4): given
  binary instrument `Z`, treatment `D`, post-treatment stratum `S`,
  and outcome `Y`, under random `Z` + monotonicity D(1)>=D(0) +
  exclusion + SUTVA, the function reports two Wald LATEs among
  Z-compliers — `τ_Y` for the effect of `D` on the outcome and `τ_S`
  for the effect of `D` on the post-treatment stratum variable —
  plus the complier share `π_C(Z)`, all with bootstrap SE/CI. A
  `RuntimeWarning` is emitted when the first stage degenerates or points
  in the wrong direction for the supplied instrument coding.
  `method=` is ignored on this path because identification comes from
  `Z`, not from the post-treatment stratum decomposition. The
  `limitations` entry is rewritten: the only remaining gap on this
  path is always-survivor SACE under encouragement design (Mealli &
  Pacini 2013, partial identification). Seven new tests in
  `tests/test_principal_strat.py`.

- **`sp.hal_tmle(variant='projection')` RFC + sharper error.** Rather
  than ship an unverified port of the Li-Qiu-Wang-vdL (2025) §3.2
  Riesz-projection step (the v1.11.x code path was a no-op on the
  point estimate — see CHANGELOG), v1.13 keeps the
  `NotImplementedError` and adds `docs/rfc/hal_tmle_projection.md`
  with the full implementation roadmap and the parity-test gates that
  must clear before the variant can be promoted to `stable`. The
  runtime exception message now points at the RFC and asks reporters
  to file an issue with the publication's headline number they'd like
  to match — so the next maintainer to pick this up has a clear
  target. Registry `limitations` entry updated with the RFC link.

- **Smart layer respects `FunctionSpec.stability`.** `sp.recommend(...)`,
  `sp.causal(...)`, and `sp.paper(...)` now accept an
  `allow_experimental: bool = False` flag (default agent-safe).  When
  ``False``, recommendations whose backing function is registered as
  ``stability='experimental'`` (or ``'deprecated'``) are dropped from
  the ranked output and the workflow's ``warnings`` /
  ``pipeline_notes`` records what was filtered.  Pass ``True`` to
  include frontier MVPs (e.g. ``did_multiplegt_dyn``,
  ``text_treatment_effect``).  This closes a gap where an LLM agent
  asking ``sp.causal(df, ...)`` for an applied analysis could
  silently land on a frontier MVP just because the recommender
  ranked it first.  Tests in `tests/test_smart_stability_gating.py`.
- **Stability reverse-audit script.** `scripts/stability_audit.py`
  cross-checks every `stability='stable'` claim in the registry
  against parity-test coverage in `tests/reference_parity/` and
  `tests/external_parity/`. Splits the catalogue into hand-written
  vs. auto-registered specs (the latter having been silently
  classified `stable` by default) and reports the count of unbacked
  claims in each bucket. `--check` mode is CI-friendly and fails when
  the unbacked-handwritten count exceeds a loose floor (currently
  220) — bumping the floor requires editing the script as a
  deliberate quality signal. Does NOT auto-downgrade; the call to
  flip a function from `stable` to `experimental` belongs to a
  maintainer who has read the code. Tests in
  `tests/test_stability_audit.py`. The audit fixed a registry bug
  along the way: auto-registered specs were never tagged `_auto=True`
  on the `FunctionSpec` instance, so `describe_function` error hints
  and the audit itself couldn't distinguish them from hand-written
  entries; that's now fixed via `object.__setattr__` inside
  `_auto_spec_from_callable`.
- **Runtime consistency tests for `FunctionSpec.limitations`.** Each
  `limitations` entry on a `FunctionSpec` is now structurally audited
  by `tests/test_limitations_consistency.py` so the registry's
  parity-grade-with-known-gaps claims cannot drift away from runtime
  behaviour: every entry must (a) use vetted vocabulary and (b) be
  classified as either runtime-testable (a curated map calls the
  function with the unimplemented value and asserts the documented
  exception) or descriptively-soft (silent fallback / caveat,
  whitelisted in `LIMITATIONS_DESCRIPTIVE_ONLY`). Adding a new
  limitation without classifying it now fails CI. Caught one drift
  bug in this pass: the `cgroup='nevertreated' + panel=False`
  limitation was attached to `wooldridge_did`, but only the `etwfe`
  alias exposes those parameters — moved to `etwfe` and surfaced the
  missing `cgroup` ParamSpec to the schema.
- **Test-coverage battery for the four worst-covered files +
  parity-grade smoke battery across `did/synth/rd/iv/tmle/bayes`.**
  The v1.12.x audit flagged six causal-family modules at low
  statement coverage (`did` 14.7%, `synth` 12.9%, `rd` 16.9%,
  `iv` 18.0%, `tmle` 14.8%, `bayes` 14.1%) with four files entirely
  unexercised: `wooldridge_did.py`, `did_imputation.py`,
  `synth/report.py`, `workflow/paper.py`. Five new test files raise
  per-file coverage to **synth/report.py 4% → 81%, wooldridge_did.py
  76% → 93%, did_imputation.py 85% → 99%, workflow/paper.py 66% →
  86%** and add a 30-test cross-family smoke battery
  (`tests/test_low_cov_battery.py`) that exercises every headline
  estimator's CI/SE/point-estimate contract:
  - `tests/test_synth_report.py` (25 tests) — full text/markdown/LaTeX
    SCM report renderer + every sensitivity sub-block + the LaTeX
    escape table.
  - `tests/test_wooldridge_did_branches.py` (31 tests) — Bacon + dCDH
    decomposition, repeated-CS / never-only / xvar dispatch branches,
    every `etwfe` validation guard, all four `etwfe_emfx`
    aggregations including `include_leads=True`.
  - `tests/test_did_imputation_branches.py` (14 tests) — every
    `ValueError` guard, the controls + horizon event-study path with
    pre-trend chi-squared test, and the `_cluster_se_horizon`
    `N_k == 0` short-circuit.
  - `tests/test_paper_branches.py` (31 tests) — every YAML/TeX/MD
    helper, all four `to_qmd` rendering branches (single vs.
    multi-format, author / bibliography / csl), `to_docx` fallback
    when `python-docx` is missing, `write()` extension dispatch, and
    the `_render_dag_section` text + mermaid branches.
- **`CausalResult.summary()` accepts both event-study column
  conventions.** The shared `summary()` previously hard-coded
  `(relative_time, att)` and crashed with `KeyError: 'relative_time'`
  on `wooldridge_did` / `etwfe` results, which carry the
  `(rel_time, estimate)` schema instead. The renderer now auto-detects
  whichever pair is present and silently skips the event-study block
  when neither is — every existing caller keeps its formatting and
  the wooldridge family no longer crashes a user's `.summary()` call.
  Regression-pinned by `test_wooldridge_did_summary_renders_event_study`.

- **Stability tiers and per-function `limitations` (parity-grade vs.
  frontier-grade visibility).** Every `FunctionSpec` now carries a
  `stability` field (`"stable"` / `"experimental"` / `"deprecated"`,
  exposed as `sp.STABILITY_TIERS`) and a `limitations` list that
  enumerates partial-implementation gaps inside otherwise stable
  functions (e.g. `hal_tmle(variant='projection')`,
  `principal_strat(instrument=...)`, `rdrobust(weights=...)`). The
  fields flow through `sp.describe_function`, `sp.agent_card`,
  `sp.function_schema` (description prefix + `Known limitations:`
  suffix so LLM tool-callers see the gap before calling),
  `sp.list_functions(stability=...)`, `sp.agent_cards(stability=...)`,
  the `STABILITY` block in `sp.help()`, the per-function detail in
  `sp.help('<name>')`, and a new `statspai list --stability ...` CLI
  flag. This closes a layering gap where users (and agents) could not
  tell which functions are numerically aligned and signature-locked
  vs. which are MVP / RFC-tracked frontier work, and where specific
  unimplemented variants were only discoverable by triggering
  `NotImplementedError` mid-pipeline. Initial tagging covers the three
  causal_text / `did_multiplegt_dyn` experimental entries plus
  variant-level limitations on `hal_tmle`, `principal_strat`,
  `rdrobust`, `callaway_santanna`, `wooldridge_did`,
  `network_exposure`, and `continuous_did`. See the new
  `docs/guides/stability.md` for the contract and promotion path.

### Changed

- **Cold-start: lazy-load `statspai.forest` (Step 1B).** `import
  statspai` previously chained
  ``from .forest.causal_forest import CausalForest, causal_forest`` plus
  three sibling eager imports for `forest_inference` /
  `multi_arm_forest` / `iv_forest` at module load, transitively pulling
  ~245 `sklearn.*` submodules into `sys.modules` (~270 ms cumulative on
  cold cache) for every session — even ones that never touch
  heterogeneous-effect forests. The four eager lines are removed; the
  ten public leaves (`CausalForest`, `causal_forest`,
  `calibration_test`, `test_calibration`, `rate`, `honest_variance`,
  `multi_arm_forest`, `MultiArmForestResult`, `iv_forest`,
  `IVForestResult`) now resolve via `_LAZY_ATTRS` keyed to dotted
  submodule paths (e.g. `forest.causal_forest`) and fault in on first
  `sp.<name>` access. `forest` does not collide with a top-level function
  (no `sp.forest` callable export) so the standard lazy path is safe;
  `sp.causal`'s callable shim and the `statspai.causal` deprecation
  shim continue to work unchanged. Pinned by three new contracts in
  `tests/test_late_bind_contracts.py` — `import statspai` must not
  pre-load any `statspai.forest.*` submodule (subprocess-isolated to
  avoid `sys.modules` pollution that would corrupt downstream
  `isinstance` checks); each of the 10 forest leaves must resolve to a
  callable on first access; and a downstream
  `from statspai.forest.causal_forest import CausalForest` must not
  re-shadow `sp.causal_forest` to the leaf module via Python's
  post-import attribute binding. Other sklearn-eager paths
  (`did/overlap_did`, `metalearners/*`, `policy_learning/*`,
  `synth/cluster`, plus ~7 conflict-prone same-name modules pinned eager
  for the late-bind contract) still pull `sklearn` on bare import; those
  are tracked separately for Step 1C and do not block this lazy-forest
  win.

- **Cold-start: drop sklearn class inheritance from HAL estimators
  (Step 1D).** ``HALRegressor`` / ``HALClassifier`` in
  ``tmle/hal_tmle.py`` previously subclassed
  ``sklearn.base.BaseEstimator`` plus a Mixin, which pulled ~39
  ``sklearn.*`` submodules into ``sys.modules`` at module-load time —
  the *only* remaining sklearn footprint after Steps 1B/1C lazy-loaded
  ``forest`` and the 18 estimator files.  The inheritance is gratuitous
  here: ``super_learner.fit`` only needs ``sklearn.base.clone(learner)``
  (which is duck-typed — ``get_params(deep=False)`` + ``cls(**params)``
  reconstruction) plus ``.fit`` / ``.predict`` / ``.predict_proba``; no
  code path calls ``.score(...)``, ``is_classifier(...)``, or
  ``is_regressor(...)`` on the HAL classes.  Replaced the inheritance
  with a minimal ``_BaseHAL`` providing the ``get_params`` /
  ``set_params`` / ``__repr__`` slice that ``clone()`` actually
  consumes (introspection via ``inspect.signature(self.__init__)``,
  with object identity preserved so sklearn's post-clone
  ``param1 is param2`` sanity check passes).  ``_estimator_type =
  "regressor"`` / ``"classifier"`` class attributes keep
  ``sklearn.base.is_regressor`` / ``is_classifier`` returning True for
  any future external caller.  After Step 1D, ``import statspai`` pulls
  **zero** sklearn submodules — full 245 → 0 — and the
  ``test_sklearn_budget_ceiling_on_bare_import_statspai`` contract is
  tightened from ``<= 50`` to ``<= 0`` to pin the floor.  152 tests
  across ``test_hal_tmle`` / ``test_tmle`` / ``test_late_bind_contracts``
  / ``test_low_cov_battery`` / ``test_metalearners`` pass cleanly.

- **Cold-start: lazy-import sklearn across 18 estimator files (Step
  1C).** Building on Step 1B, every remaining top-level
  `from sklearn.X import Y` in
  `did/overlap_did.py`, `metalearners/{auto_cate,metalearners,auto_cate_tuned}.py`,
  `policy_learning/{policy_tree,ope}.py`, `synth/cluster.py`,
  `proximal/pci_regression.py`, `bcf/{bcf,longitudinal}.py`,
  `tmle/{tmle,super_learner,ltmle,ltmle_survival}.py`,
  `dose_response/gps.py`, `multi_treatment/multi_ipw.py`,
  `mediation/four_way.py`, and `interference/orthogonal.py` was moved
  inside the function bodies that actually use it.  `BaseEstimator`
  type annotations were converted to string-literal form under
  `if TYPE_CHECKING:` so `inspect.signature` / Pyright / mypy still
  resolve them without forcing `sklearn.base` at module load.  Several
  long-standing dead imports were dropped (`BaseEstimator` /
  `is_classifier` / `cross_val_predict` in
  `metalearners/metalearners.py`; `LinearRegression` in
  `proximal/pci_regression.py`; `BaseEstimator` / `clone` /
  `GradientBoostingClassifier` in `multi_treatment/multi_ipw.py`;
  etc.).  After Step 1B + 1C, `import statspai` pulls **39** sklearn
  submodules instead of **245** — a 5.3× reduction.  The 39 are
  `sklearn.base` plus its mandatory deps, pulled by `tmle/hal_tmle.py`
  whose `HALRegressor(BaseEstimator, RegressorMixin)` /
  `HALClassifier(BaseEstimator, ClassifierMixin)` need sklearn at
  class-definition time; refactoring that inheritance hierarchy is
  out of scope.  Pinned by a new
  `test_sklearn_budget_ceiling_on_bare_import_statspai` contract in
  `tests/test_late_bind_contracts.py` (≤ 50 ceiling, ~39 floor + 11
  slack for sklearn-version drift) running in a subprocess so the
  cold-state measurement does not perturb other tests' `sys.modules`.
  248 tests across the 18 affected modules (metalearners /
  metalearner_frontiers / auto_cate / auto_cate_tuned / overlap_did /
  tmle / hal_tmle / proximal / proximal_frontiers / bcf_longitudinal /
  bcf_ordinal / conformal_bcf_bunching_mc / policy_learning / mediation
  / mediation_sensitivity / interference_extensions / late_bind_contracts
  / causal_forest_grf / forest_inference / ope_cevae / ope_extensions /
  cluster_rct) pass cleanly.

- **README lead aligned with the agent-native + parity-validated
  positioning.** `README.md` and `README_CN.md` both foreground "first
  agent-native Python platform" and surface R / Stata parity
  validation in the lead paragraph (was previously buried in the Task
  View comparison section further down).

- `sp.recommend()` now defaults to an agent-safe stability gate:
  recommendations whose registry entry is marked
  `stability='experimental'` or `stability='deprecated'` are dropped
  unless the caller passes `allow_experimental=True`. The filter keeps
  backward compatibility for unknown custom recommendation entries,
  records dropped names in `RecommendationResult.warnings`, and is
  forwarded through `sp.causal(..., allow_experimental=...)` and
  `sp.paper(..., allow_experimental=...)` so higher-level workflows
  cannot silently land on frontier MVP estimators.
- Hardened the workflow/paper orchestration layer so optional failures
  no longer disappear silently. `sp.causal(...).run(full=True)` now
  records optional-stage failures (`compare_estimators`,
  `sensitivity_panel`, `cate`) in `workflow.pipeline_notes`, and
  `sp.causal(...).report(fmt='markdown')` renders those notes in a
  dedicated section instead of silently dropping the context.
- `sp.paper(...)` now constructs its internal `CausalWorkflow` with
  `auto_run=False` and advances stages exactly once. This removes the
  prior double-execution path where the workflow could fully auto-run
  before `paper()` manually re-ran `diagnose/recommend/estimate`
  (and sometimes `robustness`) again.
- `PaperDraft` now surfaces orchestration degradations directly in a
  `Pipeline notes` section and includes `degradations` in `to_dict()`,
  so missing DAG/citation/provenance/section-rendering steps are
  visible in the artifact itself rather than only via warnings.

### Fixed

- **⚠️ Correctness — `sp.callaway_santanna(method='reg')` inference.**
  The Callaway–Sant'Anna outcome-regression (REG) path produced
  influence-function standard errors that were inconsistent with the
  IPW / DR variants because the control-regression uncertainty was
  not propagated and the per-cohort scaling in the influence-function
  aggregation was off by the cohort-size weighting. Coverage
  simulations under the `mpdta` DGP were running ~88% (nominal 95%)
  on `method='reg'`, while `'ipw'` and `'dr'` were inside the 99%
  Wilson band. The fix tightens the REG influence-function scaling
  and explicitly adds the control-regression contribution; the
  parity table at `tests/r_parity/results/parity_table_3way.md` and
  the coverage frame at `tests/coverage_monte_carlo/FINDINGS.md` are
  refreshed accordingly. Regression-pinned by
  `tests/reference_parity/test_did_parity.py` and the new
  `tests/r_parity/04_csdid.py` driver. **Re-run any v1.10–v1.13
  Callaway–Sant'Anna analyses that used `method='reg'`**;
  `'ipw'` and `'dr'` are unchanged.

- **`isinstance(res, sp.OPEResult)` no longer false-negative on results
  from `sp.ope.*`.** During the lazy-load refactor of optional families
  the eager re-export path that used to bind `sp.OPEResult` to
  `statspai.ope.estimators.OPEResult` was dropped, so `sp.OPEResult`
  silently resolved to a *parallel* class defined in
  `statspai.policy_learning.ope` — and `isinstance(sp.ope.ips(...),
  sp.OPEResult)` flipped from `True` (v1.12.2) to `False`. The eager
  `from .policy_learning import ... OPEResult` is removed so
  `sp.OPEResult` falls through to the lazy `_register_lazy("ope",
  "OPEResult", ...)` table, restoring v1.12.2 class identity.
  Regression-pinned by `tests/test_ope_cevae.py::test_ips_close_to_true_value`.
- Hand-written registry specs for `aggte` and `principal_strat` now
  exactly match their callable signatures (`na_rm`, `alpha`, `seed`),
  with a regression test guarding the new v1.13 hand-written upgrades
  against future signature drift.
- The natural-language `sp.paper(data, question, ..., include_robustness=False)`
  path no longer runs or renders the robustness section implicitly via
  `sp.causal` auto-run side effects.
- `paper_from_question()` now carries its collected degradation records
  into the returned `PaperDraft`, so late provenance/citation/DAG
  failures remain inspectable after draft construction.
- Top-level `statspai.__all__` is now de-duplicated in order-preserving
  fashion, reducing public-surface drift between the import namespace
  and registry/help tooling.
- The top-level function-first API now survives the `sp.iv` bootstrap
  path for same-name families like `bartik` and `deepiv`. The root
  package eagerly rebinds the 14 function/subpackage collisions
  (`proximal`, `principal_strat`, `bartik`, `bridge`, `causal_impact`,
  `bcf`, `bunching`, `deepiv`, `dose_response`, `frontier`,
  `interference`, `msm`, `multi_treatment`, `tmle`) while
  `statspai.iv` lazy-loads its optional `bartik` / `deepiv`
  re-exports, so `sp.bartik(...)` / `sp.deepiv(...)` stay callable
  instead of degenerating into bare module objects after import order
  changes.
- `smart.assumptions`, `smart.brief`, `smart.identification`,
  `smart.sensitivity`, and `smart.verify` now lazy-import
  `workflow._degradation` only inside failure paths. That removes a
  premature `workflow/__init__` import during `import statspai`,
  which had reintroduced partially initialized top-level symbols and
  made the lazy API order-sensitive.
- Added a committed `src/statspai/__init__.pyi` generator and pinned it
  with a regression test so IDE/type-checker visibility tracks the live
  runtime namespace. The stub generator now skips exported constants
  during leaf scanning and correctly types `STABILITY_TIERS` as
  `frozenset[str]`, avoiding duplicate/conflicting declarations.
- Pinned the two binding hazards introduced by the lazy-load refactor
  with 21 explicit contracts in `tests/test_late_bind_contracts.py`:
  the five late-bind aliases re-bound by `_article_aliases`
  (`mediation`, `policy_tree`, `dml`, `matrix_completion`,
  `causal_discovery`) plus the `sp.iv` callable dispatcher must each
  remain callable rather than degenerating to a module on import
  re-order; and the 14 function/subpackage collisions
  (`proximal`, `principal_strat`, `bridge`, `bcf`, `bunching`,
  `dose_response`, `multi_treatment`, `causal_impact`, `frontier`,
  `interference`, `tmle`, `msm`, `deepiv`, `bartik`) must survive a
  downstream `from statspai.X import Y` without the auto-bound submodule
  silently re-shadowing the function. Closes the residual gap left by
  Codex's lazy-load refactor and Claude Code's same-name eager-rebind
  follow-up.

## [1.12.2] — 2026-05-01

### Headline

ML-routing for the estimand-first DSL (`sp.causal_question`) plus a
shared robustness battery so `sp.paper(...)` renders the same audit
section regardless of entry point. The Egami et al. (2024) LLM-label
corrector graduates from binary-only to multi-class with a
bias-corrected bootstrap, and DML's IV variants (`sp.dml(model='pliv')`,
`sp.dml(model='iivm')`) now honour `sample_weight` end-to-end.
Citation metadata fixes the wrong Zenodo DOI shipped under v1.12.1 —
no estimator output changes.

### Added

- `sp.llm_annotator_correct` (`causal_text/llm_annotator.py`) — three
  v1.7-deferred upgrades to the Egami et al. (2024) measurement-error
  correction for LLM-derived treatment labels. Backward compatible:
  the binary-T numerical path is unchanged, every existing kwarg keeps
  its default behaviour, and existing diagnostics retain their keys.
  - **Multi-class treatment.** The corrector now auto-detects the class
    set from the union of LLM and human labels. For K ≥ 3 the
    confusion matrix `M[i, j] = P(T_obs=j | T_true=i)`, validation-
    marginal `π[i]`, and Bayes posterior `Q[i, j] = P(T_true=i |
    T_obs=j)` are assembled; the K×K coefficient transform `θ_obs =
    T θ_true` is inverted to recover per-class corrected contrasts.
    Headline `.estimate` reports the smallest non-reference class;
    full vector ships in `.detail` (per-class naive/corrected
    estimate, SE, CI, p-value).  Singular / near-singular `T` raises
    `IdentificationFailure`.
  - **Bias-corrected bootstrap.** Optional `bootstrap=True` jointly
    resamples the full sample (validation rows + unlabeled rows) and
    re-runs the entire correction pipeline `n_bootstrap` times
    (default 500), reporting Efron-Tibshirani bias-corrected percentile
    CIs that reflect validation-set sampling uncertainty. New kwargs:
    `bootstrap`, `n_bootstrap`, `bootstrap_seed`. First-order SE/CI
    remain available in `model_info['first_order_se' / '_ci']`;
    `bootstrap` sub-dict reports `n_valid`, `n_failed`, `seed`,
    `method`, `mean`, `median`.
  - **SE inflation factor diagnostic.** Both binary and multi-class
    paths populate `model_info['se_inflation_factor']` — a delta-
    method multiplier (≥ 1) the user can apply to the first-order SE
    for an honest accounting of validation-set noise. For binary it is
    derived analytically from the binomial variances of `p_01` and
    `p_10`; for multi-class it is a finite-difference Jacobian-based
    heuristic (use `bootstrap=True` for the rigorous version).
  - Multi-class diagnostics also expose `confusion_matrix`,
    `q_posterior`, `transform_matrix`, `condition_number`,
    `pi_validation`, `headline_contrast`.
- `sp.causal_question(..., design=...)` now accepts the four
  ML-selection-on-observables tags directly:
  `design='dml' | 'tmle' | 'metalearner' | 'causal_forest'`.
  The planner records the right identification story /
  assumptions for each, and the dispatcher now routes to the
  corresponding estimator with targeted validation (e.g. DML
  covariates required; PLIV / IIVM scalar-instrument guard;
  causal-forest binary-treatment guard for ATE inference).
- New guide:
  `docs/guides/choosing_ml_causal_estimator.md` — decision tree for
  choosing between DML / TMLE / metalearner / causal_forest, plus a
  side-by-side comparison of estimands, IV support, and inference.
- Shared robustness battery:
  `workflow/_robustness.py` + `run_robustness_battery(...)`.
  Both `sp.paper(data, question, ...)` and `sp.paper(CausalQuestion(...))`
  now render the same design-aware robustness section instead of
  splitting between a thin NL path and a placeholder estimand-first
  path.
- Weighted `sample_weight` support in `sp.dml(model='pliv')` and
  `sp.dml(model='iivm')`. The IV orthogonality moment, residualisation
  step, and downstream sandwich SE are all weighted consistently
  (`E[w · ψ(W; θ, η)] = 0`); unit weights reproduce the unweighted path
  bit-for-bit. Closes the last `sample_weight` gap in `dml/` after the
  v1.12.0 PLR / interactive audits — `sp.dml`'s four core estimators
  now all support survey / inverse-probability weights.

### Changed

- `sp.causal(...).robustness()` now delegates to the shared robustness
  battery and still preserves backwards compatibility via the legacy
  flat `robustness_findings` dict; structured per-finding records are
  additionally available under `['_findings']`.
- `paper.bib` / docs metadata filled in missing bibliographic details
  for TMLE / causal forest / meta-learner references and removed a
  duplicate van der Laan entry so the new ML-estimator guide and the
  expanded `causal_question` docstrings resolve cleanly.
- `docs/guides/causal_text_family.md` and the registry card for
  `sp.llm_annotator_correct` now describe the new multi-class,
  bootstrap, and SE-inflation-factor behaviour rather than the old
  binary-only path.

### Fixed

- `sp.paper(CausalQuestion(...))` no longer emits a placeholder
  Robustness section pointing users back to `sp.causal(...)`; it now
  runs the same substantive battery as the natural-language paper path.
- `sp.causal_question(..., estimand='CATE')` now auto-promotes to
  `metalearner` only when effect modifiers are actually declared.
  Without covariates it falls back honestly to a scalar ATE path with
  an explicit warning, so `identify()` and `estimate()` agree.
- `design='causal_forest'` now reports the population ATE summary via
  cross-fit AIPW influence-function inference instead of leaving the
  planner with a CATE-only story and no principled scalar ATE layer.

## [1.12.1] — 2026-04-30

Citation metadata polish — no numerical or API changes to any estimator.

### Added

- `sp.citation(format=...)` — package-level citation helper returning
  BibTeX (default), APA, plain text, or the raw `CITATION.cff` contents.
  Distinct from `sp.cite()`, which formats individual coefficients
  inline. `sp.__citation__` exposes the default BibTeX entry as a `str`
  for one-liners.
- `CITATION.cff` at the repository root — GitHub renders a "Cite this
  repository" button from it; bundled in the sdist via `MANIFEST.in`.
- Zenodo DOI [10.5281/zenodo.19933900](https://doi.org/10.5281/zenodo.19933900)
  (concept DOI; always resolves to the latest archived release). The
  DOI now appears in `sp.citation()` output, the README citation block,
  and a DOI badge alongside the existing review-status badge.
- `.zenodo.json` so future GitHub Releases mint version-specific DOIs
  with consistent metadata (creators, keywords, license, related
  identifiers).

## [1.12.0] — 2026-04-30

### Headline

The whole `dml/` module got a careful audit. `sp.dml` / `sp.dml_panel`
/ `sp.dml_model_averaging` all stay backwards-compatible at the
call-site level (existing scripts keep working) but several internal
numerical behaviours change — see the **⚠️ Correctness** section and
[`MIGRATION.md`](MIGRATION.md).

### ⚠️ Correctness

- `sp.dml(model='irm')` and `sp.dml(model='iivm')` now use
  `StratifiedKFold` (stratified by D and Z respectively) — the old
  `KFold` could produce a fold whose subgroup mask was empty, in which
  case the AIPW score for that fold's test rows was silently filled
  with zeros (biased point estimate, biased SE). Empty subgroups now
  raise `IdentificationFailure` with a clear remedy. Estimates may
  shift slightly on data sets where the old `KFold` happened to
  produce extreme folds.
- `sp.dml_panel(binary_treatment=True)` is now a deprecated no-op. The
  previous classifier path fit a propensity on within-demeaned features
  but raw {0,1} labels — there is no clean interpretation as
  E[D̃ | X̃] for the result. The estimator now always uses a regressor
  on D̃ (PLR-with-FE is agnostic to D's type). A `DeprecationWarning`
  is emitted, and `D ∈ {0,1}` is validated when the flag is True.
- `sp.dml_model_averaging` now drops rows with NaN in y / treat /
  covariates / sample_weight (matching every other DML class);
  previously NaNs propagated into sklearn fits and could produce NaN
  estimates undetected by the existing `denom < 1e-12` guard.
- `sp.dml_model_averaging`: the default `weight_rule` is now
  `"short_stacking"` — Ahrens, Hansen, Schaffer & Wiemann (2025, JAE)
  eq. 7 — which solves a constrained least squares stacking problem on
  cross-fitted nuisance predictions and plugs the stacked nuisance
  into a single PLR moment equation. The previous `"inverse_risk"`
  default (heuristic 1/MSE-weighted average of per-candidate θ̂_k) was
  not in the cited paper and is preserved as a clearly labelled
  baseline. New `"single_best"` matches the paper's footnote 8
  formulation. Per-nuisance stacking weights are exposed as
  `model_info["weights_g"]` / `weights_m`.
- `sp.dml(model='pliv')` raises `RuntimeError` when the
  ML-residualised partial correlation `|corr(z̃, d̃)|` falls below
  `1e-3` (was `1e-6`, too lenient to catch genuine weak-IV collapse).
  A new `model_info["diagnostics"]` block reports the partial
  correlation and an approximate first-stage F.

### Added

- All four `sp.dml(model=…)` variants now accept a `random_state=`
  argument (default 42) controlling fold assignment. Repeated splits
  use `random_state + rep` so a single seed fully determines the
  result.
- `sample_weight=` support on `sp.dml(model='plr')`, `sp.dml(model='irm')`,
  `sp.dml_panel`, and `sp.dml_model_averaging` (any weight rule). The
  weighted estimator uses a Z-estimator sandwich variance throughout.
  `sp.dml(model='pliv')` and `sp.dml(model='iivm')` raise
  `NotImplementedError` if a non-trivial weight is supplied — the
  weighted Wald-ratio variance derivation is non-trivial and lands
  in a follow-up. `sample_weight` may be passed as a 1-D array, a
  pandas Series, or a column name string.
- New `model_info["diagnostics"]` block on every variant:
  - PLR: residual scales, partial correlation y_resid·d_resid,
    within-R² of each nuisance.
  - IRM: propensity p01/p99/min/max, n clipped below/above the
    `[0.01, 0.99]` overlap clip, n times the subgroup g̃₁/g̃₀ fit
    fell back to the subgroup mean.
  - IIVM: instrument-propensity p01/p99/min/max, clipping counts,
    subgroup fallbacks for both g(z, X) and r(z, X), and
    E[ψ_b] (the LATE Wald-ratio denominator — proximity to zero
    indicates a weak first stage).
  - PLIV: first-stage partial correlation, approximate first-stage
    F, residual scales.
  - panel_dml: y/d residual std, within-R², cluster Ω, weighted flag.
- `sp.dml_panel(sample_weight=…)` does a *weighted* within transform
  (subtract weighted unit / time means) and reports a weighted
  Liang-Zeger cluster SE.

### Changed

- Internal flag rename `_BINARY_TREATMENT` → `_ML_M_TARGET_BINARY` and
  `_BINARY_INSTRUMENT` → `_ML_R_TARGET_BINARY` on the per-model DML
  classes. The new names describe the nuisance-target shape
  (the IIVM `ml_m` actually models the instrument propensity, not D).
  These flags are private (underscore-prefixed); no public API change.
- `paper.bib`: filled in the missing `volume` / `number` / `pages`
  fields on `@ahrens2025model` (40(3):249–269), verified via the Wiley
  Online Library record and the JAE issue listing.

### Internal

- Per-rep diagnostics now flow back to `model_info["diagnostics"]`
  via a new `_aggregate_diagnostics` helper on `_DoubleMLBase`. Each
  subclass populates `self._last_rep_diagnostics` inside
  `_fit_one_rep`; the base merges across reps (sum for counts, mean
  for floats, OR for booleans, concat for lists).

### ⚠️ Correctness — TMLE module audit pass

- `sp.tmle.SuperLearner` previously ran NNLS and post-hoc-normalised
  weights to sum to 1, which is **not** the simplex-constrained
  optimum (rescaling an unconstrained NNLS solution gives the simplex
  optimum only when the unconstrained sum already equals 1, a
  measure-zero event). Replaced with a direct SLSQP QP on the
  simplex; ensemble predictions are now genuinely the convex
  combination minimising squared loss. Affects every downstream
  caller — `sp.tmle`, `sp.hal_tmle`, and any user code that builds a
  Super Learner directly. Numerical results will shift slightly on
  data sets where the old NNLS solution did not happen to be on the
  simplex.
- `sp.tmle.ltmle` censoring half-implementation: the regime-following
  indicator now includes ``& (C_k_obs == 1)`` so censored units are
  excluded from the targeting equation rather than continuing to
  contribute with `1/p_c`-inflated weights. (`sp.tmle.ltmle_survival`
  was already correct on this; `ltmle.py` was the regression.)
- `sp.tmle.ltmle_survival` influence function: previously used
  ``-H * (T_k - h_star_regime)`` summed across intervals as the
  influence function for **both** the RMST contrast *and* the
  terminal risk difference at K. The proper EIF for :math:`E[S^a(t)]`
  (Cai & van der Laan 2020) needs the survival-product factor
  :math:`S^a(t)/S^a(j)` and the IC for the terminal RD at K is the
  EIF of :math:`S^a(K)` alone (NOT the cumulative-across-K RMST IC).
  Refactored ``_run_regime`` to expose the per-subject sequences
  ``S_seq``, ``h_star_seq``, ``H_seq``, ``T_seq``; the call site now
  computes the RMST and terminal-RD EIFs separately via
  ``_eif_rmst`` and ``_eif_survival_at_k``. SE estimates change —
  generally smaller for RMST (was conservative), and the terminal-RD
  SE is now correctly tied to its target functional rather than
  picking up RMST's cross-time aggregation.
- `sp.hal_tmle(variant='projection')` was a **no-op** in v1.11.x and
  earlier. The projection variant ran an ad-hoc shrinkage on
  ``model_info["eps"]`` after the point estimate had already been
  computed; the variant flag did not change the estimate. The path
  now raises :class:`NotImplementedError` honestly until the proper
  Riesz-projection step (Li-Qiu-Wang-vdL 2025 §3.2) is ported.
- `sp.hal_tmle` docstring previously claimed the basis was "rich
  enough to approximate any càdlàg function of bounded variation",
  the property of full HAL (Benkeser & van der Laan 2016). The
  implementation only builds **main-effects** indicator basis
  functions :math:`\\mathbb 1\\{x_j \\le a_j\\}` — i.e.
  L1-penalised additive piecewise-constant regression, NOT full HAL.
  Docstring is corrected; numerical behaviour unchanged.

### Fixed — TMLE convergence + overlap diagnostics

- `sp.tmle._fit_epsilon` now emits a `UserWarning` when the Newton
  iteration on the fluctuation parameter fails to converge in
  ``max_iter`` steps, instead of silently returning the last value
  (which yields a non-targeted plug-in). The warning includes the
  final score magnitude and ε for diagnosis.
- `sp.tmle` now reports `model_info['propensity_diagnostics']` (min,
  max, p01, p99, n clipped below/above, clip share) and emits a
  `UserWarning` when ≥ 5 % of propensities hit the
  `propensity_bounds` clip — same overlap convention as
  `sp.metalearner`. AIPW scores blow up at e≈0/1, so heavy clipping
  silently changes the estimand from ATE in the population to ATE
  on the trimmed sample.
- `sp.tmle.SuperLearner(task='classification')` validates that the
  target is binary (was silently dropping non-{0,1} columns of
  `predict_proba`); switches to `StratifiedKFold` so every fold has
  both classes; `predict()` clips to (1e-6, 1-1e-6) for
  classification (was inconsistent with `predict_proba` which
  already clipped).

### Fixed — TMLE / HAL-TMLE citations (§10 verification pass)

- `paper.bib` now records three previously-uncatalogued HAL-TMLE
  references with full Crossref/arXiv-verified metadata (added
  2026-04-30):
  - `@li2025regularized` — arXiv:2506.17214, verified via
    arxiv.org. Earlier inline-cited title in `hal_tmle.py` was
    `"Highly Adaptive Lasso Implementations"`; the paper's actual
    title is `"Highly Adaptive Lasso Implied Working Models"` —
    fixed in docstring + `model_info['citation']`.
  - `@vanderlaan2023efficient` — IJB 19(1):261–289,
    doi 10.1515/ijb-2019-0092, verified via degruyterbrill.com.
  - `@benkeser2016highly` — IEEE DSAA 2016, pp. 689–696,
    doi 10.1109/DSAA.2016.93, verified via Crossref API.
- `tmle.py:_CITATIONS['tmle']` now includes the `vanderlaan2006targeted`
  reference that the docstring already cites (was missing — docstring
  promised it via ``[@vanderlaan2006targeted]`` but the inline BibTeX
  registered only ``vanderlaan2007super``). Author punctuation /
  capitalisation aligned to `paper.bib`.
- `ltmle_survival.py` `cai2020step` reference reformatted to match
  paper.bib (year 2020 vs the previous docstring's 2019; the IJB
  volume's nominal year is 2020).
- Dropped the dangling "Qian-van der Laan Section 4" reference from
  `hal_tmle.py` projection-variant docstring (the paper was never in
  References section and the cited Section 4 doesn't exist in any
  HAL-TMLE paper).

### ⚠️ Correctness — `sp.metalearner` unifies ATE / SE via AIPW influence function

- ATE for **all** learners (`learner ∈ {'s','t','x','r','dr'}`) is now
  the mean of the AIPW (DR) pseudo-outcome
  :math:`\varphi_i = \hat\mu_1(X_i) - \hat\mu_0(X_i) +
  D_i(Y_i-\hat\mu_1(X_i))/\hat e(X_i) -
  (1-D_i)(Y_i-\hat\mu_0(X_i))/(1-\hat e(X_i))`,
  and SE is :math:`\sigma(\varphi)/\sqrt n`. AIPW is the
  semiparametric-efficient estimating function for :math:`E[Y(1)-Y(0)]`
  (van der Laan & Robins 2003; Kennedy 2023), so the SE is valid for
  *any* CATE estimator the user picks via `learner=`.
- Previously S/T/X/R-Learner used `mean(τ̂(X))` for ATE and a
  re-sampling bootstrap of the **fitted** CATE values for SE. That
  bootstrap silently treated τ̂ as fixed and only captured empirical-
  mean variation — completely missing the dominant component
  (estimation error in τ̂ itself). Result: SEs were systematically too
  small and CIs severely under-covered.
- DR-Learner: ATE was previously `mean(τ̂(X))` from the regularised CATE
  fit, while SE used `std(φ)/√n` from the raw pseudo-outcome — a
  finite-sample inconsistency that disappears under the new
  `mean(φ)` ATE.
- New `model_info['se_method'] = 'aipw_influence_function'` (was
  `'bootstrap'` for S/T/X/R, `'influence_function'` for DR).
  `model_info['ate_method'] = 'aipw_dr_pseudo_outcome'`.
  `n_bootstrap` parameter is **deprecated and ignored**; will be
  removed in a future minor release.
- New `model_info['aipw_diagnostics']` block reports clipped-propensity
  counts and share. `UserWarning` fires when ≥ 5 % of propensities hit
  the (0.01, 0.99) overlap clip — overlap is poor and the AIPW score
  may be biased toward the trimmed sample.

### Fixed — Künzel et al. 2019 author hallucination (§10 red line)

- `metalearners.py` previously listed `Seetharam, Liang, Athey` as
  co-authors of Künzel et al. 2019 PNAS — those are **invented
  names**. Correct authors per the canonical record in `paper.bib`:
  **Künzel, Sekhon, Bickel, Yu** (PNAS 116(10), 4156–4165,
  doi 10.1073/pnas.1804597116). Both the docstring and the inline
  BibTeX (used by `result.cite()`) now match `paper.bib` byte-for-byte.
  Verification path: `paper.bib:99` ← Crossref / doi.org / Google
  Scholar all confirm `Künzel, Sekhon, Bickel, Yu`.
- `Kennedy, Edward H` (no period) was also out of sync with
  `@kennedy2023towards` in `paper.bib` (`Edward H.`); fixed.

### Refactor — PLR variance code: `psi` → `psi_inner` / `psi_score`

- `dml/plr.py` previously named the inner residual
  `(Y − ĝ − θ̂(D − m̂))` as `psi`, even though the Neyman-orthogonal
  score is the *product* with `d_resid`. The misnomer made the
  variance line `np.mean((d_resid * psi)**2)` look wrong on a
  cursory read. Renamed to `psi_inner` (the residual) and
  `psi_score` (the actual score `psi_inner * d_resid`); math is
  unchanged. PLIV/IIVM already used the consistent `psi`-as-score
  convention.

## [1.11.4] — 2026-04-30

### Fixed — `sp.dml` accepts string learner aliases

- `sp.dml(..., ml_g='rf', ml_m='rf')` previously crashed with
  ``TypeError: Cannot clone object 'rf' (type str): it does not seem to
  be a scikit-learn estimator …`` once cross-fitting reached
  ``sklearn.base.clone``. The error surfaced in **all four** DML
  variants (PLR / IRM / PLIV / IIVM), not just PLR.
- New `dml/_learners.py` resolves user-supplied strings into
  appropriately configured scikit-learn estimators:
  ``'rf'`` / ``'gbm'`` / ``'lasso'`` / ``'ridge'`` / ``'linear'`` /
  ``'ols'`` / ``'logistic'`` / ``'xgb'`` / ``'lgbm'`` (case-insensitive,
  with common synonyms). Classifier variants are selected automatically
  for the propensity (``ml_m`` under ``model='irm'``) and instrument
  (``ml_r`` under ``model='iivm'``) roles.
- Estimator objects (anything exposing `.fit` + `.get_params`) pass
  through unchanged. Unknown aliases / wrong types now raise an
  immediate, descriptive `ValueError` / `TypeError` at construction
  time rather than the cryptic clone error mid-cross-fit.
- Optional dependencies (`xgboost`, `lightgbm`) are imported lazily —
  not installed → clean `ImportError` with install hint.

## [1.11.3] — 2026-04-30

### Fixed — output layer graceful degradation restored

- `to_excel` / `to_word`: revert optional-dependency handling from
  ``raise ImportError`` back to ``warnings.warn() + return``, restoring
  graceful degradation when ``openpyxl`` / ``python-docx`` are absent.
  (Regression introduced in v1.11.2.)

## [1.11.2] — 2026-04-29

Internal refactor only — collapses `esttab`, `modelsummary`, and
`outreg2` to thin facades over the shared `regtable` engine.  No API
changes, no estimator numerics changed.

### Changed — output layer facades collapsed into `regtable`

- `outreg2` → thin `regtable` facade; all formatting delegated to
  `regtable.py` (shared `FormatOptions` / star formatter / numeric
  formatter).  Old `outreg2.py` retained as import shim for backward
  compatibility.
- `modelsummary` → thin `regtable` facade; the summary layout logic is
  now `regtable.FormatOptions` driven.  Old `modelsummary.py` kept as
  import shim.
- `esttab` / `EstimateTable` → thin `regtable` facade; identical
  dispatch path as `outreg2` and `modelsummary`.
- The `regtable` snapshot baselines added in `c608528` ensure any
  future drift is caught by the test suite.

### Tests

- `test_regtable.py` (new): 12 snapshot cases covering every
  `fmt` variant, star placement, confidence-interval style, and
  `reorder` / `drop` / `keep` path.

## [1.11.1] — 2026-04-29

Polish patch for the v1.11 agent surface. Closes the four
"留意 / 没做" items from the v1.11 release notes: mcp_server.py
1,475-line bloat, from_stata Tier 3, deeper from_r, and the MCP
sampling abstraction layer. **No estimator numerics changed**.

### Added — `from_stata` Tier 3 (long-tail, ~95% coverage)

- `ppmlhdfe` → `sp.ppmlhdfe` (Correia-Guimarães-Zylkin Poisson-PML
  with HDFE, multi-FE absorb).
- `mlogit` / `oprobit` → `sp.glm(family='multinomial' /
  'ordered_probit')` with a translation note pointing strict
  diagnostics at `result.raw_model`.
- `xtabond` / `xtdpdsys` → `sp.xtabond` / `sp.xtdpdsys` (Arellano-
  Bond difference / Blundell-Bond system GMM).
- `bunching` → `sp.bunching` (Saez 2010, Kleven-Waseem 2013).
- `boottest` → `sp.wild_cluster_bootstrap` (Roodman-Webb wild-cluster
  bootstrap; takes a fitted result).
- `mi estimate: <inner>` → translation hint pointing at
  `sp.mi_estimate` (Stata's nested grammar isn't auto-parsed).
- 33 alias entries / 29 distinct handlers total.

### Added — `from_r` deepening (5 → 11 callables)

- `glm` with smart routing: `family=binomial` → `sp.logit`,
  `family=binomial(link="probit")` → `sp.probit`,
  `family=poisson` → `sp.poisson`, otherwise `sp.glm`.
- `lmer` → `sp.multilevel`; `glmer` → `sp.glmer`.
- `plm(formula, data=df, model='within', index=c('id','t'))` →
  `sp.panel(method='within', id='id', time='t')`.
- `matchit(treat ~ x, data=df, method='nearest')` → `sp.match`
  with method-name aliasing (nearest→nn, genetic→genmatch).
- R Synth synth() now emits a structured field-mapping note
  (predictors → predictors, dependent → outcome, unit.variable →
  unit, time.variable → time, treatment.identifier →
  treated_unit, time.predictors.prior[max] + 1 → treatment_time).

### Added — MCP `sampling/createMessage` abstraction (opt-in)

- New `agent/_sampling.py`:
  - `request_sampling(messages, max_tokens, ...)` — server-to-client
    LLM request; blocks until response or timeout.
  - `set_capability(bool)` / `get_capability()` — client capability
    advertisement flag.
  - `set_writer(callable)` / `route_response(message)` — stdio
    writer registration + reply matcher.
- Wire-up:
  - `_handle_initialize` reads `params.capabilities.sampling`.
  - `handle_request` routes JSON-RPC replies to pending sampling
    requests via `route_response`.
  - `serve_stdio` registers / clears the writer + capability flag.
- Fail-closed: `UnsupportedSamplingError` raised when no capability
  is advertised OR no writer is registered, so existing LLM helpers
  (`llm_dag_propose` / `llm_evalue` / `llm_sensitivity`) keep
  working via their user-API-key paths until clients (Claude
  Desktop, Cursor, …) advertise `sampling`.
- `STATSPAI_MCP_SAMPLING_TIMEOUT_SECONDS` (default `60`) caps every
  request; `SamplingTimeoutError` on overage.

### Changed — `mcp_server.py` split into leaf modules

- v1.11.0: `mcp_server.py` was 1,475 lines.
- v1.11.1: split into 5 leaf modules:
  - `_errors.py` (35 LOC) — `RpcError` / `InvalidParamsError` /
    `ResourceNotFoundError` typed taxonomy.
  - `_prompts.py` (344 LOC) — 10 prompt templates plus `SafeDict`,
    `handle_prompts_list`, `handle_prompts_get`.
  - `_resources.py` (313 LOC) — catalog text / function detail /
    handle reads / templates list. Handlers accept `json_default`
    plus error classes via dependency injection (no circular import).
  - `_data_loader.py` (176 LOC) — `load_dataframe` / size cap /
    LRU cache / remote-URL routing.
  - `_sampling.py` (227 LOC) — see above.
  - `mcp_server.py` shrunk to **817 lines** (well under the
    CLAUDE.md §4 ~800-line guideline).
- All v1.x private names re-exported via thin import shims so
  external code reaching for `_PROMPTS` / `_load_dataframe` /
  `_RpcError` etc. still works.
- Test fixture compatibility preserved: `monkeypatch.setattr(
  agent.tools, '_resolve_fn', …)` continues to take effect because
  the dispatch path looks up `_resolve_fn` via the parent package
  namespace at call time (carried over from the v1.11.0 split).

### Tests

- `test_mcp_sampling.py` (10 cases, new):
  - Fail-closed when capability or writer unset.
  - Round-trip via mock client thread (success + error envelope).
  - Timeout via env-var override.
  - serve_stdio integration (capability + writer lifecycle).
  - Unsolicited / malformed reply handling.
- `test_translation.py` extended:
  - 8 new Stata Tier-3 round-trips + 3 edge cases.
  - 10 new R round-trips.
  - 61 → 82 cases; coverage assertion checks every distinct handler.

424/424 pass across all agent + MCP + translation + runner +
sampling suites.

## [1.11.0] — 2026-04-29

Agent-native infrastructure follow-up to v1.10. Closes the four
follow-up items the v1.10 release notes flagged: tools.py subpackage
split, Stata/R command translators, concurrent runner with progress
notifications, and tool-call timeouts. **No estimator numerics
changed**; this is the agent-orchestration layer.

### Added — `from_stata` / `from_r` translators

- New `agent/_translation/` subpackage exposes
  `from_stata(line) → {ok, tool, arguments, python_code, notes,
  source, input}` and `from_r(line)` with the same shape.
- 21 distinct Stata handlers / 25 alias entries covering ~85% of
  real econ workflows:
  - **Tier 1** (~60% coverage): `regress` / `reg`, `xtreg`,
    `reghdfe`, `ivreg2` / `ivregress`, `csdid`, `did_imputation`,
    `synth`, `rdrobust`.
  - **Tier 2** (push to ~85%): `probit` / `logit` / `poisson` /
    `nbreg` (shared GLM scaffold), `tobit`, `heckman`, `rdplot`,
    `rddensity`, `teffects` (ipw / nnmatch / psmatch / ra /
    aipw), `margins` / `marginsplot`, `contrast`, `test`, `xtset`
    / `tsset` (no-op note).
- 5 R handlers: `feols`, `felm`, `lm`, `att_gt` / `did`. fixest's
  `y ~ x | id^year | (d ~ z) | cluster` pipe-form decomposition
  preserved.
- Returns close-match `suggestions` for unrecognised commands —
  never silently guesses.
- Surfaces `notes` for partial mappings (e.g. Stata `if` clause
  → `df.query(...)` instructions).
- Tests: `tests/test_translation.py` — 61 cases, every distinct
  Stata handler covered by ≥1 round-trip.

### Added — concurrent runner + progress notifications + timeouts

- New `agent/_runner.py`:
  - `run_with_progress(work, progress_token, timeout, drain)` —
    zero-arg `work()` runs in a worker thread; main loop drains a
    thread-safe queue.
  - `progress(value, total, message)` — tool-side helper. No-op
    when no channel is registered (safe for in-process tests /
    direct `execute_tool` callers).
  - `tool_timeout()` — reads `STATSPAI_MCP_TOOL_TIMEOUT_SECONDS`
    (default `600`; `0` disables). Hard wall-clock cap; `TimeoutError`
    surfaces as `-32000` with the env-var name embedded.
- `_handle_tools_call` runs every dispatch through the runner;
  reads `params._meta.progressToken` per MCP 2024-11-05.
- `_make_progress_drain` writes `notifications/progress` JSON-RPC
  messages to the active stdio sink mid-call.
- `serve_stdio` registers / unregisters the stdout sink so
  in-process tests don't accidentally write to a closed handle.
- Threading rather than asyncio for cross-platform reliability —
  decision documented in `_runner.py`.

### Changed — `tools.py` split into `agent/tools/` subpackage

- Pre-1.11: `agent/tools.py` was 1,024 lines.
- v1.11 layout:

  ```text
  agent/tools/
  ├── __init__.py        # public API + legacy private re-exports
  ├── _helpers.py        # _scalar_or_none / _default_serializer / _identification_serializer
  ├── _dispatch.py       # tool_manifest / execute_tool / _resolve_fn
  └── _specs/            # TOOL_REGISTRY split by family
      ├── _regression.py / _did.py / _iv.py / _rd.py
      └── _matching.py / _diag.py / _orchestrate.py
  ```

- Public API unchanged (`from statspai.agent.tools import
  tool_manifest, execute_tool, TOOL_REGISTRY`).
- Legacy private imports preserved (`from .tools import
  _default_serializer` continues to resolve).
- Test-fixture hook preserved: `monkeypatch.setattr(agent.tools,
  '_resolve_fn', …)` works because `execute_tool` looks up
  `_resolve_fn` via the parent package namespace at call time.
- `causal` / `recommend` bespoke serializers promoted from inline
  lambdas to module-level functions (readable in stack traces).

### MCP wire-up

- `from_stata` / `from_r` registered as workflow tools in
  `WORKFLOW_TOOL_NAMES`, dataless override list, and surface in
  `tools/list`.
- 393/393 pass across `test_mcp_protocol.py` /
  `test_mcp_error_envelope.py` / `test_mcp_result_handle.py` /
  `test_mcp_enrichment.py` / `test_mcp_image_content.py` /
  `test_mcp_pipelines.py` / `test_mcp_prompts_expanded.py` /
  `test_mcp_runner.py` (new) / `test_translation.py` (new) plus
  the existing agent + registry + help + exceptions suites.

### Known follow-ups (not in 1.11)

- `mcp_server.py` is now 1,475 lines — the 10 prompt-template
  dictionaries account for most of the bloat. Splitting them into
  `_prompts.py` is a half-day mechanical follow-up.
- MCP `sampling/createMessage` server-initiated LLM requests (for
  `llm_dag_propose` etc. to reuse the client's auth) deferred
  pending Claude Desktop / Cursor capability advertisement.

## [1.10.0] — 2026-04-29

Agent-native / MCP layer overhaul. Closes the chained-workflow gap that
the v1.9 stateless tools couldn't span and turns the `sp.agent` /
`statspai.agent.mcp_server` surface into a proper experimentation
workbench. **No estimator numerics changed**; this is purely the
discovery / orchestration / output layer.

### Added

- **Result handles (`as_handle=True`)** — every `execute_tool` /
  `tools/call` invocation can now return a `result_id` /
  `result_uri` (`statspai://result/<id>`) pointing to the fitted
  object. Backed by an in-process LRU cache (`agent/_result_cache.py`,
  default 32 entries, env override `STATSPAI_MCP_RESULT_CACHE_SIZE`).
  Resource read returns the agent-detail `to_dict` payload + a
  provenance block (originating tool + arguments + class name).
- **Handle-based workflow tools** — `audit_result`, `brief_result`,
  `sensitivity_from_result`, `honest_did_from_result` accept a
  `result_id` instead of forcing the LLM to ferry betas/sigma
  arrays back across turns. `honest_did_from_result` auto-extracts
  `betas` / `sigma` / `num_pre_periods` / `num_post_periods` from
  the cached result (CallawaySantanna, EventStudy, BJS, SA shapes
  supported via best-effort attribute walk).
- **First-class workflow primitives** — `audit`, `preflight`,
  `detect_design`, `brief` are now hand-curated MCP tools (previously
  surfaced only via the auto-generated manifest with one-line
  descriptions). Schemas describe expected columns explicitly.
- **`bibtex` tool** — pulls verified BibTeX entries from `paper.bib`
  (single source of truth per CLAUDE.md §10). Unknown keys return
  empty bodies + close-match suggestions, never fabricated entries.
  Closes the citation-hallucination loophole at the source.
- **Composite pipelines** — `pipeline_did` / `pipeline_iv` /
  `pipeline_rd` run preflight + estimator + audit + sensitivity +
  brief in one call, return a markdown narrative + cached
  `result_id` + per-stage status. `pipeline_rd` attaches an
  `rdplot` PNG as MCP image content.
- **Image content blocks** — `_handle_tools_call` promotes any
  `_plot_png` bytes returned by a tool to a second
  `{type: "image", mimeType: "image/png"}` content block (Claude
  vision and any MCP image-capable client renders it inline). New
  `plot_from_result` tool renders the canonical diagnostic plot for
  a cached result (event-study / rdplot / synth-gap / love-plot /
  cate-plot / coef-plot — auto-detected by class name).
- **Output enrichment** — every tool return now carries:
  - `next_calls` — pre-built `tools/call` payloads with `result_id`
    and forwarded base args; agents copy-paste verbatim.
  - `citations` — verified bib keys (static map; empty list ⇒
    intentionally absent, never invent) + BibTeX bodies pulled from
    `paper.bib`.
  - `narrative` — short markdown digest (method + estimate + CI +
    N + violations).
- **Expanded prompt templates** — `prompts/list` jumps from 3 to 10:
  `audit_did_result` (rewired to `pipeline_did`),
  `audit_iv_result`, `audit_rd_result`, `design_then_estimate`,
  `robustness_followup`, `paper_render`, `compare_methods`,
  `policy_evaluation`, `synth_full`, `decompose_inequality`.
- **Schema injections** — every MCP tool now exposes:
  - `data_path` (URL-aware: `s3://`, `gs://`, `https://`, plus
    `.dta` / `.feather` / `.arrow` / `.jsonl` in addition to the
    legacy `.csv` / `.parquet` / `.xlsx` / `.json`).
  - `data_columns` — column projection for parquet / feather / stata
    fast partial reads.
  - `data_sample_n` — deterministic uniform random subsample (seed=0)
    for fast iteration on large panels.
  - `result_id` — handle reference for chained calls.
  - `as_handle` — opt-in result caching.
- **`initialize` returns a session-level `instructions` block**
  describing the recommended workflow (detect_design → preflight →
  fit `as_handle=true` → `audit_result` → `*_from_result` →
  `bibtex`).
- **`statspai://result/{id}` URI template** advertised via
  `resources/templates/list`.

### Changed

- **`_DATALESS_TOOLS` is now registry-derived.** A new
  `_dataless_tool_names()` helper walks the registry and marks any
  spec without a required `data` parameter as dataless; the
  hand-curated `_DATALESS_OVERRIDES` set covers stub-backed tools the
  registry can't reach (workflow / handle / bibtex / plot tools). The
  legacy `_DATALESS_TOOLS` constant stays as a backward-compat alias.
- **`auto_tool_manifest(max_tools=...)` default bumped 250 → 500**
  and emits a `RuntimeWarning` when more eligible tools exist than
  the cap admits — silent truncation was hiding registry growth.
- **`tool_manifest()` no longer silently swallows auto-merge
  failures.** A `RuntimeWarning` fires before the curated-only
  fallback so operators / CI log scrapers can detect registry
  introspection regressions.
- **`_load_dataframe` is LRU-cached by `(path, mtime, columns)`**
  so repeated `tools/call` invocations on the same file are O(1)
  after the first load. New 2 GiB default file-size cap (env
  override `STATSPAI_MCP_MAX_DATA_BYTES`; set to `0` to disable).
- **Bad/missing `data_path` now surfaces as JSON-RPC `-32602`
  (invalid params)** rather than the generic `-32000`. Clients
  branching on error codes get a cleaner signal.
- **Traceback exposure on `-32000` errors gated by
  `STATSPAI_MCP_DEBUG=1`.** Production deployments no longer leak
  internal paths / class names through the JSON-RPC error envelope
  by default.
- **`_json_default` covers every type we've actually seen leak
  through** the agent / MCP wire: `np.bool_`, `np.complexfloating`,
  `np.datetime64`, `np.timedelta64`, NaN / Inf → `null`,
  `pd.Index` / `Timedelta` / `Categorical` / `Interval`,
  `set` / `frozenset`, `bytes` (b64-wrapped),
  `Decimal`, `pathlib.PurePath`, `Enum`, dataclasses.
- **`oaxaca`-style estimators with their own `detail` parameter
  shadowed via MCP.** The schema's `detail` enum is server-side
  control (forwarded to `result.to_dict(detail=...)`); collisions
  are resolved by force-overwriting the registry's version. Affected
  estimators remain reachable via the direct Python API.

### Tests

- New: `test_mcp_result_handle.py` (31 cases) — result-cache LRU,
  resource read, handle-based workflows, `_json_default` types,
  `STATSPAI_MCP_DEBUG` gating.
- New: `test_mcp_enrichment.py` (14 cases) — `next_calls` /
  `citations` / `narrative` shape; `bibtex` round-trip.
- New: `test_mcp_image_content.py` (4 cases) — PNG promotion to
  MCP image content block.
- New: `test_mcp_pipelines.py` (7 cases) — pipeline_did / iv / rd.
- New: `test_mcp_prompts_expanded.py` (5 cases) — full 10-prompt
  template surface.
- Existing `test_mcp_protocol.py` updated to use the registry-derived
  `_dataless_tool_names()` helper rather than the static override
  set.

321/321 pass across all `tests/test_*agent*.py` + `test_mcp_*.py` +
`test_registry.py` + `test_help.py` + `test_exceptions.py`.

### New modules

- `src/statspai/agent/_result_cache.py` — bounded LRU cache + entry
  metadata.
- `src/statspai/agent/auto_dispatch.py` — registry-driven dispatch
  for non-curated tools (filters kwargs against `ParamSpec`).
- `src/statspai/agent/workflow_tools.py` — handle-based + workflow
  primitive tools (audit_result / brief_result / *_from_result /
  audit / preflight / detect_design / brief / plot_from_result /
  bibtex).
- `src/statspai/agent/pipeline_tools.py` — pipeline_did / pipeline_iv
  / pipeline_rd composites.
- `src/statspai/agent/_enrichment.py` — `next_calls` + `citations` +
  `narrative` builder.

## [Unreleased]

### Changed — output module PR-B (continuation of v1.11.x cleanup)

- **`esttab` / `EstimateTableResult` are now thin facades over `regtable`.**
  ``output/estimates.py`` previously housed a ~500-line ``EstimateTable``
  class that re-implemented the full renderer pipeline (text / LaTeX /
  HTML / Markdown / CSV / DataFrame). PR-B/5c collapses it; the
  ``esttab()`` function now translates Stata-flavoured kwargs and
  forwards to ``sp.regtable``, and ``EstimateTableResult`` becomes a
  thin pass-through wrapper around the resulting ``RegtableResult``
  that preserves the legacy type identity.
  - Net code: ``output/estimates.py`` 987 → 526 lines (-47%).
    Helpers used by ``regression_table`` / ``mean_comparison`` /
    ``_inline`` (``_ModelData``, ``_extract_model_data``,
    ``_ci_bounds``, ``_format_stars`` re-exports, ``_latex_escape`` /
    ``_html_escape``, ``_STAT_ALIASES`` / ``_STAT_DISPLAY``,
    ``eststo`` / ``estclear`` global store) are kept verbatim.
  - ``EstimateTableResult.to_csv()`` is implemented via
    ``to_dataframe().to_csv()`` (regtable does not natively expose CSV;
    the dataframe path is byte-identical to what the legacy esttab
    produced).
  - The four exclusive-output flags ``se`` / ``t`` / ``p`` / ``ci``
    map to regtable's ``se_type=`` with priority ``ci > p > t > se``
    (matches legacy behaviour).
  - First call emits ``DeprecationWarning`` pointing to ``sp.regtable``.

- **`modelsummary` is now a thin facade over `regtable`.** The R-style
  ``modelsummary()`` previously shipped a ~700-line renderer pipeline
  (``_build_coef_rows`` / ``_to_text`` / ``_to_latex`` / ``_to_html``
  / ``_to_excel`` / ``_to_word``) that re-implemented coefficient
  extraction, star formatting, three-line table styling, and every
  export format — duplicating code already maintained by
  ``sp.regtable``.
  - Net code: ``output/modelsummary.py`` 845 → 378 lines (-55%;
    remainder is module docstring + ``coefplot`` kept verbatim +
    ``_extract_coefs`` for ``coefplot``).
  - Rendered output now matches ``regtable`` exactly. The dict form
    of ``stars=`` is reinterpreted (only threshold values used; symbol
    overrides dropped — use ``regtable(notation='symbols')`` for
    ``†/‡/§``). ``se_type='brackets'`` is no longer a separate render
    mode (emits ``UserWarning`` and falls back to parens; use
    ``show_ci=True`` for ``[lo, hi]``). ``se_type='none'`` likewise
    keeps the SE row.
  - First call emits ``DeprecationWarning`` pointing to ``sp.regtable``.
  - ``coefplot`` is unchanged (independent of the table renderer).

- **`outreg2` is now a thin facade over `regtable`.** The Stata-style
  `OutReg2` class and `outreg2()` function previously shipped a
  bespoke 800-line renderer that re-implemented coefficient
  extraction, star formatting, three-line table styling and
  Excel/Word/LaTeX export. Collapsed to ~150 lines that translate
  Stata-flavoured kwargs and forward to ``sp.regtable`` — single
  point of fix for rendering bugs going forward.
  - Net code: `outreg2.py` 804 → 341 lines (-58%).
  - Rendered output now matches `regtable` exactly. **Visible label
    changes**: `Variables` column header → blank (book-tab),
    `R-squared`/`Adj. R-squared`/`Observations`/`F-statistic / Trees`
    → `R²`/`Adj. R²`/`N`/`F`. LaTeX gains a proper star legend.
    Bug fixes: spurious `& None & None` LaTeX cell removed; the
    nonsensical `/ Trees` label that appeared on OLS results is gone.
  - `show_se=False` is no longer supported (regression tables without
    uncertainty are pseudo-science) — emits `UserWarning` and keeps
    the SE row.
  - First call emits `DeprecationWarning` pointing to
    `sp.regtable(...).to_excel(...)`. Plan to remove the facade in
    two minor releases.
  - See [`MIGRATION.md`](MIGRATION.md) for the side-by-side rewrite.

### Added — output module PR-B foundation (B-1)

- **`tests/test_regtable_snapshots.py` snapshot harness.** Locks down
  the byte-stable rendered output of `sp.regtable` for five
  representative fixtures (simple OLS / multi-model / custom stats /
  notes+labels / GLM-logit) across four text formats (text / HTML /
  LaTeX / Markdown) — 20 snapshots total. Whitespace-normalised so
  diffs survive editor newline handling but catch real renderer
  drift. Excel / Word are **not** snapshotted (binary archives are
  brittle); coverage there is via `test_paper_tables_export.py`.
  Update with `STATSPAI_UPDATE_SNAPSHOTS=1 pytest tests/test_regtable_snapshots.py`.

### Added — agent / dispatcher work (other sessions)

- **`sp.panel()` `method=` expanded with friendly aliases + HDFE.**
  ``sp.panel`` already supported a ``method=`` table of 10 classical
  + dynamic estimators (``fe``/``re``/``be``/``fd``/``pooled``/
  ``twoway``/``mundlak``/``chamberlain``/``ab``/``system``).  The
  table is now case-insensitive and accepts intuitive aliases that
  match what users already write (instead of forcing the two-letter
  Stata shorthand):

  - ``fe`` ← ``fixed`` / ``fixed_effects`` / ``within``
  - ``re`` ← ``random`` / ``random_effects``
  - ``be`` ← ``between`` / ``between_effects``
  - ``fd`` ← ``first_difference`` / ``first_diff``
  - ``pooled`` ← ``pooled_ols`` / ``pols`` / ``ols``
  - ``twoway`` ← ``two_way`` / ``two_way_fe`` / ``2way``
  - ``ab`` ← ``arellano_bond`` / ``gmm`` / ``diff_gmm``
  - ``system`` ← ``blundell_bond`` / ``bb`` / ``system_gmm``

  Plus a new ``method='hdfe'`` (a.k.a. ``feols`` / ``reghdfe`` /
  ``absorbed_ols``) route that delegates to ``feols.hdfe_ols`` for
  high-dimensional fixed-effects absorption.  When the formula has
  no ``|`` separator, the dispatcher bolts the ``entity`` and
  ``time`` columns on automatically, so

  ```python
  sp.panel(df, "wage ~ exp", entity='id', time='year', method='hdfe')
  ```

  is equivalent to

  ```python
  sp.hdfe_ols("wage ~ exp | id + year", data=df)
  ```

  This closes the Stata ``reghdfe`` / R ``fixest::feols`` slot in
  the ``sp.panel`` namespace without forcing users to switch APIs.

  ``sp.panel_logit`` / ``sp.panel_probit`` / ``sp.interactive_fe`` /
  ``sp.panel_unitroot`` are intentionally NOT in the ``method=``
  table — they have a different ``(data, y, x, id, time)``-style
  signature and remain accessible as standalone functions.

  Regression-guarded by ``tests/test_panel_dispatcher.py`` (37 new
  tests); 31 existing panel-family tests pass.

- **`sp.match()` `method=` expanded to cover the full matching
  toolkit.** ``sp.match`` was already a function with built-in
  ``method=`` for classical algorithms (nearest / stratify / cem /
  psm / mahalanobis); the table now reaches every
  matching/weighting estimator in ``statspai.matching`` from a
  single entry point:

  - **Classical:** ``nearest`` (default), ``stratify`` /
    ``subclass`` / ``subclassification``, ``cem`` /
    ``coarsened_exact``, ``psm``, ``mahalanobis``.
  - **Weighting:** ``ebalance`` / ``entropy`` /
    ``entropy_balancing`` (Hainmueller 2012),
    ``cbps`` (Imai-Ratkovic 2014),
    ``sbw`` / ``stable_balancing`` (Zubizarreta 2015),
    ``overlap`` / ``ow`` / ``overlap_weights`` (LMZ 2018).
  - **Genetic:** ``genmatch`` / ``genetic`` (Diamond-Sekhon 2013).
  - **Optimization-based:** ``optimal`` / ``optimal_match``
    (Rosenbaum 1989), ``cardinality`` / ``cardinality_match``
    (Zubizarreta 2014).

  The dispatcher translates ``treat`` ↔ ``treatment`` and ``y``
  ↔ ``outcome`` for the few estimators that internally use the
  alternate names (``optimal_match``, ``cardinality_match``).
  Standalone access (``sp.ebalance``, ``sp.cbps``, ``sp.genmatch``,
  ``sp.sbw``, ``sp.optimal_match``, ``sp.cardinality_match``,
  ``sp.overlap_weights``) is unchanged.

  The dispatcher refuses to silently swallow nonsense: passing a
  classical-matching kwarg (``caliper=`` / ``replace=`` /
  ``n_matches=`` / ``bias_correction=`` / ``ps_poly=`` /
  ``n_strata=`` / ``n_bins=``) with ``method='ebalance'`` etc.
  raises ``TypeError: does not accept these classical-matching
  kwargs``.

  Regression-guarded by ``tests/test_match_dispatcher.py`` (31
  new tests); 76 existing matching-family tests still pass.

- **`sp.rd()` is now callable with a unified `method=` table.** Same
  PEP 562 callable-module pattern used for ``sp.iv`` in this
  release: the ``statspai.rd`` subpackage itself dispatches calls,
  while ``sp.rd.rdrobust`` / ``sp.rd.rdplot`` / ``sp.rd.rdsummary``
  and all 35+ existing names continue to resolve. The default
  ``sp.rd(data, y, x, c)`` call equals
  ``sp.rd.rdrobust(data, y, x, c)`` (CCT 2014 local polynomial). 18
  canonical ``method=`` aliases route to:

  - **Local polynomial:** ``rdrobust`` / ``default`` / ``rd`` /
    ``robust`` / ``local_poly`` (CCT 2014).
  - **Honest CIs:** ``honest`` / ``armstrong_kolesar`` / ``ak``.
  - **Local randomisation:** ``randinf`` / ``random`` /
    ``local_randomization``.
  - **Heterogeneous effects:** ``hte`` / ``cate``.
  - **ML+RD:** ``forest`` / ``causal_forest``, ``boost`` / ``gbm``,
    ``lasso``.
  - **Bayesian HTE:** ``bayes_hte`` / ``bayes``.
  - **2D / boundary RD:** ``rd2d`` / ``2d`` / ``boundary``.
  - **Multi-cutoff:** ``rdmc`` / ``multi_cutoff``.
  - **Multi-score / geographic:** ``rdms`` / ``geographic`` /
    ``multi_score``.
  - **Kink (RKD):** ``rkd`` / ``kink``.
  - **RD-in-time:** ``rdit`` / ``time``.
  - **Extrapolation:** ``extrapolate``, ``multi_extrapolate``.
  - **Spillover/interference:** ``interference`` / ``spillover``.
  - **Distributional:** ``distribution``,
    ``distributional_design``.
  - **External validity:** ``external_validity``.

  The dispatcher normalises ``x`` ↔ ``running`` and ``c`` ↔
  ``cutoff`` for methods that use the alternate names internally
  (``rd_bayes_hte``, ``rd_interference``, ``rd_distribution``,
  ``rd_distributional_design``).

  Diagnostics-only functions (``rdbwselect``, ``rdbwsensitivity``,
  ``rdbalance``, ``rdplacebo``, ``rdsummary``, ``rdplotdensity``,
  ``rdpower``, ``rdsampsi``, ``rdwinselect``, ``rdsensitivity``,
  ``rdrbounds``) are intentionally NOT in the ``method=`` table —
  they are not estimators of treatment effects.

  Regression-guarded by ``tests/test_rd_dispatcher.py`` (22 new
  tests); 78 existing rd-family tests still pass.

### Performance

- **`import statspai` cold-start: ~2,070 ms → ~1,680 ms (-19%).**
  ``output/outreg2.py`` now imports ``openpyxl`` lazily inside
  ``_export_with_formatting`` instead of at module load. Top-level
  ``import openpyxl`` was transitively pulling ``PIL`` / ``Pillow``
  via ``openpyxl.drawing.image`` on every session even when the user
  never touched ``outreg2`` (4 references in repo vs 163 for
  ``regtable``). After the fix, no heavy modules
  (``openpyxl`` / ``docx`` / ``PIL`` / ``matplotlib``) are eagerly
  loaded at top level. Also drops unused symbol imports
  (``Border`` / ``Side`` / ``PatternFill`` / ``dataframe_to_rows`` /
  ``write_title``).

### Changed

- **Output module: shared formatter helpers.** New
  ``output/_format.py`` houses the canonical ``format_stars`` /
  ``fmt_val`` / ``fmt_int`` / ``fmt_auto`` / ``is_missing``
  implementations. ``estimates._format_stars/_fmt_val/_fmt_int/_fmt_auto``
  are now thin re-exports under their legacy underscore names; existing
  ``regression_table`` / ``_inline`` imports are unchanged.
  ``outreg2._format_number/_format_pvalue`` and
  ``modelsummary._format_num`` delegate to the canonical helpers.
  Net effect: ~80 lines of duplicate formatters removed; bug fixes in
  one place propagate to every backend.

- **Output module: ``modelsummary._stars_str`` dead code removed.**
  The original implementation had two ``for`` loops where the first
  used ``for/else`` that always overwrote ``best`` to ``''`` before
  the second loop ran — making the first loop unreachable. Cleaned
  to keep only the working logic. Behavior identical for all valid
  inputs.

- **Output module: ``MeanComparisonResult`` extracted.**
  ``output/regression_table.py`` had grown to 3,335 lines and held
  two unrelated result classes. Moved ``MeanComparisonResult`` and
  the public ``mean_comparison()`` API to ``output/mean_comparison.py``
  (510 lines). ``regression_table.py`` is now 2,831 lines (-15%).
  Re-exported from ``regression_table`` for back-compat;
  ``from statspai.output.regression_table import MeanComparisonResult``
  still works. ``sp.list_functions()`` count unchanged.

- **Output module: ``__init__.py`` reorganised.** Imports and
  ``__all__`` are grouped by purpose (regression-table renderers /
  single-table helpers / multi-table bundles / provenance /
  bibliography / adapters), with a docstring documenting that
  ``regtable`` is the canonical regression-table renderer and that
  ``esttab`` / ``modelsummary`` / ``outreg2`` are Stata/R compatibility
  surfaces (full consolidation tracked in
  [``docs/rfc/output_pr_b_consolidation.md``](docs/rfc/output_pr_b_consolidation.md)).
  No symbols added or removed; ``sp.list_functions()`` unchanged at
  973.

- **Top-level `__init__.py` deduplication.** Removed redundant
  ``from .regression.glm`` / ``logit_probit`` / ``count`` imports
  that re-bound the same names twice (lines 245-247 vs 495-501).
  No public name was added or removed; ``sp.glm``, ``sp.logit``,
  ``sp.poisson`` etc. resolve identically to the earlier (canonical)
  binding. Same number of registered functions (973). Net: −5 LOC,
  −5 redundant import statements, identical behaviour.

### Fixed

- **`sp.iv()` is now callable.** Prior to this release, the
  ``statspai.iv`` subpackage *shadowed* the function exposed at
  line 45 of ``statspai/__init__.py`` (because Python attaches an
  imported subpackage to its parent's namespace, and the subpackage
  load happened *after* the function bind). The result was that
  every advertised callsite — registry examples, agent summaries,
  MCP server docs, replication examples, and the live call in
  [`src/statspai/question/question.py:505`](src/statspai/question/question.py#L505)
  — raised ``TypeError: 'module' object is not callable``. Fixed by
  installing a tiny ``ModuleType`` subclass with ``__call__`` on
  ``statspai.iv`` (PEP 562-style) and removing ``iv`` from the
  ``regression.iv`` import line so the subpackage isn't shadowed in
  reverse. Regression-guarded by
  ``tests/test_iv_dispatcher.py::test_sp_iv_is_callable`` (33 new
  tests total).

### Changed

- **Unified IV dispatcher.** ``sp.iv(formula, data, method=...)`` now
  routes 25+ method aliases (case- and dash-insensitive) to 19
  canonical estimators across the ``regression.iv`` /
  ``regression.advanced_iv`` / ``iv/`` / ``deepiv`` / ``bartik``
  modules:

  - **K-class formula path:** ``2sls`` (a.k.a. ``tsls``, ``iv``),
    ``liml``, ``fuller``, ``gmm``, ``jive``.
  - **Modern JIVE:** ``jive1``, ``ujive``, ``ijive``, ``rjive``.
  - **Many-weak:** ``jive_mw``, ``many_weak_ar``.
  - **Lasso:** ``lasso``, ``post_lasso`` (a.k.a. ``bch``).
  - **ML/nonparametric:** ``kernel``, ``npiv``, ``ivdml``,
    ``deepiv``.
  - **Bayesian:** ``bayes``.
  - **LATE/MTE:** ``continuous_late``, ``mte``, ``ivmte_bounds``.
  - **Quantile IV:** ``ivqreg``.
  - **Plausibly exogenous sensitivity:** ``plausibly_exog_uci``,
    ``plausibly_exog_ltz``.
  - **Shift-share:** ``shift_share`` (a.k.a. ``bartik``).

  The dispatcher normalises common alias names (``endog`` →
  ``treat``/``treatment`` for kernel-style methods, ``exog`` →
  ``covariates`` for ``ivdml``, singleton ``instruments=['z']`` →
  ``instrument='z'`` for singular-instrument methods), and refuses
  ambiguous combinations with ``TypeError: Got both 'endog' and
  'treat'``. Standalone access (``sp.iv.kernel_iv``,
  ``sp.iv.bayesian_iv``, ``sp.ivreg``,
  ``from statspai.regression.iv import iv``) is unchanged.
  ``sp.iv.fit(...)`` remains as an explicit alias for the dispatcher.

  Diagnostics functions (``anderson_rubin_test``, ``effective_f_test``,
  ``kleibergen_paap_rk``, ``sanderson_windmeijer``,
  ``conditional_lr_test``) are intentionally *not* in the
  ``method=`` table — they are not estimators.

## [1.9.1] — MCP schema + JSON-RPC error polish

Patch release on top of 1.9.0. **No estimator numerical paths
changed.** Two MCP-server fixes surfaced by strict-schema clients
(Claude Desktop / Cursor) plus one docs typo.

### Fixed

- **MCP `tools/list` schema — dataless tools no longer require
  `data_path`.** Tools whose underlying StatsPAI function does not
  consume a DataFrame (currently ``honest_did`` and ``sensitivity``)
  used to be advertised with ``data_path`` in ``required``. Strict-
  schema MCP clients refused to dispatch the call without a CSV
  path the estimator never reads. ``data_path`` is still exposed as
  an optional property for clients that always send it; only the
  ``required`` list is conditional now. New
  ``_DATALESS_TOOLS = {"honest_did", "sensitivity"}`` is the single
  source of truth in
  [`src/statspai/agent/mcp_server.py`](src/statspai/agent/mcp_server.py)
  — keep in sync with ``TOOL_REGISTRY`` in ``agent/tools.py``.

- **MCP `tools/call` typed error — missing `name` returns -32602.**
  Previously a ``tools/call`` request without a ``name`` field
  raised a generic ``ValueError``, which the dispatcher surfaced as
  ``-32000`` (server fallback). 1.9.0 already promised typed
  JSON-RPC errors for invalid params (``-32602``); this fixes the
  one path that escaped the audit. Regression-guarded by
  ``test_tools_call_missing_name_returns_invalid_params``.

### Docs

- **MIGRATION.md** — fixed a typo in the 1.9.0 ``CausalResult.to_dict``
  byte-identity note: the no-kwargs default is identical to
  ``to_dict(detail="standard")``, not ``cite(detail="standard")``.

## [1.9.0] — Agent-native API surface: 12 modules across 4 phases

The 1.9.0 line ships StatsPAI's first deliberately agent-shaped API
surface — 12 new top-level entry points designed for Claude Code /
Cursor / Copilot CLI workflows where the LLM, not a human, is doing
the calling. **No estimator numerical paths changed**; all
additions are new functions or strictly additive parameters with
``"agent"`` as the default so existing behaviour is byte-identical.

### Added — Agent serialization & error envelope (Phase 1)

- **``CausalResult.to_dict(detail=...)``** and
  **``EconometricResults.to_dict(detail=...)``** — unified payload
  control with three documented levels:

  - ``"minimal"`` (~150 tokens) — bare answer; no diagnostics.
  - ``"standard"`` (~250 tokens) — current default; coefficients +
    scalar diagnostics + ``detail_head`` rows. Byte-identical to
    legacy ``to_dict()``.
  - ``"agent"`` (~620 tokens) — adds ``violations`` / ``warnings``
    / ``next_steps`` / ``suggested_functions`` so an LLM can plan
    its next call without another round-trip.

  ``for_agent()`` is now a thin alias for ``to_dict(detail="agent")``;
  ``to_agent_summary()`` is unchanged but its docstring now points
  at ``to_dict(detail="agent")`` as the canonical flat form.

- **``execute_tool`` MCP error envelope** — when an estimator raises
  a structured ``StatsPAIError`` subclass, the MCP ``tools/call``
  response now surfaces ``error_kind`` (e.g.
  ``"method_incompatibility"``) plus the full ``error_payload``
  dict (``code`` / ``recovery_hint`` / ``diagnostics`` /
  ``alternative_functions``). Legacy ``error`` / ``remediation``
  fields preserved.

### Added — MCP server polish (Phase 1)

- **``statspai-mcp`` console script** wired in ``pyproject.toml`` so
  ``pip install statspai`` exposes it on PATH.
- **``statspai://function/{name}``** per-function resources surfacing
  the registry's full agent-card (description, signature,
  assumptions, failure_modes, alternatives, typical_n_min, example).
  Listed via the new ``resources/templates/list`` handler.
- **``statspai://functions``** machine-readable JSON index for
  one-shot tool discovery.
- **Typed JSON-RPC errors** mapped to canonical MCP codes:
  ``-32002`` (resource not found), ``-32602`` (invalid params),
  ``-32000`` (server fallback). Replaces the previous blanket
  ``-32000``.
- **``notifications/*`` silenced** — Claude Desktop / Cursor send
  ``notifications/initialized`` after the handshake; the server now
  drops any method whose name starts with ``notifications/`` per
  the MCP spec, instead of replying with ``-32601`` noise on every
  session.
- **MCP-level ``detail`` parameter** on ``tools/call`` — agents pick
  ``detail="minimal" | "standard" | "agent"`` per call to control
  token cost. Validation rejects invalid values with ``-32602``.

### Added — Workflow primitives (Phases 2-4)

- **``sp.audit(result)``** — *missing-evidence* checklist (the
  read-only counterpart to ``sp.assumption_audit``): inspects what
  robustness / sensitivity diagnostics are stored on a fitted
  result and surfaces which method-family checks are still
  missing. Returns ``{checks: [{name, question, status, severity,
  importance, suggest_function, ...}], summary, coverage}`` with
  18 curated checks across DID/RD/IV/synth/matching/OLS.

- **``sp.detect_design(data, **hints)``** — heuristic design
  identifier: returns ``{design, confidence, identified, candidates,
  n_obs, columns}`` with ``design ∈ {"panel", "rd",
  "cross_section"}``. Symmetric ``(unit, time)`` pair dedup; RD
  confidence capped at 0.30 without explicit hint to avoid
  noise-data false positives.

- **``sp.preflight(data, method, **kwargs)``** — method-specific
  pre-estimation diagnostics distinct from
  ``sp.check_identification`` (design-level) and
  ``sp.assumption_audit`` (re-runs tests). Cheap shape / column /
  treatment-binarity / sample-size checks per method family;
  returns ``{verdict: "PASS" | "WARN" | "FAIL", checks, summary,
  known_method}``.

- **``CausalResult.cite(format=...)``** and
  **``sp.bib_for(result)``** — multi-format citations:
  ``"bibtex"`` (default, byte-identical to legacy ``cite()``),
  ``"apa"`` (parsed prose), ``"json"`` (structured ``{type, key,
  authors, year, title, journal, volume, number, pages,
  publisher, fields}``). LaTeX-diacritic normalisation
  (``{\\"o}`` → ``ö``); multi-entry BibTeX strings (e.g.
  ``twfe_decomposition`` cites both Goodman-Bacon 2021 AND de
  Chaisemartin & D'Haultfœuille 2020) round-trip both authors —
  zero hallucination per CLAUDE.md §10.

- **``sp.examples(name)``** — runnable code snippets for any
  registered function; 10 hand-curated flagship snippets, falls
  back to ``registry.example`` for the rest.

- **``sp.session(seed=42)``** — deterministic-RNG context manager
  snapshotting Python ``random`` and NumPy's legacy global MT19937
  generator; restores prior state on exit even when an exception
  is raised inside the block. Lazy torch / jax interop — never
  auto-imports. Documented escape hatch for
  ``np.random.default_rng()`` (which is *not* covered — pass
  ``state.seed`` explicitly).

- **``result.brief()`` / ``sp.brief(result)``** — one-line
  dashboard string (~95 chars typical, ≤ 140 hard cap) for
  multi-result agent loops.

- **MCP ``prompts/list`` + ``prompts/get``** — three curated
  workflow prompt templates (``audit_did_result`` /
  ``design_then_estimate`` / ``robustness_followup``) surfaced as
  prompt buttons in MCP-compliant clients.

### Changed

- ``CausalResult.to_dict`` / ``EconometricResults.to_dict`` now
  accept a keyword-only ``detail`` parameter. Default ``"standard"``
  preserves the legacy shape exactly. ``CausalResult``'s
  ``detail_head`` is also keyword-only now (was positional-or-
  keyword) to close the ``to_dict("agent")`` foot-gun.

- ``CausalResult.cite()`` now accepts ``format=`` keyword; zero-arg
  call still returns BibTeX, byte-identical to
  ``cite(format="bibtex")``.

### Tests

**+422 targeted tests** across the agent stack, all passing.
Token-budget assertions pin the size of every ``detail`` level so
future changes can't accidentally bloat the LLM tool-result channel.

### No numerical changes

Existing estimator coefficient / SE / CI / p-value paths are byte-
identical to 1.8.0. The 12 new modules are introspection,
serialization, prompt-rendering, and RNG-management primitives —
they read from existing result state, never recompute it.

## [1.8.0] — 2026-04-28

Internal-development version covering five `sp.regtable` rounds, the
Native Rust IRLS for `sp.fast.fepois`, twelve provenance-rollout
phases, the production-function module, the clubSandwich-equivalent
HTZ Wald, the LLM-DAG closed loop, the synth refactor, the
estimand-first paper appendix, the great_tables / CSL pipeline, and
the export trinity (numerical lineage / replication pack / Quarto).
Subsections below preserve the chronological development order.

### `sp.regtable` Round 4 (event_study_table, vcov= recompute, transpose)

Three further additions on top of Rounds 1-3. **No numerical
changes** to any estimator; the ``vcov=`` recompute reuses the
fit-time X + residuals already stored on OLS results.

#### Added

- **``sp.event_study_table(result, *, regex=None, label_fmt="t={t}",
  include_reference=False)``** — adapter that turns an event-study
  fit into a regtable input. Two extraction paths:

  - **CausalResult fast path** when ``model_info['event_study']``
    holds the canonical ``relative_time`` / ``estimate`` / ``se`` /
    ``ci_lower`` / ``ci_upper`` / ``pvalue`` DataFrame produced by
    :func:`sp.event_study`.
  - **Regex path** when raw coefficient names like ``"tau_-3"``,
    ``"lag_-2"``, ``"::-1"`` need to be parsed; the first capture
    group becomes the relative time. Rows are sorted in event-time
    order regardless of input ordering.

- **``vcov=``** parameter on :func:`sp.regtable` — recompute SE / t /
  p / 95% CI at print time without re-fitting. Currently supports
  OLS-style results that store ``data_info['X']`` and
  ``data_info['residuals']``:

  - ``"HC0"`` — White heteroskedasticity-robust
  - ``"HC1"`` / ``"robust"`` — Stata's ``robust`` (HC0 × n/(n-k))
  - ``"HC2"`` — leverage-weighted
  - ``"HC3"`` — leverage-squared (recommended for small samples;
    Long-Ervin 2000)

  Columns whose underlying result lacks the X/residuals fields emit
  a ``UserWarning`` and retain their fit-time SEs, so a
  heterogeneous mix of OLS + non-OLS does not blow up.

- **``transpose=True``** on :func:`sp.regtable` — rows become models,
  columns become variables. Single-panel only; multi-panel input or
  ``multi_se=`` is rejected with ``NotImplementedError`` to keep the
  layout pivot semantics tight. Renders in text and HTML.

#### Tests

15 new tests in ``test_regtable_round4_extensions.py`` covering all
three features, including HC0/HC1/HC2/HC3 ordering verification
under heteroskedasticity, regex extraction fallback, and pivot
guards on multi-panel / multi_se.

577 targeted tests pass (Rounds 1-4 = 528 + 20 + 14 + 15, plus broad
anchors). Zero regression.

### 2026-04-28 — Native Rust IRLS for `sp.fast.fepois` + production-function module

The headline of v1.8.0 is the **3× wall-clock improvement** on the
medium HDFE benchmark: `sp.fast.fepois` runs at **0.855 s** vs the
v1.7.x baseline's 2.61 s, and **1.34× of R `fixest::fepois`** (0.64 s)
on the project's standard medium dataset (n=1M, fe1=100k, fe2=1k).
This closes the long-standing wall-clock gap to `fixest` to 1.34× —
well under the ≤ 1.5× target set in the v1.8 design spec.

Plus a new structural-estimation module: `sp.prod_fn` ships four
production-function estimators (Olley-Pakes, Levinsohn-Petrin,
Ackerberg-Caves-Frazer, Wooldridge) + De Loecker-Warzynski markup.

#### Performance — sp.fast.fepois on medium HDFE benchmark

| stage                                          | wall    | vs fixest | shipped |
| ---------------------------------------------- | ------: | --------: | :-----: |
| v1.7.x baseline (Python np.bincount inside Python IRLS) | 2.61 s | 4.08× | — |
| Phase A (Rust scatter, no cache)               | 2.45 s  |     3.83× |    ✓    |
| Phase B0 (Rust sequential + dispatcher cache)  | 1.441 s |     2.25× |    ✓    |
| Phase B1 (native Rust IRLS, single PyO3 call)  | 0.880 s |     1.37× |    ✓    |
| **Path A (B1 + Rust separation pre-pass)**     | **0.855 s** | **1.34×** | **✓** |
| R fixest::fepois                               | 0.64 s  |     1.00× |    —    |

The closure was driven by three orthogonal contributions, each
verified with a wall-clock spike before the next was committed
(audited at `benchmarks/hdfe/AUDIT.md`):

- **Phase A primitives**: `statspai_hdfe.demean_2d_weighted` PyO3
  binding, Python `_weighted_ap_demean` dispatcher with NumPy fallback,
  `weighted_demean_matrix_fortran_inplace` crate-internal Rust API.
- **Phase B0 algorithmic primitive**: sort-by-primary-FE permutation
  (`sort_perm::primary_fe_sort_perm`) + sequential weighted sweep
  (`weighted_group_sweep_sorted`) replaces the L2-cache-miss-bound
  random-scatter inner loop on G1 = 100k bucket arrays. Plus the
  module-level FE-only-plan fingerprint cache in the dispatcher
  (avoids ~1.4 s per fepois of recomputing `np.argsort` /
  `searchsorted` / secondary perms across IRLS iters).
- **Phase B1 native Rust IRLS**: `irls.rs` hosts `fepois_loop`, the
  full IRLS state machine (working response, working weight, sort-aware
  weighted demean, hand-coded SPD Cholesky for the WLS solve, eta clip,
  step-halving, deviance + convergence). `FePoisIRLSWorkspace` holds
  scratch + Aitken history + sorted indices, allocated once per fepois
  call and reused across all IRLS iters. Single PyO3 call
  (`fepois_irls`) eliminates the 12 round-trips per fepois that
  Phase B0 still had.
- **Path A — Rust separation pre-pass**: the iterative Poisson-
  separation drop (drops rows in FE groups whose total y-sum is zero —
  Poisson cannot identify them) was the last meaningful Python-side
  O(n log n) overhead inside fepois (np.unique + np.isin per pass).
  The Rust port (`separation::separation_mask`) replaces it with an
  O(n × n_iter × K) bincount loop. ~25 ms additional wall reduction
  on the medium benchmark; closes 1.37× → 1.34× of fixest. Reusable
  by future `feglm` GLM families.

#### Numerical correctness — preserved at v1.7.x parity

- `sp.fast.fepois` vs `pyfixest.fepois` coef on the medium dataset:
  unchanged (atol < 1e-13 across IRLS-converged fits).
- Cluster-robust SE (`vcov="cr1"`): the v1.7.x integration is
  **untouched**; commit `39c94d0` (CR1 recovery from auto-checkpoint)
  remains the canonical implementation.
- The Python NumPy fallback path (when the compiled `statspai_hdfe`
  wheel is absent) is bit-for-bit identical to the v1.7.x behavior
  — verified by `test_fepois_falls_back_when_rust_unavailable`.

#### Added

- New `statspai_hdfe` v0.6.0 PyO3 entry points (Rust crate v0.5.0-alpha.1):
  - `demean_2d_weighted` — Phase A weighted variant of the K-way AP demean.
  - `demean_2d_weighted_sorted` — Phase B0 sort-aware variant.
  - `fepois_irls` — Phase B1 single-call IRLS state machine.
  - `separation_mask` — Path A iterative Poisson separation detector.

- **`sp.prod_fn`** unified production-function estimator dispatcher with
  four named entry points (`olley_pakes` / `opreg`, `levinsohn_petrin` /
  `levpet`, `ackerberg_caves_frazer` / `acf`, `wooldridge_prod`) plus
  `markup` (De Loecker-Warzynski) and `ProductionResult`. Cobb-Douglas
  default + translog functional form; firm-cluster bootstrap SE; full
  registry coverage. References: Olley-Pakes (1996), Levinsohn-Petrin
  (2003), Ackerberg-Caves-Frazer (2015), Wooldridge (2009),
  De Loecker-Warzynski (2012). 23 dedicated tests.
- `sp.fast.fepois` Python dispatcher with three-tier fallback (native
  Rust IRLS → sort-aware Rust demean → random-scatter Rust demean →
  pure NumPy) — no user-facing API change.
- `benchmarks/hdfe/run_fepois_phase_a.py`, `run_fepois_phase_b0.py`,
  `run_fepois_phase_b.py` — reproducible wall-clock harnesses with
  hard merge gates.
- `benchmarks/hdfe/AUDIT.md` — Phase A round 1 (gate failure +
  root-cause), Phase B0 round 1 PASS, Phase B1 round 1 PASS audit
  trails. The audit pattern (measure-before-commit) is the structural
  counter-measure that prevented Phase A's "assumption broke" failure
  from repeating in B0 / B1.

#### Internal

- Rust crate `statspai_hdfe` bumped 0.2.0-alpha.1 → 0.5.0-alpha.1 across
  Phase A → Phase B → Path A (4 minor crate version bumps).
- Python `__version__` in `statspai_hdfe` extension: `0.2.0` → `0.6.0`.

#### Tests — 192 fast-fepois tests pass (was 187 in v1.7.x) + 23 prod_fn tests

- Phase B1 native-vs-Python IRLS parity: coef atol ≤ 1e-10, SE atol ≤ 1e-7
  (`test_fepois_native_irls_vs_python_irls_parity`).
- Path A separation parity: 10 random seeds with synthetic zero-cluster
  injection; Rust ↔ NumPy mask agreement element-wise
  (`test_separation_rust_matches_python_fallback`).
- Cluster-SE suite intact (5 tests covering validation / NaN rejection /
  IID-baseline / closed-form / fixest-parity).
- New 23-test prod_fn suite covering OP / LP / ACF / Wooldridge / markup
  on synthetic Cobb-Douglas / translog DGPs + edge cases + bootstrap-SE
  reproducibility.

---

### `sp.regtable` Round 3 (margins_table, tests= footer, fixef_sizes)

Three further additions on top of Round 1 + Round 2. **No numerical
changes** to any estimator (margins_table is a pure adapter; tests=
formats user-supplied test results; fixef_sizes reads pre-existing
``model_info['n_fe_levels']``).

#### Added

- **``sp.margins_table(model)``** — adapter that wraps a
  :func:`sp.margins` DataFrame as a duck-typed result with
  ``.params`` / ``.std_errors`` / ``.tvalues`` / ``.pvalues`` /
  ``.conf_int_*``. Pipes straight into :func:`sp.regtable`, closing
  the "estimator → marginal-effects table" gap that previously
  required users to hand-build ``add_rows``. Mirrors the R workflow
  ``modelsummary(avg_slopes(model))``. The wrapper z-stat is mapped
  to ``tvalues`` so existing ``se_type='t'`` / ``'p'`` / ``'ci'``
  paths render unchanged.

- **``tests=``** parameter on :func:`sp.regtable` — render
  hypothesis-test rows in the diagnostic strip below the stats
  block. ``tests={"Wald F": [(12.34, 0.001), (8.91, 0.003)]}``
  → "Wald F  12.340***  8.910***". Each per-model entry can be a
  ``(stat, p)`` tuple, a bare p-value, ``None``, or a pre-formatted
  string. Stars honour the configured ``notation`` family for
  cross-table consistency. Closes the gap to Stata's ``estadd
  scalar`` / ``test`` integration where reviewers expect Wald /
  Sargan / Hansen-J / first-stage F right under the main results.

- **``fixef_sizes=True``** on :func:`sp.regtable` — auto-emit
  "# Firm: 1,234" / "# Year: 30" rows showing distinct levels per
  fixed effect. Reads ``model_info['n_fe_levels']`` from each
  result; currently populated by ``count.py`` (Poisson/NegBin) and
  the pyfixest adapter. Other estimators silently no-op. Mirrors
  R fixest's ``etable(..., fixef_sizes=TRUE)``.

#### Tests

14 new tests in ``test_regtable_round3_extensions.py`` covering
all three features across text / LaTeX / HTML renderers.

562 targeted tests pass (Rounds 1-3 = 528 + 20 + 14, plus broad
anchors); zero regression on the 33 output / regression test files
exercised.

### `sp.regtable` Round 2 (templates, notation, apply_coef, escape, Word/Excel spanners)

Five further additions on top of the Round 1 commit. **No numerical
changes** to any estimator; output-layer only.

#### Added — Five regtable parameters

- **``estimate=`` / ``statistic=``** — flexible cell templates that
  mirror R ``modelsummary``'s arguments. Placeholders: ``{estimate}``,
  ``{stars}``, ``{std_error}``, ``{t_value}``, ``{p_value}``,
  ``{conf_low}``, ``{conf_high}``. Examples:
  - ``estimate="{stars}{estimate}"`` for stars-first.
  - ``statistic="t={t_value}, p={p_value}"`` for working-paper cells.
  - ``statistic="[{conf_low}, {conf_high}]"`` for inline CI without
    needing ``se_type="ci"`` separately. Unknown placeholders raise a
    ``KeyError`` at the ``regtable()`` call site.

- **``notation=``** — alternative significance-marker family.
  ``"stars"`` (default) keeps ``("*", "**", "***")``;
  ``"symbols"`` swaps to ``("†", "‡", "§")`` for AER / JPE contexts
  where star-shaped markers conflict with footnote symbols; pass a
  custom 3-tuple for any ladder. The footer "p<0.01, ..." line
  rebuilds itself to match.

- **``apply_coef=`` / ``apply_coef_deriv=``** — generalise ``eform``
  to any callable. ``apply_coef=lambda b: 100*b`` for a percentage
  transform; ``apply_coef=np.log`` for log-scale; ``apply_coef_deriv``
  enables delta-method SE rescaling (``|f'(b)|·SE``). Mutually
  exclusive with ``eform`` — both transform the point estimate, and
  silently combining them would hide whichever the user listed second.

- **``escape=False``** — opt out of auto-escape so users can pass raw
  LaTeX (e.g. ``"$\\beta_1$"``) or HTML (``"<i>β</i>"``) as labels.
  Mirrors R ``kableExtra::escape`` and ``xtable::print``. Cell content
  (numeric estimates, computed stats) is always safe — it never
  contains user-controlled metacharacters.

- **Word + Excel ``column_spanners`` rendering** — closes the format
  parity gap left in Round 1. Word inserts an extra header row with
  merged cells across each column block; Excel uses
  ``ws.merge_cells`` and the spanner row sits above the model-label
  row inside the booktab top-rule region.

#### Tests

20 new tests in ``test_regtable_round2_extensions.py`` covering all
five features across text / LaTeX / HTML / Word / Excel renderers.

548 targeted tests pass (Round 1's 528 + 20 new), zero regression.

### `sp.regtable` publication-quality extensions

Five additions designed to close the remaining gap between
``sp.regtable`` and Stata ``esttab`` / R ``modelsummary`` /
R ``fixest::etable`` for empirical paper writing. **No numerical
changes** to any estimator; output-layer only.

#### Added — Five regtable parameters

- **``eform``** — report ``exp(b)`` (odds ratios for ``logit`` /
  ``probit``, incidence-rate ratios for ``poisson``, hazard ratios
  for Cox-style models). SE via delta method (``exp(b)·SE(b)``);
  CI bounds via ``(exp(lo), exp(hi))`` of the original endpoints;
  t and p unchanged because ``H_0: b=0`` is equivalent to
  ``H_0: exp(b)=1``. Accepts ``bool`` (apply to all) or
  ``List[bool]`` (per-model — mix logit OR with OLS coefs in the
  same table). A footer note transparently flags which columns are
  exponentiated. Mirrors Stata ``esttab, eform``.

- **``column_spanners``** — multi-row header above the model labels.
  Pass a list of ``(label, span)`` tuples whose spans partition all
  model columns, e.g. ``[("OLS", 2), ("IV", 2)]``. Renders as
  ``\multicolumn{n}{c}{label}`` + ``\cmidrule`` in LaTeX, ``colspan``
  in HTML, repeated bold cells in Markdown, and centered ASCII in
  text. Mirrors Stata ``mgroups()`` and R ``modelsummary``'s ``group``.

- **``coef_map``** — single-shot rename + reorder + drop. Pass an
  ordered dict whose keys are coefficients to keep (in display
  order) and values are display labels. Variables not in the map
  are dropped. Mutually exclusive with the legacy ``coef_labels``
  / ``keep`` / ``drop`` / ``order`` quartet. Mirrors R
  ``modelsummary``'s ``coef_map``.

- **``stats=["depvar_mean", "depvar_sd"]``** — auto rows for the
  dependent variable's sample mean and standard deviation, populated
  from the result object's ``data_info['y']`` (or ``endog`` /
  ``dep_var``) at extraction time. Rows render as "Mean of Y" and
  "SD of Y". Top-5 economics journals routinely require these so
  reviewers can sanity-check effect magnitudes against the
  outcome's scale. Aliases: ``"ymean"`` / ``"ysd"``.

- **``consistency_check``** (default ``True``) — emit a
  ``UserWarning`` when sample sizes differ across columns. Disable
  via ``consistency_check=False`` when the N-mismatch is
  intentional (IV first stage on a subsample, RD bandwidth
  restriction). Reviewer red flag silenced by default in v1.7.2,
  surfaced now.

#### Tests

23 new tests in ``test_regtable_publication_extensions.py`` covering
all six format renderers (text / LaTeX / HTML / Markdown) plus the
parameter validation paths. Existing 204 output-area tests
unchanged.

### Phase 12: provenance rollout to 66/925 (bounds + randomization + imputation)

Continues the v1.7.2 provenance rollout. **No numerical changes** to
any estimator. 5 estimators instrumented spanning bounds /
randomization inference / imputation. Coverage 61/925 → **66/925**.

#### Added — Provenance for 5 estimators

- ``sp.balke_pearl`` — Balke-Pearl bounds on ATE under monotonicity.
- ``sp.lee_bounds`` — Lee (2009) trimming bounds for selection.
- ``sp.manski_bounds`` — Manski (1990) worst-case ATE bounds.
- ``sp.fisher_exact`` — Fisher randomization test (permutation).
- ``sp.imputation.mice`` — Multiple Imputation by Chained Equations.

#### Tests

6 new (5 per-estimator + 1 multi-estimator integration). All pass.

### Phase 11: provenance rollout to 61/925 (spatial + qte + bootstrap + conformal)

Continues the v1.7.2 provenance rollout. **No numerical changes** to
any estimator. 7 estimators instrumented spanning spatial / quantile
/ distributional / bootstrap / conformal. Coverage 54/925 → **61/925**.

#### Added — Provenance for 7 estimators

- ``sp.spatial.spatial_did`` — spatial-lag DiD with spillover decomposition.
- ``sp.spatial.spatial_iv`` — spatial 2SLS.
- ``sp.qte.dist_iv`` — distributional IV / quantile LATE.
- ``sp.qte.beyond_average_late`` — quantile LATE under fuzzy compliance.
- ``sp.qte.qte_hd_panel`` — high-dim panel QTE via LASSO controls.
- ``sp.bootstrap`` — general-purpose bootstrap inference.
- ``sp.conformal_cate`` — conformal prediction intervals for CATE.

#### Tests

8 new (7 per-estimator + 1 multi-estimator integration). All pass.

### Phase 10: provenance rollout to 54/925 (panel + decomp + mediation)

Continues the v1.7.2 provenance rollout. **No numerical changes** to
any estimator. 6 estimators instrumented; ``sp.panel`` refactored
into outer wrapper + dispatcher (parallel to Phase 4 ``sp.synth`` and
Phase 7 ``sp.etwfe``). Coverage 48/925 → **54/925**.

#### Added — Provenance for 6 estimators

- ``sp.panel`` — multi-method panel dispatcher (FE / RE / BE / FD /
  pooled / twoway / CRE / GMM). Refactored: outer ``panel`` wrapper
  captures kwargs + calls ``_dispatch_panel_impl`` + attaches
  provenance once. Public signature unchanged.
- ``sp.causal_impact`` — Brodersen-Gallusser-Koehler-Remy-Scott
  (2015) BSTS-style impact.
- ``sp.mediate`` — Imai-Keele-Tingley (2010) mediation.
- ``sp.mediate_interventional`` — VanderWeele-Vansteelandt-Robins
  (2014) interventional (in)direct effects.
- ``sp.bartik`` — Goldsmith-Pinkham-Sorkin-Swift (2020) shift-share IV.
- ``sp.decompose`` — Oaxaca / FFL / DFL / RIF / gap-closing
  dispatcher; ``Provenance.function`` surfaces the dispatched method
  (e.g. ``"sp.decompose.oaxaca"``).

#### Skipped — `sp.did` top-level dispatcher

The ``sp.did`` dispatcher delegates to already-instrumented inner
estimators (``sp.did.callaway_santanna`` / ``sp.did.did_2x2`` /
``sp.did.aggte`` / ``sp.sun_abraham`` / ``sp.synth(method='sdid')``).
With the established ``overwrite=False`` semantics, the inner
record's name (more specific) wins. Wrapping the dispatcher would
add no information.

#### Tests

8 new (6 per-estimator + 1 panel method-choice variant + 1
multi-estimator integration). 111 green across the panel /
causal_impact / mediation / decomposition / bartik regression sweep.

### Phase 9: provenance rollout to 48/925 (TMLE + forest + DR)

Continues the v1.7.2 provenance rollout. **No numerical changes** to
any estimator. 12 ML-causal + classical-identification estimators
instrumented. Coverage 36/925 → **48/925**.

#### Added — Provenance for 12 estimators

ML-causal (8):

- ``sp.tmle`` — van der Laan & Rose Targeted MLE (with Super Learner).
- ``sp.tmle.ltmle`` — Longitudinal TMLE for static regime contrasts.
- ``sp.tmle.hal_tmle`` — TMLE with Highly Adaptive Lasso nuisance.
- ``sp.causal_forest`` — GRF causal forest factory.
- ``sp.multi_arm_forest`` — multi-arm causal forest.
- ``sp.iv_forest`` — Athey-Tibshirani-Wager IV causal forest.
- ``sp.metalearner`` — S/T/X/R/DR meta-learner dispatcher.
- ``sp.bcf`` — Hahn-Murray-Carvalho Bayesian Causal Forest.

Classical identification (4):

- ``sp.aipw`` — Augmented IPW (doubly robust, cross-fit).
- ``sp.ipw`` — Inverse Probability Weighting.
- ``sp.g_computation`` — parametric g-formula.
- ``sp.front_door`` — Pearl front-door adjustment.

Pattern reuse: established Phase 3 idiom — assign to ``_result``,
``attach_provenance(overwrite=False)``, return. The ``hal_tmle`` →
``tmle`` cascade is handled correctly: inner ``sp.tmle`` record wins,
matching the ``etwfe`` → ``wooldridge_did`` and ``lasso_iv`` → ``iv``
patterns from earlier rounds.

#### Tests

14 new (12 per-estimator + 1 metalearner choice variant + 1
multi-estimator integration). 103 green across the
hal_tmle / causal_forest / metalearner / bcf / front_door /
g_computation regression sweep.

### production function estimators (OP / LP / ACF / Wooldridge + translog + DLW markup)

Adds proxy-variable production function estimation — Olley-Pakes,
Levinsohn-Petrin, Ackerberg-Caves-Frazer, Wooldridge — plus
Cobb-Douglas + translog functional forms and the De Loecker-Warzynski
markup. Closes the long-standing gap that forced StatsPAI users to
drop into R `prodest` or Stata `prodest` for productivity / TFP /
markup work.

#### Added

- `sp.prod_fn(method=..., functional_form=...)` — unified dispatcher
  (`'op' | 'lp' | 'acf' | 'wrdg'`, `'cobb-douglas' | 'translog'`).
- `sp.olley_pakes` (alias `sp.opreg`) — investment-proxy estimator
  with strictly-positive-investment filter.
- `sp.levinsohn_petrin` (alias `sp.levpet`) — intermediate-input proxy
  (avoids OP zero-investment selection).
- `sp.ackerberg_caves_frazer` (alias `sp.acf`) — modern default,
  corrects the OP/LP labor-coefficient identification problem via
  lagged-labor instruments.
- `sp.wooldridge_prod` — joint stacked-NLS estimator (Cobb-Douglas
  only; translog raises NotImplementedError; full-GMM Wooldridge on
  roadmap).
- `sp.markup` — De Loecker & Warzynski (2012) firm-time markup
  μ_it = θ_v · (PQ) / (P_v V) with optional η-correction. Supports
  both Cobb-Douglas (constant θ_v) and translog (firm-time θ_v_it
  read from the elasticity panel attached to the result).
- `sp.ProductionResult` — unified result class with `coef`, `tfp`,
  `productivity_process`, `cite()`, `summary()`, plus
  `model_info["elasticities"]` for translog firm-time elasticities.
- Translog functional form: input matrix expanded to linear +
  0.5*x_j² + cross-term basis; instrument matrix expanded by the
  same polynomial; firm-time output elasticities computed from
  ∂y/∂x_j = β_j + β_jj·x_j + Σ_{k≠j} β_jk·x_k.
- Firm-cluster bootstrap SE (Wooldridge 2009 §4 convention) with
  convergence filtering on each replicate.
- Multi-start Nelder-Mead in stage 2 over 5 economic-prior starts
  (the OLS warm start is intentionally avoided — it lands in a
  spurious basin where the productivity AR overfits ω onto ω_lag at
  implausible β).
- UserWarning on non-consecutive panel time periods (lag operator
  would silently treat gaps as 1-period lags otherwise).
- 9 new registry entries (5 canonical + 3 aliases + markup) — total
  rises to 964 functions.

#### References (verified via Crossref API on 2026-04-27)

- Olley & Pakes (1996) Econometrica 64(6) 1263–1297, DOI 10.2307/2171831
- Levinsohn & Petrin (2003) RES 70(2) 317–341, DOI 10.1111/1467-937X.00246
- Ackerberg, Caves & Frazer (2015) Econometrica 83(6) 2411–2451, DOI 10.3982/ECTA13408
- Wooldridge (2009) Economics Letters 104(3) 112–114, DOI 10.1016/j.econlet.2009.04.026
- De Loecker & Warzynski (2012) AER 102(6) 2437–2471, DOI 10.1257/aer.102.6.2437

#### Tests

- `tests/test_prod_fn.py` — 23 tests:
  - Synthetic DGP recovery (ACF tight; OP/LP loose per ACF's
    identification critique; Wooldridge feasible-range)
  - Translog: 5-coef structure, dispatcher pass-through, CD-truth
    nesting (β_ll/β_kk/β_lk near 0), markup with firm-time θ_v_it,
    Wooldridge-translog raises, unknown functional_form raises
  - Dispatcher, aliases, bootstrap SE, markup CD path, edge cases
    (missing columns, too-few-obs, zero-proxy filter, time-gap
    warning, registry presence, no-bootstrap diagnostics shape).

#### Notes

- Default `productivity_degree=1` (linear AR(1)). Higher degrees can
  overfit ω given ω_lag in finite samples and flatten the GMM
  objective surface — see dispatcher docstring.
- Translog identification caveat: stage-2 instruments are polynomial
  transforms of the same raw (k, l_lag) pair, so the moment system
  can be near-singular when state and lagged-free inputs are highly
  correlated. Higher-order coefficients have larger finite-sample
  variance than linear ones — bootstrap SEs recommended.
- Gandhi-Navarro-Rivers (2020) flexible-input identification and
  full efficient-GMM Wooldridge are roadmap items, not in this
  release.

### Phase 8: provenance rollout to 36/925 (IV + matching + DML)

Continues the v1.7.2 provenance rollout from Phases 3-4-7. **No
numerical changes** to any estimator. 12 instrumentation points added
(15 user-facing functions, since the JIVE family of 4 share a single
``_run`` instrumentation). Coverage 21/925 → **36/925**.

#### Added — Provenance instrumentation for 12 more points

IV family (9 user-facing names):

- ``sp.liml`` — Limited Information Maximum Likelihood / Fuller.
- ``sp.jive`` — legacy single-method JIVE (regression/advanced_iv).
- ``sp.lasso_iv`` — Belloni-Chen-Chernozhukov-Hansen (2012). The
  pre-existing ``iv()`` API drift bug here was also repaired —
  ``lasso_iv`` now builds a formula string for the formula-only
  ``sp.iv()`` API and maps the legacy ``robust='robust'`` kwarg to
  the modern ``hc1`` enum.
- ``sp.iv.bayesian_iv`` — Chernozhukov-Hong (2003) Anderson-Rubin
  posterior with Metropolis-Hastings.
- ``sp.iv.jive1`` / ``sp.iv.ujive`` / ``sp.iv.ijive`` /
  ``sp.iv.rjive`` — all four flow through the shared ``_run``
  dispatcher; ``method`` arg discriminates and surfaces in
  ``Provenance.function`` (``"sp.iv.jive1"`` / ``"sp.iv.ujive"`` /
  …). One instrumentation point covers four user-facing names.
- ``sp.iv.mte`` — Brinch-Mogstad-Wiswall (2017) polynomial Marginal
  Treatment Effect.

Matching family (5):

- ``sp.match`` — main matching dispatcher (PSM / mahalanobis / CEM
  / strata / coarsened).
- ``sp.optimal_match`` — Hungarian-algorithm 1:1 with caliper.
- ``sp.cardinality_match`` — Zubizarreta (2014) LP with SMD
  tolerance.
- ``sp.genmatch`` — Diamond-Sekhon (2013) genetic matching.
- ``sp.sbw`` — Zubizarreta (2015) Stable Balancing Weights.

DML (1):

- ``sp.dml`` — Chernozhukov et al. (2018) Double ML dispatcher
  covering plr / irm / pliv / iivm. Single-exit pattern.

Pattern reuse: each follows the established Phase 3 idiom — assign
result to ``_result``, call ``attach_provenance(overwrite=False)``,
return. ``overwrite=False`` semantics preserve the inner-most record
when an outer wrapper (e.g. ``lasso_iv`` calling ``sp.iv``) is also
instrumented.

#### Fixed — `sp.lasso_iv` API drift (pre-existing)

Independent fix: ``sp.lasso_iv`` was calling the legacy ``iv(y=,
x_endog=, x_exog=, z=)`` signature which is no longer accepted. Now
builds a Patsy-style formula (``y ~ (endog ~ z) + exog``) for the
current formula-only ``sp.iv()`` API.

#### Tests

16 new tests (12 per-estimator + 4 JIVE variants confirming each
gets the right ``method``-discriminated function name + 1
multi-estimator integration). 155 green across the IV + matching +
DML + provenance regression sweep:

- IV: ``test_iv.py`` and ``test_iv_frontiers.py``.
- Matching: ``test_matching.py`` and ``test_matching_optimal.py``.
- DML: ``test_dml.py``, ``test_dml_iivm.py``, ``test_dml_panel.py``,
  ``test_dml_split.py``.
- Provenance: rounds 1+2+3+4.

#### Documentation

``docs/guides/replication_workflow.md`` scorecard updated to 36/925.

### production function estimators

Adds proxy-variable production function estimation — Olley-Pakes,
Levinsohn-Petrin, Ackerberg-Caves-Frazer, Wooldridge — plus the
De Loecker-Warzynski markup. Closes the long-standing gap that
forced StatsPAI users to drop into R `prodest` or Stata `prodest`
for productivity / TFP / markup work.

#### Added

- `sp.prod_fn(method=...)` — unified Cobb-Douglas dispatcher
  (`'op' | 'lp' | 'acf' | 'wrdg'`).
- `sp.olley_pakes` (alias `sp.opreg`) — investment-proxy estimator
  with strictly-positive-investment filter.
- `sp.levinsohn_petrin` (alias `sp.levpet`) — intermediate-input proxy
  (avoids OP zero-investment selection).
- `sp.ackerberg_caves_frazer` (alias `sp.acf`) — modern default,
  corrects the OP/LP labor-coefficient identification problem via
  lagged-labor instruments.
- `sp.wooldridge_prod` — joint stacked-NLS estimator (one-step GMM
  with identity weighting and instruments = regressors; full
  efficient-GMM variant on the roadmap).
- `sp.markup` — De Loecker & Warzynski (2012) firm-time markup
  μ_it = θ_v · (PQ) / (P_v V) with optional η-correction.
- `sp.ProductionResult` — unified result class with `coef`, `tfp`,
  `productivity_process`, `cite()`, `summary()`.
- Firm-cluster bootstrap SE (Wooldridge 2009 §4 convention) with
  convergence filtering on each replicate.
- Multi-start Nelder-Mead in stage 2 to dodge the upward-biased
  OLS warm start (positive selection of labor on ω).
- UserWarning on non-consecutive panel time periods (lag operator
  would silently treat gaps as 1-period lags otherwise).
- 9 new registry entries (5 canonical + 3 aliases + markup), bringing
  total to 964 functions.

#### References (verified via Crossref API on 2026-04-27)

- Olley & Pakes (1996) Econometrica 64(6) 1263–1297, DOI 10.2307/2171831
- Levinsohn & Petrin (2003) RES 70(2) 317–341, DOI 10.1111/1467-937X.00246
- Ackerberg, Caves & Frazer (2015) Econometrica 83(6) 2411–2451, DOI 10.3982/ECTA13408
- Wooldridge (2009) Economics Letters 104(3) 112–114, DOI 10.1016/j.econlet.2009.04.026
- De Loecker & Warzynski (2012) AER 102(6) 2437–2471, DOI 10.1257/aer.102.6.2437

#### Tests

- `tests/test_prod_fn.py` — synthetic DGP recovery (ACF tight, OP/LP
  loose per ACF's identification critique), dispatcher, aliases,
  bootstrap SE, markup, edge cases (missing columns, too-few-obs,
  zero-proxy filter, time-gap warning, registry presence). 18 tests.

#### Notes

- Default `productivity_degree=1` (linear AR(1)). Higher degrees can
  overfit ω given ω_lag in finite samples and flatten the GMM
  objective surface — see dispatcher docstring.
- Translog and Gandhi-Navarro-Rivers (2020) production functions
  are roadmap items, not in this release.

### Phase 7: provenance rollout to 21/925 (DiD long-tail + RD)

Continues the v1.7.2 provenance rollout established in Phases 3-4.
**No numerical changes** to any estimator. 12 more estimators
instrumented; `sp.etwfe` refactored into wrapper + dispatcher
(parallel to the Phase 4 `sp.synth` move). Coverage now **21/925**.

#### Added — Provenance instrumentation for 12 more estimators

DiD long-tail (10):

- `sp.cic` — Athey-Imbens (2006) Changes-in-Changes.
- `sp.cohort_anchored_event_study` — staggered-robust ES
  (arXiv:2509.01829).
- `sp.design_robust_event_study` (Wright 2026, arXiv:2601.18801) —
  orthogonalised event-study under staggered adoption.
- `sp.gardner_did` / `sp.did_2stage` — Gardner (2021) two-stage.
- `sp.harvest_did` — Borusyak-Harmon-Hull-Jaravel-Spiess (2025)
  harvesting.
- `sp.did_misclassified` — staggered DiD with treatment
  misclassification + anticipation (arXiv:2507.20415).
- `sp.stacked_did` — Cengiz-Dube-Lindner-Zipperer (2019) stacked.
- `sp.wooldridge_did` — Wooldridge (2021) Extended TWFE.
- `sp.etwfe` — refactored into outer wrapper + 4-branch
  `_dispatch_etwfe_impl` so the (with-xvar / never-only / notyet /
  repeated-cross-section) routing attaches provenance once on the
  way out. Same pattern as Phase 4's `sp.synth` move.
- `sp.drdid` — Sant'Anna-Zhao (2020) doubly robust DiD.

RD (2):

- `sp.rd_honest` — Armstrong-Kolesar (2018, 2020) honest CIs.
- `sp.rkd` — Card-Lee-Pei-Weber (2015) Regression Kink Design.

Each follows the established Phase 3 idiom: assign result to
`_result`, call `attach_provenance(overwrite=False)`, return.
`overwrite=False` semantics preserve the inner-most record so
estimand-first / `sp.causal` / `sp.paper` wrappers don't clobber
the more-specific call name.

#### Changed — `sp.etwfe` refactored into outer wrapper + dispatcher

Mirrors Phase 4's `sp.synth` refactor. The previous `etwfe` had 4
return sites (one per `(panel × cgroup × xvar)` branch), which
made naive instrumentation maintenance-hostile. New layout:

- `_dispatch_etwfe_impl(...)` — full dispatcher (former `etwfe`
  body), unchanged logic.
- `etwfe(...)` — thin outer wrapper that captures kwargs, calls
  impl, attaches provenance once before returning.

Public signature is bit-identical; the existing wooldridge / etwfe
test sweep passes with zero changes.

#### Tests

14 new tests (12 per-estimator + 1 `did_2stage` alias check + 1
multi-estimator `replication_pack` integration). 346 green across
the DiD + RD + paper regression sweep (DiD: 214, paper+remaining:
132). Zero regressions across either family.

#### Documentation

`docs/guides/replication_workflow.md` scorecard updated to reflect
the new 21/925 coverage. Users running `get_provenance(result)`
can verify any estimator's status locally.

### production function estimators

Adds proxy-variable production function estimation — Olley-Pakes,
Levinsohn-Petrin, Ackerberg-Caves-Frazer, Wooldridge — plus the
De Loecker-Warzynski markup. Closes the long-standing gap that
forced StatsPAI users to drop into R `prodest` or Stata `prodest`
for productivity / TFP / markup work.

#### Added

- `sp.prod_fn(method=...)` — unified Cobb-Douglas dispatcher
  (`'op' | 'lp' | 'acf' | 'wrdg'`).
- `sp.olley_pakes` (alias `sp.opreg`) — investment-proxy estimator
  with strictly-positive-investment filter.
- `sp.levinsohn_petrin` (alias `sp.levpet`) — intermediate-input proxy
  (avoids OP zero-investment selection).
- `sp.ackerberg_caves_frazer` (alias `sp.acf`) — modern default,
  corrects the OP/LP labor-coefficient identification problem via
  lagged-labor instruments.
- `sp.wooldridge_prod` — joint stacked-NLS estimator (one-step GMM
  with identity weighting and instruments = regressors; full
  efficient-GMM variant on the roadmap).
- `sp.markup` — De Loecker & Warzynski (2012) firm-time markup
  μ_it = θ_v · (PQ) / (P_v V) with optional η-correction.
- `sp.ProductionResult` — unified result class with `coef`, `tfp`,
  `productivity_process`, `cite()`, `summary()`.
- Firm-cluster bootstrap SE (Wooldridge 2009 §4 convention) with
  convergence filtering on each replicate.
- Multi-start Nelder-Mead in stage 2 to dodge the upward-biased
  OLS warm start (positive selection of labor on ω).
- UserWarning on non-consecutive panel time periods (lag operator
  would silently treat gaps as 1-period lags otherwise).

#### References (verified via Crossref API on 2026-04-27)

- Olley & Pakes (1996) Econometrica 64(6) 1263–1297, DOI 10.2307/2171831
- Levinsohn & Petrin (2003) RES 70(2) 317–341, DOI 10.1111/1467-937X.00246
- Ackerberg, Caves & Frazer (2015) Econometrica 83(6) 2411–2451, DOI 10.3982/ECTA13408
- Wooldridge (2009) Economics Letters 104(3) 112–114, DOI 10.1016/j.econlet.2009.04.026
- De Loecker & Warzynski (2012) AER 102(6) 2437–2471, DOI 10.1257/aer.102.6.2437

#### Tests

- `tests/test_prod_fn.py` — synthetic DGP recovery (ACF tight, OP/LP
  loose per ACF's identification critique), dispatcher, aliases,
  bootstrap SE, markup, edge cases (missing columns, too-few-obs,
  zero-proxy filter, time-gap warning, registry presence). 18 tests.

#### Notes

- Default `productivity_degree=1` (linear AR(1)). Higher degrees can
  overfit ω given ω_lag in finite samples and flatten the GMM
  objective surface — see dispatcher docstring.
- Translog and Gandhi-Navarro-Rivers (2020) production functions
  are roadmap items, not in this release.

### clubSandwich-equivalent HTZ Wald (independent PR)

Adds a numerically-equivalent Python implementation of R
``clubSandwich::Wald_test(..., test="HTZ")`` for cluster-robust Wald
tests under CR2 sandwich. Closes the BM-vs-HTZ gap documented in
``cluster_dof_wald_bm`` (which uses the BM 2002 simplified formula
and can drift 50–100% from clubSandwich on multi-restriction tests).

#### Added

- ``sp.fast.cluster_wald_htz()`` — full HTZ Wald test, returns
  ``WaldTestResult`` (``test, q, eta, F_stat, p_value, Q, R, r, V_R``).
- ``sp.fast.cluster_dof_wald_htz()`` — DOF-only helper mirroring the
  ``cluster_dof_wald_bm`` signature for easy substitution.
- ``sp.fast.WaldTestResult`` — frozen dataclass with ``.summary()``
  and ``.to_dict()``.
- Pustejovsky-Tipton 2018 §3.2 moment-matching DOF η computed as
  ``q(q+1) / sum(var_mat)`` with ``var_mat`` derived from
  cluster-pair contributions to ``R · V^CR2 · R^T`` under a working
  covariance Φ = I (OLS+CR2; clubSandwich's default).
- Hotelling-T² scaling: ``F_stat = (η - q + 1) / (η · q) · Q`` with
  ``p_value = 1 - F_{q, η-q+1}.cdf(F_stat)``.

#### Verification

- 3 frozen-fixture parity tests vs R clubSandwich 0.6.2 at
  ``rtol < 1e-8`` (``q ∈ {1, 2, 3}``, balanced + unbalanced panels;
  fixture in ``tests/fixtures/htz_clubsandwich.json``, no R required
  in CI).
- 3 live-R parity tests at ``rtol < 1e-8`` (skipif ``Rscript`` missing).
- 14 unit tests: validation, invariance (X rescale + cluster relabel +
  bread arg path), edge cases (singleton cluster warning, zero
  residuals short-circuit, ``η ≤ q-1`` rejection, non-uniform weights
  ``NotImplementedError``).
- Total: 23/23 tests pass.

#### Scope (v1)

- Standalone — no wiring into ``crve`` / ``feols`` / ``fepois`` /
  ``event_study``. That's the next PR.
- Working covariance ``Φ`` locked to ``I`` (OLS+CR2). Non-uniform
  weights raise ``NotImplementedError`` with a pointer to v2.
- HTZ test variant only; HTA / HTB / KZ / Naive / EDF deferred.

#### References

- ``pustejovsky2018small`` added to ``paper.bib`` after Crossref
  dual-source verification (DOI ``10.1080/07350015.2016.1247004``;
  authors / year 2018 / vol 36(4) / pp 672–683 / title — all four
  elements verified per CLAUDE.md §10).
- Implementation derived 1:1 from Pustejovsky-Tipton 2018 §3.2 +
  clubSandwich source (R Wald_testing / get_P_array / total_variance_mat).
  No GPL code copied; clubSandwich used only as black-box reference.

### Phase 5: LLM-DAG closed loop + layered credential resolver

Closes the LLM-DAG closed-loop deferred from Phases 2-4. **No
numerical changes** to any estimator. The export pipeline can now
auto-propose a DAG via a real LLM (Anthropic Claude or OpenAI GPT)
without requiring users to pre-build one — credential resolution
follows the industry-standard layered fallback pattern.

#### Added — `sp.causal_llm.get_llm_client()` layered credential resolver

Resolution order (first match wins):

1. **Explicit ``client=``** — already-built ``LLMClient``, pass through.
2. **Explicit ``provider=`` + ``api_key=``** — construct directly.
3. **Environment variable** — ``ANTHROPIC_API_KEY`` /
   ``OPENAI_API_KEY``. When both are set, tie-break to the config
   file's ``[llm].provider`` (or to Anthropic if no config).
4. **Config file** ``~/.config/statspai/llm.toml`` (XDG-compliant) —
   stores ``provider`` and ``model`` preferences. **Never stores API
   keys** — that's the documented industry-standard split (Anthropic
   SDK / OpenAI SDK / AWS CLI / kubectl all keep keys in
   environment variables, never plaintext config).
5. **Interactive prompt** — only when ``sys.stdin.isatty()`` AND
   ``allow_interactive=True``. Walks user through provider + model
   selection but never asks for the API key over stdin (security:
   leaks in shell history, no obvious env-var integration path).
6. **Hard error** with concrete remediation: lists the env vars to
   set + points at ``sp.causal_llm.configure_llm(...)`` for the
   provider+model preference part.

#### Added — `sp.causal_llm.configure_llm()` preferences setter

One-shot setter that persists provider+model to the XDG config file.
Useful when a user has both env vars set and wants to pin the choice:

```python
import statspai as sp
sp.causal_llm.configure_llm(provider="openai", model="gpt-4o")
# → ~/.config/statspai/llm.toml gets a [llm] block with the choice.
```

#### Added — `sp.paper(..., llm='auto', llm_domain=...)` auto-DAG hook

When the user doesn't pass an explicit ``dag=``, ``llm='auto'`` (or
``llm='heuristic'`` for a pinned offline path) triggers
``llm_dag_propose`` against the resolved client + the variable list.
Failures (no API key, network error, malformed JSON) silently fall
back to a no-DAG paper — auto-DAG must never break the rest of the
pipeline. Pass ``llm_client=`` to override the resolver entirely.

The proposed DAG is materialised as a ``statspai.dag.graph.DAG`` and
attached to the ``PaperDraft``, so all downstream rendering (Quarto
mermaid block, replication_pack lineage, Causal DAG appendix) flows
through the existing Phase 3 plumbing — no new branches.

#### Added — `LLMClient.complete()` alias (latent bug fix)

``llm_dag_propose`` / ``llm_dag_validate`` / ``llm_dag_constrained``
all called ``client.complete(prompt)``, but the ``LLMClient`` base
class only defined ``chat(role, prompt)`` and ``__call__(prompt)``.
Any user passing a real ``openai_client`` / ``anthropic_client``
into the LLM-DAG functions would have hit ``AttributeError``. Added
``complete()`` as an alias on the base class — both names route
through ``chat()``, so no concrete adapter needs changes.

#### Public exports

``sp.causal_llm.get_llm_client``,
``sp.causal_llm.list_available_providers``,
``sp.causal_llm.configure_llm``,
``sp.causal_llm.LLMConfigurationError``,
``sp.causal_llm.llm_config_path``,
``sp.causal_llm.load_llm_config``,
``sp.causal_llm.DEFAULT_LLM_MODELS``.

#### Tests

27 new tests (``tests/test_llm_resolver.py``):

- Config file: XDG path, missing/malformed graceful fallback, save
  round-trip, header comment warns against putting keys in the file.
- Layered fallback: explicit client → explicit provider → env →
  config tie-break → no-env-no-tty hard error → no-env-tty-no-keys
  hard error → env-set skips prompt.
- ``configure_llm`` round-trip + unknown-provider rejection.
- ``LLMClient.complete()`` alias smoke.
- ``sp.paper(llm='auto')`` integration: no-env falls back to
  heuristic; explicit ``llm_client=`` populates the DAG.

221 green across the new + adjacent paper / lineage /
replication_pack / estimator-provenance / bibliography / gt suites.

### Phase 4: synth refactor + 5 more estimator provenance hookups

Continues the v1.7.2 provenance rollout from Phase 3 (4 estimators
instrumented). This round closes the deferred ``sp.synth`` dispatcher
refactor and adds 4 more high-leverage estimators. **No numerical
changes** to any estimator — total provenance coverage now **9/925**.

#### Changed — `sp.synth` dispatcher refactored for one-shot provenance

The previous v1.7.2 instrumentation deferred ``sp.synth`` because its
13 method branches each had their own ``return X(...)`` call site —
sprinkling 13 ``attach_provenance`` calls would've been
maintenance-hostile. Refactor splits responsibility:

- ``_dispatch_synth_impl(...)`` — full dispatcher (former ``synth``
  body), unchanged logic.
- ``synth(...)`` — thin outer wrapper that captures kwargs, calls
  impl, then attaches provenance **once** before returning.

Public signature is **bit-identical**; the 145-test synth regression
sweep passes with zero changes. All 13 SCM method variants
(``classic`` / ``penalized`` / ``demeaned`` / ``unconstrained`` /
``augmented`` / ``sdid`` / ``factor`` / ``staggered`` / ``mc`` /
``discos`` / ``multi_outcome`` / ``scpi`` / ``bayesian`` / ``bsts`` /
``penscm`` / ``fdid`` / ``cluster`` / ``sparse`` / ``kernel`` /
``kernel_ridge``) now flow through the same provenance attach.

#### Added — Provenance instrumentation for 4 more estimators

- ``sp.did.did_imputation`` — Borusyak-Jaravel-Spiess (2024) imputation.
- ``sp.did.aggte`` — Callaway-Sant'Anna ATT(g, t) aggregation. Captures
  ``upstream_run_id`` and ``upstream_function`` so downstream consumers
  can trace the aggregation step back to the producing
  ``callaway_santanna`` call (``sp.replication_pack``'s
  ``lineage.json`` thus gets a chain, not just disconnected runs).
- ``sp.did.did_multiplegt`` — de Chaisemartin-D'Haultfoeuille (2020).
- ``sp.rd.rdrobust`` — Calonico-Cattaneo-Titiunik local-polynomial RD
  with robust bias correction. Captures kernel / bwselect / fuzzy /
  donut / weights for the full reproduction recipe.

Each follows the established pattern: assign result to ``_result``,
call ``attach_provenance`` with ``overwrite=False``, return. Any
upstream-instrumented estimator (``sp.causal_question`` /
``sp.paper`` / ``aggte``) preserves the inner record.

#### Tests

9 new tests (3 synth + 1 did_imputation + 1 aggte upstream-linkage +
1 did_multiplegt + 2 rdrobust + 1 multi-estimator integration). 166
green across the DiD + RD + new provenance regression sweep
(95s wall, 145 of which are synth — the refactor is paid for in
test time once and forgotten).

#### Provenance coverage scorecard

|              | v1.7.2 P3 | v1.7.2 P4 (this) |
|---           |---        |---               |
| Instrumented | 4/925     | **9/925**        |

| Estimator                            | Status   |
|---                                   |---       |
| ``sp.regress``                       | P3 ✓     |
| ``sp.callaway_santanna``             | P3 ✓     |
| ``sp.did_2x2``                       | P3 ✓     |
| ``statspai.regression.iv.iv``        | P3 ✓     |
| ``sp.synth`` (13 method dispatcher)  | **P4 ✓** |
| ``sp.did.did_imputation``            | **P4 ✓** |
| ``sp.did.aggte`` (chain-aware)       | **P4 ✓** |
| ``sp.did.did_multiplegt``            | **P4 ✓** |
| ``sp.rd.rdrobust``                   | **P4 ✓** |

Remaining 916 estimators bucket into v1.7.3 sprint themes:
DiD long-tail (~20), IV variants (~15), synth sub-modules (already
flow through dispatcher), DML / TMLE / metalearners (~50), panel /
structural (~80), and the long tail (~750).

### Phase 3: estimand-first paper + estimator provenance + DAG appendix

Layered on top of the Phase 1+2 export trinity. **No numerical changes**
to any estimator. Three additions, each gated to **opt-in** call sites
to keep blast radius small.

#### Added — Estimand-first `sp.paper(causal_question_obj)`

The Target-Trial-Protocol-shaped declaration now drives the paper end
to end. Two equivalent entry points:

```python
# Method-style:
q = sp.causal_question("trained", "wage", data=df, design="did",
                       time="year", id="worker_id")
draft = q.paper(fmt="qmd")
draft.write("paper.qmd")

# Function-style dispatch:
draft = sp.paper(q, fmt="qmd")
```

The builder routes through ``q.identify()`` + ``q.estimate()`` and
assembles Question / Data / Identification / Estimator / Results /
Robustness / References sections whose contents match the
*declaration* (not natural-language inference). Unlike the
DataFrame-first ``sp.paper(df, "natural-language question")`` path,
this preserves the user's pre-registered estimand, design, and
identification claims verbatim — agents that pre-register get
audit-grade traceability for free.

Underlying estimator's result is exposed on
``draft.workflow.result`` so ``sp.replication_pack`` and
``draft.to_qmd()``'s Reproducibility appendix pick up provenance
automatically.

#### Added — Estimator-level provenance instrumentation (4 of 5)

Top-tier estimators now ``attach_provenance()`` to their fit result
with ``overwrite=False`` semantics — outer wrappers (``sp.causal``,
``sp.paper``) preserve the inner estimator's more-specific record:

- ``sp.regress`` (regression/ols.py).
- ``sp.callaway_santanna`` (did/callaway_santanna.py).
- ``sp.did_2x2`` (did/did_2x2.py).
- ``statspai.regression.iv.iv`` — unified 2SLS / LIML / GMM / JIVE.

Each captures: function name, key kwargs (formula / estimator /
control_group / method / etc.), 12-char SHA-256 of the input frame
(column-name + dtype + value sensitive), run uuid, version stamps.

**Deferred**: ``sp.synth`` dispatcher (13 method branches, 13+ return
sites). A dedicated v1.7.3 sprint refactors ``synth`` into an inner
``_dispatch_synth`` plus an outer wrapper that attaches provenance
once, instead of sprinkling 13 attach calls.

#### Added — Causal DAG appendix in PaperDraft

Pass ``dag=`` to ``sp.paper(...)`` (or ``q.paper(dag=...)``) and the
draft gains a *Causal DAG* section that renders fmt-aware:

- **Markdown / TeX**: text-art with the variable list, edge list,
  back-door paths, adjustment sets, and any latent ``_L_*`` confounders.
- **Quarto (.qmd)**: native ``{mermaid}`` graph block (Quarto renders
  to SVG out of the box) plus the same text fallbacks below it.

Identification-relevant info (back-door paths, adjustment sets, bad
controls) is computed from the DAG via the existing
:class:`statspai.dag.graph.DAG` API; the LLM-DAG closed loop
(``sp.llm_dag_propose / validate / constrained``) integrates as a
data source — pass any DAG those return into ``dag=``. The paper
builder doesn't itself call any LLM API; that remains the user's
explicit choice.

#### Added — Public exports

- ``sp.paper_from_question`` — alternative entry point next to the
  method-style ``q.paper()`` and the dispatcher in ``sp.paper(q)``.
- DAG-section-related fields on :class:`PaperDraft`: ``dag``,
  ``dag_treatment``, ``dag_outcome``.

#### Tests

35 new tests (14 paper_from_question + 8 estimator_provenance + 13
paper_dag_section). 295 green across the full Phase 1+2+3 + adjacent
paper / registry / help / output / workflow surface.

### HDFE silent-bug fix + completeness pass

Layered on top of the v1.8 RC `sp.fast.*` HDFE stack. **One ⚠️
correctness fix** (`event_study` cluster SE), the rest is additive.

#### ⚠️ Correctness — `sp.fast.event_study` cluster SE

`sp.fast.event_study` was computing CR1 cluster-robust SEs without
charging the absorbed FE rank against residual degrees of freedom.
The small-sample factor used `(n-1)/(n-k_dummies)` instead of
`(n-1)/(n - k_dummies - Σ(G_k - 1))`, so SEs were **systematically
too small** (~3–5% on a typical balanced panel; up to ~10% on
small/uneven designs). The fix passes `extra_df = Σ(G_k - 1)` —
matching `reghdfe` / `fixest` convention — through the new `crve`
parameter (see Added below). t-statistics and CIs reported by
`sp.fast.event_study` will now be slightly **wider**; users
re-running the same data should expect modest changes in the third
decimal of SE.

#### Added — `sp.fast.feols`: native OLS HDFE estimator

The linear sister of `sp.fast.fepois`. Pure-Python orchestration on
top of the Phase 1 Rust demean kernel + Phase 4 inference primitives;
**independent of pyfixest**. Public API mirrors `sp.fast.fepois`
(formula DSL, `vcov`, `cluster`, `weights`).

- `vcov` ∈ `{"iid", "hc1", "cr1"}`. CR1 is FE-rank-aware via the
  same `extra_df = Σ(G_k - 1)` convention used elsewhere in fast/*.
- Weighted OLS path routes through the `_weighted_ap_demean` loop
  (matches pyfixest weighted feols to ~1e-12).
- Coefficient parity vs R `fixest::feols`: **4.2e-15** at n=1M / fe1=100k
  / fe2=1k (machine epsilon). Wall-time **135 ms** vs R fixest **106 ms**
  vs pyfixest **210 ms** — i.e. 1.55× faster than pyfixest, 1.27× slower
  than fixest's mature C++. See [`benchmarks/hdfe/run_feols_bench.py`](benchmarks/hdfe/run_feols_bench.py).
- Full `coef()` / `se()` / `vcov()` / `tidy()` / `summary()` surface;
  drop-in compatible with `sp.fast.etable` for side-by-side regression
  tables alongside `sp.fast.fepois` results.

#### Added — Cluster-robust SE in `sp.fast.fepois`

`sp.fast.fepois(vcov="cr1", cluster="<col>")` now ships. Score uses
the weighted Poisson form `obs_weights · (y - μ) · X̃` with the
WLS bread `(X̃' diag(μ) X̃)^{-1}`; small-sample factor charges
`Σ(G_k - 1)` via the new `crve` parameter. NaN cluster values raise.

#### Added — `extra_df` parameter on `crve` / `boottest` / `boottest_wald`

Backward-compatible `extra_df: int = 0` parameter on all three CR1
callers. Default 0 reproduces the prior behaviour bit-for-bit; HDFE
callers should pass `extra_df = Σ(G_k - 1)` to get the FE-rank-aware
small-sample factor. Documented in each docstring; rejected if `< 0`.

#### Added — Bell-McCaffrey / Imbens-Kolesar Satterthwaite DOF

Two new helpers for small-G CR2 inference:

- `sp.fast.cluster_dof_bm(X, cluster, *, contrast, ...)` — single
  1-D contrast Satterthwaite DOF, formula
  `ν = (Σ_g λ_g)² / Σ_g λ_g²` with
  `λ_g = ‖A_g · W_g · X_g · bread · c‖²`.
- `sp.fast.cluster_dof_wald_bm(X, cluster, *, R, ...)` — q-dim
  matrix Satterthwaite for joint Wald tests, formula
  `ν_W = (Σ tr(E_g E_g'))² / Σ ‖E_g E_g'‖_F²`. q=1 collapses to
  the scalar form bit-for-bit.

Honest convention note in both docstrings: these implement BM 2002
§3 simplified, **not** clubSandwich's Pustejovsky-Tipton 2018 HTZ /
generalized form. The CR2 *variance* matches clubSandwich exactly;
the DOF differs by 5–10% on typical panels (1-D contrast) and can
differ 50–100% in the q-dim matrix Satterthwaite. For tightest
small-G inference prefer `sp.fast.boottest` / `sp.fast.boottest_wald`.

#### Changed — `sp.fast.fe_interact` rejects NaN

The 2-way fast path was silently producing collision-prone packed
codes when input columns contained NaN (`pd.factorize`'s `-1`
sentinel leaking into `c0 * n1 + c1`). Now fail-fast, matching the
fail-fast convention of `sp.fast.demean` / `sp.fast.fepois` /
`sp.fast.feols`. K-way path also restructured to progressive packing
with periodic re-densification, so deeply-nested FE chains can't
overflow `int64`.

#### Changed — Registry walks `sp.fast.*`

`sp.list_functions()` / `sp.describe_function()` now surface every
public callable in the `sp.fast.*` namespace under a `fast.<name>`
key (e.g. `fast.feols`, `fast.cluster_dof_bm`). The top-level
pyfixest-backed `sp.feols` continues to coexist as a separate
registry entry — no name collision. **+27 new registry entries** on
the v1.8 stack become Agent-discoverable for the first time.

#### Documentation

- `src/statspai/fast/jax_backend.py` — added a verified-blocked note
  for Apple Silicon (Metal). `jax-metal 0.1.1` (latest, Apple-
  maintained) is incompatible with JAX 0.10.0 at the StableHLO
  bytecode level; even basic ops fail. Verified empirically on M3.
  Workaround for users with jax-metal installed: `JAX_PLATFORMS=cpu`.
- `benchmarks/hdfe/SUMMARY.md` — added v1.8.1 follow-up section with
  OLS bench numbers and full delta vs Phase 8.

#### Tests

- `tests/test_fast_feols.py` — 20 new tests (coef / SE parity vs
  pyfixest and R fixest; weighted; intercept-only; validation; hand
  closed-form for OLS).
- `tests/test_fast_inference.py` — +14 tests (extra_df backward-compat
  and direction proofs across crve/boottest/boottest_wald; BM and
  Wald BM DOF coverage).
- `tests/test_fast_event_study.py` — +2 tests (FE-rank pin via math
  identity; R fixest SE parity within 1%).
- `tests/test_fast_fepois.py` — +6 tests (cluster CR1 path + R fixest
  SE parity).
- `tests/test_fast_within_dsl.py` — +3 tests (`fe_interact` NaN
  rejection; 2-way no-collision; K-way matches pandas tuple path).
- `tests/test_fast_etable.py` — +2 tests (etable × FeolsResult; mixed
  feols + fepois side-by-side).
- `tests/test_registry_new_modules.py` — +25 tests (parametrised
  `fast.*` registry coverage; namespace coexistence with top-level).

Total: `pytest tests/test_fast_*.py tests/test_hdfe_native.py
tests/test_registry*.py tests/test_help.py` — **267 passed,
2 graceful-skip** (was 133 at end of Phase 8).

### Phase 2: great_tables + CSL pipeline + paper auto-provenance

Layered on top of the export trinity below. **No numerical changes**
to any estimator. Three additions, all opt-in, all stdlib + soft
optional deps.

#### Added — `sp.gt(result)` great_tables adapter

Posit's ``great_tables`` is the Python port of R's gt — the
publication-oriented table grammar (cell-level styling, spanners,
footnote marks, themes, multi-target HTML/LaTeX/RTF output). The new
adapter dispatches on input type:

- :class:`RegtableResult` → full-fidelity adapter (title, notes,
  journal preset → gt theme via ``opt_footnote_marks`` and
  ``tab_options(table_font_names=...)``).
- :class:`PaperTables` → flattens panels into row groups via
  ``GT(groupname_col=...)``.
- :class:`MeanComparisonResult` → ``to_dataframe()`` round-trip.
- ``pandas.DataFrame`` → wraps verbatim with optional ``rowname_col``.
- Anything with ``to_dataframe()`` → duck-typed.

Soft dep — ``great_tables`` is **not** required to import StatsPAI.
``sp.is_great_tables_available()`` reports the dep; calling
``sp.gt(...)`` without it raises a friendly ``ImportError`` pointing
to ``pip install great_tables``. All 8 journal presets (AER / QJE /
Econometrica / RestStat / JF / AEJA / JPE / RestUd) apply without
crashing.

#### Added — `sp.csl_url()` / `sp.write_bib()` CSL pipeline

Quarto needs a ``.bib`` and a ``.csl`` to render citations. StatsPAI
captures citations as free-form strings on each estimator's
``.cite()``; this layer bridges:

- **CSL URL registry** — short journal names (``"aer"`` /
  ``"econometrica"`` / ``"qje"`` / etc.) → canonical Zotero/styles
  URLs. ``sp.csl_url('aer')`` returns the URL; we deliberately do
  **not** bundle ``.csl`` files (CC-BY-SA-3.0, incompatible with
  MIT). Users ``curl`` once at project setup.
- **Citation → BibTeX** — best-effort regex parse of canonical
  "Author Y (YEAR). Title. Journal." form into ``@article``
  entries with stable ``firstauthor + year + first-title-word``
  keys. Falls back to ``@misc`` for unparseable strings rather
  than dropping them.
- **`sp.write_bib(citations, path)`** — dedupes by computed key,
  writes a clean ``paper.bib`` Quarto can resolve.
- **Replication pack integration** — ``replication_pack`` now writes
  a real BibTeX file (``paper/paper.bib``) instead of a free-text
  dump.
- **Quarto short names** — ``draft.to_qmd(csl='aer')`` now resolves
  to ``csl: "american-economic-association.csl"`` automatically;
  pre-existing ``.csl`` filenames pass through untouched.

#### Added — `sp.paper()` auto-attaches provenance

``sp.paper()`` now calls ``attach_provenance()`` on ``workflow.result``
after the estimate stage with ``overwrite=False``: estimators that
wire their own provenance at ``fit()`` keep their (more specific)
record; estimators that don't gain workflow-level provenance for
free. Downstream consequences:

- ``replication_pack`` now picks up provenance from a plain
  ``draft = sp.paper(...)`` workflow with no further work — its
  ``lineage.json`` becomes non-empty automatically.
- ``draft.to_qmd()`` emits the ``statspai:`` YAML block
  (``version`` / ``run_id`` / ``data_hash``) and the
  ``Reproducibility {.appendix}`` body section automatically.

This is the **aggregation-point pattern** for provenance rollout:
v1.7.3+ instruments individual estimators (``sp.feols``,
``sp.did.callaway_santanna``, ``sp.iv.tsls``, ``sp.rd.rdrobust``,
``sp.synth``, …); the workflow-level hook here is the bridge.

#### Added — Public `sp.*` exports

``gt``, ``is_great_tables_available``, ``csl_url``, ``csl_filename``,
``list_csl_styles``, ``parse_citation_to_bib``, ``make_bib_key``,
``citations_to_bib_entries``, ``write_bib``.

#### Tests

46 new tests (20 gt adapter + 26 bibliography); 226 passing across
the full new + adjacent surface. Fast/Rust HDFE territory still
untouched — Phase 2 is fully orthogonal to the parallel work.

### Export trinity: numerical lineage + replication pack + Quarto emitter

Pure-additive export-layer patch. **No numerical changes** to any
estimator. Closes three concrete gaps between StatsPAI's export stack
and the R / Posit publication tooling, and lays the foundation for the
v1.7.2+ "agent-native paper" line.

#### Added — `sp.replication_pack()` (replication archive)

One-liner that bundles an analysis into the layout AEA / AEJ data
editors expect:

```python
draft = sp.paper(df, "effect of trained on wage")
sp.replication_pack(draft, "submission.zip",
                    code="analysis.py", paper_format="qmd")
```

Produces a zip with `MANIFEST.json` (versions, timestamp, git SHA,
per-file SHA-256), `README.md` (replication instructions), `data/`
(CSV + schema manifest), `code/`, `env/requirements.txt` (from
`pip freeze`, fallback `importlib.metadata`), `paper/` (rendered
draft + `paper.bib`), and `lineage.json` (aggregated provenance from
any results carrying `_provenance`). Tolerant by design — every
sub-step that fails is logged in `MANIFEST.json["warnings"]` rather
than aborting the archive.

#### Added — `sp.Provenance` / `sp.attach_provenance()` (numerical lineage)

A small dataclass attached as `result._provenance` recording: function
name, summarised params, 12-char SHA-256 of the input frame, run uuid,
StatsPAI/Python versions, ISO-8601 timestamp. Hash is column-name +
dtype + value sensitive. Estimators opt in by calling
`attach_provenance(result, function="sp.did.foo", params=..., data=df)`
at the end of their fit; backwards-compatible — unrecorded estimators
still work, recorded ones gain free traceability into every downstream
artifact (`replication_pack`, the Quarto appendix, table footers).

#### Added — `PaperDraft.to_qmd()` + `sp.paper(fmt='qmd')` (Quarto emitter)

`sp.paper()` now produces a `.qmd` document directly:

```python
draft = sp.paper(df, "effect of trained on wage", fmt="qmd")
draft.write("paper.qmd")  # quarto render paper.qmd
```

YAML frontmatter auto-emits `format: {pdf,html,docx}`,
`bibliography: paper.bib` when citations exist, optional `csl:` for
journal styles, and a `statspai:` block carrying `version` / `run_id`
/ `data_hash` for machine-readable provenance. When the underlying
workflow.result has a `_provenance` record, a `Reproducibility
{.appendix}` section is appended automatically. YAML escaping is
robust against quotes / colons / newlines in the question text.

#### Added — Public `sp.*` exports

`Provenance`, `attach_provenance`, `get_provenance`,
`compute_data_hash`, `format_provenance`, `lineage_summary`,
`replication_pack`, `ReplicationPack`. Registry entry for
`replication_pack` is full agent-native (params, returns, example,
tags, assumptions, failure modes, alternatives).

#### Tests

77 new tests (32 lineage + 18 replication pack + 27 Quarto), 232
passing across new + adjacent paper/registry/help suites. Fast/ Rust
HDFE territory untouched — runs independently of this patch.

## [1.7.1] — 2026-04-26 — `fmt="auto"` magnitude-adaptive formatting + unified book-tab xlsx style

Pure-additive output-layer patch on top of v1.7.0. **No numerical changes** to
any estimator. Two themes — both close gaps that referees and AER/QJE
production editors flag in practice:

1. `sp.regtable(..., fmt="auto")` (and `sp.modelsummary(..., fmt="auto")`)
   pick decimal precision per-cell so a single table can mix dollar-magnitude
   coefficients (`1,521`) with elasticity-magnitude coefficients (`0.288`)
   without one side rounding to bare `0`.
2. Every `*.xlsx` writer in `statspai.output` now emits the strict academic
   book-tab three-rule layout (top / mid / bottom) in Times New Roman, via a
   single new shared module `statspai.output._excel_style`.

### Added — `fmt="auto"` magnitude-adaptive formatting (`regtable`, `modelsummary`)

`sp.regtable(..., fmt="auto")` (and `sp.modelsummary(..., fmt="auto")`)
now picks decimal precision per-value so a single table can mix
dollar-magnitude coefficients (e.g. `1,521`) with elasticity-magnitude
coefficients (e.g. `0.288`) without one side being rounded to zero.

Bucketing: `|x| ≥ 1000` → comma-separated integer; `≥ 100` → integer;
`≥ 10` → 1 decimal; `≥ 1` → 2 decimals; `< 1` → 3 decimals.

**Why this matters.** Before this addition, passing a single fixed format
like `fmt="%.0f"` (sensible for a wage regression where coefficients are
in dollars) would silently round any 0.X-magnitude regressor (e.g.
lagged-earnings persistence in a wages model) to bare `0` while keeping
its significance stars — producing `0***` cells that read as nonsense.
`fmt="auto"` is the recommended setting for any specification with
mixed-magnitude regressors. The default remains `fmt="%.3f"` for
backwards compatibility.

Closes the gap with R `modelsummary::fmt_significant()` and Stata
`esttab`'s `%g`-family format codes.

### Changed — Unified book-tab three-line style across **all** xlsx exports

Every ``*.xlsx`` writer in :mod:`statspai.output` now emits the strict
academic book-tab convention (thick top rule above the column header,
thin mid-rule between header and body, thick bottom rule under the
last data row, Times New Roman throughout, no internal grid lines —
mirrors LaTeX ``booktabs`` ``\toprule`` / ``\midrule`` / ``\bottomrule``
verbatim).

Affected entrypoints:

- `sp.regtable(...).save("xxx.xlsx")` (`RegtableResult.to_excel`) —
  upgraded from a two-rule layout (heavy/heavy) to strict three-rule
  (heavy/thin/heavy).
- `sp.mean_comparison(...).to_excel(...)` — was previously a styleless
  ``pandas.DataFrame.to_excel`` dump; now goes through the shared
  book-tab renderer.
- `sp.sumstats(..., output="xxx.xlsx")` — added Times New Roman, top/
  mid/bottom rules, merged panel headers for ``by=`` MultiIndex
  columns. Also adds the `by_labels` parameter and **auto-maps binary
  0/1 group keys to ``Control`` / ``Treated``** so academic Table 1
  reads correctly out of the box (previously rendered raw ``0`` / ``1``
  as panel headers).
- `sp.modelsummary(..., output="xxx.xlsx")` — Calibri → Times New
  Roman, double-line bottom border → strict ``\bottomrule``.
- `sp.outreg2(..., filename="xxx.xlsx")` — replaces the legacy Excel
  grid layout (four-edge per-cell borders) with the book-tab three-rule
  convention; drops Calibri for Times New Roman.
- `sp.tab(..., output="xxx.xlsx")` — was unstyled; now book-tab
  compliant, chi-square test row appended as italic note below the
  table.

`sp.paper_tables(...).to_xlsx()` and `sp.collect(...).save("xxx.xlsx")`
were already book-tab compliant via ``_aer_style.excel_booktab_borders``
and are unchanged in this release.

**Implementation note.** The visual conventions live in a single new
module ``statspai.output._excel_style`` (Times constants, ``write_title``
/ ``write_header`` / ``write_body`` / ``apply_booktab_borders`` / ``write_notes``
/ ``autofit_columns`` / one-shot ``render_dataframe_to_xlsx``). Future
xlsx writers should call these primitives instead of hand-rolling
borders so the book-tab look stays consistent across all of StatsPAI.

**Why this matters.** Before this change the xlsx layer was three-way
fractured — ``regtable`` / ``outreg2`` shipped two-rule or grid
layouts, ``sumstats`` / ``tab`` / ``modelsummary`` had no rules at
all, and only ``paper_tables`` / ``collection`` matched the AER/QJE
book-tab standard. Authors copying ``lalonde_sumstats.xlsx`` straight
into a manuscript got a styleless dump. Every entrypoint now produces
output a referee would accept verbatim.

## [1.7.0] — 2026-04-25 — Phase 2 output overhaul: journal presets, auto-diagnostics, multi-SE, sp.cite(), reproducibility footer

This release closes the remaining gaps between StatsPAI's table layer
and `R::modelsummary` / `fixest::etable`. Six additive features;
**no numerical changes** to any estimator. One backwards-compat note
(see "Behavior change" below) — pure OLS without clustering or FE
produces byte-identical output to v1.6.x.

### Added — Journal presets via `template=` on `regtable`

`sp.regtable(..., template="qje")` now picks up the per-journal SE-row
label (e.g. QJE → "Robust standard errors"), default summary-stat
selection (JF/AEJA add Adj. R²; QJE drops R²), and footer notes from a
single source-of-truth registry at
`statspai.output._journals.JOURNALS`. Eight presets ship: `aer`, `qje`,
`econometrica`, `restat`, `jf`, `aeja`, `jpe`, `restud`. Adding a new
journal is one dict entry — `regtable`, `paper_tables.TEMPLATES`, and
the top-level `sp.JOURNAL_PRESETS` view all light up automatically.

### Added — Auto-extracted diagnostic rows (`diagnostics="auto"` default)

`regtable` now reads `model_info` / `diagnostics` on each result and
auto-emits journal-quality rows above the summary-stats block:

- **Fixed Effects: Yes/No** when any column absorbs FE.
- **Cluster SE: `<var>`** with the variable name when any column clusters.
- **First-stage F** for IV models (Olea-Pflueger preferred, falls back to
  per-endog F from `sp.IVRegression`).
- **Hansen J p-value** for over-identified IV.
- **Pre-trend p-value**, **Treated groups** for DiD methods.
- **Bandwidth**, **Kernel**, **Polynomial order** for RD.

Rows are emitted only when at least one column produces a non-empty
cell, and user-supplied `add_rows={...}` always overrides on label
collision. Pass `diagnostics=False` (or `"off"`) to disable.

### Added — Multi-SE side-by-side

`sp.regtable(*models, multi_se={"Cluster SE": [...]})` stacks alternative
SE specs under the primary SE row. Bracket styles cycle `[]` / `{}` /
`⟨⟩` / `«»` (the fourth pair is guillemets, not pipes — Markdown-safe).
Footer notes record each label automatically. Works across
text / HTML / LaTeX / Markdown / Excel / Word / DataFrame.

### Added — `sp.cite()` inline coefficient reporter

`sp.cite(result, "treat")` returns `"0.234*** (0.041)"` for direct
embedding in manuscript prose, Jupyter Markdown cells, and Quarto inline
expressions. Mirrors `regtable`'s star / SE / CI conventions for
cross-table consistency. Modes: `output="text"|"latex"|"markdown"|"html"`,
`second_row="se"|"t"|"p"|"ci"|"none"`. Markdown stars are escaped so
they do not collide with bold delimiters.

### Added — Reproducibility metadata footer

`sp.regtable(..., repro=True)` appends `Reproducibility: StatsPAI v1.X.Y;
2026-04-25 15:30` as the last footer line. Pass a dict to record more:
`repro={"data": df, "seed": 42, "extra": "git@<sha>"}` adds
`data 50000×12 SHA256:abcd1234ef; seed=42; ...`. Hashing skips frames
over `MAX_HASH_ROWS` (1M rows) to keep the call fast.

### ⚠️ Behavior change — `diagnostics="auto"` default emits new rows

`regtable` previously rendered **only** the rows you typed via
`add_rows={...}`. With the new `diagnostics="auto"` default, tables for
clustered or fixed-effects models now include a **Cluster SE: `<var>`** /
**Fixed Effects: Yes** row that was previously absent. Pure OLS without
clustering or FE produces byte-identical output to v1.6.x. Workarounds:

- Pass `diagnostics=False` (or `"off"`) to restore the old behavior.
- Override individual rows by passing `add_rows={"Cluster SE": [...]}`.

This is the only behavior change in the release; no numerical paths are
affected.

## [1.6.6] — 2026-04-24

Two parallel sub-releases consolidated under one version: the
journal-grade output-layer overhaul (AER/QJE DOCX, paper_tables
docx/xlsx, `sp.collect`, regtable.alpha, Quarto cross-refs) and the
HDFE LSMR/LSQR solver paired with the ⚠️ Heckman two-step SE
correctness fix.

### Output-layer overhaul: AER/QJE DOCX, paper_tables docx/xlsx, sp.collect, regtable.alpha, Quarto cross-refs

This release elevates the export layer to journal-grade output. Five
additive changes; **no breaking changes, no numerical changes** to any
estimator. Existing scripts continue to produce identical numbers.

#### Added — Quarto cross-reference output for `sp.regtable`

`sp.regtable(..., quarto_label="main", quarto_caption="Wage equation")`
now emits a Quarto-cross-referenceable Markdown table via
`result.to_quarto()` (or `result.to_markdown(quarto=True)`). The
canonical `: <caption> {#tbl-<label>}` block is appended so the
manuscript can reference the table with `@tbl-<label>`.

- The `tbl-` prefix is auto-prepended when missing
  (`quarto_label="main"` → `{#tbl-main}`); already-prefixed labels are
  not double-prefixed.
- `quarto_caption` falls back to `title` when omitted; if both are
  absent a generic default is used and a `UserWarning` is emitted.
- `output="quarto"` / `output="qmd"` make `__str__` / `print()` /
  `result.save("table.qmd")` round-trip Quarto output end-to-end.
- The leading bold-title line is suppressed in Quarto output to avoid
  duplicating the caption block.

This closes the last ergonomic gap between StatsPAI's export layer and
modern reproducible-paper toolchains (Quarto is the de-facto successor
to R Markdown for academic econ workflows).

#### Added — `sp.regtable(..., alpha=...)` now controls CI width

`sp.regtable(..., se_type="ci", alpha=0.10)` now displays 90% confidence
intervals (and labels them `90% CI`); `alpha=0.01` displays 99% CIs, etc.
Previously the `alpha` parameter was documented but ignored — the
displayed CI was always the model's stored 95% CI.

When `alpha=0.05` (default) the bounds come from the result's stored
95% CI for backward-compat (typically t-based with model df). For any
other `alpha` the bounds are recomputed as `b ± crit · se`, using the
t-distribution when `df_resid` is known and the standard normal as a
fallback.

`sp.esttab(..., ci=True, alpha=...)` mirrors the same behaviour. Both
APIs raise `ValueError` for `alpha` outside `(0, 1)`.

#### Added — AER/QJE book-tab DOCX styling

`sp.regtable(...).to_word(...)`, `sp.sumstats(..., output="*.docx")`,
`sp.tab(..., output="*.docx")` and `sp.mean_comparison(...).to_word(...)`
now render in book-tab style matching *AER* / *QJE* / *Econometrica*
conventions:

- heavy top rule above column headers (`sz=12`)
- thin mid rule below the header (`sz=4`)
- heavy bottom rule above the notes (`sz=12`)
- **no** internal vertical or horizontal borders
- Times New Roman, header bold, notes italic 8pt

The shared helper lives in `src/statspai/output/_aer_style.py`.
Previous DOCX output used the boxed `Table Grid` style.

#### Added — `sp.paper_tables(...)` DOCX / XLSX export

`sp.PaperTables` gains `.to_docx(path)` and `.to_xlsx(path)` methods,
and the `sp.paper_tables(...)` constructor accepts `docx_filename=` and
`xlsx_filename=` kwargs. Multi-panel paper bundles now go to a single
Word document (one panel per page, book-tab styled) or a single
workbook (one sheet per panel) in addition to the existing Markdown
and LaTeX outputs.

#### Added — `sp.collect()` / `sp.Collection` session-level container

A new container mirroring Stata 15's `collect` and R's `gt::gtsave`
workflow — gather any number of regressions, descriptive statistics,
balance tables, and free-form text in one container, then export the
whole bundle to a single `.docx` / `.xlsx` / `.tex` / `.md` / `.html`
file.

```python
import statspai as sp
c = sp.collect("Wage analysis", template="aer")
c.add_regression(m1, m2, m3, name="main", title="Table 1: Wage equation")
c.add_summary(df, vars=["wage", "educ"], name="desc")
c.add_balance(df, treatment="treat", variables=["age", "female"], name="bal")
c.add_text("Standard errors clustered at firm level.")
c.save("appendix.docx")   # single Word doc, AER book-tab style
c.save("appendix.xlsx")   # single workbook, one sheet per item
```

`Collection` exposes fluent `add_regression / add_table / add_summary /
add_balance / add_text / add_heading` (each returns `self`), plus
`list() / get(name) / remove(name) / clear()` for inspection. The
public factory `sp.collect()` is registered with the StatsPAI registry
and visible via `sp.help("collect")`.

#### Tests

- `tests/test_regtable_alpha.py` (6 tests) — `alpha` controls CI label
  and width; `esttab` parity; recompute matches `scipy.stats.t.ppf`
  by hand.
- `tests/test_aer_word_style.py` (6 tests) — OOXML reverse-checks the
  three rules, asserts no inner vertical borders, italic notes.
- `tests/test_paper_tables_export.py` (5 tests) — multi-panel docx /
  xlsx round-trip with book-tab borders.
- `tests/test_collection.py` (18 tests) — construction, chained adds,
  duplicate-name guard, all five export formats, registry presence.

#### Files changed

- `src/statspai/output/estimates.py` — `_ModelData` gains `df_resid`
  slot; `_ci_bounds(model, var, alpha)` helper; `EstimateTable` /
  `esttab` accept `alpha`.
- `src/statspai/output/regression_table.py` — `RegtableResult` accepts
  and uses `alpha`; `to_word` rewritten to use `_aer_style` helpers;
  `MeanComparisonResult.to_word` likewise.
- `src/statspai/output/sumstats.py` — `_sumstats_to_word` uses
  `_aer_style`.
- `src/statspai/output/tab.py` — `_tab_to_word` uses `_aer_style`.
- `src/statspai/output/paper_tables.py` — `PaperTables.to_docx` /
  `to_xlsx` added; `paper_tables()` accepts `docx_filename=` /
  `xlsx_filename=`.
- `src/statspai/output/_aer_style.py` — **new**, OOXML border
  manipulation + book-tab typography helpers.
- `src/statspai/output/collection.py` — **new**, `Collection` class
  + `collect()` factory.
- `src/statspai/output/__init__.py` — export `Collection`,
  `CollectionItem`, `collect`.
- `src/statspai/__init__.py` — export `Collection`, `CollectionItem`,
  `collect`; add to public `__all__`.
- `src/statspai/registry.py` — register `collect` under
  `category="output"`.

### 2026-04-24 — HDFE LSMR/LSQR solver + ⚠️ Heckman SE correctness fix

#### ⚠️ Correctness fix — `sp.heckman` two-step standard errors

**Affected**: `sp.heckman(...)` — the Heckman (1979) two-step selection
model. Point estimates are unchanged; **standard errors, t-statistics,
p-values and confidence intervals change**, and `model_info['sigma']` /
`model_info['rho']` now use the correct Greene (2003) formula.

**What was wrong.** Before v1.6.6, `sp.heckman` reported an ad-hoc
HC1-style sandwich that (a) ignored the selection-induced
heteroskedasticity `Var(y | X, D=1) = σ²(1 − ρ² δ_i)`, and (b) treated
the inverse Mills ratio `λ̂` as a known regressor, ignoring the
first-stage probit estimation error in γ̂ — the "generated regressor"
problem. The code itself flagged this as
`"Heckman SEs are complex; robust is conservative"`. It was a known
limitation, not a false belief; this release upgrades it from
approximate-conservative to textbook-correct.

**The fix.** `sp.heckman` now computes the Heckman (1979) / Greene
(2003, eq. 22-22) / Wooldridge (2010, §19.6) analytical variance:

```text
V(β̂) = σ̂² (X*'X*)⁻¹ [ X*'(I − ρ̂² D_δ) X* + ρ̂² F V̂_γ F' ] (X*'X*)⁻¹
```

where `δ_i = λ̂_i (λ̂_i + Z_iγ̂) ≥ 0`, `D_δ = diag(δ_i)`,
`F = X*' D_δ Z`, and `V̂_γ = (Z' diag(w_i) Z)⁻¹` is the probit
information-based variance of γ̂. Consistent σ̂² is
`σ̂² = RSS/n_sel + β̂_λ² · mean(δ_i)` (Greene 22-21), replacing the
old naive `RSS/(n−k)`. The probit IRLS helper `_probit_fit` now also
returns `V̂_γ` for consumption by the second-stage SE computation.

**What you'll see.** Heckman SEs will generally be **smaller** than
before when selection is strong (the heteroskedastic factor
`1 − ρ² δ_i ≤ 1` trims the structural-error contribution) and
**larger** when the exclusion restriction is weak (generated-regressor
uncertainty dominates). Match is to Stata's `heckman ..., twostep`
output and R's `sampleSelection::heckit` to the documented formula
precision.

#### Added — test coverage (Heckman)

- `tests/reference_parity/test_heckman_se_parity.py`: three tests
  pinning β̂ and SE to a hand-computed implementation of the
  Greene (2003) formula, plus a check that `model_info['sigma']` /
  `rho` expose the consistent σ̂² estimator.

#### Fixed

- `src/statspai/regression/heckman.py::heckman` — replace naive
  HC1 sandwich with the Heckman (1979) two-step analytical variance.
- `src/statspai/regression/heckman.py::_probit_fit` — now returns
  `(γ̂, V̂_γ)`; avoids allocating an n×n `diag(w)` via broadcasting.

#### Added — HDFE LSMR/LSQR solver option (additive, pyreghdfe parity)

- `sp.hdfe_ols` / `sp.absorb_ols` / `sp.Absorber` / `sp.demean` now accept
  `solver={"map", "lsmr", "lsqr"}` (default `"map"`, unchanged).
  - `"lsmr"` / `"lsqr"` build a sparse FE design matrix and delegate the
    within-projection to `scipy.sparse.linalg.lsmr` / `lsqr`. Weighted
    regression uses the standard √w transformation applied to both the
    sparse design and the response. No new runtime dependency — scipy
    is already core.
  - Covers the feature surface of `pyreghdfe`: multi-way FE OLS,
    robust / multi-way cluster SE, singleton drop, weights, Krylov
    solvers. `pyreghdfe` can now be archived with `sp.hdfe_ols` as a
    strict replacement (see [`MIGRATION.md`](MIGRATION.md)).
- New cross-solver parity tests in `tests/test_hdfe_native.py` verify MAP
  ≡ LSMR ≡ LSQR to `atol=1e-6` on two-way FE OLS (with and without
  weights, with and without clustering).
- `MIGRATION.md` gained a "Migrating from `pyreghdfe`" section with full
  API mapping.

#### Behavior

- HDFE default solver remains `"map"` — all HDFE numerical output
  (MAP path) is byte-identical to v1.6.5.

## [1.6.5] — 2026-04-24 — ⚠️ Standalone LIML correctness fix (follow-up to v1.6.4)

### ⚠️ Correctness fix — standalone `sp.liml` / `sp.iv.liml`

**Affected**: the standalone LIML entry point
`sp.liml(...)` = `sp.iv.liml(...)` in `statspai.regression.advanced_iv`.
This is a **separate code path** from the 2SLS/LIML/Fuller dispatcher
fixed in v1.6.4 (`sp.ivreg(method='liml')` went through the correct
`_k_class_fit` implementation and was fixed in the previous release;
the standalone `sp.liml` did not).

**Not affected**: `sp.ivreg`, `sp.iv.iv`, `sp.iv.fit`,
`sp.ivreg(method='liml' | 'fuller' | '2sls')` — all already correct
as of v1.6.4.

**What was wrong.** Two independent bugs in the standalone LIML:

1. **κ (Anderson LIML eigenvalue) used non-symmetric solver**: the code
   called `np.linalg.eigvals(np.linalg.inv(A) @ B)` on a non-symmetric
   product, which can silently return complex eigenvalues and produces
   a biased κ. This is the same bug fixed in `iv.py::_liml_kappa` in an
   earlier release, but the standalone module was an orphan copy that
   never got the fix. **Point estimates β̂ were biased** as a result.
2. **Cluster / robust SE meat used raw X**: same bug as v1.6.4, just in
   a different module. Sandwich meat is now built from the k-class
   transformed regressor `AX = (I − κ M_Z) X`.

**The fix.**

1. κ now computed via `scipy.linalg.eigh(S_exog, S_full)`
   (generalized symmetric eigenvalue problem), aligned with
   `iv.py::_liml_kappa`. Falls back to 2SLS (κ = 1) with a warning if
   the solver returns an implausible κ < 1.
2. Cluster / robust SE meat now uses `AX = I_kMz @ X_all`, matching
   the FOC `X' (I − κ M_Z) (y − X β) = 0`.

**What you'll see.** Users who called `sp.liml(...)` directly will see
**both β̂ and SE change** compared to ≤ v1.6.4. After the fix,
`sp.liml(...)` and `sp.ivreg(..., method='liml')` produce byte-identical
output, and both agree with `linearmodels.IVLIML` on β̂ to machine
precision. Cluster SEs differ from `linearmodels.IVLIML` by ~0.1–0.2%
due to a convention choice (StatsPAI uses the k-class FOC-derived
meat `AX`; linearmodels uses the 2SLS-style meat `X̂ = P_Z X`
regardless of κ). The two are asymptotically equivalent and coincide
at κ = 1 (2SLS).

### Added — test coverage

- `tests/reference_parity/test_liml_se_parity.py`: four tests —
  hand-computed projected-meat formula match, `sp.liml` vs
  `sp.ivreg(method='liml')` internal consistency (byte-exact), and
  `linearmodels.IVLIML` parity with documented convention tolerance.

### Fixed

- `src/statspai/regression/advanced_iv.py::liml` — κ solver aligned to
  `scipy.linalg.eigh` on the symmetric generalized eigenvalue problem;
  cluster / robust meat now uses projected `AX = I_kMz @ X_all`.

## [1.6.4] — 2026-04-24 — ⚠️ IV SE correctness fix

### ⚠️ Correctness fix — IV cluster & robust standard errors

**Affected**: `sp.iv`, `sp.ivreg`, `sp.iv.fit(method='2sls' | 'liml' | 'fuller')`
— any call that passes `robust={'hc0','hc1','hc2','hc3'}` or `cluster=`.

**Not affected**: point estimates `β̂` are unchanged; nonrobust (default)
standard errors are unchanged; GMM (`method='gmm'`), JIVE (`method='jive'`),
and the JIVE variants (`ujive`/`ijive`/`rjive`) are unchanged (they already
used the correct formula).

**What was wrong.** The 2SLS / LIML / Fuller k-class sandwich meat was
computed with the **unprojected** regressor matrix `X = [X_exog, X_endog]`
instead of the projected `X̂ = P_W X`. The bread
`(X' A X)^{-1} = (X̂' X̂)^{-1}` was correct; the bug was in
`src/statspai/regression/iv.py::_cluster_cov` / `::_robust_cov` call
sites which passed `X_actual` where the parameter (already misleadingly
named `X_hat`) expected the projected regressor.

This deviated from Cameron & Miller (2015), Stata `ivregress`,
`ivreg2` (Baum–Schaffer–Stillman 2007), and `linearmodels`. The
magnitude of the error depends on first-stage fit: weaker instruments
→ larger inflation of the reported SE. On the audit DGP (n=1000,
40 clusters, moderate first stage) the reported SE on the endogenous
coefficient was **2.46× too large**.

**The fix.** `_k_class_fit` now computes `AX = A @ X_actual` and passes
it to `_cluster_cov` / `_robust_cov`. For 2SLS (κ=1) this yields
`AX = P_W X = X̂`; for LIML/Fuller it is the k-class transformed
regressor `X − κ M_W X` that the k-class FOC `X' A (y − X β) = 0`
dictates. Matches `linearmodels` `IV2SLS` with `debiased=True` to
machine precision.

**What you'll see.** Reported SEs for cluster / HC0 / HC1 / HC2 / HC3
under 2SLS / LIML / Fuller will decrease (or occasionally increase)
compared to v1.6.3 and earlier. t-statistics, p-values, and confidence
intervals will change accordingly. **If you cite SEs from StatsPAI IV
in a paper, re-run and update the numbers before submission.**

### Added — test coverage

- `tests/reference_parity/test_iv_se_parity.py`: six tests pinning
  2SLS cluster / HC0 / HC1 to both a hand-computed projected-meat
  formula (Cameron–Miller) and to `linearmodels.IV2SLS` with
  `debiased=True`. Closes the coverage gap that let this bug live
  in `_cluster_cov` / `_robust_cov` since the module's introduction.

### Fixed

- `src/statspai/regression/iv.py::_k_class_fit` — pass projected
  `AX = A @ X_actual` to the sandwich meat.

## [1.6.3] — 2026-04-24 — DiD frontier sprint

Additive release focused on closing gaps in the DiD module. **No numerical
changes to existing estimators** — all new work is either new functions,
new registry entries, new tests, or docstring truth-up where the existing
docstring had overstated paper fidelity.

### Added — new DiD estimators

- **`sp.lp_did`** — Local-Projections DiD (Dube, Girardi, Jordà &
  Taylor 2023). Per-horizon long-difference OLS with time FE and
  cluster-robust SE; 'not-yet-treated' or 'never-treated' control
  variants. Paper bib key pending — reference carries ``[待核验]``.
- **`sp.ddd_heterogeneous`** — Heterogeneity-robust triple differences
  (Olden & Møen 2022 / Strezhnev 2023). CS-style cohort-time
  decomposition of DDD with a placebo subgroup, aggregated via
  switcher-count weights. `[@olden2022triple]` verified via Crossref;
  Strezhnev bib key pending.
- **`sp.did_timevarying_covariates`** — DiD with covariates frozen at
  baseline (Caetano, Callaway, Payne & Rodrigues 2022 — paper version
  `[待核验]`). Avoids the bad-controls bias when treatment affects the
  covariates. Per-(g, t) OR-DiD on frozen baseline X, aggregated with
  cohort-size weights.
- **`sp.did_multiplegt_dyn`** — dCDH (2024) intertemporal event-study
  DiD **MVP**. Long-difference per-horizon estimator with not-yet-
  treated / never-treated controls, cluster-bootstrap SE, joint
  placebo and overall Wald tests. Anchored to
  `[@dechaisemartin2024difference]` (DOI verified). **Not paper-
  parity** — switch-off events and analytical IF variance are flagged
  `[待核验]`, covered in `docs/rfc/multiplegt_dyn.md`.
- **`sp.continuous_did(method='cgs')`** — Callaway-Goodman-Bacon-
  Sant'Anna (2024) ATT(d)/ACRT(d) **MVP**. 2-period design, OR only,
  Nadaraya-Watson-style local linear smoother over dose, bootstrap
  SE. Anchored to `[@callaway2024difference]`. Full CGS cohort
  aggregation + DR/IPW + analytical IF are flagged `[待核验]` and
  tracked in `docs/rfc/continuous_did_cgs.md`.

### Added — shared infrastructure

- **`statspai.did._core`** — shared DiD primitives: cluster-bootstrap
  resampling with collision-safe relabelling, canonical event-study
  DataFrame shape, influence-function → SE plumbing, joint Wald. Used
  by the new estimators above; existing estimators retain their
  in-file copies (refactor is a separate pass). 16 unit tests.

### Added — docstring truth-up (non-numerical)

- `sp.continuous_did` docstring no longer claims "equivalent to the
  methods in Callaway, Goodman-Bacon & Sant'Anna (2024)". The heuristic
  modes (`'twfe'`, `'att_gt'`, `'dose_response'`) are explicitly
  labelled as heuristic; the CGS MVP lives at `method='cgs'`. Method
  label in returned CausalResult for the dose-bin heuristic changed
  from "Continuous DID (Callaway et al. 2024)" to "Continuous DID
  (dose-bin heuristic)" with estimand name updated accordingly.
- `sp.did_multiplegt` docstring explicitly notes its `dynamic=H`
  argument is a pair-rollup extension, **not** equivalent to the dCDH
  (2024) `did_multiplegt_dyn` estimator (which is now a separate
  function, `sp.did_multiplegt_dyn`).

### Added — test coverage

- `tests/test_continuous_did_heuristics.py` — 11 tests covering
  `method='att_gt'` and `method='dose_response'` paths that previously
  had zero dedicated tests.
- `tests/test_did_core_primitives.py` — 16 unit tests for `_core.py`.
- `tests/test_lp_did.py` — 9 tests for `sp.lp_did`.
- `tests/test_ddd_heterogeneous.py` — 7 tests for
  `sp.ddd_heterogeneous`.
- `tests/test_did_timevarying_covariates.py` — 6 tests.
- `tests/test_did_multiplegt_dyn.py` — 10 tests including method-label
  MVP warning.
- `tests/test_continuous_did_cgs.py` — 8 tests including recovery on
  linear dose-response DGP.
- `tests/reference_parity/test_did_multiplegt_parity.py` — skeleton
  with R fixture script template; skipped until
  `tests/reference_parity/fixtures/did_multiplegt/*.json` committed.

### Added — registry

Rich hand-written `FunctionSpec` entries with agent-card metadata
(assumptions, failure modes with `alternative` pointers, pre-conditions,
typical_n_min) for 18 previously auto-registered DiD estimators:
`did_2x2`, `drdid`, `sun_abraham`, `did_imputation`, `wooldridge_did`,
`etwfe`, `bacon_decomposition`, `ddd`, `cic`, `stacked_did`,
`event_study`, `did_analysis`, `harvest_did`, `overlap_weighted_did`,
`cohort_anchored_event_study`, `design_robust_event_study`,
`did_misclassified`, `did_bcf`, plus rich entries for the five new
functions above. One fabricated bib key (`roth2023trustworthy`)
detected and removed during self-review; replaced with `[待核验]`.

### Added — documentation

- `docs/guides/choosing_did_estimator.md` §4.5 **Frontier estimators**
  section distinguishes shipped vs. partial vs. not-yet-landed work
  and cross-links all three RFC documents. Makes explicit that
  `sp.did_multiplegt(dynamic=H)` is **not** the dCDH (2024) `_dyn`
  estimator.

### Fixed — citation hygiene

- **`paper.bib`**: `dechaisemartin2022fixed` upgraded from the SSRN
  working-paper stub to the published *Econometrics Journal* 26(3):
  C1–C30 (2023) version, DOI `10.1093/ectj/utac017`. Verified via two
  independent Crossref queries per CLAUDE.md §10 two-source rule.

## [1.6.2] — 2026-04-23 — DiD-frontier registry coverage

Patch release. **Pure-additive: no numerical behaviour changes.** Closes a
registry-coverage gap for two already-shipping DiD estimators that were
callable but invisible to `sp.list_functions()` / `sp.describe_function()` /
agent discovery (CLAUDE.md §4). Adds the supporting RFC design documents
under `docs/rfc/` so the registry `reference` / `remedy` pointers resolve.

### Added — registry & agent discoverability

- `sp.continuous_did` is now registered. DiD with continuous treatment
  intensity, exposing three modes: (i) TWFE with dose×post interaction,
  (ii) dose-quantile group-time ATT vs. the untreated (dose=0) arm with
  bootstrap SE, (iii) local-linear dose-response of ΔY on baseline dose.
  Callaway, Goodman-Bacon & Sant'Anna (2024) analytical
  influence-function inference is on the v1.7 roadmap — see
  `docs/rfc/continuous_did_cgs.md`.
- `sp.did_multiplegt` is now registered. de Chaisemartin &
  D'Haultfœuille (2020) DID_M estimator for treatments that switch on
  *and off* (unlike Callaway–Sant'Anna which assumes staggered
  adoption). Supports placebo lags, dynamic horizons, joint placebo
  Wald test, and cluster-bootstrap SE. The full dCDH (2024)
  intertemporal event-study (Stata `did_multiplegt_dyn`) is on the v1.7
  roadmap — see `docs/rfc/multiplegt_dyn.md`.
- `docs/rfc/` — RFC directory for not-yet-landed design docs. Ships
  with `continuous_did_cgs.md`, `multiplegt_dyn.md`,
  `did_roadmap_gap_audit.md`, plus `README.md` and a sprint-prep
  handoff note (`HANDOFF_2026-04-23.md`).

### Changed

- *None. No estimator output changes. Existing `sp.continuous_did` /
  `sp.did_multiplegt` callers observe identical numerical behaviour.*

### Fixed

- *None.*

### Notes for agents

- Both estimators now surface in `sp.list_functions()` and expose full
  `ParamSpec` / `FailureMode` / `alternatives` metadata through
  `sp.describe_function()`. Registered count rises from 923 to **925**.

## [1.6.1] — 2026-04-23 — CI/CD green-up

Patch release. No user-facing behavior or numerical change — all three
fixes target CI matrix reliability. The `hashlib.md5` change is
digest-byte-identical to v1.6.0 (verified by assert); `embed_texts` /
`sp.text_treatment_effect` outputs are bit-for-bit unchanged.

### Fixed — CI/CD green-up

- **Bandit security gate** — `src/statspai/causal_text/_common.py`
  hashing call now passes `usedforsecurity=False` to `hashlib.md5`.
  The digest is used as a deterministic bucket index for hashed-token
  embeddings, not a security primitive; the flag tells Bandit B324
  (CWE-327) that weak-hash use is intentional. Digest bytes are
  identical to the prior call — no numerical change to `embed_texts`
  or `sp.text_treatment_effect`.
- **Windows path-separator parity** — `tools/audit_bib_coverage.py::_rel`
  now emits POSIX-style paths via `Path.as_posix()`, so the
  `citations_by_key` report is identical across Windows and POSIX
  runners. Fixes `test_build_report_records_citation_locations` on
  `windows-latest`.
- **Windows CLI subprocess** — `tests/test_suggest_bibkey_backfills.py`
  now merges the child-process environment with `os.environ` (so `PATH`
  survives) before invoking the tool. Windows `CreateProcess` has no
  `_CS_PATH` fallback like POSIX `execvpe`, so an empty-env child
  cannot resolve `git.exe`. Fixes `test_cli_dry_run_does_not_mutate` on
  `windows-latest`.

## [1.6.0] — 2026-04-21 — P1 Agent-Native × Frontier + Agent-Native Infrastructure

Pure-additive release pushing two competitive axes:

- **Agent-native** — closed-loop LLM-DAG, end-to-end `sp.paper()`
  pipeline, full registry/agent-card metadata for every new function,
  typed exception taxonomy (`StatsPAIError` + 6 subclasses) with
  `recovery_hint` / `diagnostics` / `alternative_functions` payloads,
  result-object `.violations()` / `.to_agent_summary()` methods, and
  auto-generated `## For Agents` blocks in every flagship guide.
- **Methodological frontier** — five post-2020 Mendelian-randomization
  estimators (`mr_lap`, `mr_clust`, `grapple`, `mr_cml`, `mr_raps`),
  long-panel Double-ML (`sp.dml_panel`), constrained LLM-assisted PC
  discovery, and two `causal_text` MVPs (text-as-treatment,
  LLM-annotator measurement-error correction).

Together: one new top-level pipeline (`sp.paper`), four new LLM-aware
dag/text estimators (`sp.llm_dag_constrained`, `sp.llm_dag_validate`,
`sp.text_treatment_effect`, `sp.llm_annotator_correct`), constrained
PC discovery (`sp.pc_algorithm(forbidden=, required=)`), five MR
frontier estimators (`sp.mr_lap` etc.), one long-panel DML estimator
(`sp.dml_panel`), 36 populated agent cards (was 0 pre-v1.5.1), and 26
`## For Agents` blocks across 19 guides.

### Added — P1-A: closed-loop LLM-assisted causal discovery

- **`sp.llm_dag_constrained`** — iterate **propose → constrained PC →
  CI-test validate → demote** until convergence or `max_iter`.  Returns
  per-edge `llm_score` + `ci_pvalue` + `source` (`required` /
  `forbidden` / `demoted` / `ci-test`) so every kept edge is justified
  by both the LLM prior and the data.  `result.to_dag()` round-trips
  into `statspai.dag.DAG` for downstream `recommend_estimator()`.
- **`sp.llm_dag_validate`** — per-edge CI-test audit of any declared
  DAG; flags edges whose implied conditional independence is
  consistent with the data (i.e. the edge looks spurious).
- **`sp.pc_algorithm(forbidden=, required=)`** — background-knowledge
  constraints injected into PC.  Default `None` preserves the prior
  contract bit-for-bit.  Required edges win over forbidden when both
  reference the same pair.
- 18 new tests (`tests/test_llm_dag_loop.py`).
- Family guide: `docs/guides/llm_dag_family.md`.

### Added — P1-C: data → publication-draft pipeline

- **`sp.paper(data, question, ...)`** — orchestrator on top of
  `sp.causal()` that parses a natural-language question, runs the full
  `diagnose → recommend → estimate → robustness` pipeline, and
  assembles a 7-section `PaperDraft` (Question / Data / Identification
  / Estimator / Results / Robustness / References).
- **`PaperDraft`** with `to_markdown()` / `to_tex()` / `to_docx()` /
  `write(path)` / `to_dict()` / `summary()` and a `parsed_hints`
  attribute exposing what the question parser extracted.
- Lightweight question parser (`statspai.workflow.paper.parse_question`)
  recognises "effect of X on Y", "Y ~ X", DiD / RD / IV / RCT design
  hints, "instrument(ing) `Z`", "discontinuity at `c`", "running
  variable `X`".  Explicit kwargs always win.
- Per-section failure isolation: a failed estimator stage yields a
  "Pipeline notes" section rather than crashing the draft.
- 27 new tests (`tests/test_paper_pipeline.py`).
- Family guide: `docs/guides/paper_pipeline.md`.

### Added — P1-B: `sp.causal_text` (experimental MVP)

- **`sp.text_treatment_effect`** — Veitch-Wang-Blei (2020 UAI, MVP)
  text-as-treatment ATE via embedding-projected OLS with HC1 SEs.
  Hash embedder default (deterministic, dependency-free); lazy `sbert`
  optional via `pip install sentence-transformers`; custom callable
  embedder also supported.
- **`sp.llm_annotator_correct`** — Egami-Hinck-Stewart-Wei (2024)
  measurement-error correction for binary LLM-derived treatments.
  Hausman-style: estimate `p_01` / `p_10` on a hand-validated subset
  (≥30 rows spanning both classes), divide naive coefficient by
  `1 - p_01 - p_10`.  First-order SE correction; raises
  `IdentificationFailure` when the LLM has no information.
- Both methods subclass `CausalResult`, surface `status: "experimental"`
  in `result.diagnostics`, and ship full agent-card metadata
  (`assumptions` / `pre_conditions` / `failure_modes` / `alternatives`).
- 20 new tests (`tests/test_causal_text.py`).
- Family guide: `docs/guides/causal_text_family.md`.

### Added — MR Frontier (`src/statspai/mendelian/frontier.py`)

- **`sp.mr_lap`** — Sample-overlap-corrected IVW (Burgess, Davies &
  Thompson 2016 closed-form bias correction; conceptually aligned with
  the Mounier-Kutalik 2023 MR-Lap).  Required inputs: `overlap_fraction`
  and `overlap_rho` (e.g. from LD-score regression).  `overlap=0`
  exactly reproduces naive IVW.
- **`sp.mr_clust`** — Clustered Mendelian randomization via finite
  Gaussian mixture on Wald ratios (Foley, Mason, Kirk & Burgess 2021).
  EM with SNP-specific measurement SE, optional null cluster at θ=0,
  BIC-selected K.  Returns per-cluster estimates, SNP-to-cluster
  responsibilities, and the BIC path.
- **`sp.grapple`** — Profile-likelihood MR with joint weak-instrument
  and balanced-pleiotropy robustness (Wang, Zhao, Bowden, Hemani et al.
  2021, single-exposure variant).  Jointly MLE over causal β and
  pleiotropy variance τ² via L-BFGS-B; SE from observed Fisher info.
- **`sp.mr_cml`** — Constrained maximum-likelihood MR with L0-sparse
  pleiotropy, MR-cML-BIC variant (Xue, Shen & Pan 2021).  Block-
  coordinate descent jointly updates causal β, true exposure effects,
  and a K-sparse pleiotropy vector; K selected by BIC.
- **`sp.mr_raps`** — Robust Adjusted Profile Score (Zhao, Wang,
  Hemani, Bowden & Small 2020, *Annals of Statistics* 48(3)).
  Profile-likelihood MR with Tukey biweight loss + log-variance
  adjustment; same structural model as GRAPPLE but resistant to
  gross pleiotropy outliers.  Sandwich SE from M-estimator formula.

### Added — v1.7 long-panel DML (`src/statspai/dml/panel_dml.py`)

- **`sp.dml_panel`** — Long-panel Double/Debiased ML for static panel
  models with fixed effects (Clarke & Polselli 2025 simplified).
  Absorbs unit (and optional time) fixed effects via within-transform,
  cross-fits ML nuisance learners with folds that **split units**
  (Liang-Zeger compatible), reports cluster-robust SE at the unit
  level.  PLR moment for continuous or binary treatment;
  empty-covariate fallback reduces to pure FE-OLS.
  *(Citation corrected post-v1.15: the original v1.7 release note
  attributed this estimator to a "Semenova-Chernozhukov 2023
  Econometrics Journal 26(2)" paper that does not exist; the actual
  reference is Clarke & Polselli (2025) ECTJ 29(1) 69-86, DOI
  10.1093/ectj/utaf011, arXiv:2312.08174. See [Unreleased] for the
  full audit.)*

### Added — dispatcher + registry wiring

- `sp.mr(method=...)` routes `mr_lap | lap | sample_overlap`,
  `mr_clust | clust | clustered`, `grapple | profile_likelihood`,
  `mr_cml | cml | constrained_ml`, `mr_raps | raps |
  robust_profile_score` to the new estimators.
- All six new functions (5 MR + `dml_panel`) registered in
  `registry.py` with full `ParamSpec` metadata, category, tags, and
  reference.  `sp.describe_function`, `sp.function_schema`, and
  `sp.agent_card` cover them.

### Added — tests

- `tests/test_mr_frontier.py` — 41 tests covering correctness,
  boundary validation, cross-method consistency (`mr_lap` with
  `overlap=0` == IVW; `mr_cml` with `K=0` ≈ IVW; `mr_clust`
  two-cluster DGP; `mr_raps` outlier-robustness vs IVW), dispatcher
  routing, and registry/schema export.
- `tests/test_dml_panel.py` — 13 tests covering recovery under
  homogeneous treatment, FE-OLS agreement in the no-confounding
  limit, cluster-SE vs iid SE under AR(1) within-unit correlation,
  time-FE option, boundary validation, and registry metadata.

### Deferred (originally scoped for v1.6)

- **CAUSE** (Morrison et al. 2020) — the full variational-Bayes
  implementation is ~5000 LOC in the R reference and cannot be
  reference-parity validated in-cycle.  **Replaced with `mr_cml`**
  (same use-case: robust to correlated and uncorrelated pleiotropy).
  CAUSE will land in a later release once reference-parity
  infrastructure is in place.

### Agent-native infrastructure (foundation for v1.6.0)

Every layer now speaks in structured data with recovery hints, not
prose — this is the foundation the P1 frontier estimators above build
on.

### Added — agent-native exception taxonomy (`statspai.exceptions`)

- `StatsPAIError` root + `AssumptionViolation` / `IdentificationFailure`
  / `DataInsufficient` / `ConvergenceFailure` / `NumericalInstability` /
  `MethodIncompatibility`, each carrying `recovery_hint`, machine-readable
  `diagnostics`, and a ranked `alternative_functions` list.
- Warning counterparts: `StatsPAIWarning` / `ConvergenceWarning` /
  `AssumptionWarning` plus a rich-payload `sp.exceptions.warn()` helper.
- Domain errors subclass `ValueError` / `RuntimeError` for backwards
  compatibility with existing `except` blocks. No estimator behavior
  changes — migration of existing `ValueError`/`RuntimeError` call
  sites will follow incrementally.

### Added — agent-native registry schema

- `FunctionSpec` extended with `assumptions` / `pre_conditions` /
  `failure_modes` / `alternatives` / `typical_n_min` (all optional).
- New `FailureMode` dataclass: `(symptom, exception, remedy, alternative)`.
- New public accessors `sp.agent_card(name)` and
  `sp.agent_cards(category=None)` returning the superset of
  `function_schema()` plus the agent-native fields.
- Flagship families populated: `sp.regress`, `sp.iv`, `sp.did`,
  `sp.callaway_santanna`, `sp.rdrobust`, `sp.synth` (was previously
  auto-registered only).

### Added — agent-native methods on result objects

- `CausalResult.violations()` and `EconometricResults.violations()` —
  inspect stored diagnostics (pre-trend p-value, first-stage F,
  McCrary, rhat/ESS/divergences, overlap, SMD) and return flagged
  items with `severity` / `recovery_hint` / `alternatives`.
- `CausalResult.to_agent_summary()` and
  `EconometricResults.to_agent_summary()` — JSON-ready structured
  payload with point estimate, coefficients, scalar diagnostics,
  violations, and next-steps. Sits alongside existing `summary()`
  (prose) and `tidy()` (DataFrame).

### Added — guide `## For Agents` sections

- Auto-rendered from registry cards via `sp.render_agent_block(name)`
  and `sp.render_agent_blocks(category=…, names=…)`.
- `scripts/sync_agent_blocks.py` regenerates in-place between
  `<!-- AGENT-BLOCK-START: <name> --> … <!-- AGENT-BLOCK-END -->`
  markers; `--check` exits non-zero on drift (CI-friendly).
- Wired into four flagship guides so far:
  `choosing_did_estimator.md` (did + callaway_santanna),
  `choosing_iv_estimator.md` (iv),
  `choosing_rd_estimator.md` (rdrobust),
  `synth.md` (synth).
- Test guard `tests/test_agent_blocks_drift.py` fails CI if a doc
  falls out of sync with the registry.

### Tests — agent-native infrastructure

- `tests/test_exceptions.py` — hierarchy, payload, raise/catch,
  `warn()` helper, top-level exposure.
- `tests/test_agent_schema.py` — schema mechanics, `agent_card` /
  `agent_cards` APIs, `FailureMode`, parametrized flagship population.
- `tests/test_agent_result_methods.py` — `violations()` /
  `to_agent_summary()` on both result classes, JSON round-trip.
- `tests/test_agent_docs.py` — renderer output, pipe escaping,
  empty / non-empty cases.
- `tests/test_agent_blocks_drift.py` — CI guard for doc/registry sync.

### Added — agent-native follow-up sprint

- **Eight more flagship agent cards populated**: `sp.dml`,
  `sp.causal_forest`, `sp.metalearner`, `sp.match`, `sp.tmle`,
  `sp.bayes_dml` (extended), `sp.bayes_did` (new hand-register),
  `sp.bayes_iv` (new hand-register). Each carries pre-conditions,
  identifying assumptions, 3–4 failure modes with recovery hints,
  ranked alternatives, and a typical minimum-N rule of thumb.
- **Seven more guide AGENT-BLOCKs** (13 total across 11 guides now):
  `choosing_matching_estimator.md` (match),
  `callaway_santanna.md` / `cs_report.md` / `mixtape_ch09_did.md`
  (callaway_santanna), `honest_did.md` / `repeated_cross_sections.md`
  (did), `synth_experimental.md` (synth).
- **`sp.recommend` now consumes agent cards**: every recommendation
  gets `agent_card` / `pre_conditions` / `failure_modes` /
  `alternatives` / `typical_n_min` fields merged in from the registry.
  When `n_obs < typical_n_min`, a dedicated warning lands in the
  top-level `warnings` list pointing to `sp.agent_card(name)`.
  Hand-coded `assumptions` / `reason` / `code` are never overwritten
  — only empty fields are promoted from the card.
- **First call-site migrations** to the typed taxonomy, with
  `recovery_hint` + `diagnostics` + `alternative_functions` attached:
  - `sp.did_2x2` treat/time cardinality → `MethodIncompatibility`
  - `sp.did_analysis(method='cs'/'sa')` missing `id` →
    `MethodIncompatibility`
  - `sp.misclassified_did` no cohorts / no never-treated →
    `DataInsufficient`
  - IV under-identification (all 3 k-class paths) →
    `MethodIncompatibility`
  - IV singular k-class matrix → `NumericalInstability`
  - `sp.bayes_dml` non-positive DML SE → `NumericalInstability`
- **Latent registry bug fixed** — `_build_registry()` used
  `if _REGISTRY: return` as its idempotence gate, which silently
  skipped hand-written specs whenever any caller ran `register()`
  first (e.g. test fixtures). Replaced with a dedicated
  `_BASE_REGISTRY_BUILT` sentinel so flagship agent-native fields
  survive arbitrary registration order.
- **New tests**: `tests/test_recommend_agent_cards.py` (5 tests),
  `tests/test_exception_migrations.py` (7 tests). All existing
  registry / help / DID / IV / synth / matching / DML / meta-learner
  / Bayesian-DID / TMLE / causal-forest / agent-native suites
  continue to pass.

### Added — agent-native round 3 (v1.6 sprint)

- **Nine more flagship agent cards**: `sp.dml_panel` (v1.7 long panel
  DML), `sp.proximal` (+ bidirectional/fortified PCI alternatives
  exposed), `sp.mr` (dispatcher for the full MR family), `sp.qdid`,
  `sp.qte`, `sp.dose_response`, `sp.spillover`, `sp.multi_treatment`,
  `sp.network_exposure`. `sp.agent_cards()` now returns **30 populated
  entries** (was 19 after the prior sprint).
- **Thirteen more guide `## For Agents` blocks** (26 total across 19
  guides): `proximal_family.md`, `mendelian_family.md`,
  `qte_family.md` (qte + qdid), `interference_family.md` (spillover +
  network_exposure), `harvest_did.md` (did + callaway_santanna),
  `causal_text_family.md` (text_treatment_effect +
  llm_annotator_correct), `llm_dag_family.md` (llm_dag_constrained +
  llm_dag_validate), `paper_pipeline.md` (paper).
- **`paper` spec cleanup** — `alternatives` entries now use bare
  function names (`"causal"`, `"recommend"`) instead of prose strings,
  so the renderer emits `sp.causal` rather than `sp.sp.causal: ...`.
- **Six more call-site exception migrations** with recovery hints:
  - `sp.match` non-binary treatment → `MethodIncompatibility`
    pointing at `sp.multi_treatment` / `sp.dose_response`
  - `sp.match` all-same treatment → `DataInsufficient`
  - `sp.ebalance` < 2 treated-or-control → `DataInsufficient`
  - `sp.dml(model='irm')` non-binary D → `MethodIncompatibility`
  - `sp.dml(model='irm')` constant D → `IdentificationFailure`
  - `sp.conformal_synth` / `sp.augsynth` insufficient pre/post
    periods → `DataInsufficient`
- **6 new migration tests** added to
  `tests/test_exception_migrations.py` (13 total now). All existing
  DID / IV / matching / DML / meta-learners / TMLE / synth / Bayesian
  family suites (363 tests total) continue to pass.

### Added — agent-native round 4 (v1.6 closed-loop)

- **Seven more flagship agent cards**: `sp.principal_strat`
  (extended), `sp.mediate`, `sp.bartik`, `sp.bayes_rd`,
  `sp.bayes_fuzzy_rd`, `sp.bayes_mte`, `sp.conformal` (extended).
  `sp.agent_cards()` now returns **36 populated entries**
  (30 → 36).
- **Two more guide `## For Agents` blocks** (28 total across 21
  guides): `conformal_family.md` (conformal),
  `shift_share_political_panel.md` (bartik). Drift-check passes.
- **Six more exception migrations** with recovery hints:
  - `sp.gsynth` < 3 pre-periods → `DataInsufficient` pointing at
    `sp.synth` / `sp.did`
  - `sp.gsynth` < 1 post-period → `DataInsufficient`
  - `sp.sbw` non-binary treatment → `MethodIncompatibility`
    pointing at `sp.multi_treatment` / `sp.dose_response`
  - `sp.optimal_match` missing control arm → `DataInsufficient`
  - `sp.synth_survival` no donor → `DataInsufficient`
- **Closed-loop `sp.diagnose_result`**: the diagnostic battery output
  now also carries:
  - `violations` — the structured output of `result.violations()`
    (already surfaces pre-trend / first-stage F / McCrary / rhat /
    ESS / divergences / overlap / SMD with severity + recovery_hint),
  - `next_steps` — the output of
    `result.next_steps(print_result=False)`.
  The printed version includes a new "Structured violations
  (agent-native)" section below the family battery so humans and
  agents see the same triage picture. Backwards compatible: the
  existing `method_type` / `checks` keys are untouched.
- **3 new migration tests** + **8 new closed-loop tests** added to
  `tests/test_exception_migrations.py` and
  `tests/test_diagnose_result_closed_loop.py`.
- **Self-audit fix**: the `rdrobust` card's alternatives list used
  `rd_donut` (not exposed as a top-level function); replaced with
  `rdrbounds`. Doc block re-synced; drift-check green.

### Final tally (rounds 1 – 4 combined)

- **36 populated agent cards** covering: regression / IV / DID /
  RD / synth / matching / DML / meta-learners / TMLE / Bayesian
  (DID/IV/DML/RD/fuzzy-RD/MTE) / proximal / MR / principal strat /
  mediation / Bartik / QTE / QDID / dose-response / spillover /
  multi-treatment / network exposure / conformal / DML panel /
  paper / causal text / LLM-DAG.
- **28 `## For Agents` blocks** across **21 guides**, rendered by
  `python scripts/sync_agent_blocks.py` with a CI drift guard.
- **19 call-site exception migrations** to the typed taxonomy
  (`MethodIncompatibility`, `DataInsufficient`,
  `IdentificationFailure`, `NumericalInstability`) across DID / IV
  / DML / matching / synth / Bayes. All still inherit from
  `ValueError` / `RuntimeError`, so existing `except` blocks work
  unchanged.
- **Closed-loop `sp.diagnose_result`** bridges fit → violations →
  next_steps in one call, merging the family battery with the
  structured agent-native view.

### Migration notes

This release is purely additive. Existing call sites that catch
`ValueError` continue to catch `AssumptionViolation` /
`DataInsufficient` / `MethodIncompatibility` /
`IdentificationFailure`; catching `RuntimeError` continues to catch
`ConvergenceFailure` and `NumericalInstability`. New code in
StatsPAI should prefer the specific subclasses and attach a
`recovery_hint` so agents can act on failures without parsing
error strings.

---
## [1.5.0] — 2026-04-21 — Interference / Conformal / Mendelian family consolidation

Minor release.  Three concurrent improvements to the interference,
conformal causal inference, and Mendelian Randomization families:
full-family documentation guides, unified dispatchers matching the
`sp.synth` / `sp.decompose` / `sp.dml` pattern, and a targeted
correctness audit that surfaced and fixed two silent-wrong-numbers
issues.

### Added — three new family guides (interference / conformal / MR)

- `docs/guides/interference_family.md` — complete walkthrough of
  `sp.spillover`, `sp.network_exposure`, `sp.peer_effects`,
  `sp.network_hte`, `sp.inward_outward_spillover`,
  `sp.cluster_matched_pair`, `sp.cluster_cross_interference`,
  `sp.cluster_staggered_rollout`, `sp.dnc_gnn_did`.  Decision tree
  covering partial / network / cluster-RCT designs with the 5
  diagnostics every interference analysis should report (exposure
  balance, identification check for peer_effects, overlap for
  network_hte, parallel trends for staggered-cluster, sensitivity to
  exposure function).
- `docs/guides/conformal_family.md` — complete walkthrough of
  `sp.conformal_cate`, `sp.weighted_conformal_prediction`,
  `sp.conformal_counterfactual`, `sp.conformal_ite_interval`,
  `sp.conformal_density_ite`, `sp.conformal_ite_multidp`,
  `sp.conformal_debiased_ml`, `sp.conformal_fair_ite`,
  `sp.conformal_continuous`, `sp.conformal_interference`.  Clarifies
  the distinction between marginal and conditional coverage, with
  per-tool "when to use it" + how-to-read-disagreement guidance.
- `docs/guides/mendelian_family.md` — complete walkthrough of all 17
  MR functions (4 point estimators + 6 diagnostics + 3 multi-exposure
  extensions + instrument-strength F + 2 plots), organised around the
  IV1 / IV2 / IV3 assumption hierarchy.  Ships the 4 sanity checks every
  MR analysis should report and a worked BMI → T2D example.

Each guide is linked from `mkdocs.yml` under Guides and surfaces via
`sp.search_functions()`.

### Added — unified family dispatchers

Three new top-level dispatchers mirroring the style of `sp.synth` /
`sp.decompose` / `sp.dml`:

- **`sp.mr(method=..., ...)`** — single entry point for the 17-function
  Mendelian Randomization family.  Supports
  `method ∈ {"ivw", "egger", "median", "penalized_median", "mode",
  "simple_mode", "all", "mvmr", "mediation", "bma", "presso", "radial",
  "leave_one_out", "steiger", "heterogeneity", "pleiotropy_egger",
  "f_statistic", ...}` with aliases.  kwargs pass through to the target
  function.  `sp.mr_available_methods()` lists all aliases.

- **`sp.conformal(kind=..., ...)`** — single entry point for the
  10-function conformal causal inference family.  Supports
  `kind ∈ {"cate", "counterfactual", "ite", "weighted", "density",
  "multidp", "debiased", "fair", "continuous", "interference", ...}`.
  `sp.conformal_available_kinds()` lists all aliases.

- **`sp.interference(design=..., ...)`** — single entry point for the
  9-function interference / spillover family.  Supports
  `design ∈ {"partial", "network_exposure", "peer_effects",
  "network_hte", "inward_outward", "cluster_matched_pair",
  "cluster_cross", "cluster_staggered", "dnc_gnn", ...}`.
  `sp.interference_available_designs()` lists all aliases.

All three dispatchers are registered with hand-written schemas so
`sp.describe_function("mr")` / `"conformal"` / `"interference"` return
agent-readable descriptions.  30 new tests in
`tests/test_dispatchers_v150.py` guarantee the dispatcher path and the
direct-call path produce byte-for-byte identical results.

### ⚠️ Breaking — `sp.mr` is now a function, not a module alias

Prior to v1.5.0 `sp.mr` was a reference to the `statspai.mendelian`
submodule (`from . import mendelian as mr`), so `sp.mr.mr_ivw(...)`
worked.  v1.5.0 replaces this with the new **dispatcher function**
`sp.mr(method=..., ...)`.

**Migration**: code that previously wrote `sp.mr.mr_ivw(bx, by, sx, sy)`
should use the top-level `sp.mr_ivw(bx, by, sx, sy)` (already exported
in every prior version) or the new `sp.mr("ivw", beta_exposure=bx, ...)`
dispatcher.  The module is still accessible as `sp.mendelian` for users
who were doing submodule-level introspection.

Updated references: the only in-repo consumer of the old
`sp.mr.mr_ivw` form was `tests/reference_parity/test_mr_parity.py`,
which has been migrated to top-level calls.  All external user code
that already uses `sp.mr_ivw` / `sp.mendelian_randomization` / etc
continues to work unchanged.

### Fixed — silent wrong numbers (correctness audit)

- **`sp.mr_egger` — slope inference used Normal, not t(n−2).**  The
  companion `sp.mr_pleiotropy_egger` correctly used `t(n−2)` for the
  Egger intercept p-value, but `mr_egger` itself used `stats.norm.cdf`
  for both the slope p-value and the slope CI's critical value.  This
  was anti-conservative at small `n_snps`: e.g. for `n_snps = 5` and a
  t-stat of 1.5, the Normal-based two-sided p is 0.134 whereas the
  correct t(3)-based p is 0.231.  `mendelian_randomization(..., methods=["egger"])`
  inherited the bug through its internal call.  The fix switches both the
  p-value and the CI critical value to `t(n−2)`.  Regression guard in
  `tests/test_correctness_v150.py::TestMREggerUsesTDistribution`.
  For `n_snps ≥ 100` the change is numerically invisible (< 1e-3 in p).

- **`sp.mr_presso` — MC p-value could equal exactly 0.**  Both the
  global test p-value and the per-SNP outlier p-values used the raw
  `mean(null >= obs)` form, which collapses to `0.0` when the observed
  statistic exceeds every simulated null.  An MC-estimated p-value
  cannot be zero — its true lower bound is `1 / (B + 1)`.  The fix
  switches to the standard `(k + 1) / (B + 1)` convention (matching
  R's `MR-PRESSO` package).  Downstream effect: reported p-values are
  now always strictly positive and in `[1/(B+1), 1]`, which prevents
  log-transforms and sensitivity analyses from silently producing
  `-inf`.  Regression guard in
  `tests/test_correctness_v150.py::TestMRPressoMCPvalueConvention`.

### Fixed — dead code

- **`sp.network_exposure._ht_estimate`** contained a dimensionally
  inconsistent `var = ...` expression that was immediately overwritten
  by the conservative Aronow-Samii Theorem 1 bound `var_as = ...`.  The
  dead line is removed; the reported SE is unchanged.

### Fixed — registry coverage

Five previously-exposed-but-unregistered family functions now surface
in `sp.list_functions()` and have agent-readable schemas via
`sp.describe_function()`:

- `sp.network_exposure` (Aronow-Samii HT)
- `sp.peer_effects` (Bramoullé-Djebbari-Fortin 2SLS)
- `sp.weighted_conformal_prediction` (TBCR 2019 primitive)
- `sp.conformal_counterfactual` (Lei-Candès Theorem 1)
- `sp.conformal_ite_interval` (Lei-Candès Eq. 3.4 nested bound)

### No other API changes

Every other public signature is byte-for-byte identical to v1.4.2.
Existing user code keeps working; upgrades reveal slightly wider Egger
CIs at small `n_snps` and strictly positive `mr_presso` p-values.

## [1.4.2] — 2026-04-21 — correctness patches + family guides

Patch release.  No breaking changes; two silent-wrong-numbers bug
fixes in `dml_model_averaging` and `gardner_did`, plus three new
family guides (Proximal / QTE / Causal RL) closing the last gaps
between the v3 reference document and the documentation.

### Fixed — silent wrong numbers

- **`sp.dml_model_averaging` — √n SE scaling bug.** The cross-candidate
  variance aggregator treated the sample-mean influence-function outer
  product as `Var(θ̂_avg)` directly, missing a final `/ n`.  Net effect:
  reported SEs were `√n` times too large; on the canonical n=400 DGP the
  95% CI width was 4.20 (nominal ≈ 0.21) and empirical coverage was
  100% (nominal 95%).  After the fix, CI width is 0.21 and coverage is
  82% (≈ nominal, with the remaining gap explained by a 4% small-sample
  bias in the point estimate — a nuisance-tuning issue, not a
  variance-formula issue).  Regression guard added to
  `tests/test_dml_model_averaging.py::test_se_on_correct_scale`.
- **`sp.gardner_did` — event-study reference-category contamination.**
  The Stage-2 dummy regression pooled never-treated units *and* treated
  units outside the event-study horizon into a single baseline,
  dragging every event-time coefficient toward the mean of that pool.
  On a synthetic panel with true τ=2 and strict parallel trends, pre-
  trends came out ≈ -0.30 (should be 0) and post ≈ +1.72 (should be 2.0).
  Replaced the Stage-2 regression in event-study mode with direct
  Borusyak-Jaravel-Spiess-style within-(cohort × relative-time)
  averaging of the imputed gap.  After the fix: pre-trends ≈ +0.01,
  post ≈ +2.02.  Non-event-study path (single ATT) was already correct
  and is unchanged.

### Added — family guides

- `docs/guides/proximal_family.md` — complete walkthrough of the
  Proximal Causal Inference family: `sp.proximal`,
  `sp.fortified_pci`, `sp.bidirectional_pci`, `sp.pci_mtp`,
  `sp.double_negative_control`, `sp.proximal_surrogate_index`,
  `sp.select_pci_proxies`.  Includes a decision tree ("got 1 Z + 1 W /
  bridges sensitive to spec / unsure which is Z vs W / continuous
  treatment + shift policy / only have negative controls / want
  long-term from short-term experiment / have candidate proxies") and
  the four diagnostics every PCI analysis should report.
- `docs/guides/qte_family.md` — the three granularity levels (mean →
  quantile → whole distribution), with cross-sectional / DiD / IV /
  panel-with-many-controls decision paths covering `sp.qte`,
  `sp.qdid`, `sp.cic`, `sp.distributional_te`, `sp.dist_iv`,
  `sp.kan_dlate`, `sp.beyond_average_late`, and `sp.qte_hd_panel`.
- `docs/guides/causal_rl_family.md` — when to use causal RL vs
  classical causal inference, with `sp.causal_bandit`, `sp.causal_dqn`,
  `sp.offline_safe_policy`, `sp.counterfactual_policy_optimization`,
  `sp.structural_mdp`, `sp.causal_rl_benchmark`.  Ships the 4
  causal-RL-specific sanity checks.

Each guide is linked from `mkdocs.yml` under Guides and surfaces via
`sp.search_functions()` since all referenced functions have
hand-written registry specs.

### Added — tests + docs hooks (from v1.4.1 cherry-picks now formally shipped)

- `tests/test_bridge_full.py`: 10 end-to-end smoke + correctness tests
  for the six `sp.bridge(kind=...)` bridging theorems — dispatches,
  finite outputs, agreement property on correctly-specified DGPs.
- `docs/guides/bridging_theorems.md`: full walkthrough of the six
  bridges with when-to-use and how-to-read-disagreement.

### No API changes

Every public signature is byte-for-byte identical to v1.4.1.  Existing
user code keeps working; upgrades reveal narrower CIs for
`dml_model_averaging` and cleaner event-study coefs for `gardner_did`.

## [1.4.1] — 2026-04-21 — v3-frontier sprint 3 (AKM SE + Claude thinking + parity suites + docs)

Additive follow-up to v1.4.0.  All v1.4.0 APIs remain stable; new
functionality is exposed through additive kwargs on existing entry
points.

### Added — shock-clustered SE for panel shift-share

- **`sp.shift_share_political_panel(..., cluster='shock')`** — new
  option computes the panel-extended Adão-Kolesár-Morales (2019)
  variance estimator recommended by Park-Xu (2026) §4.2:

  ```text
  u_k = Σ_{i, t} s_{ikt} · Z̃_{it} · ε̂_{it}
  Var(β̂) = Σ_k u_k² / (D̂'_fit · D̃)²
  ```

  Typically 3× tighter than unit-clustered SEs in settings with 10–100
  industries.  `diagnostics['akm_se']` exposes the value alongside the
  chosen cluster type, and `diagnostics['cluster']` is now a
  human-readable label (`"shock (AKM 2019)"` when the shock estimator
  is active).
  [`bartik/political.py`]

### Added — Claude extended-thinking support for Causal MAS

- **`sp.causal_llm.anthropic_client(..., thinking_budget=N)`** — opt
  into the Claude 4.5 / Opus 4.7 **extended-thinking** API.  The
  reasoning trace is captured on `client.history[-1]['thinking']` for
  auditability but is NOT included in the public answer parsed by
  `causal_mas`.  Compatible with Anthropic's `thinking` /
  `redacted_thinking` content blocks; both are handled cleanly.
  Validates `thinking_budget >= 1024` and `< max_tokens` eagerly, so
  misconfiguration fails loudly before the first API call.
  [`causal_llm/llm_clients.py`]

### Added — parity + integration test suites

- **`tests/reference_parity/test_assimilation_parity.py`** — 10 checks
  on the Kalman / particle backends:
  - static-effect posterior recovery (both backends)
  - Kalman ↔ particle agreement on three seeds (point + SD within 15%)
  - monotone posterior variance under `process_var = 0`
  - particle-filter ESS stays above threshold after resampling
  - Student-t particle beats Kalman on a contaminated stream
  - drift tracking without variance blow-up
  - `assimilative_causal(backend=...)` matches direct-backend calls

- **`tests/integration/test_causal_mas_with_fake_llm.py`** — 11
  end-to-end integration tests using the deterministic `echo_client`
  to drive the proposer / critic / domain-expert / synthesiser loop:
  proposer parsing (newlines + bullets), critic rejection,
  domain-expert endorsement lifting confidence, transcript
  auditability, confidence scaling with rounds, role overrides, DAG
  interop via `sp.dag(...)`, plus three Claude-thinking content-block
  splitter tests that mock Anthropic responses without requiring the
  `anthropic` SDK at test time.

### Documentation

Two new MkDocs guides, wired into `mkdocs.yml` nav under
*DID & Panel Methods* / guides:

- `docs/guides/shift_share_political_panel.md` — full panel-IV recipe
  including AKM shock-cluster guidance and pretrend workflow.
- `docs/guides/causal_mas.md` — multi-agent LLM causal discovery,
  real-SDK integration, Claude thinking-mode walkthrough, and
  end-to-end pipe into `sp.dag` / `sp.identify`.

### Fixed

- Integration test used `dag.edges()` but `DAG.edges` is a list-of-
  tuples **attribute** (not a method).  Corrected to `dag.edges`.

### Backwards compatibility

- All v1.4.0 APIs remain stable.  The only new surface is additive
  kwargs:
  - `sp.shift_share_political_panel(cluster='shock')`
  - `sp.causal_llm.anthropic_client(thinking_budget=N)`

## [1.4.0] — 2026-04-21 — v3-frontier sprint 2 (extensions + LLM SDK + docs)

Follow-up to v1.3.0 covering the four secondary items flagged at the
end of Sprint 1.

### Added — panel-shift-share extension

- **`sp.shift_share_political_panel`** — multi-period extension of
  `sp.shift_share_political` per Park & Xu (2026) §4.2.  Handles
  time-varying shares **and** time-varying shocks, runs pooled 2SLS
  with unit / time / two-way fixed effects, and reports a per-period
  event-study table plus aggregate Rotemberg top-K weights.  Recovers
  τ = 0.30 within 0.003 on a 30 × 4 synthetic panel.
  [`bartik/political.py`]

### Added — real-LLM adapters for Causal MAS

- **`sp.causal_llm.openai_client`** — adapter over the `openai>=1.0`
  Python SDK; supports custom `base_url` for Azure / vLLM / Ollama.
- **`sp.causal_llm.anthropic_client`** — adapter over the
  `anthropic>=0.30` Messages API; defaults to `claude-opus-4-7`.
- **`sp.causal_llm.echo_client`** — deterministic scripted-response
  client for offline unit testing.
- All three implement a single-method `LLMClient` protocol and
  integrate with `sp.causal_llm.causal_mas(client=...)` via the
  existing `chat(role, prompt)` interface.  Lazy-imports the SDKs so
  the core package has zero new runtime dependencies.
  [`causal_llm/llm_clients.py`]

### Added — particle-filter assimilation backend

- **`sp.assimilation.particle_filter`** — bootstrap-SIR particle
  filter with systematic resampling (Gordon-Salmond-Smith 1993;
  Douc-Cappé 2005).  Handles non-Gaussian priors, heavy-tailed
  observation noise, and nonlinear dynamics via pluggable
  `prior_sampler` / `transition_sampler` / `observation_log_pdf`
  callbacks.  Agrees with the exact Kalman filter to ~0.003 under
  Gaussian DGPs.
- **`sp.assimilative_causal(..., backend='particle')`** — the
  end-to-end wrapper now routes to the particle filter when
  `backend='particle'`.
  [`assimilation/particle.py`]

### Documentation

Three new MkDocs guides covering the v3-frontier estimators:

- `docs/guides/synth_experimental.md` — Abadie-Zhao inverse-SC workflow.
- `docs/guides/harvest_did.md` — Borusyak-Hull-Jaravel harvesting DID.
- `docs/guides/assimilative_ci.md` — Nature Comms 2026 streaming CI
  with both Kalman and particle backends.

All three are wired into `mkdocs.yml` nav under the *DID & Panel
Methods* / guides section.

### Registry + agent schema

- 5 new hand-written `FunctionSpec` entries:
  `shift_share_political_panel`, `particle_filter`, `openai_client`,
  `anthropic_client`, `echo_client`.

### Code-quality pass (Sprint 1 audit)

- Removed 20 unused imports / shadow variables across the Sprint 1
  modules identified by `pyflakes` (`did/harvest.py`,
  `bcf/ordinal.py`, `bcf/factor_exposure.py`,
  `causal_llm/causal_mas.py`, `bartik/political.py`,
  `assimilation/kalman.py`, `target_trial/report.py`).

### Fixed

- `tests/external_parity/test_causalml_book.py::test_forest_ate_recovers_average_tau`
  was flaking on `ubuntu-latest + Python 3.10` because only the
  data-generating RNG was seeded — the causal forest's bootstrap +
  honest-split sampling was unseeded, so the ATE estimate varied
  by ±0.3 between OS / Python matrix entries and the
  `|ATE - 0.5| < 0.3` tolerance occasionally failed. Fixed by
  passing `random_state=0` + `n_estimators=300` + bumping `n` to
  1 500 so the test is fully deterministic across the matrix.

## [1.3.0] — 2026-04-21 — v3-frontier sprint (Sprint 1 of the 知识地图 v3 roadmap)

Builds on top of the v1.2.0 doc-alignment work by implementing the
eleven highest-leverage frontier methods identified in the 2026-04-20
*Causal-Inference Method Family 万字剖析 v3* gap analysis.  Every new
public function is wired into the registry + agent schema so it
surfaces through `sp.list_functions`, `sp.describe_function`, and
`sp.all_schemas` for LLM agents.

### Added — P0 frontier (4 methods, within-sprint week 1)

- **`sp.synth_experimental_design`** — Abadie & Zhao (2025/2026)
  inverse synthetic controls: picks the best ``k`` candidate units to
  treat by minimising the sum of per-unit pre-period SC MSPEs.
  Produces a ranking table, recommended treatment assignment, and a
  variance-gain benchmark against random allocation.
  [`synth/experimental_design.py`]

- **`sp.rdrobust(..., bootstrap='rbc', n_boot=999, random_state=...)`**
  — Cavaliere, Gonçalves, Nielsen & Zanelli (arXiv:2512.00566, 2025) robust-bias-corrected
  studentised percentile bootstrap.  Empirically delivers CIs ~3–15%
  shorter than the analytic robust CI without sacrificing coverage.
  New ``model_info['rbc_bootstrap']`` block exposes the CI, p-value,
  length-ratio, and effective replicate count.

- **`sp.fairness.evidence_without_injustice`** — Loi, Di Bello & Cangiotti
  (arXiv:2510.12822, 2025) counterfactual-fairness test that freezes
  admissible-evidence features at their factual values and tests
  whether predictions still change under ``do(A = a')``.  Returns a
  bootstrap CI, p-value, and per-alternative breakdown.
  [`fairness/evidence_test.py`]

- **`sp.target_trial.to_paper(..., fmt='jama' | 'bmj')`** — renders a
  JAMA / BMJ-ready manuscript with all 21 TARGET Statement (JAMA/BMJ
  2025-09) items auto-filled where derivable plus `(supply text)`
  placeholders elsewhere.  Supports `authors`, `funding`,
  `registration`, `data_availability`, `background`, `limitations`
  keyword arguments.

### Added — P1 frontier (4 methods, within-sprint week 2)

- **`sp.harvest_did`** — Abadie, Angrist, Frandsen & Pischke, NBER WP 34550 (2025)
  Harvesting DID + event-study framework: extracts every valid 2×2
  DID comparison from a staggered panel, combines them via
  inverse-variance weights, and reports event-study + pretrend Wald
  tests.  Uses a not-yet-treated-at-max(t₁, t₂) clean-control filter
  that correctly handles placebo horizons.  [`did/harvest.py`]

- **`sp.bcf_ordinal`** — Zorzetto et al. (2026) BCF for ordered / dose
  treatments.  Chains pairwise binary BCF between consecutive levels
  to yield cumulative dose-response CATEs with per-level ATEs.
  [`bcf/ordinal.py`]

- **`sp.bcf_factor_exposure`** — arXiv:2601.16595 (2026) BCF on
  PCA-factor scores of a high-dimensional exposure vector.  SVD or
  user-supplied loadings compress the exposure to ``K`` factors; one
  BCF is fit per factor.  Returns per-factor ATEs, loadings, scores,
  and an aggregate mixture-ATE with CI.  [`bcf/factor_exposure.py`]

- **`sp.causal_llm.causal_mas`** — arXiv:2509.00987 (2025/09) multi-
  agent causal discovery framework.  Runs proposer / critic /
  domain-expert / synthesiser agents over several debate rounds with
  per-edge confidence scores and a full auditable transcript.
  Offline heuristic backend by default; accepts any
  ``chat(role, prompt)`` / ``complete(prompt)`` LLM client.
  [`causal_llm/causal_mas.py`]

- **`sp.shift_share_political`** — Park & Xu (arXiv:2603.00135, 2026)
  political-science variant of the Bartik IV.  Long-difference 2SLS
  with AKM shock-cluster SEs, Rotemberg top-K diagnostic, and
  share-balance F-test against pre-treatment covariates.
  [`bartik/political.py`]

### Added — P2 frontier + testing (2 methods + 2 test suites)

- **`sp.assimilation.causal_kalman`**,
  **`sp.assimilation.assimilative_causal`** —
  *Assimilative Causal Inference* (Nature Communications 2026): a
  Kalman filter over streaming causal-effect estimates.  Produces a
  running posterior with effective-sample-size diagnostics, pluggable
  dynamics (static or random-walk), and an end-to-end wrapper that
  runs a user-supplied per-batch estimator.  New subpackage
  [`assimilation/`].

- **`tests/reference_parity/test_mr_parity.py`** — 7 analytic-truth
  checks over the MR suite (IVW consistency, Egger intercept under
  balanced pleiotropy, Egger directional-pleiotropy detection,
  weighted-median robustness, PRESSO outlier flag, LOO stability,
  Radial-Wald exact agreement).  All 7 pass.

- **`tests/external_parity/test_causalml_book.py`** — 7 CausalMLBook
  (Chernozhukov et al. 2024–2025) canonical-DGP checks: DML-PLR,
  Causal Forest, T-learner, 2SLS, Callaway–Sant'Anna DID, rdrobust,
  and rbc-bootstrap vs analytic parity.  All 7 pass.

### Registry + agent schema

- 9 hand-written `FunctionSpec` entries for every new public function:
  `synth_experimental_design`, `evidence_without_injustice`,
  `harvest_did`, `bcf_ordinal`, `bcf_factor_exposure`, `causal_mas`,
  `shift_share_political`, `causal_kalman`, `assimilative_causal`.
  Each entry ships with NumPy-style parameter docs, examples, tags,
  and paper references for LLM-agent consumption.

### Backwards compatibility

- All v1.2.x public APIs remain stable.  The only changes to existing
  signatures are additive kwargs:
  - `sp.rdrobust` — `bootstrap`, `n_boot`, `random_state`
  - `sp.target_trial.to_paper` — `journal`, `authors`, `funding`,
    `registration`, `data_availability`, `background`, `limitations`

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
- `sp.continuous_iv_late` (top-level alias) — Zeng et al. (2025,
  arXiv:2504.03063) LATE on the maximal complier class for continuous
  instruments via quantile-bin Wald estimator. (Also fixed a summary
  formatting bug — see below.)

**TMLE**

- `sp.hal_tmle` + `sp.HALRegressor` / `sp.HALClassifier` — TMLE with
  Highly Adaptive Lasso nuisance learners (Li, Qiu, Wang & van der
  Laan, 2025, arXiv:2506.17214). Two variants: `"delta"` (plug HAL into standard
  TMLE) and `"projection"` (apply tangent-space shrinkage to the
  targeting epsilon). Recovers ATE within ~3% on n=400 with rich
  nuisance.

**Synthetic Control**

- `sp.synth_survival` — Synthetic Survival Control (Han & Shah,
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
- `question.CausalQuestion.save` — added `TYPE_CHECKING` import for
  `pathlib.Path` so the stringified return annotation stops tripping
  `flake8 F821` in CI.
- Added `tabulate>=0.9.0` to core dependencies. `pandas.to_markdown()`
  dispatches to `tabulate`, which was previously a pandas-optional
  dep; user-facing `sp.causal(...).report('markdown' | 'html')`
  triggered an `ImportError` on systems (Windows, fresh envs) that
  didn't happen to transitively install `tabulate`.

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
  `design_robust_event_study` (Wright 2026, arXiv 2601.18801),
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
- **`sp.causal_rl`** (NEW namespace): `causal_dqn` (Li-Zhang-Bareinboim
  confounding-robust Deep Q, arXiv 2510.21110), `causal_rl_benchmark`
  (5 benchmarks per Cunha-Liu-French-Mian, arXiv 2512.18135),
  `offline_safe_policy` (Chemingui et al., arXiv 2510.22027).
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

**Target trial emulation — structured report**

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
  modes raise across the parametrised five-function set.

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

  A bundled CATE learner race with honest held-out scoring. `econml`'s
  multi-metalearner pipeline is not bundled into
  a single call; `causalml`'s BaseMetaLearner comparison doesn't run
  BLP calibration per learner.

- **`sp.check_identification(..., strict=True)`** raises
  `sp.IdentificationError` when the report's verdict is `'BLOCKERS'`.
  The exception carries the complete report on `.report` for post-mortem
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

## [0.9.3.post] — 0.9.3 post-release bugfixes (rolled into a later patch)

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
  `mlogit`.
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
  Software-journal submission (`paper.md`) was already correct.

## [0.9.2] - 2026-04-16

### Decomposition Analysis — Broad Decomposition Toolkit in Python

Release focus: `statspai.decomposition`. **18 first-class decomposition methods across 13 modules (~6,200 LOC, 54 tests)** spanning mean, distributional, inequality, demographic, and causal decomposition. The release consolidated a broad Python API surface for workflows that are often split across Stata commands and R packages; numerical claims remain tied to the method-level tests and validation metadata.

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

### Regression Discontinuity — Broad RD Toolkit

Release focus: `statspai.rd`. **18+ RD estimators, diagnostics, and inference methods across 14 modules (~10,300 LOC)**. The machinery behind Calonico-Cattaneo-Titiunik (CCT), Cattaneo-Jansson-Ma density tests, Armstrong-Kolesar honest CIs, Cattaneo-Titiunik-Vazquez-Bare local randomization, Cattaneo-Titiunik-Yu boundary (2D) RD, and Angrist-Rokkanen external validity is exposed through one `import statspai as sp`; validation status is method-specific.

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
| `sp.rdsummary(df, ...)` | **Single-call dashboard** — rdrobust + density test + bandwidth sensitivity + placebo cutoffs + covariate balance |
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

### Synthetic Control — Broad SCM Toolkit

Release focus: `statspai.synth`. **20 SCM methods + 6 inference strategies + analysis workflow (compare / power / sensitivity / reports)**, all behind the unified `sp.synth(method=...)` dispatcher. This is an API-breadth statement; exact validation evidence is recorded by each function's validation metadata and the parity artifacts.

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
- `synth_report(result, format='markdown'|'latex'|'text')` — one-command structured report
- `synth_power(df, effect_sizes=[...])` — power-analysis helper for SCM designs
- `synth_mde(df, target_power=0.8)` — minimum detectable effect
- `synth_sensitivity(result)` — LOO + time placebos + donor sensitivity + RMSPE filtering
- Three canonical datasets shipped: `california_tobacco()`, `german_reunification()`, `basque_terrorism()`

#### Critical fixes from comprehensive module review

Following a 5-parallel-agent code review (correctness / numerics / API / perf / docs), nine critical review findings were fixed:

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

- **144 synth tests passing** (new: 12-method cross-method consistency benchmark verifying the benchmarked synth methods recover a known ATT within 1.5 units on a clean DGP).
- **Full suite: 1481 passed, 4 skipped, 0 failed** (5m42s).
- New guide: `docs/guides/synth.md` — complete tutorial covering all 20 methods with a method-choice decision table.

#### API migration notes

`sdid(y=, treat_unit=, treat_time=)` still works but `outcome=, treated_unit=, treatment_time=` is preferred for consistency with every other `sp.synth.*` function. A deprecation of the legacy names is planned for v1.0.

### Other Modules

Decomposition and Regression Discontinuity modules received significant upgrades in this release cycle (tier-C decomposition expansion to 18 methods + unified `sp.decompose()`; RD `_core.py` primitive centralization + bug fixes from code review). These will be highlighted in a dedicated follow-up release note.

---

## [0.8.0] - 2026-04-16

### Spatial Econometrics + 10-Domain Breadth Upgrade

**Largest release in StatsPAI history. 60+ new functions across 10 domains.**

#### Spatial Econometrics (NEW — 38 API symbols)

From 3 functions / 419 LOC to **38 functions / 3,178 LOC / 69 tests**. A unified spatial econometrics API for Python users.

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
- **`CSReport.to_latex()`** — formatted booktabs fragment
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
- **Numerical validation tests**: early selected Stata/R reference checks
  with humanized error messages; current package-wide evidence is reported
  through `validation_status`, not a blanket validation claim

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
