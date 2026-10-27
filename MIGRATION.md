# Migrating between StatsPAI versions + from PyStataR

Internal version-to-version migrations are at the top; the long-form
`PyStataR → StatsPAI` migration follows below.

---

## 1.41 → 1.42: ⚠️ `sp.estat` after weighted and likelihood fits, directed network measures (option scan round 6)

- `sp.estat` after `sp.regress(weights=)`: `hettest`, `reset`, `vif` and
  `linktest` change (they now use the weights, as Stata does after
  `[aw=]`). After `weights_as='sampling'`, `hettest` raises and `reset` is
  the robust Wald test. `white`, `imtest`, `normality` and `leverage`
  raise after any weights. Unweighted fits are unchanged.
- `sp.estat(fit, 'linktest')` after `sp.logit` / `sp.probit` /
  `sp.poisson` refits that model and reports z. The linear tests
  (`hettest`, `white`, `imtest`, `reset`, `bgodfrey`, `durbinalt`,
  `archlm`) raise after a likelihood fit; `'all'` leaves them out.
- `sp.logit_gof`, `sp.logit_influence` and `sp.estat(fit,
  'classification')` raise after a weighted fit. Refit without weights.
- Directed graphs: `sp.transitivity` / `sp.clustering` (read as undirected),
  `sp.assortativity` (out-degree to in-degree) and `sp.network_modularity`
  (Leicht-Newman) change and now match igraph. Undirected graphs are
  unchanged.
- `sp.eigenvector_centrality` is unchanged; pass `mode='in'` for igraph's
  and networkx's orientation on a directed graph.
- `sp.hits` uses tie weights on a weighted graph; pass `weighted=False`
  for the old scores.
- `sp.netlm(method='dsp')`: the intercept's p-value changes. Predictor
  p-values do not.
- `sp.ergm` with `nodefactor:<attr>` returns one term per level except the
  first (`nodefactor.<attr>.<level>`), not a single term.

---

## 1.41 → 1.42: ⚠️ table and plot intervals, `sp.binscatter`, `sp.fast` clustered inference (option scan round 6)

- `sp.regtable` / `sp.esttab` / `sp.cite` with `alpha=` other than the
  fit's level: bounds change for clustered fits (t(G - 1)) and likelihood
  fits (normal). They now equal `result.conf_int(alpha)`.
- A result fitted with `alpha=0.10` (or any non-default level) shown with
  `se_type='ci'`: the table used to print the stored 90% bounds as 95%.
  Pass `alpha=0.10` to the table to get the stored bounds.
- `sp.coefplot` / `sp.coefplot_tikz`: bars are the model's own interval,
  wider for small-sample t fits. There is no option for the old `b ± z·se`.
- `sp.binscatter(absorb=)`: points move to the means of `x` and `y`.
  `weights=`: bins, points and intervals change (weighted, as Stata). With
  `absorb=` and `controls=` the points change (the controls are absorbed).
  Unweighted calls without `absorb=` are unchanged.
- `sp.fast.event_study`: standard errors fall by the factor of the nested
  fixed effects (about 5% on a typical panel).
- `sp.fast.feols(vcov='cr1').tidy()`: p-values use t(G - 1);
  `sp.fast.etable` stars follow each fit's own p-values.
- `sp.mean_comparison(weights=)` raises. Use `sp.sumstats(data, vars,
  by=group, weights=w)` for weighted means, or `sp.regress('x ~ group',
  data=data, weights=w, vce='hc1')` for a weighted difference and its test.
  A `group=` with more than two values raises; subset the data first.

---

## 1.40.2 → 1.41: ⚠️ cluster folds in `sp.aipw`, conformal test weights, weighted R-learner (round-5 scan)

- `sp.aipw(cluster=)` with `cross_fit=True` (the default): folds are drawn
  over whole clusters, so estimates and SEs change. There is no option for
  the old row-level split, which let a cluster's own rows train the models
  that predicted it. Without `cluster=` nothing changes.
- `sp.aipw`: a cross-fitting fold whose training rows lack an arm now
  raises `DataInsufficient` (it predicted 0 for that arm). Use fewer folds
  or `cross_fit=False`.
- `sp.weighted_conformal_prediction`: pass `weights_test=w(X_test)` with
  `weights_calib=`; without it the call warns and keeps the old weight 1.
  `sp.conformal_counterfactual` / `sp.conformal_ite_interval` intervals
  change (they now use each test point's own weight).
- `sp.metalearner(learner='r', weights=)` with a continuous treatment:
  `estimate` / `se` change (weighted nuisance fits).
- `sp.subgroup_analysis`: `results_df` has a `note` column and keeps
  subgroups without an estimate as NaN rows; filter on `note == ""` for the
  estimated ones.

---

## 1.40.2 → 1.41: ⚠️ power, rlasso, cross_validate and interference (option scan round 5)

- `sp.cross_validate(weights=)`: engines now fit the weighted model; DoubleML
  and the DiD engines are skipped. A cross-check that said AGREE before was
  comparing unweighted fits. With `"y ~ x | fe | d ~ z"` the statspai engine
  now absorbs the fixed effects; estimand `"ols"` with a `| fe` block is
  read as `"feols"`. The DoubleML engine follows `model=` (`'irm'` runs
  `DoubleMLIRM`; other models are skipped). linearmodels skips multiway
  clustering. The Stata engine passes `vce()`, weights and `small`.
- `sp.iv(absorb=)`: t statistics and p-values now match the reported SEs;
  coefficients, SEs and intervals are unchanged.
- `sp.rlasso_effect` / `sp.rlasso_effects` / `sp.rlasso_iv`: `vce='robust'`
  now gives a robust SE on every path (it was ignored for partialling out
  and for `select_X` only). Defaults are unchanged. New `robust=` argument.
- `sp.rlasso` family: missing or infinite values raise `DataInsufficient`
  instead of selecting nothing.
- `sp.power_two_proportions` / `sp.power_logrank` / `sp.power_case_control`:
  `alternative` must be a two-sided or one-sided spelling. `"two.sided"`
  and `"two_sided"` are now two-sided (they were one-sided); `"greater"` /
  `"less"` raise, use `"one-sided"`.
- `sp.power(..., power=)` can return fewer than 11 units. `sp.mde` with
  `sigma` other than 1 can move in the sixth significant digit, and works
  for large `sigma`.
- `sp.cluster_staggered_rollout`: event time is counted in observed
  periods; NaN `first_treat` is never-treated. Results on consecutive
  integer periods with 0-coded controls are unchanged.
- `sp.peer_effects` raises `IdentificationFailure` in designs where the peer
  effect is not identified.
- `sp.network_hte`: SEs are HC0 sandwich SEs; point estimates unchanged.
- `sp.regress` and the other formula estimators raise on a `| fe` or
  `(d ~ z)` block they cannot fit, instead of dropping it.

---

## 1.40.2 → 1.41: ⚠️ attributable fractions, NNT intervals, NDCG ties, MR-BMA limits

- `sp.epi.attributable_risk` on a table with a zero cell: `rr`,
  `ar_exposed` and `paf` are the plug-in values on the raw counts (Stata
  `cs`). Without a case in one exposure group `paf_ci` is `(nan, nan)` with
  a warning; without any case the call raises `DataInsufficient`. Tables
  without zeros are unchanged. No option restores the Haldane-based values.
- `sp.epi.number_needed_to_treat` when the risk-difference interval contains
  zero: `ci` is `(1 / max(|L|, U), inf)` instead of `(inf, inf)`. Read
  `ci_disjoint` before treating `ci` as an ordinary interval, and use
  `ci_signed` (the epiR pair) for Altman's two-sided statement.
- `sp.ndcg` with tied scores: the value no longer depends on row order. To
  reproduce a row-order tie break (R `farr::ndcg`), subtract a tiny
  multiple of the row number from the scores first.
- `sp.mr_bma`: a `max_model_size` above the number of exposures now raises
  instead of meaning "all"; pass `None`. Duplicated exposures and invalid
  `se_y` raise.

---

## 1.40.2 → 1.41: removals, ⚠️ `sp.distributional_te` SEs, `sp.absorb_ols(cluster_df=)` announced for 1.42

**Removed** (each was deprecated with a removal version that has passed):

- `sp.kan_dlate`: call `sp.dist_iv` with the same arguments. It returns the
  numbers `sp.kan_dlate` returned.
- `sp.SurrogateResult`: no function returned it. Use the `sp.CausalResult`
  that `sp.surrogate_index`, `sp.long_term_from_short` and
  `sp.proximal_surrogate_index` return (`.estimate`, `.se`, `.ci`,
  `.model_info`).
- `sp.qte(method='quantile_regression')`: pass `method='conditional_qr'` for
  the same numbers, or `method='firpo_qte'` for the Firpo (2007) estimator.
  The old name raises `MethodIncompatibility`.
- `sp.evalue(rare_outcome=)` and `sp.evalue_from_result(rare_outcome=)` are removed; pass `rare=`, which takes the same values.
  In `sp.evalue`, `alpha` moves up one place, so a call that passed `alpha` positionally after `rare_outcome` must name it.

**⚠️ `sp.distributional_te` standard errors.** The bootstrap SEs (and the
pointwise bands built from them) use `ddof=1` like every other bootstrap SE
in `qte/`. They are larger by `sqrt(B/(B-1))`, about 0.1% at the default
`n_boot=500`. There is no option for the old value.

**`sp.absorb_ols(cluster_df=)`, announced for 1.42.** With multi-way
clustering the default `'per_term'` becomes `'min'` (Stata `reghdfe`, R
`fixest`, and `sp.hdfe_ols` since 1.41). Calls with two or more cluster
variables and no `cluster_df=` warn now. Pass `cluster_df='per_term'` to
keep the current standard errors or `cluster_df='min'` to adopt the new
ones. One-way clustering is the same either way.

**Input checks.** `sp.heckman` refuses a `select` column coded other than
0/1. `sp.ipcw` drops rows with a missing time, event or covariate (with a
warning, NaN weight, `result.n_dropped_missing`) instead of returning NaN
for every weight.

---

## 1.40.2 → 1.41: ⚠️ spatial ML search interval and `sp.impacts` stability interval

Numbers change in these calls; there is no option to restore the old ones,
which were wrong.

- `sp.sar` / `sp.sem` / `sp.sdm`: when the old fit returned `rho` or
  `lambda` at -0.99 (or at 0.99 times `1 / omega_max`, the largest
  eigenvalue of `W`), the search now continues over the admissible
  interval `(1 / omega_min, 1 / omega_max)` as `spatialreg` does, and the
  estimate, its SEs and the log-likelihood change. Fits that did not stop
  on that edge are unchanged bit for bit.
- `sp.impacts`: draws are kept when the spatial parameter lies inside
  `(1 / omega_min, 1 / omega_max)` of `W`, not inside (-0.999, 0.999).
  Simulated SEs change for an unstandardised `W` (draws past
  `1 / omega_max` are now dropped) and for a row-standardised `W` whose
  smallest eigenvalue is -1 when draws fall in (-1, -0.999) (now kept, as
  in R; near -1 the direct and indirect SEs are dominated by these draws in
  both). A fitted `rho` below -0.999 is no longer refused.
- `sp.impacts` on `sp.sem` raises `MethodIncompatibility`, a subclass of
  `ValueError`, so `except ValueError` still catches it.

---

## 1.40.2 → 1.41: ⚠️ CATE, TMLE survival, conformal and policy-tree fixes (Oct 2026 scan)

- `sp.ltmle_survival`: the survival curves are now the sequential-regression
  TMLE with every past treatment set to the regime. Numbers change whenever
  earlier treatment affects later hazards. There is no option for the old
  estimator, which was biased.
- `sp.conformal_cate`: `estimate` / `se` are the cross-fitted AIPW average
  and its SE (as `sp.metalearner`). The old point estimate is
  `model_info['cate_mean']`; the old SE has no replacement.
- `sp.conformal_fair_ite`: intervals are about twice as wide; they now cover
  the effect rather than one potential outcome.
- `sp.focal_cate`: `se_grid` is the standard error of the CATE prediction.
- `sp.policy_tree`: `value_policy`, `value_gain`, their SEs and intervals
  are cross-fitted by default. Pass `value_evaluation='in_sample'` to
  reproduce earlier numbers (it warns that they are optimistic).
- `sp.auto_cate`: `blp_beta1_pvalue` tests `beta1 = 0`.
- OPE with fitted behaviour propensities and actions not coded `0..K-1`:
  values change (the old ones used uniform propensities).
- `sp.cate_by_group` / `sp.blp_test` / `sp.gate_test` after rows were
  dropped: rows are now paired correctly; a frame of any other length than
  the estimation sample or the input frame raises instead of being sliced.

---

## 1.40.2 → 1.41: announced defaults take effect

The defaults that 1.40.0 and 1.40.1 announced with a `FutureWarning`
change now, and the warnings are gone. Code that named the argument is
unaffected; the last column of the table restores the 1.40 behaviour.

| function | 1.40 default | 1.41 default | keep 1.40 |
| --- | --- | --- | --- |
| ⚠️ `sp.event_study` | normal critical values | t(G-1) | `crit_dist="normal"` |
| ⚠️ `sp.sun_abraham` | normal critical values | t(G-1); t(N-K) with `i=None` | `crit_dist="normal"` |
| ⚠️ `sp.did_2x2` | t(N-K), also with `cluster=` | t(G-1) with `cluster=`, else t(N-K) | `crit_dist="residual"` |
| ⚠️ `sp.sun_abraham` | headline `aggregation="event_time"` | `aggregation="fixest_att"` | `aggregation="event_time"` |
| ⚠️ `sp.hdfe_ols`, two or more clusters | `cluster_df="per_term"` | `cluster_df="min"` | `cluster_df="per_term"` |
| `sp.bcf` | ensemble approximation, `CausalResult` | `sp.bart_causal(method="bcf")`, `BayesCausalResult` | `sp.BayesianCausalForest(...).fit()` |
| `sp.interference("partial" / "hudgens_halloran")` | `sp.spillover` | `sp.two_stage_interference` | `sp.interference("spillover", ...)` |
| `diagnostics["predicted_*"]` of `sp.zip_model` / `sp.zinb` / `sp.hurdle` | worked, `DeprecationWarning` | removed | `data_info["predicted_*"]` |

- **Critical values.** The rule announced in "1.40.0 → 1.40.1: announced
  for 1.41, t critical values" (below) is the default. Estimates, SEs and
  the pre-trend tests do not change. Castle-doctrine panel
  (`sp.datasets.castle_doctrine()`, outcome `l_homicide`,
  `window=(-5, 4)`, `weights="popwt"`, clustered on 50 states; the 1.41
  numbers equal Stata `reghdfe` / `lincom` / `regress` to 1e-10,
  `tests/reference_parity/test_es_crit_default.py`):

  | | estimate (SE) | 1.40: 95% CI, p | 1.41: 95% CI, p |
  | --- | --- | --- | --- |
  | `sp.event_study`, lead -3 | 0.05072 (0.02550) | [0.00074, 0.10070], p = 0.0467 | [-0.00053, 0.10196], p = 0.0523 |
  | `sp.event_study`, headline ATT | 0.09770 (0.02400) | [0.05067, 0.14474], p = 4.7e-5 | [0.04948, 0.14593], p = 1.7e-4 |
  | `sp.did_2x2`, 2006 cohort, `cluster="sid"` (42 states) | 0.06824 (0.08417) | [-0.09716, 0.23364], p = 0.418 | [-0.10174, 0.23821], p = 0.422 |

  `sp.did(crit_dist="legacy")` still means each estimator's 1.40 rule.
  Library entry points that pinned the 1.40 value (`sp.did_analysis`,
  `sp.did_summary`, `sp.auto_did`, `sp.compare_event_study_conventions`,
  `sp.causal_question`, `CausalWorkflow`, `sp.compare_estimators`) now
  follow the default.
- **Sun-Abraham headline.** `.estimate` / `.se` are the cohort-by-period
  cell-weighted ATT of `fixest::summary(agg = "att")`. On `mpdta`
  (`y="lemp"`, `g="first_treat"`, `t="year"`, `i="countyreal"`) the
  headline moves from -0.0772 to -0.0400 (fixest -0.03995). Both numbers
  stay in `model_info` (`att_event_time`, `att_fixest_att`).
- **Multi-way clustering in `sp.hdfe_ols`.** Every term of
  `V_a + V_b - V_ab` is scaled by G_min/(G_min-1), as Stata `reghdfe` and
  R `fixest` do. SEs change by a few percent when one dimension has few
  clusters (on the weighted two-way fixture of
  `tests/reference_parity/test_panel_absorb_ols_parity.py`, x2: 0.038146 →
  0.038382, reghdfe's). One-way clustering, fits without absorbed effects
  and the IV path are unchanged, and `sp.absorb_ols` keeps `"per_term"`.
- **`sp.bcf`.** It samples the Bayesian causal forest posterior. Pass
  `draws=`, `burnin=`, `chains=` (forwarded to `sp.bart_causal`) instead of
  `n_bootstrap=` / `n_folds=`, which now raise. Read `result.cate` (a
  DataFrame with the posterior mean, SD and interval per row) instead of
  `result.model_info["cate"]`. `sp.bcf_ordinal`, `sp.bcf_factor_exposure`
  and `sp.did_bcf` are unchanged.
- **`sp.interference`.** `"partial"` and `"hudgens_halloran"` need
  `group=` and `saturation=` now. A call with `sp.spillover`'s arguments
  (`cluster=`, `n_bootstrap=`, ...) under those names raises
  `MethodIncompatibility`; call `sp.interference("spillover", ...)`.

---

## 1.40.2 → 1.41: ⚠️ survey / multilevel / gmm option scan

Numbers that change, and how to get the old ones where that makes sense:

- `sp.mixed(group=[outer, inner])`: `random_effects` is indexed by
  `(outer, inner)` and `predict(data)` needs both columns. Predictions with
  class ids that repeat across schools were wrong and are not restorable.
- `sp.mixed` LR test vs. the linear model: p-value is chibar2(01) with one
  variance parameter (half the old value); under REML the statistic now
  compares REML criteria. Old p = 2 × new p for one parameter.
- `sp.xtlsdvc(bootstrap=)`: bootstrap SEs change (the old resampling world
  had no regressors and no unit effects). No option restores them.
- `sp.gmm(vcov='hac')` without `hac_bandwidth`: now `floor(4 (n/100)^(2/9))`
  lags. `hac_bandwidth=1` gives the old (no-lag) numbers.
- `sp.gmm(method='onestep')`: `diagnostics['J_stat']` / `['J_p']` are
  `None`. Fit `method='twostep'` for Hansen's J.
- `sp.meglm(..., trials=)`: `n_obs` is the row count, so BIC changes.
- `MEGLMResult.conf_int()` and the odds-ratio / IRR tables use the fit's
  `alpha`; pass `conf_int(alpha=0.05)` for 95%.
- `subpop=` arrays with missing values: those rows are outside the domain.
- Newly refused (typed errors with a recovery hint): string `subpop`
  columns, missing PSU ids, string outcomes in `sp.meologit`, `trials=`
  outside the binomial family, outcomes outside a family's support,
  `sp.xtabond(cluster=..., robust=False)`, `sp.lrtest` across samples or
  REML / ML, missing values in `sp.svymean` / `sp.svytotal` variables (use
  `subpop=data[col].notna()`).

---

## 1.40.2 → 1.41: ⚠️ postestimation after `irr=True`, unseen factor levels, stateful transforms in new data

- `sp.test` / `sp.lincom` / `sp.nlcom` / `sp.hypotheses` after
  `sp.poisson` / `sp.nbreg` / `sp.xtnbreg` with `irr=True` name and test the
  coefficients `b` (Stata `e(b)`), not `exp(b)`. A hypothesis about the IRR
  is written on `b`: IRR = 1 is `x = 0`, IRR = 2 is `x = ln(2)`. For a
  combination reported as an IRR, take `exp()` of the `lincom` estimate and
  of its interval ends.
- `result.predict(data)` after those `irr=True` fits returns the model's
  predictions (equal to the `irr=False` fit's). The old numbers used
  `exp(b)` as the coefficients and were wrong; there is no option to
  restore them.
- `sp.margins_at`, `sp.margins(at=)`, `sp.contrast`, `sp.pwcompare` and the
  marginaleffects-style functions with `newdata=` raise
  `MethodIncompatibility` for a factor level the fit never saw. Drop or
  recode those rows.
- `sp.predictions(newdata=)` and `result.predict(data)` with `center()`,
  `standardize()`, `bs()` or another stateful transform in the formula
  return the predictions the fitted model makes. The old numbers had the
  transform re-estimated on the new rows; there is no option to restore them.

---

## 1.40.2 → 1.41: ⚠️ `sp.twoway_cluster`, `sp.conley`, `sp.front_door(covariates=)`, `sp.suest(cluster=)`

- `sp.twoway_cluster`: `predict(what='confidence' / 'prediction')` use the
  two-way covariance and the error variance RSS / (n - k); p-values always
  use t(min(G1, G2) - 1), like the intervals. `data_info['df_resid']` is
  n - k again; the inference degrees of freedom are in
  `data_info['df_inference']`. Logit / probit / Poisson / GLM results raise
  `MethodIncompatibility`.
- `sp.conley` after a weighted `sp.regress` returns different standard
  errors: the weights now enter the sandwich. The old SEs were an unweighted
  sandwich around weighted coefficients, which no reference computes.
- `sp.conley` raises `MethodIncompatibility` after logit, probit, Poisson
  and other likelihood fits. Use `sp.fepois(vce='conley')` or
  `sp.ppmlhdfe(vce='conley')` for Poisson models.
- p-values of an `sp.conley` result use t(n - k), as its intervals did,
  whatever the input fit's inference was. `data_info['var_cov']` is the
  Conley matrix.
- `sp.front_door` with `covariates=` returns a different estimate when the
  covariates predict the treatment. Calls without covariates are unchanged.
- `sp.suest(cluster=)` returns different standard errors when `data` has
  rows that no equation uses whose clusters appear nowhere else. The new
  number is the one obtained after dropping those rows, as Stata `suest`.

---

## 1.40.2 → 1.41: ⚠️ `sp.spec_curve`, `sp.robustness_report`, sensitivity breakpoints

- `sp.spec_curve`: `ci_lower` / `ci_upper` are `1 - alpha` intervals (they
  were 95% for every `alpha`). The `nonrobust` / `hc1` specifications keep
  rows whose `cluster_var` is missing. A specification whose regressors are
  linearly dependent is no longer on the curve; it raises an
  `AssumptionWarning` and is listed in `model_info["dropped_specs"]`.
  `n_specs` and `share_significant` can change for such inputs.
- `sp.robustness_report`: the clustered row's p-value and interval use
  t(G - 1).
- `sp.copula_sensitivity` / `sp.survival_sensitivity`: `breakpoint` can be
  set where it was `None` (an interval that crosses zero between grid
  values).
- `sp.robustness_report` / `sp.subgroup_analysis`: formulas with `| fe`,
  `C()`, `I()`, functions or interactions raise `MethodIncompatibility`.
  Create the columns first and name them in the formula.

---

## 1.40.2 → 1.41: ⚠️ survival, epidemiology and Mendelian-randomization statistics (2026-10 option scan)

None of these has an option that restores the old number; each old number
was wrong for the quantity it was labelled as.

- `sp.cohen_kappa` (unweighted): standard errors, CIs and p-values change
  (0.060 → 0.080 on a 40/10/10/40 table; now equal to R `vcd::Kappa`).
- `sp.mantel_haenszel`: the chi-square p-value changes only when the
  exposed-case total is within 0.5 of its expectation (it now goes to 1).
- `sp.epi.incidence_rate_ratio(method="exact")` (the default): the p-value
  is now the exact conditional binomial test (R `poisson.test`). For the old
  Wald p-value pass `method="wald"` (whose interval is the Wald one too).
  A zero event count no longer yields p = 0 or p = 1.
- `sp.aft`: estimates move by up to about 1e-5 relative and standard errors
  by up to about 7e-6, to `sp.survreg`'s exact MLE. A zero duration or an
  event code other than 0/1 now raises.
- `sp.mr_multivariable`: `conditional_f_stats` is NaN unless you pass
  `exposure_se=[...]` (one standard-error column per exposure); with it, it
  is the conditional F of Sanderson, Spiller & Bowden (2021). The old values
  measured the opposite of instrument strength and should not be reused.
- `sp.mr_f_statistic`: `r2_mean` is NaN unless `n_samples=` is given.
- `sp.evalue(measure='RR' | 'OR' | 'HR', se=...)`: `se` is the standard
  error of `log(estimate)`. If you passed a ratio-scale SE, pass the
  interval instead (`ci=(lo, hi)`), which is unchanged.
- `sp.cox(robust='hc1' | 'HC3' | 'cluster' without cluster=)` and
  `sp.mendelian_randomization(methods=[unknown])` now raise; they used to
  fall back silently. `sp.survreg` / `sp.cox_frailty` refuse event codes
  other than 0/1: recode explicitly (`event = (status == k)`).

---

## 1.40.2 → 1.41: ⚠️ bounds, quantile and selection fixes (option scan, 2026-10-27)

Numbers change in these calls; there is no option to restore the old ones,
which were wrong.

- `sp.selection_bounds` equals `sp.lee_bounds(trimming='quantile')`
  (unconditional) or `sp.lee_bounds(covariates=<strata>,
  trimming='quantile')` (conditional). Bounds move by about one
  observation's mass (unconditional) or by the change of cell weights
  (conditional).
- `sp.iv_bounds(assumption='monotone_iv')` returns the Nevo-Rosen set, which
  is a half-line (`lower = -inf` or `upper = inf`) whenever the instrument
  and the treatment are positively correlated after the controls. Code that
  read `lower` / `upper` as finite numbers must handle `inf`;
  `model_info['closed']` says which case applies. State the assumed sign of
  `corr(treatment, error)` with `xu_sign=` (default `'positive'`).
  `model_info['first_stage_f']` is now the first-stage F.
- `sp.iv_bounds(assumption='less_than_late')` is deprecated
  (`DeprecationWarning`) and will be removed no earlier than the next minor
  release. It has no replacement; use `'monotone_iv'`.
- `sp.balke_pearl` monotone bounds follow the direction of the first stage.
  Calls with an instrument that raises take-up are unchanged.
- `sp.qte(method='conditional_qr')` effects are the exact quantile-regression
  coefficients (as `sp.qreg`); they move in the third decimal or so.
- `sp.best_subset` may select no candidate. Check `selected` before indexing.
- `sp.stepwise(criterion='pvalue')` may enter more terms than before (the
  BIC no longer vetoes them). It now refuses `alpha_in > alpha_out`.
- `BoundsResult.ci_lower` / `ci_upper` are centred on the estimated bounds,
  and the interval in `summary()` / `plot()` is Imbens-Manski
  (`parameter_ci()`), narrower than the old two-sided one when the set is
  wide.

---

## 1.40.2 → 1.41: ⚠️ spatial impacts, `SpatialModel`, local Moran, spatial DiD, shift-share alignment

Numbers change in these calls; there is no option to restore the old ones,
which were wrong.

- `sp.impacts(...)`: simulated SEs now use the joint covariance of the
  coefficients and `rho` (as `spatialreg::impacts`), not their variances
  alone. Indirect and total SEs typically shrink (0.428 to 0.197 for the
  total effect on the test fixture). Point impacts are unchanged.
- `sp.SpatialModel(...).fit()`: now identical to `sp.sar` / `sp.sem` /
  `sp.sdm` on the same arguments, standard errors and `alpha` included.
- `sp.moran_local(...)`: `p_sim` and `simulations` come from conditional
  randomisation; the same `seed` gives different replicates than before.
- `sp.spatial_did(...)` with missing outcomes or covariates: the spillover
  exposure now counts every observed treatment. Complete panels are
  unchanged.
- `sp.bartik` / `sp.ssaggregate`: a `shares` frame whose index holds the
  same labels as `data.index` in another order is now aligned by label; a
  partial overlap raises. A labelled `shocks` Series in `sp.ssaggregate` is
  aligned to the share columns. Missing values raise `DataInsufficient`.
- `sp.shift_share_political`: a unit with a missing outcome or regressor in
  its first or last period now raises instead of being differenced over a
  shorter window. Drop or impute it first.
- `sp.slx`: `diagnostics['Log-Likelihood']` is the ML log-likelihood.

---

## 1.40.1 → 1.40.2: ⚠️ `sp.frontier` / `sp.xtfrontier` standard errors

The observed-information SEs were inverted from a central-difference Hessian
with a fixed 1e-5 step, whose round-off was about 1e-5 relative. They now use
a Richardson-extrapolated Hessian, and the optimisation finishes with Newton
steps. SEs move by up to about 2e-5 relative and estimates by up to about
3e-6; both now agree with Stata `frontier` / `xtfrontier` to 1e-8 or better.
There is no option to restore the old numbers, which were round-off.

---

## 1.40.1 → 1.40.2: defaults announced for 1.42 (`sp.iv` GMM weight matrix, `sp.panel` `ssc`)

Nothing changes numerically in this release; calls whose numbers will
change in 1.42 emit a `FutureWarning` that names the argument to pin.

- `sp.iv(method='gmm')` over-identified, with no `wmatrix=`, `cluster=` or
  robust `robust=`: from 1.42 the weight matrix is heteroskedasticity-robust
  (Stata `ivregress gmm`), so the estimate changes. Pass
  `wmatrix='unadjusted'` to keep the current estimator (equal to 2SLS) or
  `wmatrix='robust'` to adopt the new one.
- `sp.panel(...)` with `cluster=`, with `robust='robust'` (fe / twoway / re /
  CRE), or with method re / mundlak / chamberlain: from 1.42 the default is
  `ssc='stata'` (G/(G-1)·(N-1)/(N-K) and t(G-1) under clustering, clustering
  on the panel variable for `robust='robust'` after `xtreg`, z / chi2 after
  RE). Pass `ssc='linearmodels'` to keep the current numbers.

---

## 1.40.1 → 1.40.2: ⚠️ `sp.iv(method='gmm')` with the unadjusted weight matrix and `small=True`

`ivregress gmm ..., wmatrix(unadjusted) small` multiplies the variance by
`N / (N - K)`. `sp.iv(method='gmm')` with `robust='nonrobust'` (the default)
did not, although `small=True` (also the default) reported t / F as if it
had. Standard errors under that combination grow by `sqrt(N / (N - K))`
(0.8% at N = 240, K = 4); estimates and Hansen's J do not change.

- To keep the old standard errors, pass `small=False`: they are Stata's
  `wmatrix(unadjusted) vce(robust)` (default `gmm_vcov`) or
  `wmatrix(unadjusted)` (`gmm_vcov='efficient'`), with z / chi2 inference
  instead of t / F.
- `robust='hc1'` and `cluster=` fits are unchanged.
- `summary()` gains lines that state the variance, the reference
  distribution and the small-sample convention. Code that parses the summary
  text by line position should read `result.inference_rule()` instead.

---

## 1.40.0 → 1.40.1: ⚠️ `sp.did_summary` rows share one estimand; announced `sp.did_multiplegt_dyn` defaults for 1.42

- **`sp.did_summary`.** The Sun-Abraham and stacked rows were the
  equal-weight mean of the post event-time effects, the other rows
  treated-observation ATTs. All rows are now treated-observation ATTs. On the
  267-city personal-bankruptcy panel SA moves from 0.01696 to 0.01215 and
  stacked from 0.01617 to 0.01187. The old numbers are
  `sp.sun_abraham(..., aggregation='event_time')` and `sp.stacked_did(...)`
  (its default `aggregation='event_time'` is unchanged).
- **`sp.bacon_plot(colors=)`** with a partial dict keeps the default colours
  for the types it leaves out.
- **`sp.did_multiplegt_dyn`, announced.** From 1.42 the defaults are
  `aggregation='switchers'` (Stata / R `Av_tot_eff`, switcher-weighted) and
  `se_method='analytic'` (influence-function SEs). Today, without these
  arguments, `.estimate` is the equal-weight mean of the dynamic effects and
  the SEs come from 500 bootstrap draws. The two headlines differ whenever
  later horizons rest on fewer cohorts: 0.0168 against 0.0119 on the
  bankruptcy panel. Pass `aggregation='simple', se_method='bootstrap'` to
  keep today's numbers, or `aggregation='switchers', se_method='analytic'` to
  adopt the 1.42 ones now; either silences the warning.
- **`sp.from_stata` / `sp.stata` with `did_imputation, horizons()`** now pass
  `min_n=30` when `minn()` is left out, so the translated call hides the
  horizons Stata hides. Add `minn(0)` to the Stata line to report them all.

---

## 1.40.0 → 1.40.1: ⚠️ `sp.rlasso_iv(select_Z=True, select_X=False)` solves the second stage exactly

The second stage was solved with a Moore-Penrose pseudo-inverse at
`MASS::ginv`'s tolerance, as `hdm::rlassoIVselectZ` 0.3.2 does. When the
controls are ill-conditioned the pseudo-inverse discards real singular
values and the coefficient is not the IV estimate. On the eminent-domain
application (`EminentDomain$logGDP`) the coefficient moves from 0.2274 to
0.0154 and the SE from 0.1760 to 0.0141; the new numbers equal
`AER::ivreg` with HC0 SEs on the same first stage. Well-conditioned designs
change only by rounding. There is no switch back: the old number depends on
an arbitrary singular-value cutoff (0.0054 at 1e-10).

---

## 1.40.0 → 1.40.1: ⚠️ `sp.did_multiplegt` placebo and dynamic effects with `controls=`; `robust_dynamic=`

Checked against Stata 18 `did_multiplegt_old` and R `DIDmultiplegt` 0.1.4 on
the same CSV (`tests/reference_parity/test_did_multiplegt_dynamic_conventions.py`).

- **`controls=` with `placebo=` or `dynamic=`.** These effects ignored the
  controls (1.40.0 silently, the development line after it with a warning). They
  are now adjusted like DID_M, with slopes fitted on each estimate's own
  control cells, as R does. On the 2026-10-21 test panel (cell means)
  `dynamic_1` moves from 0.59200 to 0.57119 (R 0.57119) and `placebo_1`
  (`placebo_sign='r'`) from -0.16141 to -0.11499 (R -0.11499). Stata's
  placebos reuse the DID_M slopes instead: pass `placebo_controls='main'`
  (-0.11457 there, Stata -0.11457).
- **Dynamic effects with several rows per cell** weigh a cell by its size in
  the switching period, as R does (0.66028 on the individual-level test
  panel, R 0.66028). 1.40.0 used the first row of each cell.
- **Stata's `robust_dynamic` estimators** are new, under
  `robust_dynamic=True`. The default does not change. On treatments that
  switch on and off they are a different estimator (first-time switchers
  against groups not yet switched), and `result.estimate` is then DID_0,
  not DID_M.

To get the old unadjusted placebo or dynamic numbers, drop `controls=`.

---

## 1.40.0 → 1.40.1: `parity=` of `sp.callaway_santanna` off a balanced panel

Only fits with `parity='r'` or `'stata'` on repeated cross-sections or an
unbalanced panel change; default numbers do not.

- `parity='r'` and `'stata'` set `variance_convention='reference'`. With
  `estimator='dr'` and covariates on repeated cross-sections the SEs move
  (up to 15% on the test data) and now equal R `did` and Stata `csdid`.
  An explicit `variance_convention='delta'` together with `parity=` raises.
- `parity='stata'` weights each ATT(g, t) by its treated count in every
  aggregate (calendar, event, and all four on unbalanced panels), as csdid
  does, and reports `estat simple` as the fit's estimate. The aggregates
  that used R's cohort shares before move by up to 5%.
- `parity='r'` accepts `allow_unbalanced_panel=True`; `parity='stata'`
  still refuses it.
- `sp.aggte(agg_weights='csdid')` no longer adds R's cohort-share term to
  the group-average SE; pass `agg_weights='did'` for R's aggregation.
- With sampling weights (`weights=`), `agg_weights='csdid'` (and so
  `parity='stata'` off a balanced panel) weights each cell by the weight
  mass of its treated observations, as csdid `[iw=]` does, instead of
  their head count, and no longer warns. Weighted aggregates move by up
  to 19% (single cells) and 2.6% (overall) on the test data.

---

## 1.40.0 → 1.40.1: ⚠️ `sp.did_multiplegt` with several rows per cell, and with `controls=`

Two correctness fixes, both checked against Stata 18 `did_multiplegt_old`
on the same CSV (`tests/reference_parity/test_did_multiplegt_disaggregated_stata.py`).

- **Several rows per (group, period).** The rows are now averaged to the
  cell and the cell is weighted by its row count, which is what Stata does.
  Before, the main estimate paired rows across periods and the placebo and
  dynamic estimates used the first row of each cell. On the test panel
  DID_M moves from 0.6149 to 0.5600 (Stata 0.5600). Data with one row per
  (group, period) gives the same numbers as before.
- **`controls=`.** The adjustment now uses Stata's regression (stayer cells,
  period effects, one slope per baseline treatment value). On the
  group-level test panel DID_M moves from 0.7015 to 0.6425 (Stata 0.6425).
  Placebo and dynamic effects use the controls too, see the next section.

To reproduce an old number there is no switch: both old results were
computed with rules that no reference implementation uses.

---

## 1.40.0 → 1.40.1: announced for 1.41, t critical values in `sp.event_study`, `sp.sun_abraham`, `sp.did_2x2`

In effect since 1.41 ("1.40.2 → 1.41: announced defaults take effect",
above); "today" below means 1.40.

From 1.41 the three regression-based event-study estimators turn their
standard errors into p-values and confidence intervals the way Stata
(`reghdfe`, `regress`, `eventstudyinteract`) and R `fixest` do: Student t
with `G - 1` degrees of freedom under cluster-robust SEs (`G` clusters) and
t with `N - K` otherwise. Point estimates and standard errors do not change.
The pre-trend tests do not change either (`sp.event_study` already uses the
cluster F(q, G-1) test, `sp.sun_abraham` its Wald chi-squared), and neither
do `sp.uniform_bands`, `sp.honest_did` or `sp.pretrends_test`, which read
the coefficients and their covariance.

Until then the new argument `crit_dist=` picks the rule, and a call that
leaves both `crit_dist` and `parity` unset warns (`FutureWarning`) wherever
its numbers would change:

| function | today (`crit_dist=None`) | from 1.41 | keep today's numbers |
| --- | --- | --- | --- |
| `sp.event_study` | normal | t(G-1) | `crit_dist="normal"` |
| `sp.sun_abraham` | normal | t(G-1); t(N-K) with `i=None` | `crit_dist="normal"` |
| `sp.did_2x2` | t(N-K), also with `cluster=` | t(G-1) with `cluster=`, else t(N-K) | `crit_dist="residual"` |

`crit_dist="t"` adopts the 1.41 rule now. Its critical values are those of
`parity="stata"` / `parity="r"`; for `sp.sun_abraham` it leaves
`share_variance` alone, so `crit_dist="t"` equals `parity="stata"` and
`crit_dist="t", share_variance=False` equals `parity="r"`. `sp.did_2x2`
without `cluster=` already uses t(N-K) and does not warn. A `crit_dist`
other than `"t"` together with `parity=` raises `MethodIncompatibility`.
Callaway-Sant'Anna (`sp.callaway_santanna`, `sp.aggte`) keeps the normal
distribution, as R `did` and Stata `csdid` do.

Castle-doctrine panel (`sp.datasets.castle_doctrine()`, outcome
`l_homicide`, `window=(-5, 4)`, `weights="popwt"`, clustered on 50 states;
the 1.41 numbers are Stata `reghdfe` / `lincom`'s, from
`tests/reference_parity/_fixtures/es_crit_Stata.csv`):

| | estimate (SE) | today: 95% CI, p | 1.41: 95% CI, p |
| --- | --- | --- | --- |
| `sp.event_study`, lead -3 | 0.05072 (0.02550) | [0.00074, 0.10070], p = 0.0467 | [-0.00053, 0.10196], p = 0.0523 |
| `sp.event_study`, headline ATT | 0.09770 (0.02400) | [0.05067, 0.14474], p = 4.7e-5 | [0.04948, 0.14593], p = 1.7e-4 |
| `sp.did_2x2`, 2006 cohort, `cluster="sid"` (42 states) | 0.06824 (0.08417) | [-0.09716, 0.23364], p = 0.418 | [-0.10174, 0.23821], p = 0.422 |

The lead at -3 is significant at 5% under the normal and not under t(49),
as in Stata.

```python
import statspai as sp
r = sp.event_study(df, y="l_homicide", treat_time="effyear", time="year",
                   unit="sid", window=(-5, 4), weights="popwt",
                   crit_dist="t")          # the 1.41 numbers now
r = sp.event_study(..., crit_dist="normal")   # today's numbers, no warning
r.model_info["conventions"]                   # {'parity': None, 'crit_dist': 't', 'crit_df': 49}
```

Library entry points that call these estimators for you (`sp.did_analysis`,
`sp.did_summary`, `sp.auto_did`, `sp.compare_event_study_conventions`,
`sp.causal_question`, `CausalWorkflow.compare_estimators`) pass the
pre-1.41 value and do not warn; they switch with the default.

---

## 1.40.0 → 1.40.1: `sp.drdid(group=)` deprecated in favour of `treat=`; swapped `group=` in five DiD estimators

No number changes.

- `sp.drdid`'s treatment-group parameter is now called `treat`. Calls that
  pass it by keyword as `group=` keep working with a `DeprecationWarning`;
  the keyword will be removed **no earlier than 1.42**. Replace
  `sp.drdid(df, y="y", group="d", time="post")` with
  `sp.drdid(df, y="y", treat="d", time="post")`. Positional calls and the R
  spelling `dname=` are unaffected. Everywhere else in the DiD family
  `group=` is, and stays, the unit id.
- A 0/1 column passed as the unit id (`group=` / `id=`) is caught in
  `sp.gardner_did`, `sp.stacked_did`, `sp.etwfe` (panel), `sp.wooldridge_did`
  and `sp.did_multiplegt`. When the cohort varies within it the call raises
  `MethodIncompatibility` (the first three refused it before, with a less
  specific message; `sp.wooldridge_did` used to return a number). In
  `sp.did_multiplegt` it raises when the treatment varies within a
  (group, period) cell. Otherwise a 0/1 column with several rows per period
  runs and warns with `ArgumentRoleWarning`; pass the unit identifier as
  `id=`.
- New: `sp.did(parity=)`, `sp.did(crit_dist=)` and `sp.cs_report(parity=)`.
  Defaults unchanged. `sp.did(method='2x2' | 'sun_abraham')` without
  `crit_dist=` shows the 1.41 `FutureWarning` of the estimator it runs;
  pass `crit_dist='t'` to adopt the new rule or `crit_dist='legacy'` to
  keep the current one whatever method `sp.did` picks.

---

## 1.39.6 → 1.40.0: `sp.callaway_santanna` drops always-treated units; `parity=`

- Units whose cohort is at or before the first observed period plus
  `anticipation` are now removed before estimation, with a warning, as R
  `did` and Stata `csdid` do (`always_treated='drop'`, the default;
  `'error'` raises). ATT(g, t), every aggregate and every analytic SE are
  unchanged. On a panel that has such units, `model_info['n_units']` and
  `n_obs` fall by their count, the default (Hotelling-F) pre-test p-value
  moves slightly because it counts units, and a seeded multiplier
  bootstrap gives different draws. Panels without such units are
  unaffected.
- `model_info['pretrend_test']['pvalue']` is unchanged by default (Hotelling
  F); the chi2 value that R `did` / csdid print is now alongside it as
  `pvalue_chi2`. `summary()` prints both.
- New: `parity='r' | 'stata'` on `sp.callaway_santanna` and `sp.aggte`.
  Code that reproduced R or Stata by hand
  (`base_period='varying'`, `notyet_cutoff='cohort'`,
  `aggte(..., share_variance=False)` for csdid's group average, a chi2 p-value
  computed from the statistic) can pass `parity=` instead. Defaults are
  unchanged.

---

## 1.39.6 → 1.40.0: swapped `group=` in `sp.did_imputation` / `sp.drdid`; `sp.drdid(seed=, random_state=)`

No number changes. What can now stop or warn a call that used to run:

- `sp.drdid(seed=a, random_state=b)` with `a != b` raises
  `MethodIncompatibility`. Pass only `seed=`. Any of `seed=`,
  `random_state=` or a non-default `n_boot=` now warns that it has no
  effect (the standard errors come from the influence function).
- `sp.drdid` with a two-valued `group` column not coded 0/1 (say 1/2)
  warns with `ArgumentRoleWarning` and still takes the larger value as
  treated. Recode to 0/1 to silence it. A column without exactly two
  values was already refused; the error now names `treat=` and `id=`.
- `sp.did_imputation` with a 0/1 `group` column (the treatment indicator
  where the unit id belongs): when the cohort varies within it the call
  was already refused by the cohort check and the message now names the
  swap; when it does not (one cohort), the call runs and warns with
  `ArgumentRoleWarning`. Pass the unit id as `id=`, or pass `cluster=` if
  two group fixed effects are intended.

The canonical spellings `id=`, `cohort=`, `treat=`, `seed=` are accepted
next to the existing parameter names; nothing was renamed.

---

## 1.39.x → 1.40: ⚠️ three announced default changes take effect

The three defaults that 1.39 announced with a `FutureWarning` or
`DeprecationWarning` change now, and the warnings are gone. Code that named
the argument is unaffected.

- `sp.feols(fixef_rm=)` and `sp.fepois(fixef_rm=)` default to `"singleton"`
  (was `"none"`): observations in a fixed-effect group of one are dropped,
  repeatedly until none is left, as R `fixest`, Stata `reghdfe` and
  pyfixest do. Coefficients do not change; the number of observations and
  the degrees of freedom behind the standard errors do, so standard errors
  with a small-sample factor move slightly. The extended `vce=` menu (wild
  bootstrap, CR2 / CR3, Conley) runs on the rows left after the
  singletons go. Pass `fixef_rm="none"` for the 1.39 numbers. `sp.feglm` has no `fixef_rm` and
  already dropped singletons.
- `sp.bayes_regress` without `prior` uses `prior="weakly_informative"`
  (normal priors scaled to the data, as R `rstanarm`) for the models
  normal, logit, probit, poisson and negbin when `prior_mean` and
  `prior_var` are left at their defaults and `inference="mcmc"`; every
  other call still uses the fixed prior (`"vague"`). Posterior summaries
  change most where a regressor or the intercept is in large units. Pass
  `prior="vague"` for the 1.39 numbers.
- `sp.kaplan_meier` without `conf_type` gives the log-log interval
  (`conf_type="log-log"`, Stata's `sts list`), which never leaves (0, 1).
  `ci_lower` / `ci_upper` change; the survival estimate, its standard error
  and the median do not. Pass `conf_type="plain"` for the 1.39 interval.

---

## 1.39.6 → 1.40.0: `parity=` on the regression event studies; cluster spellings in `sp.feols`

No default number changes.

- `sp.sun_abraham(share_variance=)` now defaults to `None`, meaning `True`
  unless `parity='r'` (then `False`, fixest's convention). Calls that
  pass `share_variance` keep their behaviour; passing a value that
  contradicts `parity=` raises `MethodIncompatibility`.
- `sp.hdfe_ols` with an unrecognised `vce=` raises
  `MethodIncompatibility` instead of a plain `ValueError` (it subclasses
  `ValueError`, so existing `except ValueError` still catches it).
  `vce='cluster firm'` and its variants now mean `cluster='firm'`
  instead of raising.
- `sp.feols` / `sp.fepois` / `sp.feglm`: cluster spellings that used to
  raise pyfixest's `AssertionError` now work, and other rejected `vcov`
  values raise `MethodIncompatibility`. Code that caught
  `AssertionError` should catch `MethodIncompatibility` (or `ValueError`).
- To reproduce Stata's or fixest's p-values and intervals, pass
  `parity='stata'` or `parity='r'` to `sp.event_study`,
  `sp.sun_abraham` or `sp.did_2x2`.

---

## 1.39.6 → 1.40.0: ⚠️ integer columns inside generated formula terms; new `sp.bacon_decomposition` columns; `sp.honest_did` attrs

- A formula term that computes on an integer column (`I(x**3)`,
  `np.power(x, 2)`) is now evaluated in float64. Results change only
  where the integer arithmetic overflowed (e.g. a cube of an `int32` /
  `int64` population in the millions), which used to give a wrong
  coefficient silently. Elsewhere the design matrix is the same up to
  the last bit. Columns used inside `C(...)` stay integers, so level
  names such as `C(g)[T.2]` do not change. To get the old behaviour
  there is no switch; it was a bug.
- `sp.bacon_decomposition` reads float32 outcome / treatment / weight
  columns in float64. On float32 data (a Stata `float`) the 2x2
  estimates and weights move in the seventh significant digit, toward
  Stata's; float64 data give the same numbers as before.
- `sp.bacon_decomposition(...)['decomposition']` has two new columns,
  `comparison_class` and `is_forbidden`, and the `summary` table has
  `comparison_class`. Code that selects columns by name is unaffected;
  code that compares the whole column list should add them.
- `sp.honest_did` tables carry new `attrs` (`simulated_cv`, `seed`,
  `lf_draws`, `next_steps`, and `breakdown_m` for the FLCI). The table
  columns and numbers are unchanged. The FLCI path now also computes the
  breakdown value (about half the time of the table); pass
  `breakdown=False` to skip it.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.rlasso_iv` with `select_Z=True, select_X=False`; neural estimators with `n_bootstrap < 2`

- `sp.rlasso_iv(select_Z=True, select_X=False)` now keeps an intercept in
  the second stage (unless the controls already contain a constant) and
  uses the sandwich of the estimator it computes. Point estimates change
  when `y` or `d` has a nonzero mean and the controls hold no constant;
  standard errors change whenever controls are given and the first stage
  leaves one out (EminentDomain logGDP: SE 0.2466 → 0.1760, coefficient
  unchanged). Both old values were bugs carried over from
  `hdm::rlassoIVselectZ` 0.3.2; there is no switch back. Without
  controls and with every instrument selected, the result is now robust
  2SLS (`sp.ivreg(..., robust='hc0')`).
- `sp.rlasso_iv(select_Z=True, select_X=True)`: the SE changes when no
  control is selected for the instrument (it was inflated); the
  coefficient does not. `intercept=False` is now honoured by the
  `select_X` and double-selection paths.
- `sp.tarnet` / `sp.cfrnet` with `n_bootstrap=0` or `1`: `pvalue` is NaN
  (it was 0.0); `se` and `ci` were already NaN.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.rdrobust` / `sp.rdbwselect` with `cluster=` and `vce='hc2'` / `'hc3'`; `sp.rddensity` not-computable test

- With `cluster=`, `vce='hc2'` and `vce='hc3'` are now the cluster-robust
  CR2 and CR3 variances of R rdrobust 4.0.0 (each cluster's residuals
  times `(I - H_gg)^(-1/2)` or `(I - H_gg)^(-1)`, no small-sample
  factor), and a warning says so. Before, they were HC2 / HC3-scaled
  residuals with the CR1 factor, which no reference computes; on the
  `rd_covs_discriminating` fixture the SEs were 7% (CR2) to 18% (CR3)
  smaller than R's. Because the bandwidth cascade uses the same variance,
  `h` and `b` (and therefore the point estimates) change too. `vce='cr1'`
  / `'cr2'` / `'cr3'` can now be named directly; without `cluster=` they
  fall back to `hc1` / `hc2` / `hc3` with a warning, as in R. Clustered
  results with `nn` / `hc0` / `hc1` (CR1) are unchanged. There is no
  switch back to the old numbers; for the CR1 variance pass `vce='cr1'`.
- `sp.rddensity` now reports `T` and the p-value as NaN (with an
  `AssumptionWarning` and `model_info['diagnostics']['not_computable']`)
  when the test cannot be computed on the bandwidth windows, as R
  `rddensity` reports `NA`. It used to report `T = 0, p = 1`.

---

## 1.39.6 → 1.40.0: ⚠️ DR-DiD standard errors on repeated cross-sections; never-treated cohorts beyond the sample; classical dynamic-panel GMM after FOD / `h=1`

- `sp.drdid(est_method='dr', method='trad')` without `id=` and
  `sp.callaway_santanna(estimator='dr')` with `panel=False` or
  `allow_unbalanced_panel=True` now default to
  `variance_convention='delta'` (Sant'Anna and Zhao 2020, Appendix B).
  Standard errors with covariates change; point estimates do not. To
  reproduce R `DRDID` / `did` or Stata `drdid` / `csdid`, pass
  `variance_convention='reference'`. Under a misspecified propensity score
  the reference SE under-covers (Monte Carlo coverage 0.81 vs 0.95).
- `sp.callaway_santanna`: a cohort first treated after the last observed
  period is now part of the never-treated comparison group, as in R
  `did`. Results change only for data with such cohorts and
  `control_group='nevertreated'`.
- Dynamic-panel GMM with `transform='fod'` or `h=1`, one-step and
  `robust=False`: σ̂² is no longer halved, so classical SEs grow by about
  √2 and the Sargan statistic halves. Classical AR(1)/AR(2) statistics
  after system GMM or orthogonal deviations change to xtabond2's
  formula. Robust and two-step results are unchanged.

## 1.39.6 → 1.40.0: ⚠️ design-mode `sp.rdsampsi` uses the exact power equation

`sp.rdsampsi` without `data=` now returns the smallest sample whose two-sided
power reaches `target_power`. It used the one-tail approximation before, which
gives a larger sample only when the opposite tail carries power (target power
close to `alpha`). Conventional settings (0.8 / 0.05) are unchanged. Invalid
inputs now raise `MethodIncompatibility` (a `ValueError`). Nothing to change in
calling code.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.match(method='radius', caliper_scale='sd')`

Radius matching ignored `caliper_scale='sd'` and used `caliper` as a raw
radius. It now uses `caliper * sd(score)`, as nearest-neighbour matching
does. Code that passed `caliper_scale='sd'` with `method='radius'` gets a
different (correct) estimate; to keep the old number pass
`caliper_scale='raw'`, which is what those calls actually ran.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.grapple(loss='tukey')` standard errors; `sp.mr_raps` robust log-likelihood

- `sp.grapple` with the Tukey loss (its default) now evaluates the whole
  sandwich on one scale of the loss. `tau2_se` and `loglik` are 3.658
  times (c^2/6 with c = 4.685) the previous values, which reproduced
  GRAPPLE 0.2.2 and understated the tau2 SE about 3.7-fold in simulation;
  the beta SE changes by up to about 15%, most on weak-instrument data
  with tau2 = 0. Point estimates are unchanged. There is no switch back.
  Huber and L2 results are unchanged.
- `sp.mr_raps(...).loglik_robust` for the Huber and Tukey losses is now
  `-sum rho(t)` (Zhao et al. 2020, eq. 5.1), twice the previous value.
  Estimates and SEs are unaffected; compare log-likelihoods only within
  one version.

---

## 1.39.6 → 1.40.0: `sp.interference("partial" | "hudgens_halloran")` deprecated as aliases of `spillover`

`sp.interference("partial", ...)` and `sp.interference("hudgens_halloran", ...)`
route to `sp.spillover`, which is not the Hudgens-Halloran (2008) estimator
(it splits units by the treated share of their cluster mates; it takes no
randomized saturation). They now emit a `FutureWarning`; numbers are
unchanged. **In the next minor release both names will route to
`sp.two_stage_interference`**, the Hudgens-Halloran two-stage estimator,
which needs `group=` and `saturation=` columns, so existing calls with those
names will then fail or change meaning.

- To keep the current estimator: `sp.interference("spillover", ...)` or
  `sp.spillover(...)`.
- For Hudgens-Halloran two-stage designs: `sp.interference("two_stage", ...)`
  or `sp.two_stage_interference(...)`.

The dispatcher's default `design` changes from `"partial"` to
`"spillover"` (same estimator), so calls without a design neither warn now
nor change target later.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.rdrobust` / `sp.rdbwselect` bandwidths with mass points

With `masspoints='adjust'` (the default) and 20% or more tied values of the
running variable on a side, the selected `h` and `b` are no longer floored
at the distance of the 10th (`bwcheck`-th) distinct value; only the pilot
bandwidth is, as in rdrobust 4.0.0. Bandwidths, estimates and SEs change on
such data (continuous running variables are unaffected). Pass `h=` / `b=`
to fix the bandwidths yourself. Windows with fewer distinct values than
polynomial coefficients are now refused instead of fitted with a
pseudo-inverse.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.scpi(w_constr='L1-L2')` in-sample bounds

The in-sample bounds of the L1-L2 constraint (and the combined
`subgaussian` / `ls` / `qreg` and `joint` bounds that add to them) now
agree with R `scpi` 4.0.1; they could differ from it by up to about 1e-2 on
short panels. Point estimates and the other constraints are unchanged. No
option restores the old values.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.lee_bounds(trimming='leebounds')` on tiny outcomes; `leebounds_ntall` renamed

With outcomes whose trimming threshold is below 1e-5 in absolute value,
`trimming='leebounds'` now rounds the threshold as Stata stores it (12
significant digits) and so matches Stata `leebounds`; earlier versions
could differ from Stata by up to 2.7% on a bound there. Other inputs are
unchanged. `model_info['leebounds_ntall']` (tightened bounds) is now
`model_info['between_cell_divisor']`; the old key is still filled and will
be removed in a later minor version.

## 1.39.6 → 1.40.0: ⚠️ sup-F p-values (Hansen 1997 table recomputed)

`sp.structural_break(method='sup-f')` p-values (and
`statspai.timeseries.structural_break.hansen_supf_pvalue`) use a
response-surface table that StatsPAI recomputed by simulation from Hansen
(1997) instead of a transcription of his program's table. Expect changes of
up to 0.002 for p <= .10 and 0.008 above; the third decimal of a p-value may
differ from R `strucchange` and Stata `estat sbsingle`. `k` outside 1..40,
non-integer `k`, and a trimming in (0.5, 1) or <= 0 now raise
`MethodIncompatibility`. To reproduce an old number exactly, use
StatsPAI 1.39.6.

## 1.39.6 → 1.40.0: ⚠️ `sp.breakdown_m(method='relative_magnitude')` precision and search

The default (C-LF) relative-magnitude breakdown value is now the first
Mbar at which the set contains zero, found by a log-grid scan and bisected
to 1e-3 relative; it used to bisect a doubling bracket to 1e-3 absolute.
Values move in the third or fourth decimal on nested sets and can move
more where the set is not nested in Mbar. No option restores the old
search.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.sac` standard errors (and estimates)

`sp.sac` now reports observed-information (OIM) standard errors of the
full likelihood, as Stata `spregress, ml dvarlag(W) errorlag(W)`. The old
SEs held rho and lambda fixed for the regression coefficients and were
too small, often by a factor of two or more for the constant and the
spatial parameters. The optimiser now reaches Stata's maximum (log-likelihoods equal to 1e-12), so
estimates can move in the fourth decimal. There is no switch back; the
old SEs were not a valid covariance.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.local_projections` state-dependent options, redundant instruments

- `sp.local_projections(state=...)` now applies `dfk=` and `common_sample=`
  (observed shock) and `vce=` / `dfk=` (with
  `identification='lpirfs_cholesky'`); they were silently ignored, so
  standard errors and, with `common_sample=True`, estimates change for
  calls that passed them. Calls without them are unchanged.
- An excluded instrument that duplicates a control or lag is dropped with an
  `AssumptionWarning`; the response equals the fit without it.
- Arguments of another design (`gamma=` without `state=`, `diff_shock=`
  without `id=` / `time=`, `dfk=` / `trend=` / `common_sample=` on a panel,
  `nw_lags=` on a panel unless `vce='driscoll_kraay'`) now raise
  `MethodIncompatibility` instead of being ignored. Drop them.

---

## 1.39.6 → 1.40.0: default changes and announced deprecations (owner decisions 2026-10-20)

None of these changes a number in this release, except the `'rlasso'`
classifier alias in the next section. The full record is
`docs/dev/2026-10-20-owner-decisions.md`.

- `sp.causal_forest(n_jobs=)` now defaults to `-1` (all cores) instead of `1`; estimates are byte-identical for any thread count.
  `n_jobs` is now an explicit parameter of `sp.causal_forest` (it used to
  travel through `**kwargs`), and `sp.CausalForest(n_jobs=)` has the same
  default. Pass `n_jobs=1` to keep a single thread.
- `sp.hdfe_ols` with two or more cluster variables and no `cluster_df=`
  emits a `FutureWarning`: the default small-sample factor of a multi-way
  cluster variance will change from `'per_term'` (each inclusion-exclusion
  term scaled by its own `G/(G-1)`, R `sandwich::vcovCL`) to `'min'`
  (`G_min/(G_min-1)` on every term, Stata `reghdfe` and R `fixest`) in a
  later minor release. Pass `cluster_df='per_term'` to keep today's
  standard errors, or `cluster_df='min'` to adopt the new ones now.
  One-way clustering is not affected.
- `sp.sun_abraham` without `aggregation=` emits a `FutureWarning`: the
  headline `.estimate` / `.se` will change in a later minor release from
  the equal-weight average of the post-treatment relative-time effects
  (`aggregation='event_time'`, today's default) to the cohort-by-period
  cell-weighted ATT that `fixest::summary(agg = "att")` reports
  (`aggregation='fixest_att'`). Pass `aggregation='event_time'` to keep
  today's headline, or `'fixest_att'` to adopt the new one now. The
  event-time coefficients do not depend on it, and both numbers are in
  `model_info` either way.
- `sp.bcf` emits a `FutureWarning`: in a later minor release it will route
  to `sp.bart_causal(method='bcf')`, so its result type becomes that of
  `sp.bart_causal`. Call `sp.bart_causal(method='bcf')` directly to adopt it
  now; the numbers of `sp.bcf` are unchanged in this release.
- `matplotlib>=3.5` is a core dependency (still imported only when a plot
  is drawn). `pip install statspai[plotting]` remains the way to get
  seaborn and plotly.

---

## 1.39.6 → 1.40.0: ⚠️ `ml_m='rlasso'` in a classifier role is the logistic rigorous Lasso

`sp.dml` (`irm` / `iivm` propensities, `ml_r`; any classifier role),
`sp.dml_did(ml_m=)` and `sp.dml_panel` (binary treatment) resolve the string `'rlasso'` (and
`'rigorous_lasso'`) to `sp.RlassologitClassifier` when the role is a
classifier, which is what `hdm` fits for propensities (`rlassologit`). It
used to be `sp.RlassoClassifier`, a linear-probability rigorous Lasso whose
predictions are clipped to `[clip, 1 - clip]`; estimates with that alias
change. Regressor roles (`ml_g`, a continuous `ml_m`) keep
`sp.RlassoRegressor`. For the old propensities pass
`ml_m=sp.RlassoClassifier()` or the new alias `ml_m='rlasso_lpm'`.

---

## 1.39.6 → 1.40.0: ⚠️ classifier learners in `sp.dynamic_dml`; rigorous-Lasso learners refuse `sample_weight`

- `sp.dynamic_dml(model_t=<classifier>)` (or `model_y=`) now residualises
  on the classifier's P(T = 1) instead of its hard 0/1 label; estimates
  with a classifier change. A classifier on a target that is not 0/1 is
  refused. Regressors, including the default `RidgeCV`, are unchanged.
- `sp.RlassoRegressor` / `sp.RlassoClassifier` / `sp.RlassologitClassifier`
  `.fit(..., sample_weight=w)` raises `TypeError` (the weights used to be
  ignored). Weighted `sp.dml` fits fall back to an unweighted nuisance with
  a warning, as for any learner without weights; in `sp.ipw` / `sp.aipw`
  with `weights=` choose a learner whose `fit` takes `sample_weight`.
- `sp.RlassoClassifier.predict_proba` warns when it clips. It is a
  linear-probability model; for an inverse-weighted propensity use
  `sp.RlassologitClassifier`.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.wild_cluster_boot(bootcluster=)` across one-way clusters

With one-way `cluster=` and a `bootcluster=` that cuts across it, the
bootstrap weights are drawn for the intersections of the two (Stata
`boottest`'s behaviour) instead of for `bootcluster` alone; p-values change.
A `bootcluster` coarser than the error clusters now raises
`MethodIncompatibility` (boottest fails there too). To get the old draws,
there is no switch: they did not correspond to a defined bootstrap.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.binscatter(method='binsreg')` on a discrete `x`

With `n_bins` left to the selector and an `x` that takes at most about
30 + (rule-of-thumb bins) distinct values, each value is now its own bin
(Stata / R binsreg's behaviour); the dots, `n_bins` and intervals change,
often from a single collapsed bin. Pass `n_bins=` to keep binning the
values. Continuous `x` and explicit `n_bins` are unchanged.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.xtivreg(method='re')` on balanced panels with time-invariant variables

On a balanced panel, `sp.xtivreg(method='re')` now follows Stata 18's
`xtivreg, re` where a regressor or an excluded instrument does not vary
within panels: the degrees of freedom behind `sigma_e` count only the
regressors the within regression uses, and the between regression behind
`sigma_u` leaves out the excluded instruments that are constant within
panels. `sigma_e`, `sigma_u`, theta, the coefficients and their SEs change
(by about 0.1% for `sigma_e` with one time-invariant regressor). Unbalanced
panels, and balanced ones where every variable varies within panels, are
unchanged. With no excluded instrument varying within panels the fit used
to raise; it now runs as Stata does, with a warning.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.honest_did` smoothness test-inverted sets no longer empty for large effects

`sp.honest_did(method='smoothness', honestdid_method='Conditional' or
'C-LF')`, with or without `bias_direction` / `monotonicity_direction`,
returned `NaN` bounds and `rejects_zero=False` when the estimate was far
from zero relative to its standard error: the default grid (the identified
set at `beta = 0` widened by 20 standard deviations) did not reach it. Such
rows now carry the confidence set computed on the same span centred at the
estimate. Rows that already had finite bounds are unchanged, and a grid you
pass through `grid_lb` / `grid_ub` is used as given. R `HonestDiD` still
returns `NA` for these rows.

---

## 1.39.6 → 1.40.0: ⚠️ ordered Categorical outcomes, spatial z intervals, `dose_response(method='dr')` defaults

- `sp.ologit` / `sp.oprobit` / `sp.gologit` read an ordered `pd.Categorical`
  outcome in its declared order (they sorted the labels alphabetically).
  Integer-coded outcomes are unchanged.
- The spatial maximum-likelihood fits (`sar`, `sem`, `sdm`, `sac`, `sdem`,
  `gns`) report z p-values and normal intervals, as Stata `spregress, ml`;
  estimates and standard errors are unchanged. There is no switch back.
- `sp.dose_response(method='dr')` without `treatment_model=` now uses the
  documented linear treatment model (it received gradient boosting) and
  fits the variance on cross-fitted residuals; curves from default calls
  change. Pass `treatment_model=` to choose another learner.
- `sp.msprt`: the default `min_per_arm` is 10 with the plug-in variance
  (it was 2, which broke the level); pass `min_per_arm=2` for the old
  sequence of looks.
- `sp.augsynth(progfunc='none', scm=False)` is SCM, as in R augsynth.
- `sp.lee_bounds(weights=)` drops zero-weight rows (SEs and `n_obs` change
  when there are any).

---

## 1.39.6 → 1.40.0: `sp.margins` after `fepois` / `feglm` needs `vcov=False`

After a fit with absorbed fixed effects and a non-linear link
(`sp.fepois`, `sp.feglm`), and for predictive margins after `sp.feols`,
`sp.margins` / `sp.margins_at` / `sp.contrast` / `sp.pwcompare` now raise
instead of reporting standard errors that hold the estimated effects fixed
(they did not match Stata's `margins`). Pass `vcov=False` to `margins` /
`margins_at` for the estimates, which are unchanged, or fit the effects as
`C(g)` terms to get Stata's standard errors.

Update (same release): after a Poisson fit (`sp.fepois`,
`sp.feglm(family='poisson')`) averages no longer raise. Their standard
errors count the effects as parameters, as Stata's `margins` after
`poisson y x i.g` / `ppmlhdfe ..., d`, with the fit's own small-sample
factor. Logit / probit with absorbed effects, OLS predictive margins,
`method='mem'`, unit-level tables and two-way clustering still raise.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.panel(vce='wild')` re-absorbs entity effects not nested in the clusters

With `method='fe', vce='wild'` and a `cluster=` column that does not nest
the entities (each entity in one cluster), the bootstrap now re-absorbs the
entity effects in every draw, as Stata `boottest` after `areg` does. The
wild p-values and confidence sets change for such calls (on the test panel
an 8-cluster design went from p = 0.0625 to boottest's 0.078); point
estimates and the CR1 standard errors do not. Clustering on the entity (the
default) is unchanged. There is no switch back: the old numbers came from a
bootstrap that is not valid for non-nested effects.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.feols` / `sp.hdfe_ols` finish unconverged projections exactly

When the alternating projections stopped at their sweep limit, the native
`sp.feols` and `sp.hdfe_ols` used to report the unconverged slope silently.
Those columns are now residualised by a sparse direct solve, so such fits
change (to the dummy-variable regression's numbers); converged fits do not.
`model_info['fe_solver']` (`sp.feols`) and `result.absorber.fe_solver`
(`sp.hdfe_ols`) say when it happened. There is no switch back: the old
numbers were wrong. `sp.read_data` on a large CSV now parses decimals to
the nearest double (pyarrow), which can move the last bit of a value that
pandas' default parser rounded the other way.
## 1.39.6 → 1.40.0: ⚠️ `sp.mi_estimate` degrees of freedom for clustered and likelihood fits

`sp.mi_estimate` now takes the complete-data df of the Barnard-Rubin
correction from the fit's own t reference, as Stata `mi estimate` does:
`G - 1` for a clustered `sp.regress`, and the large-sample df for z-based
fits such as `sp.logit` / `sp.poisson`. Estimates and variances are the
same; the df, p-values and confidence intervals of those fits change
(clustered fits get wider intervals). Unclustered `sp.regress` fits are
unchanged. There is no switch back: the old df were not those of any
reference implementation.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.lee_bounds` drops rows with missing indicators

`sp.lee_bounds` now drops rows whose treatment or selection indicator is
missing, and selected rows whose outcome is missing, as Stata `leebounds`
does. Calls on data without such rows are unchanged. With a missing
treatment the analytic standard errors change (they used to count those
rows); with a selected row lacking its outcome the bounds were NaN and are
now numbers. To keep the old behaviour there is nothing to pass; drop or
impute the rows yourself to control the sample.

---

## 1.39.6 → 1.40.0: ⚠️ translated two-way clustered `reghdfe` uses reghdfe's factor

`sp.from_stata('reghdfe y x, absorb(id) cluster(a b)')` now emits
`sp.hdfe_ols(..., cluster=['a', 'b'], cluster_df='min')`, so the standard
errors `sp.stata` reports for such a line equal reghdfe's. To get the old
per-term numbers, call `sp.hdfe_ols` directly without `cluster_df=`.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.augsynth(covariates=...)` now uses the covariates

`sp.augsynth` ignored `covariates=` on the native path. It now balances
them as `augsynth::augsynth` does by default (pre-period means appended to
the outcome columns after rescaling), so estimates and weights change for
every call that passed `covariates=`. To reproduce the old number, drop the
argument. Calls without covariates are unchanged.

---

## 1.39.6 → 1.40.0: ⚠️ treatment-effect estimators refuse non-binary treatments; `sp.sensemakr` / `sp.bcf` / `sp.optimal_match`

- `sp.cbps`, `sp.ipw`, `sp.aipw`, `sp.tmle`, `sp.ebalance`, `sp.optimal_match`,
  `sp.cardinality_match` and `sp.rosenbaum_bounds` raise
  `MethodIncompatibility` (a `ValueError`) for a treatment not coded 0/1.
  `sp.cbps` used to return a meaningless number there. Recode the treated arm
  to 1 and the control arm to 0.
- `sp.sensemakr` returns `SensemakrResult`, a `dict` subclass: existing
  `result['rv_q']` code is unchanged; `type(result) is dict` is now `False`.
- `sp.bcf`'s `result.method` reads "BCF-style approximation (RF + GBM; not
  BART)" and every call issues `BCFApproximationWarning`; the numbers are
  unchanged. Silence it with
  `warnings.filterwarnings("ignore", category=statspai.bcf.bcf.BCFApproximationWarning)`.
- `sp.optimal_match` with more treated than controls returns a result and
  warns instead of raising `ValueError`; check `n_unmatched_treated`.
- `sp.match(..., unknown_keyword=...)` raises `MethodIncompatibility`, which
  is still a `TypeError`.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.causal_impact` standard errors are wider

**What changed.** The standard error, confidence interval and p-value of
`sp.causal_impact` now include the estimation of the regression and of the
error process (REML AR(1)-plus-noise model, t quantile). They are wider,
most of all when the pre-period is short and the error persistent. Point
estimates are unchanged.

**What to do.** Nothing. The old (conditional) values are in
`model_info['se_avg_conditional']` and `model_info['se_total_conditional']`.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.ges` returns the GES equivalence class

**What changed.** `sp.ges` runs Greedy Equivalence Search over CPDAGs
(Chickering 2002). Edges that the old version reported as undirected but
that every DAG in the class orients the same way are now directed, and on
some data the learned skeleton itself changes (towards `pcalg::ges`).

**What to do.** Nothing; `.adjacency`, `.edges()` and `.bic` keep their
meaning. Results from earlier versions should be re-run.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.msm` weights are not truncated by default

**What changed.** `sp.msm(trim=)` defaults to `0.0` (no truncation) instead
of `0.01`, matching R `ipw::ipwtm`. Estimates and standard errors of calls
that relied on the default change; calls that passed `trim=` explicitly do
not.

**What to do.** Pass `trim=0.01` to reproduce an old number. Treat a
truncated fit as a sensitivity check: `model_info['truncation']` reports
how far truncation moved the estimate.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.did_imputation` lead-horizon covariance and lead SEs

**What changed.** With BJS pre-trend leads, `model_info['event_study_vcov']`
now has the lead-horizon cross-covariance (it was zero and flagged
`block_diagonal`), as Stata `did_imputation`'s `e(V)` does. `sp.honest_did`
and `sp.uniform_bands` on these fits therefore change: the first from the
warned worst-case-bias interval to the FLCI / ARP set, the second from a
Sidak band to the sup-t band of the joint matrix. Lead standard errors
change only where the auxiliary regression has singleton units (a unit
with one untreated period): their dof count now drops them, as `reghdfe`
does (0.1% on mpdta with `pretrends(3)`).

**What to do.** Nothing. There is no switch back to the block-diagonal
matrix; its zero cross block was not an estimate of anything.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.rdrobust` fuzzy + covariates bandwidth

**What changed.** With `fuzzy=` and `covs=` and no `h=`, the selected
bandwidth now matches R / Stata `rdrobust` (the covariates are no longer
demeaned before selection). On covariates whose means are far from zero
`h` moves by a few parts in a thousand and the estimate and standard
errors move with it. Sharp designs, fuzzy designs without covariates and
any call with a fixed `h` are unchanged.

**What to do.** Nothing; to reproduce an old number pass the old `h=`
explicitly.

---

## 1.39.6 → 1.40.0: ⚠️ staggered DiD input checks (`sun_abraham`, `etwfe`, `stacked_did`, ...)

**What changed.** (1) A cohort column that varies within a unit is now an
error in `callaway_santanna`, `sun_abraham`, `did_imputation`,
`gardner_did`, `stacked_did` and `etwfe`; before, each estimator resolved
it differently and silently. (2) `sun_abraham`, `etwfe` and `stacked_did`
drop cohorts treated in or before the first period of the panel (no
untreated row), with a warning. `sun_abraham` previously returned zero
standard errors for such panels; `etwfe` and `stacked_did` returned
estimates contaminated by the unidentified cohort. On panels without such
a cohort nothing changes. (3) The regression estimators warn on repeated
(unit, period) rows and on fewer than ten treated clusters.

**What to do.** Recompute the cohort as the first treated period per unit
(0 or NaN for never treated). To reproduce an old `etwfe` / `stacked_did`
number on a panel with an always-treated cohort there is no switch: the
old number depended on which collinear cell the solver dropped. If the
repeated rows are accidental copies, drop them; if `unit` is a group with
several individuals per period, the warning can be ignored
(`warnings.filterwarnings("ignore", category=RepeatedUnitPeriodWarning)`,
from `statspai.did._validate`).

---

## 1.39.6 → 1.40.0: ⚠️ `[pweight=]` in `sp.from_stata` / `sp.stata`

**What changed.** A Stata command with `[pweight=]` and no `vce()` is
translated as if `vce(robust)` were given, as Stata fits it (`xtreg, fe`:
clustered on the panel variable). Before, `reghdfe`, `ivreghdfe`,
`ivregress`, `ivreg2`, `xtreg, fe` and `glm` kept the non-robust variance,
so their standard errors differed from Stata's (7% for the x1 SE of a
`reghdfe` example, 21% for `xtreg, fe`). Weighted robust `reghdfe` /
`ivreghdfe` now emit `sp.feols(..., vcov='hetero', fixef_rm='singleton')`.

**What to do.** Nothing, unless you relied on the old standard errors:
re-run the line to get Stata's. Payloads also carry `dropped` and
`complete`; the old keys are unchanged.

---

## 1.39.6 → 1.40.0: ⚠️ predictions, marginal effects and wild bootstraps after fixed-effects, IV and GLM fits

**What changed.**

- `predict(newdata)` after `sp.feols` / `sp.fepois` / `sp.feglm` with
  `| fe` adds the estimated fixed effects (and, for `fepois` / `feglm`,
  applies the inverse link). Before it returned `x'b` alone.
- `sp.avg_slopes` / `sp.avg_predictions` / `sp.avg_comparisons` /
  `sp.margins` after `sp.fepois` / `sp.feglm` return marginal effects
  on the response scale with the effects in the index; before they
  returned the coefficients. The marginaleffects-style functions now
  raise for standard errors that depend on the effects (predictions, and
  any quantity of a Poisson / logit / probit fit); pass `vcov=False`.
- `sp.wild_cluster_boot` runs the bootstrap of the fitted estimator (WRE
  after `sp.ivreg`, score bootstrap after `sp.logit` / `sp.probit` /
  `sp.poisson`), refuses models without one, and residualises draws on
  absorbed effects that are not nested in the clusters. Its p-values
  after such fits change; after plain least squares they do not.
- `sp.feols(..., vce='wild')` with an IV part: WRE p-values on the
  endogenous coefficients, CRV1 standard errors kept.
- `tab.attrs["jacobian"]` / `["vcov_coefficients"]` are read-only
  array wrappers; use `np.asarray(...)` where an ndarray is needed (`@`,
  indexing and `.T` work as before).

**What to do.** Nothing, unless you relied on the old numbers. To keep
coefficient-scale numbers after `sp.fepois`, read `result.params`.

---

## 1.39.6 → 1.40: `sp.feols` / `sp.fepois` will drop fixed-effect singletons by default

**Done in 1.40.** The default is now `fixef_rm="singleton"` and the
`FutureWarning` is gone; pass `fixef_rm="none"` to keep singletons. The
text below is the 1.39.6 announcement.

**What changes.** `fixef_rm` of `sp.feols` and `sp.fepois` defaults to
`None` (was `"none"`), which still keeps observations in fixed-effect
groups of one; when a fit kept some, a `FutureWarning` says so. From
StatsPAI 1.40 the default becomes `"singleton"`, which drops them, as R
`fixest`, Stata `reghdfe` and pyfixest do by default. Coefficients do not
change; the number of observations and the degrees of freedom behind the
standard errors do.

**What to do.** Pass `fixef_rm="singleton"` to adopt the new default now
(results then match fixest's defaults), or `fixef_rm="none"` to keep
singletons. Either silences the warning.

---

## 1.39.6 → 1.40.0: `sp.feols` / `sp.fepois` / `sp.feglm` fit on StatsPAI's own engine

**What changed.** `sp.feols(backend=)` defaults to `'auto'` (was
`'pyfixest'`), and `sp.fepois` / `sp.feglm` gained the same argument. With `'auto'`, a
call of the form `y ~ x1 + ... | fe1 + ...` with numeric regressors (and,
for `sp.feols`, `i(factor, ref=v)` terms, two-way clustering and the IV
form `y ~ x | fe | d ~ z`), `vcov` in `'iid'` / `'hetero'` /
`{'CRV1': col}`, no `ssc=` and no extra pyfixest keyword runs on
`sp.fast.feols` / `sp.fast.fepois` (the Rust kernels when `statspai-hdfe`
is installed); everything else still goes to pyfixest. A native IV fit
reports fixest's first-stage Wald F (`ivwald1`); pyfixest's robust or
clustered value differed when the panel had singletons, because its first
stage dropped them. Two-way clustered `sp.fepois` / `sp.feglm` standard
errors now include fixest's eigenvalue fix, which pyfixest skipped for
GLMs; where it applies they move to fixest's values. The conventions are the delegate's (singletons kept unless
`fixef_rm='singleton'`, rows with missing values dropped, the same degrees
of freedom and R-squared definitions), so coefficients agree to about
1e-10 and standard errors to the demeaning tolerance. `sp.fepois` iterates
IRLS one Newton step further than pyfixest's stop, which moves its
standard errors by up to about 1e-5 relative, towards the exact optimum.
`sp.feglm(family='logit' | 'probit')` follows `fixest::feglm`: standard
errors move by up to about 1e-4 relative from pyfixest's (towards fixest's)
and p-values come from the normal distribution instead of a t.
`model_info['backend']` is `'statspai-native'` for these fits, and the
result has no `_pyfixest_fit`. A plain `pip install statspai` (without the
`fixest` extra) can now run these calls.

**What to do.** Nothing for the numbers. Code that reads `_pyfixest_fit`,
or needs pyfixest's own fit for another reason, passes
`backend='pyfixest'`.

---

## 1.39.6 → 1.40.0: ⚠️ `sp.panel` within / between / overall R-squared

**What changed.** After `method="fe"`, `"re"` and `"be"` (unweighted),
`diagnostics["R-squared (within)"]` and `["R-squared (between)"]` are
Stata `xtreg`'s squared correlations, and `["R-squared (overall)"]` is
new. They used to be linearmodels' 1 - SSR/SST values (a fixed-effects
between R-squared of 0.47 where `xtreg, fe` prints 0.87).

**What to do.** Nothing if you compare with Stata. For the old numbers,
read `result.model_info["r2_linearmodels"]`.

## 1.39.6 → 1.40.0: ⚠️ mixed-model standard errors in the eighth digit

**What changed.** `sp.melogit` / `meglm` / `mepoisson` / `meologit` and
`sp.icc` take the observed information from a Richardson-extrapolated
numerical Hessian. Standard errors move by about 1e-8 to 5e-8 relative
(ICC standard errors by up to 1e-6, towards Stata's); point estimates do
not move.

**What to do.** Nothing, unless a test pins mixed-model standard errors
beyond seven significant digits.

---

## 1.39.6 → 1.40.0: ⚠️ zero-inflated / hurdle NB at the dispersion boundary

**What changed.** When the NB2 dispersion estimate of `sp.zinb` or
`sp.hurdle(count_model="negbin")` sits at alpha = 0 (overdispersion
`alpha * mean(mu)` below 1e-4), the standard error of `ln_alpha` is NaN on
every platform (it used to be NaN on Linux and macOS and a finite number
on Windows), and the other standard errors hold alpha at its estimate.
They now equal the zero-inflated / hurdle Poisson standard errors; before
they came from a near-singular inverse and could differ in the fourth
digit. `model_info["dispersion_at_boundary"]` says when this applies.

**What to do.** Nothing for fits with real overdispersion. At the boundary
the data show no overdispersion; `sp.zip_model` / `sp.hurdle(count_model=
"poisson")` is the model the estimates describe.

---

## 1.39.6 → 1.40.0: ⚠️ E-values, honest DiD relative magnitudes, spatial intervals

**What changed.**

| Call | Before | Now |
| --- | --- | --- |
| `sp.evalue_from_result(result)` / `measure='SMD'` or `'MD'` | raw estimate read as a standardised difference (E-value moves with the outcome's units) | needs `sd=` (the outcome's SD; `sd=1` if already standardised) and divides by it; refuses without it |
| `sp.honest_did(method='relative_magnitude')` without `m_grid` | M-bar grid in SE units (0 to 3 SE) | HonestDiD's `Mbarvec`: ten points from 0 to 2 |
| `sp.sar` / `sem` / `sdm` / `slx` / `sac` with `alpha=` | ignored, 95% intervals | intervals at the requested level |
| `next_steps()` items | `action` / `reason` / `priority` / `category` | also `executable`; DiD steps name `sp.honest_did` / `sp.breakdown_m` instead of `sp.sensitivity_rr` |

**What to do.** Pass `sd=df[y].std()` to `sp.evalue_from_result` for a
mean difference. Re-run relative-magnitude sensitivity tables that
relied on the default grid; pass `m_grid=` to keep a specific grid.

---

## 1.39.6 → 1.40.0: figure editor (`sp.interactive`)

**What changed.** The editor was rebuilt (CHANGELOG, "Figure editor").
Figures themselves are unchanged until edited; what differs is the editor's
output.

| Behaviour | Before | Now |
| --- | --- | --- |
| `generate_code()` | bare `ax.` lines plus `fig.tight_layout()`; fonts and themes as `mpl.rcParams[...]` / `sp.set_theme(...)` | starts with `ax = fig.axes[0]`; property-level statements that reproduce the figure; no `tight_layout` unless requested |
| `apply_theme`, `set_font`, `apply_font_preset` | changed global rcParams (all later figures) | change this figure only; use `sp.set_theme` for later figures |
| `set_ylim` / `set_xlim` near error bars or bands | could clip them | widened to keep them, with a warning |
| `set_alpha(..., 0)` on data | allowed | 0.2 floor (0.05 for intervals), with a warning |
| `sp.interactive(fig)` on Agg / headless | added button axes to `fig` | returns the editor, figure untouched |
| `editor.save(...)` | Type 3 fonts in PDF | TrueType embedded; `embed_fonts=False` for the old output |
| `editor.save(...)` | recorded as an edit (undo removed it first) | not an edit |
| `apply_actions` with an unknown parameter | ignored | `ValueError` |
| `to_edits_list()` records | `target` / `property` / `old` / `new` / `code` | adds `op` and `params`; replay uses those, never `code` |

**What to do.** Regenerate code saved from the old editor if you rely on
it. Saved edit lists (version 1) still load. If a script relied on the
editor to change global fonts, call `sp.set_theme(...)` or set
`mpl.rcParams` yourself.

---

## 1.39.6 → 1.40.0: ⚠️ options off the default path in DiD, matching and IV

**What changed.** Comparing every option value of the core estimators
with R and Stata found defects in non-default options. Default calls on
the main path return what they did. The calls below return different
numbers or now refuse; recompute them.

| Call | Before | Now |
| --- | --- | --- |
| `sp.etwfe(xvar=...)`, linear model | pooled cohort × post fit with unit effects; same number under both `cgroup`; categorical moderator as continuous | R `etwfe`'s moderated design (1.4957 → 1.5691 not-yet / 1.5248 never on the option fixture) |
| `sp.event_study` on an unbalanced panel or with weights varying within a unit | one pass of unit and time means (up to 0.035 / 0.15 off `fixest`) | exact two-way projection; balanced panels unchanged |
| `sp.did_imputation(fe=...)` with a treated fixed-effect level that has no untreated row | imputed at the reference level, no warning | raises, as Stata; `autosample=True` drops those rows |
| `sp.match(bias_correction=True)` | unweighted regression on all controls | matched controls weighted by use (= `Matching`) |
| `sp.match(caliper=...)`, psmatch2 / Abadie-Imbens SE | counted unmatched treated units | excludes them (= Stata `psmatch2`) |
| `sp.match(se_method='abadie_imbens_2016', caliper=...)` where a unit lacks max(m, h) / h matches in the caliper | a standard error | raises, as `teffects psmatch` |
| `sp.match` / `sp.psm` default SE with tied scores (`ties='all'`) | same-arm neighbours in row order (Lalonde 1181.665, moves with row order) | every tied neighbour, = `psmatch2 , ai(1) ties` (1178.209 in any order) |
| `sp.callaway_santanna(panel=False, estimator='stdipw' \| 'ipw_abadie')` | unconditional cell-mean DiD | DRDID repeated-cross-section estimators (= R `did`) |
| `sp.callaway_santanna(panel=False, bstrap=True)` headline / event study / `sp.aggte` | analytic SE | multiplier bootstrap |
| `sp.honest_did(method='smoothness')` FLCI at `M = 0` | numerical centre (up to 1.4e-3 of the half-length off) | closed form |
| `sp.honest_did` with no pre-treatment coefficient | θ ± z·SE fallback | raises `DataInsufficient` |
| `sp.spec_curve(se_types=[...])` with an unknown name, or `'cluster'` without `cluster_var` | HC1 under the unknown label; cluster dropped | raises; `'classical'` / `'iid'` are the nonrobust variance |
| `sp.did_multiplegt_dyn(continuous=...)` | polynomial residualised separately; matched only when every group switched | the reference design (= `DIDmultiplegtDYN` / Stata) |
| `sp.ivreg(vce='conley')`, `sp.ivreg(cluster=[a, b])` with a negative variance | SE 0, t = inf, p = 0 | NaN with a warning (= `acreg` / `ivreg2`) |

**What to do.** Re-run analyses that use these options. `sp.feols`,
`sp.fepois` and `sp.feglm` now request pyfixest's compiled demeaner;
pass `demeaner_backend='numba'` to keep the old one (results agree to
pyfixest's `fixef_tol`).

---

## 1.39.5 → 1.39.6: ⚠️ `sp.stata` results that now match Stata

**What changed.** A snippet-by-snippet comparison with Stata 18 found
places where `sp.stata` ran and printed a different number. They now
print Stata's.

| In a do-file run through `sp.stata` | Before | Now |
| --- | --- | --- |
| `collapse ... [fw=w]` | unweighted | frequency-weighted |
| `logit`, `probit`, `poisson`, `nbreg`, `cloglog`, `ologit`, `oprobit` with `[pw=]` | robust SE with N/(N−k) | N/(N−1), as Stata |
| `margins ..., post`, then `_b[]` / `test` / `lincom` | read the model | read the margins |
| `margins, predict(xb)` | mean probability | linear index |
| `round(x)` for negative halves, `round(x, 0)` | away from zero; missing | Stata's `y*floor(x/y + .5)`; x |
| `string()`, `display %fmt`, numbers stored in macros | Python formatting | Stata's `%g` / `%f` / `%e` |
| `e(df_r)` after `vce(cluster)` | N − k | G − 1 (unset after z-inference fits, as in Stata) |
| `gsort -x` | missing first | missing last (`mfirst` for first) |
| `gen byte` / `int` / `long` out of range, `gen strN` | value kept | missing; truncated to N bytes |
| `egen group()` / `tag()` with empty strings | "" a group | "" missing unless `missing` |
| `xtile, cutpoints()` with repeated cutpoints | merged | kept, as Stata |
| `tsset t, delta(k)` | delta ignored | lags step by k |

**What to do.** Re-run do-files that use these commands. Code that
calls StatsPAI functions directly is not affected.

---

## 1.39.5 → 1.39.6: ⚠️ numbers that change after the October review

**What changed.** A review of the month's work fixed defects on inputs
off the main path. Default calls on ordinary data return what they did.
The calls below return different numbers; recompute them.

| Call | Before | Now |
| --- | --- | --- |
| `sp.sun_abraham`, `sp.stacked_did` on an integer time column with gaps | no reference period omitted (estimate off, standard error 0 or NaN) | periods numbered by rank; equals the consecutively coded panel |
| `sp.gardner_did` with treated cells that have no untreated comparison, with units treated throughout, or with never-treated coded as NaN | minimum-norm first stage; NaN rows deleted | those cells dropped with a warning; NaN is never treated |
| `sp.etwfe` with a cohort after the last period | treated as a cohort | never treated |
| `sp.did_multiplegt` with no switchers | 0, p = 1 | NaN |
| `sp.cate_gain_curve` with tied scores | ties broken by row order | tied units enter together |
| `sp.iv.kleibergen_paap_rk`, `sp.iv.sanderson_windmeijer`, and the `KP rk` / `SW conditional F` diagnostics of `sp.iv`, with two or more endogenous regressors | not the rk statistic; OLS partialling | equal to Stata `ivreg2` |
| `sp.iv.iv_diag(cluster=<array>)`, and with `absorb=` plus CLR / K sets | Anderson-Rubin and effective F unclustered; sets ignored the fixed effects | as with a column name; sets on the demeaned variables |
| `sp.panel(method='fd')` on unsorted rows or a late-starting first unit | differences between non-consecutive rows | consecutive periods of a unit |
| `sp.fisher_exact(cluster=, stratify=)` | `stratify` ignored | clusters permuted within strata |
| `sp.bootstrap(ci_method='bca', block=)` | row jackknife for the acceleration | block jackknife |
| `sp.jackknife_se` on fits with interactions, factors, weights or no intercept | refitted a different model | refits the stored design |
| `sp.local_projections(cumulative=True)`, `nw_lags=` below the horizon | one-period change; lag raised to the horizon | change since `t−1`; the lag you gave |
| `sp.garch(model='egarch', dist='t').forecast(h)` for h ≥ 2 | future shocks centred at the normal value | at the t value |
| `sp.dlm` with `constant=` or a zero state variance | first-date smoothed sd off; Gibbs paths moved within a draw | exact |
| `sp.arima`, `sp.structural_break`, `sp.cusum_test`, `sp.gee`, `sp.gam`, `sp.mswitch`, `sp.dlm`, `sp.panel_logit(method='fe')`, `sp.xtserial` with a redundant regressor | coefficient split arbitrarily, or a huge standard error | omitted with a note and listed in `model_info['omitted']` |
| `sp.chow_test(time=)`, `sp.zivot_andrews(time=)` on unsorted rows | regimes by file order | sorted by `time` |
| `sp.forecast_accuracy` with a zero actual value | MAPE over the other periods | missing, with a warning |
| `sp.factor_importance` with numerically coded discrete factors | arbitrary tied neighbour | all tied neighbours |
| `sp.mdp_policy_value` standard error | depended on state labels and on the location of the features | invariant to both |
| `sp.pscore`, `sp.marginal_policy_effect`, `sp.implied_weights` with a `category` covariate | one numeric column | indicators |
| `sp.itcv` on a weighted fit | unweighted correlations | weighted |
| `sp.cor_test` with a redundant or constant covariate | one degree of freedom too few | rank of the covariates |
| `sp.standardize(formula=)` with `log(x)`, `I(x^2)`, `C(g)` | rescaled `x`, `g` | left alone |
| `sp.bandit_allocate`, `sp.bandit_experiment` with `prob_floor > 0` | floor ignored while topping up | floor applied |
| `sp.space_filling(qualitative=)` with a run count that is not a multiple of the level combinations | unbalanced levels | balanced to within one run |
| `sp.gam` with `d` next to `s(x, by=d)`, or `x` next to `s(x)` | unidentified parametric term kept | omitted |
| `sp.bjs_pretrend_joint` given a result that is not from `sp.did_imputation` (e.g. Callaway-Sant'Anna) | that estimator's leads tested against a covariance bootstrapped from `did_imputation` refits | refused (`MethodIncompatibility`); fit `sp.did_imputation` with the same arguments |

**Calls that now raise.** Each of these returned a number that did not
mean what it appeared to.

- `predict()` with a factor level the fit never saw (`sp.regress`,
  `sp.logit`, `sp.probit`, `sp.glm`, `sp.poisson`, `sp.nbreg`,
  `sp.poststratify`). Fix the label or refit.
- `sp.influence_measures` on a weighted fit; `sp.itcv` on an IV fit.
- `sp.jackknife_se` on a fit with absorbed fixed effects, or one whose
  stored design does not reproduce its coefficients.
- `sp.ipw` under complete separation; a single arm in `sp.aipw`,
  `sp.g_computation`, `sp.pate`, `sp.fisher_exact`.
- A single cluster in `sp.bootstrap`, `sp.jackknife_se`, `sp.cr2_se`,
  `sp.wild_cluster_boot`, `sp.hdfe_ols`, `sp.iv.iv_diag`.
- `sp.hdfe_ols` with `vce='robust'` and `cluster=` together, `wild=True`
  without `cluster=`, or an `se_type` that contradicts the other
  options. `se_type='hetero'` used to return iid standard errors under
  that label; pass `vce='robust'`.
- A row with a missing value inside the sample in `sp.var`, `sp.vec`,
  `sp.varsoc`, `sp.johansen`, `sp.johansen_lrtest`, `sp.engle_granger`,
  `sp.mdp_policy_value`, `sp.marginal_policy_effect`, and an interior
  gap in `sp.panel_unitroot`. Rows missing at either end are trimmed.
- A constant, exactly fitted or linearly dependent series in the unit
  root, VAR, cointegration, break and event-study functions.
- A Series whose `DatetimeIndex` or `PeriodIndex` is not increasing, in
  the forecasting functions, `sp.arima`, `sp.garch` and `sp.unitroot`.
  Sort it first.
- `sp.mdp_policy_value` when the policy's action never occurs in a state
  it reaches.
- `sp.gee` and `sp.gam` with an outcome outside the support of the
  family; `sp.panel_logit(method='fe')` with an outcome that is not 0/1.
- `sp.nls` with a rank-deficient Jacobian at the solution.
- `sp.ets` with `damped=` contradicting the model string.
- `sp.meta_analysis`, the weak-instrument confidence sets,
  `sp.plausibly_exogenous_*` and `sp.iv.mte` with missing or non-finite
  input.
- `sp.gp_regress("y ~ x - 1")`, `sp.stochvol` on a two-column array,
  ordered and multinomial `sp.bayes_regress` fits with more
  coefficients than observations.
- `sp.rdmcplot` and the `rdplot` family with an unknown `kernel`; an
  unrecognised spelling such as `'Uniform'` used to give the triangular
  kernel.
- `alpha`, `level`, `trim` and lag arguments outside their range, in
  the functions that used to accept them.

**New warnings on results that are returned.** The default prior of
`sp.bayes_ivreg`, `sp.bayes_sur`, `sp.bayes_mixed`, the probit systems
and `sp.bayes_arima` is now checked after the fit; if the warning fires,
the estimate was being pulled toward zero and `prior_var=` should be
raised or the data rescaled. The shared propensity fit warns on
separation, which reaches `sp.match`, `sp.psmatch2`, `sp.pscore` and
`sp.full_match`.

**What to do.** Recompute anything in the table. For a call that now
raises, the message names the argument or the rows.

---

## 1.39.4 → 1.39.5: ⚠️ `sp.lasso_select(method="cv")`

**What changed.** The cross-validation refits the centring and scaling
inside each training fold and gives each fold its own intercept. Before,
the folds were cut from data standardised on the full sample and the
held-out rows were predicted without an intercept, so the held-out fold
had already entered its own fit. The error is now pooled over
observations instead of averaged over folds. `method="bic"` and
`method="aic"` are unchanged, and so is the coefficient path.

**Who is affected.** Anyone who used `method="cv"`. With hundreds of
observations per candidate variable the selected penalty is usually the
same grid point. In small samples the list of selected variables can
change. Recompute, and compare a few values of `seed`: when the selection
changes with the folds, the data do not pin it down.

**What to do.** Nothing in the call. `sp.glmnet` gives the same
cross-validated error on the same folds and also reports `lambda.1se`.

---

## 1.39.4 → 1.39.5: ⚠️ `sp.blp` diversion ratios, standard errors and outer loop

**What changed.**

- **`result.diversion_ratios()` returned wrong numbers.** It divided
  elasticities where it should divide share derivatives, so every entry
  was off by the factor `s_j / s_k`. Recompute any diversion ratio taken
  from this method. Ratios computed by hand from
  `result.elasticity_matrix()` are unaffected.
- **Standard errors change when a standard deviation is estimated at
  zero.** That parameter now has a NaN standard error and is held fixed
  for the others, which get smaller. Fits with every standard deviation
  away from zero keep their standard errors.
- **Point estimates can change.** The default outer loop is L-BFGS-B
  from four starting points. Where Nelder-Mead had converged to the
  lowest minimum the estimates agree to the tolerance it was run at (on
  the automobile data the largest standard deviation is 3.833, against
  3.841 after 3,000 Nelder-Mead iterations). Where it had stopped early
  or in a higher local minimum, the new fit has a lower objective and
  different estimates.

The logit and nested logit (`x_random=[]`) have no outer loop and are
unchanged.

**What to do.** Refit. To reproduce an earlier fit pass
`optimizer='nelder-mead'` with the `maxiter` used then; the standard
errors at a zero bound and the diversion ratios will still be the new
ones. Compare `model_info['starts']` to see whether the starting point
mattered for your data.

## 1.39.4 → 1.39.5: ⚠️ `rho` of the production-function estimators

**What changed.** `result.productivity_process["rho"]` and
`result.diagnostics["ar_rho"]` from `sp.olley_pakes`,
`sp.levinsohn_petrin`, `sp.acf` and `sp.prod_fn(method='op' | 'lp' |
'acf')` are now the slope of the linear projection of productivity on its
lag. They were the linear coefficient of the Markov polynomial, which is
a persistence only when `productivity_degree=1` or productivity is
centred at zero. `sp.wooldridge_prod` is unchanged.

Input coefficients change only where stage 2 had lost digits to the
level of log output: by 1e-5 in the capital coefficient when log output
was near 200, by less than 1e-9 on data near zero.

**What to do.** Recompute any reported `rho` that came from the default
`productivity_degree=3` on real data. The polynomial coefficients are in
`diagnostics["markov_coef"]` as before. A fit without `boot_reps` no
longer warns about a singular variance matrix; code that filtered that
warning can drop the filter.

## 1.39.3 → 1.39.4: `sp.mixed` and `sp.nbreg` after the speed rewrite

**What changed.** No argument, default or return type changed, and no
other estimator touched by the speed work returns a different number.
Two do, by small amounts.

- **`sp.mixed`.** The likelihood is evaluated from per-group summaries
  and rounds differently. When every variance component is away from
  zero and every correlation away from one, fixed effects, standard
  errors, variance components and BLUPs move by less than 1e-6 relative.
  When the optimum is on the boundary (a variance at zero, a correlation
  at one), and in the three-level model, the likelihood is flat and the
  optimiser stops at a slightly different point: variance components
  can differ by up to 2e-4 relative, fixed-effect standard errors by up
  to 8e-5, BLUPs by up to 9e-5. Where the residual variance is far below
  the random-effect variance the old likelihood was itself off by up to
  1e-7; the new one is accurate to 3e-13.
- **`sp.nbreg`.** Fits with overdispersion are unchanged (1e-14). On
  data with no overdispersion the outer loop used to run its 100 rounds
  and warn "did not converge"; it now stops when the likelihood stops
  moving, `model_info['converged']` is `True` and the warning is gone.
  Coefficients on such data move by 1e-7 or less. The standard error of
  the log dispersion can differ, because the dispersion is not
  identified at its lower bound.

**What to do.** Nothing for results reported to four or five digits. A
test that pins `sp.mixed` output of a boundary fit or of a three-level
model tighter than 1e-3 may need its reference value refreshed. Code
that treated the `sp.nbreg` non-convergence warning as a sign of
Poisson-like data should test the fitted dispersion instead.

## 1.39.3 → 1.39.4: ⚠️ options off the main path in matching, regression and forests

**What changed.** These calls returned a different number before; the
list is by what to recompute.

- **JIVE.** `sp.iv(method='jive' | 'jive1' | 'ujive' | 'ijive' | 'rjive')`
  and `sp.iv.jive_mw` regressed the outcome on the leave-one-out
  first-stage fit instead of using it as an instrument. Point estimates
  and standard errors change, typically away from zero. `sp.jive` is
  unchanged.
- **`sp.glm(robust='hac')`.** Standard errors were too small by
  `sqrt(n)`.
- **`sp.match`.** Standard errors change, estimates do not, for
  `weights=` (was 0), for `estimand='ATE'` with nearest-neighbour
  matching (was the ATT's), and for the ATT with `distance='mahalanobis'`
  or `'euclidean'` (was 25 to 40% too large). `method='cem',
  estimand='ATE'` returned the ATT and now returns the ATE. The ATT with
  `distance='propensity'`, the default, is unchanged.
- **`sp.glm(family='gamma' | 'inverse_gaussian')`.** Default standard
  errors change (the dispersion is the Pearson estimate, as in R and
  Stata, where it was deviance / df); `scale='dev'` gives the old ones.
  Coefficients do not change.
- **`sp.propensity_score`** with a repeated or redundant covariate, and
  the balance tables and plots computed from it.
- **`sp.poisson` / `sp.nbreg`** called with both `offset=` and
  `exposure=` (the offset was dropped).
- **Marginal effects of `sp.logit` / `sp.probit` / `sp.cloglog`** fitted
  with `weights=`.
- **`sp.iv(cluster=)`** on data with missing cluster labels.
- **`sp.rate_split` and `sp.forest_policy_tree`** for any forest except
  one with user-supplied nuisance models and no controls; **`sp.rate`**
  for a forest fitted with `equalize_cluster_weights=True` on clusters of
  unequal size.

**What is now an error.** A single cluster in the binary, count, ordered
and multinomial models (the standard errors were about 1e-16);
`sp.match` / `sp.psmatch2` when no unit finds a match (the effect was
reported as 0 with p = 1); `sp.glm` with an outcome outside the family's
support; a two-way fixed effects forest whose treatment the period
effects absorb; `sp.cbps` with a redundant covariate; unknown values of
`dispersion=`, `count_model=`, `marginal_effects=` and `gmm_vcov=`;
`sp.match(estimand='ATE', common_support='minmax')` and
`caliper_scale='sd'` with a covariate distance.

**What to do.** Recompute anything on the first list. Code that read a
standard error of 0, a p-value of 1 or an estimate of exactly 0 from the
calls on the second list as a result should treat the new error as the
answer: the quantity was never estimated. A forest fitted with controls
by an earlier release has to be refitted before `sp.rate_split` or
`sp.forest_policy_tree` will accept it.

## 1.39.2 → 1.39.3: ⚠️ inputs that returned a wrong number

**What changed.**

- `sp.fect(method='fe', force='none')` with no covariates returned the mean
  outcome of the treated cells. It now returns that mean minus the mean of
  the untreated cells, as R `fect` does.
- A `Period` cohort column with `NaT` for never-treated units put some
  treated units in the wrong cohort, in every staggered DiD estimator that
  accepts calendar time.
- `sp.rdrobust` raises `DataInsufficient` when the running variable has
  fewer than `p + 1` distinct values on a side. It used to warn and return
  an arbitrary number.
- `sp.drdid(id=..., weights=...)` ignored the weights on panel data. It
  now uses them, as R `DRDID` does.
- `sp.did_multiplegt` reports a placebo or dynamic effect it cannot
  estimate as missing, not as 0 with p-value 1.
- `sp.callaway_santanna` refuses two cluster choices that gave a standard
  error of zero: a within-unit-varying cluster under
  `allow_unbalanced_panel=True`, and, with `panel=False`, a cluster that
  is constant within every cohort-by-period cell.
- `sp.honest_did` and `sp.breakdown_m` under relative magnitudes no longer
  return an empty set (and a breakdown value of 0) when the estimate is
  more than 20 standard errors from zero.

**What to do.** Rerun any of these. Nothing else is affected: `fect`
with fixed effects, factors, matrix completion or covariates; DiD with
integer or `datetime64` time; `rdrobust` on a running variable with enough
support; unweighted `drdid`; `did_multiplegt` rows that were estimable.
Code that read a `did_multiplegt` placebo of exactly 0 as "no pre-trend"
should check for a missing value.

## 1.39.2 → 1.39.3: ⚠️ `sp.liml` formulas with the bracket first

**What changed.** `sp.liml("y ~ (d ~ z1 + z2) + x1 + x2", data=df)` used
to treat `x1` and `x2` as excluded instruments and leave them out of the
structural equation. It now reads them as exogenous controls, as
`sp.ivreg` and `sp.iv(method="liml")` always have. A formula that is in
neither the bracket form nor `"y ~ x | d | z"` raises
`MethodIncompatibility`.

**What to do.** Rerun any `sp.liml` call whose formula puts the bracket
before the controls: the coefficients change, and the old ones were not
the LIML estimate of the model as written. Formulas written
`"y ~ x + (d ~ z)"`, calls with `y=, x_endog=, x_exog=, z=`, and
`sp.iv(method="liml")` give the same numbers as before.

## 1.39.2 → 1.39.3: ⚠️ balance diagnostics with missing weights

**What changed.** `sp.balance_diagnostics`, `sp.ps_balance` and
`sp.love_plot` read a missing weight as zero. Before, a single missing
weight (every unmatched row of a matched frame has one) turned each
weighted standardized difference into 0.0000 and `balanced` into `True`.
Negative or infinite weights, and a treatment arm with no positive weight,
now raise `ValueError`.

**What to do.** Recompute any balance table or love plot that was built
from a matched frame's `_weight` column, including
`sp.love_plot(sp.psmatch2(...))`. Tables built from complete weights are
unchanged.

## 1.39.2 → 1.39.3: ⚠️ `itt_ci` in a fuzzy `sp.rdrandinf`, and inputs that are now refused

**What changed.**

- `model_info['itt_ci']` of a fuzzy `sp.rdrandinf` with a `ci=` grid held
  the Anderson-Rubin interval of the complier effect. It now holds the
  large-sample interval of the ITT. The Anderson-Rubin interval is under
  `model_info['anderson_rubin_ci']`.
- `sp.rd_optimized(fuzzy=)` raises `DataInsufficient` when the treatment
  does not jump at the cutoff. It used to return a ratio near 1e16.
- `sp.twowayfeweights` returns a missing standard error on a saturated
  design, where it returned one near 1e-15.
- `sp.dml(fold_indices=...)` raises `MethodIncompatibility` when a row
  that is otherwise complete has a missing fold label. It used to drop
  the row.
- `sp.synth.causal_impact` raises `DataInsufficient` when the outcome is
  missing in a post-period row. It used to return NaN.

**What to do.** Code that read `itt_ci` as the interval of the complier
effect should read `anderson_rubin_ci`, or the result's `ci`. For the
other four, supply the missing piece the message names. No estimate that
was identified before has changed.

## 1.39.1 → 1.39.2: ⚠️ `sp.mswitch` and `sp.mswitch_lrtest` search on the standardised series

**What changed.** The multi-start search runs on `y` divided by its
standard deviation and maps the estimates back. The model and the
likelihood are the same; what changes is which local maximum a start
reaches when the series is far from unit scale. Before, that depended on
the unit of measurement, and in `sp.mswitch_lrtest` it could move the
bootstrap p-value (0.20 to 0.15 on one example after multiplying `y` by
0.01, with `starts=3`).

**What to do.** Nothing for a series of roughly unit scale. For a series
measured in large or small units (levels of GDP, basis points), refit: the
reported maximum is now the same whatever the unit, and its likelihood is
not lower than before. A p-value of `sp.mswitch_lrtest` computed on such a
series with few starts should be recomputed.

## 1.38.0 → 1.39.0: `CausalForest.variable_importance()` defaults to grf's measure

**What changed.** Without `method=` the method returns the depth-weighted
split frequencies of `grf::variable_importance` (the same numbers as
`sp.variable_importance(forest)`), and no longer warns. It returned the
permutation measure, with a `FutureWarning` announcing this change since
1.31. Forests grown with `split_rule="legacy"` keep the permutation
measure, because they record no split depths.

**What to do.** Pass `method="permutation"` to keep the old numbers.

## 1.38.0 → 1.39.0: ⚠️ `sp.gsynth` with covariates

**What changed.** The coefficients on `covariates` are estimated inside
the interactive fixed effects model of the never-treated units, with unit
effects, period effects and the factors, as in Xu (2017) and R `gsynth`.
They used to come from a pooled regression of the outcome on the
covariates alone, on pre-treatment control rows. The counterfactual, the
effect and the per-period effects change whenever a covariate is
correlated with the unit effects or the loadings. On one treated unit of
the `gsynth` example data the effect moves from 6.68 to 5.70, which is
what `gsynth` returns.

Calls with covariates also change how they choose the number of factors
(the leave-one-period-out error of the treated unit before treatment) and
how they do inference (`inference=`, a bootstrap, in place of the placebo
over donors). `placebo=` and `cv_folds=` are not used on that path.

**Who is affected.** `sp.gsynth(..., covariates=[...])` and
`sp.synth(method='gsynth', covariates=[...])` on the native backend.
Calls without covariates return the same numbers as before.

**What to do.** Re-run. To hold the number of factors fixed across the
change, pass `n_factors=`. The panel has to be balanced on this path;
`sp.fect` takes unbalanced panels.

## 1.38.0 → 1.39.0: `sp.aipw` says when it clips propensity scores

**What changed.** No estimate changes. `model_info['n_propensity_clipped']`
used to be 0 always; it is now the number of fitted scores outside
`[trim, 1 - trim]`, and a `RuntimeWarning` is issued when it is positive.
The bound is the new argument `trim`, default 0.01 as before.

**What to do.** If the warning appears, look at overlap
(`sp.overlap_plot`). `trim=0` gives the unclipped estimator; restricting
the sample with `sp.trimming` changes the target population instead.

## 1.38.0 → 1.39.0: ⚠️ `sp.network_exposure` rebuilt

**What changed.** Exposure probabilities are exact under Bernoulli
assignment instead of simulated and floored. The default estimator is the
self-normalised (Hajek) mean; pass `estimator='ht'` for Horvitz-Thompson.
Standard errors come from a variance estimate over the dependency graph
of the design (`variance='hac_psd'` by default, `'hac'` without the
conservative adjustment), and contrasts account for the covariance
between the two exposure means. Units for which some exposure level is
impossible, such as units without neighbours under `mapping='as4'`, are
excluded with a warning. New arguments: `estimator`, `variance`,
`contrasts`, `thresholds`, `min_prob`, `alpha`. The result gains the
columns `min_prob` and `ess` and the attributes `estimator`, `variance`,
`n_eligible`.

**Who is affected.** Every caller. Point estimates change because the
probabilities are exact and the default is the Hajek mean. Standard
errors change by a large factor on networks with uneven degree, where
the old ones could be a hundred times the sampling spread.

**What to do.** Re-run. `n_sim` and `seed` are still accepted and are
used only when `mapping` is a callable. To reproduce the Horvitz-Thompson
point estimate, pass `estimator='ht'`; the old standard errors cannot be
reproduced and should not be.

## 1.38.0 → 1.39.0: conditional effects of the R- and DR-learners

**What changed.** When `cate_model=` is not given,
`sp.metalearner(learner='r')`, `sp.metalearner(learner='dr')`,
`sp.RLearner` and `sp.DRLearner` fit the final stage with gradient
boosting that requires 20 observations per leaf. It had no minimum.

**Who is affected.** Anyone using the fitted conditional effects of these
two learners with the default final stage: `model_info['cate']`,
`sp.predict_cate`, and summaries built on them. The reported average
effect and its standard error come from the doubly robust scores and do
not change.

**What to do.** Re-run. To reproduce earlier conditional effects pass
`cate_model=GradientBoostingRegressor(n_estimators=200, max_depth=3,
learning_rate=0.05, subsample=0.8, random_state=42)`.

## 1.38.0 → 1.39.0: ⚠️ `sp.iv.mte` ATT, ATU and standard errors

**What changed.** The ATT and ATU are computed from weights over the
whole sample (each unit in proportion to its propensity, or one minus
it) instead of over the treated or untreated units alone. Standard errors
of the MTE curve and the ATE use the full, heteroskedasticity-robust
covariance of the coefficients; the ATT and ATU get standard errors in
`extra['att_se']` and `extra['atu_se']`. The bootstrap resamples aligned
rows.

**Who is affected.** Anyone who reported an ATT or ATU from `sp.iv.mte`,
any analytic standard error from it, or any bootstrap standard error on
data where trimming removed rows. The ATE point estimate and the MTE
curve itself are unchanged.

**What to do.** Re-run. On a design with a linear marginal effect the
ATT moved from 1.22 to 1.31 (truth 1.30) and the ATE standard error from
0.075 to 0.022 (sampling spread 0.023).

## 1.38.0 → 1.39.0: ⚠️ `sp.tmle` validates `estimand`

**What changed.** `sp.tmle(estimand=...)` used to treat every value other
than the exact string `'ATE'` as the ATT and to label the result with
whatever was passed. `estimand='ate'` therefore returned the ATT. The
argument is now case-insensitive and anything outside `ATE`, `ATT`, `ATC`,
`EY1`, `EY0`, `RR`, `OR` raises.

**Who is affected.** Anyone who passed a lower-case or misspelled
`estimand` and read the result as an ATE. Results from `estimand='ATE'`
and `estimand='ATT'` spelled exactly so are unchanged.

**What to do.** Rerun. The new value is the ATE that was asked for.

## 1.38.0 → 1.39.0: ⚠️ `sp.gp_regress` hyperparameters on coarse or replicated designs

**What changed.** Before maximising the marginal likelihood the function
now evaluates it on a coarse grid of length scales and noise shares and
adds the two best grid points to the starting values. The likelihood is
flat where the length scale is far below the spacing of the data, and a
search that entered that region did not leave it.

**Who is affected.** Fits that ended on that plateau: typically one
regressor observed at few distinct, replicated or widely spaced values,
with the default starting values. Such a fit had a very small length scale
and predictions that returned to the mean between the data points. Fits
that were at the maximum are unchanged, and so is every call with
`optimize_hyper=False` or with `length_scale=` given.

**What to do.** Nothing in calling code. If a stored fit had a length
scale below the smallest gap between observed values, refit; the log
marginal likelihood of the new fit is at least as high.

## 1.38.0 → 1.39.0: the default prior of `sp.bayes_regress` will change in 1.40

**Done in 1.40.** An omitted `prior` now means `prior="weakly_informative"`
for the five models named below (when `prior_mean` and `prior_var` are not
given and `inference="mcmc"`), and the `DeprecationWarning` is gone. Pass
`prior="vague"` to keep the fixed prior. The text below is the 1.39
announcement.

**What changes, and when.** Today an omitted `prior` means a fixed
`N(0, 1000)` on every coefficient (`N(0, 100)` for logit and probit). In
1.40 it will mean `prior='weakly_informative'` for the models that have
one (normal, logit, probit, Poisson, negative binomial): normal priors
scaled to the data, as R `rstanarm` uses. Nothing changes in this release
except a `DeprecationWarning` on the calls that will be affected.

**Who is affected.** Calls to `sp.bayes_regress` for those five models
that pass neither `prior` nor `prior_mean` nor `prior_var`. With regressors
on a moderate scale the two priors give nearly the same posterior; with a
regressor or an intercept in large units the fixed prior pulls the
estimate toward zero (the function already warned about that), and those
are the fits that will move.

**What to do.** To keep today's numbers, write `prior='vague'`. To adopt
the new default now, write `prior='weakly_informative'`. Either silences
the warning.

## 1.38.0 → 1.39.0: draws of the Metropolis models of `sp.bayes_regress`

**What changed.** `model='logit'`, `'poisson'`, `'negbin'` and `'mlogit'`
are sampled with a mixture of the former random-walk proposal and an
independence proposal centred at the posterior mode.

**Who is affected.** Anyone comparing draws or summaries with a stored
run at the same seed. The posterior is unchanged, so summaries agree
within Monte Carlo error, which is now smaller: the effective sample size
is three to five times what it was.

**What to do.** Nothing. `acceptance_rate` still refers to the
random-walk moves.

## 1.38.0 → 1.39.0: ⚠️ collinear regressors are omitted by the likelihood-based estimators

**What changed.** `sp.logit`, `sp.probit`, `sp.cloglog`, `sp.glm`,
`sp.poisson`, `sp.nbreg`, `sp.ologit`, `sp.oprobit`, `sp.mlogit`,
`sp.tobit`, `sp.qreg` and `sp.svydesign(...).glm` now omit a regressor that
is an exact linear combination of earlier ones, warn with its name and
list it in `model_info['omitted']`. `sp.regress` already did. Before, the
optimiser was run on the rank-deficient design and returned one arbitrary
point of a flat likelihood, with standard errors of zero, of order 1e6 or
`nan`, sometimes without a warning.

**Who is affected.** Fits whose design was rank deficient: all the
indicators of a category next to the constant, a duplicated column, a
regressor that is the sum of others. The result now has one coefficient
fewer per dependent regressor, and the remaining coefficients and standard
errors are those of the model without it. Fits on full-rank designs are
unchanged to the last digit.

**What to do.** Code that indexes coefficients by position should index by
name. A stored coefficient for a regressor now reported as omitted was
never identified and should be discarded with the numbers around it.

## 1.38.0 → 1.39.0: ⚠️ separation warnings from `sp.logit` and `sp.glm`

**What changed.** A binary fit whose linear index classifies every
observation correctly, or whose fitted probabilities reach 0 or 1 to
machine precision, raises a `ConvergenceWarning`. `sp.logit` used to
require in addition that 99 percent of the fitted probabilities be within
0.01 of the boundary, which small samples often failed; `sp.glm` did not
check.

**Who is affected.** Fits on separated data that used to return silently.
The numbers returned are the same as before and as meaningless as before.

**What to do.** Treat the warning as a failed fit. A weakly informative
prior gives a finite answer:
`sp.bayes_regress(formula, df, model='logit', prior='weakly_informative')`.

## 1.38.0 → 1.39.0: ⚠️ aggregated SEs of weighted Callaway-Sant'Anna on repeated cross-sections

**What changed.** After `sp.callaway_santanna(..., weights=...)` with
`panel=False` or `allow_unbalanced_panel=True`, the share-estimation term
of an aggregated variance now uses the observation weights, as R `did`
does. It used head counts. The standard errors of aggregates that mix
cohorts (simple, event study, calendar, the overall row of the group
aggregation, and the headline of the fit) move by up to 1.2e-4 in relative
terms on the data used to find it.

**Who is affected.** Weighted fits on repeated cross-sections or on an
unbalanced panel kept as one. Point estimates, ATT(g, t) and their
standard errors are unchanged, and so is everything on a balanced panel or
without weights.

**What to do.** Nothing in calling code. A stored result will differ in
those standard errors after the fifth significant digit or so.

## 1.38.0 → 1.39.0: ⚠️ `sp.arima` on over-parameterised mixed models

**What changed.** A final fit also searches from zero ARMA coefficients
and from the estimates of the models with one AR or one MA term fewer, and
keeps the best. An ARMA(5,2) for Swiss GDP growth went from a
log-likelihood of -110.13 to -107.03, the value of R's `arima`. An MA(3)
for the quarterly change of Swiss log GDP went from -249.22, below the
MA(2) it nests, to -243.82.

**Who is affected.** Fits with two or more ARMA terms whose likelihood has
more than one maximum, typically over-parameterised models and series with
a seasonal pattern left in. A fit that was at its maximum is unchanged.

**What to do.** Re-run. A higher log-likelihood for the same order is the
better fit.

## 1.38.0 → 1.39.0: ⚠️ `sp.garch` with two or more lagged variances

**What changed.** `sp.garch(y, p=2, ...)` and higher orders search from
several starting values. Earlier releases could return a lagged-variance
coefficient of exactly zero, a standard error of zero for it, and a
log-likelihood below that of GARCH(1, q). On the Swiss Market Index
returns, GARCH(2,1) went from -5600.552 to -5598.548, the value of Stata's
`arch`. An estimate that still ends at zero now raises a `RuntimeWarning`.

**Who is affected.** Anyone who reported a GARCH model with `p >= 2` or
`q >= 2`, or compared such models by AIC or BIC. GARCH(1,1) and ARCH(1)
results do not change.

**What to do.** Re-run. If the warning appears, report the lower-order
model.

## 1.38.0 → 1.39.0: ⚠️ `sp.fci` returns the graph of the full algorithm

**What changed.** `sp.fci` runs the Possible-D-SEP pass and all ten
orientation rules. Graphs can lose edges the earlier version kept, and
circle marks can become tails or arrowheads. The result equals
`pcalg::fci`.

**Who is affected.** Anyone who reported a PAG from `sp.fci`.

**What to do.** Re-run. `sp.fci(..., possible_dsep=False)` keeps the
earlier skeleton (it is RFCI's); the orientation still uses the ten rules.

## 1.38.0 → 1.39.0: ⚠️ `sp.match` keeps every tied match by default

**What changed.** `sp.match` (and `sp.psm`, `sp.MatchEstimator`) with
`replace=True` now defaults to `ties='all'`: when several units are exactly
as close as the `n_matches`-th nearest, all of them are matched and share
the weight. The default was `ties='first'`, which kept the first in data
order. The ATT, ATE, their standard errors and the matched frame change
wherever the data have exact ties, which means discrete covariates or a
propensity score with few distinct values. On the Lalonde sample the ATT
goes from 1967.94 to 1968.80, the value of Stata `teffects psmatch`.

**Who is affected.** Anyone who reported a nearest-neighbour matching
estimate with replacement on data with exact ties. The earlier number
depended on the order of the rows. `sp.psmatch2` is not affected, nor is
matching without replacement, nor data with continuous covariates.

**What to do.** Re-run. `ties='first'` gives the earlier number, with a
warning that says how many units had ties.

## 1.38.0 → 1.39.0: ⚠️ `sp.bayes_mte` with `mte_method='polynomial'` or `'hv_latent'`

**What changed.** The default `mte_method='polynomial'` fitted
`Y = alpha + D * g(p)` and reported `g` as the MTE curve. That model
assumes the untreated outcome does not vary with the unobserved
resistance to treatment. When it does (selection on levels, the usual
case), `g`, the ATE, ATT, ATU and every policy effect built on it were
wrong, not just noisy. The mode now fits the local IV regression
`E[Y | p] = alpha + sum_k b_k int_0^p a(u)^k du`, in which `b_k` are the
coefficients of the MTE polynomial and `alpha` is `E[Y_0]`.

`mte_method='hv_latent'` gains a centred polynomial `b_sel` for the
untreated outcome, of the same order as the MTE polynomial.

`mte_method='bivariate_normal'` is unchanged.

The method label of the polynomial mode reads "Bayesian local-IV MTE"
(was "Bayesian treatment-effect-at-propensity").

**Who is affected.** Anyone who reported an MTE curve, ATE, ATT, ATU or
policy effect from `sp.bayes_mte` without `mte_method='bivariate_normal'`.
Results change in level and slope. Posterior intervals are wider: local
IV identifies the MTE from variation in the propensity alone.

**What to do.** Re-run. There is no switch for the earlier numbers.

## 1.38.0 → 1.39.0: ⚠️ `sp.bayes_synth` credible intervals

**What changed.** The posterior of the effect now includes the treated
unit's own post-period noise, `N(0, sigma^2 / T1)`, in addition to the
uncertainty of the donor weights. The posterior mean is unchanged. The
posterior standard deviation, the interval, `prob_positive` and the ROPE
probability change; intervals are wider, most of all with few post
periods or a noisy pre-period fit. The post-period counterfactual band in
`model_info['detail']` is now a predictive band.

**Who is affected.** Anyone who reported an interval or a probability
from `sp.bayes_synth`.

**What to do.** Re-run. The earlier interval answered a different
question (how well the weights are known) and did not cover the effect at
its nominal rate.

## 1.38.0 → 1.39.0: ⚠️ `sp.bayes_iv`, `sp.bayes_hte_iv` and `sp.bayes_fuzzy_rd` credible intervals

**What changed.** `sp.bayes_iv` and `sp.bayes_hte_iv` used first-stage
residuals computed once by OLS as a regressor in the outcome equation. The
posterior of the effect was centred on 2SLS but its standard deviation was
too small by the factor `sqrt(1 - corr(v, eps)^2)`, where the correlation
is the one between the first-stage and the structural error. The first
stage now enters the outcome equation through its parameters, which is the
joint normal model the docstring described. Posterior means barely move.
Posterior standard deviations and intervals grow, by 15 percent at a
correlation of 0.5 and by a factor of 2.3 at 0.9.

`prior_first_stage_sigma` in `sp.bayes_iv` is now the prior scale of the
first-stage coefficients. It was ignored (they used `prior_coef_sigma`).

`sp.bayes_fuzzy_rd` models the errors of the outcome and take-up equations
as jointly normal. They were independent. Posterior means of the effect
barely move; intervals become narrower in most designs and slightly wider
when the effect and the selection into treatment have opposite signs.

**Who is affected.** Anyone who reported an interval, a posterior standard
deviation, `prob_positive` or a ROPE probability from these three
functions.

**What to do.** Re-run. There is no switch for the earlier numbers; the
earlier intervals did not have their nominal coverage.

## 1.38.0 → 1.39.0: ⚠️ `CoxResult.ph_test()` and the Poisson overdispersion test

**What changed.**

- `CoxResult.ph_test()` returns the Grambsch-Therneau score test. The
  `chi2` and `p_value` columns change, a `df` column and a `GLOBAL` row are
  added, and `rho` is the correlation of the scaled Schoenfeld residuals
  with the time function instead of a Spearman correlation with time.
  `fit.model_info['ph_test']` and the proportional-hazards violation that
  `fit.violations()` reports follow the new p-values.
- `sp.poisson(...).diagnostics['Overdispersion test (C-T)']` and
  `'Overdispersion p-value'` change. The statistic is now the
  Cameron-Trivedi t ratio against the NB2 alternative.

**Who is affected.** Anyone who reported a proportional-hazards test from
`ph_test()`, or relied on the Poisson overdispersion flag to clear a model.

**What to do.** Re-run. Code that takes the row with the smallest p-value
from `ph_test()` should drop the `GLOBAL` row first, or use it. There is no
switch for the earlier numbers: neither was the statistic it was named
after. `ph_test(method='approx', transform='identity')` reproduces
Stata's `estat phtest`.

## 1.38.0 → 1.39.0: ⚠️ `sp.arima` optimiser, automatic search and regressor names

**What changed.**

1. A fit is no longer one quasi-Newton search. It is followed by simplex
   searches and a polish. Where the single search had stopped at an
   inferior local maximum, or at its limit of 50 iterations, the
   estimates change and the log likelihood rises. A model with
   differences is estimated on the differenced series; where the old
   search had found the maximum this moves the log likelihood by less
   than `1e-3` and coefficients by at most `2e-3`.
2. `aicc` of a model with a seasonal difference uses `n - d - D s`
   observations in its correction. It used `n - d`.
3. `auto=True` searches stepwise, decides by AICc whether to include a
   mean or a drift, and does not return a model with a root of modulus
   below 1.01. It used to fit the full `(p, q)` grid with the default
   constant.
4. Regressors are named after their columns in `params` and `se`. They
   were `x1`, `x2`, ...
5. `forecast()` returns a table indexed by the forecast periods when the
   series had a date or period index. With `level=` the columns are
   `forecast`, `lower_<level>`, `upper_<level>`.

**Who is affected.** Anyone who reported `sp.arima` estimates of mixed or
seasonal models, where the old search failed most often. Anyone who
relied on the order `auto=True` returned. Code that reads
`result.params["x1"]`.

**What to do.**

- Refit. A higher `log_likelihood` than before means the old estimates
  were not the maximum.
- For the old search over every `(p, q)`, pass `stepwise=False`. For the
  old treatment of the constant, pass `trend='c'` or `trend='n'`.
- Index regressor coefficients by column name.
- `forecast(horizon, alpha)` without `level=` keeps its three columns
  `forecast`, `lower`, `upper`.

## 1.38.0 → 1.39.0: `sp.ivreg` / `sp.iv` name formula terms as `sp.regress` does

**What changed.** A transformed, categorical or interacted term in an IV
formula is reported under the name the formula engine gives it. No number
changes.

| formula term | before | now |
| --- | --- | --- |
| `np.log(x)` | `np.log[x]` | `np.log(x)` |
| `I(x**2)` | `I[x ** 2]` | `I(x ** 2)` |
| `C(g)` | `g[1]`, `g[2]` | `C(g)[T.1]`, `C(g)[T.2]` |

The same holds for the keys of `diagnostics` that embed a name
(`"First-stage F (np.log(x))"`).

**Who is affected.** Code that indexes an IV result by a bracket name,
`iv.params["np.log[x]"]`.

**What to do.** Use the formula name, `iv.params["np.log(x)"]`. In
`sp.test` and `sp.lincom` both spellings are read. Plain column names
were never affected. `sp.panel` still reports the bracket names.
## 1.38.0 → 1.39.0: ⚠️ graphs: `sp.dag` parsing, adjustment sets, backdoor paths, `sp.identify`, `sp.pc_algorithm`, `sp.fci`, `sp.front_door`

Found by working through Ness's *Causal AI*
(`docs/dev/2026-10-06-ness-causal-ai-review.md`). Results that can change:

| call | before | now |
| --- | --- | --- |
| `sp.dag("X <- Z; ...")` | the right-to-left arrow was dropped | read as `Z -> X` |
| `sp.dag("A -> B <- C")` | a node named `"B <- C"` | two edges into `B` |
| `sp.dag` with `X -- Y`, `X => Y`, a cycle | skipped or accepted | `MethodIncompatibility` |
| `DAG.adjustment_sets` needing 7+ variables | `[]` | the set |
| `DAG.backdoor_paths` | every non-causal path | paths that enter the exposure; `noncausal_paths` has the rest |
| `DAG.path_status(...)[i]['type']` | `'causal'` or `'backdoor'` | also `'noncausal'` |
| `DAG.classify_variable` | a mediator on `X -> M -> C <- Y` was a `'confounder'` | it is not |
| `sp.identify` with a node named `U_...` | treated as latent | observed unless declared latent |
| `sp.identify(...).estimand` | conditioned on all predecessors | redundant conditioning variables dropped; same value |
| `sp.pc_algorithm` | edges lost to clashing colliders; union-of-neighbours search | PC-stable; clashes in `orientation_conflicts` |
| `sp.fci` | search stopped at the first level without a removal | full PC-stable skeleton |
| `sp.fci(...).edges` labels | `'>->'`, `'>--'`, `'>-o'` for an arrowhead at the left end | `'<->'`, `'<--'`, `'<-o'` |
| `sp.front_door`, mediator constant in an arm | an estimate scaled by the other arm's share | `IdentificationFailure`; `outcome_model='additive'` to extrapolate |

What to do.

- Re-read any graph specified with `<-`: the arrow was not in the graph
  before, so adjustment sets, paths and identification results computed
  from it were for a different graph.
- If code tests `path['type'] == 'backdoor'` to mean "not causal", test
  `!= 'causal'` instead.
- If a node name starts with `U_` and you meant it to be unobserved,
  declare it: `sp.dag(spec, latent=['U_x'])`.
- Rerun causal discovery. A graph from `sp.pc_algorithm` or `sp.fci` may
  gain edges it was losing and lose edges the old search could not
  separate. On Gaussian data the result now equals `pcalg::pc(...,
  skel.method = "stable")`.
- A `sp.front_door` call that now raises was returning a number that
  assumed the mediator has no effect in one arm. Use
  `outcome_model='additive'` if a no-interaction model is defensible, and
  say so.

`sp.ipw` now warns when the weights leave an arm an effective sample
under a fifth of its size. The estimate is unchanged.

## 1.38.0 → 1.39.0: ⚠️ `sp.arima` on differenced series, and `auto=True` with a drift

**What changed.** For a model with `d + D >= 1` and at least one AR or MA
term, `sp.arima` now starts the integration states from an exact diffuse
prior. It used the statsmodels default, a normal prior of variance 1e6,
which is informative when the innovation variance of the series is not
far below 1e6. Coefficients, standard errors, the log-likelihood, AIC,
BIC and AICc of such models move, and they move more the larger the
numbers in the series. For ARIMA(1, 1, 0) on quarterly U.S. GDP in
billions of dollars the AR coefficient goes from 0.170 to 0.216, which is
what R and Stata report. `method='innovations_mle'` keeps its
coefficients; its log-likelihood and information criteria change.

`sp.arima(auto=True)` now considers a constant: a mean without
differencing, a drift with one difference. It also drops candidates with
a root within 1% of the unit circle. Selected models will more often be
`(p, 1, q)` with a drift where they were `(p, 2, q)` or had an AR
coefficient near one.

AIC and BIC of differenced models use the number of observations after
differencing.

**Who is affected.** Anyone who fitted `sp.arima` with `d >= 1` to a
series not of order one in size, such as GDP, prices or an index in
levels, or who used `auto=True` on a trending series.

**What to do.** Re-run. To keep the earlier search over models without a
constant, pass `trend='n'`. There is no switch for the earlier estimates;
they changed with the unit of the data.

---

## 1.38.0 → 1.39.0: ⚠️ two-way clustered standard errors, `sp.poisson` under separation

**What changed.**

- `sp.regress(cluster=[a, b])`, `sp.panel(..., cluster=[a, b])` and
  `sp.feols(..., vcov={'CRV1': 'a + b'})` set the
  negative eigenvalues of the two-way covariance to zero when it is not
  positive semi-definite, and warn. Until now a negative variance came out
  as a standard error of 0 and the other standard errors were read off an
  indefinite matrix. The result's `.vcov()` is the two-way matrix; it
  raised before.
- `sp.poisson` stops when the deviance is flat even if a coefficient is
  still moving, and names the regressor responsible in
  `model_info['separated_terms']`. Such a model used to fail.
- `sp.from_r` on `feols(..., ~ a + b)` or `feols(..., vcov = ~ a + b)`
  returns the clustered call. It returned the unclustered one.

**Who is affected.** Anyone who clustered in two dimensions with one of
them small (years, industries) relative to the number of coefficients: a
`RuntimeWarning` now appears and the standard errors are larger. Fits whose
two-way covariance was positive semi-definite give the same numbers.
Converged Poisson fits give the same numbers.

**What to do.** Re-run. The adjusted matrix depends on the parametrisation
(which factor level is the base), as it does in `fixest` and `reghdfe`;
with few clusters in one dimension consider clustering on the other alone,
or a wild cluster bootstrap. There is no option to turn the adjustment off:
the unadjusted matrix is not a covariance matrix.

## 1.38.0 → 1.39.0: ⚠️ `sp.arima` default estimates and `auto=True` order selection

**What changed.** `sp.arima(y, order=...)` with the default
`method='statespace'` now maximises the exact Gaussian likelihood of every
observation. It used to start the state from a diffuse prior, which drops
the first `max(p, q + 1)` observations from the likelihood. Coefficients,
standard errors, the log-likelihood, AIC, BIC and AICc all move. On 223
observations of U.S. GDP growth the AR(1) constant goes from 2.942 to
2.980 and the log-likelihood from -562.07 to -565.00, the numbers Stata,
R and statsmodels' `ARIMA` report.

`sp.arima(auto=True)` chooses `d` by KPSS tests first, then `(p, q)` by
AICc with `(0, d, 0)` among the candidates. It used to skip `(0, d, 0)`
and compare AICc across values of `d`. Selected orders will often be
smaller.

`method='css'` and `method='conditional'` raise instead of returning exact
maximum likelihood estimates under another name.

**Who is affected.** Anyone who reported numbers from `sp.arima` without
`method='innovations_mle'`, or an order chosen by `auto=True`.

**What to do.** Re-run. There is no switch for the earlier numbers; they
were not the estimates of any documented estimator.
## 1.38.0 → 1.39.0: ⚠️ the base level of `C(g)`, `sp.hausman` with factors, RESET on the regressors

**What changed.**

- A formula term `C(g)` takes its reference category from the rows the
  model is fitted on. Before, a level that occurred only in rows dropped
  for a missing value elsewhere could become the base; one of the other
  indicators was then omitted as collinear. The fitted values, the
  residuals and every test of the factor as a whole are the same. The
  coefficients of the levels and the intercept are those of the coding
  Stata and R use.
- `sp.hausman(fe, re)` compares factor levels the two results name
  differently. The degrees of freedom and the statistic change when the
  models hold `C(g)` terms.
- `sp.estat(result, "reset", rhs=True)` takes the powers of each regressor
  on unit scale. The statistic changes only where it was wrong (regressors
  in the thousands and above; it could be negative).
- A formula may name columns called `class`, `return`, `yield` and other
  Python keywords inside functions. This raised an error before.

**Who is affected.** Anyone who read level coefficients from a model whose
factor had a level without complete rows; anyone who reported a Hausman
test of models with factor variables, or RESET with `rhs=True`.

**What to do.** Re-run. To get the old coding of a factor, drop the
incomplete rows yourself and set the reference with
`C(g, Treatment(reference=...))`.

## 1.38.0 → 1.39.0: ⚠️ `sp.stata` and extended missing values

**What changed.** `sp.stata` declines `x == .`, `x != .`, `x > .`,
`x <= .` and any comparison with `.a` to `.z` on a variable that may hold
an extended missing value, and `by`, `tabulate, missing` and
`collapse, by()` on such a variable when it has missing rows. In Stata
`.a != .` is true and each kind is a group of its own; the data hold all
kinds as NaN, so the old answers were those of another question.
`missing(x)`, `x < .` and `x >= .` run as before.

A variable counts as coded when the frame was read with
`sp.read_data(path, extended_missing='column')`, when
`df.attrs['_ext_missing']` lists it, when its value labels name such a
code, or once the session has assigned one (`replace x = .a if ...`,
`mvdecode x, mv(-1 = .a)`).

**What to do.** Write `missing(x)` or `x < .` where any missing value is
meant. Where the kind matters, recode it into a variable of its own before
calling `sp.stata`.
<a id="oct2026-qiu-jiaping-fixes"></a>

## 1.38.0 → 1.39.0: ⚠️ propensity score matching with a redundant covariate or tied controls; `replace` in `sp.stata`; two-step `heckman` with rho outside [-1, 1]

**Matching on an estimated propensity score.** Three things changed in
`sp.match`, `sp.psmatch2` and `teffects psmatch` run through `sp.stata`.

```python
sp.match(df, y="y", treat="d", covariates=[..., "u74", "u75"],  # u75 == u74
         distance="propensity", ties="all")
# before: scores off in the third decimal when the model also holds a
#         column such as re74**2; the ATT moved by up to a third
# now:    the redundant covariate is left out, as Stata's logit does
```

1. A covariate that is a linear combination of others no longer disturbs
   the fitted score. If your specification had one (the same dummy under
   two names, a full set of category indicators), rerun it.
2. With `ties='all'` (and `sp.psmatch2(ties=True)`) a control at exactly
   the same distance as the nearest one is always kept. About one distance
   in 600 was affected, so an estimate on data with many tied scores
   (discrete covariates) may move.
3. Scores are not clipped to [1e-6, 1 - 1e-6] before matching. This matters
   only where scores that extreme exist.

Estimates on specifications with none of the three are unchanged beyond
the last digits. Nothing in the call needs to change.

**`replace` in `sp.stata`.** A `replace` whose expression reads an earlier
row of the variable being replaced is now run row after row, as in Stata.

```python
sp.stata("""
    gen x = 0 in 1
    replace x = 0.4 * l.x + e in 2/l
""", data)
# before: x missing from the third row on
# now:    the AR(1) series
```

**`sp.heckman(method='twostep')` when rho falls outside [-1, 1].** The
standard errors are now computed with rho set to +/-1 and sigma to
`|lambda|` (Stata's rule), and an `AssumptionWarning` is raised.

```python
res = sp.heckman(df, y="y", x=["age", "educ"], z=["age", "educ", "married"])
res.model_info["rho"]            # before: -1.34; now: -1.0
res.model_info["rho_two_step"]   # -1.34
# standard errors: before a third too small; now Stata's
```

Coefficients do not change. Fits whose two-step rho is inside the interval
are untouched. If you see the warning, the model is the thing to look at:
usually the selection equation has no variable that is excluded from the
outcome equation.

**A running `sum()` under `if`, `in` or `by` in `sp.stata`.**

```python
sp.stata("gen s = sum(x) if d == 1", data)
# before: on a row with d == 1, the sum of x over every row above it
# now:    the sum of x over the rows with d == 1, as in Stata
```

A cumulative variable built this way and used later (a cumulative count
within a subsample, a running total by group under a condition) was
wrong. Rerun the snippet. `gen s = sum(x)` without a qualifier did not
change.

If a snippet relied on such a line, its later results were computed on
missing values and should be rerun. `clear all` now empties the data in
memory, so a snippet that simulates twice no longer carries the first
dataset into the second.

<a id="oct2026-facure-textbook-fixes"></a>

## 1.38.0 → 1.39.0: ⚠️ `sp.did` with repeated unit-period rows; categorical covariates; `sigma=` in power

**`sp.did` and `sp.callaway_santanna` with several rows per unit and
period.** The Callaway-Sant'Anna reshape used the first row of each (unit,
period) cell and ignored the others. Typical input: daily data with
`time='post'`.

```python
sp.did(df, y="y", treat="treated", time="post", id="unit")
# before: Callaway-Sant'Anna on the first pre day and the first post day
# now:    the 2x2 regression on every row, clustered on unit
sp.callaway_santanna(df, y="y", g="treated", t="post", i="unit")
# before: the same silent subset; now: MethodIncompatibility
```

If you ran either call on such data, rerun it. To keep Callaway-Sant'Anna,
pass the date as `t` and the first treated date as `g`, or average the
outcome within unit and period first. Panels with one row per unit and
period are unchanged.

**Categorical covariates.** A covariate of dtype `category`, a text column
or a `C(col)` entry is now expanded to indicator columns in the estimators
listed in the changelog. Only the `category` case changes a number that was
returned before: a `category` column with numeric levels used to enter as
one linear term.

```python
df["role"] = df["role"].astype("category")       # levels 1..5
sp.aipw(df, y="y", treat="d", covariates=["x", "role"])
# before: role as a number; now: four indicator columns
sp.aipw(df.astype({"role": int}), y="y", treat="d", covariates=["x", "role"])
# the old number
```

**`sigma=` in the power calculators.** `effect_size` is in the units of
the outcome and `sigma` is the outcome's standard deviation. Before,
`sigma` cancelled out of the formula, so a call with `sigma != 1` returned
the power of the standardised effect `effect_size` whatever `sigma` was.

```python
sp.power("rct", effect_size=0.08, sigma=0.2, power_target=0.8).n
# before: 4906 (the n for a standardised effect of 0.08); now: 197
```

Calls that leave `sigma` at 1 are unchanged. If you passed a standardised
effect together with `sigma`, drop `sigma`.

<a id="oct2026-barrett-textbook-fixes"></a>

## 1.38.0 → 1.39.0: ⚠️ Crump trimming, weighted balance statistics, minimal adjustment sets

**`sp.trimming(method='crump')`** and
**`sp.propensity_score(trimming='crump')`** drop more rows than before. The
old cutoff solved the wrong equation and was too small (0.064 against 0.072
on the example in the changelog). If an analysis was run on a Crump-trimmed
sample, rerun it. The new sample is a subset of the old one.
`method='sturmer'` is unchanged.

**`sp.balance_diagnostics`, `sp.ps_balance`, `sp.love_plot`.** Weighted
standardized differences and variance ratios change slightly because the
weighted variance now has the divisor `sum(w) - sum(w^2) / sum(w)`. The
change is largest when a few rows carry most of the weight. Raw columns,
weighted means and KS statistics are unchanged. No option returns the old
numbers, since they were not invariant to equal weights.

**`DAG.adjustment_sets(minimal=True)`** may return more sets than before,
because minimal sets larger than the smallest one are now included. Every
set returned before is still returned.

**`sp.from_r`.** The payload for `matchit(...)` no longer has `y` (it used
to repeat the treatment there); it has `missing_arguments: ['y']`, and for
`method = "nearest"` it now carries `replace`, `n_matches` and `estimand`.
Any payload may carry `untranslated_arguments`. Code generated earlier from
`lm(..., weights =)` ran unweighted, and code generated from
`did::att_gt(...)` used `base_period='universal'` and ignored
`control_group`, `xformla` and `anticipation`; regenerate both. The
`att_gt` payload now always has `control_group` and `base_period`.

**`sp.contrast`** on a frame in which the variable takes a single value now
raises. Pass the full data and `subset=` to average over a subpopulation.

## 1.38.0 → 1.39.0: ⚠️ `sp.cox` robust standard errors under tied event times
<a id="oct2026-linear-model-fixes"></a>

**What changed.** With `ties='efron'` (the default), `sp.cox(robust=...)`
and `sp.cox(cluster=...)` built the sandwich from Breslow's score
residuals. They now use Efron's, as `survival::coxph` and Stata `stcox,
efron` do. Robust and clustered standard errors change wherever two or
more events share a time; they are identical when no event times are tied.

**How much.** Up to 2% on a trial with daily relapse times (112 distinct
days, 800 events). Nothing changes for coefficients, hazard ratios,
model-based standard errors, or fits with `ties='breslow'`.

**What to do.** Rerun any Cox model reported with robust or clustered
standard errors on data with ties. There is no switch for the old numbers;
they did not correspond to either tie rule.

## 1.38.0 → 1.39.0: fitted values of zero-inflated models moved from `diagnostics` to `data_info`

`sp.zip_model`, `sp.zinb` and `sp.hurdle` stored one fitted value per
observation, three times over, in `result.diagnostics`, so every summary
and JSON export of the fit carried them.

```python
fit.diagnostics["predicted_overall"]   # still works, DeprecationWarning; removed in 1.41
fit.data_info["predicted_overall"]     # where it lives now
```

`for key in fit.diagnostics` and `fit.diagnostics.items()` no longer
return those keys. Scalars (`ll`, `aic`, `bic`, `vuong_stat`, ...) are
unchanged, and `Log-Likelihood`, `AIC`, `BIC` are added.

## 1.38.0 → 1.39.0: `sp.kaplan_meier` asks for `conf_type`; the default changes in 1.40

**Done in 1.40.** The default is now `conf_type="log-log"` and the
`DeprecationWarning` is gone; pass `conf_type="plain"` for the earlier
interval. The text below is the 1.39 announcement.

`sp.kaplan_meier` builds its confidence interval as `S(t) +/- z se`, cut
off at 0 and 1. From 1.40 the default is the log-log interval, the one
Stata's `sts list` prints, which never leaves (0, 1). The survival
estimate, its standard error and the median are the same either way.

```python
sp.kaplan_meier(df, "time", "event")                        # 'plain' today, warns
sp.kaplan_meier(df, "time", "event", conf_type="plain")     # today's interval, for good
sp.kaplan_meier(df, "time", "event", conf_type="log-log")   # the 1.40 default (Stata)
sp.kaplan_meier(df, "time", "event", conf_type="log")       # R survfit's default
```

Nothing changes in 1.39 except the `DeprecationWarning`. To keep reported
intervals fixed across the change, pass `conf_type` explicitly now.

## 1.38.0 → 1.39.0: `sp.regress` accepts regressions through the origin it used to refuse

`sp.regress("y ~ 0 + x + I(1 - x)", df)` and `sp.regress("y ~ 0 + one +
x", df)` raised `NumericalInstability`. They fit now. A model with an
intercept behaves as before, and proportional columns (`x` and `2 x`) are
still caught in a model without one.

## 1.38.0 → 1.39.0: ⚠️ period dummies in `sp.xtabond` / `sp.xtdpdsys`
<a id="oct2026-hansen-dynpanel"></a>

**What changed.** With `time_dummies=True` the dummies of periods that have
no row in the transformed equation are no longer regressors. When the model
has a constant (`sp.xtdpdsys`, or `constant=True`), the last remaining dummy
is also omitted from the regressors and stays among the instruments. This
is the rule of Stata's `xtdpd`. The coefficient table is shorter and the
other coefficients change: on `abdata` with two lags the first lag in a
one-step system GMM moves from 1.1548 to 1.1665, Stata's value.

**Who is affected.** Calls with `time_dummies=True`. The old fit emitted
"GMM weight matrix Z'HZ is singular" in the affected cases. Fits without
period dummies, and fits where you pass your own dummies as regressors, are
unchanged.

**What to do.** Rerun. Code that looks up `_T<period>` rows of
`result.detail` should expect fewer of them.

A two-step fit whose weight matrix is singular to rounding now warns. It
used to return unusable standard errors without a warning. A singular
weight, whether detected before or now, gets the generalized inverse Stata
takes (Mata's `invsym`) in place of the Moore-Penrose inverse, so the
two-step estimates of such a fit change and now agree with `xtdpd` and
`xtabond2`. Fits that did not warn about a singular weight and did not have
standard errors of order 1e14 are unchanged.

## 1.38.0 → 1.39.0: ⚠️ random effects with a regressor that is constant within unit
<a id="oct2026-hansen-fixes"></a>

**What changed.** `sp.panel(method='re')` computes the within residual
variance with `N - G - r` degrees of freedom, `r` being the number of
regressors that vary within unit. It used `N - G - K` with `K` the number
of regressors, which counts a time-invariant regressor as a parameter of
the within regression although that regression cannot estimate it. Stata's
`xtreg, re` and R's `plm` count the rank. `method='mundlak'` and
`'chamberlain'` use the same variance components.

**Who is affected.** Random-effects and correlated-random-effects fits that
include at least one regressor with no within-unit variation, which is the
usual reason to choose random effects. `sigma_e` was too large by the
factor `sqrt((N - G - r) / (N - G - K))`, and `sigma_u`, `theta`, the
coefficients and the standard errors moved with it. On a panel of 40 units
and 300 rows with two such regressors a slope went from 1.09305 to
1.09251. Fits in which every regressor varies within unit are unchanged.

**What to do.** Re-run. No argument reproduces the old numbers.

## 1.38.0 → 1.39.0: ⚠️ Dickey-Fuller p-values above the fitted range are 1

**What changed.** `sp.unitroot` (`test='adf'`, `test='pp'`), the
Fisher-type panel unit-root tests and anything else that calls
`mackinnon1994_pvalue` return a p-value of 1 when the statistic is above
the largest value MacKinnon's response surface was fitted on: 2.74 with a
constant, 0.70 with a constant and a trend. The approximation was
evaluated outside that range before, where its cubic bends down.

**Who is affected.** Tests on series that are explosive or trend upward
faster than a random walk, where the statistic is positive. With a trend
the old p-value was 0.99 at +1.5, 0.77 at +2.5, 0.25 at +3.0, 0.06 at +3.25
and 0.00 at +4.0, so such a series could be declared stationary. Negative
statistics, which is where unit-root tests are usually read, are unchanged.

**What to do.** Re-run any test whose statistic was positive.

## 1.38.0 → 1.39.0: ⚠️ `sp.rdrandinf` / `sp.rdwinselect` / `sp.rdsensitivity` with `p > 0` use HC3

`rdlocrand` 3.0 (2026-10-04) moved the variance behind its large-sample
test for polynomial-adjusted statistics from HC2 to HC3 and added a `vce`
argument. StatsPAI follows the reference. Large-sample p-values, standard
errors and the default grid of `ci=True` change for calls with `p > 0`.
Calls with `p = 0` are unchanged.

```python
sp.rdrandinf(df, y="y", x="x", wl=-1, wr=1, p=1)              # HC3 now
sp.rdrandinf(df, y="y", x="x", wl=-1, wr=1, p=1, vce="hc2")   # as before
```

## 1.38.0 → 1.39.0: ⚠️ `sp.arima` estimates a constant when the series is not differenced
<a id="oct2026-syllabus-fixes"></a>

**What changed.** `sp.arima(y, order=(p, 0, q))` with the default
`method='statespace'` used to fit a zero-mean model. It now estimates the
mean (`const` in `params`), as R's `stats::arima`, statsmodels' `ARIMA`
and Stata's `arima` do. A new `trend=` argument controls it: `None` (the
default) is a constant without differencing and none with it, `'c'` asks
for the constant (a `drift` once the series is differenced), `'n'` for
none. `method='innovations_mle'` already estimated the constant.

**Who is affected.** Anyone who fitted an undifferenced series whose mean
is not zero with the default method. The autoregressive coefficients were
biased towards one (0.763 against 0.619 on the differenced log wholesale
price index). Series that were demeaned first, and every model with
`d >= 1`, give the same numbers as before; they gain no parameter.

**What to do.** Re-run. `sp.arima(y, order, trend='n')` reproduces the old
fit. `params` has one more entry (`const`) in the affected case, so code
that indexes it by position should index by name.

## 1.38.0 → 1.39.0: ⚠️ `sp.tab` on a 2 x 2 table reports Pearson's chi-squared

**What changed.** The statistic `sp.tab(df, a, b)` prints as
"Pearson chi2" on a 2 x 2 table is now Pearson's. It was Yates's
continuity-corrected statistic, which is smaller. Tables with more than
one degree of freedom, and Fisher's exact p-value, are unchanged.

**Who is affected.** Anyone who reported the chi-squared statistic or its
p-value from `sp.tab` on a 2 x 2 table.

**What to do.** Re-run. The new number is the one Stata's
`tabulate, chi2` prints. If the corrected statistic is wanted, call
`scipy.stats.chi2_contingency(table)` directly.

## 1.38.0 → 1.39.0: ⚠️ Stata translation of `tobit`

**What changed.** `sp.stata` / `sp.from_stata` write both censoring limits
out. `tobit y x, ul(2)` is now `sp.tobit(..., ll=None, ul=2.0)`; it used to
leave `ll` at `sp.tobit`'s default of 0 and so censored from below as
well. `tobit y x` with no limit is the uncensored model, as in Stata.
`tobit y x, ll` (censor at the observed minimum) and a limit given as a
variable are refused instead of falling back to zero.

**Who is affected.** Results obtained by running `tobit` through
`sp.stata` without `ll()`. Commands with `ll(#)` are unchanged, and so are
direct calls to `sp.tobit`, whose default stays `ll=0`.

**What to do.** Re-run the command.

## 1.38.0 → 1.39.0: ⚠️ two-step `sp.heckman` standard errors; `sp.garch(q=0)`

**What changed.**

- `sp.heckman(method='twostep')` standard errors change in the fourth
  digit. The first-step probit now enters through its observed
  information, as in Stata and R `sampleSelection`.
- `sp.garch(p, q=0)` raises `MethodIncompatibility`. The model is not
  identified and the numbers it returned were arbitrary.

**Who is affected.** Two-step Heckman standard errors quoted to four
digits; any GARCH fit with `q=0`.

**What to do.** Re-run. An ARCH(1) model is `sp.garch(y, p=0, q=1)`.

## 1.38.0 → 1.39.0: ⚠️ Stata translation of `ivregress gmm`

`sp.stata("ivregress gmm y x (d = z1 z2)", data=df)` and the code
`sp.from_stata` prints for it change. Stata's `ivregress gmm` uses the
robust weight matrix by default; the old translation used the unadjusted
one, so its coefficients were not Stata's (0.1608 against 0.1588 for
schooling on Card's data), and `wmatrix()` was dropped.

| Stata | Before | Now |
| --- | --- | --- |
| `ivregress gmm ...` | `sp.ivreg(..., method='gmm')` | `sp.iv(..., method='gmm', robust='hc1', small=False)` |
| `..., small` | `sp.ivreg(..., method='gmm')` | `sp.iv(..., method='gmm', robust='hc1')` |
| `..., wmatrix(cluster c)` | `wmatrix` untranslated | `sp.iv(..., method='gmm', cluster='c', small=False)` |
| `..., wmatrix(unadjusted)` | `wmatrix` untranslated | `sp.iv(..., method='gmm', gmm_vcov='efficient', small=False)` |

What to do: rerun any `ivregress gmm` line you replayed through
`sp.stata`. Direct calls to `sp.iv(method='gmm')` and
`sp.ivreg(method='gmm')` are unchanged; `small=False` is newly accepted
there.

## 1.38.0 → 1.39.0: ⚠️ `sp.synth(method='classic').ci` is the interval its p-value implies

`ci` changes for every classic fit with placebos. `estimate`, `se`,
`pvalue` and the weights do not.

| | through 1.38.0 | now |
| --- | --- | --- |
| `ci` | `estimate -/+ z * sd(placebo ATTs)` | constant effects not rejected by the RMSPE-ratio rank test |
| symmetric around the estimate | yes | no |
| agrees with `pvalue` | not necessarily | zero is outside exactly when `pvalue < alpha` |
| fewer than `1 / alpha - 1` donors | finite | `(-inf, inf)` |

The old interval is `result.model_info['ci_normal']`. If a table or a
plot needs finite bounds with few donors, read that key, or raise
`alpha` (`alpha=0.1` needs 9 donors, `0.05` needs 19).
`model_info['ci_method']` is `'placebo_inversion'`, or `'placebo_sd'`
when the old interval is still what `ci` holds (no usable placebo, a
perfect pre-treatment fit, or every constant effect rejected). Other
`method=` values are not affected.

## 1.38.0 → 1.39.0: ⚠️ `sp.fisher_exact(statistic='ks' | 'rank_sum' | 't').ci` is an interval for the effect

For these statistics `FisherResult.ci` used to be the 2.5% and 97.5%
percentiles of the null distribution of the statistic. It is now the set
of constant effects the test does not reject, in outcome units, like the
interval for `statistic='ate'`. Code that read the two numbers as
critical values should take `np.percentile(result.perm_dist, [2.5, 97.5])`.
`statistic`, `p_value` and `p_one_sided` are unchanged.

## 1.38.0 → 1.39.0: ⚠️ `sp.ipw` Horvitz-Thompson ATT / ATC; the interval of `sp.fisher_exact`

**`sp.ipw(estimand='ATT', normalize=False)`** divided the weighted sums by
`n`. The effect on the treated needs the number of treated, so the result
was `P(T=1)` times the ATT (and `P(T=0)` times the ATC for
`estimand='ATC'`). Both are now divided by the size of their target group.
If you reported an unnormalised ATT or ATC, rerun it; the old number
divided by the treated (control) share is the new one. Nothing changes
under the default `normalize=True`, for the ATE, or for
`se_method='sandwich'`, which requires normalised weights.

**`sp.fisher_exact(...).ci`** is now the exact inversion of the
randomization test and is wider than before (on the LaLonde experiment
[572, 3015] against [999, 2589]). The old interval came from a coarse grid with 500 draws per point
and stopped at the last grid point inside. With `controls` the ends also
move because the shifted outcome is now residualized. p-values and the
statistic are unchanged. A design with too few assignments to reject at
`alpha` now returns `(-inf, inf)` with a warning where it used to return
the edges of the grid.

<a id="oct2026-the-effect-textbook-fixes"></a>

## 1.38.0 → 1.39.0: ⚠️ formulas on `int8` / `int16` / `int32` columns

**What changed.** Integer columns narrower than 64 bits are widened to
`int64` before a formula is evaluated. Before, `I(x**2)`, `I(x*z)` and any
other arithmetic written inside a formula was computed in the column's own
type and wrapped around without a warning: above 181 for `int16`, above 11
for `int8`, above 46,340 for `int32`.

**Who is affected.** Fits whose formula does arithmetic on a column that
came from `pd.read_stata`, a parquet file or an explicit `astype('int16')`
and holds values large enough to overflow. The functions are `sp.regress`,
`sp.feols`, `sp.fepois`, `sp.feglm`, `sp.glm`, `sp.logit`, `sp.probit`,
`sp.poisson`, `sp.ivreg`, `sp.panel`, the covariate formulas of the DiD
family and the spatial models. The old numbers were
wrong; there is nothing to reproduce them with.

**What to do.** Rerun. No argument changes.

**Unaffected.** Frames whose integers are `int64` (the pandas default for
data built in Python or read from CSV) or float; formulas without
arithmetic; factor terms, whose level names are unchanged (`C(g)[T.2]`).

## 1.38.0 → 1.39.0: ⚠️ `sp.match(method='cem')` coarsening

**What changed.** Two conventions now follow the `cem` packages for R and
Stata. The default number of bins is one fewer: Sturges' rule,
`ceil(log2(n) + 1)`, counts cut points from the minimum to the maximum.
And a bin is closed on the right, `(a, b]`, where it used to be closed on
the left.

**Who is affected.** Calls without `n_bins=` (other strata, other matched
sample, other estimate). Calls with `n_bins=` whose covariates take values
that fall exactly on a bin edge, which in practice means integer-valued
covariates.

**What to do.** Rerun. To get the old number of bins back pass
`n_bins=int(np.ceil(np.log2(n) + 1))`; the old closure is not available.

**New.** `n_bins` also takes a list (one count per covariate) or a dict
from covariate to a count or to cut edges.

## 1.38.0 → 1.39.0: ⚠️ `sp.ebalance` with redundant moments; `sp.sensemakr` benchmarks

**What changed.** `sp.ebalance(moments=2)` (or 3) on a covariate whose
higher moment repeats a lower one, such as a 0/1 indicator, used to fail
to converge and return uniform control weights with a warning. The
redundant moment is now left out and the remaining ones are balanced
exactly. `sp.sensemakr(benchmark=[...])` raises on a name that is not among
`controls`; it used to drop the row.

**Who is affected.** `sp.ebalance` calls with `moments >= 2` and a binary
covariate: the estimate changes from the unweighted difference in means to
the entropy-balanced one. `sp.sensemakr` calls with a misspelt benchmark.

**What to do.** Rerun `sp.ebalance`. Correct the benchmark name.

## 1.38.0 → 1.39.0: ⚠️ Stata translations of factor-variable notation

**What changed.** In `sp.stata` and `sp.from_stata`:

- `a##b##c` is the full factorial (it lacked the two-way terms);
- a variable without a prefix inside `#` or `##` is a factor, as in Stata
  (`reg y d##c.x` now gives `C(d) + x + C(d):x`);
- `a#b` without the main effects is refused (it was the product of the two
  columns);
- `collapse` stores means, medians and sds in single precision when Stata
  does.

**Who is affected.** Translations of commands with three-way `##`, with an
unprefixed variable of more than two levels in an interaction, or with
`a#b` alone. Coefficient names change from `d` to `C(d)[T.1]` for an
unprefixed 0/1 variable inside an interaction; the fit is the same.

**What to do.** Rerun. For `a#b` write `a##b`, or build the cell variable
with `egen cell = group(a b)` and use `i.cell`.

---

<a id="oct2026-wooldridge-textbook-fixes"></a>

## 1.38.0 → 1.39.0: ⚠️ DFBETAS, `predict` after `sp.glm`, and the Stata translation of `heckman`

**What changed.**

1. `sp.estat(result, 'leverage')['dfbetas']` is now DFBETAS as Belsley, Kuh
   and Welsch define it and as Stata's `dfbeta` reports it. Column `j` used
   to lack the division by `sqrt((X'X)^-1_jj)`. `dfbetas_flagged_obs`
   changes with it.
2. `result.predict(new_data)` after `sp.glm` returns the predicted mean. It
   used to return the linear index `x'b`, while `result.predict()` returned
   the mean.
3. `sp.from_stata` and `sp.stata` translate `heckman y x, select(...)` to
   `sp.heckman(..., method='ml')`, and to `method='twostep'` when the
   command says `twostep`. Both used to run the two-step estimator.

**Who is affected.** Code that reads `dfbetas` or the list of flagged
observations; code that predicts on new data after `sp.glm` with a log,
logit, probit or other non-identity link; do-files with `heckman` run
through `sp.stata`.

**What to do.** Rerun. For the old `sp.glm` behaviour ask for it by name:
`result.predict(new_data, what='link')`. Nothing changes for a Gaussian
identity fit, for in-sample predictions, for leverage and Cook's distance,
or for direct calls to `sp.heckman`.

<a id="oct2026-dcdh-textbook-fixes"></a>

## 1.38.0 → 1.39.0: ⚠️ `sp.did_multiplegt` with a non-binary treatment

**What changed.** Each switcher now enters with the sign of its own
treatment change, and the sum is divided by the total absolute change, so
the estimate is an effect per unit of treatment. Every switcher from a
baseline other than 0 used to be signed as a switch off, and nothing was
divided.

**Who is affected.** Calls where the treatment takes values other than 0
and 1. The old numbers were wrong: -0.00082 against 0.0057791 on the
newspaper panel of Gentzkow, Shapiro and Sinkinson (2011).

**What to do.** Rerun.

**Unaffected.** Binary treatments. `placebo_sign` keeps its default; its
description is corrected (`'r'` is the sign of a plain
`did_multiplegt_old` run, `'stata'` the sign of `did_multiplegt_old,
robust_dynamic`).

## 1.38.0 → 1.39.0: ⚠️ `sp.sun_abraham`: cohort shares and `event_window`

**What changed.** (1) The interaction weights are the cohorts' shares of
the observations at each relative time, not of the units. (2) With
`event_window=`, the regression stays saturated in every relative time and
the window only selects what is reported and aggregated. The new argument
`window_rule` chooses between that (`'report'`), pooled ends (`'bin'`) and
the previous behaviour (`'reference'`).

**Who is affected.** (1) Unbalanced panels, where the two shares differ.
(2) Every call with an `event_window` narrower than the data. The previous
estimates measured each coefficient against a reference that contained the
treated observations outside the window.

**What to do.** Rerun. Pass `window_rule='reference'` only to reproduce an
earlier number.

**Unaffected.** Balanced panels without `event_window`.

## 1.38.0 → 1.39.0: ⚠️ `sp.did_multiplegt_dyn`: `same_switchers`, `aggregation='switchers'`, `controls=`, joint tests

**What changed.** (1) `same_switchers=True` keeps the switchers whose
effect is estimable at every requested horizon. It kept those merely
observed there. (2) `aggregation='switchers'` divides by the treatment
changes in place at each horizon. (3) `controls=` fits its regression on
the never-switchers too, weighted when `weights=` is given and within
`trends_nonparam` cells, no longer uses a change across a hole in the
panel as a one-period change, and its analytic variance accounts for the
estimation of the slopes. (4) With `se_method='analytic'` the joint
tests are computed from the analytic covariance and `n_boot` is ignored;
`joint_effects_test` is new. (5) A non-binary treatment is estimated
instead of refused.

**Who is affected.** (1) Panels where some switchers have no control at
the longest horizon, which includes every staggered panel without
never-treated groups. (2) Treatments that return to their starting value
within the horizons. (3) Every fit with `controls=`: the analytic
standard errors move by a few tenths of a percent, and the estimates move
on panels with never-switchers, missing periods, weights or
`trends_nonparam`. (4) Analytic fits that read `joint_placebo_test`,
`joint_overall_test` or `effects_equal_test`: the p-values now come from
the same variance as the standard errors, and an analytic fit with
`n_boot=0`, which used to return `None` for them, returns the test.

**What to do.** Rerun. Code that relied on a `ValueError` for a non-binary
treatment should check the treatment itself.

**Unaffected.** Effects, placebos and standard errors of fits without
these options; bootstrap fits.

## 1.38.0 → 1.39.0: ⚠️ `regress, vce(hc2 clustvar, dfadjust)` in `sp.stata`

**What changed.** The command translates to `sp.regress(..., vce='cr2',
cluster='clustvar', dfadjust=True)`. It used to translate to
`robust='hc2'`, without the cluster and without the adjusted degrees of
freedom.

**Who is affected.** Anyone who ran such a line through `sp.stata` or
`sp.from_stata`. With a cluster variable the standard errors change; with
`dfadjust` the p-values and intervals do.

**What to do.** Rerun.

<a id="oct2026-causalml-textbook-fixes"></a>

## 1.38.0 → 1.39.0: ⚠️ `sp.dml` PLR / PLIV with a classifier nuisance

**What changed.** With `model='plr'` or `model='pliv'`, a nuisance learner
that is a scikit-learn classifier now contributes `predict_proba[:, 1]`.
It used to contribute `predict`, the hard 0/1 label, so the residual of a
binary treatment was `D - 1{p > 0.5}` and not `D - E[D | X]`. A classifier
whose target is not a 0/1 variable is refused. `sp.dml_model_averaging`
follows the same rule.

**Who is affected.** Calls that pass a classifier object (for example
`ml_m=RandomForestClassifier()` or `LogisticRegression()`) to PLR or PLIV.
Estimates and standard errors change, and the old ones were wrong: on the
401(k) data the PLR estimate was 6,044 against 8,754 from `DoubleML`.

**What to do.** Rerun. No argument changes.

**Unaffected.** Regressors, the string aliases (`'rf'`, `'lasso'`, ...),
which resolve to regressors under PLR and PLIV, and the `irm` / `iivm`
models, which already used probabilities.

## 1.38.0 → 1.39.0: ⚠️ `sp.dml_sensitivity` on IRM fits, and `rv_qa`

**What changed.** (1) For `model='irm'` the bias bound is computed from
`sigma^2 = E[(Y - g(D, X))^2]` and the Riesz representer of the estimand,
as in Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022). It used
the PLR formula on the IRM score. (2) `rv_qa` is the strength at which the
`1 - alpha` interval for the bound reaches the null, with the bound's own
standard error. It used `|theta| - z * se` with the unadjusted `se`.
(3) New fields `se_low`, `se_high`, `ci_low`, `ci_high`.

**Who is affected.** Anyone who reported `bias_bound`, the adjusted range
or a robustness value from an IRM fit: the old numbers were not the bound
of the paper. For PLR fits `bias_bound`, the adjusted range and `rv_q` are
unchanged, and `rv_qa` moves in the third decimal.

**What to do.** Rerun. Sample-weighted IRM fits are refused, because the
score elements are not stored for them; refit without `sample_weight`.

**Unaffected.** `sp.sensemakr`, `sp.evalue` and the other sensitivity
tools.

## 1.38.0 → 1.39.0: ⚠️ `sp.twfe_decomposition` returns the TWFE coefficient and its exact decomposition

**What changed.** `sp.twfe_decomposition` described itself as the
Goodman-Bacon (2021) decomposition with de Chaisemartin-D'Haultfoeuille
(2020) weights, and computed neither. Its 2×2 rows were weighted by the
number of units in each comparison, the timing comparisons used every
period instead of the window in which the comparison is clean, and the
"dCDH" weights came from a simplified formula that did not sum to one.
The rows therefore did not add up to the coefficient they claimed to
decompose. On a noiseless three-unit panel whose pieces can be worked out
by hand (27/11 for the coefficient; weights 7/22, 8/22, 3/22, 4/22) the
function returned 1.839 with four weights of 0.25.

It now returns

- `estimate`: the TWFE coefficient (it was the unit-weighted average of
  the 2×2 rows);
- `se`: the unit-clustered standard error of that coefficient, with the
  small-sample factor of Stata `xtreg, fe vce(cluster)` (it was the
  spread of the rows around their average, zero with one comparison);
- `detail`: the rows of `sp.bacon_decomposition`, which sum to
  `estimate` exactly. The `n_units` column is gone;
- `model_info['dcdh_weights']`: the dCDH weights summed over the units of
  each cohort and period, with `n_negative`; they sum to one;
- `model_info['n_negative_weights_dcdh']`: counted over unit × period
  cells, as R `twowayfeweights` counts them.

**Who is affected.** Anyone who reported `estimate`, `se`, the weights in
`detail` or the dCDH weights from this function. `model_info['twfe_beta']`
was right and is unchanged.

**What to do.** Rerun. On an unbalanced panel the Goodman-Bacon theorem
does not hold: the coefficient and the dCDH weights are still returned,
`detail` is empty and a warning says so (it used to return rows computed
on the unbalanced data).

## 1.38.0 → 1.39.0: ⚠️ propensity-score matching with tied scores; Stata translations of `teffects ipw` / `teffects aipw` / `boottest`

**Matching.** `sp.match(distance='propensity', ties='all')`, and the
functions built on it, decide ties by equality of the fitted score. Two
units with the same covariates have the same score, but the matrix
product that computed it could round them one unit in the last place
apart, and then only one of several identical controls was matched. The
score is now computed once per distinct covariate row. Estimates change
only where the sample has units that share every covariate (discrete
covariates): on the Lalonde data with age, education and four indicators
the ATET moves from 1199.666 to 1209.718, which is what Stata `teffects
psmatch` reports. Nothing to do but rerun; the earlier number also
depended on the order of the rows.

**Translations.** Through `sp.stata` and `sp.from_stata`:

| Stata | ran before | runs now |
| --- | --- | --- |
| `teffects ipw (y) (d x)` | `sp.ipw(...)`, bootstrap standard error | `sp.ipw(..., se_method='sandwich')`, the standard error Stata prints |
| `teffects aipw (y x) (d x)` | `sp.aipw(...)`, five-fold cross-fitting | `sp.aipw(..., cross_fit=False, se_method='sandwich')` |
| `teffects aipw ..., atet` | `sp.aipw(estimand='ATT')` | refused: `teffects aipw` has no `atet` |
| `boottest x, reps(999)` | `TypeError` | `sp.wild_cluster_boot(result, variable='x', n_boot=999)` |
| `psmatch2 d x, out(y)` | refused (probit not available) | `sp.psmatch2(..., ps_model='probit')` |

The point estimates of `teffects ipw` were right before; its standard
error and everything about `teffects aipw` were not Stata's. The direct
calls `sp.ipw` and `sp.aipw` keep their defaults.
## 1.38.0 → 1.39.0: ⚠️ `sp.survreg` now honours `robust=` and `cluster=`; zero durations are refused

**What changed.** `sp.survreg(..., robust='robust')` and
`sp.survreg(..., cluster='id')` used to return the same standard errors as
the plain call. They now return the sandwich and the cluster sandwich
(Stata's `streg, vce(robust)` / `vce(cluster id)`). Separately, a duration
that is zero or negative raises `MethodIncompatibility`; before, `1e-15`
was added to every duration.

**Who is affected.** Anyone who reported standard errors from
`sp.survreg` with `robust=` or `cluster=`: those numbers were
observed-information standard errors. Point estimates are unchanged
beyond the sixth digit.

**What to do.** Re-run. To get the old standard errors on purpose, drop
the option. Rows with a duration of zero have to be dropped or the time
origin shifted before the call.

## 1.38.0 → 1.39.0: ⚠️ the Vuong statistic of `sp.zip_model` and `sp.zinb` is smaller

**What changed.** `result.diagnostics['vuong_stat']` and `['vuong_p']`
compare the zero-inflated fit with the plain Poisson (for `sp.zinb`, the
plain negative binomial) regression. The comparison model used to be
evaluated at the zero-inflated model's count coefficients. It is now
fitted on its own by maximum likelihood, which is what the test is
defined on and what R `pscl::vuong` computes. The old statistic was too
large in favour of the zero-inflated model.

**Who is affected.** Anyone who reported that statistic. Coefficients,
standard errors, the log-likelihood, AIC and BIC do not change.

**What to do.** Re-run and report the new value. `sp.vuong(fit_a, fit_b)`
gives the same statistic for any two fitted count models, with AIC and
BIC corrections. Do not read it as a test for zero inflation: a
zero-inflated model and its plain counterpart are nested on a boundary,
and the statistic is not standard normal under that null (Wilson 2015,
doi:10.1016/j.econlet.2014.12.029).

## 1.38.0 → 1.39.0: ⚠️ `sp.rdrandinf(statistic='ksmirnov')` large-sample p-value with tied outcomes

**What changed.** `model_info['pvalue_asymptotic']` for the
Kolmogorov-Smirnov statistic is now the exact conditional p-value given
the ties in the data (below 10,000 treated-control pairs; Kolmogorov's
limit above), which is what R's `ks.test` and therefore `rdlocrand`
report. It used to come from a formula that assumes a continuous outcome.

**Who is affected.** Calls with `statistic='ksmirnov'` or `'all'` on an
outcome or covariate with tied values, binary variables above all. On a
binary covariate of the Senate data the old value was 0.965 and the new
one is 0.356. Without ties nothing changes, and the randomization p-value
(`pvalue`) was never affected.

**What to do.** Re-run if you reported that number.

## 1.38.0 → 1.39.0: `sp.rdrobust` checks for manipulation with `sp.rddensity`

**What changed.** The density check that `sp.rdrobust` runs by default
(`manipulation_test=True`) used McCrary's binned test and now uses the
local-polynomial density test of `sp.rddensity`. The point estimate,
standard error and interval do not change.

**Who is affected.** Code that reads `model_info['mccrary']['pvalue']`,
`result.violations()` or the `AssumptionWarning`: the p-value is a
different test's, so a fit may warn where it did not, or stop warning.
With a discrete or heaped running variable the old check raised false
alarms (43% of placebo cutoffs on one such dataset, against 5% now).

**What to do.** Nothing, unless a stored p-value is compared across
versions. `model_info['mccrary']['test']` is `'rddensity'` from this
version on. For McCrary's test call `sp.mccrary_test` directly.

## 1.38.0 → 1.39.0: ⚠️ `sp.rdrandinf` / `sp.rdrbounds` take the window's endpoints, not offsets

**What changed.** `wl` and `wr` are the left and right ends of the window
on the scale of the running variable, as in R / Stata `rdrandinf`. They
used to be added to the cutoff.

**Who is affected.** Calls with `c != 0`. With `c = 0` nothing changes.

**What to do.** For a window of half-width `w` around cutoff `c`, pass
`wl=c - w, wr=c + w`. A window that does not contain the cutoff now raises
`ValueError`, so an offset-style call with a cutoff far from zero fails
loudly. The case that changes silently is a cutoff smaller in magnitude
than the window (`c=0.5, wl=-1, wr=1` was `[-0.5, 1.5]` and is now
`[-1, 1]`); check any such call. `model_info['window']` holds the window
actually used.

## 1.38.0 → 1.39.0: ⚠️ `sp.rdrandinf(p > 0)` is the difference in side-specific fits

**What changed.** With `p > 0` the outcome used to be residualized on one
polynomial in the score fitted across both sides, and the difference in
means of the residuals was reported. Inside the window the score is
collinear with treatment, so that polynomial absorbed most of the effect.
The statistic is now the difference between the two sides' fitted values
at the cutoff, each side with its own polynomial, which is what
`rdlocrand` computes. On the U.S. Senate data in a ±2.5 window: 2.90
before, 13.26 now and in R.

The randomization p-value for `p > 0` permutes outcomes against (score,
assignment) pairs. This is not what `rdlocrand` does, and its p-values for
`p > 0` are much smaller; see the function's Notes for the reason and the
size simulation behind the choice. The large-sample p-value
(`model_info['pvalue_asymptotic']`) agrees with R.

**Who is affected.** Every call with `p >= 1`. Estimates were biased
toward zero.

**What to do.** Re-run. `statistic='ksmirnov'`, `'ranksum'` and `'all'`
with `p > 0` or a non-uniform kernel now raise instead of running on
residuals.

## 1.38.0 → 1.39.0: ⚠️ `sp.rdrandinf(fuzzy=)` tests the reduced form

**What changed.** `pvalue` used to come from permuting the instrument and
recomputing the Wald ratio. Most permuted first stages are near zero, so
the reference distribution was dominated by exploded ratios and the
p-value was close to uninformative (0.863 on an example where the 2SLS
p-value is 0.038). Now:

- `fuzzy_stat='itt'` (default): `pvalue` is the randomization p-value of
  the intention-to-treat effect, the Anderson-Rubin logic. It is a valid
  finite-sample test that the effect on compliers is zero.
- `fuzzy_stat='tsls'`: `pvalue` is the large-sample z-test of the Wald
  ratio, and `model_info['pvalue_permutation']` is NaN.

`estimate` is still the Wald ratio. `se` is now the heteroskedasticity-
robust 2SLS standard error (it was a delta-method approximation), and the
interval is `estimate ± z * se` unless a grid is passed through `ci=`.
The reduced-form estimate is in `model_info['itt']`.

**Who is affected.** Every call with `fuzzy=`. Conclusions drawn from the
old p-value should be revisited. `p > 0` or a kernel with `fuzzy=` now
raises.

## 1.38.0 → 1.39.0: ⚠️ `sp.rdwinselect` builds the windows the procedure describes

**What changed.** The default window sequence, the p-values and the
columns.

- Windows: the first is the smallest symmetric window with `obsmin=10`
  observations on each side; each later one adds at least `wobs=5` on
  each side (or `wstep=` in score units). The old default started at a
  fraction of the score's range and ended at its full extent.
- `p_value` is the smallest balance p-value across the covariates, from
  1,000 randomization draws (it was 500), and `variable` names the
  covariate. `binom_pvalue` is the binomial test of the split. Without
  covariates `p_value` is NaN; it used to be computed on quantile dummies
  of the score.
- `balanced` is `p_value >= alpha` (it was `>`), and the recommended
  window, in `attrs['recommended_window']`, is the largest window such
  that it and every smaller one are balanced.
- Rows with a missing covariate are dropped inside each window, not
  before the windows are built; pass `dropmissing=True` for the old
  order. The counts are of the rows the tests used.

**Who is affected.** Every call that relied on the default sequence. A
call with both `wmin=` and `wstep=` keeps its windows and counts.

**What to do.** Re-run. For reproducible, fast p-values pass
`approx=True`.

## 1.38.0 → 1.39.0: ⚠️ `sp.rdmc(cutoff_var=)` reports robust inference

**What changed.** For each cutoff, `se`, `ci_lower`, `ci_upper` and
`p_value` are now the robust bias-corrected ones that `rdmulti::rdmc`
prints; `estimate` is still the conventional point estimate. The interval
is centred on `estimate_robust`, so it is not symmetric around `estimate`.
`pooled_se` and `pooled_ci` change the same way. The old numbers are in
`se_conventional` and `ci_conventional`.

**Who is affected.** Anything that read intervals or p-values from
`cutoff_results` or `pooled_ci` on the `cutoff_var=` path. They were too
narrow. The `cutoffs=` path (one shared running variable) is unchanged.

**What to do.** Re-run. `result.normalized` holds the estimate on the
normalized score, `rdmulti`'s "Pooled" row; `pooled_estimate` remains the
effective-sample-size-weighted average of the cutoffs.

---

---

## 1.38.0 → 1.39.0: ⚠️ `sp.callaway_santanna` with time-varying covariates, `sp.etwfe_emfx` headline SEs, unidentified `sp.did_imputation` leads

**`sp.callaway_santanna` with time-varying covariates.**
*What changed.* A covariate that varies within unit is now read, for each
ATT(g, t), in the earlier of the cell's two periods. It used to be read
from the unit's first row of the data for every cell.
*Who is affected.* Calls with `x=` where at least one covariate changes
over time. Every ATT(g, t) and every aggregate can move. Covariates that
are constant within unit are unaffected.
*What to do.* Re-run. The new numbers are the ones Stata `csdid` and R
`did` give. To reproduce an old result, replace each covariate by its
value in the unit's first row before calling.

**`sp.etwfe_emfx(type='group' | 'calendar' | 'event')`.**
*What changed.* `.se`, `.pvalue` and `.ci` of the returned result now
belong to `.estimate` (the unweighted mean of the reported rows). They
used to be those of the `'simple'` aggregate. With `include_leads=True`
the mean is over the post-treatment rows only.
*Who is affected.* Code that reads the headline of a non-simple
aggregation of a `sp.etwfe` / `sp.jwdid` fit. The rows in `.detail` and
`type='simple'` are unaffected.
*What to do.* Re-run. For the overall ATT and its standard error use
`type='simple'`.

**`sp.did_imputation(pretrends=k)`.**
*What changed.* A request for leads that the design does not identify
raises `MethodIncompatibility` instead of returning numbers.
*Who is affected.* Panels without never-treated units and a large `k`.
The numbers such calls returned were not estimates of anything.
*What to do.* Request the number of leads the message names.

**`sp.synth(method='classic')` without predictors.**
*What changed.* `standardize_predictors` defaults to `None` instead of
`True`. With `covariates=` or `special_predictors=` that still means
rescaling. Without them the pre-treatment outcomes are no longer rescaled
period by period, and the weights minimise the pre-treatment squared
error.
*Who is affected.* Classic SCM fitted with neither `covariates=` nor
`special_predictors=`, and `sp.synth_loo`, `sp.synth_time_placebo`,
`sp.synth_donor_sensitivity`, `sp.synth_rmspe_filter` and
`sp.synth_sensitivity` on such a fit. Donor weights, the synthetic path,
the gaps and the placebo ratios can move. On Proposition 99 the ATT goes
from -19.76 to -19.51 and the pre-treatment RMSE from 1.72 to 1.66.
*What to do.* Re-run. For the earlier numbers pass
`standardize_predictors=True`.

**`sp.stata` / `sp.from_stata`: `csdid`, `collapse (first)`.**
*What changed.* A translated `csdid` line carries `pscore_trim=1.0`,
`csdid`'s default, where it used to leave StatsPAI's 0.995 in force. In
`collapse`, `(first)` and `(last)` take the first and last row of each
group even when it is missing; they used to skip missing values, which
is `(firstnm)` and `(lastnm)`.
*Who is affected.* A `csdid` line with covariates run through `sp.stata`
on data where a control unit's propensity score exceeds 0.995. A
`collapse (first)` or `(last)` on a variable with missing values at the
edge of a group.
*What to do.* Nothing to reproduce Stata. For the earlier results, add
`pscoretrim(0.995)` to the `csdid` line, or write `(firstnm)` /
`(lastnm)`. Direct calls to `sp.callaway_santanna` are unaffected.

---

## 1.37.0 → 1.38.0: `sp.write_data` folds `<var>__miss` columns back into `<var>` in a .dta file

**What changed.** A frame read with `sp.read_data(path,
extended_missing='column')` has, beside each variable that held `.a` ...
`.z`, a column `<var>__miss` with those codes. `sp.write_data` to .dta used
to write that column as a string variable and every missing value as `.`.
It now writes the codes back into `<var>` as Stata's extended missing
values and does not write the `__miss` column. Value-label sets also keep
the names they had in Stata (`df.attrs['_value_label_names']`) instead of
being named after each variable.

**Who is affected.** Code that reads a .dta written by `sp.write_data` and
expects a `<var>__miss` string variable in it, or expects every label set
to be named after its variable.

**What to do.** Pass `extended_missing='nan'` to `sp.write_data` for the
old file layout. To have sets named after their variables again, remove
`df.attrs['_value_label_names']` before writing.

**Unaffected.** A `__miss` column that holds anything other than `.a` ...
`.z` codes, or whose base column is missing or not numeric, is an ordinary
column and is written as before. Formats other than .dta are unchanged.

<a id="oct2026-python310"></a>

## 1.36.0 → 1.37.0: Python 3.10 or later is required

`pip install statspai` on Python 3.9 now resolves to the last release that
supported it (1.36.0). Upgrade the interpreter to get later versions. There
was no deprecation period: 3.9 has been end-of-life since October 2025, and
the functions built on `pyfixest` (`sp.feols`, `sp.xtevent` with its default
engine, `sp.ssc`) already could not run on it.

Windows users: the package installs as before, but its tests are not run on
Windows for now, and the MCP server's stdio transport failed its tests there
on 2026-10-03. See `docs/dev/2026-10-04-windows-test-gaps.md`.

---

<a id="oct2026-dta-widths"></a>

## 1.36.0 → 1.37.0: ⚠️ labels on `.a` ... `.z` move to `attrs['_missing_labels']`; `sp.tab` prints value labels

**Extended missing values.** For a .dta file that labels an extended
missing value (`label define r 1 "North" .a "Refused"`), `sp.read_data`
used to return the label in `df.attrs['_value_labels']` under a key that
depended on whether `pyreadstat` was installed: `2147483622` without it,
`'a'` with it. Neither is a code a row can hold once the missing values
are `NaN`. The label is now in `df.attrs['_missing_labels']`, keyed
`'.a'`, on both paths, and a `UserWarning` reports it.

```python
# before
df.attrs["_value_labels"]["region"]   # {1: 'North', ..., 2147483622: 'Refused'}
# after
df.attrs["_value_labels"]["region"]   # {1: 'North', ...}
df.attrs["_missing_labels"]["region"] # {'.a': 'Refused'}
```

Code that looped over `_value_labels` and special-cased the large integer
or the bare letter should read `_missing_labels` instead. To keep which
rows held `.a`, read with `extended_missing='column'`.

With `pyreadstat` installed, a numeric variable that holds `.a` ... `.z`
used to come back as an `object` column; it is now `float64`. Code that
worked around this with `pd.to_numeric` keeps working.

**`sp.tab`.** On data that carry value labels, the table's categories are
now the label texts (in the order of the codes) rather than the codes.
Counts, percentages and tests are the same. Pass `labels=False` for the
old layout. Data without value labels are unaffected.

**`sp.describe`.** The returned frame has a sixth column, `value_labels`.
Code that unpacks exactly five columns by position needs the extra name.

## 1.36.0 → 1.37.0: ⚠️ `sp.read_data` returns `int64` / `float64` for every numeric .dta column

This affects you only if `pyreadstat` is **not** installed (with it the
dtypes were already these), and it also applies to a `.dta` loaded through
the MCP server.

- A Stata `byte` / `int` / `long` used to arrive as `int8` / `int16` /
  `int32` and a `float` as `float32`. Arithmetic on the narrow integers
  wrapped silently (`df["age"] ** 2` past 127). They are now `int64` and
  `float64`. Values are unchanged; a Stata `float` keeps its float32 value.
- A frame takes more memory: up to 8 times for a file of `byte` columns.
  For a file too large to widen, read it with `pd.read_stata` directly.
- Code that tested `df[col].dtype == "int8"` to recognise a dummy should
  test the values instead.
- `sp.write_data` picks the smallest exact Stata type for each column, so
  a file read and written back is the same size as before.

---

<a id="oct2026-dta-dates"></a>

## 1.36.0 → 1.37.0: ⚠️ `sp.read_data` returns Stata dates as `datetime64` when `pyreadstat` is installed

This affects you only if `pyreadstat` is installed; without it
`sp.read_data` already behaved as described.

- A `%tm` / `%tq` / `%th` / `%tw` / `%ty` variable used to arrive as
  Stata's period count (months since 1960m1, and so on). It now arrives as
  `datetime64`, the first day of the period. Code that did arithmetic on
  the count (`ym - 720`) should work on the date instead, or read the
  counts with `sp.read_data(path, disable_datetime_conversion=True)`.
- A `%td` variable used to arrive as an `object` column of
  `datetime.date`. It is now `datetime64`; comparisons against
  `datetime.date(2020, 1, 15)` should use `pd.Timestamp("2020-01-15")`.
- `sp.write_data` writes a date read from a .dta back with its Stata unit
  instead of as `%tc`. Pass `convert_dates=` to choose another.

---

<a id="oct2026-mcp-dta-codes"></a>

## 1.35.0 → 1.36.0: the MCP data loader returns Stata value-labelled columns as codes

A `.dta` passed to an MCP tool as `data_path` used to arrive with every
value-labelled column converted to a string categorical (`foreign` became
`"Domestic"` / `"Foreign"`). It now arrives as Stata stores it and as
`sp.read_data` returns it: the numeric codes (0 / 1), with the labels in
`df.attrs['_value_labels']` and the variable labels in
`df.attrs['_labels']`.

- A formula such as `price ~ weight + foreign` now fits the same model
  Stata's `regress price weight foreign` does.
- A call that filtered or grouped on the label text (`foreign == "Foreign"`)
  should use the code, or map it through `attrs['_value_labels']`.

Python calls to `sp.read_data` are unaffected.

---

<a id="oct2026-cr2-t-reference"></a>

## 1.36.0 → 1.37.0: ⚠️ CR2 / CR3 / two-way p-values and intervals use t(G - 1)

`sp.regress`, `sp.feols` and `sp.ivreg` with `vce='cr2'`, `vce='cr3'` /
`'jackknife'`, and `sp.regress(cluster=[a, b])`, now refer the t ratio to
`t(G - 1)` (`t(G_min - 1)` for two-way clustering), as the one-way cluster
variance of the same functions and Stata's `vce(hc2 clustvar)`,
`vce(jackknife, cluster())` and `vce(cluster a b)` do. They used the
normal. Coefficients and standard errors do not change; p-values rise and
intervals widen, by a little at 40 clusters (the 97.5% critical value
goes from 1.960 to 2.023) and by a lot at 6 (2.571).

`sp.hdfe_ols` and `sp.panel` already used `t(G - 1)` for these variances
and are unchanged, as are the GLM variants (`sp.feglm`, `sp.fepois`),
which keep the normal.

---

<a id="oct2026-poisson-weights-vcov"></a>

## 1.36.0 → 1.37.0: ⚠️ weighted `sp.poisson` / `sp.ppmlhdfe` standard errors

Refit any weighted Poisson or PPML-HDFE table: the coefficients stand,
the standard errors do not.

| Call | Stata equivalent | Before | Now |
| --- | --- | --- | --- |
| `sp.poisson(..., weights=w)` | `poisson [iw=w]` (`[fw=w]` if integer) | SE 10% high (42% with integer weights) | matches |
| `sp.poisson(..., weights=w, robust='robust')` | `poisson [pw=w]` | SE 7% low | matches |
| `sp.poisson(..., weights=w, cluster=g)` | `poisson [pw=w], vce(cluster g)` | unweighted cluster sums | matches |
| `sp.ppmlhdfe(..., weights=w)` | `ppmlhdfe [pw=w], vce(robust)` | SE 8% off | matches |
| `sp.ppmlhdfe(..., weights=w, cluster=g)` | `ppmlhdfe [pw=w], vce(cluster g)` | SE 12% off | matches |
| `ll`, `ll_null`, LR chi2, pseudo R2 of a weighted `sp.poisson` / `sp.nbreg` / `sp.ppmlhdfe` | `e(ll)`, `e(ll_0)` | unweighted sums | weighted sums |

The percentages are on the test fixture. Unweighted fits, `sp.logit`,
`sp.probit`, `sp.glm` and `sp.fepois` are not affected.

---

<a id="oct2026-feols-jax-ssc"></a>

## 1.36.0 → 1.37.0: ⚠️ `sp.fast.feols_jax` standard errors follow `sp.fast.feols`; `ssc=` added

`sp.fast.feols_jax` now uses the same small-sample convention as
`sp.fast.feols` (R `fixest` defaults). Before, the two backends disagreed:
on a panel with unit effects clustered above the unit, the JAX backend's
clustered standard errors were 8% larger. Coefficients do not change.
Pass `ssc='statspai'` for the previous numbers. Results fitted on a GPU
with clustered standard errors should be refit or rescaled.

---

<a id="oct2026-fast-fepois-ssc"></a>

## 1.36.0 → 1.37.0: ⚠️ `sp.fast.fepois` standard errors follow fixest; `ssc=` added

`sp.fast.fepois` now reproduces R `fixest::fepois` standard errors by
default. Coefficients do not change. To get the previous numbers pass
`ssc='statspai'`.

| Variance | Before | Now (`ssc='fixest'`) | Size of the change on the reference panel |
| --- | --- | --- | --- |
| `cr1`, effects nested in the clusters | every absorbed level charged | nested dimensions left out of `K` | SEs fall by 8% |
| `cr1`, effects not nested | `Σ(G_k - 1)` charged | `ΣG_k - (dims - 1)` charged | SEs rise by 0.06% |
| `hc1` | `n/(n - p - Σ(G_k - 1))` | `n/(n - K)` | SEs rise by 0.06% |
| `iid` | `n/(n - p - Σ(G_k - 1))` | `(n - 1)/(n - K)` | SEs rise by 0.008% |
| `hc1` with `weights=` | weights missing from the score | weighted score | SEs change by 43% to 46% |

The last row is a defect, not a convention: `ssc='statspai'` does not
bring it back.

---

<a id="oct2026-fast-feols-one-fe-dof"></a>

## 1.36.0 → 1.37.0: ⚠️ `sp.fast.feols` standard errors with one absorbed dimension

`sp.fast.feols('y ~ x | firm', ...)` at the default `ssc='fixest'` now
uses `n - p - G` residual degrees of freedom, as R `fixest` and Stata
`reghdfe` do; it used `n - p - (G - 1)`. Standard errors rise by a factor
`sqrt((n - p - G + 1)/(n - p - G))`, which is 1.0006 on a 1,042-row panel
with 150 firms and larger on short panels. Coefficients do not change.

| Model | Variance | Before | Now |
| --- | --- | --- | --- |
| one absorbed dimension | `iid`, `hc1` | one degree of freedom too few | matches fixest |
| one absorbed dimension, nested in the clusters | `cr1` | intercept not counted | matches fixest |
| two absorbed dimensions, neither nested in the clusters | `cr1` | one degree of freedom too many | matches fixest |
| two absorbed dimensions, one nested in the clusters | `cr1` | exact | unchanged |
| any | `ssc='statspai'` | pre-1.31 convention | unchanged |

`sp.feols` and `sp.hdfe_ols` are not affected.

---

<a id="oct2026-regress-weights-cluster"></a>

## 1.36.0 → 1.37.0: ⚠️ `sp.regress` weights and missing rows under CR2, CR3 and two-way clustering

Three calls returned numbers for a different model than the one asked
for. Refit any table produced with them.

| Call | Before | Now |
| --- | --- | --- |
| `sp.regress(..., weights=w, vce='cr2'/'cr3', cluster=g)` | unweighted coefficients and SEs | weighted; CR2 matches Stata `regress [aw=w], vce(hc2 g)` |
| `sp.regress(..., weights=w, cluster=[a, b])` | unweighted coefficients and SEs | weighted |
| the same three variances on data with missing values in a formula variable | cluster keys taken from the first `n` rows | cluster keys taken from the fitted rows |
| `sp.twoway_cluster(fit, ...)`, `sp.cr2_se(fit, ...)` on a weighted `fit` | unweighted scores | weighted scores |
| `sp.regress(..., weights=w, vce='wild' / 'conley')` | weights dropped | raises `MethodIncompatibility` |

Calls without `weights=` on data without missing values are unchanged.

Two conventions are documented rather than changed. Two-way clustering in
`sp.regress(cluster=[a, b])` scales the whole meat by
`G_min/(G_min - 1)`; Stata 18's `regress, vce(cluster a b)` scales each
component by its own `G/(G - 1)`, which is what `sp.twoway_cluster`
returns. `vce='cr3'` is `sum_g (b_(g) - b)(b_(g) - b)'`; Stata's
`vce(jackknife, cluster(g) mse)` is `(G - 1)/G` times that.

---

<a id="oct2026-panel-vce"></a>

## 1.35.0 → 1.36.0: ⚠️ `sp.panel(vce=...)` raises on a value it does not implement

`sp.panel(..., vce='robust')` (and `'cluster'`, `'hc1'`, or any other
string outside `'CR2'` / `'CR3'` / `'jackknife'` / `'conley'` / `'wild'`)
used to be accepted and ignored: the result carried the **classical**
standard errors. It now raises `MethodIncompatibility`.

- You meant heteroskedasticity-robust SEs: pass `robust='robust'`.
- You meant cluster-robust SEs: pass `cluster='<column>'`.
- You meant the classical SEs: drop `vce=`.

Numbers do not change for any call that worked as intended. A table
produced with one of the ignored values reports classical SEs under a
robust label and should be refit.

---

<a id="oct2026-crossval-next-steps"></a>

## 1.35.0 → 1.36.0: `CrossValidationResult.next_steps()` returns dicts

`sp.cross_validate(...).next_steps()` returned a list of strings; every
other result class returns a list of `{"action", "reason", "priority",
"category"}` dicts. It now does the same, and so does the `next_steps`
entry of its `to_dict()`. Code that printed the strings should print
`step["action"]`. No numbers change.

---

<a id="oct2026-intro-textbook-fixes"></a>

## 1.35.0 → 1.36.0: ⚠️ model F under robust errors, fit statistics without a constant, AIC / BIC, `sp.estat` defaults

**`sp.regress(..., robust=...)` / `cluster=`.** `diagnostics['F-statistic']`
and its p-value change whenever the covariance is not the classical one.
They were the classical F. They are now the Wald test of all slopes on the
covariance of the fit, the number Stata and R print. Coefficients, standard
errors and `sp.test` are unchanged. If you need the old number, fit without
`robust=`.

**`sp.regress("y ~ x - 1")`.** `r2`, `r2_adj`, the F statistic and
`data_info['df_model']` change to the uncentred versions. The old centred
R-squared is `1 - RSS / sum((y - ybar)^2)` from `result.data_info`.

**`diagnostics['AIC']` / `['BIC']` of `sp.regress`, and `sp.estat(...,
'ic')`.** Both shift by the constant `N (log(2 pi) + 1)` and now equal
`-2 log L + 2K` / `-2 log L + K log N`. Comparisons between models fitted on
the same sample are unaffected. Code that compared these numbers with an
externally computed criterion on the old scale needs the constant added.

**`sp.estat(result, 'bgodfrey')`.** The default keeps every observation
(`fill='zero'`). Pass `fill='drop'` for the previous statistic.

**`sp.estat(result, 'white')`.** `df` and the p-value change when a
regressor is a dummy or the square of another regressor.

**`sp.estat(result, 'vif')`.** `vif_table` is no longer rounded.

**`result.bp_lm_test()` on unbalanced panels.** The statistic is smaller;
balanced panels are unchanged.

**`sp.estat(..., lags=)`.** The default is now `None` (1 for `'bgodfrey'`,
2 for `'varlmar'`) instead of 1. Calls that pass `lags` are unaffected.

**`sp.from_stata` / `sp.stata`.**

| Stata line | Before | Now |
| --- | --- | --- |
| `reg y x, noconstant` | refused (`untranslated_options`) | `sp.regress('y ~ x - 1')` |
| `xtreg y x` / `, re` / `, be` / `, mle` | refused | `sp.panel(method='re' / 'be' / 'mle', ssc='stata')` |
| `synth ...` | `sp.synth(...)` with the in-space placebo | `placebo=False`, as Stata |
| `tsset t` / `xtset id t` | recorded | recorded, and the data are sorted |
| `estimates clear` | skipped | clears the stored models |

<a id="oct2026-dbe-textbook-fixes"></a>

## 1.35.0 → 1.36.0: ⚠️ `sp.etwfe` without an untreated group, RD pilot bandwidth, LIML `kappa`, CR2 degrees of freedom, `didregress` translation

**`sp.etwfe`, every unit eventually treated.** Results from such a panel
were arbitrary before (the design was rank deficient and a warning said
so). The call now estimates on the periods before the last cohort's
adoption, with that cohort as the not-yet-treated reference, and warns.
The numbers equal R `etwfe` and what you get by dropping those periods by
hand. Panels with a never-treated unit, or a cohort adopting after the
panel ends, are unchanged.

**`sp.rdrobust` / `sp.rdbwselect` with a data-driven bandwidth.** The
bandwidth changes when `IQR / 1.349` of the running variable is smaller
than its standard deviation, typically a heavy-tailed or bounded running
variable. The change is of order 1e-4 or less in `h` and smaller in the
estimate. Results with `h=` given are unchanged. There is no switch back:
the new value is the one R and Stata `rdrobust` compute.

**`sp.iv(method='liml')` / `sp.liml`.** Estimates change by about the
relative error the old `kappa` carried, 1e-6 on a design with first-stage
F of 30 and `kappa - 1` of 7e-6, less when instruments are stronger.

**`sp.stata("didregress ...")` / `"xtdidregress ..."`.** The result is
now Stata's two-way fixed-effects ATET. Before, a panel with more than two
periods was collapsed to a 2x2 and the number was a different estimand.
Call `sp.did(method='twfe')` directly if the collapsed 2x2 is what you
want.

**`sp.synth` and other simplex-weight solvers with fewer rows than
donors.** Donor weights move by up to 2e-5 and estimates by about 1e-7
where the solution is unique, because it is now solved exactly. Fits where
the solution is not unique are unchanged.

**`sp.synth` when the donor weights are not unique.** For a treated unit
inside the donors' predictor hull (`model_info['in_predictor_hull']`), and
for placebo units in that position, the weights are now the minimum-norm
ones among those with the best fit. Before they were SLSQP's choice from a
uniform start. Estimates for such units change, and a placebo p-value that
ranks them can change with them. Fits with a unique solution do not move.
There is no switch back: the old choice was not reproducible across
solvers. `perfect_fit='exact_balance'` remains the alternative estimator
for the in-hull case.

**`sp.cr2_se` p-values and confidence intervals.** The standard errors are
unchanged. The degrees of freedom were an approximation that is not the
Bell-McCaffrey formula, and every coefficient was referred to the smallest
of them. Each coefficient now has its own exact df (those of R `estimatr`
and `clubSandwich`), so intervals change, usually to narrower ones for the
coefficient of interest. `diagnostics['satterthwaite_dof']` holds the new
values and `data_info['df_resid']` is still their minimum.

**`sp.estat(result, 'endogenous')` after `robust=` or `cluster=`.** The
headline statistic is now the robust regression-based F, as in Stata. The
Durbin and Wu-Hausman statistics it used to return assume homoskedastic
errors. They are still in the output under `wu_hausman_F`.

**`sp.pretrends_power` / `sp.pretrends_slope_for_power`.** Values move in
the sixth digit because the probability is integrated more finely.

**`sp.from_stata` / `sp.stata`.**

| Stata line | Before | Now |
| --- | --- | --- |
| `ivregress` / `ivreg2` (2SLS, LIML) without `small` | `sp.ivreg(...)`, small-sample SEs; `robust` as HC0 | `sp.iv(..., small=False)`, Stata's large-sample SEs for classical, robust and cluster VCE |
| `teffects nnmatch` | `sp.match(method='nn')`, which raised | `sp.match(method='nnmatch', ...)` |
| `teffects psmatch` | `sp.match(method='ps')`, which raised | propensity matching with ties kept; ATET with the Abadie-Imbens (2016) SE |
| `teffects ra` / `ipwra` | `sp.regress` / `sp.ipw` | refused: a different estimator |
| `reg y 1.d` | formula `y ~ 1.d` | `y ~ I(1 * (d == 1))` |

---

<a id="oct2026-stata-translation-conventions"></a>

## 1.35.0 → 1.36.0: ⚠️ `sp.stata` standard errors for robust ML commands and large-sample IV

**Who is affected.** Anyone who ran these Stata lines through `sp.stata`,
`sp.from_stata` or the MCP `from_stata` tool and reported the standard
errors. Coefficients are unchanged. Direct calls to `sp.probit`,
`sp.ivreg` and the others are unchanged.

| Stata line | Before | Now | Size of the change |
| --- | --- | --- | --- |
| `probit` / `logit` / `poisson` / `nbreg ..., vce(robust)` | `robust='hc1'`, factor `N/(N-K)` | `robust='robust'`, Stata's `N/(N-1)` | SE smaller by `sqrt((N-1)/(N-K))` |
| `ivreg2` / `ivregress ..., robust` without `small` | `robust='hc1'` | `robust='hc0'`, no factor, as Stata | SE smaller by `sqrt((N-K)/N)` |
| `regress ...; summarize x; test x` | `test` ran on the summary table | `test` runs on the regression | |
| `summarize x, detail` | linear-interpolated `p10` `p25` `p50` `p75` `p90` | Stata's percentile rule; adds `p1` `p5` `p95` `p99`, variance, skewness, kurtosis | percentiles move within the gap between two observations |

`sp.sumstats(stats=[...])` with a name it does not know used to drop that
column without a message. It now raises `MethodIncompatibility` and lists
the available statistics. Its default percentiles are unchanged (linear
interpolation); `percentile_method='stata'` selects Stata's rule.

To reproduce an old number, call the estimator directly with
`robust='hc1'`.

---

<a id="oct2026-known-truth-fixes"></a>

## 1.34.2 → 1.35.0: ⚠️ estimators that missed a known truth, and clustered CER bandwidths

**Who is affected.** Anyone who reported numbers from the functions below.
Unless a row says otherwise, the old output cannot be reproduced by an
option, because it was not an estimate of the documented quantity.

| Function | What changes | Old on a design with a known answer | New |
| --- | --- | --- | --- |
| `sp.balke_pearl` | `lower`, `upper`, `width` | interval excluded the true ATE in 5% of valid IV models | the sharp bounds |
| `sp.rd_multi_score`, `sp.multi_score_rd` | `boundary_effect`, `se` | 0.14 (truth 0.8) | 0.80 |
| `sp.design_robust_event_study` when exposure exceeds `lags` | every coefficient and the headline | 1.21 (truth 2.0) | 2.05 |
| `sp.cluster_staggered_rollout` with two or more cohorts | `event_study`, `overall_att`, all SEs | 1.22 (truth 1.5) | 1.51 |
| `sp.rd_extrapolate` | `se`, `ci`, `pvalue` of the average | 40% coverage | 92% |
| `sp.kitagawa_test` | `p_value` | near 0.5 on any data | a test with power |
| `sp.rd_forest`, `sp.rd_boost` | `estimate`, `se`, `ci`, `detail`, `estimand` | 1.29, 0% coverage (truth 1.0) | 1.03, interval covers; mean of fitted CATEs in `model_info['mean_cate']` |
| `sp.rd_distribution` | `qte`, `se` | `qte` was a CDF jump (-0.48 for a +2.0 shift) | a quantile effect; the old values are `cdf_effect`, `cdf_se` |
| `sp.did_few_treated` | `ci` | 88% coverage at nominal 95% (29 controls) | 96.6%; old interval in `model_info['quantile_interval']` |
| `sp.event_study` with a window wider than the exposure in the data | headline `estimate`, `se`; rows of `detail` | 1.56 (truth 2.0), a row of zeros | 1.95; the empty event time is dropped with a warning |
| `sp.event_study` with never-treated coded by a value just below the first period (e.g. `0`, periods from 1) | every coefficient | controls counted in the post bins | never-treated |
| `sp.power_iv` | `power` | 0.95 (simulated 0.21) | 0.21 |
| `sp.power_rd` | `power` | 0.93 (simulated 0.74) | 0.72 |
| `sp.power_did` | `power` | 0.48 (simulated 0.38) | 0.38; new `prop_treat` |
| `sp.identify` | `estimand` string (verdict unchanged) | front-door: `sum_{M} [P(Y) * P(M \| X)]` | the front-door formula |
| `sp.kernel_iv` | `h_hat`, `ci_low`, `ci_high` | RMSE 0.25 from `sin(d)` under confounding, band excluded the truth | RMSE 0.05 to 0.06; band covers |
| `sp.bcf_longitudinal` | `average_ate`, `per_time_ate`, `individual_cate` | 7.44 to 7.77 (truth 6.0, randomised treatment) | 5.99 to 6.00 |
| `sp.bcf_longitudinal` with a period where treatment does not vary | `average_ate`; that row of `per_time_ate` | the period counted as an effect of 0 | NaN for that period, left out of the average, with a warning |
| `sp.bcf_ordinal` | `ate_se`, `ate_ci` | SE 0.001 (across-seed SD 0.03 to 0.05) | 0.024 to 0.042 |
| `sp.dgp_rd(fuzzy=True)` | `treatment`, `y` for every seed | no first-stage jump; fuzzy estimates had no finite mean | jump of 0.6; fuzzy `rdrobust` averages 0.301 for a declared 0.3 |
| `sp.dgp_rd` with `cutoff != 0` or a non-default `bandwidth_relevant` | `x`, `treatment` | `x` stayed on `(-1, 1)`; the spread argument was ignored | `x` uniform on `cutoff +/- 2 * bandwidth_relevant` |
| `sp.dgp_rdit` | `y` for every seed | level shift 2.86 for a declared 2.0 | 2.0 |
| `sp.conformal_ite_multidp` | `intervals_per_stage`, `cumulative_interval` | 83% per-stage coverage of the individual effect (95% owed) | 99.8%; intervals about 2.3 times wider |
| `sp.deepiv` at its defaults | `estimate`, `detail`, `effect()` | 0.50 to 0.71 per unit (truth 1.0) | 0.90 to 1.14; pass `n_gradient_samples=0` for the old loss |
| `sp.vcnet`, `sp.scigan` with a `t_grid` that does not span the observed doses | `mu_hat`, `se`, `ci_lo`, `ci_hi` | 1.01 at `t = 0.2` (truth 1.36), 0% coverage | 1.354, 93% coverage |
| `sp.vcnet` | `se`, `ci_lo`, `ci_hi` | 78% coverage at mid-range | 93% |
| `sp.scigan` with `propensity_weights` | `mu_hat`, `se` | a weighted random resample of the rows | weighted fit; unit weights equal `sp.vcnet` |
| `sp.llm_annotator_correct`, binary label | `estimate`, `correction_factor` | 0.835 (truth 1.0) at prevalence 0.2 | 1.00 |
| `sp.llm_annotator_correct`, every path | `se`, `ci`, `pvalue` | first-order SE 0.059 (across-seed SD 0.074) | delta-method SE 0.077; first-order kept in `model_info` |
| `sp.discos_test(test='ks')` | `statistic`, `pvalue` | 57% rejection of a true null | placebo rank of the sup-norm quantile gap; NaN without placebos |
| `sp.discos_test(test='cvm' / 'stochastic_dominance')` on a result fitted with `placebo=False` | `pvalue` | a chi-squared or KS approximation over grid nodes | NaN |
| `sp.sensitivity_rr` | `ci_lower`, `ci_upper`, `breakdown_mbar` | 2% coverage at `Mbar = 0` under a linear pre-trend | covers; intervals wider |
| `sp.iv_compare` | `first_stage_F` column | NaN | the first-stage F |
| `sp.rd_compare` at its default `methods` | number of rows | three, the `randinf` row always `status = error` | two; add `randinf` with its window |
| `sp.llm_dag_validate` given an edge list or dict | `n_supported`, `n_unsupported`, `edge_evidence` | 0, 0, empty | the per-edge tests |
| `sp.rdbwselect`, `sp.rdrobust` with a `cer*` bandwidth and `cluster=` | `h`, and through it the estimate and SEs | `h_cerrd` 0.2054 (rdrobust: 0.2305) | 0.2305 |

**What to do.** Re-run. Code that read `res.qte` from `sp.rd_distribution`
as a change in probability should read `res.cdf_effect`; code that
assumed `sp.rd_forest(...).estimate == detail['cate'].mean()` should read
`model_info['mean_cate']`. For `design_robust_event_study`, results are
unchanged when `lags` already covers the longest exposure in the data
(`model_info['diagnostics']['n_obs_binned_post'] == 0`). For
`cluster_staggered_rollout`, results with a single treated cohort are
unchanged in the point estimates. `balke_pearl`'s `lower_monotone` and
`upper_monotone` are unchanged.

Separately, estimators that used to swallow an internal failure now warn
(see the changelog). Code that runs with warnings as errors will surface
them; the remedy is in each message.

---

<a id="stata-translation-grammar"></a>

## 1.34.2 → 1.35.0: ⚠️ `sp.from_stata` / `sp.stata` stop translating a different model

**Who is affected.** Anyone who translated or ran Stata lines that use
option abbreviations, prefixes, macros, `if` / `in`, or `xtreg` without
`fe`. Lines written with full option names and no qualifier are unchanged.

| Stata line | Old | New |
| --- | --- | --- |
| `reg y x, r` · `cl(id)` · `vce(cl id)` | default SEs, no note | robust / clustered SEs |
| `reghdfe y x, a(id year)` | pooled OLS, fixed effects lost | `y ~ x \| id + year` |
| `rdrobust y x, h(5) vce(cluster id) covs(a)` | options dropped | carried into `sp.rdrobust` |
| `reg y x, nocons` or an unknown option | dropped, no note | `ok=True`, listed in `untranslated_options` |
| `reg y x $ctrl` | `'y ~ x + $ctrl'`, `ok=True` | refused |
| `by g: reg y x` | pooled regression | refused |
| `qui reg y x` · `eststo: reg y x` | "unknown command" | translated |
| `xtreg y x` (no `fe`) | fixed-effects model | refused (it is random effects) |
| `xtreg y x, fe vce(robust)` | default SEs | clustered on the panel id |
| `csdid y x, ivar() time() gvar()` | covariates dropped; StatsPAI defaults | `x=['x']`, `base_period='varying'` (csdid's default; `long2` gives `'universal'`) |
| `csdid ..., method(ipw)` | `estimator='ipw'` (stabilised) | `estimator='ipw_abadie'` |
| `did_imputation ..., horizons(0/3)` | horizons dropped | `horizon=[0, 1, 2, 3]` |
| `xtabond y x` printed code | no `robust=`, so `True` when run | `robust=False`, as Stata |
| `psmatch2 d x, out(y)` (no `logit`) | translated as exact | `ok=True`, `probit` in `untranslated_options`; `sp.stata` raises |
| `psmatch2 ..., ties ate` | dropped / "use sp.match" note | `ties=True`, `ate=True` |
| `ivreghdfe y c.a##c.b (d = z), absorb(id)` | formula `a + + + b + + + a:b` | `a + b + a:b` |
| `sp.stata("reg y x if z > 0", data=df)` | ran on all rows, warned | raises `MethodIncompatibility` |
| `sp.stata("reg y x, nocons", data=df)` | ran with a constant, warned | raises `MethodIncompatibility` |

**What to do.** Numbers obtained by running abbreviated lines through
`sp.stata` before this change may have used the wrong standard errors or
dropped the fixed effects: rerun them. For `if`, filter first and drop the
qualifier, `sp.stata("reg y x", data=df[df.z > 0])`, minding that Stata
counts a missing `z` as larger than any number. For an option reported in
`untranslated_options`, run the printed `python_code` and set the option
on the sp call.

---

<a id="regress-collinear-omit"></a>

## 1.34.0: `sp.regress` omits collinear regressors instead of raising

**Who is affected.** Code that relied on `sp.regress` raising
`NumericalInstability` on a rank-deficient design.

| Area | Old | New | Old behaviour |
| --- | --- | --- | --- |
| `sp.regress` on collinear regressors | raised `NumericalInstability` | omits the later member of each set with a `UserWarning`, lists it in `model_info['omitted']` (Stata `regress`) | `sp.regress(..., collinear='raise')` |

---

<a id="ordered-cutpoint-vce"></a>

## 1.34.0: ⚠️ ordered-model cutpoint SEs and the mlogit IIA test

**Who is affected.** Anyone reading cutpoint standard errors from
`sp.oprobit` / `sp.ologit` with `robust=` or `cluster=`, the `brant_test`
of those models, or `iia_test` from `sp.mlogit`. Coefficients and their
SEs are unchanged.

| Area | Old | New |
| --- | --- | --- |
| cutpoint SEs with `robust` / `cluster` | model-based (the VCE was ignored) | the requested VCE, as Stata |
| `brant_test` SEs | BFGS `hess_inv` approximation | exact binary-logit information |
| `mlogit` `iia_test` | restricted model with the dropped category left in the softmax; `hess_inv` variance; non-PSD categories skipped | `mlogit ... if y != j`; Stata `hausman` generalised inverse and df; every category reported |

There is no switch back: the old numbers were wrong.

---

<a id="stacked-did-reghdfe-k"></a>

## 1.33.0 — ⚠️ `sp.stacked_did` standard errors follow `reghdfe`

**Who is affected.** Anyone reading standard errors, p-values or intervals
from `sp.stacked_did`. Point estimates are unchanged.

| Area | Old | New | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.stacked_did` CR1 small-sample factor `(N-1)/(N-K)` | `K` = event-time dummies only | `K` also counts the absorbed stack-by-period effects (not nested in the unit cluster), as `reghdfe` on the same stack | `se_new * sqrt((N - K)/(N - k_event))` |

Every SE widens by the same constant within a fit: 1.5% on the
castle-doctrine panel and 7.0% on the unilateral-divorce panel, where the
new SE equals `reghdfe` on the manual stack to 1e-13.

---

<a id="etwfe-labels-table-rows"></a>

## 1.33.0 — `sp.etwfe` estimand labels, blank FE / cluster cells for causal results

**Who is affected.** Code that compares `sp.etwfe(...).estimand` with the old
long label, and tables that mix a `CausalResult` with fixed-effects
regressions. No estimate or standard error changed.

| Area | Old | New | How to keep the old behaviour |
| --- | --- | --- | --- |
| `sp.etwfe(family='poisson' / 'logit', scale='link').estimand` | `"ATT (link scale: treated-observation-weighted mean of the log-point cohort x period effects)"` | `"ATT (link scale)"`; the long text is `model_info['estimand_description']` | test `"link scale" in res.estimand`, or read `estimand_description` |
| linear jwdid-design `sp.etwfe(hettype= / fe='unit').estimand` | `"ATT (linear ETWFE, treated-observation-weighted)"` | `"ATT (linear ETWFE)"` | as above |
| `sp.regtable` FE / cluster cells of a `CausalResult` without FE metadata | `No` | blank | `add_rows={"<Var> FE": [...]}` |
| `sp.regtable` rows for `sp.etwfe` | FE `No`, cluster `No` | unit (or cohort) and period FE `Yes`, cluster variable | — |

---

<a id="ppml-singletons-aggte-inherit"></a>

## 1.33.0 — Stata defaults: `ppmlhdfe` singletons, `winsor` percentiles, `logit` perfect prediction; `aggte` inherits

**Who is affected.** `sp.ppmlhdfe` users whose absorbed effects have
one-observation groups (unbalanced panels, many crossed effects), and code
that calls `sp.aggte(fit)` without `bstrap=`, `sp.winsor` users (cutoffs move
slightly), and binary-choice
fits with a category that predicts the outcome perfectly.

| Area | Old | New | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.ppmlhdfe`, singleton groups | kept; `N`, cluster count, `G/(G-1)` and pseudo R² differ from Stata | dropped with the separated rows (Stata default); `model_info['n_singletons']` | `drop_singletons=False` |
| `sp.ppmlhdfe`, regressor absorbed by the FE | `LinAlgError` | omitted with a warning, `model_info['omitted']` | — |
| `sp.ppmlhdfe`, rows with a missing estimation variable | NaN reached the solver | dropped | — |
| `sp.aggte(fit)` default SE | unseeded multiplier bootstrap, new SE every call | the fit's setting: analytic for a default `callaway_santanna` fit | `sp.aggte(fit, bstrap=True, cband=True, random_state=...)` |
| `sp.aggte(..., bstrap=True)` without a seed | seed not recorded | seed drawn and stored in `model_info['random_state']` | — |
| `sp.winsor` cutoffs | numpy linear interpolation | Stata `_pctile` (`winsor2`) | `method='linear'` |
| `sp.callaway_santanna(clustervars=..., bstrap=False)` | `MethodIncompatibility` | clustered analytic SEs (csdid) | — |
| `sp.aggte(bstrap=False)` on a clustered fit | unclustered SE | clustered SE | — |
| `sp.aggte(agg_weights='csdid')` on `panel=False` fits | cell weights treated as fixed | csdid's weight influence included (SE up to ~1% larger) | — |
| `sp.logit` / `sp.probit` / `sp.cloglog` with an indicator that predicts the outcome perfectly | rows kept, no warning; MLE does not exist | rows and indicator dropped (Stata), warning | `perfect_prediction='keep'` |

---

<a id="hdfe-t-g-minus-1"></a>

## 1.33.0 — ⚠️ `sp.hdfe_ols` clustered inference on `t(G-1)`; HonestDiD relative-magnitude sets no longer truncated; `sp.ri_test(cluster=)` checks the design

**Who is affected.** Anyone reading p-values, stars or CIs from clustered
`sp.hdfe_ols` fits (point estimates and SEs are unchanged); anyone whose
`sp.honest_did(method='relative_magnitude')` set reached the ±20 sd grid
(it warned); anyone running `sp.ri_test(cluster=)` with a treatment that
varies within clusters.

| Area | Old | New | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.hdfe_ols(cluster=)` p-values / CIs | `t(N - K - df_a)` | `t(min(G) - 1)`, as `reghdfe` | `df_inference='resid'` |
| `sp.honest_did` relative-magnitude bound at the grid edge | grid end reported (warning) | grid extended until the set closes | `grid_expand=False` |
| `sp.ri_test(cluster=)`, treatment varying within clusters | first row's value used for the cluster | `MethodIncompatibility`; `treat=<cluster-level>, interact=<pattern>` | — (the old distribution was not the design's) |
| `sp.ri_test(stat='diff_means'/'t'/'ks')`, non-binary treatment | silently compared `d == 1` to `d == 0` | `MethodIncompatibility` | use `stat='ols'` |
| `sp.hdfe_ols` with an absorbed varying slope, default `tol` | `1e-8` | `1e-12` (SEs move in the 5th-6th digit) | `tol=1e-8` |
| `sp.iv(absorb='a^b')`, rows with `a` or `b` missing | kept as a `"nan"` FE level | dropped, as `ivreghdfe` | pre-fill the missing values yourself |
| `sp.from_stata("reghdfe ...")` | `tool='feols'`, `arguments['fml']` | `tool='hdfe_ols'`, `arguments['formula']` | call `sp.feols` yourself |
| `sp.from_stata("ivreghdfe ...")` with `absorb()` | `tool='feols'` | `tool='hdfe_ols'` (IV part) | call `sp.feols` yourself |

---

<a id="ppml-separation-etwfe-link"></a>

## 1.33.0 — ⚠️ `sp.ppmlhdfe` drops separated observations; `sp.etwfe(family='poisson')` gains `scale=` / `fe=`

**Who is affected.** Anyone running `sp.ppmlhdfe` on data with fixed-effect
groups whose outcome is always zero (or a regressor that is non-zero only
where the outcome is zero), and anyone reproducing Stata `jwdid ...,
method(ppmlhdfe)` tables with `sp.etwfe`.

| Area | Old | New | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.ppmlhdfe`, separated rows | kept (warning only); slow or incomplete convergence, `N` / clusters include them | dropped (Stata default rules `fe simplex`), `model_info['n_separated']`, warning | `separation='warn'` |
| `sp.ppmlhdfe`, regressor zero after dropping | singular solve | omitted with a warning (Stata's "omitted") | `separation='warn'` |
| `sp.etwfe(family='poisson')` headline | response-scale AME only | unchanged by default; `scale='link'` gives jwdid's `estat simple, predict(xb)` number | — |
| `sp.etwfe(family='poisson')` on an unbalanced panel | cohort dummies, silently different from jwdid | same numbers plus a warning; `fe='unit'` reproduces jwdid / ppmlhdfe | `fe='cohort'` (default) |
| `sp.etwfe(family=..., cgroup='nevertreated')` | raised `MethodIncompatibility` | supported (adds pre-period cells) | — |
| nonlinear `sp.etwfe_emfx(type='group'/'event'/'calendar')` rows | `att`, `se`, `n_treated` | also `pvalue`, `ci_lower`, `ci_upper`; `scale=` selects the scale | — |

---

<a id="agent-surface-hardening"></a>

## 1.33.0 — ⚠️ MCP errors as `isError` results, remote data opt-in, `next_steps()` stops printing

**Who is affected.** MCP clients and scripts that parse `statspai-mcp`
responses, code that calls `handle_request` in-process, shell pipelines
that branch on `statspai run` exit codes, and anyone relying on
`result.next_steps()` printing. No estimate, standard error or default
seed changed.

| Area | Old | New | How to keep the old behaviour |
| --- | --- | --- | --- |
| MCP tool failure (bad / expired `data_id` or `result_id`, data-load error, invalid argument value, timeout) | JSON-RPC `error` (-32602 / -32000), message only | `result` with `isError: true`; details in `structuredContent` (`error_kind`, `message`, `hint`, `miss_reason`) | read `result.structuredContent.error_kind` instead of `error.code` |
| MCP unknown tool name | `isError` result | JSON-RPC -32602 (protocol error, per spec) | — |
| MCP stale `result_id` | silently ignored | `isError`, `error_kind: "missing_result_handle"` | pass a live handle |
| MCP `data_path` URL (`http(s)` / `s3` / `gs`) | fetched, no byte cap | refused (`remote_disabled`) unless `STATSPAI_MCP_ALLOW_REMOTE=1`; capped by `STATSPAI_MCP_MAX_DATA_BYTES` | set `STATSPAI_MCP_ALLOW_REMOTE=1` |
| MCP `transform_data` `query` / `assign` | any pandas expression (`engine="python"`) | AST allowlist: columns, literals, arithmetic / comparison / boolean operators, `in`, a fixed set of math functions and `isnull` / `isin` / `between` / `.str.contains`-style column methods; attribute chains, dunders, subscripts and `@` locals are refused (`unsafe_expression`) | — |
| In-process `handle_request` tool profile | `full` | `curated` (as the CLI) | `STATSPAI_MCP_PROFILE=full` |
| MCP text content block | pretty-printed JSON | compact JSON; oversized payloads truncated (see `truncated`) | `max_output_bytes=0` disables the budget |
| `CausalResult.next_steps()` / `EconometricResults.next_steps()` | returned the list and printed a banner | returns the list only | `r.next_steps(print_result=True)` |
| `statspai run` exit code | 3 for every estimator error | 4 input errors (`column_not_found`, `missing_arguments`, `unknown_argument`), 5 `missing_dependency`, 3 otherwise | treat any non-zero, non-2 code as an estimator error |
| Remediation category | `bad_argument` | `unknown_argument` (with `did_you_mean`) | — |
| `sp.all_schemas()` | included 320 classes | functions only | `sp.all_schemas(include_classes=True)` |
| Schema of `data` parameters | `"type": "string"` | `"type": "object"`, `x-statspai-role: "dataframe"` | — |

**Exceptions.** `ColumnNotFound` subclasses both `DataInsufficient` and
`MethodIncompatibility` (and `ValueError`), and `MissingDependencyError`
subclasses `ImportError`, so existing `except` clauses keep catching them.

---

<a id="design-and-diagnostic-fixes"></a>

## 1.33.0 — ⚠️ `optimal_design` sample sizes, Hausman test, `compare_estimators`, `estat` IV tests

**Who is affected.** Anyone who sized a study with `sp.optimal_design`, chose
between fixed and random effects with `sp.hausman_test` or
`PanelResults.hausman_test`, compared estimators with `sp.compare_estimators`,
or ran `sp.estat` after an IV fit.

| Area | Old | New | Old number, if you need it |
| --- | --- | --- | --- |
| `optimal_design` individual / stratified, `mde=` given | total sample reported as `n_per_arm`, `n_total` twice that — every design twice as large as needed | `n_total` = total sample, `n_per_arm` = largest arm (the two-sample power formula; equals `sp.power("rct")` up to per-arm rounding) | — (old sizes were twice the correct ones) |
| `optimal_design` cluster, `mde=` given | cluster count doubled the same way | smallest design reaching the target power (checked against `sp.power_cluster_rct`) | — |
| `optimal_design` with `mde=None` | MDE for a sample of one (ignored the sample size); cluster MDE used the treated arm as the total | needs `n=` (individual / stratified) or `n_clusters=` (cluster) and returns the MDE for that sample | — |
| `optimal_design` with `cost_per_cluster` / `cost_per_unit` | cost-optimal cluster size reported but not used | the cost-optimal size drives the sample size | — |
| `sp.hausman_test` statistic | own FE / RE variance estimates (223 where Stata reports 4.36 on the same panel) | same as `PanelResults.hausman_test` and Stata `hausman fe re` to 1e-6 | — |
| Hausman, negative statistic (both entry points) | clamped to 0: `pvalue=1`, `recommendation="RE"` | reported as is, `pvalue=nan`, `recommendation="inconclusive"`, `AssumptionWarning` | — |
| `sp.compare_estimators` | `matching` (default list) and `dml` silently absent (keyword errors caught as warnings); IPW SE unseeded | all run; IPW bootstrap seeded | — |
| `sp.estat` after `sp.ivreg` / `sp.iv` | `endogenous` / `overid` always "not found"; `all` ran no IV test | read the fit's Wu-Hausman F and Sargan / Hansen J | — |

**New.** `sigmamore=True` on both Hausman entry points reproduces Stata's
`hausman fe re, sigmamore` and gives a usable test where the classical statistic
is negative. `sp.test` / `sp.lincom` accept multi-coefficient restrictions on
`sp.panel` results (e.g. the Mundlak test `sp.test(res, "_mean_x1 _mean_x2")`).
`sp.dgp_bunching` now generates bunching at the kink (new `t0=` / `t1=`), so its
seeded draws differ from earlier releases.

---

<a id="sdid-staggered"></a>

## 1.33.0 — ⚠️ `sp.did` / `sp.did_analysis` with `method="sdid"` estimate staggered adoption by cohort

**Who is affected.** Anyone who ran `sp.did(..., method="sdid")` or
`sp.did_analysis(..., method="sdid")` on a panel whose `treat` column holds
more than one first-treatment period, and anyone who relied on
`sp.did(method="sdid")` standard errors being reproducible.

**What changed.** Those calls used the earliest adoption period for every
treated unit and returned a number for a different estimand. They now fit
each adoption cohort against the never-treated units and average the cohort
effects (`sp.sdid(treat=...)`, Stata `sdid`'s estimator). A panel with one
adoption cohort gives the same estimate as before. `sp.did(method="sdid")`
now seeds its placebo / bootstrap draws with `seed=` (or `random_state=`),
which were previously ignored, so its SE is reproducible and differs from
an earlier unseeded run.

**What to do.** Re-estimate `sdid` numbers from staggered panels. The
placebo SE needs more never-treated than treated units (Stata's rule); use
`se_method="bootstrap"` otherwise.

---

<a id="result-display-precision"></a>

## 1.33.0 — `.to_latex()`, `.summary()` and generic result tables use the adaptive precision

Display only; no estimate, standard error or p-value changes.

| | before | now (default) |
| --- | --- | --- |
| `CausalResult.summary()` | `0.332635` (SE `0.084764`), p `0.0001` | `0.333` (SE `0.085`), p `<0.001` |
| `CausalResult.to_latex()` | `0.3326` (`%.4f`, no argument) | `0.333`; `digits=` / `fmt=` accepted |
| generic `*Result.to_latex()` | `-0.09446` / `5.386e+05` | `-0.094` / `538,582` |
| generic `to_markdown()` / `to_word()` | `-0.094458` / `538582` | `-0.094` / `538,582` |
| `regtable(stats_fmt=)` | `"%.3f"` | `"stat"` (same for values below six figures) |
| `to_dict()` / `to_excel()` | full value | **unchanged** |

To restore the old output:

```python
result.to_latex(digits=4)
result.summary(digits=6)
sp.regtable(m1, m2, stats_fmt="%.3f")
```

---

<a id="plr-theta-label"></a>

## 1.32.0 — DML PLR estimate is labelled `theta`; `cate_summary` row `Mean`

**Who is affected.** Code that indexes a partially linear DML result by
its old label, e.g. `sp.dml(..., model="plr").params["ATE"]`,
`sp.regtable(..., keep=["ATE"])` for a PLR column, or
`sp.cate_summary(res).loc["Mean (ATE)"]`.

**What changed.** No number changes; two labels were wrong.

- `sp.dml(model="plr")` (and `sp.DoubleMLPLR`, `sp.dml_model_averaging`)
  now report `estimand == "theta"`. The PLR parameter theta in
  `Y = theta*D + g(X) + e` equals the ATE only under a constant effect;
  with heterogeneous effects of a binary `D` it is a variance-weighted
  average, and with a continuous `D` (years of schooling, a dose) it is a
  partial coefficient. `model="irm"` still reports `ATE`, and the IV
  models `LATE`.
- `sp.cate_summary` names its first row `Mean` instead of `Mean (ATE)`. It
  is the plain average of the fitted conditional effects, not the
  estimator's doubly-robust average (`cf.average_treatment_effect()` for a
  forest), and for a continuous treatment it averages partial effects.

**What to do.**

```python
res = sp.dml(df, y="y", d="d", X=X, model="plr")
res.estimate            # unchanged, prefer this
res.params["theta"]     # was res.params["ATE"]
sp.regtable(ols, res, keep=["d", "theta"])
sp.cate_summary(cf).loc["Mean"]   # was .loc["Mean (ATE)"]
```

---

<a id="qreg-t-inference"></a>

## 1.32.0 — ⚠️ `sp.qreg` p-values and intervals use t(N - k)

**Who is affected.** Anyone reading p-values, stars or confidence intervals
from `sp.qreg` (any `vce`), including `result.ci` and `detail["pvalue"]`.

**What changes.** The statistic is referred to t with N - k degrees of
freedom, as Stata `qreg` and R `quantreg` do, instead of the normal.
Coefficients and standard errors do not change; intervals widen by the
ratio of the t and normal critical values. `detail` has a new `t` column;
the old `z` column still exists and holds the same statistic.

**To reproduce old numbers.** `2 * scipy.stats.norm.sf(abs(b / se))` and
`b +/- scipy.stats.norm.ppf(0.975) * se` from `detail`.

---

<a id="panel-cre-theta"></a>

## 1.32.0 — ⚠️ `sp.panel(method="mundlak" | "chamberlain")` mean coefficients move slightly

**Who is affected.** Anyone reading the coefficients on the unit means
(`_mean_<x>`), the Chamberlain terms or the constant of a correlated
random-effects fit, or the Mundlak Wald statistic built on them.

**What changes.** The random-effects variance components no longer count
the unit-mean columns as degrees of freedom (they add no rank). theta is now
Stata's, and the mean coefficients and the constant move by a few parts in
ten thousand on typical panels. The coefficients on the original
regressors (the FE estimates) are unchanged.

**To reproduce old numbers.** Fit `linearmodels.panel.RandomEffects`
directly on the augmented design.

---

<a id="panel-weights-cluster"></a>

## 1.32.0 — ⚠️ `sp.panel(weights=, cluster=)` and `sp.did_summary(cluster=)` take effect

**Who is affected.** Anyone who passed `weights=` to `sp.panel`, passed
`cluster=` with a column name other than the panel entity (e.g. a state
id for county panels), or passed `cluster=` to `sp.did_summary`.

**What changes.** `sp.panel` weighted `fe` / `twoway` / `pooled` fits are
now WLS (they were unweighted) and a named cluster column is clustered on
(it used to cluster on the entity). Weights with `re` / `be` / `fd` /
`mundlak` / `chamberlain` / GMM now raise. The CS row of `sp.did_summary`
clusters on the requested variable through the multiplier bootstrap.

**To reproduce old numbers.** Drop `weights=`; use `cluster='entity'`;
for `did_summary`, drop `cluster=` (the CS row always clustered by unit).

---

<a id="pwcorr-pairwise"></a>

## 1.32.0 — ⚠️ `sp.pwcorr` deletes missing values pairwise

**Who is affected.** Anyone calling `sp.pwcorr` on data with missing values.

**What changes.** Each correlation now uses every row on which *that pair*
is observed (Stata `pwcorr`'s default); it used to drop any row with a gap
in any listed variable. The text output reports the range of pairwise N;
`output="dataframe"` carries `.attrs["nobs"]` and `.attrs["pvalues"]`.

**To reproduce old numbers.** `sp.pwcorr(..., listwise=True)`.

---

<a id="margins-stata-semantics"></a>

## 1.32.0 — ⚠️ `sp.margins` / `margins_at` / `contrast` / `pwcompare` follow Stata's averaging

**Who is affected.** Users of the margins family with (a) `poisson` /
`nbreg` / `glm` fits that used `exposure=` / `offset=`, (b) weighted fits,
(c) `data=` containing rows outside the estimation sample, (d)
`method="mem"` with factor variables, (e) p-values / intervals from
`margins_at` / `contrast` / `pwcompare` after `regress`, or (f) the default
`variables` list of a model with `C()` terms.

**What changes.** (a) The offset / exposure is part of the prediction --
this is the largest change (the AME scales with the mean exposure). (b) The
average is weighted by the fit's weights. (c) Rows missing the outcome or a
model / cluster / weight variable are dropped (Stata `e(sample)`), with a
warning when the remaining count differs from the fit's `nobs`; they used to
be averaged in or to raise. (d) `atmeans` sets factor indicators to their
shares. (e) t(df) instead of N(0, 1). (f) Factor variables get discrete-change
rows (`"2.g"`), as Stata's `dydx(*)`; transformations such as `I(x**2)` are
differentiated through, so an `x + I(x**2)` model reports the total effect
of `x`. `margins_at` / `contrast` / `pwcompare` now run after logit / probit
/ poisson on the response scale instead of raising.

**To reproduce old numbers.** There is no switch for (a)-(e): those were
defects against Stata (see `tests/reference_parity/test_r2_postest_parity.py`
and `test_margins_ext_parity.py`). For (f) pass `variables=[...]` listing the
continuous variables, and pre-compute a transformed column (e.g. `x2`) if
you want it treated as an independent regressor.

---

<a id="mice-categorical"></a>

## 1.32.0 — ⚠️ `sp.mice` imputes categorical and binary variables with models

**Who is affected.** Anyone using `sp.mice` on data with non-numeric
columns or binary variables, and anyone whose other variables are imputed
from data that contains categorical columns.

**What changes.** Two-level variables default to proper `'logreg'`
(parameters drawn from their posterior; the MLE was used before) and
multi-level non-numeric variables to the new `'polyreg'` (they used to get
`'sample'`, random draws from their own marginal, which attenuates every
association with them). Categorical columns now enter the other variables'
equations as dummies (they were ignored). Imputed values and pooled
estimates change.

**To reproduce old numbers.** Not exactly (the random streams differ);
`method={"var": "sample"}` restores the marginal draw for a variable.

---

<a id="rdrobust-fuzzy-comb"></a>

## 1.32.0 — ⚠️ `sp.rdrobust` fuzzy designs with a `comb` bandwidth selector

**Who is affected.** `sp.rdrobust(fuzzy=..., bwselect="msecomb1" |
"msecomb2" | "cercomb1" | "cercomb2")`.

**What changes.** The bandwidth is now the fuzzy (Wald-ratio) one, as R
`rdrobust`; the comb recursion used to drop the first stage and return the
sharp bandwidth, moving h and the estimate substantially (h 0.18 vs 0.32,
estimate 0.29 vs 0.56 on the test design).

**To reproduce old numbers.** Compute the sharp comb bandwidth
(`sp.rdrobust(data, y=..., x=..., bwselect="msecomb2")` without `fuzzy=`)
and pass it as `h=` / `b=`.

---

<a id="mcp-sample-rule"></a>

## 1.32.0 — MCP `data_sample_n` draws a different (deterministic) sample

**Who is affected.** MCP clients that pass `data_sample_n`.

**What changes.** The sample is the `data_sample_n` rows with the smallest
seed-0 uniform keys, in file order, identical whether the file is loaded
whole or streamed; it used to be `DataFrame.sample(random_state=0)` on the
loaded frame (different rows, shuffled order).

**To reproduce old numbers.** Load the file yourself and call
`df.sample(n, random_state=0)`.

---

<a id="qreg-default-se"></a>

## 1.32.0 — ⚠️ `sp.qreg` default standard errors follow Stata's `qreg`

**Who is affected.** Anyone reading standard errors, p-values or intervals
from `sp.qreg` (and `sp.sqreg`, which calls it) without choosing a variance.

**What changes.** The default SE is now Stata's `qreg` default,
`vce(iid)`: the sparsity is the difference quotient of the mean fitted
quantiles at `tau +/- h` with the Hall-Sheather bandwidth. It used to be a
Gaussian-kernel density of the residuals with a Silverman bandwidth, which
matched neither Stata's `qreg` nor R's `quantreg` (2-8% off). Coefficients are
unchanged. New choices: `vce="robust"` (Stata `vce(robust)`) and
`vce="nid"` (R `quantreg` `se="nid"`).

**To reproduce old numbers.** `sp.qreg(..., vce="powell")`.

---

<a id="tobit-polish"></a>

## 1.32.0 — ⚠️ `sp.tobit` estimates move by ~1e-6

**Who is affected.** Anyone comparing `sp.tobit` output digit by digit
with an earlier release.

**What changes.** The fit is Newton-polished to the optimum and the SEs
come from the polished observed information; coefficients move by ~1e-6
and SEs by up to ~5e-5 (relative), towards Stata `tobit` and R `censReg`
(now 1e-11 apart on the Track A fixture). There is no switch: the old
digits were an optimiser stopping short.

---

<a id="iv-weights-alpha"></a>

## 1.32.0 — ⚠️ `sp.iv(weights=)` and fit-time `alpha=` take effect

**Who is affected.** (1) Anyone who passed `weights=` to `sp.iv` or
`sp.ivreg`. (2) Anyone who passed `alpha=` to one of the 31 estimators
listed in the CHANGELOG (`logit`, `poisson`, `cox`, `liml`, `frontier`,
`cr2_se`, ...) and read intervals from the result. (3) Anyone who read
intervals from `summary(alpha=...)` on an `EconometricResults`. (4) Users of
`sp.synth(method="mc", covariates=...)` or
`sp.synth(method="augmented", placebo=False)`.

**What changes.** (1) The weights are applied: point estimates and SEs are
now the weighted ones (Stata `ivregress [aw=]`); previously the unweighted
fit was returned. (2) and (3) Intervals are at the requested level;
previously they were always 95%. Point estimates, SEs and p-values are
unchanged. (4) The covariates are used and the placebo loop is skipped.

**Also newly loud.** Unknown keywords on `sp.iv`, `sp.ivreg` (including
`small=` and `iv_diag=`), `sp.synth(method="classic")` and `sp.augsynth`
raise `TypeError`; `sp.synth_compare` warns about skipped methods; dropping
incomplete rows emits an `AssumptionWarning`; NaN standard errors emit a
`ConvergenceWarning`.

**To reproduce old numbers.** (1) Omit `weights=`. (2)-(3) Omit `alpha=` or
pass `alpha=0.05`. (4) Omit `covariates=` / call `sp.augsynth` directly.
To silence the listwise-deletion note, drop incomplete rows before the call.

---

<a id="cs-event-study-share-term"></a>

## 1.31.0 — ⚠️ Callaway--Sant'Anna convenience event study carries the cohort-share term

**Who is affected.** Anyone reading standard errors, confidence intervals or
p-values from `sp.callaway_santanna(...).model_info['event_study']` (also
shown by the fit's summary and plots) rather than from
`sp.aggte(cs, type="dynamic")`.

**What changes.** Those standard errors now include the estimation error of
the cohort shares that weight each event time, as R `did::aggte` and
`sp.aggte` do. They grow wherever an event time mixes cohorts (up to 47% on
the castle-doctrine panel); single-cohort event times and all point
estimates are unchanged. The table now equals
`sp.aggte(cs, type="dynamic", bstrap=False)`.

**To reproduce old numbers.** There is no switch: the old numbers treated
estimated weights as known and were anti-conservative. `sp.aggte` has
reported the corrected values since the share term was added there.

---

<a id="pretrends-joint-covariance"></a>

## 1.31.0 — ⚠️ `sp.pretrends_test` / `pretrends_power` / `sensitivity_rr` use the joint pre-period covariance

**Who is affected.** Anyone who passed an `sp.event_study` result (without
`expose_pre_vcov=True`) or a `sun_abraham` / `stacked_did` /
`did_imputation` result to `sp.pretrends_test`,
`sp.pretrends_power` or `sp.sensitivity_rr`. Those calls warned "treated as
MUTUALLY INDEPENDENT" and used a diagonal covariance.

**What changes.** The pre-trend statistics now use the joint cluster-robust
covariance of the pre-treatment coefficients, the same one the event
study's own `model_info['pretrend_test']` uses. The p-value can move in
either direction. On the castle-doctrine panel it moved from 0.603 to 0.277
(`type="wald"`) and 0.293 (`type="f"`, now matching Stata's `test`). On `sun_abraham` /
`stacked_did` / `did_imputation` results the pre-period block now comes
from `sp.event_study_vcov`.
`type="f"` also switches its denominator degrees of freedom from
`n_obs - K` to `G - 1` for clustered results, and the default `type` is now
`"auto"`: F(K, G - 1) on an `sp.event_study` result (its own convention and
Stata's), chi2 Wald otherwise. Pass `type="wald"` for the old default
statistic.

**To reproduce old numbers.** `sp.event_study(..., expose_pre_vcov=False)`
restores the diagonal path for event-study results. Only reproduce old
numbers this way; the diagonal ignores the covariance between coefficients
that share a reference period and fixed effects.

---

<a id="feols-ssc-default"></a>

## 1.31.0 — ⚠️ `sp.fast.feols` small-sample correction and `sp.iv(vce=)`

**`sp.fast.feols`.** The default `ssc` is now `"fixest"`, the convention of
`fixest::feols` and `reghdfe` (and of the parity rows). Coefficients do not
change. Standard errors change:

* `vcov="cr1"` with absorbed effects nested in the cluster variable (e.g.
  unit effects, clustered by unit): smaller, because those effects no longer
  enter the small-sample factor. The old factor over-covered.
* `vcov="iid"` / `"hc1"` with two or more absorbed dimensions: the residual
  degrees of freedom fall by one per extra dimension (`n - p - (ΣG_k - 1)`
  instead of `n - p - Σ(G_k - 1)`), so SEs rise slightly.

```python
sp.fast.feols("y ~ x | i + t", data=df, vcov="cr1", cluster="i")                  # new default
sp.fast.feols("y ~ x | i + t", data=df, vcov="cr1", cluster="i", ssc="statspai")  # old numbers
```

**`sp.iv`.** `vce=` / `vcov=` used to be ignored (CR1 was returned for
`vce="cr2"`). They are now honoured, bit-identically to `sp.ivreg`, or
rejected with `MethodIncompatibility` when the chosen `method=` cannot
compute them. If you passed `vce=` to `sp.iv` before, your SEs change to the
ones you asked for.

**`sp.validation_scope`.** The return value gained `outputs`, `unchecked`
and `invariant`; `status` can now be `estimate_only` or `disclosure_only`;
explicit dimension values outside the domain raise; the forest `trees`
dimension is the exact count (`"2000"`, not `">=2000"`) and the DML
`learners` dimension is `linear` / `default` / `other` (not `flexible`).
<a id="surrogate-result-deprecated"></a>

## 1.31.0 — `SurrogateResult` deprecated

**Who is affected.** Code that imports or constructs
`statspai.SurrogateResult`.

**What changed.** Nothing in StatsPAI ever returned this class:
`sp.surrogate_index`, `sp.long_term_from_short` and
`sp.proximal_surrogate_index` return `sp.CausalResult`. Constructing it now
raises a `DeprecationWarning`; it will be removed in 1.33. Use the
`CausalResult` the estimators return (`.estimate`, `.se`, `.ci`,
`.model_info`).

---

<a id="grf-family-rebuild"></a>

## 1.31.0 — ⚠️ `iv_forest`, `multi_arm_forest`, `causal_survival_forest` rebuilt on the GRF engine; engine seeding fixed

**Who is affected.** Anyone calling these three functions, and anyone who
compares `sp.causal_forest` (or any GRF-engine forest) numbers across
releases for a fixed `random_state`.

**What changed.**

1. The three estimators were stand-ins (see CHANGELOG, ⚠️ Correctness) and
   now implement the methods their names promise. Their numbers change --
   not within Monte Carlo error, but because the old estimates were of
   something else (a global Wald ratio, in-sample AIPW scores, a CATE from
   a forest that could ignore the treatment).
2. Forest options follow `sp.causal_forest`; the old spellings keep working:

   | old | new |
   | --- | --- |
   | `n_trees=` | `n_estimators=` (alias kept) |
   | `min_leaf=` | `min_samples_leaf=` (alias kept) |
   | `n_bootstrap=` (`iv_forest`) | removed from the method; ignored with `DeprecationWarning` |
   | `propensity_bounds=(0.05, 0.95)` default | default `None` (no clipping, as grf); pass bounds to clip |
   | — | `split_alpha=` for grf's `alpha`; `alpha=` is still the significance level |

   Defaults are grf's: 2000 trees, `min_samples_leaf=5`,
   `ci_group_size=2`. The old defaults (200-300 trees) are much smaller;
   pass `n_estimators=` to keep runs short.
3. Result fields are kept: `iv_forest(...).late / se / ci / pvalue / cate /
   n_obs / detail` (`late` is now the doubly-robust average conditional
   LATE, `cate` is out-of-bag); `multi_arm_forest(...).arms / ate / ate_se /
   ci / cate / n_obs / detail`; `causal_survival_forest(...).ate_rmst / se /
   ci / pvalue / cate / horizon / n_obs / n_trees / detail`. New: `predict()`,
   `get_scores()`, `average_treatment_effect()`, `best_linear_projection()`,
   `variable_importance()`, and `target=` for survival probabilities.
4. **Seeding.** GRF-engine forests used `seed + g` for tree group `g`, so
   consecutive seeds shared nearly every tree and nuisance forests shared
   subsamples with the main forest. For a given `random_state` every GRF
   forest -- including `sp.causal_forest` -- now returns a different (and
   independent) draw. Differences are Monte Carlo noise; if you pinned
   exact forest outputs in tests, re-pin them.

```python
# before
r = sp.iv_forest(df, y="y", treat="d", instrument="z", covariates=X, n_trees=300, n_bootstrap=50)
# after (same call works; this is the idiomatic form)
r = sp.iv_forest(df, y="y", treat="d", instrument="z", covariates=X, n_estimators=2000)
r.late, r.se                       # doubly-robust ACLATE and its SE
sp.best_linear_projection(r, A=df[["age"]])
```

**`CausalForest.variable_importance()`** now takes `method=`. Without it
you get the old permutation measure and a `FutureWarning`; from 1.33 the
default is `"split"` (grf's `variable_importance`, also `sp.variable_importance`).

<a id="ebalance-mestimation-se"></a>

## 1.31.0 — ⚠️ `sp.ebalance` standard errors account for the balancing

**Who is affected.** Anyone reading `.se`, `.ci` or `.pvalue` from
`sp.ebalance`. Point estimates do not change.

**What changed.** The standard error treated the entropy-balancing weights
as fixed and ignored that they balance the covariates exactly, so it
overstated the uncertainty whenever the covariates predict the outcome --
by about 2x on a design where they explain most of its variance. The default
is now the M-estimation sandwich (`vce="mestimation"`), the variance R
`WeightIt::lm_weightit(vcov = "asympt")` reports for `method = "ebal",
estimand = "ATT"`.

```python
r = sp.ebalance(df, y="y", treat="d", covariates=["x1", "x2"])          # new SE
r_old = sp.ebalance(df, y="y", treat="d", covariates=["x1", "x2"],
                    vce="naive")                                         # old SE
```

**What to do.** Re-run inference that used `sp.ebalance`; intervals will
usually be narrower. Use `vce="naive"` only to reproduce earlier numbers.

---

<a id="causal-question-forest-ate"></a>

## 1.31.0 — ⚠️ `causal_question(design="causal_forest")` reports the forest's own ATE

**Who is affected.** Anyone reading `.estimate`, `.se` or `.ci` from
`sp.causal_question(..., design="causal_forest").estimate()`.

**What changed.** The ATE, SE and interval used to come from a separate
cross-fit AIPW with gradient-boosting nuisances; the fitted forest was kept
only on `result.underlying`. They now equal
`result.underlying.average_treatment_effect(target_sample="all")`, the
forest's own doubly-robust average over its out-of-bag predictions.

**What to do.** Re-run. If you passed a very small `n_estimators` (e.g. 30),
some rows may have no out-of-bag prediction and the call raises
`DataInsufficient`; use the default 2,000 trees. `aipw_n_folds=` no longer
has an effect.

---

<a id="honest-did-native-rm"></a>

## 1.31.0 — `sp.honest_did(method="relative_magnitude")` is the Rambachan-Roth set natively

**Who is affected.** Calls with `method="relative_magnitude"` and the
default `backend="native"`.

**What changed.** The native path returned a worst-case-bias approximation
(`theta_hat +/- Mbar*max|pre| +/- z*SE`) and warned. When the fit carries the
joint event-study covariance it now returns the Andrews-Roth-Pakes confidence
set HonestDiD reports (`honestdid_method="C-LF"` by default, or
`"Conditional"`), reported on HonestDiD's +/-20 SD grid, so bounds are grid
points. `result.attrs["interval"]` says which you got (`"arp_c_lf"`,
`"arp_conditional"`, or `"worst_case_bias"` when no covariance is
available, still with a warning). `sp.breakdown_m(...,
method="relative_magnitude")` follows.

---

<a id="audit-not-applicable"></a>

## 1.31.0 — `sp.audit` reports checks that do not exist for a fit as `not_applicable`

**Who is affected.** Code that branches on `sp.audit(...)["checks"][i]
["status"]` or on `summary["n_total"]`, for IV fits.

**What changed.** An over-identification check on a just-identified IV fit
now has `status == "not_applicable"` (with a `reason`, and
`suggest_function` is `None`) instead of `"missing"`. `summary["n_total"]`
counts applicable checks only (`passed + failed + missing`), and
`summary["not_applicable"]` counts the rest, so `coverage` is no longer
diluted by checks that cannot be run.

---

<a id="forest-continuous-att"></a>

## 1.31.0 — ⚠️ causal forests with a continuous treatment refuse ATT and ATC

**Who is affected.** Code that calls `cf.att()`, or
`cf.average_treatment_effect(target_sample="treated")` / `"control"`, on a
forest fitted with a non-binary treatment (`discrete_treatment=False`).

**What changed.** These calls used to select the "treated" rows as
`T == 1`. For a dose such as years of schooling, that is the handful of units
whose dose happens to equal one, so the returned "ATT" was an average over an
arbitrary slice, printed with a zero-width interval. They now raise
`MethodIncompatibility`, as grf does for a non-binary treatment.

```python
cf.ate()                                                   # average partial effect (unchanged)
cf.average_treatment_effect(target_sample="overlap")       # partially linear coefficient (unchanged)
cf.att()                                                   # now raises MethodIncompatibility
```

**What to use instead.** `ate()` for the average partial effect. For effects
among units above some dose, fit a forest on the binary indicator you mean
(for example, `educ >= 16`) and call `att()` on that forest.

---

<a id="forest-grf-defaults"></a>

## 1.29.0 — ⚠️ `sp.causal_forest` follows grf's defaults, and fitted effects move

*(Added after release. The 1.29.0 CHANGELOG listed this under Changed.)*

**Who is affected.** Anyone who reports CATE predictions, their spread, group
effects, or calibration results from `sp.causal_forest` fitted with 1.28.0
or earlier.

**What changed.** The default forest is now grf's: `n_estimators=2000` (was
100), honest subsampling without replacement (`bootstrap=True` raises), and
nuisances from out-of-bag regression forests. Fitted effects are less noisy.
On the bundled Card data (`educ` as a continuous treatment; `exper`,
`expersq`, `black`, `south`, `smsa`; `random_state=42`):

| | 1.28.0 | 1.29.0 | R grf 2.6.1 |
| --- | --- | --- | --- |
| CATE mean | 0.0793 | 0.0766 | 0.0762 |
| CATE std. dev. | 0.0810 | 0.0164 | 0.0168 |
| CATE Q25 / Q75 | 0.0575 / 0.1031 | 0.0633 / 0.0869 | 0.0634 / 0.0869 |

The 1.28.0 spread was about five times what grf estimates on the same bytes.
The difference comes from the engine, not the tree count: all three columns
use 2,000 trees. Refit instead of reusing earlier heterogeneity results. To
reproduce an old number, pass `split_rule="legacy"` (deprecated) with the
same arguments. On this design it returns the 1.28.0 column exactly.

---

<a id="fe-forest-imputation"></a>

## 1.31.0 — ⚠️ causal forests with fixed effects: calibration uses imputation scores

**Who is affected.** Code that calls `sp.calibration_test(cf)` or
`sp.calibrate_cate(cf)` on a forest fitted with `fe="twoway"` and a binary
treatment. Pooled forests, `fe="unit"` forests and FE forests with a
continuous treatment are unchanged.

**What changed.** 1.29.0 regressed the globally within-transformed outcome
on the within-transformed treatment times the OOB prediction. Global two-way
demeaning is exact only under a constant effect, so that slope was not a
de-attenuation factor (`calibrate_cate` warned about it). The default is now
the imputation regression: each treated cell's `Y - alpha_hat_i -
gamma_hat_t` (unit and period effects fitted on untreated cells) is
regressed on the OOB prediction. Both coefficients and their standard errors
change; `calibrate_cate(...)["method"]` is `"blp_imputation"`.

```python
sp.calibration_test(cf)                   # imputation regression (new default)
sp.calibration_test(cf, method="within")  # the 1.29.0 numbers
sp.calibrate_cate(cf, method="within")    # the 1.29.0 rescaling
```

At the same time `cf.average_treatment_effect("treated")`,
`cf.best_linear_projection()` and `sp.forest_group_effects(cf)` stop raising
for FE forests and return imputation-based estimates.
`average_treatment_effect()` with `target_sample="all"`, `"control"` or
`"overlap"` and `sp.rate(cf)` still raise.

---

## 1.29.0 — DML orthogonality diagnostic is unavailable

`dml_diagnostics(result).orth_stat` and `.orth_pvalue` now return `None`.
The old calculation used a series centered by construction, so its near-one
p-value provided no evidence about nuisance quality or causal identification.
Check `.orthogonality_status` and the explanatory `.orth_warning` rather than
formatting these fields as floats. Fitted effects, standard errors, and
confidence intervals are unchanged. Retained OOF score concentration is an
optional descriptive diagnostic, not a replacement hypothesis test.

---

<a id="forest-scalar-effect-doubly-robust"></a>

## 1.29.0 — ⚠️ `CausalForest.ate()` / `.att()` return the doubly-robust estimate, not the plug-in average

**Who is affected.** Anyone who takes the *float value* of
`sp.causal_forest(...).ate()` or `.att()` — that is, `float(effect)`,
`f"{effect:.3f}"`, arithmetic on it, or its `to_dict()["estimate"]`.
`average_treatment_effect()` is unchanged, and so is any code that already
read `detail["estimate"]`.

**What changed.** The float was the plug-in average of the fitted CATE
predictions, while `.se`, `.ci` and `.pvalue` came from the doubly-robust
(AIPW) score, whose point estimate is a different number. The printed
interval was therefore not centred on the value printed beside it. On a
misspecified binary design (n = 1,500, 500 trees) the gap is 0.049 for the
ATE and 0.120 for the ATT against a standard error of 0.081; it grows where
the fitted propensity approaches its bounds, which is exactly where a
practitioner most needs the two to agree.

The float is now the doubly-robust estimate — the quantity its own interval
covers, and what `grf::average_treatment_effect` returns. Nothing else moves:

```python
e = cf.ate()
float(e)                          # doubly-robust estimate (was: plug-in mean CATE)
e.detail["estimate"]              # same number, unchanged
e.detail["plug_in_estimate"]      # the old float value
e.detail["plug_in_minus_aipw"]    # their difference, as a diagnostic
cf.average_treatment_effect()     # unchanged
```

**To keep the old number**, read `e.detail["plug_in_estimate"]`, or take the
mean of `cf.effect(X)` directly. Note that the old float does not have a
valid standard error attached to it: the dispersion of fitted effects is not
an influence-function variance for their mean.

**Continuous treatments.** *(Corrected after release. This paragraph
originally said continuous treatments were unaffected.)* 1.29.0 also
replaced the 1.25.0 plug-in fallback with grf's continuous-treatment
debiased score, so `ate()` on a continuous treatment changed as well: it
reports `method="aipw_continuous"`, and the float is that estimate. On the
bundled Card data (`educ`, five covariates, `random_state=42`) the float went
from 0.0793 in 1.28.0 to 0.0785; R grf 2.6.1 gives 0.0787. The plug-in value
is still available in `e.detail["plug_in_estimate"]`. See also
[forest-grf-defaults](#forest-grf-defaults) and, for `att()`,
[forest-continuous-att](#forest-continuous-att).

---

<a id="dml-plr-ivtype"></a>

## 1.29.0 — ⚠️ `sp.dml(model="plr", score="IV-type")` now matches DoubleML

**Who is affected.** Callers of `sp.dml(..., model="plr", score="IV-type")`
(and `sp.DoubleMLPLR(..., score="IV-type")`). The default score,
`"partialling out"`, is untouched and reproduces earlier output bit for bit.

**What changed.** The IV-type score of Chernozhukov et al. (2018) uses a third
nuisance `g(X) = E[Y − θ̃D | X]`, fitted on the outcome adjusted by a
preliminary partialling-out estimate `θ̃`. StatsPAI had substituted
`l(X) = E[Y|X]` for it while keeping the IV-type denominator `Σ v̂·D`. The
substitute is unbiased at the truth but not Neyman-orthogonal in the
propensity nuisance, so first-stage regularisation error entered the estimate
at first order, and it agreed with neither DoubleML port (0.40 s.e. off on a
fold-controlled benchmark). The option now fits `g` — cross-fitted on the same
partition with a clone of `ml_g` — and evaluates
`ψ = (Y − Dθ − ĝ)(D − m̂)`.

**How much numbers move.** With linear nuisance learners the corrected
IV-type estimate equals the partialling-out estimate exactly, so anything
previously reported under `score="IV-type"` with OLS first stages moves to the
partialling-out number (for the benchmark above, from 1.0232 to 1.0094). With
regularised or nonlinear learners the corrected estimator is distinct from
partialling out and from the old variant; expect movement of the order of the
first-stage regularisation bias. Standard errors change with the score.

**Reproducing the old numbers.** Not supported: the old moment was not a
documented estimator. Compute
`Σ v̂ û / Σ v̂ D` by hand from `result.model_info` residuals if an exact
replication of a past table is required.

**Cost.** One additional cross-fitted regression per fold (the third
nuisance), so an IV-type fit takes roughly 1.5 times as long as before.

---

<a id="cluster-markout"></a>

## 1.29.0 — ⚠️ Missing cluster variable: rows are dropped, as in Stata

**Who is affected.** Anyone who fitted `sp.ivreg` / `sp.iv` / `sp.liml` with
`cluster=` / `vce="cluster v"` on data where `v` has missing values: those
rows used to stay in the fit, so the coefficients and standard errors were not
Stata's. Other estimators raised (or already dropped the rows) and now run.

**What changes.** Rows with a missing cluster variable leave the estimation
sample on every estimator that takes the `vce()` grammar, with a
`StatsPAIWarning` and `model_info["n_missing_cluster_dropped"]`. To keep a
row, give it a cluster label (for example its own singleton cluster) before
fitting. `sp.margins(fit, data=df)` raises when `df` has missing model
variables instead of returning NaN; omit `data=` to use the estimation sample.

<a id="parity-campaign-phase3"></a>

## 1.29.0 — ⚠️ Cross-language campaign phase 3: 174 defects fixed against R / Stata references (round 1: 125, round 2: 49)

**Who is affected.** Anyone who reported numbers from the functions below.
Phase 3 compared survival / epidemiology, time series, inference and
sensitivity, panel and GLMM, treatment effects, DiD and synthetic control,
RD and IV, and spatial / survey / structural estimators with their R or
Stata reference on the same data. Where a row below changes a default,
rerun the analysis; most old numbers were not a documented quantity and
cannot be reproduced. Where an option restores the old number, the last
column names it. "—" means no route back is offered.

### Survival and epidemiology

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.cuminc` | delta-method SE / CI (`F(t) - F(t_j)`); Gray's test | cause-1 SE at t = 2.0: 0.027944 → 0.030489; Gray chi2 2.3950 → 1.8499 | — (old values were wrong) |
| `sp.finegray` | censoring weights at left limits; default SE is Fine–Gray's sandwich | coef 0.440740 → 0.440603; SE 0.084511 → 0.081090 | `vce="model"` for the old SE; old weights not kept |
| `sp.cox_frailty` | all outputs; `.theta` is now the frailty variance | beta 0.480666 → 0.528795; theta 0.500004 (precision) → 0.240201 (variance) | — (old fit ignored the frailty); for a precision use `1/res.theta` |
| `sp.roc_curve` / `sp.auc` | AUC with tied scores; `thresholds` / `tpr` / `fpr` have one entry per distinct score | 0.68376068 → 0.68389966 | — |
| `sp.kdensity` | default width is R `bw.nrd0`; `"sheather-jones"` is Sheather–Jones | 0.522176 → 0.525683; SJ 0.5222 → 0.5592 | `bw_method="stata"` gives Stata's 1.349 rule with Stata's percentiles |
| `sp.cox(ties="breslow")` | runs Breslow | x1 0.355720 → 0.335603 | `ties="efron"` is what it used to run |

### Time series

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.panel_unitroot` | LLC / IPS / Fisher are the named tests (Stata `xtunitroot` by default) | LLC −7.80 → −4.656; Fisher P 239.1 → 117.09 | — (old numbers were not these tests) |
| `sp.engle_granger` | step-2 ADF without a constant; `trend` acts in step 1; MacKinnon (2010) critical values | Z −9.6167 → −9.6364; 5% CV −3.34 → −3.3608 | — (old statistic had no valid critical values) |
| `sp.johansen` | critical values (max-eig k−r = 3, 4; separate 'n' / 'ct' tables) | max-eig 5% k−r=3: 21.12 → 20.97 | statistics unchanged |
| `sp.bvar` | Minnesota prior per equation; no column-order dependence | gdp own-lag 0.901 → 0.6382 | — |
| `sp.garch` | pre-sample `mean(eps²)`; converged optimiser; `forecast()` uses all lags | log L −2144.43397943 → −2144.43398728 | — (differences ~1e-5) |
| `sp.cusum_test` | exact boundary coefficient | 0.948 → 0.9478982 | n/a (1e-4) |
| `sp.structural_break` | sup-F grid includes strucchange's last point | changes only when the maximum is at that point | — |

### Inference and sensitivity

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.wild_cluster_bootstrap`, `sp.wild_cluster_boot`, `sp.subcluster_wild_bootstrap`, `sp.panel` feols `wild` | exact (enumerated) p with strict `>` when `2^G <= n_boot` | p 0.92569 → 0.9306640625 (G = 12) | — (MC estimate of the same quantity; `>=` was wrong) |
| `sp.wild_cluster_ci_inv` | endpoints are the jumps of p(h0) | lower −1.114049 → −1.121498336078 | — |
| `sp.fisher_exact`, `sp.ri_test` | exact p on designs with at most `n_perm(s)` assignments | 0.1443 → 132/924 | pass `n_perm(s)` below the number of assignments |
| `sp.rosenbaum_bounds` / `sp.rosenbaum_gamma` | no continuity correction; zeros ranked; correct tail | Γ = 2 upper bound 0.1077 → 0.149128 | `zero_method="wilcox"` reproduces the zero handling only |
| `sp.oster_bounds` | exact solution from data; corrected approximation from summaries | δ* 1.20492 → 0.904199433 | — (wrong formula) |
| `sp.oster_delta` | default `r_max=1.3` means min(1, 1.3 R_full); exact solution | δ* 0.16189 → 0.90420 | pass an explicit `r_max` in (R_full, 1] |
| `sp.pate(method="aipw")` | per-arm Hájek augmentation | bias −0.168 → −0.001 (simulation) | — |

### Panel and GLMM

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.xtnbreg(model="fe")` | Hausman–Hall–Griliches conditional FE (Stata `xtnbreg, fe`) | z1 0.418 → 0.3783114 | `model="ufe"` |
| `sp.xtnbreg(model="re")` | beta-dispersion RE (Stata `xtnbreg, re`) | — | `model="normal_re"` |
| `sp.menbreg` / `sp.megamma` / `sp.meologit` | observed-curvature Laplace / AGHQ | gamma `_cons` 0.6196 → 0.6372 | `curvature="expected"` |
| `sp.meglm(family="gaussian")` | σ² estimated (was fixed at 1) | SE(x1) 0.0428 → 0.0335 (Stata) | — (old was not a Gaussian mixed model) |
| GLMMs, `sp.mixed` | optimum finished with Newton steps | `mepoisson` `_cons` 0.3451384 → 0.3450937699 | — |
| `sp.meologit` | SEs from the full observed information; threshold SEs | — | — |
| `sp.icc` | Stata `estat icc` SE / CI; latent ICC for `melogit` / `meologit` (was NaN) | mixed ML SE heuristic → 0.0533785 | — |
| `MixedResult.aic` / `.bic` | residual variance counted once | AIC was 2 too high, BIC log n too high | — |
| `sp.interactive_fe` | Bai's SE (M_Λ X M_F), CR1 cluster factor, `tol` 1e-10 | xa SE 0.03194 → 0.03304 | — |
| `sp.gmm` (nonlinear) | Gauss–Newton finish | estimates move ~1e-8 | — |
| `sp.mixlogit` | positive SDs; no ε perturbation; robust × N/(N−1) | — | `small_sample=False` for the robust factor |

### Treatment effects

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.ltmle` | R `ltmle`'s algorithm; `propensity_bounds` bounds the cumulative g, default `(0.01, 1.0)` | binary ATE 0.757 → 0.373 | — (invalid) |
| `sp.gformula_ice_fn(bootstrap=0)` | SE is the M-estimation sandwich | 0.0591 → 0.0683 | — (not an SE) |
| `sp.aipw(estimand="ATT")` | SE includes the ratio term | 0.1018 → 0.0899 | — |
| `sp.principal_strat` | "Complier (LATE)" is the Wald LATE | 6.395 → 7.355 | E[Y(1)\|c] = `(mu_11 p11 − mu_01 p01)/pi_c` |
| `sp.survivor_average_causal_effect` | bounds keep rows with missing outcomes | (1.684, 1.684) → (0.725, 2.612) | — |
| `sp.lee_bounds` | Lee's sample-quantile trimming | (0.7176, 2.6187) → (0.7245, 2.6120) | —; `trimming="exact"` for fractional trimming |
| `sp.manski_bounds(assumption="mts")` | MTS alone | lower bound 0 → −0.336 | `assumption="mts_mtr"` |
| `sp.horowitz_manski` | strata missing an arm kept | [−0.332, 0.629] → [−0.337, 0.663] | — |
| `sp.mediate_interventional(tv_confounders=)` | D→L→Y path included | IDE 0.574 → 1.106 | — |
| `sp.multi_treatment` | unpenalised mlogit GPS; bootstrap refits the same outcome model | ATE 1.163538 → 1.163558 | — |
| `sp.g_estimation` | logistic propensity by default | psi 1.0564 → 1.0552 | `propensity_model="linear"` |
| `sp.ipcw` | censored rows weighted 0; summaries over uncensored rows | censored weights ≥ 0.77 → 0 | none needed (observed-row weights unchanged) |

### DiD and synthetic control

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.sdid`, `sp.synthdid_estimate`, `sp.sc_estimate`, `sp.did_estimate` | placebo / bootstrap / jackknife SE follow `synthdid::vcov`; undefined SEs are NaN | Prop. 99 placebo SE (seed 42) 2.6041 → 2.5630; five-treated jackknife 0.1248 → 0.4382 | — (old SEs were not a documented quantity) |
| `sp.breakdown_m` | inverts the FLCI; honours `method` | mpdta CS e=0 0.01925 → 0.00799 | the closed form is still returned, with a warning, when no covariance is available |
| `sp.honest_did` (native smoothness) | worst-case bias fixed; no rounding | mpdta CS e=1 M=0.02 lower −0.12136 → −0.10740 | — |
| `sp.continuous_did(method="twfe")` | SE degrees of freedom; unbalanced-panel slope | iid SE 0.042912 → 0.047212; unbalanced slope 0.403496 → 0.402988 | — |
| `sp.continuous_did(method="att_gt")` | bootstrap with multiplicity; bins bootstrapped jointly | pooled SE 0.10639 → 0.18500 | — (old bootstrap was a subsample) |
| `sp.continuous_did(method="dose_response")` | SE is a bootstrap of the average slope | 0.24308 → 0.15439 | `mean(model_info["dose_response_pointwise_se"])` |
| `sp.did_timevarying_covariates` | outcome-regression ATT(g,t); group aggregation | 1.273390 → 1.257705 | —; `aggregation="simple"` gives the treated-count-weighted aggregate of the corrected cells |
| `sp.mc_panel`, `sp.matrix_completion` | `fixed_effects="two-way"`; `tol` 1e-5 → 1e-10, `max_iter` 1000 → 5000 | ATT at lambda 8: 2.934 → 2.385 | `fixed_effects="none", tol=1e-5, max_iter=1000` |
| `sp.mc_synth`, `sp.synth(method="mc")` | `fixed_effects="two-way"`; `tol` 1e-6 → 1e-10, `max_iter` 500 → 5000; CV grid from the centred matrix | `mc_synth(seed=0)` 1.66626 → 1.32783 | `fixed_effects="none", tol=1e-6, max_iter=500` |
| `sp.spillover_did` (staggered only) | ring effects by exposure onset; SE adds the weight term | ring 1 0.7912 → 1.2376; direct SE 0.0779 → 0.0914 | — |
| `sp.harvest_did` | own cohort excluded from placebo controls; joint-covariance inference; IF cell SE | headline SE 0.0621 → 0.1015; cohort-5 e=−4 −0.3550 → −0.5104 | — |
| `sp.causal_impact` | SE uses correlated forecast errors | 0.2348 → 0.2852 | `sqrt(mean(detail.predicted_se[post]**2)/n_post)` |
| `sp.scpi` | intervals from R `scpi`'s procedure; `se` / `pvalue` NaN; `period_results` loses `in_sample_var` / `out_sample_var` | Germany 1991 [0.230, 0.774] → [−0.731, 1.316] | — |
| `sp.scest(w_constr="lasso")` | L1 ball ‖w‖₁ ≤ 1 | Germany max \|Δw\| 0.227 | — |
| `sp.scest(w_constr="ridge")` | L2 ball, Q from `shrinkage.EST` | Germany max \|Δw\| 0.194 | — |
| `lasso_lambda`, `ridge_lambda` (`scest`, `scpi`) | deprecated and ignored | — | use `Q=` / `Q2=` |
| `sp.ssaggregate` | AKM SE of `x`; z instead of t(n−k); HC1 diagnostic from the 2SLS sandwich | SE 0.06646 → 0.29044 | — |
| `sp.shift_share_se` | AKM on the recorded instrument; raises for results not from `sp.bartik` / `sp.ssaggregate` | 0.017595 → 0.290442 | — |
| `sp.bartik(robust=<other>)` | raises `ValueError` | silently HC1 → error | `robust="hc1"` |
| `sp.staggered_synth` | donors untreated through the effect window; unit-level ATT | 2.6649 → 2.5971 | — (contaminated donors) |
| `sp.staggered_synth(penalization=)` | multisynth's `lambda` on the normalised objective | — | — (the old scale had no reference) |
| `sp.discos` (individual-level data) | Gunsilius estimator; default `method="quantile"` | 1.2491 → 0.9245 | aggregate to unit-period means first (the fallback then runs, with a warning) |
| `sp.demeaned_synth` / `sp.robust_synth` | consistent MSPE-ratio placebo p-value | null DGP 0.111 → 0.222 / 0.444 | — (old statistic was inconsistent) |
| `sp.robust_synth(l1_penalty>0)` | unpenalised intercept; `l1` multiplies ‖w‖₁ in `RSS + l2‖w‖² + l1‖w‖₁` | intercept 0.53 → 20.74 at `l2=5, l1=1e-12` | old `l1` ≈ new `l1/2`, except that the intercept is no longer penalised |
| `sp.robust_synth` / `sp.demeaned_synth(covariates=)` | raises `NotImplementedError` | — | drop the argument (it was ignored) |

### RD and IV

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.tF_critical_value`, `sp.tF_adjustment` | every value between F = 4 and 106 | c(10) 3.16 → 3.4353 | — (old table was wrong) |
| `sp.iv_diag` | `tF_critical_value`, `tF_adjusted_ci`, `se_ols`, `ci_ols`, `p_ols`; tF NaN for k > 1 | tF c.v. 18.66 → 12.238 (HC1); `se_ols` 0.030699 → 0.035494 | — |
| `sp.weakrobust` | `clr_stat`, `clr_pvalue`, `clr_ci`, `k_stat`, `k_pvalue`, `k_ci`, `tF_critical_value` | CLR 0.0094457 → 0.0090722 | `clr_method="simulate"` restores the simulation only |
| `sp.jive` | `variant="jive2"` estimates; default SEs | jive2 0.18519 → −3.38682; SE 3.3019 → 7.0217 | — |
| `sp.rdrobust` | explicit `p <= deriv` honoured; `deriv > p` raises | `deriv=1, p=1`: 1.3843 → 0.9933 | pass `p=deriv+1` |
| `sp.rkd` | default bandwidth; fuzzy SE | h 0.21760 → 0.21801 | pass `h=` explicitly |
| `sp.rdplot` | bin count / positions, CI bars; default kernel string | esmv J (17, 16) → (38, 41) | `nbins=` to fix the bins |
| `sp.rdplotdensity` | every curve | left density 0.6045 → 0.2325 | — |
| `sp.mccrary_test` | estimate / SE / p; `model_info["n_bins"]` is the number of histogram cells | θ 1.3036 → 1.2707 | — |
| `sp.rdhte`, `sp.rdbwhte`, `sp.rdhte_lincom` | robust SE / CI / p, default bandwidth, binary-z output, `rdbwhte` return type for subgroups | SE 0.0697 → 0.0993; h 0.1506 → 0.3893 | `vce="hc1"` for the HC1 flavour; the conventional SE is not exposed |
| `sp.rd_bias_aware_fuzzy` | CI, M defaults, h default, nearest-neighbour SE | default set (0.62, 3.92) → (0.941, 3.143) | — |
| `sp.rdrbounds` | both bounds | Γ = 2 upper 0.159 → 0.346 (R) | — |

### Spatial, survey and structural

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `W.transform` on non-binary weights | styles re-weight the constructed weights; `"V"` sums to n | Columbus `"V"` sum 105.2985 → 49 | set `transform = "B"` first, then `"R"` |
| `sp.gwr` (exponential kernel, adaptive Gaussian / exponential) | GWmodel kernels | AICc 896.3499 → 894.1283726 | — (old kernel was not a documented quantity) |
| `sp.gwr_bandwidth` | CV is leave-one-out; GWmodel `bw.gwr` search | CV adaptive bisquare 7 → 147 | —; for the old search bounds pass `bw_min` / `bw_max` |
| `sp.mgwr` | β kept directly (was 0 where x = 0) | — | — |
| `sp.sarar_gmm` | GS2SLS (`spatialreg::gstsls`) | const 43.97302 → 43.54044357 | `w_lags=1` for PySAL `GM_Combo`; the old estimator is not reproducible |
| `sp.spatial_panel` | joint-information SEs; exact ρ; two-way SDM lags | SE(lpcap) 0.02535161 → 0.02544250 | `vce="oim"` for Stata `xsmle`; the old conditional β SEs are not available |
| `sp.svydesign` (+ `svymean` / `svytotal` SE) with `fpc=` given as population counts and clusters | Sampling fraction per stratum is #PSU sampled / N_h (R `as.fpc`), not #elements / N_h | `full` design SE: NaN → 0.49514353151367907 (R 0.4951435315136789) | Not reachable; the element-based fraction is not a documented quantity (it gives 1 − f < 0 → NaN) |
| `sp.svydesign` design df (CIs, p-values) when PSU ids repeat across strata and `nest=False` | PSUs are keyed by (stratum, PSU), so df = #PSU − #strata, as in R and Stata | df 1, CI [6.7114, 20.2148] → df 17, CI [12.342014912296227, 14.584186736399586] (R/Stata [12.34201491229616, 14.58418673639966]) | Not reachable; the old df counted raw ids globally, inconsistent with the nested variance it was paired with |
| `sp.svyglm(family="binomial" \| "poisson")` SEs | Sandwich bread is (X′ diag(w·V(μ)) X)⁻¹, not the Gaussian (X′WX)⁻¹ | Logit SE [0.1179, 0.0385] → [0.5098, 0.1731] (R; match 3.9e-14); Poisson SE [0.4900, 0.1605] → [0.2138, 0.0502]; coefficients move ~1e-9 (tighter IRLS) | Not reachable; the Gaussian bread is wrong for non-Gaussian families |
| `sp.svymean` / `sp.svytotal` DEFF | Denominator is svyvar·(N − n)/(N·n), not svyvar/Σw, so it no longer depends on weight scale | Mean DEFF 204.877 → 3.9032222795694005; total DEFF 1511.21 → 28.790859422197673 (R `deff=TRUE`) | Not reachable; the scale-dependent value is not a documented quantity (`deff="replace"` gives Stata's no-fpc DEFF, not the old number) |
| `sp.rake` | Stops when every category share is within `tol` of its target (relative), default `tol=1e-10`; was an absolute change on O(1/n) weights at 1e-6 | n = 100 000: margin error 4.3e-4 → 8.1e-11; n = 200: gap to R's fixed point 2.4e-7 (design start) / 1.9e-6 (equal start) → 1.04e-10 | Not reachable; `tol` now means a relative margin gap, so old `tol=1e-6` runs but stops on a different rule |
| `sp.lcsf` / `sp.zisf` | Newton polish after L-BFGS-B, and a Richardson-extrapolated Hessian for OIM SEs | lcsf vs sfaR (two fixture cases): estimates max rel err 1.15e-3 / 1.43e-3 → 1.1e-8 / 6.2e-9; SEs 1.44e-3 / 2.13e-3 → 8.6e-8 / 6.7e-8; log-likelihood −324.1954416 → reaches sfaR's −324.1954399 | Not reachable; the old optimum was a premature stop, not a different estimand |
| `sp.metafrontier(cost=True)` | Sign of the technology gap flipped for cost frontiers; TE_meta changes with it | TGR min / mean / max 1.0 / 1.0 / 1.0 → 0.170 / 0.445 / 1.0 | Not reachable; TGR ≡ 1 came from clipping a wrong-signed gap |
| `sp.blp` standard errors | Joint (β, σ) robust GMM sandwich with 1/N-scaled Jacobian and analytic dδ/dσ | Linear SEs (0.00434, 0.00070, 0.00187) → (0.1075, 0.0201, 0.0474); σ SE 4.99e-5 → 0.0304 (fixture, N = 609; pyblp match 5.6e-9) | Not reachable; old SEs were too small by √N (linear) and N (σ) |
| `sp.blp` elasticities when price is in `x_random` | Each simulated consumer's price coefficient is α + σ_p·ν (`sigma_price` was unused) | Rel. error vs pyblp > 1e-2 → ~1e-8 | Not reachable; the old value ignored the estimated σ_p |

### Time series (round 2)

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.structural_break(method="bai-perron")` (default) | Bai-Perron sequential procedure (`mbreaks::dosequa`): global minimum segment, sup F(l+1\|l) critical values; `p_values` is `None`, new `critical_values` / `sequential_tests`; `min_segment` and `alpha` must be tabulated values | fixture dates unchanged; statistics on the Wald scale, p-values → critical values; other data can select different dates | — (the old procedure was not Bai-Perron) |
| `sp.structural_break` sup-F p-value | Hansen (1997) approximation (strucchange, Stata `estat sbsingle`) instead of a fixed-seed Monte Carlo | `r2_wn`: default p 0.2297329322069956 (= R); the Monte-Carlo value was within 0.03 | `pvalue_method="simulate"` |

### Treatment effects (round 2)

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.policy_value` | (n, 1) `scores` or unequal lengths raise `ValueError` | mean(scores)·mean(policy) → `ValueError` | — (wrong number); pass `scores.ravel()` |

### RD and IV (round 2)

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.rd2d` / `sp.boundary_rd` | port of R `rd2d`: pointwise effects, `rdbw2d` bandwidths, robust bias-corrected inference, WBATE headline; default `approach` `"distance"` → `"location"` | one pooled effect → five pointwise effects (0.446 … 1.357), WBATE 0.891 (SE 0.0903) on `rd_open_bd.csv` | none exactly; `approach="pooled"` gives one one-score effect (now via `sp.rdrobust`) |
| `sp.rd2d_bw` | returns a DataFrame of per-point bandwidths (= `rdbw2d`) instead of a float | float → DataFrame | — |
| `sp.rd_discrete` | `'bsd'` = `RDHonest`, `'bme'` / `'bm'` = `RDHonestBME`; `K=` ignored (`DeprecationWarning`); `model_info['discrete']` drops `n_left`, `bin_means`, `honest_ci`, adds `maximum_bias`, `bandwidth`, `eff_obs` | bsd M = .05, h = 5: SE 0.0113 → 0.1121; default CI [−34.5, 35.8] → [0.464, 1.216]; bm h = 4: 0.7271 → 1.4953 | — (old SEs understated by ~√(bin size)) |

### Spatial, survey and structural (round 2)

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.malmquist` | EC = TE_{t+1}/TE_t (`efficiency="bc"`); M changes with it; TC unchanged | mean EC t→2 1.0653 → 1.0389; mean M t→2 1.0997 → 1.0730 | `efficiency="residual"` |
| `sp.zisf` JLMS efficiency | `efficiency(method="jlms")` and `model_info["mean_efficiency_jlms"]` are exp(−E[u\|ε]) | 0.81702 → 0.81279 | — (the old value was the BC number under the JLMS label) |
| `sp.mgwr` | PySAL mgwr search, kernel, SOC stop and bandwidth freeze; `tol` is the SOC threshold | Georgia bandwidths [93, 101, 156, 158] → [101, 101, 117, 157] | — (the old search matched no reference); `kernel_eps=1.0` with a small `tol` for GWmodel's kernel |

### Decomposition and QTE (round 2)

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.melly_decompose`, `sp.machado_mata` | exact LP quantile regressions; midpoint grid (j − 0.5)/J, default `n_tau_qr` 99 → 100; averaged inverse CDF | counterfactual τ ≈ .1 1.49961 → 1.48117; τ ≈ .9 3.11162 → 3.12503 | — (the old QR was not a minimiser) |
| `sp.fairlie` | Jann's rank-matched sequential decomposition; new `detailed['se']` | educ 0.010457 → 0.009463 | — (the old algorithm was not Fairlie's) |
| `sp.mediation_decompose` (with covariates) | `nde`, `total`, `propn_mediated`: covariates at the overall mean | NDE 0.717658 → 0.770783 | `nde_old = cde + theta3*(mean(M\|A=0) - E[M\|A=0, c̄])` (not exposed) |
| `sp.mediation_decompose(inference="analytical")` (default) | delta-method `se` / `ci` returned | `None` → SEs (= `paramed` 9.5e-15) | — |
| `sp.dist_iv`, `sp.distributional_te`, `sp.qte(method="firpo_*")` with covariates | unpenalised Newton logit propensities | 5th–6th digit | — |

### ML causal inference (round 2)

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.calibration_test` / `sp.test_calibration` | `grf::test_calibration` regression, HC3, grf's t against 0 with one-sided p; no `null` column; non-training rows raise; covariance chosen by `vce=` (result key `vcov_type`) | differential 0.180 → 1.187; mean 0.985 → 1.00297 | — (the old differential coefficient estimated b2 − b1) |
| `sp.rate` | grf AIPW scores, tie averaging, QINI `k/n` weights, rank-corrected SE; non-training `X` raises | AUTOC SE 0.0949 → 0.0725 (grf bootstrap 0.0744) | — |
| `sp.cate_eval` | same RATE operator as `sp.rate` | as `sp.rate` | — |
| `sp.average_treatment_effect(target_sample="overlap")` | grf R-learner OLS with intercept and HC3 | 1.12386 / 0.07687 → 1.12514 / 0.07968 | — |
| `sp.honest_variance` | `se` is the half-sample spread, not divided by √`n_splits` | 0.0038 (25 splits) → ≈ sd(tau)/√n | — (Monte-Carlo error of the average, not an SE) |
| `sp.blp_test` | CDDF weighted regression on an out-of-fold proxy; default `vce="HC1"` (result key `vcov_type`) | constant-effect DGP β2 ≈ 1.43, p ≈ 3e-52 → null DGP β2 0.137, p 0.29 | `proxy="in_sample"` for the in-sample proxy (the regression stays weighted); `vce="const"` for GenericML's default |
| `sp.gate_test` | GATES weighted regression when `y` / `treat` / `covariates` are given | ANOVA on predictions → GATES γ_k (= GenericML) | call without outcomes for the descriptive path (warns) |
| `sp.forest_diagnostics` on non-training rows | overlap reported as NaN with a warning | perfect overlap → NaN | — |
| `sp.ips` / `sp.snips` / `sp.doubly_robust` with `clip` | `clip` caps the weight only; `pi_b <= 0` raises | `clip=2` IPS 0.8496 → 1.1904 (obp) | — |
| `sp.snips` | delta-method SE (= `sp.ope.snips`) | 0.00039, against a bootstrap SE of 0.0174 (n = 2000) → delta-method SE | — |
| `sp.doubly_robust` with a 1-D policy | action vector read as actions, not probabilities | always action 2: 1.3054 → 0.8555 | — |
| `sp.dml_panel(include_time_fe=True)`, unbalanced or weighted | exact two-way within transform | 0.82050 → 0.81951 (DoubleML) | — |
| `sp.dml_model_averaging` / `sp.model_averaging_dml` | exact QP for K ≤ 12 learners | weights move ~1e-8 | — |

### Sensitivity, Mendelian randomization, transport and trial design (round 2)

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.mi_estimate` / `MICEResult.combine` | Barnard–Rubin df with `dfcom`, full within covariance, t-based p / CI | x2 df 31.25 → 24.99 | pass estimates without `df_resid` to `combine` (large-sample df) |
| `sp.mediate_sensitivity` | Imai–Keele–Yamamoto / `medsens` | crossing ρ 0.558 → 0.487; ACME(−0.9) 0.852 → 1.533 | — (wrong formula) |
| `sp.calibrate_confounding_strength` | `sensemakr::ovb_bounds`; `dof=` required | k = 5 bias bound 0.0199 → 0.4640 | — |
| `sp.survival_sensitivity` | worst case moves toward the null for HR < 1 | log HR −0.4, SE 0.15: breakpoint None → 1.2 | — |
| `sp.unified_sensitivity` | `rho_max` honoured on both paths; default None = Oster's min(1, 1.3 R²); RR CI is EValue's interval | summary-path default 1.0 → 1.3 rule | `rho_max=1.0` |
| `sp.attrition_bounds(method="lee")` | Lee's quantile rule (= `sp.lee_bounds`) | (0.70358, 1.61981) → (0.71883, 1.60455) | — ; `trimming="exact"` for `leebounds` with exact thresholds |
| `sp.mr_bma` | Zuber et al. Bayes factor (IVW scaling, prior σ 0.5); `model_averaged_estimate` added | BIC weights → Bayes-factor posterior probabilities | `method="bic"` |
| `sp.mr_multivariable` | SE × max(1, RSE) | changes only when RSE < 1 | — |
| `sp.mr_mediation` | total effect SE from random-effects IVW (`sp.mr_ivw`) | fixed-effect SE → random-effects SE | — |
| `sp.grapple` | GRAPPLE robust profile score (Tukey) with its sandwich; `RuntimeWarning` on non-convergence | β 2.694 → 2.7197; SE 0.510 → 0.5520 | — (different estimator) |
| `sp.transport_weights_fn` / `sp.transport_generalize` | SE with w² (HC0) | 0.0864 → 0.1556 | — |
| `sp.identify_transport` | descendants of X no longer admissible | X→Z→Y, S→Z: formula → "not transportable" | — |
| `sp.randomize` | `randomizr` `complete_ra` rule within strata / over clusters; re-randomization keeps the design | old: odd-stratum misfits to a fixed arm (5 → 2 treated, 7 → 4) | — (draws differ for the same seed) |

### DiD and synthetic control (round 2)

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| no-covariate SCM fits (`solve_simplex_weights`: `sp.synth` equal-V, sensitivity tools, `conformal_synth`, `multi_outcome_synth`, `synth_experimental_design`) | exact active-set weights for identified problems | pre-RMSPE 0.17714 → 0.1771399093 (moves ~1e-7) | — |
| `sp.synth_loo` / `sp.synth_time_placebo` | `se` / `pvalue` NaN when T1 < 2 or SE ≤ 0 | last time placebo p 0.0 → NaN | — |
| `sp.conformal_synth` | Chernozhukov–Wüthrich–Zhu / `scinference` procedure | p 0.0556 → 0.444; CI (−inf, inf) at α 0.05 when T = 18 | — (the old procedure had no reference) |
| `sp.synth_power` / `sp.synth_mde` | rank-p rejection rule; centred null baseline | Prop. 99 power at δ = 0 1.0 → 0.0; MDE 0 → 10.9 | — (anti-conservative) |
| `sp.synth_survival` | exact solver; band [gap − q_hi, gap − q_lo], pointwise | weights off by 0.03 → exact | — |
| `sp.shift_share_political*` Rotemberg weights | GPSS weights (signed, sum to 1), = `bartik.weight::bw` | max \|α_k\| 0.447 → 0.702 | `abs_weight` column retained |
| `sp.shift_share_political` share-balance F | df1 = rank([1, S]) − 1 | F 0.924 → 1.143; p 0.490 → 0.357 | — |
| `sp.shift_share_political_panel` | exact two-way within transform (unbalanced panels); `cluster="twoway"` is Cameron–Gelbach–Miller; `cluster="shock"` is AKM; `first_stage_F` is a partial F | balanced-panel β and unit / time SEs unchanged | — (`"twoway"` was HC0 on unit × time cells) |
| `sp.synth_experimental_design().method` | `"abadie_zhao_2025"` → `"loo_sc_fit_ranking"` (label only) | — | — |

### Defaults aligned with the reference implementation

| Function | What changes | Old → new (example) | Old number, if you need it |
| --- | --- | --- | --- |
| `sp.roc_curve` | `se_method` default `"hanley"` → `"delong"` (pROC / roctab default) | AUC SE follows DeLong | `se_method="hanley"` |
| `sp.direct_standardize` | `ci_method` default `"lognormal"` → `"gamma"` (epitools default) | interval follows `ageadjust.direct` | `ci_method="lognormal"` |
| `sp.power_case_control` | `test` default `"wald"` → `"chi2"` (Stata `power twoproportions`) | power 0.9213 → 0.9171; cases for 80% power 138 → 141 | `test="wald"` |

---

<a id="stata-vce-grammar"></a>

## 1.29.0 — ⚠️ Standard errors, p-values, `test` / `lincom` and `margins` follow Stata

**Who is affected.** Anyone who reported, from the estimators below, a robust
or clustered standard error, a p-value or confidence interval of a clustered
least-squares or likelihood-based fit, an `sp.test` / `sp.lincom` involving
more than one coefficient, or an `sp.margins` result from a nonlinear model.
Point estimates change only where noted.

**What changes, and how to get the old number back.**

| Area | Old | New (matches Stata 18) | Old number, if you need it |
| --- | --- | --- | --- |
| `vce="robust"` on logit/probit/cloglog/poisson/glm/ordered/multinomial/zero-inflated/truncreg/biprobit/betareg/fracreg | unscaled HC0 | HC0 × N/(N-1) | `vce="hc0"` |
| cluster SEs on glm, ologit, oprobit, mlogit, clogit, zip, zinb, hurdle | G/(G-1)·(N-1)/(N-K) | G/(G-1) | — |
| `nbreg` SEs (all vce) | IRLS bread, Poisson score | joint (beta, ln alpha) information | `model_info["se_conditional_on_dispersion"]` (= MASS::glm.nb) |
| `nbreg(dispersion="constant")` coefficients | profile fit, up to 8.6% off | joint MLE | — |
| p-values / CIs, clustered `regress` / `ivreg` | t(N-K) | t(G-1) | — |
| p-values / CIs, likelihood-based fits | t(N-K) | z | — |
| `sp.test` / `sp.lincom` | diagonal covariance, distribution from `df_resid` | full covariance, F/t or chi2/z by the fit | — |
| `sp.margins` after logit/probit/cloglog/poisson/nbreg/glm | index coefficients | average marginal effects on the prediction scale, delta-method SE | `result.params` |
| `sp.margins` with interaction terms | SE = std(dydx)/sqrt(n) | delta-method SE | — |
| `glm`, non-canonical link | expected information | observed information | `information="expected"` (= R `glm`) |
| `liml` / `iv(method="liml")` robust & cluster | meat (I - kappa M_Z) X; `sp.liml` robust unscaled | meat P_Z X; N/(N-K) | — |
| `clogit` SEs | BFGS `hess_inv` | analytic information | — |

**Now errors instead of a quiet fallback.** An unknown or unimplemented
`vce=` spelling; `vce="cluster"` without a cluster variable; a misspelled
coefficient in `test` / `lincom`; a multi-coefficient restriction or
`margins_at` / `contrast` / `pwcompare` on a result without a covariance
matrix; `margins` on a model whose prediction scale is not supported;
`robust=` / `cluster=` on `panel_logit(method="fe")` or `xtnbreg(model="re")`;
`betareg` outcomes on the (0, 1) boundary; `sp.etable` given a non-result.

**Newly honoured options.** `robust=` / `cluster=` on `truncreg`, `biprobit`,
`betareg`, `panel_logit` / `panel_probit` (RE) and `robust=` on
`subgroup_analysis` used to be ignored; results computed with those options
had model-based (or HC1) standard errors. Rerun them.

<a id="iv-first-stage-f-vce"></a>

## 1.29.0 — ⚠️ IV first-stage F follows the requested vcov; `sp.cross_validate` engines aligned

**Who is affected.** Anyone reading the first-stage F from `sp.ivreg` /
`sp.iv` / `sp.lasso_iv` fitted with `robust=` or `cluster=` (diagnostics,
`model_info['first_stage_f']`, weak-IV warnings, `sp.regtable`'s
"First-stage F" row), and anyone who relied on `sp.cross_validate` verdicts
for IV formulas or clustered / robust errors.

**What changes.**

| Quantity | Before | Now |
| --- | --- | --- |
| First-stage F with `robust=` / `cluster=` | classical F | Wald F under the same vcov, `F(m, N-k)` / `F(m, G-1)` (Stata `estat firststage`) |
| First-stage F, non-robust fits | classical F | unchanged |
| Classical F under robust / cluster | the headline F | `First-stage F, non-robust (<endog>)`, `first_stage[j]['f_statistic_nonrobust']` |
| `cross_validate` IV formula `y ~ w1 + w2 \| x ~ z` | statspai / linearmodels dropped `w1` | all exogenous regressors kept |
| `cross_validate(..., cluster=)` | statspai ignored it; R clustered on the first FE | every engine clusters on the requested variable |
| linearmodels engine SEs | large-sample | small-sample (`debiased=True`) |

**What to do.** Weak-instrument thresholds of Stock and Yogo apply to the
classical F; read `First-stage F, non-robust` for those, or use
`sp.effective_f_test` / `sp.anderson_rubin_ci` under heteroskedasticity.
Re-run `sp.cross_validate` checks that used IV formulas or clustering: an
old `AGREE` / `DISAGREE` / `PARTIAL` may change.

**Also.** `sp.regtable` exports no longer drop columns when `model_labels`
repeat; `to_dict()["columns"]` suffixes a repeated label with its position
(`"OLS (1)"`). Weak-IV confidence-set `summary()` / `as_intervals()` print
the root-found endpoints, so printed intervals widen slightly.

---

<a id="causal-forest-grf-engine"></a>

## 1.29.0 — ⚠️ `sp.causal_forest` grows generalized random forests

**Who is affected.** Every user of `sp.causal_forest` / `sp.CausalForest`:
CATE predictions, `effect_interval`, and every in-sample statistic
(`average_treatment_effect`, `ate()`, `att()`, `calibration_test`, `rate`,
`best_linear_projection`, `forest_diagnostics`) change numerically.

**What was wrong.** Trees were scikit-learn regression trees fitted to the
*treatment residual*, so splits followed the propensity residual rather than
effect heterogeneity (CATE RMSE 1.02 vs 0.31 for R grf on a simple design);
leaves with fewer than two honest rows kept the mean treatment residual as
their "effect"; in-sample statistics used predictions from trees that had seen
the rows; `effect_interval` returned percentiles of single-tree predictions.

**What changes.**

| | Before | Now |
| --- | --- | --- |
| Splitting | CART on `T - T_hat` | GRF causal gradient, stabilised splits |
| Default trees | 100 | 2000 |
| Nuisances | 3-fold sklearn RF | out-of-bag regression forests (or cross-fitted user models) |
| `predict()` on training rows | in-sample | out-of-bag |
| `effect_interval` | tree percentiles | little-bag variance, normal interval |
| Clusters / panel FE | not supported | `clusters=`, `fe=` |
| `calibration_test`, BLP, ATO | StatsPAI conventions | grf definitions (see CHANGELOG) |

**To reproduce old numbers** for one release:

```python
cf = sp.causal_forest(..., split_rule="legacy", n_estimators=100)  # DeprecationWarning
```

**Things that now raise.** `bootstrap=True` (GRF subsamples without
replacement); `max_samples > 0.5` together with `ci_group_size >= 2` (the
default `ci_group_size=None` drops to 1 with a warning); passing a subset of
rows as `X`/`Y`/`T` to `calibration_test`, `rate`, `average_treatment_effect`
or `best_linear_projection` (these are defined on the training rows' OOB
predictions — fit a separate forest on the evaluation sample instead).

**`sp.honest_variance` is deprecated.** Its SE shrank to zero as `n_splits`
grew. Use `forest.average_treatment_effect()`.

---

<a id="cs-notyet-cutoff"></a>

## 1.29.0 — ⚠️ `callaway_santanna(control_group="notyettreated")` control sets

**Who is affected.** Pre-treatment (placebo) cells under
`base_period="universal"` when some cohort is first treated between the cell's
period `t` and the base period `g - 1`; and every cell when `anticipation > 0`.
Post-treatment cells with `anticipation = 0` and every `base_period="varying"`
cell with `anticipation = 0` are unchanged, as are aggregates built only from
them.

**What was wrong.** Units counted as not yet treated when `G > t`. R `did`
requires `G > max(t, base) + anticipation`: a control must be untreated in both
periods of the 2x2 comparison. A cohort treated before the universal base
period had its treated base-period outcome used as a control, and anticipation
was ignored.

**What changes.** Every ATT(g, t) and SE now matches R `did` 2.3.0 to
machine precision in all tested configurations. Pre-trend tests and event
studies that include affected placebo cells can change materially — rerun them.
`notyet_cutoff="cohort"` (Stata `csdid`'s default) is unchanged.

**To reproduce the old numbers**, or Stata `csdid, ..., asinr`, pass
`notyet_cutoff="asinr"`. The docs used to say `asinr` was R's convention; that
is true only under `base_period="varying"`.

---

<a id="did-bcf-windows"></a>

## 1.29.0 — ⚠️ `sp.did_bcf` compares cohorts with controls over the same periods

**Who is affected.** All `sp.did_bcf` users; the ATT, per-cohort CATTs and
standard errors change.

**What was wrong.** Never-treated units were split into pre and post at the
median period regardless of each cohort's adoption date, so a non-linear
common trend biased the ATT; the covariate-path SE was not a valid standard
error; errors silently fell back to a difference in means.

**What changes.** Each cohort `g` is compared with never-treated units over
`t < g` and `t >= g`; SEs are influence-function based (no covariates) or
bootstrap based and conservatively combined (covariates); failures raise.
`treat=` may also be an absorbing 0/1 indicator. For heterogeneous effects with
group-time inference, prefer `sp.did_forest`.
---

<a id="gardner-did-two-stage-se"></a>

## 1.29.0 — ⚠️ `sp.gardner_did` / `sp.did_2stage`: default standard error is the did2s corrected two-stage variance

**Who is affected.** Anyone reading `.se`, `.ci`, `.pvalue` or the
event-study standard errors from `sp.gardner_did` (or the `sp.did_2stage`
alias) with the default `vce`. Point estimates are unchanged in every mode.

**What changed.** The default `vce='analytic'` clustered only the Stage-2
residuals, treating the imputed counterfactual as known data; it understated
uncertainty (~26% on `mpdta`, a median 0.71x per horizon in event studies).
It now propagates the Stage-1 fixed-effect estimation error into the
Stage-2 clustered variance -- Gardner (2022)'s corrected variance, exactly
what R `did2s::did2s` 1.2.1 and Stata `did2s` report -- with no
small-sample cluster factor, because neither reference applies one.

| fixture | statistic | before (stage-2 only) | after (corrected) | R `did2s` | Stata `did2s` |
| --- | --- | ---: | ---: | ---: | ---: |
| mpdta replica (`tests/r_parity/data/73_did2s.csv`) | static ATT SE | 0.0051177 | 0.0069036 | 0.0069036 | 0.0069036 |
| original mpdta (`02_mpdta_original.csv`) | static ATT SE | 0.0110838 | 0.0134784 | 0.0134784 | -- |

**To recover the previous numbers** pass `vce='stage2'`:

```python
import statspai as sp
res_old = sp.gardner_did(df, y="lemp", group="countyreal", time="year",
                         first_treat="first_treat", vce="stage2")
```

It reproduces the old SE exactly (`0.005117728129401425` on the fixture) and
warns that it understates. `vce='bootstrap'` is unchanged and agrees with the
new default within bootstrap noise. `model_info['se_convention']` records
which variance a result carries.

**If you compared against R or Stata `did2s` before**, the SE gap you had to
explain away is gone; drop any manual rescaling. In event studies `res.se`
for the overall ATT now accounts for the cross-horizon covariance
(`model_info['event_study']['vcov']`), and per-horizon SEs match `did2s`'s
`i(rel_time, ...)` second stage horizon by horizon (rel 2.1e-14 on the
Roth panel of `tests/test_did_event_study_conventions.py`).

---

<a id="synth-placebo-pvalue"></a>

## 1.29.0 — ⚠️ Synthetic-control placebo p-values now rank the treated unit too

**Who is affected.** Anyone who reported a placebo (permutation) p-value from a
native `sp.synth` estimator — classic SCM and the penalized, demeaned, robust,
sparse, gsynth, matrix-completion, staggered, multi-outcome, augmented, kernel
and DiSCo variants. Point estimates, weights, SEs and CIs do not change.

**What was wrong.** The old code returned the share of *placebos* at least as
extreme as the treated unit, `#{placebo >= treated} / J`, floored at
`1/(J+1)`. The permutation p-value counts the treated unit in both the numerator
and the denominator:

```text
p = (1 + #{placebo >= treated}) / (J + 1)   # = rank / (J + 1)
```

**What changes.**

| Treated rank among J+1 units | Old | New |
| --- | --- | --- |
| 1 (most extreme) | `1/(J+1)` | `1/(J+1)` — unchanged |
| r > 1 | `(r-1)/J` | `r/(J+1)` — larger |

California Prop 99 (README example): rank 3 of 39, `0.0526 → 0.0769`. If you
hand-computed `rank/(J+1)` because StatsPAI disagreed with Abadie et al., the
two now agree; no code change is needed. A result sitting just under a
significance threshold at rank `r > 1` may cross it: rerun and re-check.

**Also new.** `sp.SyntheticControl` warns (`RuntimeWarning`) when placebo fits
fail, instead of silently dropping them, and lists them in
`model_info['placebo_failures']`.

---

<a id="prodest-sweep"></a>

## 1.29.0 — ⚠️ Production functions follow Stata / R `prodest`

**Who is affected.** Every call to the following, and markups computed from
them:

- `sp.olley_pakes` (`sp.opreg`)
- `sp.levinsohn_petrin` (`sp.levpet`)
- `sp.ackerberg_caves_frazer` (`sp.acf`)
- `sp.wooldridge_prod`
- `sp.prod_fn`

| Function | What changes | Parity panel, labour / capital (truth 0.60 / 0.30) |
| --- | --- | --- |
| `olley_pakes` | labour from stage 1; capital minimises the innovations (cubic `g`) | 0.999 / 0.210 → 0.633 / 0.286 |
| `levinsohn_petrin` | same | 1.003 / 0.204 → 0.615 / 0.331 |
| `ackerberg_caves_frazer`, `prod_fn` default | exact root nearest the stage-1 coefficients; cubic `g` | 1.002 / 0.235 → 0.983 / 0.216 |
| `wooldridge_prod` | joint GMM with lagged-labour instruments; analytic SEs | 0.722 / 0.309 → 0.624 / 0.341 |
| all | calendar lags: a firm that skips a year loses the year after the gap | — |

**What to do.**

- Re-run.
- To keep the previous Markov process, pass `productivity_degree=1`. The
  previous `polynomial_degree` of `wooldridge_prod` was 2.
- For translog, use `sp.ackerberg_caves_frazer`.
- To reproduce Stata `prodest, method(wrdg)`, pass
  `convention="prodest", vce="unadjusted"`.
- Panels with duplicate (firm, year) rows or a non-numeric year now raise.

---

<a id="xtdpdsys-stata-convention"></a>

## 1.29.0 — ⚠️ `sp.xtdpdsys` now reproduces Stata's `xtdpdsys`

**Who is affected.** Anyone who called `sp.xtdpdsys` with exogenous
regressors (`x=`). Pure autoregressions are affected only through the
one-step weight (`h`), and on `abdata` moved by less than 1e-12.

| Surface | Before | Now |
| --- | --- | --- |
| exogenous regressors instrument | both equations (one column) | differenced equation only |
| one-step weight `H` | `h(3)` | `h(2)` |
| `abdata`, `L.n` / `w` | 0.686 / −0.203 | 0.542 / −0.615 (= Stata `xtdpdsys`) |

**What to do.** Re-run. To reproduce an old number, pass
`iv_equation="both", h=3` — that is `xtabond2`'s default, which
`sp.xtabond(method="system")` still uses.

---

<a id="decomp-sweep"></a>

## 1.29.0 — ⚠️ Decomposition: `das_gupta`, `gap_closing`, Yu-Elwert, Gini RIF, FFL

| Function | Affected calls | What changes | Size |
| --- | --- | --- | --- |
| `das_gupta` | more than one row per population | aggregate is `Σᵢ ∏_f f`, not `∏_f mean(f)` | factor shares 0% → 37%, +333% → −52% on Das Gupta Table 6.5 |
| `das_gupta` | data frames of unequal length | now raise; pass `by=` to pair strata | — |
| `gap_closing` | `method="ipw"`, `method="aipw"` | reweighting direction corrected | counterfactual gap −4.0 → 0.0 on a zero-truth DGP |
| `yu_elwert_decompose` | `method="efficient"` | `cdgd`'s estimator; components add up | selection 0.0010 → 0.0054 on a 2,000-row example |
| `rifreg`, `rif_values` | `statistic="gini"` | Gini RIF of the plug-in Gini (`dineq`) | coefficients up to 1% |
| `ffl_decompose` | all | `spec_error` ↔ `reweight_error` swapped back; `reference=1` sign; `gap` = RIF-mean difference | variance gap 0.12163 → 0.12157; reference-1 components now add up |
| `bauer_sinning`, `yun_nonlinear` | new outputs | `se`, `vcov`, `detailed_unexplained`, `detailed["se"]` (mvdcmp) | point estimates unchanged |
| `gelbach` | standard errors | `b1x2`'s joint covariance; new `vcov`, `total_se`, `vce=` | 0.03% on `cps_wage` |
| `ffl_decompose`, `dfl_decompose` | `stat="gini"` | plug-in Gini (was `n/(n−1)`-corrected) | 0.4% |

**What to do.** Re-run stratified `das_gupta` calls, passing `by=` with the
stratum column. Re-run `gap_closing` with `method="ipw"` or `"aipw"`
(`"regression"` is unchanged). Re-run `yu_elwert_decompose(method=
"efficient")`; `inference="analytic"` now gives `cdgd`'s standard errors. Code reading `ffl_decompose(...).spec_error` or `.reweight_error` should
swap the two names it previously relied on; re-run Gini RIF regressions.

**Evidence.** `tests/reference_parity/test_decomp_R_parity.py` against
`_fixtures/decomp_R.json`, generated by `_fixtures/_generate_decomp_R.R`.

---

<a id="mr-sweep"></a>

## 1.29.0 — ⚠️ Mendelian randomisation brought in line with the R packages

**Who is affected.** Anyone using `sp.mendelian`: the MR estimators and
diagnostics were compared with `MendelianRandomization`, `TwoSampleMR`,
`RadialMR`, `MRPRESSO` and `mr.raps`, and most differed. Changes on
`MendelianRandomization`'s 28-variant LDL-C → CHD example:

| Function | What changes | Size there | Old behaviour |
| --- | --- | --- | --- |
| `mr_ivw`, `mr("ivw")`, `mendelian_randomization` | default SE is random-effects (> 3 variants) | SE 0.276 → 0.530 | `model="fixed"` |
| `mr_leave_one_out` | each row is `mr_ivw` on the subset | SE 49% | `model="fixed"` |
| `mr_egger`, `mr_pleiotropy_egger`, `mr_heterogeneity(method="egger")` | variants oriented; RSE floored at 1 | slope 14%, intercept 38% | none — the old fit depended on allele coding |
| `mr_median` | interpolated median; penalty fixed | 0.8% (weighted), 38% (penalized) | none |
| `mr_mode` | bandwidth / grid / bootstrap as Hartwig et al.; SE is the bootstrap MAD | 7% | none |
| `mr_cml` | BIC uses the sample size `n=`; profile SE | invalid set 6 → 2 variants; SE ~12% | omit `n` (warns) |
| `mr_raps` | port of `mr.raps`; default loss Huber | τ² 3.5×, SE 47% | none; `loss="tukey"` for the old loss |
| `mr_steiger` | two-sided p from the survival function | p 0.0 → 1.8e-73 | `alternative="greater"` (one-sided) |
| `mr_presso` | port of `MRPRESSO`; Bonferroni; `k / B` p-values | corrected estimate 2.4%, outliers 7 → 2 | none |

**What to do.**

- Re-run MR analyses and report the new numbers. Where a table needs the
  old IVW standard error, pass `model="fixed"` and say so.
- `mr_cml`: pass the GWAS sample size, `sp.mr_cml(..., n=N)`. Without it the
  BIC penalty is the number of variants, which the method does not
  justify, and a warning says so.
- `mr_raps`: calls that passed `tuning_c=4.685` meant Tukey — add
  `loss="tukey"` (a `FutureWarning` flags bare `tuning_c`). Drop
  `beta_init` / `tau2_init`: they are ignored with a `DeprecationWarning`
  and will be removed in 1.29.
- `mr_presso`: a p-value of 0 now means "below 1 / B" (or `n / B` for the
  Bonferroni-adjusted outlier p-values), as in `MRPRESSO`. Use
  `n_boot ≥ n_snps / sig_threshold`; a warning fires otherwise. Code that
  relied on 1.5.0's `(k + 1) / (B + 1)` floor must test `p < 1 / n_boot`
  instead of `p > 0`.

**Evidence.** `tests/reference_parity/test_mr_R_parity.py` against
`_fixtures/mr_R.json`, generated by `_fixtures/_generate_mr_R.R`.

---

<a id="evidence-grade-closed-form"></a>

## 1.29.0 — ⚠️ 23 functions are no longer graded as matching R / Stata

**Who is affected.** Anyone who cited `sp.parity_status(fn)` /
`docs/parity.md` for one of the functions below as evidence of agreement
with R or Stata. **No numerical output changed.**

`auc`, `roc_curve`, `bootstrap`, `breakdown_frontier`, `contrast`,
`direct_standardize`, `gelbach`, `icc`, `indirect_standardize`,
`kdensity`, `lee_bounds`, `lrtest`, `manski_bounds`,
`margins_at`, `mediate_interventional`, `mediation_decompose`,
`oster_delta`, `policy_value`, `power_case_control`, `pwcompare`,
`sensitivity_specificity`, `source_decompose`, `subgroup_decompose` —
`bit-exact` → `analytical-only`.

**Why.** Their tests verify closed-form identities; none of them compares
against an R or Stata run. That is real evidence of a different kind, and
it is now labelled as such. Eight further functions (`evalue_rr`, four
centrality measures, `mr`, `das_gupta` and `kitagawa_decompose`) kept
their grade because a real comparison was built for them in this release —
for `mr` and `das_gupta` that comparison found a correctness bug (see the
Mendelian-randomisation and decomposition entries).

**What to do.** If you cited one of these as "matches R / Stata", cite it
as "verified against a closed-form identity" instead, or run your own
comparison.

---

<a id="dyadic-directed"></a>

## 1.29.0 — ⚠️ `sp.dyadic_regression` on directed dyads

**Who is affected.** Anyone who passed `sp.dyadic_regression` data in which
the same two nodes appear in more than one row — most commonly directed
dyadic data carrying both `(i, j)` and `(j, i)`, as in trade or conflict
panels. One-row-per-unordered-pair data is unaffected.

| Surface | Changes? |
| --- | --- |
| coefficients | no |
| `se_dyadic`, `z`, `p`, `ci_low`, `ci_high`, undirected data | no |
| `se_dyadic`, `z`, `p`, `ci_low`, `ci_high`, directed data | **yes — 1.8% on a 20-node design** |
| rows with `i == j` | **now raise** |

**What was wrong.** The dyadic-robust variance should let each pair of
dyads that share a member contribute once. The implementation weighted a
pair by how many members it shares, so `(i, j)` and `(j, i)` — which share
both — contributed twice.

**What to do.** Re-run directed dyadic regressions. Drop self-dyads
(`i == j`) before calling; they are not dyads and have no consistent
weight in this estimator.

---

<a id="etwfe-cohort-att"></a>

## 1.27.0 — ⚠️ ETWFE cohort-level ATTs were wrong by up to 37%

**Who is affected.** Anyone who read a *per-cohort* ATT out of the ETWFE
family, or who called `sp.wooldridge_did` at all. Specifically:

| Surface | Changes? |
| --- | --- |
| `sp.wooldridge_did(...)` — `estimate`, `se`, `pvalue`, `ci`, `detail` | **yes** |
| `sp.etwfe(...).detail` (every `cgroup` / `panel`) | **yes** |
| `sp.etwfe_emfx(fit, type='group')` | **yes** |
| `sp.etwfe_emfx(fit, ..., weighting='cohort')` | **yes** |
| `sp.etwfe(...)` — `estimate`, `se`, `ci` | no — bit-identical |
| `sp.etwfe_emfx(fit, ..., weighting='treated')` | no — bit-identical |

If you reported an `sp.etwfe` pooled ATT, nothing you published moves. If you
reported cohort-level effects, or anything from `sp.wooldridge_did`, **rerun**.

**What was wrong.** Both functions ran two regressions on the same data. One
was saturated in cohort × period — one coefficient per treated cell — and fed
the event-study output. The other carried a *single post dummy per cohort* and
fed `detail`, plus the `sp.wooldridge_did` headline. Only the first is
Wooldridge's extended TWFE. The second is not saturated, so under dynamic
effects the already-treated cohorts enter the period fixed effects and
contaminate every treatment coefficient — the forbidden comparison that
extended TWFE exists to eliminate.

On a deterministic DGP whose true cohort ATTs are 3.0 and 2.5:

| | cohort 2 | cohort 3 | headline |
| --- | ---: | ---: | ---: |
| truth | 3.0000 | 2.5000 | 2.7500 |
| through 1.26.0 | 2.6739 | 1.6343 | 2.1541 |
| from this release | 2.9951 | 2.4363 | 2.7157 |

On the committed `17_etwfe` bytes, against R `etwfe::emfx(type='group')`:

| cohort | through 1.26.0 | from this release | R reference |
| --- | ---: | ---: | ---: |
| 2004 | −0.040562 | −0.0390171350 | −0.0390171349687 |
| 2006 | −0.035247 | −0.0311139256 | −0.0311139255617 |
| 2007 | −0.037735 | −0.0274615453 | −0.0274615452587 |

The old cohort ATTs were 3.9%, 13.3% and 37.4% away from the reference; the new
ones agree to 3e-13.

**A second, smaller change.** `sp.wooldridge_did`'s clustered SEs counted only
the explicit design columns in the CR1 factor `(N-1)/(N-K)`, ignoring the fixed
effects absorbed by demeaning. `fixest`'s `ssc(fixef.K="nested")` and
`reghdfe`'s default count every absorbed effect not nested inside the cluster
variable, and `sp.event_study` / `sp.sun_abraham` were moved onto that rule in
1.24.0. `sp.wooldridge_did` now follows it too. SEs rise by a uniform ~0.08% on
`17_etwfe`; agreement with Stata `jwdid, estat group` goes to 2e-15.

**What to do.**

- Cohort-level ETWFE numbers: rerun and republish. The direction of the change
  is not uniform — it depends on the cohort's exposure length and on how much
  the other cohorts' effects grow.
- Pooled `sp.etwfe` numbers: nothing to do.
- If you want the treated-observation-weighted simple ATT that R
  `emfx(type='simple')` and Stata `jwdid, estat simple` report, call
  `sp.etwfe`. If you want the cohort-size-weighted average of `ATT(g)` under a
  never-treated comparison group, call `sp.wooldridge_did`. These are two
  documented aggregations, not two estimators. On `17_etwfe` the `sp.etwfe`
  default sits 15.9% from the `sp.wooldridge_did` headline, and
  `sp.etwfe(cgroup='nevertreated')` — the same comparison group, only
  reweighted — still sits 10.5% from it. Check which one your write-up claims.

**Evidence.** Track A module `17_etwfe` grew from one statistic to eight and is
now three-way (StatsPAI / R `etwfe` / Stata `jwdid`) on both the pooled and the
per-cohort aggregation, under both comparison groups. All eight rows fit inside
the module's pre-existing registered budget; no tolerance was widened. The
module calls `sp.wooldridge_did` directly, so its restored `certified` grade
rests on an artifact rather than on the alias assertion that 1.26.0 withdrew.

---

<a id="sqreg-rounding"></a>

## 1.27.0 — `sp.sqreg` no longer rounds its output to four decimals

**Who is affected.** Anyone who read numbers out of `sp.sqreg`. Values
change in the 5th decimal onward; nothing about the estimator changed.

`sp.sqreg` applied `round(..., 4)` to every coefficient and standard error
before returning them. Display rounding belongs in `.summary()` and
`.to_latex()`, which already do it; this was the returned value. For a
coefficient of order `1e-3` that leaves a single significant digit.

Against R's `quantreg::rq` on the repository's fixture, the rounding
capped agreement at 1.2e-04 on `x2` and 7.8e-03 on `x3`. Without it the
coefficients agree to 3.5e-14.

**What to do.** Nothing, unless you compared `sp.sqreg` output for exact
equality against previously stored 4-decimal values. Standard errors
remain on the Powell kernel sparsity convention, which differs from R's
`se="nid"` default and Stata `qreg` by one scalar per quantile — see
`tests/reference_parity/test_sqreg_parity.py`, which now pins that scalar
structure explicitly.

---

<a id="panel-re-intercept"></a>

## 1.27.0 — ⚠️ RE panel binary models had no intercept; `panel_fgls` was `igls`

**Who is affected.** Anyone who called `sp.panel_logit(method='re')`,
`sp.panel_probit(method='re')` (including `method='cre'`, which routes
through the same fitter) or `sp.panel_fgls`.

| Surface | Changes? |
| --- | --- |
| `sp.panel_logit(method='re' \| 'cre')` — every coefficient | **yes** |
| `sp.panel_probit(method='re' \| 'cre')` — every coefficient | **yes** |
| `result.params` for those models | **gains a `_cons` row** |
| `sp.panel_logit(method='fe')` | no — the constant is differenced out there |
| `sp.panel_fgls(...)` with default arguments | **yes — now two-step** |
| `sp.panel_fgls(..., igls=True)` | no — this is the old behaviour |

**The intercept.** The RE fitter shares a grouping helper with the
conditional FE logit, which correctly has no constant. The helper builds
the design from the regressor names, and nothing added one back for the RE
path, so the random-effects likelihood was maximised over slopes and
`sigma_u` with the intercept pinned at zero.

The size of the resulting bias depends on the data, not on the estimator:
0.39% on the fixture here because the regressors are centred. Nothing
guarantees that.

**`panel_fgls`.** Stata's `xtgls` takes the variance parameters from the
OLS residuals, does one GLS solve, and stops. StatsPAI re-estimated them
from the GLS residuals and iterated, which is a different estimator —
Stata calls it `igls` and it is not the default. The docstring claimed
equivalence to the plain command.

**What to do.** Re-run random-effects panel binary models and read the new
`_cons` row. For `panel_fgls`, either accept the new (Stata-matching)
default or pass `igls=True` to reproduce your previous numbers exactly.

---

<a id="weakiv-vif"></a>

## 1.27.0 — `sp.vif` and the weak-IV confidence sets stop rounding

**Who is affected.** Anyone who read a VIF, or an endpoint from
`sp.anderson_rubin_ci` / `sp.conditional_lr_ci` / `sp.k_test_ci`.

| Surface | Changes? |
| --- | --- |
| `sp.vif` — `VIF`, `1/VIF` | **yes — full precision instead of 2 / 4 decimals** |
| `sp.anderson_rubin_ci` — `lower`, `upper` | **yes — the interval widens to its true boundary** |
| `sp.conditional_lr_ci`, `sp.k_test_ci` — `lower`, `upper` | **yes, same reason** |
| the same objects' `beta_grid`, `statistic`, `in_set` | no |
| `sp.anderson_rubin_test` | no (was already exact) |

`sp.vif` rounded its output. `car::vif` gives 1.1634266096594732 where
StatsPAI gave 1.16 — a 2.9e-3 error with no source but the `round()` call.

The confidence sets inverted the test on a β grid and then reported the
extreme grid point still inside the acceptance region. That is always
*inside* the true interval, so every reported set was too narrow by up to
one grid step. Endpoints are now found by bisecting the acceptance
boundary, which puts `sp.anderson_rubin_ci` at 5e-15 against `ivmodel`
where it was 8.1e-3.

**What to do.** Re-run if you reported a weak-IV confidence set; the new
one is wider. Nothing to do for VIF beyond expecting more digits.

---

<a id="spatial-lm-tests"></a>

## 1.27.0 — ⚠️ Spatial diagnostics: LM battery, join counts, Gi, residual Moran

**Who is affected.** Anyone who used `sp.lm_tests`, `sp.join_counts`,
`sp.getis_ord_local(star=False)`, `sp.moran_residuals` or read a z-score
off `sp.geary`.

| Surface | Changes? |
| --- | --- |
| `sp.lm_tests` — all five statistics | **yes; `Robust_LM_err` flips its conclusion** |
| `sp.join_counts` — `BW` | **yes — halved** |
| `sp.join_counts` — `BB`, `WW` | no |
| `sp.getis_ord_local(star=False)` | **yes** |
| `sp.getis_ord_local(star=True)` | no |
| `sp.moran_residuals` — statistic | no |
| `sp.moran_residuals` — p-value | **yes when `X=` is supplied** |
| `sp.geary` — `C`, `p_sim` | no |
| `sp.geary` — `variance`, `z_score`, `p_norm` | **yes: NaN → a value, and z changes sign** |
| `sp.moran`, `sp.moran_local`, `sp.getis_ord_g`, `sp.slx`, `sp.sac`, `sp.impacts` | no (verified exact) |

**The one that matters most.** `sp.lm_tests` is the Anselin battery used
to choose between a spatial lag and a spatial error model. Its `T` term
was computed as `tr(WW) + Σ_ij w_ij (WW')_ij` instead of
`tr(W'W) + tr(WW)`, and its `J` term from `M(Wy)` instead of `M(WXβ̂)`.
On a 10×10 rook lattice:

| statistic | before | `spdep::lm.RStests` |
| --- | ---: | ---: |
| `LM_err` | 39.465383 | 19.578530 |
| `LM_lag` | 20.150963 | 23.898571 |
| `Robust_LM_err` | 20.489246 | 0.039707 |
| `Robust_LM_lag` | 1.174826 | 4.359748 |

Read as a decision rule, the old numbers say "robust LM-error is
overwhelming (p = 6e-6), robust LM-lag is not (p = 0.28) → fit a spatial
error model". The correct numbers say the opposite: robust LM-lag is
significant (p = 0.037) and robust LM-error is not (p = 0.84) → fit a
spatial lag model. **Re-run the diagnostic before trusting a
specification chosen with it.**

**Why none of this was caught.** The spatial subpackage's tests assert
directions and ranges — "C below 1 for smooth data", "p_sim below 0.05" —
which every one of these defects satisfies. A doubled `LM_err` is still
significant; a doubled `BW` still exceeds its expectation; Gi with the
wrong standardisation still ranks hotspots in roughly the same order.
None of them survives being held against `spdep` on the same weights.

**A note on weights, which decides every number above.** `sp.W` is
**binary** until you set `w.transform = "R"`; `spdep::nb2listw` is
row-standardised by default. Comparing the two directly is not a parity
check, it is two different statistics. The new test file states the style
each assertion uses.

**What to do.**

* Re-run `sp.lm_tests` anywhere it informed a specification choice.
* Divide any stored `BW` join count by two, or re-run.
* Re-run `sp.getis_ord_local(star=False)`; `star=True` results stand.
* Pass `X=` (the design matrix, including its constant) to
  `sp.moran_residuals` for a p-value; without it the raw-variable null is
  kept, and is now documented as a fallback rather than presented as the
  residual test.
* `sp.geary`'s z-score changes sign. If you compared it against zero in a
  one-sided direction, flip the comparison; two-sided p-values are
  unchanged.

---

<a id="etregress-mle"></a>

## 1.27.0 — ⚠️ `sp.etregress` ran the two-step and called it MLE

**Who is affected.** Anyone who called `sp.etregress`. The default path
changes.

| Surface | Changes? |
| --- | --- |
| `sp.etregress(...)` with default `method='mle'` | **yes — a different estimator** |
| `sp.etregress(..., method='twostep')` coefficients | no |
| `sp.etregress(..., method='twostep')` standard errors | **yes — ~11% wider** |
| `sp.etregress(..., robust=...)` | **yes — the argument now does something** |
| `sp.etregress(..., cluster=...)` | **yes — the argument now does something** |
| `result.diagnostics['mills_coef']`, `['mills_se']` | present only under `twostep` |

**What was wrong.** The `mle` branch and the `twostep` branch of
`regression/selection.py` were the same code. The `mle` one carried the
comment `# MLE (simplified: control function + joint estimation)` followed
by `# Use two-step as approximation for MLE`. So the documented default —
"Equivalent to Stata's `etregress y x, treat(D = z)`" — returned the
control-function estimate. On a 2,000-observation replica with two
instruments, Stata's MLE gives a treatment effect of 1.787 and the old
default gave 1.978.

Separately, the two-step's standard errors were the OLS standard errors of
the regression that includes the hazard term. That regression's regressor
is estimated, and its errors are heteroskedastic by construction, so those
standard errors are too small — 11.2% on the same replica, in the
direction that overstates significance.

And `robust=` / `cluster=` were function parameters that the body never
read.

**Why it survived.** The analytical test for `sp.etregress` checks that the
estimate recovers δ on a known DGP and sits below the upward-biased naive
difference. A two-step passes that. No test compared the *two methods* to
each other, and none compared either to Stata — so a branch that was a
copy of the other branch looked healthy from every angle the suite had.

**What to do.**

* Re-run any endogenous-treatment model. If you want the old numbers back
  for comparison, they are `method='twostep'`'s point estimates, which are
  unchanged and now verified against Stata.
* Re-derive any inference: both methods' standard errors move.
* `result.params` now carries the selection equation as
  `<treatment>:<name>` and, for ML, `athrho` / `lnsigma`; `rho`, `sigma`,
  `lambda` and `loglik` are in `result.diagnostics`.

---

<a id="fuzzy-rd-cct"></a>

## 1.27.0 — ⚠️ Fuzzy RD bandwidth and robust inference

**Who is affected.** Anyone who called `sp.rdrobust(fuzzy=...)` or
`sp.rdbwselect(fuzzy=...)`.

| Surface | Changes? |
| --- | --- |
| `sp.rdbwselect(fuzzy=...)`, **two-sided** noncompliance | **yes — 9%–16% in `h`** |
| `sp.rdbwselect(fuzzy=...)`, one-sided noncompliance | no |
| `sp.rdrobust(fuzzy=...)` robust estimate | **yes — ~1%** |
| `sp.rdrobust(fuzzy=...)` robust SE | **yes — ~2.5%** |
| `sp.rdrobust(fuzzy=...)` conventional SE | **yes — ~1.1%** |
| `sp.rdrobust(fuzzy=...)` conventional estimate | no (was already exact) |
| Sharp RD, any argument | no |

**What was wrong.** Two separate things on the same path.

`sp.rdbwselect` accepted `fuzzy=`, validated the column, dropped its
missing rows, and then called the bandwidth cascade without it. The
returned bandwidth was the sharp one. The docstring meanwhile promised
that the bandwidth "accounts for first-stage variance in the Wald / IV
estimator".

`sp.rdrobust` did reach the CCT operator, but only for the sharp
quantities; it then divided the sharp bias-corrected estimate by a
separately bias-corrected first stage and divided the SEs by
`|first stage|`. Both are different quantities from `rdrobust`'s, which
treats the ratio as the estimand from the start.

**Why it survived this long.** The repository's fuzzy fixture used
*one-sided* noncompliance — nobody below the cutoff was treated. R's
`rdbwselect` detects exactly that (`perf_comp`: a side with no first-stage
variation), drops `T`, and returns the sharp bandwidth. So the fixture
reported the bandwidth agreeing with R to 1.2e-08 and that was read as
"the cascade already handles fuzzy". It was the sharp cascade agreeing
with itself. `tests/reference_parity/test_rdrobust_fuzzy_parity.py` now
runs on a two-sided design, and asserts the fuzzy bandwidth *differs* from
the sharp one — a test that would have failed against the broken version
rather than passing quietly.

**What to do.** Re-run any fuzzy RD whose design has noncompliance on both
sides. Point estimates from the conventional row are unchanged; the robust
row and every standard error are not.

---

<a id="rdms-not-rdmulti"></a>

## 1.27.0 — ⚠️ `sp.rdms` was not computing `rdmulti::rdms`

**Who is affected.** Anyone who called `sp.rdms`. Every number it returned
changes.

### What was wrong

Its docstring claimed equivalence to `rdmulti::rdms()`. It was a different
estimator, and not a defensible variant of one:

1. The default bandwidth was `1.06 * sd(distance) * n^(-1/5)` — Silverman's
   rule for **kernel density estimation**, applied as an RD estimation
   bandwidth. A category error, and the reason the window was roughly
   thirty times too narrow.
2. The variance was homoskedastic, not the CCT heteroskedasticity-robust
   sandwich.
3. There was no bias correction, so no robust interval either.

Measured on a 6,000-row two-score design with a vertical boundary and known
effects of 1.5 / 2.0 / 2.5 at three boundary points:

| boundary point | truth | `rdmulti::rdms` (obs used) | `sp.rdms` before (obs used) |
| --- | ---: | ---: | ---: |
| (0, −0.5) | 1.5 | 1.396 (519) | 1.322 (13) |
| (0,  0.0) | 2.0 | 1.778 (725) | **4.804 (8)** |
| (0, +0.5) | 2.5 | 2.463 (743) | 2.326 (27) |

The middle point is 170% out, reported with a standard error of 4.21
against the reference's 0.13.

### What it does now

The reference collapses the two scores onto one:

```text
d = sqrt((x1 - cutoff1)^2 + (x2 - cutoff2)^2) * (2 * treat - 1)
```

— Euclidean distance to the boundary point, signed by treatment status —
and runs a sharp RD at `d = 0`. `sp.rdms` now builds that score and
delegates to `sp.rdrobust`, so the CCT bandwidth cascade, the robust bias
correction and the heteroskedasticity-robust variance all come with it.

Following the reference, the reported estimate is the **bias-corrected**
point estimate with the **robust** interval; the conventional pair is in
`model_info["conventional"]`.

### What you need to do

- **Recompute every `sp.rdms` result.**
- **Pass `treat=`.** It is the treatment indicator (R's `zvar`). Treatment
  on a two-dimensional boundary is not implied by the coordinates, which is
  why the reference requires it. Without it, `sp.rdms` assumes
  `x1 >= cutoff1` and warns.
- `bandwidth=` still forces a fixed bandwidth; omit it to get the
  MSE-optimal one, as the reference does.

### Evidence

`tests/r_parity/89_rdms.{py,R}` and `tests/stata_parity/89_rdms.do` pin
three boundary points, each with the bias-corrected and conventional
estimates, their standard errors, the selected bandwidth and the effective
sample size on each side: 7.6e-12 against R, 3.3e-9 against Stata. The six
effective sample sizes are exactly equal on all three sides — integers, so
no tolerance is involved, and the row the superseded implementation would
have failed outright.

---

<a id="rdbwselect-cct-cascade"></a>

## 1.27.0 — ⚠️ `sp.rdbwselect` returned the wrong bandwidth

**Who is affected.** Anyone who called `sp.rdbwselect`, and anyone who
called `sp.rdrobust(bwselect=...)` with one of the four `comb` selectors.
Unlike the `wooldridge_did` entry below, **this one does change numbers**.

### What was wrong

Two independent defects, both found by packaging Track A module
`88_rdbwselect` — the first cross-language evidence this function has ever
had.

**1. The published selector ran a retired formula.** `rd/bandwidth.py`
never imported `rd/_cct_bandwidth.py`, so `sp.rdbwselect` used a
single-step rule of thumb while `sp.rdrobust` used the real CCT three-stage
cascade. On the Lee 2008 senate replica:

| | `h` (left) | `b` (left) |
| --- | ---: | ---: |
| `sp.rdbwselect` (before) | 4.632539 | 7.141922 |
| `rdrobust::rdbwselect` | 17.754397 | 28.028087 |
| `sp.rdbwselect` (now) | 17.754397 | 28.028087 |

The old formula's exponent `1/5` equals CCT's `1/(2p+3)` only at `p == 1`,
so it also failed to move with the polynomial order, and it produced no
separate bias bandwidth at all.

**2. The `comb` selectors were not implemented.** `msecomb1`, `msecomb2`,
`cercomb1`, `cercomb2` silently resolved to the plain `rd` cascade. The
reference computes the `rd`, `two` and `sum` cascades to completion and
combines the finished `h` and `b` element-wise per side:

```text
comb1 = min(rd, sum)
comb2 = median(two, rd, sum)
```

`comb1` therefore *appeared* correct wherever `rd` is already the smaller of
the pair, which is why it survived undetected. `comb2` did not:

| selector | before | `rdrobust` | relative |
| --- | ---: | ---: | ---: |
| `msecomb2` | 7.4141308229 | 7.4162332312 | 2.8e-4 |
| `cercomb2` | 7.6315619595 | 7.6738301717 | 5.5e-3 |

Because this path is shared, the defect reached `sp.rdrobust` — an
already-`certified` function — for those selector values.

### What you need to do

- **Recompute any bandwidth read from `sp.rdbwselect`.** Every returned
  value changes, in most cases by a factor of three to five.
- **Recompute any `sp.rdrobust(bwselect="msecomb2" | "cercomb2")` result.**
  Estimates move by 2.8e-4 to 5.5e-3 relative on the fixture above; the
  size on your own data depends on how far the three cascades separate.
- `sp.rdrobust` at its **default** `bwselect="mserd"`, and at `msetwo`,
  `msesum`, `cerrd`, `certwo`, `cersum`, is unaffected — those paths already
  matched `rdrobust` and still do, now to 1.8e-12 across a 68-cell sweep.
- Two API surfaces widened rather than changed: `sp.rdbwselect` now accepts
  `msesum` / `cersum` (previously rejected, though `comb1`/`comb2` are
  defined in terms of them), and returns full-precision floats rather than
  values rounded to six decimals.

### Evidence

`tests/r_parity/88_rdbwselect.{py,R}` and
`tests/stata_parity/88_rdbwselect.do` pin all ten selectors, polynomial
orders 1–3, three kernels, covariate adjustment, clustering and the RKD
derivative: 1.8e-12 against R, 3.7e-9 against Stata.
`tests/reference_parity/test_rdbwselect_comb_rules.py` pins the combination
rules as identities across random designs, which the numerical fixture
alone could not do — a one-dataset fixture certifies a `comb1` that is not
computing `comb1`.

---

<a id="wooldridge-did-evidence-grade"></a>

## 1.26.0 — ⚠️ `sp.wooldridge_did` is no longer `certified`

> **Superseded by [ETWFE cohort-level ATT](#etwfe-cohort-att) (1.27.0).**
> The withdrawal below was correct; its stated reason was not. The entry is
> kept verbatim because it shipped in 1.26.0 — read the newer entry for what
> the disagreement actually was, and note that the grade has since been
> restored on direct evidence.

**Who is affected.** Anyone who cited `sp.wooldridge_did`'s registry tier,
its `validation_notes`, or `sp.parity_status("wooldridge_did")` as evidence
that the estimator had been aligned against R or Stata. **No numerical
output changed** — `sp.wooldridge_did` returns exactly what it returned
before, and no result you have computed needs recomputing.

**What changed.** The function was marked `certified` because the registry
credited it with Track A module `17_etwfe` as an alias of `sp.etwfe`. The
alias was never measured. On that module's committed bytes, with identical
arguments:

| entry point | ATT | SE |
| --- | ---: | ---: |
| `sp.wooldridge_did` | −0.0378480795 | 0.0058045845 |
| `sp.etwfe` (default `cgroup='notyet'`) | −0.0351082766 | 0.0069250918 |
| `sp.etwfe(cgroup='nevertreated')` | −0.0329765138 | 0.0077660899 |

6.1% and 12.1% apart, and no control-group setting reconciles them: a
saturated cohort×post TWFE and an ETWFE with a not-yet-treated control
group are different estimators. The alias was withdrawn and the tier fell
to `validated`, which is what the function's own known-truth evidence
supports.

**What to do.**

- If you needed the ETWFE estimand aligned against R `etwfe` / Stata, call
  `sp.etwfe` — it is `certified` and carries module `17_etwfe`.
- If you want the saturated Wooldridge (2021) TWFE, keep calling
  `sp.wooldridge_did`; nothing about it changed except the honesty of its
  label.
- To see the evidence behind any grade:
  `sp.parity_status("wooldridge_did")` and
  `sp.describe_function("wooldridge_did")["validation_notes"]`.

The refuting measurement is asserted in
`tests/reference_parity/test_track_a_alias_equivalence.py`, so the claim
cannot quietly return.

---

<a id="forest-continuous-treatment-ate"></a>

## unreleased — ⚠️ `etwfe_emfx(type='event'|'calendar')` headline excludes leads and has an SE {#etwfe-emfx-event-headline}

**Affects:** `.estimate`, `.se`, `.pvalue`, `.ci` of `sp.etwfe_emfx(..., type='event')` and `type='calendar'`. The per-row `detail` table is unchanged.

**What changed.** The headline was the unweighted mean of all reported rows; with `include_leads=True` that averaged placebo leads into a treatment effect. It is now the mean of the post-treatment event-time rows (all rows for `'calendar'`), and its SE is the delta method through the rows' joint covariance (`model_info['vcov']`) instead of `NaN`.

**To reproduce the old point estimate:** `r.detail['estimate'].mean()`.

## `lp_did` pre-treatment leads: clean-control window now matches Stata `lpdid` {#lp-did-lead-window}

**Since:** unreleased (after 1.25.1). **Affects:** only the leads `h < 0` of `sp.lp_did` (`model_info['event_study']` rows with `relative_time < 0`) and the pooled pre window `model_info['pooled']['pre']`; post-treatment horizons and the pooled post window are unchanged.

**What changed.** A control observation at lead `h` was required to be untreated over `[t+h-1, t-1]`; it is now required over `[t+h, t-1]`, the periods whose outcomes enter `Y_{t+h} - Y_{t-1}`, which is the rule of Stata `lpdid` (`CCS_m<h> = CCS_0`). The old rule discarded the earliest observable calendar year at every lead.

**Size.** On the castle-doctrine panel the lead standard errors move by 3e-4 to 1.2e-3 at leads -2 to -4 and the lead -5 estimate by 6 percent; on the no-fault-divorce panel by 2e-3 to 7e-3 and 14 percent. The new numbers match `lpdid` to 1e-7 on both panels with identical per-lead sample sizes.

**To keep the old numbers:** there is no switch; the old window was a defect (a stricter sample than the estimator's definition), not a convention.

## 1.25.0 — ⚠️ Causal-forest ATE/ATT no longer report an AIPW score for a continuous treatment

**Who is affected.** Callers of `CausalForest.ate()`, `.att()`, or
`.average_treatment_effect()` on a forest fitted with
`discrete_treatment=False`, or with any treatment that is not 0/1.
Binary-treatment results are **unchanged**, including every Track A and
Track B row, all of which fit a binary treatment.

**What changed.** The doubly-robust score

```
psi_i = tau_i + (T_i - e_i) / (e_i (1 - e_i)) * (Y_i - m_i - (T_i - e_i) tau_i)
```

divides by `e(1 - e)` and is defined only when `e` is a propensity — that
is, only for a binary treatment. With a continuous treatment the same
nuisance slot holds `E[T | X]` on the treatment's own scale, and the
propensity clip mapped it into `[0.01, 0.99]`. On the Card
returns-to-schooling design `E[educ | X]` is about 13 years, which clipped
to 0.99 and made the weight about 1,200; the reported "ATE" was −1266.6
against a mean conditional effect of 0.086, with `p = 0.0000` attached.

The aggregation now detects a non-binary treatment, emits an
`AssumptionWarning`, and returns the plug-in average of the fitted
effects with `method="plug_in"` and
`plug_in_reason="non_binary_treatment"`. R's `grf` draws the same
boundary by refusing the aggregation outright for non-binary treatment;
StatsPAI reports the quantity that is defined rather than raising on a
fitted forest a user may only want a descriptive average from.
`ScalarEffect` additionally prints `descriptive SE` rather than `SE`
whenever the attached inference is a plug-in aggregation.

**What to do.** If you reported an ATE, ATT, standard error, confidence
interval, or p-value from a continuous-treatment causal forest, re-run:
the point value was the plug-in mean all along (the float has not moved),
but the inference attached to it was not. For doubly-robust inference,
fit with a binary treatment. For a continuous treatment, the printed
value is the average of `tau(x)` and its standard error is descriptive.

---

<a id="weak-iv-rank-deficient"></a>

## 1.29.0 — ⚠️ Sun–Abraham, GLMM, and PPML-HDFE standard errors realigned to their references

Point estimates do not change. Standard errors move by 0.1%–2% because
each estimator now reproduces the reference implementation's variance
construction exactly (parity modules 05, 26, 27, 37, 47).

| Function | What changed | New keyword |
|---|---|---|
| `sp.sun_abraham` | `K` in the cluster-robust small-sample factor counts only observed cohort × relative-time cells plus the fixed effects not nested in the cluster (fixest/reghdfe nested rule). Matches Stata `eventstudyinteract` to `8e-12`. | `share_variance=True` (default, Sun–Abraham Prop. 3 / Stata) or `False` (`fixest::sunab`, shares fixed). |
| `sp.melogit`, `sp.meglm`, `sp.mepoisson`, … | Fixed-effect covariance is the observed-information block of the marginal log-likelihood over all parameters (Stata `vce(oim)`, lme4), not the conditional information with the variance components held fixed. SEs grow by up to ~2%. | `result._cov_fixed_conditional` keeps the old matrix. |
| `sp.ppmlhdfe` | Robust sandwich factor is `N/(N-1)` (Stata `ppmlhdfe`) instead of `(N-1)/(N-k)`; `robust="robust"` includes the factor (was HC0). | `ssc="stata"` (default), `"fixest"` (`N/(N-K)`), `"none"`. |
| `sp.event_study` | The cluster-robust small-sample factor counts the absorbed time effects that are not nested in the cluster (fixest/reghdfe nested rule), so the event-time SEs match `fixest::feols` and `reghdfe` with default settings; SEs grow by ~0.3% on the module-85 fixture. | none (convention now matches both references). |

To reproduce pre-1.24.0 numbers: `sp.ppmlhdfe(..., robust="hc0")` gives the
bare sandwich; the old Sun–Abraham and GLMM variances are not reproducible
because they were not a documented convention.

## 1.29.0 — ⚠️ weak-IV confidence sets refuse collinear instruments

**Who is affected.** Callers of `sp.iv.anderson_rubin_ci` and
`sp.iv.clr_ci` whose instruments are collinear after the exogenous
regressors are partialled out. Full-rank instrument sets are unaffected.

**What changed.** These routines form `(Z'Z)⁻¹`. With a singular `Z'Z`, some
LAPACK builds raised a bare `LinAlgError: Singular matrix` from inside NumPy
while others returned an inverse and the routine reported a confidence set
computed from it. The instrument matrix is now rank-checked once in `_prep`,
so every entry point raises `IdentificationFailure` naming the redundant
instruments, with `diagnostics = {"n_instruments", "rank", "instruments"}`.

**What to do.** Drop the redundant instrument — one is an exact linear
combination of the others or of the exogenous controls.

---

<a id="ges-tie-break"></a>

## 1.29.0 — ⚠️ `sp.ges` edge orientation is now deterministic

**Who is affected.** Anyone comparing `sp.ges` output across machines, or who
observed an implausible complete undirected graph.

**What changed.** BIC assigns `i → j` and `j → i` the *same* score for an
isolated pair — they are the same Markov equivalence class. The greedy search
compared raw score gains with `>`, so the orientation of the first edge was
decided by last-bit LAPACK rounding, which differs between BLAS builds. The
choice cascades: on a collider `X → Z ← Y`, taking the reverse edge makes the
spurious explaining-away edge `X — Y` score better, and the search converges
on a complete undirected graph. Candidates must now beat the incumbent by more
than a relative tolerance, so the deterministic scan order breaks exact ties.

**What to do.** Nothing, unless you pinned a CPDAG that was produced by the
old tie-break on a platform where it went the wrong way. `sp.ges` now returns
the same graph everywhere; re-run and re-pin if so.

---

<a id="did-imputation-analytic-se"></a>

## 1.29.0 — ⚠️ `sp.did_imputation` analytic standard errors changed

**Who is affected.** Anyone who used `sp.did_imputation` (or its aliases
`sp.bjs` / `sp.borusyak_jaravel_spiess`) with the default
`vce="analytic"`. Point estimates are **unchanged** everywhere. Standard
errors, confidence intervals and p-values change for the overall ATT,
every event-study horizon, and the `hetby` and `project` outputs.

**What changed.** The estimator is linear in the outcome, so the weight
vector `v` with `tau = v'y` is computable exactly. StatsPAI approximated
it: it used balanced-panel unit/time shares in place of the least-squares
projection `v* = -Z(Z0'Z0)^-1 Z1'w`, and centred treated residuals on the
global mean effect at a horizon instead of the `v²`-weighted mean within
each (cohort, relative-time) block. Both reference implementations use
the exact form.

**Direction and size.** Too small, on the headline. 36% too small on the
harness fixture, 18% on `mpdta`. Horizons moved 4.9–13% with non-uniform
sign. After the fix everything reproduces Stata `did_imputation` and R
`didimputation` to ~5e-8.

**Coverage.** Measured over 600 replications on a 60-unit
homogeneous-effect design: 0.932 at a nominal 0.95, mean SE 0.94 of the
empirical SD. The approximation gave roughly 0.87. Still short of
nominal at this cluster count — prefer `vce="bootstrap"` in small
designs.

**The warning is gone.** The path used to emit a `UserWarning` calling
itself anti-conservative at ~0.87 coverage. That described the
approximation, not the estimator, so it went with it. If you were
suppressing that warning, you can stop.

**If you need the old numbers** for a reproduction, pin StatsPAI
`<=1.22.0`. There is no option to restore the approximation: it has no
reference implementation behind it and no setting in which it is the
right answer.

---

<a id="did-imputation-pretrend-method"></a>

## 1.29.0 — ⚠️ `sp.did_imputation` pre-trend coefficients changed

**Who is affected.** Anyone who read the **pre-treatment** rows of
`sp.did_imputation(...).model_info['event_study']`, or the
`model_info['pretrend_test']` built from them, in 1.22.0 or earlier.
Post-treatment coefficients, the overall ATT, its standard error, and
every `hetby` / `project` / `saveweights` output are **unchanged**.

**What changed.** The default construction of the leads. Before, the leads
were means of the imputation residual `Y − Ŷ(0)` at pre-treatment
relative times. Now they come from the auxiliary dynamic TWFE regression
on untreated observations that Stata `did_imputation, pretrends(k)` runs,
with all earlier relative times pooled into the omitted category.

**Why this is a correctness fix and not a preference.** The old leads are
*in-sample* prediction errors: the pre-treatment outcomes of
eventually-treated units are themselves part of what fits `Ŷ(0)`. Li and
Strezhnev (2025), restated in Roth (2026, appendix A), show that in a
non-staggered design this makes them exactly `N0/N` times the symmetric
benchmark, where `N0` counts never-treated units. StatsPAI reproduces
that identity to 1e-10. The docstring meanwhile advertised the Stata
option.

**What the attenuation does and does not cost.** An earlier version of
this note said a Wald test built on the attenuated leads under-rejects.
With the variance computed exactly (see the entry above) it does not:
the standard error attenuates by the same `N0/N` factor, so t-statistics
and the joint test are unchanged — checked at 50% and 90% treated. The
damage is to magnitude-based reasoning, which is most of what a
pre-trend plot is for, and to sensitivity analyses asking how large a
violation would overturn the result. Against the BJS convention, a
different construction rather than a rescaling, the tests do move. Both could not be true.

**How to restore the old numbers.**

```python
sp.did_imputation(..., pretrend_method="in-sample")
```

This is a supported option, not a deprecated shim: it is what R `fect`
and `did2s` report, so it is the right choice when reconciling against
those packages. It carries the attenuation caveat in
`model_info['event_study_convention']`.

**What to use for a plot.** Neither default is directly comparable to a
dynamic TWFE event study, because both build the two halves of the path
against different reference periods. For a path you can read with the
usual visual heuristics, use `pretrend_method="symmetric"` (Roth's
`β̂^{BJS,new}`; non-staggered balanced designs) or
`sp.callaway_santanna(..., base_period="universal")`, which is symmetric
by construction in staggered designs too and is StatsPAI's default.
`sp.compare_event_study_conventions()` quantifies the difference on your
own panel.

**One new refusal.** Requesting leads that cover *every* pre-treatment
period of a cohort now raises `MethodIncompatibility`: those lead
indicators are collinear with that cohort's unit fixed effects, and the
old path returned numbers for the rank-deficient system. Request at most
`(shortest pre-treatment history − 1)` leads.

**Not yet changed.** `sp.gardner_did` builds its leads the same in-sample
way. It is documented in `sp.event_study_convention()` and left alone
pending a matched reference run against R `did2s`.
<a id="hdfe-iv-inference"></a>

## 1.29.0 — ⚠️ HDFE-IV standard errors and diagnostics realigned to `ivreghdfe`

**Who is affected.** Anyone who ran `sp.iv(absorb=...)`, or who quoted a
`Kleibergen–Paap rk LM`, a first-stage `KP rk Wald F`, a Sargan statistic
from a clustered 2SLS fit, or a cluster-robust GMM standard error.

**What changed, and by how much.**

| Quantity | Before | Now | Typical size of the move |
| --- | --- | --- | --- |
| `sp.iv(absorb=, cluster=)` where the cluster is an absorbed FE | raised `ValueError` | runs, matches `ivreghdfe` | n/a — it did not work |
| Absorbed IV SE, cluster nested in an absorbed FE | FE DOF charged | nested FE DOF dropped, constant charged | SE falls ~5% at 120 clusters |
| Cluster-robust GMM SE | heteroskedastic meat, no finite-sample factor | cluster meat + CR1 factor | SE rises with within-cluster correlation |
| Over-id test under `robust=`/`cluster=` | Sargan | Hansen J | different statistic, same role |
| `KP rk Wald F` on a clustered fit | heteroskedasticity-only | cluster-robust | usually falls |
| `KP rk LM` | inflated by a factor of `n` | correct | millions → tens |
| `KP rk` / effective F on an absorbed fit | one control silently dropped | full reduced form | specification-dependent |

**What you should do.**

1. Re-run any absorbed-IV specification whose standard errors appear in a
   draft. The point estimates do not move — only the variance and the
   degrees-of-freedom bookkeeping.
2. Delete any reported `KP rk LM` value produced by an earlier version;
   it was not on the χ² scale it was compared against.
3. If a table reports a Sargan statistic next to clustered standard
   errors, it now reports Hansen's J instead. That is the correct test
   for that variance assumption; the numbers are not comparable.

**Opting out.** There is no flag to restore the old behaviour: each item
above was a defect relative to the reference implementation the docstrings
already claimed parity with. To reproduce `ivreg2 gmm2s`'s *variance
formula* (as opposed to StatsPAI's more agnostic sandwich) pass
`gmm_vcov="efficient"`.

**Verification.** `tests/reference_parity/test_iv_hdfe_stata_parity.py`
pins 30 quantities against Stata 18 MP (`ivreghdfe` 1.1.4, `ivreg2`
4.1.12, `reghdfe` 6.13.1, `ranktest` 2.0.04, `acreg` 1.1.0).

---

<a id="functional-form-test-default-binning"></a>

## 1.29.0 — ⚠️ `sp.functional_form_test` default binning changed

**Who is affected.** Anyone who called `sp.functional_form_test` in
1.21.0 or 1.22.0 **without** passing `n_bins` or `binpoints`. If you
passed either one explicitly, nothing moves.

**What changed.** The default went from `n_bins=10` to `n_bins="auto"`.
`"auto"` is the rule `didFF` 0.1.0 uses:

- fewer than 20 distinct outcome values among untreated observations →
  the outcome is treated as **discrete**, one bin per value, with a
  warning;
- otherwise → cut into `min(20, n_distinct)` equal-width bins.

An explicit integer still always cuts, however few distinct values the
outcome takes.

**Why.** The binning is not a display choice, it is the test. It fixes
the resolution at which a negative implied density can be detected, so
two runs that bin differently answer different questions. Shipping a
default that differs from the reference implementation meant
`sp.functional_form_test(df, ...)` and `didFF(data, ...)` returned
different p-values on the same data with no indication why — the worst
kind of parity gap, because it looks like a disagreement about the
method rather than about the grid.

**What to do.** To reproduce a p-value computed under 1.21.0 or 1.22.0,
pass `n_bins=10` explicitly:

```python
res = sp.functional_form_test(df, y="lemp", g="first_treat",
                              t="year", i="countyreal", n_bins=10)
```

Otherwise re-run and report the new number. Expect the p-value to move
in either direction: a finer grid has more power against a localised
violation, and more bins to be uninformative about.

**Two smaller changes ride along.** `binpoints` that stop short of the
outcome range are now padded (with a warning) rather than silently
dropping the uncovered mass, and passing both `binpoints` and an
explicit `n_bins` now raises rather than silently preferring
`binpoints`. Both match `didFF`.

---

<a id="staggered-rollout-singleton-cohorts"></a>

## 1.29.0 — `sp.staggered_rollout` no longer raises on singleton cohorts

**Who is affected.** Anyone whose panel has a treatment cohort
containing exactly one unit.

**What changed.** That case used to raise `DataInsufficient`. It now
drops the offending cohort, warns, and estimates on the rest — which is
what R `staggered` 1.2.2 does. The within-cohort covariance of a
one-unit cohort is not estimable, so the cohort cannot contribute
either way; the old behaviour just refused to proceed.

**What to do.** Nothing, unless you were catching `DataInsufficient` to
detect this case. If a dropped cohort matters to your estimand, the
warning names it.

**No estimate that previously succeeded moves**: panels without
singleton cohorts take an identical path.

---

<a id="drdid-full-family"></a>

## 1.29.0 — ⚠️ `sp.drdid` on repeated cross-sections changes

**Who is affected.** Anyone calling `sp.drdid` **without** `id=`. The
`id=` panel path is unaffected except in its last few digits (see the
tilting note below).

**What was wrong.** Two things, both silent:

1. `method='imp'` and `method='trad'` computed the *same estimator*. They
   agreed to 3e-13 — floating-point reassociation, not a real choice. The
   estimator you actually got was `DRDID::drdid_rc1`, reported under an
   "Improved" or "Traditional" label picked by an argument that did
   nothing.
2. The standard error was a nonparametric bootstrap over `n_boot`
   resamples, not the Sant'Anna–Zhao influence function. It was random,
   changed with `n_boot` and `random_state`, and sat 1.5–5.5% off the
   reference.

There was also a fallback branch that, when a treatment × period cell was
too small to fit the outcome regression, quietly dropped the covariates
and returned an unadjusted 2×2 DID under the doubly-robust label.

**What changes.**

| call | before | after |
|---|---|---|
| `sp.drdid(...)` (default `method='imp'`) | `drdid_rc1`, bootstrap SE | `DRDID::drdid_imp_rc`, analytic SE |
| `sp.drdid(..., method='trad')` | `drdid_rc1`, bootstrap SE | `DRDID::drdid_rc`, analytic SE |
| too-small cells | unadjusted 2×2 DID, no warning | raises `DataInsufficient` |

**Point estimates move** whenever `method='imp'` was used or defaulted to.
**Standard errors move in every case**, and are now deterministic —
`n_boot` and `random_state` no longer affect them at all.

**What to do.** Re-run any `sp.drdid` call made without `id=`. Check
`result.model_info["engine"]`, which now names the exact DRDID function
that ran, against what you meant to estimate.

```python
r = sp.drdid(df, y="y", treat="d", time="post", covariates=["x1", "x2"])
r.model_info["engine"]            # 'drdid_imp_rc'
r.model_info["drdid_reference"]   # 'DRDID::drdid_imp_rc'
```

**Also new, nothing breaking.** `est_method=` (`'dr'`/`'ipw'`/`'reg'`/
`'twfe'`), `normalized=`, `locally_efficient=`, `weights=` and
`trim_level=` open up the rest of the DRDID family — 14 estimators in
total, all pinned against R `DRDID` 1.2.3. `sp.drdid(..., id=,
method='trad')` used to raise and now runs `DRDID::drdid_panel`.

**Improved estimators move in the last digits.** The inverse
probability tilting solver now runs a Newton refinement to a gradient of
1e-13, where it previously stopped at BFGS's tolerance. Tilting is
supposed to make the covariate-balance conditions hold *exactly*; a
1.5e-9 slack was moving `drdid_imp_rc` by 7e-4 in the ATT and 2.9% in the
SE. Anything with `method='imp'` shifts slightly, toward the reference.

---

<a id="cs-clustered-bootstrap-unequal-clusters"></a>

## 1.29.0 — ⚠️ clustered CS bootstrap SEs change on unequal clusters

**Who is affected.** Anyone calling
`sp.callaway_santanna(..., clustervars=<var>, bstrap=True)` — or
`sp.aggte(..., bstrap=True)` on such a fit — where the clusters are
**not all the same size**. Clustering counties by state, firms by
industry, schools by district: the normal case. Equal-sized clusters are
unaffected, as is every unclustered bootstrap and every analytic SE.

**What changed.** The multiplier bootstrap collapsed the influence
functions to cluster *means* and divided by `n_clusters`. It now
collapses to cluster *sums* and divides by `n` (the unit count).

**Why.** The old form is the cluster-robust variance only when all
clusters are the same size. Otherwise each cluster enters with weight
`1/|c|`, so the smallest clusters dominate the variance. Concretely, on
a 500-county panel clustered into 9 states of sizes 3–317, the old SEs
were 1.5× to 11× too large; on a 1–150 spread, ~5× too large. The
cluster-sum form reduces to the ordinary `σ²/n` when there is no
within-cluster correlation, however the units happen to be partitioned;
the cluster-mean form does not.

**This is a deliberate divergence from CRAN `did` 2.3.0.** StatsPAI's old
behaviour matched it exactly — this was a faithful port, not a
transcription error. Upstream `did` (GitHub master, post-2.3.0) has since
switched to the cluster-sum form, with a source comment stating the old
aggregation only coincides for equal-sized clusters; the `csdid` Python
port already tracks the corrected form. If you are reconciling StatsPAI
against **R `did` 2.3.0** on unequal clusters, the two will now differ,
and StatsPAI is deliberately following the corrected upstream.

**What to do.** Re-run clustered `bstrap=True` inference. SEs will
usually **shrink**, so confidence intervals narrow and p-values fall —
the old numbers were conservative, not wrong in a direction that
protected you against false positives elsewhere. Point estimates do not
move.

```python
# unchanged call; the SEs behind it are now the cluster-robust ones
fit = sp.callaway_santanna(df, y="lemp", g="first_treat", t="year",
                           i="countyreal", clustervars="state",
                           bstrap=True, biters=1000, random_state=0)
```

---

<a id="cs-anticipation-varying-base"></a>

## 1.29.0 — ⚠️ `anticipation>0` pre-treatment placebos change

**Who is affected.** Anyone calling `sp.callaway_santanna` (or `sp.did`
routing to it) with **both** `anticipation > 0` **and**
`base_period="varying"`. The default is `base_period="universal"`, which
is unaffected, as is `anticipation=0`.

**What changed.** StatsPAI shifted the base period back by
`anticipation` for every cell. It now shifts it only for
*post*-treatment cells, matching R `did`: a pre-treatment placebo keeps
the period immediately before it as its base, so its value no longer
moves when you change `anticipation`.

**What moves.** Pre-treatment ATT(g,t) — event-study leads, the joint
pre-trend test, anything built on placebo cells. The earliest
pre-treatment cells, previously dropped from the grid, now appear.
**Post-treatment ATT(g,t) do not move**, so `sp.aggte` overall/simple/
group/calendar effects and the headline ATT are unchanged.

Separately, cohorts with no period satisfying `t + anticipation < g` are
now dropped with an explicit warning rather than silently, and the
period grid is compared with strict inequalities instead of `t − 1`
arithmetic — so irregularly spaced periods (1990, 1995, 2000, …) resolve
to the neighbouring observed period instead of losing the cell.

---

<a id="cs-allow-unbalanced-panel"></a>

## 1.29.0 — `allow_unbalanced_panel` on `sp.callaway_santanna`

**Nothing breaks.** This is a new opt-in argument, defaulting to `False`,
which is the previous behaviour.

**What it is for.** With `panel=True` and units missing periods, the
default route still forms within-unit differences, so a unit missing
either the base or the comparison period drops out of *that cell* and
the effective sample varies from cell to cell. That is now stated in the
existing unbalanced-panel warning, which names the option.

```python
fit = sp.callaway_santanna(df, y="y", g="g", t="t", i="i",
                           allow_unbalanced_panel=True)
```

switches to the repeated-cross-section estimators, which never difference
within unit and therefore keep every observed row, and folds the
influence functions back to the unit level so standard errors still
account for within-unit correlation. This mirrors
`did::att_gt(allow_unbalanced_panel = TRUE)` and agrees with it to
≤7e−15 on ATT(g,t) and to machine precision on SEs.

The flag is **inert on a balanced panel** (as in R), so turning it on
cannot quietly change results that did not need it. It does not yet
combine with `weights=` or `clustervars=`; both raise rather than being
ignored.

**The two routes are different estimators, not two precisions of one.**
Expect different numbers on the same unbalanced data, and say which you
used.

---

<a id="did-weights-and-ipw-se"></a>

## 1.29.0 — DiD unit weights, and IPW standard errors

Two output-changing fixes in the DiD family, both prompted by Baker,
Callaway, Cunningham, Goodman-Bacon & Sant'Anna (2026), *JEL* 64(2),
498–557 (doi:10.1257/jel.20251650).

### 1. `weights=` was silently dropped on every staggered estimator

**Who is affected.** Anyone who called `sp.did(..., weights="pop")` with
`method` in `{cs, callaway_santanna, sun_abraham, bjs, sdid}`. The
argument was accepted, validated, and then never forwarded. You received
the **unweighted** estimand with no warning.

**Why it matters.** Unit weights are not a precision setting. They enter
the definition of the target parameter: the unweighted ATT averages over
treated *units*, the ω-weighted ATT averages over the *population those
units represent*. These can differ in sign. Comparing a weighted with an
unweighted estimate is therefore not a robustness check — it is two
different questions.

**What to do.**

```python
# Before (silently unweighted):
r = sp.did(df, y="y", treat="g", time="t", id="i",
           method="cs", weights="pop")

# Now — same call, and ω is actually applied:
r = sp.did(df, y="y", treat="g", time="t", id="i",
           method="cs", weights="pop")
r.model_info["weighted"]      # True
r.model_info["weights"]       # 'pop'

# Or directly, mirroring R did::att_gt(weightsname="pop"):
cs = sp.callaway_santanna(df, y="y", g="g", t="t", i="i", weights="pop")
sp.aggte(cs, type="dynamic")  # cohort shares are now ω-mass, not counts
```

`sp.sun_abraham` also implements ω now (projection, solve,
cluster-robust variance, and interaction weights). Estimators that still
do not (`sdid`, `bjs`, `gardner_did`, `stacked_did`, `lp_did`) **raise**
`MethodIncompatibility` rather than ignoring the argument. Either switch
to `method="cs"` or drop `weights=` and report the unweighted estimand
explicitly.

Weights must be constant within unit; a time-varying column now raises.

### 2. `estimator="ipw"` standard errors were up to 89% too large

**Who is affected.** Anyone who reported inference from
`sp.callaway_santanna(..., estimator="ipw")` or
`sp.did(..., method="cs", estimator="ipw")`. **Point estimates are
unchanged**; every SE, z-statistic, p-value, and confidence interval
moves.

**Why.** The influence function centred both arms on the ATT rather than
on each arm's own Hájek mean, and omitted the propensity-score estimation
effect. The errors always went the same way — too wide — so prior
intervals were conservative, not anti-conservative. Against
`did::att_gt(est_method="ipw")` the gap was 9–11% unweighted and up to
89% weighted; it is now 5e−11.

`estimator="reg"` is unaffected.

### 2b. `estimator="dr"` standard errors move when covariates are supplied

The same change set also completed the DR influence function by
propagating DRDID's two nuisance estimation effects. DR is
Neyman-orthogonal in each nuisance separately, so the gap was small
(≤0.9% against `DRDID::drdid_panel`) — but it was real.

**Who is affected.** Anyone reporting inference from
`estimator="dr"` **with covariates**. Without covariates the correction
is identically zero, so covariate-free results are unchanged.

```python
# Covariate-free: unchanged, correction is exactly zero.
sp.callaway_santanna(df, y="y", g="g", t="t", i="i", estimator="dr")

# With covariates: SE moves toward the R reference.
sp.callaway_santanna(df, y="y", g="g", t="t", i="i",
                     x=["x1"], estimator="dr")
```

All three covariate strategies now agree with R `did` 2.3.0 to ≤4.6e−11
across a 72-cell grid (3 strategies × 2 comparison groups × weighted and
unweighted × with and without covariates × 3 aggregations).

```python
# Re-run and compare:
old_ci = ...                      # from a pre-fix run
new = sp.callaway_santanna(df, y="y", g="g", t="t", i="i",
                           x=["x1"], estimator="ipw")
new.se                            # smaller than before; matches R did
```

### 3. Unknown keyword arguments now raise

`sp.did` used to swallow anything it did not recognise. These were all
accepted and did nothing:

| stale | correct |
| --- | --- |
| `sp.did(..., post="post")` | the post indicator is inferred from `time=` |
| `sp.did(..., repeated_cs=True)` | `panel=False` |
| `sp.did(..., d="treated")` | `treat="treated"` |
| `sp.honest_did(r, max_M=0.2)` | `m_grid=[0.0, 0.1, 0.2]` |

---

<a id="precision-vocabulary"></a>

## 1.29.0 — one precision vocabulary for every exporter

**What changed.** Precision used to be spelled differently by every
exporter, and the ``"auto"`` sentinel only worked in ``sp.regtable``.

| exporter | before | now |
| --- | --- | --- |
| ``sp.regtable`` / ``sp.esttab`` / ``sp.modelsummary`` | ``fmt=`` | unchanged, plus ``digits=`` |
| ``sp.sumstats`` / ``sp.mean_comparison`` | ``fmt="%.3f"`` / ``"%.2f"`` | ``fmt=`` / ``digits=``, adaptive default |
| ``sp.outreg2`` | ``decimal_places=3`` | ``decimal_places=`` still works, adaptive default |
| ``sp.fast.etable`` | ``digits=3`` | ``digits=`` / ``fmt=``, adaptive default |
| ``.to_markdown()`` / ``.to_html()`` | ``digits=4`` | ``digits=`` / ``fmt=``, adaptive default |
| ``.to_excel()`` | ``digits=6`` | **unchanged** — numeric, for data interchange |

Every one of them now accepts the same spellings, taken from the tools
people already use:

```python
sp.sumstats(df, digits=3)      # R modelsummary / stargazer
sp.sumstats(df, fmt="%.3f")    # Stata esttab's b(%9.3f)
sp.sumstats(df, fmt="r3")      # R fixest: round to 3 decimals
sp.sumstats(df, fmt="s3")      # R fixest: 3 significant digits
sp.sumstats(df, fmt="auto")    # StatsPAI: pair each estimate with its SE
```

**Two bugs this closes.**

1. ``fmt="auto"`` on ``sp.sumstats`` / ``sp.mean_comparison`` used to fill
   every cell with the literal word ``auto``. ``"auto" % value`` returns
   ``"auto"`` unchanged when the template has no conversion specifier, so
   nothing raised — the table just came out as garbage.
2. ``sp.etable`` returned bare coefficients with no standard errors on the
   non-pyfixest path.

**Who is affected.**

- **Called any of these with no precision argument** — output changes where
  the old fixed default was wrong for the scale. A ``$50,229`` mean printed
  as ``50228.947`` now prints as ``50,229``.
- **Passed an explicit ``fmt=`` / ``digits=`` / ``decimal_places=``** —
  nothing changes.

**How to keep the old output.** Pass the previous default explicitly:

```python
sp.sumstats(df, fmt="%.3f")             # pre-Unreleased sumstats default
sp.mean_comparison(..., fmt="%.2f")     # pre-Unreleased default
sp.outreg2(..., decimal_places=3)       # pre-Unreleased default
result.to_markdown(digits=4)            # pre-Unreleased default
```

---

<a id="table-precision-pairing"></a>

## 1.29.0 — ⚠️ regression-table decimal places change

**What changed.** Two things, both in the table renderer only. No estimator,
no standard error, no p-value is affected — this is presentation.

**1. `sp.regtable` now defaults to `fmt="auto"`** (previously `"%.3f"`).
`sp.esttab` and `sp.modelsummary` follow it (both were `"%.4f"`).

**2. `fmt="auto"` picks precision per coefficient/SE *pair*, not per cell.**
The old implementation read each value's magnitude independently, so the two
halves of one estimate could disagree:

| | old `fmt="auto"` | new |
| --- | --- | --- |
| age coefficient | `-5.22` | `-5.22` |
| its standard error | `(45.3)` | `(45.34)` |
| earnings coefficient | `24.8` | `24.8` |
| its standard error | `(140)` | `(140.2)` |

A coefficient printed to two decimals above a standard error printed to one
is not a convention any economics journal follows. Each coefficient row now
resolves to a single decimal count — the finer of what the estimate and the
standard error each need, so neither loses a significant digit — and every
model column in the panel shares it.

**Who is affected.**

- **Called `sp.regtable(...)` with no `fmt=`** — output changes only where
  fixed `"%.3f"` was wrong for the scale. For sub-unit coefficients the
  rendering is byte-identical to before (the committed snapshot fixtures did
  not move). A dollar-magnitude row changes from `2108.412*** (471.938)` to
  `2,108*** (472)`.
- **Called with `fmt="auto"`** — rows are now internally consistent, per the
  table above.
- **Called with an explicit template** (`fmt="%.3f"`, `"%.4f"`, `"%.0f"`) —
  **nothing changes.** Explicit precision is honoured verbatim, as before.

**How to keep the old output.** Pass the old default explicitly:

```python
sp.regtable(m1, m2, fmt="%.3f")       # pre-Unreleased regtable default
sp.esttab(m1, m2, fmt="%.4f")         # pre-Unreleased esttab / modelsummary
```

**Related behaviour worth knowing.**

- `fmt="auto"` no longer floors at three decimals, so a coefficient of
  `0.00042` prints as `0.00042` rather than `0.000`. Values at or above
  `0.001` are unaffected.
- Summary-statistic rows (R², adj. R², F) and `tests=` footer statistics
  never followed `fmt` — they were hard-pinned to `"%.3f"`. They now have
  their own `stats_fmt=`, still defaulting to `"%.3f"`, so existing output
  is unchanged.
- `fmt=3` now works as shorthand for `fmt="%.3f"`; previously it raised
  `TypeError: unsupported operand type(s) for +=: 'float' and 'str'` from
  inside the renderer. `digits=3` is the same knob under the R/Stata name.

---

<a id="aggte-weight-influence"></a>

## 1.29.0 — ⚠️ `sp.aggte` standard errors get larger

**What changed.** The Callaway–Sant'Anna aggregation weights are
*estimated* cohort shares $\hat p_g = \widehat{P}(G = g)$, not constants.
`sp.aggte` was treating them as fixed, which drops a term from the
variance (R `did`'s `wif`, "weight influence function"). The reported
standard errors were therefore **too small** — anti-conservative — on
every aggregation that mixes more than one adoption cohort.

**Point estimates never changed.** They already matched R `did` and Stata
`csdid` to ~1e-11. Only `se`, the confidence interval, and the p-value
move, and they move in the conservative direction (SEs widen).

Real `did::mpdta`, `bstrap=False`, `base_period='universal'`:

| aggregation | old SE | new SE | R `did` |
| --- | ---: | ---: | ---: |
| `simple` | 0.0117467 | **0.0120340** | 0.0120340 |
| `dynamic` (overall) | 0.0199587 | **0.0199650** | 0.0199650 |
| `group` (overall) | 0.0123872 | **0.0124461** | 0.0124461 |
| `calendar` (overall) | 0.0158022 | **0.0159719** | 0.0159719 |
| `calendar`, t=2006 | 0.0184354 | **0.0201259** | 0.0201259 |

The largest gap observed was **8.4%** (`calendar` t=2006). On the
Cheng–Hoekstra castle panel the `simple` SE moves 0.038602 → **0.038724**,
matching both R and Stata exactly.

**Who is affected.** Anyone quoting a CS standard error, CI, or p-value
from `sp.aggte` — including `sp.cs_report`, `sp.did_report`, and
`honest_did` inputs built off `aggte`. Re-run and re-quote. A result that
was significant at exactly 5% may no longer be.

**Who is not.** Per-cohort cells of `type='group'` were always correct:
within a single cohort the $\hat p_g$ factors cancel and the omitted term
is identically zero. That is precisely why the bug survived — every
internal consistency check passed.

**No flag restores the old behaviour.** It was not an alternative variance
convention; the term was missing. Both the analytic path and
`bstrap=True` now derive from the same corrected influence functions, so
they cannot drift apart.

**How this was found.** The Cheng–Hoekstra castle-doctrine replication
(`sp.replicate('castle_2013')`) compared StatsPAI against Stata 18 MP and
R `did` on real data. See
[the guide](docs/guides/mixtape_castle_replication.md).

<a id="vcov-silently-ignored"></a>

## 1.29.0 — ⚠️ `vcov=` on `sp.regress` / `sp.ivreg` was ignored

**What changed.** `sp.regress` and `sp.ivreg` accepted a `vcov=`
argument — the pyfixest spelling used by `sp.feols` — and silently
discarded it, reporting **default (unclustered) standard errors**. The
call raised nothing and printed nothing.

```python
# Before: returned plain OLS standard errors, no warning.
# After:  clustered, identical to cluster="firm".
sp.regress("y ~ x", df, vcov={"CRV1": "firm"})
```

In a 15-cluster example the reported SE was `0.109` where the clustered
value is `0.063` — **1.7× too small**, i.e. t-statistics inflated by the
same factor.

**Who is affected.** Anyone who passed `vcov=` to `sp.regress` or
`sp.ivreg` — most likely users moving between `sp.feols` (where `vcov=`
is native) and the other two. `sp.feols` itself was always correct.
**Re-run those regressions**; the previously reported standard errors,
t-statistics, p-values, and confidence intervals were wrong. Point
estimates are unaffected.

**How to check an archived result.** If a saved `sp.regress` result used
`vcov=` and its SEs match a plain unclustered run, it hit this bug.

**New behaviour.**

| Spelling | Maps to |
| --- | --- |
| `vcov="iid"` | `robust="nonrobust"` |
| `vcov="hetero"` / `"HC0"`–`"HC3"` | `robust="hc0"`–`"hc3"` |
| `vcov={"CRV1": "firm"}` | `cluster="firm"` |
| `vcov={"CRV2": "firm"}` | `cluster="firm", vce="CR2"` |
| `vcov={"CRV3": "firm"}` | `cluster="firm", vce="CR3"` |

Supplying `vcov=` together with a conflicting `robust=` / `cluster=` /
`vce=` now raises `MethodIncompatibility` rather than silently choosing
one.

**Unrecognised keywords now raise.** Relatedly, `sp.regress` and
`sp.ivreg` forwarded any unknown keyword into `fit(**kwargs)`, where it
vanished — a misspelled `robsut="hc1"` quietly produced default
standard errors. Both now raise `TypeError`, matching `sp.feols` and
`sp.did_2x2`. If you have code passing an option that was never
implemented, it will now fail loudly instead of being ignored.

---

<a id="hausman-integer-dtype"></a>

## 1.29.0 — ⚠️ Durbin-Wu-Hausman test with integer treatments

**What changed.** The endogeneity test reported by `sp.ivreg` /
`sp.iv` computed its first-stage residuals in the dtype of the
endogenous regressor. With an integer 0/1 treatment — what you get from
`d = (...).astype(int)` — the residuals truncated to zero and the test
returned `NaN`. With integer counts it returned a **finite but wrong**
statistic (4971.6 against a correct 5316.7) with no indication.

**Who is affected.** Anyone who read `Hausman F-stat` / `Hausman
p-value` off an IV result whose endogenous regressor was an integer
column. A `NaN` was visible; the wrong-but-finite case was not. Point
estimates, standard errors, and the first-stage F are unaffected — only
the DWH endogeneity diagnostic.

**Fix.** Computed in float regardless of input dtype. Results for
float-typed regressors are bit-identical to before; casting your
treatment with `.astype(float)` reproduced the correct value under the
old code.
<a id="sunab-share-variance"></a>

## 1.29.0 — ⚠️ `sp.sun_abraham` standard errors rise at multi-cohort event times

**What changed.** The interaction-weighted estimator is
δ̂_ℓ = Σ_g ŵ_{g,ℓ} β̂_{g,ℓ} — a product of *two* estimated objects. Sun &
Abraham (2021), Prop. 3 accordingly gives it a two-part variance:

```
Var(δ̂_ℓ) = w_ℓ' Var(β̂) w_ℓ   +   β_ℓ' Var(ŵ_ℓ) β_ℓ
            \___ regression ___/     \___ cohort shares ___/
```

StatsPAI computed only the first part. The second is dropped-to-zero
whenever a single cohort is eligible at ℓ (there ŵ ≡ 1 carries no
uncertainty), which is why the omission survived: it is *exactly* correct
at single-cohort event times and only bites where cohorts pool.

**Effect.** Point estimates, confidence-interval centres and the overall
ATT point estimate are **unchanged**. Per-event-time SEs — and hence CIs
and p-values — **increase** at event times where two or more cohorts
contribute. On `mpdta` the increase is 0.6–2.0%; it grows with the number
of pooled cohorts and with the dispersion of β̂ across them. The previous
numbers were anti-conservative.

| `mpdta`, never-treated control | old SE | new SE | Stata |
| --- | ---: | ---: | ---: |
| e = 1 (2 cohorts) | 0.016884 | **0.016978** | 0.016964 |
| e = 2 (1 cohort) | 0.036619 | 0.036619 | 0.036589 |

**Who should re-run.** Anyone quoting `sp.sun_abraham` event-study SEs,
CIs or p-values from a staggered design with more than one cohort at a
given relative time. Point estimates need no revision.

**Reference divergence — read before "fixing" this back.** R
`fixest::sunab` treats the cohort shares as fixed and reports the
first-term-only SE; Stata `eventstudyinteract` carries both terms.
StatsPAI now follows `eventstudyinteract`, which is Liyang Sun's own
implementation of her paper. So StatsPAI SEs now sit slightly *above*
`fixest`'s at multi-cohort event times and match it exactly at
single-cohort ones. That divergence is deliberate; there is no flag to
restore the old behaviour.

<a id="cardinality-match-exact"></a>

## 1.22.0 — ⚠️ `sp.cardinality_match` matched sets change

**What changed.** The estimator relaxed its binary program to a continuous
LP and rounded the weights. Rounding does not preserve the balance
constraints, so the returned sample violated the `smd_tolerance` it
advertises. It is now solved exactly (`scipy.optimize.milp`, HiGHS).

| | old (LP + rounding) | new (exact ILP) |
| --- | ---: | ---: |
| infeasible cells (12-cell grid) | **9** | **0** |
| worst breach against a 0.05 request | 0.0631 | 0.0500 |

**Do my numbers change?** Yes — matched sets, and therefore the ATE and its
SE, move. That is the point: the previous solutions sat outside the feasible
region, so they never satisfied the balance guarantee the method is defined
by. An estimate published before 1.22 should be re-run; the matched sample
it used was not the one cardinality matching specifies.

New `time_limit` (default 30s) bounds the solve. An infeasible request now
raises instead of silently returning a breach.

<a id="overlap-weights-mle"></a>

## 1.22.0 — ⚠️ `sp.overlap_weights` estimates move by ~1e-6

**What changed.** The propensity score was fitted with
`sklearn.LogisticRegression(C=1e6)` — a penalised likelihood however large
`C` is — while the rest of the matching module uses the unpenalised MLE. It
now uses that same MLE.

Against R's `glm(family = binomial)` and `WeightIt::weightit`:

| | propensity vs R | ATO / ATE / ATT / ATC vs WeightIt |
| --- | ---: | ---: |
| old (penalised) | 8.5e-06 | ~1e-6 |
| new (MLE) | 2.6e-14 | ~1e-14 |

Estimates shift by ~1e-6 relative, toward the reference. Beyond parity: the
overlap weights' exact-balance property (Li, Morgan & Zaslavsky 2018) is
derived at the *unpenalised* score equations, so regularisation broke the
guarantee the docstring cites.

<a id="weighted-ks-exact"></a>

## 1.22.0 — ⚠️ weighted KS balance statistics increase

**What changed.** The weighted Kolmogorov-Smirnov statistic was computed by
interpolating the cumulative weights linearly between order statistics. An
empirical CDF is a **step** function — `F(v) = Σ wᵢ·1[xᵢ ≤ v] / Σ w` — so
the interpolant cuts the corner at every jump and reports a smaller maximum
gap than exists. It is now evaluated exactly.

**Which numbers.** `ks_stat` in `sp.ps_balance(...).table` and
`ks_stat_weighted` in `sp.balance_diagnostics(...).table`. Both **increase**.

| | old (interpolated) | new (exact step ECDF) |
| --- | ---: | ---: |
| typical n = 30 sample | 0.0919 | **0.1492** |
| worst understatement, n = 30 | ~0.06 absolute | — |
| worst understatement, n = 400 | ~0.004 absolute | — |
| equal weights, deviation from `scipy.stats.ks_2samp` | up to 0.063 | **1.1e-16** |

The unweighted branch always delegated to `scipy.stats.ks_2samp` and is
unchanged; the point of the fix is that the two branches of one function
now compute the same statistic.

**Do my estimates change?** No. This is a balance *diagnostic*, not an
estimator — no treatment effect, standard error, weight, or matched set
moves. What changes is the reported degree of imbalance, and only upward:
a covariate you read as "KS = 0.09, fine" may now read 0.15. Re-read any
balance table published before 1.22 before quoting a KS threshold from it.
There is no flag to restore the interpolated value; it was not the defined
statistic.

Zero total weight in either arm now returns `nan` rather than dividing by
zero.

<a id="sbw-solver-status"></a>

## 1.22.0 — `sp.sbw(...).solver_status` holds the solver outcome

**What changed.** The field was assigned the *estimand* — the literal
`"att"` / `"atc"` / `"ate"` — while the real `scipy.optimize.minimize`
outcome was computed and thrown away. It now holds `"optimal"`, or
`"feasible-not-converged: <message>"` when SLSQP only got there via the
loosened-`ftol` retry. For `estimand='ate'`, which runs two solves, the
worse of the two is reported.

```python
res = sp.sbw(df, treat='d', covariates=['x1', 'x2'], y='y', estimand='att')

res.solver_status   # before 1.22: 'att'      — always, told you nothing
res.solver_status   # now:         'optimal'  — did the optimiser converge
res.estimand        # 'ATT'        (unchanged, and always carried this)
res.method          # 'SBW-ATT (variance)'
```

**Do my numbers change?** No — weights, estimates and balance are
untouched, and a returned solution was always feasible (`_solve_sbw` raises
if the balance constraints are violated). Only code that *read*
`solver_status` is affected: a test or pipeline asserting
`solver_status == estimand` will now fail, correctly. Read `result.estimand`
(or `result.method`) for the estimand.

<a id="matching-default-se"></a>

## 1.22.0 — ⚠️ `sp.match` default standard errors change

**What changed.** `se_method='auto'` resolved to `'ai'` — the simple
matched-pair standard error — for nearest-neighbour matching. It now
resolves to `'abadie_imbens'`.

**Point estimates are unchanged.** Only the standard error, and everything
derived from it (t, p, confidence interval), moves. Standard errors get
**larger**, by roughly 1/0.91 to 1/0.56 — i.e. between 10% and 79% wider.

**Why.** `'ai'` treats matched pairs as independent and ignores the extra
variance from reusing controls under matching with replacement. Measured
over 36 designs × 1000 replications (`benchmarks/matching_se_coverage.py`)
on a design whose ATT is known:

| `se_method` | SE / true sampling SD | coverage (nominal 0.95) |
| --- | :-: | :-: |
| `'ai'` (old default) | 0.56 – 0.91 | **0.71 – 0.92** |
| `'psmatch2'` | 1.50 – 1.69 | 0.994 – 1.000 |
| `'abadie_imbens'` (new default) | **0.95 – 1.04** | **0.905 – 0.956** |
| `'bootstrap'` | 0.95 – 1.23 | 0.933 – 1.000 |

The old default did not reach nominal coverage in **any** of the 36 cells;
at worst a nominal 95% interval covered 71% of the time. `'abadie_imbens'`
is the only option measured to be correctly sized.

**To reproduce a pre-1.22 number**, pass the old estimator explicitly:

```python
res = sp.match(df, y='y', treat='d', covariates=X, se_method='ai')
```

That now emits a `UserWarning` naming the coverage shortfall — the option
still works, it just no longer passes silently.

**If you have published a `sp.match` standard error** computed with the
default before 1.22, it was the `'ai'` estimator. The point estimate stands;
the interval was too narrow. Re-running with the new default gives the
correctly-sized interval.

Unaffected: `method='kernel'` / `'radius'` (already resolved to
`'psmatch2'`), `method='llr'` (resolves to `'bootstrap'`), and any call that
passed `se_method=` explicitly.

<a id="psm-did-weight-regimes"></a>

## 1.22.0 — ⚠️ `psm_did(weight=...)` regimes

**What changed.** `sp.psmatch2(...).psm_did()` handed the matching
`_weight` to `sp.feols`, which applies Stata **aweight** semantics
(`df_resid = n_rows - k`). But the option was named `'fweight'`, and both
the docstring and `docs/guides/psm_did.md` advertised the Stata line

```stata
reg y i.treat##i.post [fweight=_weight] if _support==1
```

whose residual degrees of freedom are `Σw - k`. The coefficient was
correct under either reading; the standard error was not the one the
documentation promised.

| | old `'fweight'` | new `'aweight'` (default) | new `'fweight'` |
| --- | ---: | ---: | ---: |
| DiD coefficient | 1.551163 | 1.551163 | 1.551163 |
| standard error | 0.250051 | **0.250051** | **0.214797** |
| residual df | 366 | 366 | 496 |
| Stata equivalent | `[aweight=]` | `[aweight=]` | `[fweight=]` |

**Does my number change?**

- **Default call — no.** The default moved from `'fweight'` to
  `'aweight'`, and those produce identical numbers. If you never passed
  `weight=`, nothing moves.
- **Explicit `weight='fweight'` — yes.** You now get the `fweight`
  degrees of freedom you were asking for. To keep the old numbers, pass
  `weight='aweight'` explicitly.

**Which should I use?** `'aweight'`, which is why it is the default. A
control matched three times is not three independent observations, so the
`fweight` degrees of freedom overstate the information in the matched
sample. Reach for `'fweight'` only to reproduce a specific Stata output.

`'fweight'` requires integer weights, exactly as Stata does. Matching with
`k > 1` neighbours splits weights into `1/k` shares, so `weight='fweight'`
raises there with a pointer back to `'aweight'`.

Pinned in `tests/reference_parity/test_psmdid_weight_parity.py` against
Stata 18 MP.

<a id="pstest-vs-balance"></a>

## 1.22.0 — `m.balance()` is not `pstest` (and never was)

`docs/guides/psm_did.md` stated that `m.balance()`'s `smd_weighted` column
was "exactly what Stata `pstest` reports". It is not. `pstest` keeps the
**unmatched** pooled standard deviation in the denominator of the
post-matching standardised bias; `balance()` uses the matched-sample SD.

On the reference fixture, covariate `x1` after matching:

| | value |
| --- | ---: |
| `pstest` `%bias` | 13.910 |
| `balance()` `smd_weighted × 100` | 14.727 |

Both conventions are defensible and neither number changed — only the
claim that they were the same. If you need Stata's table, use the new
`m.pstest()`, which reproduces it to 1e-14 per covariate.

---

<a id="rdrobust-bandwidth-rebuild"></a>

## 1.21.0 — ⚠️ `sp.rdrobust` numbers change

**What changed.** The CCT bandwidth selector and the bias-correction step
were both wrong. On `rdrobust`'s own `rdrobust_RDsenate` with default
settings `sp.rdrobust` reported **12.39**; R reports **7.41**.

| | old | new | R |
| --- | ---: | ---: | ---: |
| headline effect | 12.39 | **7.5065** | 7.5065 |
| bandwidth `h` (p=1, tri) | 4.633 | **17.7544** | 17.7544 |
| bandwidth `h` (p=2, tri) | 4.633 | **22.2563** | 22.2563 |
| bias bandwidth `b` | = `h` | **28.0281** | 28.0281 |

The old `h` was **identical for p=1 and p=2** because the rate exponent was
hard-coded to `1/5`, which is CCT's `1/(2p+3)` only at `p=1`.

**Effect.** Every `sp.rdrobust` / `sp.rdbwselect` number changes, and so do
the downstream diagnostics built on them (`rdbwsensitivity`, `rdbalance`,
`rdplacebo`, `rd_multi_extrapolate`). **Re-run anything whose numbers came
from these.** There is no flag restoring the old behaviour; it was not an
alternative bandwidth convention, it was the wrong formula.

**How to check an archived figure.** If you recorded the bandwidth, the old
`h` was roughly `n^{-1/5}`-scaled off the correct one and insensitive to `p`
— an `h` that does not move when you change `p` is the signature. The
conventional estimate at a *user-supplied* `h` was always correct, so
`sp.rdrobust(..., h=<your old h>)` reproduces the old point estimate.

**Behaviour changes beyond the numbers.**

| Before | After |
| --- | --- |
| `bwselect='msesum'` / `'cersum'` raised `ValueError` | Accepted; all six R variants work |
| `b` defaulted to `h` | `b` comes from the cascade when `h` is auto-selected; still `b = h` when you supply `h` yourself, matching R |

**Not fixed.** `covs=` is a silent no-op — see the Known issues section of
the CHANGELOG. If you have been passing covariates to `sp.rdrobust`, you
have been getting unadjusted estimates, and that is still true after this
release.

---

<a id="unified-sensitivity-scale"></a>

## 1.21.0 — ⚠️ `unified_sensitivity`: E-value scale and Oster inputs

Two quantities in the dashboard were computed from the wrong inputs.

**E-value from an un-standardised coefficient.** The E-value is defined on
the risk-ratio scale. `unified_sensitivity` forced `measure="RR"` and
passed a raw regression coefficient through unchanged whenever it was
positive, so a $1,548 treatment effect was read as a risk ratio of 1548:

```text
before:  RR "1548.24"  ->  E-value 3095.99      (meaningless)
after:   d = 0.2072, RR = exp(0.91*d) = 1.2075  ->  E-value 1.7082
```

A mean difference must be standardised by the outcome SD first
(`vanderweele2017sensitivity`). Migration:

```python
# supply the scale — data=/y= is usually already there for Sensemakr
sp.unified_sensitivity(fit, term="treat", data=df, y="re78", controls=X)

# or give the SD directly
sp.unified_sensitivity(fit, term="treat", outcome_sd=df["re78"].std(ddof=1))

# or declare that the estimate really is a ratio
sp.unified_sensitivity(hazard_fit, term="treat", measure="RR")
```

Without a scale the E-value is now `nan` with an explanatory note instead
of a fabricated number. If you published an E-value from a linear model,
recompute it — the old one described a risk ratio you never estimated.

**Oster inputs.** `r2_treated` / `r2_controlled` read like sensemakr's
partial R^2 but were consumed as the short- and long-regression R^2. Given
sensemakr-style values they produced `delta* = -12.765` where
`sp.oster_delta` reported `-2.339` for the same specification — two
contradictory deltas in one report. They are renamed `r2_short` /
`r2_long` (old names still work, with a `DeprecationWarning`), and when
`data`, `y`, `treat` and `controls` are available the R^2 are derived from
the data so the two paths agree by construction.

---

<a id="continuous-did-cgs"></a>

## 1.21.0 — `sp.continuous_did(method="cgs")` is superseded

**What changed.** `method="cgs"` now emits a `DeprecationWarning` and will be
removed after one minor release. Use `sp.cgs_continuous_did` instead.

**Why.** That mode was an MVP standing in for an estimator StatsPAI did not
have: outcome regression only, a bootstrap standard error, and formula
details left as `[待核验]` in `docs/rfc/continuous_did_cgs.md`. The
replacement is the actual Callaway-Goodman-Bacon-Sant'Anna estimator, with
`ATT(d)` and `ACRT(d)` from a B-spline in the dose and an
influence-function variance — and it is pinned against the authors' own
`contdid` package: both curves at four grid points and both overall
quantities, across three spline specifications, agree to 1e-12.

**What to do.**

```python
# Before
sp.continuous_did(df, y="y", dose="d", time="t", id="i", method="cgs")

# After
sp.cgs_continuous_did(df, y="y", dose="d", time="t", unit="i", cohort="g",
                      degree=3, num_knots=0)
```

The new function needs a `cohort` column (the first-treatment period, 0 for
never-treated) rather than inferring a single pre/post split, which is what
lets it handle staggered adoption at all.

The other `continuous_did` modes — `twfe`, `att_gt`, `dose_response` — are
unchanged. They are dose-bin and local-linear heuristics, useful for a quick
look, and the docstring says so.

---

<a id="multiplegt-switch-directions"></a>

## 1.21.0 — ⚠️ `sp.did_multiplegt` dynamics/placebos, and switch-off in `_dyn`

**What changed.** Four numbers move, all on non-trivial designs:

| Function | What | Why |
| --- | --- | --- |
| `sp.did_multiplegt` | dynamic effect at horizon ≥ 1 | switchers who switch again inside the window are now excluded |
| `sp.did_multiplegt` | placebo value | the pre-window stability condition is applied |
| `sp.did_multiplegt_dyn` | everything, on non-absorbing panels | switch-off events are no longer dropped |

Nothing else moves. In particular the placebo's **sign is unchanged** — see
below.

**Why it went unnoticed.** All four were invisible without a working
reference, and the reference looked broken: `DIDmultiplegt` 2.x returns `NaN`
from `mode="old"` on its own bundled example. The archived **0.1.4** works,
and against it the static DID_M effect was already bit-exact while the
dynamic and placebo paths were not.

**On the placebo's sign: dCDH's own implementations disagree, and StatsPAI
now says so instead of choosing.** On `did::mpdta` the Stata and R packages
return the same three effects to six decimals and the same
`|placebo_1| = 0.024269` — with opposite signs. The new `placebo_sign`
parameter selects between them and **defaults to the Stata convention this
function has always used**, so nothing you have reported changes.

```python
sp.did_multiplegt(df, ..., placebo=1)                      # Stata sign (default)
sp.did_multiplegt(df, ..., placebo=1, placebo_sign="r")    # DIDmultiplegt sign
```

If you compare StatsPAI output against an R script, pass `placebo_sign="r"`
or the placebos will look like they disagree when only the convention does.

**What to do.** For the two rows above: nothing at the call site; re-run and
re-read. Absorbing
panels are unaffected by the `_dyn` change — with a binary treatment every
control already shares the baseline of zero, and that is verified rather than
assumed.

```python
# Both now match the reference; the counts are worth looking at too.
res = sp.did_multiplegt(df, y="y", group="i", time="t", treatment="d",
                        placebo=1, dynamic=1)
res.model_info["event_study"]      # placebo, effect, dynamic on one scale
```

---

<a id="multiplegt-dyn-placebo"></a>

## 1.21.0 — ⚠️ `sp.did_multiplegt_dyn` placebos are now the estimator's placebos

**What changed.** The placebo at lag ℓ was computed as
`Y_{F-1-ℓ} − Y_{F-1-ℓ-1}` — a one-period difference sliding backwards
through the pre-period. de Chaisemartin & D'Haultfœuille's placebo is the
effect window reflected about `F-1`: `Y_{F-1-ℓ} − Y_{F-1}`, a long difference
the same length as the effect it mirrors, reported with the reverse sign so
it sits on the same event-study scale. Every placebo value changes. Effects
are unaffected.

**Why.** Two reasons, and the second is the one that bites:

1. It is a different quantity. A one-period difference at lag ℓ does not
   mirror the ℓ-period effect and does not test what the paper's placebo
   tests.
2. It needed one more pre-period, so it silently used fewer cohorts. On the
   parity fixture, lag 1 ran on 96 switchers where `DIDmultiplegtDYN` uses
   146 — the earliest cohort was dropped without a word.

Since the placebos feed `model_info["joint_placebo_test"]`, the module's
parallel-trends diagnostic was testing the wrong contrast on the wrong
subsample. If you reported a passed placebo test from this function, re-run
it.

**What to do.** Nothing at the call site; re-run and re-read the placebos.
They now match `DIDmultiplegtDYN` 2.3.4 to 5e-15, switcher counts included
(`tests/reference_parity/test_multiplegt_dyn_parity.py`, Track A module
`78_multiplegt_dyn`).

**Related, not a break.** `aggregation="switchers"` is new and reproduces
the R package's `Av_tot_eff`; the default stays on the equal-weight average
over horizons, so the headline number is unchanged.

```python
sp.did_multiplegt_dyn(df, y="y", group="i", time="t", treatment="d",
                      dynamic=3, aggregation="switchers")
```

---

<a id="pretrends-power-test"></a>

## 1.21.0 — ⚠️ `sp.pretrends_power` defaults to the pre-test Roth (2022) analyses

**What changed.** `sp.pretrends_power(result)` returned the power of the
*joint Wald* test that all pre-period coefficients are zero. It now returns
the power of the coefficient-by-coefficient pre-test: reject if any
pre-period coefficient is individually significant at `alpha`. That is the
practice Roth (2022) analyses, and the quantity his `pretrends` R package
reports — the paper the docstring has always cited.

**Why.** The two answer different questions and are not close. On the
reference fixture at a linear violation of slope 0.02:

| | power |
| --- | --- |
| coefficient-by-coefficient (new default) | 0.332 |
| joint Wald (old default) | 0.157 |

They are not even on the same footing: the joint test has size exactly
`alpha`, while the eyeball test rejects above `alpha` under the null because
each of the K coefficients gets its own `alpha`-level look. Reporting the
Wald number under Roth's name understated how often a real trend would have
been spotted, which is the opposite of the paper's message.

**What to do.**

```python
# Previous behaviour, explicitly:
sp.pretrends_power(res, test="joint")["power"]

# Or read it off the new default call — both are always returned:
out = sp.pretrends_power(res)
out["power"]        # coefficient-by-coefficient
out["power_joint"]  # joint Wald, unchanged from before
```

No key was removed: `noncentrality` and `critical_value` are still reported
under both settings. New keys: `power_under_null`, `bayes_factor`,
`likelihood_ratio`, `test`, `threshold_tstat`, `power_joint`.

**New in the same release.** `sp.pretrends_slope_for_power(result,
target_power=0.5)` inverts the calculation — the slope of a linear pre-trend
the pre-test would catch half the time. It is the number to quote when a
reader asks what a passed pre-test actually rules out, and mirrors
`pretrends::slope_for_power`.

Both are pinned against `pretrends` 0.1.0 in
`tests/reference_parity/test_pretrends_power_parity.py` and Track A module
`76_pretrends`.

---

<a id="unified-sensitivity-term"></a>

## 1.21.0 — ⚠️ `sp.unified_sensitivity` analysed the intercept

**What changed.** `sp.unified_sensitivity(result)` pulled the coefficient to
analyse with `params.iloc[0]` and its standard error with
`std_errors.iloc[0]`. For a formula regression those are the **intercept**,
not the treatment. On the LaLonde baseline:

```text
what it analysed:  66.51    (Intercept)
what you meant:  1548.24    (treat)
```

The standard error came from the intercept too, so the entire dashboard —
E-value, breakdown point, Rosenbaum bounds — described a parameter nobody
asked about.

**Why you may not have noticed.** It usually raised instead of answering,
but for an unrelated reason: the intercept's CI spanned zero, the
risk-ratio conversion mapped `(-4892, 5025)` to `(4893, 5026)` via
`1 + |limit|`, that interval excludes the converted point estimate `67.5`,
and an assertion inside `evalue` fired with "Point estimate should lie
inside the CI." Designs whose intercept CI stays positive skipped that
tripwire and got a confident wrong number.

**Migration.** Name the coefficient:

```python
# before — silently analysed the Intercept
sp.unified_sensitivity(ols_fit)

# after — explicit, and the only form that still works for a multi-term fit
sp.unified_sensitivity(ols_fit, term="treat")
```

If you already pass `treat=` (it names the treatment for the Sensemakr
component), that doubles as the term — no need to name the same column
twice:

```python
sp.unified_sensitivity(fit, data=df, y="re78", treat="treat", controls=X)
```

With more than one non-intercept coefficient and neither `term=` nor
`treat=`, the function now raises `MethodIncompatibility` listing the
candidates rather than guessing. A fit
with exactly one non-intercept coefficient still needs no `term=`, and
results exposing a scalar `.estimate` / `.ate` (`CausalResult` and friends)
are unaffected — they never went through the coefficient path.

If you published a robustness claim produced by the old code path on a
multi-term regression, re-run it with `term=` — the previous output did not
describe your treatment effect.

**Related.** `sp.sensitivity_dashboard` does *not* share this defect: it
already skipped intercept-like names. It did, however, return an empty
dashboard graded `overall_stability='?'` when called without `data=`,
because most of its dimensions re-estimate on perturbed samples. That case
now emits a `RuntimeWarning` instead of looking like a pass.

---

<a id="aipw-default-seed"></a>

## 1.21.0 — ⚠️ `sp.aipw` was not reproducible; default `seed` is now 42

**What changed.** `sp.aipw(..., seed=...)` defaulted to `None`, which reached
`np.random.default_rng(None)` and therefore seeded the cross-fitting fold
split from OS entropy. The default is now `42`.

**Why it mattered.** Three identical calls on the same 614-row LaLonde frame:

```text
+308.87    +149.84    +905.21
```

That spread is wider than the treatment effect being estimated, so which
number reached your paper depended on when you happened to run the script.
It could not be pinned from the outside either — `np.random.default_rng`
does not consult the legacy global RNG, so `np.random.seed(7)` before every
call changed nothing.

It propagated one level up: `sp.causal_question(...).estimate()` resolves a
selection-on-observables plan to cross-fitted AIPW, so the headline estimate
of a whole estimand-first pipeline moved between runs.

**Migration.**

```python
# Reproducible (the new default) — nothing to do:
sp.aipw(df, y="re78", treat="treat", covariates=X)

# Old behaviour: a fresh random fold split on every call.
sp.aipw(df, y="re78", treat="treat", covariates=X, seed=None)
```

If you published a number produced by the old default, you cannot reproduce
it by re-running — the fold split that generated it is gone. Re-estimate
with the new default and report that number instead.

**Scope.** An audit of the stochastic surface found `sp.dml`, `sp.tmle` and
`sp.metalearner` already deterministic by default; `sp.tmle`, `sp.bcf` and
`sp.super_learner` already defaulted to `42`. `sp.aipw` was the only
offender. The seed actually used is now recorded in
`result.model_info['seed']`, and `tests/test_estimator_determinism.py` pins
the convention for the whole family.

---

<a id="nsw-lalonde-default-simulated-false"></a>

## 1.21.0 — bundled datasets now default to the real published data

**What changed.** Five loaders ship a real extract in
`statspai/datasets/data/`. Four of them also offer a calibrated replica
behind `simulated=`, and their defaults now all point at the real data:
`card_1995`, `lee_2008_senate`, `california_prop99` and `nsw_lalonde`.

The rule is now uniform: **if StatsPAI ships the real published data, a
bare call returns it.** Previously only `nsw_lalonde` behaved that way, so
what you got depended on which dataset you reached for.

Why the real extract is the better default: it is what reproduces the
papers. `card_1995` returns OLS 0.074 / IV 0.132 against Table 2's
0.075 / 0.132 — the replica gives 0.110 / 0.142.

`nsw_lalonde` in detail:

| | old default (`simulated=True`) | new default (`simulated=False`) |
| --- | --- | --- |
| shape | `(445, 10)` | `(614, 11)` |
| columns | no `hispanic` | adds `black`, `hispanic` |
| naive OLS ATT | ≈ **+$1,794** | ≈ **−$635** |
| `df.attrs['data_source']` | `'simulated'` | `'real'` |

**Why.** The old default was a quiet correctness trap. `sp.datasets`'s own
first example is a bare `nsw_lalonde()`, so a reader following the docs got
simulated numbers that match no published table while believing they were
looking at LaLonde's data. Defaulting to the real extract makes the honest
path the default one; the replica stays available and is still the right
choice when you want the *experimental* subset.

**Migration.**

```python
# The real extract is now the default for all five bundled datasets:
df = sp.datasets.card_1995()          # (3010, 9) real NLSYM
df = sp.datasets.lee_2008_senate()    # (1390, 2) rdrobust RDsenate
df = sp.datasets.california_prop99()  # (1209, 8) ADH smoking panel
df = sp.datasets.nsw_lalonde()        # (614, 11) MatchIt::lalonde

# The calibrated replicas remain available:
df = sp.datasets.nsw_lalonde(simulated=True)
```

**Shapes change where the two variants differ.** `california_prop99` keeps
its columns (order only) and `card_1995` gains `nearc2`, so those are
near-transparent. `lee_2008_senate` differs materially — the real
`rdrobust::rdrobust_RDsenate` extract is 1,390 rows with columns `x`
(lagged Democratic margin, percent points) and `y` (current vote share),
against the replica's 6,558 rows of `margin` / `voteshare_next` on a 0-1
scale. The two are genuinely different frames on different scales; they
are deliberately *not* forced into a shared vocabulary, because giving
them the same column names on different units would be a worse trap than
the shape change. Pass `simulated=True` to keep the replica.

The signature is a plain `simulated: bool = False` — no sentinel, no
warning to silence. If you relied on the old default, pass
`simulated=True`.

`sp.datasets.list_datasets()` now carries a `source` column saying which
variant each bare `name()` call returns, so you can see at a glance
whether you are getting a real extract or a calibrated replica.

**Offline note.** The real extract is a CSV bundled in the wheel under
`statspai/datasets/data/`, so this default needs no network. Verified by
loading it in a clean venv built from the wheel with `socket` hard-blocked.

---

<a id="ltmle-influence-curve-martingale-term"></a>

## 1.21.0 — ⚠️ `sp.ltmle` standard errors were 250–400× too small

**What changed.** The efficient influence curve for LTMLE is

    sum_k H_k (Q*_{k+1} - Q*_k)  +  (Q*_1 - psi)

Only the second term was being computed. What the function reported as a
standard error was therefore the dispersion of a fitted conditional mean, not
the sampling variability of the estimator. The martingale sum is now
accumulated across time points.

**Effect.** Point estimates are **unchanged**. Standard errors, confidence
intervals, p-values and any significance marks in `.summary()` all change, and
the old ones were not usable: on a two-period DGP with a known ATE the
reported SE was 0.00024 where the estimator's actual Monte-Carlo standard
deviation was 0.059 (n = 500). The discrepancy grew with sample size — 353× at
n = 2000, 405× at n = 8000 — because the reported quantity was not converging
at the √n rate at all.

**What to do.** Re-run anything that used `sp.ltmle` for inference. Any
conclusion that rested on an `sp.ltmle` confidence interval or p-value should
be treated as unsupported until recomputed; intervals will be roughly two
orders of magnitude wider.

**Remaining limitation, now quantified.** The module targets with a one-step
fluctuation rather than iterating to convergence, so the martingale sum is
near zero but not identically zero and the SE stays mildly anti-conservative.
Over 200 replications it runs ~13% below the Monte-Carlo standard deviation at
n = 1000 (nominal-95% coverage 0.905) and ~7% below at n = 4000 (coverage
0.930). This is stated with those numbers in the function's Notes. For
inference needing honest coverage with flexible ML nuisances, a full
CV-LTMLE / iterated-targeting path is still the right tool and is tracked as a
follow-up.

---

<a id="gmm-unadjusted-variance-and-conventions"></a>

## 1.21.0 — ⚠️ `sp.gmm` variance, closed form, and HAC conventions

**What changed.** Three things in `sp.gmm`.

1. `se='unadjusted'` returned `(D'WD)⁻¹/n` for every weighting matrix. That
   formula is the estimator's variance only at the efficient weight `W = S⁻¹`;
   under any other weight it is smaller than the truth. It now warns.
   `se='robust'` — the default — was already the sandwich and is unchanged.
2. Moment conditions affine in θ were minimised with BFGS. They have a closed
   form, which is now used; `diagnostics['n_iter'] == 0` reports it.
3. The Bartlett HAC kernel is now evaluated at `lag/bandwidth` (vanishing at
   `lag == bandwidth`), matching R `sandwich`.

**Effect.** `se='unadjusted'` results are unchanged numerically but now carry
a warning wherever they were wrong. Point estimates for linear moment
conditions change in the last digits — they are now the exact minimiser
rather than BFGS's approximation, agreeing with R's analytic two-step to
1e-12. HAC standard errors change by percent-level amounts if you were
relying on the previous bandwidth convention.

**What to do.** If you reported `se='unadjusted'` standard errors from a
one-step fit or with an explicit `W=`, re-run: those numbers were too small.
Switch to `se='robust'`, or use a weight-updating method (`'twostep'`,
`'iterative'`, `'cue'`) so the efficient formula actually applies.

**New parameters.** `jacobian=` (analytic `D(theta)`), `vcov=` (`'mds'`,
`'iid'`, `'hac'`, `'cluster'`), `cluster=`, `hac_bandwidth=`, and `center=` —
moment centring, which R `gmm` does by default and Stata does not. `center`
defaults to Stata's convention, so existing results are unchanged.

---

<a id="xtabond-twostep-ar-test"></a>

## 1.21.0 - `sp.xtabond` two-step AR(1)/AR(2) statistics changed

**What changed.** The Arellano-Bond serial-correlation test variance
decomposes into three terms, the last of which is
`(W'q)' Avar(beta) (W'q)`. StatsPAI always evaluated it at the uncorrected
robust sandwich. When `twostep=True` the *reported* VCE is either the
Windmeijer-corrected one (`robust=True`) or the conventional
`(W'ZA2Z'W)^-1` (`robust=False`), so the test was using a variance the
coefficient table did not.

**Who is affected.** Only `twostep=True` fits, and only the `ar1_z` /
`ar1_p` / `ar2_z` / `ar2_p` fields - coefficients and standard errors are
untouched. One-step fits are bit-identical: there the reported and naive
VCEs are the same matrix and the correction is exactly zero.

**How large was the error.** On Stata's `abdata`:

| spec | old AR(1) z | new | Stata |
| --- | --- | --- | --- |
| `lags=1, twostep=True` | -2.2438 | -2.1000 | -2.1000 |
| AB(1991) Table 4, `twostep=True` | -4.3229 | -3.1030 | -3.1030 |

**What to do.** Re-read any AR(2) conclusion drawn from a two-step fit. The
direction of the change is not uniform - it can move the statistic either
way - so a previously "passing" AR(2) test is not automatically safe.

---

<a id="gmm-unadjusted-se"></a>

## 1.21.0 - `sp.gmm(se='unadjusted')` now reports the efficient-GMM variance

**What changed.** `se='unadjusted'` used to return `(D'WD)^-1/n` for
whatever weight matrix `W` was in force. That expression is the variance of
the GMM estimator **only when `W` is efficient** (`W = S^-1`); with any
other weight the estimator's variance is the sandwich
`(D'WD)^-1 D'W S W D (D'WD)^-1 / n`, which is generally larger. The
reported standard errors were therefore too small in exactly the case a
user reaches for a custom `W`.

It now returns `(D' S^-1 D)^-1/n` - the efficiency bound - and warns when
the weight actually used is not efficient, pointing at `se='robust'`.

**Who is affected.** Only calls that combined `se='unadjusted'` with a
non-efficient weight: `method='onestep'` with an explicit `W=`, or the
identity default. Two-step, iterated and CUE fits are **unchanged**,
because at the efficient weight `(D'WD)^-1` and `(D'S^-1 D)^-1` are the
same matrix.

**What to do.** Nothing if you used the default `se='robust'`. If you
relied on `se='unadjusted'` with a custom weight, switch to `se='robust'`:
that is the variance of the estimator you actually computed.

---

<a id="xtabond-listwise-deletion"></a>

## 1.21.0 — `sp.xtabond` no longer deletes instruments on covariate `NaN`s

**What changed.** `sp.xtabond` used to start with

```python
df = data[[id, time, y] + x].dropna()
```

Listwise deletion across *all* columns means a missing value in any covariate
at period *t* removes that row entirely — and with it `y_{i,t}` as a GMM
instrument and as a lag source, not merely as an estimation observation.
Availability is now evaluated **per variable**: a covariate that is
unobserved early costs only the equations that need it.

**Who is affected.** Only fits where some covariate had missing values in the
estimation window — most commonly because the user built lagged regressors by
hand, which necessarily leaves leading `NaN`s. If every covariate was complete
over the periods used, nothing changes; `tests/test_xtabond_golden.py` locks
that.

**How large was the error.** On Stata's `abdata` panel with the Arellano-Bond
(1991) Table 4 specification (`n` on two lags of `n`, `l(0/1).w`, `l(0/2).k`):

| | old `sp.xtabond` | new | Stata `xtabond` |
| --- | --- | --- | --- |
| observations | 331 | 611 | 611 |
| instruments | 19 | 32 | 32 |
| ρ̂₁ | 0.660 | 0.849 | 0.849 |

**What to do.** Nothing, unless you have published numbers from an affected
fit; re-run those. Hand-built lag columns are no longer necessary either —
`x=["l(0/1).w", "l(0/2).k"]` is now accepted directly.

---

<a id="qte-firpo-mislabel"></a>

## 1.21.0 — `sp.qte` method names and default

**What changed.** `sp.qte(method='quantile_regression')` was documented,
labelled and registered as Firpo (2007). It is not. It returns the
coefficient on `D` in a quantile regression of `Y` on `D + controls` — a
**conditional** QTE, which absent rank invariance is not a treatment effect
on any quantile of the outcome distribution. Firpo (2007) is the
**unconditional** estimator, which reweights by the propensity score and
compares the marginal quantiles of `Y(1)` and `Y(0)`.

| Old | New | Numbers |
| --- | --- | --- |
| `method='quantile_regression'` | `method='conditional_qr'` | **unchanged** |
| — (did not exist) | `method='firpo_qte'` | new — the actual Firpo QTE |
| — (did not exist) | `method='firpo_qtt'` | new — Firpo QTT |
| `method='distribution'` | `method='distribution'` | **unchanged**, but now labelled QTT rather than QTE, which is what it always computed |

**The default changed** from `'quantile_regression'` to `'firpo_qte'`. A
call that relied on the default now returns a different estimand. Pass
`method='conditional_qr'` explicitly to keep the old numbers.

`method='quantile_regression'` still works and emits a
`DeprecationWarning`; it is removed in 1.23.0.

**Which should you use?** If you want "the effect on the median worker",
that is the unconditional `'firpo_qte'` (or `'firpo_qtt'` for the effect on
treated units). `'conditional_qr'` answers "holding covariates fixed, how
does the τ-th conditional quantile shift" — a within-cell statement that
does not aggregate to a distributional effect.

**`sp.qdid` reference correction.** `sp.qdid` was described as Athey &
Imbens (2006) changes-in-changes in its docstring, its method label and the
registry. It implements **QDiD** — the DiD contrast applied to quantiles —
which is the estimator Athey & Imbens propose CiC *in place of*, and
criticise directly. **No numbers change**; only the attribution. For
changes-in-changes use `sp.cic`.

---

<a id="dist-iv-quantile-wald-ratio"></a>

## 1.21.0 — ⚠️ `sp.dist_iv` estimated the wrong object

**What changed.** `sp.dist_iv` (and its alias `sp.kan_dlate`) computed a
*Wald ratio of quantiles*:

```text
LATE_q(τ) = [Q(τ | Z=1) − Q(τ | Z=0)] / [E(D | Z=1) − E(D | Z=0)]
```

The quantile operator is not linear, so the mean-Wald rescaling that makes
the ordinary LATE work does not carry over. That expression is inconsistent
for any quantile estimand — it is not a noisier version of the complier QTE,
it converges to something else. It now uses Abadie (2002, 2003) κ-weighted
complier CDFs and returns

```text
QTE_c(τ) = F⁻¹_{Y(1)|complier}(τ) − F⁻¹_{Y(0)|complier}(τ)
```

**Effect.** Every `sp.dist_iv` / `sp.kan_dlate` number changes. The old bias
was multiplicative in the first stage: on a design with a true complier
`QTE(τ) ≡ 2.0` and `Δp = 0.5`, the old code returned ≈ 4.0 at n = 200,000.

**Approximate back-conversion.** The old estimator was roughly

```text
old(τ)  ≈  [Q(τ|Z=1) − Q(τ|Z=0)] / Δp
```

so when the treated and control quantile curves are near-parallel you can
sanity-check an archived figure with `new(τ) · Δp ≈ Q(τ|Z=1) − Q(τ|Z=0)`,
i.e. **`old(τ) ≈ new(τ) / Δp`** only in the special case of a homogeneous
shift among compliers with no always-takers. With always-takers present
there is no exact conversion — the old quantiles mixed compliers,
always-takers and never-takers in proportions that depend on τ. **Re-run
the estimation.** There is no flag restoring the old behaviour; it was not
an alternative convention.

**Other behaviour changes in the same release.**

| Before | After |
| --- | --- |
| `covariates=` accepted, then silently discarded | Selects Frölich & Melly (2013) unconditional IV-QTE weighting; changes the estimate |
| Constant instrument → all-`NaN` result object + warning | Raises `ValueError` |
| Near-zero first stage → silent estimate | `UserWarning` naming the complier share and first-stage *t* |
| Bootstrap SE only | Analytic influence-function SE by default (`se='auto'`); bootstrap when covariates are supplied |

**`sp.kan_dlate` is deprecated** (removal in 1.23.0). It was always a pure
alias for `sp.dist_iv` and never implemented a Kolmogorov-Arnold bridge
function. Its docstrings also attributed arXiv:2506.12765 to two different
authors; verification against arXiv and the DataCite DOI registry shows the
paper is *Model Risk in Machine-Learning Distributional IV Estimation* by
**Charles Shaw** alone, and neither its title nor its v1 abstract mentions a
KAN. Call `sp.dist_iv` directly.

---

<a id="genmatch-variance-basis"></a>

## 1.21.0 — ⚠️ `sp.genmatch` distance uses full-sample variances

**What changed.** The genetic-matching kernel computed its generalised
distance after standardising covariates by the **control group's**
variances. Its own module docstring specified `D' S^(-1/2) W S^(-1/2) D`,
and the metric `Matching::Match(Weight = 3, Weight.matrix = W)` implements
is the diagonal of the **full-sample** variances. The kernel now uses
those.

**Effect.** Genetic-matching weights, matched pairs and ATT all change.
Given a fixed diagonal `W`, the kernel now reproduces `Matching::Match`'s
assignment on every uniquely matched treated unit of `MatchIt::lalonde`
(163/163).

**What to do.** Re-run any `sp.genmatch` analysis. Note the genetic
*search* is stochastic and was never reproducible across languages or
seeds; only the deterministic distance-and-assignment kernel is pinned, in
`tests/reference_parity/test_matching_r_parity.py`.

---

<a id="sbw-tolerance-scale"></a>

## 1.21.0 — `sp.sbw` balance tolerance now names its units

**What changed.** `delta` was always interpreted against the full-sample
standard deviation. `sbw::sbw` quotes the same tolerance against either
the target group (`bal_std="target"`, treated units under ATT) or the
group being reweighted (`bal_std="group"`, controls), so a tolerance alone
did not determine the estimator. `tolerance_scale` now selects among
`'sd'` (the previous behaviour, still the default), `'target'`, `'group'`
and `'raw'`.

**Effect.** None by default. But the conventions are not interchangeable:
on `MatchIt::lalonde` at `delta = 0.05` the ATT is 1330.30 under
`'target'`, 1335.87 under `'sd'` and 1342.89 under `'group'`.

**What to do.** When reconciling with `sbw::sbw`, set `tolerance_scale` to
match its `bal_std`. When reporting a tolerance, report the scale too.

---

<a id="match-with-replacement-ties"></a>

## 1.21.0 — `sp.match` can now pool tied controls

**What changed.** Under matching *with replacement*, `sp.match` kept only
the lowest-index control among equidistant candidates. `ties='all'` pools
them and splits the weight, and `tie_tolerance` sets how close squared
distances must be to count as tied.

**Effect.** None by default (`ties='first'`). With
`ties='all', tie_tolerance=1e-5` the ATT and the Abadie-Imbens population
standard error match `Matching::Match` exactly.

**What to do.** Nothing is required. Use `ties='all'` if you would rather
not have row order decide which of several equally good controls is used,
and add `tie_tolerance=1e-5` when reconciling with `Matching::Match`.

---

<a id="cbps-solver-rewrite"></a>

## 1.21.0 — ⚠️ `sp.cbps` now solves the Imai-Ratkovic problem

**What changed.** The CBPS GMM was posed in the raw covariate basis with an
empirical outer-product weighting matrix. CBPS is defined in a standardised,
orthonormalised basis with the *model-implied* moment covariance frozen at the
starting value. Neither the just-identified quadratic form nor the GMM
weighting is invariant to that change of basis, so the old code minimised a
different objective and returned a different estimator — not a less precise
version of the same one.

**Effect.** Every `sp.cbps` estimate changes. On `MatchIt::lalonde` the
old `estimand='ATE', variant='over'` result was **8.6x** off `CBPS::CBPS`
(1585.99 vs 165.88); `ATT`/`over` was 25% off; coefficients differed by up
to 170%. After the rewrite, ATE (both variants) and ATT/`exact` agree with
R to ≤5e-3 relative.

**What to do.** Re-run any analysis whose numbers came from `sp.cbps`. If
you need to reproduce an old figure, there is no flag for the previous
behaviour — it was not an alternative convention, it was the wrong problem.
For a sanity check on the new results, `variant='exact'` must now balance
covariates to |SMD| < 1e-6; that identity holds only for the corrected
solver.

**Note on `estimand='ATT', variant='over'`.** StatsPAI deliberately does
*not* reproduce `CBPS::CBPS` here. CBPS's analytic ATT gradient divides the
balance block by `n_1` where the moment's Jacobian carries `1/n`,
overstating it by `n/n_1`, and its `optim` call stops at a non-stationary
point as a result. StatsPAI uses the correct Jacobian and attains both a
lower GMM objective and better covariate balance (max |SMD| 0.037 vs 0.106
on lalonde). This is asserted in
`tests/reference_parity/test_matching_r_parity.py`.

---

<a id="ebalance-exact-balance"></a>

## 1.21.0 — ⚠️ `sp.ebalance` now achieves exact moment balance

**What changed.** The entropy-balancing dual was minimised with L-BFGS-B on
unscaled constraints. When covariates live on different scales the dual
Hessian is badly conditioned and the optimiser stops early, so the weights
did not match the targeted moments — which is the one property entropy
balancing is defined by. The convergence check could not catch it either:
it compared an *absolute* moment gap against 0.01, which is meaningless
when one constraint is a 0/1 indicator and the next is annual earnings in
dollars.

**Effect.** On `MatchIt::lalonde` the reweighted control mean of `re74` was
2.66 away from the treated mean (1.3e-3 relative); it is now 1e-15
relative. ATT estimates move in the 3rd significant figure (1269.45 →
1273.26, against `ebal::ebalance`'s 1273.26).

**What to do.** Re-run affected analyses. The convergence warning now fires
on a standardised gap above 1e-6, so a result that previously passed
silently may now warn — that warning is correct and means the treated
moments are likely outside the convex hull of the control moments.
`model_info['max_standardized_moment_gap']` records the achieved gap.

---

<a id="tmle-shared-nuisance-and-fluctuation"></a>

## 1.21.0 — `sp.tmle` gains `Q` / `g1W` / `fluctuation`

**What changed.** Three new parameters. `Q` takes an `(n, 2)` matrix of
`[Q(0,W), Q(1,W)]` and `g1W` a propensity vector; supplying either bypasses
the corresponding Super Learner stage, so the targeting step can run on
externally-estimated nuisances. `fluctuation` selects the submodel used in
the targeting step.

**Effect on existing code: none.** The default `fluctuation='single'` is the
one-clever-covariate submodel `H(A,W) = A/g - (1-A)/(1-g)` StatsPAI has
always used, and its numbers are unchanged. `model_info['epsilon']` keeps its
scalar type on that path.

**What is new.** `fluctuation='per_arm'` fits two clever covariates, `A/g`
and `-(1-A)/(1-g)`, jointly — the submodel the R `tmle` package uses. Both
are valid TMLEs solving the efficient-influence-function equation and are
asymptotically equivalent, but they differ at finite *n*: about 1.3e-3
relatively on the Track A module-72 fixture. Use `'per_arm'` when
reconciling against `tmle::tmle`, which it reproduces to ~1e-9 on a shared
initial fit.

`model_info` gains `epsilon_vec` (the full fluctuation vector in both
modes), `fluctuation`, and `nuisance_source`. Under `'per_arm'` the scalar
`model_info['epsilon']` is `None`, because no scalar fluctuation parameter
exists there; read `epsilon_vec` instead. `sl_outcome_weights` /
`sl_propensity_weights` are `None` when the corresponding nuisance was
supplied — there is no ensemble to report weights for.

**What to do.** Nothing. If you compare StatsPAI against `tmle::tmle`, pass
`fluctuation='per_arm'` and supply the same `Q` / `g1W` to both.

---

<a id="dml-panel-learner-aliases"></a>

## 1.21.0 — `sp.dml_panel` accepts `sp.dml`'s learner aliases

**What changed.** `sp.dml(ml_g='linear')` resolved short learner names;
`sp.dml_panel(ml_g='linear')` did not, and failed inside scikit-learn's
`clone()` with `TypeError: Cannot clone object ''linear''`, a message that
named neither the offending parameter nor the accepted values. Both now
route through the same `resolve_learner`.

**Effect.** Strictly additive — calls that passed estimator instances are
unaffected, and calls that passed strings previously raised.

**What to do.** Nothing.

---

<a id="dml-sensitivity-structural-residual"></a>

## 1.21.0 — ⚠️ `sp.dml_sensitivity` uses the structural outcome residual

**What changed.** The DML omitted-variable-bias bound scales by
`S = sqrt(σ²ν²)`, where for the PLR coefficient
`σ² = E[(Y − ℓ(X) − θ(D − m(X)))²]` and `ν² = 1/E[(D − m(X))²]`. StatsPAI
computed the numerator as `sd(Y − ℓ(X))`, i.e. without removing the treatment's
own contribution. Because `sd(Y − ℓ)² = σ² + θ²·sd(D − m)²`, the scaling factor
was systematically too large.

**Effect.** `rv_q`, `rv_qa`, `bias_bound`, `adjusted_estimate_low/high`, `s`,
and every row of `benchmarks` change. The direction is consistent: the old
code **overstated the bias bound and understated the robustness value**, so it
portrayed estimates as *more* fragile to unobserved confounding than the bound
warrants. On a linear-nuisance PLR fit (n = 1500) the bias bound was 0.0671
instead of 0.0529 (27% too large) and `RV_1` was 0.454 instead of 0.533.

After the fix, `bias_bound` and the adjusted `theta` bounds match
`doubleml`'s `sensitivity_analysis` to 2.5e-15 and `RV` to 9.2e-8 on a shared
fold partition (`tests/external_parity/test_dml_sensitivity_parity.py`).

`model='irm'` is **unaffected**: it stores `y_resid` as the score residual
`ψ − θ̂`, which is already centred, so subtracting `θ·d_resid` again would
double-count.

**What to do.** Re-run any reported robustness values. If you previously
concluded that a DML estimate was *not* robust on the basis of a low `RV_q`,
recheck — the corrected value is higher. `rv_qa` remains a StatsPAI
convention: it exhausts `|θ| − z·se` using the unadjusted standard error,
whereas `doubleml` lets the standard error move with the confounding scenario;
the two differ by about 0.14% on the pinned fixture and that gap is asserted
rather than hidden.

---

<a id="dml-irm-iivm-se-normalisation"></a>

## 1.21.0 — ⚠️ `sp.dml` IRM / IIVM standard errors normalise by `n`

**What changed.** The unweighted IRM and IIVM branches divided the
influence-function variance by `n − 1` (`ddof=1`). Nothing else in the module
did: PLR and PLIV use `n`, the weighted IRM/IIVM branches use `n`, and even
the `normalize_ipw`/ATTE branch *inside `irm.py`* uses `mean(psi**2)`, i.e.
`n`. So `sp.dml(model='plr')` and `sp.dml(model='irm')` on the same data
reported standard errors under two different conventions, and which one you
got for IRM depended on whether you passed `normalize_ipw`. `DoubleML`
normalises by `n`; all paths now agree.

**Effect.** IRM and IIVM standard errors shrink by exactly `sqrt((n−1)/n)`.
That is 0.025% at n = 2000 and about 1% at n = 50 — immaterial for most
reported results, but it is the difference between matching `DoubleML` and
not. On the Track A module-71 fixture the IRM/IIVM standard errors now agree
with `DoubleML` 1.0.2 to 1.1e-10 on a shared fold partition (they were off by
the `sqrt(n/(n−1))` factor, observed ratio 1.00025009389849 against
`sqrt(2000/1999) = 1.00025009378908`). **Point estimates are unchanged**, as
are PLR and PLIV in full.

**What to do.** Nothing. Confidence intervals narrow very slightly; if you
need the old figures, multiply the reported SE by `sqrt(n/(n−1))`.

---

<a id="dml-fold-indices-all-models"></a>

## 1.21.0 — `sp.dml(fold_indices=...)` now works for IRM / PLIV / IIVM

**What changed.** `fold_indices=` was accepted only for `model='plr'`; the
other three model classes raised `MethodIncompatibility` rather than silently
ignore the argument. All four now route caller-supplied folds through a
shared `_make_splits` helper, so cross-fitting can be pinned to an explicit
partition for every model.

**Effect.** No change to any existing result — the parameter previously
raised, so nothing depended on it. With folds supplied, the estimate becomes
independent of `random_state`, which is what makes a bit-exact comparison
against `DoubleML` possible (Track A module 71).

**What to do.** Nothing is required. If you supply folds for `irm` or
`iivm`, note that you are bypassing the built-in `StratifiedKFold`: each
training set must contain both classes of the binary nuisance target, and a
partition that violates this now raises `DataInsufficient` naming the
offending fold rather than fitting a degenerate classifier.

---

<a id="causal-forest-grf-att-convention"></a>

## 1.21.0 — ⚠️ Causal-forest ATT/ATC now use grf's estimator

**What changed.** `sp.causal_forest(...).average_treatment_effect(
target_sample='treated')` (and `'control'`) computed the mean of a single
Robins doubly-robust score divided by `p̂₁`. `grf::average_treatment_effect`
does something structurally different: it reports the **plug-in CATE average
over the target arm plus a Hájek-normalised doubly-robust correction**, and
reports the standard error as the square root of the *sum* of the two
components' variances — the plug-in dispersion
`Σ_{i:Tᵢ=1}(τ̂ᵢ - τ̄)² / n₁²` plus `n/(n-1) · Σᵢ Δᵢ² / n²` — rather than the
dispersion of one score vector. StatsPAI's docstring claimed GRF-style
ATT/ATC aggregation, so this was a documentation/implementation mismatch,
not a deliberate alternative convention.

**Effect.** ATT and ATC point estimates change slightly and their standard
errors change materially. Given `grf`'s own forest outputs — so that the
forest is held fixed and the comparison isolates the formula — the old
route matched `grf`'s ATT point estimate to 9.3e-5 but produced a standard
error **12% larger**. The new code reproduces `grf` 2.6.1's ATT estimate and
`std.err` to 1e-15 on those same inputs. **ATE and ATO are unchanged**, and
the ATE score vector was already elementwise identical to `grf::get_scores`.

On the Track A module-13 clean-overlap fixture, where the two sides grow
independent forests, the ATT standard-error gap against `grf` fell from
14.6% to 0.087%.

**What to do.** Nothing is required; the new numbers are the ones the
documentation always described. If you reported causal-forest ATT/ATC
standard errors from an earlier version, re-run — the previous figures were
conservative (too wide) rather than anti-conservative, so significance
claims do not flip in the dangerous direction, but they were not `grf`'s.
The two operators are now exposed directly as
`statspai.forest.forest_inference.aipw_scores` (the ATE influence function,
elementwise equal to `grf::get_scores`) and `grf_att_atc` (the ATT/ATC
decomposition), so the formula can be inspected and reused without going
through a fitted forest.

---

<a id="policy-tree-exact-depth2-search"></a>

## 1.21.0 — ⚠️ `sp.policy_tree` now solves the depth-2 problem exactly

**What changed.** The module documented an exhaustive depth-1/depth-2 search
but implemented a greedy one: each candidate root split was scored as though
both of its children were terminal leaves, and only then did the routine
recurse. That one-step lookahead is exact for a depth-1 stump, but for
`max_depth=2` — the default — the root split that scores best with terminal
children is routinely *not* the root split that admits the best pair of
depth-1 subtrees. Candidate thresholds were also subsampled to at most 50
quantiles per covariate, so the returned tree was not even the greedy optimum
over the full split grid.

Depth ≤ 2 now maximises the Athey–Wager objective
`sum_i Gamma_i * pi(X_i)` exactly, by exhaustive search over the complete
grid of distinct covariate values, matching `policytree::policy_tree`'s
`x <= t` split convention and its "smallest permitted terminal node" reading
of `min_leaf_size`.

**Effect.** For `max_depth=2`, the learned policy and every quantity derived
from it — `policy`, `value_policy`, `value_gain`, `fraction_treated`,
`rules` — can change. On the Track A module-70 fixture the old search fell
0.70% short of the welfare optimum and assigned 78 of 1200 units to the
wrong arm. `max_depth=1` is unaffected (greedy is exact for a stump), except
where the 50-quantile threshold subsample previously missed the best split.
Depth ≥ 3 is unchanged in kind — still greedy, because exhaustive search is
combinatorially infeasible — but now searches the full threshold grid.

**What to do.** Nothing is required; the new numbers are the ones the
documentation always promised, and they now agree with `policytree` 1.2.4 to
9.6e-16 with all per-row policy decisions identical. If you need to
reproduce an earlier figure, pass `search='greedy'` — but note that this
still searches the full grid, so it does not reproduce the old
50-quantile behaviour exactly. Check `result['search_mode']` to see which
search actually ran; `search='auto'` (the default) falls back to greedy with
a `UserWarning` when the exact sweep would exceed its cost budget, and
`split_step=k` thins the candidate grid the way `policytree`'s `split.step`
does.

---

<a id="mahalanobis-pooled-covariance"></a>

## 1.21.0 — ⚠️ Mahalanobis matching uses the pooled within-group covariance

**What changed.** `sp.match(distance='mahalanobis')` and
`sp.optimal_match(metric='mahalanobis')` built the metric from `cov(X)` over
the pooled sample. The Mahalanobis matching metric of Rubin (1980) — the
reference the module already cited, and the one `MatchIt` uses — is the
pooled *within-group* covariance `[(n₁−1)S₁ + (n₀−1)S₀] / (n₁+n₀−2)`. The
total covariance is inflated along the direction in which the group means
differ, which is exactly the direction matching needs to resolve most
finely, so it systematically under-weights the covariates that separate the
groups.

**Effect.** All Mahalanobis matching estimates change. The new default
reproduces `MatchIt:::mahalanobis_dist` to 1e-15.

**What to do.** Pass `mahalanobis_cov='total'` to restore the previous
metric if you need to reproduce an earlier figure. New work should keep the
default.

---

<a id="match-m-order"></a>

## 1.21.0 — `sp.match` greedy matching order is now explicit

**What changed.** Nearest-neighbour matching *without replacement* is
order-dependent: each treated unit consumes a control, so who is matched
first changes who is left. StatsPAI processed treated units
closest-pair-first without documenting it. That is now the `m_order`
parameter.

**Effect.** None by default — `m_order='smallest_min_dist'` is the previous
behaviour. But the choice is material: on `MatchIt::lalonde` with
Mahalanobis distance the ATT ranges over more than 5x across orders, so if
you are comparing against another package you should set it explicitly.
`m_order='data'` and `'closest'` reproduce the MatchIt rules of the same
name exactly.

**What to do.** Nothing is required. When reconciling with R, set
`m_order='data'` (MatchIt's default for non-propensity distances) or
`'closest'`.

---

<a id="honest-did-flci"></a>

## 1.21.0 — ⚠️ `sp.honest_did(method='smoothness')` now returns the real FLCI

**What changed.** The native smoothness path returned
`θ̂ ± M·(e+1) ± z·SE`: the worst-case bias added to an ordinary Wald interval.
That is not the Rambachan-Roth confidence set — it ignores the pre-period
covariance and was *narrower* than the reference at every M, overstating how
robust a result is to parallel-trends violations. It now solves the actual
fixed-length confidence interval.

Separately, `backend='r'` was building `sigma <- diag(ses^2)` before calling
`HonestDiD`, discarding the cross-period covariance. Both backends now receive
the full event-study covariance recovered from the influence functions.

**Effect.** All `method='smoothness'` intervals move. Two changes will look
surprising and are correct:

- **The interval is no longer centred on the event-study coefficient.** Its
  centre is the optimal affine estimator, which extrapolates the pre-trend.
- **`M=0` no longer equals the Wald interval.** `Δ^SD(0)` still permits an
  arbitrary *linear* pre-trend, so the M=0 FLCI prices in that extrapolation.
  R `HonestDiD` behaves identically.

**Who is affected.** Anyone reporting `sp.honest_did(method='smoothness')`.
Re-run; the new numbers agree with R `HonestDiD` to ~7e-5 on width. If the
event-study covariance cannot be recovered (a result carrying no influence
functions), the old approximation is still used and now warns.
`method='relative_magnitude'` is unchanged and still approximate.

<a id="cs-rcs-reg-covariates"></a>

## 1.21.0 — ⚠️ `callaway_santanna(panel=False, estimator='reg', x=[...])` changed estimator

**What changed.** Repeated cross-sections with covariates previously used a
StatsPAI-specific approximation: the outcome was residualised on the covariates
using the never-treated pool with period fixed effects, and then plain 2×2
cell-mean differences were taken. R `did` instead calls
`DRDID::reg_did_rc`, which fits period- and group-specific outcome regressions
inside each (g, t) cell. StatsPAI now does the same.

**Effect.** ATT and SE both move for `panel=False` **with covariates**. In
exchange the estimator now reproduces R `did::att_gt(panel=FALSE,
est_method="reg", xformla=...)` to ~1e-11 on `did::mpdta`
(−0.0419686124 with never-treated controls).

**Who is affected.** Only `panel=False` calls that pass `x=`. Repeated
cross-sections *without* covariates are unchanged (still the cell-mean DiD, and
still equal to the unconditional panel simple ATT). Panel calls are entirely
unaffected. Re-run and use the new values; the old path was an approximation
with no reference implementation behind it.

<a id="cs-varying-base-period-e-minus-1"></a>

## 1.21.0 — ⚠️ `base_period='varying'` now reports the `e = −1` placebo

**What changed.** `sp.callaway_santanna`'s (g, t) grid builder skipped
`t == g − 1 − anticipation` under *every* base-period scheme. Under
`base_period='universal'` that is correct — it is the reference period and
ATT(g, g−1) is zero by construction. Under `base_period='varying'` it is not
the reference: the base for `t = g−1` is `g−2`, so ATT(g, g−1) is an estimable
pre-treatment placebo. R `did` and Stata `csdid` both report it; StatsPAI
dropped it.

**Effect.** Under `base_period='varying'` the event study gains one row at
`e = −1`. On canonical `did::mpdta` that cell is −0.024459, matching R `did`
2.3.0 and Stata `csdid` to the printed precision — and with it restored the
*entire* varying event study now agrees with both references, where previously
only the post-treatment half did.

Post-treatment coefficients do not move, under either scheme. The default
`base_period='universal'` path is completely unchanged.

**Who is affected.** Users of `base_period='varying'`, and in particular
anything that consumes the pre-period vector:

- `sp.honest_did` / `sp.sensitivity_rr` — one more pre-period enters the
  Rambachan–Roth restriction set, so breakdown values and robust CIs shift.
- `sp.pretrends_test` / `sp.pretrends_power` — the joint pre-trend test gains a
  degree of freedom.

Re-run these if you have recorded output from `base_period='varying'`. If you
need the old event-time grid for comparison, `base_period='universal'` is
unchanged, but note it is a *different* placebo estimand, not the old buggy
one. There is no flag to restore the omission — it was a bug.

<a id="aggte-group-overall-weighting"></a>

## 1.21.0 — ⚠️ `sp.aggte(type='group')` overall ATT is now cohort-size weighted

**What changed.** The per-cohort effects θ(g) were correct, but collapsing them
into the single reported `.estimate` used equal `1/K` weights. R
`did::aggte(type="group")` weights each cohort by its share of treated units:
`sum_g (p_g / sum_g p_g) * θ(g)`. The `cohort_sizes` series needed for this was
already being computed by `sp.callaway_santanna` and passed into the weight
builder, where it was silently ignored.

**Effect.** Only the headline scalar (`.estimate`, `.se`, `.pvalue`, `.ci`) of
`sp.aggte(type='group')` moves. The `.detail` frame — one row per cohort — is
unchanged. The size of the shift depends on how unequal the cohorts are and how
much the effect varies across them; it is exactly zero when all treated cohorts
are the same size. On a 300-unit simulated panel the overall went from
0.4315153 to 0.4317301.

**Who is affected.** Anyone who read the overall number off
`sp.aggte(type='group')`. `type='simple'`, `'dynamic'`, and `'calendar'` are
unaffected — `simple` was already cohort-share weighted, and R reports the
dynamic and calendar overalls as unweighted means across event times and
calendar periods respectively, which is what StatsPAI already did. Re-run and
use the new value; there is no flag to restore the old behavior.

<a id="aggte-analytic-se-covariance"></a>

## 1.21.0 — ⚠️ `sp.aggte(bstrap=False)` standard errors were ~0.64× too small

**What changed.** With `bstrap=False`, `sp.aggte` combined the per-cell
standard errors as `sqrt(Σ wₖ² seₖ²)` — the formula for *independent* cells.
ATT(g, t) cells are not independent: they are built from overlapping sets of
control units, so the omitted covariance terms are large and positive. Both the
per-cell and the overall SE now aggregate through the influence functions,
`sqrt(mean((Ψw)²)/n)`, matching R `did` and matching what this function's own
`bstrap=True` branch already used for the overall estimate.

**Effect.** SEs on the `bstrap=False` path get larger — on simulated staggered
panels the old value averaged **0.635×** the multiplier-bootstrap SE, so a
nominal 5% Wald test of a true null was rejecting about 21–23% of the time.
Confidence intervals widen accordingly and some previously "significant"
aggregations will stop being significant. Point estimates do not move.

**Who is affected.**

- Callers who passed `bstrap=False` explicitly.
- Callers whose result carried no influence-function matrix, where `aggte`
  forces `bstrap=False` internally.
- **Not** the default path: `sp.aggte` ships `bstrap=True`.
- **Not** `sp.callaway_santanna`'s own headline SE, which already aggregated
  through the influence functions and is numerically unchanged.

If you have recorded SEs or p-values from `sp.aggte(..., bstrap=False)`,
re-run them. The new values are the correct ones; there is no flag to restore
the old behavior. As before, `bstrap=True` additionally gives you the uniform
sup-t bands, which the analytic path cannot produce.

<a id="cic-athey-imbens-step2"></a>

## 1.21.0 — ⚠️ `sp.cic` now reproduces the Athey-Imbens estimator

**What changed.** The step-2 counterfactual in `sp.cic` had two defects: it
composed the empirical CDFs with the control-post (`y01`) and treated-pre
(`y10`) cells transposed relative to Athey & Imbens (2006) eq. 9, and it used
linearly-interpolated CDF / quantile functions on a finite τ grid instead of
the step-function ECDF and its generalized inverse. It now computes the
counterfactual map `k(y) = F_01⁻¹(F_00(y))` on the step ECDF.

**Effect.** The unconditional ATT converged ~0.5% away from the reference
(2.8% with covariates); it now matches Kranker's Stata `cic` (a direct port of
the A&I Matlab) to the printed digits — e.g. 2.999904 on the test fixture, where the old
grid-dependent code gave 3.01792 at the default `n_grid=200` (3.01388 in the
large-grid limit). Every `sp.cic` point estimate and QTE moves
slightly.

**Who is affected.** Anyone who ran `sp.cic`. If you have recorded CIC numbers
from an earlier release, re-run them; the new values are the correct A&I
estimates. There is no flag to restore the old behavior — it was a bug.

<a id="panel-hdfe-multiway-cluster-nul"></a>

## 1.21.0 — ⚠️ Panel HDFE multiway cluster SEs no longer collapse

**What changed.** `sp.hdfe_ols` / `sp.feols`' native N-way cluster sandwich
formed its intersection clusters by joining the dimension labels with a `"\0"`
separator, but `pd.factorize` truncates object strings at an embedded NUL
byte. Every intersection therefore collapsed onto its first cluster variable,
so distinct specifications such as `cluster(prov, year)` and
`cluster(pref, year)` returned *identical* standard errors. Replaced with a
mixed-radix integer code combination, mirroring the fix already applied to the
standalone `sp.multiway_cluster_vcov` inference path in v1.17.0.

**Effect.** Two-way and higher cluster-robust SEs from the panel HDFE path
change, toward Stata `reghdfe`. One-way clustering and non-clustered SEs are
unaffected.

**Who is affected.** Anyone using `vcov={"CRV1": [a, b]}`-style multiway
clustering through `sp.hdfe_ols` / native `sp.feols`. Re-run affected models.

<a id="conley-non-psd-nan"></a>

## 1.21.0 — ⚠️ Conley non-PSD variances report `nan`, not `0`

**What changed.** Kernel-weighted spatial HAC is not positive semi-definite in
finite samples; with a uniform kernel `S'WS` routinely has negative diagonal
entries. Every Conley path used `sqrt(max(V, 0))`, which turned a negative
variance into `se = 0` — reported downstream as `t = ∞`, `p = 0`. Affected
terms now return `nan` with a loud `RuntimeWarning` (Stata `acreg` reports the
same terms as missing). Rounding-level negatives are still clamped to 0
silently.

**Effect.** Where the Conley covariance was non-PSD, SEs that used to read
`0.0` now read `nan`. The covariance itself is unchanged (and still matches
`acreg` to ~1e-12 where it is PSD), so any SE that was previously non-zero is
unchanged.

**Remedies** named in the warning: widen/narrow the distance cutoff, use
`kernel="bartlett"` (tapered kernels are far better behaved than the uniform
indicator), or check whether the coordinates are collinear with the absorbed
fixed effects.

<a id="event-study-pre-vcov-optin"></a>

## 1.21.0 — `sp.event_study` pre-period covariance (opt-in this release)

**What changed.** `sp.event_study` now computes the full cluster-robust
covariance of the event-time coefficients (always available in
`model_info['vcov']`). The pre-period submatrix `model_info['vcv_pre']` —
which `pretrends_test`, `pretrends_power`, `sensitivity_rr`, and `honest_did`
use in place of the historical diagonal (independent-pre-coefficients)
approximation — is written **only** when you pass `expose_pre_vcov=True`.

**Why opt-in for now.** Switching the default would move published honest-DiD
and pre-trend numbers during the live JOSS review. By default the diagonal
fallback still fires — but it now **warns loudly** that it is assuming the
pre-period coefficients are independent (they are not; they share the omitted
reference period and the fixed effects). The correct full covariance becomes
the default in a future release, at which point it will be logged as a flagged
⚠️ correctness fix.

**What to do.** For statistically correct honest-DiD / power today, pass
`expose_pre_vcov=True` to `sp.event_study`. To reproduce numbers from an
earlier release, do nothing — the default is unchanged.

<a id="event-study-headline-att-se"></a>

## 1.21.0 — ⚠️ Event-study headline ATT SE uses the full covariance

**What changed.** Three event-study estimators reported a headline ("overall
ATT") standard error that treated the post-period event-time coefficients as
independent:

- `sp.event_study`: `sqrt(mean(se²)/m)` → now `sqrt(w'Vw)` with `w = 1/m`
  over the post-period block of the cluster-robust `model_info["vcov"]`.
- `sp.design_robust_event_study`: same formula swap on its cluster-robust
  vcov (validated against a 400-draw cluster bootstrap of the full
  procedure: analytic 0.3065 vs bootstrap 0.3101; the old formula gave
  0.2414).
- `sp.cohort_anchored_event_study`: the per-event-time bootstrap loops were
  merged into one **joint** cluster bootstrap; the headline SE is now the
  bootstrap SD of the post-period average itself.

Event-time coefficients share a reference period and fixed effects, so their
covariance is large and positive; the independence approximation understated
the headline SE — by ~2× on a realistic staggered test panel.

Additionally, `sp.event_study`'s `model_info["pretrend_test"]` is now a
**cluster-robust Wald test** (`F(q, G-1)` on the same vcov as the printed
SEs) instead of a classical homoskedastic F-test, and the `or 1e-6`
fabricated-SE fallback in `design_robust` / `cohort_anchored` is gone (an
unavailable SE is `NaN` plus a warning).

**Effect.** Headline `se` / `pvalue` / `ci` change (typically wider /
less significant) for every call to these three functions; headline
`estimate` and the per-coefficient rows are unchanged. Downstream,
`sp.parallel_trends_robustness` breakdown values shift slightly (e.g.
0.504 → 0.485 on the covariance-export test fixture). `pretrend_test`
statistics/p-values change under clustering; the dict gains `df_denom`.

**Who is affected.** Anyone quoting the overall ATT inference or the inline
pre-trend test from these estimators. Re-run affected models; the new values
are the correct ones. There is no flag to restore the old behavior — it was
a bug. (This is separate from the `expose_pre_vcov` opt-in above, which
remains opt-in during the JOSS review: that flag governs what the
*downstream pre-trend tools* consume, whereas this fix governs the headline
aggregation, whose correct covariance was already computed unconditionally.)

<a id="did2x2-ddd-weighted-robust"></a>

## 1.21.0 — ⚠️ Weighted robust SEs in `sp.did_2x2` / `sp.ddd`

**What changed.** With analytic weights, the `robust=True` (HC1) branch
built the sandwich meat as `X'diag(w·e²)X`; the WLS score is `w·x·e`, so the
correct meat is `Σ w²e²xx'` (Stata aweight-robust / R `sandwich`
convention). The cluster branch always squared the score correctly.

**Effect.** `sp.did_2x2(..., weights=, robust=True)` and
`sp.ddd(..., weights=, robust=True)` SEs change (~9% on dispersed weights)
and now match Stata 18 MP `regress ..., [aw=w] robust` to machine precision
(pinned in `tests/reference_parity/test_did2x2_ddd_weighted_robust_parity.py`).
Point estimates, unweighted SEs, and clustered SEs are unchanged.

**Who is affected.** Only weighted + `robust=True` calls. Re-run them.

<a id="parallel-trends-robustness-inf-verdict"></a>

## 1.21.0 — ⚠️ `sp.parallel_trends_robustness` verdict at `Mbar* = ∞`

**What changed.** When the honest CI still excludes zero at the top of the
search range (`Mbar = 1e4`), the breakdown is `inf` — maximal robustness.
The verdict builder's `not np.isfinite(...)` guard routed that case into the
"NOT robust: the CI already includes zero at M = 0" sentence — the exact
opposite conclusion. `inf` now yields a "robust over the entire searched
range" verdict; failed (NaN) families are excluded from the binding-family
comparison and listed in an explicit note instead of silently (and
order-dependently) participating in the `min`.

**Effect.** Only the `verdict` string changes. The `breakdown` and
`ci_grid` tables were always correct.

**Who is affected.** Anyone (human or agent) who read `result.verdict` for
a large effect measured in raw units. Re-read those verdicts.

<a id="conley-duplicate-unit-time"></a>

## 1.21.0 — ⚠️ `sp.conley` rejects duplicated `(unit, time)` rows

**What changed.** The spatio-temporal path (`time=` + `unit=`) resolves the
cross-unit block through a single-valued `(unit, time) → row` lookup. With
more than one row per unit-period (e.g. plant-level rows with
`unit="county"`), the cross-unit terms silently kept only the last duplicate
row while the within-unit terms kept all rows — wrong SEs with no signal.
`sp.conley` now raises a `ValueError` naming the offending unit, matching
Stata `acreg`'s repeated-id-time restriction.

**Effect.** Previously-silent wrong answers become an immediate error.

**What to do.** Aggregate your data to one row per `(unit, time)`, or pass
the true row-level identifier as `unit=` if each row is its own location.

<a id="proximal-surrogate-index-bridge-2sls"></a>

## 1.21.0 — ⚠️ `sp.proximal_surrogate_index` bridge is now proper 2SLS

**What changed.** The linear bridge `h(s, x)` used to be read off a
second-stage regression of `Y` on `[1, W, S_hat, X]`. Because `S_hat` — the
first-stage projection of `S` on `[1, W, X]` — is an exact affine function of
those same columns, that design matrix is rank-deficient, and the reported
"bridge slope" was whatever minimum-norm split `np.linalg.lstsq` happened to
return. Concretely, the point estimate depended on the *units* of the proxy
`W`: on a fixed persistent-confounding DGP with true ATE 1.32, the estimate
was 0.49 with `W` as given, 0.008 with `W×10`, and 1.22 with `W×0.01`. The
second stage now excludes `W` (`Y ~ [1, S_hat, X]`), which solves the correct
bridge moment `E[(Y - h(S,X)) · (1, W, X)'] = 0` — classical 2SLS with the
proxies as excluded instruments. Estimates are now invariant to rescaling `W`
and recover the true long-term ATE in the linear model.

**Why.** A point estimate that changes by two orders of magnitude when a proxy
switches from dollars to cents is not an estimate of anything (§7 — numerical
correctness is the floor).

**Who is affected.** Every previous `sp.proximal_surrogate_index` call —
earlier point estimates, SEs, and CIs were unit-dependent artifacts and should
be discarded, not compared against the new output. `sp.surrogate_index` and
`sp.long_term_from_short` are untouched.

**Action.** Re-run affected analyses. Two calls that previously "worked" now
raise: fewer proxies than surrogates raises `MethodIncompatibility`
(under-identified order condition), and proxies whose first-stage projections
are collinear raise `DataInsufficient` (rank condition). Both used to return
minimum-norm artifacts silently.

---

<a id="callaway-santanna-nevertreated-no-control"></a>

## 1.21.0 — ⚠️ `sp.callaway_santanna` fails loudly with an empty never-treated control

**What changed.** `sp.callaway_santanna(control_group="nevertreated")` on a
panel where every unit is eventually treated (no `g=0` units) used to return a
silent `ATT = 0.0`: each `ATT(g,t)` had no comparison cell, returned `0.0`, and
those aggregated to a headline `0.0` with no warning. It now raises
`MethodIncompatibility`.

**Why.** `0.0` is a specific wrong number that reads as "no treatment effect,"
so a mis-specified control group produced a plausible-looking but meaningless
estimate instead of an error (§7 — fail loudly).

**Who is affected.** Only calls that requested `control_group="nevertreated"`
on a panel with zero never-treated units. Any panel with at least one
never-treated unit (including `NaN`/`inf`-coded, which are treated as
never-treated) is unchanged, and `control_group="notyettreated"` is unchanged.

**Action.** Use `control_group="notyettreated"` (later-treated cohorts serve as
controls), or add never-treated units to the panel. No previously-valid
estimate changes.

---

<a id="eigenvector-centrality-bipartite"></a>

## 1.21.0 — ⚠️ `sp.eigenvector_centrality` fixed on bipartite graphs

**What changed.** Eigenvector centrality was computed by naive power iteration
`x <- A x`. On a bipartite graph the adjacency spectrum is symmetric
(`lambda_max = -lambda_min`), so the iteration oscillates between the two
sign-partitions and never converges; after `max_iter` steps it returned a
near-uniform vector (a star scored ~`1/sqrt(n)` for every node). The leading
eigenvector is now obtained by direct eigendecomposition, so the dominant nodes
score correctly (star hub `1/sqrt(2)`, leaves `1/sqrt(8)`).

**Why.** The returned centralities were qualitatively wrong on any bipartite or
near-bipartite network — the whole point of the measure (ranking nodes by
recursive influence) was lost.

**Who is affected.** Any `sp.eigenvector_centrality` call on a bipartite or
near-bipartite graph. Non-bipartite connected graphs (where power iteration did
converge) are numerically unchanged up to normalization.

**Action.** Re-run; the new scores are the correct leading eigenvector. The
`max_iter` / `tol` arguments remain accepted for backward compatibility.

---

<a id="ges-collider-acyclicity"></a>

## 1.21.0 — ⚠️ `sp.ges` no longer adds a spurious collider-parent edge

**What changed.** Greedy Equivalence Search searched over edge additions with no
acyclicity constraint. On a v-structure `X -> Z <- Y` it could add an edge into
`Z` and another out of `Z`; scoring a parent of `X`/`Y` then conditioned on the
collider and made the two independent parents look dependent, so a false
`X -- Y` edge entered the graph. The search now rejects cycle-creating edges and
returns the DAG's CPDAG (v-structures directed, reversible edges undirected).

**Why.** The recovered skeleton was wrong — colliders came back fully connected
instead of `X -> Z <- Y`, contradicting the d-separation structure.

**Who is affected.** Any `sp.ges` result on data containing a collider (common).
Recovered graphs may lose spurious edges and gain correct v-structure
orientations; chains now read as undirected CPDAG edges rather than a single
arbitrary orientation.

**Action.** Re-run `sp.ges`; the new adjacency is the correct CPDAG. No API
change (`.edges()`, `.adjacency`, `.to_frame()` unchanged in shape).

---

<a id="dist-iv-binary-instrument-nan"></a>

## 1.21.0 — ⚠️ `sp.dist_iv` / `sp.kan_dlate` no longer NaN on binary instruments

**What changed.** The distributional-IV Wald estimator split the instrument at
`Z > median(Z)`. For a binary `Z` with more 1s than 0s the median is 1, so the
high group (`Z > 1`) was empty and `late_q` came back all-`NaN` with no error —
about half of ordinary data draws. The split now falls back to `Z >= median`
when the strict split is degenerate, so a binary instrument always separates
into its two levels; it returns NaN only when `Z` is constant.

**Why.** A silently all-NaN point estimate is a correctness failure — the
function ran to completion and returned a result object full of NaNs.

**Who is affected.** Any `sp.dist_iv` / `sp.kan_dlate` call whose instrument is
binary (or discrete with the median on the top support point). Draws that
already produced finite estimates are numerically unchanged.

**Action.** Re-run affected calls; previously-NaN quantiles now carry the
correct Wald LATE. No API change.

---

<a id="contrast-pwcompare-categorical"></a>

## 1.21.0 — ⚠️ `sp.contrast` / `sp.pwcompare` now fire `C(var)` factor dummies

**What changed.** `sp.contrast` and `sp.pwcompare` previously returned all-zero
contrasts (and zero SEs / p-values) when the model was fit with a
formula-encoded categorical such as `y ~ C(g) + x`. The predictive-margin
engine matched coefficient terms to raw data columns, so design terms named
`C(g)[T.1]` never responded to setting the raw `g` column to a level. The margin
builder now parses treatment-coded factor terms (`C(var)[T.level]`, including
string levels), so reference/adjacent/pairwise contrasts equal the
corresponding dummy coefficients exactly.

**Why.** All-zero contrasts are a silent correctness failure — the function ran
without error but every reported difference was wrong. The fix restores the
documented Stata `margins, contrast(...)` behaviour for factor-encoded models.

**Who is affected.** Any `sp.contrast` / `sp.pwcompare` call on a model fit with
`C(...)` factor notation. Models that coded the categorical as a plain numeric
column were already correct and are unchanged.

**Action.** Re-run affected contrasts; the new numbers are the correct ones
(they equal the treatment-dummy coefficients). No API change.

---

<a id="did-multiplegt-baseline-conditioning"></a>

## 1.21.0 — ⚠️ `sp.did_multiplegt` now baseline-conditions switcher/stayer cells

**What changed.** `sp.did_multiplegt` now computes DID_M, dynamic effects, and
placebo effects within each baseline-treatment cell `d_{t-1}`. Switchers are
compared only to stayers with the same baseline treatment; switch-off cells are
sign-flipped so the reported estimand is the effect of gaining treatment. The
dynamic path additionally uses robust stayers that keep the baseline treatment
unchanged through the full horizon `[t, t+h]`, and placebo effects use the Stata
`did_multiplegt (old)` mirror sign convention.

**Why.** Pooling all stayers in a period let already-treated stayers contaminate
untreated control trends and mixed switch-on / switch-off effects under one
majority sign. The static path is pinned to Stata reference values, and the
dynamic/placebo path is guarded by small hand-computable panels that isolate the
robust-stayer and placebo-sign requirements.

**Who is affected.** Any `sp.did_multiplegt` run with multiple baseline
treatment values, switch-off events, dynamic effects, or placebo effects can
change. Designs with only switch-on events and a single valid same-baseline
stayer set may be unchanged.

**Action required.** Re-run reported `did_multiplegt` estimates, especially if
they used `dynamic=` or `placebo=`. No call-site change is required; this is a
numerical correction. For release/JOSS notes, flag this as a correctness fix
that can change point estimates.

---

<a id="spatial-ml-fullinfo-se"></a>

## 1.21.0 — ⚠️ `sp.sar` / `sp.sdm` report full-information coefficient SEs

**What changed.** The coefficient standard errors from `sp.sar` (spatial lag)
and `sp.sdm` (spatial Durbin) now come from the inverse of the full
`(β, ρ, σ²)` maximum-likelihood information matrix — the same asymptotic
covariance `spatialreg::lagsarlm` reports — instead of the concentrated
`σ²(XᵀX)⁻¹`. The bounded `ρ`/`λ` line-search was also tightened to
`xatol=1e-10`.

**Why.** The concentrated formula treats the spatial parameter `ρ` as known,
dropping the `β`–`ρ` covariance and understating the coefficient SEs; on a
row-standardised `W` the intercept SE came out roughly half its correct value.
The full information matrix was already being formed and inverted to produce the
`ρ` SE, so the correct `Var(β)` is the leading block of that same inverse.
Module `65_spatial` now grades `sar`/`sem`/`sdm` **bit-exact** against
`spatialreg` (worst relative error 8.3e-8 on estimates, 2.0e-8 on SEs).

**Who is affected.** Any `sp.sar` / `sp.sdm` result whose reported coefficient
standard errors, t/z-statistics, p-values, or confidence intervals were used;
the intercept SE moves most. Point estimates move only at the ≲1e-5 level from
the tighter optimiser. `sp.sem` and `sp.slx` standard errors are unchanged.

**Action required.** Re-run any `sp.sar` / `sp.sdm` inference; coefficient point
estimates are substantively unchanged, but SEs (hence significance) can differ.
No call-site change is required — this is a numerical correction.

---

<a id="etwfe-cgroup-simple-att"></a>

## 1.21.0 — ⚠️ `sp.etwfe` now honors `cgroup` and reports the R/Stata simple ATT

**What changed.** The public `sp.etwfe` headline now matches R
`etwfe::emfx(type="simple")` and Stata `jwdid, estat simple`: a
treated-observation-weighted simple ATT over post-treatment cohort-time effects.
The default `cgroup="notyet"` now uses not-yet-treated comparisons, while
`cgroup="nevertreated"` matches R `etwfe(cgroup="never")`.

**Why.** The previous public default was labelled `cgroup="notyet"` but behaved
like a never-treated-style estimand under a different aggregation. On the
canonical `did::mpdta` panel, this produced about `-0.0385` for the default
instead of the R/Stata not-yet-treated simple ATT `-0.047709918`. The corrected
`cgroup="nevertreated"` path matches the R never-treated value
`-0.039951275`.

**Who is affected.** Any code using `sp.etwfe(...).estimate`, `se`, `pvalue`, or
CI can change. The lower-level `sp.wooldridge_did` helper keeps its historical
saturated-TWFE cohort headline. `sp.etwfe_emfx` now defaults to
`weighting="treated"`; pass `weighting="cohort"` when you need the historical
cohort-share aggregation for comparison.

**Migration.**

| Before | After |
|---|---|
| R/Stata-compatible simple ATT via `sp.etwfe(..., panel=False)` + `sp.etwfe_emfx(..., weighting="treated")` | `sp.etwfe(...)` directly for the default not-yet-treated panel estimand |
| Previous never-treated-style default comparison | `sp.etwfe(..., cgroup="nevertreated")` |
| Historical saturated-TWFE helper output | `sp.wooldridge_did(...)` |
| Historical cohort-share emfx aggregation | `sp.etwfe_emfx(fit, weighting="cohort")` |

For release/JOSS notes, flag this as a correctness fix because the public
default point estimate changes on staggered panels.

---

<a id="bch-post-lasso-iv-deprecation"></a>

## 1.21.0 — Deprecation: `iv.bch_post_lasso_iv` → `sp.rlasso_iv`

**What changed.** `statspai.iv.bch_post_lasso_iv` now emits a
`DeprecationWarning`. It was StatsPAI's original, from-memory reconstruction
of the Belloni–Chen–Chernozhukov–Hansen (2012) post-Lasso IV estimator and
does **not** agree numerically with R's `hdm`: on the canonical eminent-domain
application it returns ≈0.013 where `hdm::rlassoIV` returns 0.227 (~17× off),
because it uses the asymptotic penalty `λ = 2c√{2n log(2p/α)}` and selects
only instruments (no control selection).

**Why.** `sp.rlasso_iv` is a faithful, parity-tested port of `hdm::rlassoIV`
(verified to ~1e-6 against `hdm` 0.3.2, exact on eminent domain). It supports
all four selection regimes (instruments, controls, both, neither).

**Migration.**

| Before | After |
|---|---|
| `iv.bch_post_lasso_iv(y='y', endog='d', instruments=z_cols, data=df)` | `sp.rlasso_iv(y='y', d='d', z=z_cols, data=df, select_Z=True, select_X=False)` |
| `iv.bch_post_lasso_iv(..., exog=x_cols)` | `sp.rlasso_iv(..., x=x_cols, select_Z=True, select_X=True)` |

The result object differs (`RLassoIVResult` exposes `.coef` / `.se` / `.tstat`
/ `.pvalue` / `.conf_int()` / `.summary()` / `.cite()`). `bch_post_lasso_iv`
keeps its original numerics during the deprecation window; nothing about
existing call sites breaks, but new code should use `sp.rlasso_iv`. See
[`docs/guides/rigorous_lasso_hdm.md`](docs/guides/rigorous_lasso_hdm.md).

---

<a id="cusum-boundary"></a>

## 1.20.0 — ⚠️ `sp.cusum_test` used the wrong CUSUM boundary

**What changed.** The recursive-residual CUSUM test compared the CUSUM path
against a **constant** critical value (`1.358` at 5%). That constant is the
`sup|Brownian bridge|` quantile of the *OLS-CUSUM* (Ploberger–Krämer 1992), a
different test; the Brown–Durbin–Evans recursive CUSUM crosses a **linear**
boundary `a·[1 + 2 s/(n−k)]` (`a = 0.948` at 5%) that widens from `a` to `3a`
across the sample. The old constant over-rejected late breaks and
under-rejected early ones — empirically it rejected ≈32% of stable series at a
nominal 5% level (now ≈4%).

**Who is affected.** Anyone reading `cusum_test(...)["reject"]` or
`["critical_value"]`. **`critical_value` changed from a scalar to the boundary
array**; `reject` is now True iff the path crosses that boundary anywhere.

**Action required.** If you compared `max_cusum` to a hard-coded `1.358`, use
the returned `reject` instead. Point estimates / the CUSUM path are unchanged.

---

<a id="lee-imbens-manski"></a>

## 1.20.0 — ⚠️ `sp.lee_bounds` reported a Horowitz–Manski CI labelled "Imbens–Manski"

**What changed.** The confidence interval padded *both* bound endpoints by the
two-sided `z_{1−α/2}`. That is the Horowitz–Manski interval for the identified
**set**, which over-covers the partially identified **parameter**, yet it was
labelled "Imbens–Manski". It is now the genuine Imbens & Manski (2004) interval:
a critical value `C_n` solving `Φ(C_n + Δ/σ_max) − Φ(−C_n) = 1 − α` that
interpolates between the one-sided `z_{1−α}` (wide bounds) and the two-sided
`z_{1−α/2}` (point identification). Refs verified via Crossref + RePEc/IDEAS
(Econometrica 72(6):1845–1857, doi:10.1111/j.1468-0262.2004.00555.x).

**Who is affected.** Anyone reading the CI from `sp.lee_bounds`. The interval is
**narrower** (correct). Point bounds, midpoint estimate, and bound width are
unchanged.

**Action required.** None beyond noting that previously reported CIs were
conservative (too wide).

---

<a id="rd-hc-variance"></a>

## 1.20.0 — ⚠️ RD heteroskedasticity-robust standard errors were inflated

**What changed.** The local-polynomial HC ("conventional"/"robust") variance
built its sandwich *meat* with the kernel weight to the **first** power
(`Σ w_i x_i x_i' e_i²`) instead of **squared** (`Σ w_i² x_i x_i' e_i²`) as the
Calonico–Cattaneo–Titiunik (2014) variance requires. Every HC-robust RD
standard error was therefore inflated — ≈1.4× for a uniform kernel versus R
`rdrobust` `vce="hc0"`. Affects `sp.rdrobust`, `sp.rd2d`, RD heterogeneous-
effects, and `sp.rd_bias_aware_fuzzy`. (Cluster-robust RD SEs were already
correct.)

**Who is affected.** Anyone using RD HC-robust SEs / CIs / p-values. Point
estimates are **unchanged**; SEs/CIs are now **smaller** and match R `rdrobust`
to the documented HC1-vs-HC0 d.o.f. convention.

**Action required.** None beyond noting that prior HC-robust RD intervals were
conservative (too wide). Re-run if you reported their exact width.

---

<a id="cs-pretrend-f"></a>

## 1.20.0 — ⚠️ `sp.callaway_santanna` pre-trend Wald test over-rejected

**What changed.** The joint pre-trend test (`model_info["pretrend_test"]`)
referred its Wald statistic `W = θ̂'V̂⁻¹θ̂` to `χ²(k)`. Because the pre-period
ATT(g,t) are strongly correlated (shared base period and control group) and
`V̂` is estimated, the plug-in χ² over-rejected in finite samples — empirical
size ≈0.15 at a nominal 5% level for ~60 units. It now applies the Hotelling-T²
correction, referring `W·(G−k)/(k·(G−1))` to `F(k, G−k)` (`G` = number of
units), which is exact under normal influence functions and → χ²(k)/k as
`G → ∞` (empirical size ≈0.07).

**Who is affected.** Anyone reading the pre-trend test p-value. ATT point
estimates and SEs are unchanged. The new p-value is (weakly) larger (less
likely to spuriously reject parallel trends).

**Action required.** None beyond noting prior pre-trend p-values were too small.

---

<a id="gardner-es-weighting"></a>

## 1.20.0 — ⚠️ `gardner_did(event_study=True)` overall ATT was unweighted

**What changed.** In event-study mode the overall ATT was the *unweighted* mean
of the post-period coefficients. It is now the treated-observation-**weighted**
mean (the `did2s` aggregated-ATT convention), which equals the
non-event-study `gardner_did` ATT exactly. The unweighted mean disagreed with
the non-ES path under heterogeneous effects / unbalanced horizon support
(e.g. 1.63 vs the correct 1.75).

**Who is affected.** Anyone using `gardner_did(event_study=True).estimate` (the
headline overall ATT). The per-horizon event-study coefficients are unchanged;
only their aggregation into the overall ATT changed.

**Action required.** Re-run if you reported the event-study overall ATT; it now
matches the obs-weighted (non-event-study) value.

---

<a id="regress-weights"></a>

## 1.20.0 — ⚠️ `sp.regress` ignored `weights=` (silently fit unweighted OLS)

**What changed.** `sp.regress(..., weights=col)` accepted the `weights`
argument through `**kwargs` and then never used it — the returned fit was
plain unweighted OLS, with no warning. As of this fix it solves the weighted
least squares problem with Stata `aweight` semantics, so `weights=` changes the
coefficients, standard errors (classical / HC-robust / clustered), and R²
exactly as a weighted regression should. Verified against **Stata 18 MP**
`regress y x [aw=w]` (+ `, robust` / `, vce(cluster …)`) to machine precision.

**Who is affected.** Anyone who called `sp.regress(..., weights=w)`. Calls
*without* `weights=` are numerically identical (the unweighted code path is
byte-for-byte unchanged).

**Action required.** Re-run any weighted `sp.regress` fits — prior results were
the unweighted OLS solution. The new path also raises `ValueError` on
non-finite, non-positive, wrong-length, or unknown-column weights instead of
silently proceeding. This mirrors the `sp.feols` no-FE weights fix below; the
same fail-silently bug existed independently in the OLS estimator.

---

<a id="hdfe-cluster-nested-fe"></a>

## 1.20.0 — ⚠️ `sp.hdfe_ols` cluster-robust SE inflated when an absorbed FE was nested in the cluster

**What changed.** The native HDFE backend (`sp.hdfe_ols` / `sp.absorb_ols`,
**not** the pyfixest path) built the CRV1 finite-sample factor
`(N−1)/(N−K) · G/(G−1)` with `K` counting **every** absorbed fixed-effect level
(plus the regressors). When a fixed-effect dimension is fully nested in the
cluster variable — the canonical `absorb(unit + time) + cluster(unit)` case,
where each `unit` maps to exactly one cluster — the cluster-robust sandwich
already accounts for arbitrary within-cluster correlation, so counting that
FE's `(G−1)` levels again in `K` double-penalises the degrees of freedom and
inflates the standard error. The backend now detects nested dimensions (every
FE level maps to a single cluster level) and drops their levels from the cluster
DOF, matching Stata `reghdfe`, `pyfixest`, and `sp.feols`. Non-nested
dimensions (e.g. `time` under `cluster(unit)`) are charged exactly as before;
when *all* absorbed FEs are nested, one degree of freedom is retained for the
intercept.

**Who is affected.** Anyone reading clustered standard errors / t-stats /
p-values / CIs from `sp.hdfe_ols(..., cluster=…)` or `sp.absorb_ols(...,
cluster=…)` where an absorbed FE is nested in the cluster — the very common
`absorb(entity, time) + cluster(entity)` design. The inflation grew with the
ratio of absorbed FE levels to `N`: ≈5.4% on the reporter's MRE panel and ≈6.3%
on a 37,869-row firm-year panel. Point estimates and `iid` / `hetero` SEs are
unchanged; only clustered SEs change, and they get **smaller** (less
conservative), so results that were marginally non-significant may now cross
conventional thresholds.

**Action required.** Re-run any `sp.hdfe_ols` / `sp.absorb_ols` fits that
combined absorbed FEs with clustering on (or nested in) one of those FE
dimensions; the previous clustered SEs were systematically too large. The
corrected FE dof charged to CRV1 is exposed as `dof_fe_cluster`, and the
detected nested dimensions as `nested_fe` (in `cluster_info`) /
`nested_fe_in_cluster` (raw result), so you can confirm what was reclassified.
`sp.feols` (pyfixest backend) was already correct and is unchanged.

---

<a id="feols-nofe-weights"></a>

## 1.18.0 — ⚠️ `sp.feols` ignored `weights=` when no fixed effects were absorbed

**What changed.** Called with regressors but **no** fixed effects, `sp.feols`
took an intercept-only OLS fallback that accepted the `weights=` argument but
never used it — the fit was unweighted. As of 1.18.0 the fallback solves the
weighted least squares (WLS) normal equations, so `weights=` now changes the
coefficients, standard errors, and R² exactly as a weighted regression should.

**Who is affected.** Anyone who called `sp.feols(..., weights=w)` **without**
fixed effects. Calls *with* fixed effects were already weighted correctly and
are unaffected; calls without `weights=` are numerically identical.

**Action required.** Re-run any no-FE weighted `sp.feols` fits — prior results
were the unweighted OLS solution. The new path also raises on non-finite,
negative, or zero-total-mass weight vectors instead of silently proceeding.

---

<a id="evalue-hr-ci-parity"></a>

## 1.18.0 — ⚠️ `sp.evalue` HR / CI E-value parity with R `EValue`

**What changed.** Two numerical behaviours of `sp.evalue` (and
`sp.evalue_from_result`) changed so that StatsPAI now reproduces the R
`EValue` package exactly (#21):

1. **Hazard ratios.** `measure='HR'` was always treated as a *rare-outcome*
   ratio (`OR ≈ RR ≈ HR`). It now uses the exact common-outcome conversion
   `(1 − 0.5^√HR)/(1 − 0.5^√(1/HR))` by default, matching `EValue::evalues.HR`.
   HR E-values change for non-rare outcomes.
2. **Confidence intervals that cross the null.** The CI E-value is now exactly
   `1` whenever the interval already contains the null (or a user-supplied
   `true` value), instead of a spurious value > 1 computed from the limit.

**How to get the old numbers.** Pass `rare=True` for the rare-HR
approximation. There is **no** flag to restore the un-clamped CI E-value — the
old value was incorrect (it claimed "confounding needed" for a result already
compatible with the null).

**Parameter rename.** `rare_outcome` → `rare`. The old name still works (emits
`DeprecationWarning`) and will be removed no earlier than the next minor.

**Who is affected.** Anyone computing an E-value from a hazard ratio with a
non-rare outcome, or reading the CI E-value of a non-significant result. RR-
and OR-based point E-values are unchanged.

**JOSS / JSS.** This is parity module `23_evalue` in the JSS cross-language
table (`Paper-JSS/manuscript/tables/appendix_b_parity.tex`); the change
*increases* agreement with R `EValue` and the row remains a machine-precision
**PASS** (worst relative difference 5.8e-14 over 26 rows). No JOSS (#10604)
numeric figure uses an HR or CI E-value.

---

<a id="matching-nearest-tie-break"></a>

## 1.18.0 — ⚠️ `sp.match` nearest-neighbor tie-breaking stabilised

**What changed.** `sp.match(method='nearest')` now resolves exact equal-distance
nearest-neighbor ties by the source DataFrame index. Previously the
Euclidean/propensity nearest-neighbor path delegated tie selection to
`argpartition` and incidental row order, so ties on discrete or binary
covariates could move the ATT across environments. Lower-index control units
are now selected first; when matching without replacement and multiple treated
units have the same best distance, lower-index treated units are assigned first.

**Who is affected.** Only users whose matching data contain exact
equal-distance ties. Continuous covariates without exact ties are unchanged.
For tied designs, results are now deterministic across row order and backend as
long as the DataFrame index preserves unit identity. One caveat: distances that
are merely *near*-equal (differing at the ~1e-13 ULP level because the BLAS
build computes propensity scores slightly differently) are still resolved by
strict comparison, so a residual backend sensitivity of that magnitude remains
— on LaLonde it amounts to ~$4.5 (vs. ~$150 before the fix), and all GitHub CI
platforms (ubuntu/windows/macos) now agree bitwise.

**Action required.** None for code. If you previously recorded a nearest-match
estimate on tied discrete covariates, re-run it once and treat the new value as
the stable pin. The bundled LaLonde 1:1 NN PSM guard now pins the two observed
fixed points exactly (`1967.94` on GitHub CI, `1963.43` under Accelerate on
macOS 26) instead of allowing the old ~$300 cross-backend tie band.

---

<a id="blp-maxiter-fix"></a>

## 1.18.0 — ⚠️ `sp.blp` functionality fix (was non-functional)

**What changed.** `sp.blp` (BLP random-coefficients logit demand) now runs.
Previously its GMM inner loop called `_gmm_objective(..., maxiter=1000)` while
the parameter is named `maxiter_inner`, so **every** `sp.blp` call raised
`TypeError: _gmm_objective() got an unexpected keyword argument 'maxiter'` as
soon as the outer optimiser evaluated the objective — i.e. on every estimation
path (`contraction`, `mpec`, `gmm`).

**Who is affected.** Anyone who tried to call `sp.blp`. Because the function
produced *no* output before (it crashed), this fix cannot move any
previously-correct number — JOSS (#10604) / JSS dossier figures are unaffected.
`sp.blp` (and `BLPResult`) appears in the JSS manuscript only as a
function-inventory catalog row (`function_inventory_full.tex`), never in any
numeric or parity table, and the fix does not change that row. **Note the name
collision:** the "BLP" entries in the JSS parity-change log (`05-parity.tex`,
`05-parity-compact.tex`) refer to the **Best Linear Projection of CATE**
(`best_linear_projection` / `blp_test` / `test_calibration`,
Chernozhukov-Demirer et al.) — a different feature, not this
Berry-Levinsohn-Pakes demand estimator. A regression guard now recovers the
known linear price/characteristic coefficients on a pure-logit DGP
(`tests/test_tierD_structural_analytic.py::TestBLPAnalytic`).

**Action required.** None, beyond noting that `sp.blp` is now usable. Found by
the Tier D analytic special-case test campaign (CLAUDE.md §5).

<a id="dag-dseparation-fix"></a>

## 1.18.0 — ⚠️ d-separation corrected (forks & colliders)

**What changed.** `statspai.dag`'s d-separation engine (`_d_separated`, behind
`DAG.d_separated`, `adjustment_sets`, `backdoor_paths`, `do_rule1/2/3`,
`do_calculus_apply`, `swig`, `dag_recommend_estimator`) moralised the ancestral
graph incorrectly — it married *siblings* instead of *co-parents*. So the two
non-trivial d-separation rules were backwards: conditioning on a common cause
did **not** block a fork (`A ⊥ C | M` on `M→A, M→C` wrongly returned `False`),
and conditioning on a collider did **not** open it (`A ⊥ C | K` on `A→K←C`
wrongly returned `True`). Chains were unaffected. Moralisation now connects
every pair of a node's parents, so all three canonical structures and the
adjustment-set / do-calculus routines built on them are correct.

**Who is affected.** Anyone who used `DAG.d_separated`, `adjustment_sets`,
`backdoor_paths`, the do-calculus rule checkers, `swig`, or
`dag_recommend_estimator`. **Re-derive any adjustment sets / identification
conclusions** obtained from these — previous fork/collider answers were
unreliable. No API change. None of the package's reference-parity or
JOSS/JSS-dossier numbers come from these graph routines, and all nine
dag-touching test files pass unchanged (none had encoded the broken behaviour).

**Action required.** None code-wise; re-check any DAG-derived adjustment sets.
Found by the Tier D analytic special-case campaign (CLAUDE.md §5); guard:
`tests/test_tierD_p2_dag_dsep_analytic.py`.

<a id="granger-wald-variance-fix"></a>

## 1.18.0 — ⚠️ `sp.granger_causality` test statistic corrected

**What changed.** `sp.granger_causality` now computes the correct Wald
statistic. The coefficient covariance used in the test was a placeholder
`V = sigma2 * I` (the caused equation's residual variance, not its coefficient
covariance), which omitted the design-matrix factor `(X'X)⁻¹`. The reported
F-statistic was too small by a factor of roughly `T·Var(regressors)`, so the
test essentially never rejected — even a textbook-strong lagged link went
undetected (true F≈326 reported as ≈0.36). `VARResult` now stores `(X'X)⁻¹` and
the test forms `Var(β̂_caused) = σ²_caused·(X'X)⁻¹`; the F now equals the
standard restricted-vs-unrestricted OLS F-test.

**Who is affected.** Anyone who called `sp.granger_causality` (directly or via
`VARResult.granger_test`). **Re-run any Granger conclusions** — prior runs
almost certainly failed to reject and were not trustworthy. There is no API
change. No JOSS (#10604) / JSS table uses this function, and the previous output
was statistically meaningless, so no valid published result is invalidated.

**Action required.** None code-wise; re-run Granger tests and expect them to
detect real causal directions now. Found by the Tier D analytic special-case
campaign (CLAUDE.md §5); guard: `tests/test_tierD_p2_timeseries_analytic.py`.

<a id="ols-qr-kernel"></a>

## 1.18.0 — ⚠️ OLS kernel switched to a QR solve (numerical accuracy)

**What changed.** The core OLS kernel — `ols_fit` for coefficients and
`OLSEstimator.estimate` for the variance-covariance matrix, both in
`src/statspai/` — now solves least squares via the **QR factorisation** of the
design matrix `X = QR` (`b = R⁻¹Qᵀy`, `(X'X)⁻¹ = R⁻¹R⁻ᵀ`). The previous
implementation solved the normal equations `(X'X) b = X'y` and formed
`inv(X'X)` directly. Forming `X'X` squares the condition number of `X`, so on
ill-conditioned designs roughly half of the available digits are lost — and on
the worst cases the result is meaningless.

**Why.** The new NIST StRD certification suite
(`tests/numerical_accuracy/test_nist_strd_ols.py`) showed the normal-equations
path produced **0 correct digits** on the NIST Filippelli dataset (a degree-10
polynomial fit, `cond(X) ≈ 1e10`) and only ~6 digits on several Wampler
polynomials. The QR path tracks `cond(X)` rather than `cond(X)²` and lifts
those to ~7 and ~9–13 digits respectively, matching the published certified
values.

**Who is affected.**

- **Well-conditioned regressions (the overwhelming majority): no action.**
  Coefficients, standard errors, R², F all match the old output to ≈1e-12 —
  far below any reporting precision. The full `reference_parity` and
  `external_parity` (JOSS reproduction) suites pass unchanged.
- **Regressions on near-collinear or high-degree-polynomial designs:** you will
  now get **different — and correct — numbers**. If you previously fit, say, a
  high-order polynomial trend or a strongly collinear specification directly
  (without centring/orthogonalising) and recorded the coefficients, re-run and
  expect them to move. The old numbers were the unstable ones.

There is **no API change** and nothing to rewrite; this note exists only so the
numerical shift on ill-conditioned designs is on the record.

Separately, exact-fit OLS (`R² == 1`) now reports the F-statistic as `inf`
(matching NIST's certified "Infinity") instead of emitting a divide-by-zero
`RuntimeWarning`; non-exact fits are unaffected.

Separately, `sp.regress` now fits intercept models in mean-centred
(Frisch-Waugh-Lovell) coordinates and reconstructs the intercept afterward.
This is algebraically identical to the raw fit in exact arithmetic, but it
avoids catastrophic cancellation when `y` or a regressor has a very large
constant offset. Well-conditioned fits remain unchanged to machine precision;
the visible effect is on pathological offset designs such as the NIST StRD
ANOVA `SmLs07/08/09` cases, where F/R² accuracy improves down to the float64
input-representation floor.

---

<a id="regress-collinearity-guard"></a>

## 1.18.0 — ⚠️ `sp.regress` raises on perfect collinearity; `sp.logit`/`sp.probit` warn on separation

**What changed.** Two silent-failure corners now fail loudly:

- **Perfect collinearity** in `sp.regress` — duplicate or proportional
  regressors, the dummy-variable trap (complementary 0/1 dummies plus an
  intercept), or a constant non-intercept regressor — previously returned
  enormous unidentified coefficients (e.g. `~1e14`) with no warning. It now
  raises `statspai.exceptions.NumericalInstability`, with the offending columns
  in `error.diagnostics`.
- **Perfect / quasi-complete separation** in `sp.logit` / `sp.probit` —
  where the outcome is perfectly predicted and the maximum-likelihood estimate
  does not exist — previously returned large finite coefficients with no
  signal. It now emits a `statspai.exceptions.ConvergenceWarning`.

**Why.** The project rule is "fail loudly": returning wrong numbers silently is
the cheapest way to hide a correctness problem. Neither case has a meaningful
answer to return.

**What you need to do.**

- If you *intended* collinear regressors, drop one (or remove the intercept for
  a single constant regressor). The exception names them.
- For separation, use penalized (Firth) logistic regression, drop the
  separating predictor, or pool sparse categories.

**Scope / non-goals.** Collinearity detection is deliberately *structural*
(duplicate/proportional columns, zero-variance regressors), not based on the
condition number or matrix rank. A rank tolerance loose enough to catch real
collinearity also flags legitimately ill-conditioned but full-rank designs —
the NIST StRD Filippelli benchmark is numerically *more* singular
(`s_min/s_max ~ 6e-16`) than an exactly duplicated column yet must fit. So a
general exact linear dependence among 3+ columns that is not reducible to a
pairwise duplicate or a constant column is **not** auto-detected; inspect the
design's condition number if you suspect one.

---

<a id="drdid-traditional-normalisation"></a>

## 1.17.0 — ⚠️ `sp.drdid(method='trad')` ATT correctness fix

**What changed.** The traditional doubly-robust DiD branch of `sp.drdid`
(Sant'Anna & Zhao 2020) divided each of its four cell terms — treated/control ×
post/pre, each a weighted average of the outcome-regression residual — by the
**full sample size** `n` rather than by that cell's weight mass. On a balanced
2×2 each cell holds ~¼ of the sample, so every term was scaled down by its
sample share and the ATT was biased toward zero by ~50%. Concretely, on a 2×2
with true ATT 2.0 (raw DiD 1.96) `method='trad'` returned ≈1.04. Each term is
now normalised by its own weight total. The traditional estimator therefore now
reduces **exactly** to the raw 2×2 DiD when no covariates are supplied, and
recovers the true ATT with covariates.

Separately, `sp.drdid` now **raises `ValueError`** when `method` is neither
`'imp'` nor `'trad'`. Previously any other string (e.g. `'ipw'`, `'reg'`,
`'dr'`) fell through silently to the traditional branch; such calls were never
distinct estimators and now fail loudly.

**Who is affected.** Callers of `sp.drdid(..., method='trad')` (or any non-`imp`
string, which silently ran the traditional branch). The **default**
`method='imp'` (improved, locally efficient) already normalised correctly and
is **unchanged** — its point estimates, standard errors, the
source-audit / R-`DRDID` parity numbers (which pin
`drdid_imp_panel`), and every other default-path result are **not** affected.

**Action.** If you relied on `method='trad'` output, re-run; the corrected ATT
is ~2× the previously reported (downward-biased) value and now matches the raw
DiD / `method='imp'`. Replace any `method` value other than `'imp'`/`'trad'`
with one of those two.

---

<a id="multiway-cluster-intersection"></a>

## 1.17.0 — ⚠️ `sp.multiway_cluster_vcov` multiway-cluster SE correctness fix

**What changed.** `sp.multiway_cluster_vcov` forms the Cameron-Gelbach-Miller
(2011) variance by inclusion-exclusion over the clustering dimensions, which
requires an *intersection* cluster: the unique combinations of the dimensions'
levels (e.g. the distinct `(firm, year)` pairs). The intersection key was built
by joining the dimensions into one string with a `"\0"` separator, but NumPy
fixed-width unicode strips the embedded NUL byte, so `(1, 23)` and `(12, 3)`
both collapsed to `"123"`. On a 40×50 crossed-cluster DGP this merged 1733 true
intersection clusters into 1639, which inflated the `G/(G-1)` finite-sample
factor on the subtracted intersection term and biased the multiway SE by ~0.2%
(two-way) to ~0.5% (three-way) away from the canonical estimator. The
intersection key is now built collision-free via `np.unique(axis=0)` on
per-dimension integer codes. `sp.multiway_cluster_vcov` now reproduces
`sandwich::vcovCL(cluster = ~ g1 + g2 + ...)` and `sp.twoway_cluster` to machine
precision (two-way exact; three-way relative error ~4e-7).

**Who is affected.** Callers of `sp.multiway_cluster_vcov` with **two or more**
clustering dimensions, and the multiway-clustered standard errors of
`did.harvest` and `panel.feols`. One-way clustering is unaffected.
`sp.twoway_cluster` is **not** affected — it used a separate, collision-free
intersection key and already matched `sandwich::vcovCL` to machine precision.
Point estimates are unchanged; only multiway-cluster SEs/CIs/p-values move
(typically by tenths of a percent, always toward the canonical value).

**Action.** Re-run any analysis that reported `sp.multiway_cluster_vcov`-based
multiway-cluster SEs with ≥2 dimensions; the corrected SEs now agree with
`sandwich::vcovCL` (R) and Stata multiway-cluster conventions.

---

<a id="structural-break-supf-null"></a>

## 1.17.0 — ⚠️ `sp.structural_break` sup-F p-value null distribution correctness fix

**What changed.** The sup-F / Chow statistic in `sp.structural_break(...)` is a
*supremum* of the Chow F statistic over all candidate break points. Under the
null of no break it therefore follows the Andrews (1993) sup-F limiting law,
**not** the ordinary `F(k, n-2k)` distribution. The previous code referred the
maximised statistic to the F CDF (`1 - scipy.stats.f.cdf(best_f, k, n-2k)`),
which ignored the search over break points and produced p-values that were far
too small. Measured false-positive rate on Gaussian white noise at the 5%
level: **33–37%** (n ∈ {100, 200, 400}) — a roughly 7× inflation. P-values are
now drawn from the Andrews (1993) null (a q-vector Brownian-bridge functional,
sampled by a deterministic seeded simulation cached per `(q, grid, trimming)`),
which restores **nominal size (~5%)** with no material loss of power. The same
correct critical value now governs the Bai-Perron sequential `supF(l+1|l)`
stopping rule, so `method='bai-perron'` stops over-segmenting noise.

**Who is affected.** Anyone who relied on the `p_values` / `break_dates` of
`sp.structural_break` with `method` in `{'sup-f', 'chow', 'bai-perron'}`.
Previously-reported breaks (and their tiny p-values) were anti-conservative;
some "significant" breaks were spurious. The point estimate of the *location*
of the most likely break (`break_dates[0]` for sup-F) is unchanged — only its
significance and the break **count** change.

**What to do.** Re-run any structural-break tests and re-check significance
against the corrected p-values. A break that was marginal under the old
(inflated) test may no longer reject. `method='bai-perron'` may now return
fewer breaks. The result object additionally gained populated `f_stats` /
`p_values` for the Bai-Perron path (one entry per detected break, sorted by
date), where it previously returned `None`.

**Reference.** Andrews, D.W.K. (1993). "Tests for Parameter Instability and
Structural Change with Unknown Change Point." *Econometrica*, 61(4), 821-856.
doi:10.2307/2951764 (verified via Crossref, the Econometric Society, and
RePEc).

---

<a id="msm-singleperiod-iptw"></a>

## 1.17.0 — ⚠️ `sp.stabilized_weights` / `sp.msm` single-period IPTW correctness fix

**What changed.** On a **single-period (point-treatment) panel**,
`sp.stabilized_weights(...)` (and therefore `sp.msm(...)`) previously returned
stabilized weights that were all exactly `1.0`. The within-unit lagged-
treatment column is all-zero in that setting, which made the logistic
treatment-model design singular; the failure was silently caught and the
weights fell back to the marginal mean for both the numerator and denominator,
cancelling to `1.0`. The MSM then silently reduced to an unweighted,
**confounded** regression. The fix drops zero-variance columns before fitting,
so the confounders are now used and the weights are computed correctly.

**Who is affected.** Anyone who called `sp.stabilized_weights` / `sp.msm` on a
panel with **one period per unit** (point treatment). Multi-period panels —
the intended MSM use case — are **unaffected** (their weights already varied
correctly and are numerically identical before and after).

**What to do.** Re-run any single-period MSM analyses: the previous output was
equivalent to an unadjusted regression and should not be relied on. The fixed
weights match a textbook stabilized-IPTW computation to machine precision. If
the treatment model genuinely cannot be fit (e.g. perfect separation), you now
get a `RuntimeWarning` instead of a silent fallback.

---

<a id="sp-synth-default-classic"></a>

## 1.16.1 — ⚠️ `sp.synth()` default method restored to `'classic'`

**What changed.** A bare `sp.synth(...)` call (no `method=`) now runs
`method='classic'` — canonical Abadie–Diamond–Hainmueller (2010) synthetic
control with convex, non-negative, sum-to-one donor weights. The signature
default had silently drifted to `method='augmented'` (Augmented SCM,
Ben-Michael, Feller & Rothstein 2021), which deliberately allows negative
donor weights by extrapolating outside the donor convex hull. That
contradicted the documented default (`sp.synth` docstring: `method : str,
default 'classic'`), the migration-from-R mapping (`Synth::synth` is
classic), and the canonical Prop99 examples shipped in the docs.

**Who is affected.** Anyone calling `sp.synth(...)` **without** an explicit
`method=`. Estimated effects, donor weights, and synthetic trajectories
revert from ASCM to classic SCM. Every call that already passes `method=`
(including `method='augmented'` / `'ascm'`) is **unchanged**.

**What to do.** To keep Augmented SCM, pass it explicitly:

```python
res = sp.synth(df, outcome=..., unit=..., time=...,
               treated_unit=..., treatment_time=..., method='augmented')
```

Otherwise no action is needed — the default now matches the documentation and
the R `Synth` reference implementation.

Guarded by
`tests/test_synth.py::TestSyntheticControl::test_weights_non_negative`.

---

<a id="sp-causal-forest-aipw-fix"></a>

## 1.16.0+source.20260531 — ⚠️ Causal-forest ATE/ATT now doubly-robust (AIPW)

**What changed.** `CausalForest.average_treatment_effect(...)` previously
returned a plug-in average of the forest's CATE predictions. Forest
regularisation shrinks those predictions, so the plug-in mean is biased
(≈ 15 % high on a clean-overlap design) and is *not* the estimand
`grf::average_treatment_effect` reports. It now returns the doubly-robust
AIPW influence-function mean built from the forest's own cross-fitted
nuisances (`Γ_i = τ̂ + (T−ê)/(ê(1−ê))·(Y − m̂ − (T−ê)τ̂)`), with the
influence-function standard error `sd(Γ)/√n`.

**Who is affected.** Anyone reading
`cf.average_treatment_effect(...)['estimate']` or `['se']` (any
`target_sample`: `all`/`treated`/`control`/`overlap`). The plug-in
convenience methods `cf.ate()` / `cf.att()` are **unchanged**.

**What to do.** Re-run any analysis that reported a causal-forest ATE/ATT
from `average_treatment_effect`. The new estimate is closer to truth and
agrees with `grf` within combined Monte Carlo error.

```python
ate = cf.average_treatment_effect(target_sample="all")  # ['method']=='aipw'
ate_plugin = cf.ate()                                   # still available, plug-in
```

Guarded by `tests/reference_parity/test_causal_forest_aipw_recovery.py`
and `tests/reference_parity/test_grf_parity.py`.

---

## 1.16.0 — ⚠️ `sp.xtabond` Arellano-Bond GMM correctness fix

**What broke.** `sp.xtabond` (and `sp.panel(method='ab')`) used a flat,
fixed block of lagged-level instrument columns and then dropped every
row that was missing any of them — on a short panel this discards most
of the sample — and weighted with `W = (Z'Z)⁻¹`. The correct
Arellano-Bond estimator uses a **block-diagonal** GMM instrument matrix
(each available deeper lag `Y_{i,s}`, `s ≤ t-2`, is a period-specific
moment; missing lags are zero-filled, no rows dropped) and the one-step
weight `W = (Σᵢ Zᵢ'H Zᵢ)⁻¹`, with `H` the first-difference MA(1)
structure (2 on the diagonal, −1 on the first off-diagonals). The old
code returned `β_{y₋₁}=0.264 (se 0.224)` where Stata returns
`0.391 (se 0.046)` — a 48 % estimate gap and an 80 % SE gap.

**Who is affected.** Anyone who called `sp.xtabond(...)` or
`sp.panel(..., method='ab'|'system')` on an earlier release. **Both the
point estimates and the standard errors change** — point estimates are
*not* preserved here (unlike the qreg fix).

**What to do.**

| Surface | Pre-fix | Action |
| --- | --- | --- |
| `res.estimate`, `detail["coefficient"]` | biased (instrument set wrong) | Rerun |
| `res.se`, `detail["se"]`, `res.ci`, `res.pvalue` | wrong | Rerun |
| `gmm_lags` default | `(2, 5)` | now `(2, None)` = all deeper lags (Stata default); pass an explicit max to cap |
| `method='system'` | returned a number | now raises `NotImplementedError`; use `method='difference'` |
| `twostep=True` SEs | uncorrected | now Windmeijer (2005)-corrected when `robust=True` |

**Verification.** One-step robust `sp.xtabond` now matches Stata
`xtabond y x, lags(1) vce(robust)` to machine precision on the parity
DGP (`tests/r_parity/50_xtabond`, rel ≈ 1e-15 on both β and SE);
guarded by `tests/test_gmm.py::TestArellanoBond::test_parity_matches_stata_xtabond`.

---

<a id="sp-qreg-se-fix"></a>

## 1.16.0 — ⚠️ `sp.qreg` Powell sandwich SE correctness fix

**What broke.** The Powell (1991) kernel sandwich for quantile
regression standard errors was implemented with an extra factor of
`n` in the denominator: `V = τ(1−τ) / (n · f̂(0)²) · (X'X)⁻¹`. The
textbook formula (Koenker 2005, eq. 3.7) is
`V = τ(1−τ) / f̂(0)² · (X'X)⁻¹` — no `n`. The reported SE was
therefore the correct SE divided by √n. On the parity dataset with
n = 500 (`tests/r_parity/40_qreg`), the bug under-reported SE by
~20× and produced z-statistics in the 6–30 range for null
covariates.

**Who is affected.** Anyone who used the `se`, `pvalue`, `ci`, or
`z` columns of `sp.qreg(...).detail` (or the top-level `res.se` /
`res.pvalue` / `res.ci`) on an earlier release. Point estimates
(`res.estimate`, `detail["coefficient"]`) are **unchanged at machine
precision** and do not need to be rerun.

**What to do.** Pull the patch, then rerun any analysis that
referenced an `sp.qreg` standard error. Concretely:

| Surface | Pre-fix value | Action |
| --- | --- | --- |
| `res.se`                                       | SE / √n   | Multiply by √n to recover, or just rerun |
| `res.pvalue`                                   | ~0        | Rerun — most pre-fix p-values were spuriously zero |
| `res.ci`                                       | too narrow | Rerun |
| `res.detail["se" / "z" / "pvalue"]`            | as above  | Rerun |
| `res.estimate`, `res.detail["coefficient"]`    | correct   | No change needed |

**Verification.** The cross-language parity table in
`tests/r_parity/results/parity_table_3way.md` for module `40_qreg`
shows the post-fix SE matching `quantreg::rq` (Powell `nid` kernel)
within 1.4–6.8 % and Stata `qreg` (Koenker-Bassett) within 2.9 %.
This is the expected residual gap between three different
implementations of the same sandwich.

**Why was it not caught earlier.** No 3-way Stata parity test
existed for quantile regression before the 2026-05-28 session, and
the unit tests in `tests/test_quantile.py` checked only point
estimates and that SEs were finite — never against an external
reference value.

---

<a id="sp-rdrobust-bwselect-cct-r-parity-opt-in"></a>

## v1.15.2 → v1.15.3 — doc-only PyPI hero-banner fix

**No code changes, no migration step.** The v1.15.2 PyPI project page
rendered the hero banner as a broken image because the `<img>` tag in
`README.md` / `README_CN.md` used a repo-relative path
(`docs/logo/readme-1.png`) that PyPI's long-description renderer
cannot resolve. v1.15.3 swaps the path for the absolute raw GitHub
URL so the banner loads on PyPI / TestPyPI / off-GitHub mirrors.
Module hashes match v1.15.2 bit-for-bit; only the long-description
metadata baked into the wheel + sdist changes.

---

## v1.15.1 → v1.15.2 — strict-JSON MCP wire, dual-track replicate, packaging

**No estimator numerical path changes.** Three classes of consumers
should take note:

- **`sp.agent.mcp_server` clients** (Claude Desktop / Codex / any
  RFC 8259-strict JSON parser). v1.15.1 could leak the non-standard
  literals `NaN` / `Infinity` / `-Infinity` into responses whenever an
  estimator surfaced a degenerate float (`np.nan` standard errors on a
  singular covariate, `inf` log-likelihood on a saturated model, etc).
  v1.15.2 walks all containers before `json.dumps` and serialises with
  `allow_nan=False`, replacing those values with `null`. **Action**:
  none — strict parsers that previously failed now succeed; lenient
  parsers see `null` where they used to see `NaN`. Update your
  downstream JSON Schema if it explicitly typed those fields as
  `number` (they should be `["number", "null"]`).

- **`sp.causal_text` users.** The MVP relied on a soft import of
  `sentence-transformers`. v1.15.2 adds an explicit
  `pip install statspai[text]` extra. The lazy import path is
  preserved, but the `ImportError` message now points at the extra
  instead of suggesting a bare `pip install sentence-transformers`.

- **`sp.replicate` users.** Entries for Card (1995), Abadie-Diamond-
  Hainmueller (2010), Lalonde (1986) / DW (1999), and Lee (2008) now
  return classic + modern recipes computed on the bundled real CSVs
  instead of single-track simulated stubs. If you were pinning to the
  v1.15.1 simulated numbers in CI, switch to the published-paper
  benchmarks now exposed via `df.attrs['paper_original']` (see
  `sp.datasets.nsw_lalonde(simulated=False)` and
  `sp.datasets.lee_2008_senate(simulated=False)`).

Existing `sp.rdrobust` / `sp.nbreg` / `sp.xtnbreg` / `sp.menbreg`
call sites carry over unchanged from v1.15.1.

---

## v1.15.0 → v1.15.1 — `sp.rdrobust(bwselect='cct')` R-parity opt-in

**No breaking change.** `sp.rdrobust` keeps `bwselect='mserd'` (StatsPAI's
own MSE-optimal recipe) as the default — every existing call returns the
same numbers. A new opt-in value `bwselect='cct'` is added for users who
need bit-equal R `rdrobust::rdrobust` parity.

`sp.nbreg`, `sp.xtnbreg`, and `sp.menbreg` also get clearer README /
release-note documentation in v1.15.1. Their call signatures and
numerical paths are unchanged, so there is no migration step for
negative-binomial regression users.

### When to switch from `'mserd'` to `'cct'`

Use `bwselect='cct'` when **any** of these apply:

- You're replicating a CCT 2014 / Cattaneo-Idrobo-Titiunik (2018, 2020)
  paper and need the published numbers to the 4th decimal.
- A reviewer asks for "the same number R `rdrobust` gives".
- Your data has features that stress StatsPAI's internal pilot bandwidth
  (heavy tails, small `n`, mass points). On the canonical Lee/CCT Senate
  replication, `'mserd'` gives `Conv = 12.62 / h = 4.6` while `'cct'`
  gives `Conv = 7.41 / h = 17.75` — the latter matches R bit-equal.

Keep the default `bwselect='mserd'` when:

- You don't need exact R parity, **and**
- You don't want a soft dependency on the `rdrobust` package, **and**
- Your downstream tests / pipelines have already been calibrated against
  StatsPAI's `'mserd'` numbers.

### How to switch

```python
import statspai as sp

# Before — StatsPAI internal MSE-optimal (kept stable)
res = sp.rdrobust(data=df, y='y', x='x', c=0)
# After — R-bit-equal via official rdrobust delegation
res = sp.rdrobust(data=df, y='y', x='x', c=0, bwselect='cct')
```

Install the optional dependency once:

```bash
pip install statspai[rd-cct]   # adds rdrobust>=1.3
```

Calling `bwselect='cct'` without it raises a clear `ImportError` that
points you to the install command — no silent fallback.

### Why we didn't change `'mserd'` itself

Aligning the internal `'mserd'` to R `rdbwselect`'s recursive 3-step
recipe would shift point estimates on every dataset that exercises
StatsPAI's RD path (5+ test classes, `r_parity` scripts, downstream
docs / notebooks). The additive `'cct'` route gives anyone who wants R
parity an immediate path **and** preserves the 1.x line's numerical
stability. A future major version may flip the default.

---

## v1.11 → v1.12 — DML module hardening

`sp.dml`, `sp.dml_panel`, `sp.dml_model_averaging` keep all of their
existing call signatures (every old script imports the same way and
runs without code changes), but several internal numerical behaviours
shift on the boundaries of the input space. The full release-note
discussion lives in [`CHANGELOG.md`](CHANGELOG.md) under
`[1.12.0]`; the breaking points are summarised here.

### What can change in your numbers

| Estimator | What changed | When you'll notice |
| --- | --- | --- |
| `sp.dml(model='irm')` | `KFold` → `StratifiedKFold` (stratified by D). Empty subgroup folds were silently filled with zeros for `g(1, X)` / `g(0, X)`; they now raise `IdentificationFailure`. | Small N, imbalanced D, or small `n_folds` may give point estimates a hair different from before — folds are no longer drawn from the un-stratified KFold sequence. |
| `sp.dml(model='iivm')` | Same — `StratifiedKFold` on Z, plus empty-subgroup `IdentificationFailure`. | Small N or imbalanced Z. |
| `sp.dml(model='pliv')` | Weak-IV floor on the ML-residualised partial correlation: `1e-6 → 1e-3`. | When your instrument's first-stage corr after ML residualisation is in `[1e-6, 1e-3]`, the call now raises `RuntimeError` with a clear hint to consult `sp.weakrobust` / `sp.anderson_rubin_test`. |
| `sp.dml_model_averaging` | Default `weight_rule="inverse_risk"` → `"short_stacking"`. | Different default point estimate. To preserve the v1.11 number, pass `weight_rule="inverse_risk"` explicitly. |
| `sp.dml_model_averaging` | NaN rows in `y` / `treat` / `covariates` are now dropped instead of being passed to sklearn. | If your data had NaNs you may have been getting `RuntimeError("No candidate produced a finite estimate")` or, worse, NaN θ̂; now you'll silently lose those rows but the estimate will be finite. The dropped count is reported in `model_info["n_dropped_missing"]`. |
| `sp.dml_panel(binary_treatment=True)` | Now a deprecated no-op — the previous classifier path was incorrect. The estimator runs as `binary_treatment=False` (regressor on D̃) regardless. | Different θ̂ when you used `binary_treatment=True`; a `DeprecationWarning` fires so you see it. |

### Recovering the v1.11 default for `dml_model_averaging`

```python
# v1.11 default behaviour (inverse-MSE-weighted average of per-candidate θ̂)
result = sp.dml_model_averaging(
    df, y="y", treat="d", covariates=cov_list,
    weight_rule="inverse_risk",   # v1.12 default is "short_stacking"
)

# v1.12 default — Ahrens et al. (2025, JAE) eq. 7 short-stacking
result = sp.dml_model_averaging(
    df, y="y", treat="d", covariates=cov_list,
    # weight_rule="short_stacking" (now the default)
)
result.model_info["weights_g"]   # CLS stacking weights for E[Y|X]
result.model_info["weights_m"]   # CLS stacking weights for E[D|X]
```

### Recovering the v1.11 `dml_panel(binary_treatment=True)` semantics

There is no recovery — the v1.11 path was incorrect (classifier on
within-demeaned features but raw {0,1} labels). For DR-style ATE on
binary D in panels, prefer one of:

```python
# (a) sp.dml IRM with unit dummies as covariates
import pandas as pd
unit_dummies = pd.get_dummies(df["unit"], drop_first=True)
df_aug = pd.concat([df, unit_dummies], axis=1)
sp.dml(df_aug, y="y", treat="d",
       covariates=[*cov_list, *unit_dummies.columns.tolist()],
       model="irm")

# (b) sp.etwfe (extended TWFE for staggered binary treatment in panels)
sp.etwfe(df, yname="y", tname="t", gname="treatment_cohort",
         idname="unit", covariates=cov_list)

# (c) sp.callaway_santanna (staggered DR-DiD)
sp.callaway_santanna(df, yname="y", tname="t",
                     gname="treatment_cohort", idname="unit")
```

### New capabilities (no migration needed — purely additive)

- `sample_weight=` is now accepted on `sp.dml(model='plr' | 'irm')`,
  `sp.dml_panel`, and `sp.dml_model_averaging`. Pass a 1-D array, a
  pandas Series, or a column name. The weighted estimator uses a
  Z-estimator sandwich variance throughout. `sp.dml(model='pliv' | 'iivm')`
  raise `NotImplementedError` if a non-trivial weight is supplied.
- `random_state=` (default 42) on every `sp.dml(model=...)` call
  controls fold assignment deterministically.
- `model_info["diagnostics"]` is populated on every variant — propensity
  distribution, n clipped, subgroup-fallback counts, partial correlation,
  approximate first-stage F, etc.
- String learner aliases (already shipped in 1.11.4) still work:
  `sp.dml(..., ml_g='rf', ml_m='lasso')`.

---

## v1.11 → v1.12 — `esttab` becomes a thin facade over `regtable`

The Stata-style `esttab()` previously shipped a ~500-line
`EstimateTable` class that re-implemented the full renderer pipeline.
PR-B/5c in v1.12 collapses it to a thin facade that translates
Stata-flavoured kwargs and forwards to `sp.regtable`.

**API is unchanged**, including `eststo()` / `estclear()` global store,
`isinstance(x, EstimateTableResult)` type identity, and all
`esttab(*results, se=, t=, p=, ci=, stats=, output=, ...)` keyword
spellings. Rendered output now matches `regtable`'s book-tab style.
A `DeprecationWarning` is emitted on first use; plan to migrate to
`sp.regtable(...)` directly within the next two minor releases.

### Behaviour changes

| Old | New |
| --- | --- |
| `se=True/t=True/p=True/ci=True` exclusive flags | translated to `regtable(se_type='se' \| 't' \| 'p' \| 'ci')`. Priority `ci > p > t > se` if multiple are passed (matches legacy). |
| `output='csv'` | implemented via `result.to_dataframe().to_csv()`. |
| `output='markdown'` / `'md'` / `'tex'` aliases | unchanged, all forward to the corresponding regtable renderer. |
| `filename=` extension auto-detect | unchanged (`.tex` → latex, `.html` → html, `.md` → markdown, `.csv` → csv). |

### Side-by-side migration

```python
# Before — Stata-style stateful workflow
sp.eststo(m1, name="(1)")
sp.eststo(m2, name="(2)")
sp.esttab(stats=["N", "R2", "adj_R2"], output="latex",
          filename="table1.tex")
sp.estclear()

# After — direct regtable call (same LaTeX, no global state)
sp.regtable(
    [m1, m2],
    model_labels=["(1)", "(2)"],
    stats=["N", "R2", "adj_R2"],
    filename="table1.tex",
)
```

---

## v1.11 → v1.12 — `modelsummary` becomes a thin facade over `regtable`

The R-style `modelsummary()` previously shipped a ~700-line renderer
pipeline that re-implemented coefficient extraction, star formatting,
three-line table styling and every export format. PR-B/5b in v1.12
collapses it to a thin facade that translates R-flavoured kwargs and
forwards to `sp.regtable`.

**API is unchanged**, but rendered output now matches `regtable` (book-tab
three-line, publication-quality star legend). A `DeprecationWarning` is
emitted on first use; plan to migrate to `sp.regtable(...)` directly
within the next two minor releases.

### Behaviour changes

| Old | New |
| --- | --- |
| `stars={"*": 0.10, "**": 0.05, "***": 0.01}` | only the threshold *values* are kept; the symbol overrides are dropped (regtable's ladder is `*/**/***` by convention; use `regtable(notation='symbols')` for `†/‡/§`) |
| `se_type='brackets'` | downgraded to parens with `UserWarning`; use `show_ci=True` for `[lo, hi]` if you want brackets to convey actual information |
| `se_type='none'` | downgraded to parens with `UserWarning`; the SE row stays |
| Stat keys `nobs/r_squared/adj_r_squared/f_stat` | translated to regtable canonical (`N`/`r2`/`adj_r2`/`F`) |
| Stat keys `method`/`bandwidth`/`estimand` | silently dropped (modelsummary-only; build a custom `add_rows={}` if needed) |

`coefplot` is unchanged — independent of the table renderer.

### Side-by-side migration

```python
# Before — R-style functional API
sp.modelsummary(m1, m2, m3,
                model_names=["Base", "Mid", "Full"],
                stats=["nobs", "r_squared", "adj_r_squared"],
                output="latex")

# After — direct regtable call (same LaTeX output, full control)
sp.regtable(
    [m1, m2, m3],
    model_labels=["Base", "Mid", "Full"],
    stats=["N", "r2", "adj_r2"],
).to_latex()
```

---

## v1.11 → v1.12 — `outreg2` becomes a thin facade over `regtable`

The Stata-style `OutReg2` class and `outreg2()` function previously
shipped a bespoke 800-line renderer that re-implemented coefficient
extraction, star formatting, three-line table styling, and Excel /
Word / LaTeX export. PR-B in v1.12 collapses that to ~150 lines of
glue that translates Stata-flavoured kwargs and forwards to
`sp.regtable`.

**API is unchanged**, but rendered output now matches `regtable`'s
canonical book-tab style. The visible label changes are listed below.
A `DeprecationWarning` is emitted on first use; plan to migrate to
`sp.regtable(...)` directly within the next two minor releases.

### Label / format changes

| Legacy outreg2 output | New (regtable canonical) |
| --- | --- |
| `Variables` column header | blank (book-tab convention) |
| `R-squared` | `R²` |
| `Adj. R-squared` | `Adj. R²` |
| `Observations` | `N` |
| `F-statistic / Trees` | `F` *(bug fix: "/ Trees" only applied to causal-forest results)* |
| LaTeX missing star legend | proper `\multicolumn` legend below the rule |
| LaTeX `& None & None \\` junk row | gone *(bug fix: spurious empty ATE row)* |

### Removed parameter

| Old | New |
| --- | --- |
| `show_se=False` | no longer supported. Emits `UserWarning`; the SE row stays. Use `sp.regtable(..., se_type='t' \| 'p' \| 'ci')` directly if you need a different cell. |

### Side-by-side migration

```python
# Before — Stata-style stateful builder
o = sp.OutReg2()
o.set_title("Wage Regressions")
o.add_model(m1, "Baseline")
o.add_model(m2, "Full")
o.add_note("Robust SE in parentheses")
o.to_excel("table1.xlsx")

# After — direct regtable call (same Excel output, full control)
sp.regtable(
    [m1, m2],
    title="Wage Regressions",
    model_labels=["Baseline", "Full"],
    notes=["Robust SE in parentheses"],
).to_excel("table1.xlsx")
```

---

## Migrating from `pyreghdfe`

`pyreghdfe` (`pip install pyreghdfe`) is a Python port of Stata's
`reghdfe` maintained as a standalone package. Its scope — multi-way FE
OLS with robust / multi-way cluster SEs, singleton dropping, weighted
regression — is now a strict subset of `sp.hdfe_ols` / `sp.absorb_ols`
in StatsPAI.

### API mapping (pyreghdfe → StatsPAI)

| `pyreghdfe` | StatsPAI (`import statspai as sp`) |
| --- | --- |
| `reghdfe(data=df, y='y', x=['x'], fe=['firm','year'], cluster=['firm'])` | `sp.absorb_ols(y=df['y'].values, X=df[['x']].values, fe=df[['firm','year']], cluster=df['firm'].values, solver='lsmr')` |
| Stata-style formula via pyreghdfe is not supported | `sp.hdfe_ols("y ~ x \| firm + year", data=df, cluster="firm")` (formula interface via pyfixest backend) |
| `solver='lsmr'` / `'lsqr'` | `solver='lsmr'` / `'lsqr'` — same Krylov paths (scipy.sparse.linalg) |
| Krylov-based solvers (LSMR/LSQR) | default `solver='map'` — alternating projections + Irons-Tuck acceleration, typically faster on well-conditioned panels. LSMR/LSQR remain opt-in for pathological FE structures. |
| weighted regression | `weights=` kwarg; LSMR path uses the standard √w transformation on both the sparse design and the response |
| singleton drop | `drop_singletons=True` (default) |
| multi-way cluster SE | `cluster=[firm_arr, year_arr]` (inclusion-exclusion CGM with PSD correction) |

### What you also get

- `sp.ppmlhdfe` — Poisson pseudo-ML with HDFE (not available in `pyreghdfe`).
- Rust-accelerated mean-sweep kernel ([rust/statspai_hdfe/](rust/statspai_hdfe/)).
- Formula interface and unified result object (`summary()`, `to_latex()`, `to_excel()`).
- One-line cross-solver parity check (all three solvers exposed under the
  same API — see `tests/test_hdfe_native.py::test_demean_alt_solver_matches_map_two_way`).

### Numerical parity

Default MAP and `solver='lsmr'` / `'lsqr'` agree on identical data to
`atol=1e-6` on two-way FE OLS (with and without weights, with and
without clustering). See the cross-solver parity suite in
`tests/test_hdfe_native.py`. We do not take a runtime dependency on
`pyreghdfe`; correctness is anchored to scipy's well-established
`scipy.sparse.linalg.lsmr` / `lsqr` plus the internal MAP baseline.

### When to prefer which solver

- **Default (`solver='map'`)**: almost everything. MAP + Aitken is
  typically 2–5× faster than LSMR on canonical firm × year panels.
- **`solver='lsmr'`**: ill-conditioned / highly nested FE structures
  where MAP shows slow convergence (`converged=False`,
  `iters==maxiter`). LSMR is more robust to near-redundancy between FE
  dimensions.
- **`solver='lsqr'`**: exposed for users migrating from code that
  explicitly requested LSQR. For new work prefer LSMR, which scipy
  implements on the same interface and generally offers better
  numerical stability on sparse least-squares.

---

## v1.8.0 → v1.9.0 — Agent-native API surface (no breaking changes)

**Strictly additive release.** Twelve new agent-shaped APIs land
under ``sp.``: ``audit``, ``bib_for``, ``brief``, ``detect_design``,
``examples``, ``preflight``, ``session`` (the seven new top-level
functions), plus ``result.brief()`` / ``result.cite(format=...)``
methods, plus three MCP-server features (``statspai-mcp`` console
script, ``prompts/list``, per-function ``statspai://function/{name}``
resources). **No estimator numerical paths changed**; every
coefficient / SE / CI / p-value is byte-identical to v1.8.0. See
the v1.9.0 [CHANGELOG](CHANGELOG.md#190--agent-native-api-surface-12-modules-across-4-phases)
entry for the full surface.

### Backward-compat invariants the test suite pins

The 422 new tests include explicit regression guards on these
contracts. If your code depended on any of them, nothing changes.

- ``CausalResult.to_dict()`` with no kwargs is **byte-identical**
  to ``to_dict(detail="standard")`` — the legacy default. The new
  ``detail`` parameter is keyword-only and adds three documented
  levels (``"minimal"`` / ``"standard"`` / ``"agent"``).
- ``CausalResult.cite()`` with no kwargs still returns a BibTeX
  string. The new ``format=`` keyword adds ``"apa"`` / ``"json"``
  options without changing the default.
- ``result.for_agent()`` is now a thin alias for
  ``result.to_dict(detail="agent")`` and produces the same dict.
  Existing callers see no change; new code should prefer the
  explicit form for readability.
- ``result.to_agent_summary()`` is unchanged. Its docstring now
  cross-references ``to_dict(detail="agent")`` so future readers
  know the distinction (``to_agent_summary`` is the *nested*
  schema with a ``point`` sub-dict; ``to_dict(detail="agent")`` is
  the *flat* schema). Both round-trip through ``json.dumps``.
- ``execute_tool``'s exception envelope still carries the legacy
  ``error`` / ``tool`` / ``arguments`` / ``remediation`` fields
  unchanged. Two new fields — ``error_kind`` and ``error_payload``
  — are added **only** when the caught exception is a
  ``StatsPAIError`` subclass, so any agent that previously branched
  on ``"error_kind" in out`` to detect structured errors gets a
  clean signal.

### One subtle widening to be aware of

- ``sp.agent.execute_tool``'s default serializer now invokes
  ``r.to_dict(detail="agent")`` instead of ``r.to_dict()``. The
  result dict is a strict superset of the previous shape — every
  pre-1.9 key is still present at the same path; ``violations``,
  ``warnings``, ``next_steps``, and ``suggested_functions`` are
  added. The MCP ``tools/call`` payload is therefore ~3× larger by
  default. Agents that need the smaller form should pass
  ``detail="standard"`` (or ``"minimal"``) in the ``tools/call``
  arguments — the MCP input schema documents this.

### New entry points worth knowing about

- Agents handed unfamiliar data → ``sp.detect_design(df)``.
- Before an expensive call → ``sp.preflight(df, "did", y=..., ...)``.
- After fitting → ``result.brief()`` for dashboards,
  ``sp.audit(result)`` for the missing-evidence checklist,
  ``result.cite(format="apa")`` for prose citations.
- Reproducible RNG → ``with sp.session(seed=42): ...``.
- One-shot install for MCP clients → ``pip install statspai`` now
  exposes ``statspai-mcp`` on PATH (Claude Desktop /
  ``claude_desktop_config.json`` example in
  [agent/mcp_server.py](src/statspai/agent/mcp_server.py)).

---

## v1.6.5 → v1.6.6 — ⚠️ Heckman two-step SE correctness fix (+ HDFE solver option)

**Two-part release.** (1) Correctness fix for `sp.heckman` standard
errors — point estimates unchanged, **SE / t / p / CI change**.
(2) Additive HDFE LSMR/LSQR solver option — all HDFE MAP output is
byte-identical to v1.6.5.

### What changed numerically (Heckman two-step)

`sp.heckman(...)` previously reported an HC1-style sandwich that the
source code itself flagged as
`"Heckman SEs are complex; robust is conservative"`. This was a known
limitation, not a secret bug — but it meant reported SEs, t-stats,
p-values and CIs were off by an amount that depended on (a) how
strongly selection induced heteroskedasticity `σ²(1 − ρ² δ_i)` and
(b) how uncertain the probit first-stage estimate γ̂ was.

v1.6.6 replaces it with the textbook Heckman (1979) / Greene (2003, eq.
22-22) / Wooldridge (2010, §19.6) analytical two-step variance:

```text
V(β̂) = σ̂² (X*'X*)⁻¹ [ X*'(I − ρ̂² D_δ) X* + ρ̂² F V̂_γ F' ] (X*'X*)⁻¹
```

- `X*`: second-stage design matrix including λ̂ as its last column.
- `δ_i = λ̂_i (λ̂_i + Z_iγ̂) ≥ 0` (Mills' ratio inequality).
- `D_δ = diag(δ_i)`; `F = X*' D_δ Z` (`k × q`).
- `V̂_γ = (Z' diag(w_i) Z)⁻¹` with probit information weights
  `w_i = φ(Z_iγ̂)² / [Φ(Z_iγ̂)(1 − Φ(Z_iγ̂))]`.
- `σ̂² = RSS / n_sel + β̂_λ² · mean(δ_i)` (Greene 22-21) —
  replaces the old naive `RSS / (n_sel − k)`.
- `ρ̂² = β̂_λ² / σ̂²`.

`model_info['sigma']` / `model_info['rho']` now also use this
consistent σ̂², so downstream code reading those fields will see
slightly different numbers.

### Who is affected

- Any caller of `sp.heckman(...)` — SEs, t-stats, p-values, CIs change.
- Point estimates `β̂` **do not change** (OLS of y on [X, λ̂]
  is unaffected by the variance formula).
- Callers that pin SE values in their own test suites against a
  pre-v1.6.6 StatsPAI will need to re-baseline.

### What you should do

1. **If you cited a Heckman SE / t / p / CI from StatsPAI ≤ 1.6.5**,
   re-run and update. The direction of change depends on whether
   selection-induced heteroskedasticity (reduces SE) or
   generated-regressor uncertainty (increases SE) dominates.
2. **Cross-validation**: compare the new output against Stata
   `heckman y x, select(z) twostep` or R
   `sampleSelection::heckit(...)`. Both implement the same Heckman
   (1979) formula; agreement should be to the documented precision.
3. **If you want the old conservative HC1 sandwich** for any reason
   (e.g. replicating a legacy pipeline), there is no supported way to
   get it. The old formula was not a convention choice — it was a
   known approximation the project had not yet replaced.

### Reference formula

Same as above, with the influence-function derivation:

```text
β̂ − β = (X*'X*)⁻¹ [ X*' e − β̂_λ · X*' D_δ Z · (γ̂ − γ) ] + o_p(n^{-1/2})
```

The first term gives the heteroskedastic `X*'(I − ρ̂² D_δ) X*`
contribution; the second gives the `ρ̂² F V̂_γ F'` generated-regressor
contribution, since `∂λ / ∂γ' = −λ(λ + Zγ) Z' = −δ · Z'`.

---

## v1.6.4 → v1.6.5 — ⚠️ Standalone LIML correctness fix

**Narrow correctness follow-up to v1.6.4.** If your codebase only uses
`sp.ivreg`, `sp.iv.iv`, `sp.iv.fit`, or `sp.ivreg(method='liml')` you
are **not affected** — those paths were fixed in v1.6.4. This release
closes an orphan copy of the same bug that lived in the standalone
`sp.liml` / `sp.iv.liml` entry point.

### What changed numerically

Anything calling `sp.liml(...)` directly will see both **β̂ and SE
change** compared to ≤ v1.6.4. Two independent bugs were fixed:

1. **κ_LIML solver**: switched from the non-symmetric
   `np.linalg.eigvals(inv(A) @ B)` (which can silently return complex
   eigenvalues and a biased κ) to the proper generalized symmetric
   eigenvalue problem `scipy.linalg.eigh(S_exog, S_full)`. Point
   estimates β̂ shift to the correct κ.
2. **Sandwich meat**: the cluster / robust meat used raw `X` instead of
   the k-class transformed `AX = (I − κ M_Z) X`. Same bug family as
   v1.6.4 for 2SLS; same fix (use the influence-function regressor in
   the meat).

### Post-fix consistency checks

- `sp.liml(...)` now produces **byte-identical** output to
  `sp.ivreg(..., method='liml')`.
- β̂ agrees with `linearmodels.IVLIML` to machine precision.
- Cluster SEs differ from `linearmodels.IVLIML` by ~0.1–0.2% because
  StatsPAI uses the k-class FOC-derived meat `AX = (I − κ M_Z) X`,
  while `linearmodels` uses the 2SLS-style meat `X̂ = P_Z X`
  regardless of κ. Both estimators are asymptotically equivalent and
  coincide exactly at κ = 1 (2SLS). The convention is documented in
  the new test file `tests/reference_parity/test_liml_se_parity.py`.

### What you should do

1. **If you have published LIML results** from a version ≤ v1.6.4 via
   `sp.liml(...)`, re-run and update — the old κ could be materially
   off and the old SE was built from the wrong meat.
2. **If you want LIML and only used `sp.ivreg(method='liml')`**, no
   action needed; v1.6.4 already has the correct formula.
3. **If you pinned SE or coefficient values** against the standalone
   `sp.liml` in your test suite, re-baseline to the v1.6.5 numbers.

### Reference formula (same as v1.6.4 for the k-class meat)

```text
β̂ − β = (X' A X)⁻¹ (AX)' u ,  A = (1 − κ) I + κ P_Z
Meat (cluster):  Σ_c (Σ_{i∈c} (AX)_i u_i)(·)'
Bread         :  (X' A X)⁻¹  = (AX' X)⁻¹
```

For 2SLS (κ = 1) `AX = P_Z X = X̂`; for LIML/Fuller `AX` is the
k-class transformed regressor.

---

## v1.6.3 → v1.6.4 — ⚠️ IV SE correctness fix

**Correctness-fix release.** No API surface changes, no new functions,
no docstring renames. **Numerical output of IV cluster / robust SE
changes** — this is the whole point of the release.

### What changed numerically

`sp.iv`, `sp.ivreg`, and `sp.iv.fit(method='2sls' | 'liml' | 'fuller')`
produce different standard errors when called with `robust={'hc0',
'hc1', 'hc2', 'hc3'}` or `cluster=...`. The fix restores the textbook
Cameron–Miller (2015) / Stata `ivregress` / `linearmodels` formula —
meat uses the projected regressor `X̂ = P_W X` rather than the raw
`X = [X_exog, X_endog]`.

Concretely the sandwich is now

```text
V̂ = (X̂'X̂)⁻¹ · [ Σ_c (X̂_c' û_c)(û_c' X̂_c) ] · (X̂'X̂)⁻¹
```

for the cluster case, and analogously for HC0/HC1/HC2/HC3. Before v1.6.4
the bread used `X̂` but the meat used `X`, which is a strictly incorrect
estimator for 2SLS — it happens to coincide with the correct formula
only when the first stage is a perfect fit (never, in practice).

### Who is affected

- Any IV workflow using `robust=` or `cluster=` with 2SLS, LIML, or Fuller.
- **Not affected**: point estimates (`β̂` is algebraically unchanged by
  the projection in the meat), nonrobust default SE, `method='gmm'`,
  `method='jive'`, and `sp.iv.ujive` / `ijive` / `rjive`.

### What you should do

1. **If you have published results** citing an IV SE / t-stat / p-value
   / CI from StatsPAI ≤ 1.6.3, re-run and update. The bias in the
   reported SE can be several-fold depending on first-stage fit —
   **not a rounding issue**.
2. **If you have pinned SE values in your test suite** against an
   earlier StatsPAI version, expect a mismatch. You can verify the new
   numbers by cross-checking with `linearmodels.IV2SLS(...).fit(
   cov_type='clustered', debiased=True)` — they should now agree to
   machine precision.
3. **If you were intentionally trying to reproduce the old (wrong)
   numbers**, don't. There is no supported way to get the
   pre-v1.6.4 behaviour because it was not a convention choice — it
   was a bug.

### Reference formula

For k-class with parameter κ (2SLS → κ=1, LIML → κ=κ_LIML, Fuller →
κ_LIML − α/(n−K)):

- Bread: `(X' A X)⁻¹` with `A = (1−κ) I + κ P_W`
- Meat: uses `A X` (the k-class transformed regressor); for 2SLS
  `A X = P_W X = X̂`
- FOC: `X' A (y − X β) = 0`, so the influence function is
  `β̂ − β = (X'AX)⁻¹ (AX)' u`, and the cluster/robust variance
  plugs `(AX)_i u_i` into the moment sum.

Pre-v1.6.4 the implementation plugged `X_i u_i` instead of `(AX)_i u_i`.

---

## v1.6.2 → v1.6.3 — DiD frontier sprint

**Strictly additive** plus one docstring / label truth-up. No existing
estimator's numerical path changes.

### User-visible changes worth noting

1. **`sp.continuous_did(method='att_gt')` result labels** —
   - ``result.method`` changed from
     `"Continuous DID (Callaway et al. 2024)"` to
     `"Continuous DID (dose-bin heuristic)"`.
   - ``result.estimand`` changed from
     `"ACRT (Average Causal Response on Treated)"` to
     `"Sample-weighted mean of dose-bin 2x2 DIDs (not CGS 2024 ATT(d|g,t))"`.
   - Why: the previous labels claimed paper fidelity with CGS (2024)
     that the implementation did not deliver. Numerical output is
     unchanged. If you were parsing these strings in a pipeline, update
     the matcher.
   - If you actually want a CGS (2024)-style estimator: the new
     `method='cgs'` is an **MVP** (2-period design, OR only) with
     paper formulas flagged `[待核验]`. See
     `docs/rfc/continuous_did_cgs.md`.

2. **`sp.did_multiplegt(dynamic=H)` semantic clarification** — the
   docstring now states explicitly that this is a pair-rollup
   extension, **not** the dCDH (2024) `did_multiplegt_dyn` estimator.
   Numerical output is unchanged; if you were using `dynamic=H` and
   calling it "dCDH 2024", switch to the new `sp.did_multiplegt_dyn`
   (also MVP — see `docs/rfc/multiplegt_dyn.md`).

### New functions (no migration needed, just additive)

`sp.lp_did`, `sp.ddd_heterogeneous`, `sp.did_timevarying_covariates`,
`sp.did_multiplegt_dyn` (MVP), `sp.continuous_did(method='cgs')` (MVP).

### Bib key updates

`paper.bib` entry `dechaisemartin2022fixed` upgraded from SSRN to the
published *Econometrics Journal* 26(3):C1–C30 (2023) version. Any
downstream uses of the bib key via `[@dechaisemartin2022fixed]` are
unaffected; the expanded citation will now render to the journal
version.

---

## v1.5.x → agent-native infrastructure (Unreleased)

Pure-additive release. **No migration required** for existing code.
New agent-native surface area documented here for adopters.

### 1. Exception taxonomy (new public module)

```python
from statspai.exceptions import (
    AssumptionViolation, IdentificationFailure,
    DataInsufficient, ConvergenceFailure,
    NumericalInstability, MethodIncompatibility,
)
```

Domain errors subclass the right stdlib base (`ValueError` /
`RuntimeError`), so existing `try / except ValueError` blocks still
catch `AssumptionViolation` and `DataInsufficient`, and
`except RuntimeError` still catches `ConvergenceFailure` and
`NumericalInstability`. No call-site changes required.

New code should prefer the specific subclass + attach a
`recovery_hint`:

```python
raise AssumptionViolation(
    "Parallel trends rejected at p=0.003",
    recovery_hint="Run sp.sensitivity_rr for Rambachan-Roth honest CI.",
    diagnostics={"test": "pretrends", "pvalue": 0.003},
    alternative_functions=["sp.sensitivity_rr", "sp.callaway_santanna"],
)
```

### 2. Agent-native result methods

- `result.violations()` — structured list of assumption /
  diagnostic issues with `severity` / `recovery_hint` / `alternatives`.
- `result.to_agent_summary()` — JSON-ready structured payload.
- Complement (do not replace) existing `summary()` / `tidy()` /
  `next_steps()`.

### 3. Registry agent cards

- `sp.agent_card(name)` — full metadata including pre-conditions,
  assumptions, failure modes with recovery hints, ranked
  alternatives, typical minimum N.
- `sp.agent_cards(category=None)` — bulk export of entries that
  have at least one agent-native field populated (currently:
  `regress`, `iv`, `did`, `callaway_santanna`, `rdrobust`, `synth`).

### 4. Guide `## For Agents` blocks

Run `python scripts/sync_agent_blocks.py` after any change to a
registered spec's agent-native fields. The `--check` flag is
CI-friendly and fails non-zero on drift.

---

## v1.4.x → v1.5.0

Minor release.  Only one change requires any migration:

### `sp.mr` is now a dispatcher function, not a module alias

Before v1.5.0, `sp.mr` was a reference to the `statspai.mendelian`
submodule, and `sp.mr.mr_ivw(...)` worked as attribute access on the
module.

In v1.5.0, `sp.mr` is the new **unified dispatcher** for the MR family,
matching the pattern of `sp.synth` / `sp.decompose` / `sp.dml`:

```python
sp.mr("ivw",   beta_exposure=bx, beta_outcome=by,
       se_exposure=sx, se_outcome=sy)
sp.mr("egger", beta_exposure=bx, beta_outcome=by,
       se_exposure=sx, se_outcome=sy)
sp.mr("mvmr",  snp_associations=snp_df,
       outcome="beta_y", outcome_se="se_y",
       exposures=["beta_bmi", "beta_ldl"])
```

| Old (<= v1.4.2) | New (>= v1.5.0) |
| --- | --- |
| `sp.mr.mr_ivw(...)` | `sp.mr_ivw(...)` (already available since v0.9) or `sp.mr("ivw", ...)` |
| `sp.mr.mr_egger(...)` | `sp.mr_egger(...)` or `sp.mr("egger", ...)` |
| `sp.mr.mr_presso(...)` | `sp.mr_presso(...)` or `sp.mr("presso", ...)` |
| `sp.mr` (as module alias) | `sp.mendelian` (module access preserved under this name) |

**Rule of thumb:** if your code uses `sp.mr_*` (underscore form) it
already works unchanged in v1.5.0.  Only the uncommon
`sp.mr.<attribute>` pattern needs rewriting.

### Output numerical differences you may notice after upgrading

- `sp.mr_egger` / `sp.mendelian_randomization(..., methods=["egger"])`
  slope p-values and CIs now use `t(n − 2)` rather than `Normal`, matching
  `sp.mr_pleiotropy_egger` and R's `MendelianRandomization` package.
  Effect is invisible for `n_snps ≥ ~100`.  For very small `n_snps` (say
  5 or 6) CIs widen by ~1.6×.
- `sp.mr_presso` p-values now use the `(k + 1) / (B + 1)` MC convention,
  so they are strictly positive (floor `1 / (B + 1)`).  No change for
  non-extreme cases; fixes `-inf` propagation through `log(p)` downstream.

---

## From PyStataR to StatsPAI

`PyStataR` is deprecated. All of its functionality is now available in
[StatsPAI](https://github.com/brycewang-stanford/StatsPAI), under a
unified `sp.*` namespace.

```bash
pip install statspai
```

```python
import statspai as sp
```

## API mapping

| PyStataR | StatsPAI |
|---|---|
| `pdtab.tab1(df, 'x')` / `tab2(df, 'x', 'y')` | `sp.tab(df, 'x')` / `sp.tab(df, 'x', 'y')` |
| `pywinsor2.winsor2(df, ['x'], cuts=(1,99))` | `sp.winsor(df, ['x'], cuts=(1,99))` |
| `pywinsor2.outlier_indicator(df, ['x'])` | `sp.outlier_indicator(df, ['x'])` |
| `pyoutreg.outreg(models, 'out.xlsx')` | `sp.outreg2(models, filename='out.xlsx')` |
| `pyegen.rowmean(df, ['x1','x2'])` | `sp.rowmean(df, ['x1','x2'])` |
| `pyegen.rowtotal(df, ['x1','x2'])` | `sp.rowtotal(df, ['x1','x2'])` |
| `pyegen.rowmax/rowmin(df, [...])` | `sp.rowmax(df, [...])` / `sp.rowmin(df, [...])` |
| `pyegen.rowsd(df, [...])` | `sp.rowsd(df, [...])` |
| `pyegen.rownonmiss(df, [...])` | `sp.rowcount(df, [...])` |
| `pyegen.rank(df, 'x', by='g')` | `sp.rank(df, 'x', by='g')` |

## Why migrate

- **One package, one namespace.** `sp.*` covers everything PyStataR did,
  plus DID, RD, synthetic control, IV, matching, DML, causal forest,
  meta-learners, and more.
- **Actively maintained.** PyStataR is frozen; new features land only in
  StatsPAI.
- **Cleaner naming.** No "Stata" in the name — StatsPAI is Python-native.

## Questions

Open an issue on
[StatsPAI/issues](https://github.com/brycewang-stanford/StatsPAI/issues).
