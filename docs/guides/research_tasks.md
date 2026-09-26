# Research tasks: where to start

Ten common applied-econometrics tasks, each with the functions to call, the
options that change the answer, how to check what evidence covers *your*
configuration, and the mistakes that are easy to make. Every function named
here exists (`tests/test_research_tasks_doc.py` checks), and every page it
links to goes deeper.

Two habits apply to all of them:

* **Read the result card.** `sp.result_card(fit)` (also in every MCP
  response) lists the estimand, the rows actually used, the variance
  convention, the backend, which configuration-level evidence covers this
  call, and the assumptions the data cannot verify.
* **Evidence is per configuration.** A certified function is not certified
  for every option. `sp.validation_scope(fit)` says which outputs of *this*
  call were compared with R / Stata; the generated
  [capability table](../capabilities.md) summarises the flagship estimators.

---

## 1. Moving a Stata `.do` file

**Call:** `sp.stata(...)` runs supported estimation / postestimation
commands; `sp.from_stata(...)` translates. See [translator](translator.md)
and [grammar](grammar.md).

**Semantics to check, not just syntax:**

* missing values -- estimators drop incomplete rows listwise (and rows with a
  missing cluster, as `vce(cluster)`); `sp.pwcorr` is pairwise like Stata's
  `pwcorr`, `listwise=True` for `pwcorr, listwise`;
* weights -- `sp.regress(weights=)` is `[aw=]`; matching estimators take
  frequency weights (`[fw=]`); survey work goes through `sp.svydesign`;
* factor variables -- `C(g)` is `i.g` with the lowest level as base;
  `sp.margins` reports `"2.g"`-style discrete changes as Stata does;
* `margins` averages over the estimation sample (`e(sample)`) with the fit's
  weights, like Stata.

## 2. Two-way / high-dimensional fixed effects

**Call:** `sp.feols("y ~ x | firm + year", df, cluster="state")` (pyfixest
backend), `sp.hdfe_ols` (native, Stata `reghdfe` conventions),
`sp.fast.feols`. See [panel data](panel_data.md).

**Options that matter:** clustering (`cluster=` or `vcov={"CRV1": col}`;
intervals use t(G - 1)); varying slopes `firm + state[year]` (run natively,
`model_info["backend"] == "statspai-native"`); `weights=`.

**Pitfalls:** with only slopes absorbed (`state[[year]]`) `sp.feols` keeps
an intercept as R fixest does, while `sp.hdfe_ols` / Stata `reghdfe
absorb(i.state#c.year)` does not.

## 3. Staggered policy adoption

**Call:** `sp.callaway_santanna`, then `sp.aggte(type="dynamic")`,
`sp.uniform_bands`, `sp.honest_did`. Choosing among CS / Sun-Abraham / BJS /
ETWFE: [choosing a DiD estimator](choosing_did_estimator.md);
[Callaway-Sant'Anna](callaway_santanna.md), [HonestDiD](honest_did.md).

**Options that matter:** `control_group`, `base_period`, `anticipation`,
`weights`, bootstrap vs analytic inference; repeated cross-sections via
`panel=False` ([guide](repeated_cross_sections.md)).

**Pitfalls:** a non-rejected pre-trend test is not evidence of parallel
trends -- check its power (`sp.pretrends_power`) and report
`sp.honest_did`. The joint event-study covariance behind bands and HonestDiD
is compared with R `did` for the default configuration
(`sp.validation_scope(fit)["outputs"]["vcov"]`).

## 4. Instrumental variables and weak instruments

**Call:** `sp.iv(...)`, `sp.effective_f_test`, `sp.anderson_rubin_ci`;
[choosing an IV estimator](choosing_iv_estimator.md).

**Pitfalls:** switching to LIML does not fix weak-instrument inference;
report a weak-IV-robust interval. The exclusion restriction is untestable --
`sp.recommend(...).identification` lists it with the questions to answer.

## 5. Regression discontinuity (sharp, fuzzy, weighted)

**Call:** `sp.rdrobust`, `sp.rddensity`, `sp.rdplot`;
[choosing an RD estimator](choosing_rd_estimator.md).

**Options that matter:** `weights=` (observation weights, as R
`rdrobust`), `fuzzy=`, `covs=`, `cluster=`, `bwselect=`.

**Pitfalls:** check manipulation (`sp.rddensity`) and covariate balance at
the cutoff; `validation_scope` distinguishes the configurations compared
with R `rdrobust` from those that were not.

## 6. Selection on observables, matching and DML

**Call:** `sp.psm` / `sp.match`, `sp.aipw`, `sp.dml`;
[choosing a matching estimator](choosing_matching_estimator.md),
[DML vs DoubleML](sp_dml_vs_doubleml.md).

**Pitfalls:** every estimator here assumes no unmeasured confounding --
check overlap (`sp.overlap_plot`) and sensitivity (`sp.sensemakr`,
`sp.evalue`); the recommendation's identification brief says so explicitly.

## 7. Survey data, calibration and multiple imputation

**Call:** `sp.svydesign(...)`, `design.calibrate(margins=... | totals=...)`,
`sp.mice(...)` then `sp.mi_estimate(...)` and `sp.mi_test(...)` for joint
tests; [survey guide](survey_ph.md).

**Pitfalls:** calibrated weights passed to a new design are treated as fixed
-- use `design.calibrate` for calibration-aware standard errors (R
`survey::calibrate` convention by default, Stata's with
`variance="stata"`). Categorical variables are imputed with
`'logreg'` / `'polyreg'`; `'sample'` ignores every other variable. Joint
tests after MI are not ordinary Wald tests on the pooled covariance:
`sp.mi_test` is Stata's `mi test, nosmall` (the small-sample df is not
implemented and is refused, not approximated).

## 8. Marginal effects and predictions

**Call:** `sp.margins`, `sp.margins_at`, `sp.contrast`, `sp.pwcompare`.

**Pitfalls:** `I(x**2)` and interactions are differentiated through (the
effect of `x` includes `2 b2 x`); a pre-computed `x2` column is treated as an
independent regressor. After logit / probit / poisson the scale is Pr(y=1) /
the expected count including exposure.

## 9. Tables and replication

**Call:** `sp.regtable(...)`, `sp.result_card(fit)`,
`sp.replication_pack(..., strict=True)` then `sp.verify_replication_pack`;
[replication workflow](replication_workflow.md),
[tables](exporting-regression-tables.md).

**Pitfalls:** a zip that opens is not a replication -- the verifier reruns
the script in a fresh directory and compares every recorded estimate.

## 10. Working through an agent (MCP)

**Call:** `sp.recommend(...)` -- each card has `ready` /
`missing_arguments`; `.identification` states what the design cannot
establish. See [agent-native workflow](agent_native_workflow.md) and
[economist MCP workflow](economist_mcp_workflow.md).

**Pitfalls:** a recommendation chooses a tool; it does not identify an
effect. When `identification["claim"]` is `"descriptive_only"` or a card is
`blocked`, the right output is a question to the user, not an estimate.
