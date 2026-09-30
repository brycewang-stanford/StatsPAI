# Modern estimators and inference (1.29 – 1.32 era)

> Reference file of the `statspai-analysis` skill. Read the section you need; every `sp.*` name, signature and result attribute here is checked by `validate_api_claims.py` against the installed StatsPAI.

`method-catalog.md` lists the classic call shapes. This file covers the newer
estimators and inference tools an agent should reach for once the design is
fixed, and the traps that make them fail loudly. When in doubt, route first:
`sp.route('did', design='staggered', ...)` names the registered call.

## Staggered DID: the robust estimator family

All of these take the same long panel (`y`, unit, period, first-treatment
period with `0` for never-treated) and return a `CausalResult`. Run two or
three of them side by side. They should agree up to their documented
comparison-group choices.

```python
cs  = sp.callaway_santanna(df, y="y", g="first_treat", t="year", i="id")
es  = sp.aggte(cs, type="dynamic")                                    # event-study aggregate
bjs = sp.did_imputation(df, y="y", group="id", time="year",
                        first_treat="first_treat", horizon=[0, 1, 2]) # Borusyak-Jaravel-Spiess
g2s = sp.gardner_did(df, y="y", group="id", time="year",
                     first_treat="first_treat", event_study=True)     # two-stage DID
ew  = sp.etwfe(df, y="y", group="id", time="year", first_treat="first_treat")  # extended TWFE
stk = sp.stacked_did(df, y="y", group="id", time="year",
                     first_treat="first_treat", window=(-3, 3))
lp  = sp.lp_did(df, y="y", unit="id", time="year", treatment="d", horizons=(-3, 3))
dch = sp.did_multiplegt_dyn(df, "y", group="id", time="year", treatment="d",
                            dynamic=3, placebo=2, seed=0)             # de Chaisemartin-D'Haultfoeuille
```

- `sp.did_multiplegt_dyn` is the one to use when treatment switches **on and
  off** or is non-binary. The others assume absorbing treatment.
- `sp.etwfe(..., family="poisson")` handles count / non-negative outcomes.
  With `fe="unit"` say which count-scale SE you report. The default
  `response_se="profile"` is for the sample's own units. Use
  `"unconditional"` when the claim is a population ATT, since profile runs
  narrow when units are heterogeneous. `"margins"` matches Stata `jwdid`'s
  `estat`.
- `sp.compare_event_study_conventions(df, y=, unit=, time=, first_treat=)`
  explains why TWFE and the robust estimators disagree on the reference
  period. It is defined for **one treatment cohort only** and raises
  `MethodIncompatibility` on staggered timing; use
  `sp.event_study_convention()` there instead.
- `sp.did_design_contract(cs)` returns the checklist a referee will ask for
  (target parameter, comparison group, anticipation, covariates) and marks
  each item `determined` / `undetermined` from what the result records.

## Event-study inference: one covariance, simultaneous bands

```python
V  = sp.event_study_vcov(es)            # EventStudyVcov: .beta, .vcov, .times, .as_frame()
ub = sp.uniform_bands(es, alpha=0.05)   # DataFrame with ci_* AND sup-t cband_lower / cband_upper
eq = sp.pretrends_equivalence(es)       # equivalence test: "pre-trends are small", not "not rejected"
pw = sp.pretrends_power(es)             # dict: power, bayes_factor, likelihood_ratio, ...
hd = sp.honest_did(es, e=0, method="relative_magnitude")  # or method="smoothness"
```

- `sp.event_study_vcov` is the single entry for the **joint** covariance of
  every event-study estimator above. Pointwise CIs on an event-study figure
  do not license a statement about the whole path; plot `cband_lower` /
  `cband_upper` from `sp.uniform_bands` for that.
- Pre-trend reporting is three numbers, not one p-value: the equivalence
  test, the power of the pre-test, and the honest-DID breakdown.

## Few treated clusters

With one or a handful of treated units, cluster-robust SEs over-reject badly.
Use a permutation-style test built from the controls:

```python
r = sp.did_few_treated(df, y="y", id="id", time="year", treat="treated_post",
                       method="conley_taber")   # or method="ferman_pinto" (group-size heteroskedasticity)
```

The point estimate is still the TWFE coefficient. The function fixes the
inference and does not claim consistency. For few clusters in general:
`sp.wild_cluster_bootstrap(df, y=, x=[...], cluster=, test_var=)` and
`sp.cr2_se(result, df, cluster=)`.

## Instrumental variables: the reporting bundle

```python
diag = sp.iv_diag(df, y="y", endog="d", instruments="z", exog="w", cluster="state")
print(diag.summary())    # first-stage F, Olea-Pflueger effective F, KP rk, tF, AR set
                         # (include_clr_ci=True adds the CLR / K sets)
sp.effective_f_test(df, endog="d", instruments=["z"], exog=["w"])   # dict: F_eff, strength, ...
sp.weakrobust(df, y="y", endog="d", instruments=["z"], exog=["w"])  # AR / K / CLR together
```

`sp.iv_diag` is the default IV report. It covers everything the "first-stage
F before the 2SLS coefficient" hard rule asks for in one call.

## Heterogeneity with forests (GRF family)

`sp.causal_forest` is a native GRF engine. Every statistic on the training
rows uses out-of-bag predictions.

```python
cf = sp.causal_forest("y ~ d | x1 + x2 + x3", data=df, random_state=0)
cf.ate()                                             # AIPW ATE with SE
sp.calibrate_cate(cf)                                # dict: is the CATE signal real?
sp.best_linear_projection(cf, A=df[["x1", "x2"]])    # BLP of CATE on chosen covariates
sp.forest_group_effects(cf, by=df["x1"] > 0)         # group ATEs with SEs
sp.rate(cf, target="AUTOC")                          # dict: RATE / TOC prioritisation value
```

Panel data with unit / period fixed effects:

```python
fcf = sp.causal_forest("y ~ d | x1", data=panel, fe="twoway", id="id", time="year",
                       random_state=0)
fcf.att()                                            # equals sp.did_imputation's ATT
sp.rate_split(fcf, target="AUTOC")                   # heterogeneity test, split by unit
sp.forest_policy_tree(fcf, depth=2, cost=1.0)        # dict: rules, value, gain_over_treat_all
```

- **Do not test heterogeneity with the forest's own OOB ranking on a panel.**
  `sp.rate` on an `fe=` forest over-rejects under zero heterogeneity. Use
  `sp.rate_split`, which refits on half the units and evaluates on the rest.
  Both `rate_split` and `forest_policy_tree` aggregate 21 splits by default.
- `sp.forest_policy_tree` accepts **only `fe=` forests** and raises
  `MethodIncompatibility` on a pooled forest. For a cross-sectional policy use
  `sp.policy_tree(data, y, d, X)`.
- Report `res["diagnostics"]["split_stability"]` from `forest_policy_tree`. If the root split
  variable changes across splits, a narrow value interval is not evidence for
  that particular rule.

Other family members share the engine and the result API
(`.average_treatment_effect()`, `.best_linear_projection()`,
`.variable_importance()`, `.to_latex()`):

```python
sp.iv_forest(df, y="y", treat="d", instrument="z", covariates=["x1", "x2"])      # local LATE
sp.multi_arm_forest(df, y="y", treat="arm", covariates=["x1", "x2"])             # several arms
sp.causal_survival_forest(df, time="t", event="event", treat="d",
                          covariates=["x1", "x2"], horizon=4)                    # RMST / survival
```

In these functions `split_alpha` is the GRF split-balance parameter. `alpha`
is always the significance level.

## Time-varying treatment: dynamic DML

When treatment today changes the state that drives treatment tomorrow,
static DML (`sp.dml`, `sp.dml_panel`) is wrong, not just inefficient.

```python
dd = sp.dynamic_dml(panel, y="y", treat="d", id="id", time="t", covariates=["s"])
print(dd.summary())      # per-period effects on the final outcome + joint covariance
```

Keep the default `lags=1`. Without treatment history in the state the
estimator becomes confidently wrong, not merely noisier.

## Factor models, honest RD, RD heterogeneity

```python
sp.fect(df, y="y", treat="d", unit="id", time="t", method="ife", r=1)   # interactive FE counterfactual
sp.gsynth(df, outcome="y", unit="id", time="t", treated_unit=1, treatment_time=2000)  # ONE treated unit
sp.rd_honest(df, y="y", x="score", c=0)            # Armstrong-Kolesar bias-aware CI
sp.rdhte(df, y="y", x="score", z="female", c=0)    # RD effect heterogeneity by covariate
```

`sp.gsynth` takes a single `treated_unit` (a list is not supported). With
several treated units, or treatment that switches on and off, use `sp.fect`.

## What exactly has been validated

```python
sp.validation_scope(cs)      # evidence for THIS configuration: estimate / se / coverage / vcov
```

It lists, per output, whether the exact options you ran are covered by a
Stata / R parity test (T2), a stochastic screen (S), or nothing
(`not_covered`). It also lists the nearest evidence that differs in one
dimension. Quote this instead of a blanket "matches Stata / R" claim.
