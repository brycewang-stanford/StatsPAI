# Common mistakes and the agent integration pattern

> Reference file of the `statspai-analysis` skill. Read the section you need; every `sp.*` name, signature and result attribute here is checked by `validate_api_claims.py` against the installed StatsPAI.

## Common Mistakes

| Anti-pattern | Correct form |
|---|---|
| Reporting Table 2 without writing the estimating equation | Step 2 — write the equation + identifying assumption to `artifacts/empirical_strategy.md` *before* estimating |
| Skipping the event-study figure and going straight to the DID coefficient | Step 3.1 — `sp.event_study(...)` + `sp.enhanced_event_study_plot(...)` precedes the regression table |
| Reporting IV without first-stage F | Step 3.2 — `iv.summary()` reports first-stage F; bench-mark F ≥ 10 (≥ 23 for AR-equivalent inference) |
| Reporting RD without McCrary + binscatter | Step 3.3 — `sp.rddensity` + `sp.binscatter` |
| Single-spec main result with no robustness panel | Step 7 — placebo, Oster, honest_did, alt-SE, spec_curve are *expected*, not optional |
| Cluster at observation level when treatment is at firm/state level | Cluster at the level of treatment assignment; use `sp.twoway_cluster` if multi-dim |
| Raw panel → staggered DID without balance check | Run Step 0 `data_contract`; inspect `sp.balance_panel` output and cohort sizes |
| `spec_curve(controls=["a","b","c"])` (flat list) | `controls=[["a"], ["a","b"], ["a","b","c"]]` — each inner list = one spec |
| `sp.rdrobust(..., cutoff=0)` | Kwarg is `c=0` across `rdrobust` / `rkd` / `rdplacebo` / `rdbwsensitivity` |
| `sp.evalue(result)` | `sp.evalue(estimate=<point>, ci=(lo, hi), measure="RR")` |
| `sp.match(df, treat="t", y="y", ...)` | Signature is `(df, y, treat, covariates, ...)` — **y before treat** |
| `sp.sun_abraham(df, y, g, t)` — no unit id | Staggered DID **requires** `i=<unit_id>` |
| `sp.synth(..., treated_period=2000)` | Kwarg is `treatment_time=` (singular) |
| `sp.panel(df, formula, fe=True)` | Kwarg is `method="fe"` |
| `sp.robustness_report(result, ...)` | Takes `(data, formula, x, ...)` — not a result object |
| `sp.mediation(df, y, treat, mediator)` | Kwargs are `(df, y, d, m, X)` — `d` for treatment, `m` for mediator |
| Pre-computed embeddings to `text_treatment_effect` | Pass `text_col=<column_name>`; control vectorisation via `embedder=` |
| `llm_annotator_correct(df)` | Takes aligned `pd.Series` (not DataFrame); NaN for unlabelled rows |
| `sp.callaway_santanna(..., covariates=[...])` | Kwarg is `x=[...]`, not `covariates=` |
| `sp.subgroup_analysis(..., cluster=...)` | Kwarg is `robust='hc1'` (or `'hc0'`/`'hc2'`/`'hc3'`); no cluster slot |
| `sp.oster_delta(..., treat=, controls=, r_max=)` | Real signature: `(data, y, x_base, x_controls, r_max)` |
| `sp.power_did(..., power_target=...)` | Wrappers don't auto-solve. Use dispatcher: `sp.power('did', ..., power_target=..., n_periods=, n_treated_periods=)` |
| `sp.power_cluster_rct(n_clusters=..., power_target=...)` | Use dispatcher: `sp.power('cluster_rct', cluster_size=, icc=, effect_size=, power_target=)` |
| `sp.cate_group_plot(forest, group=...)` | Takes a DataFrame: `g = sp.cate_by_group(ml, df, by=..., n_groups=4); sp.cate_group_plot(g)`. Forest result lacks per-row CATEs — use `sp.metalearner(..., learner='dr')` |
| `sp.cate_plot(causal_forest_result, ...)` | Needs a `metalearner` (or any X/DR/R-learner) result. Per-row CATEs live at `result.model_info["cate"]` (ndarray) — there is **no `.cate_estimates` attribute**. For a causal forest, use `cf.effect(X)` to get the CATE vector |
| `sp.bjs_pretrend_joint(es)` | Real signature: `(cs_or_sa_result, data, y=, group=, time=, first_treat=, controls=)` — NOT `event_study()` output |
| `sp.honest_did(ols_result, ...)` | Only accepts CS / SA / `did_multiplegt` / `aggte(..., 'dynamic')` results — pass a `callaway_santanna` object |
| `sp.sumstats(df, groups={...}, ...)` | No `groups=` kwarg; loop `sp.sumstats(vars=v_panel, ...)` per panel and concat |
| `sp.sumstats(..., by="treat")` always shows numeric "0" / "1" panel headers | Binary 0/1 `by=` auto-renders as **Control / Treated** (no kwarg needed). For non-binary or alternative wording, pass `by_labels={0:"Untrained", 1:"Trained"}` |
| Fixing `fmt="%.0f"` (or any fixed format) on a regtable that mixes dollar-magnitude (~$1500) and elasticity-magnitude (~0.09) coefficients | Silently rounds the elasticities to `0` while stars survive — the LaLonde precision trap. Use `fmt="auto"` for magnitude-adaptive precision: thousands separator for ≥1000, integer for ≥100, 1 dp for ≥10, 2 dp for ≥1, 3 dp below |
| `plan.population` / `plan.equation` / `plan.threats` | Not exposed on `IdentificationPlan`. Available: `assumptions / estimand / estimator / fallback_estimators / identification_story / warnings / summary()`. Use `q.population / q.treatment / q.outcome` from the `CausalQuestion` |
| `sp.regtable(..., output="docx")` / `output="xlsx"` | Enum is `{"text","latex","tex","html","markdown","md","qmd","quarto","word","excel"}`. Either use `output="word"`/`"excel"` or — preferred — drop `output=` and call `.to_word(filename)` / `.to_excel(filename)` on the result |
| `sp.sumstats(..., output="docx")` returns plain text | `sumstats` doesn't natively emit binary docx/xlsx. For Word/Excel use `sp.collect().add_summary(...).save("file.docx")` or convert via `sp.mean_comparison(...).to_word(...)` |
| Hand-rolling Word from `pandas.DataFrame.to_string()` / writing LaTeX manually | `RegtableResult.to_word/.to_excel/.to_latex/.to_markdown/.to_html` already apply book-tab borders, AER stars, and the right SE label. `sp.collect()` bundles many such tables into one file |
| Forgetting `template="aer"` (or `qje`/`econometrica`/`restat`/`jf`/`jpe`/`restud`/`aeja`) on `regtable` | Without `template=`, you lose the journal-correct SE label, star levels, and notes. List presets via `sp.list_journal_templates()` |
| Saving each regression to its own `.tex` and stitching by hand in LaTeX | Use `sp.paper_tables(main=, heterogeneity=, robustness=, placebo=)` for a single multi-panel `.docx` / `.xlsx`, or `sp.collect()` for a full Word/Excel/Markdown bundle (Step 8.4) |
| `sp.regtable(..., keep=[focal_var])` (or `drop=["Intercept"]`) as the *default* for every table | AER convention is to **show every estimated parameter verbatim — controls AND the intercept** so the reader can verify the full spec. `regtable()` does this when you pass NEITHER `keep=` NOR `drop=`. Reserve `drop=["Intercept"]` for when you actively want to suppress the constant; reserve `keep=[focal]` for intentionally focal-only tables (IV first-stage triplet, interaction-form heterogeneity) — each with a comment explaining why |
| `sp.regress("y ~ x \| firm_id", df, cluster="firm_id")` for FE | **Silently produces wrong numbers** — `sp.regress` is a thin statsmodels OLS wrapper that does NOT parse `\|` as a FE separator; it interprets `x \| firm_id` as a single garbage variable name. Use `sp.feols("y ~ x \| firm_id", df, vcov={"CRV1":"firm_id"})` for any formula containing `\|`. Two-way cluster: `vcov={"CRV1":"firm_id+year"}` |
| `sp.feols(..., cluster="firm_id")` | feols uses pyfixest convention: `vcov={"CRV1":"firm_id"}` (one-way) or `vcov={"CRV1":"firm_id+year"}` (two-way). The `cluster=` kwarg is for `sp.regress` / `sp.ivreg` (statsmodels) only |
| `sp.twoway_cluster(feols_result, ...)` or `sp.conley(feols_result, ...)` | Both consume **statsmodels-backed** results only (`sp.regress`/`sp.ivreg`); a pyfixest `feols` result raises (`KeyError`). For feols two-way cluster pass `vcov={"CRV1":"firm_id+year"}` directly; for Conley SE on an FE spec, re-fit that spec via `sp.regress(...)` and pass that |
| Trusting SEs without checking convergence / weak-IV / overlap | Always read `result.summary()` warnings and `result.diagnostics` |
| `sp.<plot>(...).savefig(path)` (chaining `.savefig` on a plot call) | Plotters return a `(fig, ax)` tuple — unpack: `fig, ax = sp.coefplot(...); fig.savefig(path, dpi=300)`. `sp.binscatter` → `(fig, ax, df)`; `sp.kaplan_meier(...).plot()` → bare `Axes` (use `ax.figure.savefig`) |
| `sp.enhanced_event_study_plot(sp.event_study(...))` for the event-study figure | `enhanced_event_study_plot` needs a **CS/SA** result (`KeyError: 'att'` otherwise). Build the figure from `cs = sp.callaway_santanna(...)`: `fig, ax = cs.plot()` (or `sp.ggdid(cs)` / `sp.group_time_plot(cs)`). Keep `sp.event_study(...)` for the numerical pre-trends test only |
| `sp.did_summary_plot(callaway_santanna_result)` | `did_summary_plot` only accepts a `sp.did_summary()` result. For a CS/SA dynamic-effects figure use `cs.plot()` / `sp.ggdid(cs)` / `sp.group_time_plot(cs)` |
| `sp.spec_curve(..., y_transforms=["log","ihs"])` (list) | `y_transforms` is a **dict** `{name: callable}`, e.g. `{"log": np.log, "ihs": np.arcsinh}`. `se_types` accepts only `'nonrobust'`/`'hc1'`(=`'robust'`)/`'cluster'` |
| `sp.unified_sensitivity(...).results` / `sp.sensitivity_dashboard(r).plot()`/`.savefig()` | Both return a **text** `SensitivityDashboard` — use `.summary()` and numeric attrs (`.e_value_point`, `.oster`, …); no `.results`/`.plot()`/`.savefig()`. The sensitivity **figure** (`sp.sensitivity_plot`) consumes `sp.honest_did(cs, ...)` output |
| `sp.gformula(df, ...)` / `sp.bounds(df, ...)` | Both are **modules**, not functions. Point-treatment g-formula → `sp.g_computation(df, y=, treat=, covariates=)` (time-varying → `sp.gformula.gformula_mc(...)`). Bounds → `sp.bounds.manski_bounds(...)` / `sp.bounds.lee_bounds(...)` |
| `sp.target_trial_emulate(df, protocol=, id=, time=, treat=, event=)` | Real signature: `(protocol, data, outcome_col, treatment_col, time_zero_filter=None, weights=None)`. `eligibility` is applied as `data.query(...)` unless you pass a `time_zero_filter` callable |
| `TargetTrialProtocol(assignment="...free text...", causal_contrast="...free text...")` | `assignment ∈ {"randomization","observational emulation"}`, `causal_contrast ∈ {"ITT","per-protocol","as-treated","observational-analogue"}` — free text raises `ValueError`. Put prose in `notes=` |
| `sp.dag(["a","b",...])` / `dag.add_edges([...])` / `dag.adjustment_set(...)` | `sp.dag("a -> b; c -> b")` parses an edge **string**; add edges with chained `.add_edge(parent, child)` (singular); back-door sets via `.adjustment_sets(exposure, outcome)` (**plural**, positional) → list of sets |
| `sp.aft("Surv(time, event) ~ x", ...)` | AFT formula LHS is `"duration + event"`: `sp.aft("followup_days + mace ~ x", df, family="weibull")` |
| `sp.hal_tmle(..., variant="ate")` | Only `variant="delta"` is implemented (`"projection"` is NotImplemented) |
| `sp.principal_strat(..., strata=<3-level>, instrument=<continuous>)` | Both `strata` and `instrument` must be **binary 0/1** columns |
| `sp.evalue(estimate=result.point_estimate, ...)` | `CausalResult` exposes `.estimate` and `.ci` (no `.point_estimate`). Econometric results use `.params[name]` / `.conf_int().loc[name]` |
| `sp.dml(..., ml_g=, ml_m=)` or passing a `SuperLearner` as nuisance | dml nuisance kwargs are `model_y=` / `model_d=`, each a sklearn estimator OR alias `'gbm'/'rf'/'lasso'/'xgb'/...`. `metalearner` `outcome_model=`/`propensity_model=` need sklearn **objects** (no string aliases). `sp.super_learner(...)` output is a standalone predictor, not a nuisance arg |
| `sp.causal_question(..., estimand="ate")` / `q.identify(strategy=, X=)` | `estimand` is UPPERCASE (`'ATE'/'ATT'/'LATE'/...`); set strategy via `design=`/`covariates=` on `causal_question`; `q.identify()` takes **no** arguments |
| `sp.offline_safe_policy(state=X_cols, ...)` | `state` and `action` must each be a **single discrete column name** — encode multi-feature state into one segment column first |
| `sp.ope.doubly_robust(X, A, R, pi_b=<1-D>, pi_e=<1-D>, reward_model=<model>)` | `pi_b`/`pi_e` must be `(n, K)` probability matrices (one-hot for deterministic policies); `reward_model` is a **callable** `reward_model(X, a) -> length-n vector`. `ips`/`snips` need no reward model |
| `sp.fairness.fairness_audit(..., predictions=<continuous>, labels=<continuous>)` | Both `predictions` and `labels` must be **binary 0/1**; it audits a binary classifier. A meta-learner result has no `.predict` |
| `pol_tree.plot()` / `cf.local_effects()` | `PolicyTreeResult` uses `.plot_tree()` (→ `(fig, ax)`); `CausalForest` has no `.local_effects()` — get per-row CATEs via `cf.effect(X)` |
| `sp.feols(...)` without installing pyfixest | `sp.feols`/`fepois`/`feglm` need `pip install "statspai[fixest]"`; neural causal needs `[neural]` (torch); plots need `[plotting]` |
| `sp.ivreg("y ~ (d ~ z) + x \| industry + year", ...)` for FE-IV | **Silently drops the `\| fe`** (identical β̂ with/without it; FE never appear in output). `sp.ivreg` does not absorb `\|` FE or parse `C(fe)`. Keep all IV-triplet columns on the same low-dim controls, or pre-build dummy columns in pandas |
| `sp.regtable(ivw, egger, median, ...)` for MR results | `mr_ivw`/`mr_egger`/`mr_median` return **dicts** (`estimate/se/ci_lower/ci_upper/p_value/...`), not result objects. Build a `pd.DataFrame({...}).T` and `.to_excel()`/`.to_latex()` it |
| `aft.to_word(...)` / hand-rolling a `pd.DataFrame` for an AFT table | `sp.regtable(aft, ...)` works **directly** — `AFTResult` exposes `.params` + `.std_errors`, so regtable renders SEs/stars and the `RegtableResult` exports to Word/Excel/LaTeX. `AFTResult` itself still has no `.to_word`/`.to_latex`/`.conf_int`; read `.n`/`.n_events`/`.aic`/`.family`/`.summary()` for the footer. For a causal survival estimand use `sp.ltmle_survival(...)` |
| `sp.causal(..., dag=discovered.dag)` | `LLMConstrainedDAGResult` has no `.dag` — use `discovered.to_dag()` (or inspect `.final_edges`) |
| `result.data_info["n_obs"]` / `result.conf_int().loc["treat"]` on a `CausalResult` | `CausalResult` exposes `.estimate` / `.ci` (tuple) / `.n_obs` (alias `.nobs`) / `.estimand`; `data_info`'s key is `"nobs"`. Its `.conf_int()` has a single row labelled by `.estimand` (`.conf_int().loc[result.estimand]`), not by the treatment column |

---

## Agent Integration Pattern

```python
import statspai as sp

sp.list_functions()                                        # discover
info   = sp.describe_function("callaway_santanna")         # understand
schema = sp.function_schema("callaway_santanna")           # structured call spec

result = sp.callaway_santanna(df, y="y",
                               g="first_treat_year", t="year", i="firm_id")
print(result.summary())
result.to_latex("tables/did_results.tex")
```

---
