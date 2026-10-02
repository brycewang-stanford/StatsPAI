# Method catalog and when to use StatsPAI

> Reference file of the `statspai-analysis` skill. Read the section you need; `validate_api_claims.py` checks it against the installed StatsPAI: every `sp.*` name resolves, every `sp.*(...)` call in a code block binds to the real signature (keywords of functions that take `**kwargs` cannot be checked), and the result attributes on the gate's smoke-fit list exist.

## Method Catalog

### Classical

**Choose by FE structure:**
- **No FE / single low-cardinality FE** → `sp.regress` (statsmodels OLS wrapper)
- **High-dim FE absorption (`y ~ x | fe1 + fe2`)** → `sp.feols` (pyfixest backend, AER workhorse)
- **Two-way panel (entity × time)** → `sp.panel(...)` (linearmodels backend, standard panel diagnostics)

```python
sp.regress("y ~ x1 + x2", df, cluster="firm_id")                       # OLS — `|` is NOT FE here
sp.feols  ("y ~ x1 + x2 | firm_id + year", df, vcov={"CRV1":"firm_id"})# OLS + 2-way FE absorbed
sp.feols  ("y ~ x1 + x2 | firm_id",        df, vcov={"CRV1":"firm_id+year"})  # 2-way cluster
sp.fepois ("count ~ x1 + x2 | firm_id",    df, vcov={"CRV1":"firm_id"})# Poisson + FE (count outcomes)
sp.feglm  ("y ~ x1 + x2 | firm_id", df, family="logit", vcov={"CRV1":"firm_id"})  # Logit + FE
sp.ivreg  ("y ~ (x1 ~ z1 + z2) + x2", df, cluster="state")             # IV/2SLS — (endog ~ instruments) + exog
sp.panel  (df, "y ~ x1 + x2", entity="firm", time="year", method="fe") # Panel FE (within / between / RE / FD)
sp.heckman(df, y="wage", x=["age", "edu"],
           select="in_labor_force", z=["marital", "kids"])              # Heckman selection
sp.qreg   (df, formula="y ~ x1 + x2", quantile=0.5)                     # Quantile regression
```

> **`sp.regress` does NOT parse `|` as a FE separator** — it forwards the formula to statsmodels which treats `edu | firm_id` as a single garbage variable name. Use `sp.feols` (or `sp.panel`) whenever your formula has `|`. Models from `sp.regress`, `sp.feols`, `sp.ivreg`, `sp.panel`, `sp.fepois`, `sp.feglm`, `sp.qreg`, `sp.heckman` all flow through `sp.regtable / sp.coefplot / sp.collect / sp.paper_tables` — mix freely in the same table.

### Difference-in-Differences
```python
sp.did(df, y="y", treat="treated", time="post")                              # 2×2 DID (time = 2 values)
sp.callaway_santanna(df, y="y", g="first_treat_year", t="year", i="firm_id") # CS 2021
sp.sun_abraham(df, y="y", g="first_treat_year", t="year", i="firm_id")       # SA 2021 event study
sp.bacon_decomposition(df, y="y", treat="treated", time="year", id="firm_id")# TWFE diagnostic
sp.continuous_did(df, y="y", dose="dose", time="year", id="firm_id")         # Continuous treatment
sp.honest_did(cs_result, method="smoothness")                                # PT sensitivity (RR 2023) — needs CS/SA result
sp.event_study(df, y="y", treat_time="first_treat_year",
               time="year", unit="firm_id", window=(-4, 4))                  # Event-study coefficients
```

> The robust staggered family (`sp.did_imputation`, `sp.gardner_did`, `sp.etwfe`, `sp.stacked_did`, `sp.lp_did`, `sp.did_multiplegt_dyn`), simultaneous event-study bands and few-treated inference are in `modern-methods.md`.

### Regression Discontinuity
```python
sp.rdrobust(df, y="y", x="running_var", c=0)                      # Sharp RD (CCT 2014)
sp.rdrobust(df, y="y", x="running_var", c=0, fuzzy="treatment")   # Fuzzy RD
sp.rddensity(df, x="running_var", c=0)                            # McCrary density test
sp.rdmc(df, y="y", x="running_var", cutoffs=[0, 5, 10])           # Multi-cutoff RD
sp.rkd(df, y="y", x="running_var", c=0)                           # Regression kink
sp.rdplacebo(df, y="y", x="running_var", c=0,
             placebo_cutoffs=[-2, -1, 1, 2])                       # RD placebo
sp.rdbwsensitivity(df, y="y", x="running_var", c=0,
                    bw_grid=[0.5, 1.0, 1.5, 2.0])                  # Bandwidth sensitivity
```

### Matching & Reweighting
```python
sp.match(df, y="wage", treat="training", covariates=["age", "edu"], method="nearest")  # PSM (default)
sp.match(df, y="wage", treat="training", covariates=["age", "edu"], method="cem")      # Coarsened EM
sp.ebalance(df, y="wage", treat="training", covariates=["age", "edu"])                 # Entropy balancing
```

### Synthetic Control
```python
sp.synth(df, outcome="y", unit="unit", time="time",
         treated_unit=1, treatment_time=2000)              # ADH SCM (method='classic' default; 'augmented'/'sdid'/'mc' opt-in)
sp.sdid(df, outcome="y", unit="unit", time="time",
        treated_unit=1, treatment_time=2000)               # Synthetic DID (Arkhangelsky et al. 2021)
sp.synth_time_placebo(df, outcome="y", unit="unit", time="time",
                      treated_unit=1, treatment_time=2000,
                      n_placebo_times=10)                  # SCM in-time placebo
```

### ML Causal
```python
sp.dml(df, y="wage", treat="training", covariates=["age", "edu"], model="plr")       # DML
sp.causal_forest(formula="wage ~ training | age + edu", data=df)                      # Causal Forest (formula API)
sp.metalearner(df, y="wage", treat="training", covariates=["age", "edu"], learner="dr")  # DR-Learner
sp.tmle(df, y="wage", treat="training", covariates=["age", "edu"])                   # Targeted MLE
sp.aipw(df, y="wage", treat="training", covariates=["age", "edu"])                   # Augmented IPW
```

### Neural Causal
```python
# Requires torch: pip install "statspai[neural]"  (tarnet / cfrnet / dragonnet / cevae)
sp.tarnet(df,    y="wage", treat="training", covariates=["age", "edu"])
sp.cfrnet(df,    y="wage", treat="training", covariates=["age", "edu"])
sp.dragonnet(df, y="wage", treat="training", covariates=["age", "edu"])
```

### Text Causal (v1.6 P1, experimental)
```python
sp.causal_text.text_treatment_effect(
    df, text_col="doc", outcome="y", treatment="t",
    covariates=["age", "edu"], embedder="hash", n_components=20)      # Veitch–Wang–Blei 2020

sp.causal_text.llm_annotator_correct(
    annotations_llm=df["t_llm"],                                      # aligned pd.Series (all rows)
    annotations_human=df["t_true"],                                   # NaN where unlabelled
    outcome=df["y"], covariates=df[["age", "edu"]],
    method="hausman")                                                 # Egami et al. 2023
```

### Mechanisms / Decomposition
```python
sp.mediation(df, y="wage", d="training", m="hours_worked",
             X=["age", "edu"])                                      # ACME / ADE
sp.decompose(...)                                                    # Oaxaca-Blinder / RIF / FFL / KOB
```

### Robustness, Sensitivity & Inference
```python
sp.spec_curve(df, y="wage", x="training",
              controls=[["age"], ["age", "edu"], ["age", "edu", "tenure"]])
sp.robustness_report(df, formula="wage ~ training + age + edu",
                     x="training", cluster_var="firm_id")
sp.subgroup_analysis(df, formula="wage ~ training + age + edu",
                     x="training", by={"gender": "female", "age_bin": "age_quartile"})
sp.oster_bounds(df, y="wage", treat="training",
                controls=["age", "edu"], r_max=1.3)                  # Oster 2019
sp.unified_sensitivity(result, r2_treated=0.05, r2_controlled=0.10,
                       include_oster=True).summary()                  # text dashboard (.summary(); no figure)
sp.sensitivity_dashboard(result).summary()                           # Cinelli-Hazlett + Oster + E-value (text)
sp.evalue(estimate=..., ci=(..., ...), measure="RR")
sp.twoway_cluster(result, df, cluster1="firm_id", cluster2="year")    # two-way SE (statsmodels results)
sp.conley(result, df, lat="lat", lon="lon", dist_cutoff=100)          # spatial HAC

fig, ax = result.plot()                                              # plot() → (fig, ax)
sp.interactive(fig)                                                   # WYSIWYG editor, 29 academic themes
```

---

## When to Use StatsPAI vs Alternatives

| Scenario | Use StatsPAI | Alternative |
|---|---|---|
| One-stop EDA → estimand → DAG → estimate → robustness pipeline | ✅ single import covers all eight AER sections | assemble pyfixest + econml + causalml + differences + ... |
| Agent-driven analysis with self-describing API | ✅ `list_functions` / `describe_function` / `function_schema` | statsmodels / pyfixest (no agent API) |
| Estimand-first "DID vs RD vs IV?" decision | ✅ `sp.causal_question` + `sp.causal` | manual judgement call |
| Stata → Python migration (same API names) | ✅ `sp.regress`, `sp.estat`, `sp.sumstats`, `sp.feols`, `sp.panel` (Stata `xtreg` → `sp.feols("y ~ x | id + year", df)` or `sp.panel(..., method="fe"/"re")`) | linearmodels (partial) |
| Full AER-style robustness gauntlet from one package | ✅ Oster / honest_did / E-value / Conley / 2-way / spec_curve / placebo all in `sp.*` | manually wire 5+ packages |
| **Epidemiology / public health** (target-trial emulation, IPTW + g-formula + TMLE triplet, MR, KM/AFT survival, E-value, STROBE/TRIPOD reporting) | ✅ `sp.target_trial.TargetTrialProtocol` + `sp.target_trial_emulate` + `sp.gformula` + `sp.msm` + `sp.tmle` + `sp.hal_tmle` + `sp.mendelian` (`sp.mr_ivw`/`sp.mr_egger`/`sp.mr_median`) + `sp.kaplan_meier` + `sp.aft` + `sp.evalue` + `sp.principal_strat` — see §A. | hand-stitched zEpid + lifelines + statsmodels + manual MR scripts |
| **ML causal inference** (DML / S/T/X/R/DR-Learner / causal forest / Dragonnet / TARNet / CEVAE / BCF / matrix completion / policy learning / OPE / conformal CATE / fairness audit / DAG learning) | ✅ `sp.dml` + `sp.metalearner` + `sp.causal_forest` + `sp.dragonnet`/`tarnet`/`cevae` + `sp.bcf` + `sp.matrix_completion` + `sp.policy_tree` + `sp.offline_safe_policy` + `sp.ope.*` + `sp.conformal_causal.*` + `sp.fairness.fairness_audit` + `sp.causal_discovery`/`pc_algorithm`/`notears`/`llm_dag_propose`+`llm_dag_validate` — see §B. | EconML + DoWhy + CausalML + GRF + zEpid + dowhy-gcm assembled by hand |
