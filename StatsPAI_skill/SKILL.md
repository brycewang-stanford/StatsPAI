---
name: StatsPAI_skill
description: Use when the user asks to run a full empirical / causal analysis in Python, pick between DID / RD / IV / SCM / DML / matching, estimate treatment effects, build a DAG, evaluate policy, or produce regression / robustness tables for a paper. Also triggers on keywords "StatsPAI", "statspai", "causal_question", "estimand-first", "honest_did", "spec_curve", "callaway_santanna", "dragonnet", or "text as treatment".
triggers:
  - causal inference in python
  - DID IV RD SCM
  - callaway_santanna
  - synthetic control
  - double machine learning
  - causal forest
  - honest_did
  - spec_curve
  - estimand-first DSL
  - LLM-assisted DAG discovery
  - text as treatment
  - StatsPAI
  - statspai
---

# StatsPAI: Agent-Native Causal Inference & Econometrics

StatsPAI is the agent-native Python package for causal inference and applied econometrics: one `import statspai as sp`, 900+ functions behind a self-describing API, and `CausalResult` objects that export to LaTeX / Word / Excel / BibTeX.

- **Source**: https://github.com/brycewang-stanford/StatsPAI
- **Install**: `pip install statspai` (>= 1.6)
- **Paper**: submitted to JOSS (under review)

## Why for Agents

1. **Self-describing**: `sp.list_functions()` / `sp.describe_function(name)` / `sp.function_schema(name)` — every public symbol is discoverable without doc lookup.
2. **Unified result**: every estimator returns `CausalResult` with `.summary()`, `.plot()`, `.diagnostics`, `.to_latex()`, `.to_word()`, `.cite()`.
3. **One import, full pipeline**: data contract → EDA → estimand-first DSL → DAG discovery → estimation → diagnostics → robustness → paper artifacts.
4. **Estimand-first**: `sp.causal_question(...).identify()` forces the "DID vs RD vs IV?" decision before estimation, not after.

## Pipeline (the canonical agent loop)

```
Step -1 Design & power       sp.power.*  (pre-data: sample-size / MDE planning)
Step  0 Data contract        5 sanity checks + MCAR/MAR hint → contract dict (go / no-go gate)
Step  1 EDA / Table 1        sp.sumstats · sp.describe · sp.balance_table
Step  2 Pre-flight           sp.diagnose (leverage, overlap, missing pattern)
Step  3 Research question    sp.causal_question(...).identify()       → IdentificationPlan (PRE-REGISTRATION)
Step  4 DAG exploration      sp.llm_dag_propose → validate → constrained
Step  5 Estimation           sp.causal(...)   OR   sp.<specific_estimator>(...)
Step  6 Diagnostics/robust   diagnose_result · honest_did · spec_curve · evalue · sequential blocks
Step  7 Deliverables         result.to_latex() · .to_word() · .cite()  → paper artifacts
```

> **All code blocks below share one running example (`training → wage`, with `worker_id / firm_id / year / age / edu / tenure`) purely for readability.** Column names, `population`, `estimand`, and `design` values are **illustrative** — substitute the user's actual columns and research question. Only `sp.*` function names and argument *shapes* are normative.

---

## Step −1 — Design & power (pre-data; skip if data already collected)

`sp.power(design, n=..., effect_size=..., power_target=...)` is a unified dispatcher — leave one argument `None` to solve for it (sample size, MDE, or power). Convenience wrappers: `sp.power_rct`, `sp.power_did`, `sp.power_rd`, `sp.power_iv`, `sp.power_cluster_rct`, `sp.power_ols`.

```python
sp.power("rct", effect_size=0.3, power_target=0.80)                # → PowerResult(n=349, power=0.80)
sp.power("did", n=200, effect_size=0.15, power_target=0.80)        # DID: solves MDE / n / power
sp.power_cluster_rct(n_clusters=30, cluster_size=50,
                     icc=0.05, effect_size=0.2, power_target=0.80) # Cluster RCT
sp.pretrends_power(result)                                          # Roth (2022) pre-trends power
```

Persist the `PowerResult` next to `data_contract.json` — a referee will ask.

## Step 0 — Data contract (go / no-go gate)

StatsPAI assumes an **analysis-ready DataFrame**. Do ETL (missing-value imputation, type coercion, merges, transforms) in pandas first, then run this 5-check contract and feed the result into downstream `sp.*` calls:

```python
import pandas as pd, numpy as np, statspai as sp

def data_contract(df, *, y, treatment, id=None, time=None, covariates=()):
    """Return a go/no-go dict. Stop the pipeline if any required check fails."""
    keys = [y, treatment] + ([id, time] if id and time else []) + list(covariates)
    c = {
        "n_obs":       len(df),                                           # 1. shape
        "dtypes":      df[keys].dtypes.astype(str).to_dict(),             # 2. dtypes on keys
        "n_missing":   df[keys].isna().sum().to_dict(),                   # 3. missing pattern
        "n_dupes_on_keys": 0,
        "panel_balanced":  None,
        "cohort_sizes":    None,
    }

    if id and time:
        c["n_dupes_on_keys"] = int(df.duplicated([id, time]).sum())       # 4. duplicate (id,time)
        balanced = sp.balance_panel(df, entity=id, time=time)              # 5. panel balance
        c["panel_balanced"]        = len(balanced) == len(df)
        c["n_dropped_by_balance"]  = len(df) - len(balanced)

        if "first_treat_year" in df.columns:                               # staggered cohorts
            c["cohort_sizes"] = (
                df.drop_duplicates(id).groupby("first_treat_year").size().to_dict()
            )

    c["y_range"]          = (float(df[y].min()), float(df[y].max()))
    c["treatment_share"]  = float(df[treatment].mean())

    # Missingness mechanism hint (Rubin): compare covariate means between
    # rows missing-on-y vs observed. Any p < 0.05 ⇒ NOT MCAR → use MI / IPW,
    # not listwise deletion.
    from scipy import stats
    miss_y = df[y].isna()
    c["mcar_hint"] = "likely MCAR (listwise OK)"
    if miss_y.any() and (~miss_y).any():
        for cov in covariates:
            if df[cov].dtype.kind in "fi":
                _, p = stats.ttest_ind(df.loc[miss_y, cov].dropna(),
                                        df.loc[~miss_y, cov].dropna(),
                                        equal_var=False)
                if p < 0.05:
                    c["mcar_hint"] = f"NOT MCAR (y-miss differs on {cov}, p={p:.3f}) → use MI / IPW"
                    break
    return c

contract = data_contract(df, y="wage", treatment="training",
                         id="worker_id", time="year",
                         covariates=["age", "edu", "tenure"])

assert contract["n_dupes_on_keys"] == 0, "duplicate (id, time) — fix before panel methods"
assert all(v == 0 for v in contract["n_missing"].values()), \
       f"NaNs on keys: {contract['n_missing']}"
```

If any assertion fires, **stop** and fix it in pandas — StatsPAI estimators silently drop NaN rows, which is the most common source of "mysterious sample-size shrinkage" bugs. Persist the contract alongside results for reproducibility:

```python
import json; json.dump(contract, open("artifacts/data_contract.json", "w"), indent=2, default=str)
```

## Step 1 — Descriptive statistics & Table 1

```python
sp.sumstats(df, vars=["wage", "edu", "exp"], by="treated", output="tables/table1.docx")
sp.describe(df).to_markdown("references/codebook.md")   # auto-codebook (types, N, unique, stats)
sp.balance_table(df, treat="treated",
                 covariates=["age", "edu", "income"], test="ttest")
```

## Step 2 — Pre-flight checks (identification-independent)

```python
report = sp.diagnose(df, y="wage", x=["age", "edu", "tenure"])   # leverage, overlap, missing
```

Identification-specific checks (parallel trends for DID, weak-IV F, bandwidth for RD, common support for matching) run **automatically inside `sp.causal(...)` in Step 5** — don't duplicate them here.

## Step 3 — Estimand-first research question (= pre-registration)

`sp.causal_question` declares the five-tuple (population, treatment, outcome, estimand, design) and `.identify()` picks the estimator with its assumptions. **Treat the `IdentificationPlan` as the pre-registration artifact** — freeze it *before* running `q.estimate()` so the analysis plan is a dated document, not a post-hoc rationalization.

**Design decision** — when `design="auto"` is too opaque, use this decision tree:

```
                 ┌─ running var + cutoff ───────────────── RDD   (sp.rdrobust)
                 │
                 ├─ exogenous instrument Z ─────────────── IV    (sp.ivreg, sp.dml)
data + question ─┤
                 ├─ pre/post × treat/control ─┬ 2 periods  ── 2×2 DID (sp.did)
                 │                            └ staggered  ── CS / SA  (sp.callaway_santanna)
                 │
                 ├─ 1 treated unit + donor pool + long pre ── SCM   (sp.synth, sp.sdid)
                 │
                 ├─ high-dim X, selection-on-observables ── DML / Causal Forest
                 │
                 └─ none of the above ──────────────────── matching + E-value (sp.match, sp.evalue)
```

```python
q = sp.causal_question(
    treatment="training", outcome="wage", data=df,
    population="manufacturing workers, 2010–2020",
    estimand="ATT",
    design="auto",                 # 'auto' | 'did' | 'event_study' | 'regression_discontinuity'
                                   # | 'iv' | 'rct' | 'selection_on_observables'
                                   # | 'synthetic_control' | 'natural_experiment'
                                   # | 'policy_shock' | 'longitudinal_observational'
    time_structure="panel", time="year", id="worker_id",
    covariates=["age", "edu", "tenure"],
)
plan = q.identify()                # IdentificationPlan: estimator + assumptions + fallbacks
print(plan.summary())              # human-readable Methods paragraph
print(plan.identification_story)   # narrative of why this estimator identifies the estimand

# FREEZE the plan to disk BEFORE estimating — this is your pre-registration:
from pathlib import Path
bullets = lambda xs: "\n".join(f"- {x}" for x in xs)
Path("artifacts/preregistration.md").write_text(
    f"# Pre-registration\n\n"
    f"**Estimand**: {plan.estimand}\n"
    f"**Estimator**: {plan.estimator}\n\n"
    f"## Identification story\n{plan.identification_story}\n\n"
    f"## Assumptions\n{bullets(plan.assumptions)}\n\n"
    f"## Fallback estimators\n{bullets(plan.fallback_estimators)}\n"
)

result = q.estimate()              # run only after the plan is committed to disk / git
```

## Step 4 — Model exploration via LLM-assisted DAG

```python
proposal   = sp.llm_dag_propose(
    variables=df.columns.tolist(),
    domain="labor economics: training, wages, tenure",
    client=my_llm_client,                          # .complete(prompt) -> str; None = heuristic
)
validation = sp.llm_dag_validate(proposal, df, alpha=0.05)
print(validation.edge_evidence)

discovered = sp.llm_dag_constrained(
    df,
    descriptions={"wage": "monthly wage USD", "training": "0/1 program"},
    oracle=my_llm_client.suggest_edges,            # optional; falls back to plain PC
    max_iter=3,
)
# Pass into Step 5 as:  sp.causal(..., dag=discovered.dag)
```

## Step 5 — Estimation

**One-call orchestration (recommended for agents):**

```python
w = sp.causal(df, y="wage", treatment="training",
              id="worker_id", time="year", design="did",
              covariates=["age", "edu", "tenure"],
              dag=discovered.dag)                  # optional
print(w.diagnostics)                               # identification verdict
print(w.recommendation)                            # which estimator + why
print(w.result.summary())                          # point estimate + SE + CI
print(w.robustness_findings)                       # automated robustness battery
```

**Direct estimator call** — see *Method Catalog* below.

## Step 6 — Diagnostics & robustness

```python
sp.diagnose_result(result)                                # PT / weak-IV / overlap / leverage
sp.honest_did(result, method="smoothness")                # Rambachan–Roth PT sensitivity

sp.evalue(estimate=result.params["training"],             # E-value takes point + CI, NOT result
          ci=tuple(result.conf_int().loc["training"]),
          measure="RR")

sp.spec_curve(df, y="wage", x="training",                 # controls = List[List[str]]
              controls=[["age"], ["age", "edu"], ["age", "edu", "tenure"]],
              subsets={"all": None, "manuf": df["industry"].eq("manufacturing")})

# Sequential confounder blocks — the Oster (2019) / referee-proof pattern:
# add theoretically grouped blocks one at a time and watch coefficient stability.
blocks = {
    "M1 base":           [],
    "M2 +demographics":  ["age", "edu"],
    "M3 +labor-market":  ["age", "edu", "tenure", "firm_size"],
    "M4 +psychosocial":  ["age", "edu", "tenure", "firm_size", "motivation"],
}
models = [sp.regress(f"wage ~ training + {' + '.join(cov) or '1'}", df,
                     cluster="firm_id")
          for cov in blocks.values()]
sp.regtable(*models,                                      # side-by-side publication table
            model_labels=list(blocks),
            keep=["training"],                            # show only the coefficient of interest
            output="latex",
            filename="tables/confounder_blocks.tex")

sp.bacon_decomposition(df, y="wage", treat="training",
                       time="year", id="worker_id")       # TWFE diagnostic for staggered DID
sp.estat(result, test="all")                              # Stata-style postestimation
```

## Step 7 — Deliverables

Every `CausalResult` exports to paper-ready artifacts in three lines:

```python
result.to_latex("tables/main.tex")                        # booktabs LaTeX table
result.plot().savefig("figures/main.png", dpi=300)        # publication-quality figure
print(result.cite())                                      # BibTeX entry for the method

# Reproducibility stamp — pair with the Step-0 data_contract.json
import json
json.dump({
    "statspai":        sp.__version__,
    "seed":            42,
    "n_obs":           result.data_info["n_obs"],
    "estimand":        result.estimand,
    "estimate":        float(result.params["training"]),
    "ci95":            list(result.conf_int().loc["training"]),
}, open("artifacts/result.json", "w"), indent=2)
```

For full-draft generation (abstract + methods + results + bibliography), see `sp.paper(result, ...)` — out of scope for this skill; call it only when the user explicitly asks for a paper draft.

---

## Method Catalog

### Classical
```python
sp.regress("y ~ x1 + x2", df, cluster="firm_id")                       # OLS (formula-first)
sp.ivreg("y ~ (x1 ~ z1 + z2) + x2", df, cluster="state")               # IV/2SLS — (endog ~ instruments) + exog
sp.panel(df, "y ~ x1 + x2", entity="firm", time="year", method="fe")   # Panel FE
sp.heckman(df, y="wage", x=["age", "edu"],
           select="in_labor_force", z=["marital", "kids"])              # Heckman selection
sp.qreg(df, formula="y ~ x1 + x2", quantile=0.5)                        # Quantile regression
```

### Difference-in-Differences
```python
sp.did(df, y="y", treat="treated", time="post")                              # 2×2 DID (time = 2 values)
sp.callaway_santanna(df, y="y", g="first_treat_year", t="year", i="firm_id") # CS 2021
sp.sun_abraham(df, y="y", g="first_treat_year", t="year", i="firm_id")       # SA 2021 event study
sp.bacon_decomposition(df, y="y", treat="treated", time="year", id="firm_id")# TWFE diagnostic
sp.continuous_did(df, y="y", dose="dose", time="year", id="firm_id")         # Continuous treatment
sp.honest_did(result, method="smoothness")                                   # PT sensitivity (RR 2023)
```

### Regression Discontinuity
```python
sp.rdrobust(df, y="y", x="running_var", c=0)                      # Sharp RD (CCT 2014)
sp.rdrobust(df, y="y", x="running_var", c=0, fuzzy="treatment")   # Fuzzy RD
sp.rddensity(df, x="running_var", c=0)                            # McCrary density test
sp.rdmc(df, y="y", x="running_var", cutoffs=[0, 5, 10])           # Multi-cutoff RD
sp.rkd(df, y="y", x="running_var", c=0)                           # Regression kink
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
         treated_unit=1, treatment_time=2000)              # ADH SCM (method='augmented' default)
sp.sdid(df, outcome="y", unit="unit", time="time",
        treated_unit=1, treatment_time=2000)               # Synthetic DID (Arkhangelsky et al. 2021)
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
    method="hausman")                                                 # Egami et al. 2024
```

### Robustness & Workflow
```python
sp.spec_curve(df, y="wage", x="training",
              controls=[["age"], ["age", "edu"], ["age", "edu", "tenure"]])
sp.robustness_report(df, formula="wage ~ training + age + edu",
                     x="training", cluster_var="firm_id")
sp.subgroup_analysis(df, formula="wage ~ training + age + edu",
                     x="training", by={"gender": "female", "age_bin": "age_quartile"})

fig = result.plot()
sp.interactive(fig)                                                   # WYSIWYG editor, 29 academic themes
```

---

## Common Mistakes

| Anti-pattern | Correct form |
|---|---|
| Raw panel → staggered DID without balance check | Run Step 0 `data_contract`; inspect `sp.balance_panel` output and cohort sizes |
| `spec_curve(controls=["a","b","c"])` (flat list) | `controls=[["a"], ["a","b"], ["a","b","c"]]` — each inner list = one spec |
| `sp.rdrobust(..., cutoff=0)` | Kwarg is `c=0` across `rdrobust` / `rkd` |
| `sp.evalue(result)` | `sp.evalue(estimate=<point>, ci=(lo, hi), measure="RR")` |
| `sp.match(df, treat="t", y="y", ...)` | Signature is `(df, y, treat, covariates, ...)` — **y before treat** |
| `sp.sun_abraham(df, y, g, t)` — no unit id | Staggered DID **requires** `i=<unit_id>` |
| `sp.synth(..., treated_period=2000)` | Kwarg is `treatment_time=` (singular) |
| `sp.panel(df, formula, fe=True)` | Kwarg is `method="fe"` |
| `sp.robustness_report(result, ...)` | Takes `(data, formula, x, ...)` — not a result object |
| Pre-computed embeddings to `text_treatment_effect` | Pass `text_col=<column_name>`; control vectorisation via `embedder=` |
| `llm_annotator_correct(df)` | Takes aligned `pd.Series` (not DataFrame); NaN for unlabelled rows |
| Trusting SEs without checking convergence / weak-IV / overlap | Always read `result.summary()` warnings and `result.diagnostics` |

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

## When to Use StatsPAI vs Alternatives

| Scenario | Use StatsPAI | Alternative |
|---|---|---|
| One-stop EDA → estimand → DAG → estimate → robustness pipeline | ✅ single import covers all seven steps | assemble pyfixest + econml + causalml + differences + ... |
| Agent-driven analysis with self-describing API | ✅ `list_functions` / `describe_function` / `function_schema` | statsmodels / pyfixest (no agent API) |
| Estimand-first "DID vs RD vs IV?" decision | ✅ `sp.causal_question` + `sp.causal` | manual judgement call |
| Stata → Python migration (same API names) | ✅ `sp.regress`, `sp.estat`, `sp.sumstats`, `sp.xtreg` | linearmodels (partial) |
