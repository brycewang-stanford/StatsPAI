# The AER-style empirical pipeline, step by step (default mode)

> Reference file of the `statspai-analysis` skill. Read the section you need; `validate_api_claims.py` checks it against the installed StatsPAI: every `sp.*` name resolves, every `sp.*(...)` call in a code block binds to the real signature (keywords of functions that take `**kwargs` cannot be checked), and the result attributes on the gate's smoke-fit list exist.

## Step −1 — Pre-Analysis Plan (pre-data; AEA RCT Registry style)

`sp.power(design, n=..., effect_size=..., power_target=...)` is a unified dispatcher — leave one argument `None` to solve for it (sample size, MDE, or power). Convenience wrappers: `sp.power_rct`, `sp.power_did`, `sp.power_rd`, `sp.power_iv`, `sp.power_cluster_rct`, `sp.power_ols`.

```python
# Always go through the dispatcher when you want auto-solve. The
# `sp.power_<design>` wrappers (power_rct / power_did / power_rd /
# power_iv / power_cluster_rct / power_ols) accept *only* the design's
# native arguments — they will NOT solve for power_target / n / effect
# unless you go via `sp.power(design, ..., power_target=...)`.

sp.power("rct", effect_size=0.3, power_target=0.80)                  # → PowerResult(n=349, power=0.80)
sp.power("did", n=200, effect_size=0.15, power_target=0.80,
         n_periods=4, n_treated_periods=2)                            # DID: solves MDE / n / power
sp.power("cluster_rct", cluster_size=50, icc=0.05,
         effect_size=0.2, power_target=0.80)                          # Cluster RCT: solves n_clusters
# Roth (2022) pre-trends power is a POST-estimation diagnostic — it needs an estimated
# event-study result, so run it in §3 once you have `es = sp.event_study(...)`:
#   sp.pretrends_power(es)
```

Persist the `PowerResult` next to `data_contract.json` and `empirical_strategy.md` — a referee will ask whether the design was powered before data collection, not after.

## Step 0 — Sample construction & data contract (Section "Data")

An AER §1 *Data* section has three jobs: (a) describe sources, (b) document **every** sample restriction (the "footnote 4" sample log), (c) lock the panel structure. StatsPAI assumes an **analysis-ready DataFrame** — do ETL (imputation, type coercion, merges, transforms) in pandas first, then run the 5-check contract.

### 0.1 Sample-construction log (footnote 4)

```python
sample_log = []
df0 = df_raw.copy();                                       sample_log.append(("0. raw",                len(df0)))
df1 = df0.dropna(subset=["wage"]);                          sample_log.append(("1. drop missing wage",  len(df1)))
df2 = df1[df1["age"].between(18, 65)];                      sample_log.append(("2. drop age outside 18-65", len(df2)))
df3 = df2[df2["industry"].isin(MANUF_CODES)];               sample_log.append(("3. keep manufacturing", len(df3)))
df  = df3
import json; json.dump(sample_log, open("artifacts/sample_construction.json", "w"), indent=2)
```

Paste this log verbatim as footnote 4 of your paper. AER reviewers use it to reconstruct the analysis sample.

### 0.2 Five-check data contract (go / no-go gate)

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

If any assertion fires, **stop** and fix it in pandas — StatsPAI estimators silently drop NaN rows, the most common source of "mysterious sample-size shrinkage" bugs. Persist:

```python
import json; json.dump(contract, open("artifacts/data_contract.json", "w"), indent=2, default=str)
```

## Step 1 — Descriptive statistics (Table 1)

The signature AER Table 1 has three column blocks plus a difference column:

| | (1) Full | (2) Treated | (3) Control | (4) Δ (t-test) |

The Imbens–Rubin rule of thumb: a normalized difference `|Δ| / √((s²₁+s²₀)/2) > 0.25` flags substantive imbalance and should trigger matching / reweighting *before* you trust an OLS comparison.

```python
# Quick text/LaTeX preview (use sumstats `output=` for a string-only render).
# When `by=` is binary 0/1 and you don't pass `by_labels=`, sumstats auto-fills
# the panel headers as **Control / Treated** so the academic Table 1 reads
# correctly out of the box. For non-0/1 codings or different wording, pass
# `by_labels={0:"Untrained", 1:"Trained"}` (or `{"A":"Control","B":"Treated"}`).
print(sp.sumstats(df, vars=["wage","edu","exp","tenure","age"],
                  by="training", output="text"))

# AER-style balance table → Word + Excel + LaTeX in three lines.
# `mean_comparison` returns a MeanComparisonResult that exposes the full
# export chain (.to_word / .to_excel / .to_latex / .to_markdown / .to_html).
mc = sp.mean_comparison(df,
                        ["age","edu","tenure","firm_size"],
                        group="training",
                        test="ttest",
                        title="Table 1. Summary statistics by treatment status")
mc.to_word ("tables/table1_summary.docx")     # editable in Word
mc.to_excel("tables/table1_summary.xlsx")     # editable in Excel
open("tables/table1_summary.tex", "w").write(mc.to_latex())
sp.describe(df).to_markdown("references/codebook.md")              # auto-codebook
```

### 1.1 Multi-panel Table 1 (AER convention)
Group rows into **Panel A: Outcomes**, **Panel B: Treatment intensity**, **Panel C: Controls**, **Panel D: Sample composition**. The cleanest path is to push each panel into a `sp.collect()` bundle — one `.save("file.docx")` call then writes the whole multi-panel Table 1 with AER book-tab borders, in Word **and** Excel **and** LaTeX from one source.

```python
panels = {
    "A. Outcomes":             ["wage", "log_wage", "weeks_employed"],
    "B. Treatment":            ["training", "training_hours"],
    "C. Demographic controls": ["age", "edu", "female", "married"],
    "D. Labor market":         ["tenure", "firm_size", "industry_id"],
}

c1 = sp.collect("Table 1. Summary statistics", template="aer")
for label, vs in panels.items():
    c1.add_heading(f"Panel {label}", level=2)
    c1.add_summary(df, vars=vs, stats=["mean", "sd", "n"])
c1.save("tables/table1_summary.docx")          # editable Word, AER book-tab borders
c1.save("tables/table1_summary.xlsx")          # one sheet per panel (heading drives the sheet name)
c1.save("tables/table1_summary.tex")           # multi-panel LaTeX

# Plain-text alternative (no Collection): one `sp.sumstats` per panel, concat strings.
# Useful when you only need the .tex preview without a binary export.
import io; buf = io.StringIO()
for label, vs in panels.items():
    buf.write(f"\n% Panel {label}\n")
    buf.write(sp.sumstats(df, vars=vs, by="training",
                          stats=["mean", "sd", "n"], output="latex"))
open("tables/table1_summary_flat.tex", "w").write(buf.getvalue())
```

### 1.2 Figure 1 — raw trends / treatment rollout
For DID / event-study designs, the *first* figure of an applied paper is almost always either (a) raw treated-vs-control means over time, or (b) the staggered rollout heat-strip showing which units are treated when. Both are one-liners:

```python
# (a) Raw trends with vertical line at treatment start (DID Figure 1 style)
fig, ax = sp.parallel_trends_plot(df, y="wage", time="year", treat="training",
                                  treat_time=2015, ci=True,
                                  labels={"treated":"Trained", "control":"Untrained"})
fig.savefig("figures/fig1a_raw_trends.png", dpi=300)

# (b) Treatment rollout heatmap (staggered DID convention; Goodman-Bacon-friendly)
fig, ax = sp.treatment_rollout_plot(df, time="year", treat="training", id="worker_id",
                                    sort_by="first_treat_year",
                                    title="Figure 1. Treatment timing")
fig.savefig("figures/fig1b_rollout.png", dpi=300)
```

For matching designs, also produce a **love plot** of standardized differences pre/post matching (Step 3.4).

## Step 2 — Empirical strategy (Section "Identification")

This is the heart of an AER paper. Before any code, **write down the equation explicitly** and **state the identifying assumption**. Vague identification language is the single most common reason a referee rejects an applied paper.

### 2.1 Equation × identifying assumption table

| Design | Estimating equation | Identifying assumption |
|---|---|---|
| 2×2 DID | `Y_it = α_i + λ_t + β·D_it + X'γ + ε_it` | parallel trends conditional on X |
| Event-study (CS / SA) | `Y_it = α_i + λ_t + Σ_{e≠-1} β_e · 1{t-G_i = e} + ε_it` | no anticipation + group-time PT |
| 2SLS | `Y_i = α + β·D_i + X'γ + ε_i;  D_i = π·Z_i + X'δ + u_i` | exclusion + relevance + monotonicity |
| Sharp RD | `Y_i = α + β·1{X_i ≥ c} + f(X_i) + ε_i` (local poly) | continuity of E[Y(0)\|X] at c, no manipulation |
| SCM | `Ŷ_1t(0) = Σ_j ŵ_j Y_jt`, τ_t = `Y_1t − Ŷ_1t(0)` for t≥T_0 | pre-period fit + interpolation validity |
| DML / unconfoundedness | `Y_i = m(X_i) + β·D_i + ε_i` (Robinson partialling-out) | unconfoundedness \| X + overlap |

### 2.2 Design picker

When `design="auto"` is too opaque, use this decision tree:

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

### 2.3 Estimand-first DSL = pre-registration

`sp.causal_question` declares the five-tuple (population, treatment, outcome, estimand, design) and `.identify()` picks the estimator with its assumptions written down. **Treat the `IdentificationPlan` as your pre-registration artifact** — freeze it *before* running `q.estimate()` so the analysis plan is a dated document, not a post-hoc rationalization.

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

# FREEZE the plan to disk BEFORE estimating — this is your pre-registration.
# `q` (CausalQuestion) carries the question (population / treatment / outcome).
# `plan` (IdentificationPlan) carries the strategy (estimator / story /
# assumptions / fallbacks / warnings). The estimating equation is *your*
# job to write down — paste it from the §2.1 table that matches plan.estimator.
from pathlib import Path
bullets = lambda xs: "\n".join(f"- {x}" for x in xs) if xs else "- (none)"
Path("artifacts/empirical_strategy.md").write_text(
    f"# Empirical Strategy (pre-registration)\n\n"
    f"**Population**: {q.population}\n"
    f"**Treatment**: `{q.treatment}`    **Outcome**: `{q.outcome}`\n"
    f"**Estimand**: {plan.estimand}\n"
    f"**Estimator**: `sp.{plan.estimator}`\n\n"
    f"## Estimating equation (paste from §2.1 row matching `{plan.estimator}`)\n"
    f"```\n<paste here>\n```\n\n"
    f"## Identification story\n{plan.identification_story}\n\n"
    f"## Identifying assumptions (must defend in §2)\n{bullets(plan.assumptions)}\n\n"
    f"## Auto-flagged warnings\n{bullets(plan.warnings)}\n\n"
    f"## Fallback estimators (Step 7 robustness)\n{bullets(plan.fallback_estimators)}\n"
)
# Machine-readable sidecar (full question, replayable):
Path("artifacts/causal_question.yaml").write_text(q.to_yaml())

result = q.estimate()              # run only after the plan is committed to disk / git
```

### 2.5 (Optional) LLM-assisted DAG addendum

Useful when the user wants an explicit DAG to defend in §2 or §7. Pipe the discovered DAG into `sp.causal(..., dag=...)`.

```python
proposal   = sp.llm_dag_propose(
    variables=df.columns.tolist(),
    domain="labor economics: training, wages, tenure",
    client=my_llm_client,                          # .complete(prompt) -> str; None = heuristic
)
validation = sp.llm_dag_validate(proposal, df, alpha=0.05)   # (dag, data) positional
print(validation.edge_evidence)

discovered = sp.llm_dag_constrained(
    df,
    descriptions={"wage": "monthly wage USD", "training": "0/1 program"},
    oracle=my_llm_client.suggest_edges,            # optional; falls back to plain PC
    max_iter=3,
)
# The result is an LLMConstrainedDAGResult — it has NO `.dag` attribute. Get a DAG with
# `.to_dag()` (or inspect `.final_edges`). Pass into Step 4 as:
#   sp.causal(..., dag=discovered.to_dag())
```

## Step 3 — Identification graphics (Section "Identification, graphical evidence")

AER convention: **the identification figure precedes the regression table**. The reader should see graphical evidence that PT holds / first stage is strong / RD jumps cleanly *before* you ask them to trust your point estimate.

### 3.1 Event-study plot + numerical pre-trends test (DID identification)
Pre-period coefficients ≈ 0 (with the −1 reference period normalized to zero) is the visual evidence for parallel trends. Pair the **figure** with a **numerical** pre-trends test so reviewers don't have to eyeball it.

```python
# --- The event-study FIGURE comes from a Callaway–Sant'Anna (or sun_abraham)
#     result, NOT from sp.event_study(). The figure plotters
#     (enhanced_event_study_plot / cs.plot() / ggdid / group_time_plot) consume a
#     CS/SA result; feeding them sp.event_study() output raises KeyError('att').
#     Use `x=` for covariates (NOT `covariates=` — that kwarg does not exist on CS).
cs = sp.callaway_santanna(df, y="wage", g="first_treat_year",
                          t="year", i="worker_id",
                          x=["age", "edu"])

# Figure 2a — dynamic ATT / event-study coefficient plot. Plotters return (fig, ax).
fig, ax = sp.enhanced_event_study_plot(
    cs, shade_pre=True,
    title="Figure 2a. Event-study coefficients (95% CI; ref. period = −1)")
fig.savefig("figures/fig2a_event_study.png", dpi=300)
# (equivalently: `fig, ax = cs.plot()` or `fig, ax = sp.ggdid(cs)` /
#  `fig, ax = sp.group_time_plot(cs)` — all consume the CS result and return (fig, ax).)

# Numerical pre-trends test (Roth 2022 power) for the table footnote. THIS is what
# sp.event_study() is for — the coefficient/pre-trend numerics, not the figure.
es = sp.event_study(df, y="wage", treat_time="first_treat_year",
                    time="year", unit="worker_id",
                    window=(-4, 4), ref_period=-1,
                    covariates=["age", "edu"])
print(sp.pretrends_summary(es))                       # F-stat, p-value, max-PT bound
# es.model_info["pretrend_test"] holds the same numbers machine-readably.

# Bacon decomposition figure for staggered DID (Figure 2a-bis)
bd = sp.bacon_decomposition(df, y="wage", treat="training",
                            time="year", id="worker_id")
fig, ax = sp.bacon_plot(bd, title="Figure 2a-bis. Goodman-Bacon weights")
fig.savefig("figures/fig2a2_bacon.png", dpi=300)

# Borusyak–Jaravel–Spiess joint pre-trends test — needs the CS/SA result
# AND the underlying panel (NOT the event_study() output):
sp.bjs_pretrend_joint(cs, df, y="wage", group="first_treat_year",
                      time="year", first_treat="first_treat_year",
                      controls=["age", "edu"])
```

### 3.2 First-stage F-statistic + scatter (IV identification)
Rule of thumb: first-stage F ≥ 10 for OLS-style inference; F ≥ 23 for AR-equivalent inference (Stock–Yogo / Lee 2022).

```python
iv = sp.ivreg("wage ~ (training ~ Z1 + Z2) + age + edu", df, cluster="firm_id")
print(iv.summary())                                    # reports first-stage F (Cragg–Donald / KP)
fig, ax, _binned = sp.binscatter(df, y="training", x="Z1",   # binscatter → 3-tuple
                                 controls=["age", "edu"],
                                 n_bins=20, ci=True)
fig.savefig("figures/fig_first_stage.png", dpi=300)
```

### 3.3 RD: McCrary density + canonical RD plot + binscatter
The signature RD figure is `sp.rdplot` (CCT-style binned scatter with local-polynomial fit on each side), paired with the McCrary manipulation test. Together they answer: (a) is there a visual jump? (b) is the density continuous at the cutoff?

```python
# Figure 2b — canonical RD plot (binned means + local poly fit on each side)
fig, ax = sp.rdplot(df, y="y", x="running_var", c=0,
                    p=4, kernel="triangular", binselect="esmv",
                    shade_ci=True, ci_level=0.95)
fig.savefig("figures/fig2b_rdplot.png", dpi=300)

# Figure 2b-bis — McCrary density (manipulation test). .plot() → (fig, ax)
fig, ax = sp.rddensity(df, x="running_var", c=0).plot()
fig.savefig("figures/fig2b2_mccrary.png", dpi=300)

# Optional: covariate-adjusted binscatter (continuity in covariates is also testable)
fig, ax, _ = sp.binscatter(df, y="age", x="running_var", n_bins=40, ci=True)
fig.savefig("figures/fig2b3_cov_binscatter.png", dpi=300)
```

### 3.4 Matching: love plot (standardized differences)
```python
m = sp.match(df, y="wage", treat="training",
             covariates=["age", "edu", "tenure"], method="nearest")
fig, ax = m.plot()                                    # |std diff| pre vs post; target |Δ|<0.1
fig.savefig("figures/fig2c_love_plot.png", dpi=300)
```

### 3.5 SCM: synthetic-control trajectory + gap plot
For synthetic-control designs the canonical Figure 2 is the treated-vs-synthetic time-series with treatment time annotated. `synthdid_plot` does this in one line.

```python
sc = sp.synth(df, outcome="y", unit="unit", time="time",
              treated_unit=1, treatment_time=2000)
fig, ax = sc.plot()                                   # treated vs synthetic + gap
fig.savefig("figures/fig2d_synth_trajectory.png", dpi=300)
sd = sp.sdid(df, outcome="y", unit="unit", time="time",
             treated_unit=1, treatment_time=2000)
fig, ax = sp.synthdid_plot(sd, title="Figure 2d. Synthetic DID")
fig.savefig("figures/fig2d2_sdid.png", dpi=300)
```

### 3.6 Generic pre-flight (identification-independent)
```python
sp.diagnose(df, y="wage", x=["age", "edu", "tenure"])  # leverage, overlap, missing
```

> Identification-specific checks (PT for DID, weak-IV F, density for RD, common support for matching) **are also auto-run inside `sp.causal(...)`** in Step 4 — don't duplicate the numerics here, but DO produce the figures: a referee scans the figures first.

## Step 4 — Main results (multi-regression tables, AER style)

This is the densest section of an applied paper. A modern AER §4 typically contains **2–3 multi-regression tables and one coefficient plot**:

- **Table 2** (main): progressive controls, 4–6 columns
- **Table 2-bis** (design horse race): same coefficient under OLS / 2SLS / DID / DML
- **Table 2-ter** (multi-outcome): same treatment, several outcomes side-by-side
- **Figure 3** (coefplot): visual summary of β̂ and 95% CI across specs

> **Estimator routing** (memorize this — getting it wrong silently produces nonsense):
> - **No FE** → `sp.regress("y ~ x1 + x2", df, cluster="firm_id")`
> - **High-dim FE** → `sp.feols("y ~ x1 + x2 | fe1 + fe2", df, vcov={"CRV1":"firm_id"})`
> - **Two-way cluster** → `sp.feols(..., vcov={"CRV1":"firm_id+year"})`
> - **2SLS / IV** → `sp.ivreg("y ~ (x ~ z) + controls", df, cluster=...)`
> - **DID / event-study** → `sp.callaway_santanna(...)` / `sp.sun_abraham(...)`
>
> **Never** write `sp.regress("y ~ x | firm_id")` — `sp.regress` does not parse `|` and silently treats `x | firm_id` as a single variable name. Use `sp.feols` for any formula containing `|`.

`sp.regtable(*models, ...)` is the workhorse. Useful kwargs:

```
keep              : list of coef names to display (e.g. ["training"])
drop              : list of coef names to suppress (controls)
model_labels      : column labels   ["(1) Baseline", "(2) +Demog", ...]
dep_var_labels    : dep-var-row labels (for multi-outcome tables)
panel_labels      : panel-A / panel-B layout for stacked tables
coef_labels       : pretty-print names for coefficients
stars             : "aer" → * 0.10 ** 0.05 *** 0.01  (or "default", "none")
stats             : footer rows ["N","R2","Cluster","FE","DV mean", ...]
output            : "latex" | "html" | "markdown" | "text"
filename          : path to write the table
```

### 4.1 Pattern A — Progressive controls (the canonical Table 2)
Stable β̂ across columns ⇒ less concern that selection on observables is driving the estimate (Oster 2019 selection-stability logic; quantified in Step 7.5). **`sp.regtable(*models)` is the StatsPAI equivalent of Stata `outreg2` / `esttab` and R `modelsummary::msummary` / `summary_col` — it consolidates N models into ONE table with one column per model.**

| | (1) Baseline | (2) +Demographics | (3) +Labor-market | (4) +Region×Industry FE | (5) +Worker FE |
|---|---|---|---|---|---|
| Controls | none | age, edu | + tenure, firm_size | high-dim FE | individual FE |

```python
# RULE: pure OLS → sp.regress; high-dim FE absorption → sp.feols
# (sp.regress does NOT parse `|` as FE — it's a thin OLS wrapper. Use
# `sp.feols("y ~ x | fe1 + fe2", df, vcov={"CRV1":"firm_id"})` for FE.)
M1 = sp.regress("wage ~ training",                                  df, cluster="firm_id")
M2 = sp.regress("wage ~ training + age + edu",                      df, cluster="firm_id")
M3 = sp.regress("wage ~ training + age + edu + tenure + firm_size", df, cluster="firm_id")
M4 = sp.feols  ("wage ~ training + age + edu + tenure + firm_size | region + industry + year",
                df, vcov={"CRV1": "firm_id"})
M5 = sp.feols  ("wage ~ training + age + edu + tenure + firm_size | worker_id + year",
                df, vcov={"CRV1": "firm_id"})

# Consolidate 5 models into ONE table (= Stata `outreg2 [M1..M5] using ..., replace`).
# **Default = show ALL coefficients verbatim — controls AND the intercept**
# (AER convention; readers verify the full spec). Pass NO `keep=`/`drop=` and
# `regtable` will surface every estimated parameter. Add `drop=["Intercept"]`
# only if you want to suppress the constant for paper aesthetics; add
# `keep=[focal]` only when a focal-coefficient-only table is intentional.
rt = sp.regtable(M1, M2, M3, M4, M5,
                 template="aer",                  # auto-applies SE label, star levels, font
                 coef_labels={"training": "Job training"},
                 model_labels=["(1) Baseline", "(2) +Demog.", "(3) +Labor-mkt",
                               "(4) Region×Ind. FE", "(5) Worker FE"],
                 stats=["N", "R2", "Cluster", "FE", "DV mean"],
                 title="Table 2. Effect of training on wages")
# Variants (all opt-in — the default above is preferred):
#   • drop intercept only:    sp.regtable(..., drop=["Intercept"])
#   • focal-coefficient only: sp.regtable(..., keep=["training"])
#   • mixed-magnitude table:  sp.regtable(..., fmt="auto")
#       Use whenever a single table mixes dollar-magnitude coefficients
#       (e.g. earnings ≈ 1500) with elasticity-magnitude coefficients
#       (e.g. log-earnings ≈ 0.09). The default fmt="%.3f" pads the dollar
#       side; a fixed fmt="%.0f" rounds the elasticity side to "0" while
#       significance stars survive — the silent LaLonde-style precision
#       trap. fmt="auto" picks per-value precision: thousands separator
#       for |β|≥1000, integer for ≥100, 1 dp for ≥10, 2 dp for ≥1, 3 dp
#       below — so neither magnitude is killed.

# Export to ALL THREE in three lines — Word for co-authors, Excel for editors, LaTeX for build:
rt.to_word ("tables/table2_main.docx")
rt.to_excel("tables/table2_main.xlsx")
open("tables/table2_main.tex", "w").write(rt.to_latex())
```

### 4.2 Pattern B — Design horse race (Table 2-bis)
Show the same coefficient of interest under multiple identification strategies. This is *the* AER credibility move: convergent evidence across designs each making different identifying assumptions.

```python
ols  = sp.feols  ("wage ~ training + age + edu + tenure | industry + year",
                   df, vcov={"CRV1": "firm_id"})                                          # OLS + 2-way FE
ivr  = sp.ivreg("wage ~ (training ~ Z1 + Z2) + age + edu + tenure",
                 df, cluster="firm_id")                                                    # 2SLS
did  = sp.callaway_santanna(df, y="wage", g="first_treat_year",
                             t="year", i="worker_id",
                             x=["age","edu","tenure"])                                     # CS-DID (kwarg is x=)
dml  = sp.dml(df, y="wage", treat="training",
               covariates=["age","edu","tenure","firm_size"], model="plr")                 # DML
mtch = sp.match(df, y="wage", treat="training",
                 covariates=["age","edu","tenure"], method="nearest")                      # PSM

rt = sp.regtable(ols, ivr, did, dml, mtch,
                 template="aer",
                 coef_labels={"training": "Job training (β̂)"},
                 model_labels=["(1) OLS+FE", "(2) 2SLS", "(3) CS-DID",
                               "(4) DML-PLR", "(5) PSM"],
                 stats=["Estimator", "Identifying assumption",
                        "N", "R2 / Pseudo-R2", "Cluster"],
                 title="Table 2-bis. Convergent evidence across designs")
rt.to_word ("tables/table2b_design_race.docx")
rt.to_excel("tables/table2b_design_race.xlsx")
open("tables/table2b_design_race.tex", "w").write(rt.to_latex())
```

### 4.3 Pattern C — Multi-outcome table (same X, several Y's)
A single treatment, several outcomes. Use `dep_var_labels` so each column carries the Y name.

```python
ys = ["wage", "log_wage", "weeks_employed", "left_firm", "promoted"]
multi_y = [sp.feols(f"{y} ~ training + age + edu + tenure | industry + year",
                     df, vcov={"CRV1": "firm_id"})
           for y in ys]

rt = sp.regtable(*multi_y,
                 template="aer",
                 dep_var_labels=ys,                    # column header: dep var
                 model_labels=["(1)","(2)","(3)","(4)","(5)"],
                 stats=["N","R2","DV mean","Cluster"],
                 title="Table 2-ter. Effect of training on multiple outcomes")
rt.to_word ("tables/table2c_multi_outcome.docx")
rt.to_excel("tables/table2c_multi_outcome.xlsx")
open("tables/table2c_multi_outcome.tex", "w").write(rt.to_latex())
```

### 4.4 Pattern D — Stacked Panel A / Panel B table
Same model family, two horizons (short-run / long-run) or two samples (pre-2015 / post-2015) stacked vertically. Use `panel_labels`.

```python
panelA = [sp.feols("wage_t1 ~ training + X | industry + year",  df, vcov={"CRV1":"firm_id"}),
          sp.feols("wage_t1 ~ training + X | worker_id + year", df, vcov={"CRV1":"firm_id"})]
panelB = [sp.feols("wage_t5 ~ training + X | industry + year",  df, vcov={"CRV1":"firm_id"}),
          sp.feols("wage_t5 ~ training + X | worker_id + year", df, vcov={"CRV1":"firm_id"})]

rt = sp.regtable(*panelA, *panelB,
                 template="aer",
                 panel_labels=["Panel A. Short-run (1 year)",
                               "Panel A. Short-run (1 year)",
                               "Panel B. Long-run (5 years)",
                               "Panel B. Long-run (5 years)"],
                 model_labels=["(1) Industry FE","(2) Worker FE"]*2,
                 stats=["N","R2"],
                 title="Table 2-quater. Short- vs long-run effects")
rt.to_word ("tables/table2d_horizons.docx")
rt.to_excel("tables/table2d_horizons.xlsx")
open("tables/table2d_horizons.tex", "w").write(rt.to_latex())
```

### 4.5 Pattern E — IV reporting triplet (first-stage / reduced-form / 2SLS)
The textbook AER IV table presents the **first stage**, the **reduced form**, and the **2SLS** in three columns so the reader can verify Wald-ratio = RF / FS.

> **Trap:** `sp.ivreg` does **not** absorb `| fe` and does **not** parse `C(fe)` — it **silently drops** a `| industry + year` term (identical β̂ with or without it), so a 2SLS column written that way would not control for the FE the first-stage/reduced-form columns absorb. Keep the IV triplet on the **same low-dim control set** in all three columns; to control for fixed effects in a 2SLS, pre-build dummy columns in pandas and add them explicitly, or partial the FE out first.

```python
fs = sp.feols("training ~ Z + age + edu", df, vcov={"CRV1":"firm_id"})   # 1st stage
rf = sp.feols("wage     ~ Z + age + edu", df, vcov={"CRV1":"firm_id"})   # reduced form
iv = sp.ivreg("wage ~ (training ~ Z) + age + edu", df, cluster="firm_id")  # 2SLS (same controls)

rt = sp.regtable(fs, rf, iv,
                 template="aer",
                 keep=["Z", "training"],               # IV triplet is intentionally focal:
                                                       # show only Z + endog so the reader can
                                                       # eyeball Wald-ratio = RF / FS. For the
                                                       # full coef list, drop the kwarg entirely.
                 dep_var_labels=["training", "wage", "wage"],
                 model_labels=["(1) First stage", "(2) Reduced form", "(3) 2SLS"],
                 stats=["First-stage F", "N", "R2", "Cluster"],
                 title="Table 2-quinto. IV reporting triplet")
rt.to_word ("tables/table2e_iv_triplet.docx")
rt.to_excel("tables/table2e_iv_triplet.xlsx")
open("tables/table2e_iv_triplet.tex", "w").write(rt.to_latex())
```

### 4.6 Pattern F — Causal-design main via `sp.causal(...)`
For DID / IV / RD / SCM mains, the `sp.causal(...)` orchestrator returns a `CausalResult` plus diagnostics and an automatic robustness preview. Pipe `.result` into `regtable`:

```python
w = sp.causal(df, y="wage", treatment="training",
              id="worker_id", time="year", design="did",
              covariates=["age", "edu", "tenure"],
              dag=discovered.to_dag())             # optional (LLMConstrainedDAGResult.to_dag())
print(w.diagnostics)                               # PT verdict + warnings
print(w.recommendation)                            # which estimator + why
print(w.result.summary())                          # point estimate + cluster-robust SE + CI
print(w.robustness_findings)                       # automated robustness battery preview
```

### 4.7 Figure 3 — coefficient plot of the main table
Replace one of the wall-of-numbers tables with a coefplot in the body, push the table to the appendix. Modern AER papers increasingly do this.

```python
fig, ax = sp.coefplot(M1, M2, M3, M4, M5,
                      model_names=["(1)","(2)","(3)","(4)","(5)"],
                      variables=["training"],
                      title="Figure 3. β̂ on training across specifications (95% CI)",
                      alpha=0.05)
fig.savefig("figures/fig3_coefplot.png", dpi=300)
```

### Reporting checklist for the Table 2 footnote (AER house style)
- Standard-error cluster level (and whether it's two-way / Conley)
- Fixed-effects absorbed — `regtable` auto-adds **one footer row per FE name** (e.g. `Industry FE: Yes / Year FE: Yes / Worker_id FE: No`) whenever any column comes from `sp.feols(... | fe1 + fe2 ...)`. Don't hand-roll these rows.
- Sample size **and number of clusters**
- Estimator (OLS / 2SLS / CS-DID / SCM / DML)
- Stars convention `* 0.10  ** 0.05  *** 0.01`
- Mean of dependent variable in the estimation sample (so β̂ can be read as a % of the base rate)

## Step 5 — Heterogeneity (Table 3 + Figure 4)

The AER §5 *Heterogeneity* combines (a) a **subgroup regression table** with one column per subgroup (binary moderators + interaction terms), and (b) a **CATE / dose-response figure** for continuous moderators. Both should appear; they answer different questions.

### 5.1 Pattern G — Subgroup `regtable` (Table 3)
One column per subgroup, with the same specification re-run on each slice. Clean, easy to read, expected by referees.

```python
slices = {
    "(1) All":        df,
    "(2) Female":     df[df["female"] == 1],
    "(3) Male":       df[df["female"] == 0],
    "(4) Low skill":  df[df["skill_quartile"].isin([1, 2])],
    "(5) High skill": df[df["skill_quartile"].isin([3, 4])],
    "(6) Small firm": df[df["firm_size"] < 100],
    "(7) Large firm": df[df["firm_size"] >= 100],
}
gmodels = [sp.feols("wage ~ training + age + edu + tenure | industry + year",
                     d, vcov={"CRV1": "firm_id"}) for d in slices.values()]

rt = sp.regtable(*gmodels,
                 template="aer",
                 coef_labels={"training": "Training"},
                 model_labels=list(slices),
                 stats=["N","R2","DV mean"],
                 title="Table 3. Heterogeneous effects of training")
rt.to_word ("tables/table3_heterogeneity.docx")
rt.to_excel("tables/table3_heterogeneity.xlsx")
open("tables/table3_heterogeneity.tex", "w").write(rt.to_latex())
```

### 5.2 Interaction-form heterogeneity (alternative Table 3)
Test moderation formally with interaction terms — referees often ask whether the gap between subgroups is statistically significant, which requires the interaction p-value.

```python
H1 = sp.feols("wage ~ training*female + age + edu + tenure | industry + year",
              df, vcov={"CRV1": "firm_id"})
H2 = sp.feols("wage ~ training*C(skill_quartile) + age + edu + tenure | industry + year",
              df, vcov={"CRV1": "firm_id"})
H3 = sp.feols("wage ~ training*log_firm_size + age + edu + tenure | industry + year",
              df, vcov={"CRV1": "firm_id"})

rt = sp.regtable(H1, H2, H3,
                 template="aer",
                 keep=["training", "training:female", # interaction-form heterogeneity
                       "training:C(skill_quartile)[T.2]",   # is intentionally focal:
                       "training:C(skill_quartile)[T.3]",   # only the main effect + interactions
                       "training:C(skill_quartile)[T.4]",   # are reported. Drop this kwarg
                       "training:log_firm_size"],           # entirely to show full controls.
                 model_labels=["(1) ×Female", "(2) ×Skill quartile", "(3) ×log(Firm size)"],
                 stats=["N","R2"],
                 title="Table 3-bis. Interaction-form heterogeneity")
rt.to_word ("tables/table3b_interactions.docx")
rt.to_excel("tables/table3b_interactions.xlsx")
open("tables/table3b_interactions.tex", "w").write(rt.to_latex())
```

### 5.3 Figure 4 — dose-response (continuous treatment)
```python
dr = sp.dose_response(df, y="wage", treat="training_hours",
                      covariates=["age","edu","tenure","firm_size"],
                      n_dose_points=20)
fig, ax = dr.plot(title="Figure 4a. Dose-response: training hours → wage")
fig.savefig("figures/fig4a_dose_response.png", dpi=300)

# DID-flavored continuous treatment (de Chaisemartin–D'Haultfœuille):
fig, ax = sp.continuous_did(df, y="wage", dose="training_hours",
                            time="year", id="worker_id").plot()
fig.savefig("figures/fig4a2_continuous_did.png", dpi=300)
```

### 5.4 Figure 4-bis — CATE distribution (DR-Learner / causal forest)
The CATE plotters read per-row conditional effects out of the result's
`model_info["cate"]` array. **There is no `.cate_estimates` attribute** — the raw
per-row CATE vector lives at `ml.model_info["cate"]` (an ndarray of length *n*),
and summary stats at `model_info["cate_mean"] / cate_q25 / cate_q75 / ...`.
`sp.causal_forest` returns a *summary* result that does not populate
`model_info["cate"]`, so for the CATE histogram and grouped bar chart use a
meta-learner (or any DR-/X-/R-learner) and pass its result to the plotters.

```python
ml = sp.metalearner(df, y="wage", treat="training",
                    covariates=["age","edu","tenure","firm_size"], learner="dr")

# Raw per-row CATE vector (if you need the numbers, not just the figure):
cate_i = ml.model_info["cate"]                        # ndarray, length n  (NOT ml.cate_estimates)

fig, ax = sp.cate_plot(ml, kind="hist",
                       title="Figure 4b. Distribution of conditional ATE")
fig.savefig("figures/fig4b_cate_hist.png", dpi=300)

# CATE by group bar chart: first compute the group-level table, THEN plot it.
# `cate_group_plot` takes a DataFrame (from cate_by_group), not the result object.
g = sp.cate_by_group(ml, df, by="skill_quartile", n_groups=4)
fig, ax = sp.cate_group_plot(g, title="Figure 4c. CATE by skill quartile")
fig.savefig("figures/fig4c_cate_by_group.png", dpi=300)

# Tabular summary for the appendix
print(sp.cate_summary(ml))
print(g)                                              # group-level CATE table
```

### 5.5 Subgroup-analysis dispatcher (one-liner)
```python
sp.subgroup_analysis(df, formula="wage ~ training + age + edu + tenure",
                     x="training",
                     by={"gender": "female", "skill": "skill_quartile"},
                     robust="hc1")                 # quick subgroup β̂ table (HC1 by default; no cluster arg)
```

For continuous moderators or many subgroups, prefer:
- `sp.continuous_did(...)` — dose-response under DID
- `sp.metalearner(..., learner="dr")` + `sp.cate_plot` / `sp.cate_by_group` — DR-Learner CATE (recommended for plotting)
- `sp.causal_forest(formula="wage ~ training | X", data=df)` — CATE summary only (does not populate `model_info["cate"]`; use a meta-learner for per-row CATEs)

## Step 6 — Mechanisms / channels

```python
sp.mediation(df, y="wage", d="training", m="hours_worked",
             X=["age", "edu", "tenure"])           # ACME / ADE / total effect
sp.decompose(...)                                   # Oaxaca-Blinder / RIF / FFL / KOB
```

## Step 7 — Robustness gauntlet (the AER referee gauntlet)

The seven canonical robustness blocks of an applied paper. A modern AER paper expects most of these in the body or appendix — assemble a Table A1-style robustness panel from the outputs.

### 7.1 Placebo tests
```python
sp.rdplacebo(df, y="y", x="running_var", c=0,
             placebo_cutoffs=[-2, -1, 1, 2])                      # RD: fake cutoffs
sp.synth_time_placebo(df, outcome="y", unit="unit", time="time",
                      treated_unit=1, treatment_time=2000,
                      n_placebo_times=10)                          # SCM in-time placebo
sp.synthdid_placebo(...)                                           # SDID placebo
# For DID: re-run with a fake treat year before actual treatment and confirm β̂ ≈ 0.
```

### 7.2 Alternative samples
```python
result_no_outliers = sp.causal(df.query("wage < wage.quantile(0.99)"), ...)
result_drop_early  = sp.causal(df.query("first_treat_year > 2008"),  ...)
result_balanced    = sp.causal(sp.balance_panel(df, entity="worker_id", time="year"), ...)
```

### 7.3 Alternative specifications (spec curve)
```python
sp.spec_curve(df, y="wage", x="training",
              controls=[["age"], ["age", "edu"], ["age", "edu", "tenure"]],
              subsets={"all": None, "manuf": df["industry"].eq("manufacturing")})
```

### 7.4 Alternative standard errors
Cluster-level choice is itself a robustness check — show the result is not driven by an over-narrow cluster.

```python
# For statsmodels-backed sp.regress / sp.ivreg results:
sp.twoway_cluster(M3, df, cluster1="firm_id", cluster2="year")     # two-way clustering
sp.conley(M3, df, lat="lat", lon="lon",
          dist_cutoff=100, kernel="uniform")                        # spatial HAC (Conley 1999)

# For pyfixest-backed sp.feols results, set 2-way cluster directly in `vcov`:
sp.feols("y ~ x | firm_id + year", df,
         vcov={"CRV1": "firm_id+year"})                              # 2-way: firm × year
```

### 7.5 Oster (2019) selection bound
"How big would unobserved selection have to be for β to flip sign / vanish?" The Oster δ tells you whether the bound on selection on unobservables, relative to selection on observables, has to exceed an implausible value to overturn the result.

```python
sp.oster_bounds(data=df, y="wage", treat="training",
                controls=["age", "edu", "tenure"],
                r_max=1.3)                          # β* assuming δ=1, R̃²=1.3·R²
# `oster_delta` uses x_base / x_controls (NOT treat= / controls=):
sp.oster_delta(data=df, y="wage",
               x_base=["training"],                 # treatment(s) of interest
               x_controls=["age", "edu", "tenure"], # observed controls
               r_max=1.3)                           # δ for which β=0
```

### 7.6 Honest DID — Rambachan–Roth (2023) PT sensitivity
`honest_did` only consumes a CS / SA / `did_multiplegt` event-study result
(or `aggte(result, type='dynamic')`). Pass the `cs` object built in §3.1,
not a generic OLS/FE main-table result:

```python
sp.honest_did(cs, method="smoothness")              # bound β under bounded PT violation
```

### 7.7 E-value & unified sensitivity (unmeasured confounding)
```python
sp.evalue(estimate=result.params["training"],       # E-value takes point + CI, NOT result
          ci=tuple(result.conf_int().loc["training"]),
          measure="RR")
sp.unified_sensitivity(result, r2_treated=0.05,
                       r2_controlled=0.10,
                       include_oster=True)          # Cinelli-Hazlett + Oster combined
sp.sensitivity_dashboard(result)                    # one-page sensitivity figure
```

### 7.8 RD-specific bandwidth / kernel sensitivity
```python
sp.rdbwsensitivity(df, y="y", x="running_var", c=0,
                    bw_grid=[0.5, 1.0, 1.5, 2.0])   # is β̂ stable across bandwidths?
```

### 7.9 TWFE diagnostic (staggered DID)
Goodman-Bacon decomposition flags when the TWFE estimate is contaminated by forbidden 2×2's (already-treated as control).

```python
sp.bacon_decomposition(df, y="y", treat="training",
                       time="year", id="worker_id")
```

### 7.10 Sequential confounder blocks (Oster-style robustness table)
```python
blocks = {
    "M1 base":           [],
    "M2 +demographics":  ["age", "edu"],
    "M3 +labor-market":  ["age", "edu", "tenure", "firm_size"],
    "M4 +psychosocial":  ["age", "edu", "tenure", "firm_size", "motivation"],
}
models = [sp.regress(f"wage ~ training + {' + '.join(c) or '1'}",
                     df, cluster="firm_id")
          for c in blocks.values()]
rt = sp.regtable(*models,
                 template="aer",
                 model_labels=list(blocks),
                 title="Table 7. Selection-stability across confounder blocks")
rt.to_word ("tables/table_robust_blocks.docx")
rt.to_excel("tables/table_robust_blocks.xlsx")
open("tables/table_robust_blocks.tex", "w").write(rt.to_latex())
```

### 7.11 Pattern H — Robustness master table (Table A1, one row per check)
The canonical AER appendix Table A1 stacks every robustness specification next to the baseline so reviewers see at a glance that β̂ survives. `sp.regtable` accepts any mix of `EconometricResults` / `CausalResult`, so build the list dynamically:

```python
baseline = sp.feols("wage ~ training + age + edu + tenure | industry + year",
                     df, vcov={"CRV1": "firm_id"})

rob = {
    "(1) Baseline":            baseline,
    "(2) Drop top 1% wage":    sp.feols("wage ~ training + age + edu + tenure | industry + year",
                                        df.query("wage < wage.quantile(0.99)"),
                                        vcov={"CRV1": "firm_id"}),
    "(3) Balanced panel":      sp.feols("wage ~ training + age + edu + tenure | industry + year",
                                        sp.balance_panel(df, entity="worker_id", time="year"),
                                        vcov={"CRV1": "firm_id"}),
    "(4) Drop early cohorts":  sp.feols("wage ~ training + age + edu + tenure | industry + year",
                                        df.query("first_treat_year > 2008"),
                                        vcov={"CRV1": "firm_id"}),
    "(5) Worker FE":           sp.feols("wage ~ training + age + edu + tenure | worker_id + year",
                                        df, vcov={"CRV1": "firm_id"}),
    "(6) 2-way cluster":       sp.feols("wage ~ training + age + edu + tenure | industry + year",
                                        df, vcov={"CRV1": "firm_id+year"}),  # 2-way: firm × year
    # sp.conley needs a STATSMODELS-backed result (sp.regress/sp.ivreg) — it raises
    # KeyError on a pyfixest feols result. Re-fit the spec via sp.regress for this row.
    "(7) Conley spatial SE":   sp.conley(sp.regress("wage ~ training + age + edu + tenure",
                                                    df, cluster="firm_id"),
                                         df, lat="lat", lon="lon", dist_cutoff=100),
    "(8) Log outcome":         sp.feols("log_wage ~ training + age + edu + tenure | industry + year",
                                        df, vcov={"CRV1": "firm_id"}),
    "(9) IHS outcome":         sp.feols("ihs_wage ~ training + age + edu + tenure | industry + year",
                                        df, vcov={"CRV1": "firm_id"}),
    "(10) PSM-weighted":       sp.match(df, y="wage", treat="training",
                                         covariates=["age","edu","tenure","firm_size"],
                                         method="nearest"),
    "(11) Entropy balance":    sp.ebalance(df, y="wage", treat="training",
                                            covariates=["age","edu","tenure","firm_size"]),
    "(12) DML-PLR":            sp.dml(df, y="wage", treat="training",
                                       covariates=["age","edu","tenure","firm_size"], model="plr"),
}

# Robustness master = AER Table A1 — readers MUST see every coefficient
# across every spec to verify nothing is hiding behind `keep=`. Default to
# the full coef table (intercept included); only switch to
# `keep=["training"]` if a referee has explicitly asked for a focal-only
# summary, or add `drop=["Intercept"]` if you want the constant suppressed.
rt = sp.regtable(*rob.values(),
                 template="aer",
                 coef_labels={"training": "Training (β̂)"},
                 model_labels=list(rob),
                 stats=["N", "R2", "Cluster", "FE"],
                 title="Table A1. Robustness of the main estimate")
rt.to_word ("tables/tableA1_robustness.docx")
rt.to_excel("tables/tableA1_robustness.xlsx")
open("tables/tableA1_robustness.tex", "w").write(rt.to_latex())

# Equivalent one-shot via the paper-format multi-panel API — produces a
# single .docx / .xlsx that you can hand a co-author, with main + robustness
# (+ heterogeneity / placebo if you have them) auto-laid-out per AER style:
sp.paper_tables(main=[M1, M2, M3, M4, M5],
                robustness=list(rob.values()),
                template="aer",
                coef_labels={"training": "Training"},
                model_labels_main=["(1)","(2)","(3)","(4)","(5)"],
                model_labels_robustness=list(rob),
                # paper_tables only accepts `keep=`, not `drop=`. Omit both to
                # show every coefficient (AER convention). Pass `keep=["training"]`
                # only when a focal-only summary is desired.
                ).to_docx("tables/paper_tables.docx")
```

### 7.12 Figure 5 — coefficient forest plot of all robustness specs
A single visual summary that an AER referee can parse in 5 seconds: every β̂ and 95% CI on one axis. Confirms the estimate is not knife-edge.

```python
fig, ax = sp.coefplot(*rob.values(),
                      model_names=list(rob),
                      variables=["training"],
                      title="Figure 5. β̂ on training across robustness specifications",
                      alpha=0.05)
fig.savefig("figures/fig5_robustness_forest.png", dpi=300)
```

### 7.13 Figure 5-bis — spec curve
The Simonsohn et al. (2020) specification curve plots β̂ across **every combination** of {controls × subsamples × outcome transforms × SE types}. Useful when you want to head off "what about specification X?" referee letters.

```python
# se_types accepts only: 'nonrobust', 'hc1' (alias 'robust'), 'cluster' (needs cluster_var).
# y_transforms is a DICT {name: callable} — NOT a list of strings.
sc = sp.spec_curve(df, y="wage", x="training",
                   controls=[["age"], ["age","edu"], ["age","edu","tenure"],
                             ["age","edu","tenure","firm_size"]],
                   se_types=["nonrobust", "robust", "cluster"],   # 'cluster' uses cluster_var below
                   y_transforms={"level": lambda s: s,
                                 "log":   np.log,
                                 "ihs":   np.arcsinh},
                   subsets={"all": None,
                            "manuf":  df["industry"].eq("manufacturing"),
                            "no99":   df["wage"] < df["wage"].quantile(0.99)},
                   cluster_var="firm_id")
fig, ax = sc.plot(title="Figure 5-bis. Specification curve")
fig.savefig("figures/fig5b_spec_curve.png", dpi=300)
```

### 7.14 Figure 6 + sensitivity dashboard
`sp.unified_sensitivity(...)` and `sp.sensitivity_dashboard(...)` both return a
**text/numeric** `SensitivityDashboard` (Cinelli–Hazlett + Oster + Rosenbaum +
E-value). It is **not** a figure — it has `.summary()` and numeric attributes
(`.e_value_point`, `.e_value_ci`, `.oster`, `.rosenbaum`, `.sensemakr`, `.breakdown`),
**no `.plot()` / `.savefig()` / `.results`**. The sensitivity *figure*
(`sp.sensitivity_plot`) is a Rambachan–Roth honest-DID plot and consumes the
DataFrame returned by `sp.honest_did(...)` (columns `M / ci_lower / ci_upper / rejects_zero`).

```python
# (a) Numeric sensitivity dashboard — print the summary, read the attributes.
dash = sp.unified_sensitivity(baseline, r2_treated=0.05, r2_controlled=0.10,
                              include_oster=True)
print(dash.summary())                                 # one-page text dashboard
print(dash.e_value_point, dash.e_value_ci)            # numeric fields for the §7 prose
# sp.sensitivity_dashboard(baseline).summary() is the auto-dimensioned variant.

# (b) Sensitivity FIGURE (honest-DID PT sensitivity) — needs a CS/SA event-study
#     result (`cs` from §3.1) and its honest_did() DataFrame.
sens_df = sp.honest_did(cs, method="smoothness")      # → DataFrame (M, ci_lower, ci_upper, ...)
fig, ax = sp.sensitivity_plot(sens_df,
                              original_estimate=cs.estimate,
                              original_ci=cs.ci,
                              title="Figure 6. Sensitivity to PT violations (Rambachan–Roth)")
fig.savefig("figures/fig6_sensitivity.png", dpi=300)
```

### 7.15 One-stop robustness reporter
```python
sp.diagnose_result(result)                          # PT / weak-IV / overlap / leverage verdict
sp.robustness_report(df, formula="wage ~ training + age + edu",
                     x="training", cluster_var="firm_id")
sp.estat(result, test="all")                        # Stata-style postestimation battery
```

## Step 8 — Replication package

The agent's job at §8 is to produce a **single artifact a co-author can open in Word, Excel, or LaTeX without further StatsPAI calls**. There are three packaging tiers, picked by what you need to ship:

### 8.1 Per-result export (one estimator → one Word/Excel file)
```python
result.to_docx("tables/main_result.docx",
               title="Table 2. Main result")          # CausalResult → .docx
result.to_latex(caption="Main result", label="tab:main")
fig, ax = result.plot()                               # publication-quality figure → (fig, ax)
fig.savefig("figures/main.png", dpi=300)
print(sp.cite(result, "training"))                    # → "1.239*** (0.153)"  ← inline citation
```

### 8.2 Per-table export (already covered in Steps 4 / 5 / 7)
Every `sp.regtable(*models)` returns a `RegtableResult` with `.to_word()` / `.to_excel()` / `.to_latex()` / `.to_markdown()` / `.to_html()`. Use these in §4–§7 so that by the time you reach §8 the `tables/` folder already has parallel `.docx` / `.xlsx` / `.tex` for every numbered table.

### 8.3 Multi-panel paper-format (Tier 2 — Tables 2 + 3 + A1 + A2 in one file)
```python
sp.paper_tables(
    main          = [M1, M2, M3, M4, M5],            # → "Table 2. Main results"
    heterogeneity = [g_full, g_fem, g_male],         # → "Table 3. Heterogeneity"
    robustness    = list(rob.values()),              # → "Table A1. Robustness"
    placebo       = [pb1, pb2],                      # → "Table A2. Placebo tests"
    template      = "aer",
    coef_labels   = {"training": "Training"},
    keep          = ["training"],
).to_docx("replication/paper_tables.docx")           # → 4 panels in one .docx
# .to_xlsx(...) writes one sheet per panel; .to_latex(...) one .tex with section breaks.
```

### 8.4 Full session bundle (Tier 3 — the Stata `collect` equivalent)
The single most efficient §8 deliverable: descriptives + balance + main + heterogeneity + robustness + prose **in one Word file**. `sp.collect()` is the agent-native counterpart of Stata 15's `collect` and R's `gtsave`.

```python
c = sp.collect("Effect of Training on Wages — Replication", template="aer")

c.add_heading("§1. Descriptive statistics", level=1)
c.add_summary(df, vars=["wage","age","edu","tenure"],
              stats=["mean","sd","n"],
              title="Table 1. Summary statistics")
c.add_balance(df, treatment="training",
              variables=["age","edu","tenure","firm_size"],
              title="Table 1b. Balance by treatment")

c.add_heading("§4. Main results",        level=1)
c.add_regression(M1, M2, M3, M4, M5,
                 model_labels=["(1)","(2)","(3)","(4)","(5)"],
                 stats=["N","R2","Cluster","FE"],
                 title="Table 2. Effect of training on wages")

c.add_heading("§5. Heterogeneity",       level=1)
c.add_regression(*gmodels,
                 model_labels=list(slices),
                 title="Table 3. Heterogeneous effects")

c.add_heading("§7. Robustness",          level=1)
c.add_regression(*rob.values(),
                 model_labels=list(rob),
                 title="Table A1. Robustness")

c.add_text(
    "Standard errors clustered at the firm level. *** p<0.01, ** p<0.05, * p<0.10. "
    "Sample restrictions and full variable definitions are documented in "
    "artifacts/sample_construction.json and artifacts/data_contract.json.",
    title="Notes",
)

# One artifact, three formats — auto-detected from the path extension:
c.save("replication/paper.docx")   # editable Word, page-break between tables
c.save("replication/paper.xlsx")   # one sheet per add_*() item
c.save("replication/paper.tex")    # multi-section LaTeX
c.save("replication/paper.md")     # GitHub-flavoured Markdown for the README
```

Inspect the bundle before saving:

```python
print(c)                # → <Collection title='...' template='aer' items=8 kinds=['heading','summary','balance',...]>
print(c.list())         # DataFrame with name / kind / title for every item
```

### 8.5 Reproducibility stamp

> A `CausalResult` (from DID / CS / IV-causal / DML / TMLE / …) exposes `.estimate` (scalar), `.ci` (tuple), `.estimand`, and `.n_obs` — plus `.nobs` (alias of `.n_obs`) and `.conf_int()` (one row, labelled by `.estimand`, equal to `.ci`); `.data_info`'s key is `"nobs"`, not `"n_obs"`. An `EconometricResults` (regress / feols / ivreg) instead exposes `.params[name]` and `.conf_int().loc[name]` — use that branch for an OLS/FE main result.

```python
import json
ci = result.ci                                    # CausalResult: (lo, hi) tuple
json.dump({
    "statspai":          sp.__version__,
    "seed":              42,
    "n_obs":             int(result.n_obs),
    "estimand":          result.estimand,
    "estimate":          float(result.estimate),
    "ci95":              [float(ci[0]), float(ci[1])],
    # Econometric (feols/regress) main result instead:
    #   "estimate": float(M.params["training"]),
    #   "ci95":     list(M.conf_int().loc["training"]),
    "pre_registration":  "artifacts/empirical_strategy.md",
    "data_contract":     "artifacts/data_contract.json",
    "sample_log":        "artifacts/sample_construction.json",
    "paper_bundle":      "replication/paper.docx",
}, open("artifacts/result.json", "w"), indent=2)
```

For full-draft generation (abstract + methods + results + bibliography), see `sp.paper(result, ...)` — out of scope for this skill; call it only when the user explicitly asks for a paper draft.

---
