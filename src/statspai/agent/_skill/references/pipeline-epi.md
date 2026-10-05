# Mode A: epidemiology / public health pipeline

> Reference file of the `statspai-analysis` skill. Read the section you need; `validate_api_claims.py` checks it against the installed StatsPAI: every `sp.*` name resolves, every `sp.*(...)` call in a code block binds to the real signature (for a function that takes `**kwargs`, to its agent schema or forwarding target), and the result attributes on the gate's smoke-fit list exist.

## §A. Epidemiology / public health pipeline (Mode A)

> **Convention**: STROBE (observational) / TRIPOD-AI (prediction) reporting. A modern epidemiology reference design is **target-trial emulation** (Hernán & Robins): write the protocol of the hypothetical RCT first, then emulate it with observational data using a doubly-robust estimator. Outcomes are commonly **risk differences, risk ratios, hazard ratios, or restricted mean survival time**, not just OLS coefficients. The skill mirrors the AER 8-section flow but swaps the Step-4 estimator stack and adds survival/MR-specific reporting rows.

Running example: `statin_initiation → 5-yr_MACE` in an EHR cohort (`patient_id / index_date / age / sex / ldl_baseline / comorbidity_index / followup_days / event`). The exposure is time-varying, confounders are time-varying, and competing-risk censoring matters — the canonical setting where naïve OLS / Cox-with-baseline-adjustment is biased.

### A.0 Cohort construction & target-trial protocol

```python
import statspai as sp

# Eligibility, treatment-strategy, time-zero, follow-up, outcome — written down BEFORE estimation.
# NOTE the two validated enum fields:
#   • assignment    ∈ {"randomization", "observational emulation"}
#   • causal_contrast ∈ {"ITT", "per-protocol", "as-treated", "observational-analogue"}
# (free-text in these two fields raises ValueError). Put the prose description in `notes=`.
protocol = sp.target_trial.TargetTrialProtocol(
    eligibility           = "adults 40-75, LDL ≥ 130, no prior MI/stroke, no statin in 12mo washout",
    treatment_strategies  = ["initiate statin within 30d of index", "no statin within 30d"],
    assignment            = "observational emulation",       # enum — not free text
    time_zero             = "index_date (first eligible cardiology visit)",
    followup_end          = "first MACE / death / disenrollment / index_date + 5yr",
    outcome               = "first MACE (composite: MI, stroke, cardiovascular death)",
    causal_contrast       = "per-protocol",                  # enum — not free text
    analysis_plan         = "IPTW-MSM + g-formula + TMLE triplet; report all three with CIs",
    baseline_covariates   = ["age","sex","ldl_baseline","comorbidity_index","smoker"],
    time_varying_covariates = ["ldl_current"],
    notes                 = "emulate randomization via IPTW + g-formula; 5-yr risk difference",
)
# Signature: target_trial_emulate(protocol, data, outcome_col, treatment_col,
#                                 time_zero_filter=None, weights=None) -> TargetTrialResult.
# Eligibility is applied as `data.query(protocol.eligibility)` UNLESS you pass a
# `time_zero_filter` callable (which then defines the eligible/time-zero rows and
# lets `eligibility` stay human-readable prose). Use the callable for non-query-able rules:
cohort_res = sp.target_trial_emulate(
    protocol, df, outcome_col="mace", treatment_col="statin_initiation",
    time_zero_filter=lambda d: d["age"].between(40, 75) & (d["ldl_baseline"] >= 130),
)
cohort = df   # downstream estimators run on the eligible analysis frame you constructed
```

### A.1 Table 1 — baseline characteristics by exposure

```python
# Same sumstats stack as AER mode; binary 0/1 by= auto-renders Control/Treated.
mc = sp.mean_comparison(cohort, ["age","sex","ldl_baseline","comorbidity_index","smoker"],
                        group="statin_initiation", test="ttest",
                        title="Table 1. Baseline characteristics by statin initiation")
mc.to_word ("tables/table1_epi.docx")
mc.to_excel("tables/table1_epi.xlsx")
```

### A.2 Identification — DAG, propensity overlap, KM curves

```python
# 2.1 DAG (manual or LLM-assisted). sp.dag(spec) parses an edge STRING
# ("A -> B; C -> B"); build edges with the string spec or chained .add_edge(parent, child)
# (singular — there is no .add_edges). Back-door sets come from .adjustment_sets(exposure,
# outcome) (PLURAL, positional) and return a LIST of valid sets.
dag = sp.dag(
    "age -> ldl_baseline; age -> statin_initiation; "
    "ldl_baseline -> statin_initiation; ldl_baseline -> mace; "
    "comorbidity_index -> statin_initiation; comorbidity_index -> mace; "
    "statin_initiation -> mace"
)
adj = dag.adjustment_sets("statin_initiation", "mace")    # list of back-door sets, e.g. [{...}]

# 2.2 Propensity-score overlap (positivity check; epi convention before any IPW)
# Returns a pd.Series of fitted PS — draw mirrored histograms by exposure.
ps = sp.propensity_score(cohort, treatment="statin_initiation",
                          covariates=["age","sex","ldl_baseline","comorbidity_index","smoker"],
                          method="logit")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(ps[cohort["statin_initiation"]==1], bins=40, alpha=0.5, label="Treated")
ax.hist(ps[cohort["statin_initiation"]==0], bins=40, alpha=0.5, label="Control")
ax.set_xlabel("Estimated propensity score"); ax.legend()
fig.savefig("figures/figA1_ps_overlap.png", dpi=300)

# 2.3 Crude KM curves by exposure (descriptive identification graphic).
# KMResult.plot() returns a bare Axes (NOT a (fig, ax) tuple) — save via ax.figure.
km = sp.kaplan_meier(cohort, duration="followup_days", event="mace", group="statin_initiation", conf_type="log-log")
ax = km.plot()
ax.figure.savefig("figures/figA2_km.png", dpi=300)
```

### A.3 Main estimate — IPTW · g-formula · TMLE triplet (the modern epi standard)

Report **all three** in one `regtable` so the reader sees convergent doubly-robust evidence — this is the epi equivalent of the AER design horse race:

```python
# (1) IPTW marginal structural model
iptw = sp.msm(cohort, y="mace", treat="statin_initiation",
              id="patient_id", time="month",
              time_varying=["ldl_current","comorbidity_index"],
              baseline=["age","sex"])

# (2) Parametric g-formula (g-computation). `sp.gformula` is a MODULE, not a function.
# For a point-treatment g-formula use the top-level `sp.g_computation`:
gcomp = sp.g_computation(cohort, y="mace", treat="statin_initiation",
                         covariates=["age","sex","ldl_baseline","comorbidity_index","smoker"])
# For a TIME-VARYING treatment/confounder g-formula (the Robins setting) use the
# Monte-Carlo g-formula in the module:
#   sp.gformula.gformula_mc(cohort, treatment_cols=["statin_t1","statin_t2",...],
#                           confounder_cols=[["ldl_t1"],["ldl_t2"],...],
#                           outcome_col="mace", strategy=(1,1,1), control_strategy=(0,0,0),
#                           id_col="patient_id", time_col="month")

# (3) TMLE -- doubly robust targeted learning estimator.
# Pass an sklearn-style library list for nuisance learners; statspai stacks them
# internally via SuperLearner. Keep `outcome_library` and `propensity_library`
# explicit so the reviewer can see your nuisance choices.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
sl_lib = [LogisticRegression(max_iter=1000),
          GradientBoostingClassifier(),
          RandomForestClassifier()]
tmle = sp.tmle(cohort, y="mace", treat="statin_initiation",
               covariates=["age","sex","ldl_baseline","comorbidity_index","smoker"],
               outcome_library=sl_lib, propensity_library=sl_lib)

# (3-bis) HAL-TMLE if you want a fully nonparametric variant.
# `variant=` only accepts "delta" (the default; "projection" is NotImplemented).
hal = sp.hal_tmle(cohort, y="mace", treat="statin_initiation",
                  covariates=["age","sex","ldl_baseline","comorbidity_index","smoker"],
                  variant="delta")

# Convergent-evidence table — risk difference at 5 years
rt = sp.regtable(iptw, gcomp, tmle, hal,
                 model_labels=["(1) IPTW-MSM","(2) g-formula","(3) TMLE","(4) HAL-TMLE"],
                 stats=["N","Effect type","Risk diff. (RD)","Risk ratio (RR)"],
                 title="Table 2. Effect of statin initiation on 5-yr MACE — convergent estimators")
rt.to_word ("tables/table2_epi.docx"); rt.to_excel("tables/table2_epi.xlsx")
```

### A.4 Survival outcomes — KM / AFT / restricted mean

```python
import pandas as pd

# AFT formula LHS is "duration + event" (NOT R-style Surv(time, event)).
aft = sp.aft("followup_days + mace ~ statin_initiation + age + sex + ldl_baseline",
             cohort, family="weibull")
print(aft.summary())                                  # AFTResult exposes .summary() (text)

# AFTResult exposes `.params` (a pd.Series) + `.std_errors`, so it drops STRAIGHT into
# sp.regtable → SEs, stars, and one-line Word/Excel/LaTeX export via the RegtableResult.
# (AFTResult itself still has no `.to_word`/`.to_latex`/`.conf_int` — go through regtable.)
aft_tbl = sp.regtable(aft, model_labels=["Weibull AFT"],
                      title="Table 3. Survival (accelerated failure time)")
aft_tbl.to_word("tables/table3_survival.docx"); aft_tbl.to_excel("tables/table3_survival.xlsx")
# Read N / events / AIC straight off the result for the table footer:
print(f"N={aft.n}, events={aft.n_events}, family={aft.family}, AIC={aft.aic:.1f}")
# Manual fallback only if you need a custom layout (regtable is preferred):
#   pd.DataFrame({"coef": aft.beta, "se": aft.se}, index=aft.var_names)

# For a CAUSAL survival estimand (risk/RMST contrast under unconfoundedness), use the
# doubly-robust longitudinal-TMLE survival estimator instead of a raw AFT:
#   sp.ltmle_survival(cohort, ...)   # returns an LTMLESurvivalResult
```

### A.5 Mendelian randomization (genetic IV — when relevant)

```python
import pandas as pd

# Standard MR triple: IVW → Egger → weighted median, on summary statistics.
# Each mr_* returns a DICT (keys: estimate, se, ci_lower, ci_upper, p_value, ...) —
# NOT a result object, so it does NOT go into sp.regtable. Assemble a DataFrame instead.
ivw    = sp.mr_ivw   (beta_exposure, beta_outcome, se_exposure, se_outcome)
egger  = sp.mr_egger (beta_exposure, beta_outcome, se_exposure, se_outcome)   # 'intercept'(_p) = pleiotropy test
median = sp.mr_median(beta_exposure, beta_outcome, se_exposure, se_outcome, penalized=True)

mr_table = pd.DataFrame(
    {"IVW": ivw, "MR-Egger": egger, "Weighted median": median}
).T[["estimate", "se", "ci_lower", "ci_upper", "p_value"]]
mr_table.to_excel("tables/table4_mr.xlsx")        # or .to_latex() / .to_markdown()
print(mr_table)
# Egger intercept ≠ 0 (egger["intercept_p"] < 0.05) flags directional pleiotropy.
```

### A.6 Robustness — E-value, bounds, principal stratification

```python
# E-value: minimum strength of unmeasured confounding to explain away the result.
# CausalResult exposes `.estimate` and `.ci` (there is NO `.point_estimate`).
ev = sp.evalue(estimate=tmle.estimate, ci=tmle.ci, measure="RR")
# → "E-value 1.84; CI E-value 1.42" (a confounder must be ~2x associated with both
#   exposure and outcome to nullify the effect — interpret in your domain)

# Manski / Lee bounds — `sp.bounds` is a MODULE; call the specific estimator:
bds  = sp.bounds.manski_bounds(cohort, y="mace", treat="statin_initiation")
# selection bias (truncation-by-death / attrition): sp.bounds.lee_bounds(..., selection="observed")

# Principal stratification — BOTH `strata` and `instrument` must be BINARY (0/1) columns
# that already exist in the frame (build them first; they are NOT created for you).
cohort["high_density_zip"] = (cohort["zip_pharmacy_density"] > 0).astype(int)   # binary instrument
cohort["adherent"]         = (cohort["adherence_score"] > 0.8).astype(int)      # 0/1 stratum indicator
ps_strat = sp.principal_strat(cohort, y="mace", treat="statin_initiation",
                              instrument="high_density_zip",
                              strata="adherent")
```

### A.7 Reporting checklist (epi-specific footer for `notes=`)

When producing the Table-2 footer, include — in addition to the AER stars/SE language:

- Cohort size, person-years of follow-up, event count
- **Adjustment set** (variables in the back-door set, not just "controls")
- **Positivity diagnostic** (PS truncation rule, % of cohort with extreme weights)
- **E-value** for the main effect and its CI bound
- For survival: **proportional-hazards check** (Schoenfeld residuals p-value) or "PH violated, RMST reported instead"
- STROBE checklist completion (cite as a supplementary file)

> **Output path stays identical**: every estimator above returns a `CausalResult` and slots straight into `sp.regtable(...) / sp.collect(...) / sp.paper_tables(...)`. Doubly-robust estimators (TMLE, HAL-TMLE, AIPW) are preferred over single-robust IPTW or g-formula alone — report all three for transparency, but treat TMLE as the primary.

---
