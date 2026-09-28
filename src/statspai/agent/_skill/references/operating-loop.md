# Operating loop, pipeline map and domain modes

> Reference file of the `statspai-analysis` skill. Read the section you need; every `sp.*` name, signature and result attribute here is checked by `validate_api_claims.py` against the installed StatsPAI.

## Why for Agents

1. **Self-describing**: `sp.list_functions()` / `sp.describe_function(name)` / `sp.function_schema(name)` — registered symbols are discoverable without doc lookup.
2. **Structured results**: mature estimators return result objects with methods such as `.summary()`, `.plot()`, `.diagnostics`, `.to_latex()`, `.to_word()`, `.cite()` when supported.
3. **One import, full pipeline**: data contract → Table 1 → estimand-first DSL → identification graphs → main table → heterogeneity → mechanisms → robustness → replication package.
4. **Estimand-first**: `sp.causal_question(...).identify()` forces the "DID vs RD vs IV?" decision *before* estimation, with the identifying assumption written down — the way a referee expects to read it.

## SkillOpt-derived operating loop (read before the playbook)

SkillOpt's useful lesson for this skill is procedural, not cosmetic: a skill is a
bounded decision policy that should improve from rollout evidence while preserving
verified behavior. Treat every StatsPAI request as a mini rollout:

1. **Route the mode first**: choose Default/AER, Mode A/epi, Mode B/ML-causal, or
   a narrow export-only path from the user's words. Do not run the full paper
   pipeline when the request is only "make Table 1" or "export this regression".
2. **Freeze the contract before estimating**: name `y`, treatment/exposure,
   unit/time ids, estimand, design, required artifacts, and install extras. If any
   field is missing, infer only when the column names make the choice obvious;
   otherwise produce a short blocking checklist instead of hallucinating columns.
3. **Start from the smallest verified call shape**: prefer the skeleton and the
   relevant section-specific snippet over ad hoc API guesses. For an unfamiliar
   function, call `sp.describe_function(name)` / `sp.function_schema(name)` before
   writing code.
4. **Widen one block at a time**: data contract → plan → diagnostic figure →
   main estimate → robustness/export. After each block, read warnings and object
   attributes before passing the result downstream.
5. **Gate the answer on artifacts, not intentions**: final responses should list
   the files produced, the identifying assumption, the estimator class, and any
   failed or skipped gate. Never claim "paper-ready" if Word/Excel/LaTeX exports
   or required diagnostics were not actually generated.
6. **Turn failures into bounded corrections**: if a call raises, fix the smallest
   wrong rule (signature, result type, optional extra, plot return shape) and
   continue from the last verified artifact. Do not rewrite the pipeline wholesale.

### SkillOpt-style execution gate (task-local card)

Before generating or revising StatsPAI analysis code, compress the request into a task-local `best_skill` card:

```text
best_skill: <mode + design + artifact target>
train_signal: <current failure, user goal, or missing evidence>
selection_split: <focal dataset/spec/output used to judge the candidate>
heldout_gate: <checks the patch must pass beyond the focal example>
accepted_patterns: <rules to reuse after validation>
rejected_patterns: <failed shortcuts not to retry without new evidence>
patch_scope: <one estimator/sample/export/robustness change>
reject_if: <conditions that force rollback to the last passing spec>
```

1. **Route card**: record the mode (`econ`, `epi`, or `ml-causal`), estimand, identification design, focal outcome/treatment, StatsPAI install extras, and required artifacts.

2. **Bounded edit**: change one decision at a time (sample rule, estimator, optional extra, plot return shape, export format, or robustness check). Prefer the smallest patch that can pass validation.

3. **Selection split discipline**: treat the user's immediate failure or requested artifact as the selection split. Reserve at least one alternate outcome, sample window, estimator family, or export target as the held-out gate.

4. **Held-out gate**: define checks before running code: row counts, key uniqueness, treatment support, missingness thresholds, expected table/figure files, and one non-focal robustness/specification that the change must not break.

5. **Reject buffer**: if a candidate spec fails the gate, log the failure, code diff, and gate output in `analysis_log.md`; revert to the last passing spec and do not retry the same unchecked pattern.

6. **Slow/meta update**: at the end of the task, write down `accepted_patterns` and `rejected_patterns` from the trajectory. Do not widen the canonical project template from a single passing run.

7. **Promote only after validation**: only turn a one-off fix into reusable project boilerplate after it passes the current data and at least one alternate outcome/sample/specification.

### Acceptance gates by request type

| Request type | Minimum gates before final answer |
|---|---|
| Export-only / `outreg2` equivalent | At least one `RegtableResult` or `Collection` object is created; requested `.docx` / `.xlsx` / `.tex` paths are written or the exact missing optional dependency is reported |
| AER DID / event study | `sp.causal_question(...).identify()` saved or printed; CS/SA result used for the event-study figure; numerical pre-trends checked separately with `sp.event_study(...)` or equivalent; Table 2 and at least one robustness/sensitivity artifact produced |
| IV | First-stage F and instrument story reported before the 2SLS coefficient; no `\| fe` formula is passed to `sp.ivreg`; FE-IV needs explicit dummy construction or a stated limitation |
| RD | McCrary/manipulation check plus RD plot are produced before the treatment-effect table; bandwidth/kernel sensitivity is in the robustness block |
| Matching / weighting | Balance or love plot is produced before outcome estimation; weights are carried into the Table 1 / balance export when applicable |
| Epi / target-trial | Target-trial protocol is written before modeling; positivity/overlap is checked; IPTW/g-formula/TMLE estimates are compared when data support them; E-value or equivalent sensitivity is reported |
| ML causal / CATE | Train/holdout split and nuisance learners are explicit; per-row CATE source is valid (`model_info["cate"]` for meta-learners or `cf.effect(X)` for forests); policy/OPE claims use holdout data |
| Stata/R migration | Use StatsPAI's self-description or translator surface first; preserve semantic notes for unsupported options instead of silently pretending full parity |

### Maintenance rule for future skill edits

When improving this skill itself, follow a SkillOpt-style accept rule: propose a
small add/delete/replace edit, then accept it only if it helps a concrete failure
case and does not regress the verified skeleton, export cookbook, or Common
Mistakes table. Use `EVALS.md` as the held-out gate set for future skill edits.
Keep reusable fixes near the earliest section where an agent will need them; keep
rare API traps in Common Mistakes.

## The AER-style empirical pipeline

The skill mirrors the canonical sections of an applied AER / QJE / AEJ paper. Each step below is one paper section and one set of artifacts on disk.

```
Paper section               Step  StatsPAI moves
─────────────────────────── ───── ────────────────────────────────────────────────
Pre-Analysis Plan           −1    sp.power.* + freeze IdentificationPlan to disk
§1. Data                     0    data_contract + sample-construction log (footnote 4)
§1.1 Descriptives (Table 1)  1    sp.sumstats · sp.balance_table · sp.describe
§2. Empirical Strategy       2    write equation + identifying assumption + sp.causal_question
   (LLM-DAG addendum)        2.5  sp.llm_dag_propose · validate · constrained
§3. Identification graphics  3    event-study · first-stage F · McCrary · love plot
§4. Main Results (Table 2)   4    progressive controls + FE  (sp.regtable / sp.causal)
§5. Heterogeneity (Table 3)  5    sp.subgroup_analysis · sp.continuous_did · CATE
§6. Mechanisms               6    sp.mediation · sp.decompose
§7. Robustness gauntlet      7    placebo · Oster · honest_did · E-value · 2-way / Conley SE · spec_curve
§8. Replication package      8    .to_latex() · .plot() · reproducibility stamp
```

> **All code blocks below share one running example (`training → wage`, with `worker_id / firm_id / year / age / edu / tenure`) purely for readability.** Column names, `population`, `estimand`, and `design` values are **illustrative** — substitute the user's actual columns and research question. Only `sp.*` function names and argument *shapes* are normative.

## Three domain modes (default = AER econ; alternates = epi & ML-causal)

The default playbook above is **AER-style applied econometrics** — the AEA convention: written-out estimating equation, identifying assumption table, design horse-race, full robustness gauntlet. The skill **also** ships two parallel sub-pipelines for the other two big causal-inference traditions, each reusing the same export stack (`sp.regtable / sp.collect / sp.paper_tables`) and result objects:

| Mode | Reader convention | Identification stack | Reporting stack | Jump to |
|---|---|---|---|---|
| **Default — Applied Econ (AER / QJE / AEJ)** | "Show the equation + identifying assumption + design horse-race; controls visible; clustered SE" | DID / IV / RD / SCM / matching / `feols` HDFE | AER house-style multi-column `regtable` + 8-section paper layout | §−1 → §8 (entire playbook above) |
| **Mode A — Epidemiology / Public Health** | "STROBE / TRIPOD-AI; target trial protocol; doubly-robust estimand; absolute & relative risk; KM survival" | Target-trial emulation · IPTW · g-formula · TMLE · Mendelian randomization · KM/AFT | Same `regtable` + `collect`, with risk-difference / hazard-ratio / E-value rows | §A. Epidemiology pipeline |
| **Mode B — ML Causal Inference** | "DML / meta-learners / causal forest / DR-learner; CATE distribution; policy value" | DML · S/T/X/R/DR-Learner · GRF causal forest · Dragonnet/TARNet/CEVAE · BCF · matrix completion | `regtable` ML horse-race + `cate_plot` + policy-value table + `conformal_causal` PI | §B. ML causal pipeline |

**How to invoke a non-default mode** (Claude / agent picks this up from the user's wording):

| User says... | Mode the skill switches to |
|---|---|
| "Run a DID / IV / RD / event study", "AER table", "applied micro" | Default (AER econ) |
| "Target trial emulation", "g-formula", "IPTW", "TMLE", "Mendelian randomization", "STROBE / TRIPOD", "公共健康 / 流行病学", "epi pipeline", "RWE study", "cohort study", "case-control" | Mode A (Epi) |
| "DML", "double machine learning", "causal forest", "meta-learner", "CATE", "Dragonnet", "BCF", "policy learning", "conformal causal", "ML causal", "uplift modeling", "因果机器学习" | Mode B (ML causal) |
| "Mix" (e.g. "estimate DID + then ML CATE on the heterogeneity") | Default + Mode B in sequence — every estimator returns the same `CausalResult`, drop them all into one `sp.regtable(...)` for the horse-race column |

The three modes share **the same export stack, the same `CausalResult` interface, and the same `sp.causal_question(...).identify()` estimand-first DSL** — switching modes only changes which Step 4 estimators you reach for, not the surrounding scaffolding. If you only want descriptive stats / Table 1 / a balance check, the AER `sp.sumstats` / `sp.mean_comparison` / `sp.collect` calls work in all three modes.

## Paper-ready figure & table inventory (what to produce by section)

A modern AER paper has **5–7 figures** and **3–5 main tables** + an appendix robustness table. Every step below should leave at least one numbered artifact on disk. Default file names assume parallel `.tex` / `.docx` / `.xlsx` exports (the agent should produce all three so co-authors can edit in Word / Excel and the build system can use LaTeX):

| § | Artifact | StatsPAI primitive | Filenames (write all three) |
|---|---|---|---|
| §1 | **Figure 1**: raw trends / treatment rollout | `sp.parallel_trends_plot` · `sp.treatment_rollout_plot` | `figures/fig1_trends.png` |
| §1 | **Table 1**: summary stats (full / treated / control + Δ) | `sp.sumstats` + `sp.mean_comparison(...).to_word()/.to_excel()` (or `sp.collect().add_summary().add_balance()`) | `tables/table1_summary.{tex,docx,xlsx}` |
| §3 | **Figure 2**: identification graphic (event-study / first-stage / McCrary / RD scatter / SCM trajectory) | `sp.enhanced_event_study_plot` · `sp.binscatter` · `sp.rdplot` · `sp.rddensity().plot()` · `sp.synthdid_plot` | `figures/fig2_identification.png` |
| §4 | **Table 2**: main results — progressive controls | `rt = sp.regtable(M1...M5, template="aer"); rt.to_word(...); rt.to_excel(...)` | `tables/table2_main.{tex,docx,xlsx}` |
| §4 | **Table 2-bis**: design horse-race (OLS / IV / DID / DML) | `sp.regtable(ols, iv, did, dml, ...).to_word/.to_excel` | `tables/table2b_designs.{tex,docx,xlsx}` |
| §4 | **Figure 3** (optional): coefficient plot across specs | `sp.coefplot(M1, M2, M3, M4)` | `figures/fig3_coef.png` |
| §5 | **Table 3**: heterogeneity by subgroup | `sp.regtable(g_full, g_male, g_fem, g_q1...q4).to_word/.to_excel` | `tables/table3_heterogeneity.{tex,docx,xlsx}` |
| §5 | **Figure 4**: dose-response / CATE | `sp.dose_response(...).plot()` · `sp.cate_plot` · `sp.cate_group_plot` | `figures/fig4_cate.png` |
| §6 | **Table 4**: mechanisms (mediation / decomposition) | `sp.regtable(total, direct, indirect).to_word/.to_excel` | `tables/table4_mechanisms.{tex,docx,xlsx}` |
| §7 | **Table A1**: robustness master (one row per check) | `sp.regtable(rob1...robN, panel_labels=[...]).to_word/.to_excel` — or `sp.paper_tables(robustness=[...]).to_docx()` | `tables/tableA1_robustness.{tex,docx,xlsx}` |
| §7 | **Figure 5**: spec curve | `sp.spec_curve(...).plot()` | `figures/fig5_spec_curve.png` |
| §7 | **Figure 6**: honest-DID sensitivity plot (+ text dashboard) | `sp.sensitivity_plot(sp.honest_did(cs, ...))` for the figure; `print(sp.sensitivity_dashboard(result).summary())` for the Cinelli–Hazlett/Oster/E-value numbers (text, not a figure) | `figures/fig6_sensitivity.png` |
| §8 | **Replication bundle**: all tables in one Word/Excel/LaTeX file | `sp.collect("Paper").add_summary(...).add_regression(...)...save("paper.{docx,xlsx,tex}")` — or `sp.paper_tables(main=, heterogeneity=, robustness=, placebo=).to_docx/.to_xlsx` | `replication/paper.{docx,xlsx,tex}` |

> Every `CausalResult` and OLS model can be passed straight into `sp.regtable(...)`, `sp.coefplot(...)`, **and `sp.collect()`**. Don't hand-roll LaTeX, and don't render Word/Excel from pandas — the export functions apply book-tab borders, AER-style stars, and the right SE label automatically.

---
