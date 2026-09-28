---
name: statspai-analysis
description: Run a full empirical / causal analysis in Python with StatsPAI (import statspai as sp) — applied-economics style (DID / RD / IV / synthetic control / DML / matching with estimating equation, identifying assumption, Table 1, main table, event-study figure, robustness gauntlet, Word / Excel / LaTeX export), epidemiology style (target-trial emulation, IPTW + g-formula + TMLE, Mendelian randomization, survival, E-values, STROBE / TRIPOD reporting), ML-causal style (DML, meta-learners, causal forest, CATE, policy learning, conformal, fairness), or decomposition style (Oaxaca-Blinder, Kitagawa, DFL, Gelbach, RIF). Use when the user names StatsPAI / statspai, asks for an AER / QJE-style empirical pipeline, event study, honest DID, spec curve, regression tables to Word / Excel (outreg2 / esttab / modelsummary equivalent), a replication bundle, a target-trial or g-formula analysis, or CATE / policy learning in Python.
---

# StatsPAI: agent-native causal inference, paper-ready

StatsPAI is one `import statspai as sp` over 1,200+ registered functions for
causal inference and applied econometrics, numerically aligned with Stata / R
reference implementations, with self-describing schemas and result objects that
export to LaTeX / Word / Excel / BibTeX. This skill drives it through the
canonical pipeline of an applied empirical paper, in one of three modes
(default applied econ; epidemiology; ML-causal), and leaves paper-ready
artifacts on disk at every step.

- **Source**: https://github.com/brycewang-stanford/StatsPAI
- **Install**: `pip install "statspai[fixest,plotting]"` (default pipeline);
  add `neural` for Dragonnet / TARNet / CEVAE, `text` for text-as-treatment.
  The bare install cannot run `sp.feols` (any `y ~ x | fe` regression) or make
  figures — see `references/operating-loop.md` for the extras matrix.
- **Verified**: every `sp.*` name, signature and result attribute in this skill
  is checked by `validate_api_claims.py` against the installed package
  (`statspai skill validate`). There is no version stamp to go stale.
- **Paper**: Wang & Rozelle (2026), *Journal of Open Source Software* 11(125),
  10604, <https://doi.org/10.21105/joss.10604>.

## Operating loop (every request)

1. **Route the mode first** from the user's words — default applied econ,
   Mode A epi, Mode B ML-causal, or an export-only path. Do not run the whole
   paper pipeline for "make Table 1" or "export this regression".
2. **Freeze the contract before estimating**: `y`, treatment / exposure,
   unit / time ids, estimand, design, required artifacts, install extras. Infer
   a missing field only when the column names make it obvious; otherwise return
   a short blocking checklist instead of inventing columns.
3. **Pick the estimator with the tables, not from memory**:
   `sp.route('did', design='staggered', timing_random='no', covariates='none')`
   (also `iv`, `rd`, `matching`, `ml_causal`, `qte`, `dynamic_panel`) returns
   the registered call, an example, why, and the assumptions it adds;
   `sp.decision_guide(family)` lists the questions. `sp.recommend(df, ...)`
   routes from the data itself.
4. **Start from the smallest verified call shape**: the skeleton in
   `references/verified-skeleton.md`, then the section snippet. For anything
   unfamiliar, `sp.describe_function(name)` / `sp.function_schema(name)` before
   writing code; `sp.search_functions("weak instrument robust CI")` to find
   the name.
5. **Widen one block at a time**: data contract → plan → diagnostic figure →
   main estimate → robustness / export. After each block read the result's
   `violations()`, `next_steps()`, `degradations` and `model_info` before
   passing it downstream.
6. **Gate the answer on artifacts**: the final message lists the files
   produced, the identifying assumption, the estimator class, and any failed or
   skipped gate. Never say "paper-ready" when an export or a required
   diagnostic did not actually run.
7. **Turn failures into bounded corrections**: a `StatsPAIError` carries
   `code`, `recovery_hint`, `diagnostics` and `alternative_functions`; fix the
   smallest wrong rule and continue from the last verified artifact.

## Pipeline map (default mode)

```
Paper section               Step  StatsPAI moves
─────────────────────────── ───── ────────────────────────────────────────────
Pre-analysis plan            −1   sp.power(...) + freeze the IdentificationPlan
§1 Data                       0   data contract + sample-construction log
§1.1 Table 1                  1   sp.sumstats · sp.mean_comparison · sp.balance_table
§2 Empirical strategy         2   equation + assumption + sp.causal_question(...).identify()
   (LLM-DAG addendum)        2.5  sp.llm_dag_propose · validate · constrained
§3 Identification graphics    3   event study · first-stage F · McCrary · love plot
§4 Main results (Table 2)     4   progressive controls + FE (sp.regtable / sp.causal)
§5 Heterogeneity (Table 3)    5   sp.subgroup_analysis · sp.continuous_did · CATE
§6 Mechanisms                 6   sp.mediation · sp.decompose
§7 Robustness gauntlet        7   placebo · Oster · honest_did · E-value · Conley · spec_curve
§8 Replication package        8   sp.collect / sp.paper_tables · reproducibility stamp
```

Step-by-step code: `references/pipeline-econ.md`. Mode A:
`references/pipeline-epi.md`. Mode B: `references/pipeline-ml-causal.md`.

## Hard rules (read before writing code)

- **Estimand first.** `sp.causal_question(...).identify()` (or `sp.route`)
  decides DID vs RD vs IV before any estimate, and the identifying assumption
  is written down where a referee will read it.
- **`y ~ x | fe` goes through `sp.feols`** (pyfixest). `sp.regress` is OLS
  only and does not parse `|`; `sp.ivreg` takes no `| fe` either.
- **Staggered DID**: never a static TWFE coefficient. `sp.callaway_santanna`
  (+ `sp.aggte`) or `sp.sun_abraham`; event-study figures come from those
  results. A passed pre-trend test is not evidence for parallel trends —
  report `sp.honest_did` and `sp.pretrends_power`.
- **Covariates do not go in a TWFE / 3WFE regression** under conditional
  parallel trends: `sp.drdid`, `sp.callaway_santanna(..., x=[...],
  estimator='dr')`, `sp.ddd(..., method='dr')`.
- **IV**: report the first-stage / effective F before the 2SLS coefficient;
  below 10, `sp.anderson_rubin_ci` — not a 2SLS t-ratio.
- **RD**: `sp.rddensity` and `sp.rdplot` before the effect table; bandwidth
  sensitivity in robustness.
- **Matching / weighting**: balance (`sp.love_plot`, `sp.ps_balance`) before
  outcomes; say which estimand (ATT / ATE / ATO).
- **Export all three formats** (`.to_word`, `.to_excel`, `.to_latex`) from the
  same `RegtableResult` / `Collection`; never hand-roll LaTeX.
- **Citations only from `result.cite()` / `sp.bibtex(keys=[...])`** — never
  from memory (paper.bib is the single source of truth).
- **Plots return `(fig, ax)`**: unpack, then `fig.savefig(...)`.

## Where to read next

| Need | File |
| --- | --- |
| operating loop in full, acceptance gates by request type, mode table, figure & table inventory | `references/operating-loop.md` |
| the minimal end-to-end skeleton to copy | `references/verified-skeleton.md` |
| Steps −1 → 8 with code, one paper section each | `references/pipeline-econ.md` |
| target-trial / IPTW / g-formula / TMLE / MR / survival pipeline | `references/pipeline-epi.md` |
| DML / meta-learners / causal forest / policy / conformal / fairness pipeline | `references/pipeline-ml-causal.md` |
| Word / Excel / LaTeX export, regtable recipes, the 12 standard figures, notebook setup | `references/export.md` |
| method catalog by family; when to use StatsPAI vs alternatives | `references/method-catalog.md` |
| API traps that cost agents real time; agent integration pattern | `references/common-mistakes.md` |
| driving StatsPAI over MCP (`statspai-mcp`) or the shell (`statspai run`), data handles, routing, the result contract | `references/mcp-and-cli.md` |

## Validation

`statspai skill validate` (or `python validate_api_claims.py`) runs the gate:
every `sp.<name>` in this skill exists, the documented argument names bind, and
the documented result attributes hold on a live fit. `EVALS.md`-style held-out
checks for editing the skill itself live in the repository under
`StatsPAI_full_data_analysis_skill/EVALS.md`.
