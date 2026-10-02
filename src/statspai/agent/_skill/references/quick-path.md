# The short path: one question, one fit, one table

> Reference file of the `statspai-analysis` skill. Read the section you need; `validate_api_claims.py` checks it against the installed StatsPAI: every `sp.*` name resolves, every `sp.*(...)` call in a code block binds to the real signature (for a function that takes `**kwargs`, to its agent schema or forwarding target), and the result attributes on the gate's smoke-fit list exist.

Use this when the request is one estimate, one check or one export. Do not
open the paper pipeline for it. Five steps, each one call, and three places
where the right answer is to stop.

## 1. Route

```python
import statspai as sp

sp.search_functions("staggered adoption event study")   # find the name
sp.route("did", design="staggered", timing_random="no", covariates="none")
```

`sp.route(family, ...)` returns the registered call and the assumptions it
adds. If you cannot answer a question it asks (is adoption staggered? is
there a never-treated group?), **stop and ask the user**. Do not guess a
design from column names that do not settle it.

## 2. Describe

```python
card = sp.describe_function("callaway_santanna")
card["assumptions"], card["failure_modes"], card["pre_conditions"]
sp.function_schema("callaway_santanna")["parameters"]["required"]
```

Read `required` before writing the call. If a required column is not in the
data (no unit id, no first-treatment period), **stop**: return the list of
what is missing instead of inventing a column.

## 3. Fit, minimally

```python
df = sp.datasets.mpdta()
fit = sp.callaway_santanna(df, y="lemp", g="first_treat", t="year", i="countyreal")
```

Defaults first. Add an option only when the user asked for it or a
diagnostic says it is needed.

## 4. Inspect before you report

```python
fit.violations()            # assumption violations found in what was run
card = sp.result_card(fit)
card["assumptions"]["checks_summary"]   # passed / failed / not_run / not_applicable
card["evidence"]["outputs"]             # which outputs of THIS configuration were validated
card["provenance"]                      # versions, data hash, seed / reproducible
```

Three things to say out loud when they are true:

- **A check was not run.** `checks_summary["not_run"] > 0` with an empty
  `violations()` means nothing was found *because nothing looked*. Name the
  checks (`card["assumptions"]["checks"]`) and either run the function each
  entry gives under `run_with` or report them as not run. Never write
  "passes the diagnostics".
- **This configuration's standard error was not validated.**
  `card["evidence"]["outputs"]["se"]` other than `"reference"` means the
  function-level parity claim does not cover the options used. Say which
  outputs are covered; do not call the result aligned with Stata / R.
- **A step degraded.** A `degradations` entry on a workflow, pipeline or
  MCP result means a sub-step was skipped. Report it.

## 5. Export

```python
table = sp.regtable(fit, title="Effect on log employment")
table.to_word("table1.docx")
table.to_latex()
print(fit.cite())                      # BibTeX of the method, from paper.bib
```

List the files written. If an export raised, say so; a table that was not
written is not "ready".

## When to leave the short path

| The request mentions | Go to |
| --- | --- |
| a full paper, Table 1 through robustness | `pipeline-econ.md` |
| target trial, g-formula, TMLE, survival | `pipeline-epi.md` |
| CATE, policy learning, causal forest | `pipeline-ml-causal.md` |
| several tables, journal styling, Word / Excel bundles | `export.md` |
| a staggered design, weak instruments, RD inference choices | `modern-methods.md` |
| an error you do not recognise | `common-mistakes.md` |
| driving StatsPAI through MCP or the shell | `mcp-and-cli.md` |
