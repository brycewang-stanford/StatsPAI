# MCP workflow for empirical economists

This guide is for the common agent workflow: a researcher has a local `.dta`,
CSV, Parquet, or Arrow file and wants Claude Code, Codex, Cursor, or another
MCP client to run a defensible empirical design without hand-copying arrays
between tools.

StatsPAI's MCP server is deliberately local-first. It loads the file you point
it at, fits StatsPAI estimators, returns strict JSON, and caches fitted result
handles so follow-up diagnostics can reuse the same object.

## Configure the server

After installing StatsPAI, the package exposes `statspai-mcp` and the module
entry point `python -m statspai.agent.mcp_server`.

Claude Desktop-style configuration:

```json
{
  "mcpServers": {
    "statspai": {
      "command": "python",
      "args": ["-m", "statspai.agent.mcp_server"]
    }
  }
}
```

For clients that prefer a console script, use:

```json
{
  "mcpServers": {
    "statspai": {
      "command": "statspai-mcp",
      "args": []
    }
  }
}
```

The server speaks stdio JSON-RPC, advertises the `2025-06-18` MCP protocol
revision, and returns `structuredContent` with a compact `outputSchema`. Older
clients still receive the text JSON payload.

## Data handoff

Every data-bound tool accepts `data_path`.

Supported local and remote formats include:

| Format | Notes |
| --- | --- |
| `.dta` | Native Stata files through `pandas.read_stata` |
| `.csv`, `.tsv`, `.txt` | Delimited text |
| `.parquet`, `.pq`, `.feather`, `.arrow` | Column projection supported |
| `.xlsx`, `.xls` | Spreadsheet inputs |
| `.json`, `.jsonl` | JSON records |
| `file://`, `https://`, `http://`, `s3://`, `gs://` | Remote URLs through pandas/fsspec |

For large files, pass:

```json
{
  "data_columns": ["y", "d", "id", "year", "x1", "x2"],
  "data_sample_n": 50000
}
```

`data_columns` narrows the read when the backend supports it. `data_sample_n`
uses deterministic random sampling for quick exploration. Raise or disable the
loader cap with `STATSPAI_MCP_MAX_DATA_BYTES` only when the host has enough
memory.

When the MCP server loads a local file, tool results include
`data_provenance`: source path, format, requested columns/sample, file size,
mtime, and SHA-256. `statspai://result/<id>` exposes the same provenance through
the cached result metadata. Remote URLs are recorded after dropping query tokens;
StatsPAI does not hash remote bytes unless the data are first saved locally.

## The core loop

Use result handles. They keep the agent from copying fitted objects through
chat text.

```text
detect_design -> preflight/recommend -> fit(as_handle=true)
              -> audit_result -> sensitivity_from_result / plot_from_result
              -> bibtex
```

The fitted estimator returns `result_id`. Pass it to:

| Tool | Purpose |
| --- | --- |
| `audit_result` | Reviewer checklist of missing diagnostics |
| `brief_result` | One-line estimate summary |
| `interpret_result` | Grounded explanation, optionally using MCP sampling |
| `plot_from_result` | Inline PNG diagnostic plot |
| `sensitivity_from_result` | E-value / Oster / Cinelli-Hazlett style checks |
| `honest_did_from_result` | Rambachan-Roth sensitivity from DID/event-study results |
| `bibtex` | Verified BibTeX from StatsPAI's citation registry |

## One-call empirical pipelines

StatsPAI includes high-level MCP pipeline tools for the most common designs.

### DID

```json
{
  "name": "pipeline_did",
  "arguments": {
    "data_path": "/abs/cfps_panel.dta",
    "y": "lwage",
    "treat": "treated",
    "time": "year",
    "id": "pid",
    "cohort": "first_treat",
    "covariates": ["age", "age2", "edu", "industry"],
    "as_handle": true
  }
}
```

When `id` and `cohort` are supplied, the pipeline dispatches the
Callaway-Sant'Anna path. Otherwise it falls back to the 2x2 DID path. It then
adds the audit, honest-DID sensitivity where possible, Bacon diagnostics where
available, a narrative, and follow-up calls.

### IV

```json
{
  "name": "pipeline_iv",
  "arguments": {
    "data_path": "/abs/card.dta",
    "formula": "lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4)",
    "as_handle": true
  }
}
```

The IV pipeline reports the fitted estimate and weak-IV diagnostics, and
prioritizes Anderson-Rubin-style inference when first-stage evidence is weak.

### RD

```json
{
  "name": "pipeline_rd",
  "arguments": {
    "data_path": "/abs/lee_senate.dta",
    "y": "voteshare_next",
    "x": "margin",
    "c": 0,
    "as_handle": true
  }
}
```

The RD pipeline fits `rdrobust`, attempts the canonical RD plot, checks density
manipulation, and returns bandwidth-sensitivity follow-ups.

## Stata and R command migration

StatsPAI ships translator tools for one command at a time:

| Source | MCP tool | Examples |
| --- | --- | --- |
| Stata | `from_stata` | `regress`, `xtreg`, `reghdfe`, `ivreg2`, `ivreghdfe`, `csdid`, `did_imputation`, `synth`, `rdrobust`, `psmatch2`, count-panel commands |
| R | `from_r` | `feols`, `felm`, `lm`, `glm`, `plm`, `matchit`, `att_gt`, `did`, `synth` |

Use the built-in prompts:

| Prompt | Use |
| --- | --- |
| `stata_command_workflow` | Translate a single Stata command, fit the translated StatsPAI tool, then audit |
| `r_command_workflow` | Translate a single R expression, fit, then audit |
| `cross_language_command_check` | Translate Stata and R snippets, compare estimand/covariance conventions, then fit comparable StatsPAI calls |

These prompts are conservative. If a translator returns `ok=false`, the agent
should report suggestions instead of guessing. If Stata and R snippets imply
different controls, fixed effects, cohorts, or covariance conventions, treat
that as a mismatch before fitting.

For `psmatch2`, `from_stata` maps the common nearest-neighbor, kernel, radius,
`common`, and `ai()` paths onto `sp.psmatch2`. Convention-changing options such
as Stata's `probit` propensity score or ATE-focused requests are surfaced as
notes rather than silently claimed as exact parity; use `sp.match` directly for
ATE-oriented matching.

For `ivreghdfe`, `from_stata` maps the IV-with-fixed-effects command to the
same StatsPAI/fixest shape produced by R `feols(... | fe | endog ~ instr)`:
the formula contains the IV block and `fe=[...]` carries the absorbed fixed
effects. This is a migration contract for dispatching StatsPAI; it is not a
live Stata run.

## Cross-software verification discipline

StatsPAI already stores committed R and Stata parity artifacts under
`tests/r_parity/` and `tests/stata_parity/`. The Track A 3-way report compares
Python, R, and Stata outputs with pre-registered tolerances. Use it to decide
whether a StatsPAI result is:

| Status | Meaning |
| --- | --- |
| machine-level agreement | Point estimates match at tight tolerance |
| iterative/cross-fit agreement | Random folds or optimizers need a wider registered tolerance |
| convention gap | Backends target different estimands or covariance conventions |
| unavailable | No committed external artifact yet |

In MCP clients, read `statspai://parity/track-a-summary` for a compact JSON
summary of the committed Track A report. It includes strictness-tier counts,
module ids, Stata command labels, convention notes, and a `tool_evidence` index
keyed by common StatsPAI tool names, without loading the full markdown artifact
into the model context.

Do not call a live Stata/R comparison unless a separate Stata or R MCP server is
configured and actually invoked. The StatsPAI translators check whether command
semantics align; they are not themselves Stata or R runtimes.

## External data MCP servers

For World Bank, OECD, FRED, IMF, or OpenEcon-style data MCP servers, keep the
responsibilities separate:

1. Use the data MCP server to search and retrieve indicators.
2. Save the returned table to CSV, Parquet, Arrow, or `.dta`.
3. Pass the saved file path to StatsPAI through `data_path`.
4. Preserve the source metadata in your notebook, table notes, or paper
   appendix: provider, indicator id, query, retrieval date, and transformation
   code.

StatsPAI should analyze the bytes it receives. It should not invent missing
indicator values, source names, or retrieval provenance.

## Suggested first prompt

```text
Use the statspai MCP server. Load /abs/cfps_panel.dta.
Run pipeline_did with y=lwage, treat=treated, time=year, id=pid,
cohort=first_treat, controls age age2 edu industry, as_handle=true.
Then audit the result, run the first feasible high-importance follow-up,
render the canonical plot, and return a short methods paragraph plus a
regression-table export suggestion.
```
