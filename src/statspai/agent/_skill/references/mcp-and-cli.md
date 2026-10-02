# Driving StatsPAI over MCP and the shell; the result contract

> Reference file of the `statspai-analysis` skill. The Python API, the MCP
> server and the CLI share one dispatch layer, so a payload obtained one way
> is the payload obtained the others.

## Three ways in

| Surface | Start | Discover | Run |
| --- | --- | --- | --- |
| Python | `import statspai as sp` | `sp.search_functions("...")`, `sp.describe_function(name)`, `sp.function_schema(name)`, `sp.route(family, ...)` | `sp.<name>(df, ...)` → result object |
| MCP (Claude Code / Desktop, Cursor) | `claude mcp add statspai -- statspai-mcp` | tools `search_functions` → `describe_function` → `route_estimator`; resources `statspai://function/{name}`, `statspai://guide/{family}` | `tools/call` on a listed tool, or `call_function(function=..., arguments={...})` for any of the 1,200+ |
| Shell | `pip install statspai` | `statspai search ...`, `statspai describe <name>`, `statspai route <family> --answer k=v` | `statspai run <name> --data f.csv --arg k=v`, `statspai did --data f.csv --y ... --treat ...` |

The MCP server lists the hand-curated tools by default (`--profile curated`,
the default for the CLI and for in-process `handle_request` alike);
`--profile core` is a smaller set and `--profile full` lists every
auto-generated tool (several hundred, megabytes of schema — only for clients
that page their tool list). Every registered function stays callable under
every profile, and `statspai://functions` always indexes all of them.

## Data over MCP

- First call: `load_data(data_path="/abs/panel.dta")` → `data_id` (also
  shape, dtypes, missing counts, head, numeric summary). Inline tables:
  `data_records=[{...}, ...]` or `data_csv="y,x\n1,2\n..."`.
- Every later call passes `data_id` instead of the path.
- `transform_data(data_id, operations=[{"op": "dropna", "columns": ["wage"]},
  {"op": "query", "expr": "year >= 2005"}, {"op": "assign", "column": "lwage",
  "expr": "log(wage)"}, {"op": "winsor", "columns": ["lwage"], "cuts": [1, 99]}])`
  returns a new handle; the chain is recorded and rides in every result's
  `data_provenance`, so a table note can state how the sample was built. Other
  ops: `select`, `drop`, `rename`, `fillna`, `sort`, `sample`, `wide_to_long`,
  `long_to_wide`, `mice`, `function` (any DataFrame-returning `sp.<fn>`).
- `query` / `assign` expressions are allowlisted: columns (backticks for odd
  names), constants, operators, `in` / `not in`, math functions (`log`, `exp`,
  `sqrt`, `abs`, …) and `col.isnull()` / `isin()` / `between()` /
  `str.contains()`. Attribute access, subscripts and `@local` references fail
  with `error_kind: "unsafe_expression"`.
- `describe_data(data_id)` profiles a handle; `statspai://data/{id}` reads it.
- Fit with `as_handle=true` to get a `result_id`; chain it into
  `audit_result`, `honest_did_from_result`, `sensitivity_from_result`,
  `interpret_result`, `plot_from_result`.

## Errors, limits and cancellation over MCP

- **Tool failures are results, not protocol errors.** A bad or expired
  `data_id` / `result_id`, an unreadable or refused file, a timeout or an
  estimator error returns a normal `tools/call` result with `isError: true`;
  `structuredContent` carries `error_kind` (`missing_data_handle`,
  `missing_result_handle`, `file_not_found`, `data_load_error`,
  `path_not_allowed`, `remote_disabled`, `invalid_arguments`, `timeout`,
  `internal_error`, or an estimator code such as `assumption_violation`),
  `message`, `hint` and, for handles, `miss_reason` (`ttl` / `lru` / `bytes`
  / `explicit` / `unknown`). Only protocol faults — malformed JSON-RPC, an
  unknown method or tool name, `params` / `arguments` that are not objects —
  are JSON-RPC errors.
- **`result_id` is never ignored**: a stale handle is `missing_result_handle`
  on every tool; a handle given to a tool that cannot use a fitted result is
  listed under `_unsupported_args`.
- **Output budget.** Results are capped at `max_output_bytes` (argument; server
  default `STATSPAI_MCP_MAX_OUTPUT_BYTES`, 256 KiB; `0` = no limit). Over
  budget, the longest lists / tables are cut first and each cut is listed
  under `truncated: [{path, total, shown}]`; estimate / SE / CI / p-value are
  never cut. The `text` block is the same object as `structuredContent`,
  serialised compactly.
- **Shortened risk lists.** `violations`, `runtime_warnings`, `degradations`
  and `warnings` are cut last. If `risk_details_complete` is `false`, the
  lists shown are not all the risks raised: read `risk_summary` (`total`,
  `omitted`, `by_severity`, `categories` per field) and report those counts,
  or repeat the call with a larger `max_output_bytes`. Never describe such a
  result as free of violations.
- **`replay_completeness`.** Only `level: "standalone"` means the `replay`
  line re-runs in a new Python process (given the file in `needs`).
  `session_replayable` depends on a handle in this server; `call_only`
  documents the call but cannot re-run it. Say which when you hand a user
  reproduction code.
- **`output_budget`.** Present when the result did not fit untouched.
  `status: "unavoidable_overflow"` means the never-cut fields exceed the
  budget and the response is larger than asked for.
- **`server_busy`.** An `isError` result with this `error_kind` means the call
  was not started (queue full, waited too long in the queue, or timed-out
  computations still running). Wait and retry; do not treat it as an
  estimation failure.
- **`isolation: {"mode": "process"}`.** The operator runs self-contained
  calls in a killable child process. Results are the same; a handle is only
  kept when you pass `as_handle=true`, which keeps the call in the server.
- **NaN / Inf** are sent as `null`; their JSON Pointers are listed under
  `_nonfinite` (`[{path, value: "NaN" | "Infinity" | "-Infinity"}]`), so an
  infinite SE is not mistaken for a missing one.
- **`replay`**: estimator results (and cached result handles, via
  `statspai://result/{id}`) carry the `sp.<fn>(data=data, ...)` call that
  reproduces them, with a comment naming the data source.
- **Liveness and cancel.** `ping` answers `{}`. `tools/call` runs on a worker
  pool (1 worker by default, `STATSPAI_MCP_WORKERS`), so `ping`, `tools/list`
  and resource reads are answered while an estimator runs.
  `notifications/cancelled` stops a call at its next progress checkpoint and
  suppresses its response. Timeouts (`STATSPAI_MCP_TOOL_TIMEOUT_SECONDS`,
  default 600) return `error_kind: "timeout"` with
  `worker_may_still_be_running: true` — Python threads cannot be killed, so a
  computation without checkpoints finishes in the background.
- **Operator controls.** `STATSPAI_MCP_DATA_ROOTS` (`os.pathsep`-separated)
  restricts which directories `data_path` may read (symlinks resolved;
  `file://` URLs included). Network URLs (`s3://`, `gs://`, `https://`) are
  off unless `STATSPAI_MCP_ALLOW_REMOTE=1`, and remote reads obey the same
  `STATSPAI_MCP_MAX_DATA_BYTES` cap as local files. The data cache is bounded
  by count (`STATSPAI_MCP_DATA_CACHE_SIZE`) and bytes
  (`STATSPAI_MCP_DATA_CACHE_BYTES`, default 2 GiB). Tools that can write a
  file carry `readOnlyHint: false`; `openWorldHint` is true only when remote
  loading is on.

## The result contract (what to read before trusting a number)

`result.to_dict(detail="agent")` (the MCP default) and `statspai run`
return, beyond `estimate / se / pvalue / ci / n_obs / method / estimand`:

| Key | Meaning | What to do |
| --- | --- | --- |
| `diagnostics` | scalar and one-level nested diagnostics from `model_info` (pre-trend test, McCrary p-value, first-stage F, …) | report them; a missing test is a test that did not run |
| `violations` | assumption checks that *failed* (kind, severity, test, value, threshold, recovery_hint, alternatives) | act on `severity: error` before reporting |
| `degradations` | a diagnostic that *crashed* while building the payload (`section`, `error_type`, `message`) | an empty `violations` with a non-empty `degradations` is not a clean result |
| `next_steps` / `suggested_functions` | the estimator's own checklist | run the `essential` ones |
| `runtime_warnings` (MCP / CLI) | Python warnings raised during the call (`ConvergenceWarning`, `AssumptionWarning`, few-cluster, weak-IV, LIML→2SLS fallback) | read them; they never reach stdout otherwise |
| `_unsupported_args` (MCP / CLI) | arguments the function could not bind and did **not** apply | a misspelt `cluster=` changed the standard errors — fix and re-run |
| `data_provenance` | file hash / inline hash / handle lineage | cite it in the replication stamp |
| `result_card` | estimand, sample, specification, inference, provenance, evidence tier | attach to the appendix |

Errors are structured: `error`, `error_kind` (`assumption_violation`,
`identification_failure`, `data_insufficient`, `convergence_failure`,
`numerical_instability`, `method_incompatibility`), `error_payload`
(`recovery_hint`, `diagnostics`, `alternative_functions`) and `remediation`.
On the CLI they go to stderr with an exit code by kind: 2 usage, 4 input
errors (`column_not_found` / `missing_arguments` / `unknown_argument`),
5 `missing_dependency`, 3 any other estimator error.

## Routing without data

```python
sp.decision_guide("did")          # questions + every route
sp.route("did", design="staggered", timing_random="no", covariates="yes")
# -> callaway_santanna(..., x=[...], estimator='dr'), why, assumptions added,
#    the guide heading to read, unanswered questions, next question to ask
```

Families: `did`, `iv`, `rd`, `matching`, `ml_causal`, `qte`, `dynamic_panel`.
Over MCP the same is `route_estimator(family=..., answers={...})`; the full
prose guide is `statspai://guide/{family}`.

## Shell examples

```bash
statspai run callaway_santanna --data panel.csv \
    --arg y=lemp --arg g=first_treat --arg t=year --arg i=countyreal --out cs.json
statspai did --data panel.csv --y lemp --treat treated --time year --id id \
    --covariates '["pop"]' --format summary
statspai run regress --data df.csv --arg "formula=y ~ x1 + x2" --arg 'vce="hc1"' --detail standard
statspai route iv --answer strength=weak
statspai skill install            # copy this skill to ~/.claude/skills/statspai-analysis
statspai skill validate           # re-run the API-claim gate against the installed package
```
