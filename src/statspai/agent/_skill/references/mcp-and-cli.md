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

The MCP server lists ~35 curated tools by default (`--profile curated`);
`--profile full` lists every auto-generated tool (~580, ~2 MB — only for
clients that page their tool list). Every registered function stays callable
under every profile.

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
- `describe_data(data_id)` profiles a handle; `statspai://data/{id}` reads it.
- Fit with `as_handle=true` to get a `result_id`; chain it into
  `audit_result`, `honest_did_from_result`, `sensitivity_from_result`,
  `interpret_result`, `plot_from_result`.

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
On the CLI they go to stderr with exit code 3 (usage errors exit 2).

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
