# AGENTS.md — using StatsPAI from an AI agent

This file is for agents that *use* the package (coding assistants, MCP
clients, shell agents). Contributors to the repository read `CLAUDE.md`.

## One import, one alias

```python
import statspai as sp
```

Every public function is `sp.<name>` — no second-level imports. Examples,
docstrings and tests all use `sp.`.

## Discover → describe → call

```python
sp.search_functions("staggered adoption event study")   # ranked matches
sp.describe_function("callaway_santanna")                # params, assumptions, failure modes, example
sp.function_schema("callaway_santanna")                  # JSON-Schema tool spec (agent_native=True adds x_statspai)
sp.route("did", design="staggered", timing_random="no")  # data-free routing to the right call
sp.recommend(df, y="y", treatment="d", id="id", time="t")  # routing from the data
```

`sp.list_functions()` enumerates the 1,200+ registered names;
`sp.describe_function` reports `kind: "class"` for result / exception classes,
which are not tools.

## Read results the same way every time

```python
r = sp.did(df, y="y", treat="d", id="id", time="t")
r.to_dict(detail="agent")   # estimate / se / ci / diagnostics / violations / degradations / next_steps
r.violations()              # failed assumption checks only; empty + degradations ≠ clean
r.next_steps()              # follow-up checklist as a list of dicts; prints nothing (print_result=True to print)
r.result_card()             # estimand, sample, specification, inference, provenance (incl. seed), evidence tier
r.cite()                    # verified BibTeX; never write a citation from memory
```

Every result class carries these five methods, not only `CausalResult` /
`EconometricResults`: domain results get them from `ResultProtocolMixin`, with
fields the class does not record returned as `null` rather than guessed
(`scripts/result_protocol_audit.py --check` lists the few documented gaps).

`sp.audit(r)` lists the robustness checks still missing. Errors are
`StatsPAIError` subclasses with `code`, `recovery_hint`, `diagnostics`,
`alternative_functions` and `to_dict()`.

## Other surfaces

- **MCP**: `claude mcp add statspai -- statspai-mcp` (curated profile by
  default; `search_functions` / `describe_function` / `call_function` /
  `route_estimator` reach everything; `load_data` → `data_id` handles;
  `statspai://guide/{family}` serves the estimator-choice guides).
- **Shell**: `statspai run <function> --data FILE --arg k=v`, family
  shortcuts (`statspai did --data FILE --y ...`), `statspai route`,
  `statspai mcp`.
- **Claude Code skill**: `statspai skill install` copies the packaged
  `statspai-analysis` skill (pipeline playbooks for applied econ, epi and
  ML-causal analyses) into `~/.claude/skills`.

## Rules an agent must keep

1. Estimand and identifying assumption before estimation (`sp.causal_question`
   or `sp.route`).
2. Staggered DiD never as a static TWFE coefficient; covariates never inside a
   TWFE regression under conditional parallel trends.
3. Weak instruments → Anderson-Rubin sets, not 2SLS t-ratios.
4. Report `violations`, `degradations` and `runtime_warnings`; a missing
   diagnostic is a diagnostic that did not run.
5. Citations only from `result.cite()` / `sp.bibtex(...)`.

Full guide: `docs/guides/agent_api.md`; MCP workflow:
`docs/guides/economist_mcp_workflow.md`; machine-readable index: `llms.txt`.
