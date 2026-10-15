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

`sp.list_functions(kind="function")` enumerates the callable functions
(`sp.list_functions()` also lists result and exception classes, which are not
tools). `sp.search_functions` understands Chinese queries too.

## Example data

```python
sp.list_datasets()            # name, design, source, simulated, columns
df = sp.load_dataset("card_1995")
df.attrs["simulated"]          # False: the real NLSYM extract
```

Several datasets are **simulated replicas that carry the names of real ones**
(`mpdta`, `nsw_dw`, `basque_terrorism`, `german_reunification`,
`angrist_krueger_1991`, `currency_union_panel`). An estimate on them describes
the simulation, not the published study: never report it as a finding about
the real data. Every frame carries `attrs["simulated"]`, and the first load of
a replica in a process warns.

## Read results the same way every time

```python
r = sp.did(df, y="y", treat="d", id="id", time="t")
r.to_dict(detail="minimal") # the headline: estimate / se / pvalue / ci / alpha / n_obs
r.to_dict(detail="agent")   # + diagnostics / violations / degradations / next_steps
r.violations()              # failed assumption checks only; empty + degradations ≠ clean
r.next_steps()              # follow-ups as dicts; `executable: True` runs as written with result / df bound
r.result_card()             # estimand, sample, specification, inference, provenance (incl. seed), evidence tier
r.cite()                    # verified BibTeX; never write a citation from memory
```

Every result class carries these five methods, not only `CausalResult` /
`EconometricResults`: domain results get them from `ResultProtocolMixin`, with
fields the class does not record returned as `null` rather than guessed
(`scripts/result_protocol_audit.py --check` lists the few documented gaps).

Every level of `to_dict` is strict JSON (no NaN). For a regression
(`sp.regress`, `sp.iv`, `sp.logit`, ...) the envelope reports the coefficient
named in `focal_term` (the single endogenous regressor of an IV fit, a recorded
treatment, or the only regressor); with several regressors and none
designated it reports `null` and a short `coef_head`, never a guess. Some
fields say what scale they are on (`model_info["scale"]`, e.g. `sp.rddensity`)
or which sample they describe (`sp.match`: `balance_before` /
`balance_after`); read those before quoting a number.

`sp.audit(r)` lists the robustness checks still missing.

## When a call fails

Errors are `StatsPAIError` subclasses with `code`, `recovery_hint`,
`diagnostics`, `alternative_functions` and `to_dict()`. A misspelled column
raises `ColumnNotFound` with `diagnostics["did_you_mean"]`; a misspelled
keyword raises `TypeError` naming the parameter you probably meant
(`seed` -> `random_state`, `clustr` -> `cluster`). Read the hint and retry;
arguments are never silently dropped.

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
6. Numbers from a simulated dataset (`df.attrs["simulated"]`) are not findings.
7. A test of heterogeneity graded on the same forest that produced the ranking
   is not a test: use `sp.rate_split`, not `sp.rate(cf)` on its own ranking.

Full guide: `docs/guides/agent_api.md`; MCP workflow:
`docs/guides/economist_mcp_workflow.md`; machine-readable index: `llms.txt`.
