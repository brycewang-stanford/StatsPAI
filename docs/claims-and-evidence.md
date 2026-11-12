---
description: Every public claim StatsPAI makes about itself, the evidence behind it, and the command that re-checks it.
---

# Claims and evidence

This page lists what StatsPAI says about itself on its website, README, PyPI
page, documentation and MCP listing, and how each statement can be checked.
A statement that cannot be checked here should not appear elsewhere. If you
find one, please [open an issue](https://github.com/brycewang-stanford/StatsPAI/issues).

Last reviewed: **2026-11-12**, StatsPAI 1.43.1. Counts are regenerated at each
release; the commands give the live values.

## What StatsPAI is

> **StatsPAI is an open-source, agent-native Python toolkit for econometrics
> and causal inference, validation-tiered against Stata and R.**

| Claim | Evidence | Re-check |
| --- | --- | --- |
| Open source, MIT licence | [`LICENSE`](https://github.com/brycewang-stanford/StatsPAI/blob/main/LICENSE); no GPL / AGPL runtime dependencies (project rule) | `pip show statspai` → `License-Expression: MIT` |
| Python 3.10–3.13 | `requires-python = ">=3.10"`; CI matrix | `python -c "import statspai"` on each version |
| One import covers the API | every public function is `sp.<name>` | `sp.list_functions(kind="function")` |
| 1,680 registered functions, 92 submodules (1.43.1) | the function registry | `python scripts/registry_stats.py` |
| Agent-native: every public function has a machine-readable schema | `schemas/functions.json`, `sp.function_schema(name)` | `sp.function_schema("feols")` |
| Agent-native: an MCP server ships with the package | `statspai-mcp` console script, `server.json` | `statspai-mcp --version`; `statspai mcp doctor` |
| Structured results | every result class has `to_dict`, `violations`, `next_steps`, `result_card`, `cite` | `scripts/result_protocol_audit.py --check` |
| Peer-reviewed software paper | Wang and Rozelle (2026), *Journal of Open Source Software* 11(125), 10604, [doi:10.21105/joss.10604](https://doi.org/10.21105/joss.10604) | `sp.citation()` |

## Numerical evidence

"Validation-tiered" means each function carries a grade saying what its
numbers have been checked against. Only the first row is a claim of agreement
with Stata or R.

| Grade | Meaning | Functions (1.43.1) |
| --- | --- | ---: |
| `bit-exact` (T2) | matches a named R / Stata reference on identical data, estimate and SE relative error ≤ 1e-6 | 501 |
| `aligned` | matches within a documented, looser tolerance | 75 |
| `analytical-only` | recovers a known true value on a deterministic design | 395 |
| `external-replication` | reproduces numbers from a published paper | 57 |
| `unverified` | registered API with no numerical evidence attached yet | 652 |

Re-check with `sp.parity_summary()`, or for one function
`sp.parity_status("feols")` (reference, tolerance, test file). The full table
is the [parity matrix](parity.md); interval coverage and speed, including
failures, are in the [evidence scorecard](scorecard.md).

**Wording rule.** Say "matches Stata / R" only for a function graded
`bit-exact` or `aligned`, and name the reference. Do not describe the whole
package as "matching Stata and R".

## Agent integration

| Claim | Evidence | Re-check |
| --- | --- | --- |
| Works with Claude Code | tested 2026-11-12: tools listed and `ivreg` called from a Claude Code session | [AI agents and MCP](tasks/ai-agents-mcp.md) |
| Works with any stdio MCP client | tested 2026-11-12 against StatsPAI 1.43.0 from PyPI in a fresh environment: `initialize`, `tools/list`, `tools/call` | same page |
| Claude Desktop, Cursor, Windsurf, VS Code, Codex | `statspai mcp config` generates their entries; not re-tested end to end on that date | `statspai mcp config --client <name>` |

## Performance

| Claim | Evidence | Re-check |
| --- | --- | --- |
| HDFE timings relative to `fixest` and `pyfixest` | [performance guide](guides/performance.md), with hardware, versions and load average recorded | `python benchmarks/run_all.py` |

No other speed claim is made. "Fastest" is not claimed.

## Claims we do not make

- "The first" or "the only" agent-native econometrics library: not
  established, so not claimed.
- "Best" Python econometrics library: the
  [library comparison](tasks/python-econometrics-libraries.md) explains where
  other packages are the better choice.
- "Replaces Stata / R" without qualification: StatsPAI offers a replacement
  path for many workflows, with the evidence grades above.
- User counts, ratings or endorsements.
