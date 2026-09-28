# Agent-native API surface (v1.9.0)

StatsPAI v1.9.0 ships a 12-piece API surface designed for the case
where the **caller is a language-model agent** — Claude Code, Cursor,
Copilot CLI, or a custom workflow that uses StatsPAI through the
[Model Context Protocol](https://modelcontextprotocol.io). The
underlying estimators are unchanged; what's new is everything around
them: shape detection, pre-flight checks, structured exceptions,
token-budgeted serialization, missing-evidence audits, multi-format
citations, deterministic RNG sessions, MCP prompts, and a one-line
dashboard view.

This guide is a quickstart for agent authors. Human researchers can
use these too — they just turn out to be the right primitives for
agents to chain.

---

## Why agent-native?

When an LLM is the caller, three things differ from human use:

1. **The agent can't see the DataFrame.** It needs APIs that report
   structure (panel? RD running variable? cross-section?) without a
   visualisation step.
2. **Token budget matters per call.** A 4 000-character "tidy
   summary" may be useful to a notebook but burns context the agent
   needs for reasoning. We expose every result at three sizes —
   `minimal` / `standard` / `agent` — and an even smaller one-line
   `brief()`.
3. **Errors should be machine-readable.** A free-text "weak
   instrument F=2.1, try LIML" is great for a human but the agent
   has to regex-parse it. v1.9.0's exception envelope ships
   `error_kind` / `recovery_hint` / `diagnostics` /
   `alternative_functions` as discrete fields.

Everything below is additive — no estimator numerical path changed,
default behaviour is byte-identical to v1.8.0.

---

## The 12-piece surface at a glance

```python
import statspai as sp

# Discovery — "what is this data?"
sp.detect_design(df)                       # cross-section / panel / RD
sp.preflight(df, "did", y=..., treat=...)  # cheap pre-estimation check
sp.examples("did")                         # runnable code snippets

# Estimation — unchanged, plus richer envelope
result = sp.did(df, y='y', treat='t', time='post')

# Serialization — pick payload size per call
result.to_dict(detail="minimal")           # ~150 tokens — answer only
result.to_dict(detail="standard")          # ~250 tokens — coefs + diagnostics
result.to_dict(detail="agent")             # ~620 tokens — + violations + next_steps
result.brief()                             # ~95 chars — dashboard view

# Reviewer-grade follow-up
sp.audit(result)                           # what robustness checks are missing?
result.cite(format="apa")                  # APA / BibTeX / JSON citations
sp.bib_for(result)                         # structured citation dict

# Reproducibility
with sp.session(seed=42):
    result_a = sp.did(df, ...)             # deterministic across runs
    result_b = sp.bayes_did(df, ...)
```

---

## End-to-end agent workflow

Concrete example: an agent receives an unfamiliar CSV and is asked
"is there a treatment effect?". Five calls, each with a clear
purpose:

```python
import statspai as sp
import pandas as pd

df = pd.read_csv("/path/to/dataset.csv")

# 1. Identify the study design.
design = sp.detect_design(df)
# {'design': 'panel', 'confidence': 1.0,
#  'identified': {'unit': 'firm_id', 'time': 'year'}, ...}

# 2. Pre-flight a candidate estimator before paying for it.
report = sp.preflight(df, 'did',
                      y='sales', treat='treated', time='year')
if report['verdict'] == 'FAIL':
    # The verdict carries structured failure info — agent can
    # decide whether to fix args, switch method, or stop.
    for c in report['checks']:
        if c['status'] == 'failed':
            print(f"  blocked by {c['name']}: {c['message']}")
    raise SystemExit
elif report['verdict'] == 'WARN':
    print("warnings present but proceeding")

# 3. Run the estimator. If it raises a structured StatsPAIError,
#    the MCP layer surfaces error_kind + alternative_functions.
result = sp.did(df, y='sales', treat='treated', time='year')

# 4. One-line dashboard summary for logs / multi-result loops.
print(result.brief())
# [Difference-in-Differences (2x2)]  estimand=ATT  est=0.412
# (se=0.087)  95% CI [0.241, 0.583]  ***  N=2,000

# 5. Reviewer checklist — which robustness checks are still
#    MISSING from the result's evidence base?
audit_card = sp.audit(result)
for c in audit_card['checks']:
    if c['status'] == 'missing' and c['importance'] == 'high':
        print(f"  follow-up: {c['suggest_function']}  ({c['name']})")
# follow-up: sp.pretrends_test  (parallel_trends)
# follow-up: sp.honest_did       (rambachan_roth)
```

The agent now has enough structured information to plan its next
call — no prose parsing, no "did you remember to test parallel
trends?" loops.

---

## Token-budget control

Every fitted result exposes the same payload at three sizes. Agents
choose per call:

| level       | shape                                             | typical size |
| ----------- | ------------------------------------------------- | -----------: |
| `brief()`   | one-line string (`[METHOD] estimand= est=… ci ⚠`) |     ~95 char |
| `"minimal"` | dict: method / estimand / estimate / SE / CI / N  |  ~150 tokens |
| `"standard"`| `"minimal"` + scalar diagnostics + detail rows    |  ~250 tokens |
| `"agent"`   | `"standard"` + violations + next_steps            |  ~620 tokens |

```python
result.to_dict()                       # = "standard" (legacy default)
result.to_dict(detail="minimal")       # cheap sub-step
result.to_dict(detail="agent")         # full agent envelope
```

Through MCP, the same control is exposed as a `detail` argument on
every `tools/call`:

```json
{
  "method": "tools/call",
  "params": {
    "name": "did",
    "arguments": {
      "data_path": "/abs/path.csv",
      "y": "sales", "treat": "treated", "time": "year",
      "detail": "minimal"
    }
  }
}
```

---

## `sp.audit(result)` — the missing-evidence view

`sp.audit()` is intentionally distinct from three neighbours:

| function                                | answers                                                                |
| --------------------------------------- | ---------------------------------------------------------------------- |
| `result.violations()`                   | "what evidence is **on the result and failing**?"                       |
| `result.next_steps()`                   | "what should the user **do next** to publish this result?"             |
| `sp.assumption_audit(result, data)`     | "given the data, do the assumptions actually hold?" (re-runs tests)     |
| `sp.audit(result)`                      | "what reviewer-grade evidence is **still missing** from this result?"   |

`audit` is read-only and runs in microseconds: it inspects
`result.model_info` for the diagnostics each method family
expects, and reports each as `passed` / `failed` / `missing`. Each
`missing` check carries a `suggest_function` so the agent knows
exactly what to call next.

```python
{
    "method": "did_2x2",
    "method_family": "did",
    "checks": [
        {
            "name": "parallel_trends",
            "question": "Are pre-treatment trends statistically parallel?",
            "status": "missing",
            "severity": "warning",
            "importance": "high",
            "suggest_function": "sp.pretrends_test",
            "rationale": "DID identification rests on parallel trends; "
                         "without a pre-trend test the design is "
                         "unfalsifiable.",
            ...
        },
        ...
    ],
    "summary": {"passed": 0, "failed": 0, "missing": 5, "n_total": 5},
    "coverage": 0.0,
}
```

`coverage` is `passed / n_total` — agents can sort multiple
results by reviewer-readiness.

---

## Citations: zero-hallucination, three formats

`result.cite(format=...)` and `sp.bib_for(result)` parse the
canonical BibTeX entry stored on the result class and reformat it.
Bibliographic facts come **only** from the parsed BibTeX — the
formatter never invents authors, years, journals, or publishers
(per [CLAUDE.md §10](https://github.com/brycewang-stanford/StatsPAI/blob/main/CLAUDE.md)).

```python
r = sp.callaway_santanna(df, ...)

r.cite()                              # default — BibTeX
# @article{callaway2021difference, ...}

r.cite(format="apa")
# Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-
# differences with multiple time periods. Journal of Econometrics,
# 225(2), 200–230.

sp.bib_for(r)                         # structured dict
# {'type': 'article', 'key': 'callaway2021difference',
#  'authors': [{'last': 'Callaway', 'first': 'Brantly'}, ...],
#  'year': '2021', 'title': '...', 'journal': '...', ...}
```

Methods that cite **multiple** papers (e.g.
`twfe_decomposition` cites both Goodman-Bacon 2021 and
de Chaisemartin & D'Haultfœuille 2020) round-trip every author —
the parser walks every `@type{...}` head in the source string.

---

## `sp.session(seed=42)` — reproducible blocks

A standard frustration: an agent reruns `sp.bootstrap(...)`
twice and gets different intervals because Python `random` and
NumPy's legacy global drifted between calls. `sp.session` snapshots
both, applies the seed for the duration of the block, and restores
prior state on exit (even when an exception is raised inside):

```python
with sp.session(seed=42):
    boot = sp.bootstrap(df, statistic=lambda d: d['y'].mean(), n_boot=1000)
    perm = sp.ri_test(df, y='y', treat='d', n_perms=1000)
# state outside the block is byte-identical to before the with
```

What's covered: Python `random`, NumPy legacy global (`np.random.randn`,
`np.random.choice`, …). Lazy interop with PyTorch / JAX (only seeded
if those libraries are already imported — never auto-installed).

What's **not** covered: `np.random.default_rng()` instances. Those
have no process-global state; pass `state.seed` explicitly if you
need them deterministic:

```python
with sp.session(seed=42) as state:
    rng = np.random.default_rng(state.seed)   # explicit seed
    x = rng.normal(size=5)
```

Not thread-safe — for parallel workloads, use one
`np.random.default_rng(seed)` per thread.

---

## Shell: `statspai run`

A shell agent gets the same payload an MCP client gets, without a
server:

```bash
statspai run callaway_santanna --data panel.csv \
    --arg y=lemp --arg g=first_treat --arg t=year --arg i=countyreal
statspai did --data panel.csv --y lemp --treat treated --time year --id id \
    --covariates '["pop"]' --format summary
statspai route rd --answer assignment=sharp --answer running=continuous
statspai mcp --profile curated
```

`run` takes any registered function and `--arg key=value` pairs (values
parse as JSON when they can — numbers, booleans, lists, quoted strings);
the family shortcuts (`did`, `callaway_santanna`, `event_study`,
`regress`, `feols`, `ivreg`, `rdrobust`, `synth`, `dml`, `match`, `ipw`,
`aipw`) build their `--flags` from the registry schema, so they cannot
drift from the signature. `--data` accepts every format the MCP loader
reads (`--columns` / `--sample` for large files). Output is the agent
payload (`--detail minimal|standard|agent`, `--out file.json`), or
`--format summary` for the result's text summary. Arguments the function
cannot bind are reported under `_unsupported_args`; estimator errors are
the structured `StatsPAIError` payload on stderr with exit code 3, usage
errors exit 2.

## MCP server: drop-in for Claude Desktop / Cursor

`pip install statspai` exposes a `statspai-mcp` console script.
Wire it into your MCP-capable client by adding to the client's
config:

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "statspai": {
      "command": "statspai-mcp"
    }
  }
}
```

**Cursor** / generic stdio MCP client:

```bash
statspai-mcp     # speaks JSON-RPC 2.0 over stdio
```

What the server exposes:

- **`tools/list`** — typed tools with JSON-Schema inputs. The shape
  depends on the profile: `statspai-mcp --profile curated` (the CLI
  default) lists the ~35 hand-curated estimator / workflow / pipeline
  tools plus three discovery meta-tools — `search_functions`,
  `describe_function`, `call_function` — through which every one of the
  ~1,250 registered functions stays reachable; `--profile core` trims
  that to ~20; `--profile full` advertises every auto-generated tool
  (~580 entries, ~2 MB — larger than most client context windows, so
  only for clients that page or defer tool loading). `tools/call`
  accepts any tool name under every profile.
- **`tools/call`** — runs the estimator. Data arrives as `data_path`
  (CSV, Parquet, Stata, … read server-side), `data_id` (a handle from
  `load_data` / `transform_data` — no re-upload, lineage recorded),
  `data_records` or `data_csv` (a small inline table); plus the
  estimator's own kwargs and `detail` to control payload size. Payloads
  carry `data_provenance`, `runtime_warnings` and `_unsupported_args`.
- **Data handles** — `load_data` → `data_id`; `transform_data(data_id,
  operations=[...])` derives a new handle (query / select / rename /
  dropna / assign / winsor / reshape / mice / any DataFrame-returning
  `sp.<fn>`) and records the chain; `describe_data` profiles one.
- **Routing without data** — `route_estimator(family, answers)` turns
  answers to a family's decision questions (did / iv / rd / matching /
  ml_causal / qte / dynamic_panel) into registered calls with example,
  reason, added assumptions and the guide section; the same tables back
  `sp.route` / `sp.decision_guide` in Python.
- **`resources/list`** — `statspai://catalog` (Markdown index) and
  `statspai://functions` (JSON `[{name, description}]`).
- **`resources/templates/list`** — `statspai://function/{name}` →
  per-function rich agent card (description, signature,
  assumptions, failure_modes, alternatives, `typical_n_min`,
  example); `statspai://guide/{family}` → the full "choosing an
  estimator" guide; `statspai://data/{id}` → a dataset handle with its
  lineage; `statspai://result/{id}` → a fitted result.
- **`prompts/list` / `prompts/get`** — curated workflow templates
  (`audit_did_result`, `design_then_estimate`, `robustness_followup`,
  `stata_command_workflow`, `r_command_workflow`, …) MCP clients
  surface as direct action buttons.

When an estimator raises a structured `StatsPAIError`, the
`tools/call` response carries the full payload alongside legacy
fields:

```json
{
  "error": "MethodIncompatibility: treatment has 3 unique values...",
  "error_kind": "method_incompatibility",
  "error_payload": {
    "code": "method_incompatibility",
    "message": "...",
    "recovery_hint": "Use sp.callaway_santanna or sp.multi_treatment.",
    "diagnostics": {"n_unique_values": 3, "expected": 2},
    "alternative_functions": ["sp.callaway_santanna",
                               "sp.multi_treatment"]
  },
  "tool": "did", "arguments": {...}, "remediation": {...}
}
```

Agents branch on `error_kind` (typed) instead of regex-parsing
`error` (free text).

---

## Deciding which API to call when

Quick decision tree for agents:

```
unfamiliar data?      → sp.detect_design(df)
known data, want method advice?  → sp.recommend(df, outcome=…, treatment=…)
chosen method, before fitting?   → sp.preflight(df, method, **args)
fitting succeeded, want a quick view?     → result.brief()
fitting succeeded, want structured agent payload?  → result.to_dict(detail="agent")
fitting succeeded, want to find missing evidence?  → sp.audit(result)
need to cite the method?                   → result.cite(format="apa")
running multiple estimators, want determinism?  → with sp.session(seed=42): …
need a code snippet?                       → sp.examples(name)
```

---

## See also

- [`CHANGELOG.md`](https://github.com/brycewang-stanford/StatsPAI/blob/main/CHANGELOG.md#190--agent-native-api-surface-12-modules-across-4-phases)
  — full v1.9.0 release notes.
- [`MIGRATION.md` v1.8.0 → v1.9.0](https://github.com/brycewang-stanford/StatsPAI/blob/main/MIGRATION.md#v180--v190--agent-native-api-surface-no-breaking-changes)
  — backward-compatibility invariants pinned by the test suite.
- [`agent/mcp_server.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/src/statspai/agent/mcp_server.py)
  — the JSON-RPC 2.0 stdio MCP server source.
- [`smart/audit.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/src/statspai/smart/audit.py),
  [`smart/preflight.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/src/statspai/smart/preflight.py),
  [`smart/detect_design.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/src/statspai/smart/detect_design.py),
  [`smart/citations.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/src/statspai/smart/citations.py),
  [`smart/examples.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/src/statspai/smart/examples.py),
  [`smart/session.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/src/statspai/smart/session.py),
  [`smart/brief.py`](https://github.com/brycewang-stanford/StatsPAI/blob/main/src/statspai/smart/brief.py)
  — the seven new `sp.smart` modules.
