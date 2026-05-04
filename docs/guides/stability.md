# Stability tiers — parity-grade vs. frontier-grade

> Why a single "function exists" bit is not enough.

StatsPAI ships **975+ public functions** across 50+ subpackages. Some are
numerically aligned with R/Stata or a published reference implementation
and have a locked signature. Others are recent or first-class
implementations of frontier methods — they work, but they are not yet
parity-tested or the API may shift.

Until v1.13 these were treated identically by `sp.list_functions()`,
`sp.help()`, and `sp.describe_function()`. That made it hard for a
human (or an LLM agent) to answer the most basic catalogue question:

> *Which of these can I trust for a publication-grade analysis, and
> which are research-grade conveniences I should report with caveats?*

This page documents the contract that makes that question answerable.

---

## The two-axis model

A function's maturity has two distinct dimensions, surfaced as two
separate fields on every `FunctionSpec`:

| Field | Scope | Question it answers |
| --- | --- | --- |
| `stability` | **Whole function** | "Is the *function as a whole* parity-grade or frontier-grade?" |
| `limitations` | **Inside an otherwise-stable function** | "Which `param=value` combinations are documented as not-yet-implemented?" |

Conflating these is what made the older catalogue hard to navigate. A
function can be perfectly parity-grade for its primary code path and
still raise `NotImplementedError` on a less-common variant — that is a
**limitation on a stable function**, not a reason to demote the whole
function to experimental.

---

## Stability tiers

The `stability` field takes one of three values
(`statspai.STABILITY_TIERS`):

### `stable` (default)

- Numerically aligned with R / Stata / a published reference, or with
  an analytic ground-truth result.
- Public signature is locked under SemVer minor releases.
- Safe to use for publication-grade analysis without a methodological
  caveat (beyond the function's documented assumptions, which are
  already exposed via `sp.describe_function(name)['assumptions']`).

This is the **default** — most of the catalogue (~970 functions) lives
here. You don't need to set it explicitly when registering a new
function.

### `experimental`

- Method is implemented and tested for correctness on synthetic /
  illustrative data, but it is **not (yet) parity-tested** against a
  published reference; **or** the function is an MVP whose signature
  may shift between minor versions.
- Safe to *try*, but report results as "based on StatsPAI
  experimental implementation of <method>" if you publish them, and
  pin the StatsPAI version.

Current entries (v1.13):

- `text_treatment_effect` — Veitch-Wang-Blei (2020) text-as-treatment,
  hash-embedder fallback.
- `llm_annotator_correct` — Egami et al. (2024) LLM-annotator
  measurement-error correction.
- `did_multiplegt_dyn` — dCDH (2024) intertemporal event-study DiD MVP
  (switch-on only, bootstrap SE, no analytical influence-function
  variance).

### `deprecated`

- Scheduled for removal. The function still works for the duration of
  the deprecation window and emits a `DeprecationWarning` at call
  time. The replacement is documented in
  [`MIGRATION.md`](https://github.com/brycewang-stanford/StatsPAI/blob/main/MIGRATION.md).
- No entries are deprecated as of v1.13 — the field exists so the
  registry can express deprecation explicitly when we need to.

---

## `limitations` — variant-level gaps inside stable functions

`limitations` is a list of one-line strings describing **specific
parameter values or feature combinations that are not yet
implemented** inside an otherwise stable function. Each entry should
be agent-readable and follow the pattern:

```
"<param>=<value>: <what's missing>"
```

Examples in v1.13:

```python
>>> import statspai as sp
>>> sp.describe_function('hal_tmle')['limitations']
["variant='projection' raises NotImplementedError — the Riesz-projection
  targeting step from Li-Qiu-Wang-vdL (2025) §3.2 is not yet ported …"]

>>> sp.describe_function('principal_strat')['limitations']
["instrument= (explicit two-layer IV + treatment setup) is not yet
  implemented; for encouragement-design LATE use sp.iv or sp.dml(model='iivm')"]

>>> sp.describe_function('rdrobust')['limitations']
["observation-level weights are not yet supported — passing a weight
  column raises NotImplementedError"]
```

Why surface these as data and not just as runtime exceptions?

1. **Agents read schemas before calling.** An agent that only learns
   `variant='projection'` is unsupported by triggering
   `NotImplementedError` mid-pipeline burns a tool-call round-trip and
   may pollute downstream reasoning. The same fact in the schema lets
   the agent route around the gap on the *first* attempt.
2. **Humans read help.** `sp.help('hal_tmle')` now shows a "Known
   limitations" section between the description and the parameter
   table — you see the gap before you read the signature.
3. **`function_schema()` carries the limitation into the LLM
   tool-call description automatically**, so any framework that feeds
   `sp.all_schemas()` to an LLM (OpenAI / Anthropic / LangChain /
   LangGraph) inherits the gap visibility for free.

---

## How to filter

### Python

```python
import statspai as sp

sp.list_functions()                            # all (default)
sp.list_functions(stability='stable')          # parity-grade only
sp.list_functions(stability='experimental')    # frontier-grade only
sp.list_functions(category='causal',
                  stability='stable')          # parity-grade causal subset
sp.agent_cards(stability='stable')             # bulk agent-card export, parity-grade only

# Per-function inspection
spec = sp.describe_function('hal_tmle')
spec['stability']      # 'stable'
spec['limitations']    # ['variant=projection ...']

# Search results now carry stability so you can post-filter
hits = sp.search_functions('treatment')
[h for h in hits if h['stability'] == 'stable']
```

### CLI

```bash
$ statspai list --stability experimental
did_multiplegt_dyn
llm_annotator_correct
text_treatment_effect

$ statspai list --category causal --stability stable | head
$ statspai describe hal_tmle    # shows the Known limitations section
```

### sp.help() overview

`sp.help()` (no arguments) now prints a `STABILITY` block with the
count per tier, immediately under the `CATEGORIES` block. Per-function
detail (`sp.help('hal_tmle')`) prints a `Stability:` line as the first
substantive content and a `Known limitations` section if applicable.

---

## What to do if you maintain a StatsPAI extension

If you register your own functions via `statspai.registry.register`,
set `stability` and `limitations` deliberately:

```python
from statspai.registry import register, FunctionSpec, ParamSpec

register(FunctionSpec(
    name="my_estimator",
    category="causal",
    description="Doe (2026) double-debiased estimator for setting X.",
    params=[...],
    # Pick the right tier:
    stability="experimental",   # not yet parity-tested
    # And surface partial-implementation gaps explicitly:
    limitations=[
        "method='kernel' is implemented; method='spline' raises "
        "NotImplementedError pending the v2.0 release.",
    ],
))
```

A typo in `stability` raises `ValueError` at `register()` time, so
you find out at import time — not at the moment an agent tries to
filter on it.

---

## Promotion path

- An `experimental` function is promoted to `stable` once it has at
  least one **parity test** in `tests/reference_parity/` matching a
  published number, plus its public signature has been stable across
  one minor version.
- A `limitation` is removed once the unimplemented variant lands and
  has its own test (parity or analytic).
- `deprecated` enters via a `DeprecationWarning` + a `MIGRATION.md`
  entry, with the removal version explicit in the warning text.

---

*Last updated: v1.13 (2026-05-03).*
