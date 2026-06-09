# Agent-Infra Campaign

> Work line: branch `agent-infra`, **physically isolated** from the JOSS paper review
> ([joss-reviews#10604](https://github.com/openjournals/joss-reviews/issues/10604)).
> Goal: make the agent-native surface (registry / schema / result objects / MCP
> result cache / DSL plumbing) *solid* — not the estimators, not the paper.

## JOSS isolation contract

A change on this branch is **safe for the paper review** iff it does **not** touch
any of these four classes of artifact:

1. `paper.md` / `paper.bib`
2. The **numerical output** of any estimator (`.estimate` / `.se` / `.ci` /
   coefficient tables of `did iv rd synth dml panel …`).
3. The reference-/external-parity expected numbers
   (`tests/reference_parity/`, `tests/external_parity/`).
4. The **existing signature or default behavior** of any public function
   (new params/methods may be *added* only with defaults equal to current behavior).

Every commit on this branch is audited against this contract before merge.
Metadata-only changes (registry specs, schema plumbing, additive result methods,
MCP runtime cache, tests) are all in-bounds: they describe or wrap the estimators,
they do not change a single estimated number.

## Items (ranked by value)

| # | Item | Risk to paper | Status |
|---|------|---------------|--------|
| 1 | Full schema↔signature drift CI guard (+ fix existing drift) | none (metadata + tests) | **done** |
| 2 | `EconometricResults.to_dict(detail=)` + `.cite()` (additive) | ~none (additive, default=current) | pending |
| 3 | Workflow tools → registry single-source (sync `registry_stats`) | none (metadata; counts re-synced) | pending |
| 4 | MCP result-cache TTL + structured invalidation | none (runtime only) | pending |
| 5 | Docstring parsing multi-format (lazy/extras only) | none (metadata extraction) | pending |

## Item #1 — schema↔signature drift

**Problem (measured 2026-06-09).** Of 224 hand-written `FunctionSpec`s, a CI-grade
audit found two *agent-breaking* invariant violations:

- **Invariant A** — spec advertises a parameter the function cannot accept
  (param not in signature, and no `**kwargs` to absorb it): **21 functions**.
  An agent that reads the schema and calls with that kwarg gets a `TypeError`.
- **Invariant B** — a *required* signature parameter is missing from the spec
  (agent omits it → `TypeError`): **15 functions** (overlapping set).

Root cause: hand-written specs duplicated the param list, then signatures evolved
(e.g. `metalearner` spec said `treatment`/`method`; the signature — and the spec's
own `example` — use `treat`/`learner`).

The 4 genuine `**kwargs` dispatchers (`synth`/`iv`/`dml`/`did`) are *correctly
excluded* by Invariant A: their specs document logical params routed through kwargs.

**Fix.** (a) Add `tests/test_registry_signature_contract.py` enforcing Invariants
A & B over *all* hand-written specs, with a precise per-function failure message.
(b) Correct the ~24 drifted specs to match real signatures, preserving curated
descriptions/enums under the right param name.

**Why not a runtime reconciliation engine?** It would rewrite all 224 specs at
import time — large blast radius, opaque to a reviewer reading `registry.py`, and
it would silently drop curated enums attached to renamed params. Hand-fix + CI
guard gives the same permanence with a 24-spec diff instead of a 224-spec one.
