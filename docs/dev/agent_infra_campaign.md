# Agent-Infra Campaign

> Work line to make the agent-native surface (registry / schema / result objects /
> MCP result cache / docstring plumbing) *solid* — **not** the estimators, not the
> paper. Physically isolated from the JOSS paper review
> ([joss-reviews#10604](https://github.com/openjournals/joss-reviews/issues/10604)).

## JOSS isolation contract

A change in this work line is **safe for the paper review** iff it does **not**
touch any of these four artifact classes:

1. `paper.md` / `paper.bib`
2. The **numerical output** of any estimator (`.estimate` / `.se` / `.ci` /
   coefficient tables of `did iv rd synth dml panel …`).
3. The reference-/external-parity expected numbers
   (`tests/reference_parity/`, `tests/external_parity/`).
4. The **existing signature or default behavior** of any public function
   (new params/methods may be *added* only with defaults equal to current behavior).

Every agent-infra commit is metadata / result-object additions / MCP runtime /
generated schemas / tests — it describes or wraps the estimators, it does not
change a single estimated number.

## Items (ranked by value)

| # | Item | Risk to paper | Status |
|---|------|---------------|--------|
| 1 | schema↔signature drift CI guard (+ fix 18 drifted specs) | none | **done** |
| 2 | `EconometricResults.cite()` (`to_dict(detail=)` already shipped) | ~none (additive) | **done** |
| 3 | workflow-tool dispatch contract guard (MCP layer) | none (test-only) | **done** |
| 4 | MCP result-cache TTL + reason-aware misses | none (runtime; default off) | **done** |
| 5 | docstring parser: type-less NumPy headers (no new dep) | none | **done** |
| 6 | `inherits_from` graph integrity guard | none (test-only) | **done** |
| 7 | auto-spec type-resolution integrity guard | none (test-only) | **done** |
| 8 | spec default/enum consistency guard (+ fix 11 specs) | none | **done** |

A recurring pattern: three of the original eight framings (#2 `detail=`, #5
`docstring-parser` dep, #7 stringifier) were **overstated on measurement** — the
capability already existed or the "fix" would have added risk for no gain. Each
was redirected to the real, measured gap. Decisions are evidence-first.

## Provenance / git reality

The first slice of this work — **Item #1 + an unrelated in-flight BLP structural
fix** — was already squash-merged to `origin/main` as **PR #20 "Agent infra"
(`43198c9`)** *before* this follow-up. Consequences:

- **BLP is already live on `origin/main`.** `fix(structural): blp passes
  maxiter_inner to _gmm_objective` (+ its analytic test) touches an **estimator
  numeric path** and rode in on PR #20. It is **not** agent-infra work and must be
  evaluated for JOSS impact **separately** — flagged here so it is not mistaken
  for metadata. (Revert is available if it should come out; left to a conscious
  call, since it is the author's own fix.)
- **PR #20 changed `registry.py` param names but did not regenerate the
  `schemas/` bundle**, leaving `scripts/dump_schemas.py --check` red on main. This
  follow-up regenerates it.

Items #2–#8 are cherry-picked clean onto `origin/main` (no BLP), schemas
regenerated, and form the merge candidate.

---

## Item #1 — schema↔signature drift

Of 224 hand-written `FunctionSpec`s, two *agent-breaking* invariants were violated:
**A** spec advertises a param the function cannot accept (21 functions), **B** a
required signature param is missing from the spec (15, overlapping). An agent
following `describe_function` would `TypeError`. Root cause: specs duplicated the
param list, signatures drifted (e.g. `metalearner` spec said `treatment`/`method`;
the signature — and its own `example` — use `treat`/`learner`). Fixed 18 specs;
`tests/test_registry_signature_contract.py` locks A & B over all hand-written
specs (the 4 `**kwargs`/`*args` dispatchers are correctly skipped).

## Item #2 — `EconometricResults.cite()`

`to_dict(detail=minimal/standard/agent)` already existed. The gap was `.cite()`:
only `CausalResult` had it, so `sp.bib_for(regression_result)` raised. Added
`EconometricResults.cite(format=)`, resolving the bib key **exactly** from
`model_info['citation_key'] → model_type → method` against the shared
`CausalResult._CITATIONS`. Zero-hallucination (§10): OLS/logit/probit/poisson
return a placeholder, never a fuzzy/fabricated match. Pure addition.

## Item #3 — workflow-tool dispatch contract

These are MCP-only tools (no `sp.<name>` callable), so registering them in the
function registry would pollute `sp.list_functions()` — the single source of truth
correctly stays `WORKFLOW_TOOL_SPECS`. Real gap: `execute_workflow_tool` falls
through to a *silent error dict* when a spec has no branch, and the existing test
only checked name membership. `tests/test_workflow_tool_dispatch_contract.py`
exercises the dispatcher for every spec, rejecting the fall-through sentinel.

## Item #4 — result-cache TTL + reason-aware misses

Opt-in TTL via `STATSPAI_MCP_RESULT_CACHE_TTL` (default `None` = no expiry, prior
behaviour byte-identical); expired entries swept lazily on access / eagerly on
insert. A bounded ledger records why a handle left (`ttl`/`lru`/`explicit`) so
`_need_result` gives a tailored re-fit hint. New `evict()/purge_expired()/stats()`.

## Item #5 — docstring parser: type-less NumPy headers

Scan of all 1020 functions: zero Sphinx `:param:` docstrings → the planned
`docstring-parser` dep was cancelled. Real gap: the type-less NumPy convention
(`name` then indented description, no `: type`) was unmatched, silently dropping
**65 param descriptions across 41 functions** (`feols`, `causal_forest`, `match`,
…). Added a column-0 `barename_re` branch; false-positive boundary pinned by
`tests/test_docstring_param_parser.py`.

## Item #6 — `inherits_from` graph integrity

Locks the 42-edge inheritance graph against dangling parents, self-references,
cycles, and runaway depth — any of which breaks the agent-card metadata merge.
Currently healthy; forward guard (`tests/test_inherits_from_integrity.py`).

## Item #7 — auto-spec type-resolution integrity

Measured: the 7.2% `Any` rate is *honest* — 321/397 params have no source
annotation, 76 are explicitly `: Any`; **zero** concrete annotations are dropped.
Name-guessing a narrower type would mislead agents, so no speculative enrichment.
`tests/test_auto_spec_type_resolution.py` locks the real invariant (a concrete
annotation never collapses to `Any`) + a loose 12% regression backstop.

## Item #8 — spec default/enum consistency

Extends Item #1 from names to *values*. Two invariants over hand-written specs:
a pinned default must not contradict the signature, and must be a member of its
own enum. Fixed **11** misleading specs where `describe_function` reported a
default the estimator does not use — e.g. `drdid` method `'dr'`→`'imp'` with enum
`['dr','or','ipw','reg','stdipw']`→`['imp','trad']`; `did_bcf` n_trees `200`→`50`;
`harvest_did` reference `'pre'`(str)→`-1`(int); `iv_compare`/`iv_diag` defaults
stored as string-reprs-of-tuples → real lists. Also surfaced a latent duplicate
`harvest_did` registration (the wrong spec was effective; corrected — dedup left
as a separate hygiene item). Guard: `tests/test_registry_default_contract.py`.

## Schema bundle

The committed `schemas/` + `src/statspai/schemas/` JSON bundle (checked by
`scripts/dump_schemas.py --check`) is regenerated whenever the registry changes.
Generated artifact; no hand-edits.
