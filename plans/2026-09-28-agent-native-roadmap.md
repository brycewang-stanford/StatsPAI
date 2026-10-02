# Agent-native roadmap — 2026-09-28 → 2026-10-28

> Status tracker for the six long-tail items left open by the 2026-09-28
> agent-native audit (CHANGELOG "Unreleased", commit `c18b118`). One
> workstream at a time; each lands as its own commit series on `main` with
> tests, CHANGELOG entries and, where an MCP surface changes, a regenerated
> schema bundle. Checkboxes are the live state — update them in the same
> commit as the work.

**Goal:** an LLM agent driving StatsPAI over MCP, the CLI, or plain Python can
(1) find the right estimator from a task description, (2) learn what it
assumes and returns without reading source, (3) move data through a
multi-step analysis without re-uploading files, and (4) never be misled by a
payload that hides a failure. The 2026-09-28 commits fixed the acute cases;
this plan closes the structural ones.

**Ordering rationale.** Correctness and plumbing first (a deadlock and the
data-handoff gap block real multi-step sessions), then the metadata that
makes discovery trustworthy, then the routing / CLI / skill surfaces that sit
on top of it. Each workstream is independently shippable; none is a
prerequisite for another except where noted.

---

## W1 — MCP stdio: server-side sampling without a deadlock (week 1)

**Problem.** `serve_stdio` reads stdin on the main thread and dispatches
`tools/call` synchronously; a tool that asks the client's LLM a question
(`interpret_result` via `sampling/createMessage`) waits for a reply that can
only be routed by the same loop, so it times out after 60 s. Tests inject
`route_response` from a side thread and never hit the real path.

**Design.**
- Reader thread: stdin lines go onto a queue; the main loop pulls requests
  from the queue, so responses to server-initiated requests are routed while
  a tool call is in flight (`_sampling.route_response` is already
  thread-safe).
- Tool calls keep running on the `_runner` worker; the progress drain and
  the sampling writer share one lock on stdout.
- A real subprocess test: launch `statspai-mcp` with `subprocess.Popen`,
  negotiate `initialize`, call `interpret_result` with a fake client that
  answers `sampling/createMessage`, assert the answer lands in the payload.
- Also: `tools/list` size budget test per profile; wheel smoke test that
  `statspai-mcp --help` runs; MCP protocol tests move into the push-time
  fast gate in `ci-cd.yml`.

- [x] reader-thread stdio loop with a single stdout lock
- [x] subprocess end-to-end test (initialize → tools/list → tools/call → sampling round-trip)
- [x] `tools/list` byte budget test (`curated` < 300 KB, `core` < 100 KB)
- [x] MCP suites in the push-time CI gate; `statspai-mcp --help` in the wheel smoke test — landed in `.github/workflows/ci-cd.yml` (fast gate + wheel smoke); the 2026-10-02 review found this line still marked pending. The gate was widened the same day to the whole agent chain (stdio chain, hardening, output-budget risk, handles, error envelope, metadata ratchet, validation scope, packaged skill).
- [x] CHANGELOG

## W2 — Data handles, inline data and transform chains (week 1–2)

**Problem.** The only way to hand data to a tool is an absolute file path.
An agent that reshapes, filters or imputes cannot pass the derived frame to
the next call, and small hand-made tables need a temp file.

**Design.**
- `DATA_CACHE` (LRU, mirrors `RESULT_CACHE`): `load_data(data_path=...)`
  → `data_id` + shape + dtypes + head; every tool accepts `data_id` wherever
  it accepts `data_path`; `statspai://data/{id}` resource for the profile.
- Inline input: `data_records` (list of row objects) or `data_csv` (string)
  on any tool, capped by `max_data_bytes`; recorded in `data_provenance` as
  `inline`.
- `transform_data` meta-tool: `query` (pandas `DataFrame.query`), `select`,
  `rename`, `dropna`, `winsor`, `mice`, `reshape` (`wide_to_long` /
  `long_to_wide`), plus `sp.<function>` returning a DataFrame → new
  `data_id`, with the lineage (`parent_id`, operation, args) stored on the
  handle and echoed in provenance.
- Results fitted from a handle carry `data_id` in `result_card.provenance`.

- [x] `agent/_data_cache.py` + `load_data` / `describe_data` tools
- [x] `data_id` accepted by every tool; `data_records` / `data_csv` inline input
- [x] `transform_data` with lineage; `statspai://data/{id}` resource
- [x] provenance: handle lineage in `data_provenance` and `result_card`
- [x] tests (in-process + subprocess), session-instructions text, CHANGELOG, schema regen

## W3 — Registry metadata for the 982 auto-generated entries (week 2–3)

**Problem.** 66 % of registry entries carry no `returns`, `example`,
`assumptions`, `failure_modes` or `alternatives`; 31 % of parameter
descriptions are the placeholder "`<name> parameter.`". Search, `describe`
and the agent cards are only as good as this metadata.

**Design — three passes, each with a ratchet test.**
1. *Harvest*: `_auto_spec_from_callable` parses NumPy-style docstrings
   (`numpydoc`-compatible parser, no new dependency) for parameter
   descriptions, `Returns`, the first `Examples` block, `References` bib
   keys, and `Notes` / `Warnings` lines that state assumptions. Enum values
   are inferred from `{'a', 'b'}` / `Literal[...]` annotations; array item
   types from `Sequence[float]`; DataFrame / ndarray typed as `object` with
   a `x-statspai-type` marker instead of `string`.
2. *Inherit*: family members without a card inherit `assumptions` /
   `failure_modes` / `alternatives` from their dispatcher (`inherits_from`
   auto-derived by module + naming: `rd_*` → `rdrobust`, `did_*` → `did`,
   `synth_*` → `synth`, `dml_*` → `dml`, `iv_*` → `iv`, …), with an
   `inherited_from` marker so `describe_function` can say so.
3. *Curate*: hand-written cards for the ~120 estimators the parity ledgers
   cover (every Track A / original-data module's entry point) — assumptions,
   pre-conditions, failure modes with remedies, `not_recommended_when`,
   `typical_n_min`, `cost_profile`. Evidence tier (T1–T4 / S) becomes a
   structured field derived from `_parity_index.json`, not prose.
- Ratchets: `tests/test_registry_metadata_ratchet.py` pins the share of
  entries with `returns`, `example`, non-placeholder param descriptions,
  and agent-native fields; the floor only moves up.
- Fix the schema contradictions found in the audit (enum on numeric types,
  default not in enum, wrong default type, classes exported as functions:
  a `kind: class` marker so tools exclude them).

- [x] docstring harvest in `_auto_spec_from_callable` + ratchet test
- [x] family inheritance + `inherited_from` marker
- [x] schema contradictions fixed; `kind` marker for classes
- [x] curated cards, batch 1: family cards for every parity-ledger entry point without one (28 families, 272 members; card coverage 46 % → 75 %)
- [x] curated cards, batch 2: per-function refinement of the highest-traffic members (80 cards for did / iv / rd / synth / dml; the family card is the floor, the per-function card overrides field by field, hand-written registry entries keep the last word)
- [x] structured evidence tier field (`evidence` in describe_function / agent_card / x_statspai); `describe_function` merges inheritance
- [x] CHANGELOG, schema regen, docs/stats

## W4 — Machine-readable estimator routing (week 3)

**Problem.** The seven `docs/guides/choosing_*_estimator.md` decision
tables exist only as Markdown that is not shipped in the wheel; `sp.recommend`
needs a live DataFrame; 17 `alternatives` point at names that do not resolve.

**Design.**
- `src/statspai/_routing/*.yaml` (or Python literals): one decision table per
  family — `question`, `answers`, `route` (function + arguments), `why`,
  `assumptions_added`, `read_more` (guide anchor). The Markdown guides are
  regenerated *from* the tables (`scripts/build_choosing_guides.py`), so the
  two cannot drift.
- `sp.route(family, **answers)` and `sp.decision_guide(family)` in Python;
  `route_estimator` MCP tool; `statspai://guide/{family}` resource serving
  the rendered guide; guides added to `package-data`.
- Test: every `route` target and every registry `alternatives` entry
  resolves to a registered callable.

- [x] decision tables for did / iv / rd / matching / ml_causal / qte / dynamic_panel
- [x] `sp.route` / `sp.decision_guide` + registry entries
- [x] guides packaged (byte-synced with docs/guides, pre-push hook); MCP resource + `route_estimator` tool. (Regenerating the Markdown *from* the tables was dropped: the guides carry prose the tables do not; the tests bind the two instead.)
- [x] resolvability test for routes and `alternatives`
- [x] CHANGELOG, schema regen

## W5 — CLI that runs estimators (week 3–4)

**Problem.** `statspai` only lists / describes / searches. An agent in a
shell cannot run an analysis and get JSON back.

**Design.**
- `statspai run <function> --data x.csv [--data-id ...] --arg y=wage
  --arg treat=treated --json` → `to_dict(detail=...)`; `--detail`,
  `--as-handle` writing a result file, `--out result.json`.
- Family shortcuts `statspai did|iv|rd|synth|dml ...` mapping flags to the
  dispatcher signature from the registry schema (no hand-written flag
  tables).
- `statspai mcp [--profile ...]` as the documented entry to the server;
  non-TTY stdout defaults to JSON.
- Errors: structured `StatsPAIError.to_dict()` on stderr, exit code by
  `error_kind`.

- [x] `run` subcommand + schema-driven argument parsing
- [x] family shortcuts; `mcp` subcommand; JSON default when piped
- [x] tests (subprocess), docs/guides/agent_api.md, CHANGELOG

## W6 — Skill package and agent docs (week 4)

**Problem.** `StatsPAI_full_data_analysis_skill/` breaks the skill naming
rules (uppercase, underscores), its description is over the 1,024-character
limit, `SKILL.md` is a 2,263-line monolith stamped 1.19.0, it never mentions
MCP, and it is not shipped. There is no `AGENTS.md` / `llms.txt`.

**Design.**
- Rename to `skills/statspai-analysis/` with valid frontmatter; split into
  `SKILL.md` (< 300 lines, workflow + when-to-use) plus `references/`
  (families, MCP loop, result contract, citations); version stamp read from
  `sp.__version__` by the drift script.
- `AGENTS.md` at repo root (how an agent should use the package: import
  alias, discover → describe → call, result contract, citations rule) and
  `llms.txt` pointing at the packaged guides and schema bundle.
- Install path: `statspai skill install [--target ~/.claude/skills]` copies
  the packaged skill; skill ships in `package-data`.
- `validate_api_claims.py` runs in CI on the split files.

- [x] rename + frontmatter + split; drift script updated
- [x] `AGENTS.md`, `llms.txt`; README pointers
- [x] `statspai skill install`; packaged; CI check
- [x] CHANGELOG

---

## Cross-cutting rules for every workstream

- Registry / schema / count lines regenerate in the order
  `build_parity_index.py` → `dump_schemas.py` → `registry_stats.py --check`
  (CLAUDE.md §9.2).
- Any change to a file on a Track A estimation path re-runs the provenance
  trace in a `site-packages` venv (the container's `dist-packages` layout
  hides third-party boundaries — see the 2026-09-28 entry in
  `docs/dev/jss_review_changes.md`).
- New public functions are registered, have an `example`, a correctness test
  and a boundary test; `examples-coverage` must stay at 0 missing.
- No silent degradation: MCP / workflow best-effort paths use
  `record_degradation`.

## Log

- 2026-09-28 — plan written; W1 started.
- 2026-09-28 — W1 done (reader-thread stdio loop, subprocess e2e test, CI gate).
- 2026-09-28 — W2 done (data handles, inline tables, transform chains with lineage).
- 2026-09-28 — W3 pass 1–2 done (docstring harvest, family inheritance, schema reconciliation, ratchet).
- 2026-09-28 — W3 pass 3 batch 1 done (28 family cards). Evidence-tier field and batch 2 remain.
- 2026-09-28 — W4 done (routing tables, sp.route / sp.decision_guide, route_estimator tool, packaged guides).
- 2026-09-28 — W5 done (statspai run / family shortcuts / route / mcp).
- 2026-09-28 — W3 evidence-tier field done (structured `evidence` in every discovery view).
- 2026-09-28 — W6 done (packaged `statspai-analysis` skill, `statspai skill install|validate|path`, AGENTS.md, llms.txt). Remaining across the plan: W3 per-function card refinement (batch 2); the CI workflow patch in `plans/pending-workflow-patches/` needs a user push.
- 2026-09-28 — W3 per-function cards, batch 2 done (`statspai._function_cards`, 80 entries; seed precedence fixed and tested). Remaining across the plan: the CI workflow patch in `plans/pending-workflow-patches/` needs a user push.

---

## Follow-up audit (2026-09-28, second pass) — W7–W10

A second agent-native audit (hands-on probes + static review) found gaps
beneath the W1–W6 surfaces. All four landed together.

## W7 — Result contract on every result class

- [x] `to_dict(detail=)` / `violations()` / `next_steps()` / `result_card()` on `ResultProtocolMixin`; legacy `to_dict` wrapped; NamedTuple / plain results via `attach_result_protocol` (298 / 300; gaps listed in `result_protocol_audit.py::AGENT_CONTRACT_GAPS`)
- [x] `next_steps()` no longer prints by default
- [x] `result_card` provenance: `result_class`, `seed` / `reproducible` / `seed_source`
- [x] audit ratchet + `tests/test_result_agent_contract.py`

## W8 — MCP protocol, errors as results, bounded data access

- [x] tool failures are `isError` results with `error_kind`; stale `result_id` errors on every path
- [x] `ping`; worker pool; `notifications/cancelled`; honest timeout reporting
- [x] output byte budget with `truncated`; `_nonfinite`; compact text block; `replay`
- [x] `STATSPAI_MCP_DATA_ROOTS`; remote data opt-in + byte cap; data-cache byte cap; schema-derived `readOnlyHint`; `transform_data` AST allowlist
- [x] one default profile (`curated`); no hard-coded counts in help / docs

## W9 — Error taxonomy, CLI and pipeline errors

- [x] `ColumnNotFound` with `did_you_mean`; `MissingDependencyError` with install command
- [x] remediation: `missing_arguments`, `unknown_argument`, `missing_dependency`, `column_not_found`
- [x] pipeline stages keep structured payloads; fallbacks record degradations
- [x] CLI strict JSON, `runtime_warnings`, exit codes by kind

## W10 — Discovery, schemas, cards

- [x] search: estimand / design vocabulary, dispatcher-first, card text, partial-match fallback
- [x] `result_class` + `x_statspai.returns`; enums; `x-statspai-role`; `x-aliases` / `x-canonical`; array item types; classes out of `all_schemas()`
- [x] `alias_of` (aliases listed once in the MCP manifest); per-field card `provenance`
- [x] re-frozen coverage floors; placeholder-description ceiling 0.55 → 0.39; 28 new estimator cards

**Left open:** default seeds still differ across stochastic estimators
(`dml` / `bcf` default 42, `causal_forest` / `callaway_santanna` /
`rdrobust` default `None`) — unifying them changes seeded numbers and needs
its own ⚠️ entry; `did_2x2` and other hand-written provenance blocks do not
record `seed`; `CrossValidationResult.next_steps()` returns `List[str]`.
