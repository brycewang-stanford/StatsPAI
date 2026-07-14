# Agent-Card Coverage Specification

> Contract for what counts as a "complete" agent card on a registered
> StatsPAI function.  Read this before adding new functions or filling
> in metadata for existing ones.  The CI ratchet in
> [`tests/test_agent_card_coverage.py`](../tests/test_agent_card_coverage.py)
> enforces "only-up" against the floor in
> [`scripts/agent_card_coverage_floor.json`](../scripts/agent_card_coverage_floor.json).

## Why this file exists

StatsPAI has ~1000 public functions.  Each is exposed to LLM agents
through [`sp.describe_function`](../src/statspai/registry.py),
[`sp.agent_card`](../src/statspai/registry.py), and
[`sp.function_schema`](../src/statspai/registry.py).  An agent calling
`sp.did(...)` blindly is a defect — the wrapper has to know
*identifying assumptions*, *common failure modes*, and *which alternative
estimator to fall back to*.  This document defines, per field, what
"populated" means; the coverage script measures it.

Two views of the same data exist:

- **Curated (raw) view** — what a human has written into
  [`registry.py`](../src/statspai/registry.py) or the auto-card layer.
  Measured by [`scripts/agent_card_coverage.py`](../scripts/agent_card_coverage.py).
  Drives the work-to-do.
- **Effective (post-fallback) view** — what an LLM agent sees after
  `_param_description` synthesizes placeholder text for empty fields.
  Measured by [`scripts/schema_quality.py`](../scripts/schema_quality.py).
  Feeds JSS Table 9.

These two **must not be conflated**.  The fallback layer is a UX cushion,
not curation; treating it as curated is how silent quality regressions
hide.

## Three tiers, cumulative

### Tier-B — baseline (target: 100% coverage)

A Tier-B card is the *minimum* viable surface for an agent to decide
"do I call this function?".  All five fields are required:

| Field | Definition | Source of truth |
| --- | --- | --- |
| `description` | More than 30 characters of human-written prose. | `FunctionSpec.description` |
| `tags` | At least one tag from a controlled vocabulary (estimator family, design, output). | `FunctionSpec.tags` |
| `example` | One runnable `sp.xxx(...)` call. | `FunctionSpec.example` |
| `reference` | A bib key from [`paper.bib`](../paper.bib) **or** a verified DOI/arXiv ID.  Never a hand-written citation string — see [§10 in CLAUDE.md](../CLAUDE.md). | `FunctionSpec.reference` |
| `param.description` | At least one parameter has a non-empty `ParamSpec.description`.  (Per-parameter coverage is a separate Tier-B+ goal.) | `ParamSpec.description` |

The `description_30` lower bound exists because the auto-registration
pass can produce 10-character placeholders like `"Internal helper."` —
those don't count.

### Tier-A — agent-native (target: ≥ 70% of independent design points)

A Tier-A card lets an agent recover from misuse.  Required *on top of*
Tier-B:

| Field | Definition | Notes |
| --- | --- | --- |
| `assumptions` | One human-readable line per identifying / statistical assumption. | E.g. `"Conditional mean independence: E[u\|X] = 0"`. |
| `pre_conditions` | One line per data-shape requirement the agent should verify *before* calling. | E.g. `"data is a panel keyed by (unit, time) with at least 2 periods"`. |
| `failure_modes` | At least one `FailureMode(symptom, exception, remedy, alternative)`. | Recovery hints — what to try when this estimator breaks. |
| `alternatives` | Ranked list of `sp.xxx` fallbacks for when this is a poor fit. | E.g. for `did`: `["did_callaway_santanna", "did_sun_abraham"]`. |
| `typical_n_min` | Rule-of-thumb minimum sample size; `None` allowed only when no rule exists. | Documented in `validation_notes` if `None`. |

The target is **70% of independent design points**, not 70% of the
1,139-function surface.  Many functions are method variants behind a
dispatcher (`sp.synth(method=...)` has 20+) and should inherit from
their parent via `FunctionSpec.inherits_from` (Sprint 2).  See the
collapse-strategy doc once it lands.

### Tier-S — certified/validated evidence (target: flagship 50+)

Tier-A *and* `validation_status ∈ {certified, validated}`:

- **certified** — cross-language or published-reference parity evidence,
  typically from `tests/r_parity/`, `tests/stata_parity/`, or original-data
  published-reference fixtures.
- **validated** — known-truth, reference-parity, external-parity, coverage,
  or explicit convention evidence in this checkout.

This tier is gated by the parity-test work tracked separately in
[`scripts/stability_audit.py`](../scripts/stability_audit.py) and by the
JSS validation-evidence audit. It does not block Tier-A completion, and it
is not a marketing ratchet: if evidence weakens, a function should be
demoted to `api_stable` until the numerical evidence is repaired.

## The seven flagship families (must reach Tier-S)

1. `regress` (and OLS variants)
2. `did` / `did_callaway_santanna` / `did_sun_abraham` / `did_chaisemartin`
3. `ivregress` / `iv_2sls` / `liml`
4. `rdrobust` / `rd_donut` / `rdmulti`
5. `synth` family (via dispatcher)
6. `dml_plr` / `dml_irm` / `dml_ate`
7. `matching` family (psmatch, ip-weighting, entropy balancing)

These are the most frequent agent invocations.  If any of them slip from
Tier-S, treat it as a release-gate issue.

## Where to write the data

- **Hand-curated specs (Tier-A/S, ~200 functions)** — directly in
  [`src/statspai/registry.py::_build_registry`](../src/statspai/registry.py).
  These always win over the auto-pass.
- **Auto-baseline (Tier-B fill)** — emitted by
  [`scripts/gen_baseline_cards.py`](../scripts/gen_baseline_cards.py)
  into [`src/statspai/_baseline_cards.py`](../src/statspai/_baseline_cards.py),
  read by `_ensure_full_registry` after `_build_registry`.  The auto-pass
  **only writes empty fields**; it never overwrites curated content.
- **Dispatcher inheritance** — declared via `_INHERITANCE_SEEDS` in
  [`registry.py`](../src/statspai/registry.py); the rules are explained
  in the next section.

## Dispatcher inheritance

Many StatsPAI estimators come in families: a canonical parent
(`did`, `iv`, `rdrobust`, `synth`) and a long tail of method variants
(`callaway_santanna`, `kernel_iv`, `rd_honest`, `synthdid_estimate`).
Restating the four parallel-trends assumptions on every DiD variant
is both error-prone and bad metadata hygiene.

`FunctionSpec.inherits_from` solves this:

```python
FunctionSpec(
    name="callaway_santanna",
    category="causal",
    description="Callaway–Sant'Anna staggered DiD with group-time ATTs.",
    inherits_from="did",            # ⬅︎ absorbs did's assumptions etc.
    assumptions=[
        "No anticipation in any cohort.",   # method-specific addendum
    ],
    failure_modes=[...],            # method-specific failures
    ...
)
```

What inherits, what doesn't:

| Field | Inherits? | Why |
| --- | :-: | --- |
| `description` | ✗ | Always method-specific. |
| `example` | ✗ | Variant has its own call signature. |
| `params` | ✗ | Signatures diverge. |
| `reference` | ✗ | Variant cites its own paper. |
| `tags` | ✗ | Variant tags are usually narrower. |
| `assumptions` | ✓ | Union — child first, parent fills the rest, deduped. |
| `pre_conditions` | ✓ | Union, deduped. |
| `failure_modes` | ✓ | Union, deduped by `(symptom, exception)`. |
| `alternatives` | ✓ | Union, deduped. |
| `typical_n_min` | ✓ | Child wins if set; else first non-`None` ancestor. |
| `validation_status` | ✗ | Each variant carries its own parity evidence. |
| `limitations` | ✗ | Variant-specific gaps. |

**Cycles** are silently capped at depth 5 — they should never occur
in practice but the renderer won't hang if someone declares one.

**Coverage measurement** uses the *merged* view: a child whose
`inherits_from` resolves to a Tier-A parent counts as Tier-A even if
its own `assumptions` field is empty.  This is intentional — what
matters is what an agent sees from `FunctionSpec.agent_card`.

To add a new variant link, edit `_INHERITANCE_SEEDS` in
[`registry.py`](../src/statspai/registry.py).  Use sparingly:
**only canonical estimator children belong there**.  Output / plot /
diagnostic helpers should *not* inherit estimator assumptions, because
those assumptions don't apply to "render a coefficient plot".

## Citations are the only red line

Repeating [§10 of CLAUDE.md](../CLAUDE.md) because it is the cheapest
quality killer:

> Any new citation (docstring / `paper.bib` / `MIGRATION.md` /
> `CHANGELOG.md` / docs) must be verified against **two independent
> sources** before landing.  **Never** rely on LLM memory for DOI,
> author order, year, or journal — even for "obvious" papers like
> Abadie (2003) or Callaway & Sant'Anna (2021).

For the agent-card work specifically:

- The auto-baseline script only emits a `reference` field if it can
  match a `paper.bib` bib key already present in the repo.  Otherwise
  the field stays empty.
- The LLM-assist phase (Sprint 3) **must** hard-block any `reference`
  string that is not a `paper.bib` bib key.  No DOI guessing.
- PRs that touch citations include `refs verified via <source1>,
  <source2>` in the body.

A fake DOI in one card metastasizes — once a reviewer sees one bogus
reference, every estimator becomes suspect.  Cite a bib key or cite
nothing.

## CI ratchet

The floor in
[`scripts/agent_card_coverage_floor.json`](../scripts/agent_card_coverage_floor.json)
tracks card-completeness counters: per-tier totals, per-field counts, and
the evidence-tier counts exposed to agents. Non-validation card coverage is
a ratchet. Validation-status counts are evidence-audit outputs, not vanity
floors: they may go down when a function is honestly demoted from
`certified`/`validated` to `api_stable`. To intentionally raise the
card-completeness bar, run:

```bash
python scripts/agent_card_coverage.py --write-floor
```

after the new content is merged.  CI runs `--check` and fails if any
non-validation coverage counter dropped. Do not lower card-completeness
floors to hide missing metadata; do update validation-status expectations
when the evidence audit justifies a demotion.

## Status

Historical snapshot at v1.15.5 — Tier-B 127, Tier-A 84, Tier-S 78; denominators
are preserved in `scripts/agent_card_coverage_floor.json`.

| Sprint | Goal | Status |
| --- | --- | --- |
| 0 | Coverage script, spec doc, CI ratchet | **complete** — 5-mode `scripts/agent_card_coverage.py`, 15-counter floor, 9-test ratchet suite, `docs/agent_cards_spec.md`. |
| 1 | Auto-baseline → fill empty Tier-B fields from docstrings | **complete (mechanically saturated)** — `scripts/gen_baseline_cards.py` + `src/statspai/_baseline_cards.py` lifts `tags` to 100% and `example` to 36.6%. Further gains require docstring rewrites, not script changes. |
| 2 | Dispatcher inheritance → collapse to ~200 design points | **complete (PoC + propagation)** — `FunctionSpec.inherits_from` field, `_merge_inherited_view` helper, `_INHERITANCE_SEEDS` wires 41 variants (DiD / IV / RD / synth / MR). |
| 3 | Hand-curate Tier-A by category | **in progress** — 14 flagship estimators curated in this batch (`panel`, `feols`, `fepois`, `decompose`, `dfl_decompose`, `ffl_decompose`, `oaxaca`, `sar`, `sem`, `sdm`, 4× `mr_*`). Remaining ~56 design points to reach 70%-of-200 target are tracked as a separate follow-up sprint. |
| 4 | Tier-S parity evidence (parallel, long-running) | **pending** — gated by `scripts/stability_audit.py` and `tests/reference_parity/` work, independent of Tier-A curation. |
