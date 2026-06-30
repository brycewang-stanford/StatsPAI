# Parity Status — making cross-language alignment a first-class, auditable property

> Goal: turn StatsPAI's 1,100+ functions from a *liability* ("can I trust
> function #847?") into an *auditable asset* — every function carries a
> queryable parity grade backed by a committed, version-pinned artifact, and
> every gap is honestly marked rather than hidden.

This is the single highest-leverage trust investment for an econometrics
audience. Stata's 40-year reputation rests on *Methods and Formulas* +
traceability; every R package is reverse-dependency-checked and JSS-reviewed.
A reviewer who opens function #800 must be able to see, in one call, *what it
was aligned against, to what tolerance, on which test, and how closely.*

## What already existed (the raw material)

Parity evidence was real but **fragmented across three subsystems and not
queryable**:

1. `tests/r_parity/` — the **Track A 3-way harness** (StatsPAI ↔ R ↔ Stata),
   64 modules, each with `NN_method_{py,R}.json` full-precision results, a
   `NN_method.do` Stata sibling, pinned R versions (`renv.lock` +
   per-run `provenance`), a pre-registered tolerance budget
   (`compare.py::TOLERANCES`), and headline-statistic selection
   (`compare.py::HEADLINE`) feeding the JSS Appendix B table.
2. `tests/reference_parity/` — 48 pytest files: frozen-R fixtures (bit-exact)
   + deterministic-DGP recovery tests (analytical), documented in
   `REFERENCES.md`.
3. `tests/external_parity/` — 6 files pinning replicas to published paper
   numbers, documented in `PUBLISHED_REFERENCE_VALUES.md`.

The registry's `validation_status` field (`certified`/`validated`/
`api_stable`) was a *coarse one-word tier* — no reference package+version, no
command, no tolerance, no test id, no relative error. That is the gap this
work closes.

## Status taxonomy (user-facing parity grade)

| grade | meaning | maps to existing tier |
| --- | --- | --- |
| `bit-exact` | matches a named R/Stata reference to machine tolerance (headline rel ≤ 1e-6) | certified |
| `aligned` | matches a named reference within a documented, pre-registered looser tolerance (cross-fit / convention disagreement) | certified |
| `analytical-only` | recovers a known population parameter on a deterministic DGP, or a closed-form identity (no cross-package reference) | validated |
| `external-replication` | reproduces published paper numbers on a calibrated replica | validated |
| `unverified` | registered public API, no qualifying numerical-parity evidence yet — **the honest gap** | api_stable |

## Architecture

```
committed parity artifacts (tests/r_parity, reference_parity, external_parity)
        │   scripts/build_parity_index.py   (single producer; zero-hallucination:
        ▼                                    every field traces to an artifact)
src/statspai/_parity_index.json             (frozen snapshot; works in wheels)
        │   src/statspai/parity.py
        ▼
sp.parity_status(name) · sp.parity_matrix() · sp.parity_summary()
        │
        ├── docs/parity.md          (public, auto-generated matrix)
        ├── README badge            (honest verified/total counts)
        └── CI drift gate           (snapshot must match artifacts)
```

Key integrity rule (CLAUDE.md §7 + §10): the index **consumes the project's own
`collect()` + `HEADLINE` + `TOLERANCES`** — the exact selection the JSS paper
reports — so it can never claim a tighter grade than the committed comparison
supports. The generator fails loud if any committed golden under-performs its
own registered budget.

## Week plan

- **Day 1 ✅** — extractor + `_parity_index.json` + `sp.parity_status` /
  `parity_matrix` / `parity_summary`, wired into the public API and
  auto-registered.
- **Day 2 ✅** — aggregated `reference_parity` (DGP recovery → analytical-only;
  curated frozen-R promotions ipw/g_computation/tmle → bit-exact) and
  `external_parity` → external-replication; dispatcher families handled with a
  variant-specificity note; standalone aliases (oaxaca/dfl_decompose/mediate)
  credited from their backing Track A module; scan-noise + registry-gap
  detection. **111 verified records (58 bit-exact, 4 aligned, 45
  analytical-only, 4 external-replication).**
- **Day 2–3 ✅** — `_parity_index.json` packaged in the wheel (pyproject +
  MANIFEST); reconciled against `registry.validation_status` — **106/108 agree**;
  the 5 divergences are the index being *more accurate*; 2 reverse cases are
  the old seed over-marking datasets. No `validation_status` mutation (JSS-safe).
- **Day 3–4 ✅** — `tests/test_parity_index.py` (14 tests): API contract,
  taxonomy validity, artifact existence (no phantom evidence), snapshot +
  doc drift gates (run in normal pytest CI), budget-guard, and the
  validation_status reconciliation invariant with a pinned benign-divergence
  allowlist. Schema bundle regenerated so agents see the new tools.
- **Day 4–5 ✅** — auto-generated public [`docs/parity.md`](../parity.md)
  (mkdocs nav already wired) + README "Cross-language parity, made queryable"
  section (qualitative pointer, no hardcoded counts to avoid a drift surface).
- **Day 5–6 ✅** — honest [gap inventory](parity_gap_inventory.md): the real
  metric is **111/964 estimators = 11.5%** (171 infra functions are parity-N/A);
  gap mapped by family with Tier 1/2/3 alignment prioritization.
- **Day 6–7** — JSS-safety review: the public function count moved 1,130 →
  1,135 by adding the parity introspection API, and the schema bundle +
  registry at-a-glance counts changed — the inventory tables and any headline
  count in `paper.md` / README / docs must be regenerated *consciously*, never
  silently. MCP tool-exposure check; full test pass; summary.

## Long-term goals (beyond the week)

1. **Close the gap, honestly.** Every `unverified` function with a named
   R/Stata sibling becomes a tracked alignment target; `parity_summary()`'s
   verified fraction is the north-star metric, reported per release.
2. **Parity as a release gate.** No estimator ships `certified` without a
   committed, version-pinned artifact; `--check` blocks drift in CI.
3. **Reviewer-facing.** `docs/parity.md` is the public matrix a JSS / Stata /
   R reviewer can audit function-by-function; the paper's Appendix B is
   generated from the same index.
4. **Agent-facing.** Agents call `sp.parity_status()` before relying on a
   number, and downgrade confidence on `unverified` functions automatically.
