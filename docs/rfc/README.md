# RFC — `docs/rfc/`

Design documents for not-yet-landed StatsPAI work. Each RFC describes a proposed API + identification/estimation blueprint + test plan, **before** code lands on `main`.

An RFC ≠ a guide:

- **RFC** lives here, owns the design, may contain `[待核验]` placeholders for paper formulas pending verification, gets deleted or archived once the feature ships.
- **Guide** lives in `docs/guides/`, is user-facing, assumes the feature works, cites verified references only.

When an RFC graduates (feature implemented + tested + reference-aligned), move the user-facing content to `docs/guides/` and either archive or delete the RFC.

## Current RFCs

| RFC | Status | Owner | Target |
|---|---|---|---|
| `did_roadmap_gap_audit.md` | draft 2026-04-23 | — | survey |
| `continuous_did_cgs.md` | draft 2026-04-23 | — | v1.7 |
| `multiplegt_dyn.md` | draft 2026-04-23 | — | v1.7 |

## Rules

1. Any paper formula in an RFC **must** carry `[待核验 — <source>, <location>]` until two-source verification completes (CLAUDE.md §10).
2. RFCs do not register functions, do not ship code, do not add to `paper.bib`. They only *propose*.
3. When you lift an RFC into production code, grep this directory and delete the stale copy — never ship both.
