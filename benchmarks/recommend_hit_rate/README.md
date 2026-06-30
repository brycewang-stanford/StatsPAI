# Recommendation Hit-Rate Benchmark

> The moat is not "the agent loop runs." The moat is **"the agent loop is
> right."** A plausible-but-wrong recommendation is worse than no
> recommendation, because it carries authority. This benchmark measures, across
> real published designs, whether `sp.recommend` proposes the estimator the
> paper actually used, and whether `sp.audit` knows the robustness checks a
> referee would demand. The hit-rate is a public, agent-native quality metric —
> the single number that lets the field trust StatsPAI's judgment.

## What it measures

Two metrics, scored against a ground-truth corpus of published empirical
designs (`corpus.yaml`):

1. **recommend hit-rate** (dynamic). Run `sp.recommend` on the real data and
   check whether an econometrically-*acceptable* estimator appears in the
   ranked output.
   - `HIT` — top-1 is acceptable.
   - `PARTIAL` — acceptable appears in top-k but not top-1.
   - `MISS` — no acceptable estimator surfaced.
   - `HARD-MISS` — top-1 is a *disqualifying* estimator (e.g. TWFE on a
     staggered design). This is the failure that destroys trust; the `--check`
     gate fails the build on any hard-miss.

2. **audit catalog coverage** (static). For the design's method family, does the
   `sp.audit` check catalog contain the ground-truth robustness checks? This
   answers "does the audit even know to ask what a referee would ask?"
   A later phase adds **dynamic audit recall**: fit the primary estimator, run
   `sp.audit`, and confirm it flags the missing checks as `MISSING` with a
   `suggest_function`.

## Architecture

The scoring engine is a registered, agent-native function —
`sp.recommend_benchmark()` (in `statspai/smart/recommend_benchmark.py`) — so
humans and agents query the hit-rate the same way. This directory holds the
corpus, the CLI wrapper, and the generated artifacts.

```
benchmarks/recommend_hit_rate/
├── README.md       — this file (design + roadmap)
├── corpus.yaml     — ground-truth corpus (versioned, citation-disciplined)
├── harness.py      — thin CLI over sp.recommend_benchmark() → writes scorecard
├── scorecard.md    — generated, human-readable
├── scorecard.json  — generated, machine-readable (CI / dashboards)
└── FINDINGS.md     — concrete correctness gaps surfaced, most-actionable first

statspai/smart/recommend_benchmark.py   — the scoring engine (sp.recommend_benchmark)
.github/workflows/recommend-benchmark.yml — CI ratchet (hard-miss=0 + hit-rate floor)
docs/recommend_benchmark.md             — public methodology + scorecard page
```

Query it like any other agent-native function:

```python
import statspai as sp
card = sp.recommend_benchmark()           # full run
card["summary"]["hit_rate_top1"], card["summary"]["hard_miss_rate"]
```

### Corpus tiers
- **Tier A — data-backed.** The dataset is bundled in `sp.datasets`; recommend
  runs on the **real** data. The rigorous core. (Seed: 8 entries.)
- **Tier B — design-card.** We encode the design fingerprint; recommend runs on
  a faithful stub. Scales to the 50-paper canon, especially the staggered-DiD
  era (Goodman-Bacon / Callaway-Sant'Anna / Sun-Abraham / honest-DiD). (Roadmap
  Phase 3.)

### Scoring judgment lives in two explicit places
- `corpus.yaml` `ground_truth.estimator.{acceptable, disqualifying}` — the
  econometric equivalence class per design (e.g. CS ≈ Sun-Abraham ≈
  did_imputation are all valid-staggered; TWFE on staggered is disqualifying).
- `harness._NORMALIZE_RULES` — prose method label → controlled estimator tag.

Both are deliberately readable and reviewable; the benchmark's credibility
rests on these being defensible, not hidden.

## Citation discipline (CLAUDE.md §10 — zero-hallucination red line)

Every paper in the corpus cites a `bib_key` that **must** resolve in
`paper.bib` (the DOI-verified single source of truth). The harness asserts this
and reports `citation_errors`. Robustness-check `provenance` labels where each
"a referee would ask this" claim comes from (`paper_section`,
`methodological_critique`, `methodological_followup`, `replication_package`) and
never fabricates a referee report. Concerns we cannot yet source are tagged
`citation_needed` and are **excluded from scoring** until promoted. Expanding to
50 papers routes every new citation through Crossref/DOI verification before it
enters `corpus.yaml`.

## Run it

```bash
python3 benchmarks/recommend_hit_rate/harness.py          # human report + scorecard
python3 benchmarks/recommend_hit_rate/harness.py --json    # scorecard.json only
python3 benchmarks/recommend_hit_rate/harness.py --check    # CI gate (exit 1 on regression)
```

## Results

| metric | Day-1 (8) | Phase 2 (8) | Phase 3 (17) | + F-004 frontier (22) |
| --- | --- | --- | --- |
| entries | 8 Tier-A | 8 Tier-A | 17 core | 17 core + 5 frontier |
| core top-1 hit-rate | 0.625 (5/8) | 1.0 (8/8) | 1.0 (17/17) | **1.0** (17/17) |
| top-k hit-rate | 0.625 | 1.0 | 1.0 | 1.0 |
| hard-miss rate | 0.0 | 0.0 | 0.0 | **0.0** |
| errors | 0 | 0 | 0 | 0 |
| audit catalog mean recall (static) | 1.0 | 1.0 | 1.0 | 1.0 |
| audit dynamic mean recall (fit+audit) | — | 1.0 (8/8) | 1.0 (17/17) | **1.0** (17/17) |
| frontier coverage (gap-probe) | — | — | — | **1.0** (5/5) |

Phase 3 added 7 Tier-B adversarial design archetypes (synthetic stubs via
`sp.dgp_*`, each anchored to a DOI-verified method/critique paper in
`paper.bib`): the TWFE negative-weights trap (staggered + heterogeneous),
weak-instrument (→ LIML) vs strong-instrument (→ 2SLS), sharp vs fuzzy RD, a
clean 2×2, and strong-confounding selection-on-observables (→ PSM). **The
engine resisted every trap — 0 hard-misses across all 17 designs.**

**What works (verified on real data):** staggered DiD → Callaway-Sant'Anna
(never TWFE); weak & moderate IV → 2SLS with live first-stage F; sharp RD →
local-polynomial; single-treated-unit case study → synthetic control (after
F-001 fix); selection-on-observables → SOO hierarchy. When the top-1 estimator
is fitted and audited, `sp.audit` surfaces the ground-truth referee checks with
an actionable `suggest_function` for 7 of 8 designs.

**What the benchmark caught (and drove to a fix):**
- **F-001 (FIXED).** All three synthetic-control designs were misrouted to
  staggered DiD — `recommend` was blind to the synth design family despite 20+
  synth estimators shipping. Phase 2 added synth detection + a ranked synth
  recommendation block; all three flipped MISS→HIT with no regression. Locked
  by `tests/test_recommend_synth_detection.py`.
- **F-002 (OPEN).** On the selection-on-observables design, `recommend` leads
  with bare OLS, so the realized audit asks only regression checks and misses
  overlap/balance/OVB — the textbook plausible-but-wrong loop. Needs a design
  decision (see `FINDINGS.md`).

---

## Roadmap — one week to a trustworthy public benchmark

The week is organized so that **every day ends with a regenerated scorecard**
and the numbers only move in defensible ways.

### Phase 0 — Foundation ✅ (done)
- Corpus schema + 8 Tier-A entries (all §10-verified citations).
- Harness: recommend hit-rate (dynamic) + audit catalog coverage (static),
  scorecard.{md,json}, `--check` CI gate.
- First scorecard + first finding (F-001).

### Phase 1 — Harden the scorer & adversarial traps (Day 2)
- Expand `_NORMALIZE_RULES` coverage; add a unit test asserting every label
  `recommend` can emit normalizes to a known tag (no silent `unknown:` leaks).
- Add explicit **adversarial trap** entries that must be `HARD-MISS`-proof:
  staggered-with-TWFE-bait, weak-IV-with-2SLS-bait, RD-with-manipulation,
  single-treated-unit-with-DiD-bait. Assert recommend resists each.
- Add a **design-detection accuracy** sub-metric (detected vs. true design).

### Phase 2 — Close F-001 + dynamic audit recall (Days 2–3)
- Implement the synth branch in `detect_design` + `recommend` (see F-001).
  Re-run; Prop99/Basque/German must flip `MISS → HIT` with **no regression** on
  the staggered/IV/RD/SOO entries. This is a real correctness improvement the
  benchmark drives.
- Add dynamic audit recall: fit each entry's primary estimator with
  `as_handle=True`, run `sp.audit(result_id=...)`, and score whether it flags
  the ground-truth checks as `MISSING` with the right `suggest_function`. This
  is the true "does the loop close correctly" metric.

### Phase 3 — Scale to the 50-paper canon (Days 3–5)
- Build Tier-B design cards for the staggered-DiD era and beyond, each with a
  Crossref/DOI-verified citation added to `paper.bib` first (§10). Target mix:
  ~20 staggered-DiD (the highest-risk regime), ~10 IV (incl. weak-IV &
  shift-share), ~8 RD, ~7 synth, ~5 matching/DML. Ground-truth estimator +
  referee robustness checks sourced from each paper's own robustness section
  and the published methodological-critique literature.
- For each Tier-B paper, encode a faithful design fingerprint → synthetic stub
  (via the existing `sp.dgp_*` generators) so recommend runs without the real
  (often unavailable) microdata. Clearly label Tier-B as fingerprint-based.

### Phase 4 — Make it public & non-regressing (Days 5–6)
- Promote the scorer to an agent-native surface: `sp.recommend_benchmark()`
  (registered, schema'd) so humans and agents query the hit-rate the same way
  they query everything else. Mirror the `verify_benchmark` pattern.
- CI job (scheduled, like `parity-guards.yml`): regenerate the scorecard, run
  `--check`, and fail on any hard-miss, error, citation error, or top-1
  hit-rate drop below a pinned floor. The hit-rate becomes a ratcheted,
  un-regressable property — like the parity tests.
- A `docs/` page publishing the scorecard + methodology, so the number is
  externally citable: *"StatsPAI recommends the paper's estimator X% of the
  time across N published designs, and never leads with a disqualifying one."*

### Phase 5 — Adversarial / red-team hardening (Day 7)
- A panel of independent "skeptic" checks per recommendation: for each HIT,
  an adversarial pass that tries to argue the recommendation is wrong for the
  design (mirrors the user's adversarial-validation framing). Promotes any
  surviving objection to a FINDINGS entry.
- Stress the corpus with near-miss designs (staggered-but-no-variation,
  fuzzy-RD, sharp-RD-with-bunching, RDiT) where the *right* answer is subtle.

## Definition of done (the trust bar)
1. ≥ 40 published designs scored, every citation §10-verified in `paper.bib`.
2. **Zero hard-misses** across the corpus (the non-negotiable invariant).
3. Top-1 hit-rate published, with every miss explained in `FINDINGS.md`.
4. Dynamic audit recall scored, not just static catalog coverage.
5. CI gate makes the hit-rate a ratcheted property; a regression fails the build.
6. A public docs page + queryable `sp.recommend_benchmark()`.
