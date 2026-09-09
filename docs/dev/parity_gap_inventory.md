# Parity coverage gap inventory (dev / planning)

> **Judgment only — no counts.** Live coverage numbers come from
> `sp.parity_summary()` and the auto-generated
> [parity matrix](../parity.md). This page holds the prioritization and the
> candidate references that the machine index deliberately does not assert.

## Where the numbers live

**This page carries judgment, not counts.** Every count it used to hardcode
is now generated from the committed artifacts:

- `docs/parity.md` — "Coverage at a glance" (evidence kinds split), "Honest
  denominators" (estimator / infrastructure / result-class strata), and
  "Coverage by estimator family" (the gap map that used to be transcribed
  below).
- `sp.parity_summary()` — the same numbers at call time, including
  `by_evidence_kind` and `denominators`.

The snapshot this page previously froze (2026-07-01, 1.20.0) drifted within
weeks: `network` was recorded as `EMPTY 0/33` and now holds four
cross-language rows, `robustness`'s three bit-exact closed forms had moved
to the `causal` category, and the "213 / 964 estimators" denominator no
longer matched any live computation. A hand-maintained coverage table in a
repository that regenerates its own parity index is a drift surface with no
compensating benefit, so the counts were removed rather than re-pinned.

Two framing rules survive from that snapshot because they are judgment, not
arithmetic:

1. **Report cross-language coverage separately from known-truth coverage.**
   Only the first answers "does StatsPAI agree with Stata/R". Summing them
   lets the smaller claim borrow the larger one's authority.
2. **Use the estimator denominator.** Result classes can never carry a
   parity grade and infrastructure functions render tables or load data;
   including them dilutes the metric without making it more honest.

> Candidate reference ecosystems named below are **leads to verify before
> alignment**, not parity claims (CLAUDE.md §10).

## Prioritization — where to spend alignment effort

**Tier 1 — high leverage, clear cross-language sibling, large family.**
One module here verifies many functions and closes an `EMPTY` row.
- **spatial** (30 gap) — SAR/SEM/SDM ML and SAR-2SLS/SEM-GMM now bit-exact vs
  `spatialreg` (modules 65--66). Remaining leads to verify: spatial panels
  (R `splm`, Stata `spxtregress`), GWR (`GWmodel`), and the SARAR GMM /
  heteroskedastic-GM estimators (reconcile the joint moment sequence against
  `spatialreg::gstsls` / `sphet`).
- **panel** (29 gap) — extend the existing Track A panel module: dynamic
  (`xtdpdgmm`, `plm::pgmm` beyond `xtabond`), spatial panels.
- **epi** (20) — candidate refs: R `epiR` / `survival` / `metafor`, Stata
  `epitab` / `st` suite (several already have `external_parity` via NHEFS).
- **survival** (11 gap) — R `survival` (KM/AFT), `cmprsk` (Fine-Gray), Stata
  `stcox` / `streg` / `stcrreg`.
- **timeseries** (17 gap) — R `vars` / `urca` / `rugarch`, Stata `var` /
  `vec` / `arch`.

**Tier 2 — alignable, partial families to finish.**
- **decomposition** (27 gap) — extend the `_common.py`-backed family
  (Gelbach, Das-Gupta, inequality) against R `oaxaca` / `dineq` / `ddecompose`.
- **inference** (23 gap) — bootstrap / wild-cluster / MHT vs R `fwildclusterboot`,
  `sandwich`, `multcomp` (CR2/CR3/multiway already bit-exact).
- **mendelian** (31 gap) — R `MendelianRandomization` / `TwoSampleMR`
  (MR core already has analytical recovery).
- **frontier / structural / transport / survey / bartik** — established R/Stata
  siblings exist for most; verify per-method.

**Tier 3 — frontier methods, analytical/simulation is the honest ceiling.**
`neural_causal`, `conformal_causal`, `causal_llm`, `causal_rl`, `causal_text`,
`fairness`, `ope`, `surrogate`, `bridge`, most of `bayes` (where the right
evidence is convergence diagnostics + Monte-Carlo coverage, not bit-for-bit
parity). For these the target is a documented **`analytical-only`** record
(DGP recovery / closed-form / MC calibration), not a cross-package grade —
and that is the honest top grade, stated as such.

## The closing loop

1. Pick a Tier-1 family; add a Track A module (`tests/r_parity/NN_*.{py,R}` +
   Stata `.do`) **or** a `reference_parity` frozen fixture.
2. Regenerate: `python scripts/build_parity_index.py` — the new function(s)
   flip from `unverified` to a graded record automatically; the matrix,
   summary, and `docs/parity.md` update; the drift gate stays green.
3. `sp.parity_summary()`'s estimator-verified fraction is the metric of record;
   report it per release in `CHANGELOG.md`.
