# Cross-language coverage campaign

> Working plan for the parity campaign that follows the RD-family sweep
> (`docs/dev/rd_parity_sweep_findings.md`). Same method, applied family by
> family. Every claim here is a measurement or a target, never a summary of
> intent.

## Where the number stands

`sp.parity_summary()` at the head of this campaign:

| denominator | cross-language (T2) | any evidence | total |
| --- | ---: | ---: | ---: |
| estimator callables | 177 | 398 | 773 |
| infrastructure (parity N/A) | 0 | 0 | 124 |
| result / exception classes | 0 | 0 | 285 |

**596 estimator callables carry no cross-language evidence.** That is the
working set. It is not a list of suspected defects — many of these methods
have no R or Stata sibling at all and can never exceed `analytical-only`.
The campaign's job is to separate those two cases, module by module, and
convert every function that *does* have a canonical reference.

## Why this is worth three months

The RD sweep is the argument. Nine functions were audited against the
reference they already named in their own docstrings; six carried
correctness defects, one of which (`sp.rdrandinf` publishing `p = 0.000`
from an estimate that never computed) is the worst class of failure a
statistics package can have — a number that is not merely wrong but wrong
in the direction that gets published.

None of the six were found by unit tests. All six were found by holding the
implementation against the reference on the same bytes. That is the yield
rate this campaign is built on: **a function whose docstring names an R or
Stata command, and which has never been compared to it, is a coin flip.**

## Method (per family)

Unchanged from the RD sweep, in this order:

1. **Read the reference's own source**, not its documentation and not
   memory. `deparse()` in R, `viewsource` in Stata. The `sp.rdms` defect
   was two naming traps deep (R's `rdms$B` is a bias-corrected *estimate*,
   Stata's `e(B)` is a bias *bandwidth*); no amount of reading the manual
   would have surfaced it.
2. **Compare on identical bytes.** One CSV, written once, read by all three
   sides.
3. **Bisect the first divergence** — design matrix, weights, residuals,
   meat matrix — before proposing any explanation. CLAUDE.md 5.1's decision
   tree, step 1: assume we are wrong.
4. **Fix, or classify.** A gap that is real gets a T2 module. A gap that is
   a documented convention difference gets a registered tolerance and a
   written mechanism. A gap that cannot be located gets grade C and stays
   visibly open — it does not get a widened tolerance.
5. **Register through the gates, not the checklist.**
   `tests/test_parity_harness_contract.py`, `scripts/tier_a_fixture_lock.py`,
   `make -C Paper-JSS audit`.

## Phase 1 — close what is already known to be open

| item | state |
| --- | --- |
| Fuzzy RD on the CCT path (4 strict xfails) | the last cross-language xfail in the suite |
| `black` pin 23.9.1 vs the 26.5.1 the code reflects | O7; the hook rewrites files as a side effect |
| Paper-JSS regeneration after the RD merge | O3; includes one hard-coded count that the claims mechanism does not reach |

## Phase 2 — family sweeps, ordered by yield

Ordering is (functions reachable at T2) x (probability the reference
disagrees), highest first. Counts are estimator callables with no
cross-language evidence today.

| # | family | uncovered | canonical reference | reachable at T2 |
| --- | --- | ---: | --- | ---: |
| 1 | regression | 9 | Stata `etregress` `sqreg` `jive` `mixlogit`, R `quantreg` | ~7 |
| 2 | network | 21 | R `igraph`, `sna`, `ergm` | ~15 |
| 3 | spatial | 29 | R `spdep` / `spatialreg` / `GWmodel`, Stata `spregress` | ~20 |
| 4 | mendelian | 23 | R `MendelianRandomization`, `TwoSampleMR` | ~15 |
| 5 | panel | 18 | Stata `xtdpdsys` `xtlsdvc` `xtnbreg` `xtgls`, R `plm` | ~12 |
| 6 | diagnostics + postestimation | 26 | Stata `estat` / `hausman` / `vif`, R `sensemakr` | ~10 |
| 7 | decomposition | 18 | Stata `oaxaca` `fairlie` `rifhdreg`, R `dineq` | ~8 |
| 8 | structural | 10 | Stata `opreg` `levpet` `prodest`, R `prodest` | ~6 |

Families deliberately **not** on this list, and why: `dag` (15),
`neural_causal` (11), `causal_llm` (4), `causal_rl` (3), `causal_text` (2)
have no deterministic external sibling — their ceiling is `analytical-only`
and pretending otherwise is the failure mode this whole apparatus exists to
prevent. `bayes` (14) has R siblings but MCMC puts it at T3 by
construction.

If every reachable row lands, cross-language coverage moves from 177 to
roughly 270 of 773 (~35%). That number is a target, not a promise; the
honest outcome of a sweep is sometimes "this family has no reference" or
"the reference disagrees with itself".

## Ledger

Findings are appended to `docs/dev/rd_parity_sweep_findings.md`'s successor
per family, and every correctness fix reaches `CHANGELOG.md` under
**⚠️ Correctness fixes** with the recompute advice a user needs.
