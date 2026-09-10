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
| Fuzzy RD on the CCT path (4 strict xfails) | **closed** — the last cross-language xfail in the suite |
| `black` pin 23.9.1 vs the 26.5.1 the code reflects | open (O7 in the RD findings) |
| Paper-JSS regeneration after the RD merge | open (O3) |

## Phase 2 — family sweeps, ordered by yield

Ordering was (functions reachable at T2) x (probability the reference
disagrees). Spatial jumped the queue because `spdep` and `spatialreg` were
already installed; the others followed the same rule.

| # | family | uncovered at start | state | certified | defects found |
| --- | --- | ---: | --- | ---: | ---: |
| 0 | RD (pre-campaign sweep) | — | done | 2 modules | 13 |
| 1 | regression | 9 | partial | 2 (`etregress`, `sqreg`) | 4 |
| 2 | spatial | 29 | done | 13 | 4 + 1 gap |
| 3 | weak-IV / diagnostics / meta | 26 | partial | 5 | 2 |
| 4 | panel | 18 | partial | 3 | 2 |
| 5 | network | 21 | not started | — | — |
| 6 | mendelian | 23 | not started | — | — |
| 7 | decomposition | 18 | not started | — | — |
| 8 | structural | 10 | not started | — | — |

**Cross-language coverage: 169 → 201 of 773 estimator callables
(21.9% → 26.0%).**

Families 5 and 6 need R packages that are not installed here (`igraph`,
`sna`, `MendelianRandomization`); everything landed so far used a reference
that was already present.

## What the yield rate actually turned out to be

The plan predicted "a function whose docstring names an R or Stata command,
and which has never been compared to it, is a coin flip". Across five
families, 25 defects in roughly 55 functions examined — close to the guess,
and the severity distribution is worth recording because it is not what a
test-coverage metric would predict:

| class | count | examples |
| --- | ---: | --- |
| Wrong estimator under the right name | 4 | `etregress` MLE was the two-step; `panel_fgls` was `igls`; `getis_ord_local(star=False)` used Gi*'s moments; `rdms` built a different score |
| Formula error where the comment was right | 3 | `lm_tests` T term and J term; `join_counts` BW |
| Silently ignored argument | 4 | `rdbwselect(fuzzy=)`; `etregress(robust=, cluster=)`; `rdbwselect` comb selectors |
| Rounding a returned value | 3 | `sqreg` (4 dp), `vif` (2 dp), `rdbwselect` / `rdrobust` bandwidths (6 dp) |
| Missing term | 2 | RE panel binary intercept; two-step `etregress` Heckman correction |
| Wrong null distribution | 2 | `moran_residuals`; `rdrandinf` publishing p = 0.000 from a NaN |
| Approximation where an exact form existed | 2 | AR / CLR grid endpoints |

**None of these was found by a unit test, and the suite is not small — it
is over 17,000 tests.** The reason is structural: unit tests assert
directions, ranges and shapes ("C below 1 for smooth data", "the estimate
recovers delta", "p_sim below 0.05"), and every defect above satisfies
those. A doubled `LM_err` is still significant. A two-step still recovers
delta. Gi with the wrong standardisation still ranks hotspots in the same
order. What separates them is holding the implementation against the
reference it names, on the same bytes.

## Method notes worth keeping

* **The comment is often right and the line below it wrong.** Twice in one
  function (`lm_tests`). The author knew the formula; the transcription
  failed. No amount of reading the code catches this, because reading the
  comment feels like reading the code.
* **A gap that does not shrink when you refine the approximation is not an
  approximation error.** The RE panel binary defect was found by noticing
  0.39% at 12 quadrature points and 0.39% at 30.
* **Fixture design can hide a whole code path.** The repository's fuzzy RD
  fixture used one-sided noncompliance, which makes `rdbwselect` fall back
  to the sharp bandwidth — so the fixture could not tell a correct fuzzy
  bandwidth from a missing one, and recorded 1.2e-08 agreement while the
  argument was being discarded entirely.
* **Assert the identity, not just the reference.** `BB + WW + BW = S0/2`
  catches the join-count defect with no R installed. Tests of that shape
  survive fixture regeneration.

## Ledger

Per-family findings: `rd_parity_sweep_findings.md`,
`spatial_parity_sweep_findings.md`, `weakiv_parity_sweep_findings.md`,
`panel_parity_sweep_findings.md`. Every correctness fix reaches
`CHANGELOG.md` under **⚠️ Correctness fixes** with the recompute advice a
user needs, and `MIGRATION.md` with a table of what moves.
