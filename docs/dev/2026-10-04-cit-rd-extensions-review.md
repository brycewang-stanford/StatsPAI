# Cattaneo, Idrobo and Titiunik (2024), *RD Designs: Extensions*: what its replication files showed about StatsPAI

2026-10-04. Worktree `.claude/worktrees/cit-rd-extensions`, branch
`wt/cit-rd-extensions`.

## What was examined

The replication package of *A Practical Introduction to Regression
Discontinuity Designs: Extensions* (Cambridge Elements, 2024;
`cattaneo2024extensions` in `paper.bib`): six R / Stata / Python scripts
and their datasets, plus the arXiv text. They sit in
`改进建议-收集整理/Cattaneo-RD-Extensions-2024/`, which is gitignored.
Neither the scripts nor the data are redistributed; the committed tests
use the Senate file already in `tests/reference_parity/_fixtures/`.

| Section | Topic | Commands |
| --- | --- | --- |
| 2 | Local randomization (U.S. Senate) | `rdrandinf`, `rdwinselect`, `rdrobust` |
| 3 | Fuzzy RD (Ser Pilo Paga) | `rdrobust fuzzy()`, `rdrandinf fuzzy()`, `rddensity bino_w()` |
| 4 | Discrete running variable (academic probation) | `rddensity`, `rdrobust vce(cluster)`, collapsed data, `rdwinselect wmin() wstep()` |
| 5 | Multiple cutoffs, multiple scores, geographic RD | `rdmc`, `rdms`, `rdmcplot` |

## Method

The six R scripts were run as shipped (R 4, `rdrobust` 4.0.0, `rdlocrand`
2.0, `rddensity` 2.6, `rdmulti` 2.0.0). Every call was then repeated
through its return value rather than its printed table, 53 result sets in
all, and compared number by number with the StatsPAI call. Where a
formula was not documented it was identified from outputs on synthetic
data. The R packages are GPL; their source was not read.

The book is from 2024 and prints output from an older `rdlocrand`. That
turned out to matter, see D2.

## What agreed and was left alone

- `sp.rdrobust`: 20 calls (sharp, fuzzy, `cerrd`, clustered on the score,
  mass points, collapsed data). Robust intervals agree to 1e-11; the
  collapsed-data call to 3e-7.
- `sp.rddensity`: statistic, p-value and bandwidths on the fuzzy and the
  discrete data.
- `sp.rdms` at a boundary point, and `sp.rdrobust` on the perpendicular
  distance.

## Defects fixed

| # | Function | What was wrong | Evidence |
| --- | --- | --- | --- |
| F1 | `sp.rdrandinf(p>=1)` | Residualized on one polynomial pooled across the cutoff, which absorbs the effect | Senate, ±2.5: 2.90 (p = 0.107) against 13.26 in R |
| F2 | `sp.rdrandinf(fuzzy=)` | Permutation p-value of the Wald ratio | Book's Snippet 3.7: 0.863 against a 2SLS p-value of 0.038 |
| F3 | `sp.rdrandinf` / `sp.rdrbounds` | `wl` / `wr` were offsets from the cutoff | The placebo-cutoff example (`c=1`) raised |
| F4 | `sp.rdrandinf(kernel=)` | Accepted and ignored | Triangular: 10.358 in R, 9.167 before |
| F5 | `sp.rdwinselect` | Not the window-selection procedure: wrong default sequence, no binomial test, no covariate name, p = 1 without covariates | Senate: windows of 5 to 100 points against 0.53 to 1.42 |
| F6 | `sp.rdmc(cutoff_var=)` | Conventional interval and p-value | Third cutoff: p = 0.033 against the robust 0.112 |
| F7 | `sp.from_stata` | `rdrandinf`, `rdwinselect`, `rdmc` unknown; `rddensity` `bino_*` dropped | Each returned "unknown / unsupported Stata command" |

After the fixes every deterministic number agrees with the reference to
1e-11 or better: observed statistics and large-sample p-values of
`rdrandinf` for each kernel, order, evaluation point and null value;
2SLS; window counts and binomial p-values; `rdmc` per cutoff, weighted
and pooled. `sp.rdwinselect` reproduces the book's Snippet 2.5 (ten
windows, counts, binomial tests, recommended window [-0.7652, 0.7652]).

## Where StatsPAI now departs from the reference

**D1. Randomization p-values with `p > 0`.** `rdlocrand` re-randomizes
labels with scores fixed. On a covariate with no discontinuity its
60-seed mean p-value is 0.0009 where its own large-sample p-value is
0.224. A size simulation (no effect, n = 40, `p = 1`) puts that test at
37% rejection for a nominal 5% (400 replications); permuting outcomes
against (score, assignment) pairs gives 4.7% (3,000 replications; the
large-sample test gives 6.9%). StatsPAI does the latter. Tests:
`test_rdrandinf_p1_holds_its_level`,
`test_label_permutation_overrejects_with_p1`. Worth reporting upstream.

**D2. First window of `rdwinselect`.** `rdlocrand` 2.0 starts with 9
observations on the left for `obsmin = 10`; its help text and the book's
printed output (10 on the left) give the rule StatsPAI implements. The
default sequences therefore differ from current R and agree with the
book. With `wmin=` and `wstep=` they agree with R.

**D3. Smaller items.** With `p > 0` and a non-zero `nulltau`, R's
large-sample p-value ignores `nulltau`; StatsPAI's tests the stated
null. R's `rdwinselect(approx = TRUE, p = 1)` stops with an error on the
Senate data, so that path has no reference rows. R's randomization
p-value for the Kolmogorov-Smirnov statistic on a binary covariate was
1.000 on all 60 seeds where the difference in means gives 0.147; not
investigated further.

## Left open

- **`rdwinselect(wmasspoints)`**: windows at successive mass points. The
  reference errored on the synthetic check, so its rule could not be
  established. `sp.from_stata` reports the option as untranslated.
- **`rdmcplot`**: no StatsPAI equivalent; `RDMultiResult.plot()` is a
  forest plot of the estimates, not the binned scatter.
- **`sp.rdms` takes one boundary point per call** and has no `xnorm`
  pooled row. The book's three-point call is three calls; the pooled row
  is `sp.rdrobust` on the perpendicular distance.
- **`sp.rdrobust`'s automatic density check uses McCrary's test.** On the
  academic-probation data (heaped GPA) it warns with p = 6.5e-11 where
  `sp.rddensity` gives 0.082, the number the book reports. On simulated
  discrete scores without heaping the two tests have similar size (4.5%
  and 6.5%), so the default was not changed on this evidence. Switching
  the automatic check to `sp.rddensity` is the candidate fix.
- **`sp.rddensity` binomial table, default first window.** StatsPAI and
  Stata use the smallest window holding 20 observations in total; R uses
  20 on each side. Both references are by the same authors. Kept as is.
- **`interfci`** (interval under interference) and `rdwinselect`'s
  `hotelling` statistic are not implemented.
- **`bitesti`** has no translation.
