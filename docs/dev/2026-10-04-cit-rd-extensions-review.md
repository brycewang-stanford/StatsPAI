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
| F8 | `sp.rdrobust` automatic density check | McCrary's binned test raised false alarms on a heaped score; now `sp.rddensity` | Academic-probation data: p = 6.5e-11 against 0.082; 43% rejection at 60 placebo cutoffs against 5% |

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
`test_label_permutation_overrejects_with_p1`. Run on `rdlocrand` itself
(400 replications, `reps = 200`): 4.75% at `p = 0`, 34.75% at `p = 1`,
with its own large-sample test at 6%. Worth reporting upstream.

**D2. Default windows of `rdwinselect`: a regression in the reference,
not a divergence.** `rdlocrand` 2.0 starts with 9 observations on the
left for `obsmin = 10`. Four CRAN releases were installed side by side
and given the same checks:

| | 0.9 | 1.0 | 1.1 | 2.0 |
| --- | --- | --- | --- | --- |
| `p = 1` randomization test, rejection of a true null at 5% | 42% | 42% | 42% | 36% |
| default first window holds `obsmin` on each side | yes | yes | no | no |
| `wmasspoints`: k-th support point on each side | yes | yes | no | no |
| KS randomization p-value on a binary variable | ok | ok | ok | stuck at 1 |
| `rdwinselect(approx, p = 1)` | error | error | error | error |
| large-sample p-value honours `nulltau` when `p = 1` | no | no | no | no |

`sp.rdwinselect` matches release 1.0 to 1e-9 in windows, counts,
binomial tests, balance p-values and the covariate named, for the default
sequence, `wobs`, `obsmin` and mass-point windows
(`tests/reference_parity/test_rdlocrand_v1_parity.py`). The mass-point
regression was reported by a user on the read-only CRAN mirror
(`cran/rdlocrand` issue 1, 2025-08-12) and closed there without reaching
the maintainers; `rdpackages/rdlocrand` has no issues at all.

**D3. Smaller items.** With `p > 0` and a non-zero `nulltau`, R's
large-sample p-value ignores `nulltau`; StatsPAI's tests the stated
null. R's `rdwinselect(approx = TRUE, p = 1)` stops with an error on the
Senate data, so that path has no reference rows. R's randomization
p-value for the Kolmogorov-Smirnov statistic on a binary covariate was
1.000 on all 60 seeds, where the observed statistic is 0.215 and its own
exact p-value is 0.150. That is new in 2.0: in a wider window where 1.0
averages 0.358 over 40 seeds, StatsPAI averages 0.358 and 2.0 gives 1.
Checking against 1.0 also showed that StatsPAI's own
large-sample KS p-value was wrong with ties (0.965 against 0.356): it is
now the exact conditional p-value, as in R.

## Left open

Closed after the first pass, all on main: `sp.rdms` with several boundary
points and `xnorm=`; `sp.rdmcplot`; `sp.rdwinselect(wmasspoints=True)`;
`sp.rdwinselect(statistic='hotelling')`; `sp.rdrandinf(interfci=)`; `sp.rdms` with one score and cumulative
cutoffs, and its translation with `rdmcplot`'s; `sp.bitest` with the `bitest` / `bitesti`
translations; `rdms` in `sp.stata`; and `sp.rdrobust`'s density check
(F8).

Still open:

- **`sp.rddensity` binomial table, default first window.** StatsPAI and
  Stata use the smallest window holding 20 observations in total; R uses
  20 on each side. Both references are by the same authors. Kept as is.
- **`rdmcplot, genvars`** is refused by `sp.stata`: it creates variables
  that the do-file then plots by hand. `fig.rdmcplot_data` holds the same
  numbers.
- **Upstream.** A report for the `rdlocrand` maintainers is drafted in
  the materials folder (`run/upstream_report_draft.md`) and not sent. It
  carries the four-release table and reproductions (`run/versions.R`).
