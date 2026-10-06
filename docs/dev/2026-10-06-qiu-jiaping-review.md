# Qiu Jiaping, *Practical Econometric Methods for Causal Inference*: what its Stata code showed about StatsPAI

Done 2026-10-06. Worktree `.claude/worktrees/qiu-jiaping`, branch
`wt/qiu-jiaping`.

## What was examined

The material that accompanies 邱嘉平《因果推断实用计量方法》(上海财经大学出版社).
It is one transcribed text file of Stata code for chapters 3 to 11, sixteen
`.dta` files, the twelve lecture decks and a scan of the book. The files sit
in `改进建议-收集整理/29-邱嘉平/`, which is gitignored. Neither the code nor
the data are redistributed.

The book is a methods course for empirical finance and economics. It is
older than the current practice in two places, and it was read as a probe
and not as a specification. Chapter 6 teaches propensity score matching with
Becker and Ichino's `pscore` and `attnd` (2002) next to `psmatch2` and
`teffects psmatch`. Chapter 9 stops at the two-period, two-group design.

| Chapter | Content | Commands |
| --- | --- | --- |
| 3 | regression in practice | `summarize`, `tabstat`, `correlate`, `regress`, `test`, `esttab` |
| 4 | standard errors | simulated heteroskedastic and AR(1) errors, GLS by hand, `newey`, `loneway`, `regress, cluster()` |
| 5 | treatment effects (STAR) | `regress` with school dummies, clustering |
| 6 | matching | `pscore`, `attnd`, `psmatch2`, `bootstrap: psmatch2`, `teffects psmatch`, `pstest` |
| 7 | matching against regression | saturated regressions |
| 8 | panel data | `xtsum`, within and first-difference by hand, `xtreg, fe` |
| 9 | difference in differences | 2x2 by means and by regression, LSDV, dynamic effects, pre-trend terms |
| 10 | instrumental variables | `ivregress 2sls, first`, `estat endogenous / firststage / overid` |
| 11 | selection | probit and the inverse Mills ratio by hand, `heckman, twostep`, `etregress` by ML and two-step |
| 12 | regression discontinuity | `rdplot`, `DCdensity`, `rddensity`, global polynomial, `rdrobust`, `rdbwselect` |

Two things about the material itself.

- The text file was transcribed from the book and has slips that stop Stata
  (`gen err` followed by `replace e`, `1.g1schid` for `i.g1schid`,
  `normalden(2)` for `normalden(z)`). `tests/external_parity/qiu_jiaping_prepare.py`
  lists each repair with its reason.
- Chapter 12 uses union election data (Campello, Gao, Qiu and Zhang 2018)
  that were not released, and the text file has no code for it. The
  commands were read off the scanned pages and applied to the two Lee (2008)
  House election files that ship in their place.

## Method

The same as for the other textbooks. The code was run once in Stata 18 with
a log per chapter. Each log was replayed through one `sp.stata` session by
`scripts/stata_log_replay.py`, which compares every number Stata printed
with what StatsPAI returns, to the precision Stata printed it. Each gap was
then traced to its first point of divergence on the same rows.

| | Before | After |
| --- | --- | --- |
| Numbers compared and reproduced | 1,561 | 2,007 |
| Numbers different | 6 | 1, documented below |
| Commands `sp.stata` refused | 26 | 0 |
| Numbers with no counterpart | 2 | 2, documented below |
| Numbers on simulated data (not comparable) | 40 | 41 |

Eleven kinds of command ran at the start without any of their numbers being
compared (`pscore`, `attnd`, `pstest`, `heckman`, `etregress`, `vif`,
`rdplot`, `rdbwselect`, `DCdensity`, `loneway`, `esttab`). The replay script
now has a comparator for each but `esttab`, which is a formatted copy of
models compared where they were fitted.

Run it with

```bash
python tests/external_parity/qiu_jiaping_prepare.py <folder>   # then run _master.do in Stata
STATSPAI_QIU_DIR=<folder>/run pytest tests/external_parity/test_qiu_jiaping_logs.py
```

Every fix is also pinned on data that can be redistributed.
`tests/reference_parity/test_qiu_jiaping_methods_stata.py` holds Stata 18
output on `sp.datasets.nsw_lalonde()`. `tests/test_stata_qiu_jiaping_syntax.py`
holds what Stata lists after the data steps.

## What was wrong, and what changed

### Wrong numbers

| # | Finding | Status |
| --- | --- | --- |
| 1 | The logit behind `sp.match`, `sp.psmatch2` and the translation of `teffects psmatch` solved its Newton step from a singular Hessian when a covariate was redundant. Next to a column of order 1e9 the error reached the scores. On chapter 6 the ATT was 1562.29 where `attnd`, `psmatch2` and `teffects psmatch` all give 1627.36 | fixed |
| 2 | `sp.match(ties='all')` missed controls at exactly the same distance. The cut-off was squared with the C library's `pow` and the distances with numpy, and the two differ in the last place for about one number in 600 | fixed |
| 3 | The score the matching runs on was clipped to [1e-6, 1 - 1e-6], so every control below 1e-6 was at the same distance from a treated unit | not clipped |
| 4 | `sp.stata`: `replace x = a * x[_n-1] + ...` and `replace x = a * L.x + ...` were evaluated on the whole column at once. Stata replaces row after row, so the line builds a recursion. The result here was missing from the third row on, with no error | fixed |
| 5 | `sp.stata`: a lag column made for `L.x` was reused after `x` had changed | recomputed each time it is named |
| 6 | `sp.stata`: `clear all` was skipped as a display setting and left the data in memory | clears data, stored estimates, scalars and programs |
| 7 | `sp.from_stata("reg y x, cluster (id)")` dropped the clustering and reported `id` as an untranslated option (`sp.stata` refused the line) | a blank before the parenthesis is read as Stata reads it |

Item 1. The textbook's specification lists `un74` and `un75`, which are the
same column in the file it ships. Stata prints `note: un75 omitted because
of collinearity`. `numpy.linalg.solve` returns a step from the singular
Hessian without complaint, and the error of that step lies along the
redundant direction. With columns of similar size it does not move the
fitted score. With `re74^2` in the model it does. The scores were off by
3e-4 and the estimate by 4%. The fit now runs on the standardised design and
leaves out a covariate that is a combination of earlier ones
(`matching/_binary_fit.py`). The case is reproduced on the bundled Lalonde
data with a repeated dummy, where the old estimate was 605.80 and Stata
gives 468.10.

Item 2 was found while pinning item 1. Two controls with the same
covariates have the same score to the last bit. `_extend_with_ties`
compared `d**2 / scale` with `cutoff**2 / scale`, the first a numpy product
and the second Python's `pow`. On the polynomial specification one treated
unit lost one of its two tied controls and the ATET was 1148.68 against
Stata's 1170.95.

Item 4 is the one most likely to have gone unnoticed elsewhere. The idiom
is how a simulated AR(1) series, a running total and a carried-forward
value are written in Stata.

```stata
gen x = 0 in 1
replace x = 0.4 * l.x + rnormal(0, 5) in 2/200
bysort id (t): replace v = v[_n-1] if missing(v)
replace z = z / z[1]
```

The last line is a known trap. Row 1 becomes 1 first, and every later row
is then divided by 1. Stata leaves `z` unchanged from row 2 on, and so does
`sp.stata` now. A reference from row `i` to row `j` of the variable being
replaced sees the new value when `j < i` and the old one otherwise. Random
draws in the expression are drawn once.

### Refused or missing

| # | Finding | Status |
| --- | --- | --- |
| 8 | `pscore` (Becker and Ichino): the score, its blocks and the balancing test | `sp.pscore` |
| 9 | `attnd` | translated to `sp.psmatch2(..., ties=True)`, same estimate and analytic standard error |
| 10 | `psmatch2 d, pscore(ps)` and `attnd y d, pscore(ps)`: matching on a score already in the data | `pscore=` on `sp.match` and `sp.psmatch2` |
| 11 | `attnd, comsup`: controls outside the range of the treated scores are set aside | `common_support='treated'` |
| 12 | `attnd, bootstrap` | `se='bootstrap'` together with `ties=True` |
| 13 | `pstest` after `psmatch2` in `sp.stata` | reads `_treated`, `_weight`, `_support` from the data, as Stata does |
| 14 | `r(att)`, `r(seatt)` after `psmatch2`, so that `bootstrap r(att): psmatch2 ...` runs | stored |
| 15 | `loneway` | `sp.loneway` |
| 16 | `DCdensity` (McCrary) | translated to `sp.mccrary_test` |
| 17 | `heckman, twostep` did not return the selection equation or the Wald test of the outcome equation | `model_info['selection_equation']`, `wald_chi2` |
| 18 | `estat firststage` returned the F alone | adds its p-value, degrees of freedom and the partial R-squared |
| 19 | `vif` (the name before `estat vif`), `d` / `des` / `desc` | run, skipped |
| 20 | A Stata log with comments in Chinese could not be replayed. The script read it as Latin-1 | UTF-8 first |

Items 8 to 12. Becker and Ichino's commands are from 2002 and their
standard errors ignore the estimation of the score. They are still what a
large share of applied papers in Chinese journals cite for "the balancing
property is satisfied". `sp.pscore` reproduces the block search and the
tests of the Stata routine: number of blocks, the unbalanced covariates and
the count of treated and controls in every block agree on the three
specifications of the chapter and on three more on the Lalonde data.
`attnd` turned out to be `psmatch2, ties` under another name. Both keep
every control tied at the smallest distance and share the analytic standard
error to nine digits. The one difference is documented below.

## Documented differences

1. **Model F of a clustered regression on fourth-order polynomials** (chapter
   12, the global polynomial with `cluster(yearel)`). Stata prints
   42621.11. The statistic computed in 60-digit arithmetic from the same
   rows is 42620.8909, and StatsPAI returns 42620.8899. There are 15
   clusters and nine restrictions, and the covariance matrix of the slopes
   has a condition number of 5e7. Coefficients and standard errors agree
   to every printed digit.
2. **F of a regression on the constant alone** (chapter 4, `reg score`).
   Stata prints `F(0, 29) = 0.00` with a missing p-value. StatsPAI reports
   the statistic as missing.
3. **`attnd` with a treated unit exactly half way between two controls.**
   `attnd` picks the control below or the one above at random.
   `sp.psmatch2(ties=True)` keeps both, as `psmatch2, ties` does. The
   textbook's data and the Lalonde data have no such case.
4. **Standard errors of the logit inside `pscore`.** The command fits its
   model under `version 8`. With observations that are completely
   determined its standard errors differ from those of `logit` in Stata 18
   in the sixth digit. StatsPAI returns the Stata 18 ones. The same holds
   for `probit` in general. Stata stops when its scaled gradient is below
   1e-5 and three independent fits here agree with each other to machine
   precision.
5. **The first block of `pscore, comsup`.** When the balancing property
   holds Stata prints the lower bound of the first block as the smallest
   treated score, and as 0 when it does not. `sp.pscore` reports the bound
   of the block search (0) in both cases and the region separately in
   `support_range`.

## What the book does not cover

Listed for the reader guide (`docs/guides/qiu_jiaping.md`), which gives the
call for each. Standard errors that account for the estimated score
(Abadie and Imbens 2016), overlap and trimming, doubly robust estimators,
staggered adoption and event studies with heterogeneous effects,
sensitivity of the parallel-trends assumption, weak-instrument-robust
inference, bias-corrected robust inference in regression discontinuity.

## Left open

1. **A recursion longer than 20,000 rows is refused.** The ordered
   `replace` is computed by repeated passes over the column, one per link
   of the chain, which is quadratic in the length of the chain. A chain of
   5,000 rows takes two seconds. A single pass for expressions that are
   linear in the lagged value would remove the limit.
2. **Stock and Yogo critical values in `estat firststage`.** Stata prints
   them. StatsPAI has the effective F and its critical values
   (`sp.effective_f_test`), which is the current recommendation.
3. **The summary of `sp.heckman` does not print the selection equation.** It
   is in `model_info`.
4. **`attk`, `attr`, `atts`**, the kernel, radius and stratification
   companions of `attnd`. Not in this book. `sp.psmatch2(method='kernel' |
   'radius')` and `sp.match(method='stratify')` are the estimators.
5. **`esttab` in `sp.stata` prints the constant of `regress` as `Intercept`
   and that of `etregress` as `_cons`**, on two rows of one table.
