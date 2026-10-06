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

## Second round: the five items left open

The first round ended with five open items. All were taken up the same day.

| # | Item | Outcome |
| --- | --- | --- |
| 21 | A recursion longer than 20,000 rows was refused | runs: one small evaluation per link |
| 22 | `sp.heckman` summary did not print the selection equation | printed; and a wrong variance was found on the way (item 23) |
| 23 | ⚠️ Two-step `heckman` with an estimate of rho outside [-1, 1] | Stata's truncation, with a warning |
| 24 | `attk`, `atts`, `attr` | `attk` and `atts` translated and equal to Stata; `attr` declined with its reason |
| 25 | `esttab` in `sp.stata` showed `Intercept` and `_cons` on two rows, and no N for `etregress` | one `_cons` row, N shown |
| 26 | Stock and Yogo critical values in `estat firststage` | returned, from Stata's own table |

Item 21. The ordered `replace` evaluated the whole column once per link of
the chain, which is quadratic. It now rewrites the expression so that the
value on a row depends on that row's columns alone. What counts rows is
written into columns once: `_n` and `_N` (within the group under `by`), a
subscript of another variable, every random draw. After one evaluation of
the column only the rows that read a row whose value has just changed are
evaluated again. A chain of 50,000 rows takes five seconds. A panel of
300,000 rows with chains of 30 takes seven. The one form left on the old
path is a running `sum()` of the variable being replaced, which reads the
whole column by construction. It is still declined beyond 20,000 rows.

Item 23. Writing the test for item 22 on the Lalonde data gave a two-step
rho of -1.34. The variance formula of the two-step estimator weights each
row by `1 - rho^2 delta_i`. With rho^2 above one those weights turn
negative and the formula is no longer a variance. StatsPAI used it as it
stood, and the standard errors were a third smaller than Stata's. Stata
sets rho to +/-1 and sigma to `|lambda|` and prints a note. The same is
done now, with an `AssumptionWarning` that names the usual cause (no
excluded variable in the selection equation). The four standard errors,
rho, sigma and the Wald statistic agree with Stata to six digits, which is
where Stata's probit stops.

Item 24. `attk` is kernel matching on the score with a Gaussian kernel and
bandwidth 0.06. `sp.psmatch2(method='kernel', kernel='normal')` gives the
same number (1157.9545 on the Lalonde data, to the eight digits `attk`
holds). `atts` is the difference of means in each block of `pscore`,
weighted by the treated in the block. `sp.match(method='stratify')` had the
formula and took its strata from quantiles of the score. It now takes them
from a column (`strata=`), and estimate and standard error equal Stata's to
13 digits. `attr` is declined. It weights each control by the number of
treated units within the radius of it, so a treated unit with many controls
nearby counts more than one with few. The radius estimator of `psmatch2`
averages the controls of each treated unit first and gives a different
number (1157.14 against 770.77 on the same score). Copying the first would
mean shipping an estimator whose weights nobody would choose on purpose.

Item 26. The values were not typed in. `estat firststage` returns them in
`r(mineigcv)`, and the matrix was read off by running the command for one
to three endogenous regressors and up to 30 excluded instruments (199
filled cells of three tables; the empty ones are empty in Stata too). They
sit in `diagnostics/_stock_yogo.py` and `sp.estat(result, 'firststage')`
attaches the row that applies. On the book's own example (chapter 10, one
instrument) the first-stage F is 13.69. It passes the rule of thumb of 10
and does not reach 16.38, the value for a 5% Wald test with true size at
most 10%. The label "Stock-Yogo rule of thumb" that the output used for the
threshold of 10 was dropped: the threshold is a rule of thumb and the
critical values are something else.

## Still open

1. **`attr`**, for the reason above.
2. **Cragg-Donald statistic with two or three endogenous regressors.** The
   critical values are in the table. `sp.estat(result, 'firststage')`
   attaches them for one endogenous regressor only, where the statistic is
   the first-stage F.
3. **A running `sum()` of the variable being replaced** beyond 20,000 rows.
