# Kohler, Kreuter and Haensch, *Data Analysis Using Stata* (4th ed.): what it showed about StatsPAI

Done 2026-10-05. Worktree `.claude/worktrees/kohler-kreuter`, branch
`wt/kohler-kreuter`.

## What was examined

Ulrich Kohler, Frauke Kreuter and Anna-Carolina Haensch, *Data Analysis
Using Stata*, Fourth Edition (Stata Press, 2026). The files of the book
(`net from http://www.stata-press.com/data/kkh4/`, package `daus4`) are in
`改进建议-收集整理/18-Kohler-Kreuter-DataAnalysisUsingStata-4e/`, which is
gitignored. They hold 20 datasets (a teaching sample of the German
Socio-Economic Panel, the Titanic, Anscombe's quartet and others) and 50
do-files. Most of the do-files build the datasets from the raw panel, which
is not distributed, or draw figures. The commands the book teaches are typed
into its running text.

So the book was used as a syllabus, as with the Xu and Lan text. It is the
standard first course in Stata, and it differs from the econometrics texts
audited so far. Two thirds of it is about what happens before the first
regression.

| Chapter | Content | Commands the audit ran |
| --- | --- | --- |
| 1 | a first session | `summarize`, `by`, `mvdecode`, `tabulate`, `generate`, `label`, `regress` |
| 3 | the grammar of Stata | variable lists, `if` / `in`, 30 functions, `foreach`, `forvalues`, the four kinds of weights |
| 5 | creating and changing variables | `recode`, `egen`, `_n` / `_N`, string functions, dates and clock times, storage types, extended missing values |
| 7 | describing and comparing distributions | `tabulate` with tests, `table`, `tabstat`, `xtile`, `centile`, `ranksum`, `kwallis`, `ksmirnov`, `oneway`, `robvar` |
| 8 | statistical inference | `mean`, `proportion`, `total`, `ratio`, `svyset` and `svy:`, `ci`, `ttest`, `teffects`, a regression tree by hand |
| 9 | linear regression | `predict` diagnostics, `dfbeta`, `estat`, `margins` over grids, `xtreg`, `ivregress` |
| 10 | categorical outcomes | `logit`, `logistic`, `estat gof`, `lroc`, Pregibon's diagnostics, `mlogit`, `ologit` |
| 11 | reading and combining data | `merge`, `append`, frames, `collapse`, `reshape`, `contract` |
| 12 | programming | macros, extended macro functions, `program`, `syntax`, `tokenize` |

Chapters 2, 4 and 6 (organising do-files, Python inside Stata, graphs) have
nothing to compute.

## Method

One do-file per chapter was written for this audit
(`tests/external_parity/kohler_kreuter_syllabus/`), on the book's own
datasets. They were run in Stata 18 with a text log per chapter, and each
log was replayed through one `sp.stata` session by
`scripts/stata_log_replay.py`, which compares every number Stata printed.

The replay script had compared estimation tables and a few named
statistics. Most of what this book prints is neither. It gained a
comparator that takes every number of a printed table and looks for it in
what StatsPAI returned, and one for the text `display` prints.

| | Before | After |
| --- | --- | --- |
| Numbers compared and reproduced | 1,619 | 5,442 |
| Numbers that differ | 101 | 25 |
| Commands declined | 375 | 45 |
| Commands that ran without the printed number | 54 | 34 |

The first column is the state at the start, after a crash of the replay
script on value labels with an empty text was fixed (the panel labels its
missing codes that way). Most of its 101 differences came after a declined
`mvdecode`, not from wrong estimates.

What was added or corrected is pinned without the book's data. A synthetic
dataset of 420 rows (`tests/reference_parity/_fixtures/kk_syllabus.csv`)
went through a reference do-file in Stata 18, which wrote 196 numbers at
full precision. `tests/reference_parity/test_kohler_kreuter_stata_parity.py`
runs the same do-file through `sp.stata`. 153 numbers are held to 1e-9.
41 are held to 1e-5 and are of two kinds, both named in the test: 29
values behind a maximum-likelihood fit, where Stata stops iterating
earlier, and 12 values Stata passes through a single-precision variable
(`kwallis`, `estat ovtest`, `dfbeta`, `collapse`, `statsby`). The two
exact confidence limits of the odds ratio are held to 1e-4 (see below).

## What was wrong

These changed a number or a result that StatsPAI returned without saying so.

1. **A regression on a column named like a Python keyword failed.** The
   Titanic data call the passenger class `class`. `sp.regress("survived ~
   C(class)")` and `sp.logit` raised a syntax error, because the formula
   engine evaluates each term as Python. `return` and `yield` are as common
   in finance data. Fixed in `core/utils.py::create_design_matrices`.
2. **The reference category of a factor could be an empty cell.** The
   levels of `C(g)` were read from every row where `g` is observed, also
   the rows dropped because the outcome is missing. A level that occurs
   only there became the base, the remaining indicators added up to the
   constant, and one of them was dropped with a note. The fit was the same;
   the coefficients were those of another coding than Stata's and R's. The
   book's rent regression has this shape (owners pay no rent).
3. **`sp.estat(result, "reset", rhs=True)` could return a negative F.**
   The powers of a regressor in the thousands reach 1e14, and the
   unrestricted fit lost them to rounding. Stata's F(8, 2148) = 59.24 came
   out as -214.
4. **`sp.hausman` left factor levels out of the test.** The fixed-effects
   and the random-effects result spell a level differently, so only the
   coefficients with the same name were compared: chi2(2) = 100.26 where
   Stata reports chi2(6) = 236.26.
5. **`regress y x, noconstant` with a constant `x` was refused** as
   collinear with an intercept the model does not have. This is the
   first-difference regression of the book's panel chapter (`D.age` is 1 in
   every row). The pass over Ding's *Linear Model and Extensions* found the
   same on the same day and its fix is the one on main.
6. **`table g, statistic(frequency) statistic(mean y)` returned one
   statistic.** A repeated option survived only once in the parsed
   command, so the translation carried the last `statistic()` and dropped
   the others silently.
7. **`collapse ..., by(g)` dropped the rows with a missing `g`.** Stata
   keeps them as a group.
8. **`r()` after `summarize x [aweight = w]` held the unweighted moments**
   once weights were carried over at all.

## What was added

Public functions, each with Stata 18 reference numbers:

- `sp.ranksum`, `sp.signrank`, `sp.kwallis`, `sp.spearman`, `sp.ktau`,
  `sp.ksmirnov`, `sp.median_test`, `sp.robvar`, `sp.oneway`. StatsPAI had no
  rank test and no one-way analysis of variance.
- `sp.influence_measures` (Cook's distance, DFFITS, DFBETAs, COVRATIO,
  Welsch distance, studentized residuals), `sp.logit_influence` (Pregibon's
  diagnostics by covariate pattern) and `sp.logit_gof` (Pearson and
  Hosmer-Lemeshow).
- `sp.sumstats(weights=, total=)`: analytic weights with Stata's weighted
  percentiles, and the Total panel of `tabstat, by()`.
- Goodman and Kruskal's gamma and Kendall's tau-b with their asymptotic
  standard errors in the association tests of a two-way table.

In `sp.stata`:

- **Expressions.** About 100 more function names: strings (byte semantics for
  `strlen` / `substr`, characters for the `u` functions), `recode` /
  `irecode` / `autocode`, clock times, the remaining calendar parts,
  discrete distributions, `float()`, `real()`, `string()`, `c()` values.
  String variables can be generated and replaced.
- **Extended missing values.** The data keep `.a` to `.z` as NaN like `.`.
  The session now tracks which variables may hold them. On those, the
  comparisons that tell the kinds apart (`x == .`, `x != .`, `x > .`) are
  declined with an explanation, and so are `by`, `tabulate, missing` and
  `collapse, by()`, where Stata makes a group of each kind. `missing(x)`,
  `x < .` and `x >= .` are the same for every kind and run. Before, `if x
  != .` kept rows on the book's data that Stata drops.
- **Data management.** `recode`, `mvdecode` with a list of rules,
  `mvencode`, `xtile`, `pctile`, `tostring`, `destring`, `order`, `expand`,
  `contract`, `separate`, `split`, `sample`, `splitsample`, `mark`,
  `markout`, `levelsof`, `ds`, `unab`, `assert`, `confirm`, `isid`,
  `joinby`, `duplicates report` / `tag`, `rename (a b) (c d)`, `collapse`
  with weights, `reshape, string`, `append, generate()`.
- **Descriptive commands.** `tabulate` with `row` / `column` / `cell` /
  `expected`, weights, `gamma`, `taub`, `summarize()`; `tab1`, `tab2`;
  `table` with several statistics and two dimensions; `centile`, `ameans`,
  `cii`, `correlate, covariance`, `misstable summarize`.
- **Estimation of means.** `mean`, `proportion`, `total` and `ratio` with
  `over()`, the four kinds of weights and `vce(cluster)`. Without weights
  and clusters Stata treats the groups of `over()` as separate simple
  random samples; with probability weights or clusters it linearizes. Both
  are reproduced to the last printed digit.
- **Survey data.** `svyset` (strata, sampling units, `fpc()`, the four
  `singleunit()` rules, poststratification) and `svy:` before the four
  commands above, `tabulate` (with the Rao-Scott second-order correction of
  Pearson's statistic) and `regress` / `logit` / `poisson`; `estat
  effects`. With Stata's default, a stratum with a single sampling unit
  leaves the standard errors missing, here too. `sp.svymean` already
  agreed with Stata under all three other rules; only the translation was
  missing.
- **After estimation.** `predict` with the influence statistics of a
  linear and of a logistic fit, `dfbeta`, `estat gof`, `lroc`, `lrtest`,
  `linktest`, `estat vce`, `estat summarize`, `logistic`, `ologit`,
  `cloglog`; `margins` for predictive margins, factor levels, grids in
  `at()`, several `at()` options, `atmeans` and `over()`; `e(sample)`.
- **The `by` prefix** runs any command group by group and filters rows
  (`by id: keep if _n == 1`).
- **Programming.** Extended macro functions (`word count`, `variable
  label`, `label`, `subinstr`, the macro-list functions, `display`),
  `` `i++' ``, `syntax` with its option types and minimal abbreviations,
  `marksample`, `tokenize`, `gettoken`, `macro shift`, `return local`,
  `exit`, `if exp command` on one line, `_rc` after `capture`, and
  `display` with text, formats and several items.

## Second round: the open decisions

The first round left eight items open. The owner asked for them to be
decided. What was decided and done:

| Item | Decision |
| --- | --- |
| `test` / `lincom` after `mean`, `proportion`, `total`, `ratio` | Done. Any expression linear in the estimates, with the command's covariance matrix and degrees of freedom. A nonlinear one is refused. |
| `nestreg: regress y (a) (b) ...` | Done: each block's F test given the blocks before it, on the rows complete on all blocks. |
| `anova` | `anova y g` is the one-way layout and runs `sp.oneway`. With more terms it is refused with the advice to fit `regress y i.a i.b` and use `testparm`: a second analysis-of-variance engine beside the regression one would have to agree with it in every unbalanced case. |
| `cc`, `cs` | Done: odds ratio with exact limits, risk difference and risk ratio, attributable and prevented fractions. `tabodds` stays declined. |
| `predict p1 ... pk` after `mlogit` / `ologit` / `oprobit` | Done: the probability of each outcome. |
| `statsby` | Done for named results with `by()` and `clear`. |
| `margins` after `mlogit` / `ologit`, tests across equations | Declined. `sp.margins` has no prediction scale for these models; that is a change to the function, not to the translation. |
| `mi` | Declined. Imputations are random draws, so no number could be checked against Stata, and Stata's `mi` data styles (`wide`, `mlong`, `flong`) have no counterpart in one DataFrame. `sp.mice` and `sp.mi_estimate` do the work in Python. |
| `infile`, `infix`, `import` | Declined, on purpose. `sp.stata` runs on data it is given and never reads a file; that is what keeps a session free of side effects. |

One finding came with it. **Stata's exact limits for the odds ratio are
not exact to the printed digits.** `cc` solves for them by iteration and
stops early: at its lower and upper limit the tail probability of the
conditional distribution is 0.024993, not 0.025. StatsPAI solves the same
two equations to 1e-13; the limits differ from Stata's in the fifth digit.

## What differs and stays

- **Stata's `logit` with probability weights did not converge on the
  book's data.** `logit owner age_c hhinc_k east [pweight = xweights]`
  ends with "convergence not achieved" (r(430)) in Stata 18 and prints the
  estimates of its last iteration. StatsPAI's coefficients are the ones
  Stata's own `glm, family(binomial) link(logit)` returns for the same
  weights, to seven digits, and have the higher pseudolikelihood. The
  syllabus do-file runs both commands.
- **`regress y x, noconstant` with a constant `x`.** Stata reports the
  uncentered R-squared of a model without a constant (0.845). StatsPAI
  sees that the model holds a constant and reports the centered one (0),
  as statsmodels does.
- **`levelsof` on a `float` variable** lists 16.1 as `16.10000038146973`,
  which is no longer the stored value, so a loop over the levels with
  `if x == `level'` skips that level. StatsPAI writes the same 16 digits
  and skips the same level. The book's section 5.7 is about this.
- **`strlen()` on text that is not UTF-8.** One dataset of the book stores
  names in Latin-1. Stata counts one byte for an umlaut there; the text
  pandas decodes has two.
- **`swilk`** on 2,160 observations differs in the sixth digit of V. The
  approximation is defined for 4 to 2,000 observations.
- **Storage types.** A numeric variable with missing values arrives as
  float64, so `: type` says `double` where Stata says `long`.

## Declined, and why

45 commands of the logs are declined (41 distinct ones). The opt-in test
lists each with its reason. The groups:

- extended missing values where the kinds matter (6 commands);
- `mi` (see the decisions above);
- `margins, pwcompare`; `margins` and tests across equations after
  `mlogit` / `ologit`; `tabodds`;
- `import`, `infile`, `infix`: `sp.stata` does not read files;
- `merge, update`, `xi, noomit`, `teffects ra, pomeans`,
  `tebalance summarize` after `teffects ipw`, `misstable patterns`,
  `tabstat, by() missing`, `egen rank(), unique`.

## Open items

| Item | Why it is open |
| --- | --- |
| model F test of `svy: regress` | `sp.svyglm` does not expose the coefficient covariance |
| `margins` after `mlogit` / `ologit` | `sp.margins` has no prediction scale for these models |
| `margins, pwcompare`; `tabodds` | not implemented |
| storage types of numeric variables | the frame does not keep `byte` / `int` / `long` once a value is missing |

## How to rerun

See `tests/external_parity/kohler_kreuter_syllabus/README.md`. The run
folder used here is
`改进建议-收集整理/18-Kohler-Kreuter-DataAnalysisUsingStata-4e/run`.
