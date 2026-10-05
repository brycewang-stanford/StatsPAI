# Huntington-Klein, *The Effect* (2nd ed.): review

Source: `causalbook` (the figure scripts and data behind the book) and
`causaldata` (its datasets for R, Stata and Python). The repository holds
no chapter code, so the code blocks were read from the online edition at
theeffectbook.net: 150 blocks in R, Stata and Python across chapters 4 and
13 to 21 (regression, matching, simulation, fixed effects, event studies,
difference-in-differences, instrumental variables, regression
discontinuity, partial identification). The partial identification chapter
is new in the second edition.

Method. Every Stata block except the simulation chapter was run in Stata 18
MP with the packages the book names (`reghdfe`, `ivreghdfe`, `rdrobust`,
`rddensity`, `cem`, `ebalance`, `psmatch2`, `sensemakr`, `rbounds`,
`rmhbounds`). The logs were replayed through `sp.stata` with
`scripts/stata_log_replay.py`, which compares every printed coefficient,
standard error and summary statistic. Where the numbers disagreed the first
step was to find the first point of divergence, as §5.1 of `CLAUDE.md`
asks. Estimators the translator did not reach were called directly and
compared with Stata's stored results.

| | reproduced | differ | not run |
| --- | --- | --- | --- |
| first replay | 148 | 37 | 56 |
| after this pass | 836 | 5 | 6 |

The five that still differ are explained below. Twenty-two more numbers come
from `rnormal()` and cannot be compared draw for draw.

## What was wrong in StatsPAI

1. **Integer overflow inside a formula.** `pd.read_stata` returns a Stata
   `int` as `int16` and a `byte` as `int8`. numpy keeps arithmetic in that
   type, so `I(x**2)` wrapped around for values above 181 (or 11) and the
   regression was fitted on garbage with no warning. Chapter 13 squares
   `numberoflocations`, an `int` up to 646: the coefficient was -0.0171
   against Stata's -0.0844. This is not a translation problem. It hit
   `sp.regress`, `sp.feols`, `sp.fepois`, `sp.feglm`, `sp.glm`, `sp.logit`,
   `sp.ivreg`, `sp.panel`, the covariate formulas of the DiD family and the
   spatial models, for anyone who loads a `.dta` or a parquet file. Narrow
   integer columns are now widened to `int64` before the formula is
   evaluated. Columns that hold only -1, 0 and 1 are left alone, so the
   indicator columns of a large frame are not copied.
2. **`a##b##c` lost its two-way terms.** The translator wrote the main
   effects and the three-way product only. Chapter 20 runs
   `i.participation##c.income_centered##c.income_centered`; the fitted
   model was missing `participation#c.income_centered` and the square. Now
   the full factorial, and a repeated continuous variable is a power also
   next to a factor.
3. **A variable without a prefix inside `#` or `##` was read as
   continuous.** Stata reads it as a factor. For a 0/1 variable the two
   coincide, for three levels they are different models. `reg y d##c.x`
   with a three-level `d` now reproduces Stata (101 numbers on a synthetic
   check, among them `d##g##c.x`).
4. **`collapse (mean)` kept double precision.** Stata stores the mean,
   median or sd of a byte, an int or a float as a float, and of a long or a
   double as a double (checked for twelve statistics and five types). The
   regression on the collapsed data in chapter 13 differed from Stata in
   the sixth digit. The session now stores what Stata stores.
5. **`sp.ebalance(moments=2)` with a binary covariate returned uniform
   weights.** The square of a 0/1 variable is the variable, so the dual was
   singular, the solver gave up and the "entropy balanced" estimate was the
   raw difference in means (-0.032 against 0.091), behind a warning. A
   moment that repeats another one is now left out and listed in
   `model_info['redundant_moments']`.
6. **`sp.sensemakr` skipped a benchmark it did not know.** A misspelt name
   in `benchmark=` gave a table without that row and no message. It is now
   an error.
7. **`sp.match(method='cem')` did not follow `cem`.** Sturges' rule was
   read as a number of bins and a bin was closed on the left. In the `cem`
   packages for R and Stata the rule counts cut points (one bin fewer) and
   a bin is closed on the right. With integer covariates the difference in
   closure moves units between cells. Both now follow `cem`; on the NSW
   data the default reproduces Stata's matched sample and estimate.

## What was missing

- `sp.ebalance(moments=[2, 2, 1])`: one order per covariate, which is
  `targets(2 2 1)`. `dof_adjust=True` matches the sample variance and
  skewness as Stata's `ebalance` does. The default stays the raw moments
  of R `ebal` and `WeightIt`, which the existing parity test pins.
- `sp.match(method='cem', n_bins=...)` per covariate: a list, or a dict
  from covariate to a number of bins or to cut edges. The book's R code
  sets quantile cuts for income and two bins for the party indicator.
- `sp.sensemakr(benchmark={'label': [columns]})`: several controls
  benchmarked as one group (`gbenchmark()` with `gname()` in Stata,
  a named list in R).
- `sp.rdrobust` names a string covariate. It used to stop with
  `could not convert string to float: 'New Jersey'`.
- In `sp.stata`:
  - commands `xi`, `ebalance`, `cem`, `sensemakr`, and
    `table g, statistic(mean x)`;
  - `encode x, g(new)`, `date(s, "YMD")` and the other orderings,
    `egen cut(x), group(#)`, `margins, at(x = 100)` with blanks,
    `collapse (sd) s = x` with blanks;
  - `reghdfe y d##ib3.t` and squares in `reghdfe`;
  - `_b[1.d#3.t]` and `_se[...]` in factor notation, with a zero for the
    base level, so the book's loop that collects event-study coefficients
    runs;
  - abbreviated variable names in varlists and expressions
    (`statessquire`, `_treat`), refused when ambiguous;
  - a weight that is missing or zero drops the observation;
  - `regress ... [iw = w]` when the weights sum to the number of
    observations, which is how `cem_weights` are built;
  - `i.` covariates in `psmatch2` and `teffects`, and the `_<outcome>`
    variable `psmatch2` leaves;
  - `program def` in any abbreviation, `return scalar` of a local macro,
    and a program that calls another one;
  - `bstat, stat()` on the replications `simulate` leaves in memory, so
    the bootstrap program of chapter 14 runs to its last line (observed
    IPWRA estimate 0.0769429, Stata's value);
  - `rbounds diff, gamma()`.
- `sp.rosenbaum_bounds(estimates=True)`: the range of the Hodges-Lehmann
  estimate and the outer ends of its confidence interval for each Gamma,
  the four right-hand columns of `rbounds`. The book's example of it is
  degenerate (a binary outcome), so the references were taken on the NSW
  data. Significance levels agree with Stata and with Rosenbaum's
  `DOS2::senWilcox` (eight digits against R), the Hodges-Lehmann bounds
  with Stata to its six printed digits, the confidence bounds with
  `senWilcox` to 1e-3. Stata's confidence bounds use the variance formula
  for untied data and differ from both in the third digit (482.046
  against 483.865 at Gamma 1.5 on data with tied differences); that is a
  difference between the two references, and StatsPAI follows the
  author's.

## Agreement found, nothing to change

| chapter | commands | numbers |
| --- | --- | --- |
| 13 | `regress` with `vce(hc3)`, `vce(cluster)`, `[aw]`; `logit` and `margins, dydx()`; `newey, lag(3)` | 113 |
| 14 | `teffects nnmatch` (Mahalanobis, AI robust SE), `teffects ipw`, `teffects ipwra`, `logit` and trimming by hand | 71 |
| 16 | within regression by hand, `regress i.country`, `reghdfe` one and two ways, clustered | 308 |
| 17 | market-model abnormal returns with `date()` | 9 |
| 18 | `reghdfe` TWFE, placebo treatments, event-study interactions | 22 |
| 19 | `ivregress 2sls` with village indicators, clustered | 106 |
| 20 | RD by OLS with a triangular kernel, `rdrobust`, `ivreghdfe` and `ivregress` fuzzy RD, `rddensity` | 158 |
| 21 | `sensemakr` with group benchmarks, `psmatch2` | 26 |

Direct calls, against Stata's stored results:

- `sp.sensemakr`: robustness values, partial R2 and every bound to 1e-12.
- `sp.ebalance(dof_adjust=True)` against `ebalance, tolerance(1e-10)`:
  estimate and robust SE to 1e-10 for `targets(1)`, `(2)`, `(3)` and
  `(2 2 1)`.
- `sp.match(method='nnmatch')` against `teffects nnmatch`: -0.0160021 and
  0.1164078, all digits.
- `sp.psmatch2` with five factor covariates: ATT 0.199466226, and the
  moments of `_weight`, `_pscore` and the matched outcome.

## Numbers that still differ, and why

- **`reg responded leg_black [pw = wt]` after `ebalance`** (four numbers).
  Stata's `ebalance` stops at `tolerance(.015)`, before the moments are
  balanced. Its weights give 0.0907039; run with `tolerance(1e-10)` it
  gives 0.0909340, which is what StatsPAI returns. The default run is the
  one that is off.
- **Adjusted R2 of `regress [iweight = cem_weights]`.** Stata prints
  0.0008 and 5,125 observations for weights that sum to 5,126. The
  coefficients, standard errors, F and R2 agree.

## Things the book does that Stata lets through

1. **`cem ... leg_democrat(#2)` does not match on party.** The book says
   `(#2)` means two bins. In `cem`, `#k` is the number of cut points, so
   `#2` is one interval and Democrats are pooled with Republicans. The
   imbalance table in the log shows it (L1 of `leg_democrat` 0.081 after
   matching). `(#3)` or `(#0)` would split the variable. The R and Python
   blocks do split it, so the three languages estimate different things.
2. **`rdrobust ..., covs(nonwhite bpl qob)`** stops with r(2000): `bpl`
   is a string.
3. **`causaldata mortgates.dta`** in the fuzzy RD block is a typo for
   `mortgages`.
4. **`local N = _N` after `simulate`** is the number of replications, not
   the sample size, so `bstat, n()` is told 2,000.
5. **`statessquire`** in the matching regression is an abbreviation of
   `statessquireindex`. Stata accepts it, a reader copying the name into R
   or Python gets an error.
6. **`rbounds diff`** on a binary outcome prints point estimates of
   -3.8e-07 and 0.5. The book says the output is not meaningful.

## Where the book is behind current practice

The request for this pass noted that the book may be dated. These are the
places, with what StatsPAI offers.

- **Difference-in-differences** (ch. 18). The code is two-way fixed
  effects and an event study by interaction. The text describes the
  staggered-adoption problem and names Callaway and Sant'Anna and
  Wooldridge without code. `sp.callaway_santanna`, `sp.did_imputation`,
  `sp.etwfe`, `sp.sun_abraham`, `sp.bacon_decomposition` and
  `sp.honest_did` cover it.
- **Matching in Python** (ch. 14). The book uses `causalinference`, which
  is no longer maintained, and writes CEM by hand in forty lines of pandas.
  `sp.match(method='nnmatch' | 'cem')`, `sp.ebalance` and
  `sp.g_computation` replace both, with Stata's standard errors.
- **Standard errors of weighted estimators.** The book bootstraps IPW by
  hand because `reg [pw = ipw]` ignores the estimated propensity score.
  `sp.g_computation(se_method='analytic')` and `sp.ebalance` (default
  `vce='mestimation'`) carry that uncertainty analytically.
- **Regression discontinuity** (ch. 20). The polynomial-by-OLS blocks
  precede `rdrobust`; the book then moves to `rdrobust`, which is the
  current standard and is what `sp.rdrobust` reproduces.
- **Partial identification** (ch. 21). `sensemakr` and Rosenbaum bounds.
  StatsPAI also has `sp.oster_bounds`, `sp.lee_bounds`, `sp.manski_bounds`
  and `sp.honest_did`.

## Left out on purpose

- `table` beyond one row variable and one `statistic()`. That form is
  translated to `sp.sumstats(by=)` and reproduces the book's two tables
  (3.583539 and 5.343309 by college attendance). `sp.sumstats` heads the
  two levels of a 0/1 `by` variable "Control" and "Treated", its
  documented default for a balance table. A college indicator is not a
  treatment, so the translations of `table` and `tabstat, by()` pass
  `by_labels={}` and the groups are headed by their values, as in Stata.
  The default of the direct call is unchanged.
- `lasso linear ..., sel(cv)` and `lassocoef`. Cross-validation folds come
  from Stata's random-number generator. `sp.lasso_select` and `sp.rlasso`
  exist for direct use.
- `vce(bootstrap, reps())` on `regress`. Refused as before, because the
  draws cannot be Stata's.
- `rmhbounds` (Mantel-Haenszel bounds for a binary outcome). StatsPAI has
  no estimator behind it; adding one is a new public function and was not
  taken on in a pass over a textbook whose own example of it is
  degenerate.
- `generate()` of `teffects nnmatch` (observation numbers of the matches).
  The estimate runs without it.
- A cross of two factors without their main effects (`a#b`). Stata fits
  one indicator per cell. The formula gives the same fit under other
  coefficients, so `_b[1.a#1.b]` would not be Stata's number. It is refused
  with a pointer to `a##b` or `egen group()`. Before this pass it was
  translated to the product of two columns, a different model.
- A user guide that walks through the book. The mapping from its commands
  to `sp.*` calls is short enough to live in this note.

## Files

- Fixes: `core/utils.py` (`_widen_narrow_integers`), `fixest/wrapper.py`,
  `did/_core.py`, `spatial/models/_base.py`, `matching/ebalance.py`,
  `matching/match.py`, `diagnostics/sensemakr.py`, `rd/rdrobust.py`.
- Translator: `agent/_translation/_stata.py` (factor notation, abbreviated
  names, `reghdfe` products), `_stata_expr.py` (`date()`,
  `coefficient_key`), `_stata_datastep.py` (`collapse` storage),
  `_stata_egen.py` (`cut, group()`), `_stata_run.py` (weights, factor
  covariates, importance weights), `_stata_programs.py`, `_stata_flow.py`,
  and two new modules: `_stata_balance.py` (`xi`, `ebalance`, `cem`) and
  `_stata_sensitivity.py` (`sensemakr`).
- Tests: `tests/reference_parity/test_the_effect_stata_parity.py` (Stata
  references on `sp.datasets.nsw_dw()`, generated by
  `_fixtures/_generate_the_effect_Stata.do`),
  `tests/test_narrow_integer_formula.py`,
  `tests/test_stata_the_effect_constructs.py`.
- `scripts/stata_log_replay.py`: labels defined by `encode`, labels with
  blanks, rows of crossed factors, `teffects` contrasts printed by label
  and `sensemakr`. Before these, 400 numbers of chapters 16, 19 and 20
  were reported as having no counterpart and went unchecked.
