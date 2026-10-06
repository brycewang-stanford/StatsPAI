# Das, *Causal Inference in R* (Packt): a chapter-by-chapter rerun

*2026-10-06. Worktree `wt/das-causal-r`.*

## What was done

The book's repository has 32 R scripts in 13 chapters. Each was rerun in
R 4.5.2 with the packages the book loads, the data written to CSV, and the
same computation repeated with StatsPAI on those bytes. Stata 18 served as a
second reference where it has the command (`esize`, `power`, `estat phtest`).
Nearly all the data are simulated inside the scripts with R's generator, so
the comparison is on R's draws, not on regenerated ones.

The book is an introduction. Much of the code is `lm`, `glm` and plots, and
several scripts contain mistakes of their own (listed at the end, since a
user following the book will bring them along). The value of the pass was
less in the book's methods than in running a wide, shallow sweep over
functions that the econometrics textbooks do not touch: causal discovery,
proportional-hazards diagnostics, effect sizes, power.

## Results by chapter

| Ch. | Book computes | StatsPAI | Outcome |
| --- | --- | --- | --- |
| 3 | ECLS: logit propensity score, `matchit(method="nearest")`, Welch t, `lm` | `sp.logit`, `sp.match`, `sp.ttest`, `sp.regress` | Logit to 1e-8. `sp.match` default gave an arbitrary ATT on tied scores. **Warning added.** |
| 3 | Paired t, `lm` with factors, correlation matrix | `sp.ttest(paired=True)`, `sp.regress`, `sp.pwcorr` | Equal to all printed digits. |
| 4 | `dagitty` implied independencies, `localTests` | `sp.dag(...).implied_independencies()`, `.test_implications()` | Same seven implications. A string node crashes the test with a raw `ValueError`. Left to the parallel pass (see below). |
| 5 | Collider and d-separation demos: `cor.test`, partial correlation, `ggm::dSep`, nested `anova` | `DAG.d_separated`, `sp.cor_test` (new), `sp.lrtest`, `sp.test` | d-separation agrees. No function returned r with its test and interval. **`sp.cor_test` added.** |
| 6 | PSM, `cobalt::bal.tab`, Rosenbaum bounds, PS quartile stratification, Hajek IPW | `sp.match`, `sp.ps_balance`, `sp.rosenbaum_bounds`, `sp.match(method='stratify')`, `sp.ipw` | Rosenbaum bounds equal `rbounds::psens` to four digits at every Gamma. Stratified ATE equals `MatchIt` subclass to 1e-14. |
| 7 | `lm`, logit, Poisson, `glm.nb`, `coxph`, `cox.zph`, `lmer`, `gam` | `sp.regress`, `sp.logit`, `sp.poisson`, `sp.nbreg`, `sp.cox`, `sp.mixed` | Coefficients equal throughout. **`ph_test` was not the Grambsch-Therneau test. Overdispersion test was wrong.** Both fixed. GAM: see "Not done", item 3. |
| 8 | `t.test`, `shapiro.test`, `var.test`, `effsize::cohen.d`, `pwr.t.test`, `p.adjust` | `sp.ttest`, `sp.swilk`, `sp.sdtest`, `sp.esize` (new), `sp.power_ttest` (new), `sp.adjust_pvalues` | Tests equal. No effect size, no exact t-test power, three `p.adjust` methods missing. **All added.** |
| 9 | AIPW by hand, HC1, `tmle` | `sp.aipw`, `sp.regress(robust='hc1')`, `sp.tmle` | AIPW (no cross-fitting) to 2e-9, HC1 to 1e-8, Hajek IPW to 2e-7, TMLE with a GLM library to 3e-4 (6e-6 relative; the two propensity fits are not the same optimizer). |
| 10 | `AER::ivreg` with diagnostics, `gmm`, Wald test | `sp.ivreg`, `sp.iv_diag`, `sp.test` | Coefficients, SEs, weak-instrument F, Wu-Hausman, Sargan all equal. The `y ~ x \| z` formula was refused. **Now read.** |
| 11 | `mediation::mediate`, `lavaan::sem` | `sp.mediate` | Linear product of coefficients equal. The multiple-mediator SEM now runs in `sp.path_analysis` and equals lavaan. Logit outcome model not available. |
| 12 | Complete-case and mean-imputed t tests, E-value from an odds ratio | `sp.ttest`, `sp.evalue` | Equal, and equal to the `EValue` package for the common-outcome and rare-outcome conversions. |
| 13 | `matchit` with factor covariates, `fixest::feols` | `sp.match`, `sp.feols` | `feols` coefficients and SEs equal to the six digits compared. |
| 14 | `grf::causal_forest` | `sp.causal_forest` | ATE 2.1005 (0.0796) against grf's 2.094 to 2.114 (0.079 to 0.080) over five seeds. Screen only, as the forest is random. |
| 15 | `pcalg::pc`, `bnlearn` (`hc`, `pc.stable`, `mmhc`, `si.hiton.pc`, `boot.strength`), `causaleffect` | `sp.pc_algorithm`, `sp.ges`, `sp.identify` | **PC drops an edge pcalg keeps on both data sets.** Handed over (see below). ID algorithm agrees on the front-door, back-door and bow graphs. |

## Fixes

### 1. `CoxResult.ph_test()`
The method computed the Spearman correlation between event time and the raw
Schoenfeld residual and returned `n * rho^2`. That is not a test anyone
publishes. On the book's two maintenance data sets it gave 0.135 and
0.00003 where `cox.zph` gives 1.030 and 0.163.

It is now the score test of adding `theta_j * g(t)` to coefficient `j`,
evaluated at the fitted model, which is what `survival::cox.zph` has
computed since version 3.0. The score and information contributions are
accumulated per stratum and event time under the fit's own tie rule.
Against `cox.zph` (survival 3.8-3) on 32 configurations (two tie rules, with
and without strata, tied and untied times, four time functions) the largest
relative difference in chi-square is 1.3e-6, on a statistic of 0.08.

Stata's `estat phtest` reports the 1994 approximation, which uses the
average information matrix. `method='approx'` reproduces it for `identity`
(Stata's default), `km`, `rank` and `log`, to the digits Stata prints and to
1e-8 on the one statistic read from `r()`. Two conventions had to be pinned:
Stata ranks the time among failures, `cox.zph` among all observations; and
Stata's default time function is the identity while `cox.zph`'s is the
Kaplan-Meier transform.

### 2. The Poisson overdispersion test

`sp.poisson` regressed `(y - mu)^2 - y` on a constant and `mu` and tested
the slope. With the constant in the regression the slope picks up only how
the excess variance changes with the mean. The book's accident counts have
a Pearson dispersion of 12.1 and the test returned t = 0.77, p = 0.44.

The Cameron-Trivedi test against `Var = mu + alpha mu^2` is the no-intercept
regression of `((y - mu)^2 - y) / mu` on `mu`. On the same counts it is
3.9747 with alpha = 0.546, the values of `AER::dispersiontest(trafo = 2)`.
Rejection rate under a true Poisson, 400 draws: 5.75%. The p-value is
two-sided from t(n - 1); AER prints the one-sided normal one.

### 3. `sp.match` and tied propensity scores

The ECLS propensity score has three covariates and takes 91 distinct values
over 9,289 children. `ties='first'` (the default, the convention of
`psmatch2` without its `ties` option) keeps the
lowest-index control at each distance, so every treated child with a given
score gets the same partner:

| call | ATT | SE |
| --- | --- | --- |
| default (`ties='first'`, with replacement) | -0.502 | 0.246 |
| `n_matches=5` | 0.002 | 0.145 |
| `ties='all'` | -0.125 | 0.027 |
| `replace=False` | -0.085 | 0.034 |
| R `matchit(method="nearest")` (without replacement) | -0.071 | 0.035 |

The first two rows are functions of the row order. Nothing was changed in
the numbers. The fit now warns with the count of affected units, the number
of tied candidates left out and, for the ATT, how many distinct controls
the estimate rests on (52 here), and records the counts in `model_info`.

**Decided 2026-10-06 (Bryce delegated the call): `ties='all'` is now the
default with replacement.** What settled it was the Lalonde sample of the
original-data parity suite. Under `ties='first'` the ATT was 1967.94 and
became 2012.47 when the rows were shuffled; with every tie kept it is
1968.80 in either order, and that is Stata's `teffects psmatch` to the last
digit (1968.799715855857), standard error included (1126.3212, Abadie-Imbens
2016). The earlier default matched no reference: R's `MatchIt` returns
2006.86 on the same data, with its own arbitrary tie order. The NSW-DW
Track A row has no ties and does not move. `sp.psmatch2` passes
`ties='first'` explicitly, because `psmatch2` does.

## Found here, fixed elsewhere: `sp.pc_algorithm`, `sp.fci`, `DAG.test_implications`

On the chapter 15 data `pcalg::pc` returns ten edges and `sp.pc_algorithm`
nine; on the case-study data eleven against ten. The cause was traced:

- Conditioning sets for a pair `(X, Y)` are drawn from
  `adj(X) ∪ adj(Y)`. PC tests subsets of `adj(X) \ {Y}` and of
  `adj(Y) \ {X}` separately. The union adds sets that mix the two
  neighbourhoods, each another chance to accept independence and delete a
  true edge.
- Adjacency sets are updated inside a level, so the result depends on
  column order, although the docstring cites PC-stable.
- Two colliders that ask for opposite directions on one edge zero both
  entries, and the edge vanishes from the CPDAG while staying in the
  skeleton.
- Meek's third rule is missing. Non-numeric columns are dropped in silence.

On 48 simulated linear-Gaussian data sets (5 to 12 nodes, 150 or 600 rows)
against `pcalg::pc` 2.7-12, the skeleton differed in 17, the CPDAG in 31,
27 of 420 reference edges were missing and 106 skeleton edges were absent
from the returned graph.

A rewrite was written and reproduced `pcalg::pc` edge for edge on all 48
and on a harder 24-case set, under both `skel.method` settings. Before it
was pushed it turned out that the pass on Ness, *Causal AI* (worktree
`ness-causal-ai`) had rewritten the same function, with discrete
conditional-independence tests and a fix to the stopping rule of `sp.fci`
as well, and had reworked `DAG.test_implications`. Two rewrites of one file
cannot both land, and theirs covers more. **This pass therefore leaves
`causal_discovery/pc.py`, `causal_discovery/fci.py` and `dag/graph.py` as
they are on main.** What it leaves for that line:

- **A harder reference set.** 24 data sets with 120 or 400 rows, where the
  tests err and colliders conflict. Run against the Ness implementation as
  it stood in its worktree on 2026-10-06 01:20: skeleton equal to
  `pcalg::pc` in 24 of 24, CPDAG equal in 10 of 24, and every one of the 14
  differences is in a case where that implementation reports
  `orientation_conflicts`. So the two agree wherever the algorithm
  determines the answer and differ in the convention for a conflict. pcalg
  lets the later collider (in variable order) overwrite the earlier one.
  **Closed 2026-10-06** after the Ness pass landed (`159d0cb5`): the
  default stays "first collider keeps the edge", which that pass wrote
  into the conventions, and `sp.pc_algorithm(collider_conflict='last')`
  reproduces pcalg in 24 of 24. The fixture and its test are on main
  (`tests/reference_parity/test_pc_pcalg_parity.py`).
- **Where the first rewrite is.** Local branch `das-pc-rewrite-backup` in
  this repository (not pushed) holds it, the fixture
  (`tests/reference_parity/_fixtures/pc_pcalg_data.csv`, `pc_pcalg_R.json`,
  the two generator scripts) and
  `tests/reference_parity/test_pc_pcalg_parity.py`. The test file runs
  against any implementation that returns `cpdag` and `skeleton`.
- **`sp.fci` is not full FCI**, with or without the stopping-rule fix: the
  skeleton is the PC skeleton, the Possible-D-SEP pass and Zhang's rules
  R5 to R10 are absent, and the docstring describes a `refine_dsep` option
  that does not exist. On the 24 data sets the skeleton equals
  `pcalg::rfci` in 22 and `pcalg::fci` in 10, every difference from the
  latter an extra edge.
- **`DAG.test_implications`** raises `could not convert string to float` on
  a string node (chapter 4's `interests` column) and reports no interval
  for the partial correlation. A Fisher-z interval,
  `tanh(atanh(r) -/+ z / sqrt(n - |Z| - 3))`, is one line.
- **`sp.causal_discovery(method='fci')`** is refused though `sp.fci` exists.

## Added

- `sp.power_ttest`. Noncentral-t power for one-sample, two-sample and
  paired t tests; solves for the missing one of `n`, `delta`, `power`.
  `n` is the total over both groups, as in Stata and the rest of
  `sp.power_*`; `params['n_exact']` holds the fractional solution that
  `pwr.t.test` prints per group. Ten Stata 18 `power` calls reproduced, the
  continuous ones to 1e-10.
- `sp.esize`. Stata `esize twosample, all` reproduced to 1e-12 including
  the `unequal` intervals. Glass's Delta keeps its own group's degrees of
  freedom under `unequal`, as in Stata's output.
- `sp.cor_test`. Pearson and partial correlation with t test and Fisher
  interval; equal to `cor.test` and `ggm::pcor.test`.
- `sp.adjust_pvalues(method='hochberg' | 'hommel' | 'by' | 'sidak')`,
  equal to `p.adjust` to 2e-15.
- The `AER::ivreg` two-part formula in `sp.iv` / `sp.ivreg`.

## Documented differences, not changed

- **Welch degrees of freedom.** `sp.ttest(welch=True)` is Stata's `welch`
  option (Welch 1947). R's "Welch Two Sample t-test" is Satterthwaite's
  approximation, `unequal=True`. The module docstring says so; an R user
  who reaches for `welch=True` gets 2689.4 where R prints 2687.5.
- **Standardized mean differences.** `cobalt::bal.tab` standardizes by the
  treated group's SD for the ATT and reports raw differences in proportion
  for binary covariates. `sp.ps_balance` uses the pooled SD for both. The
  Barrett pass documented this.
- **`glm.nb` standard errors.** `MASS::glm.nb` treats theta as known when it
  computes coefficient SEs. `sp.nbreg` uses the joint information, as Stata
  does. Coefficients and log-likelihood agree to 1e-9; SEs differ in the
  second digit on 100 observations.
- **`lmer` at the boundary.** With a true random-intercept variance of zero
  `lmer` reports exactly 0 and `sp.mixed` reports 14.1 on a residual
  variance of 8.8e7. Fixed effects, SEs and the REML criterion agree to six
  digits.
- **`sp.power_rct`** is a normal approximation by design and its reference
  test says so. `sp.power_ttest` is the exact counterpart.

## Not done

1. **Structural equation models.** Closed 2026-10-06 for observed
   variables: `sp.path_analysis` takes the lavaan model string of chapter
   11 as written (after dropping `missing = "ML"`, which the complete data
   do not need) and returns lavaan's estimates and standard errors for the
   paths and for the three defined effects. Parity fixture:
   `tests/reference_parity/test_path_analysis_lavaan_parity.py`. Closed
   the same day for latent variables (`=~`), mean structures (`y ~ 1`,
   `meanstructure=True`), `std_lv=True` and the chapter's `growth()` call
   (`growth=True`): one engine now fits all of them, and nine further
   models equal lavaan under ML and MLM, factor scores included
   (`test_sem_latent_lavaan_parity.py`). Writing the fixture showed that
   the first version's claim to reproduce `sem()` was too broad:
   `sem()` lets the disturbances of terminal outcomes covary without being
   asked, and `sp.path_analysis` does not. That default stays (it is
   Stata's, and it fits the model the user wrote); `auto_cov_y=True`
   gives lavaan's, and the docstring now says so. Still open:
   full-information ML for missing data, multiple groups, categorical
   indicators.
2. **Mediation with a binary outcome.** Closed 2026-10-06:
   `sp.mediate(inference='robust', outcome_model='logit')`, with
   `treat_values=(0, 1)` for the book's count treatment. On the book's data
   the indirect effect is 4.8e-5 (SE 1.8e-4) against 5.8e-5 from R's
   simulation-based `mediate`, both zero to any precision that matters.
   The reference is Stata 18 `mediate`, matched on ten model pairs.
3. **Generalized additive models.** `sp.gam` landed on main the same night
   from another line (`c41e6bfc`). On the chapter's wage data it selects
   the smoothing parameter by REML and agrees with
   `mgcv::gam(method = "REML")`; the book's call uses mgcv's default GCV.
4. **Discrete Bayesian networks.** Closed 2026-10-06: `sp.hill_climb`
   (categorical and continuous data, bnlearn's BIC to 1e-12) and
   `sp.bootstrap_edges` (`boot.strength`). On the chapter's factor data the
   three arcs and the score (-28729.85) are bnlearn's. `mmhc` and
   `si.hiton.pc` have no counterpart; `sp.bayes_net` fits and queries a
   given graph.
5. **`sp.fci`.** Closed 2026-10-06: Possible-D-SEP and rules R1 to R10,
   equal to `pcalg::fci` on 36 data sets.
6. **Forest tuning.** Closed 2026-10-06: `sp.tune_causal_forest`. Not a
   parity item: grf smooths the trial losses, this takes the best trial
   with a noise margin. Its measured effect is in the docstring.
7. **Proportion mediated** now has a percentile interval in the bootstrap
   path.

Still open after this: full-information likelihood, multiple groups and
categorical indicators in `sp.path_analysis`, and the hybrid and local
discovery algorithms of bnlearn.

## Mistakes in the book's code

A reader who translates the scripts line by line will hit these. None is a
StatsPAI issue, but each is a reason not to treat the book's printed output
as a reference.

- **Chapter 9, "doubly robust" estimate.** The formula multiplies the IPW
  weight into a difference of observed and predicted outcomes in a way that
  is not the AIPW estimator. It returns 93.0 for a true effect of 50; AIPW
  on the same fits returns 46.5.
- **Chapter 10, first stage.** The `ivreg` formula makes `renovated`
  endogenous with `investor_interest` as the instrument, which by
  construction has nothing to do with `renovated`: first-stage F = 0.066.
  The script then checks relevance by regressing `sqft` on the instrument
  and reports F = 8,695. `sp.ivreg` warns on the actual first stage.
- **Chapter 12, E-value of the confidence limit.** Computed as 1.43 from a
  limit of 0.91. The interval contains 1, so the E-value is 1, as
  `sp.evalue` and the `EValue` package return.
- **Chapter 14, confidence intervals.** Built as prediction ± variance
  rather than ± 1.96 standard errors; coverage 9.8% where the correct
  interval covers 90%. The "RMSE" used for tuning compares the outcome with
  the treatment effect.
- **Chapter 6, `senmv(data)`** is called on the unmatched data frame.
- **Chapter 4** passes a character column to `lavCor`.

## Rerun recipe

R reference scripts and the CSVs they write are not committed (the data are
R's draws of the book's simulations). To rebuild: for each chapter run the
book's data-generating lines under the same `set.seed`, `write.csv` the
frame, and call the StatsPAI function named in the table above. The
committed fixture and tests cover the fixes:

```bash
Rscript tests/reference_parity/_fixtures/_generate_cox_ph_test.R
pytest tests/reference_parity/test_cox_ph_test_parity.py \
       tests/reference_parity/test_effect_size_power_ttest_parity.py \
       tests/test_poisson_overdispersion_test.py \
       tests/test_iv_two_part_formula.py
```

R packages beyond CRAN defaults: `pcalg` (for the handed-over comparison)
needs Bioconductor's `graph` and `RBGL`.
