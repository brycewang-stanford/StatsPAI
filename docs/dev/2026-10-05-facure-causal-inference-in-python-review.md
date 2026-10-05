# Facure, *Causal Inference in Python* (2023): review

Source: the book's notebooks at
<https://github.com/matheusfacure/causal-inference-in-python-code> (MIT
licence), 11 chapter notebooks and 26 data files. They cover A/B tests
(chapter 2), graphs (3), regression (4), propensity scores (5), effect
heterogeneity and its evaluation (6), meta-learners (7),
difference-in-differences (8), synthetic control (9), geo and switchback
experiments (10), and instruments and discontinuities (11).

Method. The notebooks keep their printed outputs, so the answer key is the
book itself. Each analysis was rerun with the `sp.*` function a user would
reach for, on the book's data, and compared with the stored number. Where
the two disagreed the first step was to find the first point of divergence
(§5.1 of `CLAUDE.md`). The book writes most estimators by hand with pandas,
statsmodels, scikit-learn and cvxpy. That makes it a good probe for a
different kind of problem than the Stata and R textbooks found. Its data
look like industry data (daily dates, string categories, a `treated` flag
next to a `post` flag, several treated units), and several of the findings
below are about what StatsPAI did with inputs of that shape.

The book is from 2023. Some of its choices are dated or simplified, and
those were not adopted (see "Where StatsPAI differs on purpose").

## What was wrong in StatsPAI

1. **`sp.did` returned an estimate of the wrong sign on a block design
   with a `post` column.** `sp.did(df, y=, treat='treated', time='post',
   id='city')` on daily data went to Callaway-Sant'Anna, whose reshape
   kept the first row of each (unit, period) cell and dropped the rest
   without a word. Chapter 8's marketing data: -0.659 before, 0.6917 after
   (the book's number, the difference of the four cell means).
   `sp.callaway_santanna` now refuses repeated (unit, period) rows on a
   panel. `sp.did(method='auto')` recognises a 0/1 group flag with a 0/1
   time and repeated rows as the 2×2 design and clusters on `id`.
2. **The DAG recommender could not find an instrument.**
   `sp.dag_recommend_estimator` tested exogeneity as "Z is d-separated from
   Y given the exposure" in the original graph. Conditioning on the
   exposure opens the collider `Z -> T <- U`, so every instrument was
   rejected in exactly the confounded graphs an instrument is for. The
   canonical graph of chapter 11 came back as "not identifiable". The test
   now runs in the graph with the arrows into the exposure removed, and a
   conditional instrument is found with its conditioning set.
3. **The same recommender proposed the front door when it does not hold.**
   A heuristic accepted any mediator set unless a `<->` latent touched it.
   A direct edge `T -> Y` next to `T -> M -> Y` was accepted, and latents
   declared with `latent=[...]` were not looked at. It now calls
   `DAG.frontdoor_sets`, which checks the three conditions path by path.
4. **Three of the four `sp_call` strings the recommender emits did not
   run.** `sp.ipw(..., outcome=)`, `sp.front_door(df, exposure=, outcome=,
   mediators=[...])` and `sp.dag.identify(...)` raise `TypeError` or
   `AttributeError`; `sp.iv('y ~ [d ~ z]')` was rejected by the IV parser.
   The strings now use the real signatures and a test evaluates each one.
5. **A `category` column with numeric levels was read as a number.**
   `covariates=['tenure', 'role']` with `role` of dtype `category` and
   levels 1 to 5 entered every selection-on-observables estimator as one
   linear term. Chapter 5: `sp.aipw` gave 0.2757 instead of 0.2712. A text
   column failed in numpy with `could not convert string to float`, and
   `C(role)` with `columns not found`.
6. **`sp.cate_eval` scored a ranking against a dose.** A continuous
   treatment went through the AIPW formulas for a 0/1 treatment and came
   back as an AUTOC of 19,592 with a standard error and an interval.
7. **`sp.balance_table` and `sp.balance_check` returned an empty table for
   three arms.** With `treat='cross_sell_email'` (three string levels) both
   groups were empty and the output was a table of NaN and "N = 0".
   `balance_table` also skipped covariates that are not in the data.
8. **`sp.synth(inference=)` accepted anything.** The docstring listed
   `'bootstrap'` and `'jackknife'`; on the classic method they were dropped
   and placebo inference was returned.
9. **`sigma=` had no effect in the power calculators.** `power_rct`,
   `power_did`, `power_rd`, `power_iv`, `power_cluster_rct` and `power_ols`
   multiplied the effect by `sigma` and divided by a standard error
   proportional to `sigma`. Chapter 2's sample-size question
   (`effect_size=0.08, sigma=0.203`) returned 4,906 instead of 203.

## What was missing

- **Dates as the time variable of a staggered design.** `sp.event_study`
  took a `datetime64` column; `sp.callaway_santanna`, `sp.sun_abraham`,
  `sp.did_imputation` and `sp.etwfe` failed with `TypeError`, or with "no
  never-treated units" when the never-treated cohort was coded as a far
  date. They now number the observed dates and store the numbering in
  `model_info['calendar_time']`.
- **Meta-learners with a continuous treatment.** Chapter 7 fits its
  heterogeneous effect of a discount with the residual-on-residual
  regression. `sp.metalearner(learner='r')` now takes a continuous
  treatment. Given the same folds and learners its CATE is equal to
  `econml.dml.NonParamDML`'s to the last bit, and its headline is the
  partially linear coefficient of `sp.dml(model='plr')`.
- **Evaluating a ranking when the treatment is a dose.** `sp.cate_gain_curve`
  gives the effect by quantile of the prediction, the cumulative effect and
  gain curves and their area, for a randomized treatment of any type.
- **A t-test for synthetic control.** `sp.synth_ttest` and
  `sp.synth(inference='ttest')` implement the debiased K-fold estimator of
  Chernozhukov, Wüthrich and Zhu (2026), the method of the book's inference
  section. It agrees with the authors' R package `scinference` to 1e-15 on
  the estimate and the standard error.
- **Switchback experiments.** `sp.switchback_design` and `sp.switchback`
  implement the design and the Horvitz-Thompson analysis of Bojinov,
  Simchi-Levi and Zhao (2023): the optimal randomization points, the exact
  randomization test, and the conservative variance under the optimal
  design. Enumerating every assignment path of a 12-period design confirms
  unbiasedness and the paper's variance formula to machine precision.
- `sp.iv` reads the linearmodels block `[endog ~ instruments]`.
- `sp.synth` with a list of treated units names the functions that take
  one, instead of failing in numpy with a shape error.

## Reproduced without change

| chapter | the book computes | StatsPAI | agreement |
| --- | --- | --- | --- |
| 2 | difference in conversion, unequal variances | `sp.ttest(unequal=True)` | all printed digits |
| 3 | d-separation queries on four graphs | `sp.dag(...).d_separated` | all 9 queries |
| 4 | OLS with `C()`, `np.sqrt()`, `I()`, interactions; prediction on new rows | `sp.regress`, `.predict` | 1e-12 against statsmodels |
| 4 | size-weighted average of group slopes | `sp.margins` on the saturated model | 4.4904e-06, 1e-8 |
| 5 | Horvitz-Thompson IPW | `sp.ipw(normalize=False)` | 0.2659787, 1e-5 (logit tolerance) |
| 5 | 1-NN matching on the propensity score, both arms | `sp.match(method='psm', estimand='ATE')` | 0.28777443474, 1e-9 |
| 5 | doubly robust estimator | `sp.aipw(cross_fit=False)` | 0.27116, 1e-4 (logit tolerance) |
| 8 | 2×2 and two-way fixed effects ATT | `sp.did`, `sp.feols` | 0.69173595364, 1e-10 |
| 8 | cohort-by-date saturated TWFE | `sp.did_imputation`, `sp.etwfe` | 2.2597661447, 1e-8 |
| 9 | synthetic control of the treated average | `sp.geolift` | 1e-6 against a tight solver |
| 9 | debiased ATT, SE and 90% interval | `sp.synth(inference='ttest')` | 1e-6 |
| 10 | switchback Horvitz-Thompson estimates and interval | `sp.switchback` | 1e-12 |
| 11 | Wald and 2SLS with two instruments | `sp.iv` | 1e-10; SE up to `sqrt(n/(n-k))` |

## Where StatsPAI differs on purpose

- **Synthetic control weights.** The book's cvxpy solution stops at the
  solver's default accuracy (its weights contain entries of -8e-6) and gives
  an ATT of 0.003327. A constrained solver run to 1e-16 gives 0.0033467,
  which is what `sp.geolift` returns. The t-test numbers agree to 1e-6
  because the folds average the difference out.
- **Synthetic difference-in-differences.** The book's version has no ridge
  penalty on the unit weights and fits them without an intercept (0.004086).
  `sp.sdid` follows Arkhangelsky et al. and the `synthdid` package
  (0.003981), which is Track A parity evidence.
- **The cutoff of a discontinuity.** The book codes the 319 accounts with a
  balance of exactly 5,000 as below the threshold (`balance > 0`).
  `rdrobust`'s rule is `x >= c`, here and in the R and Stata packages. On
  the same two global lines that is 727.34 against 732.85.
- **2SLS standard errors.** `sp.iv` applies `n / (n - k)`, like Stata's
  `ivregress, small` and R's `ivreg`. `linearmodels` with
  `cov_type='unadjusted'` does not. The ratio is 1.0001 at n = 10,000.
- **Staggered adoption.** The book's cohort-versus-never-treated estimate
  (2.2247) compares full pre and post means. `sp.callaway_santanna` uses the
  last pre-period as the base (2.094 on the same data, noisier with 9 to
  18 treated cities per cohort). The imputation and extended TWFE estimators
  reproduce the book's saturated regression exactly.
- **The gain curve.** The book's curve takes `row + 1` units at each step
  (`index <= row`) and weights them by `row / n`. `sp.cate_gain_curve` takes
  `row` units. The area is 181.91 against the book's 181.75; the
  effect-by-quantile table is identical.
- **A default of three folds for the t-test.** `scinference` defaults to
  `K = 2`, which gives a t distribution with one degree of freedom. The
  book uses three. `n_folds=2` reproduces the package.

## Not adopted

- The continuous-treatment IPW of chapter 5 (weights from a normal density
  of the treatment given covariates, 1/f or stabilised). It rests on the
  normality of the treatment residual and its weights are heavy-tailed.
  `sp.dose_response` and `sp.dml(model='plr')` cover the same question; the
  three give -0.79, -0.80 and the book's -0.78 on the interest-rate data.
- The S-learner for a continuous treatment. The book itself shows that it
  shrinks the effect toward zero.
- The synthetic control design of chapter 10 (searching random sets of
  treated cities for the pair of synthetic controls that best tracks the
  market). `sp.synth_experimental_design` ranks candidates by a different
  criterion. This is an open item.

## Open items

- A market-representativeness criterion for
  `sp.synth_experimental_design`, as in chapter 10.
- `sp.switchback` reports a variance only under the optimal design, which
  is what the paper derives. Other regular designs get the randomization
  p-value alone.
- `sp.cate_gain_curve` has no standard errors.
- Categorical covariates are expanded in the selection-on-observables
  family, `sp.dml`, `sp.metalearner`, `sp.drdid` and
  `sp.callaway_santanna`. The forests and the remaining estimators that take
  a covariate list still need numeric columns.
- Date-typed time columns are handled in the four staggered estimators
  above. `sp.gardner_did`, `sp.stacked_did`, `sp.lp_did` and the
  de Chaisemartin-D'Haultfoeuille family still need period numbers.

## Rerun

```bash
export STATSPAI_FACURE_DIR=/path/to/causal-inference-in-python/data
pytest tests/external_parity/test_facure_causal_inference_in_python.py   # book data
pytest tests/test_facure_textbook_pass.py                                # simulated
```

The `scinference` reference numbers in `tests/test_facure_textbook_pass.py`
were produced by the R call quoted next to them.
