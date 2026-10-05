# Barrett, D'Agostino McGowan and Gerke, *Causal Inference in R*: review

Source: the Quarto source of the book (<https://www.r-causal.org>,
repository `r-causal/causal-inference-in-R`, last commit 2026-08-06). The
book is unfinished. Chapters 1 to 11 are complete or being polished, 12 to
16 are drafts, and 17 to 24 (mediation, longitudinal data, time to event,
doubly robust estimation, machine learning, instruments,
difference-in-differences) are placeholders of twenty lines each.

Method. The book has no stored outputs, so every analysis of chapters 2 and
6 to 16 was rerun in R 4.5.2 with the packages the book uses (`propensity`
0.1.0, `halfmoon` 0.2.0, `tipr` 1.0.2, `MatchIt`, `WeightIt`, `optweight`,
`lmw`, `marginaleffects`, `survey`, `sandwich`, `dagitty`) and then in
StatsPAI on the same rows. Where the two disagreed the first step was to
find the first point of divergence, as section 5.1 of `CLAUDE.md` asks, and
where two R packages disagreed with each other a third reference or the
definition decided. The book's running example is
`touringplans::seven_dwarfs_train_2018` (MIT licence, 354 days at 9 am).

The book was treated as a reference for a workflow and not as an authority
on every number. Three of its tools turned out to be approximations, and
they are listed under "Not adopted" below.

## What was wrong in StatsPAI

1. **Crump trimming solved the wrong equation.** With `g = 1 / (e (1 - e))`
   the rule of Crump, Hotz, Imbens and Mitnik (2009) keeps `g <= gamma`
   where `gamma = 2 E[g | g <= gamma]`, so the propensity cutoff satisfies
   `alpha (1 - alpha) = 1 / gamma`. `_crump_alpha` compared `alpha` itself
   with `1 / (2 E[g])` and walked a grid of 500 points. The cutoff was
   always too small. Book data: 0.0638 and 329 days kept before, 0.0716
   and 318 days after. `propensity::ps_trim(method = "adaptive")` gives
   0.07157928 and drops the same 36 days. The sample equation is piecewise
   linear and is now solved exactly, so the cutoff agrees to 1e-15.
   Affects `sp.trimming(method='crump')` and
   `sp.propensity_score(trimming='crump')`.
2. **Weighted variances in the balance tables were not on the scale of the
   unweighted ones.** `_smd` and `_variance_ratio` used `sum(w (x - m)^2) /
   sum(w)` with weights and the `n - 1` variance without, so
   `weights=np.ones(n)` did not reproduce the raw columns. The divisor is
   now `sum(w) - sum(w^2) / sum(w)` (Austin and Stuart 2015), which is
   free of the scale of the weights, equals `n - 1` for equal weights and
   is what `cobalt` and `halfmoon` use. Weighted variance ratio of the
   temperature variable: 0.7794 before, 0.7938 after, `cobalt` 0.79376367.
3. **`DAG.adjustment_sets(minimal=True)` returned only the smallest
   minimal sets.** The search stopped at the first size with a valid set.
   On `W -> X; P -> W; Q -> W; P -> Y; Q -> Y; X -> Y` it gave `{W}`;
   `dagitty` gives `{W}` and `{P, Q}`. Sets are now kept when no valid
   proper subset exists.
4. **`sp.contrast` on a frame with one level returned an empty table.** It
   now raises and points to `subset=`.

## What was missing

| chapter | the book uses | StatsPAI now |
| --- | --- | --- |
| 8, 10 | `propensity::wt_ate / wt_att / wt_atu / wt_atm / wt_ato`, `stabilize = TRUE`, `ps_trunc` | `sp.ps_weights` |
| 9, 10 | `halfmoon::ess`, `plot_ess` | `sp.ess(weights, by=)` |
| 9 | `halfmoon::check_balance(...)` energy row, `bal_energy` | `sp.energy_distance`, and in `sp.balance_diagnostics().summary_stats` |
| 9 | `check_model_auc(.weights =)` | `sp.auc(weights=)`, `sp.roc_curve(weights=)` |
| 10 | `lmw::lmw` | `sp.implied_weights` |
| 11 | `propensity::ipw` for five estimands | `sp.ipw(estimand='ATO' / 'ATM' / 'ATU', se_method='sandwich')` |
| 13, 14 | `avg_comparisons(newdata = subset)`, `comparison = "lnratioavg" / "lnoravg"` | `sp.contrast(subset=, effect=)`, `sp.margins(subset=)`, `sp.g_computation(estimand='ATC')` |
| 16 | `tipr::adjust_coef`, `tip_coef`, `adjust_coef_with_binary`, the `rr` / `or` / `hr` families | `sp.confounder_adjust`, `sp.confounder_tip` |
| 16 | `dagitty::equivalentDAGs`, `equivalenceClass`, `adjustmentSets(type = "all")` | `DAG.equivalent_dags()`, `DAG.equivalence_class()`, `adjustment_sets(minimal=False)` |

Six of these are new public functions (`ps_weights`, `ess`,
`energy_distance`, `implied_weights`, `confounder_adjust`,
`confounder_tip`); the rest are options on existing ones. The registry goes
from 1,316 to 1,322.

## Reproduced without change

Propensity scores (4e-14), ATE / ATT / ATC point estimates of `sp.ipw`
(1e-11), weighted KS statistics, raw standardized differences of continuous
covariates (`cobalt`, 1e-9), the unweighted AUC, `sp.margins` and
`sp.margins_at` after OLS and logit with `C(treat)` (risk difference and
its standard error equal `marginaleffects` to 1e-8), implied conditional
independencies of the chapter 16 DAG (five, as `dagitty`), `sp.evalue(1.5)`
= 2.366.

## Documented differences (not bugs)

- **`propensity::ipw` multiplies its variance by `n / (n - 1)`.**
  `sp.ipw(se_method='sandwich')` uses divisor `n`, as Stata `teffects ipw`
  does. With the factor applied the standard errors agree to 1e-13 for all
  five estimands.
- **Standardized differences have three conventions for the weighted
  case.** StatsPAI's default standardizes by the weighted variances.
  `cobalt` keeps the unweighted `n - 1` variances
  (`sd_denom='unweighted'` reproduces it to 1e-10). `halfmoon`, through
  the `smd` package, keeps the unweighted variances with divisor `n`, and
  uses `p (1 - p)` for binary covariates. Raw differences therefore differ
  from `halfmoon` by a factor near `sqrt(n / (n - 1))` within groups
  (0.1562 against 0.1569 on temperature).
- **The book's continuous-exposure weights.** Chapter 12 uses
  `mean(.sigma)` from `broom::augment`, the average of leave-one-out
  residual standard deviations, in both densities. `sp.ps_weights(
  exposure='continuous')` takes `sigma=` from the caller and uses the
  sample standard deviation of the exposure in the numerator. The two
  agree to 2e-4 on the book's data. `propensity::wt_ate` for a continuous
  exposure returns something else again (ratio up to 1.28 to the book's
  own formula) and was not used as a reference.
- **Bootstrap intervals.** The book uses `rsample::int_t` (studentized).
  `sp.bootstrap` offers percentile, normal and BCa. Not compared draw by
  draw.

## Not adopted from the references

- **`halfmoon`'s AUC is not the Mann-Whitney probability.** On the
  simulated fixture, which has no tied scores, `check_model_auc` gives
  0.80530 unweighted where `wilcox.test` and `pROC::auc` both give
  0.80558. StatsPAI equals the latter two to 1e-15, and its weighted AUC
  equals a brute-force weighted pair count to 1e-12. On the book's data
  the unweighted values happen to coincide and the weighted ones differ
  by 6e-5. Not yet reported upstream.
- **`propensity::ipw` does not accept `estimand = "atu"`** although
  `wt_atu` exists. `sp.ipw(estimand='ATC')` has a sandwich variance.
- **Chapter 11's bootstrap function refits the propensity model on the
  full data** (`data = seven_dwarfs_9` inside `fit_ipw`), so the bootstrap
  does not carry the uncertainty of the score. Chapter 2 does it right.
  `sp.ipw(se_method='bootstrap')` refits on each resample.

## Open items

1. **Natural splines.** The book uses `splines::ns(x, df)` in propensity
   and outcome models. `sp.regress` accepts `bs()` and `cr()` through the
   formula engine; `cr()` fails when `sp.margins_at` rebuilds the design,
   and no basis here equals `ns()` column for column. The contrast of two
   exposure levels in chapter 13 (30 against 60 minutes) can be read off
   `sp.margins_at`, but there is no single call that returns the
   difference with its standard error.
2. **`MatchIt` nearest-neighbour matching without replacement** was not
   compared pair by pair. The order in which treated units are matched
   differs between implementations and the matched sets are not unique.
3. **Stable balancing weights and energy balancing** (`optweight`,
   `WeightIt(method = "energy")`). `sp.sbw` exists and was not compared;
   there is no energy-balancing estimator.
4. **Studentized bootstrap interval** (`int_t`).
5. **`tipr`'s R-squared parameterisation** (`adjust_coef_with_r2`) is
   covered by `sp.sensemakr` and was not duplicated.
6. **Calibration plot of a propensity model**
   (`plot_model_calibration`), mirrored histograms with weights, and ECDF
   plots. `sp.overlap_plot` draws the mirrored density without weights.
7. Chapters 17 to 24 of the book are empty. Revisit when they are written.

## Rerun

```bash
# simulated fixture (committed)
Rscript tests/reference_parity/_fixtures/_generate_barrett_causal_inference_in_r.R
pytest tests/reference_parity/test_barrett_causal_inference_in_r_parity.py

# the book's data (not redistributed)
export STATSPAI_BARRETT_DIR=/some/empty/folder
Rscript tests/external_parity/barrett_causal_inference_in_r_reference.R
pytest tests/external_parity/test_barrett_causal_inference_in_r.py
```

Citations added to `paper.bib`, each checked against Crossref and a second
record (OpenAlex, or Semantic Scholar where Crossref was incomplete):
`li2013weighting`, `austin2015moving`, `szekely2013energy`,
`huling2024energy`, `chattopadhyay2023implied`, `lin1998assessing`,
`schlesselman1978assessing`, `mcgowan2022tipr`.
