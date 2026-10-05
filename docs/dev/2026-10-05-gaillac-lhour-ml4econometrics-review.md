# Gaillac and L'Hour, *Machine Learning for Econometrics*: what its companion code showed about StatsPAI

Started 2026-10-05. Worktree `.claude/worktrees/ml4econ-textbook`, branch
`wt/ml4econ-textbook`.

## What was examined

The companion repository of Gaillac and L'Hour (2025), *Machine Learning
for Econometrics*, Oxford University Press. It sits in
`改进建议-收集整理/Gaillac-LHour-ML4Econometrics/`, which is gitignored.
Nothing from it is redistributed.

| Chapter | Material | Methods |
| --- | --- | --- |
| 2 | `ridge_regression.ipynb` | ridge, ridgeless limit |
| 4 | `leeb-potscher.ipynb` | density of the post-selection estimator |
| 5 | `double-selection.ipynb`, `lalonde.ipynb`, package `ml4econometrics` | Lasso (FISTA), BCH penalty, double selection, cross-fitted ATT |
| 6 | `CardIV.R`, `NevoIV.R`, `BonusBLPhdm.R`, `BonusAngristKrueger.R`, `SimulationsIV.R` | `hdm::rlassoIV`, `hdm::tsls`, `ivmodel` (LIML, Fuller) |
| 7 | `CrimeIV.R`, `simulations_panel_IV.R` | within IV, cluster-Lasso |
| 8 | `GRF_appli_JTPA.R`, `test_Rlearner_fig8.1.R` | `grf` causal and instrumental forests, calibration test |
| 9 | `policy_learning_jtpa.R` | `policytree` on doubly robust scores |
| 10 | `fisher_tests.ipynb`, `synthetic control application.ipynb` | Fisher tests, intervals by test inversion, `Synth`, placebo inference |
| 11 | `Nowcasting_application.R` | sparse-group Lasso MIDAS (`midasml`), HAC Lasso |
| 12 to 14 | text notebooks | tokenisers, embeddings |

The text chapters were not examined. The JTPA files of chapters 8 and 9
and the Nevo and news-attention files are not in the repository, so those
applications could not be run.

The book's code is older than the book. Several scripts do not run as
written (`CardIV.R` refers to an undefined `data0`; `lalonde.ipynb`
predicts the outcome with the treatment model and gives the Lasso to the
treatment and the logit Lasso to the outcome; `simulations_panel_IV.R`
sources a file that is not in the repository). They were read for what
they intend, and the comparisons below are against the reference packages
themselves.

## Method

For each chapter the reference packages were run on the book's data and
StatsPAI on the same bytes. R 4.x with `hdm`, `ivmodel`, `plm`, `grf`,
`policytree`, `Synth`; Stata 18 with `lassopack` and `pdslasso`. Where a
known truth was available (the simulations of chapters 5, 7 and 10) the
simulation was repeated with StatsPAI.

## What agreed

| Item | Reference | Result |
| --- | --- | --- |
| LIML and Fuller(1), Card data, 3 and 64 instruments | `ivmodel` | estimate and SE to 1e-9 and 5e-8 |
| `rlasso_iv`, Card data, all three selection modes | `hdm::rlassoIV` | estimate and SE identical to printed precision |
| `rlasso_iv`, BLP automobile data | `hdm::rlassoIV` | identical |
| Double selection, chapter 5 design (n = 200, p = 300) | the book's Table (bias 0.012, RMSE 0.186, coverage 0.942) | bias 0.017, RMSE 0.194, coverage 0.949 over 777 draws |
| `rlasso` with heteroskedastic loadings, Crime panel | `rlasso, robust` (lassopack) | penalty level, support and post-Lasso coefficients identical |

LIML and Fuller standard errors under `robust='robust'` are 0.63% larger
than `ivmodel(heteroSE = TRUE)`. The ratio is `sqrt(n / (n - k))`:
StatsPAI applies the HC1 factor, `ivmodel` does not. A convention, not a
discrepancy.

## What did not, and what was done

### 1. No cluster-Lasso (chapter 7)

Chapter 7 treats panels: the Lasso loadings must allow for dependence
within unit (Belloni, Chernozhukov, Hansen and Kozbur 2016). `hdm` has no
such option; the book sources its own `rlasso_cluster.R`. StatsPAI's
`rlasso` is a port of hdm and had none either.

Added `cluster=` to `sp.rlasso`, `sp.rlasso_effect`, `sp.rlasso_effects`
and `sp.rlasso_iv`. The loading of column `j` is
`sqrt(sum_g (sum_{i in g} x_ij e_i)^2 / n)`, `gamma` defaults to
`0.1 / log(G)`, and the final-stage variance sums the score within
cluster. No small-sample factor, which is what `pdslasso` and `ivlasso`
do.

Evidence.

- Same bytes against Stata (`tests/reference_parity/test_cluster_lasso_stata.py`,
  fixture and do-file in `tests/reference_parity/_fixtures/`). `rlasso,
  cluster()`: penalty level to 1e-12, same support, post-Lasso
  coefficients to 1e-10. `pdslasso, cluster()`, double selection and
  partialling out: estimate and variance to 1e-9. `ivlasso, cluster()`
  with selection among instruments only and among both: the same.
  lassopack reports `lambda = lambda0 * rmse` and loadings divided by
  `rmse`; the products are equal.
- One observation per cluster reproduces the heteroskedastic fit exactly.
- Known truth. Within-transformed panel, 100 units, 8 periods, 60
  controls, AR(1) regressors and errors with coefficient 0.8, true effect
  0.5, 1,167 draws. Double selection without `cluster=` keeps 6.1
  controls and covers 78.6%; with it 4.6 controls and 93.3%.

One thing to know. The loadings depend on the residuals and the residuals
on the support, so the iteration can have more than one fixed point. hdm
(first pass at half the penalty, 15 iterations) and lassopack (two
iterations by default) can stop at different ones. Of eight simulated
panels, three differed in one marginal variable. This is true without
clustering as well. The test keeps one such panel and shows that each
package's support maps to itself under StatsPAI's loadings.

### 2. `sp.fisher_exact` confidence interval (chapter 10)

The book builds the interval by inverting the test over a constant effect
`C`. `sp.fisher_exact` did the same on a 101-point grid, took the first
and last grid point with `p >= alpha`, and drew new permutations at each
grid point. Measured coverage of the 95% interval with 12 treated of 60:
92.7% over 300 draws. A randomization interval should be at or above
nominal.

The pass over Ding's *First Course* found the same defect on the same day
and its fix reached `main` first (`06b0dd78`): for the difference in
means the statistic under `C` is linear in `C`, so the interval is
inverted on the draws that gave the p-value. That implementation was
kept. A version written here, with closed-form ends instead of bisection,
gave the same intervals and was dropped.

What this pass adds is the other statistics. For `statistic='ks'`,
`'rank_sum'` and `'t'` the interval was the percentiles of the null
distribution of the statistic. That is an acceptance region for the
statistic, not an interval for the effect. It is now the same inversion
in outcome units: assignments are stored (at most 2e7 cells), the
statistic is recomputed on `Y - C D` for all of them at once, and each
end is bisected. Coverage of a constant effect at 95% nominal: 96.0%
(rank sum) and 96.7% (KS) over 150 draws; zero is outside the interval
exactly when the test rejects, in every draw tried
(`tests/test_fisher_exact_interval.py`, which also checks the
difference-in-means interval against brute force on an enumerated
design).

### 3. Classic synthetic control: interval and p-value from different procedures (chapter 10)

`sp.synth(method='classic')` reports the ADH rank p-value and, as `ci`,
`estimate -/+ z * sd(placebo ATTs)`. Question 5 of the chapter is exactly
this: a test that rejects and an interval that contains zero cannot come
from the same procedure. Here they do not.

`model_info['ci_permutation']` now holds the dual interval
(Firpo and Possebom 2018, constant effect, RMSPE-ratio statistic). The
placebo donor pools contain the treated unit, so under `H0: C` placebo
`j`'s gap moves by `w_j1 C`. Each comparison is a quadratic in `C`; the
ends are exact and match a brute-force grid.

`ci` is now that interval. `se` and `pvalue` are unchanged and the normal
interval stays in `model_info['ci_normal']`. Bryce delegated the choice.
The reasons for switching: a reported interval should be the one the
reported test implies, and with fewer than `1 / alpha - 1` donors the
honest answer is that nothing can be rejected, which the normal interval
hid behind finite numbers. `summary()` printed an infinite end as a
blank; it now prints `-inf` / `inf`.

### 4. `sp.datasets.nsw_dw()` (chapter 5)

The LaLonde application needs the NSW treated and the PSID comparison
group. `sp.datasets.nsw_dw()` has that shape and its docstring said it
"combines the 185 NSW treated (from the experiment) with 2,490
non-experimental PSID males". It is simulated. The docstring now says so
in its first line and `attrs['simulated']` is set. Rows unchanged, so the
Track A modules that use it (11, 22 and others) are unaffected.

`sp.datasets.nsw_dw(simulated=False)` now returns the real sample (the
file of the original-data ledger, module 04b, which agrees row by row
with Dehejia's NBER files to the rounding of earnings). The bare call
stays the replica. Making the real data the default would follow the
datasets rule but would move the fixtures of Track A modules 11 and 22;
that belongs to the next re-anchor of the JSS paper, not to this pass.

## Left open

1. **Default of `nsw_dw()`.** Real data by default at the next JSS
   re-anchor, with modules 11 and 22 regenerated. See above.
2. **Other `sp.synth` methods.** Several report a normal interval around
   a placebo spread next to a rank p-value in the same way. Only
   `method='classic'` was changed, because only there are the placebo
   donor pools and weights that the inversion needs kept in the result.
3. **Sparse-group Lasso and MIDAS (chapter 11).** No counterpart. A new
   public function; held back under the new-API threshold. `midasml` is
   the reference. Decided not to add during the paper submissions.
4. **`hdm::rlassoATE / rlassoATET / rlassoLATE / rlassoLATET`.** Not
   ported. `sp.dml(model='irm', score='ATTE')` with `RlassoRegressor` /
   `RlassologitClassifier` learners covers the estimand with
   cross-fitting; numbers will not equal hdm's, which does not cross-fit.
   On the `nsw_dw` replica every learner pair tried (rigorous Lasso, OLS
   with logit, the defaults) lands between 3,300 and 5,300 with standard
   errors of 850 to 2,700, against a latent effect of 1,794: the
   propensity score is within 0.002 of zero or one for most rows and
   earnings are censored at zero. That is the LaLonde problem the replica
   was built to show, not a defect of the estimator, but nobody should
   read 1,794 as the number this data returns.
5. **Translators.** `sp.from_stata` refuses `rlasso`, `pdslasso`,
   `ivlasso`; `sp.from_r` refuses the hdm, grf and policytree calls. The
   clustered `pdslasso` / `ivlasso` now have an exact target. The
   unclustered default of `pdslasso` (homoskedastic loadings, classical
   variance) and `robust` have not been compared.
6. **Fixed-point multiplicity of the iterated loadings.** StatsPAI follows
   hdm's path. A `control={'path': 'lassopack'}` switch would let users
   reproduce Stata on the panels where the two differ.
7. **JTPA applications of chapters 8 and 9.** Data not in the repository.
   The forest and policy-tree operators were verified in earlier passes.
8. **`sp.ivreg` takes only a formula** while `sp.liml` also takes
   `y= / x_endog= / x_exog= / z=`. With 64 generated instruments the
   formula is unwieldy.

## Rerunning

```bash
PYTHONPATH="$(pwd)/src" python3 -m pytest \
  tests/reference_parity/test_cluster_lasso_stata.py \
  tests/test_fisher_exact_interval.py \
  tests/test_synth_placebo_inversion_ci.py -q
# Stata side (needs lassopack, pdslasso)
cd tests/reference_parity/_fixtures && stata -b do _generate_cluster_lasso_Stata.do
```
