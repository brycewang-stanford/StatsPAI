# Schuler and van der Laan, *Modern Causal Inference*: review record

Date: 2026-10-07. Worktree `wt/schuler-mci`.

## Material

`改进建议-收集整理/34-Schuler-vanderLaan-ModernCausalInference/` holds two
clones of <https://github.com/alejandroschuler/mci>.

- `mci/`: the first edition, *Introduction to Modern Causal Inference*
  (2022), exported from Notion as HTML. Four chapters: inference and
  statistics, causality and identification, efficiency theory, building
  efficient estimators.
- `mci-v2/`: the second edition, *Foundations of Modern Causal Inference*
  (Quarto sources, last commit 2025-11-24). Models and estimands,
  counterfactuals, structural causal models, regularity, efficiency,
  deriving canonical gradients. The plug-in chapter is a stub; estimation
  has not been rewritten yet.

Neither edition has data, replication code, or a number to reproduce. The R
chunks in the second edition draw illustrations. So this was a syllabus
audit: read what the book says an estimator must satisfy, then test
StatsPAI against those statements, against known truth, and against the R
package the second author's group maintains.

"Outdated" in practice means two things. The first edition predates the
rewrite and has a sign slip in the AIPW display of section 4.2 (the last
term should be added). Its estimation chapter is otherwise current.

## Method

1. Read chapters 2.3, 4.1 to 4.5 of the first edition and 3-4 of the second
   in full; the rest for their examples.
2. Monte Carlo with known truth for `sp.tmle`, `sp.aipw`, `sp.ipw`,
   `sp.g_computation`, `sp.dml(model='irm')`: 300 to 400 replications,
   `n = 800`, binary and continuous outcomes, good and poor overlap.
3. `tmle::tmle` 2.1.1 as a black box with the initial fits supplied, so
   only targeting, plug-in and variance are compared. `tmle` is GPL; its
   source was not read in this round. What this round says about its behaviour was
   inferred from its outputs.

## Findings

### 1. `sp.tmle(estimand=)` was not validated (⚠️ fixed)

Any value other than the exact string `'ATE'` took the ATT branch, and the
result carried the string as its label. `estimand='ate'` returned 0.1795
(the ATT) where the ATE is 0.1566. `'ATC'`, `'RR'` and typos did the same.
Now case-insensitive, with a closed list.

### 2. The ATT: examined, an alternative tried, nothing changed

Section 4.4 defines a TMLE as a plug-in at a fit that solves
`P_n phi* = 0`. `sp.tmle(estimand='ATT')` targets `Q` along
`A - (1 - A) g / (1 - g)`, leaves `g` alone, and reports the solution of
the ATT estimating equation. Its docstring said this was "not the plug-in",
and the ATT gradient does have an `A | W` component, so the first reading
was that the estimator fell short of the book's definition.

That reading was wrong, and the mistake is recorded here because it cost
half a session. The fluctuation solves `sum H (Y - Q*) = 0`, which makes
the estimating-equation form identical to the mean of
`Q*(1, W) - Q*(0, W)` over the treated. It is a plug-in at the empirical
distribution of `W` among the treated, and the influence function has mean
zero at it (1e-16 on the fixtures). An existing test,
`tests/reference_parity/test_r2_teffects_parity.py`, already asserts both.

Before that was found, an alternative was implemented and made the
default: fluctuate `logit Q` and `logit g` together with one shared
parameter and iterate. It converged in four passes, solved the equation to
1e-10, and in 400 replications matched the existing estimator to the
fourth digit with the same coverage (0.927 / 0.953). With no gain to
show, it was withdrawn. The docstring now states the identity. No ATT
number moved.

`estimand='ATC'` was added as the same computation with the arms
relabelled.

### 3. R `tmle` on the ATT and ATC (documented, not matched)

R reports a different TMLE of the same parameter. The existing test file
above pins its mechanism for the ATT: a small-step path that updates both
`Q` and `g`, on the rows with `g >= min(g | A = 1)`, stopped when the
likelihood stops improving. The black-box experiments of this round agree
with that description. Aligning R's influence curve against ours shows
that the missing rows are exactly the controls below the smallest treated
propensity (checked at `n` = 500, 1000, 1500, 2000; for the ATC, the
treated above the largest control propensity). The mean of R's curve at
its reported solution is 8e-5 to 4e-3 in absolute value on the fixtures
here, up to 0.035 standard errors, and for a continuous outcome the curve
carries a constant `min(Y)`.

On the new fixture the two ATTs and the two ATCs differ by at most 0.014
standard errors and their variances by at most 0.8%. Grade: T4, two valid
estimators, recorded side by side. Not a parity row.

### 4. Estimands the targeted fit already supports (added)

`EY1`, `EY0`, `RR`, `OR`, `ATC`. The first four need each arm's score
equation solved, which the single clever covariate does not do, so they
force `fluctuation='per_arm'`. With shared initial fits the means, the
difference, the risk ratio and the odds ratio agree with R to 1e-11 in
estimate, variance, interval and p-value, in four designs (plain, cluster,
weights, both). Grade: T2.

One exception, located. With observation weights R's log odds ratio curve
is `w * (uncentred terms)`. The centring constant
`1 / (1 - EY1) - 1 / (1 - EY0)` is harmless when `w = 1` and adds
`c^2 var(w)` otherwise: 0.27% on the weighted fixture, 2% with clusters.
The risk ratio is immune because its constant is zero. Adding the term to
our curve reproduces R's variance to 1e-12. We report the centred
variance.

### 5. Known treatment mechanism (added)

The second edition's worked projection shows the ATE gradient is the same
in a randomised trial, and remarks that the known propensity should still
be used. `sp.aipw(propensity=)` and a scalar `g1W` in `sp.tmle` do that.
Test: 200 replications with a deliberately wrong outcome regression and
the true propensity; the mean estimate is within four Monte Carlo standard
errors of the truth.

### 6. Critical causal gap (added)

Section 2.3 of the first edition. `sp.causal_gap` reproduces the book's
example (estimate 1, half-width 0.5: gap 0.5 against the null, 0.2 against
a practical threshold of 0.3) and extends it to ratios.

### 7. Checked and left alone

- `sp.tmle` ATE against R: already at 1e-10 (Track A module 72).
- `sp.aipw`, cross-fitted or not: bias below 0.003, coverage 0.92 to 0.96.
- `sp.g_computation`: unbiased for the ATE. With the default additive
  outcome model its ATT equals its ATE, so it is biased for the ATT when
  effects vary with covariates (-0.17 in the continuous design). That is
  the book's naive plug-in doing what a misspecified plug-in does;
  `by_arm=True` fits each arm.
- `sp.dml(model='irm')` with default learners: coverage 0.977, standard
  deviation 35% to 55% above the parametric estimators at `n = 800`. The
  default forests are slow to converge at this size. Conservative, not
  wrong.
- `sp.ipw(se_method='influence')` is not an option. Not needed.

## Second round (same day): the open items

Bryce delegated the decisions. What was done with each.

1. **Weak overlap: bootstrap standard errors.** 1,000 replications with
   the initial fits correctly specified show the estimates are unbiased
   under weak overlap and the standard error is what fails (ATT: sd 0.166,
   mean influence-function se 0.131). Truncating the propensity at 1e-6,
   0.025, R's `5 / (sqrt(n) ln n)`, 0.05 or 0.1 leaves ATT coverage
   between 0.74 and 0.87; none repairs it. Cross-fitting the nuisances
   moves it from 0.81 to 0.84. A nonparametric bootstrap of the whole fit
   gives 0.89 (normal interval) and 0.90 (percentile), and 0.92 for the
   ATE. So `sp.tmle(se_method='bootstrap', n_boot=)` was added, and no
   adaptive truncation. The percentile interval is the one reported.
   The shipped option was then run end to end in the same design with an
   additive (misspecified) outcome regression and a correct propensity,
   250 replications, 150 resamples: influence-function coverage 0.908
   (ATE) and 0.824 (ATT), bootstrap 0.940 and 0.920.
2. **Weights for the ATT / ATC**: implemented. Evidence is two identities:
   integer weights equal row replication, and a weighted saturated fit
   equals the weighted stratification formula.
3. **`sp.ipw(propensity=)`**: implemented, with the sandwich that drops
   the first-stage term. With one probability for everyone it is the
   difference in means with its HC0 standard error, exactly. **Not done
   for `sp.dml(model='irm')`**: that estimator is a Track A parity module
   with retained out-of-fold records, and a second propensity path there
   is more risk than a convenience warrants. `sp.aipw(propensity=)` is
   the efficient estimator with a known propensity.
4. **Note to the `tmle` maintainers**: drafted in
   `docs/dev/2026-10-07-tmle-weighted-odds-ratio-note-draft.md`, with a
   self-contained R snippet that was run. Not sent.
5. The stability-audit failure on main (`mswitch_lrtest`, `tvp_var_sv`)
   was fixed by the line that owned it before this round started.

## Third round (same day)

- `se_method='bootstrap'` now works with `fold_indices`: a copy of a row
  keeps that row's fold.
- `sp.hal_tmle` accepts every estimand of `sp.tmle`.
- `sp.aipw(outcome_model='logit' | 'poisson')`. Section 4.2 notes that a
  bias-corrected estimator can leave the range of the parameter; a linear
  outcome regression for a binary outcome makes that worse than it has to
  be. Reference is Stata 18 `teffects aipw (y x, logit | poisson) (d x)`:
  ATE, both potential-outcome means and their standard errors agree to
  4e-9 in three designs. `teffects aipw` refuses `[pw=]`; the weighted row
  uses `[iw=]` with one cluster per row, the convention already pinned in
  `test_teffects_design_stata_parity.py`. Grade: T2.

## Fourth round (same day)

- `sp.aipw(outcome_model='probit')`. The link is not canonical, so the
  stacked sandwich uses the probit's own score and observed Hessian.
  Against Stata 18 `teffects aipw (y x, probit) (d x)`: 5e-9 at worst in
  the same three designs.

## Collaborative targeting: measured, a shortcut rejected

The question left from the second round was whether a collaborative choice
of the propensity model would do better than the bootstrap. Same design as
before (`x2` drives treatment only, so it is an instrument; 9% of true
propensities outside [0.025, 0.975]; `n = 800`).

| | sd of the ATT | mean se | coverage |
| --- | --- | --- | --- |
| `g` on all covariates, influence function | 0.152 | 0.124 | 0.86 |
| the same, bootstrap percentile (200 resamples) | 0.152 | 0.158 | 0.94 |
| oracle: `g` without the instrument, influence function | 0.108 | 0.103 | 0.92 |

(120 replications for this table, so coverage has a Monte Carlo standard
error near 0.02.) Dropping the instrument from `g` cuts the standard
deviation by 29% (23% for the ATE) and makes the influence-function
standard error accurate. That is the gain a collaborative TMLE is after,
and it is real. The bootstrap fixes the interval but not the variance.

A cheap way to get there was tried: fit the propensity on the initial
outcome predictions `(Q(0,W), Q(1,W))` instead of on the covariates. With
1,000 replications:

| outcome model | `g` on | bias of the ATE | sd | coverage |
| --- | --- | --- | --- | --- |
| correct | covariates | -0.002 | 0.128 | 0.89 |
| correct | outcome predictions | -0.002 | 0.106 | 0.92 |
| wrong (omits a confounder) | covariates | -0.081 | 0.193 | 0.83 |
| wrong (omits a confounder) | outcome predictions | +1.470 | 0.107 | 0.00 |

When the outcome model is right it recovers most of the oracle's gain.
When it is wrong the estimator loses double robustness entirely, because
the propensity can no longer correct what the outcome model missed. A
default-on or lightly documented option with that failure mode has no
place in the package. It was not added.

The full algorithm (a greedy, cross-validated sequence of propensity
models, each retargeted) exists to get the gain without that failure. It
was implemented in the fifth round below.

## Fifth round: `sp.ctmle`

Bryce asked for the full algorithm. R `ctmle` 0.1.2 was installed and used
as a black box (GPL; source not read). The implementation follows van der
Laan and Gruber (2010) and Gruber and van der Laan (2010).

**Algorithm.** Step 0 targets with an intercept-only propensity. Each
later step adds the covariate whose targeted fit has the lowest criterion.
When none improves on the current targeted fit, that fit becomes the
initial fit and the search continues. The step is chosen by
cross-validation, rebuilding the sequence on every training fold. The
criterion is the residual sum of squares plus the variance of the
influence function, and in the cross-validation also `n` times its
squared mean.

**Agreement with R.** On six fixtures (three data sets, a correct and a
wrong initial fit) the order of entry is identical and the estimate at
every step agrees to 3e-9. Two things had to be found for that. The
greedy search has to use the variance-penalised criterion, not the sum of
squares alone: with the latter our order differed from R's whenever the
outcome fit was right and the losses were flat. And `ctmle`'s
cross-validated criterion is that same sum, with a zero bias term.
Grade for the sequence: T2.

**Not matched, not located.** Given the same folds, the cross-validated
losses the two report are different numbers (5.8756 against 5.8888 at
step 0 of the first fixture), and they select different steps in four of
six cases. The cause was not found. It is recorded in the test and is a
C-level open item. Over 200 data sets the two have the same profile:

| initial fit | | bias | sd | mean se | coverage |
| --- | --- | --- | --- | --- | --- |
| correct | `sp.tmle`, full propensity | -0.012 | 0.129 | 0.108 | 0.87 |
| correct | R `ctmle` | -0.012 | 0.102 | 0.077 | 0.85 |
| correct | `sp.ctmle` | -0.009 | 0.093 | 0.073 | 0.88 |
| omits a confounder | `sp.tmle`, full propensity | -0.113 | 0.205 | 0.162 | 0.78 |
| omits a confounder | R `ctmle` | +0.066 | 0.206 | 0.116 | 0.68 |
| omits a confounder | `sp.ctmle` | +0.137 | 0.193 | 0.121 | 0.61 |

R's variance is also not ours once covariates are in the propensity
(0.0118 against 0.0151 at the same estimate on one fixture). At step 0
the two agree. Ours is the plain variance of the efficient influence
function, the larger of the two, and still too small.

**Inference.** The influence-function standard error is 20% to 40% too
small for both implementations. With the propensity deliberately
misspecified, that function is not the estimator's influence function,
and the selection is ignored. `se_method='bootstrap'` reruns everything.
End to end with a refitted outcome regression (100 replications, 80
resamples): `sp.tmle` rmse 0.130 and coverage 0.91; `sp.ctmle` rmse 0.095,
influence-function coverage 0.89, bootstrap coverage 0.94.

**The ATT, tried and withdrawn.** The same construction with the ATT
clever covariate was implemented. In the end-to-end run, where the
outcome regression is additive and the effect varies with a confounder,
it was biased by -0.17 (rmse 0.251 against 0.195 for `sp.tmle`). The
intercept-only propensity already gives the lowest criterion, and nothing
in the criterion rewards a step that would repair the missing
heterogeneity. `sp.ctmle` therefore accepts `estimand='ATE'` only and
says why. R `ctmle` is ATE-only too.

**One claim corrected along the way.** The first draft said the loss
falls along the whole sequence. It does not: the fluctuation maximises a
Bernoulli likelihood, so the sum of squares can rise slightly after a
restart. A test caught it; the text and the comment were fixed.

## Still open

- Why `ctmle`'s cross-validated losses differ from ours on the same folds.
- A collaborative estimator for the ATT that is safe when the outcome
  model misses heterogeneity.
- The scalable (pre-ordered) and glmnet variants of C-TMLE.
- The second edition's exercises on estimand-restricted models have no
  counterpart and little practical use, as the book says itself.

## Rerun

```bash
cd tests/reference_parity && Rscript _generate_tmle_parameters_R.R   # needs R tmle
pytest tests/reference_parity/test_tmle_parameters_R_parity.py \
       tests/test_tmle_targeting_properties.py \
       tests/test_aipw_known_propensity.py tests/test_ipw_known_propensity.py \
       tests/test_ctmle.py tests/reference_parity/test_ctmle_R_parity.py \
       tests/reference_parity/test_causal_gap_known_values.py -q
```
