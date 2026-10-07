# Schuler and van der Laan, *Modern Causal Inference*, in StatsPAI

Alejandro Schuler and Mark van der Laan's online book
(<https://alejandroschuler.github.io/mci/>) is a book about how estimators
are built, not a book of recipes. Its first edition, *Introduction to
Modern Causal Inference*, runs from estimands and identification through
efficiency theory to the three ways of constructing an efficient
estimator. The second edition, *Foundations of Modern Causal Inference*,
rewrites the theory chapters and so far stops before estimation. Neither
ships data or replication code. The running example throughout is the
average treatment effect from `(Y, A, X)`.

So this guide does not map chapters to numbers. It maps the book's ideas to
what you can check on a StatsPAI result, and says which call corresponds to
which construction.

## The three constructions

The book expands the error of a plug-in estimate into four terms,

```text
psi(P_hat) - psi(P) =  P_n phi            the efficient CLT term
                     - P_n phi_hat        plug-in bias
                     + (P_n - P)(phi_hat - phi)   empirical process
                     + R                  second-order remainder
```

and shows that the three constructions differ only in how they remove the
plug-in bias.

| book (first edition) | idea | StatsPAI |
| --- | --- | --- |
| 4.1 naive plug-in | evaluate the estimand at a fitted distribution | `sp.g_computation` |
| 4.2 bias correction (one-step) | add `P_n phi_hat` to the plug-in | `sp.aipw` |
| 4.3 estimating equations, "DML" | solve `P_n phi(psi, eta_hat) = 0` for `psi` | `sp.aipw`; `sp.dml(model='irm')` with machine-learned nuisances |
| 4.4 targeted maximum likelihood | perturb the fit until `P_n phi* = 0`, then plug in | `sp.tmle`; `sp.hal_tmle`; `sp.ltmle`; `sp.ctmle` |
| 4.5 inference | variance of the estimated influence function over `n` | every function above; `model_info['influence_function']` on `sp.tmle` |

For the ATE the one-step and estimating-equation estimators are the same
formula, which is why one function serves both rows. `sp.aipw` fits a
logit propensity and per-arm outcome regressions that are linear by
default; `outcome_model='logit'`, `'probit'` or `'poisson'` keeps the fitted
regression inside the range of a binary or count outcome, as Stata
`teffects aipw (y x, logit)` does.

```python
import numpy as np, pandas as pd, statspai as sp

rng = np.random.default_rng(0)
n = 2000
x1, x2 = rng.normal(size=n), rng.normal(size=n)
a = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2))))
y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 0.7 * a + 0.6 * x1))))
df = pd.DataFrame({"x1": x1, "x2": x2, "a": a, "y": y})

plug = sp.g_computation(df, y="y", treat="a", covariates=["x1", "x2"], seed=0)
onestep = sp.aipw(df, y="y", treat="a", covariates=["x1", "x2"])
tmle = sp.tmle(df, y="y", treat="a", covariates=["x1", "x2"])
```

## Two properties you can check

**The targeted fit solves the influence-function equation.** That is the
whole point of the targeting step (section 4.4), and it is observable:

```python
ic = tmle.model_info["influence_function"]
abs(ic.mean())                       # about 1e-17
ic.std(ddof=1) / np.sqrt(len(ic))    # equals tmle.se
```

The same vector gives the covariance between two estimates fitted on the
same rows, which is what a joint test of several estimands needs.

**A TMLE is a plug-in.** The book's argument for TMLE over the one-step
estimator is that a plug-in cannot leave the range of the parameter. With a
rare binary outcome and poor overlap an AIPW estimate of a risk can be
negative. The targeted means cannot:

```python
means = sp.tmle(df, y="y", treat="a", covariates=["x1", "x2"], estimand="EY1")
means.detail
#   parameter  estimate   se   ci_lower  ci_upper  pvalue  log_estimate  se_log
#   EY1, EY0, ATE, RR, OR
```

`tests/test_tmle_targeting_properties.py` asserts both properties for every
estimand, and adds a third check: with a saturated model the initial fit is
already the nonparametric maximum likelihood estimate, nothing is left to
target, and every estimand must equal its stratification formula exactly.

## Estimands beyond the ATE

Section 1.3 warns that the exponentiated coefficient of a logistic
regression is a conditional odds ratio, which is rarely the question. The
marginal versions are functions of the two treatment-specific means, and
the chain rule for gradients (section 3.3) gives their influence functions.

| estimand | call | note |
| --- | --- | --- |
| `E[Y(1)]`, `E[Y(0)]` | `sp.tmle(estimand='EY1' / 'EY0')` | each arm is targeted separately |
| marginal risk ratio | `estimand='RR'` | interval built on the log scale |
| marginal odds ratio | `estimand='OR'` | outcome in [0, 1] |
| effect on the treated | `estimand='ATT'` | see below |
| effect on the controls | `estimand='ATC'` | the ATT computation with the arms relabelled |

With the same initial fits these agree with R `tmle` 2.1.1 to 1e-11 for the
means, the difference, the risk ratio and the odds ratio, with and without
observation weights and clusters
(`tests/reference_parity/test_tmle_parameters_R_parity.py`). There is one
documented exception. With observation weights, the influence curve R
`tmle` uses for the log odds ratio is not centred. It carries an extra
`w * (1 / (1 - EY1) - 1 / (1 - EY0))`, which is a constant without weights
and adds variance with them. StatsPAI reports the centred one, and the test
rebuilds R's number from it to 1e-9.

### The effect on the treated

The second edition asks the reader to derive the gradient of the ATT. It is

```text
D = H1 (Y - Q(A, W)) + (A / p) (Q(1, W) - Q(0, W) - psi)
H1 = A / p - (1 - A) g / (p (1 - g)),   p = P(A = 1)
```

The second term has a component in the tangent space of `A | W`, and there
are two ways to make its empirical mean vanish. `sp.tmle` takes the
distribution of `W` among the treated to be the empirical one. It then
fluctuates the outcome regression along `H1` with `g` held fixed, and the
score equation of that fluctuation makes the estimate equal to the mean of
`Q*(1, W) - Q*(0, W)` over the treated. That is a plug-in, and the mean of
`D` at it is zero to machine precision. `estimand='ATC'` is the same
computation with the arms relabelled.

R `tmle` takes the other route. It models the treated distribution through
`g`, so it updates `g` as well, along a small-step path, after dropping the
controls whose propensity lies below the smallest propensity among the
treated. It stops when the likelihood stops improving, which leaves the
mean of its influence curve at 1e-4 to 4e-3. Both are TMLEs of the same
parameter. On the test fixtures they are within 0.02 standard errors of
each other. `tests/reference_parity/test_r2_teffects_parity.py` pins R's
mechanism, and `test_tmle_parameters_R_parity.py` records the two side by
side for the ATT and the ATC.

## Randomised trials

The second edition projects the gradient of the ATE onto the tangent space
of a trial with known assignment probabilities and finds that nothing
changes. Knowing the treatment mechanism does not lower the efficiency
bound. It does help in finite samples, and it buys
robustness: with the true propensity the estimator is consistent whatever
the outcome regression is.

```python
sp.aipw(trial, y="y", treat="a", covariates=["x"], propensity=0.5)
sp.aipw(trial, y="y", treat="a", covariates=["x"], propensity="design_p")
sp.tmle(trial, y="y", treat="a", covariates=["x"], g1W=0.5)
sp.ipw(trial, y="y", treat="a", covariates=["x"], propensity=0.5)
```

No propensity model is fitted and nothing is clipped. One of the second
edition's exercises is the reason `sp.ipw` documents a caveat: an IPW
estimator that fits the propensity is more precise than one that uses the
true propensity, because the fit absorbs chance imbalance. The efficient
estimators do not have that dependence. For the regression
estimator the book analyses in section 1.3, and its sandwich variance, see
`sp.lm_lin`.

## Sample splitting

Section 4.1 gives two ways to control the empirical-process term. Sample
splitting needs only consistency of the nuisance fits. A Donsker condition
restricts the learners.

| | default | cross-fitted |
| --- | --- | --- |
| `sp.aipw` | cross-fitted, 5 folds | `cross_fit=False` for the classical estimator |
| `sp.dml(model='irm')` | cross-fitted | repeat with `n_rep=` |
| `sp.tmle` | **not** cross-fitted; `n_folds` is the Super Learner's internal CV | `fold_indices=` gives CV-TMLE |
| `sp.hal_tmle` | highly adaptive lasso, the learner the book singles out for its rate | |

## When the estimand is not identified

Section 2.3 separates the statistical error `psi_hat - psi` from the causal
gap `psi - psi*`, and offers three responses.

| book | StatsPAI |
| --- | --- |
| partial identification | `sp.manski_bounds`, `sp.lee_bounds`, `sp.iv_bounds` |
| sensitivity through a parametrised violation | `sp.evalue`, `sp.sensemakr`, `sp.confounder_tip`, `sp.rosenbaum_bounds` |
| critical causal gap | `sp.causal_gap` |

The critical causal gap is the book's own proposal. It asks how large the
gap would have to be for the interval to reach the null, or a threshold of
practical interest, and says nothing about where the gap comes from.

```python
sp.causal_gap(1.0, ci=(0.5, 1.5), null=[0.0, 0.3])
#    null  estimate  ...  critical_gap  gap_to_point  share_of_estimate
#     0.0       1.0                0.5           1.0               0.50
#     0.3       1.0                0.2           0.7               0.29

tbl = sp.causal_gap(tmle)            # a fitted result works too
tbl.attrs["curve"]                   # implied causal estimate and interval by gap
sp.causal_gap(rr, scale="ratio")     # gap as a bias factor
```

## What the simulations say

Known truth, 400 replications, `n = 800`, logistic and linear nuisances
that are correctly specified. "Poor overlap" multiplies the propensity
coefficients by 2.25, which puts about 9% of the true propensities outside
[0.025, 0.975].

| | good overlap | poor overlap |
| --- | --- | --- |
| ATE, coverage of the 95% interval | 0.94 to 0.96 | 0.91 to 0.92 |
| ATT | 0.93 to 0.95 | 0.82 to 0.83 |
| risk ratio, odds ratio | 0.93 | 0.90 to 0.91 |

The second column is the book's warning about inverse weights made
concrete. The influence-function variance is too small when propensities
approach zero, for every construction. The estimates themselves are
unbiased here; it is the standard error that falls short, by 15% to 21%.
Moving the truncation bound does not help. Section 4.5 mentions the
bootstrap as the alternative, and it does help:

```python
sp.tmle(df, y="y", treat="a", covariates=["x1", "x2"], estimand="ATT",
        se_method="bootstrap", n_boot=200)
```

| poor overlap | influence function | cross-fitted | bootstrap percentile |
| --- | --- | --- | --- |
| ATE | 0.91 | 0.92 | 0.92 |
| ATT | 0.81 | 0.84 | 0.90 |

### Collaborative targeting

Section 4.4 of the book mentions C-TMLE as a variant. It attacks weak
overlap at its source. A TMLE needs the propensity only to remove the bias
the outcome model left, so a covariate that drives treatment but not the
outcome (an instrument) has no business in it: it buys no bias reduction
and makes the weights extreme. `sp.ctmle` builds the propensity model one
covariate at a time, keeping a covariate only if targeting with it
improves the outcome fit, and lets cross-validation choose where to stop.

```python
res = sp.ctmle(df, y="y", treat="a", covariates=["x1", "x2", "x3"],
               se_method="bootstrap")
res.model_info["selected_covariates"]     # what entered the propensity
res.detail                                # every step of the sequence
res.model_info["selection_frequency"]     # across bootstrap resamples
```

With many candidates, pass `order=[...]` to offer them in a fixed order
(the pre-ordered variant of Ju et al.): one propensity fit per step where
the greedy search needs one per remaining candidate. `penalty='search'`
reproduces the default criterion of R `ctmle`.

In the poor-overlap design above, where `x2` affects treatment only:

| ATE, `n = 800` | bias | sd | 95% coverage |
| --- | --- | --- | --- |
| `sp.tmle`, full propensity, influence function | -0.01 | 0.130 | 0.91 |
| `sp.ctmle`, influence function | -0.01 | 0.095 | 0.89 |
| `sp.ctmle`, bootstrap | -0.01 | 0.095 | 0.94 |

Three cautions. Report the bootstrap interval: the influence-function
standard error ignores the selection and is 20% to 40% too small, in R
`ctmle` as well. Do not expect a gain when the outcome model is badly
wrong: with a confounder omitted from it, the method brings that
confounder into the propensity, as it should, and ends up about as
accurate as `sp.tmle`. And it is offered for the ATE only. A
collaborative ATT was tried and was biased when the outcome model missed
effect heterogeneity, because no step of the sequence is rewarded for
repairing that.

`sp.tmle` warns when more than 5% of the propensities hit
`propensity_bounds`. Take the warning seriously, look at `sp.overlap_plot`,
and prefer the bootstrap interval in that regime.

## What a paper written today would add

- Cross-fit. `sp.tmle(fold_indices=)` or `sp.dml`.
- Report the estimand you mean. A marginal risk ratio from
  `sp.tmle(estimand='RR')` is not a logistic coefficient.
- State the causal gap that would overturn the result, next to any
  model-based sensitivity analysis.

## References

- Schuler, A. and van der Laan, M. *Introduction to Modern Causal
  Inference*. [@schuler2022introduction]
- van der Laan, M. J. and Rose, S. (2011). *Targeted Learning*.
  [@vanderlaan2011targeted]
- Gruber, S. and van der Laan, M. J. (2012). tmle: An R Package for
  Targeted Maximum Likelihood Estimation. [@gruber2012tmle]
