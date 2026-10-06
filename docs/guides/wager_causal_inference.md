# Wager, *Causal Inference: A Statistical Learning Approach*, in StatsPAI

Stefan Wager's textbook (Stanford, draft of September 2026) builds causal
inference from randomized trials outward. Each chapter relaxes one
assumption of the trial and asks which estimator still has a central limit
theorem. It has fifteen chapters and no code, so there is nothing to
replicate line by line. This guide maps each chapter to the StatsPAI call
that computes its estimator, and shows the parts that were added or rebuilt
while the book was read against the package.

## Chapter by chapter

| chapter | the book defines | StatsPAI |
| --- | --- | --- |
| 1 | difference in means; interacted regression adjustment | `sp.ttest(unequal=True)`; `sp.lm_lin(superpopulation=True)` |
| 2 | stratification on the propensity score; inverse-propensity weighting | `sp.match(method='stratify', estimand='ATE')`; `sp.ipw(se_method='sandwich')` |
| 3 | augmented IPW with cross-fitting | `sp.aipw`; `sp.dml(model='irm')`; `sp.tmle` |
| 4 | the R-learner and other CATE methods; what to do after a CATE fit | `sp.metalearner(learner='r')`; `sp.causal_forest`; `sp.rate`, `sp.calibration_test`, `sp.best_linear_projection` |
| 5 | policy value by IPW and AIPW; empirical welfare maximization | `sp.ips`, `sp.doubly_robust`; `sp.policy_tree`, `sp.policy_value` |
| 6 | UCB, Thompson sampling; inference after adaptive collection | `sp.bandit_experiment`, `sp.bandit_allocate`; `sp.adaptive_inference` |
| 7 | covariate-balancing propensity scores; approximate balance with an augmented estimator | `sp.cbps(variant='exact')`, `sp.ebalance`, `sp.sbw`; `sp.residual_balance` |
| 8 | local linear regression discontinuity; optimized weights and bias-aware intervals | `sp.rdrobust`; `sp.rd_honest`, `sp.rd_optimized` |
| 9 | back-door and front-door formulas, do-calculus; instrumental variables with a learned instrument | `sp.dag`, `sp.identify`, `sp.frontdoor`; `sp.ivreg`, `sp.dml(model='pliv')` |
| 10 | the local average treatment effect; marginal treatment effects | `sp.ivreg`; `sp.dml(model='iivm')`; `sp.iv.mte` |
| 11 | exposure mappings; permutation tests for spillovers | `sp.interference_test` |
| 12 | exposure effects with network-robust variance | `sp.network_exposure` |
| 13 | difference in differences with staggered adoption; synthetic controls | `sp.did_imputation`, `sp.callaway_santanna`; `sp.synth`, `sp.sdid` |
| 14 | sequential unconfoundedness; doubly robust evaluation of a dynamic policy | `sp.ltmle(regime_treated=callable)`; `sp.msm`; `sp.gformula_ice_fn` |
| 15 | Markov decision processes; switchback experiments | `sp.switchback`, `sp.switchback_design` |

Chapters 1 to 3 were run on designs with a known answer. The estimators
recover the truth and their intervals cover at the nominal rate. Two
defaults are worth knowing. `sp.lm_lin` reports the design-based standard
error, which targets the average effect in the sample; the book's Theorem
1.3 is the population version, which is `superpopulation=True`. And
`sp.match(method='stratify')` estimates the effect on the treated unless
told otherwise.

## Chapter 4: the R-learner

`sp.metalearner(learner="r")` fits the R-learner: cross-fitted outcome and
propensity models, then a regression of the outcome residual on the
treatment residual with a flexible model for the effect.

```python
fit = sp.metalearner(df, "y", "w", covariates, learner="r")
fit.estimate, fit.se            # average effect, from doubly robust scores
fit.model_info["cate"]          # fitted conditional effects
```

The final regression has a heavy-tailed target, since it divides by the
treatment residual. Its default model is gradient boosting with at least
20 observations per leaf. Without that minimum the trees give single
extreme values their own leaves, and the fitted effects were less
accurate than a constant on our test design. Pass `cate_model=` to use
anything else. After fitting, check that the heterogeneity is real with
`sp.rate` or `sp.calibration_test` on a causal forest, or `sp.cate_eval`
on any set of predictions.

## Chapter 6: adaptive experiments

A bandit rule shifts assignment towards the arms that look best so far.
`sp.bandit_experiment` runs one on a reward function or on a table of
potential outcomes, and records the probability with which each arm was
assigned in each period.

```python
import statspai as sp

means = [0.0, 0.3, 0.3]
exp = sp.bandit_experiment(
    lambda k, rng: means[k] + rng.normal(),
    n_periods=1000, n_arms=3,
    algorithm="thompson", sigma=1.0,
    prob_floor=0.02,          # keep every arm alive
    true_means=means, seed=0,
)
exp.arms          # pulls and mean reward by arm
exp.regret        # shortfall relative to always playing the best arm
```

In a live experiment, `sp.bandit_allocate(df, "y", "arm", ...)` gives the
probabilities for the next subject from the data collected so far.

The data are not independent draws, and the usual interval around a
sample mean is not valid. Under Thompson sampling the sample mean of an
arm is biased downwards, because an arm that looks bad early is sampled
less and never gets the chance to recover. In 600 simulated experiments
with two equal arms the nominal 95% interval covered 87% of the time for
the contrast. `sp.adaptive_inference` weights each observation by the
inverse square root of its assignment probability, which makes the
variance of every term the same and restores a central limit theorem.

```python
r = sp.adaptive_inference(exp.data, "reward", "arm", "prob")
r.estimates       # arm means, standard errors, intervals
r.contrasts       # each arm against the first
```

The price is paid at design time. A rule that drives probabilities to
zero quickly earns more during the experiment and leaves less to learn
from afterwards. With `prob_floor=0.02` the weighted interval covered
95.5%; with no floor it covered 93% for two arms and 88% for the worst
arm of three. `sp.adaptive_inference` warns when an assigned arm had
probability below `1/T` and refuses data from a deterministic rule such
as UCB.

## Chapter 7: balancing with many covariates

With more covariates than a propensity model can carry, exact balance is
out of reach. `sp.residual_balance` finds weights that make the worst
covariate imbalance small, fits a sparse linear outcome model in each
arm, and applies the weights to its residuals.

```python
r = sp.residual_balance(df, "y", "w", covariates)       # ATE
r = sp.residual_balance(df, "y", "w", covariates, estimand="ATT")
r.model_info["imbalance_treated"]     # worst remaining imbalance, in sd
r.model_info["effective_sample_size"]
```

The regression does most of the work and the weights clean up what it
missed, which is why neither has to be exact. Weighting alone
(`outcome_model="none"`) is biased in high dimensions: with 150
covariates and 300 units it was off by a full standard error, while the
augmented estimate was unbiased with a 93% interval. The standard error
treats the covariates as fixed, so the interval is for the average
effect in the sample.

With few covariates, prefer exact balance. One finding from the audit:
the over-identified variant of the covariate-balancing propensity score,
the default of `sp.cbps` and of R's `CBPS`, was biased upwards by about
0.09 (0.8 standard errors) on a design with a heterogeneous effect and a
correctly specified logistic propensity, in both packages. The exactly
identified variant (`variant='exact'`), entropy balancing and AIPW were
unbiased on the same data.

## Chapter 8: optimized regression discontinuity

A sharp regression discontinuity estimate is a weighted sum of outcomes.
Local linear regression picks the weights with a kernel and a bandwidth.
`sp.rd_optimized` picks them directly: among all weights, the ones with
the smallest worst-case mean squared error over conditional mean
functions whose second derivative is at most `M`.

```python
r = sp.rd_optimized(df, "y", "x", c=0, M=0.1)
r.estimate, r.se, r.ci
r.model_info["max_bias"]              # exact worst case for these weights
r.model_info["effective_bandwidth"]   # farthest unit with weight
r.model_info["local_linear"]          # the same numbers for local linear
```

The interval is the bias-aware one of `sp.rd_honest`: the estimate plus
or minus a critical value that grows with the ratio of worst-case bias to
standard error. It is valid whether the running variable is continuous
or takes a handful of values, because nothing in it relies on a density
at the cutoff. On a design whose mean function sits at the curvature
bound, the interval covered 96% in 300 samples.

For a continuous running variable the gain over local linear regression
with a triangular kernel is small, as theory says it should be: on the
Lee (2008) election data the half-length is 2.996 against 3.002. The
case for the method is the discrete running variable and the fact that
the weights and the worst-case bias are exact objects that can be
inspected. `M` is the assumption that carries the result. Report it, and
show the interval at twice its value.

## Chapters 11 and 12: spillovers on a network

Start by asking whether there are spillovers at all. Under the hypothesis
that a unit's outcome depends only on its own treatment, the outcomes of a
set of focal units do not change when the treatments of the *other* units
are reshuffled. `sp.interference_test` uses that to build an exact test.

```python
t = sp.interference_test(Y, Z, adjacency=A)     # H0: no spillovers
t.pvalue
sp.interference_test(Y, Z, null="no_effect")    # H0: no effect at all
```

The two hypotheses are nested, so testing them in that order needs no
multiplicity correction. The test is valid for Bernoulli and completely
randomized assignment.

If spillovers are present, `sp.network_exposure` estimates average
outcomes by exposure level. The default mapping crosses a unit's own
treatment with whether any neighbour is treated.

```python
r = sp.network_exposure(Y, Z, adjacency=A, p_treat=0.5)
r.contrasts     # direct, spillover, composite
r.estimates     # mean outcome, se, smallest probability and
                # effective sample size at each exposure level
```

Three things to watch.

- **Overlap.** A unit with ten neighbours has no treated neighbour with
  probability `0.5 ** 10`. If such a unit is ever observed in that state
  it carries a weight of a thousand. The function warns, and
  `min_prob=0.01` restricts the averages to units for which every
  exposure has at least that probability. This changes the population
  the estimate describes, which is why it is not done silently.
- **The estimator.** The default divides by the sum of the weights
  (Hajek). The Horvitz-Thompson form (`estimator="ht"`) is unbiased but
  changes when a constant is added to the outcome, and on a 600-node
  network its spread was five times larger.
- **The variance.** Two units have dependent exposures when they share a
  neighbour. `variance="hac"` sums over such pairs. It is close to
  unbiased when overlap is good and can be too small otherwise, because
  the dependency graph is not positive semidefinite. The default
  `variance="hac_psd"` uses its positive semidefinite part, which is
  conservative for the randomization variance in every case.

## Chapter 10: marginal treatment effects

`sp.iv.mte` fits polynomial marginal treatment response functions and
reports the ATE, the effect on the treated and on the untreated, each
with a standard error.

```python
m = sp.iv.mte("y", "d", ["z"], exog=["x"], data=df, poly_degree=1)
m.ate, m.ate_se
m.att, m.extra["att_se"]
```

The analytic standard errors take the fitted propensity score and the
covariates as given. `bootstrap=200` re-estimates both in each draw.
Observations with a fitted propensity outside `(trim, 1 - trim)` are
dropped, and the averages describe the units that remain.

## Chapter 14: dynamic policies

A policy that decides each period from the history so far is a callable
regime in `sp.ltmle`.

```python
def treat_if_high(k, history):
    return (history[f"L{k}"] > 0).astype(int)

r = sp.ltmle(df, "Y", ["A0", "A1"], [["L0"], ["L1"]],
             regime_treated=treat_if_high, regime_control=[1, 1])
```

On a two-period design where the second treatment helps only when the
intermediate covariate is high, the value of the dynamic policy and its
contrast with always treating were recovered without bias and the
interval covered 95%.

## Not in StatsPAI

- Optimized regression discontinuity with a multivariate running
  variable or a fuzzy design (chapter 8.2 treats the univariate sharp
  case, which is `sp.rd_optimized`).
- Doubly robust estimation of the long-run value of a policy in a Markov
  decision process (chapter 15.1), and marginal policy effects (15.2).
- Contextual bandits (chapter 6 covers the case without covariates).
- Tests of the richer hypotheses in the chapter 11 hierarchy, such as
  spillovers that depend only on the share of treated neighbours.
