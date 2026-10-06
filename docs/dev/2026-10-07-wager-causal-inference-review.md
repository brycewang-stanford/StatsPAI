# Wager, *Causal Inference: A Statistical Learning Approach*: what the book needs and what StatsPAI had

*2026-10-07. Worktree `wt/wager-causal`.*

## What was done

The material is one PDF, 270 pages, dated 8 September 2026. There are no
scripts and no data. The book is fifteen chapters of estimators and central
limit theorems, plus exercises that are all pencil-and-paper. So it was read
as a syllabus, the way the Xu-Lan book was. For each chapter the estimators
it defines were listed, looked up in StatsPAI, and run on a design where the
answer is known. Where the method's authors publish an R package, StatsPAI
was compared with it on the same bytes.

The note that the book might be dated did not apply. The bibliography runs
to 2026 and the chapters on interference, adaptive experiments and Markov
decision processes follow papers from the last three years.

Four kinds of result came out.

1. Two existing functions were wrong: `sp.network_exposure` (chapter 12) and
   the aggregate parameters of `sp.iv.mte` (chapter 10).
2. Three topics had no function: permutation tests for spillovers
   (chapter 11), adaptive experiments and inference after them (chapter 6),
   approximate balancing in high dimensions (chapter 7.2).
3. One default was poor: the final stage of the R- and DR-learners
   (chapter 4).
4. Everything else the book defines was already there and checked out.

## Results by chapter

| Chapter | Estimator | StatsPAI | Outcome |
| --- | --- | --- | --- |
| 1 | Difference in means, Lin's interacted regression | `sp.ttest`, `sp.lm_lin` | Correct. The default standard error of `sp.lm_lin` is design-based and covers the sample effect; against the population effect it covers 81% on a design with strong heterogeneity, and `superpopulation=True` covers 95%. Documented default, same as `estimatr`. |
| 2 | Propensity stratification, IPW | `sp.match(method='stratify')`, `sp.ipw` | Correct. Hajek IPW with the sandwich or the bootstrap covers 94 to 95%. Stratification defaults to the effect on the treated. |
| 3 | AIPW with cross-fitting | `sp.aipw`, `sp.dml(model='irm')`, `sp.tmle` | Correct. Coverage 96%, 95.5%, 92.5%. |
| 4 | R-learner, CATE estimation | `sp.metalearner`, `sp.causal_forest` | Average effects correct for all five learners. The default final stage of the R- and DR-learners was changed (below). |
| 5 | Policy value by IPW and AIPW, policy trees | `sp.ips`, `sp.doubly_robust`, `sp.policy_tree` | Correct. Doubly robust value: bias -0.02 (0.3 standard errors), coverage 96%. |
| 6 | UCB, Thompson sampling, adaptively weighted inference | none | Added: `sp.bandit_allocate`, `sp.bandit_experiment`, `sp.adaptive_inference`. |
| 7.1 | Covariate-balancing propensity scores | `sp.cbps`, `sp.ebalance` | Equal to R `CBPS` to 6e-4 on 60 data sets, both variants. The over-identified variant is biased on this design in both packages (below). |
| 7.2 | Approximate balance with an augmented estimator | none | Added: `sp.residual_balance`. Weights equal to `balanceHD` to 1e-6. |
| 8 | Local linear RD, bias-aware intervals, optimized weights | `sp.rdrobust`, `sp.rd_honest` | `sp.rd_honest` equals R `RDHonest` 1.0.1 on the Lee data to all printed digits, for fixed and optimal bandwidths, two kernels and the rule-of-thumb curvature. Added in a second round: `sp.rd_optimized` for the optimized weights of section 8.2. |
| 9 | Back-door, front-door, do-calculus, IV | `sp.dag`, `sp.identify`, `sp.ivreg`, `sp.dml(model='pliv')` | Covered by the Ness and Hansen passes; not rerun. |
| 10 | LATE, marginal treatment effects | `sp.ivreg`, `sp.iv.mte` | **Fixed.** ATT and ATU used the wrong weights, standard errors ignored coefficient covariances, and the bootstrap misaligned rows. |
| 11 | Permutation tests under interference | none | Added: `sp.interference_test`. |
| 12 | Exposure effects, network HAC variance | `sp.network_exposure` | **Rebuilt.** |
| 13 | Imputation estimator for staggered adoption, synthetic controls, SDID | `sp.did_imputation`, `sp.synth`, `sp.sdid` | Covered by the DiD and synthetic-control passes; not rerun. |
| 14 | Sequential IPW, g-formula, doubly robust dynamic evaluation | `sp.ltmle`, `sp.msm`, `sp.gformula_ice_fn` | `sp.ltmle` with a callable regime: bias 0.0002 on the dynamic policy value, coverage 95.3% for its contrast with always treating (300 runs). Two labelling fixes. |
| 15 | Long-run policy value in an MDP, switchbacks | `sp.switchback` | Switchbacks covered. The doubly robust MDP estimator is not implemented. |

## What was wrong

### `sp.network_exposure`

The function estimated exposure probabilities by simulation (2,000 draws by
default) and floored them at 0.001, although under Bernoulli assignment
they are a closed-form function of a unit's degree. The variance was

    (1 / n^2) * sum over all i of  Y_i^2 (1 - pi_i) / pi_i^2

summed over every unit, including units not observed at the exposure level
in question, and with no term for the dependence between units that share
a neighbour. Contrasts added the two variances.

On a 400-node network with mean degree 4 and maximum degree 10, with fixed
potential outcomes and 400 re-randomizations:

| | truth | mean of estimates | sampling sd | mean reported se |
| --- | --- | --- | --- | --- |
| mean outcome, untreated with no treated neighbour (old) | 7.03 | 8.60 | 6.26 | 97.9 |
| spillover contrast (old) | 1.00 | -0.52 | 6.22 | 97.9 |
| spillover contrast (new, Hajek, `hac_psd`) | 1.03 | 1.12 | 0.51 | 0.51 |

The existing tests did not catch this because they checked algebraic
identities among the reported numbers and a Monte Carlo mean on a small
regular network, where every unit has the same exposure probability.

The rebuilt function follows chapter 12 of the book. Exposure
probabilities are exact for the built-in mappings. The estimator is the
self-normalised mean by default. The variance is `v' G v / n^2`, where `v`
holds the linearised terms and `G` marks pairs of units whose exposures
share a source of randomness. `G` is not positive semidefinite in general
(the book's three-node example), so the raw form can be too small and is
occasionally negative on small networks. The default replaces `G` by its
positive semidefinite part, which makes the estimate conservative.

Evidence, in `tests/test_wager_textbook_pass.py`:

- Exposure probabilities equal a full enumeration of 128 assignments to
  1e-14, for both mappings.
- On a ten-node ring all 1,024 assignments are enumerated. The
  Horvitz-Thompson contrast is exactly unbiased, the expectation of the raw
  variance form equals the true variance plus `delta' G delta / n^2` to
  1e-10 (the identity in the proof of the book's Theorem 12.4), and the
  adjusted form has an expectation above the true variance.
- On a 400-node network with good overlap the raw interval covers 93 to
  95% and the adjusted one 98% (1,500 draws in development, 300 in the
  test).

With poor overlap the raw form covers 75 to 82%, the adjusted one 91 to
96%. That is the reason for the default and for the warning when an
exposure probability is below 0.01.

### `sp.iv.mte`

Three separate problems.

**Weights.** With `D = 1{U < P}`, the effect on the treated is the average
over units of the integral of the marginal effect from 0 to `P_i`, divided
by the mean of `P_i`. The function built the weight at each `u` from the
share of *treated* units with a propensity above `u`, and integrated over
the observed range of propensities only. On a design with marginal effect
`2 - 2u` and a probit first stage, over 100 samples of 4,000:

| | truth | old | new |
| --- | --- | --- | --- |
| ATE | 1.000 | 0.999 | 0.999 |
| ATT | 1.297 | 1.222 | 1.306 |
| ATU | 0.629 | 0.725 | 0.621 |

**Standard errors.** The variance of the ATE was assembled from the
variances of the polynomial coefficients as if they were uncorrelated.
They are strongly negatively correlated. The reported standard error was
0.075 against a sampling spread of 0.024. It is now a heteroskedasticity-
robust delta-method standard error with the full covariance (0.022, 95%
coverage). The ATT and ATU did not have standard errors and now do.

**Bootstrap.** The outcome and treatment arrays were trimmed to the
common-support sample and the instrument and covariate arrays were not.
A bootstrap draw indexed all four with the same indices, so whenever
trimming removed a row the outcome of one unit was paired with the
instrument of another. The standard errors were in the hundreds of
thousands on a unit-scale outcome.

The analytic standard errors condition on the fitted propensity score and
on the covariates. With a covariate in the model the sampling spread of
the ATE was 0.030 and the analytic standard error 0.024; the bootstrap
gave 0.029. The docstring says which is which.

## New functions

Seven, in four groups (the fourth, `sp.rd_optimized`, is described under
"Second round"). The guide `docs/guides/wager_causal_inference.md`
shows them in use.

**Interference.** `sp.interference_test` tests the sharp null and the
no-spillover null by holding a focal set fixed and permuting the other
treatments. It enumerates all assignments when there are no more than
`n_perm` of them. The test file recomputes the p-value from the definition
on a 14-node example and checks the rejection rate under a null with a
large heterogeneous direct effect (4% at a nominal 5%).

**Adaptive experiments.** `sp.bandit_allocate` gives the next assignment
probabilities; `sp.bandit_experiment` runs a rule sequentially;
`sp.adaptive_inference` gives intervals afterwards. The probability that
each arm is best is computed by quadrature, exactly for two arms. Over 600
experiments of 1,000 subjects with two equal arms and Thompson sampling:

| estimator | coverage, arm means | coverage, contrast |
| --- | --- | --- |
| sample mean | 0.907, 0.910 | 0.868 |
| adaptively weighted, no floor | 0.932, 0.948 | 0.928 |
| adaptively weighted, floor 0.02 | 0.955, 0.953 | 0.955 |
| augmented, floor 0.02 | 0.942, 0.952 | 0.938 |

The sample mean is biased downwards by 0.2 to 0.4 standard errors, as the
book's Figure 6.1 shows. No reference implementation was run against
these functions. The evidence is the formula check, the coverage runs and
the refusal cases.

**Balancing.** `sp.residual_balance` solves the balancing programme through
its dual, which has one variable per covariate, and finishes with an
active-set step that solves the optimality conditions exactly. `balanceHD`
(GitHub `swager/balanceHD`, version 1.0) was installed in a private
library and used as a black box. It is GPL-3 and its source was not read.

- Weights agree to 1e-6 at five configurations. The programme is strictly
  convex, so the solution is unique, and our objective value is below
  `balanceHD`'s at every one of them by 1e-10 to 1e-8. Against scipy's
  `trust-constr` our value was also the lower. The 2e-5 tolerance in the
  test is the reference solver's accuracy.
- The objective is `(1 - zeta) * sum(gamma^2) + zeta * imbalance^2`. This
  was read off the outputs: the imbalance falls as `zeta` rises.
- The augmented estimate uses a cross-validated elastic net on random
  folds in both packages and is not comparable number for number. On the
  one data set tried it was 0.996 (se 0.160) against 1.003 (se 0.190).
- With weights allowed to be negative and `zeta` near one the estimator
  equals `sp.lm_lin` to 1e-6, the identity noted in the book's chapter 7.

## A default that was changed

**The final stage of the R- and DR-learners.** On a 2,000-unit design with
a strong, simple effect function the root-mean-square error of the fitted
conditional effects was 0.33 (S), 0.35 (X), 0.58 (T), 0.75 (R) and 1.04
(DR), where a constant scores 1.03. Average effects were fine for all
five. The fitted values of the R- and DR-learners contained points at -24
and +27. Both learners regress a pseudo-outcome with heavy tails, and the
default final stage was gradient boosting with no minimum leaf size, so
it gave single extreme values their own leaves.

Six final-stage settings were compared over three seeds and two effect
functions (heterogeneous and constant):

| final stage | R, heterogeneous | DR, heterogeneous | R, constant | DR, constant |
| --- | --- | --- | --- | --- |
| default before | 0.762 | 1.006 | 0.724 | 1.055 |
| 20 per leaf | 0.457 | 0.594 | 0.443 | 0.569 |
| 20 per leaf, Huber loss | 0.458 | 0.451 | 0.433 | 0.438 |
| random forest, 10 per leaf | 0.562 | 0.671 | 0.503 | 0.625 |

The default is now 20 observations per leaf with squared-error loss. The
Huber loss does better still for the DR-learner in this table, but it
changes the estimand of the final regression from the conditional mean of
the pseudo-outcome, whose noise is skewed, so it was not made the default.
It remains one argument away (`cate_model=`). The S-, T- and X-learners
were left alone, as were average effects, which come from the doubly
robust scores and not from the final stage.

## Second round: optimized regression discontinuity

`sp.rd_optimized` implements section 8.2 for a sharp design with one
running variable.

**How it is computed.** The weights must sum to one on the treated side
and minus one on the control side and be orthogonal to the running
variable on each side. Given that, a Taylor expansion with integral
remainder writes the bias as an integral of the second derivative against
a piecewise linear function `G`, so the largest bias over `|mu''| <= M`
is `M` times the integral of `|G|` on each side. That integral is
computed exactly, for any weights. The optimal weights are proportional
to a least favourable function: the pair with a unit jump and curvature
at most `kappa` that has the smallest weighted sum of squares at the
data. With the function written through its level, slope and a piecewise
constant second derivative this is least squares with box constraints,
solved by scipy's active-set method. The ratio `kappa` is chosen by a
one-dimensional search on the exact criterion. A first attempt with a
quasi-Newton method on the multiplier form did not converge: the Hessian
integrates twice and is too badly conditioned.

**Evidence.**

- The bias formula reduces to RDHonest's closed form `M/2 |sum w x^2|` for
  local linear weights, to 1e-10 at six bandwidth and kernel combinations.
- The bound is attained: the function with second derivative
  `M sign(G)`, built by numerical integration and evaluated at the data,
  gives a realised bias equal to the reported one (0.13398059789 by three
  routes).
- On the same data, curvature and variance estimates the criterion is
  never above that of local linear regression at its optimal bandwidth.
- Against `optrdd` 1.0.2 (GitHub `swager/optrdd`, used as a black box,
  GPL-3, source not read) with a common variance, on a continuous design
  at two curvature bounds and a discrete one: estimates within 0.008
  standard errors, worst-case bias within 0.3%, weights correlated above
  0.9998. The two packages discretise the same programme differently, so
  this is agreement of estimators and not a bit-for-bit comparison. One
  difference is one-sided: `optrdd`'s weights are orthogonal to the
  running variable only to 5e-5, so their exact worst-case bias is
  unbounded and the figure it reports is that of its discretised problem.
- At the curvature bound (`mu'' = M` on one side, `-M` on the other, the
  hard case) the interval covered 96.0% over 300 samples of 600, against
  95.3% for `sp.rd_honest`, with the same mean half-length to four digits.
  With sixteen support points it covered 92.5% over 120 samples (Monte
  Carlo error 2.4 points).

**What it buys.** For a continuous running variable, very little over
`sp.rd_honest`, as the theory predicts (the triangular kernel is close to
optimal): on the Lee data the half-length is 2.996 against 3.002. The
function is there for discrete running variables and because its weights
and bias are exact, inspectable objects. A fit takes two to three seconds
with a continuous running variable and a tenth of a second with a
discrete one.

## Findings that are not bugs

**Over-identified CBPS is biased on a heterogeneous design, in R too.** On
a design with a logistic propensity in two of three covariates and an
effect that varies with the first, `sp.cbps` with its default
`variant='over'` was off by +0.095 (sampling sd 0.115) and covered 87%.
The exactly identified variant, entropy balancing, AIPW and Hajek IPW were
all unbiased. R `CBPS` gives the same numbers on the same 60 data sets:
mean 1.0848 against our 1.0850 for the over-identified variant, 1.0057
against 1.0056 for the exact one, largest difference 6.6e-4. So this is
the estimator, not the implementation. The default follows R and was left
alone. The guide recommends the exact variant.

**Adaptive weighting needs a probability floor.** Without one, Thompson
sampling retires the losing arm too fast for the central limit theorem,
and the weighted interval covers 88% for the worst of three arms. The book
says as much at the end of chapter 6. The function warns below `1/T`.

## Open

1. **Optimized regression discontinuity beyond the univariate sharp
   case.** `optrdd` also handles a two-dimensional running variable
   (geographic designs); fuzzy designs follow Noack and Rothe. Neither is
   in `sp.rd_optimized`.
2. **Doubly robust long-run value in a Markov decision process**
   (section 15.1) and **marginal policy effects** (15.2).
3. **`design='complete'` in `sp.network_exposure`.** The exposure
   probabilities are hypergeometric and easy. The variance theory in the
   book is for Bernoulli designs, since under complete randomization every
   pair of units is weakly dependent.
4. **Cross-fitting in `sp.doubly_robust`.** The outcome model is a random
   forest fitted and evaluated on the same rows. The bias this leaves was
   0.3 standard errors in our run.
5. **Contextual bandits**, and tests of the richer hypotheses in the
   chapter 11 hierarchy.

## Rerun

```bash
cd .claude/worktrees/wager-causal        # or main, once merged
export PYTHONPATH="$(pwd)/src"
pytest tests/test_wager_textbook_pass.py -q                       # 2 minutes
pytest tests/test_wager_textbook_pass.py -q -m slow               # coverage under Thompson sampling
pytest tests/reference_parity/test_residual_balance_balancehd.py -q
pytest tests/reference_parity/test_rd_optimized_optrdd.py -q
```

The `optrdd` fixture is regenerated the same way by
`_generate_rd_optimized_data.py` and `_generate_rd_optimized_R.R` (needs
`remotes::install_github("swager/optrdd")`; it is not on CRAN for R 4.5).

The `balanceHD` fixture is regenerated by
`tests/reference_parity/_fixtures/_generate_residual_balance_data.py`
followed by `_generate_residual_balance_R.R` (needs
`remotes::install_github("swager/balanceHD")` and `quadprog`).
