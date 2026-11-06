# Glossary

Terms used across the StatsPAI documentation, function cards and result
objects. Each entry says what the word means here and where to read more.
Statistical terms follow the usual textbook meaning; the entries only pin
down the choices StatsPAI makes.

## Estimands and effects

**Estimand.** The quantity a method targets, defined before any data are
seen: for example the average effect of treatment on the treated in a
given population. An estimator is the rule that turns data into a number
for it, and the estimate is that number. Two estimators of the same
estimand can be compared; two estimates of different estimands cannot.
Result objects name their estimand in `result.estimand`.

**ATE.** Average treatment effect, `E[Y(1) - Y(0)]` over the whole
population, where `Y(1)` and `Y(0)` are the potential outcomes with and
without treatment.

**ATT.** Average treatment effect on the treated, `E[Y(1) - Y(0) | D = 1]`.
Most difference-in-differences and synthetic-control estimators target an
ATT or a group-time ATT, `ATT(g, t)`, for the cohort first treated in
period `g` at period `t`.

**ATC / ATU.** Average effect on the controls (untreated).

**ATO.** Average effect in the overlap population, the target of overlap
weights.

**CATE.** Conditional average treatment effect, `E[Y(1) - Y(0) | X = x]`.
Meta-learners, causal forests and DML with heterogeneity estimate it.

**LATE.** Local average treatment effect: the average effect for compliers,
the units whose treatment the instrument moves. A just-identified IV
estimate has this meaning only under monotonicity; see
`sp.iv.iv_diag` for the checks.

**ITT.** Intention-to-treat effect: the effect of assignment, not of
treatment received.

**Identifying assumption.** The condition under which the estimand can be
recovered from the data (parallel trends, exclusion, continuity at the
cutoff, unconfoundedness). Function cards list them under `assumptions`.

## Inference

**Standard error (SE).** The estimated standard deviation of the
estimator's sampling distribution. How it is computed is chosen with
`vce=`; see [Standard errors](standard_errors.md).

**Heteroskedasticity-robust (HC0 to HC3).** Sandwich variance estimators
that do not assume a constant error variance. HC1 is Stata's `robust`.

**Cluster-robust (CR1, CR2, CR3).** Sandwich variance estimators that let
errors be correlated within clusters and independent across them. CR1 is
Stata's `vce(cluster ...)` with its small-sample factor. CR2 and CR3 are
bias-reduced versions for few clusters. Multiway clustering allows
correlation along two or more dimensions at once.

**Small-sample correction (ssc).** The degrees-of-freedom factor applied
to a variance estimate. Stata, R `fixest` and R `sandwich` use different
defaults, which is the most common reason two packages report different
SEs for the same fit. `result.inference_rule()` states the rule a result
used.

**Wild cluster bootstrap.** A resampling test that flips the sign of
cluster residuals. It keeps size close to nominal with few clusters.

**Confidence interval (CI).** An interval that covers the true value in
the stated share of repeated samples (95% by default). Coverage is a
property of the procedure, which is what the Track B studies measure.

**p-value.** The probability, under the null hypothesis, of a test
statistic at least as extreme as the one observed.

## Evidence and validation

**Evidence grade (T1 to T4, S).** How a function's numbers were checked.

- **T1**: recovers a known truth (a closed form or a simulated design
  with a known parameter). No external reference is involved.
- **T2**: strict cross-language parity. StatsPAI and an R or Stata
  reference run on the same CSV bytes and agree within a relative error of
  `1e-6` on the estimate and on the SE. Only T2 is described as "aligned
  with R / Stata".
- **T3**: seed-replicated equivalence for a stochastic estimator. Both
  sides run many seeds on fixed data and the seed distributions are shown
  equivalent by a pre-registered TOST test.
- **T4**: a documented difference. Either the two sides compute a
  documented different quantity, or the references themselves disagree.
- **S**: a stochastic screen. A random comparison that falls short of T3
  (one or a few seeds). It is reported, not graded.

**Parity grade.** The label the parity index gives each function, shown
by `sp.parity_status(name)`:

- **bit-exact**: headline relative error at most `1e-6` against a named
  R or Stata reference.
- **aligned**: agrees within a looser tolerance that is registered in
  advance together with its mechanism.
- **analytical-only**: recovers a known data-generating truth or a
  closed-form identity, with no cross-language comparison.
- **external-replication**: reproduces numbers published in a paper.
- **unverified**: registered, with no parity evidence attached yet.

The full matrix is on the [parity page](../parity.md).

**Validation status.** The evidence tier of a function, in
`sp.describe_function(name)["validation_status"]`:

- **certified**: compared with a named external reference implementation
  on identical inputs, within a pre-registered tolerance.
- **validated**: known-truth recovery or replication of published
  numbers, without a same-byte cross-language comparison.
- **api_stable**: the public API is stable and unit-tested, but there is
  no numerical validation claim.
- **experimental**: the method or its API may still change.
- **deprecated**: scheduled for removal; the replacement is in
  `MIGRATION.md`.

The tier is derived from the committed evidence, not assigned by hand.
See [Stability and validation tiers](stability.md).

**Stability.** The API life-cycle of a function, separate from its
evidence: `stable`, `experimental` or `deprecated`. The 32 core entry
points are **frozen**: their signatures are snapshot-tested.

**Track A.** The same-byte R / Stata parity harness in `tests/r_parity/`
and `tests/stata_parity/`. Each module fits one estimator in Python, R
and Stata on one committed CSV and compares the numbers.

**Track B.** Monte Carlo coverage studies in
`tests/coverage_monte_carlo/`: `B = 1000` replications per known-truth
design, checking that 95% intervals cover at their nominal rate.

**Track C.** Timing comparisons against the reference implementation on
the same task, in `tests/perf/`.

**Golden file.** A committed reference result (`*_R.json`,
`*_Stata.json`) that parity tests compare against. Golden files are only
regenerated by rerunning the reference, never edited by hand.

**Provenance trace.** A record binding a Track A result to the hashes of
the StatsPAI source files that produced it, so a stale result is caught.

## Data and design

**Panel, unit, period.** Repeated observations of the same units (`id=`)
over time (`time=`). A balanced panel has every unit in every period.

**Cohort.** In staggered designs, the units first treated in the same
period.

**Never-treated / not-yet-treated.** The two usual comparison groups in
staggered difference-in-differences.

**Running variable, cutoff, bandwidth.** In regression discontinuity, the
score that assigns treatment, the threshold, and the window around it
used for estimation.

**Donor pool.** In synthetic control, the untreated units whose weighted
combination forms the counterfactual.

**Listwise deletion.** Dropping every row with a missing value in any
variable the model uses. StatsPAI does this by default and reports how
many rows it dropped; `inf` values are refused.
