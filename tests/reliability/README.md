# Reliability studies

Simulations that ask how an inference method behaves where no reference
implementation can answer, with the design fixed before the first run.
They are separate from the Track B coverage experiments under
`tests/coverage_monte_carlo/`, which the JSS manuscript reads.

## Few clusters (`few_clusters.py`)

Rejection rate of a true null at a nominal 5% level, 2,000 replications
per cell (Monte Carlo standard error about 0.005 at 5%, 0.010 at 30%).
The regressor is a cluster-level treatment; the intra-cluster correlation
is 0.3. "Unbalanced" means one cluster holds half the sample. CR1, CR2
and CR3 are `sp.regress` with a t(G - 1) reference; the wild cluster
bootstrap is `sp.wild_cluster_bootstrap`.

| G | treated clusters | cluster sizes | CR1 | CR2 | CR3 | wild bootstrap |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 6 | half | balanced | 0.081 | 0.057 | 0.030 | 0.073 |
| 6 | half | unbalanced | 0.146 | 0.084 | 0.034 | 0.076 |
| 6 | two | balanced | 0.118 | 0.090 | 0.051 | 0.062 |
| 6 | two | unbalanced | 0.183 | 0.112 | 0.042 | 0.036 |
| 10 | half | balanced | 0.068 | 0.056 | 0.033 | 0.060 |
| 10 | half | unbalanced | 0.174 | 0.100 | 0.033 | 0.076 |
| 10 | two | balanced | 0.161 | 0.126 | 0.086 | 0.008 |
| 10 | two | unbalanced | 0.275 | 0.182 | 0.086 | 0.044 |
| 20 | half | balanced | 0.057 | 0.050 | 0.041 | 0.051 |
| 20 | half | unbalanced | 0.249 | 0.134 | 0.044 | 0.117 |
| 20 | two | balanced | 0.245 | 0.190 | 0.128 | 0.001 |
| 20 | two | unbalanced | 0.330 | 0.208 | 0.105 | 0.038 |
| 40 | half | balanced | 0.049 | 0.048 | 0.042 | 0.048 |
| 40 | half | unbalanced | 0.359 | 0.154 | 0.044 | 0.126 |
| 40 | two | balanced | 0.314 | 0.234 | 0.170 | 0.000 |
| 40 | two | unbalanced | 0.350 | 0.232 | 0.129 | 0.028 |

What the table says:

- With clusters of similar size and half of them treated, every method is
  at its nominal size by 40 clusters. At 6 clusters CR1 rejects 8% of the
  time, the wild cluster bootstrap 7%, CR2 6% and CR3 3%.
- With one cluster holding half the sample, more clusters do not help
  CR1 (15% at 6 clusters, 36% at 40) or CR2 (8% to 15%). The wild
  bootstrap also over-rejects there (12% to 13% at 20 and 40 clusters).
  CR3 stays between 3% and 5%.
- With two treated clusters the wild bootstrap almost never rejects in
  the balanced design (6% at 6 clusters, 0.8% at 10, 0.0% at 40), so a
  non-rejection carries no information, while CR1 rejects 12% to 31% of
  the time and CR3 5% to 17%. None of the four is reliable there.
- The reference distribution matters at small G. The first run of this
  study had CR2 and CR3 on a normal reference, as `sp.regress` then
  reported them: CR2 rejected 12% of the time at 6 balanced clusters and
  CR3 7%. With t(G - 1), which is what Stata reports for the same
  variances and what `sp.regress` now uses, they are at 6% and 3%.
- The warning `sp.regress` and `sp.panel` raised was keyed on the number
  of clusters alone (fewer than 30), and neither hard case triggers it at
  40 clusters. The next section is the answer to the first of them.

### What the number of clusters hides

Half the clusters treated, sizes unequal. The effective number of
clusters by size is `(sum n_g)^2 / sum n_g^2`.

| G | cluster sizes | effective clusters | CR1 | CR3 |
| ---: | --- | ---: | ---: | ---: |
| 40 | two_to_one | 36.0 | 0.060 | 0.051 |
| 40 | lognormal_0.5 | 31.6 | 0.061 | 0.046 |
| 40 | lognormal_1 | 18.1 | 0.100 | 0.056 |
| 60 | two_to_one | 54.0 | 0.056 | 0.048 |
| 60 | lognormal_0.5 | 47.1 | 0.056 | 0.046 |
| 60 | lognormal_1 | 25.9 | 0.084 | 0.059 |

CR1 is at 6% when the effective number is above 30 and at 8% to 10% when
it is 26 or 18, whatever the count. `sp.regress` therefore records
`model_info['n_clusters_effective']` and warns when there are 30 or more
clusters but fewer than 30 in effect, which covers the dominant-cluster
case above (effective number 3.8 at 40 clusters). For the few-treated
case it records `model_info['few_treated_clusters']` and warns when a
cluster-level 0/1 regressor has fewer than 10 clusters, and under a
quarter of them, on one side.

### Fixed-effects panels

The same question where the regressor varies within units: 40 units,
unit effects absorbed, AR(1) regressor and error, clustered by unit with
`sp.panel(method='fe', ssc='stata')`.

| panel | effective clusters | rejection rate |
| --- | ---: | ---: |
| every unit has 8 periods | 40.0 | 0.055 |
| one unit has 312 periods, 39 have 8 | 3.9 | 0.218 |

The warning is therefore raised by `sp.panel`, `sp.hdfe_ols` and
`sp.feols` as well, on one-way clustered fits. The few-treated
diagnostic is in `sp.regress` only.

Rerun with `python tests/reliability/few_clusters.py` (about twenty
minutes). `tests/test_reliability_few_clusters.py` recomputes one cell on
its first 60 replications and checks the statements above against the
stored file.

### Few units ever treated in a difference-in-differences

The fourth block: 40 units over 10 periods, unit and period effects,
AR(1) errors, and a treatment that switches on in period 6 for some of
the units with a true effect of zero. Rejection rate of the 5% test,
2,000 replications per row.

| treated units of 40 | two-way FE, clustered on unit | `sp.did_few_treated` |
| ---: | ---: | ---: |
| 1 | 0.749 | 0.028 |
| 2 | 0.306 | 0.070 |
| 5 | 0.101 | 0.097 |
| 10 | 0.059 | 0.139 |
| 20 | 0.058 | 0.000 |

- The cluster-robust test is unusable with one or two treated units and
  still rejects 10% with five. It is back at its nominal size with ten.
- The placebo test of `sp.did_few_treated` holds with one or two treated
  units and over-rejects beyond that: its placebo distribution is built
  from the treated units' paths alone, which is accurate only when they
  are a small share. With as many treated as controls it never rejects.
- Between three and nine treated units neither test is at its nominal
  size.

`sp.panel`, `sp.hdfe_ols`, `sp.feols` and `sp.regress` now report a 0/1
regressor that is ever 1 in fewer than 10 clusters (and under a quarter
of them) in `few_treated_clusters`, and warn. The hint offers
`sp.did_few_treated` only for one or two treated clusters.
`sp.did_few_treated` itself now warns when the treated groups are more
than a tenth of the controls.

## Extreme weights (`extreme_weights.py`)

Coverage of the 95% interval for a slope in `sp.regress(weights=)`,
2,000 replications per cell (Monte Carlo standard error about 0.005).
Weights are log-normal; "precision" errors have variance `1 / w` (what
analytic weights assume), "sampling" errors have the same variance for
every row (the survey reading). Kish n is `(sum w)^2 / sum w^2`.

| errors | n | sigma of log w | Kish n (median) | classical | HC1 | HC2 | HC3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| precision | 200 | 0 | 200 | 0.946 | 0.941 | 0.942 | 0.942 |
| precision | 200 | 1 | 82 | 0.943 | 0.938 | 0.941 | 0.946 |
| precision | 200 | 2 | 17 | 0.948 | 0.910 | 0.928 | 0.942 |
| precision | 1000 | 0 | 1000 | 0.946 | 0.945 | 0.945 | 0.945 |
| precision | 1000 | 1 | 382 | 0.954 | 0.955 | 0.956 | 0.957 |
| precision | 1000 | 2 | 52 | 0.943 | 0.930 | 0.935 | 0.943 |
| sampling | 200 | 0 | 200 | 0.945 | 0.944 | 0.946 | 0.947 |
| sampling | 200 | 1 | 82 | 0.783 | 0.929 | 0.934 | 0.941 |
| sampling | 200 | 2 | 17 | 0.471 | 0.842 | 0.895 | 0.939 |
| sampling | 1000 | 0 | 1000 | 0.950 | 0.953 | 0.953 | 0.953 |
| sampling | 1000 | 1 | 383 | 0.770 | 0.938 | 0.940 | 0.942 |
| sampling | 1000 | 2 | 53 | 0.340 | 0.878 | 0.911 | 0.932 |

What the table says:

- With equal weights every variance covers at its nominal level.
- The classical weighted variance, which is what `weights=` alone gives,
  is right under precision weights at any dispersion and wrong under
  sampling weights: 77% to 78% coverage when the Kish size is about 0.4
  of n, 34% to 47% when it is 0.05 to 0.09 of n. The two cases cannot be
  told apart from the data; the user has to know what the weights are.
- HC1 covers 93% to 95% when the Kish size is 80 or more and falls to
  88% at 53 and 84% at 17. HC2 is in between. HC3 stays at 93% to 94%
  throughout.

`sp.regress` therefore records `model_info['n_effective_weights']` and
warns in two cases: classical standard errors with a Kish ratio under
0.5, and HC0 / HC1 / HC2 with a Kish size under 100 that is also under
half of n.

Rerun with `python tests/reliability/extreme_weights.py` (a few minutes).
`tests/test_reliability_extreme_weights.py` recomputes one cell on its
first 60 replications and checks the statements above.

## A discrete running variable (`rd_mass_points.py`)

Coverage of the robust 95% interval of `sp.rdrobust` for a jump of 1,
1,000 replications per cell (Monte Carlo standard error about 0.007 at
95%). The running variable is uniform on (-1, 1), continuous or rounded
to a number of equally spaced support points on each side; the regression
function is a smooth cubic. "Refused" counts fits that raised a
`NumericalInstability` or `DataInsufficient` error; coverage is over the
fits that ran.

| n | support points per side | default (`adjust`) | `masspoints='off'` | clustered on the support points |
| ---: | --- | ---: | ---: | ---: |
| 1000 | continuous | 0.944 | 0.944 | n/a |
| 1000 | 50 | 0.955 | 0.955 | 0.856 |
| 1000 | 20 | 0.955 | 0.956 | 0.691 |
| 1000 | 10 | 0.888 | 0.949 (13 refused) | 0.631 |
| 1000 | 5 | 0.891 (1 refused) | 0.733 (708 refused) | 0.320 |
| 4000 | continuous | 0.943 | 0.943 | n/a |
| 4000 | 50 | 0.941 | 0.940 | 0.844 |
| 4000 | 20 | 0.937 | 0.936 | 0.648 |
| 4000 | 10 | 0.689 | 0.919 (382 refused) | 0.451 |
| 4000 | 5 | 0.710 | 0.604 (598 refused) | 0.224 |

What the table says:

- The default holds its coverage with a continuous running variable and
  with 50 or 20 support points a side.
- At 10 and 5 support points a side the default covers 89% at n = 1000
  and 69% to 71% at n = 4000. The mass-point adjustment floors the pilot
  bandwidth at the tenth distinct value from the cutoff, which at 10
  points a side is the whole side: the conventional estimate is biased by
  0.21 to 0.26 on a jump of 1, and more data narrows the interval around
  the biased value. This is the reference behaviour (R `rdrobust` does
  the same); `sp.rdrobust` already warns when the running variable has
  fewer than 30 distinct values and points at `sp.rd_discrete`.
- Clustering on the support points makes it worse at every level: 84% to
  86% at 50 points a side, 65% to 69% at 20, 22% to 32% at 5.
  `sp.rdrobust` now warns when each cluster holds a single value of the
  running variable.
- `masspoints='off'` treats the data as continuous. With 5 points a side
  most fits are refused, and the ones that run have intervals thousands
  of units long. Before this study those refusals were a
  `TypeError: float() argument must be ... not 'complex'` and, in a few
  cases, a NaN interval returned without an error.

Rerun with `python tests/reliability/rd_mass_points.py` (about half an
hour). `tests/test_reliability_rd_mass_points.py` recomputes one cell on
its first 40 replications and checks the statements above.

## Extreme weights beyond OLS (`extreme_weights_models.py`)

The same question as `extreme_weights.py`, for a weighted fixed-effects
regression (`sp.panel(method='fe')`, unit-level weights, 5 periods) and
a weighted Poisson regression (`sp.poisson`). The errors do not depend
on the weight, as with sampling weights. Coverage of the 95% interval
for the slope, 2,000 replications per cell.

Fixed effects (Kish size is over units):

| units | sigma of log weight | Kish size | classical | robust | cluster on unit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0 | 50 | 0.956 | 0.957 | 0.951 |
| 50 | 1 | 23 | 0.800 | 0.932 | 0.908 |
| 50 | 2 | 7 | 0.542 | 0.889 | 0.777 |
| 200 | 0 | 200 | 0.945 | 0.945 | 0.944 |
| 200 | 1 | 82 | 0.779 | 0.954 | 0.936 |
| 200 | 2 | 17 | 0.445 | 0.924 | 0.871 |

Poisson (Kish size is over observations):

| n | sigma of log weight | Kish size | classical | robust |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 0 | 200 | 0.960 | 0.953 |
| 200 | 1 | 82 | 0.679 | 0.914 |
| 200 | 2 | 17 | 0.207 | 0.840 |
| 1000 | 0 | 1000 | 0.951 | 0.947 |
| 1000 | 1 | 382 | 0.652 | 0.943 |
| 1000 | 2 | 53 | 0.143 | 0.882 |

What the tables say:

- The default variance fails under sampling weights in both models, as
  it does for OLS: 45% to 80% with fixed effects, 14% to 68% for
  Poisson. It is the right variance only when the weights are precisions
  (linear model) or frequencies (Poisson).
- The robust variance is right in large effective samples and short in
  small ones: Poisson covers 84% at a Kish size of 17 and 88% at 53.
- Clustering on the unit does not repair unit-level weights. Coverage
  follows the number of units the weights leave in effect: 78% at 7, 87%
  at 17, 91% at 23, nominal from 50 up. The count of clusters (50 or
  200) and the size-based effective count say nothing here, because the
  clusters are equal in size.

`sp.panel`, `sp.hdfe_ols`, `sp.feols`, `sp.poisson` and `sp.ppmlhdfe` now
record `n_effective_weights`, and `n_clusters_effective_weights` for a
one-way clustered fit, and warn in the three cases above (classical
variance with a Kish ratio under 0.5; robust variance with a Kish size
under 100; fewer than 30 weight-effective clusters). `sp.nbreg` and
`sp.fast.*` were not simulated and carry no warning.

Rerun with `python tests/reliability/extreme_weights_models.py` (about
eight minutes).

## Staggered adoption on an unbalanced panel (`unbalanced_panel.py`)

Bias and coverage of the 95% interval for the overall ATT (truth 1 in
every treated cell), 1,000 replications per cell. Eight periods,
adoption in periods 4 and 6, AR(1) errors within unit. Entries are
bias / coverage.

| missing cells | units | `sp.callaway_santanna` | with `allow_unbalanced_panel=True` | `sp.did_imputation` | two-way FE, clustered |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 100 | +0.00 / 0.945 | +0.00 / 0.945 | -0.00 / 0.946 | +0.00 / 0.966 |
| none | 400 | +0.00 / 0.938 | +0.00 / 0.938 | +0.00 / 0.942 | +0.00 / 0.954 |
| at random (30%) | 100 | +0.02 / 0.938 | +0.01 / 0.939 | +0.02 / 0.947 | +0.02 / 0.969 |
| at random (30%) | 400 | +0.00 / 0.951 | +0.00 / 0.946 | +0.00 / 0.954 | -0.00 / 0.969 |
| treated units with a high level leave | 100 | -0.00 / 0.936 | -0.41 / 0.501 | -0.01 / 0.935 | -0.01 / 0.953 |
| treated units with a high level leave | 400 | +0.00 / 0.943 | -0.40 / 0.030 | +0.00 / 0.947 | +0.00 / 0.965 |
| treated cells with a low outcome | 100 | +0.38 / 0.462 | +0.50 / 0.274 | +0.44 / 0.272 | +0.42 / 0.266 |
| treated cells with a low outcome | 400 | +0.38 / 0.023 | +0.50 / 0.000 | +0.44 / 0.000 | +0.42 / 0.000 |

What the table says:

- Estimators that compare a unit with itself (the Callaway-Sant'Anna
  default, imputation, two-way fixed effects) keep their coverage when
  cells are missing at random and when units leave according to their
  level, which the unit effect absorbs.
- `allow_unbalanced_panel=True` keeps every observed row by comparing
  group means. That is as good as the default when cells are missing at
  random, and biased by 40% of the effect when treated units with a
  high level leave: the group mean falls for a reason that is not the
  treatment. Coverage is 50% with 100 units and 3% with 400. This is a
  property of the estimator (R `did` computes the same thing), not a
  bug; the option's docstring and the unbalanced-panel warning now say
  what it assumes, and the result records it in
  `model_info['unbalanced_assumption']`.
- When cells go missing according to the outcome itself, every
  estimator is biased by 38% to 50% of the effect and none covers. No
  choice among them repairs it.

No fit was refused in any cell. Rerun with
`python tests/reliability/unbalanced_panel.py` (about ten minutes).

## DML as the learner changes (`dml_learners.py`)

Bias and coverage of the 95% interval of `sp.dml(model='plr')` for a
treatment coefficient of 0.5, 300 replications per cell (Monte Carlo
standard error about 0.013 at 95%). Ten covariates; the confounding is
either linear or a smooth nonlinear function shared by treatment and
outcome. The last column is `sp.dml_model_averaging` at its defaults.
Entries are bias / coverage.

| confounding | n | OLS | Lasso | random forest | gradient boosting (the default) | model averaging |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | 500 | -0.00 / 0.953 | -0.00 / 0.950 | -0.01 / 0.930 | -0.02 / 0.903 | -0.00 / 0.950 |
| linear | 2000 | -0.00 / 0.943 | +0.00 / 0.947 | -0.01 / 0.913 | -0.01 / 0.917 | +0.00 / 0.943 |
| nonlinear | 500 | +0.72 / 0.000 | +0.72 / 0.000 | +0.15 / 0.433 | +0.06 / 0.867 | +0.09 / 0.727 |
| nonlinear | 2000 | +0.72 / 0.000 | +0.72 / 0.000 | +0.05 / 0.667 | +0.02 / 0.900 | +0.02 / 0.870 |

What the table says:

- No learner is right in both rows. Linear learners are exact when the
  confounding is linear and miss by 0.72 on a coefficient of 0.5 when it
  is not, at either sample size: that bias is not a small-sample matter.
- Tree learners pay a small regularisation bias when the truth is
  linear (90% to 93%) and are slow to shed a larger one when it is not.
  The forest covers 43% at n = 500 and 67% at n = 2000; boosting, the
  `sp.dml` default, 87% and 90%.
- Model averaging is the one choice that is never badly wrong: nominal
  under linear confounding, 73% and 87% under nonlinear. It is not a
  cure, and at n = 500 it does worse than boosting alone.
- The reported standard error matches the spread of the estimates in
  every cell (ratio 0.8 to 1.2). The intervals fail because of bias in
  the nuisance fits, which no standard error reports. Comparing the
  estimate across learners is the available check: in the nonlinear
  rows they disagree by many standard errors.

This complements the Track B mechanism experiment
(`tests/coverage_monte_carlo/mechanisms/dml_plr_learners.py`), on which
the note in `sp.dml`'s result rests; that note is unchanged. Rerun with
`python tests/reliability/dml_learners.py` (about an hour and a half).
