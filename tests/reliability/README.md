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
| 1000 | 20 | 0.956 | 0.956 | 0.657 |
| 1000 | 10 | 0.942 (3 refused) | 0.948 (21 refused) | 0.410 |
| 1000 | 5 | 0.903 (969 refused) | 0.903 (969 refused) | 0.065 (969 refused) |
| 4000 | continuous | 0.943 | 0.943 | n/a |
| 4000 | 50 | 0.941 | 0.940 | 0.844 |
| 4000 | 20 | 0.931 | 0.936 | 0.610 |
| 4000 | 10 | 0.938 (4 refused) | 0.957 (770 refused) | 0.466 |
| 4000 | 5 | 0.875 (976 refused) | 0.875 (976 refused) | 0.042 (976 refused) |

What the table says:

- The default holds its coverage with a continuous running variable and
  with 50 or 20 support points a side.
- At 10 support points a side the default covers 94% at both sample
  sizes. The mass-point adjustment floors only the pilot bandwidth at the
  tenth distinct value from the cutoff, as R `rdrobust` does. Until the
  2026-10-20 fix StatsPAI floored the final bandwidths too, which held h
  at the whole side, biased the conventional estimate by 0.21 to 0.26 on
  a jump of 1, and pulled coverage down to 69% to 89%.
- At 5 support points a side the bandwidth that matches R is narrower
  than the spacing of the support points, so the window holds fewer
  distinct values than the local polynomial has coefficients. Those fits
  are refused (`DataInsufficient` or `NumericalInstability`, with a hint
  to `sp.rd_discrete` and `sp.rdrandinf`) instead of returning a
  pseudo-inverse number: 969 of 1,000 at n = 1000 and 976 at n = 4000.
  `sp.rdrobust` also warns when the running variable has fewer than 30
  distinct values and points at `sp.rd_discrete`.
- Clustering on the support points makes it worse at every level: 84% to
  86% at 50 points a side, 61% to 66% at 20, 41% to 47% at 10.
  `sp.rdrobust` warns when each cluster holds a single value of the
  running variable.
- `masspoints='off'` treats the data as continuous. At 10 points a side
  it refuses 21 fits at n = 1000 and 770 at n = 4000; at 5 points it
  refuses the same fits as the default.

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
`sp.fast.*` are covered by the next study.

Rerun with `python tests/reliability/extreme_weights_models.py` (about
eight minutes).

## Extreme weights in the fast and negative-binomial entry points (`extreme_weights_fast.py`)

The same question for `sp.fast.feols`, `sp.fast.fepois` and `sp.nbreg`.
A regressor with a true coefficient of zero is tested at 5% with the
package's own p-value; the entry is the share of 2,000 replications
that do not reject, which is the coverage of the 95% interval at the
truth.

`sp.fast.feols`, unit fixed effects, unit-level weights:

| units | sigma of log weight | Kish size | `iid` | `hc1` | `cr1` on unit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0 | 50 | 0.953 | 0.948 | 0.946 |
| 50 | 1 | 23 | 0.806 | 0.930 | 0.917 |
| 50 | 2 | 7 | 0.542 | 0.870 | 0.793 |
| 200 | 0 | 200 | 0.949 | 0.950 | 0.951 |
| 200 | 1 | 81 | 0.807 | 0.952 | 0.949 |
| 200 | 2 | 17 | 0.427 | 0.907 | 0.859 |

`sp.fast.fepois`, the same panel with a Poisson outcome:

| units | sigma of log weight | Kish size | `iid` | `hc1` | `cr1` on unit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0 | 50 | 0.972 | 0.943 | 0.943 |
| 50 | 1 | 23 | 0.753 | 0.922 | 0.908 |
| 50 | 2 | 7 | 0.314 | 0.882 | 0.796 |
| 200 | 0 | 200 | 0.974 | 0.957 | 0.958 |
| 200 | 1 | 81 | 0.730 | 0.946 | 0.936 |
| 200 | 2 | 18 | 0.223 | 0.900 | 0.863 |

`sp.nbreg`, observation weights:

| n | sigma of log weight | Kish size | classical | robust |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 0 | 200 | 0.951 | 0.942 |
| 200 | 1 | 81 | 0.690 | 0.913 |
| 200 | 2 | 17 | 0.221 | 0.841 |
| 1000 | 0 | 1000 | 0.947 | 0.945 |
| 1000 | 1 | 385 | 0.656 | 0.939 |
| 1000 | 2 | 52 | 0.140 | 0.879 |

The three findings of the earlier weight studies repeat here. The
default variance fails under sampling weights (22% to 81%), the robust
one is short when the effective sample is small, and the clustered one
follows the number of clusters the weights leave in effect (79% at 7,
86% at 17, 92% at 23). The `sp.fast.feols` `cr1` column was rerun after
its p-values moved from t(N - K) to t(G - 1) (option scan round 6); the
`iid` and `hc1` columns reproduced exactly. The three entry points now carry the same
diagnostics as the others; the fast result objects hold them in
`weight_info`. One negative-binomial fit of 84,000 did not converge (one
observation held 80% of the weight); `sp.nbreg` said so and reported no
standard errors, and the fit counts as a miss.

## Extreme weights in the binary-outcome and IV entry points (`extreme_weights_binary_iv.py`)

The last four regression entry points that take `weights=`. A regressor
with a true coefficient of zero is tested at 5% with the package's own
p-value (for `sp.iv` it is the endogenous regressor, with a strong
instrument); the entry is the share of 2,000 replications that do not
reject.

| entry point | n | sigma of log weight | Kish size | default | robust |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sp.logit` | 200 | 0 | 200 | 0.947 | 0.951 |
| `sp.logit` | 200 | 1 | 82 | 0.679 | 0.936 |
| `sp.logit` | 200 | 2 | 17 | 0.201 | 0.855 |
| `sp.logit` | 1000 | 0 | 1000 | 0.946 | 0.948 |
| `sp.logit` | 1000 | 1 | 382 | 0.673 | 0.937 |
| `sp.logit` | 1000 | 2 | 53 | 0.160 | 0.897 |
| `sp.probit` | 200 | 0 | 200 | 0.948 | 0.946 |
| `sp.probit` | 200 | 1 | 81 | 0.675 | 0.928 |
| `sp.probit` | 200 | 2 | 17 | 0.205 | 0.814 |
| `sp.probit` | 1000 | 0 | 1000 | 0.957 | 0.957 |
| `sp.probit` | 1000 | 1 | 382 | 0.650 | 0.940 |
| `sp.probit` | 1000 | 2 | 53 | 0.149 | 0.879 |
| `sp.glm` (binomial) | 200 | 0 | 200 | 0.948 | 0.945 |
| `sp.glm` (binomial) | 200 | 1 | 82 | 0.664 | 0.921 |
| `sp.glm` (binomial) | 200 | 2 | 17 | 0.197 | 0.835 |
| `sp.glm` (binomial) | 1000 | 0 | 1000 | 0.948 | 0.946 |
| `sp.glm` (binomial) | 1000 | 1 | 382 | 0.641 | 0.935 |
| `sp.glm` (binomial) | 1000 | 2 | 52 | 0.146 | 0.890 |
| `sp.iv` | 200 | 0 | 200 | 0.952 | 0.947 |
| `sp.iv` | 200 | 1 | 82 | 0.787 | 0.917 |
| `sp.iv` | 200 | 2 | 17 | 0.503 | 0.841 |
| `sp.iv` | 1000 | 0 | 1000 | 0.947 | 0.948 |
| `sp.iv` | 1000 | 1 | 382 | 0.785 | 0.951 |
| `sp.iv` | 1000 | 2 | 53 | 0.389 | 0.894 |

The same three findings once more. Under dispersed sampling weights the
default variance covers 15% to 68% for the likelihood models and 39% to
79% for two-stage least squares; the robust one is nominal in large
effective samples and 81% to 90% in small ones. `sp.logit`, `sp.probit`,
`sp.glm` and `sp.iv` now record `n_effective_weights` and
`n_clusters_effective_weights` and warn as the other entry points do.
With these four, every regression entry point that takes `weights=` and
reports a model-based variance by default carries the diagnostic.

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

## Split-sample evaluation of fixed-effects forests (`forest_split_evaluation.py`)

Share of 95% intervals that exclude zero, 400 replications per cell
(Monte Carlo standard error 0.011 at 5%). A staggered panel, 8 periods,
adoption selected on the unit effect, `tau = 0.3 + b z`; the population
AUTOC of the ideal ranking is `0.9032 b` and QINI `0.2821 b`.

| quantity | `b` | method | mean | excludes zero |
| --- | ---: | --- | ---: | ---: |
| AUTOC | 0 | `sp.rate`, forest's own ranking | -0.028 | 0.160 |
| AUTOC | 0 | `sp.rate_split` | -0.000 | 0.005 |
| QINI | 0 | `sp.rate`, forest's own ranking | -0.006 | 0.122 |
| QINI | 0 | `sp.rate_split` | +0.000 | 0.000 |
| AUTOC (ideal 0.452) | 0.5 | `sp.rate_split` | 0.383 | 1.000 |
| QINI (ideal 0.141) | 0.5 | `sp.rate_split` | 0.128 | 1.000 |
| policy gain | 0 | `sp.forest_policy_tree` | +0.008 | 0.000 |
| policy gain (oracle 0.319) | 0.8 | `sp.forest_policy_tree` | 0.307 | 1.000 |

What the table says:

- Ranking the cells with the forest that was grown on them rejects a true
  null three times too often and reports a negative AUTOC where the truth
  is zero.
- Splitting the units removes that. With the default 21 splits the test is
  conservative (the aggregation over splits is built to be), not exact.
- Under real heterogeneity the split estimate is below the ideal ranking's
  value because it measures the rule that was fitted on half the units.

## A cross-sectional forest grading its own ranking (`rate_cross_section.py`)

No treatment-effect heterogeneity at all (constant effect 1, n = 800, four
covariates, randomised treatment), so every ranking has population RATE 0.
`sp.rate(cf)` ranks units by the forest's own out-of-bag predictions and
scores them with its own doubly robust scores; `sp.rate_split` fits the
ranking on one half and scores it on the other.

| method | replications | mean AUTOC | rejection rate at 5% |
| --- | ---: | ---: | ---: |
| `sp.rate(cf)`, own ranking | 400 | -0.104 | 0.360 |
| `sp.rate_split(cf, n_splits=5)` | 100 | 0.009 | 0.000 |

The own-ranking test rejects a true null seven times as often as it
should, and its estimate is pulled below zero. The split version holds
its size (conservatively, as VEIN aggregation does). `sp.rate` now warns
on this path, as it already did for forests with fixed effects.

## Standard-error calibration screen (`se_screen/`)
Parity work cannot see a wrong standard error where no reference exists.
This screen asks one question of every estimator call it lists: on data
that satisfy the call's stated assumptions, with a known true parameter,
does the reported 95% interval cover 95% of the time and does the
reported standard error match the sampling spread of the estimate?

One script per family (`se_screen/<family>.py`) on a shared harness
(`se_screen/_harness.py`). Designs are fixed in the script before the
first run. Each row is 400 independent draws (seeds
`SeedSequence(base).spawn(B)`, `base` derived from the row name) unless
the row says otherwise. Three numbers per row:

- coverage of the interval the estimator reports;
- SE/SD, the mean reported standard error over the Monte Carlo standard
  deviation of the estimate;
- bias/SD, the mean error of the estimate in the same units.

A row is flagged when coverage is outside [0.918, 0.985], SE/SD is
outside [0.88, 1.12] or |bias/SD| is above 0.25. Flagged rows are rerun
at 1,000 draws and every confirmed flag carries a mechanism note in the
results file. Rows screened on 200 draws (slow estimators) use the Wilson
99% band for coverage, [0.894, 0.985], and [0.83, 1.17] for SE/SD.
Weak-instrument robust sets and randomization intervals report no
standard error; those rows are coverage only.

    python tests/reliability/se_screen/regression.py              # all rows
    python tests/reliability/se_screen/regression.py --only tobit # some rows
    python tests/reliability/se_screen/regression.py --confirm    # flagged rows, B = 1000
    python tests/reliability/se_screen/regression.py --table      # the table below

`tests/test_reliability_se_screen.py` reads the committed
`<family>_results.json` files (the causal families are in `se_screen/causal/`, with their own test). It asserts that every unflagged row is in
band, that every flagged row has a mechanism note, and that one row per
family reproduces its stored first 20 draws.

### Regression family (`se_screen/regression.py`)

Correctly specified likelihood and moment models, n = 1,000 unless the
row's design says otherwise, target the slope on `x1` (0.5). Robust rows
use heteroskedastic or overdispersed data, cluster rows 100 clusters of 10
with the marginal model kept exact. The second half of the table is the
options: robust, cluster, weights, exposure and alternative variance
estimators.

| row | call | n | B | coverage | SE/SD | bias/SD | flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `regress/default` | `regress("y ~ x1 + x2")` | 1000 | 400 | 0.950 | 1.00 | +0.04 |  |
| `regress/hc1` | `regress(..., robust="hc1")` | 1000 | 400 | 0.940 | 1.02 | +0.08 |  |
| `regress/hc3` | `regress(..., robust="hc3")` | 1000 | 400 | 0.950 | 0.99 | -0.00 |  |
| `regress/cluster` | `regress(..., cluster="g")` | 1000 | 400 | 0.945 | 0.98 | -0.05 |  |
| `logit/default` | `logit("y ~ x1 + x2")` | 1000 | 400 | 0.940 | 0.98 | +0.09 |  |
| `logit/robust` | `logit(..., robust="robust")` | 1000 | 400 | 0.950 | 0.99 | -0.03 |  |
| `logit/cluster` | `logit(..., cluster="g")` | 1000 | 400 | 0.963 | 1.01 | +0.07 |  |
| `probit/default` | `probit("y ~ x1 + x2")` | 1000 | 400 | 0.935 | 0.98 | +0.06 |  |
| `probit/robust` | `probit(..., robust="robust")` | 1000 | 400 | 0.953 | 1.04 | -0.01 |  |
| `cloglog/default` | `cloglog("y ~ x1 + x2")` | 1000 | 400 | 0.920 | 0.94 | +0.02 |  |
| `poisson/default` | `poisson("y ~ x1 + x2")` | 1000 | 400 | 0.930 | 0.96 | +0.03 |  |
| `poisson/robust` | `poisson(..., robust="robust")` | 1000 | 400 | 0.935 | 0.96 | -0.01 |  |
| `nbreg/default` | `nbreg("y ~ x1 + x2")` | 1000 | 400 | 0.963 | 1.00 | -0.03 |  |
| `nbreg/robust` | `nbreg(..., robust="robust")` | 1000 | 400 | 0.948 | 0.99 | +0.00 |  |
| `glm/gamma-log` | `glm(..., family="gamma", link="log")` | 1000 | 400 | 0.940 | 0.95 | +0.05 |  |
| `glm/gamma-log-robust` | `glm(..., family="gamma", link="log", robust="robust")` | 1000 | 400 | 0.960 | 1.03 | -0.05 |  |
| `glm/binomial` | `glm(..., family="binomial")` | 1000 | 400 | 0.943 | 0.98 | +0.03 |  |
| `glm/poisson-overdispersed` | `glm(..., family="poisson", robust="robust")` | 1000 | 400 | 0.948 | 1.01 | -0.08 |  |
| `tobit/default` | `tobit(formula="y ~ x1 + x2", ll=0)` | 1000 | 400 | 0.943 | 0.98 | -0.01 |  |
| `tobit/robust` | `tobit(..., vce="robust")` | 1000 | 400 | 0.940 | 0.92 | -0.04 |  |
| `truncreg/default` | `truncreg(formula="y ~ x1 + x2", ll=0)` | 1000 | 400 | 0.950 | 0.97 | +0.03 |  |
| `intreg/default` | `intreg(y_lower=, y_upper=, x=)` | 1000 | 400 | 0.940 | 0.97 | +0.09 |  |
| `heckman/twostep` | `heckman(..., method="twostep")` | 1000 | 400 | 0.940 | 0.99 | +0.04 |  |
| `heckman/mle` | `heckman(..., method="mle")` | 1000 | 400 | 0.958 | 1.07 | -0.10 |  |
| `heckprobit/default` | `heckprobit(y, x, select, z)` | 2000 | 400 | 0.940 | 0.97 | +0.13 |  |
| `ologit/default` | `ologit("y ~ x1 + x2")` | 1000 | 400 | 0.950 | 1.01 | +0.02 |  |
| `oprobit/default` | `oprobit("y ~ x1 + x2")` | 1000 | 400 | 0.925 | 0.93 | +0.01 |  |
| `gologit/parallel` | `gologit(y, x, pl=True)` | 1000 | 400 | 0.958 | 1.01 | +0.06 |  |
| `mlogit/default` | `mlogit("y ~ x1 + x2")` | 1000 | 400 | 0.932 | 0.95 | -0.01 |  |
| `clogit/default` | `clogit("y ~ x1 + x2", group="g")` | 1000 | 400 | 0.932 | 0.94 | -0.01 |  |
| `qreg/default` | `qreg(formula="y ~ x1 + x2")` | 1000 | 400 | 0.938 | 1.01 | -0.01 |  |
| `qreg/robust-het` | `qreg(..., vce="robust")` | 1000 | 400 | 0.945 | 1.00 | +0.06 |  |
| `qreg/q75` | `qreg(..., quantile=0.75)` | 1000 | 400 | 0.958 | 1.06 | -0.02 |  |
| `sqreg/median` | `sqreg(quantiles=[.25,.5,.75], reps=100)` | 1000 | 200 | 0.940 | 1.01 | -0.00 |  |
| `zip_model/default` | `zip_model("y ~ x1", inflate=["x2"])` | 1000 | 400 | 0.958 | 1.05 | +0.02 |  |
| `zinb/default` | `zinb("y ~ x1", inflate=["x2"])` | 2000 | 400 | 0.938 | 0.96 | +0.05 |  |
| `hurdle/default` | `hurdle("y ~ x1 + x2")` | 1000 | 400 | 0.953 | 1.02 | -0.05 |  |
| `fracreg/default` | `fracreg(y, x)` | 1000 | 400 | 0.943 | 0.95 | +0.09 |  |
| `betareg/default` | `betareg(y, x)` | 1000 | 400 | 0.955 | 1.03 | -0.02 |  |
| `biprobit/default` | `biprobit(y1, y2, x1, x2)` | 1000 | 400 | 0.938 | 0.92 | +0.05 |  |
| `etregress/mle` | `etregress(..., method="mle")` | 1000 | 400 | 0.958 | 0.99 | -0.05 |  |
| `etregress/twostep` | `etregress(..., method="twostep")` | 1000 | 400 | 0.950 | 1.01 | -0.01 |  |
| `etpoisson/default` | `etpoisson(y, x, treat, z)` | 2000 | 200 | 0.955 | 0.99 | -0.07 |  |
| `ivprobit/mle` | `ivprobit(..., method="mle")` | 1000 | 400 | 0.940 | 0.99 | +0.06 |  |
| `ivprobit/twostep` | `ivprobit(..., method="twostep")` | 1000 | 400 | 0.945 | 0.99 | +0.07 |  |
| `ivtobit/mle` | `ivtobit(..., method="mle")` | 1000 | 400 | 0.945 | 0.95 | -0.06 |  |
| `ivtobit/twostep` | `ivtobit(..., method="twostep")` | 1000 | 400 | 0.950 | 0.98 | -0.02 |  |
| `ivpoisson/additive` | `ivpoisson(..., errors="additive")` | 2000 | 400 | 0.945 | 0.97 | -0.07 |  |
| `ivpoisson/multiplicative` | `ivpoisson(..., errors="multiplicative")` | 2000 | 400 | 0.953 | 0.99 | +0.08 |  |
| `gee/gaussian-exchangeable` | `gee(..., corstr="exchangeable")` | 1000 | 400 | 0.960 | 0.99 | -0.11 |  |
| `gee/binomial-exchangeable` | `gee(..., family="binomial", corstr="exchangeable")` | 1000 | 400 | 0.950 | 0.95 | +0.09 |  |
| `gee/binomial-independence` | `gee(..., family="binomial")` | 1000 | 400 | 0.960 | 1.00 | +0.09 |  |
| `prais/prais` | `prais("y ~ x1", time="t")` | 500 | 400 | 0.955 | 0.98 | -0.01 |  |
| `prais/corc` | `prais(..., method="corc")` | 500 | 400 | 0.943 | 0.97 | -0.11 |  |
| `cnsreg/default` | `cnsreg("y ~ x1 + x2", "x1 = x2")` | 1000 | 400 | 0.968 | 1.01 | +0.00 |  |
| `nls/default` | `nls("y = {b0}*exp({b1}*x1)")` | 1000 | 400 | 0.940 | 0.98 | +0.01 |  |
| `robreg/mm` | `robreg("y ~ x1 + x2")` | 1000 | 400 | 0.950 | 0.98 | -0.07 |  |
| `eivreg/default` | `eivreg(..., reliability={"x1": 0.8})` | 1000 | 400 | 0.945 | 1.00 | +0.04 |  |
| `sureg/fgls` | `sureg({eq1, eq2})` | 1000 | 400 | 0.963 | 1.05 | -0.01 |  |
| `three_sls/default` | `three_sls({eq1, eq2})` | 1000 | 400 | 0.945 | 0.97 | +0.05 |  |
| `ppmlhdfe/default` | `ppmlhdfe("y ~ x1", absorb="g")` | 1000 | 400 | 0.920 | 0.91 | +0.06 |  |
| `gam/parametric-term` | `gam("y ~ x1 + s(x2)")` | 1000 | 400 | 0.950 | 1.04 | +0.00 |  |
| `fmm/gaussian-2` | `fmm("y ~ x1", k=2)` | 1000 | 200 | 0.945 | 1.07 | +0.02 |  |
| `regress/hc2` | `regress(..., robust="hc2")` | 1000 | 400 | 0.950 | 1.00 | +0.07 |  |
| `regress/hc4` | `regress(..., robust="hc4")` | 1000 | 400 | 0.958 | 1.04 | +0.08 |  |
| `regress/cr2` | `regress(..., cluster="g", vce="cr2")` | 1000 | 400 | 0.955 | 1.05 | +0.11 |  |
| `regress/cr3` | `regress(..., cluster="g", vce="cr3")` | 1000 | 400 | 0.948 | 1.05 | -0.02 |  |
| `regress/jackknife` | `regress(..., cluster="g", vce="jackknife")` | 1000 | 400 | 0.953 | 1.00 | -0.02 |  |
| `regress/aweights` | `regress(..., weights="w")` | 1000 | 400 | 0.927 | 0.92 | +0.00 |  |
| `regress/pweights` | `regress(..., weights="w", weights_as="sampling")` | 1000 | 400 | 0.960 | 1.02 | -0.01 |  |
| `logit/hc1` | `logit(..., robust="hc1")` | 1000 | 400 | 0.948 | 1.01 | +0.03 |  |
| `logit/pweights` | `logit(..., weights="w", weights_as="sampling")` | 1000 | 400 | 0.968 | 1.08 | +0.05 |  |
| `probit/cluster` | `probit(..., cluster="g")` | 1000 | 400 | 0.935 | 1.01 | +0.08 |  |
| `cloglog/robust` | `cloglog(..., robust="robust")` | 1000 | 400 | 0.955 | 1.01 | -0.02 |  |
| `poisson/cluster` | `poisson(..., cluster="g")` | 1000 | 400 | 0.920 | 0.94 | +0.03 |  |
| `poisson/pweights` | `poisson(..., weights="w", weights_as="sampling")` | 1000 | 400 | 0.935 | 0.92 | +0.00 |  |
| `poisson/exposure` | `poisson(..., exposure="e")` | 1000 | 400 | 0.955 | 1.00 | +0.08 |  |
| `nbreg/cluster` | `nbreg(..., cluster="g")` | 1000 | 400 | 0.938 | 0.98 | -0.03 |  |
| `nbreg/constant-dispersion` | `nbreg(..., dispersion="constant")` | 1000 | 400 | 0.948 | 0.99 | -0.13 |  |
| `glm/binomial-cluster` | `glm(..., family="binomial", cluster="g")` | 1000 | 400 | 0.958 | 0.99 | +0.09 |  |
| `glm/poisson-exposure` | `glm(..., family="poisson", exposure="e")` | 1000 | 400 | 0.948 | 0.97 | -0.05 |  |
| `tobit/cluster` | `tobit(..., vce="cluster g")` | 1000 | 400 | 0.940 | 0.99 | -0.03 |  |
| `tobit/two-limit` | `tobit(..., ll=0, ul=2)` | 1000 | 400 | 0.943 | 0.98 | +0.02 |  |
| `tobit/scls` | `tobit(..., method="scls")` | 1000 | 400 | 0.970 | 1.08 | +0.13 |  |
| `truncreg/robust` | `truncreg(..., robust="robust")` | 1000 | 400 | 0.960 | 1.06 | -0.01 |  |
| `intreg/robust` | `intreg(..., vce="robust")` | 1000 | 400 | 0.948 | 1.02 | +0.10 |  |
| `heckman/mle-robust` | `heckman(..., method="ml", vce="robust")` | 1000 | 400 | 0.920 | 0.91 | -0.03 |  |
| `heckman/mle-cluster` | `heckman(..., method="ml", vce="cluster g")` | 1000 | 1000 | 0.943 | 0.97 | -0.08 |  |
| `ologit/robust` | `ologit(..., robust="robust")` | 1000 | 400 | 0.950 | 1.01 | +0.06 |  |
| `ologit/cluster` | `ologit(..., cluster="g")` | 1000 | 400 | 0.955 | 1.03 | -0.00 |  |
| `oprobit/robust` | `oprobit(..., robust="robust")` | 1000 | 400 | 0.925 | 0.98 | +0.07 |  |
| `mlogit/robust` | `mlogit(..., robust="robust")` | 1000 | 400 | 0.945 | 0.95 | +0.03 |  |
| `mlogit/cluster` | `mlogit(..., cluster="g")` | 1000 | 400 | 0.943 | 0.97 | -0.00 |  |
| `clogit/robust` | `clogit(..., robust="robust")` | 1000 | 400 | 0.970 | 1.04 | +0.08 |  |
| `qreg/nid` | `qreg(..., vce="nid")` | 1000 | 400 | 0.945 | 0.98 | +0.09 |  |
| `qreg/kernel` | `qreg(..., vce="kernel")` | 1000 | 400 | 0.968 | 1.07 | +0.03 |  |
| `qreg/cluster` | `qreg(..., vce="cluster g")` | 1000 | 400 | 0.948 | 1.03 | +0.02 |  |
| `zip_model/robust` | `zip_model(..., robust="robust")` | 1000 | 400 | 0.943 | 1.03 | -0.07 |  |
| `zinb/robust` | `zinb(..., robust="robust")` | 2000 | 400 | 0.940 | 0.93 | +0.04 |  |
| `hurdle/robust` | `hurdle(..., robust="robust")` | 1000 | 400 | 0.953 | 1.00 | -0.07 |  |
| `hurdle/negbin` | `hurdle(..., count_model="negbin")` | 2000 | 400 | 0.955 | 1.01 | -0.07 |  |
| `fracreg/probit` | `fracreg(..., link="probit")` | 1000 | 400 | 0.973 | 1.03 | +0.04 |  |
| `betareg/robust` | `betareg(..., robust="robust")` | 1000 | 400 | 0.940 | 1.00 | +0.11 |  |
| `biprobit/robust` | `biprobit(..., robust="robust")` | 1000 | 400 | 0.950 | 1.00 | +0.06 |  |
| `etregress/robust` | `etregress(..., robust="robust")` | 1000 | 400 | 0.958 | 1.02 | +0.04 |  |
| `ivprobit/robust` | `ivprobit(..., vce="robust")` | 1000 | 400 | 0.945 | 0.97 | +0.07 |  |
| `ivtobit/robust` | `ivtobit(..., vce="robust")` | 1000 | 400 | 0.955 | 1.01 | +0.01 |  |
| `ivpoisson/onestep` | `ivpoisson(..., method="onestep")` | 2000 | 400 | 0.943 | 0.97 | -0.08 |  |
| `gee/gaussian-ar1` | `gee(..., corstr="ar1")` | 1000 | 400 | 0.935 | 0.91 | +0.04 |  |
| `gee/gaussian-ar1-model` | `gee(..., corstr="ar1", vce="model")` | 1000 | 1000 | 0.930 | 0.98 | +0.02 |  |
| `gee/gaussian-unstructured` | `gee(..., corstr="unstructured")` | 1000 | 1000 | 0.916 | 0.89 | -0.03 | **yes** |
| `gee/poisson-exchangeable` | `gee(..., family="poisson", corstr="exchangeable")` | 1000 | 400 | 0.938 | 0.95 | +0.06 |  |
| `prais/robust` | `prais(..., vce="robust")` | 500 | 400 | 0.948 | 1.03 | -0.03 |  |
| `prais/twostep` | `prais(..., twostep=True)` | 500 | 400 | 0.948 | 1.01 | -0.08 |  |
| `cnsreg/robust` | `cnsreg(..., vce="robust")` | 1000 | 400 | 0.980 | 1.05 | -0.01 |  |
| `nls/robust` | `nls(..., vce="robust")` | 1000 | 400 | 0.948 | 0.98 | -0.04 |  |
| `nls/hc3` | `nls(..., vce="hc3")` | 1000 | 400 | 0.950 | 1.00 | -0.02 |  |
| `robreg/mm-contaminated` | `robreg("y ~ x1 + x2")` | 1000 | 400 | 0.955 | 1.03 | -0.03 |  |
| `robreg/m` | `robreg(..., method="m")` | 1000 | 400 | 0.945 | 1.03 | -0.11 |  |
| `gam/robust` | `gam(..., vce="robust")` | 1000 | 400 | 0.953 | 1.00 | -0.04 |  |
| `sureg/mle` | `sureg({eq1, eq2}, method="mle")` | 1000 | 400 | 0.943 | 0.96 | -0.02 |  |

121 rows, 1 flagged. What the table says:

- The default variance of every likelihood model in the family is inside
  the band, and so are the robust, cluster and sampling-weight variants.
- The one flagged row is GEE with an unstructured working correlation of
  dimension 10 on 100 clusters. The exchangeable, AR(1) and independence
  rows on the same kind of data are calibrated.
- `sureg`, `three_sls` and `gam` report standard errors but no interval
  for the screened coefficient, so those rows use a normal interval
  built from the reported SE.
- Closest to the edge without a flag: `ppmlhdfe` with 100 absorbed groups,
  `tobit(vce="robust")`, `biprobit` and `heckman(vce="robust")`, all with
  SE/SD between 0.91 and 0.93 on 400 draws.

Flagged rows and their mechanism:

- `gee/gaussian-unstructured`: Known small-sample property of the sandwich with an unstructured working correlation: 45 correlations are estimated from 100 clusters, the fitted weights are noisy and the robust SE is biased down (SE/SD 0.89). In 7% of the draws the estimated correlation matrix is not positive definite and sp.gee refuses it with a typed error. The AR(1) and exchangeable rows on the same data are calibrated.

### Panel estimators (`se_screen/panel.py`)

Balanced panels, 200 units by 8 periods unless the row says otherwise;
target the slope on `x1` or the autoregressive coefficient (both 0.5).
Each design gives the row's variance estimator what it assumes:
independent errors for the conventional variance, AR(1) errors for the
cluster rows, a random-walk error for conventional first differences.

| row | call | n | B | coverage | SE/SD | bias/SD | flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `panel/fe` | `panel(method="fe")` | 1600 | 400 | 0.953 | 1.04 | +0.02 |  |
| `panel/fe-cluster` | `panel(method="fe", cluster="id")` | 1600 | 400 | 0.958 | 1.00 | -0.06 |  |
| `panel/fe-robust` | `panel(method="fe", robust=...)` | 1600 | 400 | 0.978 | 1.04 | +0.01 |  |
| `panel/re` | `panel(method="re")` | 1600 | 400 | 0.940 | 0.99 | -0.08 |  |
| `panel/re-cluster` | `panel(method="re", cluster="id")` | 1600 | 400 | 0.963 | 1.02 | -0.05 |  |
| `panel/fd` | `panel(method="fd")` | 1600 | 400 | 0.932 | 0.94 | -0.05 |  |
| `panel/fd-cluster` | `panel(method="fd", cluster="id")` | 1600 | 400 | 0.945 | 1.00 | +0.08 |  |
| `panel/between` | `panel(method="between")` | 1600 | 400 | 0.938 | 0.97 | +0.13 |  |
| `panel/pooled-cluster` | `panel(method="pooled", cluster="id")` | 1600 | 400 | 0.948 | 0.96 | -0.02 |  |
| `panel/twoway` | `panel(method="twoway")` | 1600 | 400 | 0.932 | 0.97 | +0.03 |  |
| `panel/mundlak` | `panel(method="mundlak")` | 1600 | 400 | 0.970 | 1.06 | -0.08 |  |
| `xtabond/onestep` | `xtabond(y, x) (one-step, robust)` | 4000 | 400 | 0.938 | 0.98 | -0.15 |  |
| `xtabond/twostep` | `xtabond(twostep=True)` | 4000 | 400 | 0.938 | 0.97 | -0.25 |  |
| `xtabond/onestep-nonrobust` | `xtabond(robust=False)` | 4000 | 400 | 0.940 | 0.97 | -0.15 |  |
| `xtabond/collapse` | `xtabond(collapse=True)` | 4000 | 400 | 0.940 | 0.99 | -0.06 |  |
| `xtdpdsys/onestep` | `xtdpdsys(y, x)` | 4000 | 400 | 0.945 | 1.03 | +0.14 |  |
| `xtdpdsys/twostep` | `xtdpdsys(twostep=True)` | 4000 | 400 | 0.970 | 1.07 | +0.01 |  |
| `xtlsdvc/default` | `xtlsdvc(y, x)` | 4000 | 1000 | 0.821 | 0.79 | -0.59 | **yes** |
| `xtlsdvc/consistent` | `xtlsdvc(plug_in="consistent")` | 4000 | 1000 | 0.882 | 0.77 | -0.07 | **yes** |
| `xtlsdvc/bootstrap` | `xtlsdvc(bootstrap=50)` | 4000 | 200 | 0.885 | 0.93 | -0.58 | **yes** |
| `xtlsdvc/consistent-bootstrap` | `xtlsdvc(plug_in="consistent", bootstrap=50)` | 4000 | 200 | 0.900 | 0.94 | -0.19 |  |
| `xtivreg/fe` | `xtivreg(method="fe")` | 1600 | 400 | 0.927 | 0.93 | +0.04 |  |
| `xtivreg/re` | `xtivreg(method="re")` | 1600 | 400 | 0.945 | 0.97 | -0.11 |  |
| `xtivreg/fd` | `xtivreg(method="fd")` | 1600 | 400 | 0.955 | 1.00 | -0.04 |  |
| `xtivreg/fd-cluster` | `xtivreg(method="fd", vce="cluster")` | 1600 | 400 | 0.965 | 1.05 | -0.03 |  |
| `xtivreg/be` | `xtivreg(method="be")` | 1600 | 400 | 0.970 | 1.04 | +0.06 |  |
| `xtpcse/default` | `xtpcse(y, x, id, time)` | 600 | 400 | 0.960 | 1.03 | -0.01 |  |
| `xtpcse/hetonly` | `xtpcse(hetonly=True)` | 800 | 400 | 0.940 | 0.95 | +0.00 |  |
| `panel_fgls/heteroskedastic` | `panel_fgls(y, x, id, time)` | 800 | 400 | 0.927 | 0.93 | -0.08 |  |
| `xtregar/fe` | `xtregar(method="fe")` | 2000 | 400 | 0.943 | 0.98 | +0.00 |  |
| `xtregar/re` | `xtregar(method="re")` | 2000 | 400 | 0.963 | 1.01 | -0.03 |  |
| `xttobit/default` | `xttobit(y, x, id, ll=0)` | 1000 | 400 | 0.965 | 1.03 | +0.01 |  |
| `xtintreg/default` | `xtintreg(y_lower, y_upper, x, id)` | 1000 | 200 | 0.950 | 1.00 | -0.04 |  |
| `xthtaylor/ht` | `xthtaylor(endog=[x2, z2])` | 2500 | 400 | 0.935 | 0.99 | +0.05 |  |
| `xthtaylor/robust` | `xthtaylor(vce="robust")` | 2500 | 400 | 0.945 | 0.96 | -0.03 |  |
| `fama_macbeth/default` | `fama_macbeth("y ~ x1", time="t")` | 5000 | 400 | 0.955 | 0.99 | +0.08 |  |
| `hdfe_ols/default` | `hdfe_ols("y ~ x1 + x2 | id + t")` | 1600 | 400 | 0.943 | 0.97 | -0.01 |  |
| `hdfe_ols/cluster` | `hdfe_ols(..., cluster="id")` | 1600 | 400 | 0.945 | 0.98 | +0.04 |  |
| `feols/default` | `feols("y ~ x1 + x2 | id")` | 1600 | 400 | 0.943 | 1.01 | -0.04 |  |
| `feols/cluster` | `feols(..., vcov={"CRV1": "id"})` | 1600 | 400 | 0.955 | 1.03 | -0.06 |  |
| `feols/hetero` | `feols(..., vcov="hetero")` | 1600 | 1000 | 0.925 | 0.96 | +0.02 |  |
| `fepois/default` | `fepois("y ~ x1 | id")` | 1600 | 400 | 0.955 | 1.05 | -0.05 |  |
| `interactive_fe/default` | `interactive_fe(n_factors=1)` | 3000 | 400 | 0.938 | 1.02 | +0.07 |  |
| `panel_logit/fe` | `panel_logit(method="fe")` | 1600 | 400 | 0.943 | 1.00 | +0.11 |  |
| `panel_logit/re` | `panel_logit(method="re")` | 1600 | 400 | 0.940 | 0.98 | -0.01 |  |
| `panel_probit/re` | `panel_probit(y, x)` | 1600 | 400 | 0.955 | 1.00 | +0.05 |  |
| `xtnbreg/fe` | `xtnbreg(model="fe")` | 1600 | 400 | 0.943 | 0.97 | +0.04 |  |
| `mmqreg/median` | `mmqreg(absorb="id", quantile=0.5)` | 1600 | 400 | 0.943 | 0.94 | -0.04 |  |
| `panel/fe-cr2` | `panel(method="fe", cluster="id", vce="CR2")` | 1600 | 400 | 0.940 | 0.93 | +0.04 |  |
| `panel/fe-cr3` | `panel(method="fe", cluster="id", vce="CR3")` | 1600 | 400 | 0.953 | 0.98 | +0.09 |  |
| `panel/fe-jackknife` | `panel(method="fe", cluster="id", vce="jackknife")` | 1600 | 400 | 0.950 | 0.98 | -0.05 |  |
| `panel/fe-driscoll-kraay` | `panel(method="fe", robust="driscoll-kraay")` | 3000 | 1000 | 0.879 | 0.84 | +0.04 | **yes** |
| `panel/re-robust` | `panel(method="re", robust="robust")` | 1600 | 400 | 0.945 | 0.98 | -0.02 |  |
| `panel/twoway-cluster` | `panel(method="twoway", cluster="id")` | 1600 | 400 | 0.958 | 1.05 | -0.04 |  |
| `panel/chamberlain` | `panel(method="chamberlain")` | 1600 | 400 | 0.948 | 1.02 | -0.03 |  |
| `panel/between-robust` | `panel(method="be", robust="robust")` | 1600 | 400 | 0.960 | 1.06 | +0.03 |  |
| `xtabond/twostep-nonrobust` | `xtabond(twostep=True, robust=False)` | 4000 | 400 | 0.943 | 0.96 | -0.20 |  |
| `xtabond/ah` | `xtabond(method="ah")` | 4000 | 400 | 0.945 | 0.97 | -0.01 |  |
| `xtabond/orthogonal` | `xtabond(orthogonal=True)` | 4000 | 1000 | 0.955 | 0.98 | -0.20 |  |
| `xtdpdsys/collapse` | `xtdpdsys(collapse=True)` | 4000 | 400 | 0.948 | 0.99 | -0.07 |  |
| `xtivreg/ec2sls` | `xtivreg(method="re", ec2sls=True)` | 1600 | 400 | 0.958 | 1.04 | +0.05 |  |
| `xtivreg/fe-robust` | `xtivreg(method="fe", vce="robust")` | 1600 | 400 | 0.958 | 1.07 | -0.10 |  |
| `xtivreg/re-cluster` | `xtivreg(method="re", vce="cluster")` | 1600 | 400 | 0.940 | 0.95 | +0.02 |  |
| `xtpcse/ar1` | `xtpcse(correlation="ar1")` | 600 | 400 | 0.940 | 0.97 | -0.03 |  |
| `xtpcse/psar1` | `xtpcse(correlation="psar1")` | 600 | 400 | 0.968 | 1.04 | +0.06 |  |
| `panel_fgls/ar1` | `panel_fgls(corr="ar1")` | 600 | 400 | 0.925 | 0.91 | +0.05 |  |
| `panel_fgls/correlated` | `panel_fgls(panels="correlated")` | 600 | 400 | 0.945 | 1.00 | -0.07 |  |
| `xthtaylor/amacurdy` | `xthtaylor(method="amacurdy")` | 2500 | 400 | 0.948 | 0.98 | +0.08 |  |
| `fama_macbeth/newey-west` | `fama_macbeth(lags=4)` | 10000 | 400 | 0.925 | 0.93 | +0.01 |  |
| `feols/twoway-cluster` | `feols("y ~ x1", vcov={"CRV1": "id + t"})` | 3600 | 400 | 0.930 | 0.93 | +0.09 |  |
| `hdfe_ols/twoway-cluster` | `hdfe_ols("y ~ x1", cluster=["id", "t"])` | 3600 | 400 | 0.963 | 0.98 | -0.07 |  |
| `fepois/cluster` | `fepois(..., vcov={"CRV1": "id"})` | 1600 | 400 | 0.938 | 0.96 | +0.02 |  |
| `fepois/hetero` | `fepois(..., vcov="hetero")` | 1600 | 1000 | 0.933 | 0.92 | -0.02 |  |
| `hdfe_ols/robust` | `hdfe_ols(..., vce="robust")` | 1600 | 400 | 0.938 | 0.94 | +0.04 |  |
| `interactive_fe/nonrobust` | `interactive_fe(robust=False)` | 3000 | 400 | 0.953 | 0.98 | +0.00 |  |
| `panel_logit/re-cluster` | `panel_logit(method="re", cluster="id")` | 1600 | 200 | 0.940 | 0.98 | -0.09 |  |
| `mmqreg/q25` | `mmqreg(quantile=0.25)` | 1600 | 1000 | 0.896 | 0.95 | +0.45 | **yes** |
| `mmqreg/robust` | `mmqreg(vce="robust")` | 1600 | 400 | 0.945 | 0.98 | +0.01 |  |

78 rows, 5 flagged. What the table says:

- Static estimators, their cluster and small-sample variants, dynamic
  panel GMM (one-step, two-step with the Windmeijer correction, system,
  collapsed), panel IV, PCSE, FGLS, Hausman-Taylor, Fama-MacBeth, the
  absorbing estimators and the random-effects binary models are in band.
- `sp.xtlsdvc` is the finding. Its default reproduces Stata's `xtlsdvc`,
  and that estimator keeps a bias that does not shrink with the number of
  units. `plug_in="consistent"` (new) removes it; with `bootstrap=` it is
  the calibrated combination (coverage 0.90 on 200 draws, SE/SD 0.94).
- Driscoll-Kraay standard errors are 16% too small with 60 periods and
  serially correlated common shocks.
- `sp.mmqreg` away from the median has a bias of order 1/T with absorbed
  unit effects.

Flagged rows and their mechanism:

- `xtlsdvc/default`: Two things. (1) Bias: the Stata plug-in for the deterministic part of the lagged outcome uses within-demeaned residuals, which keep each unit's mean error; the first-order correction then converges to -0.2195 where the exact LSDV inconsistency is -0.2353 (rho 0.5, seven usable periods), a bias that does not shrink with N. sp.xtlsdvc matches Stata xtlsdvc here; plug_in='consistent' removes it (tests/test_lsdvc_plug_in.py). (2) SE: the analytic SE is the LSDV one and ignores the estimated correction, as in Stata; the function warns and offers bootstrap=.
- `xtlsdvc/consistent`: SE only. plug_in='consistent' removes the bias (bias/SD -0.07) but the analytic SE is still the LSDV one, which ignores the estimated correction (documented, warned). Row xtlsdvc/consistent-bootstrap is the calibrated combination.
- `xtlsdvc/bootstrap`: Bias only. The parametric bootstrap SE is calibrated (SE/SD 0.93) but the default Stata plug-in leaves a bias of 0.58 SD at 500 units; see xtlsdvc/default.
- `panel/fe-driscoll-kraay`: Known small-T property of the Driscoll-Kraay HAC. The variance is a Bartlett HAC of the 60 period sums of the scores at the default lag 3 (a hand computation on one draw gives 0.0444 against the reported 0.0448). With serially correlated period shocks in the regressor and the error (rho 0.5 each) the kernel truncates and down-weights the autocovariances and the HAC is biased down at T = 60.
- `mmqreg/q25`: Incidental-parameter bias of order 1/T, not the SE (SE/SD 0.95). The scale equation is fitted on absolute within residuals with unit effects estimated from 8 observations each. Mean error at the 0.25 quantile: 0.069 with 4 periods, 0.035 with 8, 0.010 with 20, none with 40 (1,600 observations throughout). The median row is unaffected. Now stated in the mmqreg notes.

### Instrumental variables (`se_screen/iv.py`)

n = 1,000, one endogenous regressor with coefficient 0.5. Strong designs
(first-stage F in the hundreds), a many-instrument design (30
instruments), and weak designs where only the weak-instrument robust
sets are screened, for coverage.

| row | call | n | B | coverage | SE/SD | bias/SD | flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `iv/2sls` | `iv("y ~ (d ~ z1) + x1")` | 1000 | 400 | 0.953 | 0.98 | +0.01 |  |
| `iv/2sls-robust` | `iv(..., robust="robust")` | 1000 | 400 | 0.938 | 0.97 | -0.08 |  |
| `iv/2sls-cluster` | `iv(..., cluster="g")` | 1000 | 400 | 0.943 | 0.96 | -0.08 |  |
| `ivreg/2sls-over` | `ivreg("y ~ (d ~ z1 + z2 + z3) + x1")` | 1000 | 400 | 0.953 | 0.98 | +0.06 |  |
| `iv/liml` | `iv(..., method="liml")` | 1000 | 400 | 0.943 | 0.97 | -0.07 |  |
| `iv/fuller` | `iv(..., method="fuller")` | 1000 | 400 | 0.960 | 1.02 | +0.10 |  |
| `iv/gmm` | `iv(..., method="gmm")` | 1000 | 400 | 0.938 | 0.98 | -0.02 |  |
| `iv/gmm-robust` | `iv(..., method="gmm", robust="robust")` | 1000 | 400 | 0.950 | 0.98 | +0.03 |  |
| `iv/jive` | `iv(..., method="jive")` | 1000 | 400 | 0.938 | 0.94 | -0.16 |  |
| `liml/default` | `liml(y, x_endog, x_exog, z)` | 1000 | 400 | 0.938 | 0.96 | -0.09 |  |
| `liml/robust` | `liml(..., robust="robust")` | 1000 | 400 | 0.948 | 0.99 | -0.08 |  |
| `liml/many` | `liml(...), 30 instruments` | 1000 | 400 | 0.940 | 0.95 | -0.02 |  |
| `liml/fuller1` | `liml(..., fuller=1)` | 1000 | 400 | 0.927 | 0.94 | +0.02 |  |
| `jive/jive1` | `jive(y, x_endog, x_exog, z)` | 1000 | 400 | 0.963 | 1.02 | -0.14 |  |
| `jive/robust` | `jive(..., robust="robust")` | 1000 | 400 | 0.968 | 1.06 | -0.17 |  |
| `iv.ujive/default` | `iv.ujive(y, d, z, exog)` | 1000 | 400 | 0.970 | 1.02 | -0.02 |  |
| `iv.ijive/default` | `iv.ijive(y, d, z, exog)` | 1000 | 400 | 0.950 | 0.99 | -0.10 |  |
| `iv.rjive/default` | `iv.rjive(y, d, z, exog)` | 1000 | 400 | 0.920 | 0.90 | -0.15 |  |
| `iv.jive1/default` | `iv.jive1(y, d, z, exog)` | 1000 | 400 | 0.958 | 0.98 | -0.09 |  |
| `iv.jive_mw/default` | `iv.jive_mw(y, endog, instruments, exog)` | 1000 | 400 | 0.950 | 1.03 | -0.09 |  |
| `gmm/linear-iv-twostep` | `gmm(moment_fn, theta0) with Z'(y - Xb)` | 1000 | 400 | 0.948 | 1.00 | -0.01 |  |
| `lasso_iv/default` | `iv.lasso_iv(y, x_endog, x_exog, z)` | 1000 | 400 | 0.950 | 0.96 | +0.12 |  |
| `rlasso_iv/default` | `rlasso_iv(y, d, z, x)` | 1000 | 400 | 0.940 | 1.02 | -0.09 |  |
| `anderson_rubin_ci/weak-1` | `anderson_rubin_ci(y, d, [z1])` | 1000 | 400 | 0.948 | n/a | n/a |  |
| `anderson_rubin_ci/weak-3` | `anderson_rubin_ci(y, d, [z1, z2, z3])` | 1000 | 400 | 0.935 | n/a | n/a |  |
| `anderson_rubin_ci/strong` | `anderson_rubin_ci(y, d, [z1, z2, z3])` | 1000 | 400 | 0.965 | n/a | n/a |  |
| `conditional_lr_ci/weak-3` | `conditional_lr_ci(y, d, [z1, z2, z3])` | 1000 | 200 | 0.945 | n/a | n/a |  |
| `k_test_ci/weak-3` | `k_test_ci(y, d, [z1, z2, z3])` | 1000 | 400 | 0.948 | n/a | n/a |  |
| `many_weak_ar/many` | `iv.many_weak_ar(y, endog, instruments)` | 1000 | 400 | 0.955 | n/a | n/a |  |
| `ivreg/hc3` | `ivreg(..., robust="hc3")` | 1000 | 400 | 0.953 | 1.00 | +0.02 |  |
| `ivreg/hc0` | `ivreg(..., robust="hc0")` | 1000 | 400 | 0.950 | 0.99 | +0.01 |  |
| `ivreg/pweights` | `ivreg(..., weights="wt", weights_as="sampling")` | 1000 | 400 | 0.945 | 0.94 | -0.06 |  |
| `iv/liml-cluster` | `iv(..., method="liml", cluster="g")` | 1000 | 400 | 0.953 | 0.94 | -0.17 |  |
| `iv/gmm-cluster` | `iv(..., method="gmm", cluster="g")` | 1000 | 400 | 0.925 | 0.96 | -0.02 |  |
| `liml/cluster` | `liml(..., cluster="g")` | 1000 | 400 | 0.963 | 1.04 | -0.03 |  |
| `jive/cluster` | `jive(..., cluster="g")` | 1000 | 400 | 0.932 | 0.96 | -0.10 |  |
| `jive/jive2` | `jive(..., variant="jive2")` | 1000 | 400 | 0.953 | 0.95 | -0.17 |  |
| `iv.jive_mw/many-weak` | `iv.jive_mw(y, endog, instruments, exog)` | 1000 | 400 | 0.968 | 0.96 | -0.18 |  |
| `many_weak_ar/many-weak` | `iv.many_weak_ar(y, endog, instruments)` | 1000 | 400 | 0.950 | n/a | n/a |  |
| `anderson_rubin_ci/cluster` | `anderson_rubin_ci(..., cluster="g")` | 1000 | 400 | 0.927 | n/a | n/a |  |
| `conditional_lr_ci/weak-1` | `conditional_lr_ci(y, d, [z1])` | 1000 | 400 | 0.955 | n/a | n/a |  |

41 rows, 0 flagged. What the table says:

- Every row is in band: 2SLS, LIML, Fuller, GMM, the JIVE family, the
  lasso IV estimators, and their robust and cluster variances.
- The Anderson-Rubin, conditional likelihood ratio and K sets hold their
  level with a concentration parameter of 5.
- `sp.iv.many_weak_ar` covered 70% in the many-weak design before this
  round: its default grid was 101 points within 0.5 of the OLS estimate
  and the constant was left out. It now inverts the statistic exactly
  (coverage 0.950 on the same seeds).

### Survival analysis (`se_screen/survival.py`)

n = 1,000, independent censoring near 30%. Regression rows target the
coefficient on `x1`; the others a survival probability, a cumulative
incidence, a restricted mean or a mean survival time with a closed-form
truth.

| row | call | n | B | coverage | SE/SD | bias/SD | flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `cox/default` | `cox("time + event ~ x1 + x2")` | 1000 | 400 | 0.955 | 1.00 | +0.02 |  |
| `cox/breslow` | `cox(..., ties="breslow")` | 1000 | 400 | 0.973 | 1.07 | +0.12 |  |
| `cox/robust` | `cox(..., robust="robust")` | 1000 | 400 | 0.932 | 0.98 | +0.15 |  |
| `cox/cluster` | `cox(..., cluster="g")` | 1000 | 400 | 0.948 | 1.00 | +0.06 |  |
| `cox/strata` | `cox(..., strata="s")` | 1000 | 400 | 0.953 | 1.00 | +0.06 |  |
| `survreg/weibull` | `survreg(dist="weibull")` | 1000 | 400 | 0.950 | 0.98 | +0.00 |  |
| `survreg/exponential` | `survreg(dist="exponential")` | 1000 | 400 | 0.948 | 0.99 | +0.03 |  |
| `survreg/weibull-robust` | `survreg(dist="weibull", robust="robust")` | 1000 | 400 | 0.948 | 1.00 | -0.03 |  |
| `survreg/lognormal` | `survreg(dist="lognormal")` | 1000 | 400 | 0.930 | 0.95 | -0.00 |  |
| `survreg/loglogistic` | `survreg(dist="loglogistic")` | 1000 | 400 | 0.963 | 1.07 | -0.03 |  |
| `aft/weibull` | `aft(family="weibull")` | 1000 | 400 | 0.963 | 1.04 | -0.01 |  |
| `aft/lognormal` | `aft(family="lognormal")` | 1000 | 400 | 0.943 | 0.99 | -0.01 |  |
| `aft/loglogistic` | `aft(family="loglogistic")` | 1000 | 400 | 0.955 | 1.00 | -0.04 |  |
| `cox_frailty/default` | `cox_frailty(..., cluster="g")` | 1000 | 400 | 0.965 | 1.00 | -0.03 |  |
| `finegray/default` | `finegray(duration, event, x, cause=1)` | 1000 | 400 | 0.938 | 0.98 | +0.06 |  |
| `finegray/model` | `finegray(..., vce="model")` | 1000 | 400 | 0.948 | 0.99 | +0.07 |  |
| `rmst/difference` | `rmst(group=, tau=2)` | 1000 | 400 | 0.938 | 1.00 | +0.04 |  |
| `kaplan_meier/loglog` | `kaplan_meier(duration, event)` | 1000 | 400 | 0.955 | 1.04 | -0.06 |  |
| `kaplan_meier/plain` | `kaplan_meier(conf_type="plain")` | 1000 | 400 | 0.953 | 1.05 | +0.07 |  |
| `cuminc/default` | `cuminc(duration, event)` | 1000 | 400 | 0.963 | 0.99 | -0.01 |  |
| `stteffects/ra` | `stteffects(method="ra")` | 2000 | 400 | 0.945 | 0.98 | -0.09 |  |
| `stteffects/ipw` | `stteffects(method="ipw")` | 2000 | 400 | 0.940 | 0.98 | +0.04 |  |
| `stteffects/ipwra` | `stteffects(method="ipwra")` | 2000 | 400 | 0.950 | 0.98 | -0.02 |  |
| `stteffects/wra` | `stteffects(method="wra")` | 2000 | 400 | 0.930 | 0.93 | +0.10 |  |
| `stteffects/ipw-heavy-censoring` | `stteffects(method="ipw")` | 2000 | 1000 | 0.953 | 0.38 | +0.05 | **yes** |
| `cox/pweights` | `cox(..., weights="w", weights_as="sampling")` | 1000 | 400 | 0.953 | 1.04 | +0.03 |  |
| `cox/robust-stata` | `cox(..., robust="robust", ssc="stata")` | 1000 | 400 | 0.938 | 0.95 | +0.04 |  |
| `cox/efron-ties` | `cox(..., ties="efron")` | 1000 | 400 | 0.953 | 1.03 | -0.01 |  |
| `cox/breslow-ties` | `cox(..., ties="breslow")` | 1000 | 1000 | 0.883 | 1.07 | -0.91 | **yes** |
| `cox/delayed-entry` | `cox(..., start="entry")` | 1000 | 400 | 0.932 | 0.95 | +0.01 |  |
| `survreg/cluster` | `survreg(dist="weibull", cluster="g")` | 1000 | 400 | 0.930 | 0.98 | -0.04 |  |
| `survreg/lognormal-robust` | `survreg(dist="lognormal", robust="robust")` | 1000 | 400 | 0.965 | 1.04 | +0.01 |  |
| `aft/exponential` | `aft(family="exponential")` | 1000 | 400 | 0.965 | 1.08 | +0.01 |  |
| `kaplan_meier/log` | `kaplan_meier(conf_type="log")` | 1000 | 400 | 0.958 | 1.03 | -0.13 |  |
| `kaplan_meier/ij` | `kaplan_meier(se_method="ij")` | 1000 | 400 | 0.955 | 1.01 | +0.01 |  |
| `rmst/ij` | `rmst(group=, tau=2, se_method="ij")` | 1000 | 400 | 0.958 | 0.98 | -0.04 |  |
| `cuminc/gray` | `cuminc(variance="gray")` | 1000 | 400 | 0.963 | 1.06 | +0.03 |  |
| `cuminc/log-log` | `cuminc(conf_type="log-log")` | 1000 | 400 | 0.960 | 1.01 | -0.04 |  |
| `finegray/small-sample` | `finegray(..., small_sample=True)` | 1000 | 400 | 0.930 | 0.95 | +0.05 |  |
| `stteffects/ra-atet` | `stteffects(method="ra", estimand="ATET")` | 2000 | 400 | 0.948 | 1.00 | +0.03 |  |

40 rows, 2 flagged. What the table says:

- Cox (Efron, robust, cluster, strata, delayed entry, sampling weights),
  the parametric and AFT models, shared frailty, Fine-Gray, Kaplan-Meier,
  cumulative incidence, RMST and `stteffects` are in band.
- Breslow's approximation is biased with heavily tied times; Efron, the
  default, is not.
- Censoring-weighted `stteffects` needs the probability of remaining
  uncensored bounded away from zero over the whole range of failure times.
  The heavy-censoring row shows what happens otherwise.

Flagged rows and their mechanism:

- `stteffects/ipw-heavy-censoring`: Design breaks the overlap that censoring weights need. With censoring rate 0.15 the second moment of T / G(T) does not exist for units with x1 > 1.2: long failure times are almost always censored and the rare uncensored one carries a huge weight. The estimator is heavy-tailed, so SE/SD does not settle (0.66, 0.27 and 0.38 in three runs) while coverage stays at 92% to 96%. At rate 0.04 (row stteffects/ipw) SE/SD is 0.98. sp.stteffects now records the censoring weights in model_info and its notes describe the condition.
- `cox/breslow-ties`: Known property of the Breslow approximation with heavy ties, not the SE (SE/SD 1.07). Event times are recorded on a grid of width 0.1, so about a tenth of the risk set fails at each tied time, and Breslow's partial likelihood attenuates the coefficient (bias -0.9 SD). The Efron default on the same data (row cox/efron-ties) is inside the band.

### Time series (`se_screen/timeseries.py`)

Stationary designs, T = 500 (2,000 for GARCH, 200 for the interrupted
time series), persistence 0.5.

| row | call | n | B | coverage | SE/SD | bias/SD | flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `arima/ar1` | `arima(y, order=(1, 0, 0))` | 500 | 400 | 0.955 | 1.00 | -0.14 |  |
| `arima/ar1-innovations` | `arima(y, order=(1, 0, 0), method="innovations_mle")` | 500 | 400 | 0.953 | 1.01 | -0.16 |  |
| `arima/arma11-ar` | `arima(y, order=(1, 0, 1))` | 500 | 400 | 0.945 | 0.99 | -0.14 |  |
| `arima/arma11-ma` | `arima(y, order=(1, 0, 1))` | 500 | 400 | 0.958 | 1.01 | +0.07 |  |
| `arima/regression-ar1-errors` | `arima(y, order=(1, 0, 0), exog=x1)` | 500 | 400 | 0.963 | 1.01 | -0.02 |  |
| `regress/hac-4` | `regress(..., vce="hac", hac_lags=4)` | 500 | 400 | 0.922 | 0.91 | -0.05 |  |
| `regress/hac-12` | `regress(..., vce="hac", hac_lags=12)` | 500 | 400 | 0.920 | 0.91 | -0.06 |  |
| `var/coefficient` | `var(lags=1)` | 500 | 400 | 0.945 | 1.00 | -0.14 |  |
| `irf/asymptotic` | `irf(var, ci="asymptotic")` | 500 | 400 | 0.938 | 0.97 | -0.00 |  |
| `irf/bootstrap` | `irf(var, ci="bootstrap", reps=200)` | 500 | 200 | 0.950 | 1.06 | -0.09 |  |
| `local_projections/h2` | `local_projections(outcome, shock)` | 500 | 400 | 0.940 | 0.97 | -0.03 |  |
| `garch/alpha-oim` | `garch(y)` | 2000 | 400 | 0.955 | 1.07 | +0.09 |  |
| `garch/alpha-robust` | `garch(y, vce="robust")` | 2000 | 400 | 0.965 | 1.06 | +0.05 |  |
| `ardl/lag-coefficient` | `ardl(y, x, lags=1, x_lags=1)` | 500 | 400 | 0.960 | 1.07 | -0.11 |  |
| `its/level-change` | `its(y, time, intervention)` | 200 | 1000 | 0.905 | 0.86 | +0.01 | **yes** |
| `vec/beta` | `vec(lags=2, rank=1)` | 500 | 1000 | 0.933 | 0.86 | -0.03 | **yes** |
| `arima/ar1-no-constant` | `arima(y, order=(1, 0, 0), trend="n")` | 500 | 400 | 0.953 | 1.01 | -0.05 |  |
| `arima/arima110` | `arima(y, order=(1, 1, 0))` | 500 | 400 | 0.958 | 1.07 | +0.05 |  |
| `var/se-df-r` | `var(lags=1, se_df="r")` | 500 | 400 | 0.935 | 0.91 | -0.25 |  |
| `var/overfitted` | `var(lags=2)` | 500 | 400 | 0.958 | 1.00 | -0.07 |  |
| `irf/non-orthogonal` | `irf(var, orthogonal=False, ci="asymptotic")` | 500 | 400 | 0.958 | 1.00 | -0.06 |  |
| `irf/bootstrap-efron` | `irf(var, ci="bootstrap", boot="efron", reps=200)` | 500 | 200 | 0.935 | 1.03 | +0.01 |  |
| `irf/bootstrap-hall` | `irf(var, ci="bootstrap", boot="hall", reps=200)` | 500 | 200 | 0.945 | 0.99 | -0.08 |  |
| `local_projections/h0` | `local_projections(outcome, shock), h = 0` | 500 | 400 | 0.953 | 0.99 | -0.01 |  |
| `local_projections/h4` | `local_projections(outcome, shock), h = 4` | 500 | 400 | 0.930 | 0.98 | -0.06 |  |
| `local_projections/robust` | `local_projections(vce="robust")` | 500 | 400 | 0.958 | 1.04 | -0.06 |  |
| `garch/alpha-opg` | `garch(y, vce="opg")` | 2000 | 400 | 0.943 | 1.07 | +0.09 |  |
| `garch/t-innovations` | `garch(y, dist="t")` | 2000 | 400 | 0.925 | 0.99 | -0.05 |  |
| `garch/normal-qmle-robust` | `garch(y, vce="robust")` | 2000 | 1000 | 0.930 | 0.96 | +0.08 |  |
| `its/level-change-lag8` | `its(..., hac_lag=8)` | 200 | 1000 | 0.880 | 0.83 | -0.01 | **yes** |
| `its/level-change-small-sample` | `its(..., hac_small_sample=True)` | 200 | 1000 | 0.914 | 0.87 | +0.02 | **yes** |

31 rows, 4 flagged. What the table says:

- ARIMA, VAR coefficients, asymptotic and bootstrap impulse-response
  bands, local projections, GARCH (OIM, OPG, robust, t innovations) and
  ARDL are in band.
- Newey-West standard errors are too small in the interrupted time-series
  design at every lag choice (SE/SD 0.83 to 0.87), and 0.91 in a plain
  regression with AR(1) errors (`regress/hac-*`, not flagged). This is the
  estimator's known small-sample behaviour; `sp.its` equals statsmodels.
- `garch`, `var` and `vec` report no interval for the screened
  coefficient; those rows use a normal interval from the reported SE.

Flagged rows and their mechanism:

- `its/level-change`: Known property of Newey-West at a fixed lag. The SE equals statsmodels HAC (maxlags 4) to 1e-12 on one draw. With an AR(1) error (rho 0.3) the Bartlett kernel at lag 4 recovers 0.936 of the true SE in population; the rest of the gap is the small-sample bias of HAC with estimated residuals at T = 200.
- `vec/beta`: Heavy tails, not the SE: the median SE over the MAD-based spread is 1.01 and coverage is inside the band. The reduced-rank estimate of the cointegrating vector has occasional large errors that inflate the Monte Carlo SD.
- `its/level-change-lag8`: Same mechanism as its/level-change, and more lags make it worse, not better: the kernel truncation matters less at lag 8 (0.96 of the true SE in population) but the small-sample downward bias of a HAC built from residuals grows with the lag when the regressors are a step and a trend.
- `its/level-change-small-sample`: Same mechanism as its/level-change. The n / (n - k) factor of hac_small_sample=True is 1.02 at T = 200 and does not close a gap of 14%.

### Spatial regression and nonparametric fits (`se_screen/spatial.py`)

A 25 by 25 lattice with rook contiguity for the spatial rows; n = 1,000
and the regression function at x = 0.5 for the nonparametric rows.

| row | call | n | B | coverage | SE/SD | bias/SD | flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `sar/beta` | `sar(W, data, "y ~ x1")` | 625 | 400 | 0.960 | 1.03 | -0.04 |  |
| `sar/rho` | `sar(...), spatial lag coefficient` | 625 | 400 | 0.963 | 1.02 | -0.08 |  |
| `sem/beta` | `sem(W, data, "y ~ x1")` | 625 | 400 | 0.965 | 1.05 | -0.12 |  |
| `sem/lambda` | `sem(...), spatial error coefficient` | 625 | 400 | 0.958 | 1.00 | -0.13 |  |
| `sdm/beta` | `sdm(W, data, "y ~ x1")` | 625 | 400 | 0.930 | 0.95 | -0.01 |  |
| `slx/beta` | `slx(W, data, "y ~ x1")` | 625 | 400 | 0.968 | 1.01 | +0.02 |  |
| `sac/beta` | `sac(W, data, "y ~ x1")` | 625 | 400 | 0.968 | 1.04 | -0.09 |  |
| `sar_gmm/beta` | `sar_gmm(W, data, "y ~ x1")` | 625 | 400 | 0.960 | 1.03 | -0.05 |  |
| `sar_gmm/rho` | `sar_gmm(...), spatial lag coefficient` | 625 | 400 | 0.948 | 0.99 | -0.09 |  |
| `lpoly/linear-truth` | `lpoly(y, x) at x = 0.5` | 1000 | 400 | 0.958 | 1.01 | -0.02 |  |
| `lpoly/curved-truth` | `lpoly(y, x) at x = 0.5` | 1000 | 400 | 0.940 | 0.96 | -0.09 |  |
| `lprobust_at_point/curved` | `lprobust_at_point(x, y, 0.5, h=0.15)` | 1000 | 400 | 0.953 | 0.99 | -0.12 |  |

12 rows, 0 flagged. What the table says:

- All rows are in band, including the spatial parameters themselves.

### Regression discontinuity beyond sharp `rdrobust` (`se_screen/rd.py`)

n = 2,000, uniform running variable, constant effect 1.0. Sharp
`rdrobust` with defaults is a Track B row and is not repeated.

| row | call | n | B | coverage | SE/SD | bias/SD | flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `rdrobust/fuzzy` | `rdrobust(fuzzy="d")` | 4000 | 400 | 0.948 | 1.01 | -0.09 |  |
| `rdrobust/covs` | `rdrobust(covs=["z"])` | 2000 | 1000 | 0.937 | 0.95 | -0.04 |  |
| `rdrobust/cluster` | `rdrobust(cluster="g")` | 2000 | 400 | 0.938 | 0.99 | -0.03 |  |
| `rdrobust/kink` | `rdrobust(deriv=1)` | 4000 | 400 | 0.945 | 0.91 | +0.07 |  |
| `rdrobust/uniform-cerrd` | `rdrobust(kernel="uniform", bwselect="cerrd")` | 2000 | 400 | 0.955 | 1.01 | +0.06 |  |
| `rdrobust/hc3` | `rdrobust(vce="hc3")` | 2000 | 400 | 0.932 | 0.92 | -0.01 |  |
| `rdrobust/p2` | `rdrobust(p=2)` | 2000 | 400 | 0.927 | 0.91 | -0.01 |  |
| `rkd/default` | `rkd(y, x)` | 4000 | 1000 | 0.941 | 0.79 | +0.23 | **yes** |
| `rd_honest/M-known` | `rd_honest(M=0.5)` | 2000 | 400 | 0.960 | 0.99 | -0.02 |  |
| `rdrandinf/ci` | `rdrandinf(wl=-0.15, wr=0.15), test-inversion interval` | 2000 | 400 | 0.927 | n/a | n/a |  |
| `rdrandinf/size` | `rdrandinf(wl=-0.15, wr=0.15), p > 0.05 under no effect` | 2000 | 400 | 0.963 | n/a | n/a |  |
| `rdmc/pooled` | `rdmc(cutoffs=[-0.4, 0.4])` | 3000 | 400 | 0.948 | 1.02 | -0.09 |  |
| `rdrobust/hc0` | `rdrobust(vce="hc0")` | 2000 | 400 | 0.935 | 0.96 | +0.05 |  |
| `rdrobust/hc1` | `rdrobust(vce="hc1")` | 2000 | 400 | 0.935 | 0.97 | -0.00 |  |
| `rdrobust/hc2` | `rdrobust(vce="hc2")` | 2000 | 400 | 0.950 | 0.95 | +0.07 |  |
| `rdrobust/msetwo` | `rdrobust(bwselect="msetwo")` | 2000 | 1000 | 0.923 | 0.88 | +0.01 |  |
| `rdrobust/certwo` | `rdrobust(bwselect="certwo")` | 2000 | 400 | 0.932 | 0.91 | +0.03 |  |
| `rdrobust/msecomb2` | `rdrobust(bwselect="msecomb2")` | 2000 | 400 | 0.953 | 0.95 | -0.04 |  |
| `rdrobust/epanechnikov` | `rdrobust(kernel="epanechnikov")` | 2000 | 400 | 0.960 | 1.03 | +0.06 |  |
| `rdrobust/manual-h` | `rdrobust(h=0.3)` | 2000 | 400 | 0.945 | 1.00 | -0.09 |  |
| `rdrobust/fuzzy-covs` | `rdrobust(fuzzy="d", covs=["z"])` | 4000 | 400 | 0.960 | 0.98 | -0.01 |  |
| `rdrobust/cluster-cr2` | `rdrobust(cluster="g", vce="cr2")` | 2000 | 400 | 0.968 | 1.02 | +0.11 |  |
| `rdrobust/donut` | `rdrobust(donut=0.02)` | 2000 | 400 | 0.930 | 0.93 | +0.09 |  |
| `rd_honest/taylor` | `rd_honest(M=0.5, sclass="T")` | 2000 | 1000 | 0.984 | 1.02 | +0.02 |  |
| `rd_honest/flci` | `rd_honest(M=0.5, opt_criterion="flci")` | 2000 | 400 | 0.968 | 1.00 | +0.01 |  |
| `rd_honest/uniform` | `rd_honest(M=0.5, kernel="uniform")` | 2000 | 400 | 0.975 | 1.02 | -0.03 |  |

26 rows, 1 flagged. What the table says:

- Fuzzy, covariate-adjusted, clustered and kink `rdrobust`, every `vce`
  and the bandwidth selectors are in band on the robust interval. Several
  sit near the lower edge (coverage 0.92 to 0.94), as the sharp Track B
  row does.
- `sp.rd_honest` covers at or above its level, `sp.rdrandinf` holds its
  size and its test-inversion interval covers.
- `sp.rkd` reports the conventional estimate as its headline.
- `sp.rdmc` prints the conventional pooled estimate next to the robust SE
  and interval; the row pairs the SE with the bias-corrected estimate it
  belongs to.

Flagged rows and their mechanism:

- `rkd/default`: Property of the conventional estimate at an MSE-optimal bandwidth. sp.rkd reports the conventional kink estimate and its SE as the headline; that bandwidth leaves a smoothing bias of the same order as the SE (bias/SD 0.23), and the data-driven bandwidth gives the estimate heavier tails than its SE describes (median SE over MAD-based spread 1.06). The robust bias-corrected interval, which sp.rkd puts in model_info['robust'] and sp.rdrobust(deriv=1) reports as its headline (row rdrobust/kink), is inside the band. Owner decision: make it the headline of sp.rkd.

### Difference-in-differences and synthetic control (`se_screen/did.py`)

Staggered adoption (300 units, 10 periods, cohorts 5 and 7 plus never
treated), AR(1) errors within unit, constant effect 1.0; a block design
with one factor for the synthetic-control rows; one treated unit and 30
controls for `sp.synth`.

| row | call | n | B | coverage | SE/SD | bias/SD | flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `did_imputation/default` | `did_imputation(y, group, time, first_treat)` | 3000 | 400 | 0.958 | 0.99 | +0.02 |  |
| `gardner_did/default` | `gardner_did(y, group, time, first_treat)` | 3000 | 400 | 0.948 | 0.98 | -0.01 |  |
| `etwfe/default` | `etwfe(y, group, time, first_treat)` | 3000 | 400 | 0.938 | 0.98 | +0.03 |  |
| `wooldridge_did/default` | `wooldridge_did(y, group, time, first_treat)` | 3000 | 400 | 0.932 | 0.98 | +0.10 |  |
| `stacked_did/default` | `stacked_did(y, group, time, first_treat)` | 3000 | 400 | 0.943 | 0.99 | +0.01 |  |
| `event_study/lag1` | `event_study(window=(-4, 4)), event time +1` | 3000 | 400 | 0.948 | 0.99 | -0.01 |  |
| `event_study/lead2` | `event_study(window=(-4, 4)), event time -2` | 3000 | 400 | 0.948 | 1.01 | -0.04 |  |
| `did_multiplegt/default` | `did_multiplegt(n_boot=50)` | 1200 | 200 | 0.960 | 1.07 | +0.08 |  |
| `sdid/bootstrap` | `sdid(se_method="bootstrap")` | 800 | 200 | 0.970 | 1.16 | +0.03 |  |
| `sdid/jackknife` | `sdid(se_method="jackknife")` | 800 | 1000 | 0.904 | 0.91 | +0.00 | **yes** |
| `sdid/placebo-10-treated` | `sdid(se_method="placebo")` | 800 | 200 | 0.905 | 0.94 | -0.14 |  |
| `sdid/did-jackknife` | `sdid(method="did", se_method="jackknife")` | 800 | 1000 | 0.793 | 0.56 | +0.01 | **yes** |
| `sdid/did-jackknife-parallel` | `sdid(method="did", se_method="jackknife")` | 800 | 400 | 0.955 | 0.96 | -0.10 |  |
| `gsynth/default` | `gsynth(n_factors=1)` | 800 | 400 | 0.935 | 1.00 | -0.02 |  |
| `did_imputation/bootstrap` | `did_imputation(vce="bootstrap", n_boot=50)` | 1200 | 200 | 0.905 | 0.95 | +0.01 |  |
| `gardner_did/stage2` | `gardner_did(vce="stage2")` | 3000 | 1000 | 0.807 | 0.66 | +0.02 | **yes** |
| `gardner_did/bootstrap` | `gardner_did(vce="bootstrap", n_boot=50)` | 1200 | 200 | 0.945 | 1.01 | +0.02 |  |
| `etwfe/nevertreated` | `etwfe(cgroup="nevertreated")` | 3000 | 400 | 0.958 | 1.01 | +0.01 |  |
| `stacked_did/pooled` | `stacked_did(spec="pooled")` | 3000 | 400 | 0.943 | 1.01 | +0.08 |  |
| `stacked_did/treated-obs` | `stacked_did(aggregation="treated_obs")` | 3000 | 400 | 0.940 | 0.97 | +0.01 |  |
| `event_study/lag1-normal` | `event_study(crit_dist="normal"), event time +1` | 3000 | 400 | 0.943 | 0.98 | -0.02 |  |
| `sdid/sc-jackknife` | `sdid(method="sc", se_method="jackknife")` | 800 | 1000 | 1.000 | 4.23 | -0.02 | **yes** |
| `gsynth/nonparametric` | `gsynth(n_factors=1, inference="nonparametric")` | 800 | 1000 | 0.917 | 0.96 | -0.03 | **yes** |
| `synth/classic-placebo-ci` | `synth(treated_unit, treatment_time)` | 1085 | 400 | 0.985 | n/a | n/a |  |
| `synth/augmented` | `synth(method="augmented")` | 1085 | 1000 | 0.948 | 0.78 | +0.01 | **yes** |
| `synth/conformal` | `synth(inference="conformal")` | 1085 | 200 | 0.980 | n/a | n/a |  |

26 rows, 6 flagged. What the table says:

- `did_imputation`, `gardner_did`, `etwfe`, `wooldridge_did`,
  `stacked_did`, event-study leads and lags and `did_multiplegt` are in
  band, with analytic and bootstrap variances.
- `sp.sdid`: bootstrap and placebo are in band with 10 treated units; the
  jackknife is 9% too small there; the jackknife with `method="sc"` is
  four times too large. All three match R `synthdid`.
- `gardner_did(vce="stage2")` is the uncorrected variance and behaves
  like one.

Flagged rows and their mechanism:

- `sdid/jackknife`: Few treated units. The jackknife variance is driven by the 10 treated units, so it is itself noisy and the normal critical value is too small for it (a t with 9 degrees of freedom would give 2.26). The SE equals R synthdid's on one draw (0.0959). The bootstrap and placebo rows on the same design are inside the band.
- `sdid/did-jackknife`: Design, not estimator. Treated units load 0.5 more on the factor, so parallel trends fail given the factor path: the DiD estimate moves with the realised post-minus-pre factor change, which is common to all units and which a jackknife over units cannot see. The estimate is unbiased over draws and its unit jackknife SE equals R synthdid's (0.0773 on one draw). With the same loading distribution in both groups (row sdid/did-jackknife-parallel) the SE is calibrated.
- `gardner_did/stage2`: Documented option. vce='stage2' is the second-stage clustered variance without the correction for the estimated first-stage fixed effects; the default vce='analytic' (row gardner_did/default) carries the correction and is calibrated.
- `sdid/sc-jackknife`: Property of the fixed-weight jackknife applied to synthetic control, shared with R synthdid 0.0.9 (same SE to four digits on one draw: 0.5185, against 0.107 from the placebo method). The jackknife holds the unit weights fixed. SC has no intercept, so dropping one treated unit changes the treated level that the fixed weights were chosen to match, and the pseudo-values pick up the spread of the unit effects, which a refitted SC would absorb. The SE is about four times the sampling SD and the interval always covers. Use se_method='placebo' or 'bootstrap' with method='sc'. sp.sdid does not warn about this combination (open item).
- `gsynth/nonparametric`: At the edge of the band (0.917 on 1,000 draws against a limit of 0.918) with a calibrated SE (SE/SD 0.96). The reported interval is the percentile interval of 200 bootstrap draws over 40 units; a normal interval from the same SE covers 0.93.
- `synth/augmented`: Heavy tails, not the SE: one treated unit. The median SE over the MAD-based spread is 1.05 and coverage is inside the band; a few draws with a poor pre-period fit inflate the Monte Carlo SD.

### Multilevel models (`se_screen/multilevel.py`)

100 or 200 groups of 10, random intercept sd 0.7, within-group regressor
with conditional coefficient 0.5.

| row | call | n | B | coverage | SE/SD | bias/SD | flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `mixed/random-intercept` | `mixed(y, x_fixed, group)` | 1000 | 400 | 0.945 | 0.97 | -0.00 |  |
| `mixed/ml` | `mixed(..., method="ml")` | 1000 | 400 | 0.948 | 0.94 | +0.00 |  |
| `mixed/random-slope` | `mixed(..., x_random=["x1"])` | 1000 | 400 | 0.940 | 0.94 | +0.07 |  |
| `meglm/gaussian` | `meglm(..., family="gaussian")` | 1000 | 400 | 0.948 | 1.01 | +0.04 |  |
| `melogit/laplace` | `melogit(y, x_fixed, group)` | 2000 | 400 | 0.958 | 1.02 | +0.07 |  |
| `melogit/agq7` | `melogit(..., nAGQ=7)` | 2000 | 400 | 0.948 | 1.02 | -0.02 |  |
| `mepoisson/laplace` | `mepoisson(y, x_fixed, group)` | 1000 | 400 | 0.930 | 0.96 | -0.01 |  |
| `menbreg/laplace` | `menbreg(y, x_fixed, group)` | 2000 | 400 | 0.932 | 0.97 | +0.01 |  |
| `megamma/laplace` | `megamma(y, x_fixed, group)` | 1000 | 400 | 0.965 | 1.07 | +0.06 |  |
| `meologit/laplace` | `meologit(y, x_fixed, group)` | 2000 | 200 | 0.960 | 0.99 | +0.02 |  |
| `mixed/random-slope-diagonal` | `mixed(..., x_random=["x1"], cov_type="diagonal")` | 1000 | 400 | 0.943 | 1.02 | +0.03 |  |
| `mixed/three-level` | `mixed(..., group=["school", "klass"])` | 1000 | 400 | 0.945 | 0.97 | +0.02 |  |
| `mixed/between-regressor` | `mixed(y, x_fixed, group), group-level x2` | 1000 | 400 | 0.970 | 1.03 | -0.05 |  |
| `mepoisson/agq7` | `mepoisson(..., nAGQ=7)` | 1000 | 400 | 0.968 | 1.02 | -0.06 |  |
| `mepoisson/random-slope` | `mepoisson(..., x_random=["x1"])` | 1000 | 200 | 0.950 | 0.96 | -0.02 |  |
| `meglm/binomial` | `meglm(..., family="binomial")` | 2000 | 400 | 0.960 | 1.00 | +0.07 |  |
| `melogit/between-regressor` | `melogit(y, x_fixed, group), group-level x2` | 2000 | 400 | 0.922 | 0.93 | -0.04 |  |

17 rows, 0 flagged. What the table says:

- Linear mixed models (REML, ML, random slopes, three levels) and the
  generalized models under Laplace and adaptive quadrature are in band,
  for within-group and group-level regressors.
