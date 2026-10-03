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

Rerun with `python tests/reliability/few_clusters.py` (about a quarter of
an hour). `tests/test_reliability_few_clusters.py` recomputes one cell on
its first 60 replications and checks the statements above against the
stored file.
