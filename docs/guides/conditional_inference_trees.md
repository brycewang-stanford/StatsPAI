# Conditional inference trees

`sp.ctree` grows a tree that decides whether to split with a hypothesis
test. It is the method of Hothorn, Hornik and Zeileis (2006), known to R
users as `party::ctree` and `partykit::ctree`, and it is the tree in
Varian's (2014) mortgage-lending example.

## What it does

In each node, every covariate is tested for association with the outcome.
The test is a permutation test in its asymptotic form, so no resampling is
done. The covariate with the smallest p-value is selected. The cut point is
then the one with the largest two-sample statistic. When the smallest
p-value, adjusted for the number of covariates tested, is not below
`alpha`, the node becomes a leaf.

Two consequences follow.

- The size of the tree is set by the significance level. There is no
  pruning step and no cross-validation.
- The variable is chosen by a test and the cut point afterwards. A
  covariate with many possible cut points therefore has no advantage over a
  binary one. A tree that searches all cut points of all covariates at once
  favours the former.

## Example

```python
import statspai as sp

df = sp.cps_wage()
fit = sp.ctree(
    df, y="log_wage",
    covariates=["female", "education", "experience", "tenure", "union", "married"],
    max_depth=2,
)
print(fit)
```

```text
Conditional inference tree (regression)
----------------------------------------------------
Outcome        : log_wage
Observations   : 3000
Inner nodes    : 3
Terminal nodes : 4
Test           : bonferroni, alpha = 0.05

[1] root  (p = 0)
|   [2] experience <= 18.3723  (p = 3.89e-79)
|   |   [3] education <= 14.9973: 3.2715 (n = 1583, var = 0.23066)
|   |   [4] education > 14.9973: 3.73729 (n = 357, var = 0.262561)
|   [5] experience > 18.3723  (p = 2.36e-73)
|   |   [6] experience <= 31.7972: 4.05617 (n = 838, var = 0.260537)
|   |   [7] experience > 31.7972: 4.75367 (n = 222, var = 0.326441)
```

`fit.nodes` has one row per node and `fit.tests` one row per test that was
run, with the statistic, its degrees of freedom, and the raw and adjusted
p-values. `fit.predict(new_data)` returns the leaf mean, or the majority
class for a classification tree. `type="prob"` gives class shares and
`type="node"` the leaf itself.

For a 0/1 outcome stored as a number, pass `task="classification"` to get
class predictions. The tree is the same either way.

## Coming from R

| `partykit::ctree_control` | `sp.ctree` |
| --- | --- |
| `alpha` (`1 - mincriterion`) | `alpha` |
| `testtype = "Bonferroni"` / `"Univariate"` | `test="bonferroni"` / `"univariate"` |
| `minsplit`, `minbucket`, `minprob` | `min_split`, `min_bucket`, `min_prob` |
| `maxdepth` | `max_depth` |
| `splitstat = "quadratic"` / `"maximum"` | `split_stat=` |

With the same data and settings the two programs return the same tree.
Split variables, cut points and the leaf of every row are identical, test
statistics agree to 1e-9 and adjusted p-values to 1e-7
(`tests/reference_parity/test_ctree_partykit_parity.py`, twelve settings).
Nodes are numbered as in `partykit`. A factor in R corresponds to a string
or categorical column, and an ordered factor to an ordered categorical.

Not implemented: the maximum-type test statistic (`teststat = "maximum"`),
Monte Carlo p-values, surrogate splits for missing values, case weights,
and multivariate outcomes. Rows with missing values are refused.

## What the p-values are not

The p-values belong to the tests that grew the tree. They are not valid
for inference on the fitted tree, because the splits were chosen on the
same data. A variable that does not appear in the tree may still matter.
The tests are marginal, so an outcome that depends on two covariates only
through their interaction produces no split at all. And a variable that
predicts poorly can still have a non-zero partial association with the
outcome. In Varian's example race does not change the tree's predictions,
yet the logit coefficient on race is far from zero. For that second
question use a regression or `sp.dml`.

For honest inference on subgroups found by a tree, split the sample or use
`sp.causal_forest` with `sp.rate_split`. For prediction accuracy a forest
usually does better than a single tree. See `sp.regression_forest` and
`sp.probability_forest`, whose defaults (`honest=True`, `split_alpha=0.05`)
are chosen for inference and should be relaxed for pure prediction.

## References

- Hothorn, T., Hornik, K. and Zeileis, A. (2006). Unbiased Recursive
  Partitioning: A Conditional Inference Framework. *Journal of
  Computational and Graphical Statistics* 15(3), 651-674.
  doi:10.1198/106186006X133933
- Varian, H. R. (2014). Big Data: New Tricks for Econometrics. *Journal of
  Economic Perspectives* 28(2), 3-28. doi:10.1257/jep.28.2.3
