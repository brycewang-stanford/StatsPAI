# Draft issue for r-causal/halfmoon: `check_model_auc()` is slightly below the Mann-Whitney AUC

Not sent. For Bryce to post at <https://github.com/r-causal/halfmoon/issues>
if he wants to. Found during the pass recorded in
`2026-10-05-barrett-causal-inference-in-r-review.md`.

---

**Title:** `check_model_auc()` / `bal_model_auc()` underestimates the AUC: `compute_auc()` sorts by FPR without breaking ties on TPR

**Body:**

Thanks for halfmoon. While checking another package against it I found that
`check_model_auc()` returns a value a little below the Mann-Whitney
probability, also without weights and without tied scores.

```r
library(halfmoon)
set.seed(1)
n <- 500
x <- rnorm(n)
z <- rbinom(n, 1, plogis(1.2 * x))
d <- data.frame(z = factor(z), p = fitted(glm(z ~ x, family = binomial())))

check_model_auc(d, .exposure = z, .fitted = p)$auc
#> 0.7993391048

unname(wilcox.test(d$p[z == 1], d$p[z == 0])$statistic) / (sum(z == 1) * sum(z == 0))
#> 0.7995551359

as.numeric(pROC::auc(z, d$p, quiet = TRUE))
#> 0.7995551359
```

halfmoon 0.2.0, R 4.5.2.

**Cause.** The ROC points from `compute_roc_curve_imp()` are correct. In
`compute_auc(x, y)` the points arrive in order of decreasing FPR, so
`is.unsorted(x)` is true and they are sorted with `order(x)`. Wherever the
curve rises vertically, several points share one FPR, and the stable sort
leaves them in their incoming order, which is decreasing TPR. The trapezoid
to the next FPR then starts from the lowest TPR of the group, not the
highest, and each vertical step loses a sliver of area.

Sorting on both coordinates gives the rank statistic exactly:

```r
roc <- check_model_roc_curve(d, z, p)
fpr <- 1 - roc$specificity
tpr <- roc$sensitivity
trap <- function(o) sum((tpr[o][-1] + tpr[o][-length(o)]) / 2 * diff(fpr[o]))

trap(order(fpr))        # what compute_auc() does
#> 0.7993391048
trap(order(fpr, tpr))
#> 0.7995551359
```

**Suggested fix.** In `compute_auc()`, replace `ord <- order(x)` by
`ord <- order(x, y)`. The same applies with weights: the weighted AUC then
equals the weighted Mann-Whitney probability
`sum_ij w_i w_j [1(s_i > s_j) + 1(s_i = s_j) / 2] / (W1 * W0)`.

The size of the gap depends on the data (3e-4 here, 6e-5 for the weighted
AUC of the Seven Dwarfs example in *Causal Inference in R*), so no
conclusion in the book changes. It matters for anyone comparing numbers
across packages.
