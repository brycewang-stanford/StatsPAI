Dear Yiqing,

I hope you are well. While checking our Python implementation of the fect
estimators against fect 2.4.1, we ran into one thing in the pre-trend F
test under the jackknife. I wanted to bring it to you directly rather than
open an issue, in case I am missing something. Everything below is from
fect 2.4.1 on R 4.5.2.

With `vartype = "jackknife"` the pre-trend F statistic in `test.out` and
`loo.test.out` appears to use the sample covariance of the leave-one-unit-out
replicates, `cov(t(pre.att.boot))`. The jackknife covariance is
`(n - 1)^2 / n` times larger. fect's own standard errors in `pre.est.att`
are on the jackknife scale, so the F statistic and the reported standard
errors disagree. With 24 units the factor is 22.0, and the F statistic is
inflated by the same factor.

Reproduction. A public CSV with 24 units and 14 periods (staggered
adoption plus units that switch off).

```r
library(fect)
url <- paste0("https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/",
              "6fdc22bef90d03defabfc45b5b26efd9cde60e8f/",
              "tests/reference_parity/_fixtures/synth_family_fect.csv")
fd <- read.csv(url)
n <- length(unique(fd$id))
fit <- fect(Y ~ D, data = fd, index = c("id", "time"), method = "fe",
            force = "two-way", se = TRUE, vartype = "jackknife", tol = 1e-12,
            max.iteration = 20000, parallel = FALSE, loo = TRUE, nboots = n)
pre <- fit$pre.est.att
rep <- fit$pre.att.boot                     # rows as in pre.est.att
rep <- rep[, apply(!is.na(rep), 2, all), drop = FALSE]
v_cov <- apply(rep, 1, var)                 # diagonal of cov(t(rep))
cat("units:", n, "  replicates:", ncol(rep), "  (n - 1)^2 / n =", (n - 1)^2 / n, "\n")
print(cbind(se_reported = pre[, 2], sd_replicates = sqrt(v_cov),
            ratio_of_squares = pre[, 2]^2 / v_cov))
cat("F from fect:", fit$test.out$f.stat, " p =", fit$test.out$f.p, "\n")
f_jk <- fit$test.out$f.stat * n / (n - 1)^2
cat("F with the jackknife covariance:", f_jk, " p =",
    pf(f_jk, fit$test.out$df1, fit$test.out$df2, lower.tail = FALSE), "\n")
```

Output on our machine.

```
units: 24   replicates: 24   (n - 1)^2 / n = 22.04167
   se_reported sd_replicates ratio_of_squares
-7   0.4636565    0.09875841         22.04167
-6   0.1993939    0.04247073         22.04167
-5   0.2021877    0.04306579         22.04167
-4   0.1699788    0.03620533         22.04167
-3   0.1175938    0.02504738         22.04167
-2   0.1399086    0.02980040         22.04167
-1   0.1601192    0.03410525         22.04167
0    0.1351062    0.02877749         22.04167
F from fect: 15.99313  p = 0.0002071337
F with the jackknife covariance: 0.7255862  p = 0.6562724
```

The ratio of the squared reported SE to the replicate variance is exactly
`(n - 1)^2 / n` in every row. In `diagtest` the F statistic is built from
`S <- cov(t(coef_mat))` with `coef_mat` the rows of `pre.att.boot`. Under
the bootstrap that is the right covariance. Under the jackknife the
replicates are leave-one-out estimates, and the covariance needs the factor
`(n - 1)^2 / n`. With the factor, F here is 0.73 (p = 0.66) instead of
16.0 (p = 0.0002). `loo.test.out` shows the same factor (16.46 in fect).

A possible fix is to scale `S` by `(n - 1)^2 / n` when
`vartype = "jackknife"`, where `n` is the number of replicates used. The
equivalence test in `f.equiv.p` uses the same F statistic and would change
with it.

Is the unscaled covariance intended for the jackknife? If it is a bug, I
am happy to open an issue on the repository or send a pull request,
whichever is easier for you. And if there is a reason for it that I have
missed, I would be glad to learn it.

Thank you, as always, for fect.

Best,
Bryce
