rlassoIV(select.X = FALSE) omits the second-stage intercept

Hello, and thank you for maintaining hdm.

With `select.X = FALSE` and `select.Z = TRUE`, `rlassoIV` fits the first
stage with an intercept but the second stage without one. When the outcome
or the endogenous regressor has a nonzero mean, the slope is then
inconsistent. A minimal example:

```r
library(hdm); library(AER)
set.seed(1)
n <- 2000
z1 <- rnorm(n); z2 <- rnorm(n); u <- rnorm(n)
d <- 5 + 0.8*z1 + 0.5*z2 + 0.5*u + rnorm(n)
y <- 10 - 2*d + u                      # true effect is -2
f <- rlassoIV(x = NULL, d = d, y = y, z = cbind(z1, z2),
              select.X = FALSE, select.Z = TRUE)
coef(f)                                 # -0.0654
coef(ivreg(y ~ d | z1 + z2))["d"]       # -2.049
```

Both instruments are kept by the lasso here, so the two estimates should
coincide. Centring `y` and `d` before the call, or adding an intercept to
the second-stage regression, recovers about -2.05.

We also saw a related difference in the reported variance when some
controls are selected out of the first stage. We are happy to send details
and a separate example if useful.

Thank you.
