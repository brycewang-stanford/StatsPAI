Tukey loss: tau2.se looks about 3.7 times too small

Dear GRAPPLE maintainers,

Thank you for GRAPPLE. We have been checking an independent implementation
of the robust profile likelihood against `grappleRobustEst`. With the L2
and Huber losses its standard errors agree with ours to 1e-6 or better
when both are evaluated at GRAPPLE's own estimates. With
the Tukey loss they do not, and the difference has a simple form.

What we see. With `loss.function = "tukey"` and the default `k = 4.685`,
`tau2.se` is exactly `6 / k^2` times the value we get from the sandwich in
S2 Text when the whole sandwich uses one scale of the loss. That factor is
0.273, so the consistent value is 3.658 times larger. The point estimates
agree. The beta standard error also moves, by about 0.1% on typical data
and by up to 15% on weak-instrument data with `tau2.hat = 0`.

Our reading is that two scales of the biweight are mixed. The tau2
equation and `Var rho(Z)` behave as if `rho` were scaled to a maximum of 1,
while the derivative `rho'` behaves as if it were on the `k^2 / 6` scale of
the paper. The estimating equations do not care about that factor, but the
sandwich does. If both parts use the same scale, the variance does not
depend on how `rho` is scaled, which is what we would expect.

A quick sign without any simulation. On one simulated data set with 40
variants and pleiotropy, the three losses give almost the same `tau2.hat`
(1.32e-4 Tukey, 1.32e-4 Huber, 1.20e-4 L2). Their `tau2.se` values are
3.42e-5 (Tukey), 1.26e-4 (Huber) and 1.20e-4 (L2). Multiplying the Tukey
value by 3.658 gives 1.25e-4. The data and outputs are in the fixture
linked below (scenario `hetero_tau`).

Monte Carlo. With 60 variants, heterogeneous standard errors, beta = 0.5
and tau = 0.02, our implementation over 400 replicates gives a mean
`tau2` standard error of 1.55e-4. The empirical SD of `tau2.hat` is 1.59e-4.
The Tukey `tau2.se` on GRAPPLE's scale averages 4.2e-5 in the same study.
The snippet below repeats a smaller version of this check with GRAPPLE
alone.

```r
## GRAPPLE 0.2.2, commit 317e837. The estimator file is sourced because
## the package does not install with the archived haploR.
url <- paste0("https://raw.githubusercontent.com/jingshuw/GRAPPLE/",
              "317e837340a29129c52c9d9b22f5f8bae72e27d0/R/mr_fun_general.R")
src <- tempfile(fileext = ".R"); download.file(url, src, quiet = TRUE)
source(src)

reps <- 100                       # about 4 minutes; 400 for a tighter check
set.seed(20261021)
m <- 60; beta <- 0.5; tau <- 0.02
gam <- runif(m, 0.05, 0.2); sx <- runif(m, 0.01, 0.03); sy <- runif(m, 0.01, 0.03)
one <- function(loss) {
  bx <- rnorm(m, gam, sx)
  by <- rnorm(m, beta * gam + rnorm(m, 0, tau), sy)
  gd <- data.frame(SNP = paste0("v", 1:m), gamma_exp1 = bx, se_exp1 = sx,
                   gamma_out1 = by, se_out1 = sy)
  g <- suppressWarnings(grappleRobustEst(gd, loss.function = loss,
                                         diagnosis = FALSE, plot.it = FALSE))
  c(tau2 = g$tau2.hat, tau2_se = g$tau2.se)
}
for (loss in c("tukey", "l2")) {
  set.seed(20261022)
  r <- t(replicate(reps, one(loss)))
  cat(sprintf("%-6s sd(tau2.hat) = %.3g  mean(tau2.se) = %.3g  ratio = %.2f\n",
              loss, sd(r[, "tau2"]), mean(r[, "tau2_se"]),
              sd(r[, "tau2"]) / mean(r[, "tau2_se"])))
}
cat(sprintf("k^2/6 at k = 4.685: %.3f\n", 4.685^2 / 6))
```

Output on our machine:

```
tukey  sd(tau2.hat) = 0.000187  mean(tau2.se) = 4.48e-05  ratio = 4.18
l2     sd(tau2.hat) = 0.000181  mean(tau2.se) = 0.000161  ratio = 1.13
k^2/6 at k = 4.685: 3.658
```

With 100 replicates the L2 SE is within Monte Carlo noise of the spread of
the estimates. The Tukey SE is about four times too small, close to the
factor `k^2 / 6`.

The data set and the outputs we compared against are here, pinned to one
commit:
<https://github.com/brycewang-stanford/StatsPAI/blob/857c4559906e8bba5144cfa0fe8a47464099581c/tests/reference_parity/_fixtures/grapple_R.json>
(data: `grapple_sim.csv` in the same folder).

We may have misread something. If the Tukey scale is intended, we would be
glad to know the reasoning, and we will document it on our side.

With best regards,
Bryce Wang
