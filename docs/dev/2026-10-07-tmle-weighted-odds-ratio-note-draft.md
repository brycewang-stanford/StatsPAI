# Draft note to the `tmle` maintainers: weighted odds-ratio variance

Status: draft, not sent. Bryce sends messages himself. The R snippet was
run on 2026-10-07 and prints the numbers quoted under it.

Suggested venue: an email to the maintainer named in the package's
DESCRIPTION, Susan Gruber `<sgruber@cal.berkeley.edu>` (read from the
installed `tmle` 2.1.1; not otherwise checked). The DESCRIPTION lists no
issue tracker, and the only GitHub repository is the read-only CRAN
mirror, so there is nowhere to open a pull request.

---

**Title:** `tmle()`: variance of the odds ratio with `obsWeights` uses an
uncentred influence curve

Hello,

Thank you for maintaining `tmle`. I maintain StatsPAI, an open-source
Python package for causal inference and econometrics, and I use `tmle` as
the reference for its TMLE: with the same initial fits, the two are
compared number by number. In that comparison against `tmle` 2.1.1 I found
one number that I could not reproduce until I dropped a centring term, and
I think it is worth a look.

**What I see.** With the initial `Q` and `g1W` supplied and a binary
outcome, the treatment-specific means, the additive effect, the risk ratio
and the odds ratio agree with my calculation to about 1e-11, in estimate
and variance, with and without `id`. With `obsWeights`, everything still
agrees except `estimates$OR$var.log.psi`, which is larger than mine, by
2.4% in the example below.

**What reproduces it.** Write `w` for the weights normalised to mean one,
`EY1` and `EY0` for the targeted means, and

```
D1 = A / g * (Y - Q*(A, W)) + Q*(1, W) - EY1
D0 = (1 - A) / (1 - g) * (Y - Q*(A, W)) + Q*(0, W) - EY0
```

The delta method gives the influence curve of the log odds ratio as

```
IC = w * ( D1 / (EY1 (1 - EY1)) - D0 / (EY0 (1 - EY0)) )
```

`var(IC) / n` is my number. I get `tmle`'s number to 1e-12 from

```
IC + w * ( 1 / (1 - EY1) - 1 / (1 - EY0) )
```

That is what one obtains if `Q*(1, W)` and `Q*(0, W)` enter the curve
without `EY1` and `EY0` subtracted. For the risk ratio the corresponding
constant is `EY1 / EY1 - EY0 / EY0 = 0`, which would explain why the risk
ratio agrees. Without weights the extra term is a constant and `var()`
removes it, which would explain why the unweighted odds ratio agrees.
With weights it is a constant times `w`, so it adds roughly
`c^2 * var(w) / n` to the variance, where
`c = 1 / (1 - EY1) - 1 / (1 - EY0)`.

**Why I think the centred version is the intended one.** An influence
curve has mean zero. The uncentred one has mean `c`, and
`estimates$IC$IC.logOR` has that mean in the weighted fit.

**Reproduction.**

```r
library(tmle)
set.seed(1)
n <- 960
x1 <- rnorm(n); x2 <- rnorm(n); x3 <- runif(n)
w  <- exp(0.4 * x2 + 0.3 * runif(n))
d  <- rbinom(n, 1, plogis(-0.2 + 0.6 * x1 - 0.4 * x2 + 0.5 * x3))
y  <- rbinom(n, 1, plogis(-0.3 + (0.8 + 0.4 * x2) * d + 0.5 * x1 - 0.3 * x3))
dat <- data.frame(x1, x2, x3, d, y)
g1 <- predict(glm(d ~ x1 + x2 + x3, binomial, dat), type = "response")
qf <- glm(y ~ d + x1 + x2 + x3, binomial, dat)
Q  <- cbind(predict(qf, transform(dat, d = 0), type = "response"),
            predict(qf, transform(dat, d = 1), type = "response"))
f  <- tmle(Y = y, A = d, W = dat[, 1:3], Q = Q, g1W = g1,
           family = "binomial", obsWeights = w)
ic <- f$estimates$IC$IC.logOR
mean(ic)                       # not zero
e1 <- f$estimates$EY1$psi; e0 <- f$estimates$EY0$psi
wn <- w / mean(w)
centred <- ic - wn * (1 / (1 - e1) - 1 / (1 - e0))
mean(centred)                  # zero
c(reported = f$estimates$OR$var.log.psi, centred = var(centred) / n)
```

With `tmle` 2.1.1 this prints a mean of 1.082 for the reported curve,
5e-16 for the centred one, and variances of 0.021947 (reported) against
0.021428 (centred), a difference of 2.4%.

I may be misreading the intended convention. If the uncentred form is
deliberate I would be glad to know the reasoning.

For context, the comparison lives in StatsPAI's test suite, where the
weighted odds ratio is the one row that does not match and the test
rebuilds your number from the centred curve:
https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/reference_parity/test_tmle_parameters_R_parity.py

Best regards,
Bryce Wang
Stanford REAP, Stanford University
brycew6m@stanford.edu
StatsPAI: https://github.com/brycewang-stanford/StatsPAI
(Wang and Rozelle, Journal of Open Source Software, 2026,
https://doi.org/10.21105/joss.10604)
