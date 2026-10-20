# Draft upstream report: DRDID `drdid_rc` / `drdid_rc1` standard error

For Bryce to send himself (an issue on the DRDID repository, or e-mail to
Pedro Sant'Anna). Not posted. Written from black-box runs of DRDID 1.3.0
only; the package source was not read.

---

**Title:** `drdid_rc` / `drdid_rc1`: sign of the pre-period control
regression term in the influence function?

Hi Pedro,

While checking our Python implementation against DRDID 1.3.0 we found that
the standard errors of `drdid_rc` and `drdid_rc1` differ from what we get
by applying the delta method to the estimators in Sant'Anna and Zhao (2020,
eqs. 3.3-3.4; Appendix B). Treating the package as a black box:

- the ATT agrees to machine precision;
- `att.inf.func` minus our delta-method influence function is an exact
  linear function of the influence function of the pre-period control
  outcome regression (R² = 1 - 3e-15), with coefficient
  `-2 (M_10 - M_00)`, where `M_dt` is the arm-weighted covariate mean of
  cell (d, t). In other words, that one estimation-effect term enters with
  the opposite sign; the post-period regression, treated-cell and
  propensity-score terms agree.

The two versions coincide asymptotically when the propensity score is
correctly specified (`M_10 - M_00 -> 0`), but not otherwise, and the
estimator is meant to be consistent in that case. A reproducible example
in R alone (misspecified propensity score, correct control outcome
regressions, true ATT = 1):

```r
library(DRDID)
set.seed(11)
B <- 300; n <- 2000
est <- se <- numeric(B)
for (b in seq_len(B)) {
  x <- rnorm(n)
  p <- plogis(-1.5 + 3 * pmax(x, 0)^2)       # the logit in x is misspecified
  d <- as.numeric(runif(n) < p)
  post <- as.numeric(runif(n) < 0.5)
  e <- rnorm(n) * ifelse(d == 0 & post == 0, 4, 0.5)
  y <- 1 + 3 * x + 0.5 * d + post * (0.5 + 1 * x + 1 * d) + e
  r <- drdid_rc(y = y, post = post, D = d, covariates = cbind(1, x))
  est[b] <- r$ATT; se[b] <- r$se
}
c(sd = sd(est), mean_se = mean(se),
  coverage = mean(abs(est - 1) / se < qnorm(0.975)))
```

On our machine (R 4.5, DRDID 1.3.0) this gives sd(ATT) = 0.303, mean SE =
0.188 and 95% coverage 0.757. In our 1,000-draw version of the same design
the delta-method SE averages 0.279 against an sd of 0.277 (coverage 0.952),
and the SE with the package's sign averages 0.188 (coverage 0.813). The
same holds for `drdid_rc1`, and for `did::att_gt(panel = FALSE,
est_method = "dr")`, which calls it. Stata's `drdid, dripw` on repeated
cross-sections gives the same numbers as the R package.

Could you check whether the sign of that term is intended? If it would
help, we can share the identity test (the difference between the two
influence functions equals `2 l_00' (M_10 - M_00)` to 1e-12 on our data).

Thanks, and thanks for the package,
Bryce

---

Notes for Bryce (not part of the message):

- StatsPAI's default is the delta-method sign since this commit
  (`variance_convention='delta'`); `'reference'` reproduces DRDID / did /
  Stata. Evidence: `docs/dev/cleanroom/did_drdid.md`,
  `tests/reference_parity/_fixtures/_drdid_rc_pre_regression_mc.py`.
- The `drdid_imp_rc` / `drdid_imp_rc1` estimators and every panel
  estimator agree with DRDID; only the traditional DR repeated-cross-section
  estimators are affected.
