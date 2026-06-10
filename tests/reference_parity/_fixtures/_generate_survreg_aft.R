#!/usr/bin/env Rscript
# R reference values for sp.survreg (parametric AFT survival) parity.
#
#   sp.survreg(formula, event=, dist=) -> survival::survreg(Surv(time, status)
#   ~ ., dist=) for the four AFT distributions weibull / exponential /
#   lognormal / loglogistic. Both use the AFT (log-time) parameterization;
#   sp's `log(sigma)` parameter equals R's `Log(scale)` = log(survreg$scale).
#
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_survreg_aft.R
suppressMessages({
  library(survival)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/survreg_aft_data.csv")
dists <- c("weibull", "exponential", "lognormal", "loglogistic")

out <- list()
for (d in dists) {
  f <- survreg(Surv(time, status) ~ x1 + x2, data = df, dist = d)
  co <- coef(f)
  se <- sqrt(diag(vcov(f)))
  entry <- list(
    coef = list(
      "_cons" = unname(co["(Intercept)"]),
      x1 = unname(co["x1"]),
      x2 = unname(co["x2"])
    ),
    se = list(
      "_cons" = unname(se["(Intercept)"]),
      x1 = unname(se["x1"]),
      x2 = unname(se["x2"])
    ),
    log_scale = if (d == "exponential") 0.0 else log(f$scale)
  )
  if ("Log(scale)" %in% names(se)) {
    entry$se[["log_sigma"]] <- unname(se[["Log(scale)"]])
  }
  out[[d]] <- entry
}

writeLines(
  toJSON(out, auto_unbox = TRUE, digits = 15),
  "tests/reference_parity/_fixtures/survreg_aft_R.json"
)
cat("wrote survreg_aft_R.json\n")
