#!/usr/bin/env Rscript
# Frozen reference for sp.glm vs base R stats::glm (binomial logit + Poisson log).
# IRLS converges to the same unpenalized MLE, so coefficients and model-based
# SEs match to machine precision. Regenerate:
#   Rscript tests/reference_parity/_generate_glm_R.R
suppressPackageStartupMessages({library(jsonlite)})
df <- read.csv("tests/reference_parity/_fixtures/glm_data.csv")

fit_b <- glm(yb ~ x1 + x2, data = df, family = binomial())
fit_p <- glm(yc ~ x1 + x2, data = df, family = poisson())

pack <- function(fit) {
  s <- summary(fit)
  list(
    coef = as.list(coef(fit)),
    se = as.list(s$coefficients[, "Std. Error"]),
    deviance = unname(fit$deviance),
    null_deviance = unname(fit$null.deviance),
    aic = unname(fit$aic),
    loglik = unname(as.numeric(logLik(fit)))
  )
}

out <- list(
  binomial = pack(fit_b),
  poisson = pack(fit_p),
  provenance = list(
    r_version = R.version.string,
    generated_by = "tests/reference_parity/_generate_glm_R.R"
  )
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           "tests/reference_parity/_fixtures/glm_R.json")
cat("wrote glm_R.json\n")
