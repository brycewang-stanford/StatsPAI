#!/usr/bin/env Rscript
# Frozen reference for sp.three_sls vs R systemfit::systemfit(method="3SLS")
# on a 2-equation simultaneous system (each equation's other endogenous variable
# instrumented by the full exogenous set). Regenerate:
#   Rscript tests/reference_parity/_generate_threesls_R.R
suppressPackageStartupMessages({library(systemfit); library(jsonlite)})
df <- read.csv("tests/reference_parity/_fixtures/threesls_data.csv")

eq1 <- y1 ~ y2 + x1
eq2 <- y2 ~ y1 + x2
system <- list(eq1 = eq1, eq2 = eq2)
inst <- ~ x1 + x2

fit <- systemfit(system, method = "3SLS", inst = inst, data = df)
s <- summary(fit)

pack <- function(eq) {
  co <- coef(summary(fit$eq[[eq]]))
  list(
    coef = as.list(co[, "Estimate"]),
    se = as.list(co[, "Std. Error"])
  )
}

out <- list(
  eq1 = pack(1),
  eq2 = pack(2),
  provenance = list(
    r_version = R.version.string,
    systemfit_version = as.character(packageVersion("systemfit")),
    generated_by = "tests/reference_parity/_generate_threesls_R.R"
  )
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           "tests/reference_parity/_fixtures/threesls_R.json")
cat("wrote threesls_R.json\n")
