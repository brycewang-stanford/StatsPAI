#!/usr/bin/env Rscript
# Frozen reference for sp.het_test / sp.reset_test vs R lmtest.
#   * het_test  : lmtest::bptest  (studentized Breusch-Pagan, Koenker — the
#                 lmtest default that sp.het_test reproduces)
#   * reset_test: lmtest::resettest(power = 2:3, type = "fitted")
# Regenerate: Rscript tests/reference_parity/_generate_diagnostics_R.R
suppressPackageStartupMessages({library(lmtest); library(jsonlite)})
df <- read.csv("tests/reference_parity/_fixtures/diagnostics_data.csv")
m  <- lm(y ~ x1 + x2, data = df)
bp <- bptest(m)
rs <- resettest(m, power = 2:3, type = "fitted")
out <- list(
  breusch_pagan = list(statistic = unname(bp$statistic),
                       df = unname(bp$parameter),
                       pvalue = unname(bp$p.value)),
  reset = list(statistic = unname(rs$statistic),
               df1 = unname(rs$parameter[1]),
               df2 = unname(rs$parameter[2]),
               pvalue = unname(rs$p.value)),
  provenance = list(r_version = R.version.string,
                    lmtest_version = as.character(packageVersion("lmtest")),
                    generated_by = "tests/reference_parity/_generate_diagnostics_R.R")
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           "tests/reference_parity/_fixtures/diagnostics_R.json")
cat("wrote diagnostics_R.json\n")
