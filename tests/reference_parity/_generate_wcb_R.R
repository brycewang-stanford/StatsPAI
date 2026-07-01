#!/usr/bin/env Rscript
# Frozen reference for sp.wild_cluster_bootstrap vs R fwildclusterboot::boottest
# (Rademacher weights, B=999, seed=42). Regenerate: Rscript .../_generate_wcb_R.R
suppressPackageStartupMessages({library(fwildclusterboot); library(jsonlite)})
df <- read.csv("tests/reference_parity/_fixtures/wcb_data.csv")
# Use lm + boottest with B=999, seed=42 (Rademacher)
m <- lm(y ~ d, data = df)
res <- boottest(
  m, param = "d", B = 999, seed = 42, clustid = "clust", engine = "R"
)
out <- list(
  t_stat = unname(res$statistic),
  p_value = unname(res$p_val),
  ci_lo = unname(res$conf_int[1]),
  ci_hi = unname(res$conf_int[2])
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           "tests/reference_parity/_fixtures/wcb_R.json")
cat("wrote wcb_R.json\n")
