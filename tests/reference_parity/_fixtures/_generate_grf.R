#!/usr/bin/env Rscript
# R grf reference: causal_forest ATE on the shared seed=42 DGP.
suppressMessages({
  library(grf)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/grf_data.csv")
X <- as.matrix(df[, paste0("X", 1:5)])
y <- df$y
W <- df$W

set.seed(42)
cf <- causal_forest(X, y, W, num.trees = 2000, seed = 42)

# Doubly-robust ATE estimate
ate <- average_treatment_effect(cf, target.sample = "all")

out <- list(
  meta = list(
    R_version = R.version.string,
    grf_version = as.character(packageVersion("grf")),
    num_trees = 2000,
    seed = 42L
  ),
  ate = list(
    estimate = unname(ate["estimate"]),
    se       = unname(ate["std.err"])
  ),
  n_obs = nrow(df)
)
write_json(out, "tests/reference_parity/_fixtures/grf_R.json",
           pretty = TRUE, auto_unbox = TRUE, digits = NA)
cat(sprintf("ATE: estimate=%.6f  se=%.6f\n", ate["estimate"], ate["std.err"]))
