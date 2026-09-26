#!/usr/bin/env Rscript
# Frozen reference for sp.rdrobust(weights=) vs R rdrobust(weights=).
# Regenerate from the repository root:
#   Rscript tests/reference_parity/_generate_rd_weights_R.R
# Data: _fixtures/rd_weights_data.csv (written once by the test module's
# generator snippet; 3% of rows carry weight 0).
suppressPackageStartupMessages({library(rdrobust); library(jsonlite)})
d <- read.csv("tests/reference_parity/_fixtures/rd_weights_data.csv")
run <- function(...) {
  r <- rdrobust(...)
  list(coef = unname(as.numeric(r$coef)), se = unname(as.numeric(r$se)),
       h = unname(as.numeric(r$bws[1, ])), b = unname(as.numeric(r$bws[2, ])),
       N_h = unname(as.numeric(r$N_h)))
}
res <- list(
  sharp_mserd      = run(d$y, d$x, weights = d$w),
  sharp_unweighted = run(d$y, d$x),
  sharp_msetwo_hc1 = run(d$y, d$x, weights = d$w, bwselect = "msetwo", vce = "hc1"),
  sharp_cerrd_p2   = run(d$y, d$x, weights = d$w, bwselect = "cerrd", p = 2),
  sharp_covs       = run(d$y, d$x, weights = d$w, covs = d$z1),
  sharp_cluster    = run(d$y, d$x, weights = d$w, cluster = d$cl),
  sharp_hc3_uniform = run(d$y, d$x, weights = d$w, vce = "hc3", kernel = "uniform"),
  sharp_fixed_h    = run(d$y, d$x, weights = d$w, h = 0.4, b = 0.6),
  fuzzy_mserd      = run(d$yf, d$x, fuzzy = d$d, weights = d$w),
  fuzzy_comb2      = run(d$yf, d$x, fuzzy = d$d, weights = d$w, bwselect = "msecomb2"),
  fuzzy_comb2_unweighted = run(d$yf, d$x, fuzzy = d$d, bwselect = "msecomb2"),
  provenance = list(r_version = R.version.string,
                    rdrobust_version = as.character(packageVersion("rdrobust")),
                    generated_by = "tests/reference_parity/_generate_rd_weights_R.R"))
writeLines(toJSON(res, digits = NA, auto_unbox = TRUE, pretty = TRUE),
           "tests/reference_parity/_fixtures/rd_weights_R.json")
cat("wrote rd_weights_R.json\n")
