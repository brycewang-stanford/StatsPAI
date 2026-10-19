# Reference values for sp.ebalance(weights=) (sampling weights) vs R
# WeightIt's s.weights. Checked in
# tests/reference_parity/test_ebalance_sweights_parity.py.
# Run from this folder:  Rscript _generate_ebalance_sweights_R.R
# WeightIt is GPL: used as a black box (outputs only).
# Data: the CSVs written by _generate_ebalance_weightit_R.R. Sampling
# weights sw = 0.4 + (row %% 5) * 0.3 and base weights
# bw = 0.5 + (row %% 4) / 4 (rows numbered from 1) are added here and on
# the Python side.
suppressPackageStartupMessages({ library(WeightIt); library(jsonlite) })
cases <- list(sim_cia = c("X1", "X2"),
              lalonde = c("age", "educ", "black", "hispan", "married",
                          "nodegree", "re74", "re75"))
out <- list()
for (nm in names(cases)) {
  df <- read.csv(paste0("ebalance_", nm, ".csv"))
  df$sw <- 0.4 + (seq_len(nrow(df)) %% 5) * 0.3
  df$bw <- 0.5 + (seq_len(nrow(df)) %% 4) / 4
  f <- as.formula(paste("d ~", paste(cases[[nm]], collapse = " + ")))
  for (est in c("ATT", "ATE", "ATC")) {
    for (bw in c(FALSE, TRUE)) {
      W <- if (bw) weightit(f, data = df, method = "ebal", estimand = est,
                            s.weights = df$sw, base.weights = df$bw)
           else weightit(f, data = df, method = "ebal", estimand = est,
                         s.weights = df$sw)
      fit <- lm_weightit(y ~ d, data = df, weightit = W, vcov = "asympt")
      out[[paste(nm, est, if (bw) "bw" else "nobw", sep = "_")]] <- list(
        data = paste0("ebalance_", nm, ".csv"), covariates = cases[[nm]],
        estimand = est, base_weights = bw,
        est = unname(coef(fit)["d"]), se = unname(sqrt(diag(vcov(fit)))["d"]),
        w = unname(W$weights))
    }
  }
}
out$`_provenance` <- list(WeightIt = as.character(packageVersion("WeightIt")),
                          R = R.version.string, vcov = "asympt")
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE),
           "ebalance_sweights_R.json")
