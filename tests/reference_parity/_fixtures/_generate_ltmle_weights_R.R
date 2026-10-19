# Reference values for sp.ltmle(weights=) vs R ltmle's observation.weights;
# checked in tests/reference_parity/test_ltmle_weights_parity.py.
# Run from this folder:  Rscript _generate_ltmle_weights_R.R
# ltmle is GPL-2: used as a black box (outputs only).
# Data: teffects_wide.csv / teffects_wide_cens.csv; observation weights
# sw = 0.4 + (row %% 5) * 0.3 (rows numbered from 1), added here and on the
# Python side. Conventions as _generate_teffects_R.R: SL.library = NULL
# (glm), default gbounds, variance.method = "ic".
suppressPackageStartupMessages({ library(ltmle); library(jsonlite) })
wide <- read.csv("teffects_wide.csv")
widec <- read.csv("teffects_wide_cens.csv")
sw <- function(d) 0.4 + (seq_len(nrow(d)) %% 5) * 0.3
summ <- function(r) {
  s <- summary(r)
  list(psi1 = s$effect.measures$treatment$estimate,
       psi0 = s$effect.measures$control$estimate,
       ate = s$effect.measures$ATE$estimate,
       ate_se = s$effect.measures$ATE$std.dev,
       psi1_se = s$effect.measures$treatment$std.dev)
}
out <- list()
for (ycol in c("Yb", "Y")) {
  d <- wide[, c("v", "L0", "A0", "L1", "A1", ycol)]
  r <- suppressMessages(ltmle(d, Anodes = c("A0", "A1"), Lnodes = "L1",
                              Ynodes = ycol, abar = list(c(1, 1), c(0, 0)),
                              SL.library = NULL, estimate.time = FALSE,
                              variance.method = "ic", observation.weights = sw(d)))
  out[[paste0("nocens_", ycol)]] <- summ(r)
  d <- widec[, c("v", "L0", "A0", "C0", "L1", "A1", "C1", ycol)]
  w <- sw(d)
  d$C0 <- BinaryToCensoring(is.uncensored = d$C0)
  d$C1 <- BinaryToCensoring(is.uncensored = d$C1)
  r <- suppressMessages(ltmle(d, Anodes = c("A0", "A1"), Cnodes = c("C0", "C1"),
                              Lnodes = "L1", Ynodes = ycol,
                              abar = list(c(1, 1), c(0, 0)), SL.library = NULL,
                              estimate.time = FALSE, variance.method = "ic",
                              observation.weights = w))
  out[[paste0("cens_", ycol)]] <- summ(r)
}
out$versions <- list(R = R.version.string, ltmle = as.character(packageVersion("ltmle")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null"),
           "ltmle_weights_R.json")
