# Reference values for sp.ltmle(design=) -- survey linearisation of the
# ltmle influence curve; checked in
# tests/reference_parity/test_ltmle_design_parity.py.
# Run from this folder:  Rscript _generate_ltmle_design_R.R
# ltmle (GPL-2) and survey (GPL-2 | GPL-3) are used as black boxes
# (documentation and outputs only).
#
# Data: teffects_wide.csv / teffects_wide_cens.csv with the observation
# weights of _generate_ltmle_weights_R.R, sw = 0.4 + (row %% 5) * 0.3, and a
# design added here and on the Python side: stratum = (row - 1) %/% 100 + 1
# (10 strata), psu = (row - 1) %/% 20 + 1 (5 PSUs of 20 per stratum),
# fpc = 40 PSUs per stratum in the population.
# The ATE's influence curve is the difference of the IC$tmle of two
# single-regime fits (abar = c(1, 1) and c(0, 0)); its sd / sqrt(n) equals
# the ATE std.dev of the two-regime fit (stored as a check). The design SE
# is sqrt(svyrecvar(IC_ate / n, design)).
suppressPackageStartupMessages({ library(ltmle); library(survey); library(jsonlite) })
wide <- read.csv("teffects_wide.csv")
widec <- read.csv("teffects_wide_cens.csv")
add_design <- function(d) {
  i <- seq_len(nrow(d))
  d$sw <- 0.4 + (i %% 5) * 0.3
  d$stratum <- (i - 1) %/% 100 + 1
  d$psu <- (i - 1) %/% 20 + 1
  d$fpc <- 40
  d
}
designs <- function(d) list(
  full = svydesign(ids = ~psu, strata = ~stratum, fpc = ~fpc, weights = ~sw, data = d),
  nofpc = svydesign(ids = ~psu, strata = ~stratum, weights = ~sw, data = d),
  psu_only = svydesign(ids = ~psu, weights = ~sw, data = d)
)
out <- list()
for (cens in c(FALSE, TRUE)) for (ycol in c("Yb", "Y")) {
  base <- add_design(if (cens) widec else wide)
  if (cens) {
    d <- base[, c("v", "L0", "A0", "C0", "L1", "A1", "C1", ycol)]
    d$C0 <- BinaryToCensoring(is.uncensored = d$C0)
    d$C1 <- BinaryToCensoring(is.uncensored = d$C1)
    fit <- function(abar) suppressMessages(ltmle(d, Anodes = c("A0", "A1"),
           Cnodes = c("C0", "C1"), Lnodes = "L1", Ynodes = ycol, abar = abar,
           SL.library = NULL, estimate.time = FALSE, variance.method = "ic",
           observation.weights = base$sw))
  } else {
    d <- base[, c("v", "L0", "A0", "L1", "A1", ycol)]
    fit <- function(abar) suppressMessages(ltmle(d, Anodes = c("A0", "A1"),
           Lnodes = "L1", Ynodes = ycol, abar = abar, SL.library = NULL,
           estimate.time = FALSE, variance.method = "ic",
           observation.weights = base$sw))
  }
  s <- summary(fit(list(c(1, 1), c(0, 0))))
  ic <- fit(c(1, 1))$IC$tmle - fit(c(0, 0))$IC$tmle
  n <- nrow(base)
  key <- paste0(if (cens) "cens_" else "nocens_", ycol)
  res <- list(ate = s$effect.measures$ATE$estimate,
              ate_se_ic = s$effect.measures$ATE$std.dev,
              ic_se_check = sd(ic) / sqrt(n))
  for (nm in names(designs(base))) {
    des <- designs(base)[[nm]]
    v <- svyrecvar(matrix(ic / n, ncol = 1), des$cluster, des$strata, des$fpc,
                   postStrata = des$postStrata)
    res[[nm]] <- list(se = sqrt(as.numeric(v)), degf = degf(des))
  }
  out[[key]] <- res
}
out$versions <- list(R = R.version.string, ltmle = as.character(packageVersion("ltmle")),
                     survey = as.character(packageVersion("survey")))
writeLines(toJSON(out, auto_unbox = TRUE, digits = I(17), pretty = TRUE, na = "null"),
           "ltmle_design_R.json")
