#!/usr/bin/env Rscript
# R HonestDiD on its own array example (BCdata_EventStudy: betahat, sigma,
# 4 pre / 4 post periods): relative magnitudes (Conditional) and smoothness
# (FLCI) for the first post period and for the post-period average.
#   Rscript tests/reference_parity/_fixtures/_generate_honest_did_moments_R.R
suppressMessages({library(HonestDiD); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
d <- HonestDiD::BCdata_EventStudy
b <- d$betahat; S <- d$sigma
npre <- length(d$prePeriodIndices); npost <- length(d$postPeriodIndices)
Mbar <- c(0, 0.5, 1); M <- c(0, 0.01, 0.02)
out <- list(betahat = unname(b), sigma = unname(S), num_pre = npre, num_post = npost,
            Mbar = Mbar, M = M)
for (tgt in c("e0", "avg")) {
  lv <- if (tgt == "e0") basisVector(1, npost) else matrix(rep(1 / npost, npost), ncol = 1)
  rm <- createSensitivityResults_relativeMagnitudes(betahat = b, sigma = S,
          numPrePeriods = npre, numPostPeriods = npost, Mbarvec = Mbar,
          l_vec = lv, method = "Conditional")
  sd <- createSensitivityResults(betahat = b, sigma = S, numPrePeriods = npre,
          numPostPeriods = npost, Mvec = M, l_vec = lv, method = "FLCI")
  out[[paste0("rm_", tgt)]] <- list(lb = rm$lb, ub = rm$ub)
  out[[paste0("sd_", tgt)]] <- list(lb = sd$lb, ub = sd$ub)
}
out$meta <- list(R = R.version.string, HonestDiD = as.character(packageVersion("HonestDiD")))
write_json(out, file.path(here, "honest_did_moments_R.json"), digits = NA, auto_unbox = TRUE)
cat("wrote honest_did_moments_R.json\n")
