#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_cbps_R_parity.py (sp.cbps).
#
# CBPS (GPL >= 2) is run as a black box: only its documented arguments and
# returned values are used; its source was not read. Run from the repository
# root after _generate_cbps_medsens_data.py:
#   Rscript tests/reference_parity/_fixtures/_generate_cbps_R.R
# Writes _fixtures/cbps_R.json (digits = 17).
#
# For each data set, estimand (ATT = 0 / 1) and method ("over" / "exact"):
#   * the default fit (twostep = TRUE): coefficients, J, mle.J, fitted
#     propensity scores, Horvitz-Thompson weights (standardize = FALSE) and
#     the within-arm normalised weights (standardize = TRUE);
#   * the continuously-updated fit (twostep = FALSE): coefficients and J,
#     used to check the moment covariance (Imai & Ratkovic 2014, eqs. 16-17)
#     independently of any optimiser;
#   * the logit MLE from glm, the starting point of both fits.
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({ library(CBPS); library(jsonlite) })
FIX <- "tests/reference_parity/_fixtures"
sets <- list(
  sim = list(file = "cbps_sim.csv", fml = d ~ x1 + x2 + x3),
  bound = list(file = "cbps_bound.csv", fml = d ~ x1 + x2),
  lalonde = list(file = "matching_lalonde.csv",
                 fml = treat ~ age + educ + married + nodegree + re74 + re75 + black + hispan)
)
out <- list(versions = list(R = paste(R.version$major, R.version$minor, sep = "."),
                            CBPS = as.character(packageVersion("CBPS"))))
# CBPS prints a note for ATT fits; keep the log clean.
quiet <- function(expr) {
  v <- NULL
  invisible(capture.output(v <- suppressWarnings(expr)))
  v
}
for (nm in names(sets)) {
  s <- sets[[nm]]
  df <- read.csv(file.path(FIX, s$file))
  mle <- glm(s$fml, data = df, family = binomial())
  res <- list(mle = as.numeric(coef(mle)))
  for (att in c(0, 1)) for (meth in c("over", "exact")) {
    f_ht <- quiet(CBPS(s$fml, data = df, ATT = att, method = meth, standardize = FALSE))
    f_st <- quiet(CBPS(s$fml, data = df, ATT = att, method = meth, standardize = TRUE))
    f_cu <- quiet(CBPS(s$fml, data = df, ATT = att, method = meth, standardize = FALSE,
                       twostep = FALSE))
    res[[paste0(if (att == 1) "att" else "ate", "_", meth)]] <- list(
      coef = as.numeric(coef(f_ht)), J = f_ht$J, mle_J = f_ht$mle.J,
      converged = f_ht$converged, fitted = as.numeric(f_ht$fitted.values),
      weights_ht = as.numeric(f_ht$weights), weights_std = as.numeric(f_st$weights),
      cue_coef = as.numeric(coef(f_cu)), cue_J = f_cu$J, cue_mle_J = f_cu$mle.J)
  }
  out[[nm]] <- res
}
writeLines(toJSON(out, digits = 17, auto_unbox = TRUE, pretty = TRUE),
           file.path(FIX, "cbps_R.json"))
cat("wrote", file.path(FIX, "cbps_R.json"), "\n")
