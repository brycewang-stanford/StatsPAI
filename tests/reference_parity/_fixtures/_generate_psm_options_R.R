#!/usr/bin/env Rscript
# R reference for tests/reference_parity/test_psm_options_parity.py:
# Matching::Match (Sekhon) on the options Stata's matching commands do not
# cover, or cover with conventions of their own.
#
# Run _generate_psm_options_data.py first, then, from the repository root,
#   Rscript tests/reference_parity/_fixtures/_generate_psm_options_R.R
# Writes psm_options_R.json.
#
# Matching is GPL-3: it is run as a black box, its source is not read.
# Every fit matches on the logit propensity score (X = ps) for the ATT with
# M = 1, and reports the Abadie-Imbens (2006) variance of the sample ATT
# (sample = TRUE) with sigma^2(X) from Var.calc = 1 same-arm neighbour. The
# data have no ties, and distance.tolerance is set far below the closest
# pair of scores so that no two distances are pooled as a tie.

suppressPackageStartupMessages({
  library(Matching)
  library(jsonlite)
})

FIX <- "tests/reference_parity/_fixtures"
df <- read.csv(file.path(FIX, "psm_options_data.csv"))
COV <- c("x1", "x2", "x3", "x4")
ps <- as.numeric(fitted(glm(d ~ x1 + x2 + x3 + x4, data = df,
                            family = binomial(link = "logit"))))
Z <- as.matrix(df[, COV])
TOL <- 1e-12

fit <- function(...) {
  m <- Matching::Match(Y = df$y, Tr = df$d, X = ps, estimand = "ATT",
                       M = 1L, ties = TRUE, distance.tolerance = TOL,
                       sample = TRUE, Var.calc = 1L, ...)
  list(est = unname(m$est[1]), se = if (is.null(m$se)) NA_real_ else unname(m$se[1]),
       n_treated_matched = length(unique(m$index.treated)),
       n_dropped = if (is.null(m$ndrops)) 0L else unname(m$ndrops))
}

res <- list()
res$match_nn1 <- c(fit(), cmd = "Match(X = ps)")
# bias adjustment on the four covariates (Z), the matching on the score
res$match_nn1_biasadj <- c(fit(BiasAdjust = TRUE, Z = Z),
                           cmd = "Match(X = ps, Z = covariates, BiasAdjust = TRUE)")
# a caliper of 0.01 on the score: Match reads caliper in standard
# deviations of X, so 0.01 / sd(ps)
for (cal in c(0.01, 0.005)) {
  key <- sprintf("match_nn1_cal%s", sub(".", "p", format(cal), fixed = TRUE))
  res[[key]] <- c(fit(caliper = cal / sd(ps)),
                  cmd = sprintf("Match(X = ps, caliper = %s / sd(ps))", format(cal)))
}
# without replacement; the rows are processed in data order
res$match_nn1_norepl <- c(fit(replace = FALSE), cmd = "Match(X = ps, replace = FALSE)")

res$`_meta` <- list(
  matching_version = as.character(packageVersion("Matching")),
  r_version = R.version.string,
  sd_ps = sd(ps)
)
writeLines(toJSON(res, auto_unbox = TRUE, digits = I(17), na = "null", pretty = TRUE),
           file.path(FIX, "psm_options_R.json"))
cat("wrote psm_options_R.json\n")
