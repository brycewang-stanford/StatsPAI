#!/usr/bin/env Rscript
# synthdid placebo and bootstrap standard errors at several seeds, for the
# stochastic screen (S) in test_sdid_resampling_screen.py.
#
# synthdid 0.0.9 (Arkhangelsky, Athey, Hirshberg, Imbens & Wager), the
# authors' own package. synthdid_se(estimate, method, replications = 200)
# after set.seed(s), s = 1..5, for sc_estimate / did_estimate /
# synthdid_estimate on two committed panels: the Prop. 99 replica of Track A
# module 12 (one treated unit: placebo only, the bootstrap is undefined) and
# did_synth_sdid_panel.csv (five treated units).
#
# These are Monte Carlo draws from R's stream. They pin nothing to the last
# digit: the per-draw map from an index vector to an estimate is pinned
# exactly in test_did_synth_R_parity.py, and these numbers only screen the
# end-to-end SE.
#
#   Rscript tests/reference_parity/_generate_sdid_resampling_R.R
suppressPackageStartupMessages({
  library(synthdid)
  library(jsonlite)
})

script_dir <- tryCatch(
  dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))),
  error = function(e) "tests/reference_parity"
)
fixtures <- file.path(script_dir, "_fixtures")
repo <- normalizePath(file.path(script_dir, "..", ".."))

SEEDS <- 1:5
REPS <- 200L

one_dataset <- function(path, unit, time, outcome, treatment, methods) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  panel <- panel.matrices(df, unit = unit, time = time, outcome = outcome,
                          treatment = treatment)
  out <- list(N0 = panel$N0, N1 = nrow(panel$Y) - panel$N0)
  for (m in c("sdid", "sc", "did")) {
    est <- switch(m,
      sdid = synthdid_estimate(panel$Y, panel$N0, panel$T0),
      sc = sc_estimate(panel$Y, panel$N0, panel$T0),
      did = did_estimate(panel$Y, panel$N0, panel$T0))
    res <- list(estimate = as.numeric(est))
    for (se_method in methods) {
      res[[se_method]] <- vapply(SEEDS, function(s) {
        set.seed(s)
        as.numeric(synthdid_se(est, method = se_method, replications = REPS))
      }, numeric(1))
    }
    out[[m]] <- res
  }
  out
}

res <- list(
  provenance = list(
    R = R.version.string,
    synthdid = as.character(packageVersion("synthdid")),
    seeds = SEEDS,
    replications = REPS
  ),
  prop99 = one_dataset(file.path(repo, "tests", "r_parity", "data", "12_sdid.csv"),
                       "state", "year", "cigsale", "treated", c("placebo")),
  panel = one_dataset(file.path(fixtures, "did_synth_sdid_panel.csv"),
                      "unit", "year", "y", "treated", c("placebo", "bootstrap"))
)

writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           file.path(fixtures, "sdid_resampling_R.json"))
cat("wrote", file.path(fixtures, "sdid_resampling_R.json"), "\n")
