# StatsPAI Honest-DiD parity (R side) -- Module 10.
#
# Mirrors the hand-crafted event study used by 10_honest_did.py and
# runs HonestDiD::createSensitivityResults under "FLCI" (the
# finite-sample length-optimised CI under the smoothness restriction
# Delta^SD), which is the closest match to sp.honest_did's smoothness
# restriction. Tolerance: abs < 0.05 on the CI bounds.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(HonestDiD)
})

MODULE <- "10_honest_did"
M_GRID <- c(0.0, 0.05, 0.1, 0.2, 0.5)

# Match the Python event study exactly:
#   pre  rel-time {-3,-2,-1} : att = (0.01, -0.02, 0.0), SE = 0.05
#   post rel-time { 0, 1, 2} : att = (0.5,   0.4,   0.3), SE = 0.10
betahat <- c(0.01, -0.02, 0.0,   0.5, 0.4, 0.3)
ses     <- c(0.05, 0.05,  0.05,  0.10, 0.10, 0.10)
sigma   <- diag(ses^2)

# Suppress the cone-solver "may be inaccurate" warnings that come
# from the small synthetic example.
sens <- suppressWarnings(
  HonestDiD::createSensitivityResults(
    betahat = betahat,
    sigma   = sigma,
    numPrePeriods  = 3,
    numPostPeriods = 3,
    Mvec    = M_GRID,
    method  = "FLCI",
    alpha   = 0.05
  )
)

rows <- list()
for (i in seq_along(M_GRID)) {
  m <- M_GRID[i]
  lb <- sens$lb[i]; ub <- sens$ub[i]
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("ci_lower_M_%g", m),
    estimate  = lb, n = 1000
  )
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("ci_upper_M_%g", m),
    estimate  = ub, n = 1000
  )
}

write_results(MODULE, rows,
              extra = list(method = "FLCI", alpha = 0.05))
