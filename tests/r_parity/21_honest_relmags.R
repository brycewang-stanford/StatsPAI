# StatsPAI Honest-DiD relative-magnitudes parity (R side) -- Module 21.
#
# Mirrors the hand-crafted event study used by 21_honest_relmags.py
# and runs HonestDiD::createSensitivityResults_relativeMagnitudes.
# Tolerance: abs < 0.10 on each CI bound.

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

MODULE <- "21_honest_relmags"
MBAR_GRID <- c(0.0, 0.5, 1.0, 1.5, 2.0)

betahat <- c(0.01, -0.02, 0.0,   0.5, 0.4, 0.3)
ses     <- c(0.05, 0.05,  0.05,  0.10, 0.10, 0.10)
sigma   <- diag(ses^2)

sens <- suppressWarnings(
  HonestDiD::createSensitivityResults_relativeMagnitudes(
    betahat = betahat,
    sigma   = sigma,
    numPrePeriods  = 3,
    numPostPeriods = 3,
    Mbarvec = MBAR_GRID,
    alpha   = 0.05
  )
)

rows <- list()
for (i in seq_along(MBAR_GRID)) {
  m <- MBAR_GRID[i]
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("ci_lower_Mbar_%g", m),
    estimate  = sens$lb[i], n = 1000
  )
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("ci_upper_Mbar_%g", m),
    estimate  = sens$ub[i], n = 1000
  )
}

write_results(MODULE, rows,
              extra = list(method = "relative_magnitudes", alpha = 0.05))
