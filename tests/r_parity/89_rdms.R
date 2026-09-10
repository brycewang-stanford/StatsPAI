# StatsPAI multi-score RD parity (R side) -- Module 89.
#
# Reads data/89_rdms.csv and runs rdmulti::rdms at three boundary points
# along a vertical boundary at x1 = 0.
#
# rdms returns, per boundary point:
#   B      bias-corrected point estimate  (rdrobust Estimate[2])
#   Coefs  conventional point estimate    (rdrobust Estimate[1])
#   V      robust variance                (so sqrt(V) is the robust SE)
#   H      selected bandwidth (left, right)
#   Nh     effective sample size (left, right)
#
# The printed table shows Coefs with the robust CI, which is why the two
# point estimates are pinned separately here rather than trusting the
# display.
#
# Tolerance: rel < 1e-6.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(rdmulti)
})

MODULE <- "89_rdms"

df <- read_csv_strict(MODULE)

BOUNDARY_X2 <- c(-0.5, 0.0, 0.5)

tag_for <- function(v) {
  s <- sprintf("%+.1f", v)
  s <- sub("\\+", "p", s)
  s <- sub("-", "m", s)
  s <- gsub("\\.", "", s)
  paste0("b", s)
}

fit <- rdmulti::rdms(
  Y = df$y,
  X = df$x1,
  C = rep(0, length(BOUNDARY_X2)),
  X2 = df$x2,
  C2 = BOUNDARY_X2,
  zvar = df$z
)

rows <- list()
for (i in seq_along(BOUNDARY_X2)) {
  tag <- tag_for(BOUNDARY_X2[i])
  add <- function(rows, stat, est, se = NA) {
    rows[[length(rows) + 1L]] <- parity_row(
      module    = MODULE,
      statistic = sprintf("%s_%s", tag, stat),
      estimate  = est,
      se        = if (is.na(se)) NULL else se,
      n         = nrow(df)
    )
    rows
  }
  rows <- add(rows, "biascorrected_est", fit$B[1, i], sqrt(fit$V[1, i]))
  rows <- add(rows, "conventional_est", fit$Coefs[1, i], sqrt(fit$V_cl[1, i]))
  rows <- add(rows, "bandwidth_h", fit$H[1, i])
  rows <- add(rows, "n_eff_left", fit$Nh[1, i])
  rows <- add(rows, "n_eff_right", fit$Nh[2, i])
}

write_results(
  MODULE, rows,
  extra = list(
    boundary_points = sprintf("(0, %g)", BOUNDARY_X2),
    n_rows = length(rows),
    rdmulti_version = as.character(utils::packageVersion("rdmulti"))
  )
)
