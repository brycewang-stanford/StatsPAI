# StatsPAI continuous-treatment DiD parity (R side) -- Module 80.
#
# Runs contdid::cont_did (Callaway, Goodman-Bacon & Sant'Anna's own package)
# on the panel dumped by 80_contdid.py, across three spline specifications.
#
# contdid is not on CRAN:
#   remotes::install_github("bcallaway11/contdid")
#
# Standard errors are not emitted: contdid routes them through the pte
# package's aggregation layer, which StatsPAI does not replicate.
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

suppressPackageStartupMessages(library(contdid))

MODULE <- "80_contdid"
GRID_POINTS <- c(1, 31, 61, 90)  # 1-based; the Python side uses 0, 30, 60, 89

df <- as.data.frame(read_csv_strict(MODULE))

rows <- list()
for (spec in list(c(1, 0), c(3, 0), c(3, 2))) {
  degree <- spec[1]
  num_knots <- spec[2]
  tag <- sprintf("d%dk%d", degree, num_knots)
  res <- suppressWarnings(cont_did(
    yname = "Y", tname = "time_period", idname = "id", dname = "D",
    data = df, gname = "G", target_parameter = "slope",
    aggregation = "dose", treatment_type = "continuous",
    dose_est_method = "parametric", control_group = "nevertreated",
    degree = degree, num_knots = num_knots, bstrap = FALSE, cband = FALSE
  ))
  for (k in seq_along(GRID_POINTS)) {
    j <- GRID_POINTS[k]
    rows[[length(rows) + 1]] <- parity_row(
      module = MODULE, statistic = sprintf("%s_att_d_%d", tag, j - 1),
      estimate = as.numeric(res$att.d[j]), se = NA_real_, n = nrow(df) / 2
    )
    rows[[length(rows) + 1]] <- parity_row(
      module = MODULE, statistic = sprintf("%s_acrt_d_%d", tag, j - 1),
      estimate = as.numeric(res$acrt.d[j]), se = NA_real_, n = nrow(df) / 2
    )
  }
  rows[[length(rows) + 1]] <- parity_row(
    module = MODULE, statistic = sprintf("%s_overall_att", tag),
    estimate = as.numeric(res$overall_att), se = NA_real_, n = nrow(df) / 2
  )
  rows[[length(rows) + 1]] <- parity_row(
    module = MODULE, statistic = sprintf("%s_overall_acrt", tag),
    estimate = as.numeric(res$overall_acrt), se = NA_real_, n = nrow(df) / 2
  )
}

write_results(MODULE, rows, extra = list(
  reference = "contdid::cont_did",
  contdid = as.character(utils::packageVersion("contdid"))
))
