# StatsPAI functional-form test parity (R side) -- Module 79.
#
# Runs didFF::didFF (Sant'Anna's own package, the reference implementation
# for Roth & Sant'Anna 2023 Econometrica) on the two panels dumped by
# 79_didff.py.
#
# Two designs: `pt` does not reject, `rej` does. Both are needed -- the
# p-value saturates at 1 whenever the max-t statistic is negative, so an
# accept-only fixture would leave the critical value untested.
#
# didFF is not on CRAN:
#   remotes::install_github("pedrohcgs/didFF")
#
# control_group is passed explicitly. didFF 0.1.0 forwards its default,
# the length-two c("nevertreated", "notyettreated"), to did. did <= 2.3.0
# took the first element; did >= 2.5.0 rejects a non-scalar value. The
# documented didFF default is "nevertreated", so the estimand is unchanged.
#
# Tolerance: rel < 1e-6 on the per-bin implied densities.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages(library(didFF))

MODULE <- "79_didff"
N_BINS <- 8

rows <- list()
for (tag in c("pt", "rej")) {
  df <- as.data.frame(read_csv_strict(sprintf("%s_%s", MODULE, tag)))
  res <- suppressWarnings(didFF(
    data = df, yname = "y", tname = "t", idname = "id", gname = "g",
    control_group = "nevertreated",
    nbins = N_BINS, seed = 0, numSims = 100000
  ))
  for (k in seq_len(nrow(res$table))) {
    rows[[length(rows) + 1]] <- parity_row(
      module    = MODULE,
      statistic = sprintf("%s_density_%d", tag, k),
      estimate  = res$table$implied_density[k],
      se        = NA_real_,
      n         = length(unique(df$id))
    )
  }
  rows[[length(rows) + 1]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("%s_pvalue", tag),
    estimate  = as.numeric(res$pval),
    se        = NA_real_,
    n         = length(unique(df$id))
  )
}

write_results(MODULE, rows, extra = list(
  reference = "didFF::didFF",
  didFF = as.character(utils::packageVersion("didFF"))
))
