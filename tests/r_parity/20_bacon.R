# StatsPAI Goodman-Bacon decomposition parity (R side) -- Module 20.
#
# Reads data/20_bacon.csv (the StatsPAI mpdta replica) and runs
# bacondecomp::bacon. Tolerance: rel < 1e-3 on the overall TWFE
# coefficient.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(bacondecomp)
})

MODULE <- "20_bacon"

df <- read_csv_strict(MODULE)

# bacondecomp::bacon returns either a tibble of (type, weight,
# avg_est) -- when summary_only is TRUE -- or a list of two tibbles
# when summary_only is FALSE. Default behaviour returns a list and
# prints the summary.
res <- bacondecomp::bacon(lemp ~ treat, data = df,
                          id_var = "countyreal", time_var = "year",
                          quietly = TRUE)

# `res` is the per-pair data.frame.
beta_twfe <- sum(res$weight * res$estimate)
weighted_sum <- beta_twfe   # same identity
neg_weight_share <- sum(res$weight[res$weight < 0]) / sum(abs(res$weight))
n_comparisons <- nrow(res)

rows <- list(
  parity_row(MODULE, "beta_twfe", estimate = beta_twfe, n = nrow(df)),
  parity_row(MODULE, "weighted_sum", estimate = weighted_sum, n = nrow(df)),
  parity_row(MODULE, "negative_weight_share",
             estimate = neg_weight_share, n = nrow(df))
)

# Per-pair estimates. R uses 99999 to denote untreated.
for (i in seq_len(nrow(res))) {
  treated_cohort <- as.integer(res$treated[i])
  ctrl_raw <- as.integer(res$untreated[i])
  ctrl_str <- if (ctrl_raw == 99999) "never" else as.character(ctrl_raw)
  rows[[length(rows) + 1L]] <- parity_row(
    module    = MODULE,
    statistic = paste0("pair_", treated_cohort, "_vs_", ctrl_str, "_est"),
    estimate  = res$estimate[i],
    n         = nrow(df)
  )
}

write_results(MODULE, rows, extra = list(n_comparisons = n_comparisons))
