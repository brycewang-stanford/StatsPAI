# StatsPAI BJS imputation parity (R side) -- Module 16.
#
# Reads data/16_bjs.csv (the StatsPAI mpdta replica) and runs
# didimputation::did_imputation. Tolerance: rel < 1e-3.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(didimputation)
})

MODULE <- "16_bjs"

df <- read_csv_strict(MODULE)
df$first_treat <- as.numeric(df$first_treat)

# didimputation::did_imputation expects a data.frame with the
# standard mpdta-like columns; treated cohorts are non-zero
# first_treat rows, never-treated rows have first_treat = 0 (we
# pass through unchanged).
fit <- didimputation::did_imputation(
  data    = df,
  yname   = "lemp",
  gname   = "first_treat",
  tname   = "year",
  idname  = "countyreal"
)

# fit is a tibble with columns: lhs, term, estimate, std.error, ...
# When horizon is unspecified the package returns the simple ATT
# under term == "treat".
agg <- fit[fit$term == "treat", ]

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "att_bjs",
    estimate  = agg$estimate[1],
    se        = agg$std.error[1],
    ci_lo     = agg$estimate[1] - qnorm(0.975) * agg$std.error[1],
    ci_hi     = agg$estimate[1] + qnorm(0.975) * agg$std.error[1],
    n         = nrow(df)
  )
)

write_results(MODULE, rows, extra = list(method = "did_imputation"))
