# StatsPAI design-based staggered-rollout parity (R side) -- Module 82.
#
# Runs staggered::staggered / staggered_cs / staggered_sa (Roth & Sant'Anna's
# own package, the reference implementation for their 2023 JPE Micro paper) on
# the randomised rollout panel dumped by 82_staggered.py.
#
# Unlike every other DiD module in this harness, this one reconciles a
# *design-based* estimator: identification comes from random adoption timing,
# not parallel trends. The package reports two standard errors -- a
# conservative Neyman bound and an adjusted one that subtracts the variance
# the randomisation identifies -- and both are emitted, since reconciling only
# one would leave half the inference path unchecked.
#
#   install.packages("staggered")
#
# Tolerance: rel < 1e-6 on every estimate and both standard errors.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages(library(staggered))

MODULE <- "82_staggered"
EVENT_TIMES <- c(0, 1, 2)

df <- as.data.frame(read_csv_strict(MODULE))
n_units <- length(unique(df$unit))

rows <- list()

add_rows <- function(rows, statistic, res) {
  rows[[length(rows) + 1]] <- parity_row(
    module = MODULE, statistic = statistic,
    estimate = res$estimate, se = NA_real_, n = n_units
  )
  rows[[length(rows) + 1]] <- parity_row(
    module = MODULE, statistic = sprintf("%s_se_neyman", statistic),
    estimate = res$se_neyman, se = NA_real_, n = n_units
  )
  # The package prints the adjusted SE as plain `se`.
  rows[[length(rows) + 1]] <- parity_row(
    module = MODULE, statistic = sprintf("%s_se_adjusted", statistic),
    estimate = res$se, se = NA_real_, n = n_units
  )
  rows
}

for (estimand in c("simple", "cohort", "calendar")) {
  for (tag in c("efficient", "plugin")) {
    beta <- if (tag == "efficient") NULL else 1
    res <- staggered::staggered(
      df = df, i = "unit", t = "time", g = "first_treat", y = "y",
      estimand = estimand, beta = beta
    )
    rows <- add_rows(rows, sprintf("%s_%s", estimand, tag), res)
  }
}

for (e in EVENT_TIMES) {
  res <- staggered::staggered(
    df = df, i = "unit", t = "time", g = "first_treat", y = "y",
    estimand = "eventstudy", eventTime = e
  )
  rows <- add_rows(rows, sprintf("eventstudy_e%d", e), res)
}

res_cs <- staggered::staggered_cs(
  df = df, i = "unit", t = "time", g = "first_treat", y = "y",
  estimand = "simple"
)
rows <- add_rows(rows, "cs_simple", res_cs)

res_sa <- staggered::staggered_sa(
  df = df, i = "unit", t = "time", g = "first_treat", y = "y",
  estimand = "simple"
)
rows <- add_rows(rows, "sa_simple", res_sa)

write_results(MODULE, rows, extra = list(
  reference = "staggered::staggered",
  staggered = as.character(utils::packageVersion("staggered"))
))
