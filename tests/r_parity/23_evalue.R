# StatsPAI E-value parity (R side) -- Module 23.
#
# Runs EValue::evalues.RR on three canonical inputs and writes the
# E-value for the point estimate plus the E-value for the CI limit.
# Tolerance: rel < 1e-6 (closed-form).

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(EValue)
})

MODULE <- "23_evalue"

CASES <- list(
  list(rr = 2.5, lo = 1.8, hi = 3.2, label = "moderate"),
  list(rr = 4.0, lo = 2.5, hi = 6.0, label = "strong"),
  list(rr = 1.3, lo = 1.0, hi = 1.6, label = "borderline")
)

rows <- list()
for (case in CASES) {
  ev <- EValue::evalues.RR(est = case$rr, lo = case$lo, hi = case$hi)
  # ev is a 2x3 matrix-like with rows c("RR", "E-values") and cols
  # c("point", "lower", "upper")
  ev_point <- ev["E-values", "point"]
  ev_ci    <- ev["E-values", "lower"]
  if (is.na(ev_ci)) ev_ci <- ev["E-values", "upper"]
  rows[[length(rows) + 1L]] <- parity_row(
    module = MODULE,
    statistic = paste0("evalue_est_", case$label),
    estimate = ev_point, n = 1
  )
  rows[[length(rows) + 1L]] <- parity_row(
    module = MODULE,
    statistic = paste0("evalue_ci_", case$label),
    estimate = ev_ci, n = 1
  )
}

write_results(MODULE, rows, extra = list(measure = "RR"))
