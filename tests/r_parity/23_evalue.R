# StatsPAI E-value parity (R side) -- Module 23.
#
# Runs EValue::evalues.{RR,OR,HR,MD,OLS,RD} on the same inputs as the
# Python side (23_evalue.py) and writes, per case, the E-value for the
# point estimate and the E-value for the CI limit closest to the null.
# Tolerance: rel < 1e-6 (closed-form / deterministic grid).

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

# Extract (point, ci) from an EValue::evalues.* matrix result. The CI
# E-value is the limit nearest the null (lower if present, else upper).
mat_pair <- function(e) {
  pt <- e["E-values", "point"]
  lo <- e["E-values", "lower"]
  hi <- e["E-values", "upper"]
  ci <- if (!is.na(lo)) lo else hi
  c(point = unname(pt), ci = unname(ci))
}

# label -> function returning c(point=, ci=)
cases <- list(
  # First three keep their original labels so the committed Stata anchor
  # (23_evalue.do) still joins in compare.py.
  moderate      = function() mat_pair(evalues.RR(2.5, 1.8, 3.2)),
  strong        = function() mat_pair(evalues.RR(4.0, 2.5, 6.0)),
  borderline    = function() mat_pair(evalues.RR(1.3, 1.0, 1.6)),
  rr_protective = function() mat_pair(evalues.RR(0.6, 0.4, 0.9)),
  rr_crossnull  = function() mat_pair(evalues.RR(1.1, 0.9, 1.3)),
  rr_nonnull    = function() mat_pair(evalues.RR(2.5, 1.8, 3.2, true = 1.5)),
  or_common     = function() mat_pair(evalues.OR(2.0, 1.5, 2.7, rare = FALSE)),
  or_rare       = function() mat_pair(evalues.OR(2.0, 1.5, 2.7, rare = TRUE)),
  hr_common     = function() mat_pair(evalues.HR(1.5, 1.1, 2.0, rare = FALSE)),
  hr_rare       = function() mat_pair(evalues.HR(1.5, 1.1, 2.0, rare = TRUE)),
  md            = function() mat_pair(evalues.MD(0.3, 0.1)),
  ols           = function() mat_pair(evalues.OLS(0.5, 0.1, sd = 2.0, delta = 1.0)),
  rd            = function() {
    e <- evalues.RD(200, 150, 100, 250)
    c(point = unname(e$est.Evalue), ci = unname(e$lower.Evalue))
  }
)

rows <- list()
for (label in names(cases)) {
  pr <- suppressMessages(cases[[label]]())
  rows[[length(rows) + 1L]] <- parity_row(
    module = MODULE, statistic = paste0("evalue_est_", label),
    estimate = pr["point"], n = 1
  )
  rows[[length(rows) + 1L]] <- parity_row(
    module = MODULE, statistic = paste0("evalue_ci_", label),
    estimate = pr["ci"], n = 1
  )
}

write_results(MODULE, rows, extra = list(reference = "EValue package"))
cat(sprintf("[%s] wrote %d rows across %d measures\n",
            MODULE, length(rows), length(cases)))
