#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R reference for tests/reference_parity/test_evalue_rr_parity.py
#
# Requires: R + EValue. Run from any directory.
#
# Until 1.27.0 that test compared sp.evalue_rr against a closed form typed
# into the test and then *asserted* that R's EValue implements the same
# formula. It was graded as cross-language evidence without ever consulting
# R. This fixture makes the comparison real: every point / CI case below is
# EValue::evalues.RR's own output, including the branches that decide which
# limit is used (RR < 1, CI crossing the null).
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({library(EValue); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."
cases <- list(
  c(1.5, NA, NA), c(2.0, NA, NA), c(3.0, NA, NA), c(5.0, NA, NA),
  c(0.6, NA, NA), c(0.25, NA, NA),
  c(2.0, 1.3, 3.1),            # CI above the null: lower limit used
  c(0.6, 0.4, 0.9),            # CI below the null: upper limit used
  c(1.5, 0.8, 2.4),            # CI crosses the null: CI E-value is 1
  c(0.7, 0.45, 1.2)            # crosses from below
)
rows <- lapply(cases, function(z) {
  e <- if (is.na(z[2])) evalues.RR(z[1]) else evalues.RR(z[1], z[2], z[3])
  ev <- e["E-values", ]
  ci <- if (is.na(z[2])) NA_real_ else suppressWarnings(max(as.numeric(ev[c("lower", "upper")]), na.rm = TRUE))
  list(rr = z[1], lo = z[2], hi = z[3], e_point = unname(ev["point"]), e_ci = ci)
})
out <- list(cases = rows,
            provenance = list(r = R.version.string, EValue = as.character(packageVersion("EValue"))))
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE, na = "null"),
           file.path(OUT, "evalue_rr_R.json"))
cat("wrote evalue_rr_R.json\n")
