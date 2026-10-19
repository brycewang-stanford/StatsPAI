#!/usr/bin/env Rscript
# pcalg::ges essential graphs for tests/reference_parity/test_ges_pcalg_parity.py.
# Usage: Rscript _generate_ges_pcalg_R.R <dir with g00.csv ... g29.csv>
# pcalg is GPL: StatsPAI compares its output only, never its source.
# Environment: R 4.5 / pcalg 2.7.12
suppressPackageStartupMessages({library(pcalg); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."
src <- commandArgs(trailingOnly = TRUE)[1]
out <- list()
for (f in sort(list.files(src, pattern = "csv$", full.names = TRUE))) {
  d <- as.matrix(read.csv(f))
  fit <- ges(new("GaussL0penObsScore", d), phase = c("forward", "backward"),
             iterate = FALSE)
  out[[basename(f)]] <- list(amat = as(fit$essgraph, "matrix") * 1, n = nrow(d))
}
writeLines(toJSON(out, digits = NA), file.path(OUT, "ges_pcalg_R.json"))
