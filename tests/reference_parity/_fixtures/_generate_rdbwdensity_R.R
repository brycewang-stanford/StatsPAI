#!/usr/bin/env Rscript
# =====================================================================
#  Default bandwidths of sp.rddensity vs R rddensity 2.6 / rdbwdensity
#  in the regimes where the local sample-size rules bind.
#
#  rddensity is GPL-2 and was used as a black box (no source read). The
#  five designs make the selector's corner cases visible:
#    gap      - a hole around the cutoff: pilots and final bandwidths are
#               widened to hold 20 + p + 1 observations per side;
#    tinyleft - 15 observations below the cutoff: the common bandwidths
#               take the farthest one;
#    heap     - two mass points at and just below the cutoff;
#    integer  - 9 support points: the bias constant cannot be estimated
#               (rdbwdensity reports NA) and the bandwidth falls back to
#               the sample-size bound;
#    shifted  - a non-zero cutoff.
#  Records rdbwdensity()$h (bw, variance, biassq for l, r, diff, sum) and
#  the rddensity() test at the default bandwidths.
#
#  Writes rdbwdensity_designs.csv and rdbwdensity_R.json.
#  Environment: R 4.5 / rddensity 2.6
# =====================================================================
suppressPackageStartupMessages({library(rddensity); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

set.seed(20261021)
g <- rnorm(1200); g <- c(g[abs(g) > 0.6][1:500], runif(8, -0.6, 0), runif(8, 0, 0.6))
designs <- list(
  gap = list(x = g, c = 0),
  tinyleft = list(x = c(runif(15, -1, 0), runif(300, 0, 1)), c = 0),
  heap = list(x = c(rnorm(400), rep(0, 30), rep(-0.1, 20)), c = 0),
  integer = list(x = sample(-4:4, 300, replace = TRUE), c = 0),
  shifted = list(x = rnorm(500, 1, 2), c = 0.8)
)
df <- do.call(rbind, lapply(names(designs), function(k)
  data.frame(design = k, x = designs[[k]]$x)))
write.csv(df, file.path(OUT, "rdbwdensity_designs.csv"), row.names = FALSE)
df <- read.csv(file.path(OUT, "rdbwdensity_designs.csv"))  # the committed bytes

out <- list()
for (k in names(designs)) {
  x <- df$x[df$design == k]; cc <- designs[[k]]$c
  for (p in 1:3) {
    b <- tryCatch(rdbwdensity(X = x, c = cc, p = p)$h, error = function(e) conditionMessage(e))
    if (is.character(b)) {  # recorded, not compared: R stops on this cell
      out[[sprintf("%s_p%d", k, p)]] <- list(c = cc, p = p, r_error = b)
      next
    }
    r <- rddensity(X = x, c = cc, p = p)
    out[[sprintf("%s_p%d", k, p)]] <- list(
      c = cc, p = p,
      bw = unname(b[, 1]), variance = unname(b[, 2]), biassq = unname(b[, 3]),
      h_left = r$h$left, h_right = r$h$right,
      f_left = r$hat$left, f_right = r$hat$right,
      T_jk = unname(r$test$t_jk), N_eff_left = r$N$eff_left, N_eff_right = r$N$eff_right)
  }
}
out[["_meta"]] <- list(
  generated_by = "_generate_rdbwdensity_R.R", r_version = R.version.string,
  rddensity_version = as.character(packageVersion("rddensity")),
  targets = c("l", "r", "diff", "sum"))
write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null"),
      file.path(OUT, "rdbwdensity_R.json"))
cat("[rdbwdensity] wrote", length(out) - 1, "cells\n")
