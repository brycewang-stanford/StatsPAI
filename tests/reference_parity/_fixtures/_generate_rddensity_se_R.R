#!/usr/bin/env Rscript
# =====================================================================
#  Standard errors of sp.rddensity vs R rddensity 2.6.
#
#  rddensity_R.json (from _generate_rddensity_R.R) pins the statistic,
#  the p-value, the bandwidths and the one-sided densities, but not the
#  standard errors the statistic is built from. sp.rddensity reports the
#  jackknife SE of the density difference as `se` and the two one-sided
#  jackknife SEs in model_info; this fixture holds R's `sd_jk` for the
#  same six cells (three designs x p in {2, 3}), and the order-p fit
#  (`hat_p`, `sd_jk_p`, `test_p`) that sp.rddensity reports as
#  model_info['conventional'].
#
#  Reads the committed CSVs (rdsenate.csv, rddensity_null.csv,
#  rddensity_manip.csv), so both sides estimate on the same bytes.
#  Writes rddensity_se_R.json.
#
#  Environment: R 4.5.2 / rddensity 2.6
# =====================================================================
suppressPackageStartupMessages({library(rddensity); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

X <- list(
  senate = read.csv(file.path(OUT, "rdsenate.csv"))$margin,
  null   = read.csv(file.path(OUT, "rddensity_null.csv"))$x,
  manip  = read.csv(file.path(OUT, "rddensity_manip.csv"))$x
)

grab <- function(r) list(
  T_jk = unname(r$test$t_jk),
  se_left = unname(r$sd_jk$left), se_right = unname(r$sd_jk$right),
  se_diff = unname(r$sd_jk$diff),
  conv_density_left = unname(r$hat_p$left),
  conv_density_right = unname(r$hat_p$right),
  conv_se_left = unname(r$sd_jk_p$left), conv_se_right = unname(r$sd_jk_p$right),
  conv_se = unname(r$sd_jk_p$diff),
  conv_T = unname(r$test_p$t_jk), conv_p = unname(r$test_p$p_jk)
)

out <- list()
for (p in c(2, 3))
  for (design in names(X)) {
    x <- X[[design]]
    x <- x[!is.na(x)]
    out[[sprintf("%s_p%d", design, p)]] <- grab(rddensity(X = x, c = 0, p = p, all = TRUE))
  }

out[["_meta"]] <- list(
  generated_by = "_generate_rddensity_se_R.R",
  r_version = R.version.string,
  rddensity_version = as.character(packageVersion("rddensity"))
)
write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
      file.path(OUT, "rddensity_se_R.json"))
cat("[rddensity-se] wrote", length(out) - 1, "cells\n")
