#!/usr/bin/env Rscript
# =====================================================================
#  Reference values for sp.rdrobust at polynomial orders 0, 3 and 4.
#
#  Writes rdrobust_porder_R.json. Same data as _generate_rdrobust_R.R
#  (rdrobust_RDsenate, NA-dropped; that script writes rdsenate.csv).
#
#  Grid: bwselect (6) x p (0,3,4) x kernel (3) = 54 specifications.
#  The grid in _generate_rdrobust_R.R covers p = 1, 2; this one covers
#  the orders sp.rdrobust accepts beyond them, where the pilot
#  regressions run to degree p + 3.
#
#  Environment: R 4.5.2 / rdrobust 4.0.0
# =====================================================================
suppressPackageStartupMessages({library(rdrobust); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

data(rdrobust_RDsenate)
d <- rdrobust_RDsenate
d <- d[!is.na(d$margin) & !is.na(d$vote), ]

out <- list()
for (bw in c("mserd", "msetwo", "msesum", "cerrd", "certwo", "cersum"))
  for (p in c(0, 3, 4))
    for (k in c("triangular", "uniform", "epanechnikov")) {
      key <- paste0(bw, "_p", p, "_", k)
      r <- try(rdrobust(y = d$vote, x = d$margin, c = 0, p = p,
                        kernel = k, bwselect = bw), silent = TRUE)
      if (inherits(r, "try-error")) { cat("FAILED", key, "\n"); next }
      out[[key]] <- list(
        bwselect = bw, p = p, kernel = k,
        coef_conventional = r$coef[1],
        coef_biascorrected = r$coef[2],
        coef_robust       = r$coef[3],
        se_conventional   = r$se[1],
        se_biascorrected  = r$se[2],
        se_robust         = r$se[3],
        ci_robust_lower   = r$ci[3, 1],
        ci_robust_upper   = r$ci[3, 2],
        pv_robust         = r$pv[3],
        h_left = r$bws[1, 1], h_right = r$bws[1, 2],
        b_left = r$bws[2, 1], b_right = r$bws[2, 2],
        N_h_left = r$N_h[1], N_h_right = r$N_h[2]
      )
    }

out[["_meta"]] <- list(
  generated_by = "_generate_rdrobust_porder_R.R",
  r_version = R.version.string,
  rdrobust_version = as.character(packageVersion("rdrobust")),
  dataset = "rdrobust::rdrobust_RDsenate (NA-dropped)",
  n = nrow(d),
  n_specs = length(out) - 1
)
write(toJSON(out, digits = 14, auto_unbox = TRUE), file.path(OUT, "rdrobust_porder_R.json"))
cat("[rdrobust-fixture] wrote", length(out) - 1, "specs, n =", nrow(d), "\n")
