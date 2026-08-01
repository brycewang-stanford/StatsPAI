#!/usr/bin/env Rscript
# =====================================================================
#  Reference values for sp.rdrobust / sp.rdbwselect vs R rdrobust 3.0.0.
#
#  Writes rdrobust_R.json and rdsenate.csv (the package's own
#  rdrobust_RDsenate, NA-dropped, so Python reads identical bytes).
#
#  Grid: bwselect (6) x p (1,2) x kernel (3) = 36 specifications.
#
#  WHY THE GRID VARIES p AND bwselect
#  ----------------------------------
#  The MSE-optimal bandwidth depends on the polynomial order through the
#  rate exponent 1/(2p+3) and on the bwselect variant through which
#  side(s) are pooled.  A selector that ignores either will produce the
#  SAME h across cells that R separates -- which is exactly the defect
#  this fixture was built to pin (see docs/rfc/rd_three_month_plan.md
#  section 0.2 A).  Any grid holding p fixed would miss it.
#
#  Environment: R 4.5.2 / rdrobust 3.0.0
# =====================================================================
suppressPackageStartupMessages({library(rdrobust); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

data(rdrobust_RDsenate)
d <- rdrobust_RDsenate
d <- d[!is.na(d$margin) & !is.na(d$vote), ]
write.csv(d, file.path(OUT, "rdsenate.csv"), row.names = FALSE)

out <- list()
for (bw in c("mserd", "msetwo", "msesum", "cerrd", "certwo", "cersum"))
  for (p in c(1, 2))
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
  generated_by = "_generate_rdrobust_R.R",
  r_version = R.version.string,
  rdrobust_version = as.character(packageVersion("rdrobust")),
  dataset = "rdrobust::rdrobust_RDsenate (NA-dropped)",
  n = nrow(d),
  n_specs = length(out) - 1
)
write(toJSON(out, digits = 14, auto_unbox = TRUE), file.path(OUT, "rdrobust_R.json"))
cat("[rdrobust-fixture] wrote", length(out) - 1, "specs, n =", nrow(d), "\n")
