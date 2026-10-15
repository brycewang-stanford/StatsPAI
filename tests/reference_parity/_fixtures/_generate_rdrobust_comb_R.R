#!/usr/bin/env Rscript
# =====================================================================
#  Reference values for sp.rdrobust under the four "comb" bandwidth
#  selectors: msecomb1, msecomb2, cercomb1, cercomb2.
#
#  Writes rdrobust_comb_R.json. Reads the committed CSVs, so R and the
#  Python test estimate on the same bytes:
#    rdsenate.csv        (written by _generate_rdrobust_R.R)
#    rdsenate_fuzzy.csv  (written by _generate_rdrobust_fuzzy_R.R;
#                         treat2 has two-sided noncompliance)
#
#  Grid:
#    sharp: 4 selectors x p (1, 2) x kernel (3)       = 24 specifications
#    fuzzy: 4 selectors x p = 1, triangular, treat2   =  4 specifications
#
#  The sharp grid in _generate_rdrobust_R.R covers mserd / msetwo /
#  msesum / cerrd / certwo / cersum; tests/reference_parity/
#  test_rdbwselect_comb_rules.py pins comb1 = min(rd, sum) and comb2 =
#  median(two, rd, sum) as identities, and Track A module 88 pins the
#  comb bandwidths of sp.rdbwselect. Neither compares the estimates and
#  standard errors sp.rdrobust reports at a comb bandwidth; this does.
#
#  Environment: R 4.5.2 / rdrobust 4.0.0
# =====================================================================
suppressPackageStartupMessages({library(rdrobust); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

sharp <- read.csv(file.path(OUT, "rdsenate.csv"))
sharp <- sharp[!is.na(sharp$margin) & !is.na(sharp$vote), ]
fuzzy <- read.csv(file.path(OUT, "rdsenate_fuzzy.csv"))

grab <- function(r, bw, p, k, design) list(
  design = design, bwselect = bw, p = p, kernel = k,
  coef_conventional = r$coef[1],
  coef_robust       = r$coef[3],
  se_conventional   = r$se[1],
  se_robust         = r$se[3],
  h_left = r$bws[1, 1], h_right = r$bws[1, 2],
  b_left = r$bws[2, 1], b_right = r$bws[2, 2],
  N_h_left = r$N_h[1], N_h_right = r$N_h[2]
)

SELECTORS <- c("msecomb1", "msecomb2", "cercomb1", "cercomb2")
out <- list()
for (bw in SELECTORS) {
  for (p in c(1, 2))
    for (k in c("triangular", "uniform", "epanechnikov")) {
      key <- paste0("sharp_", bw, "_p", p, "_", k)
      r <- rdrobust(y = sharp$vote, x = sharp$margin, c = 0, p = p,
                    kernel = k, bwselect = bw)
      out[[key]] <- grab(r, bw, p, k, "sharp")
    }
  key <- paste0("fuzzy_", bw, "_p1_triangular")
  r <- rdrobust(y = fuzzy$vote, x = fuzzy$margin, c = 0, p = 1,
                fuzzy = fuzzy$treat2, kernel = "triangular", bwselect = bw)
  out[[key]] <- grab(r, bw, 1, "triangular", "fuzzy")
}

out[["_meta"]] <- list(
  generated_by = "_generate_rdrobust_comb_R.R",
  r_version = R.version.string,
  rdrobust_version = as.character(packageVersion("rdrobust")),
  data = c("rdsenate.csv", "rdsenate_fuzzy.csv"),
  n_sharp = nrow(sharp),
  n_fuzzy = nrow(fuzzy),
  n_specs = length(out)
)
write(toJSON(out, digits = NA, auto_unbox = TRUE),
      file.path(OUT, "rdrobust_comb_R.json"))
cat("[rdrobust-comb] wrote", length(out) - 1, "specs\n")
