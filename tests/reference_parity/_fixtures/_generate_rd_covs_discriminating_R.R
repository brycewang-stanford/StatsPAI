#!/usr/bin/env Rscript
# =====================================================================
#  DISCRIMINATING covs / cluster reference for the CCT bandwidth cascade.
#
#  Why a third RD fixture exists
#  -----------------------------
#  rdrobust_RDsenate's covariates barely bind. Twice that let a wrong
#  implementation look right:
#
#    * a Z-projection in the bandwidth cascade "improved" senate from
#      7.3e-3 to 2.0e-3 while making h 3x too narrow on a DGP where
#      covariates matter;
#    * a regression that discarded covs entirely showed up on senate as a
#      ~1e-2 gap that read like "the bandwidth doesn't handle covs yet",
#      when the SE was actually off by 6.4x.
#
#  So this fixture is built so that covariates and clusters CANNOT be
#  ignored without the numbers moving a lot:
#
#    z1 carries a coefficient of 2.0 against residual noise of 0.3
#    -> adjusting must cut the SE several-fold
#    cluster effects are large relative to the idiosyncratic term
#    -> clustering must widen the SE substantially
#
#  Writes rd_covs_discriminating_R.json and rd_covs_discriminating.csv.
#
#  Environment: R 4.5.2 / rdrobust 4.0.0
# =====================================================================
suppressPackageStartupMessages({library(rdrobust); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

set.seed(20260802)
n <- 4000
x  <- runif(n, -1, 1)
z1 <- rnorm(n)                       # binds hard
z2 <- rnorm(n)                       # irrelevant, guards over-fitting
# Clusters must be CONTIGUOUS IN x, or units inside the bandwidth rarely
# share one and the cluster-robust SE barely moves (measured: 1.10x with
# clusters assigned by row index -- too weak to discriminate a wrong
# implementation from a missing one).
g  <- as.integer(cut(x, breaks = 80, labels = FALSE))
ug <- rnorm(80, sd = 1.2)            # large cluster effects
y  <- 0.5 * x + 3.0 * (x >= 0) + 2.0 * z1 + ug[g] + rnorm(n, sd = 0.3)
d  <- data.frame(y = y, x = x, z1 = z1, z2 = z2, g = g)
write.csv(d, file.path(OUT, "rd_covs_discriminating.csv"), row.names = FALSE)

grab <- function(r) list(
  coef_conventional = r$coef[1], coef_robust = r$coef[3],
  se_conventional = r$se[1], se_robust = r$se[3],
  h_left = r$bws[1, 1], h_right = r$bws[1, 2],
  b_left = r$bws[2, 1], b_right = r$bws[2, 2]
)
out <- list()
add <- function(k, e) {
  r <- try(e, silent = TRUE)
  if (inherits(r, "try-error")) { cat("FAILED", k, "\n"); return(invisible()) }
  out[[k]] <<- grab(r); cat("ok", k, "\n")
}
for (p in c(1, 2)) {
  add(paste0("plain_p", p),   rdrobust(y = d$y, x = d$x, c = 0, p = p))
  add(paste0("covs1_p", p),   rdrobust(y = d$y, x = d$x, c = 0, p = p, covs = d$z1))
  add(paste0("covs2_p", p),   rdrobust(y = d$y, x = d$x, c = 0, p = p,
                                       covs = cbind(d$z1, d$z2)))
  add(paste0("cluster_p", p), rdrobust(y = d$y, x = d$x, c = 0, p = p, cluster = d$g))
  add(paste0("covs_cluster_p", p),
      rdrobust(y = d$y, x = d$x, c = 0, p = p, covs = d$z1, cluster = d$g))
}
out[["_meta"]] <- list(
  generated_by = "_generate_rd_covs_discriminating_R.R",
  rdrobust_version = as.character(packageVersion("rdrobust")),
  r_version = R.version.string, n = n, n_clusters = 80,
  note = paste("z1 coefficient 2.0 vs noise sd 0.3; cluster sd 1.2.",
               "Ignoring covs or clusters moves the SE several-fold.")
)
write(toJSON(out, digits = 14, auto_unbox = TRUE),
      file.path(OUT, "rd_covs_discriminating_R.json"))
cat("[discriminating-fixture] wrote", length(out) - 1, "specs\n")
