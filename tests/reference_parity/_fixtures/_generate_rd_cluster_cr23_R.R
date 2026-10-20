# Reference values for sp.rdrobust(vce='cr2' / 'cr3', cluster=...) and the
# R rdrobust 4.0.0 mapping of vce='hc2' / 'hc3' with clusters to CR2 / CR3.
#
# R rdrobust is GPL-3 and used as a black box: only its documented
# arguments and its outputs on this fixture.
#
# Data: rd_covs_discriminating.csv (real intra-cluster correlation in g).
# The fuzzy treatment and the observation weights are deterministic
# functions of z2, so Python rebuilds them bit for bit:
#   t  = 1{x >= 0, z2 > -0.8} + 1{x < 0, z2 > 0.8}
#   wt = 1 + 0.5 * tanh(z2)
#
# Run: Rscript _generate_rd_cluster_cr23_R.R
library(rdrobust)
library(jsonlite)

d <- read.csv("rd_covs_discriminating.csv")
d$t <- as.numeric((d$x >= 0 & d$z2 > -0.8) | (d$x < 0 & d$z2 > 0.8))
d$wt <- 1 + 0.5 * tanh(d$z2)

grab <- function(r) {
  list(
    h_left = r$bws[1, 1], h_right = r$bws[1, 2],
    b_left = r$bws[2, 1], b_right = r$bws[2, 2],
    coef_conventional = r$coef[1, 1], coef_robust = r$coef[3, 1],
    se_conventional = r$se[1, 1], se_robust = r$se[3, 1],
    vce_type = r$vce
  )
}

run <- function(v, ...) grab(suppressWarnings(
  rdrobust(y = d$y, x = d$x, c = 0, vce = v, cluster = d$g, ...)
))

out <- list()
for (v in c("cr1", "cr2", "cr3")) {
  out[[paste0(v, "_fixed_hb")]] <- run(v, p = 1, h = 0.5, b = 0.5)
  out[[paste0(v, "_fixed_hneb")]] <- run(v, p = 1, h = 0.4, b = 0.7)
  out[[paste0(v, "_h_gt_b")]] <- run(v, p = 1, h = 0.7, b = 0.4)
  out[[paste0(v, "_covs")]] <- run(v, p = 1, h = 0.5, b = 0.6, covs = d$z1)
  out[[paste0(v, "_fuzzy")]] <- run(v, p = 1, h = 0.5, b = 0.6, fuzzy = d$t)
  out[[paste0(v, "_weights")]] <- run(v, p = 1, h = 0.5, b = 0.6, weights = d$wt)
  out[[paste0(v, "_auto_p1")]] <- run(v, p = 1)
  out[[paste0(v, "_auto_p2")]] <- run(v, p = 2)
  out[[paste0(v, "_auto_covs")]] <- run(v, p = 1, covs = d$z1)
  out[[paste0(v, "_auto_uniform")]] <- run(v, p = 1, kernel = "uniform")
}
# R maps the HC kinds onto the cluster kinds when cluster is given.
for (v in c("hc2", "hc3")) {
  out[[paste0(v, "_auto_p1")]] <- run(v, p = 1)
}

out[["_meta"]] <- list(
  n = nrow(d),
  n_clusters = length(unique(d$g)),
  rdrobust_version = as.character(packageVersion("rdrobust")),
  R_version = R.version.string
)

write_json(out, "rd_cluster_cr23_R.json", auto_unbox = TRUE, digits = 15,
           pretty = TRUE)
cat("wrote rd_cluster_cr23_R.json with", length(out) - 1, "cells\n")
