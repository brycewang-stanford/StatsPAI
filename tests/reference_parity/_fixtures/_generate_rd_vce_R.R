# Reference values for sp.rdrobust(vce=..., cluster=...) vs R rdrobust 4.0.0.
#
# Uses the SAME discriminating design as rd_covs_discriminating: covariates
# that genuinely shift the bandwidth, and a cluster structure with real
# intra-cluster correlation. A design without both lets a wrong
# implementation pass -- which is exactly how the covs bandwidth defect
# survived against the senate fixture for two months.
#
# Run: Rscript _generate_rd_vce_R.R
library(rdrobust)
library(jsonlite)

d <- read.csv("rd_covs_discriminating.csv")
stopifnot(all(c("y", "x", "z1", "z2", "g") %in% names(d)))

grab <- function(r) {
  list(
    h_left  = r$bws[1, 1], h_right = r$bws[1, 2],
    b_left  = r$bws[2, 1], b_right = r$bws[2, 2],
    coef_conventional = r$coef[1, 1],
    coef_robust       = r$coef[3, 1],
    se_conventional   = r$se[1, 1],
    se_robust         = r$se[3, 1],
    vce_type = r$vce
  )
}

out <- list()

# ---- vce variants, no clusters -------------------------------------------
for (v in c("nn", "hc0", "hc1", "hc2", "hc3")) {
  for (p in c(1, 2)) {
    out[[sprintf("%s_p%d", v, p)]] <- grab(
      rdrobust(y = d$y, x = d$x, c = 0, p = p, vce = v)
    )
  }
}

# ---- vce variants WITH covariates ----------------------------------------
for (v in c("nn", "hc0", "hc2", "hc3")) {
  out[[sprintf("%s_covs_p1", v)]] <- grab(
    rdrobust(y = d$y, x = d$x, c = 0, p = 1, vce = v, covs = d$z1)
  )
}

# ---- clusters ------------------------------------------------------------
# vce is left at its default so the fixture pins R's silent nn -> cr1
# promotion, not just the explicit cr* path.
out[["cluster_default_p1"]] <- grab(
  rdrobust(y = d$y, x = d$x, c = 0, p = 1, cluster = d$g)
)
out[["cluster_default_p2"]] <- grab(
  rdrobust(y = d$y, x = d$x, c = 0, p = 2, cluster = d$g)
)
out[["cluster_covs_p1"]] <- grab(
  rdrobust(y = d$y, x = d$x, c = 0, p = 1, cluster = d$g, covs = d$z1)
)

out[["_meta"]] <- list(
  n = nrow(d),
  n_clusters = length(unique(d$g)),
  rdrobust_version = as.character(packageVersion("rdrobust")),
  R_version = R.version.string
)

write_json(out, "rd_vce_R.json", auto_unbox = TRUE, digits = 15, pretty = TRUE)
cat("wrote rd_vce_R.json with", length(out) - 1, "cells\n")
