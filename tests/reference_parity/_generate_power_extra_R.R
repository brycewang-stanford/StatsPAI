#!/usr/bin/env Rscript
# Frozen reference for sp.mde / sp.power_cluster_rct vs base-R closed form.
#   * mde('rct')        : minimum detectable effect = (z_{1-a/2}+z_pow)/sqrt(n_g/2)
#   * power_cluster_rct : design-effect-inflated z-approx (both tails since 1.42),
#                         N_eff = N / (1 + (m-1)*icc), power = Phi(s - z) + Phi(-s - z), s = d*sqrt(N_eff/4)
# Regenerate: Rscript tests/reference_parity/_generate_power_extra_R.R
suppressPackageStartupMessages(library(jsonlite))
za <- qnorm(0.975); zp <- qnorm(0.8)
mde <- function(n) (za + zp) / sqrt((n / 2) / 2)
clus <- function(nc, m, d, icc) {
  N <- nc * m; Neff <- N / (1 + (m - 1) * icc)
  shift <- d * sqrt(Neff / 4); pnorm(shift - za) + pnorm(-shift - za)
}
out <- list(
  mde = list(
    list(n = 128, effect_size = mde(128)),
    list(n = 200, effect_size = mde(200))
  ),
  cluster = list(
    list(nc = 40, m = 20, d = 0.3, icc = 0.05, power = clus(40, 20, 0.3, 0.05)),
    list(nc = 60, m = 15, d = 0.25, icc = 0.02, power = clus(60, 15, 0.25, 0.02))
  ),
  provenance = list(r_version = R.version.string,
                    generated_by = "tests/reference_parity/_generate_power_extra_R.R")
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 16, pretty = TRUE),
           "tests/reference_parity/_fixtures/power_extra_R.json")
cat("wrote power_extra_R.json\n")
