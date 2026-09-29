#!/usr/bin/env Rscript
# ri2 0.4 randomization inference with blocks (and clusters), enumerated:
# difference in means and the regression coefficient y ~ Z + x + block.
#   python tests/reference_parity/_fixtures/_generate_ri_strata_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_ri_strata_R.R
suppressMessages({library(ri2); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
out <- list(meta = list(R = R.version.string, ri2 = as.character(packageVersion("ri2"))))
u <- read.csv(file.path(here, "ri_strata_units.csv")); u$block <- factor(u$block)
dec <- declare_ra(N = nrow(u), blocks = u$block, block_m = rep(3, 3))
for (spec in c("dim", "ols")) {
  f <- if (spec == "dim") y ~ Z else y ~ Z + x + block
  r <- conduct_ri(f, declaration = dec, assignment = "Z", sharp_hypothesis = 0,
                  data = u, sims = 10000)
  s <- summary(r)
  out[[paste0("units_", spec)]] <- list(obs = s$estimate, p = s$two_tailed_p_value,
                                         n = nrow(r$sims_df))
}
cdat <- read.csv(file.path(here, "ri_strata_clusters.csv")); cdat$block <- factor(cdat$block)
dec2 <- declare_ra(clusters = cdat$clust, blocks = cdat$block, block_m = rep(2, 4))
r <- conduct_ri(y ~ Z + x + block, declaration = dec2, assignment = "Z",
                sharp_hypothesis = 0, data = cdat, sims = 10000)
s <- summary(r)
out$clusters_ols <- list(obs = s$estimate, p = s$two_tailed_p_value, n = nrow(r$sims_df))
write_json(out, file.path(here, "ri_strata_R.json"), digits = NA, auto_unbox = TRUE)
cat("wrote ri_strata_R.json\n")
