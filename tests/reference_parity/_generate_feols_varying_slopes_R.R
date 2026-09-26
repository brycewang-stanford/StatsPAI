#!/usr/bin/env Rscript
# Frozen reference for sp.feols varying-slope fixed effects vs R fixest.
# Regenerate from the repository root:
#   Rscript tests/reference_parity/_generate_feols_varying_slopes_R.R
suppressPackageStartupMessages(library(fixest)); library(jsonlite)
d <- read.csv("tests/reference_parity/_fixtures/feols_varying_slopes_data.csv")
out <- list()
fs <- c("y ~ x | g[z]", "y ~ x | g[[z]]", "y ~ x | h + g[z]", "y ~ x | h + g[[z]]", "y ~ x | g[z, z2]", "y ~ x + z2 | h + g[z]")
for (f in fs) {
  for (v in c("iid", "hetero", "cl")) {
    m <- if (v == "cl") feols(as.formula(f), d, cluster = ~cl) else feols(as.formula(f), d, vcov = v)
    out[[paste(f, v)]] <- list(coef = unname(coef(m)), se = unname(se(m)), names = names(coef(m)), p = unname(pvalue(m)))
  }
  m <- feols(as.formula(f), d, weights = ~w, cluster = ~cl)
  out[[paste(f, "w_cl")]] <- list(coef = unname(coef(m)), se = unname(se(m)), names = names(coef(m)), p = unname(pvalue(m)))
}
out$provenance <- list(r_version = R.version.string, fixest_version = as.character(packageVersion("fixest")))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), "tests/reference_parity/_fixtures/feols_varying_slopes_R.json")
