# R did references for tests/reference_parity/test_cs_balance_full_did_parity.py
# An unbalanced panel under R did's default (allow_unbalanced_panel = FALSE):
# units not observed in every period are dropped before estimation.
#
#   Rscript _generate_cs_balance_full_R.R      (from this directory)
suppressPackageStartupMessages({
  library(did)
  library(jsonlite)
})
d <- read.csv("../../r_parity/data/04_csdid.csv")
# Deterministic holes, rebuilt the same way in the test.
d <- d[(d$countyreal * 7 + d$year) %% 11 != 0, ]
d$first_treat <- as.numeric(d$first_treat)
out <- list(did_version = as.character(packageVersion("did")), n_rows = nrow(d))
for (cg in c("nevertreated", "notyettreated")) {
  fit <- suppressWarnings(att_gt(
    yname = "lemp", tname = "year", idname = "countyreal", gname = "first_treat",
    data = d, control_group = cg, base_period = "universal", est_method = "dr",
    allow_unbalanced_panel = FALSE, bstrap = FALSE, cband = FALSE
  ))
  res <- list(n_units = fit$n, cell_group = fit$group, cell_time = fit$t, cell_att = fit$att, cell_se = fit$se)
  for (ty in c("simple", "dynamic", "group", "calendar")) {
    a <- aggte(fit, type = ty, bstrap = FALSE, cband = FALSE)
    res[[ty]] <- list(overall_att = a$overall.att, overall_se = a$overall.se)
  }
  out[[cg]] <- res
}
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, pretty = TRUE, na = "null"),
           "cs_balance_full_R.json")
