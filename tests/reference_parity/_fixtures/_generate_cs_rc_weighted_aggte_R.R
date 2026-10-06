# R did references for tests/reference_parity/test_cs_rc_weighted_aggte_did_parity.py
# Repeated cross-sections (panel = FALSE) with observation weights: the
# aggregated SEs carry the influence of the estimated cohort shares
# (did:::wif), which under weights is built from w_i * 1{G_i = g}.
#
#   Rscript _generate_cs_rc_weighted_aggte_R.R      (from this directory)
suppressPackageStartupMessages({
  library(did)
  library(jsonlite)
})
d <- read.csv("csdid2_rc_cluster.csv")
# Deterministic weights, rebuilt the same way in the test.
d$w <- 0.5 + (d$id %% 7) / 7
out <- list(did_version = as.character(packageVersion("did")))
for (cl in c("none", "cy")) {
  fit <- att_gt(
    yname = "y", tname = "year", gname = "g", data = d, panel = FALSE,
    weightsname = "w", control_group = "nevertreated",
    base_period = "universal", est_method = "dr",
    clustervars = if (cl == "none") NULL else cl,
    bstrap = FALSE, cband = FALSE
  )
  res <- list()
  for (ty in c("simple", "dynamic", "group", "calendar")) {
    a <- aggte(fit, type = ty, bstrap = FALSE, cband = FALSE)
    res[[ty]] <- list(
      overall_att = a$overall.att, overall_se = a$overall.se,
      egt = if (ty == "simple") NULL else a$egt,
      att = if (ty == "simple") NULL else a$att.egt,
      se = if (ty == "simple") NULL else a$se.egt
    )
  }
  out[[cl]] <- res
}
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE, pretty = TRUE, na = "null"),
           "cs_rc_weighted_aggte_R.json")
