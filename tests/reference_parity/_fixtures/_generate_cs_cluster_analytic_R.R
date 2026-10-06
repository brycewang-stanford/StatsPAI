#!/usr/bin/env Rscript
# R did reference for analytic cluster-robust standard errors of
# Callaway-Sant'Anna on UNEQUAL cluster sizes.
#
# did <= 2.3.0 had no analytic clustered standard error, and its clustered
# bootstrap aggregated the influence function to cluster means, which is
# the cluster-robust variance only when every cluster has the same size
# (see the note in ../_generate_cs_gaps_R.R). did 2.5.0 added analytic
# cluster-robust standard errors from the cluster sums of the influence
# function, at the group-time level and for every aggregation. This script
# records them.
#
# Panels: _fixtures/cs_gaps_panel.csv (balanced) and
# _fixtures/cs_gaps_unbalanced_panel.csv, both written by
# ../_generate_cs_gaps_R.R. `state` has nine clusters of sizes
# 150, 90, 45, 30, 20, 15, 6, 3, 1.
#
#   Rscript tests/reference_parity/_fixtures/_generate_cs_cluster_analytic_R.R
#
# Run from the repository root. Needs did >= 2.5.0.
suppressMessages({library(did); library(jsonlite)})
stopifnot(utils::packageVersion("did") >= "2.5.0")
here <- "tests/reference_parity/_fixtures"
cases <- list()
for (panel in c("cs_gaps_panel", "cs_gaps_unbalanced_panel")) {
  d <- read.csv(file.path(here, paste0(panel, ".csv")))
  d$g <- as.numeric(d$g)
  for (cg in c("nevertreated", "notyettreated")) for (base in c("varying", "universal")) {
    a <- suppressMessages(suppressWarnings(att_gt(
      yname = "y", tname = "t", idname = "i", gname = "g", data = d,
      est_method = "dr", control_group = cg, base_period = base,
      clustervars = "state", bstrap = FALSE, cband = FALSE,
      allow_unbalanced_panel = (panel != "cs_gaps_panel"))))
    agg <- list()
    for (type in c("simple", "dynamic", "group", "calendar")) {
      s <- suppressWarnings(aggte(a, type = type, bstrap = FALSE, cband = FALSE,
                                  na.rm = TRUE))
      agg[[type]] <- list(att = s$overall.att, se = s$overall.se,
                          egt = s$egt, att_egt = s$att.egt,
                          se_egt = as.numeric(s$se.egt))
    }
    keep <- !is.na(a$se)
    cases[[length(cases) + 1]] <- list(
      panel = panel, control_group = cg, base_period = base,
      group = a$group[keep], time = a$t[keep], att = a$att[keep],
      se = a$se[keep], agg = agg)
  }
}
write_json(list(meta = list(R_version = R.version.string,
                            did_version = as.character(packageVersion("did")),
                            DRDID_version = as.character(packageVersion("DRDID"))),
                cases = cases),
           file.path(here, "cs_cluster_analytic_R.json"),
           digits = NA, auto_unbox = TRUE, na = "null")
cat("wrote cs_cluster_analytic_R.json\n")
