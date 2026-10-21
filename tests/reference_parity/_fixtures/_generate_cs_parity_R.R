#!/usr/bin/env Rscript
# R did reference for sp.callaway_santanna(parity="r") / sp.aggte.
#
# R did is used as a black box (GPL-3): only its outputs are recorded.
# Panels (written by _generate_cs_parity_data.py):
#   cs_parity_mpdta.csv   the real mpdta, plus state = countyreal %/% 1000
#   cs_parity_always.csv  the same with 35 never-treated counties recoded as
#                         always treated (g = 2003 and g = 2001)
# Every case uses R's defaults (base_period = "varying", est_method = "dr")
# except bstrap = FALSE / cband = FALSE, so the standard errors are the
# analytic ones and can be compared exactly. Recorded per case: every
# ATT(g, t) and SE, the pre-test Wald statistic and p-value, the number of
# units R keeps, and the four aggregations (overall and per-cell estimate
# and SE).
#
#   Rscript tests/reference_parity/_fixtures/_generate_cs_parity_R.R
#
# Run from the repository root. Needs did >= 2.5.0 (analytic clustered SEs).
suppressMessages({library(did); library(jsonlite)})
stopifnot(utils::packageVersion("did") >= "2.5.0")
here <- "tests/reference_parity/_fixtures"
cases <- list()
grid <- expand.grid(panel = c("cs_parity_mpdta", "cs_parity_always"),
                    covariates = c(FALSE, TRUE),
                    control_group = c("nevertreated", "notyettreated"),
                    cluster = c(FALSE, TRUE), anticipation = 0,
                    stringsAsFactors = FALSE)
grid <- rbind(grid, data.frame(panel = "cs_parity_always", covariates = FALSE,
                               control_group = "nevertreated", cluster = FALSE,
                               anticipation = 1))
for (k in seq_len(nrow(grid))) {
  p <- grid[k, ]
  d <- read.csv(file.path(here, paste0(p$panel, ".csv")))
  msgs <- character(0)
  a <- withCallingHandlers(
    suppressMessages(att_gt(
      yname = "lemp", tname = "year", idname = "countyreal",
      gname = "first_treat", data = d,
      xformla = if (p$covariates) ~lpop else ~1,
      control_group = p$control_group, anticipation = p$anticipation,
      clustervars = if (p$cluster) "state" else NULL,
      bstrap = FALSE, cband = FALSE)),
    warning = function(w) {msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning")})
  agg <- list()
  for (type in c("simple", "dynamic", "group", "calendar")) {
    s <- suppressWarnings(aggte(a, type = type, bstrap = FALSE, cband = FALSE,
                                na.rm = TRUE))
    agg[[type]] <- list(att = s$overall.att, se = s$overall.se,
                        egt = if (type == "simple") NULL else s$egt,
                        att_egt = if (type == "simple") NULL else s$att.egt,
                        se_egt = if (type == "simple") NULL else as.numeric(s$se.egt))
  }
  keep <- !is.na(a$se)
  cases[[length(cases) + 1]] <- list(
    panel = p$panel, covariates = p$covariates,
    control_group = p$control_group, cluster = p$cluster,
    anticipation = p$anticipation, n = a$n, warnings = msgs,
    group = a$group[keep], time = a$t[keep], att = a$att[keep],
    se = a$se[keep], W = a$W, Wpval = a$Wpval, agg = agg)
}
write_json(list(meta = list(R_version = R.version.string,
                            did_version = as.character(packageVersion("did")),
                            DRDID_version = as.character(packageVersion("DRDID"))),
                cases = cases),
           file.path(here, "cs_parity_R.json"),
           digits = NA, auto_unbox = TRUE, na = "null")
cat("wrote cs_parity_R.json\n")
