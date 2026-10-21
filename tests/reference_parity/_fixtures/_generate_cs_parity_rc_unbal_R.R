#!/usr/bin/env Rscript
# R did reference for sp.callaway_santanna(parity="r") / sp.aggte on a
# repeated cross-section and on an unbalanced panel.
#
# R did is used as a black box (GPL-3): only its outputs are recorded.
# Data (written by _generate_cs_parity_rc_unbal_data.py):
#   cs_parity_rcs.csv    mpdta county-years sampled per year, one id per row
#                        (rid); att_gt(panel = FALSE)
#   cs_parity_unbal.csv  mpdta with 12% of county-years deleted;
#                        att_gt(panel = TRUE) with R's default (incomplete
#                        units dropped) and with allow_unbalanced_panel = TRUE
# Each case runs unweighted and with the sampling weight w (weightsname =
# "w"; a row's weight on the repeated cross-section, a county's on the
# panel). A third design, "bal", is the balanced cs_parity_mpdta.csv with
# each county's w taken from cs_parity_unbal.csv (weighted cases only; the
# unweighted balanced panel is in test_cs_parity_switch.py). Every case uses
# R's defaults (base_period = "varying", est_method = "dr")
# except bstrap = FALSE / cband = FALSE, so the standard errors are the
# analytic ones and can be compared exactly. Recorded per case: every
# ATT(g, t) and SE, the pre-test Wald statistic and p-value, n, and the four
# aggregations (overall and per-cell estimate and SE).
#
#   Rscript tests/reference_parity/_fixtures/_generate_cs_parity_rc_unbal_R.R
#
# Run from the repository root.
suppressMessages({library(did); library(jsonlite)})
stopifnot(utils::packageVersion("did") >= "2.5.0")
here <- "tests/reference_parity/_fixtures"
cases <- list()
grid <- expand.grid(design = c("rcs", "unbal_full", "unbal_allow", "bal"),
                    covariates = c(FALSE, TRUE),
                    control_group = c("nevertreated", "notyettreated"),
                    cluster = c(FALSE, TRUE),
                    weighted = c(FALSE, TRUE),
                    stringsAsFactors = FALSE)
grid <- grid[!(grid$design == "bal" & !grid$weighted), ]
unbal <- read.csv(file.path(here, "cs_parity_unbal.csv"))
county_w <- unique(unbal[, c("countyreal", "w")])
stopifnot(!anyDuplicated(county_w$countyreal))
for (k in seq_len(nrow(grid))) {
  p <- grid[k, ]
  rcs <- p$design == "rcs"
  if (p$design == "bal") {
    d <- merge(read.csv(file.path(here, "cs_parity_mpdta.csv")), county_w,
               by = "countyreal")
  } else {
    d <- read.csv(file.path(here, if (rcs) "cs_parity_rcs.csv" else "cs_parity_unbal.csv"))
  }
  msgs <- character(0)
  a <- withCallingHandlers(
    suppressMessages(att_gt(
      yname = "lemp", tname = "year", idname = if (rcs) "rid" else "countyreal",
      gname = "first_treat", data = d, panel = !rcs,
      allow_unbalanced_panel = p$design == "unbal_allow",
      xformla = if (p$covariates) ~lpop else ~1,
      control_group = p$control_group,
      clustervars = if (p$cluster) "state" else NULL,
      weightsname = if (p$weighted) "w" else NULL,
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
    design = p$design, covariates = p$covariates,
    control_group = p$control_group, cluster = p$cluster,
    weighted = p$weighted,
    n = a$n, warnings = msgs,
    group = a$group[keep], time = a$t[keep], att = a$att[keep],
    se = a$se[keep], W = a$W, Wpval = a$Wpval, agg = agg)
}
write_json(list(meta = list(R_version = R.version.string,
                            did_version = as.character(packageVersion("did")),
                            DRDID_version = as.character(packageVersion("DRDID"))),
                cases = cases),
           file.path(here, "cs_parity_rc_unbal_R.json"),
           digits = NA, auto_unbox = TRUE, na = "null")
cat("wrote cs_parity_rc_unbal_R.json\n")
