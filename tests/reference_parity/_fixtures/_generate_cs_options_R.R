#!/usr/bin/env Rscript
# R did references for sp.callaway_santanna option values that had no
# same-byte reference of their own:
#
#   estimator='drimp'       est_method = DRDID::drdid_imp_panel / drdid_imp_rc
#   estimator='ipw_abadie'  est_method = DRDID::ipw_did_panel / ipw_did_rc
#                           (Abadie 2005; Stata csdid's method(ipw))
#   estimator='stdipw'      est_method = "ipw" (DRDID::std_ipw_did_panel /
#                           std_ipw_did_rc), the same estimator as 'ipw' under
#                           a second name
#   anticipation > 0        with weights and a covariate (never-treated), and
#                           with clustering (not-yet-treated, varying base),
#                           through all four aggregations
#   clustering              with est_method = "ipw" and "reg"
#
# att_gt accepts a function as est_method and calls it with the same
# arguments as its built-in estimators, so the improved doubly robust and the
# Abadie IPW estimators of Sant'Anna's DRDID package are reached through the
# canonical did pipeline (cells, influence functions, aggte).
#
# Data: tests/orig_parity/data/02_mpdta_original.csv (did::mpdta, byte-locked
# by its SHA-256 in test_aggte_mpdta_parity.py) with two derived columns that
# the Python test computes by the same integer arithmetic:
#   state = countyreal %/% 1000        (the state FIPS code; 29 clusters)
#   w     = 1 + (countyreal %% 7) / 7  (a time-invariant unit weight)
#
#   Rscript tests/reference_parity/_fixtures/_generate_cs_options_R.R
suppressMessages({library(did); library(DRDID); library(jsonlite)})
stopifnot(utils::packageVersion("did") >= "2.5.0")
here <- "tests/reference_parity/_fixtures"
d <- read.csv("tests/orig_parity/data/02_mpdta_original.csv")
d$first_treat <- as.numeric(d$first_treat)  # integer gname loses never-treated
d$state <- d$countyreal %/% 1000
d$w <- 1 + (d$countyreal %% 7) / 7

est <- list(
  drimp = DRDID::drdid_imp_panel, drimp_rc = DRDID::drdid_imp_rc,
  ipw_abadie = DRDID::ipw_did_panel, ipw_abadie_rc = DRDID::ipw_did_rc
)
cases <- list(
  list(id = "drimp_nev_universal", est = "drimp", panel = TRUE, x = TRUE,
       control_group = "nevertreated", base_period = "universal"),
  list(id = "drimp_nyt_varying", est = "drimp", panel = TRUE, x = TRUE,
       control_group = "notyettreated", base_period = "varying"),
  list(id = "drimp_rcs", est = "drimp_rc", panel = FALSE, x = TRUE,
       control_group = "nevertreated", base_period = "universal"),
  list(id = "ipw_abadie_nev_universal", est = "ipw_abadie", panel = TRUE, x = TRUE,
       control_group = "nevertreated", base_period = "universal"),
  list(id = "ipw_abadie_rcs", est = "ipw_abadie_rc", panel = FALSE, x = TRUE,
       control_group = "nevertreated", base_period = "universal"),
  list(id = "stdipw_rcs", est = "ipw", panel = FALSE, x = TRUE,
       control_group = "nevertreated", base_period = "universal"),
  list(id = "ipw_cluster", est = "ipw", panel = TRUE, x = TRUE,
       control_group = "nevertreated", base_period = "universal",
       clustervars = "state"),
  list(id = "ant1_dr_weighted", est = "dr", panel = TRUE, x = TRUE,
       control_group = "nevertreated", base_period = "universal",
       anticipation = 1, weights = "w"),
  list(id = "ant1_reg_cluster_nyt", est = "reg", panel = TRUE, x = FALSE,
       control_group = "notyettreated", base_period = "varying",
       anticipation = 1, clustervars = "state")
)

out <- list()
for (cs in cases) {
  method <- if (cs$est %in% names(est)) est[[cs$est]] else cs$est
  a <- suppressMessages(suppressWarnings(att_gt(
    yname = "lemp", tname = "year", idname = "countyreal", gname = "first_treat",
    data = d, xformla = if (isTRUE(cs$x)) ~lpop else NULL,
    est_method = method, panel = cs$panel,
    control_group = cs$control_group, base_period = cs$base_period,
    anticipation = if (is.null(cs$anticipation)) 0 else cs$anticipation,
    weightsname = cs$weights, clustervars = cs$clustervars,
    bstrap = FALSE, cband = FALSE)))
  agg <- list()
  for (type in c("simple", "dynamic", "group", "calendar")) {
    s <- suppressWarnings(aggte(a, type = type, bstrap = FALSE, cband = FALSE,
                                na.rm = TRUE))
    agg[[type]] <- list(att = s$overall.att, se = s$overall.se,
                        egt = s$egt, att_egt = s$att.egt,
                        se_egt = as.numeric(s$se.egt))
  }
  keep <- !is.na(a$se)
  out[[cs$id]] <- c(
    cs[setdiff(names(cs), "id")],
    list(group = a$group[keep], time = a$t[keep], att = a$att[keep],
         se = a$se[keep], agg = agg)
  )
  cat(cs$id, "done\n")
}
write_json(list(meta = list(R_version = R.version.string,
                            did_version = as.character(packageVersion("did")),
                            DRDID_version = as.character(packageVersion("DRDID")),
                            data = "tests/orig_parity/data/02_mpdta_original.csv",
                            state = "countyreal %/% 1000",
                            w = "1 + (countyreal %% 7) / 7"),
                cases = out),
           file.path(here, "cs_options_R.json"),
           digits = NA, auto_unbox = TRUE, na = "null")
cat("wrote cs_options_R.json\n")
