#!/usr/bin/env Rscript
# R did::att_gt + aggte(na.rm = TRUE) for (a) a panel without never-treated
# units under control_group = "notyettreated" and (b) an unbalanced panel
# (allow_unbalanced_panel = TRUE) under both control groups.
#
#   python tests/reference_parity/_fixtures/_generate_cs_no_never_unbalanced_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_cs_no_never_unbalanced_R.R
suppressMessages({library(did); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
fit <- function(df, ...) {
  r <- suppressWarnings(att_gt(yname = "y", tname = "t", idname = "id", gname = "g",
                               data = df, bstrap = FALSE, ...))
  s <- aggte(r, type = "simple", na.rm = TRUE, bstrap = FALSE)
  keep <- !is.na(r$se)
  list(group = r$group[keep], time = r$t[keep], att = r$att[keep], se = r$se[keep],
       simple_att = s$overall.att, simple_se = s$overall.se)
}
nn <- read.csv(file.path(here, "cs_no_never_data.csv")); nn$g <- as.numeric(nn$g)
ub <- read.csv(file.path(here, "cs_unbalanced_notyet_data.csv")); ub$g <- as.numeric(ub$g)
cases <- list()
for (panel in c(TRUE, FALSE)) for (base in c("varying", "universal"))
  for (ant in c(0, 1)) for (est in c("reg", "dr")) {
    cases[[length(cases) + 1]] <- c(
      list(data = "no_never", panel = panel, base_period = base, anticipation = ant,
           est_method = est, control_group = "notyettreated"),
      fit(nn, panel = panel, base_period = base, anticipation = ant, est_method = est,
          control_group = "notyettreated"))
  }
for (cg in c("nevertreated", "notyettreated")) for (est in c("reg", "dr")) {
  cases[[length(cases) + 1]] <- c(
    list(data = "unbalanced", panel = TRUE, base_period = "varying", anticipation = 0,
         est_method = est, control_group = cg),
    fit(ub, allow_unbalanced_panel = TRUE, est_method = est, control_group = cg))
}
write_json(list(meta = list(R_version = R.version.string,
                            did_version = as.character(packageVersion("did"))),
                cases = cases),
           file.path(here, "cs_no_never_unbalanced_R.json"), digits = NA, auto_unbox = TRUE)
cat("wrote cs_no_never_unbalanced_R.json\n")
