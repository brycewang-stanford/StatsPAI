# StatsPAI Wooldridge ETWFE parity (R side) -- Module 17.
#
# Reads data/17_etwfe.csv (mpdta replica) and runs etwfe::etwfe.
#
# Pins three aggregations, not one:
#   att_etwfe               emfx(type="simple"), default cgroup="notyet"
#   att_group_notyet_<g>    emfx(type="group"),  default cgroup="notyet"
#   att_etwfe_never         emfx(type="simple"), cgroup="never", ivar set
#   att_group_never_<g>     emfx(type="group"),  cgroup="never", ivar set
#
# The group rows exist because the simple row did not constrain them: a
# headline aggregation can match to 1e-9 while every cohort ATT beneath it
# is wrong. See the note in 17_etwfe.py.
#
# Tolerance: rel_est < 1e-6, rel_se < 1e-3.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(etwfe)
})

MODULE <- "17_etwfe"

df <- read_csv_strict(MODULE)
df$first_treat <- as.numeric(df$first_treat)
df$year <- as.integer(df$year)

# etwfe::etwfe API: yvar / tvar / gvar (cohort) / data
fit <- etwfe::etwfe(
  fml  = lemp ~ 0,
  tvar = year,
  gvar = first_treat,
  data = df,
  vcov = ~ countyreal
)

# Aggregate to a single pooled ATT.
agg <- etwfe::emfx(fit, type = "simple")

rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "att_etwfe",
    estimate  = agg$estimate[1],
    se        = agg$std.error[1],
    ci_lo     = agg$conf.low[1],
    ci_hi     = agg$conf.high[1],
    n         = nrow(df)
  )
)

# Per-cohort ATTs under the same (not-yet-treated) fit.
grp <- as.data.frame(etwfe::emfx(fit, type = "group"))
for (i in seq_len(nrow(grp))) {
  rows[[length(rows) + 1]] <- parity_row(
    module    = MODULE,
    statistic = paste0("att_group_notyet_", as.integer(grp$first_treat[i])),
    estimate  = grp$estimate[i],
    se        = grp$std.error[i],
    n         = nrow(df)
  )
}

# Never-treated comparison group. `ivar` is set because the StatsPAI
# counterpart (sp.etwfe(cgroup='nevertreated') -> sp.wooldridge_did)
# absorbs unit fixed effects by two-way demeaning. Point estimates are
# invariant to this choice; the clustered SE is not.
fit_never <- etwfe::etwfe(
  fml    = lemp ~ 0,
  tvar   = year,
  gvar   = first_treat,
  ivar   = countyreal,
  data   = df,
  vcov   = ~ countyreal,
  cgroup = "never"
)

agg_never <- etwfe::emfx(fit_never, type = "simple")
rows[[length(rows) + 1]] <- parity_row(
  module    = MODULE,
  statistic = "att_etwfe_never",
  estimate  = agg_never$estimate[1],
  se        = agg_never$std.error[1],
  ci_lo     = agg_never$conf.low[1],
  ci_hi     = agg_never$conf.high[1],
  n         = nrow(df)
)

grp_never <- as.data.frame(etwfe::emfx(fit_never, type = "group"))
for (i in seq_len(nrow(grp_never))) {
  rows[[length(rows) + 1]] <- parity_row(
    module    = MODULE,
    statistic = paste0("att_group_never_", as.integer(grp_never$first_treat[i])),
    estimate  = grp_never$estimate[i],
    se        = grp_never$std.error[i],
    n         = nrow(df)
  )
}

write_results(MODULE, rows, extra = list(method = "etwfe::etwfe + emfx(simple|group)"))
