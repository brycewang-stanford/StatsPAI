# StatsPAI stacked-DiD parity (R side) -- Module 75.
#
# There is no CRAN package for Cengiz-Dube-Lindner-Zipperer stacking, so this
# is a hand-written reference: one sub-experiment per treated cohort with its
# clean controls over the event window, stacked, then TWFE with
# cohort-specific unit and time fixed effects and k = -1 as the reference
# period. Estimated with fixest::feols, clustered on the unit id.
#
# Both control-group conventions are emitted: never-treated only (StatsPAI's
# default) and never + not-yet-treated.
#
# Tolerance: rel < 1e-6 on the event-study coefficients and the post-period
# mean.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(fixest)
  library(data.table)
})

MODULE <- "75_stacked"
W_LO <- -3
W_HI <- 3

df <- as.data.table(read_csv_strict(MODULE))
cohorts <- sort(unique(df[first_treat > 0, first_treat]))
never <- unique(df[first_treat == 0, id])

stack_fit <- function(never_only) {
  frames <- list()
  for (g in cohorts) {
    t_lo <- g + W_LO
    t_hi <- g + W_HI
    coh <- unique(df[first_treat == g, id])
    nyt <- unique(df[first_treat > t_hi & first_treat > 0, id])
    ctrl <- if (never_only) never else union(never, nyt)
    if (!length(ctrl)) next
    sub <- df[id %in% union(coh, ctrl) & year >= t_lo & year <= t_hi]
    if (!nrow(sub)) next
    sub[, `:=`(
      cohort = g,
      rel = year - g,
      treated_unit = as.integer(id %in% coh)
    )]
    frames[[length(frames) + 1]] <- sub
  }
  st <- rbindlist(frames)
  st[, `:=`(uc = paste0(id, "_", cohort), tc = paste0(year, "_", cohort))]
  st[, rel_f := factor(rel, levels = sort(unique(rel)))]
  fit <- feols(y ~ i(rel_f, treated_unit, ref = "-1") | uc + tc,
               data = st, cluster = ~ id)
  list(fit = fit, n = nrow(st))
}

rows <- list()
for (spec in c("never", "nyt")) {
  res <- stack_fit(spec == "never")
  co <- coef(res$fit)
  se <- se(res$fit)
  rels <- as.numeric(sub(".*rel_f::(-?[0-9]+):.*", "\\1", names(co)))
  for (k in seq_along(co)) {
    rows[[length(rows) + 1]] <- parity_row(
      module    = MODULE,
      statistic = sprintf("%s_att_rel_%d", spec, rels[k]),
      estimate  = unname(co[k]),
      se        = unname(se[k]),
      n         = res$n
    )
  }
  rows[[length(rows) + 1]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("%s_ATT_post", spec),
    estimate  = mean(co[rels >= 0]),
    se        = NA_real_,
    n         = res$n
  )
}

write_results(MODULE, rows, extra = list(
  reference = "hand-written stacked DiD via fixest::feols",
  fixest = as.character(utils::packageVersion("fixest"))
))
