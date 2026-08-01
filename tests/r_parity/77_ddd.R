# StatsPAI triple-differences parity (R side) -- Module 77.
#
# Runs triplediff::ddd (Ortiz-Villavicencio & Sant'Anna 2025) on the panel
# dumped by 77_ddd.py, with no covariates so the doubly-robust DDD reduces
# to the unconditional cell means sp.ddd_heterogeneous computes, and
# agg_ddd(type = "simple") for the overall number.
#
# Only post-treatment cells are emitted: base_period = "varying" also
# produces (g, t) for t < g, which the Python side does not build.
#
# Standard errors are not compared -- triplediff reports analytical
# influence-function SEs and sp.ddd_heterogeneous only has a cluster
# bootstrap.
#
# Tolerance: rel < 1e-6.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(triplediff)
  library(data.table)
})

MODULE <- "77_ddd"

df <- as.data.table(read_csv_strict(MODULE))

res <- ddd(
  yname = "y", tname = "time", idname = "id", gname = "state",
  pname = "partition", xformla = ~1, data = df,
  control_group = "nevertreated", base_period = "varying",
  est_method = "dr", panel = TRUE, boot = FALSE
)

rows <- list()
post <- res$periods >= res$groups
for (k in which(post)) {
  rows[[length(rows) + 1]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("ddd_g%d_t%d", res$groups[k], res$periods[k]),
    estimate  = res$ATT[k],
    se        = NA_real_,
    n         = nrow(df)
  )
}

agg <- agg_ddd(res, type = "simple")$aggte_ddd
rows[[length(rows) + 1]] <- parity_row(
  module    = MODULE,
  statistic = "simple_ATT_cohort_weights",
  estimate  = agg$overall.att,
  se        = NA_real_,
  n         = nrow(df)
)

# Conditional DDD across the three nuisance combinations. Standard errors
# ARE compared here: sp.ddd_heterogeneous(se="analytic") uses the same
# influence-function variance, so a gap would be a real disagreement rather
# than two different estimators of the same quantity.
for (method in c("dr", "ipw", "reg")) {
  cond <- ddd(
    yname = "y", tname = "time", idname = "id", gname = "state",
    pname = "partition", xformla = ~ cov1 + cov2, data = df,
    control_group = "nevertreated", base_period = "varying",
    est_method = method, panel = TRUE, boot = FALSE
  )
  post_c <- cond$periods >= cond$groups
  for (k in which(post_c)) {
    rows[[length(rows) + 1]] <- parity_row(
      module    = MODULE,
      statistic = sprintf("ddd_%s_g%d_t%d", method, cond$groups[k], cond$periods[k]),
      estimate  = cond$ATT[k],
      se        = cond$se[k],
      n         = nrow(df)
    )
  }
  agg_c <- agg_ddd(cond, type = "simple")$aggte_ddd
  rows[[length(rows) + 1]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("simple_ATT_%s", method),
    estimate  = agg_c$overall.att,
    se        = agg_c$overall.se,
    n         = nrow(df)
  )
}

write_results(MODULE, rows, extra = list(
  reference = "triplediff::ddd + agg_ddd(type='simple')",
  triplediff = as.character(utils::packageVersion("triplediff"))
))
