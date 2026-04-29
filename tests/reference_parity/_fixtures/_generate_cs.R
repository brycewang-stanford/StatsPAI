#!/usr/bin/env Rscript
# R did::att_gt reference values for the staggered DGP.
suppressMessages({
  library(did)
  library(jsonlite)
})

df <- read.csv("tests/reference_parity/_fixtures/cs_data.csv")

# CS att_gt: 0 in gname means "never-treated control group" only when
# the package recognises it as outside the observation window.  Our
# data has years 1-6, so 0 is technically a valid earlier year.  Use
# a sentinel year FAR beyond the panel window — same effect, no
# ambiguity.
df$first_treat[df$first_treat == 0] <- 9999

# CS Doubly-Robust ATT(g, t)
out <- att_gt(
  yname        = "y",
  tname        = "year",
  idname       = "id",
  gname        = "first_treat",
  data         = df,
  control_group = "nevertreated",
  est_method   = "dr"
)

# Aggregations
agg_simple <- aggte(out, type = "simple")
agg_dynamic <- aggte(out, type = "dynamic")
agg_group   <- aggte(out, type = "group")

# Save
res <- list(
  meta = list(
    R_version = R.version.string,
    did_version = as.character(packageVersion("did")),
    est_method = "dr",
    control_group = "nevertreated"
  ),
  att_simple = list(
    estimate = unname(agg_simple$overall.att),
    se       = unname(agg_simple$overall.se)
  ),
  att_by_group = list(
    groups = unname(agg_group$egt),
    estimates = unname(agg_group$att.egt),
    ses       = unname(agg_group$se.egt)
  ),
  att_dynamic = list(
    e        = unname(agg_dynamic$egt),
    estimate = unname(agg_dynamic$att.egt),
    se       = unname(agg_dynamic$se.egt)
  ),
  att_gt_groups = unname(out$group),
  att_gt_times  = unname(out$t),
  att_gt_estimates = unname(out$att),
  att_gt_ses    = unname(out$se),
  n_obs = nrow(df)
)
write_json(res, "tests/reference_parity/_fixtures/cs_R.json",
           pretty = TRUE, auto_unbox = TRUE, digits = NA, na = "null")
cat(sprintf("Simple ATT: estimate=%.4f  se=%.4f\n",
            agg_simple$overall.att, agg_simple$overall.se))
cat(sprintf("Group ATTs: g=2 -> %.4f, g=3 -> %.4f\n",
            agg_group$att.egt[1], agg_group$att.egt[2]))
