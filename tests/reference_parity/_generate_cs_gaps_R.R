#!/usr/bin/env Rscript
# Reference values for three Callaway-Sant'Anna behaviours that StatsPAI
# either got wrong or did not implement before 1.23.0:
#
#   A. anticipation > 0 under base_period = "varying". R shifts the base
#      period only for POST-treatment cells; pre-treatment placebos keep
#      the period immediately before them (compute.att_gt.R, `pret` block).
#      StatsPAI used to shift the placebos too, which moved every
#      pre-treatment cell and dropped the earliest ones.
#
#   B. allow_unbalanced_panel = TRUE. R switches to the repeated
#      cross-section estimators and folds the influence functions to the
#      unit level (`.rowid <- idname`). StatsPAI had no such option.
#
#   C. clustervars= with the multiplier bootstrap on UNEQUAL cluster
#      sizes.
#
#      NOTE: no clustered reference is emitted here on purpose. CRAN did
#      2.3.0 -- the version this script runs against -- aggregates the
#      influence function to cluster *means* over n_clusters
#      (`rowsum(inf.func, cluster)/cluster_n`, `se <- bSigma/sqrt(n_clusters)`),
#      which is what StatsPAI mirrored and which only equals the
#      cluster-robust variance when every cluster is the same size. On the
#      lopsided cluster sizes below it inflates the SEs ~5x, because the
#      size-1 and size-3 clusters enter with weight 1/|c|. Upstream did
#      (GitHub master, post-2.3.0) switched to cluster *sums* with
#      `se <- bSigma * sqrt(n_clusters)/n` for exactly this reason, and
#      csdid tracks the corrected form. Pinning 2.3.0's clustered numbers
#      would freeze the superseded convention, so only the UNCLUSTERED
#      bootstrap -- identical in both -- is recorded, and the clustered
#      path is covered by property-based tests instead.
#
# Writes:
#   _fixtures/cs_gaps_panel.csv            balanced panel + unequal `state`
#   _fixtures/cs_gaps_unbalanced_panel.csv same panel, 18% of rows deleted
#   _fixtures/cs_gaps_reference.csv        one row per (case, g, t)
#
# R 4.x, did 2.3.0.

suppressPackageStartupMessages(library(did))

set.seed(20260811)

fixtures <- "tests/reference_parity/_fixtures"

# ---------------------------------------------------------------------
# Panel: 6 periods, cohorts {3, 5, never}, one covariate.
# `state` is deliberately lopsided (one huge cluster, several tiny ones)
# so the sums-vs-means bootstrap bug cannot hide.
# ---------------------------------------------------------------------
n_units <- 360
periods <- 1:6
cohort_pool <- c(rep(3, 110), rep(5, 110), rep(0, 140))
g_by_unit <- sample(cohort_pool)

# cluster sizes 150, 90, 45, 30, 20, 15, 6, 3, 1
state_sizes <- c(150, 90, 45, 30, 20, 15, 6, 3, 1)
state_by_unit <- rep(seq_along(state_sizes), times = state_sizes)

x_by_unit <- rnorm(n_units)
unit_fe <- rnorm(n_units)
state_shock <- rnorm(length(state_sizes), sd = 0.4)[state_by_unit]

rows <- vector("list", n_units * length(periods))
k <- 0
for (u in seq_len(n_units)) {
  g <- g_by_unit[u]
  for (tt in periods) {
    k <- k + 1
    e <- if (g > 0 && tt >= g) tt - g + 1 else 0
    y <- unit_fe[u] + state_shock[u] + 0.3 * tt + 0.5 * x_by_unit[u] +
      1.2 * e + rnorm(1, sd = 0.8)
    rows[[k]] <- data.frame(
      i = u, t = tt, g = g, state = state_by_unit[u],
      x1 = x_by_unit[u], y = y
    )
  }
}
panel <- do.call(rbind, rows)
panel <- panel[order(panel$i, panel$t), ]
# did 2.3.0 recodes never-treated `g == 0` to Inf internally. If `g` is an
# INTEGER column (which it becomes on a read.csv round-trip) that assignment
# silently fails to NA and every never-treated unit is dropped as "missing
# data", leaving the last treated cohort coerced into the control group.
# Keep the reference columns double so the fixture and any re-run agree.
panel$g <- as.numeric(panel$g)
panel$i <- as.numeric(panel$i)
panel$t <- as.numeric(panel$t)
write.csv(panel, file.path(fixtures, "cs_gaps_panel.csv"), row.names = FALSE)

# Unbalanced copy: delete 18% of rows, but never a unit's whole history.
drop_idx <- sample(seq_len(nrow(panel)), size = floor(0.18 * nrow(panel)))
unb <- panel[-drop_idx, ]
keep_units <- names(which(table(unb$i) >= 2))
unb <- unb[unb$i %in% as.numeric(keep_units), ]
unb <- unb[order(unb$i, unb$t), ]
write.csv(unb, file.path(fixtures, "cs_gaps_unbalanced_panel.csv"),
          row.names = FALSE)

collect <- function(case, res) {
  data.frame(
    case = case,
    group = res$group,
    time = res$t,
    att = as.numeric(res$att),
    se = as.numeric(res$se),
    stringsAsFactors = FALSE
  )
}

out <- list()

# ---- A. anticipation x base_period ---------------------------------
for (bp in c("varying", "universal")) {
  for (a in c(0, 1, 2)) {
    res <- att_gt(
      yname = "y", tname = "t", idname = "i", gname = "g",
      xformla = ~1, data = panel, panel = TRUE,
      control_group = "nevertreated", anticipation = a,
      base_period = bp, est_method = "dr", bstrap = FALSE, cband = FALSE
    )
    out[[length(out) + 1]] <- collect(
      sprintf("anticipation:%s:a%d", bp, a), res
    )
  }
}

# ---- B. allow_unbalanced_panel --------------------------------------
for (em in c("dr", "ipw", "reg")) {
  for (xf in c("none", "x1")) {
    form <- if (xf == "none") ~1 else ~x1
    res <- att_gt(
      yname = "y", tname = "t", idname = "i", gname = "g",
      xformla = form, data = unb, panel = TRUE,
      allow_unbalanced_panel = TRUE,
      control_group = "nevertreated", base_period = "varying",
      est_method = em, bstrap = FALSE, cband = FALSE
    )
    out[[length(out) + 1]] <- collect(
      sprintf("unbalanced:%s:%s", em, xf), res
    )
  }
}

# ---- C. unclustered multiplier bootstrap ----------------------------
# The clustered variant is deliberately NOT recorded (see the header
# note). The unclustered path is identical in 2.3.0 and master, and was
# already correct in StatsPAI, so it is pinned as a control: the cluster
# fix must not have disturbed it.
set.seed(4242)
res <- att_gt(
  yname = "y", tname = "t", idname = "i", gname = "g",
  xformla = ~1, data = panel, panel = TRUE,
  control_group = "nevertreated", base_period = "varying",
  est_method = "dr", bstrap = TRUE, biters = 200000, cband = FALSE
)
out[[length(out) + 1]] <- collect("cluster_boot:none", res)

reference <- do.call(rbind, out)
write.csv(reference, file.path(fixtures, "cs_gaps_reference.csv"),
          row.names = FALSE)

cat("wrote", nrow(reference), "reference rows\n")
print(table(reference$case))
