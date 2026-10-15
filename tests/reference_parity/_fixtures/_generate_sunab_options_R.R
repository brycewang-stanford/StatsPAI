# Reference for sp.sun_abraham(control_group='lastcohort') and
# sp.sun_abraham(window_rule='reference') with cohort shares held fixed.
#
#   feols(lemp ~ sunab(first_treat, year, ref.p = ..., ref.c = ...)
#         | countyreal + year, cluster = ~countyreal)
#
# on mpdta (tests/orig_parity/data/02_mpdta_original.csv):
#
# * ref.c = 2007 makes the last cohort a reference cohort next to the
#   never-treated (sp.sun_abraham(control_group='lastcohort'), which leaves
#   the never-treated units in the regression without dummies too);
# * ref.p = c(-1, <relative times outside the window>) puts the relative
#   times outside the window into the omitted category
#   (sp.sun_abraham(event_window=..., window_rule='reference')).
#
# For each fit: the per-relative-time coefficients and SEs fixest reports,
# the cohort-size-weighted ATT (summary(agg = "att")), and the
# equal-weighted average of the post-treatment event-time effects with SE
# sqrt(w' V w) from the aggregated covariance (the construction of
# _generate_sunab_event_time_R.R, whose aggregation map is first required
# to reproduce fixest's own aggregated coefficients and SEs).
#
# Run from the repository root:
#   Rscript tests/reference_parity/_fixtures/_generate_sunab_options_R.R
suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
})

df <- read.csv("tests/orig_parity/data/02_mpdta_original.csv")
df$first_treat <- as.numeric(df$first_treat)

one <- function(ref_p, ref_c) {
  fit <- if (is.null(ref_c)) {
    feols(lemp ~ sunab(first_treat, year, ref.p = ref_p) | countyreal + year,
          data = df, cluster = ~countyreal)
  } else {
    feols(lemp ~ sunab(first_treat, year, ref.p = ref_p, ref.c = ref_c) |
            countyreal + year, data = df, cluster = ~countyreal)
  }
  b_full <- coef(fit, agg = FALSE)
  b_full <- b_full[grepl("cohort::", names(b_full))]
  V_full <- vcov(fit)[names(b_full), names(b_full)]
  cell_rel <- as.integer(sub("^year::(-?\\d+):cohort::.*$", "\\1", names(b_full)))
  cell_coh <- as.numeric(sub("^.*:cohort::(\\d+)$", "\\1", names(b_full)))
  n_cell <- mapply(function(g, e) sum(df$first_treat == g & df$year - g == e),
                   cell_coh, cell_rel)
  b_agg <- coef(fit)
  rel <- as.integer(sub("^year::(-?\\d+)$", "\\1", names(b_agg)))
  A <- t(sapply(rel, function(e) {
    on <- cell_rel == e
    out <- rep(0, length(b_full))
    out[on] <- n_cell[on] / sum(n_cell[on])
    out
  }))
  stopifnot(max(abs(A %*% b_full - b_agg)) < 1e-12)
  stopifnot(max(abs(sqrt(diag(A %*% V_full %*% t(A))) - se(fit)) / se(fit)) < 1e-10)
  V_agg <- A %*% V_full %*% t(A)
  post <- which(rel >= 0)
  w <- rep(0, length(b_agg))
  w[post] <- 1 / length(post)
  att <- summary(fit, agg = "att")$coeftable["ATT", ]
  list(
    ref_p = ref_p,
    ref_c = ref_c,
    event_time = setNames(
      lapply(seq_along(rel), function(j) list(att = unname(b_agg[j]),
                                              se = unname(se(fit)[j]))),
      as.character(rel)),
    att_event_time = sum(w * b_agg),
    se_event_time = sqrt(as.numeric(t(w) %*% V_agg %*% w)),
    att_fixest_att = unname(att["Estimate"]),
    se_fixest_att = unname(att["Std. Error"]),
    n = nobs(fit)
  )
}

out <- list(
  lastcohort = one(-1, 2007),
  reference_m3_p2 = one(c(-1, -4, 3), NULL),
  reference_m2_p1 = one(c(-1, -4, -3, 2, 3), NULL),
  reference_m3_p2_lastcohort = one(c(-1, -4, 3), 2007),
  `_meta` = list(
    generated_by = "_generate_sunab_options_R.R",
    fixest = as.character(packageVersion("fixest")),
    r_version = R.version.string
  )
)
write_json(out, "tests/reference_parity/_fixtures/sunab_options_R.json",
           digits = NA, auto_unbox = TRUE, pretty = TRUE)
cat("[sunab-options] wrote", length(out) - 1, "fits\n")
