#!/usr/bin/env Rscript
# =====================================================================
#  Reference values for sp.event_study's options, from fixest::feols.
#
#  Reads event_study_options.csv (written by
#  _generate_event_study_options_data.py), writes
#  event_study_options_R.json.
#
#  Each specification is a two-way fixed-effects regression on event-time
#  regressors, built here from the definitions sp.event_study documents:
#
#  * relative time outside `window` is clipped into the endpoints;
#  * with `bin`, relative time is grouped into bins of that width anchored
#    at the treatment boundary ([0, k-1], [k, 2k-1], ... and [-k, -1],
#    [-2k, -k-1], ...), labelled by their left edge after clipping to the
#    window;
#  * every bin in `ref` is omitted (fixest's i(..., ref = c(...)));
#  * with `intensity`, the regressor is the bin indicator times the
#    intensity column (an exposure x period event study);
#  * never-treated units are parked at a reference level and interacted
#    with a zero, so they stay in the sample with no regressor of their
#    own (see tests/r_parity/85_twfe_event_study.R).
#
#  Fixed effects: unit + time, plus `absorb` (fixest syntax a^b for the
#  cell of two columns). Clustered on `cluster` with fixest's default
#  small-sample correction (adj = TRUE, fixef.K = "nested",
#  cluster.adj = TRUE).
#
#  Environment: R 4.5.2 / fixest 0.14
# =====================================================================
suppressPackageStartupMessages({library(fixest); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."
d0 <- read.csv(file.path(OUT, "event_study_options.csv"))

spec <- function(name, sample = "staggered", y = "y", treat_time = "g",
                 window = c(-4, 4), ref = -1, bin = NA, covariates = NULL,
                 weights = NULL, cluster = "unit", intensity = NULL,
                 absorb = NULL, adoption = NULL) {
  list(name = name, sample = sample, y = y, treat_time = treat_time,
       window = window, ref = ref, bin = bin, covariates = covariates,
       weights = weights, cluster = cluster, intensity = intensity,
       absorb = absorb)
}

specs <- list(
  # adoption: staggered, at the default window and reference period
  spec("stag_default"),
  spec("stag_cov_wt", covariates = "x", weights = "w"),
  # window: other
  spec("stag_w-2_2", window = c(-2, 2)),
  spec("stag_w-6_3", window = c(-6, 3)),
  spec("stag_w-3_7", window = c(-3, 7)),
  spec("stag_w-1_3", window = c(-1, 3)),
  spec("stag_w-9_8", window = c(-9, 8)),
  spec("single_w-2_4", sample = "single", window = c(-2, 4)),
  # ref_period: other
  spec("stag_ref-2", ref = -2),
  spec("stag_ref-4", ref = -4),
  spec("stag_ref_span-3_-1", window = c(-5, 4), ref = c(-3, -2, -1)),
  spec("stag_ref_le-3", window = c(-6, 4), ref = c(-6, -5, -4, -3)),
  spec("stag_ref_ge5", window = c(-4, 6), ref = c(5, 6)),
  spec("single_ref-3", sample = "single", ref = -3),
  # cluster: other
  spec("stag_cl_state", cluster = "state"),
  spec("stag_cl_time", cluster = "time"),
  spec("single_cl_state_wt", sample = "single", cluster = "state",
       weights = "w"),
  # bins
  spec("stag_bin2", window = c(-6, 5), bin = 2, ref = -2),
  spec("stag_bin3", window = c(-6, 6), bin = 3, ref = -3),
  spec("stag_bin2_ref_le-3", window = c(-6, 5), bin = 2, ref = c(-6, -4)),
  spec("single_bin2_cov", sample = "single", window = c(-4, 3), bin = 2,
       ref = -2, covariates = "x"),
  # intensity (common adoption date 7, outcome y_int; one staggered)
  spec("int_common", y = "y_int", treat_time = 7, intensity = "dose"),
  spec("int_common_w-6_5_cl_state", y = "y_int", treat_time = 7,
       window = c(-6, 5), intensity = "dose", cluster = "state"),
  spec("int_common_wt", y = "y_int", treat_time = 7, intensity = "dose",
       weights = "w"),
  spec("int_staggered", intensity = "dose"),
  # absorb
  spec("stag_absorb_region_time", absorb = "region^time"),
  spec("stag_absorb_state_time_cov", absorb = "state^time",
       covariates = "x"),
  spec("single_absorb_region_time_wt", sample = "single",
       absorb = "region^time", weights = "w"),
  spec("stag_bin2_absorb", window = c(-6, 5), bin = 2, ref = -2,
       absorb = "region^time"),
  # unbalanced panels and weights that vary over time
  spec("unbal_stag_default", sample = "unbalanced"),
  spec("unbal_stag_wt", sample = "unbalanced", weights = "w"),
  spec("unbal_stag_cov_cl_state", sample = "unbalanced", covariates = "x",
       cluster = "state"),
  spec("unbal_single", sample = "unbalanced_single"),
  spec("stag_tv_weights", weights = "wt"),
  spec("unbal_stag_tv_weights_w-3_5", sample = "unbalanced",
       weights = "wt", window = c(-3, 5))
)

bin_start <- function(t, k) {
  ifelse(t >= 0, (t %/% k) * k, -(((-t - 1) %/% k) + 1) * k)
}

out <- list()
for (s in specs) {
  d <- switch(s$sample,
    staggered = d0,
    single = d0[d0$g %in% c(0, 8), ],
    unbalanced = d0[d0$keep == 1, ],
    unbalanced_single = d0[d0$keep == 1 & d0$g %in% c(0, 8), ])
  win <- s$window
  tt <- if (is.character(s$treat_time)) {
    ifelse(d[[s$treat_time]] > 0, d[[s$treat_time]], NA)
  } else rep(s$treat_time, nrow(d))
  d$treat <- as.integer(!is.na(tt))
  rel <- pmin(pmax(d$time - tt, win[1]), win[2])
  lab <- if (is.na(s$bin)) rel else pmax(bin_start(rel, s$bin), win[1])
  ref_labs <- if (is.na(s$bin)) s$ref else s$ref
  d$rel_f <- ifelse(is.na(lab), ref_labs[1], lab)
  d$z <- if (is.null(s$intensity)) d$treat else d$treat * d[[s$intensity]]
  rhs <- paste0("i(rel_f, z, ref = c(", paste(ref_labs, collapse = ", "), "))")
  if (!is.null(s$covariates)) rhs <- paste(c(rhs, s$covariates), collapse = " + ")
  # A cell effect a^time spans the time effects. fixest's nested K counts
  # every level of every fixed effect not nested in the cluster and does not
  # detect that redundancy (K = 67 for stag_absorb_region_time, where the
  # rank is 55); reghdfe does, and so does sp.event_study's absorb path
  # (sp.hdfe_ols). Writing the same model without the redundant time
  # effect gives fixest the exact count. The coefficients are unchanged.
  time_fe <- if (!is.null(s$absorb) && grepl("\\^time$", s$absorb)) NULL else "time"
  fe <- paste(c("unit", time_fe, s$absorb), collapse = " + ")
  fml <- as.formula(paste0(s$y, " ~ ", rhs, " | ", fe))
  cl <- as.formula(paste0("~", s$cluster))
  fit <- if (is.null(s$weights)) {
    feols(fml, data = d, cluster = cl)
  } else {
    feols(fml, data = d, cluster = cl, weights = as.formula(paste0("~", s$weights)))
  }
  ct <- summary(fit)$coeftable
  coefs <- list()
  for (nm in rownames(ct)) {
    if (!grepl("rel_f", nm)) next
    k <- as.integer(sub("^rel_f::(-?[0-9]+).*$", "\\1", nm))
    coefs[[as.character(k)]] <- list(
      estimate = unname(ct[nm, "Estimate"]),
      se = unname(ct[nm, "Std. Error"]))
  }
  out[[s$name]] <- list(
    sample = s$sample, y = s$y, treat_time = s$treat_time,
    window = win, ref = s$ref,
    bin = if (is.na(s$bin)) NULL else s$bin,
    covariates = s$covariates, weights = s$weights, cluster = s$cluster,
    intensity = s$intensity, absorb = s$absorb,
    n = nobs(fit), n_coef = length(coefs), coefs = coefs)
}
out[["_meta"]] <- list(
  generated_by = "_generate_event_study_options_R.R",
  r_version = R.version.string,
  fixest_version = as.character(packageVersion("fixest")),
  n_specs = length(out),
  ssc = "fixest default: adj = TRUE, fixef.K = 'nested', cluster.adj = TRUE")
write(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null"),
      file.path(OUT, "event_study_options_R.json"))
cat("[event-study-options] wrote", length(out) - 1, "specs\n")
