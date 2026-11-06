#!/usr/bin/env Rscript
# =====================================================================
#  R Synth on the Basque replica with equal, unscaled predictor weights,
#  for tests/reference_parity/test_synth_basque_equal_v_parity.py.
#
#  Reads tests/r_parity/data/07_scm.csv (hash-locked Track A data, the
#  same bytes module 07 runs). Module 07 runs the nested V search, where
#  R Synth and Stata synth land on different local optima (T4). The
#  default sp.synth call has no predictors and equal V: the donor
#  weights are simplex least squares on the fifteen pre-period outcomes
#  (1955-1969), a convex problem. Synth standardises each predictor by
#  its standard deviation across units; custom.v = that variance undoes
#  the standardisation, so its inner QP is the unweighted problem.
#
#  Writes synth_basque_equal_v_R.json with
#    treated -- Basque Country against the other sixteen regions: donor
#      weights, average post-period gap (1970-1997), pre-period RMSPE.
#    placebos -- the in-space placebo loop sp.synth runs: each donor in
#      turn is treated and the other units (Basque Country included, as
#      sp.synth and Stata synth2 do) are its donors; average gaps,
#      weights, and their population standard deviation, which sp.synth
#      reports as `se`.
#  Every optimum here is unique (the Python test certifies it), so any
#  correct solver must return the same weights.
#
#  ipop is tightened (Sigf 20, Margin 1e-12) wherever it solves; the
#  setting that solved is recorded with each fit. See IPOP below.
#  Environment: R 4.5.2 / Synth 1.1.10
# =====================================================================
suppressPackageStartupMessages({library(Synth); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

df <- read.csv(file.path(OUT, "..", "..", "r_parity", "data", "07_scm.csv"),
               stringsAsFactors = FALSE)
names_all <- sort(unique(df$region))
df$id <- match(df$region, names_all)
YEARS <- sort(unique(df$year))
PRE <- YEARS[YEARS < 1970]
POST <- YEARS[YEARS >= 1970]

#: ipop settings, tightest first; a vertex optimum (all weight on one
#: unit) makes ipop's KKT system singular at the tight settings.
IPOP <- list(c(20, 1e-12), c(12, 1e-10), c(10, 1e-8), c(7, 5e-4))

fit_one <- function(tr, ctrl) {
  dp <- dataprep(
    foo = df, predictors = NULL, dependent = "gdppc",
    unit.variable = "id", time.variable = "year",
    treatment.identifier = tr, controls.identifier = ctrl,
    special.predictors = lapply(PRE, function(p) list("gdppc", p, "mean")),
    time.predictors.prior = PRE, time.optimize.ssr = PRE,
    time.plot = YEARS, unit.names.variable = "region"
  )
  custom.v <- apply(cbind(dp$X0, dp$X1), 1, var)
  out <- NULL
  for (setting in IPOP) {
    out <- tryCatch(
      synth(dp, verbose = FALSE, custom.v = custom.v, Sigf.ipop = setting[1],
            Margin.ipop = setting[2], Bound.ipop = 10),
      error = function(e) NULL)
    if (!is.null(out)) break
  }
  stopifnot(!is.null(out))
  gap <- as.numeric(dp$Y1plot) - as.numeric(dp$Y0plot %*% out$solution.w)
  list(w = setNames(as.numeric(out$solution.w), names_all[ctrl]),
       att = mean(gap[YEARS >= 1970]),
       pre_rmspe = sqrt(mean(gap[YEARS < 1970]^2)),
       sigf_ipop = setting[1], margin_ipop = setting[2])
}

tr <- match("Basque Country", names_all)
treated <- fit_one(tr, setdiff(seq_along(names_all), tr))

placebos <- list()
for (d in setdiff(seq_along(names_all), tr)) {
  r <- fit_one(d, setdiff(seq_along(names_all), d))
  placebos[[names_all[d]]] <- list(att = r$att, weights = as.list(r$w),
                                   sigf_ipop = r$sigf_ipop,
                                   margin_ipop = r$margin_ipop)
}
atts <- sapply(placebos, function(p) p$att)

out <- list(
  treated = list(weights = as.list(treated$w), att = treated$att,
                 pre_rmspe = treated$pre_rmspe,
                 sigf_ipop = treated$sigf_ipop,
                 margin_ipop = treated$margin_ipop),
  placebos = placebos,
  sd_population = sqrt(mean((atts - mean(atts))^2)),
  `_meta` = list(
    generated_by = "_generate_synth_basque_equal_v_R.R",
    data = "tests/r_parity/data/07_scm.csv",
    r_version = R.version.string,
    synth_version = as.character(packageVersion("Synth"))
  )
)
write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
      file.path(OUT, "synth_basque_equal_v_R.json"))
cat("[synth-basque-equal-v] att", treated$att, " placebo sd",
    out$sd_population, "\n")
