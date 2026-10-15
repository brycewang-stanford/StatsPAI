#!/usr/bin/env Rscript
# =====================================================================
#  R Synth on the uniquely identified SCM design of Track A module 52,
#  for tests/reference_parity/test_synth_unique_options_parity.py.
#
#  Reads tests/r_parity/data/52_scm_unique.csv (hash-locked Track A
#  data: the treated unit is exactly 0.5 donor0 + 0.3 donor1 + 0.2 donor2
#  in every pre-period; the effect is 2 in every post-period).
#
#  Writes synth_unique_R.json with two blocks.
#
#  nested -- Synth's own nested V search on seven special predictors
#    (five four-period means and two single periods). The treated unit
#    is a convex combination of the donors in every predictor, so the
#    zero-loss donor weights are unique whatever V is: the true weights,
#    and a gap of exactly 2.
#
#  placebo_equal_v -- the in-space placebo loop sp.synth runs with equal
#    predictor weights and no predictors: each donor in turn is the
#    treated unit, the other units (the actual treated unit included,
#    as sp.synth and Stata synth2 do) are its donors, and the weights
#    are simplex least squares on the twenty pre-period outcomes. Synth
#    standardises each predictor by its standard deviation across units;
#    custom.v = that variance undoes the standardisation, so its inner
#    QP is the unweighted problem. Each placebo optimum is unique (the
#    Python test checks the KKT conditions and that no zero-loss-change
#    direction is feasible). Reports every placebo's average post-period gap and their
#    population standard deviation, which sp.synth reports as `se`.
#
#  ipop is tightened (Sigf 20, Margin 1e-12, as in module 52) wherever it
#  solves; see IPOP below.
#  Environment: R 4.5.2 / Synth 1.1.10
# =====================================================================
suppressPackageStartupMessages({library(Synth); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

df <- read.csv(file.path(OUT, "..", "..", "r_parity", "data", "52_scm_unique.csv"))
names_all <- c(paste0("donor", 0:4), "treated")
df$id <- match(df$region, names_all)
PRE <- 0:19
POST <- 20:29

#: ipop settings, tightest first. A placebo whose optimum is a vertex of
#: the simplex (all weight on one unit) makes ipop's KKT system singular
#: at the tight settings; the first setting that solves is recorded with
#: the placebo, and the Python test holds each placebo to the precision
#: its setting attains.
IPOP <- list(c(20, 1e-12), c(12, 1e-10), c(10, 1e-8), c(7, 5e-4))

fit_one <- function(tr, ctrl, sp_list, custom.v = NULL) {
  dp <- dataprep(
    foo = df, predictors = NULL, dependent = "y",
    unit.variable = "id", time.variable = "year",
    treatment.identifier = tr, controls.identifier = ctrl,
    special.predictors = sp_list,
    time.predictors.prior = PRE, time.optimize.ssr = PRE,
    time.plot = 0:29, unit.names.variable = "region"
  )
  if (identical(custom.v, "unscaled")) {
    custom.v <- apply(cbind(dp$X0, dp$X1), 1, var)
  }
  out <- NULL
  for (setting in IPOP) {
    out <- tryCatch(
      synth(dp, verbose = FALSE, custom.v = custom.v, Sigf.ipop = setting[1],
            Margin.ipop = setting[2], Bound.ipop = 10),
      error = function(e) NULL)
    if (!is.null(out)) break
  }
  stopifnot(!is.null(out))
  path <- as.numeric(dp$Y0plot %*% out$solution.w)
  gap <- as.numeric(dp$Y1plot) - path
  list(w = setNames(as.numeric(out$solution.w), names_all[ctrl]),
       att = mean(gap[POST + 1]),
       pre_rmspe = sqrt(mean(gap[PRE + 1]^2)),
       sigf_ipop = setting[1], margin_ipop = setting[2])
}

tr <- 6L
ctrl <- 1:5
spec <- c(lapply(seq(0, 16, by = 4), function(a) list("y", a:(a + 3), "mean")),
          list(list("y", 19, "mean"), list("y", 3, "mean")))
set.seed(52)
nested <- fit_one(tr, ctrl, spec)

per_period <- lapply(PRE, function(p) list("y", p, "mean"))
placebos <- list()
for (d in 1:5) {
  r <- fit_one(d, setdiff(1:6, d), per_period, custom.v = "unscaled")
  placebos[[names_all[d]]] <- list(att = r$att, weights = as.list(r$w),
                                   sigf_ipop = r$sigf_ipop,
                                   margin_ipop = r$margin_ipop)
}
atts <- sapply(placebos, function(p) p$att)

out <- list(
  nested = list(weights = as.list(nested$w), att = nested$att,
                pre_rmspe = nested$pre_rmspe, sigf_ipop = nested$sigf_ipop,
                margin_ipop = nested$margin_ipop),
  placebo_equal_v = list(placebos = placebos,
                         sd_population = sqrt(mean((atts - mean(atts))^2))),
  `_meta` = list(
    generated_by = "_generate_synth_unique_R.R",
    data = "tests/r_parity/data/52_scm_unique.csv",
    r_version = R.version.string,
    synth_version = as.character(packageVersion("Synth"))
  )
)
write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
      file.path(OUT, "synth_unique_R.json"))
cat("[synth-unique] nested att", nested$att, " placebo sd",
    out$placebo_equal_v$sd_population, "\n")
