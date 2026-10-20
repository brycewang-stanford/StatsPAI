#!/usr/bin/env Rscript
# Black-box fixture for statspai.spatial.gwr.bandwidth (clean-room, 2026-10-20).
# GWmodel 2.4-1 is GPL (>= 2): only its exported functions are run, on
# instrumented objectives and on committed data; no GWmodel source was read.
#   1. gold(fun, xL, xU, adapt.bw): the points the search evaluates and its
#      result, for objectives that record their argument.
#   2. bw.gwr on the simulated points of _generate_gwr_golden_search_data.py
#      and on the Georgia data of the spatial survey fixture, exponential
#      kernel included (the spatial survey fixture has bisquare / gaussian).
#   python tests/reference_parity/_fixtures/_generate_gwr_golden_search_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_gwr_golden_search_R.R
suppressPackageStartupMessages({library(GWmodel); library(sp); library(jsonlite)})
here <- "tests/reference_parity/_fixtures"
searches <- list()
record <- function(xL, xU, adaptive, kind, par) {
  calls <- c()
  g <- switch(kind,
              inc = function(x) x,
              dec = function(x) -x,
              quad = function(x) (x - par)^2,
              wav = function(x) (x - par)^2 + 30 * sin(x / 3),
              flat = function(x) round((x - par)^2, 3))
  fn <- function(x, ...) { calls <<- c(calls, x); g(x) }
  res <- gold(fn, xL, xU, adapt.bw = adaptive)
  searches[[length(searches) + 1]] <<- list(xL = xL, xU = xU, adaptive = adaptive,
                                            kind = kind, par = par, calls = calls, res = res)
}
for (r in list(c(0, 1000), c(20, 159), c(100, 1100), c(20, 300), c(5, 60))) {
  record(r[1], r[2], TRUE, "inc", 0); record(r[1], r[2], TRUE, "dec", 0)
  for (t in c(0.1, 0.3, 0.5, 0.77, 0.9)) record(r[1], r[2], TRUE, "quad", round(r[1] + t * (r[2] - r[1]), 1))
  record(r[1], r[2], TRUE, "wav", round(r[1] + 0.4 * (r[2] - r[1])))
}
for (r in list(c(0, 1), c(111.78, 558903.09), c(0.001, 3), c(20, 159))) {
  record(r[1], r[2], FALSE, "inc", 0); record(r[1], r[2], FALSE, "dec", 0)
  for (t in c(0.1, 0.37, 0.77)) record(r[1], r[2], FALSE, "quad", r[1] + t * (r[2] - r[1]))
  record(r[1], r[2], FALSE, "flat", r[1] + 0.4 * (r[2] - r[1]))
}
select <- function(pts, fm) {
  out <- list()
  for (k in c("bisquare", "gaussian", "exponential")) for (ad in c(TRUE, FALSE)) for (ap in c("AICc", "CV")) {
    invisible(capture.output(bw <- bw.gwr(fm, data = pts, approach = ap, kernel = k, adaptive = ad)))
    out[[length(out) + 1]] <- list(kernel = k, adaptive = ad, approach = ap, bw = bw)
  }
  out
}
simd <- read.csv(file.path(here, "gwr_golden_search_points.csv"))
sim_pts <- SpatialPointsDataFrame(cbind(simd$u, simd$v), simd)
geo <- read.csv(file.path(here, "spatial_survey_georgia.csv"))
geo_pts <- SpatialPointsDataFrame(cbind(geo$X, geo$Y), geo)
out <- list(
  meta = list(R = R.version.string, GWmodel = as.character(packageVersion("GWmodel"))),
  searches = searches,
  bw_gwr_sim = select(sim_pts, y ~ x1 + x2),
  bw_gwr_georgia = select(geo_pts, PctBach ~ PctRural + PctPov + PctBlack)
)
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE), file.path(here, "gwr_golden_search_R.json"))
cat("wrote gwr_golden_search_R.json\n")
