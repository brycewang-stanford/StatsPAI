#!/usr/bin/env Rscript
# =====================================================================
#  Reference values for the grid density estimator behind sp.rdplotdensity
#  (statspai.rd._lpdensity.lpdensity_numbers) vs R lpdensity 2.5.
#
#  lpdensity is GPL-2 and was used as a black box: only its outputs on
#  the two samples below are recorded (no source was read). Each cell is
#  lpdensity(data, grid, bw, p, q, v, kernel, scale, massPoints)$Estimate.
#
#  Writes lpdensity_cont.csv / lpdensity_tied.csv (the samples, so both
#  sides estimate on the same bytes) and lpdensity_R.json.
#
#  Environment: R 4.5 / lpdensity 2.5
# =====================================================================
suppressPackageStartupMessages({library(lpdensity); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

set.seed(20261020)
cont <- rnorm(300)
tied <- round(rnorm(300), 1)
write.csv(data.frame(x = cont), file.path(OUT, "lpdensity_cont.csv"), row.names = FALSE)
write.csv(data.frame(x = tied), file.path(OUT, "lpdensity_tied.csv"), row.names = FALSE)
cont <- read.csv(file.path(OUT, "lpdensity_cont.csv"))$x   # the committed bytes
tied <- read.csv(file.path(OUT, "lpdensity_tied.csv"))$x

grid <- c(min(cont), -1.5, -0.3, 0, 0.45, 1.6, max(cont))
grid_t <- c(-2, -0.5, 0, 0.3, 1.2)
cells <- list(
  list(id = "tri_p2",   data = "cont", grid = grid, bw = 0.8, p = 2, q = 3, v = 1, kernel = "triangular"),
  list(id = "uni_p2",   data = "cont", grid = grid, bw = 0.8, p = 2, q = 3, v = 1, kernel = "uniform"),
  list(id = "epa_p2",   data = "cont", grid = grid, bw = 0.8, p = 2, q = 3, v = 1, kernel = "epanechnikov"),
  list(id = "tri_p1",   data = "cont", grid = grid, bw = 0.6, p = 1, q = 2, v = 1, kernel = "triangular"),
  list(id = "tri_p3v2", data = "cont", grid = grid, bw = 1.1, p = 3, q = 4, v = 2, kernel = "triangular"),
  list(id = "tri_cdf",  data = "cont", grid = grid, bw = 0.8, p = 2, q = 3, v = 0, kernel = "triangular"),
  list(id = "tri_bwvec", data = "cont", grid = grid, bw = c(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1),
       p = 2, q = 3, v = 1, kernel = "triangular"),
  list(id = "tri_scale", data = "cont", grid = grid, bw = 0.8, p = 2, q = 3, v = 1,
       kernel = "triangular", scale = 0.37),
  list(id = "tied_mp",   data = "tied", grid = grid_t, bw = 0.7, p = 2, q = 3, v = 1,
       kernel = "triangular", massPoints = TRUE),
  list(id = "tied_nomp", data = "tied", grid = grid_t, bw = 0.7, p = 2, q = 3, v = 1,
       kernel = "triangular", massPoints = FALSE)
)

out <- list()
for (cl in cells) {
  x <- if (cl$data == "cont") cont else tied
  sc <- if (is.null(cl$scale)) 1 else cl$scale
  mp <- if (is.null(cl$massPoints)) TRUE else cl$massPoints
  est <- lpdensity(data = x, grid = cl$grid, bw = cl$bw, p = cl$p, q = cl$q, v = cl$v,
                   kernel = cl$kernel, scale = sc, massPoints = mp)$Estimate
  out[[cl$id]] <- list(
    args = list(data = cl$data, grid = cl$grid, bw = cl$bw, p = cl$p, q = cl$q, v = cl$v,
                kernel = cl$kernel, scale = sc, mass_points = mp),
    nh = unname(est[, "nh"]), f_p = unname(est[, "f_p"]), f_q = unname(est[, "f_q"]),
    se_p = unname(est[, "se_p"]), se_q = unname(est[, "se_q"]))
}
out[["_meta"]] <- list(
  generated_by = "_generate_lpdensity_R.R", r_version = R.version.string,
  lpdensity_version = as.character(packageVersion("lpdensity")),
  n_cont = length(cont), n_tied = length(tied))
write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE, na = "null"),
      file.path(OUT, "lpdensity_R.json"))
cat("[lpdensity] wrote", length(out) - 1, "cells\n")
