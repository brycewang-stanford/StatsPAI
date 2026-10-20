# Reference conventional SEs for the nearest-neighbour RD variance
# (statspai.rd._nn_variance.nn_residuals) from R rdrobust 4.0.0
# (GPL-3; used as a black box, outputs only); checked in
# tests/test_rd_nn_residuals.py.
# Run from this folder:  Rscript _generate_rd_nn_residuals_R.R
#
# Each case is a sharp RD at c = 0 with a manual bandwidth h = b,
# p = 1, triangular kernel, vce = "nn".  Designs cover a continuous score,
# a score rounded to one decimal (mass points whose gaps are not exactly
# equal in floating point), a score on a dyadic grid (exactly equidistant
# mass points), and few distinct values.  "wide" cases use an h that
# covers every observation; "narrow" cases use an h inside the support,
# which tells whether the neighbours are searched on the whole side or
# only inside the window.  The sparse designs (about one observation per
# distinct value) make the J-th neighbour distance land on ties, which
# exercises the "all equidistant neighbours" rule; "ongrid" puts h on a
# mass point (with h = 29/64 the unit at -28/64 has two equidistant
# neighbours, one of them at |x| = h, so the case pins whether
# observations with |x| = h, which get zero kernel weight, still serve as
# neighbours).
suppressMessages({library(rdrobust); library(jsonlite)})

make_case <- function(name, x, y, h, J) {
  r <- rdrobust(y, x, c = 0, h = h, b = h, p = 1, kernel = "triangular",
                vce = "nn", nnmatch = J)
  list(name = name, x = x, y = y, h = h, nnmatch = J,
       se_conventional = r$se[1],
       V_cl_l = r$V_cl_l[1, 1], V_cl_r = r$V_cl_r[1, 1],
       N_h = as.integer(r$N_h), bws_h = as.numeric(r$bws[1, ]))
}

set.seed(20261020)
designs <- list()
n <- 160
x <- runif(n, -1, 1)
designs$continuous <- list(x = x, y = 1 + 0.5 * x + 0.8 * (x >= 0) + rnorm(n))
x <- round(runif(240, -1, 1), 1)
designs$rounded1 <- list(x = x, y = sin(2 * x) + 0.6 * (x >= 0) + rnorm(240))
x <- sample(seq(-1, 1, by = 0.125), 200, replace = TRUE)
designs$dyadic <- list(x = x, y = x^2 + 0.4 * (x >= 0) + rnorm(200))
x <- sample(c(-0.75, -0.5, -0.25, 0, 0.25, 0.5), 90, replace = TRUE)
designs$few_values <- list(x = x, y = 0.3 * x + (x >= 0) + rnorm(90))
x <- sample(seq(-1, 1, by = 1 / 64), 120, replace = TRUE)
designs$dyadic_sparse <- list(x = x, y = cos(x) + 0.5 * (x >= 0) + rnorm(120))
x <- round(runif(150, -1, 1), 2)
designs$rounded2_sparse <- list(x = x, y = x + 0.5 * (x >= 0) + rnorm(150))

cases <- list()
for (nm in names(designs)) {
  d <- designs[[nm]]
  for (J in c(1, 3, 5)) {
    cases[[length(cases) + 1]] <- make_case(paste0(nm, "_wide_J", J),
                                            d$x, d$y, 2.0, J)
    if (nm != "few_values")
      cases[[length(cases) + 1]] <- make_case(paste0(nm, "_narrow_J", J),
                                              d$x, d$y, 0.45, J)
    if (nm == "dyadic_sparse") {  # h on the grid: observations at |x| = h
      cases[[length(cases) + 1]] <- make_case(paste0(nm, "_ongrid_J", J),
                                              d$x, d$y, 0.5, J)
      cases[[length(cases) + 1]] <- make_case(paste0(nm, "_ongrid29_J", J),
                                              d$x, d$y, 29 / 64, J)
    }
  }
}
out <- list(cases = cases,
            versions = list(R = R.version.string,
                            rdrobust = as.character(packageVersion("rdrobust"))))
write_json(out, "rd_nn_residuals_R.json", digits = I(17), auto_unbox = TRUE)
