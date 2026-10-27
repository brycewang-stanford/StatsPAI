#!/usr/bin/env Rscript
# Reference numbers for sp.spatial_treatment from the method author's own
# code: Pollmann, spatialTreat-example (MIT licence), spatial_treat_estimator.R
# + sptreat_var.cpp at commit a73bca91e96a959bc19019f1300ba062aefe60bf,
# downloaded at run time. Rings (0,.05], (.05,.10], (.10,.15]; the exposure
# radius is 0.15 in design 1 and 0.10 in design 2.
#   python tests/reference_parity/_fixtures/_generate_spatial_treatment_data.py
#   Rscript tests/reference_parity/_fixtures/_generate_spatial_treatment_R.R
suppressMessages({library(data.table); library(Rcpp); library(jsonlite)})
here <- normalizePath("tests/reference_parity/_fixtures")
sha <- "a73bca91e96a959bc19019f1300ba062aefe60bf"
src <- file.path(tempdir(), "spatialTreat")
dir.create(src, showWarnings = FALSE)
base <- paste0("https://raw.githubusercontent.com/michaelpollmann/spatialTreat-example/", sha, "/")
for (f in c("spatial_treat_estimator.R", "sptreat_var.cpp"))
  download.file(paste0(base, f), file.path(src, f), quiet = TRUE)
# The .cpp declares an unused RcppArmadillo dependency (its include is
# commented out); dropping the declaration avoids linking Fortran runtimes.
cpp <- file.path(src, "sptreat_var.cpp")
writeLines(grep("Rcpp::depends", readLines(cpp), value = TRUE, invert = TRUE), cpp)
owd <- setwd(src); source("spatial_treat_estimator.R"); setwd(owd)

units <- read.csv(file.path(here, "spatial_treatment_units.csv"))
locs <- read.csv(file.path(here, "spatial_treatment_locations.csv"))
out <- list(meta = list(R = R.version.string, source_commit = sha,
                        data.table = as.character(packageVersion("data.table")),
                        Rcpp = as.character(packageVersion("Rcpp"))))
d0 <- c(0.15, 0.10)
for (dz in 1:2) {
  U <- units[units$design == dz, ]; L <- locs[locs$design == dz, ]
  D <- expand.grid(i = U$i, s = L$s)
  D$d <- sqrt((U$x[D$i + 1] - L$x[D$s + 1])^2 + (U$y[D$i + 1] - L$y[D$s + 1])^2)
  D <- D[D$d < 0.5, ]
  res <- sptreat_loop(dat_Y = data.frame(i = U$i, Y = U$Y), dat_D = D,
                      dat_S = data.frame(s = L$s, p = L$p, W = L$W),
                      dd = c(0.025, 0.075, 0.125), hh = 0.05, maxdeffect = d0[dz])
  out[[paste0("d", dz)]] <- list(tau = res$tau_hat, se = res$se, Vt = res$Vt,
                                 Vc = res$Vc, Vx = res$Vx, mut = res$mut,
                                 muc = res$muc)
}
write_json(out, file.path(here, "spatial_treatment_R.json"), digits = I(17),
           auto_unbox = TRUE)
