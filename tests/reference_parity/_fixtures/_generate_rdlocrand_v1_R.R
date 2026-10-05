# Reference values from rdlocrand 1.0 (CRAN, 2022-06-22), the last release
# before two window-construction regressions:
#
#   * from 1.1 (2025-05-23) the default first window of rdwinselect holds
#     obsmin - 1 observations below the cutoff, and wmasspoints pairs the
#     k-th support point above the cutoff with the (k-1)-th below it;
#   * from 2.0 (2026-05-14) the randomization p-value of the
#     Kolmogorov-Smirnov statistic is 1 on a binary variable.
#
# Version 1.0 does what the help page and Cattaneo, Idrobo & Titiunik (2024)
# describe, and is what sp.rdwinselect is held to here.
#
# To regenerate, install the archived release into a library of its own:
#   curl -O https://cran.r-project.org/src/contrib/Archive/rdlocrand/rdlocrand_1.0.tar.gz
#   mkdir rlib_1.0 && R CMD INSTALL -l rlib_1.0 rdlocrand_1.0.tar.gz
#   RDLOCRAND_V1_LIB=rlib_1.0 Rscript _generate_rdlocrand_v1_R.R
lib <- Sys.getenv("RDLOCRAND_V1_LIB")
stopifnot(nzchar(lib))
.libPaths(c(lib, .libPaths()))
library(rdlocrand); library(jsonlite)
# 3.0 (rdpackages/rdlocrand, 2026-10-04) restores the 1.0 behaviour. Run
# this script with a library that holds 3.0 to write rdlocrand_v3_R.json,
# which the tests hold to the same targets.
ver <- as.character(packageVersion("rdlocrand"))
stopifnot(ver %in% c("1.0", "3.0"))
outfile <- if (ver == "1.0") "rdlocrand_v1_R.json" else "rdlocrand_v3_R.json"
q <- function(e) { sink("/dev/null"); on.exit(sink()); suppressWarnings(e) }
d <- read.csv("rdsenate.csv")
X <- d$margin
covs <- cbind(d$class, d$termshouse, d$termssenate)
win <- function(r) list(
  w_left = unname(r$results[, "w_left"]), w_right = unname(r$results[, "w_right"]),
  Nl = unname(r$results[, "Obs<c"]), Nr = unname(r$results[, "Obs>=c"]),
  binom = unname(r$results[, "Bi.test"]), p_value = unname(r$results[, "p-value"]),
  variable = unname(r$results[, "Variable"]))
out <- list()
out$default  <- win(q(rdwinselect(X, covs, approx = TRUE)))
out$wobs2    <- win(q(rdwinselect(X, covs, wobs = 2, approx = TRUE)))
out$obsmin5  <- win(q(rdwinselect(X, covs, obsmin = 5, wobs = 3, nwindows = 6, approx = TRUE)))
Rm <- c(rep(-(1:10), each = 3), rep((0:9) + 0.5, each = 2))
set.seed(3); Xm <- matrix(rnorm(50), 50, 1)
out$masspoints_toy <- win(q(rdwinselect(Rm, Xm, wmasspoints = TRUE, nwindows = 4, approx = TRUE)))
out$masspoints_toy$covariate <- as.numeric(Xm)
# Kolmogorov-Smirnov on a binary covariate: mean randomization p-value
z <- d$dopen
ks <- sapply(1:40, function(s)
  as.numeric(q(rdrandinf(z, X, wl = -2, wr = 2, statistic = "ksmirnov", seed = s))$p.value))
o <- q(rdrandinf(z, X, wl = -2, wr = 2, statistic = "ksmirnov"))
out$ks_binary <- list(seedmean = mean(ks), obs_stat = unname(o$obs.stat),
                      asy_pvalue = unname(o$asy.pvalue))
out[["_meta"]] <- list(rdlocrand_version = as.character(packageVersion("rdlocrand")))
write_json(out, outfile, auto_unbox = TRUE, digits = 15, pretty = TRUE,
           na = "null")
cat("wrote", outfile, "\n")
