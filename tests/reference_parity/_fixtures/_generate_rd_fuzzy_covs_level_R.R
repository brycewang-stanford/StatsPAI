#!/usr/bin/env Rscript
# =====================================================================
#  Fuzzy RD with covariates whose MEANS ARE NOT ZERO, against rdrobust.
#
#  Why: at nu = 0 the fuzzy bandwidth cascade (rdrobust_bw) forms its
#  delta-method weights from each side's LEVELS tau_Y / tau_T at the
#  cutoff, and those levels move with the covariates' location. Every
#  earlier fuzzy + covs fixture used standardised covariates
#  (rdsenate_fuzzy.csv scales cov1 / cov2), so a StatsPAI parser that
#  demeaned covs before the cascade matched R there and missed it here
#  by 4e-3 to 5e-3 in h (b and all fixed-h estimates exact).
#
#  Writes rd_fuzzy_covs_level.csv and rd_fuzzy_covs_level_R.json.
#  Environment: R 4.5 / rdrobust 4.0.0
# =====================================================================
suppressPackageStartupMessages({library(rdrobust); library(jsonlite)})
.a <- commandArgs(trailingOnly = FALSE)
.f <- sub("^--file=", "", .a[grep("^--file=", .a)])
OUT <- if (length(.f)) dirname(normalizePath(.f[1])) else "."

set.seed(20261019)
n <- 2000
x <- runif(n, -1, 1)
g <- sample(0:79, n, replace = TRUE)
z1 <- 3 + rnorm(n) + 0.3 * x            # mean 3, not 0
z2 <- rbinom(n, 1, 0.4)                 # mean 0.4
pt <- 0.2 + 0.5 * (x >= 0) + 0.03 * (z1 - 3)
d <- as.numeric(runif(n) < pmin(pmax(pt, 0), 1))
ge <- rnorm(80) * 0.3
y <- 0.5 + 1.5 * d + 0.8 * x - 0.4 * x^2 + 0.5 * z1 + 0.3 * z2 + ge[g + 1] + rnorm(n) * 0.7
df <- data.frame(y = y, x = round(x, 12), d = d, z1 = round(z1, 12), z2 = z2, g = g)
stopifnot(var(df$d[df$x < 0]) > 0, var(df$d[df$x >= 0]) > 0)
write.csv(df, file.path(OUT, "rd_fuzzy_covs_level.csv"), row.names = FALSE)
df <- read.csv(file.path(OUT, "rd_fuzzy_covs_level.csv"))   # the bytes Python reads

grab <- function(r) list(
  coef_conventional = r$coef[1], coef_robust = r$coef[3],
  se_conventional = r$se[1], se_robust = r$se[3],
  h_left = r$bws[1, 1], h_right = r$bws[1, 2],
  b_left = r$bws[2, 1], b_right = r$bws[2, 2]
)
Z <- cbind(df$z1, df$z2)
out <- list(
  fuzzy_covs_p1 = grab(rdrobust(df$y, df$x, fuzzy = df$d, covs = Z)),
  fuzzy_covs_p2 = grab(rdrobust(df$y, df$x, fuzzy = df$d, covs = Z, p = 2)),
  fuzzy_covs_cluster = grab(rdrobust(df$y, df$x, fuzzy = df$d, covs = Z, cluster = df$g)),
  fuzzy_covs_hc1 = grab(rdrobust(df$y, df$x, fuzzy = df$d, covs = Z, vce = "hc1")),
  fuzzy_covs_cerrd = grab(rdrobust(df$y, df$x, fuzzy = df$d, covs = Z, bwselect = "cerrd")),
  fuzzy_covs_msetwo = grab(rdrobust(df$y, df$x, fuzzy = df$d, covs = Z, bwselect = "msetwo"))
)
out$meta <- list(rdrobust = as.character(packageVersion("rdrobust")),
                 R = paste(R.version$major, R.version$minor, sep = "."))
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           file.path(OUT, "rd_fuzzy_covs_level_R.json"))
