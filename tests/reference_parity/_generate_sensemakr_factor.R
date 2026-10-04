#!/usr/bin/env Rscript
# Reference values for sp.sensemakr with a factor control, benchmark
# multiples (kd = 1:3) and a grouped benchmark, from R sensemakr
# (Cinelli & Hazlett 2020 [@cinelli2020making]).
#
# Writes, under tests/reference_parity/_fixtures/:
#   sensemakr_factor.csv      simulated data: y, d, x1, x2, region (6 levels)
#   sensemakr_factor_R.json   sensemakr::sensemakr output, full precision
#
# Re-run only when the contract changes:
#   Rscript tests/reference_parity/_generate_sensemakr_factor.R

suppressMessages({
  library(sensemakr)
  library(jsonlite)
})

FIX <- "tests/reference_parity/_fixtures"
set.seed(20261004)
n <- 300
region <- sample(c("north", "south", "east", "west", "centre", "coast"), n, replace = TRUE)
shift <- c(north = 0.4, south = -0.3, east = 0.1, west = 0, centre = -0.5, coast = 0.6)[region]
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.4)
d <- 0.5 * x1 + 0.4 * x2 + shift + rnorm(n)
y <- 0.3 * d + 0.8 * x1 - 0.5 * x2 + 0.7 * shift + rnorm(n)
dat <- data.frame(y = y, d = d, x1 = x1, x2 = x2, region = region)
write.csv(dat, file.path(FIX, "sensemakr_factor.csv"), row.names = FALSE)

dat <- read.csv(file.path(FIX, "sensemakr_factor.csv"), stringsAsFactors = TRUE)
m <- lm(y ~ d + x1 + x2 + region, data = dat)
s <- sensemakr(m, treatment = "d", benchmark_covariates = "x1", kd = 1:3)
region_cols <- grep("^region", names(coef(m)), value = TRUE)
g <- ovb_bounds(m, treatment = "d", benchmark_covariates = list(region = region_cols), kd = c(1, 2))

out <- list(
  sensemakr_version = as.character(packageVersion("sensemakr")),
  stats = as.list(s$sensitivity_stats[1, c("estimate", "se", "t_statistic", "r2yd.x", "rv_q", "rv_qa", "dof")]),
  bounds_x1 = s$bounds,
  bounds_region = g
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, dataframe = "rows"),
           file.path(FIX, "sensemakr_factor_R.json"))
cat("wrote sensemakr_factor fixtures\n")
