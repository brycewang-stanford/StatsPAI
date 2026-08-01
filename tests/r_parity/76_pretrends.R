# StatsPAI pre-trends power parity (R side) -- Module 76.
#
# Mirrors the hand-crafted event study in 76_pretrends.py and runs Roth's
# `pretrends` package on it: pretrends() for the power / Bayes factor /
# likelihood ratio at a hypothesised linear trend, and slope_for_power()
# for the trend a pre-test would catch a given fraction of the time.
#
# The inputs are literals rather than a CSV because the estimator is a
# function of (betahat, sigma) alone.
#
# Tolerance: rel < 1e-3 -- mvtnorm::pmvnorm's Genz-Bretz integrator is
# randomised, so the R side itself is only reproducible to ~1e-4. See the
# Python module docstring.
#
# `pretrends` is not on CRAN; install with
#   remotes::install_github("jonathandroth/pretrends")

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages(library(pretrends))

MODULE <- "76_pretrends"

tVec <- c(-4, -3, -2, 0, 1, 2)
se <- c(0.050, 0.045, 0.040, 0.100, 0.110, 0.120)
beta <- c(0.012, -0.008, 0.021, 0.180, 0.240, 0.310)
rho <- 0.5

idx <- seq_along(se)
corr <- rho^abs(outer(idx, idx, "-"))
sigma <- corr * outer(se, se)

tag <- function(x) gsub("\\.", "p", format(x, trim = TRUE))

rows <- list()
for (slope in c(0.02, 0.05)) {
  r <- pretrends(
    betahat = beta, sigma = sigma, deltatrue = slope * (tVec + 1),
    tVec = tVec, referencePeriod = -1
  )
  vals <- c(
    power = r$df_power$Power,
    bayes_factor = r$df_power$Bayes.Factor,
    likelihood_ratio = r$df_power$Likelihood.Ratio
  )
  for (nm in names(vals)) {
    rows[[length(rows) + 1]] <- parity_row(
      module    = MODULE,
      statistic = sprintf("%s_slope_%s", nm, tag(slope)),
      estimate  = unname(vals[[nm]]),
      se        = NA_real_,
      n         = 1000L
    )
  }
}

for (target in c(0.5, 0.8)) {
  sl <- slope_for_power(
    sigma = sigma, targetPower = target, tVec = tVec, referencePeriod = -1
  )
  rows[[length(rows) + 1]] <- parity_row(
    module    = MODULE,
    statistic = sprintf("slope_for_power_%s", tag(target)),
    estimate  = as.numeric(sl),
    se        = NA_real_,
    n         = 1000L
  )
}

write_results(MODULE, rows, extra = list(
  reference = "pretrends::pretrends / pretrends::slope_for_power",
  pretrends = as.character(utils::packageVersion("pretrends"))
))
