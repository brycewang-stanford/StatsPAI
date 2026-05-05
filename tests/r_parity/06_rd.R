# StatsPAI RD CCT bias-corrected parity (R side) -- Module 06.
#
# Reads data/06_rd.csv (the StatsPAI Lee 2008 senate replica) and
# runs rdrobust::rdrobust with package defaults (kernel = triangular,
# p = 1, q = p + 1 = 2, bwselect = mserd). Tolerance: rel < 1e-3.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(rdrobust)
})

MODULE <- "06_rd"

df <- read_csv_strict(MODULE)

make_row <- function(stat_name, fit, label) {
  parity_row(
    module    = MODULE,
    statistic = stat_name,
    estimate  = fit$coef[label, "Coeff"],
    se        = fit$se[label, "Std. Err."],
    ci_lo     = fit$ci[label, "CI Lower"],
    ci_hi     = fit$ci[label, "CI Upper"],
    n         = nrow(df)
  )
}

# Default mserd bandwidth selector.
fit <- rdrobust::rdrobust(y = df$voteshare_next, x = df$margin, c = 0.0)

# rdrobust returns coef and se as 3-row matrices: Conventional /
# Bias-Corrected / Robust. The "robust_est" row pairs the bias-
# corrected coef with the robust SE/CI, matching StatsPAI's
# convention.
rows <- list(
  parity_row(
    module    = MODULE,
    statistic = "default_conventional_est",
    estimate  = fit$coef["Conventional", "Coeff"],
    se        = fit$se["Conventional", "Std. Err."],
    ci_lo     = fit$ci["Conventional", "CI Lower"],
    ci_hi     = fit$ci["Conventional", "CI Upper"],
    n         = nrow(df)
  ),
  parity_row(
    module    = MODULE,
    statistic = "default_robust_est",
    estimate  = fit$coef["Bias-Corrected", "Coeff"],
    se        = fit$se["Robust",          "Std. Err."],
    ci_lo     = fit$ci["Robust",          "CI Lower"],
    ci_hi     = fit$ci["Robust",          "CI Upper"],
    n         = nrow(df)
  ),
  parity_row(
    module    = MODULE,
    statistic = "default_bandwidth_h",
    estimate  = fit$bws["h", "left"],
    n         = nrow(df)
  ),
  parity_row(
    module    = MODULE,
    statistic = "default_bandwidth_b",
    estimate  = fit$bws["b", "left"],
    n         = nrow(df)
  )
)

# Forced-bandwidth replicate at h = b = 0.042287 so the bandwidth-
# selector convention difference is isolated from the local-
# polynomial estimator math.
H_FORCED <- 0.042287
fit_forced <- rdrobust::rdrobust(
  y = df$voteshare_next, x = df$margin, c = 0.0,
  h = H_FORCED, b = H_FORCED
)
rows[[length(rows) + 1L]] <- parity_row(
  module    = MODULE,
  statistic = sprintf("forced_h%g_conventional_est", H_FORCED),
  estimate  = fit_forced$coef["Conventional", "Coeff"],
  se        = fit_forced$se["Conventional", "Std. Err."],
  ci_lo     = fit_forced$ci["Conventional", "CI Lower"],
  ci_hi     = fit_forced$ci["Conventional", "CI Upper"],
  n         = nrow(df)
)
rows[[length(rows) + 1L]] <- parity_row(
  module    = MODULE,
  statistic = sprintf("forced_h%g_robust_est", H_FORCED),
  estimate  = fit_forced$coef["Bias-Corrected", "Coeff"],
  se        = fit_forced$se["Robust",          "Std. Err."],
  ci_lo     = fit_forced$ci["Robust",          "CI Lower"],
  ci_hi     = fit_forced$ci["Robust",          "CI Upper"],
  n         = nrow(df)
)

write_results(MODULE, rows,
              extra = list(kernel = fit$kernel,
                           p = fit$p, q = fit$q,
                           bwselect = fit$bwselect))
