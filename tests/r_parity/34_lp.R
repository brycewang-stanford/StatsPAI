# StatsPAI local projections parity (R side) -- Module 34.
#
# Reads data/34_lp.csv and runs lpirfs::lp_lin. Tolerance:
# rel < 1e-2 on impulse-response coefficients.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(lpirfs)
})

MODULE <- "34_lp"
H_MAX <- 5L

df <- read_csv_strict(MODULE)

# lpirfs::lp_lin expects an endog_data data.frame; the shock variable
# is identified via the order of columns when shock_type = 0 (linear
# combination). We use shock_type = 1 (a Cholesky-orthogonalised
# shock) to match a single-equation regression of y on x.
endog_data <- df[, c("y", "x")]

# Run linear LP with no exogenous controls beyond the lag of y
# (controls=lags_lin). lp_lin uses the same OLS-by-horizon
# regressor structure as sp.local_projections when configured this
# way.
fit <- lpirfs::lp_lin(
  endog_data = endog_data,
  lags_endog_lin = 1L,
  trend = 0L,
  shock_type = 1L,
  confint = 1.96,
  hor = H_MAX + 1L
)

# fit$irf_lin_mean is a 3D array (response, horizon, shock). We
# want response = y (row 1), shock = x (column 2 in chol order).
irf_y_to_x <- fit$irf_lin_mean[1, , 2]
irf_y_to_x_up <- fit$irf_lin_up[1, , 2]
irf_y_to_x_low <- fit$irf_lin_low[1, , 2]
# se from confint=1.96: half-width / 1.96
se <- (irf_y_to_x_up - irf_y_to_x_low) / (2 * 1.96)

rows <- list()
for (h in 0:H_MAX) {
  rows[[length(rows) + 1L]] <- parity_row(
    module = MODULE,
    statistic = paste0("irf_h", h),
    estimate = irf_y_to_x[h + 1L],
    se = se[h + 1L],
    n = nrow(df))
}

# --- direct-OLS path -------------------------------------------------
# sp.local_projections(identification="direct") and the Stata side both
# fit the textbook Jorda regression y_{t+h} = a + b x_t + c y_lag + e
# horizon by horizon. lpirfs has no equivalent entry point, so without
# these rows the Stata artifact for this module joins nothing: compare.py
# builds the three-way table by walking the Python rows and requires an R
# counterpart, so `irf_direct_ols_h*` existed on the Python and Stata
# sides but was silently dropped from the comparison. This is a plain
# lm() on the same CSV bytes -- the same estimator all three sides claim
# to compute -- so it belongs in the table rather than in a footnote.
for (h in 0:H_MAX) {
  y_lead <- if (h == 0L) df$y else c(df$y[-seq_len(h)], rep(NA_real_, h))
  d_h <- data.frame(y_lead = y_lead, x = df$x, y_lag = df$y_lag)
  fit_h <- stats::lm(y_lead ~ x + y_lag, data = d_h)
  rows[[length(rows) + 1L]] <- parity_row(
    module = MODULE,
    statistic = paste0("irf_direct_ols_h", h),
    estimate = unname(stats::coef(fit_h)[["x"]]),
    se = unname(sqrt(diag(stats::vcov(fit_h)))[["x"]]),
    n = stats::nobs(fit_h))
}

write_results(MODULE, rows,
              extra = list(method = "lpirfs::lp_lin",
                           shock_type = "Cholesky",
                           direct_ols = "stats::lm(y_{t+h} ~ x + y_lag)",
                           direct_ols_se = paste(
                             "classical OLS; sp.local_projections applies",
                             "Newey-West on this path, so the Python side",
                             "emits se = NULL and no SE row is joined")))
