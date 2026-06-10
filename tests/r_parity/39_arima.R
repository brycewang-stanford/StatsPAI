# StatsPAI ARIMA parity (R side) -- Module 39.
#
# Reads data/39_arima.csv and fits ARIMA(2,0,0) via stats::arima.
# Tolerance: rel < 1e-6 on AR coefficients and log-likelihood.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

MODULE <- "39_arima"

df <- read_csv_strict(MODULE)

fit <- stats::arima(
  df$y,
  order = c(2, 0, 0),
  include.mean = TRUE,
  method = "ML"
)

co <- coef(fit)
sigma2 <- fit$sigma2

rows <- list(
  parity_row(MODULE, "ar1",
             estimate = unname(co["ar1"]), n = nrow(df)),
  parity_row(MODULE, "ar2",
             estimate = unname(co["ar2"]), n = nrow(df)),
  parity_row(MODULE, "sigma2",
             estimate = sigma2, n = nrow(df)),
  parity_row(MODULE, "logLik",
             estimate = as.numeric(logLik(fit)), n = nrow(df))
)

write_results(MODULE, rows,
              extra = list(order = "(2,0,0)",
                           method = "ML",
                           package = "stats::arima"))
