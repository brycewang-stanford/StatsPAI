# Reference values for tests/reference_parity/test_winsor_pctile_parity.py
# Run from this folder:  Rscript _generate_winsor_infinite_R.R
# farr::winsorize() and farr::truncate() (R quantile type 2, na.rm = TRUE)
# on vectors holding -Inf / Inf, written to winsor_infinite_R.json.
# Infinite and missing values are stored as the strings "Inf", "-Inf", "NA".
suppressMessages({library(farr); library(jsonlite)})
set.seed(20261124)
r <- c(rnorm(200), rep(Inf, 7), rep(-Inf, 4), rep(NA, 10))[sample(221)]
cases <- list(
  both_tails      = list(x = c(-Inf, 0:99, Inf), lo = 0.01, hi = 0.99),
  one_upper       = list(x = c(1:100, Inf), lo = 0.01, hi = 0.99),
  cut_is_infinite = list(x = c(1:10, Inf, Inf), lo = 0.1, hi = 0.9),
  mean_with_inf   = list(x = c(1:9, Inf), lo = 0.1, hi = 0.9),
  opposite_mean   = list(x = c(-Inf, Inf), lo = 0.5, hi = 0.5),
  lower_mean      = list(x = c(-Inf, -Inf, -Inf, 1, 2, NA, Inf), lo = 0.5, hi = 0.9),
  all_infinite    = list(x = c(Inf, Inf, Inf), lo = 0.01, hi = 0.99),
  lower_cut_inf   = list(x = c(1, Inf, Inf, Inf), lo = 0.5, hi = 1),
  random_01       = list(x = r, lo = 0.01, hi = 0.99),
  random_05       = list(x = r, lo = 0.05, hi = 0.95),
  random_025      = list(x = r, lo = 0.025, hi = 0.975)
)
out <- lapply(cases, function(k) {
  x <- as.numeric(k$x)
  modifyList(k, list(x = x,
            cuts = unname(quantile(x, c(k$lo, k$hi), type = 2, na.rm = TRUE)),
            w = winsorize(x, p_low = k$lo, p_high = k$hi),
            t = truncate(x, p_low = k$lo, p_high = k$hi)))
})
out$versions <- list(R = R.version.string, farr = as.character(packageVersion("farr")))
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, na = "string", pretty = FALSE),
           "winsor_infinite_R.json")
