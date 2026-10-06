# Reference values for the ARMA tests in
# tests/reference_parity/test_beveridge_nelson_parity.py
#
# No R package computes the Beveridge-Nelson decomposition of an ARMA model,
# so the reference is its definition evaluated with stats::arima. For each
# order (p, q):
#   1. fit ARMA(p, q) with a mean to the first difference by exact ML
#      (stats::arima, method = "ML", tight optimiser tolerance);
#   2. for every date t, hand the first t differences to arima() with all
#      coefficients fixed at the estimates and forecast H = 1500 periods ahead
#      (KalmanForecast: the exact conditional expectation given the data to
#      t), and set trend[t] = y[t] + sum(forecast - mean).
# Writes beveridge_nelson_arma_R.json. Requires: jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_beveridge_nelson_arma_R.R
suppressMessages(library(jsonlite))
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
y = read.csv(file.path(here, "beveridge_nelson_arma.csv"))$y
dy = diff(y); n = length(dy); H = 1500
run = function(p, q) {
  fit = arima(dy, order = c(p, 0, q), method = "ML",
              optim.control = list(reltol = 1e-14, maxit = 5000))
  co = coef(fit); mu = unname(co["intercept"])
  phi = if (p > 0) unname(co[1:p]) else numeric(0)
  theta = if (q > 0) unname(co[(p + 1):(p + q)]) else numeric(0)
  trend = rep(NA_real_, n + 1)
  first = max(p, q) + 2          # arima() needs a few observations
  for (t in first:n) {
    ft = arima(dy[1:t], order = c(p, 0, q), fixed = co, transform.pars = FALSE,
               method = "ML")
    pr = predict(ft, n.ahead = H)$pred
    trend[t + 1] = y[t + 1] + sum(pr - mu)
  }
  psi = ARMAtoMA(ar = phi, ma = theta, lag.max = 5000)
  list(p = p, q = q, mean = mu, phi = as.list(phi), theta = as.list(theta),
       sigma2 = fit$sigma2, loglik = fit$loglik, trend = trend, first = first,
       psi1 = (1 + sum(theta)) / (1 - sum(phi)), gamma0_unit = 1 + sum(psi^2),
       residuals = as.numeric(residuals(fit)))
}
out = list(versions = list(R = R.version.string))
out$ma1 = run(0, 1)
out$arma11 = run(1, 1)
out$arma21 = run(2, 1)
out$arma12 = run(1, 2)
out$ar2 = run(2, 0)
write_json(out, file.path(here, "beveridge_nelson_arma_R.json"), digits = NA, auto_unbox = TRUE, na = "null")
