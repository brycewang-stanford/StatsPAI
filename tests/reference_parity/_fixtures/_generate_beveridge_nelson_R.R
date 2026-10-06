# Reference values for tests/reference_parity/test_beveridge_nelson_parity.py
#
# Runs on the committed synthetic file beveridge_nelson.csv and writes
# beveridge_nelson_R.json. No CRAN package computes the Beveridge-Nelson
# decomposition of an AR model, so the reference is the definition itself:
# the AR(p) for the first difference is fitted with lm(), the level is
# forecast H periods ahead by iterating the fitted equation, and
#   trend[t] = forecast of y[t + H] made at t  -  H * drift.
# Requires: stats (base R), jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_beveridge_nelson_R.R
suppressMessages(library(jsonlite))
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
y = read.csv(file.path(here, "beveridge_nelson.csv"))$y
dy = diff(y); n = length(dy); H = 400
bn = function(p) {
  E = embed(dy, p + 1)
  fit = lm(E[, 1] ~ E[, -1, drop = FALSE])
  b = as.numeric(coef(fit)); c0 = b[1]; phi = b[-1]
  drift = c0 / (1 - sum(phi))
  trend = rep(NA_real_, length(y))
  for (t in p:n) {                      # dy[t] is the change ending at y[t + 1]
    hist = dy[t:(t - p + 1)]; level = y[t + 1]
    for (h in 1:H) { nxt = c0 + sum(phi * hist); level = level + nxt; hist = c(nxt, hist)[1:p] }
    trend[t + 1] = level - H * drift
  }
  list(order = p, intercept = c0, phi = phi, drift = drift, trend = trend,
       sigma2 = summary(fit)$sigma^2, psi1 = 1 / (1 - sum(phi)),
       gamma0_unit = sum(ARMAtoMA(ar = phi, lag.max = 2000)^2) + 1)
}
out = list(versions = list(R = R.version.string), ar1 = bn(1), ar2 = bn(2), ar4 = bn(4))
write_json(out, file.path(here, "beveridge_nelson_R.json"), digits = NA, auto_unbox = TRUE, na = "null")
