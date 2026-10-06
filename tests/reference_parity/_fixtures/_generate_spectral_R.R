# Reference values for tests/reference_parity/test_spectral_parity.py
#
# Runs on the committed synthetic file spectral.csv and writes
# spectral_R.json. Requires: stats (base R), jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_spectral_R.R
suppressMessages(library(jsonlite))
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "spectral.csv"))
x = d$x; x127 = d$x[1:127]
pg = function(...) {
  s = spec.pgram(..., plot = FALSE)
  lim = s$df / qchisq(c(0.975, 0.025), s$df)
  list(freq = s$freq, spec = as.numeric(s$spec), df = s$df, bandwidth = s$bandwidth,
       lower = as.numeric(s$spec) * lim[1], upper = as.numeric(s$spec) * lim[2])
}
sar = function(...) {
  s = spec.ar(..., plot = FALSE)
  list(freq = s$freq, spec = as.numeric(s$spec), method = s$method)
}
out = list(versions = list(R = R.version.string))
out$default = pg(x)
out$textbook = pg(x, taper = 0, detrend = FALSE, demean = TRUE, fast = FALSE)
out$none = pg(x, taper = 0, detrend = FALSE, demean = FALSE, fast = FALSE)
out$spans35 = pg(x, spans = c(3, 5))
out$spans7_taper25 = pg(x, spans = 7, taper = 0.25, detrend = FALSE, demean = TRUE)
out$spans4 = pg(x, spans = 4)
out$pad1 = pg(x, spans = 5, pad = 1)
out$odd_fast = pg(x127, spans = c(3, 3))
out$odd_nofast = pg(x127, spans = c(3, 3), fast = FALSE)
out$ar_aic = sar(x)
out$ar_aic_order = ar(x, method = "yule-walker")$order
out$ar_aic_varpred = ar(x, method = "yule-walker")$var.pred
out$ar_fixed3 = sar(x, order = 3, n.freq = 101)
out$ar_max4 = sar(d$z, order.max = 4)
out$ar_max4_order = ar(d$z, order.max = 4, method = "yule-walker")$order
write_json(out, file.path(here, "spectral_R.json"), digits = NA, auto_unbox = TRUE)
