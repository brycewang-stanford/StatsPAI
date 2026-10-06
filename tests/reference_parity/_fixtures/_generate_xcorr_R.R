# Reference values for tests/reference_parity/test_xcorr_parity.py
#
# Runs on the committed synthetic file xcorr.csv and writes xcorr_R.json.
# Requires: stats (base R), jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_xcorr_R.R
suppressMessages(library(jsonlite))
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "xcorr.csv"))
x = d$x; y = d$y; M = 10
out = list(versions = list(R = R.version.string), lags = -M:M)
cc = function(a, b) as.numeric(ccf(a, b, lag.max = M, plot = FALSE, na.action = na.omit)$acf)
out$raw = cc(x, y)
## Yule-Walker, order by AIC (the default of ar())
fx = ar(x); fy = ar(y)
out$yw_aic = list(order_x = fx$order, order_y = fy$order, phi_x = as.numeric(fx$ar),
                  phi_y = as.numeric(fy$ar), aic_x = as.numeric(fx$aic), aic_y = as.numeric(fy$aic),
                  ccf = cc(fx$resid, fy$resid))
## Yule-Walker, fixed orders (3, 2); residual pairs exist from t = 4
fx = ar(x, order.max = 3, aic = FALSE); fy = ar(y, order.max = 2, aic = FALSE)
keep = !is.na(fx$resid) & !is.na(fy$resid)
out$yw_fixed = list(phi_x = as.numeric(fx$ar), phi_y = as.numeric(fy$ar),
                    ccf = cc(fx$resid[keep], fy$resid[keep]))
## least squares, fixed orders (3, 2)
fx = ar(x, order.max = 3, aic = FALSE, method = "ols")
fy = ar(y, order.max = 2, aic = FALSE, method = "ols")
keep = !is.na(fx$resid) & !is.na(fy$resid)
rho = cc(fx$resid[keep], fy$resid[keep]); n = sum(keep)
out$ols_fixed = list(phi_x = as.numeric(fx$ar), phi_y = as.numeric(fy$ar), ccf = rho, n = n,
                     S = n * sum(rho^2), S_adj = n^2 * sum(rho^2 / (n - abs(-M:M))),
                     p = pchisq(n * sum(rho^2), 2 * M + 1, lower.tail = FALSE))
## Box-Jenkins: the AR(3) filter of x applied to both series
phi = as.numeric(fx$ar)
fl = function(z) as.numeric(stats::filter(z - mean(z), c(1, -phi), sides = 1))[-(1:3)]
out$bj = list(ccf = cc(fl(x), fl(y)))
write_json(out, file.path(here, "xcorr_R.json"), digits = NA, auto_unbox = TRUE)
