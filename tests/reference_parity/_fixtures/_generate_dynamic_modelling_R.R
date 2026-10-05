# Reference values for tests/reference_parity/test_dynamic_modelling_parity.py
#
# Runs on the committed synthetic file dynamic_modelling.csv and writes
# dynamic_modelling_R.json next to it.
#
# Requires: forecast, strucchange, urca, vars, jsonlite.
#   Rscript tests/reference_parity/_fixtures/_generate_dynamic_modelling_R.R
suppressMessages({
  library(forecast); library(strucchange); library(urca); library(vars); library(jsonlite)
})
here = dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
d = read.csv(file.path(here, "dynamic_modelling.csv"))
out = list(versions = list(R = R.version.string,
  forecast = as.character(packageVersion("forecast")),
  strucchange = as.character(packageVersion("strucchange")),
  urca = as.character(packageVersion("urca")), vars = as.character(packageVersion("vars"))))

## ARIMA by exact maximum likelihood, tight optimiser
tight = list(reltol = 1e-14, maxit = 2000)
fit1 = function(y, order, drift = FALSE, mean = TRUE, seasonal = c(0, 0, 0), period = 1) {
  y = ts(y, frequency = period)
  f = Arima(y, order = order, seasonal = seasonal, include.drift = drift,
            include.mean = mean, method = "ML", optim.control = tight)
  list(coef = as.list(coef(f)), loglik = f$loglik, aic = f$aic, bic = f$bic,
       aicc = f$aicc, sigma2_ml = sum(residuals(f)^2, na.rm = TRUE) / f$nobs,
       forecast = unname(as.numeric(forecast(f, h = 4)$mean)))
}
out$arima = list(
  gdp_110 = fit1(d$gdp, c(1, 1, 0)),
  gdp_011 = fit1(d$gdp, c(0, 1, 1)),
  gdp_021 = fit1(d$gdp, c(0, 2, 1)),
  gdp_111 = fit1(d$gdp, c(1, 1, 1)),
  gdp_110_drift = fit1(d$gdp, c(1, 1, 0), drift = TRUE),
  gdp_010_drift = fit1(d$gdp, c(0, 1, 0), drift = TRUE),
  gdp_011_011_4 = fit1(d$gdp, c(0, 1, 1), seasonal = c(0, 1, 1), period = 4),
  x_100 = fit1(d$x, c(1, 0, 0)),
  # the same series in other units: the estimates must not move
  gdp_110_thousandth = fit1(d$gdp / 1000, c(1, 1, 0)),
  gdp_110_thousandfold = fit1(d$gdp * 1000, c(1, 1, 0)))

## exhaustive automatic selection
auto1 = function(y) {
  f = auto.arima(y, seasonal = FALSE, stepwise = FALSE, approximation = FALSE,
                 max.p = 3, max.q = 3, max.order = 6, ic = "aicc")
  list(order = unname(f$arma[c(1, 6, 2)]), terms = names(coef(f)), aicc = f$aicc)
}
out$auto = list(gdp = auto1(d$gdp), x = auto1(d$x), y = auto1(d$y), c2 = auto1(d$c2),
                ret = auto1(d$ret), dgdp = auto1(diff(d$gdp)))

## Chow test at a known date, OLS-CUSUM
ch = sctest(y ~ x, data = d, type = "Chow", point = 100)
ch2 = sctest(y ~ x, data = d, type = "Chow", point = 60)
oc = sctest(efp(y ~ x, data = d, type = "OLS-CUSUM"))
ocg = sctest(efp(gdp ~ t, data = d, type = "OLS-CUSUM"))
out$stability = list(chow100 = list(stat = unname(ch$statistic), p = ch$p.value),
  chow60 = list(stat = unname(ch2$statistic), p = ch2$p.value),
  ols_cusum = list(stat = unname(oc$statistic), p = oc$p.value),
  ols_cusum_trend = list(stat = unname(ocg$statistic), p = ocg$p.value))

## portmanteau tests on ARMA(1,1) residuals of x
r = residuals(arima(d$x, order = c(1, 0, 1), method = "ML", optim.control = tight))
lb = Box.test(r, lag = 10, type = "Ljung-Box", fitdf = 2)
bp = Box.test(r, lag = 10, type = "Box-Pierce", fitdf = 2)
lb0 = Box.test(d$ret, lag = 8, type = "Ljung-Box")
bp0 = Box.test(d$ret, lag = 8, type = "Box-Pierce")
out$portmanteau = list(resid = unname(as.numeric(r)),
  lb = list(stat = unname(lb$statistic), p = lb$p.value),
  bp = list(stat = unname(bp$statistic), p = bp$p.value),
  lb_raw = list(stat = unname(lb0$statistic), p = lb0$p.value),
  bp_raw = list(stat = unname(bp0$statistic), p = bp0$p.value))

## VECM: forecasts and impulse responses through the implied VAR in levels
Y = d[, c("c1", "c2", "c3")]
vec1 = function(ecdet) {
  v = vec2var(ca.jo(Y, type = "trace", ecdet = ecdet, K = 3, spec = "transitory"), r = 1)
  fc = predict(v, n.ahead = 6)$fcst
  ir = irf(v, n.ahead = 8, ortho = TRUE, boot = FALSE)$irf
  iu = irf(v, n.ahead = 8, ortho = FALSE, boot = FALSE)$irf
  list(forecast = lapply(fc, function(m) unname(m[, "fcst"])),
       forecast_lower = lapply(fc, function(m) unname(m[, "lower"])),
       irf_orth = lapply(ir, function(m) unname(as.matrix(m))),
       irf_unit = lapply(iu, function(m) unname(as.matrix(m))))
}
out$vec = list(c = vec1("none"), rc = vec1("const"))

write_json(out, file.path(here, "dynamic_modelling_R.json"), digits = NA, auto_unbox = TRUE, pretty = TRUE)
