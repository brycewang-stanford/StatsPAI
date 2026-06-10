* tests/stata_parity/39_arima.do
*
* Module 39: ARIMA(2,0,0).
*   StatsPAI:  sp.arima (statsmodels innovations MLE)
*   R:         stats::arima (ML)
*   Stata:     arima ..., ar(1/2), tight MLE tolerances
*
* Tolerance: rel < 1e-6 on AR coefficients and log-likelihood.

version 18
clear all

do _common.do
stata_parity_init, module(39_arima)
stata_parity_open, module(39_arima)

import delimited "${STATA_PARITY_DATA}/39_arima.csv", clear case(preserve)

gen t = _n
tsset t

local n = _N

arima y, ar(1/2) nolog iterate(1000) tolerance(1e-14) ///
    ltolerance(1e-14) nrtolerance(1e-14)

local ar1 = _b[ARMA:L.ar]
local ar2 = _b[ARMA:L2.ar]
* sigma is reported as the sigma equation coefficient; sigma2 = sigma^2
local sigma_hat = _b[sigma:_cons]
local sigma2 = `sigma_hat' * `sigma_hat'
local ll = e(ll)

stata_parity_row, stat(ar1)    est(`ar1') nob(`n')
stata_parity_row, stat(ar2)    est(`ar2') nob(`n')
stata_parity_row, stat(sigma2) est(`sigma2') nob(`n')
stata_parity_row, stat(logLik) est(`ll') nob(`n')

stata_parity_extra, key(order) val("(2,0,0)")
stata_parity_extra, key(method) val("exact MLE")
stata_parity_extra, key(stata_command) val("arima y, ar(1/2), tight MLE tolerances")

stata_parity_close, module(39_arima)
