* Stata 18 reference values quoted in
* tests/reference_parity/test_dynamic_modelling_parity.py (STATA dict).
* Reads the committed dynamic_modelling.csv; run from this directory.
clear all
import delimited "dynamic_modelling.csv", clear
tsset t

arima gdp, arima(1,1,0) noconstant vce(oim)
di %18.10f e(ll) "  " %14.10f _b[ARMA:L.ar] "  " %14.6f e(sigma)^2
arima gdp, arima(1,1,0) vce(oim)
di %18.10f e(ll) "  " %14.10f _b[ARMA:L.ar] "  " %14.8f _b[gdp:_cons]
arima gdp, arima(0,2,1) noconstant vce(oim)
di %18.10f e(ll) "  " %14.10f _b[ARMA:L.ma]
gen gdpk = gdp / 1000
arima gdpk, arima(1,1,0) noconstant vce(oim)
di %18.10f e(ll) "  " %14.10f _b[ARMA:L.ar]

regress y x
estat sbknown, break(101)
di %18.12f r(chi2) "  " %18.12g r(p)
estat sbcusum, ols
di %18.12f r(cusum)

var c1 c2 c3, lags(1/2) small
vargranger
matrix list r(gstats)
var c1 c2 c3, lags(1/2)
vargranger
matrix list r(gstats)
