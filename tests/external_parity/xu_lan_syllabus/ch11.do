use wpi1, clear
tsset t
corrgram wpi, lags(12)
corrgram D.wpi, lags(12)
ac D.wpi, lags(12) generate(acd)
dfuller wpi
dfuller wpi, lags(4) trend
dfuller wpi, lags(4) trend regress
dfuller D.wpi, lags(3)
dfuller ln_wpi, noconstant lags(2)
dfuller D.ln_wpi, drift lags(2)
pperron wpi
pperron wpi, trend lags(4)
pperron D.wpi
dfgls wpi, maxlag(4)
kpss wpi, maxlag(4)
kpss D.wpi, notrend maxlag(4)
arima D.ln_wpi, ar(1)
arima D.ln_wpi, ar(1) ma(1)
arima ln_wpi, arima(1,1,1)
estat ic
arima ln_wpi, arima(2,1,0)
predict res, residuals
wntestq res
arima D.ln_wpi, ma(1 4)
regress D.ln_wpi L.D.ln_wpi
regress D.ln_wpi L(1/2).D.ln_wpi
estat archlm, lags(1)
estat archlm, lags(1/4)
arch D.ln_wpi, arch(1)
arch D.ln_wpi, arch(1) garch(1)
arch D.ln_wpi, ar(1) arch(1) garch(1)
arch D.ln_wpi, arch(1) garch(1) nolog vce(robust)
use lutkepohl2, clear
tsset qtr
varsoc dln_inv dln_inc dln_consump if qtr<=tq(1978q4), maxlag(4)
regress dln_consump L(1/2).dln_consump L(1/2).dln_inc
test L1.dln_inc L2.dln_inc
regress ln_consump ln_inc
predict uhat, residuals
dfuller uhat, noconstant lags(1)
regress D.ln_consump D.ln_inc L.uhat
egranger ln_consump ln_inc
egranger ln_consump ln_inc, lags(2)
egranger ln_consump ln_inc, ecm lags(1)
vecrank ln_inv ln_inc ln_consump, lags(2)
vec ln_inc ln_consump, lags(2)
