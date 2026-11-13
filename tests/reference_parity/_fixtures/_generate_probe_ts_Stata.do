* Stata reference numbers for tests/reference_parity/test_probe_ts_parity.py
* (2026-11 macro / finance time-series probe). Run from
* tests/reference_parity/_fixtures after _generate_probe_ts_R.R has written
* the probe_ts_*.csv files:
*   stata-mp -b do _generate_probe_ts_Stata.do
* Writes probe_ts_Stata.csv (key, value). Stata 18 MP.
version 18
clear all
set more off
set type double

capture program drop pout
program define pout
    args key val
    file write fh "`key'," %21.15e (`val') _n
end
file open fh using "probe_ts_Stata.csv", write replace
file write fh "key,value" _n

* ---- ARIMA: residuals of a differenced model, OPG and OIM standard errors
import delimited "probe_ts_airpassengers.csv", clear
gen ly = ln(y)
gen tt = tm(1949m1) + _n - 1
tsset tt, monthly
arima ly, arima(0,1,1) sarima(0,1,1,12) noconstant
pout air.opg.ma1 _se[ARMA:L1.ma]
pout air.opg.sma1 _se[ARMA12:L1.ma]
predict double e, resid
quietly count if missing(e)
pout air.n_missing_resid r(N)
pout air.resid14 e[14]
pout air.resid15 e[15]
pout air.resid144 e[144]
wntestq e, lags(24)
pout air.q24 r(stat)
pout air.q24_p r(p)
arima ly, arima(0,1,1) sarima(0,1,1,12) noconstant vce(oim)
pout air.oim.ma1 _se[ARMA:L1.ma]
pout air.oim.sma1 _se[ARMA12:L1.ma]

import delimited "probe_ts_nile.csv", clear
gen tt = _n
tsset tt
arima y, arima(0,1,1) noconstant
pout nile.opg.ma1 _se[ARMA:L1.ma]
arima y, arima(0,1,1) noconstant vce(oim)
pout nile.oim.ma1 _se[ARMA:L1.ma]

import delimited "probe_ts_lynx.csv", clear
gen ly = log10(y)
gen tt = _n
tsset tt
arima ly, arima(2,0,0) vce(oim)
pout lynx.oim.ar1 _se[ARMA:L1.ar]
pout lynx.oim.ar2 _se[ARMA:L2.ar]
pout lynx.oim.const _se[ly:_cons]

* ---- VAR(2) on Canada: z p-values, fcast compute, varnorm
import delimited "probe_ts_canada.csv", clear
rename u U
gen tt = _n
tsset tt
var e prod rw U, lags(1/2)
foreach eq in e prod rw U {
    foreach v in e prod rw U {
        forvalues L = 1/2 {
            pout var.b.`eq'.L`L'.`v' _b[`eq':L`L'.`v']
            pout var.se.`eq'.L`L'.`v' _se[`eq':L`L'.`v']
            pout var.p.`eq'.L`L'.`v' 2*normal(-abs(_b[`eq':L`L'.`v']/_se[`eq':L`L'.`v']))
        }
    }
    pout var.b.`eq'._cons _b[`eq':_cons]
    pout var.p.`eq'._cons 2*normal(-abs(_b[`eq':_cons]/_se[`eq':_cons]))
}
varnorm
matrix J = r(jb)
matrix S = r(skewness)
matrix K = r(kurtosis)
local i = 0
foreach eq in e prod rw U ALL {
    local ++i
    pout varnorm.jb.`eq'.chi2 J[`i',1]
    pout varnorm.jb.`eq'.p J[`i',3]
    pout varnorm.skewness.`eq'.chi2 S[`i',2]
    pout varnorm.skewness.`eq'.p S[`i',4]
    pout varnorm.kurtosis.`eq'.chi2 K[`i',2]
    pout varnorm.kurtosis.`eq'.p K[`i',4]
    if `i' < 5 {
        pout varnorm.skewness.`eq'.value S[`i',1]
        pout varnorm.kurtosis.`eq'.value K[`i',1]
    }
}
fcast compute f_, step(8)
forvalues h = 1/8 {
    local r = 84 + `h'
    foreach v in e prod rw U {
        pout fcast.`h'.`v' f_`v'[`r']
        pout fcast.`h'.`v'_se f_`v'_se[`r']
        pout fcast.`h'.`v'_lower f_`v'_lb[`r']
        pout fcast.`h'.`v'_upper f_`v'_ub[`r']
    }
}

* ---- VEC on the Danish money data: fcast and irf use Sigma over T - d
foreach tr in rconstant constant {
    import delimited "probe_ts_denmark.csv", clear
    rename (lrm lry ibo ide) (LRM LRY IBO IDE)
    gen tt = _n
    tsset tt
    vec LRM LRY IBO IDE, lags(2) rank(1) trend(`tr')
    pout vec.`tr'.df_m e(df_m)
    fcast compute f_, step(4)
    forvalues h = 1/4 {
        local r = 55 + `h'
        foreach v in LRM LRY IBO IDE {
            pout vec.`tr'.fcast.`h'.`v' f_`v'[`r']
            pout vec.`tr'.fcast.`h'.`v'_se f_`v'_se[`r']
        }
    }
    irf create v1, set(_probe_ts_irf, replace) step(6)
    use _probe_ts_irf.irf, clear
    foreach i in LRM LRY IBO IDE {
        foreach j in LRM LRY IBO IDE {
            forvalues h = 0/6 {
                quietly summarize oirf if impulse=="`i'" & response=="`j'" & step==`h'
                pout vec.`tr'.oirf.`i'.`j'.`h' r(mean)
            }
        }
    }
}
capture erase _probe_ts_irf.irf
file close fh
