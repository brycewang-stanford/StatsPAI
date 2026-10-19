* Reference values for tests/reference_parity/test_ts_causal_parity.py
* Stata 18: lpirf (official); xtgcause (SSC, Lopez and Weber) installed into a
* private ado folder, e.g. sysdir set PLUS "<scratch>/ado" before running.
* Run from this folder:
*     do _generate_ts_causal_Stata.do
* Writes ts_causal_Stata.csv: spec, coefficient name, b, se, N.
clear all
set more off
set type double
tempname fh
file open `fh' using "ts_causal_Stata.csv", write replace
file write `fh' "spec,name,b,se,N" _n
import delimited using "ts_causal_series.csv", clear asdouble
tsset t
foreach spec in ols robust hac2 hacd dfk dfkrob dfkhac skip two {
    if "`spec'"=="ols" qui lpirf y, exog(g) lags(1/2) step(5)
    if "`spec'"=="robust" qui lpirf y, exog(g) lags(1/2) step(5) vce(robust)
    if "`spec'"=="hac2" qui lpirf y, exog(g) lags(1/2) step(5) vce(hac nwest 2)
    if "`spec'"=="hacd" qui lpirf y, exog(g) lags(1/2) step(5) vce(hac nwest)
    if "`spec'"=="dfk" qui lpirf y, exog(g) lags(1/2) step(5) dfk
    if "`spec'"=="dfkrob" qui lpirf y, exog(g) lags(1/2) step(5) vce(robust) dfk
    if "`spec'"=="dfkhac" qui lpirf y, exog(g) lags(1/2) step(5) vce(hac nwest 2) dfk
    if "`spec'"=="skip" qui lpirf y, exog(g) lags(1 3) step(5)
    if "`spec'"=="two" qui lpirf y w, exog(g z) lags(1/3) step(5) vce(robust)
    matrix b = e(b)
    matrix V = e(V)
    local names : colfullnames b
    local j = 0
    foreach nm of local names {
        local ++j
        file write `fh' "`spec',`nm'," %21.17g (b[1,`j']) "," %21.17g (sqrt(V[`j',`j'])) "," (e(N)) _n
    }
}
import delimited using "ts_causal_panel.csv", clear asdouble
xtset id time
foreach K in 1 2 {
    qui xtgcause y x, lags(`K')
    file write `fh' "xtgcause`K',wbar," %21.17g (r(wbar)) ",," _n
    file write `fh' "xtgcause`K',zbar," %21.17g (r(zbar)) "," %21.17g (r(zbar_pv)) "," _n
    file write `fh' "xtgcause`K',zbart," %21.17g (r(zbart)) "," %21.17g (r(zbart_pv)) "," _n
}
file close `fh'
