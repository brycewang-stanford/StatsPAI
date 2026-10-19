* Stata xtevent (SSC v3.1.0) trend() and proxy() / proxyiv() cases for
* sp.xtevent. xtevent is installed into the gitignored _ado_xtevent/.
*   python tests/reference_parity/_fixtures/_generate_xtevent_binary_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_xtevent_extensions_Stata.do
version 18
clear all
local ado "tests/reference_parity/_fixtures/_ado_xtevent"
capture mkdir "`ado'"
adopath + "`ado'"
capture which xtevent
if _rc {
    net set ado "`ado'"
    ssc install xtevent, replace
    net set ado PLUS
}
capture program drop _dump
program define _dump
    args fh key
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    file write `fh' `"  "`key'": {"N": "' (e(N)) `", "komit": ""' "`e(komit)'" `"", "names": ""' "`names'" `"", "b": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ","
        file write `fh' %24.16e (`b'[1, `j'])
    }
    file write `fh' `"], "se": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ","
        file write `fh' %24.16e (sqrt(`V'[`j', `j']))
    }
    file write `fh' "]}"
end
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/xtevent_extensions_Stata.json", write replace
file write `fh' "{" _n
import delimited using "tests/reference_parity/_fixtures/xtevent_continuous.csv", clear asdouble
xtset id t
qui xtevent y x, policyvar(z) window(3) trend(-3)
_dump `fh' cont_trend3_gmm
file write `fh' "," _n
qui xtevent y x, policyvar(z) window(3) trend(-2) vce(cluster cl)
_dump `fh' cont_trend2_gmm_cluster
file write `fh' "," _n
import delimited using "tests/reference_parity/_fixtures/xtevent_binary.csv", clear asdouble
xtset id t
qui xtevent y, policyvar(z) window(4) trend(-4, method(ols))
_dump `fh' bin_trend4_ols
file write `fh' "," _n
qui xtevent y, policyvar(z) window(4) trend(-3, method(ols)) vce(cluster cl)
_dump `fh' bin_trend3_ols_cluster
file write `fh' "," _n
qui xtevent y, policyvar(z) window(4) trend(-4)
_dump `fh' bin_trend4_gmm
file write `fh' "," _n
qui xtevent y, policyvar(z) window(4) proxy(x)
_dump `fh' bin_proxy_select
file write `fh' "," _n
qui xtevent y, policyvar(z) window(4) proxy(x) proxyiv(3)
_dump `fh' bin_proxy_lead3
file write `fh' "," _n
qui xtevent y, policyvar(z) window(4) proxy(x) proxyiv(1)
_dump `fh' bin_proxy_lead1
file write `fh' "," _n
qui xtevent y, policyvar(z) window(4) proxy(x) proxyiv(2 3) vce(cluster cl)
_dump `fh' bin_proxy_lead23_cluster
file write `fh' "," _n
qui xtevent y, policyvar(z) window(4) proxy(x x2) proxyiv(2 3)
_dump `fh' bin_proxy2_lead23
file write `fh' "," _n
qui xtevent y, policyvar(z) window(4) proxy(x) proxyiv(w)
_dump `fh' bin_proxy_external
file write `fh' _n "}" _n
file close `fh'
