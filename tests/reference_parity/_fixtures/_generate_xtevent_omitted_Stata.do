* Stata xtevent (SSC v3.1.0) with a collinear (omitted) left endpoint, for
* sp.xtevent. xtevent is installed into the gitignored _ado_xtevent/.
*   python tests/reference_parity/_fixtures/_generate_xtevent_omitted_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_xtevent_omitted_Stata.do
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
    file write `fh' `"  "`key'": {"N": "' (e(N)) `", "names": ""' "`names'" `"", "b": ["'
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
file open `fh' using "tests/reference_parity/_fixtures/xtevent_omitted_Stata.json", write replace
file write `fh' "{" _n
import delimited using "tests/reference_parity/_fixtures/xtevent_omitted.csv", clear asdouble
xtset id t
qui xtevent y, policyvar(z) window(3)
_dump `fh' w3
file write `fh' "," _n
qui xtevent y, policyvar(z) window(3) vce(cluster id)
_dump `fh' w3_cluster
file write `fh' "," _n
qui xtevent y, policyvar(z) window(3) trend(-3, method(ols))
_dump `fh' w3_trend3_ols
file write `fh' _n "}" _n
file close `fh'
