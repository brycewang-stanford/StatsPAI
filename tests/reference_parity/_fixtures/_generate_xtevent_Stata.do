* Stata xtevent (SSC v3.1.0; Freyaldenhoven, Hansen, Perez Perez, Shapiro)
* on a continuous policy that changes several times per unit, for
* sp.xtevent. xtevent is installed into the gitignored _ado_xtevent/.
*   python tests/reference_parity/_fixtures/_generate_xtevent_continuous_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_xtevent_Stata.do
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
import delimited using "tests/reference_parity/_fixtures/xtevent_continuous.csv", clear asdouble
xtset id t
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/xtevent_Stata.json", write replace
file write `fh' "{" _n
qui xtevent y x, policyvar(z) window(2)
_dump `fh' w2
file write `fh' "," _n
qui xtevent y x, policyvar(z) window(-3 4) vce(cluster cl)
_dump `fh' w34_cluster
file write `fh' "," _n
qui xtevent y x, policyvar(z) window(2) norm(-2) vce(robust)
_dump `fh' w2_norm2_robust
file write `fh' "," _n
qui xtevent y x, policyvar(z) window(2) reghdfe vce(cluster cl)
_dump `fh' w2_reghdfe_cluster
file write `fh' "," _n
qui xtevent y x, policyvar(z) window(2) diffavg vce(cluster cl)
qui lincom (_k_eq_p0 + _k_eq_p1 + _k_eq_p2 + _k_eq_p3)/4 - (_k_eq_m3 + _k_eq_m2)/3
file write `fh' `"  "w2_diffavg_cluster": {"estimate": "' %24.16e (r(estimate)) `", "se": "' %24.16e (r(se)) "}," _n
qui xtevent y x, policyvar(z) static vce(cluster cl)
_dump `fh' static_cluster
file write `fh' _n "}" _n
file close `fh'
