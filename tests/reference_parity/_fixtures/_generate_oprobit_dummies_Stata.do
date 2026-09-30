* Ordered probit / logit with 119 group dummies and an unscaled regressor
* (x ~ 1000 +- 50): coefficients, cutpoints and their SEs under the default,
* robust and cluster VCEs, for the analytic Newton engine.
*   python tests/reference_parity/_fixtures/_generate_oprobit_dummies_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_oprobit_dummies_Stata.do
version 18
clear all
capture program drop _dump
program define _dump
    args fh key
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local k = colsof(`b')
    file write `fh' `"  "`key'": {"ll": "' %24.16e (e(ll)) `", "b": ["'
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
import delimited using "tests/reference_parity/_fixtures/oprobit_dummies.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/oprobit_dummies_Stata.json", write replace
file write `fh' "{" _n
qui oprobit y x z i.g, nrtolerance(1e-13) tolerance(1e-12) ltolerance(1e-14)
_dump `fh' oprobit
file write `fh' "," _n
qui oprobit y x z i.g, vce(robust) nrtolerance(1e-13) tolerance(1e-12) ltolerance(1e-14)
_dump `fh' oprobit_robust
file write `fh' "," _n
qui oprobit y x z i.g, vce(cluster cl) nrtolerance(1e-13) tolerance(1e-12) ltolerance(1e-14)
_dump `fh' oprobit_cluster
file write `fh' "," _n
qui ologit y x z i.g, vce(cluster cl) nrtolerance(1e-13) tolerance(1e-12) ltolerance(1e-14)
_dump `fh' ologit_cluster
file write `fh' _n "}" _n
file close `fh'
