* Which member of a collinear set Stata's -regress- omits, and the
* coefficients of the rest, for sp.regress(collinear='omit').
*   python tests/reference_parity/_fixtures/_generate_regress_collinear_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_regress_collinear_Stata.do
version 18
clear all
capture program drop _dump
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/regress_collinear_Stata.json", write replace
file write `fh' "{" _n

program define _dump
    args fh key
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    file write `fh' `"  "`key'": {"names": ""' "`names'" `"", "b": ["'
    local k = colsof(`b')
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ","
        file write `fh' %24.16e (`b'[1, `j'])
    }
    file write `fh' `"], "se": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ","
        file write `fh' %24.16e (sqrt(`V'[`j', `j']))
    }
    file write `fh' `"], "r2": "' %24.16e (e(r2)) `", "rmse": "' %24.16e (e(rmse)) "}"
end

import delimited using "tests/reference_parity/_fixtures/regress_collinear_cs.csv", clear asdouble
qui regress y x i.g d1
_dump `fh' fv_last
file write `fh' "," _n
qui regress y d1 i.g x
_dump `fh' fv_first
file write `fh' "," _n
qui regress y x d0 d1 d2
_dump `fh' plain_dummies
file write `fh' "," _n
qui regress y x w1 wsum
_dump `fh' plain_continuous
file write `fh' "," _n

import delimited using "tests/reference_parity/_fixtures/regress_collinear_es.csv", clear asdouble
qui regress y em7 em6 em5 em4 em3 em2 ep0 ep1 ep2 ep3 ep4 ep5 ep6 i.u i.t, vce(cluster u)
_dump `fh' event_study
file write `fh' _n "}" _n
file close `fh'
