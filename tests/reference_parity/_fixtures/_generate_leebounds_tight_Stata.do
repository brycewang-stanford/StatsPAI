* Stata leebounds (Tauchmann, SSC v1.5), tight(): tightened bounds and their
* analytic variances, for sp.lee_bounds(covariates=..., trimming='leebounds').
* leebounds is installed into the gitignored _ado_leebounds/, not PLUS.
*   python tests/reference_parity/_fixtures/_generate_leebounds_tight_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_leebounds_tight_Stata.do
version 18
clear all
local ado "tests/reference_parity/_fixtures/_ado_leebounds"
capture mkdir "`ado'"
adopath + "`ado'"
capture which leebounds
if _rc {
    net set ado "`ado'"
    ssc install leebounds, replace
    net set ado PLUS
}
import delimited using "tests/reference_parity/_fixtures/leebounds_tight.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/leebounds_tight_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach cov in "x" "x z" {
    qui leebounds y d, select(s) tight(`cov')
    matrix b = e(b)
    matrix V = e(V)
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  "`cov'": {"lower": "' %24.16e (b[1,1]) `", "upper": "' %24.16e (b[1,2])
    file write `fh' `", "var_lower": "' %24.16e (V[1,1]) `", "var_upper": "' %24.16e (V[2,2])
    file write `fh' `", "cells": "' (e(cells)) `", "cellsel": ""' "`e(cellsel)'" `""}"'
}
file write `fh' _n "}" _n
file close `fh'
