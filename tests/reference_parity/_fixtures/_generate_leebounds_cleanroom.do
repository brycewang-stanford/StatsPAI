* Stata leebounds (Tauchmann, SSC v1.5) run as a black box for the clean-room
* rewrite of sp.lee_bounds(trimming='leebounds') (docs/dev/cleanroom/leebounds.md).
* set type double: generated variables in double precision, so the analytic
* variance is compared without single-precision storage artefacts (the
* default-type fixtures cover those).
* leebounds is installed into the gitignored _ado_leebounds/, not PLUS.
*   python tests/reference_parity/_fixtures/_generate_leebounds_cleanroom_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_leebounds_cleanroom.do
version 18
clear all
set type double
local ado "tests/reference_parity/_fixtures/_ado_leebounds"
capture mkdir "`ado'"
adopath + "`ado'"
capture which leebounds
if _rc {
    net set ado "`ado'"
    ssc install leebounds, replace
    net set ado PLUS
}
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/leebounds_cleanroom_stata.json", write replace
file write `fh' "{" _n `"  "samples": {"' _n
import delimited using "tests/reference_parity/_fixtures/leebounds_cleanroom.csv", clear asdouble
local first 1
forvalues k = 1/6 {
    foreach cov in "none" "x" {
        if `k' <= 2 & "`cov'" == "x" continue
        if inlist(`k', 3, 4, 5) & "`cov'" == "none" continue
        preserve
        keep if sample == `k'
        if "`cov'" == "none" qui leebounds y d, select(s)
        else qui leebounds y d, select(s) tight(x)
        matrix b = e(b)
        matrix V = e(V)
        if !`first' file write `fh' "," _n
        local first 0
        file write `fh' `"    "`k'_`cov'": {"lower": "' %24.16e (b[1,1]) `", "upper": "' %24.16e (b[1,2])
        file write `fh' `", "var_lower": "' %24.16e (V[1,1]) `", "var_upper": "' %24.16e (V[2,2])
        file write `fh' `", "trim": "' %24.16e (e(trim)) `", "trimmed": ""' "`e(trimmed)'" `"""'
        if "`cov'" == "x" file write `fh' `", "cells": "' (e(cells)) `", "cellsel": ""' "`e(cellsel)'" `"""'
        file write `fh' "}"
        restore
    }
}
file write `fh' _n "  }," _n `"  "macro": ["' _n
import delimited using "tests/reference_parity/_fixtures/leebounds_cleanroom_macro.csv", clear asdouble
forvalues i = 1/`=_N' {
    local m = v[`i']
    if `i' > 1 file write `fh' "," _n
    file write `fh' `"    ["' %24.16e (v[`i']) `", ""' "`m'" `""]"'
}
file write `fh' _n "  ]" _n "}" _n
file close `fh'
