* Stata med4way 2.3.2 (Discacciati et al., GitHub anddis/med4way; GPL-3,
* used as a black box: only its outputs) reference for
* sp.four_way_decomposition(yreg=, mreg=, vce='ml').
* Installs med4way into the PRIVATE _ado_teffects/ directory (never PLUS).
*   python _generate_four_way_nonlinear_data.py
*   cd tests/reference_parity/_fixtures && stata-mp -b do _generate_four_way_nonlinear_stata.do
* The regressions are run to tight tolerances so that the comparison is
* not limited by Stata's default convergence criteria.
version 18
clear all
set more off
local here "`c(pwd)'"
local ado "`here'/_ado_teffects"
capture mkdir "`ado'"
net set ado "`ado'"
adopath ++ "`ado'"
capture which med4way
if _rc net install med4way, from("https://raw.githubusercontent.com/anddis/med4way/master/") replace
import delimited using "four_way_nonlinear.csv", clear asdouble
local tol "yregoptions(nrtolerance(1e-13) tolerance(1e-14)) mregoptions(nrtolerance(1e-13) tolerance(1e-14))"
local c1 "ybc a mc c1 c2, yreg(logistic) mreg(linear) a0(0) a1(1) m(0) fulloutput"
local c2 "ybb a mb c1 c2, yreg(logistic) mreg(logistic) a0(0) a1(1) m(0) fulloutput"
local c3 "ypc a mc c1 c2, yreg(poisson) mreg(linear) a0(0) a1(1) m(1) fulloutput"
local c4 "ypb a mb c1 c2, yreg(poisson) mreg(logistic) a0(0) a1(1) m(1) fulloutput"
local c5 "yc a mb c1 c2, yreg(linear) mreg(logistic) a0(0) a1(1) m(0) fulloutput"
local c6 "ybb a mb c1 c2, yreg(logistic) mreg(logistic) a0(0) a1(1) m(0) c(0.5 1) fulloutput"
local c7 "ybc a mc, yreg(logistic) mreg(linear) a0(0) a1(1) m(0.5) fulloutput"
local c8 "ypc a mc c1 c2, yreg(poisson) mreg(linear) a0(-0.5) a1(1.5) m(2) c(. 0) fulloutput"
tempname fh
file open `fh' using "four_way_nonlinear_Stata.json", write replace
file write `fh' "{" _n
forvalues k = 1/8 {
    qui med4way `c`k'' `tol'
    matrix b = e(b)
    matrix V = e(V)
    local names : colnames b
    if `k' > 1 file write `fh' "," _n
    file write `fh' `"  "`k'": {"N": "' (e(N)) `", "b": {"'
    local j = 0
    foreach nm of local names {
        local ++j
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'": "' %23.16e (b[1,`j'])
    }
    file write `fh' `"}, "se": {"'
    local j = 0
    foreach nm of local names {
        local ++j
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'": "' %23.16e (sqrt(V[`j',`j']))
    }
    file write `fh' "}}"
}
file write `fh' _n "}" _n
file close `fh'
