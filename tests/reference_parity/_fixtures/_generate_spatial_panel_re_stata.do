* Stata 18 spxtregress, re: reference for sp.spatial_panel(effects='re').
*   python tests/reference_parity/_fixtures/_generate_spatial_panel_re_data.py
*   cd tests/reference_parity/_fixtures && stata-mp -b do _generate_spatial_panel_re_stata.do
version 18
clear all
set more off
import delimited using "spatial_panel_re_W.csv", clear
spset id
spmatrix fromdata W = w1-w48, normalize(row)
import delimited using "spatial_panel_re.csv", clear asdouble
xtset id t
spset id
local tol "nrtolerance(1e-13) tolerance(1e-13) ltolerance(1e-15)"
tempname fh
file open `fh' using "spatial_panel_re_Stata.json", write replace
file write `fh' "{" _n
local first = 1
foreach m in sar sem sac sdm {
    if "`m'" == "sar" qui spxtregress y x x2, re dvarlag(W) `tol'
    if "`m'" == "sem" qui spxtregress y x x2, re errorlag(W) `tol'
    if "`m'" == "sac" qui spxtregress y x x2, re dvarlag(W) errorlag(W) `tol'
    if "`m'" == "sdm" qui spxtregress y x x2, re dvarlag(W) ivarlag(W: x x2) `tol'
    matrix b = e(b)
    matrix V = e(V)
    local names : colfullnames b
    if `first' == 0 file write `fh' "," _n
    local first = 0
    file write `fh' `"  "`m'": {"ll": "' %23.16e (e(ll)) `", "coef": {"'
    local j = 0
    foreach nm of local names {
        local ++j
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'": ["' %23.16e (b[1,`j']) ", " %23.16e (sqrt(V[`j',`j'])) "]"
    }
    file write `fh' "}}"
}
file write `fh' _n "}" _n
file close `fh'
