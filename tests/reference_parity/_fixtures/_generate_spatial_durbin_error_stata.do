* Stata 18 spregress, ml: reference for sp.sdem / sp.gns.
*   python tests/reference_parity/_fixtures/_generate_spatial_durbin_error_data.py
*   cd tests/reference_parity/_fixtures && stata-mp -b do _generate_spatial_durbin_error_stata.do
version 18
clear all
set more off
tempname fh
file open `fh' using "spatial_durbin_error_Stata.json", write replace
file write `fh' "{" _n
local first = 1
forvalues s = 1/2 {
    import delimited using "spatial_durbin_error_`s'.csv", clear asdouble
    local N = _N
    spset id
    spmatrix fromdata W = w1-w`N', normalize(row) replace
    foreach m in gns sdem {
        if "`m'" == "gns" qui spregress y x1 x2, ml dvarlag(W) errorlag(W) ivarlag(W: x1 x2) nrtolerance(1e-13) tolerance(1e-13) ltolerance(1e-15)
        if "`m'" == "sdem" qui spregress y x1 x2, ml errorlag(W) ivarlag(W: x1 x2) nrtolerance(1e-13) tolerance(1e-13) ltolerance(1e-15)
        matrix b = e(b)
        matrix V = e(V)
        local names : colfullnames b
        if `first' == 0 file write `fh' "," _n
        local first = 0
        file write `fh' `"  "`s'_`m'": {"ll": "' %23.16e (e(ll)) `", "coef": {"'
        local j = 0
        foreach nm of local names {
            local ++j
            if `j' > 1 file write `fh' ", "
            file write `fh' `""`nm'": ["' %23.16e (b[1,`j']) ", " %23.16e (sqrt(V[`j',`j'])) "]"
        }
        file write `fh' "}}"
    }
}
file write `fh' _n "}" _n
file close `fh'
