* Stata 18 spregress, ml dvarlag(W) errorlag(W): reference for sp.sac.
* Reuses the sp.sdem / sp.gns fixtures (spatial_durbin_error_{1,2}.csv).
*   cd tests/reference_parity/_fixtures && stata-mp -b do _generate_spatial_sac_stata.do
version 18
clear all
set more off
tempname fh
file open `fh' using "spatial_sac_Stata.json", write replace
file write `fh' "{" _n
forvalues s = 1/2 {
    import delimited using "spatial_durbin_error_`s'.csv", clear asdouble
    local N = _N
    spset id
    spmatrix fromdata W = w1-w`N', normalize(row) replace
    qui spregress y x1 x2, ml dvarlag(W) errorlag(W) nrtolerance(1e-13) tolerance(1e-13) ltolerance(1e-15)
    matrix b = e(b)
    matrix V = e(V)
    local names : colfullnames b
    if `s' > 1 file write `fh' "," _n
    file write `fh' `"  "`s'_sac": {"ll": "' %23.16e (e(ll)) `", "coef": {"'
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
