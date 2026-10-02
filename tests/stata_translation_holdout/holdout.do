*! holdout.do -- Stata side of the translation holdout.
*! Written by build_holdout.py; do not edit. Requires reghdfe, ftools.
version 17
clear all
set more off
* Double precision: see tests/stata_parity/option_parity/README.md.
set type double

program define _dump
    args fh id comma
    if _rc {
        file write `fh' `"  ""' "`id'" `"": {"rc": "' (_rc) "}`comma'" _n
        exit
    }
    matrix HB = e(b)
    matrix HV = e(V)
    local k = colsof(HB)
    file write `fh' `"  ""' "`id'" `"": {"rc": 0, "N": "' (e(N)) `", "b": ["'
    local first 1
    forvalues j = 1/`k' {
        local se = sqrt(HV[`j', `j'])
        if !(HB[1, `j'] == 0 & `se' == 0) {
            if !`first' file write `fh' ", "
            file write `fh' %24.16e (HB[1, `j'])
            local first 0
        }
    }
    file write `fh' `"], "se": ["'
    local first 1
    forvalues j = 1/`k' {
        local se = sqrt(HV[`j', `j'])
        if !(HB[1, `j'] == 0 & `se' == 0) {
            if !`first' file write `fh' ", "
            file write `fh' %24.16e (`se')
            local first 0
        }
    }
    file write `fh' `"], "names": ["'
    local names : colfullnames HB
    local first 1
    forvalues j = 1/`k' {
        local se = sqrt(HV[`j', `j'])
        if !(HB[1, `j'] == 0 & `se' == 0) {
            local nm : word `j' of `names'
            if !`first' file write `fh' ", "
            file write `fh' (char(34)) "`nm'" (char(34))
            local first 0
        }
    }
    file write `fh' "]}`comma'" _n
end

tempname fh
file open `fh' using "holdout_Stata.json", write replace
file write `fh' "{" _n
import delimited "holdout_cross.csv", clear asdouble case(preserve)
capture noisily regress y x1 x2
_dump `fh' "ols" ","
capture noisily regress y x1 x2, vce(robust)
_dump `fh' "ols_robust" ","
capture noisily reg y x1 x2, r
_dump `fh' "ols_abbrev_r" ","
capture noisily regress y x1 x2, vce(cluster g)
_dump `fh' "ols_cluster" ","
capture noisily reg y x1 x2, cl(g)
_dump `fh' "ols_abbrev_cl" ","
capture noisily regress y x1 x2, vce(hc3)
_dump `fh' "ols_hc3" ","
capture noisily regress y x1 x2 if d == 1
_dump `fh' "ols_if" ","
capture noisily regress y x1 x2 in 1/200
_dump `fh' "ols_in" ","
capture noisily regress y x1 x2 if d == 1 & x1 > 0
_dump `fh' "ols_if_and" ","
capture noisily regress y x1 x2 [aweight=w]
_dump `fh' "ols_aweight" ","
capture noisily regress y x1 x2 [pweight=w]
_dump `fh' "ols_pweight" ","
capture noisily regress y x1 x2 [fweight=fw]
_dump `fh' "ols_fweight" ","
capture noisily regress y c.x1##c.x2
_dump `fh' "ols_interaction" ","
capture noisily regress y i.k x1
_dump `fh' "ols_factor" ","
capture noisily regress y x1 i.d#c.x2
_dump `fh' "ols_factor_slope" ","
capture noisily regress y x1 x2, noconstant
_dump `fh' "ols_noconstant" ","
capture noisily regress y x1 x2, level(90)
_dump `fh' "ols_level" ","
capture noisily regress y x1 x2, beta
_dump `fh' "ols_beta_display" ","
capture noisily quietly regress y x1 x2
_dump `fh' "ols_quietly" ","
capture noisily regress y x1 xm
_dump `fh' "ols_missing" ","
capture noisily ivregress 2sls y x1 (x2 = z)
_dump `fh' "iv_2sls" ","
capture noisily ivregress 2sls y x1 (x2 = z), vce(robust)
_dump `fh' "iv_2sls_robust" ","
capture noisily ivregress liml y x1 (x2 = z z2)
_dump `fh' "iv_liml" ","
capture noisily logit yb x1 x2
_dump `fh' "logit" ","
capture noisily probit yb x1 x2, vce(robust)
_dump `fh' "probit_robust" ","
capture noisily poisson cnt x1 x2
_dump `fh' "poisson" ","
capture noisily poisson cnt x1 x2, vce(robust)
_dump `fh' "poisson_robust" ","
capture noisily qreg y x1 x2
_dump `fh' "qreg" ","
capture noisily areg y x1 x2, absorb(g)
_dump `fh' "areg" ","
capture noisily reghdfe y x1 x2, absorb(g) vce(cluster g)
_dump `fh' "reghdfe_cluster" ","
import delimited "holdout_panel.csv", clear asdouble case(preserve)
xtset id t
capture noisily xtreg y x, fe
_dump `fh' "xtreg_fe" ","
capture noisily xtreg y x, fe vce(cluster id)
_dump `fh' "xtreg_fe_cluster" ","
capture noisily xtreg y x, re
_dump `fh' "xtreg_re" ","
file write `fh' `"  "_meta": {"stata": "18 MP", "precision": "double"}"' _n
file write `fh' "}" _n
file close `fh'
set type float
