* Reference for the joint event-study covariance of sp.did_imputation with
* BJS pre-trend coefficients (pretrend_method='bjs'): Stata did_imputation
* (Borusyak, SSC, version November 22, 2023), horizons(0/3) pretrends(3),
* on Track A module 05's CSV bytes (sp.datasets.mpdta()).
*
* did_imputation estimates the leads in an auxiliary regression on the
* untreated observations (reghdfe Y pre1..pre3, a(unit year) cluster(unit))
* and the horizons by imputation, and then forms e(V) with
*     matrix accum V = eps_tau0..eps_tau3 pre_eps1..pre_eps3
* over clusters: the per-cluster scores of both blocks side by side, so e(V)
* carries the lead-horizon cross-covariance too (Part 7 of the ado). The
* lead scores are the Frisch-Waugh weights of each lead times the auxiliary
* regression's residual, scaled by sqrt of reghdfe's cluster dof
* adjustment.
*
* The imputation weights are tightened to tol(1e-12) / maxit(100000) as in
* _generate_did_imputation_vcov_Stata.do; reghdfe's own solver tolerance
* (1e-8) still limits e(b) to ~1e-7.
*
* Run from the repository root with Stata 18:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_did_imputation_pretrend_vcov_Stata.do
version 18
set type double
clear all

import delimited "tests/r_parity/data/05_sunab.csv", clear asdouble
replace first_treat = . if first_treat == 0

did_imputation lemp countyreal year first_treat, horizons(0/3) pretrends(3) ///
    autosample tol(1e-12) maxit(100000)
matrix b = e(b)
matrix V = e(V)
local names : colnames b
local k = colsof(b)

local ver : di "`c(stata_version)'"
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/did_imputation_pretrend_vcov_Stata.json", ///
    write replace text
file write `fh' "{" _n
file write `fh' `"  "names": ["'
forvalues j = 1/`k' {
    local nm : word `j' of `names'
    file write `fh' `""`nm'""'
    if `j' < `k' file write `fh' ", "
}
file write `fh' "]," _n
file write `fh' `"  "b": ["'
forvalues j = 1/`k' {
    file write `fh' %24.16e (b[1, `j'])
    if `j' < `k' file write `fh' ", "
}
file write `fh' "]," _n
file write `fh' `"  "V": ["' _n
forvalues i = 1/`k' {
    file write `fh' "    ["
    forvalues j = 1/`k' {
        file write `fh' %24.16e (V[`i', `j'])
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "]"
    if `i' < `k' file write `fh' ","
    file write `fh' _n
}
file write `fh' "  ]," _n
file write `fh' `"  "options": "did_imputation lemp countyreal year first_treat, horizons(0/3) pretrends(3) autosample tol(1e-12) maxit(100000)","' _n
file write `fh' `"  "stata_version": "`ver'","' _n
file write `fh' `"  "did_imputation_version": "November 22, 2023""' _n
file write `fh' "}" _n
file close `fh'
