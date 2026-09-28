* Reference for sp.event_study_vcov on sp.did_imputation: the joint
* covariance of the horizon effects, from Stata did_imputation (Borusyak,
* SSC), on Track A module 05's CSV bytes (sp.datasets.mpdta()).
*
* did_imputation iterates twice:
*   - its imputation weights, to tol(1e-6) / maxit(100) by default. These
*     drive e(V) (and enter e(b)); tightened here to tol(1e-12) /
*     maxit(100000) so e(V) is the exact solution.
*   - reghdfe's fixed-effect solver, tolerance 1e-8, which did_imputation
*     does not expose. That leaves e(b) ~1e-7 relative from the exact
*     least-squares imputation.
* To show the second gap is only that solver, b_exact recomputes the
* horizon effects with a dense regress on unit and year dummies over the
* untreated rows (the exact imputation), all in Stata.
*
* Run from the repository root with Stata 18:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_did_imputation_vcov_Stata.do
version 18
set type double
clear all

import delimited "tests/r_parity/data/05_sunab.csv", clear asdouble
replace first_treat = . if first_treat == 0

did_imputation lemp countyreal year first_treat, horizons(0/3) autosample ///
    tol(1e-12) maxit(100000)
matrix b = e(b)
matrix V = e(V)

* Exact imputation by dense least squares.
gen byte D = !missing(first_treat) & year >= first_treat
regress lemp i.countyreal i.year if D == 0
predict double y0 if D == 1, xb
gen double tau = lemp - y0
gen int rel = year - first_treat if D == 1
matrix b_exact = J(1, 4, .)
forvalues h = 0/3 {
    quietly summarize tau if rel == `h', meanonly
    matrix b_exact[1, `h' + 1] = r(mean)
}

local ver : di "`c(stata_version)'"
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/did_imputation_vcov_Stata.json", ///
    write replace text
file write `fh' "{" _n
file write `fh' `"  "horizons": [0, 1, 2, 3],"' _n
foreach m in b b_exact {
    file write `fh' `"  "`m'": ["'
    forvalues j = 1/4 {
        file write `fh' %24.16e (`m'[1, `j'])
        if `j' < 4 file write `fh' ", "
    }
    file write `fh' "]," _n
}
file write `fh' `"  "V": ["' _n
forvalues i = 1/4 {
    file write `fh' "    ["
    forvalues j = 1/4 {
        file write `fh' %24.16e (V[`i', `j'])
        if `j' < 4 file write `fh' ", "
    }
    file write `fh' "]"
    if `i' < 4 file write `fh' ","
    file write `fh' _n
}
file write `fh' "  ]," _n
file write `fh' `"  "options": "did_imputation lemp countyreal year first_treat, horizons(0/3) autosample tol(1e-12) maxit(100000)","' _n
file write `fh' `"  "stata_version": "`ver'","' _n
file write `fh' `"  "did_imputation_version": "November 22, 2023""' _n
file write `fh' "}" _n
file close `fh'
