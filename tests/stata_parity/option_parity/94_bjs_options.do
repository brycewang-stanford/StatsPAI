*! 94_bjs_options.do
*! Golden numbers for tests/reference_parity/test_bjs_options_parity.py
*!
*! Pins did_imputation (Borusyak's implementation, the canonical one for
*! the BJS estimator) against sp.did_imputation under the options that
*! fixture 84 does not reach:
*!
*!   fe(countyreal region#year)       -> fe=['countyreal', 'region#year']
*!   fe(region year)              -> fe=['region', 'year']
*!   controls(x)                  -> controls=['x']           (x varies)
*!   controls(x) timecontrols(lpop) -> controls= + time_covariates=
*!   controls(x) unitcontrols(year) -> controls= + unit_covariates=
*!                                     (identified subset, as in 84)
*!   [aw=w]                       -> weights='w'
*!   [aw=w] horizons(0/3)         -> weights='w', horizon=[0, 1, 2, 3]
*!   [aw=w] fe(countyreal region#year) controls(x)
*!   fe(countyreal region#year) horizons(0/3)
*!   fe(year) without the 2003 rows of the 2004 cohort
*!                                -> fe=['year'] on the same rows
*!
*! horizons() runs pass minn(0): did_imputation otherwise suppresses
*! tau2 and tau3 (effective sample size below its default minn(30)), and
*! sp.did_imputation reports every horizon unless min_n= is given.
*!
*! Data: data_94_bjs_options.csv (mpdta with st, region, x and w added, written
*! by _generate_data_94_bjs_options.py).
*!
*! Requires: did_imputation, reghdfe, ftools. Stata 18 MP.

version 17
clear all
set more off

* Double precision throughout (see option_parity/README.md).
set type double
import delimited "data_94_bjs_options.csv", clear asdouble
local tight "tol(1e-12) maxit(10000)"

* did_imputation wants the cohort variable missing for never-treated.
gen Ei = first_treat
replace Ei = . if first_treat == 0

tempname fh
file open `fh' using "results/94_bjs_options_Stata.json", write replace
file write `fh' "{" _n

program define _dump_b
    args fh label
    matrix b = e(b)
    matrix V = e(V)
    local cn : colnames b
    file write `fh' `"  ""' "`label'" `"": {"' _n
    local k = colsof(b)
    forvalues j = 1/`k' {
        local nm : word `j' of `cn'
        file write `fh' `"    ""' "`nm'" `"": {"att": "' %25.18f (b[1,`j']) ///
            `", "se": "' %25.18f (sqrt(V[`j',`j'])) "}," _n
    }
    file write `fh' `"    "_N": "' %12.0f (e(N)) _n
    file write `fh' "  }," _n
end

qui did_imputation lemp countyreal year Ei, fe(countyreal region#year) `tight'
_dump_b `fh' "fe_unit_region_year"

qui did_imputation lemp countyreal year Ei, fe(region year) `tight'
_dump_b `fh' "fe_region_year"

qui did_imputation lemp countyreal year Ei, controls(x) `tight'
_dump_b `fh' "controls_x"

qui did_imputation lemp countyreal year Ei, controls(x) timecontrols(lpop) `tight'
_dump_b `fh' "mixed_controls_timecontrols"

qui did_imputation lemp countyreal year Ei [aw=w], `tight'
_dump_b `fh' "weights"

qui did_imputation lemp countyreal year Ei [aw=w], horizons(0/3) minn(0) `tight'
_dump_b `fh' "weights_horizons"

qui did_imputation lemp countyreal year Ei [aw=w], fe(countyreal region#year) controls(x) `tight'
_dump_b `fh' "weights_fe_unit_region_year_controls_x"

qui did_imputation lemp countyreal year Ei, fe(countyreal region#year) horizons(0/3) minn(0) `tight'
_dump_b `fh' "fe_unit_region_year_horizons"

* --- fe(year) with units treated in every period they are observed ------
* Without a unit effect such units need no untreated history: both
* packages estimate (StatsPAI refused before 2026-10-15).
preserve
drop if first_treat == 2004 & year == 2003
qui did_imputation lemp countyreal year Ei, fe(year) `tight'
_dump_b `fh' "fe_time_only_always_treated"
restore

* --- identified subset for unitcontrols (see 84_bjs_fe_covariates.do) ---
gen untreated = missing(Ei) | year < Ei
egen nuntr = total(untreated), by(countyreal)
keep if nuntr >= 2

qui did_imputation lemp countyreal year Ei, controls(x) unitcontrols(year) `tight'
_dump_b `fh' "mixed_controls_unitcontrols_subset"

file write `fh' `"  "_meta": {"command": "did_imputation", "stata": "18 MP", "precision": "double", "tol": "1e-12"}"' _n
file write `fh' "}" _n
file close `fh'

set type float
di as txt "wrote results/94_bjs_options_Stata.json"
