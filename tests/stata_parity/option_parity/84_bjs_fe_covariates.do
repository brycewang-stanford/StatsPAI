*! 84_bjs_fe_covariates.do
*! Golden numbers for tests/reference_parity/test_bjs_fe_covariates_parity.py
*!
*! Pins did_imputation's Y(0)-model options against StatsPAI:
*!
*!   fe(i t)          -> fe=None / fe=['countyreal','year']   (default)
*!   fe(t)            -> fe=['year']
*!   fe(.)            -> fe=[]
*!   unitcontrols(x)  -> unit_covariates=['x']
*!   timecontrols(x)  -> time_covariates=['x']
*!   controls(x)      -> controls=['x']
*!
*! NOTE on unitcontrols: on the full mpdta panel Stata REFUSES this
*! (rc 481, "some absorbed variables/FEs are collinear in the D==0
*! subsample but not in the full sample"), because 100 treated
*! observations sit in cohort-2004 units that have exactly one untreated
*! period — too few for an intercept plus a slope. StatsPAI raises its own
*! error there rather than returning lsqr's minimum-norm answer. The
*! parity number is therefore taken on the >=2-untreated-period subset,
*! where both packages agree the model is identified.
*!
*! Requires: did_imputation, reghdfe, ftools. Stata 18 MP.

version 17
clear all
set more off

import delimited "../../orig_parity/data/02_mpdta_original.csv", clear

* did_imputation wants the cohort variable missing for never-treated.
gen Ei = first_treat
replace Ei = . if first_treat == 0

tempname fh
file open `fh' using "results/84_bjs_fe_covariates_Stata.json", write replace
file write `fh' "{" _n

program define _dump_tau
    args fh label comma
    file write `fh' `"  ""' "`label'" `"": {"att": "' %20.15f (_b[tau]) ///
        `", "se": "' %20.15f (_se[tau]) "}`comma'" _n
end

qui did_imputation lemp countyreal year Ei
_dump_tau `fh' "default" ","

qui did_imputation lemp countyreal year Ei, fe(year)
_dump_tau `fh' "fe_time_only" ","

qui did_imputation lemp countyreal year Ei, fe(.)
_dump_tau `fh' "fe_none" ","

qui did_imputation lemp countyreal year Ei, timecontrols(lpop)
_dump_tau `fh' "timecontrols_lpop" ","

qui did_imputation lemp countyreal year Ei, controls(lpop)
_dump_tau `fh' "controls_lpop" ","

* --- identified subset for unitcontrols -------------------------------
gen untreated = missing(Ei) | year < Ei
egen nuntr = total(untreated), by(countyreal)
preserve
keep if nuntr >= 2

qui did_imputation lemp countyreal year Ei
_dump_tau `fh' "default_identified_subset" ","

qui did_imputation lemp countyreal year Ei, unitcontrols(year)
_dump_tau `fh' "unitcontrols_year_subset" ","

restore

file write `fh' `"  "_meta": {"command": "did_imputation", "stata": "18 MP"}"' _n
file write `fh' "}" _n
file close `fh'

di as txt "wrote results/84_bjs_fe_covariates_Stata.json"
