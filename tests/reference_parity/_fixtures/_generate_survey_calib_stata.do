* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_survey_calib_R_parity.py
* (Stata-side block). Built-in svycal (Stata 18) -- no SSC packages.
*
* Reads survey_calib_data.csv (written once by _generate_survey_calib_R.R)
* and writes survey_calib_stata.json. Run from this directory:
*
*   /Applications/Stata/StataMP.app/Contents/MacOS/stata-mp -b do _generate_survey_calib_stata.do
*
* Conventions pinned here
* -----------------------
* * svycal rake: multiplicative (raking-ratio) distance, Newton on the
*   Lagrange multipliers; totals are population counts with _cons = N.
*   Same fixed point as IPF / R calibrate(calfun = "raking").
* * svycal regress: additive (linear, chi-squared) distance;
*   noconstant for the no-intercept case.
* * svyset ..., rake() / regress(): svy: mean then reports the
*   calibration-adjusted linearisation SE (residuals of y on the
*   calibration variables) -- recorded to size the gap to the
*   fixed-weights SE that sp.svydesign reports for calibrated weights.
* * Weights are written per observation in file order.
* ---------------------------------------------------------------------------
version 18
set more off
set type double
import delimited using "survey_calib_data.csv", clear asdouble case(preserve)
generate byte sexn = (sex == "M")
generate byte agen = cond(agegrp == "a", 1, cond(agegrp == "b", 2, 3))

local rake_totals "_cons=10000 1.sexn=4900 2.agen=4500 3.agen=2500"
svycal rake i.sexn i.agen [pw=d], generate(w_rake) totals(`rake_totals')
svycal regress income age [pw=d], generate(w_lin_noint) ///
    totals(income=345000 age=440000) noconstant
svycal regress i.sexn income [pw=d], generate(w_lin_int) ///
    totals(_cons=10000 1.sexn=4900 income=345000)

tempname fh
file open `fh' using "survey_calib_stata.json", write replace text
file write `fh' "{" _n
foreach v in w_rake w_lin_noint w_lin_int {
    file write `fh' `"  "`v'": ["'
    forvalues i = 1/`=_N' {
        if `i' > 1 file write `fh' ", "
        file write `fh' %23.16e (`v'[`i'])
    }
    file write `fh' "]," _n
}

* calibration-adjusted SE (svyset rake / regress) vs fixed calibrated weights
svyset psu [pw=d], strata(stratum) rake(i.sexn i.agen, totals(`rake_totals'))
quietly svy: mean y
file write `fh' `"  "rake_mean_y": "' %23.16e (e(b)[1,1]) ", " `""rake_se_calibrated": "' %23.16e (sqrt(e(V)[1,1])) "," _n
svyset psu [pw=w_rake], strata(stratum)
quietly svy: mean y
file write `fh' `"  "rake_se_fixed_weights": "' %23.16e (sqrt(e(V)[1,1])) "," _n
svyset psu [pw=d], strata(stratum) regress(income age, totals(income=345000 age=440000) noconstant)
quietly svy: mean y
file write `fh' `"  "linear_noint_se_calibrated": "' %23.16e (sqrt(e(V)[1,1])) "," _n

file write `fh' `"  "provenance": {"stata_version": ""' "`c(stata_version)'" `"", "edition": ""' "`c(edition_real)'" `"", "generated_by": "tests/reference_parity/_fixtures/_generate_survey_calib_stata.do"}"' _n
file write `fh' "}" _n
file close `fh'
