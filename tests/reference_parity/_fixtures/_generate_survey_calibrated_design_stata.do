* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_survey_calibrated_design_parity.py
* (SurveyDesign.calibrate(variance="stata")). Built-in svyset rake()/regress()
* (Stata 18), no SSC packages. Reads survey_calib_data.csv, writes
* survey_calibrated_design_stata.json. Run from this directory.
*
* svyset ..., rake() / regress(): svy estimates report the calibration-
* adjusted linearisation (residuals from a regression weighted by the
* calibrated weights).
* ---------------------------------------------------------------------------
version 18
set more off
set type double
import delimited using "survey_calib_data.csv", clear asdouble case(preserve)
generate byte sexn = (sex == "M")
generate byte agen = cond(agegrp == "a", 1, cond(agegrp == "b", 2, 3))
local rake_totals "_cons=10000 1.sexn=4900 2.agen=4500 3.agen=2500"

tempname fh
file open `fh' using "survey_calibrated_design_stata.json", write replace text
file write `fh' "{" _n

foreach case in raking linear {
    if "`case'" == "raking" {
        svyset psu [pw=d], strata(stratum) rake(i.sexn i.agen, totals(`rake_totals'))
    }
    else {
        svyset psu [pw=d], strata(stratum) regress(income age, totals(income=345000 age=440000) noconstant)
    }
    quietly svy: total y
    local ty = e(b)[1,1]
    local sty = sqrt(e(V)[1,1])
    quietly svy: regress y income age
    file write `fh' `"  "`case'": {"total_y": "' %23.16e (`ty') `", "se_total_y": "' %23.16e (`sty')
    file write `fh' `", "reg_coef": ["' %23.16e (e(b)[1,3]) ", " %23.16e (e(b)[1,1]) ", " %23.16e (e(b)[1,2]) "]"
    file write `fh' `", "reg_se": ["' %23.16e (sqrt(e(V)[3,3])) ", " %23.16e (sqrt(e(V)[1,1])) ", " %23.16e (sqrt(e(V)[2,2])) "]"
    file write `fh' `", "df_r": "' %23.16e (e(df_r)) "}," _n
}
file write `fh' `"  "provenance": {"stata_version": ""' "`c(stata_version)'" `"", "generated_by": "tests/reference_parity/_fixtures/_generate_survey_calibrated_design_stata.do"}"' _n
file write `fh' "}" _n
file close `fh'
display "wrote survey_calibrated_design_stata.json"
