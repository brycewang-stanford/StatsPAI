* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_survey_design_R_parity.py
* (Stata-side block). Built-in svy only -- no SSC packages.
*
* Reads survey_design_data.csv (written once by _generate_survey_design_R.R)
* and writes survey_design_stata.json. Run from this directory:
*
*   /Applications/Stata/StataMP.app/Contents/MacOS/stata-mp -b do _generate_survey_design_stata.do
*
* Numbers are written as %23.16e (%g drops the leading zero, which JSON
* forbids).
*
* Conventions pinned here
* -----------------------
* * svyset identifies PSUs within strata, so the psu column (ids restart at
*   1 in every stratum) needs no relabelling -- the R side needs nest=TRUE.
* * e(df_r) = #PSU - #strata for every svy estimator, regressions included
*   (R's summary.svyglm uses degf + 1 - rank instead).
* * fpc() takes either the population number of PSUs in the stratum or the
*   sampling fraction.
* * singleunit(certainty | scaled | centered) are the lonely-PSU rules; R's
*   survey.lonely.psu "certainty"/"remove" (0 contribution), "average"
*   (scale by #strata / #strata with >1 PSU) and "adjust" (centre at the
*   grand mean) are the counterparts that are tested.
* * logit / poisson are run with tolerance(1e-13) ltolerance(1e-13)
*   nrtolerance(1e-13) so that e(b), e(V) are the converged MLE rather
*   than the default-tolerance iterate (the default leaves ~2e-9 in the
*   logit slope here).
* * estat effects: deff = V_design / V_srs. With an fpc declared V_srs is
*   the without-replacement (1 - n/N) s^2 / n, N the sum of weights (R
*   deff = TRUE); without an fpc it is s^2 / n (R deff = "replace").
* ---------------------------------------------------------------------------
version 18
set more off
set type double
import delimited using "survey_design_data.csv", clear asdouble case(preserve)
generate double psu_global = stratum * 100 + psu

tempname fh
file open `fh' using "survey_design_stata.json", write replace text
file write `fh' "{" _n

capture program drop dumpest
program define dumpest
    * dumpest <handle> <label> [deff]
    args fh label dodeff
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local k = colsof(`b')
    file write `fh' `"  "`label'": {"b": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ", "
        file write `fh' %23.16e (`b'[1,`j'])
    }
    file write `fh' "], " `""V": ["'
    forvalues i = 1/`k' {
        if `i' > 1 file write `fh' ", "
        file write `fh' "["
        forvalues j = 1/`k' {
            if `j' > 1 file write `fh' ", "
            file write `fh' %23.16e (`V'[`i',`j'])
        }
        file write `fh' "]"
    }
    file write `fh' "], " `""df_r": "' %23.16e (e(df_r)) ", " `""N_psu": "' %23.16e (e(N_psu)) ", " `""N_strata": "' %23.16e (e(N_strata))
    if "`dodeff'" == "deff" {
        quietly estat effects
        tempname d
        matrix `d' = r(deff)
        file write `fh' ", " `""deff": ["'
        forvalues j = 1/`k' {
            if `j' > 1 file write `fh' ", "
            file write `fh' %23.16e (`d'[1,`j'])
        }
        file write `fh' "]"
    }
    file write `fh' "}," _n
end

* ---- full: strata + PSU + weights + fpc (PSU counts) ----------------------
svyset psu [pw=w], strata(stratum) fpc(fpc_N)
quietly svy: mean y
dumpest `fh' full_mean_y deff
quietly svy: mean y x
dumpest `fh' full_mean_yx
quietly svy: total y
dumpest `fh' full_total_y deff
quietly svy: regress y x
dumpest `fh' full_regress
quietly svy: logit yb x, tolerance(1e-13) ltolerance(1e-13) nrtolerance(1e-13)
dumpest `fh' full_logit
quietly svy: poisson yc x, tolerance(1e-13) ltolerance(1e-13) nrtolerance(1e-13)
dumpest `fh' full_poisson

* ---- frac: fpc as sampling fraction -----------------------------------------
svyset psu [pw=w], strata(stratum) fpc(fpc_f)
quietly svy: mean y
dumpest `fh' frac_mean_y

* ---- nofpc ------------------------------------------------------------------
svyset psu [pw=w], strata(stratum)
quietly svy: mean y
dumpest `fh' nofpc_mean_y deff
quietly svy: total y
dumpest `fh' nofpc_total_y
quietly svy: regress y x
dumpest `fh' nofpc_regress
quietly svy: logit yb x, tolerance(1e-13) ltolerance(1e-13) nrtolerance(1e-13)
dumpest `fh' nofpc_logit
quietly svy: poisson yc x, tolerance(1e-13) ltolerance(1e-13) nrtolerance(1e-13)
dumpest `fh' nofpc_poisson

* ---- element-sampled stratified design with element fpc ---------------------
svyset _n [pw=w], strata(stratum) fpc(fpc_el)
quietly svy: mean y
dumpest `fh' element_fpc_mean_y deff
quietly svy: regress y x
dumpest `fh' element_fpc_regress

* ---- clusters without strata --------------------------------------------------
svyset psu_global [pw=w]
quietly svy: mean y
dumpest `fh' cluster_only_mean_y deff
quietly svy: regress y x
dumpest `fh' cluster_only_regress

* ---- lonely PSU ---------------------------------------------------------------
foreach su in certainty scaled centered {
    svyset psu [pw=w], strata(stratum_l) singleunit(`su')
    quietly svy: mean y x
    dumpest `fh' lonely_`su'_mean_yx
    quietly svy: regress y x
    dumpest `fh' lonely_`su'_regress
}
svyset psu [pw=w], strata(stratum_l)
* default singleunit(missing): the reported SE is missing (e(V) holds 0)
quietly svy: mean y
tempname Tm
matrix `Tm' = r(table)
file write `fh' `"  "lonely_missing_se": ""' (cond(missing(`Tm'[2,1]), "missing", "nonmissing")) `"","' _n

file write `fh' `"  "provenance": {"stata_version": ""' "`c(stata_version)'" `"", "edition": ""' "`c(edition_real)'" `"", "generated_by": "tests/reference_parity/_fixtures/_generate_survey_design_stata.do"}"' _n
file write `fh' "}" _n
file close `fh'
