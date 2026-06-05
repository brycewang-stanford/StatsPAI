* tests/stata_parity/55_hc2_hc3.do
*
* Module 55: OLS + HC2 / HC3 heteroskedasticity-robust SE (MacKinnon-White 1985).
*   StatsPAI:  sp.regress(robust="hc2"/"hc3")
*   R:         lm + sandwich::vcovHC(type="HC2"/"HC3")
*   Stata:     regress, vce(hc2)  /  regress, vce(hc3)
*
* Module 01 pins HC1; this module materializes the two small-sample
* variants. Stata's vce(hc2)/vce(hc3) are the native MacKinnon-White
* adjustments, so betas and SEs match R/Python to machine precision.
*
* Tolerance: rel < 1e-6 (closed-form covariance, no convention gap).

version 18
clear all

do _common.do
stata_parity_init, module(55_hc2_hc3)
stata_parity_open, module(55_hc2_hc3)

import delimited "${STATA_PARITY_DATA}/55_hc2_hc3.csv", clear case(preserve)

foreach kind in hc2 hc3 {
    regress lemp treat year, vce(`kind')

    local n = e(N)
    matrix B = e(b)
    matrix V = e(V)
    local vars : colnames B
    foreach v of local vars {
        local bv = B[1, "`v'"]
        local sv = sqrt(V["`v'", "`v'"])
        local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
        local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
        if "`v'" == "_cons" local stat "`kind'_(Intercept)"
        else                local stat "`kind'_`v'"
        stata_parity_row, stat("`stat'") est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')
    }
}

stata_parity_extra, key(formula) val("lemp ~ treat + year")
stata_parity_extra, key(vcov) val("HC2 + HC3 (MacKinnon-White 1985)")
stata_parity_extra, key(stata_command) val("regress lemp treat year, vce(hc2) | vce(hc3)")

stata_parity_close, module(55_hc2_hc3)
