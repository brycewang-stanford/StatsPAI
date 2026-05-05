* tests/stata_parity/01_ols.do
*
* Module 01: OLS + HC1 robust SE.
*   StatsPAI:  sp.regress(formula, robust="hc1")
*   R:         lm() + sandwich::vcovHC(type="HC1")
*   Stata:     regress, vce(robust)        <-- this file
*
* Tolerance: rel < 1e-6 (closed-form estimator).
* Note: Stata's vce(robust) on regress is HC1 by default (Stata 18).

version 18
clear all

cd "`c(pwd)'"
do _common.do

stata_parity_init, module(01_ols)
stata_parity_open, module(01_ols)

import delimited "${STATA_PARITY_DATA}/01_ols.csv", clear case(preserve)

regress lwage educ exper expersq black south smsa, vce(robust)

local n = e(N)
matrix B = e(b)
matrix V = e(V)
local vars : colnames B
foreach v of local vars {
    local bv = B[1, "`v'"]
    local sv = sqrt(V["`v'", "`v'"])
    local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
    local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
    * Stata calls intercept "_cons"; map to R/Python "(Intercept)" so
    * compare.py joins row-for-row across all three sides.
    if "`v'" == "_cons" {
        local stat "beta_(Intercept)"
    }
    else {
        local stat "beta_`v'"
    }
    stata_parity_row, statname("`stat'") estimate(`bv') stderr(`sv') cilo(`lo') cihi(`hi') nobs(`n')
}

stata_parity_extra, key(formula) val("lwage ~ educ + exper + expersq + black + south + smsa")
stata_parity_extra, key(vcov) val(HC1)
stata_parity_extra, key(stata_command) val("regress, vce(robust)")

stata_parity_close, module(01_ols)
