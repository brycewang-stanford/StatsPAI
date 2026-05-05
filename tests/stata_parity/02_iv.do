* tests/stata_parity/02_iv.do
*
* Module 02: 2SLS + HC1.
*   StatsPAI:  sp.iv (HC1)
*   R:         AER::ivreg + sandwich::vcovHC(type="HC1")
*   Stata:     ivregress 2sls, vce(robust) small
*
* `small` requests the (N-1)/(N-K) df adjustment so Stata's robust
* matches R/Python's HC1.

version 18
clear all

do _common.do
stata_parity_init, module(02_iv)
stata_parity_open, module(02_iv)

import delimited "${STATA_PARITY_DATA}/02_iv.csv", clear case(preserve)

ivregress 2sls lwage exper expersq black south smsa (educ = nearc4), vce(robust) small

local n = e(N)
matrix B = e(b)
matrix V = e(V)
local vars : colnames B
foreach v of local vars {
    local bv = B[1, "`v'"]
    local sv = sqrt(V["`v'", "`v'"])
    local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
    local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
    if "`v'" == "_cons" local stat "beta_(Intercept)"
    else                local stat "beta_`v'"
    stata_parity_row, stat("`stat'") est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')
}

stata_parity_extra, key(formula) val("lwage ~ exper + expersq + black + south + smsa + educ | nearc4")
stata_parity_extra, key(vcov) val(HC1)
stata_parity_extra, key(stata_command) val("ivregress 2sls, vce(robust) small")

stata_parity_close, module(02_iv)
