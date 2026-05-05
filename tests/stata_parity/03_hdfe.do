* tests/stata_parity/03_hdfe.do
*
* Module 03: 2-way HDFE with iid SE.
*   StatsPAI:  sp.fast.feols (vcov="iid")
*   R:         fixest::feols(... | firm + year, vcov="iid")
*   Stata:     reghdfe ..., absorb(firm year) vce(unadjusted)
*
* reghdfe matches fixest's iid SE up to a documented 1-df small-sample
* adjustment (which is what r_parity calls a "1-df conv. gap" and
* tolerates at rel < 1e-2 on SE while keeping rel < 1e-6 on point estimate).

version 18
clear all

do _common.do
stata_parity_init, module(03_hdfe)
stata_parity_open, module(03_hdfe)

import delimited "${STATA_PARITY_DATA}/03_hdfe.csv", clear case(preserve)

reghdfe y x1 x2, absorb(firm year) vce(unadjusted)

local n = e(N)
matrix B = e(b)
matrix V = e(V)
foreach v in x1 x2 {
    local bv = B[1, "`v'"]
    local sv = sqrt(V["`v'", "`v'"])
    local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
    local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
    stata_parity_row, stat(beta_`v') est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')
}

stata_parity_extra, key(formula) val("y ~ x1 + x2 | firm + year")
stata_parity_extra, key(vcov) val(iid)
stata_parity_extra, key(stata_command) val("reghdfe, absorb(firm year) vce(unadjusted)")

stata_parity_close, module(03_hdfe)
