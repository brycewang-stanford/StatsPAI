* tests/stata_parity/04_csdid.do
*
* Module 04: CS-DiD simple ATT.
*   StatsPAI:  sp.callaway_santanna(...).simple_att
*   R:         did::att_gt + did::aggte(type="simple"), method=reg
*   Stata:     csdid + estat simple, method(reg)
*
* csdid uses Wald-style asymptotic SE; matches did::aggte with
* bstrap=FALSE.

version 18
clear all

do _common.do
stata_parity_init, module(04_csdid)
stata_parity_open, module(04_csdid)

import delimited "${STATA_PARITY_DATA}/04_csdid.csv", clear case(preserve)

* csdid expects the never-treated cohort coded as 0 (or as "Inf" string,
* not supported in Stata). Inspect: r_parity uses first_treat = 0 for
* never-treated. csdid syntax: outcome ivar(id) time(t) gvar(g).
* Default control group = never-treated (matches R's control_group="nevertreated").
csdid lemp, ivar(countyreal) time(year) gvar(first_treat) method(reg)

* `estat simple` aggregates ATT(g,t) over post-treatment cells.
estat simple

* The post-`estat simple` ereturn matrix is r(b)/r(V) (not e()).
matrix B = r(b)
matrix V = r(V)
local bv = B[1, 1]
local sv = sqrt(V[1, 1])
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'

count
local n = r(N)

stata_parity_row, stat(simple_ATT) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')

stata_parity_extra, key(estimator) val(reg)
stata_parity_extra, key(control_group) val(nevertreated)
stata_parity_extra, key(stata_command) val("csdid, ivar(countyreal) time(year) gvar(first_treat) method(reg) | estat simple")

stata_parity_close, module(04_csdid)
