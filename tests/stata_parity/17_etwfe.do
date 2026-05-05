* tests/stata_parity/17_etwfe.do
*
* Module 17: Wooldridge ETWFE.
*   StatsPAI:  sp.wooldridge_did(...).att
*   R:         etwfe::etwfe + etwfe::emfx
*   Stata:     jwdid (Wooldridge port by Fernando Rios-Avila)
*
* Tolerance: rel < 1e-3 on pooled ATT.

version 18
clear all

do _common.do
stata_parity_init, module(17_etwfe)
stata_parity_open, module(17_etwfe)

import delimited "${STATA_PARITY_DATA}/17_etwfe.csv", clear case(preserve)

* jwdid: Wooldridge ETWFE estimator.
* Syntax: jwdid depvar [if] [in], ivar(panelvar) tvar(timevar) gvar(cohortvar)
jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat) cluster(countyreal)

* Aggregate to a single pooled ATT.
estat simple

local n = e(N)
matrix B = r(b)
matrix V = r(V)
local bv = B[1, 1]
local sv = sqrt(V[1, 1])
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'

stata_parity_row, stat(att_etwfe) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')

stata_parity_extra, key(method) val(jwdid)
stata_parity_extra, key(stata_command) val("jwdid lemp, ivar(countyreal) tvar(year) gvar(first_treat) | estat simple")

stata_parity_close, module(17_etwfe)
