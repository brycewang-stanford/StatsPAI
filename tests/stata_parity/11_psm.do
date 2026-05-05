* tests/stata_parity/11_psm.do
*
* Module 11: PSM 1:1 NN with replacement (NSW-DW replica).
*   StatsPAI:  sp.psm
*   R:         MatchIt::matchit(method="nearest", distance="glm", replace=TRUE)
*   Stata:     teffects psmatch (Stata built-in, with replacement by default)
*
* Tolerance: rel < 1e-2 on ATT.

version 18
clear all

do _common.do
stata_parity_init, module(11_psm)
stata_parity_open, module(11_psm)

import delimited "${STATA_PARITY_DATA}/11_psm.csv", clear case(preserve)

teffects psmatch (re78) (treat age education black hispanic married re74 re75, logit), atet nneighbor(1)

local n = e(N)
local att = _b[r1vs0.treat]
local se  = _se[r1vs0.treat]
local lo = `att' - ${STATA_PARITY_Z95} * `se'
local hi = `att' + ${STATA_PARITY_Z95} * `se'

stata_parity_row, stat(att_psm) est(`att') std(`se') cilo(`lo') cihi(`hi') nob(`n')

count if treat == 1
local n_treated = r(N)
count if treat == 0
local n_control = r(N)

stata_parity_row, stat(n_treated) est(`n_treated') nob(`n')
stata_parity_row, stat(n_control) est(`n_control') nob(`n')

stata_parity_extra, key(distance) val(logit)
stata_parity_extra, key(method)   val(nearest)
stata_parity_extra, key(replace)  val(TRUE)
stata_parity_extra, key(ratio)    val(1)
stata_parity_extra, key(stata_command) val("teffects psmatch (re78) (treat ..., logit), atet nneighbor(1)")

stata_parity_close, module(11_psm)
