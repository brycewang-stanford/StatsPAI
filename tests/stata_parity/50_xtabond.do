* tests/stata_parity/50_xtabond.do
*
* Module 50: Arellano-Bond difference GMM (one-step, robust).
*   StatsPAI: sp.xtabond(method="difference", twostep=False)
*   Stata:    xtabond y x, lags(1) twostep(no) vce(robust)
*
* Tolerance: rel < 5e-2 on coefficients. Both sides use the FULL set of
* available deeper lags as GMM instruments: Stata's `xtabond` default is
* GMM-style L(2/.).y, and sp.xtabond is called with gmm_lags=(2, None).
* (Pinning sp to the old (2,5) cap was the source of session finding #12.)

version 18
clear all
do _common.do
stata_parity_init, module(50_xtabond)
stata_parity_open, module(50_xtabond)

import delimited "${STATA_PARITY_DATA}/50_xtabond.csv", clear case(preserve)
xtset id time

local n = _N

* xtabond defaults: lags(1), GMM-style instruments for level vars
* (i.e., L.y as endogenous → L(2/.).y as instruments).
xtabond y x, lags(1) vce(robust)

local bx = _b[x]
local sex = _se[x]
local bL = _b[L.y]
local seL = _se[L.y]

stata_parity_row, stat(beta_y_lag) est(`bL') std(`seL') nob(`n')
stata_parity_row, stat(beta_x)     est(`bx') std(`sex') nob(`n')

stata_parity_extra, key(method) val("Arellano-Bond difference GMM")
stata_parity_extra, key(step) val("one-step")
stata_parity_extra, key(stata_command) val("xtabond y x, lags(1) vce(robust)")

stata_parity_close, module(50_xtabond)
