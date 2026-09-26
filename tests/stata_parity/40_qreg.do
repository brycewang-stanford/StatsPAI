* tests/stata_parity/40_qreg.do
*
* Module 40: Quantile regression (median).
*   StatsPAI:  sp.qreg(quantile=0.5, vce="nid")
*   R:         quantreg::rq(tau=0.5, summary(se="nid"))
*   Stata:     qreg y x1 x2, vce(robust)
*
* All three are the Hendricks-Koenker sandwich with the Hall-Sheather
* bandwidth (vce(robust) uses the default fitted density method).

version 18
clear all

do _common.do
stata_parity_init, module(40_qreg)
stata_parity_open, module(40_qreg)

import delimited "${STATA_PARITY_DATA}/40_qreg.csv", clear case(preserve)

local n = _N

qreg y x1 x2, vce(robust)

local b0  = _b[_cons]
local se0 = _se[_cons]
local b1  = _b[x1]
local se1 = _se[x1]
local b2  = _b[x2]
local se2 = _se[x2]

stata_parity_row, stat(beta_intercept) est(`b0') std(`se0') nob(`n')
stata_parity_row, stat(beta_x1)        est(`b1') std(`se1') nob(`n')
stata_parity_row, stat(beta_x2)        est(`b2') std(`se2') nob(`n')

stata_parity_extra, key(quantile) val(0.5)
stata_parity_extra, key(se_method) val("Hendricks-Koenker sandwich, Hall-Sheather bandwidth")
stata_parity_extra, key(stata_command) val("qreg y x1 x2, vce(robust)")

stata_parity_close, module(40_qreg)
