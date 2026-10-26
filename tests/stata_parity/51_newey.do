* tests/stata_parity/51_newey.do
* Module 51: Newey-West HAC OLS.
*   StatsPAI: sp.regress(robust="hac", lags=4)
*   R:        sandwich::NeweyWest(lag=4, prewhite=FALSE, adjust=FALSE / TRUE)
*
* newey scales the HAC covariance by N/(N-K). The beta_* rows join the
* unscaled default of StatsPAI and sandwich (a registered convention gap);
* the beta_*_small rows join hac_small=True and adjust=TRUE.
*   Stata:    newey y x, lag(4)

version 18
clear all
do _common.do
stata_parity_init, module(51_newey)
stata_parity_open, module(51_newey)

import delimited "${STATA_PARITY_DATA}/51_newey.csv", clear case(preserve)
tsset t

local n = _N

newey y x, lag(4)

local b0 = _b[_cons]
local se0 = _se[_cons]
local bx = _b[x]
local sex = _se[x]

stata_parity_row, stat(beta_intercept) est(`b0') std(`se0') nob(`n')
stata_parity_row, stat(beta_x)         est(`bx') std(`sex') nob(`n')
stata_parity_row, stat(beta_intercept_small) est(`b0') std(`se0') nob(`n')
stata_parity_row, stat(beta_x_small)         est(`bx') std(`sex') nob(`n')

stata_parity_extra, key(vcov) val(NeweyWest)
stata_parity_extra, key(lag) val(4)
stata_parity_extra, key(stata_command) val("newey y x, lag(4)")

stata_parity_close, module(51_newey)
