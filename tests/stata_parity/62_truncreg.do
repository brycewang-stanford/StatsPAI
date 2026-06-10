* tests/stata_parity/62_truncreg.do
*
* Module 62: Truncated regression (left-truncated at 0).
*   StatsPAI:  sp.truncreg(ll=0)   (ln_sigma; sigma row delta-mapped)
*   R:         truncreg::truncreg(point=0, direction='left')
*   Stata:     truncreg y x1 x2, ll(0)

version 18
clear all
do _common.do
stata_parity_init, module(62_truncreg)
stata_parity_open, module(62_truncreg)

import delimited "${STATA_PARITY_DATA}/62_truncreg.csv", clear case(preserve)
local n = _N

truncreg y x1 x2, ll(0)

local b0 = _b[eq1:_cons]
local se0 = _se[eq1:_cons]
local b1 = _b[eq1:x1]
local se1 = _se[eq1:x1]
local b2 = _b[eq1:x2]
local se2 = _se[eq1:x2]
local sg = _b[/sigma]
local sesg = _se[/sigma]

stata_parity_row, stat(beta_intercept) est(`b0') std(`se0') nob(`n')
stata_parity_row, stat(beta_x1)        est(`b1') std(`se1') nob(`n')
stata_parity_row, stat(beta_x2)        est(`b2') std(`se2') nob(`n')
stata_parity_row, stat(sigma)          est(`sg') std(`sesg') nob(`n')

stata_parity_extra, key(truncation) val("left at 0")
stata_parity_extra, key(stata_command) val("truncreg y x1 x2, ll(0)")

stata_parity_close, module(62_truncreg)
