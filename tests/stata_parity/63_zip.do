* tests/stata_parity/63_zip.do
*
* Module 63: Zero-inflated Poisson (logit inflation).
*   StatsPAI:  sp.zip_model(inflate=['z'])
*   R:         pscl::zeroinfl(dist='poisson', link='logit')
*   Stata:     zip y x1 x2, inflate(z)

version 18
clear all
do _common.do
stata_parity_init, module(63_zip)
stata_parity_open, module(63_zip)

import delimited "${STATA_PARITY_DATA}/63_zip.csv", clear case(preserve)
local n = _N

zip y x1 x2, inflate(z)

local bc0 = _b[y:_cons]
local sc0 = _se[y:_cons]
local bc1 = _b[y:x1]
local sc1 = _se[y:x1]
local bc2 = _b[y:x2]
local sc2 = _se[y:x2]
local bi0 = _b[inflate:_cons]
local si0 = _se[inflate:_cons]
local biz = _b[inflate:z]
local siz = _se[inflate:z]

stata_parity_row, stat(beta_count_intercept)   est(`bc0') std(`sc0') nob(`n')
stata_parity_row, stat(beta_count_x1)          est(`bc1') std(`sc1') nob(`n')
stata_parity_row, stat(beta_count_x2)          est(`bc2') std(`sc2') nob(`n')
stata_parity_row, stat(beta_inflate_intercept) est(`bi0') std(`si0') nob(`n')
stata_parity_row, stat(beta_inflate_z)         est(`biz') std(`siz') nob(`n')

stata_parity_extra, key(count_dist) val(poisson)
stata_parity_extra, key(inflate_link) val(logit)
stata_parity_extra, key(stata_command) val("zip y x1 x2, inflate(z)")

stata_parity_close, module(63_zip)
