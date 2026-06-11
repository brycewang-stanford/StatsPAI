* tests/stata_parity/64_zinb.do
*
* Module 64: Zero-inflated negative binomial (logit inflation).
*   StatsPAI:  sp.zinb(inflate=['z'])
*   R:         pscl::zeroinfl(dist='negbin', link='logit')
*   Stata:     zinb y x1 x2, inflate(z)
*
* Dispersion conventions differ (pscl theta vs Stata lnalpha), so the
* alpha row is exported on the common alpha = 1/theta = exp(lnalpha)
* scale as a point-estimate diagnostic.

version 18
clear all
do _common.do
stata_parity_init, module(64_zinb)
stata_parity_open, module(64_zinb)

import delimited "${STATA_PARITY_DATA}/64_zinb.csv", clear case(preserve)
local n = _N

* nrtolerance tightens the scaled-gradient stop rule (default 1e-5) so
* the ML optimum matches the tightly converged sp/R references.
zinb y x1 x2, inflate(z) nrtolerance(1e-13)

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
local alpha = exp(_b[/lnalpha])

stata_parity_row, stat(beta_count_intercept)   est(`bc0') std(`sc0') nob(`n')
stata_parity_row, stat(beta_count_x1)          est(`bc1') std(`sc1') nob(`n')
stata_parity_row, stat(beta_count_x2)          est(`bc2') std(`sc2') nob(`n')
stata_parity_row, stat(beta_inflate_intercept) est(`bi0') std(`si0') nob(`n')
stata_parity_row, stat(beta_inflate_z)         est(`biz') std(`siz') nob(`n')
stata_parity_row, stat(alpha)                  est(`alpha') nob(`n')

stata_parity_extra, key(count_dist) val(negbin)
stata_parity_extra, key(inflate_link) val(logit)
stata_parity_extra, key(alpha_note) val("alpha = 1/theta (Stata lnalpha scale)")
stata_parity_extra, key(stata_command) val("zinb y x1 x2, inflate(z) nrtolerance(1e-13)")

stata_parity_close, module(64_zinb)
