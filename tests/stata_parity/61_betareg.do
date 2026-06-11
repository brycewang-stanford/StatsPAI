* tests/stata_parity/61_betareg.do
*
* Module 61: Beta regression (logit mean link, log-link scale).
*   StatsPAI:  sp.betareg(link='logit')      (log-link precision _cons_phi)
*   R:         betareg::betareg(link='logit', link.phi='log')
*   Stata:     betareg y x1 x2               (link(logit) scalelink(log) defaults)
*
* All three sides estimate the same (beta, ln phi) parameterisation, so
* the ln_phi row including its SE is directly comparable.

version 18
clear all
do _common.do
stata_parity_init, module(61_betareg)
stata_parity_open, module(61_betareg)

import delimited "${STATA_PARITY_DATA}/61_betareg.csv", clear case(preserve)
local n = _N

* nrtolerance tightens the scaled-gradient stop rule (default 1e-5) so
* the ML optimum matches the tightly converged sp/R references.
betareg y x1 x2, nrtolerance(1e-13)

local b0 = _b[y:_cons]
local se0 = _se[y:_cons]
local b1 = _b[y:x1]
local se1 = _se[y:x1]
local b2 = _b[y:x2]
local se2 = _se[y:x2]
local lnphi = _b[scale:_cons]
local selnphi = _se[scale:_cons]

stata_parity_row, stat(beta_intercept) est(`b0') std(`se0') nob(`n')
stata_parity_row, stat(beta_x1)        est(`b1') std(`se1') nob(`n')
stata_parity_row, stat(beta_x2)        est(`b2') std(`se2') nob(`n')
stata_parity_row, stat(ln_phi)         est(`lnphi') std(`selnphi') nob(`n')

stata_parity_extra, key(link) val(logit)
stata_parity_extra, key(phi_link) val(log)
stata_parity_extra, key(stata_command) val("betareg y x1 x2, nrtolerance(1e-13)")

stata_parity_close, module(61_betareg)
