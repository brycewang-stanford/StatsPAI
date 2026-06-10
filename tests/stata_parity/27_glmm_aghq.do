* tests/stata_parity/27_glmm_aghq.do
*
* Module 27: GLMM logit (Adaptive Gauss-Hermite Quadrature, n=8).
*   StatsPAI:  sp.melogit(nAGQ=8)
*   R:         lme4::glmer(..., nAGQ=8)
*   Stata:     melogit, intpoints(8)
*
* Tolerance: rel < 1e-6 on betas; rel < 5e-2 on SE.

version 18
clear all

do _common.do
stata_parity_init, module(27_glmm_aghq)
stata_parity_open, module(27_glmm_aghq)

import delimited "${STATA_PARITY_DATA}/27_glmm_aghq.csv", clear case(preserve)

local n = _N

* AGHQ with 8 integration points
melogit y x1 || gid:, intpoints(8)

local b0  = _b[y:_cons]
local se0 = _se[y:_cons]
local b1  = _b[y:x1]
local se1 = _se[y:x1]
local ll  = e(ll)

stata_parity_row, stat(beta_intercept) est(`b0') std(`se0') nob(`n')
stata_parity_row, stat(beta_x1)        est(`b1') std(`se1') nob(`n')
stata_parity_row, stat(logLik)         est(`ll') nob(`n')

stata_parity_extra, key(family) val("binomial(logit)")
stata_parity_extra, key(intpoints) val(8)
stata_parity_extra, key(stata_command) val("melogit y x1 || gid:, intpoints(8)")

stata_parity_close, module(27_glmm_aghq)
