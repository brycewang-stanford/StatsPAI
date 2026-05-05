* tests/stata_parity/25_lmm.do
*
* Module 25: Linear mixed model.
*   StatsPAI:  sp.mixed (REML)
*   R:         lme4::lmer (REML)
*   Stata:     mixed (REML)
*
* Tolerance: rel < 1e-3 on fixed effects and ICC.

version 18
clear all

do _common.do
stata_parity_init, module(25_lmm)
stata_parity_open, module(25_lmm)

import delimited "${STATA_PARITY_DATA}/25_lmm.csv", clear case(preserve)

mixed y x1 || gid:, reml

local n = e(N)
matrix B = e(b)
matrix V = e(V)
local b_int = B[1, "y:_cons"]
local se_int = sqrt(V["y:_cons", "y:_cons"])
local b_x1  = B[1, "y:x1"]
local se_x1 = sqrt(V["y:x1", "y:x1"])

* Variance components: e(b)["lns1_1_1:_cons"] is log(sd_group); square+exp twice.
* Easier: estat icc gives ICC directly.
local sigma2_group = (exp(B[1, "lns1_1_1:_cons"]))^2
local sigma2_resid = (exp(B[1, "lnsig_e:_cons"]))^2
local icc_val = `sigma2_group' / (`sigma2_group' + `sigma2_resid')

local ll = e(ll)

* mixed posts e(N_g) as a 1xL matrix when there is at least one
* grouping level (one column per level). Fetch the scalar via matrix.
matrix NG = e(N_g)
local n_groups = NG[1, 1]

stata_parity_row, stat(beta_intercept) est(`b_int') std(`se_int') nob(`n')
stata_parity_row, stat(beta_x1)        est(`b_x1')  std(`se_x1')  nob(`n')
stata_parity_row, stat(logLik)         est(`ll')                  nob(`n')
stata_parity_row, stat(icc)            est(`icc_val')             nob(`n')

stata_parity_extra, key(method) val(REML)
stata_parity_extra_num, key(n_groups) val(`n_groups')
stata_parity_extra, key(stata_command) val("mixed y x1 || gid:, reml")

stata_parity_close, module(25_lmm)
