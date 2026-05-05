* tests/stata_parity/22_sensemakr.do
*
* Module 22: sensemakr robustness (Cinelli-Hazlett).
*   StatsPAI:  sp.sensemakr
*   R:         sensemakr::sensemakr
*   Stata:     sensemakr (Cinelli's port)
*
* Tolerance: rel < 5e-2 on RV_q and partial R^2.

version 18
clear all

do _common.do
stata_parity_init, module(22_sensemakr)
stata_parity_open, module(22_sensemakr)

import delimited "${STATA_PARITY_DATA}/22_sensemakr.csv", clear case(preserve)

* Stata sensemakr syntax: sensemakr depvar regressors, treat(...) benchmark(...).
* The regressor list must include the treatment variable. Option name
* is `treat()`, not `treatment()`.
* Limit kd to 1 (default 1 2 3) to avoid the iterate_bounds() error
* on this NSW-DW DGP at kd=2 (implied R^2_yz.dx > 1, terminates the
* command before posting e() returns).
sensemakr re78 treat age education black hispanic married re74 re75, ///
    treat(treat) benchmark(re74) alpha(0.05) kd(1) ky(1)

count
local n = r(N)

* sensemakr e()-returns:
*   e(treat_coef), e(treat_se), e(r2yd_x), e(rv_q), e(rv_qa), e(dof)
* Benchmark partial R^2 lives in e(bounds) [kd ky r2dz_x r2yz_dx ...]
local b_t   = e(treat_coef)
local se_t  = e(treat_se)
local r2yd_x = e(r2yd_x)
local rvq   = e(rv_q)
local rvqa  = e(rv_qa)
local dof   = e(dof)
local t_t   = `b_t' / `se_t'

matrix BM = e(bounds)
local r2dz_x  = BM[1, 3]
local r2yz_dx = BM[1, 4]

stata_parity_row, stat(beta_treat)    est(`b_t') std(`se_t') nob(`n')
stata_parity_row, stat(t_treat)       est(`t_t') nob(`n')
stata_parity_row, stat(partial_r2_yd) est(`r2yd_x') nob(`n')
stata_parity_row, stat(rv_q)          est(`rvq') nob(`n')
stata_parity_row, stat(rv_q_alpha)    est(`rvqa') nob(`n')
stata_parity_row, stat(benchmark_re74_partial_r2_Y) est(`r2dz_x') nob(`n')
stata_parity_row, stat(benchmark_re74_partial_r2_D) est(`r2yz_dx') nob(`n')

stata_parity_extra, key(benchmark) val(re74)
stata_parity_extra, key(alpha) val(0.05)
stata_parity_extra, key(q) val(1)
stata_parity_extra, key(stata_command) val("regress ... ; sensemakr, treat(treat) benchmark(re74) q(1) alpha(0.05)")

stata_parity_close, module(22_sensemakr)
