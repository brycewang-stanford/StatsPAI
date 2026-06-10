* tests/stata_parity/34_lp.do
*
* Module 34: Local projections (Jorda 2005).
*   StatsPAI:  sp.local_projections(outcome="y", shock="x",
*                                   controls=["y_lag"], horizons=5)
*   R:         lpirfs::lp_lin (Cholesky-orthogonalised, identification gap)
*   Stata:     equation-by-equation regress (direct-OLS diagnostic);
*              also captures Stata's `lpirf` IRF for the canonical
*              Stata reference under Cholesky identification.
*
* Tolerance: direct-OLS point diagnostics match sp's direct path at
*   machine precision. The R/Python headline rows use lpirfs Cholesky.

version 18
clear all

do _common.do
stata_parity_init, module(34_lp)
stata_parity_open, module(34_lp)

import delimited "${STATA_PARITY_DATA}/34_lp.csv", clear case(preserve)

gen t = _n
tsset t

* Build forward leads of y so we can regress y_{t+h} on x_t with the
* y_lag control. This mirrors sp.local_projections exactly.
gen y_h0 = y
forvalues h = 1/5 {
    gen y_h`h' = F`h'.y
}

local n_obs = _N

* Horizon loop: irf_h0 .. irf_h5 are the OLS coefficients on x in
* the regression y_{t+h} = a + b * x_t + c * y_lag + e.
forvalues h = 0/5 {
    regress y_h`h' x y_lag
    local b  = _b[x]
    local se = _se[x]
    local n_h = e(N)
    stata_parity_row, stat(irf_direct_ols_h`h') est(`b') std(`se') nob(`n_h')
}

* Also fit Stata's canonical lpirf for completeness and record
* coefficients in the extras block.
cap lpirf y x, step(5) lags(1)
if _rc == 0 {
    local lpirf_h1 = _b[y:F.x]
    local lpirf_h2 = _b[y:F2.x]
    local lpirf_h3 = _b[y:F3.x]
    local lpirf_h4 = _b[y:F4.x]
    local lpirf_h5 = _b[y:F5.x]
    stata_parity_extra_num, key(lpirf_h1) val(`lpirf_h1')
    stata_parity_extra_num, key(lpirf_h2) val(`lpirf_h2')
    stata_parity_extra_num, key(lpirf_h3) val(`lpirf_h3')
    stata_parity_extra_num, key(lpirf_h4) val(`lpirf_h4')
    stata_parity_extra_num, key(lpirf_h5) val(`lpirf_h5')
}

stata_parity_extra, key(method) val("regress y_{t+h} on x_t + y_lag")
stata_parity_extra, key(stata_command) val("for h=0/5: regress y_h# x y_lag; lpirf y x, step(5) lags(1)")
stata_parity_extra, key(stata_direct_ols_reference) ///
    val("Rows named irf_direct_ols_h* record regress y_{t+h} x y_lag point estimates; Stata lpirf Cholesky coefficients are stored in extras as lpirf_h*.")

stata_parity_close, module(34_lp)
