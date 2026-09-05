* tests/stata_parity/29_panel_sfa.do
*
* Module 29: Panel SFA (Pitt-Lee 1981, time-invariant inefficiency).
*   StatsPAI:  sp.xtfrontier(model="pittlee")
*   R:         frontier::sfa(..., timeEffect=FALSE, truncNorm=FALSE)
*   Stata:     xtfrontier ..., ti
*
* Tolerance: rel < 1e-3 on headline slopes; intercept and sigma rows
* are retained as Stata scale diagnostics.

version 18
clear all

do _common.do
stata_parity_init, module(29_panel_sfa)
stata_parity_open, module(29_panel_sfa)

import delimited "${STATA_PARITY_DATA}/29_panel_sfa.csv", clear case(preserve)

xtset unit year

local n = _N

* xtfrontier with `ti` = time-invariant Pitt-Lee.
* Default distribution is half-normal.
* xtfrontier's ti model is Battese-Coelli (1988) with a truncated-normal
* u_i (free /mu). StatsPAI xtfrontier(model="ti") and R frontier::sfa
* (truncNorm = FALSE) fit the half-normal Pitt-Lee (1981) model, so the
* like-for-like Stata reference constrains mu = 0. Without it the
* intercept, sigma_u, and their SEs answer a different question (the old
* "xtfrontier scale" diagnostic rows).
constraint 1 [mu]_cons = 0
xtfrontier lny lnk lnl, ti constraints(1)

local b0  = _b[lny:_cons]
local se0 = _se[lny:_cons]
local bk  = _b[lny:lnk]
local sek = _se[lny:lnk]
local bl  = _b[lny:lnl]
local sel = _se[lny:lnl]

* sigma_u and sigma_v are reported as e(b)[1, sigma_u:_cons] etc.
* xtfrontier stores them as scalars in e().
local sigma_u = e(sigma_u)
local sigma_v = e(sigma_v)
* fallback: extract from coefficient vector using log-parameterisation
if "`sigma_u'" == "" {
    local ln_sig_u2 = _b[lnsig2u:_cons]
    local sigma_u  = sqrt(exp(`ln_sig_u2'))
}
if "`sigma_v'" == "" {
    local ln_sig_v2 = _b[lnsig2v:_cons]
    local sigma_v  = sqrt(exp(`ln_sig_v2'))
}

stata_parity_row, stat(beta_intercept) est(`b0') std(`se0') nob(`n')
stata_parity_row, stat(beta_lnk)       est(`bk') std(`sek') nob(`n')
stata_parity_row, stat(beta_lnl)       est(`bl') std(`sel') nob(`n')
stata_parity_row, stat(sigma_u)        est(`sigma_u') nob(`n')
stata_parity_row, stat(sigma_v)        est(`sigma_v') nob(`n')

stata_parity_extra, key(distribution) val("half-normal")
stata_parity_extra, key(timeinvariant) val("true")
stata_parity_extra, key(stata_command) val("constraint 1 [mu]_cons = 0; xtfrontier lny lnk lnl, ti constraints(1)")
stata_parity_extra, key(model_note) val("half-normal Pitt-Lee via mu = 0 constraint on the Battese-Coelli ti model; ml_method d2 (analytic Hessian)")

stata_parity_close, module(29_panel_sfa)
