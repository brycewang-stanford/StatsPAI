* tests/stata_parity/28_frontier.do
*
* Module 28: Cross-sectional half-normal stochastic frontier.
*   StatsPAI:  sp.frontier(udist="hnormal")
*   R:         sfaR::sfacross(udist="hnormal", S=1)
*   Stata:     frontier (built-in) or sfcross (SSC). Use built-in
*              `frontier` for parsimony; default is half-normal,
*              production frontier (cost option for cost frontier).
*
* Tolerance: rel < 1e-2 on betas; rel < 5e-2 on sigma_u/sigma_v.

version 18
clear all

do _common.do
stata_parity_init, module(28_frontier)
stata_parity_open, module(28_frontier)

import delimited "${STATA_PARITY_DATA}/28_frontier.csv", clear case(preserve)

frontier lny lnk lnl, distribution(hnormal)

local n = e(N)
matrix B = e(b)
matrix V = e(V)

* B layout: lny:lnk lny:lnl lny:_cons lnsig2v:_cons lnsig2u:_cons
local b_int = B[1, "lny:_cons"]
local se_int = sqrt(V["lny:_cons", "lny:_cons"])
local b_lnk = B[1, "lny:lnk"]
local se_lnk = sqrt(V["lny:lnk", "lny:lnk"])
local b_lnl = B[1, "lny:lnl"]
local se_lnl = sqrt(V["lny:lnl", "lny:lnl"])

* Stata's `frontier` parameterises log(sigma_u^2) / log(sigma_v^2).
local lnsig2v = B[1, "lnsig2v:_cons"]
local lnsig2u = B[1, "lnsig2u:_cons"]
local sigma_u = sqrt(exp(`lnsig2u'))
local sigma_v = sqrt(exp(`lnsig2v'))
local lambda  = `sigma_u' / `sigma_v'

* Mean efficiency via JLMS estimator (predict, te).
predict te_jlms, te
quietly summarize te_jlms
local mean_eff = r(mean)

stata_parity_row, stat(beta_intercept) est(`b_int') std(`se_int') nob(`n')
stata_parity_row, stat(beta_lnk)       est(`b_lnk') std(`se_lnk') nob(`n')
stata_parity_row, stat(beta_lnl)       est(`b_lnl') std(`se_lnl') nob(`n')
stata_parity_row, stat(sigma_u)        est(`sigma_u') nob(`n')
stata_parity_row, stat(sigma_v)        est(`sigma_v') nob(`n')
stata_parity_row, stat(lambda)         est(`lambda')  nob(`n')
stata_parity_row, stat(mean_efficiency) est(`mean_eff') nob(`n')

stata_parity_extra, key(distribution) val(half-normal)
stata_parity_extra, key(S) val(1)
stata_parity_extra, key(package) val("Stata frontier (built-in)")
stata_parity_extra, key(stata_command) val("frontier lny lnk lnl, distribution(hnormal)")

stata_parity_close, module(28_frontier)
