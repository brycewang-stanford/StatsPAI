* tests/stata_parity/16_bjs.do
*
* Module 16: BJS imputation (Borusyak-Jaravel-Spiess).
*   StatsPAI:  sp.bjs_pretrend_joint(...).att
*   R:         didimputation::did_imputation
*   Stata:     did_imputation (Borusyak's Stata port)
*
* Tolerance: rel < 1e-3.

version 18
clear all

do _common.do
stata_parity_init, module(16_bjs)
stata_parity_open, module(16_bjs)

import delimited "${STATA_PARITY_DATA}/16_bjs.csv", clear case(preserve)

* The shared R/Python CSV encodes never-treated units as first_treat==0.
* Borusyak's Stata did_imputation expects missing Ei for never-treated
* units, so recode before estimation; otherwise those units are read as
* treated from period 0 and the pooled tau is not like-for-like.
replace first_treat = . if first_treat == 0

* did_imputation syntax: did_imputation depvar id time gvar [, options]
* `autosample` drops never-imputable observations (those that have no
* untreated comparison) silently.
did_imputation lemp countyreal year first_treat, autosample

* did_imputation posts e(b) and e(V); the headline pooled ATT row is
* e(b)["tau", *] -- check specific layout below. Default with no
* horizons supplied is a single pooled "tau" row.
local n = e(N)
matrix B = e(b)
matrix V = e(V)
local rownames : colnames B

* The estimator stores a single tau row.
local nm : word 1 of `rownames'
local bv = B[1, "`nm'"]
local sv = sqrt(V["`nm'", "`nm'"])
stata_parity_row, stat(att_bjs) est(`bv') nob(`n')
stata_parity_row, stat(se_stata_did_imputation) est(`sv') nob(`n')

stata_parity_extra, key(method) val(did_imputation)
stata_parity_extra, key(never_treated_coding) val("replace first_treat = . if first_treat == 0")
stata_parity_extra, key(stata_command) val("replace first_treat = . if first_treat == 0; did_imputation lemp countyreal year first_treat, autosample")

stata_parity_close, module(16_bjs)
