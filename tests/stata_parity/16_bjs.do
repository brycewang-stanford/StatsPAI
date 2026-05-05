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

* did_imputation syntax: did_imputation depvar id time gvar [, options]
* `autosample` drops never-imputable observations (those that have no
* untreated comparison) silently, matching the R didimputation default.
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
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
stata_parity_row, stat(att_bjs) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')

stata_parity_extra, key(method) val(did_imputation)
stata_parity_extra, key(stata_command) val("did_imputation lemp countyreal year first_treat")

stata_parity_close, module(16_bjs)
