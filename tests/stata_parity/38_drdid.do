* tests/stata_parity/38_drdid.do
*
* Module 38: Doubly robust DiD, Sant'Anna-Zhao improved estimator.
*   StatsPAI:  sp.drdid(..., method="imp")
*   R:         DRDID::drdid_imp_panel
*   Stata:     drdid, ivar(...) time(...) treatment(...) drimp
*
* The committed results/38_drdid_Stata.json artifact was materialized with
* Stata/MP 18 from this script after the licensed Stata executable became
* available locally.

version 18
clear all

do _common.do
stata_parity_init, module(38_drdid)
stata_parity_open, module(38_drdid)

import delimited "${STATA_PARITY_DATA}/38_drdid.csv", clear case(preserve)

* drdid defaults to drimp; pass it explicitly to match
* DRDID::drdid_imp_panel and StatsPAI method="imp".
drdid y x, ivar(id) time(post) treatment(treated) drimp

matrix B = e(b)
matrix V = e(V)
local bv = B[1, 1]
local sv = sqrt(V[1, 1])
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'

count
local n = r(N)

stata_parity_row, stat(att) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')
stata_parity_row, stat(ci_lower) est(`lo') nob(`n')
stata_parity_row, stat(ci_upper) est(`hi') nob(`n')

stata_parity_extra, key(method) val(drdid drimp)
stata_parity_extra, key(stata_command) val("drdid y x, ivar(id) time(post) treatment(treated) drimp")
stata_parity_extra, key(stata_bridge_status) val("materialized with licensed Stata/MP 18")

stata_parity_close, module(38_drdid)
