* tests/stata_parity/09_rddensity.do
*
* Module 09: RD density manipulation test (Cattaneo-Jansson-Ma).
*   StatsPAI:  sp.rddensity
*   R:         rddensity::rddensity
*   Stata:     rddensity (CCT authors, SSC)
*
* Tolerance: rel < 1e-3 (iterative bandwidth selection).

version 18
clear all

do _common.do
stata_parity_init, module(09_rddensity)
stata_parity_open, module(09_rddensity)

import delimited "${STATA_PARITY_DATA}/09_rddensity.csv", clear case(preserve)

rddensity margin, c(0)

count
local n = r(N)

* rddensity Stata e()-returns:
*   e(f_ql), e(f_qr) -- bias-corrected density at cutoff (left/right)
*   e(pv_q)          -- robust p-value
*   e(h_l), e(h_r)   -- left/right bandwidths
local dl = e(f_ql)
local dr = e(f_qr)
local pj = e(pv_q)
local hl = e(h_l)
local hr = e(h_r)
local diff = `dr' - `dl'

stata_parity_row, stat(density_diff)   est(`diff') nob(`n')
stata_parity_row, stat(density_left)   est(`dl')   nob(`n')
stata_parity_row, stat(density_right)  est(`dr')   nob(`n')
stata_parity_row, stat(test_pvalue)    est(`pj')   nob(`n')
stata_parity_row, stat(bandwidth_left) est(`hl')   nob(`n')
stata_parity_row, stat(bandwidth_right)est(`hr')   nob(`n')

stata_parity_extra, key(test_kind) val("Cattaneo-Jansson-Ma (2020)")
stata_parity_extra, key(stata_command) val("rddensity margin, c(0)")

stata_parity_close, module(09_rddensity)
