* tests/stata_parity/83_lpdid.do
*
* Module 83: LP-DiD event study.
*   StatsPAI:  sp.lp_did
*   R:         direct transcription (no LP-DiD package installed on the R side)
*   Stata:     lpdid (Busch & Girardi, SSC S459273) -- the authors' own
*              companion package to Dube, Girardi, Jorda & Taylor (2025, JAE)
*
* This is the reference that matters: the Stata side is the published
* implementation, not a transcription. Post-treatment horizons agree with
* sp.lp_did to ~1e-9 on both the coefficient and the standard error.
*
* The pre-treatment horizon is reported but NOT registered as a headline
* comparison: lpdid and sp.lp_did select different samples for the placebo
* lead (see 83_lpdid.py), so the two report the same coefficient off
* different row counts. That difference is recorded rather than tuned away.

version 18
clear all

do _common.do
stata_parity_init, module(83_lpdid)
stata_parity_open, module(83_lpdid)

import delimited "${STATA_PARITY_DATA}/83_lpdid.csv", clear case(preserve)
xtset unit time

quietly lpdid y, unit(unit) time(time) treat(d) pre_window(2) post_window(3)

matrix R = e(results)
local rownames : rownames R

* Rows are pre2, pre1, tau0, tau1, tau2, tau3. pre1 is the omitted base
* period and carries no estimate, so it is skipped.
local horizons "-2 . 0 1 2 3"

local i = 1
foreach h of local horizons {
    if "`h'" != "." {
        local est = R[`i', 1]
        local se  = R[`i', 2]
        local lo  = R[`i', 5]
        local hi  = R[`i', 6]
        local nob = R[`i', 7]
        stata_parity_row, stat(lpdid_h`h'_att) est(`est') std(`se') ///
            cilo(`lo') cihi(`hi') nob(`nob')
    }
    local ++i
}

stata_parity_close, module(83_lpdid)
