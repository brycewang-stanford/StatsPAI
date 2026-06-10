* tests/stata_parity/52_scm_unique.do
*
* Module 52: Classical SCM on an identified unique-solution DGP.
*   StatsPAI:  sp.synth(method="classic")
*   R:         Synth::synth + Synth::dataprep
*   Stata:     synth (Abadie/Diamond/Hainmueller, SSC)
*
* The fixture is the strict-parity counterpart to module 07: the treated
* unit's pre-period path is exactly a unique convex combination of donor0,
* donor1, and donor2.  The Stata bridge uses the same per-period predictor
* recipe as the R Synth reference.

version 18
clear all

do _common.do
stata_parity_init, module(52_scm_unique)
stata_parity_open, module(52_scm_unique)

import delimited "${STATA_PARITY_DATA}/52_scm_unique.csv", clear case(preserve) ///
    stringcols(1)

gen unit_num = .
replace unit_num = 1 if region == "donor0"
replace unit_num = 2 if region == "donor1"
replace unit_num = 3 if region == "donor2"
replace unit_num = 4 if region == "donor3"
replace unit_num = 5 if region == "donor4"
replace unit_num = 6 if region == "treated"
xtset unit_num year

synth y y(0) y(1) y(2) y(3) y(4) y(5) y(6) y(7) y(8) y(9) ///
        y(10) y(11) y(12) y(13) y(14) y(15) y(16) y(17) y(18) y(19), ///
    trunit(6) trperiod(20)

matrix Yt = e(Y_treated)
matrix Ys = e(Y_synthetic)

local n_rows = rowsof(Yt)
local sum_post = 0
local n_post = 0
local sum_pre_sq = 0
local n_pre = 0
forvalues r = 1/`n_rows' {
    local y = Yt[`r', 1]
    local s = Ys[`r', 1]
    local gap = `y' - `s'
    local rname : word `r' of `:rownames Yt'
    if `rname' >= 20 {
        local sum_post = `sum_post' + `gap'
        local n_post = `n_post' + 1
    }
    else {
        local sum_pre_sq = `sum_pre_sq' + `gap' * `gap'
        local n_pre = `n_pre' + 1
    }
}
local avg_post_gap = `sum_post' / `n_post'
local pre_rmse     = sqrt(`sum_pre_sq' / `n_pre')

count
local n = r(N)

stata_parity_row, stat(avg_post_gap) est(`avg_post_gap') nob(`n')
stata_parity_row, stat(pre_treatment_rmse) est(`pre_rmse') nob(`n')

matrix W = e(W_weights)
forvalues d = 0/4 {
    local row = `d' + 1
    local w = W[`row', 2]
    stata_parity_row, stat(weight_donor`d') est(`w') nob(`n')
}

stata_parity_extra, key(method) val("classic (per-period predictors)")
stata_parity_extra, key(stata_command) val("synth y y(0..19), trunit(6) trperiod(20)")
stata_parity_extra, key(identification_note) val("unique convex-hull SCM DGP; treated_pre = 0.5*donor0 + 0.3*donor1 + 0.2*donor2")

stata_parity_close, module(52_scm_unique)
