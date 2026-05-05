* tests/stata_parity/20_bacon.do
*
* Module 20: Goodman-Bacon decomposition.
*   StatsPAI:  sp.bacon_decomposition(...)
*   R:         bacondecomp::bacon
*   Stata:     bacondecomp (Goodman-Bacon's Stata port)
*
* Tolerance: rel < 1e-3 on weighted-sum TWFE; per-pair reporting.

version 18
clear all

do _common.do
stata_parity_init, module(20_bacon)
stata_parity_open, module(20_bacon)

import delimited "${STATA_PARITY_DATA}/20_bacon.csv", clear case(preserve)

* bacondecomp: depvar treatment, panel(panelvar)
xtset countyreal year
bacondecomp lemp treat, ddetail

* bacondecomp posts r(sigma) but the key matrix is e(sumdd) (weighted
* sum) and the per-pair list. Output the weighted sum + share of
* negative weights, plus per-pair rows.
local n = e(N)

* e(sumdd) is the per-comparison detail matrix [Beta, TotalWeight].
matrix M = e(sumdd)
local n_pairs = rowsof(M)
local twfe_sum = 0
local abs_w_sum = 0
local neg_w_sum = 0
forvalues i = 1/`n_pairs' {
    local est = M[`i', 1]
    local w   = M[`i', 2]
    local twfe_sum  = `twfe_sum' + `w' * `est'
    local abs_w_sum = `abs_w_sum' + abs(`w')
    if `w' < 0 {
        local neg_w_sum = `neg_w_sum' + `w'
    }
}
if `abs_w_sum' > 0 {
    local neg_share = `neg_w_sum' / `abs_w_sum'
}
else {
    local neg_share = 0
}

stata_parity_row, stat(beta_twfe)             est(`twfe_sum')  nob(`n')
stata_parity_row, stat(weighted_sum)          est(`twfe_sum')  nob(`n')
stata_parity_row, stat(negative_weight_share) est(`neg_share') nob(`n')

stata_parity_extra, key(n_comparisons) val(`n_pairs')
stata_parity_extra, key(stata_command) val("bacondecomp lemp treat, ddetail")

stata_parity_close, module(20_bacon)
