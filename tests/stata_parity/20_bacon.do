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
local rn_list : rownames M
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

* ------------------------------------------------------------------
* Per-pair timing comparisons.
*
* e(sumdd) carries one row per 2x2 comparison, but its rownames are only
* "Early_v_Late" / "Late_v_Early" / "Never_v_timing" -- the cohort years are
* not in the labels, so the mapping to the Python/R statistic names
* (pair_<g>_vs_<h>_est) has to come from the row ORDER. bacondecomp emits
* the timing pairs as consecutive Early/Late couples, iterating unordered
* cohort pairs (g < h) in ascending order:
*
*   row 1  Early_v_Late   g1 vs h1        row 2  Late_v_Early   h1 vs g1
*   row 3  Early_v_Late   g1 vs h2        row 4  Late_v_Early   h2 vs g1
*   row 5  Early_v_Late   g2 vs h2        row 6  Late_v_Early   h2 vs g2
*
* Reconstructing the cohort list from the data and walking it in the same
* order recovers the labels. The guards below make that inference falsifiable
* rather than assumed: the number of timing rows must equal twice the number
* of unordered treated-cohort pairs, and every timing row must carry the
* Early/Late label the position predicts.
*
* The three never-treated comparisons are NOT emitted: bacondecomp collapses
* them into a single `Never_v_timing` row (weight 0.467 here), so there is
* nothing to join against the Python side's per-cohort
* pair_<g>_vs_never_est rows.
* ------------------------------------------------------------------
qui levelsof first_treat if first_treat > 0 & !missing(first_treat), local(cohorts)
local n_cohorts : word count `cohorts'
local n_timing = `n_cohorts' * (`n_cohorts' - 1)

local n_never = 0
forvalues i = 1/`n_pairs' {
    local lab : word `i' of `rn_list'
    if "`lab'" == "Never_v_timing" {
        local n_never = `n_never' + 1
    }
}
if `n_pairs' - `n_never' != `n_timing' {
    display as error "bacondecomp returned `=`n_pairs'-`n_never'' timing rows; `n_cohorts' cohorts imply `n_timing'"
    exit 459
}

local r = 0
forvalues a = 1/`n_cohorts' {
    local g : word `a' of `cohorts'
    local b1 = `a' + 1
    forvalues b = `b1'/`n_cohorts' {
        local h : word `b' of `cohorts'
        local r = `r' + 1
        local lab : word `r' of `rn_list'
        if "`lab'" != "Early_v_Late" {
            display as error "row `r' expected Early_v_Late, got `lab'"
            exit 459
        }
        stata_parity_row, stat("pair_`g'_vs_`h'_est") est(`=M[`r',1]') nob(`n')
        local r = `r' + 1
        local lab : word `r' of `rn_list'
        if "`lab'" != "Late_v_Early" {
            display as error "row `r' expected Late_v_Early, got `lab'"
            exit 459
        }
        stata_parity_row, stat("pair_`h'_vs_`g'_est") est(`=M[`r',1]') nob(`n')
    }
}

stata_parity_extra, key(n_comparisons) val(`n_pairs')
stata_parity_extra, key(stata_command) val("bacondecomp lemp treat, ddetail")
stata_parity_extra, key(never_treated_rows) val("bacondecomp collapses the never-treated comparisons into one Never_v_timing row, so the per-cohort pair_<g>_vs_never_est rows stay py<->R")

stata_parity_close, module(20_bacon)
