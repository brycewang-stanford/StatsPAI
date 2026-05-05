* tests/stata_parity/07_scm.do
*
* Module 07: Classical SCM on Basque replica.
*   StatsPAI:  sp.synth(method="classic")
*   R:         Synth::synth + Synth::dataprep
*   Stata:     synth (Abadie/Diamond/Hainmueller, SSC)
*
* Tolerance: rel < 1e-3 on avg post-1970 gap. Donor pool: all
* non-treated regions; outcomes-only spec (every pre-1970 year of
* gdppc as a special predictor).

version 18
clear all

do _common.do
stata_parity_init, module(07_scm)
stata_parity_open, module(07_scm)

import delimited "${STATA_PARITY_DATA}/07_scm.csv", clear case(preserve) ///
    stringcols(1)

* synth needs numeric panel id.
encode region, generate(unit_num)
xtset unit_num year
quietly summarize unit_num if region == "Basque Country"
local treated_id = r(min)

* Pre-period 1955..1969; treatment year 1970.
synth gdppc gdppc(1955) gdppc(1956) gdppc(1957) gdppc(1958) gdppc(1959) ///
            gdppc(1960) gdppc(1961) gdppc(1962) gdppc(1963) gdppc(1964) ///
            gdppc(1965) gdppc(1966) gdppc(1967) gdppc(1968) gdppc(1969), ///
    trunit(`treated_id') trperiod(1970) nested fig

* synth e()-returns:
*   e(Y_treated)   -- T x 1
*   e(Y_synthetic) -- T x 1 (rows ordered as time)
matrix Yt = e(Y_treated)
matrix Ys = e(Y_synthetic)

* Compute gap and post-1970 mean. The matrix rows are time-ordered
* but synth labels them with the year. Iterate, summing for years
* >= 1970.
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
    if `rname' >= 1970 {
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
stata_parity_row, stat(pre_rmse)     est(`pre_rmse')     nob(`n')

stata_parity_extra, key(predictors) val("gdppc(1955)..gdppc(1969) outcomes-only")
stata_parity_extra, key(stata_command) val("synth gdppc gdppc(1955..1969), trunit(.) trperiod(1970) nested")

stata_parity_close, module(07_scm)
