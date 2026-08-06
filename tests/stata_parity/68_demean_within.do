* tests/stata_parity/68_demean_within.do
*
* Module 68: the within (entity-mean) transformation.
*   StatsPAI:  sp.demean(solver="map")
*   R:         hand-written mean-within loop
*   Stata:     bysort id: egen mean + subtract           <-- this file
*
* This module checks an exact algebraic identity rather than an estimator:
* M y and M X with M = I - P_id, the entity-mean projection. Its previous
* skip reason -- "algorithmic module ... no Stata artifact is materialized"
* -- was accurate about what the module is but drew the wrong conclusion.
* An identity is precisely the case where a third implementation is cheap
* and the expected agreement is exact, so a disagreement here would be
* unambiguous: not a convention difference, not an optimiser, a bug. Two
* implementations agreeing on an identity is weaker evidence than three,
* and `bysort id: egen` is four lines.
*
* The statistic names mirror the R side: `demean_y` is the first row of the
* demeaned outcome, and the `demean_x*_row{k}` rows sample each demeaned
* covariate at the first, middle and last observation (0-based, matching R's
* k in {1, n/2, n} converted to row indices).
*
* Tolerance: rel < 1e-6, and in practice this should be exact to the last
* bit -- the only floating-point freedom is the summation order inside the
* group mean.

version 18
clear all

do _common.do
stata_parity_init, module(68_demean_within)
stata_parity_open, module(68_demean_within)

import delimited "${STATA_PARITY_DATA}/68_demean_within.csv", clear case(preserve)
count
local n = r(N)

* Preserve the CSV row order: the R side samples rows positionally, so the
* comparison is only meaningful if both sides index the same observations.
gen long _row = _n

foreach v in y x1 x2 x3 {
    bysort id (_row): egen double _m_`v' = mean(`v')
    gen double dm_`v' = `v' - _m_`v'
}
sort _row

* All three sides sample the same observations: Python's 0-based
* (0, n//2, n-1) is Stata/R's 1-based (1, n/2 + 1, n). The emitted names are
* the 0-based indices.
local k1 = 1
local k2 = floor(`n' / 2) + 1
local k3 = `n'

stata_parity_row, stat(demean_y) est(`=dm_y[`k1']') nob(`n')

foreach k in `k1' `k2' `k3' {
    local idx = `k' - 1
    foreach v in x1 x2 x3 {
        stata_parity_row, stat("demean_`v'_row`idx'") est(`=dm_`v'[`k']') nob(`n')
    }
}

stata_parity_extra, key(stata_command) val("bysort id (_row): egen double _m = mean(v); gen double dm = v - _m")
stata_parity_extra, key(transform) val("within (entity-mean projection), M = I - P_id")
stata_parity_extra, key(row_order) val("the CSV row order is preserved with an explicit _row index so all three sides sample the same observations")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(68_demean_within)
