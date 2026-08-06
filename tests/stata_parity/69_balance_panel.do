* tests/stata_parity/69_balance_panel.do
*
* Module 69: the balanced-panel filter.
*   StatsPAI:  sp.balance_panel
*   R:         base-R ave(counts) == n_periods row selection
*   Stata:     bysort id: count distinct years, keep the full ones  <-- this file
*
* Like module 68 this is an exact identity, not an estimator, and the same
* argument applies: an identity is where a third implementation is cheapest
* and a disagreement is least ambiguous. The fixture drops exactly one
* entity (id = 2, observed in 3 of 4 periods), so a filter that is off by one
* period in either direction changes `n_obs_balanced` and fails loudly.
*
* Convention detail worth stating: the reference counts *distinct* periods
* per entity, not rows. Counting rows would keep an entity that has a
* duplicated year and a missing one, which is a different filter. This
* script counts distinct years to match.
*
* Tolerance: exact. These are counts and index values.

version 18
clear all

do _common.do
stata_parity_init, module(69_balance_panel)
stata_parity_open, module(69_balance_panel)

import delimited "${STATA_PARITY_DATA}/69_balance_panel.csv", clear case(preserve)
count
local n_in = r(N)

* Number of distinct periods in the panel, and per entity.
qui levelsof year, local(all_years)
local n_periods : word count `all_years'

bysort id year: gen byte _first_yr = (_n == 1)
bysort id: egen int _n_years = total(_first_yr)

qui keep if _n_years == `n_periods'
sort id year

count
local n_bal = r(N)
qui levelsof id, local(kept_ids)
local n_units : word count `kept_ids'

stata_parity_row, stat(n_obs_balanced) est(`n_bal')   nob(`n_in')
stata_parity_row, stat(n_units_kept)   est(`n_units') nob(`n_in')

* R samples k in {1, n/2 + 1, n} over the *balanced* frame; the emitted
* names are 0-based row indices.
local k1 = 1
local k2 = floor(`n_bal' / 2) + 1
local k3 = `n_bal'

foreach k in `k1' `k2' `k3' {
    local idx = `k' - 1
    stata_parity_row, stat("row`idx'_id")   est(`=id[`k']')   nob(`n_in')
    stata_parity_row, stat("row`idx'_year") est(`=year[`k']') nob(`n_in')
}

stata_parity_extra, key(stata_command) val("bysort id year: gen _first_yr = (_n==1); bysort id: egen _n_years = total(_first_yr); keep if _n_years == n_periods")
stata_parity_extra, key(filter) val("keep only entities observed in every period; periods are counted DISTINCT, so a duplicated year does not substitute for a missing one")
stata_parity_extra, key(dropped) val("this fixture drops exactly one entity (id=2, observed in 3 of 4 periods)")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(69_balance_panel)
