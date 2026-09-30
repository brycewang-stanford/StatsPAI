* Stacked DiD with repeated (non-absorbing) events, built independently in
* Stata: one sub-experiment per event (unit, g), window [g-3, g+4]; controls
* are units with no event inside that window; sub-experiments whose treated
* unit has another event in its window are dropped. Then reghdfe with
* unit x event and period x event effects, [aw=w], clustered by unit.
*   python tests/reference_parity/_fixtures/_generate_stacked_events_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_stacked_events_Stata.do
version 18
clear all
set more off
local lo -3
local hi 4
import delimited using "tests/reference_parity/_fixtures/stacked_events.csv", clear asdouble
tempfile panel stack
save `panel'
keep if event == 1
keep unit t
rename t g
tempfile events
save `events'
local n_ev = _N
local kept 0
local dropped 0
forvalues e = 1/`n_ev' {
    use `events', clear
    local u = unit[`e']
    local g = g[`e']
    use `panel', clear
    * treated unit's other events inside its window
    quietly count if unit == `u' & event == 1 & t != `g' & t >= `g' + `lo' & t <= `g' + `hi'
    if r(N) > 0 {
        local ++dropped
        continue
    }
    * units with any event inside the window are not clean controls
    bysort unit: egen _dirty = max(event == 1 & t >= `g' + `lo' & t <= `g' + `hi')
    keep if t >= `g' + `lo' & t <= `g' + `hi'
    keep if unit == `u' | _dirty == 0
    gen event_id = `e'
    gen treated = unit == `u'
    gen rel = t - `g'
    drop _dirty
    if `kept' == 0 save `stack'
    else {
        append using `stack'
        save `stack', replace
    }
    local ++kept
}
use `stack', clear
forvalues k = `lo'/`hi' {
    if `k' == -1 continue
    local nm = cond(`k' < 0, "m" + string(-`k'), "p" + string(`k'))
    gen D_`nm' = treated * (rel == `k')
}
egen ue = group(unit event_id)
egen te = group(t event_id)
reghdfe y D_*, absorb(ue te) vce(cluster unit)
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/stacked_events_Stata.json", write replace
file write `fh' "{" _n `"  "N": "' (e(N)) `", "kept": "' (`kept') `", "dropped": "' (`dropped') "," _n
local first 1
foreach k in m3 m2 p0 p1 p2 p3 p4 {
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  "`k'": ["' %24.16e (_b[D_`k']) ", " %24.16e (_se[D_`k']) "]"
}
file write `fh' "," _n
reghdfe y D_* [aw=w], absorb(ue te) vce(cluster unit)
local first 1
foreach k in m3 m2 p0 p1 p2 p3 p4 {
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  "w_`k'": ["' %24.16e (_b[D_`k']) ", " %24.16e (_se[D_`k']) "]"
}
file write `fh' _n "}" _n
file close `fh'
