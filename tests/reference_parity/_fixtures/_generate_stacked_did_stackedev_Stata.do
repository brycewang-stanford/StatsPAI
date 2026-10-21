* stackedev (Bleiberg, version 0.0 24jun2021) and reghdfe (6.12.3) on
* stacked_did_stackedev.csv: event-time dummies F2-F5 / L0-L4 (k = -1
* omitted), control x, unit x stack and year x stack effects.
*   "stackedev"  : stackedev's own clustering, on stack * id (unit x stack)
*   "unit"       : the same stack by hand, clustered on id
* Each block also reports the equal-weight and the treated-observation-
* weighted average of L0-L4 by lincom.
*   python tests/reference_parity/_fixtures/_generate_stacked_did_stackedev_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_stacked_did_stackedev_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/stacked_did_stackedev.csv", clear asdouble
gen never = missing(first_treat)
gen K = year - first_treat
forvalues l = 0/4 {
    gen L`l'event = K == `l'
}
forvalues l = 2/5 {
    gen F`l'event = K == -`l'
}
* treated rows at each post event time (each treated unit sits in one stack)
local nsum 0
local wexpr ""
forvalues l = 0/4 {
    qui count if K == `l'
    local n`l' = r(N)
    local nsum = `nsum' + r(N)
    local wexpr "`wexpr' + `n`l''*L`l'event"
}
local wexpr = substr("`wexpr'", 4, .)
tempfile panel
save `panel'

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/stacked_did_stackedev_Stata.json", write replace
file write `fh' "{" _n
file write `fh' `"  "n_treated": {"0": `n0', "1": `n1', "2": `n2', "3": `n3', "4": `n4'},"' _n

capture program drop _dump
program define _dump
    args fh name wexpr nsum last
    file write `fh' `"  "`name'": {"N": "' (e(N)) `", "N_clust": "' (e(N_clust))
    foreach v in F2event F3event F4event F5event L0event L1event L2event L3event L4event x {
        file write `fh' `", "b_`v'": "' %25.17f (_b[`v']) `", "se_`v'": "' %25.17f (_se[`v'])
    }
    qui lincom (L0event + L1event + L2event + L3event + L4event) / 5
    file write `fh' `", "att_eq": "' %25.17f (r(estimate)) `", "att_eq_se": "' %25.17f (r(se))
    qui lincom (`wexpr') / `nsum'
    file write `fh' `", "att_obs": "' %25.17f (r(estimate)) `", "att_obs_se": "' %25.17f (r(se)) "}"
    if "`last'" == "" file write `fh' ","
    file write `fh' _n
end

use `panel', clear
stackedev y F*event L*event, cohort(first_treat) time(year) never_treat(never) ///
    covariates(x) unit_fe(id) clust_unit(id)
_dump `fh' stackedev "`wexpr'" `nsum'

* the same stack by hand, clustered on the unit
tempfile stk
local first 1
foreach g in 2014 2015 2016 {
    use `panel', clear
    keep if first_treat == `g' | never == 1
    gen stack = `g'
    if !`first' append using `stk'
    save `stk', replace
    local first 0
}
use `stk', clear
qui reghdfe y F*event L*event x, absorb(i.id#i.stack i.year#i.stack) cluster(id)
_dump `fh' unit "`wexpr'" `nsum' last
file write `fh' "}" _n
file close `fh'
