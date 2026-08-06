*! 83_sunab_control_cohort.do
*! Golden numbers for tests/reference_parity/test_sunab_control_cohort_parity.py
*!
*! Pins eventstudyinteract's control_cohort(varname) against StatsPAI's
*! sun_abraham(control_cohort=), under two reference groups:
*!
*!   control_cohort(never)  -> StatsPAI default / control_cohort=0
*!   control_cohort(c2007)  -> StatsPAI control_cohort=2007
*!
*! Also the source of the SE numbers that exposed the missing
*! cohort-share variance term (SA 2021 Prop. 3) in StatsPAI's IW SE.
*! eventstudyinteract carries that term; fixest::sunab does not.
*!
*! Requires: eventstudyinteract (v0.1), avar, reghdfe. Stata 18 MP.

version 17
clear all
set more off

import delimited "../../orig_parity/data/02_mpdta_original.csv", clear

* eventstudyinteract wants cohort missing for never-treated units.
gen cohort = first_treat
replace cohort = . if first_treat == 0
gen rel = year - first_treat if first_treat > 0

* Relative-time dummies, omitting e = -1 as the reference. They must be
* zero (not missing) for never-treated units.
local dumlist ""
foreach k in -4 -3 -2 0 1 2 3 {
    local nm = cond(`k' < 0, "g_m" + string(abs(`k')), "g_p" + string(`k'))
    gen `nm' = (rel == `k') & !missing(rel)
    local dumlist "`dumlist' `nm'"
}

gen never  = (first_treat == 0)
gen c2007  = (first_treat == 2007)

tempname fh
file open `fh' using "results/83_sunab_control_cohort_Stata.json", write replace
file write `fh' "{" _n

program define _dump_iw
    args fh label
    matrix b = e(b_iw)
    matrix V = e(V_iw)
    local cn : colnames b
    file write `fh' `"  ""' "`label'" `"": {"' _n
    local k = colsof(b)
    forvalues j = 1/`k' {
        local nm : word `j' of `cn'
        local est = b[1,`j']
        local se  = sqrt(V[`j',`j'])
        local comma ","
        if `j' == `k' local comma ""
        file write `fh' `"    ""' "`nm'" `"": {"att": "' %20.15f (`est') ///
            `", "se": "' %20.15f (`se') "}`comma'" _n
    }
    file write `fh' "  }," _n
end

* --- never-treated as control (StatsPAI default) ---------------------
eventstudyinteract lemp `dumlist', cohort(cohort) control_cohort(never) ///
    absorb(countyreal year) vce(cluster countyreal)
_dump_iw `fh' "control_cohort_never"

* --- 2007 cohort as control ------------------------------------------
* NOTE: g_m4 comes back exactly 0 here (no estimable cohort at that lead
* once 2007 is the reference). StatsPAI omits the row rather than
* reporting a spurious zero, so that cell is excluded from the test.
eventstudyinteract lemp `dumlist', cohort(cohort) control_cohort(c2007) ///
    absorb(countyreal year) vce(cluster countyreal)
_dump_iw `fh' "control_cohort_2007"

file write `fh' `"  "_meta": {"eventstudyinteract": "0.1", "stata": "18 MP"}"' _n
file write `fh' "}" _n
file close `fh'

di as txt "wrote results/83_sunab_control_cohort_Stata.json"
