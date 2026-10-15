*! 93_sunab_options.do
*! Golden numbers for tests/reference_parity/test_sunab_options_parity.py
*!
*! Pins eventstudyinteract (Sun's own implementation) against
*! sp.sun_abraham(window_rule='reference'): relative-time dummies only
*! inside the event window, so the relative times outside it join the
*! omitted category next to e = -1. This is how the command is commonly
*! run, and what StatsPAI's window did before 1.39.0.
*!
*!   window_m3_p2_never  -> event_window=(-3, 2), window_rule='reference'
*!   window_m2_p1_never  -> event_window=(-2, 1), window_rule='reference'
*!   window_m3_p2_c2007  -> the same with control_group='lastcohort'
*!                          (2007 is the last cohort on mpdta)
*!
*! Requires: eventstudyinteract (v0.1), avar, reghdfe. Stata 18 MP.

version 17
clear all
set more off

* Double precision throughout (see option_parity/README.md).
set type double
import delimited "../../orig_parity/data/02_mpdta_original.csv", clear asdouble

* eventstudyinteract wants cohort missing for never-treated units.
gen cohort = first_treat
replace cohort = . if first_treat == 0
gen rel = year - first_treat if first_treat > 0

* Relative-time dummies, zero (not missing) for never-treated units.
foreach k in -4 -3 -2 0 1 2 3 {
    local nm = cond(`k' < 0, "g_m" + string(abs(`k')), "g_p" + string(`k'))
    gen `nm' = (rel == `k') & !missing(rel)
}

gen never  = (first_treat == 0)
gen c2007  = (first_treat == 2007)

tempname fh
file open `fh' using "results/93_sunab_options_Stata.json", write replace
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
        file write `fh' `"    ""' "`nm'" `"": {"att": "' %25.18f (`est') ///
            `", "se": "' %25.18f (`se') "}`comma'" _n
    }
    file write `fh' "  }," _n
end

eventstudyinteract lemp g_m3 g_m2 g_p0 g_p1 g_p2, cohort(cohort) ///
    control_cohort(never) absorb(countyreal year) vce(cluster countyreal)
_dump_iw `fh' "window_m3_p2_never"

eventstudyinteract lemp g_m2 g_p0 g_p1, cohort(cohort) ///
    control_cohort(never) absorb(countyreal year) vce(cluster countyreal)
_dump_iw `fh' "window_m2_p1_never"

eventstudyinteract lemp g_m3 g_m2 g_p0 g_p1 g_p2, cohort(cohort) ///
    control_cohort(c2007) absorb(countyreal year) vce(cluster countyreal)
_dump_iw `fh' "window_m3_p2_c2007"

file write `fh' `"  "_meta": {"eventstudyinteract": "0.1", "stata": "18 MP", "precision": "double"}"' _n
file write `fh' "}" _n
file close `fh'

set type float
di as txt "wrote results/93_sunab_options_Stata.json"
