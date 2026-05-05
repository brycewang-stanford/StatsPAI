* tests/stata_parity/05_sunab.do
*
* Module 05: Sun-Abraham event study (post-treatment weighted ATT).
*   StatsPAI:  sp.sun_abraham(...).att
*   R:         fixest::feols(Y ~ sunab(g, t) | id + t, cluster=~id)
*   Stata:     eventstudyinteract Y rel_dummies, cohort(g) ///
*                  control_cohort(never) absorb(i.id i.t) vce(cluster id)
*
* eventstudyinteract returns interaction-weighted event-study coeffs;
* we record the simple post-treatment weighted average to match the
* aggregated ATT reported by the R side and StatsPAI.

version 18
clear all

do _common.do
stata_parity_init, module(05_sunab)
stata_parity_open, module(05_sunab)

import delimited "${STATA_PARITY_DATA}/05_sunab.csv", clear case(preserve)

* Build relative-time and never-treated indicators expected by
* eventstudyinteract.
generate relyear = year - first_treat if first_treat > 0
generate nevertreat = (first_treat == 0)

* Construct event-time dummies for relyear in {-T..-1, 1..T}.
quietly summarize relyear if relyear != .
local rmin = r(min)
local rmax = r(max)

* Drop the relyear == -1 reference omission.
forvalues r = `rmin'/`rmax' {
    if `r' != -1 {
        local rstr = cond(`r' < 0, "m" + string(-`r'), string(`r'))
        generate byte g_`rstr' = (relyear == `r')
        local dummies `dummies' g_`rstr'
    }
}

* Run eventstudyinteract.
* `cohort(first_treat)` provides the cohort variable; `control_cohort`
* must be a 0/1 indicator of the never-treated controls.
eventstudyinteract lemp `dummies', ///
    cohort(first_treat) control_cohort(nevertreat) ///
    absorb(i.countyreal i.year) vce(cluster countyreal)

* eventstudyinteract returns e(b_iw) / e(V_iw): one column per dummy.
matrix B = e(b_iw)
matrix V = e(V_iw)
local cols : colnames B

* Identify post-treatment dummies (g_<positive integer>).
local post_dummies
foreach c of local cols {
    if regexm("`c'", "^g_[0-9]+$") {
        local post_dummies `post_dummies' `c'
    }
}

* Cohort-share weighted average over post-treatment dummies = headline
* ATT. fixest::sunab default reports per-relative-time; we average
* equally weighted (same as r_parity does post-hoc).
local k : word count `post_dummies'
local sum_est = 0
local sum_var = 0
foreach d of local post_dummies {
    local est = B[1, "`d'"]
    local var = V["`d'", "`d'"]
    local sum_est = `sum_est' + `est' / `k'
    local sum_var = `sum_var' + `var' / (`k' * `k')
}
local headline_est = `sum_est'
local headline_se  = sqrt(`sum_var')
local lo = `headline_est' - ${STATA_PARITY_Z95} * `headline_se'
local hi = `headline_est' + ${STATA_PARITY_Z95} * `headline_se'
count
local n = r(N)

stata_parity_row, stat(weighted_avg_ATT) est(`headline_est') std(`headline_se') cilo(`lo') cihi(`hi') nob(`n')

* Per-relative-time IW-aggregated ATT, like r_parity does.
foreach c of local cols {
    if regexm("`c'", "^g_(m?)([0-9]+)$") {
        local sign  = regexs(1)
        local r_num = regexs(2)
        if "`sign'" == "m" local rt = -`r_num'
        else               local rt =  `r_num'
        local est = B[1, "`c'"]
        local sv  = sqrt(V["`c'", "`c'"])
        stata_parity_row, stat(att_rel_`rt') est(`est') std(`sv') nob(`n')
    }
}

stata_parity_extra, key(cluster) val(countyreal)
stata_parity_extra, key(stata_command) val("eventstudyinteract Y rel_dummies, cohort(first_treat) control_cohort(nevertreat) absorb(i.countyreal i.year) vce(cluster countyreal)")

stata_parity_close, module(05_sunab)
