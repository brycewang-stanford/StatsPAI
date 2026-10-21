* Critical values of regression event studies in Stata 18 (parity= of
* sp.event_study / sp.sun_abraham / sp.did_2x2). Coefficient tables with
* their p-values and confidence limits, as Stata reports them (r(table)),
* on the bundled castle-doctrine CSV and the divorce fixture.
* Run from the repository root:
*   stata-mp -q -b do tests/reference_parity/_fixtures/_generate_es_crit_Stata.do
* Needs reghdfe and eventstudyinteract (+ avar) from SSC.
version 18
clear all
set more off
set type double
cap program drop dumptab
program define dumptab
    syntax, Tag(string) [Prefix(string)]
    matrix T = r(table)
    local names : colnames T
    local J = colsof(T)
    forvalues j = 1/`J' {
        local nm : word `j' of `names'
        file write fh "`tag',`nm'," %24.16e (T[1,`j']) "," %24.16e (T[2,`j']) "," %24.16e (T[4,`j']) "," %24.16e (T[5,`j']) "," %24.16e (T[6,`j']) "," %24.16e (T[7,`j']) _n
    }
end
file open fh using "tests/reference_parity/_fixtures/es_crit_Stata.csv", write replace
file write fh "tag,term,b,se,pvalue,ll,ul,df" _n

* ---- castle doctrine (Cheng-Hoekstra 2013), bundled CSV ---------------
import delimited using "src/statspai/datasets/data/castle_2013.csv", clear asdouble
gen region = 1*northeast + 2*midwest + 3*south + 4*west
gen rel = year - effyear
gen relb = rel
replace relb = -5 if rel < -5 & !missing(rel)
replace relb =  4 if rel >  4 & !missing(rel)
local lst ""
foreach k in -5 -4 -3 -2 0 1 2 3 4 {
    local nm = cond(`k' < 0, "m" + string(-`k'), "p`k'")
    gen es_`nm' = (relb == `k')
    local lst `lst' es_`nm'
}
* TWFE event study, window [-5, 4] with binned ends, reference -1
reghdfe l_homicide `lst' [aw=popwt], absorb(sid year) vce(cluster sid)
dumptab, tag(es_castle_cl)
* headline: equal-weight average of the post coefficients
lincom (es_p0 + es_p1 + es_p2 + es_p3 + es_p4)/5
file write fh "es_castle_cl_att,att," %24.16e (r(estimate)) "," %24.16e (r(se)) "," %24.16e (r(p)) "," %24.16e (r(lb)) "," %24.16e (r(ub)) "," %24.16e (r(df)) _n
reghdfe l_homicide `lst' [aw=popwt], absorb(sid year) vce(cluster sid) level(90)
dumptab, tag(es_castle_cl90)
reghdfe l_homicide `lst' [aw=popwt], absorb(sid year) vce(cluster region)
dumptab, tag(es_castle_region)

* Sun-Abraham, saturated in relative time, never-treated control cohort
gen never = missing(effyear)
gen gco = cond(missing(effyear), 0, effyear)
forvalues k = 9(-1)2 {
    gen L`k' = (rel == -`k')
}
forvalues k = 0/5 {
    gen F`k' = (rel == `k')
}
eventstudyinteract l_homicide L9-L2 F0-F5 [aw=popwt], cohort(effyear) ///
    control_cohort(never) absorb(i.sid i.year) vce(cluster sid)
dumptab, tag(sa_castle_cl)
* repeated cross sections: cohort and year effects, robust SEs
eventstudyinteract l_homicide L9-L2 F0-F5 [aw=popwt], cohort(effyear) ///
    control_cohort(never) absorb(i.gco i.year) vce(robust)
dumptab, tag(sa_castle_rob)

* 2x2: the 2006 cohort against the never treated, post = year >= 2006
keep if effyear == 2006 | missing(effyear)
gen tr = !missing(effyear)
gen po = year >= 2006
gen trpo = tr*po
reg l_homicide tr po trpo, vce(cluster sid)
dumptab, tag(2x2_castle_cl)
reg l_homicide tr po trpo, vce(robust)
dumptab, tag(2x2_castle_rob)
reg l_homicide tr po trpo [aw=popwt], vce(cluster sid)
dumptab, tag(2x2_castle_clw)

* ---- divorce (Stevenson-Wolfers 2006 / Goodman-Bacon 2021) ------------
import delimited using "tests/fixtures/divorce_no_fault_panel.csv", clear asdouble
bys stfips: egen first_post = min(cond(post == 1, year, .))
gen cohort = cond(missing(first_post), 0, first_post)
drop if cohort == 1964
gen rel = cond(cohort > 0, year - cohort, .)
replace rel = -10 if rel < -10 & !missing(rel)
replace rel =  15 if rel >  15 & !missing(rel)
forvalues k = 10(-1)2 {
    gen byte lead`k' = (rel == -`k')
}
forvalues k = 0/15 {
    gen byte lag`k' = (rel == `k')
}
reghdfe asmrs lead* lag*, absorb(stfips year) vce(cluster stfips)
dumptab, tag(es_divorce_cl)
file close fh
