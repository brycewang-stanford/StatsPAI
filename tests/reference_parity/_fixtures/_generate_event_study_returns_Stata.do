* Reference values for tests/reference_parity/test_event_study_returns_parity.py
* Stata 18 with estudy (Pacicco, Vena and Venegoni) from SSC.
* Run from this folder:
*     do _generate_event_study_returns_Stata.do
* Reads event_study_returns_wide.csv and event_study_events.csv, writes
* event_study_returns_Stata.csv (one row per model, test, window, security
* and quantity) and event_study_returns_Stata_ar.csv (the abnormal returns
* of the market model, one column per security).
clear all
set more off
import delimited using "event_study_events.csv", clear case(preserve) varnames(1) stringcols(_all)
gen double evd = date(event_date, "YMD")
format evd %td
gen n = _n
keep n id evd
rename id secname
save "_es_events_tmp.dta", replace
import delimited using "event_study_returns_wide.csv", clear case(preserve) asdouble
gen double d = date(date, "YMD")
format d %td
drop date
rename d date
gen n = _n
merge 1:1 n using "_es_events_tmp.dta", nogen
sort date

tempname H
postfile `H' str8 model str12 test str8 window str8 security str8 term double value using "_es_stata_tmp.dta", replace
local w1 "m1_p1"
local w2 "0_0"
local w3 "m5_p5"
foreach model in SIM MAM HMM MFM {
    local index = cond("`model'" == "MFM", "mkt smb hml", "mkt")
    foreach test in Norm Patell ADJPatell BMP KP {
        quietly estudy s01-s12, datevar(date) evdate(secname evd) modt(`model') indexlist(`index') diagn(`test') lb1(-1) ub1(1) lb2(0) ub2(0) lb3(-5) ub3(5) eswlb(-200) eswub(-11)
        matrix C = r(cars)
        matrix S = r(sd)
        matrix Z = r(stats)
        matrix P = r(pv)
        forvalues i = 1/13 {
            local sec = cond(`i' == 13, "group", "s" + string(`i', "%02.0f"))
            forvalues j = 1/3 {
                post `H' ("`model'") ("`test'") ("`w`j''") ("`sec'") ("car") (C[`i', `j'])
                post `H' ("`model'") ("`test'") ("`w`j''") ("`sec'") ("sd") (S[`i', `j'])
                post `H' ("`model'") ("`test'") ("`w`j''") ("`sec'") ("stat") (Z[`i', `j'])
                post `H' ("`model'") ("`test'") ("`w`j''") ("`sec'") ("pv") (P[`i', `j'])
            }
        }
    }
}
postclose `H'
* estudy drops the matrices in memory on every run, so the abnormal
* returns of the market model are taken from a last run
quietly estudy s01-s12, datevar(date) evdate(secname evd) modt(SIM) indexlist(mkt) diagn(Norm) lb1(-1) ub1(1) lb2(0) ub2(0) lb3(-5) ub3(5) eswlb(-200) eswub(-11)
matrix AR = r(ar)
clear
svmat double AR, names(col)
format _all %24.17g
export delimited using "event_study_returns_Stata_ar.csv", replace
use "_es_stata_tmp.dta", clear
format value %24.17g
export delimited using "event_study_returns_Stata.csv", replace
erase "_es_stata_tmp.dta"
erase "_es_events_tmp.dta"
