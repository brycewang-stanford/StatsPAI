* Reference values for tests/reference_parity/test_accounting_research_parity.py
* Stata 18 with xtfmb, robreg (and moremata), konfound, winsor2 from SSC.
* Run from this folder:
*     do _generate_accounting_research_Stata.do
* Reads accounting_research_{panel,cross}.csv, writes
* accounting_research_Stata.csv with one row per (model, term).
clear all
set more off
tempname H
postfile `H' str32 model str24 term double value using "_ar_stata_tmp.dta", replace

* ---------------------------------------------------------------- panel
import delimited using "accounting_research_panel.csv", clear case(preserve) asdouble
xtset firm year
foreach L in 0 2 {
    if `L' == 0 xtfmb y x z
    else xtfmb y x z, lag(`L')
    matrix V = e(V)
    foreach v in x z _cons {
        post `H' ("xtfmb_lag`L'") ("b_`v'") (_b[`v'])
        post `H' ("xtfmb_lag`L'") ("V_`v'") (V[rownumb(V, "`v'"), colnumb(V, "`v'")])
    }
    post `H' ("xtfmb_lag`L'") ("r2") (e(r2))
    post `H' ("xtfmb_lag`L'") ("df_r") (e(df_r))
}
foreach L in 1 3 {
    newey y x z, lag(`L') force
    foreach v in x z _cons {
        post `H' ("newey_lag`L'") ("se_`v'") (_se[`v'])
    }
}
preserve
keep if gap == 0
xtset firm year
newey y x z, lag(2) force
foreach v in x z _cons {
    post `H' ("newey_gap_lag2") ("se_`v'") (_se[`v'])
}
xtfmb y x z
foreach v in x z _cons {
    post `H' ("xtfmb_gap") ("b_`v'") (_b[`v'])
    post `H' ("xtfmb_gap") ("se_`v'") (_se[`v'])
}
restore

* ------------------------------------------------------- cross-section
import delimited using "accounting_research_cross.csv", clear case(preserve) asdouble
local est1 "m"
local est2 "m"
local est3 "s"
local est4 "mm"
local est5 "mm"
local opt2 "biweight"
local opt5 "efficiency(95)"
local name1 "robreg_m_huber"
local name2 "robreg_m_biweight"
local name3 "robreg_s"
local name4 "robreg_mm85"
local name5 "robreg_mm95"
forvalues i = 1/5 {
    local est "`est`i''"
    robreg `est' y_out x1 x2 x3, `opt`i'' tolerance(1e-14)
    matrix b = e(b)
    matrix V = e(V)
    local eq = cond("`est'" == "m", "", cond("`est'" == "s", "S:", "MM:"))
    foreach v in x1 x2 x3 _cons {
        post `H' ("`name`i''") ("b_`v'") (b[1, colnumb(b, "`eq'`v'")])
        post `H' ("`name`i''") ("V_`v'") (V[rownumb(V, "`eq'`v'"), colnumb(V, "`eq'`v'")])
    }
    post `H' ("`name`i''") ("scale") (e(scale))
    post `H' ("`name`i''") ("k") (e(k))
}

* ITCV and RIR from the estimate, its standard error, n and the number of
* other covariates (pkonfound uses df = n - ncov - 2)
regress y x1 x2 x3 i.ind
local n = e(N)
local ncov = e(df_m) - 1
foreach v in x2 x3 {
    local b = _b[`v']
    local s = _se[`v']
    pkonfound `b' `s' `n' `ncov', indx(IT)
    post `H' ("pkonfound_`v'") ("r_obs") (r(obs_r))
    post `H' ("pkonfound_`v'") ("r_crit") (r(critical_r))
    post `H' ("pkonfound_`v'") ("itcv") (r(itcvGz))
    post `H' ("pkonfound_`v'") ("beta_threshold") (r(beta_threshold))
    post `H' ("pkonfound_`v'") ("percent_bias") (r(perc_bias_to_change))
    post `H' ("pkonfound_`v'") ("df") (e(df_r))
    * the RIR as pkonfound reports it by default (indx(IT) rounds its copy)
    pkonfound `b' `s' `n' `ncov'
    post `H' ("pkonfound_`v'") ("rir") (r(RIR))
}

* winsorize and trim. The data are read as doubles: winsor2 compares the
* variable with a percentile held in a macro, and on a float variable that
* comparison also trims the two observations that sit exactly on the cuts.
winsor2 tail, cuts(1 99) suffix(_w)
winsor2 tail, cuts(1 99) trim suffix(_tr)
winsor2 tail, cuts(5 95) trim suffix(_tr5)
foreach v in tail_w tail_tr tail_tr5 {
    quietly summarize `v'
    post `H' ("winsor2") ("`v'_n") (r(N))
    post `H' ("winsor2") ("`v'_sum") (r(sum))
    post `H' ("winsor2") ("`v'_min") (r(min))
    post `H' ("winsor2") ("`v'_max") (r(max))
}

postclose `H'
use "_ar_stata_tmp.dta", clear
format value %24.17g
export delimited using "accounting_research_Stata.csv", replace
erase "_ar_stata_tmp.dta"
