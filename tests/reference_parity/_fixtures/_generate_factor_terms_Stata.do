* Reference numbers for tests/reference_parity/test_native_factor_terms_stata.py
* (agent probe 2026-11-13, item #33). Stata 18 MP on the castle-doctrine
* panel bundled with StatsPAI; one "tag,kind,name,value" row per number.
* Run from the repository root:
*   stata-mp -q -b do tests/reference_parity/_fixtures/_generate_factor_terms_Stata.do
* Needs from SSC: reghdfe (+ ftools).
version 18
clear all
set more off
set type double

cap program drop dumpe
program define dumpe
    args tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    forvalues i = 1/`k' {
        local nm : word `i' of `names'
        file write fh "`tag',b,`nm'," %24.17e (`b'[1,`i']) _n
        file write fh "`tag',se,`nm'," %24.17e (sqrt(`V'[`i',`i'])) _n
    }
    file write fh "`tag',s,N," %24.17e (e(N)) _n
end

file open fh using "tests/reference_parity/_fixtures/factor_terms_Stata.csv", write replace
file write fh "tag,kind,name,value" _n

import delimited using "src/statspai/datasets/data/castle_2013.csv", clear asdouble
xtset sid year

* i.year beside the unit effects, classical and clustered
xtreg l_homicide post unemployrt i.year, fe
dumpe xtreg_iyear_iid
xtreg l_homicide post unemployrt i.year, fe vce(cluster sid)
dumpe xtreg_iyear_cluster

* a named base level
xtreg l_homicide post unemployrt ib2005.year, fe vce(cluster sid)
dumpe xtreg_ib2005_cluster

* factor by continuous, every level (no main effect of the continuous term)
xtreg l_homicide post i.year#c.unemployrt, fe vce(cluster sid)
dumpe xtreg_year_by_c_cluster

* full factorial of a continuous term and the year
xtreg l_homicide post c.unemployrt##i.year, fe vce(cluster sid)
dumpe xtreg_c_hash_year_cluster

* reghdfe on the same models
reghdfe l_homicide post unemployrt i.year, absorb(sid) vce(cluster sid)
dumpe reghdfe_iyear_cluster
reghdfe l_homicide post unemployrt i.year, absorb(sid)
dumpe reghdfe_iyear_iid

file close fh
