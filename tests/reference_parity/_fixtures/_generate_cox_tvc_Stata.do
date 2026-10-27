* Reference numbers for sp.cox(tvc=, texp=) -- Stata stcox, tvc() texp().
* Run from tests/reference_parity/_fixtures:
*   stata-mp -b do _generate_cox_tvc_Stata.do
* Writes cox_tvc_Stata.csv (spec, coefficient name, b, se, N, ll).
* Convergence is tightened so the reference is the maximiser, not a
* point within ml's default tolerance of it.
version 18
clear all
set more off
set type double

tempname fh
file open `fh' using "cox_tvc_Stata.csv", write replace
file write `fh' "spec,name,b,se,N,ll" _n

capture program drop _dump
program define _dump
    args fh spec
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        file write `fh' "`spec',`nm'," %24.17g (`b'[1,`j']) "," ///
            %24.17g (sqrt(`V'[`j',`j'])) "," (e(N)) "," %24.17g (e(ll)) _n
    }
end

local tol "nrtolerance(1e-13) ltolerance(1e-15) tolerance(1e-13)"

* ---- continuous times, no ties ------------------------------------
import delimited using "cox_tvc_cont.csv", clear asdouble
stset time, failure(died)

stcox x1 race, tvc(race) breslow `tol'
_dump `fh' c1_breslow_t
stcox x1 race, tvc(race) texp(ln(_t)) efron `tol'
_dump `fh' c2_efron_ln
stcox x1, tvc(race x1) texp(_t^2) efron vce(robust) `tol'
_dump `fh' c3_efron_t2_robust
stcox x1 race, tvc(race) texp(min(_t, 4)) breslow `tol'
_dump `fh' c4_breslow_min

* ---- integer times, heavy ties ------------------------------------
import delimited using "cox_tvc_ties.csv", clear asdouble
stset time, failure(died)

stcox x1 race, tvc(race) efron `tol'
_dump `fh' t1_efron_t
stcox x1 race, tvc(race x1) texp(ln(_t)) breslow vce(robust) `tol'
_dump `fh' t2_breslow_ln_robust
stcox x1 race, tvc(race) efron strata(grp) vce(cluster cl) `tol'
_dump `fh' t3_efron_strata_cluster
stcox x1 race, tvc(race) texp(_t > 5) efron `tol'
_dump `fh' t5_efron_step

stset time [pweight = w], failure(died)
stcox x1 race, tvc(race) breslow `tol'
_dump `fh' t4_breslow_pw

stset time, failure(died) time0(entry)
stcox x1 race, tvc(race) texp(ln(_t)) efron `tol'
_dump `fh' t6_efron_entry_ln

file close `fh'
