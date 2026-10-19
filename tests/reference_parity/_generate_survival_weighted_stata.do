* Frozen-reference generator: Stata 18 stcox with [pw=], vce(robust),
* vce(cluster), enter() and multi-record (time0() + id()) data, for the
* sp.cox(ssc='stata') tests. Run from the repository root:
*   stata-mp -b do tests/reference_parity/_generate_survival_weighted_stata.do
* Stata refuses pweights with efron ties ("weights not allowed with efron
* option"), so the weighted fits use breslow (stcox's default).
version 18
set type double
local fx "tests/reference_parity/_fixtures"
import delimited using "`fx'/survival_weighted_data.csv", clear
gen double zw = w
replace zw = 0 in 1/3
tempname fh
file open `fh' using "`fx'/survival_weighted_Stata.csv", write replace
file write `fh' "tag,j,coef,se,N" _n
capture program drop dump
program define dump
    args fh tag
    matrix b = e(b)
    matrix V = e(V)
    forvalues j = 1/`=colsof(b)' {
        file write `fh' "`tag'," (`j') "," %21.0g (b[1,`j']) "," %21.0g (sqrt(V[`j',`j'])) "," (e(N)) _n
    }
end
stset time [pw=w], failure(event)
stcox x1 x2 a, nohr breslow
dump `fh' pw_breslow
stcox x1 x2 a, nohr breslow vce(cluster cl)
dump `fh' pw_cluster_breslow
stset time, failure(event)
stcox x1 x2 a, nohr efron vce(robust)
dump `fh' unw_robust_efron
stcox x1 x2 a, nohr breslow vce(robust)
dump `fh' unw_robust_breslow
stset time [pw=zw], failure(event)
stcox x1 x2 a, nohr breslow
dump `fh' zw_breslow
stset time, failure(event) enter(time entry)
stcox x1 x2 a, nohr efron
dump `fh' entry_efron
stcox x1 x2 a, nohr efron vce(robust)
dump `fh' entry_robust_efron
import delimited using "`fx'/survival_counting_data.csv", clear
stset tstop, failure(event) time0(tstart) id(id)
stcox z x2 a, nohr efron vce(robust)
dump `fh' cp_id_efron
file close `fh'
