* Stata boottest (Roodman et al. 2019) reference for sp.hdfe_ols(wild=True).
* areg with i.year as regressors, unit effects absorbed, clusters of units:
* the year effect is not nested in the cluster.  Rademacher weights with
* reps above 2^12 enumerate every sign vector, so the p-value is exact.
*   python tests/reference_parity/_fixtures/_generate_hdfe_wild_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_hdfe_wild_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/hdfe_wild_panel.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/hdfe_wild_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach spec in one two {
    if "`spec'" == "one" local rhs d
    if "`spec'" == "two" local rhs d x2
    qui areg y `rhs' i.year, absorb(unit) cluster(cl)
    foreach v of local rhs {
        qui boottest `v', reps(10000) weight(rademacher) nograph
        local p = r(p)
        matrix C = r(CI)
        if !`first' file write `fh' "," _n
        local first 0
        file write `fh' `"  "`spec'_`v'": {"p": "' %25.17f (`p')
        file write `fh' `", "lo": "' %25.17f (C[1,1]) `", "hi": "' %25.17f (C[1,2])
        file write `fh' `", "reps": "' (r(reps)) "}"
    }
}
file write `fh' _n "}" _n
file close `fh'
