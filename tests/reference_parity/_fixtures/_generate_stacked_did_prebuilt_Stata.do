* reghdfe (6.12.3) on the pre-built CDLZ stack: event-time dummies for the
* treated units (k = -1 omitted), unit x event and year x event effects,
* clustered by unit; unweighted and [aw = pop].
*   python tests/reference_parity/_fixtures/_generate_stacked_did_prebuilt_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_stacked_did_prebuilt_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/stacked_did_prebuilt.csv", clear asdouble
foreach k of numlist -3 -2 0 1 2 3 {
    local nm = cond(`k' < 0, "m" + string(-`k'), "p" + string(`k'))
    gen D_`nm' = (rel == `k') * treated
}
egen uc = group(id event)
egen tc = group(year event)
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/stacked_did_prebuilt_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach w in none pop {
    if "`w'" == "none" qui reghdfe y D_*, absorb(uc tc) cluster(id)
    else qui reghdfe y D_* [aw = pop], absorb(uc tc) cluster(id)
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  "`w'": {"N": "' (e(N))
    foreach nm in m3 m2 p0 p1 p2 p3 {
        file write `fh' `", "b_`nm'": "' %25.17f (_b[D_`nm']) `", "se_`nm'": "' %25.17f (_se[D_`nm'])
    }
    qui lincom (D_p0 + D_p1 + D_p2 + D_p3) / 4
    file write `fh' `", "att": "' %25.17f (r(estimate)) `", "att_se": "' %25.17f (r(se)) "}"
}
file write `fh' _n "}" _n
file close `fh'
