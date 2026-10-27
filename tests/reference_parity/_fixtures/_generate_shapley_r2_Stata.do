* Reference for sp.shapley_r2: Stata shapley2 (SSC, F. Chavez Juarez,
* version 1.5 10jun15) with stat(r2) after regress, ungrouped and with its
* group() option. Data: shapley_r2_data.csv (n = 300).
*
* shapley2 re-runs `regress depvar subset if e(sample)` for every subset
* and stores e(r2); it ignores weights and has no always-included
* controls, so those two cases are referenced against R only.
*
* Install shapley2 into a private ado tree first (never the user's PLUS):
*   sysdir set PLUS "<scratch>/ado/plus"
*   ssc install shapley2
* Run from the repo root:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_shapley_r2_Stata.do
version 18
clear all
set type double
import delimited using "tests/reference_parity/_fixtures/shapley_r2_data.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/shapley_r2_Stata.json", write replace
file write `fh' "{" _n

* ungrouped, six players
qui regress y x1 x2 x3 x4 x5 x6
local r2 = e(r2)
shapley2, stat(r2)
matrix S = e(shapley)
file write `fh' `"  "ungrouped": {"r2": "' %24.16e (`r2') `", "shapley": ["'
forvalues j = 1/6 {
    if `j' > 1 file write `fh' ", "
    file write `fh' %24.16e (S[`j', 1])
}
file write `fh' "]}," _n

* grouped: {x1 x2}, {x3}, {x4 x5 x6}
qui regress y x1 x2 x3 x4 x5 x6
shapley2, stat(r2) group(x1 x2, x3, x4 x5 x6)
matrix G = e(shapley)
file write `fh' `"  "grouped": {"r2": "' %24.16e (`r2') `", "shapley": ["'
forvalues j = 1/3 {
    if `j' > 1 file write `fh' ", "
    file write `fh' %24.16e (G[`j', 1])
}
file write `fh' "]}," _n

* the subset R2 values that shapley2 averages, for the full player set:
* regress on each single regressor (a check on the per-subset inputs)
file write `fh' `"  "single_r2": ["'
forvalues j = 1/6 {
    qui regress y x`j'
    if `j' > 1 file write `fh' ", "
    file write `fh' %24.16e (e(r2))
}
file write `fh' "]" _n "}" _n
file close `fh'
