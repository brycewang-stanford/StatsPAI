* Cluster-robust Anderson-Rubin statistic of ivreg2 (4.1.x) at a grid of
* null values b0: ivreg2 on y - b0*x reports AR(b0) as its e(arf) / e(arfp).
* Data: iv_wild_data.csv (10 clusters).
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_ar_ci_cluster_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/iv_wild_data.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/ar_ci_cluster_Stata.json", write replace
file write `fh' "[" _n
local first 1
foreach b0 of numlist -0.5 0 0.1 0.3 0.5 0.8 1.2 {
    capture drop yt
    gen double yt = y - `b0' * x
    qui ivreg2 yt w (x = z), cluster(cl) ffirst
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  {"b0": "' %25.17f (`b0') `", "arf": "' %25.17f (e(arf)) `", "arfp": "' %25.17f (e(arfp)) "}"
}
file write `fh' _n "]" _n
file close `fh'
