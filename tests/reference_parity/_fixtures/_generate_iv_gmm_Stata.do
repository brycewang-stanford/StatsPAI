* ivregress gmm on the Card data (over-identified: nearc4, nearc2), for
* sp.iv(method='gmm'). Two-step GMM under each weight matrix Stata
* offers, with and without -small-: coefficient and SE of educ, and
* Hansen's J with its p-value.
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_iv_gmm_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/iv_card.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/iv_gmm_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach wm in "unadjusted" "robust" "cluster cl" {
    foreach sm in "" "small" {
        qui ivregress gmm lwage exper expersq black south smsa (educ = nearc4 nearc2), wmatrix(`wm') `sm'
        local key = word("`wm'", 1) + cond("`sm'" == "", "", "_small")
        local b = _b[educ]
        local se = _se[educ]
        local bx = _b[exper]
        local sex = _se[exper]
        qui estat overid
        local J = cond(r(HansenJ) < ., r(HansenJ), .)
        local Jp = cond(r(p_HansenJ) < ., r(p_HansenJ), .)
        if !`first' file write `fh' "," _n
        local first 0
        file write `fh' `"  "`key'": {"b": "' %24.16e (`b') `", "se": "' %24.16e (`se') `", "b_exper": "' %24.16e (`bx') `", "se_exper": "' %24.16e (`sex') `", "J": "' %24.16e (`J') `", "J_p": "' %24.16e (`Jp') `", "N": "' %9.0f (e(N)) "}"
    }
}
file write `fh' _n "}" _n
file close `fh'
