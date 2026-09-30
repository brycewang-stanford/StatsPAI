* psacalc after xtreg, fe for tests/reference_parity/test_oster_absorb_Stata_parity.py
version 18
clear all
import delimited using "reghdfe_fitstats.csv", clear asdouble
xtset id year
tempname fh
file open `fh' using "oster_absorb_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach v in plain cluster {
    if "`v'" == "plain" qui xtreg y x z i.year i.city#i.year, fe
    else qui xtreg y x z i.year i.city#i.year, fe vce(cluster city)
    * plain: R_max = 1.3 x within R2 (e(r2_a) < 0 on this small panel); cluster: 1.3 x e(r2_a)
    if "`v'" == "plain" local r = e(r2_w)*1.3
    else local r = e(r2_a)*1.3
    local r2a = e(r2_a)
    psacalc delta x, rmax(`r') beta(0)
    local d = r(delta)
    psacalc beta x, rmax(`r') delta(-1)
    local b = r(beta)
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `""`v'": {"r2_a": "' %24.17e (`r2a') `", "rmax": "' %24.17e (`r') `", "delta": "' %24.17e (`d') `", "beta": "' %24.17e (`b') `"}"'
}
file write `fh' _n "}" _n
file close `fh'
