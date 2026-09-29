* Stata reference for tests/reference_parity/test_logit_perfect_prediction_Stata_parity.py
version 18
clear all
import delimited using "logit_perfect_prediction.csv", clear asdouble
tempname fh
file open `fh' using "logit_perfect_prediction_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach m in logit probit {
    `m' y x i.g
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `""`m'": {"N": "' %12.0f (e(N)) `", "r2_p": "' %24.17e (e(r2_p)) `", "ll": "' %24.17e (e(ll)) `", "b_x": "' %24.17e (_b[x]) `", "se_x": "' %24.17e (_se[x]) `", "b_g3": "' %24.17e (_b[3.g]) `", "se_g3": "' %24.17e (_se[3.g]) `"}"'
}
file write `fh' _n "}" _n
file close `fh'
