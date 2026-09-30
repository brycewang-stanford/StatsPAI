* reghdfe fit statistics for tests/reference_parity/test_hdfe_fitstats_Stata_parity.py
version 18
clear all
import delimited using "reghdfe_fitstats.csv", clear asdouble
egen cy = group(city year)
tempname fh
file open `fh' using "reghdfe_fitstats_Stata.json", write replace
file write `fh' "{" _n
local specs `" "A|id year|" "B|id year|id" "C|id city#year|cy" "'
local first 1
foreach s of local specs {
    tokenize "`s'", parse("|")
    local tag `1'
    local ab `3'
    local cl `5'
    if "`cl'" == "|" local cl
    if "`cl'" == "" reghdfe y x z, absorb(`ab')
    else reghdfe y x z, absorb(`ab') vce(cluster `cl')
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `""`tag'": {"N": "' %12.0f (e(N)) `", "df_a": "' %12.0f (e(df_a)) `", "df_a_nested": "' %12.0f (e(df_a_nested)) `", "r2": "' %24.17e (e(r2)) `", "r2_a": "' %24.17e (e(r2_a)) `", "r2_within": "' %24.17e (e(r2_within)) `", "r2_a_within": "' %24.17e (e(r2_a_within)) `", "rmse": "' %24.17e (e(rmse)) `"}"'
}
file write `fh' _n "}" _n
file close `fh'
