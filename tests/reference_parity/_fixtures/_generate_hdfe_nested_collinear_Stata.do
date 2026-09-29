* Stata reghdfe reference for sp.hdfe_ols: absorbed degrees of freedom with a
* FE nested in another (year inside region x year) and regressors collinear
* with the FEs (omitted).
*   python tests/reference_parity/_fixtures/_generate_hdfe_nested_collinear_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_hdfe_nested_collinear_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/hdfe_nested_collinear.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/hdfe_nested_collinear_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach spec in "a|x x2|year unit regionxyear|unit" "b|x x2|unit regionxyear|unit" ///
                "c|x x2|year unit regionxyear|" "d|x x2|unit regionxyear|" ///
                "e|x x2 v_exact|unit regionxyear|unit" "f|x x2 v_near|unit regionxyear|unit" ///
                "g|x x2 v_near|year unit regionxyear|" {
    tokenize "`spec'", parse("|")
    local key `1'
    local rhs `3'
    local fe `5'
    local cl `7'
    if "`cl'" == "|" local cl ""
    if "`cl'" != "" qui reghdfe y `rhs', absorb(`fe') vce(cluster `cl')
    else qui reghdfe y `rhs', absorb(`fe')
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  "`key'": {"df_a": "' (e(df_a)) `", "N": "' (e(N))
    file write `fh' `", "b_x": "' %25.17f (_b[x]) `", "se_x": "' %25.17f (_se[x])
    file write `fh' `", "b_x2": "' %25.17f (_b[x2]) `", "se_x2": "' %25.17f (_se[x2]) "}"
}
file write `fh' _n "}" _n
file close `fh'
