* Stata 18 reference for sp.mi_test (tests/reference_parity/test_mi_test_parity.py).
* Input: mi_test_flong.csv, a flong export (_mi_m = 0 original, 1..8 imputed by
* `mi impute chained (regress) x2 x3 = y x1, add(8) rseed(11)`, exported %21.0g).
* Run from this directory: stata -b do _generate_mi_test_stata.do
version 18
clear all
import delimited using "mi_test_flong.csv", clear asdouble encoding(utf-8)
rename (_mi_m _mi_id) (imp id)
mi import flong, m(imp) id(id) imputed(x2 x3) clear
tempname fh
file open `fh' using "mi_test_stata.json", write replace
file write `fh' "{" _n
local first 1
foreach spec in "m8 1/8" "m3 1/3" {
    gettoken tag imps : spec
    local imps = strtrim("`imps'")
    quietly mi estimate, imputations(`imps'): regress y x1 x2 x3
    foreach test in "equal x2 x3" "unres x2 x3" "equal3 x1 x2 x3" {
        gettoken kind terms : test
        local opt nosmall
        if "`kind'" == "unres" local opt nosmall ufmitest
        quietly mi test `terms', `opt'
        if !`first' file write `fh' "," _n
        local first 0
        file write `fh' `"  "`tag'_`kind'": {"F": "' %21.17g (r(F)) `", "df1": "' %3.0f (r(df)) `", "df2": "' %21.17g (r(df_r)) `", "p": "' %21.17g (r(p)) "}"
    }
}
file write `fh' _n "}" _n
file close `fh'
