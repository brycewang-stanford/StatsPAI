* Stata 18 reference for sp.mi_test (tests/reference_parity/test_mi_test_parity.py).
* Input: mi_test_flong.csv, a flong export (_mi_m = 0 original, 1..8 imputed by
* `mi impute chained (regress) x2 x3 = y x1, add(8) rseed(11)`, exported %21.0g).
* Run from this directory: stata -b do _generate_mi_test_stata.do
*
* Two blocks:
*   "m8_*" / "m3_*"  the original large-sample (nosmall) cells, full sample.
*   "grid"           every combination of sample size (id <= 400 / 30 / 15 / 7, so
*                    the complete-data df is 396 / 26 / 11 / 3), imputations
*                    (first 2 / 3 / 4 / 5 / 8), test (equal FMI / ufmitest),
*                    tested terms (k = 1..4) and df (Stata default small-sample
*                    / nosmall). k(M-1) and M(k-1) straddle 4 in both orders,
*                    so the grid separates the df branches of the equal FMI test.
*                    Grid numbers are written %24.17e (Stata's %g drops the
*                    leading zero, which is not valid JSON).
version 18
clear all
set more off
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
file write `fh' "," _n `"  "grid": ["' _n
local first 1
foreach n in 400 30 15 7 {
    foreach m in 2 3 4 5 8 {
        quietly mi estimate, imputations(1/`m'): regress y x1 x2 x3 if id <= `n'
        local dfcom = e(df_r)
        foreach kind in equal unres {
            foreach terms in "x2" "x2 x3" "x1 x2 x3" "x1 x2 x3 _cons" {
                foreach small in 1 0 {
                    local opt
                    if !`small' local opt nosmall
                    if "`kind'" == "unres" local opt `opt' ufmitest
                    quietly mi test `terms', `opt'
                    if !`first' file write `fh' "," _n
                    local first 0
                    file write `fh' `"    {"n": `n', "m": `m', "kind": "`kind'", "terms": "`terms'", "small": `small', "dfcom": "' %24.17e (`dfcom') `", "F": "' %24.17e (r(F)) `", "df1": "' %3.0f (r(df)) `", "df2": "' %24.17e (r(df_r)) `", "p": "' %24.17e (r(p)) "}"
                }
            }
        }
    }
}
file write `fh' _n "  ]" _n "}" _n
file close `fh'
