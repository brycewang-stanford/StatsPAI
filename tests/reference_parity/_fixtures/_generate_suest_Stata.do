* Stata suest after three regress fits (different samples), robust and
* clustered: the x coefficients, their joint covariance and two tests.
*   python tests/reference_parity/_fixtures/_generate_suest_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_suest_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/suest_data.csv", clear asdouble
qui reg y1 x w
est store e1
qui reg y2 x
est store e2
qui reg y3 x w
est store e3
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/suest_Stata.json", write replace
file write `fh' "{" _n
foreach v in robust cluster {
    if "`v'" == "robust" qui suest e1 e2 e3, vce(robust)
    else qui suest e1 e2 e3, vce(cluster cl)
    matrix V = e(V)
    matrix b = e(b)
    local names e1_mean:x e1_mean:w e1_mean:_cons e2_mean:x e2_mean:_cons e3_mean:x e3_mean:w e3_mean:_cons
    file write `fh' `"  "`v'": {"b": ["'
    local first 1
    foreach r of local names {
        if !`first' file write `fh' ", "
        local first 0
        file write `fh' %25.17f (b[1, "`r'"])
    }
    file write `fh' "], " _n `"    "V": ["'
    local first 1
    foreach r of local names {
        foreach c of local names {
            if !`first' file write `fh' ", "
            local first 0
            file write `fh' %25.17f (V["`r'", "`c'"])
        }
    }
    file write `fh' "], " _n
    qui test [e1_mean]x = [e2_mean]x = [e3_mean]x
    file write `fh' `"    "eq_chi2": "' %25.17f (r(chi2)) `", "eq_df": "' (r(df))
    qui test [e1_mean]x [e2_mean]x [e3_mean]x
    file write `fh' `", "zero_chi2": "' %25.17f (r(chi2)) "}"
    if "`v'" == "robust" file write `fh' ","
    file write `fh' _n
}
file write `fh' "}" _n
file close `fh'
