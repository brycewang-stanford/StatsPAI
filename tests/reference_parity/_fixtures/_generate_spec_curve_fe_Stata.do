* Coefficient, SE, p-value and N of x for the sp.spec_curve(fe=) fixture:
* reghdfe for the fixed-effect specifications, regress for the others.
*   python tests/reference_parity/_fixtures/_generate_spec_curve_fe_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_spec_curve_fe_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/spec_curve_fe.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/spec_curve_fe_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach fe in none firm firmyear {
    foreach vce in unadjusted robust cluster {
        local v `vce'
        if "`vce'" == "cluster" local v "cluster ind"
        if "`vce'" == "unadjusted" & "`fe'" == "none" local v "ols"
        if "`fe'" == "none" qui regress y x w, vce(`v')
        if "`fe'" == "firm" qui reghdfe y x w, absorb(firm) vce(`v')
        if "`fe'" == "firmyear" qui reghdfe y x w, absorb(firm year) vce(`v')
        local df = e(df_r)
        local p = 2 * ttail(`df', abs(_b[x] / _se[x]))
        if !`first' file write `fh' "," _n
        local first 0
        file write `fh' `"  "`fe'_`vce'": {"b": "' %25.17f (_b[x]) `", "se": "' %25.17f (_se[x]) `", "p": "' %25.17f (`p') `", "N": "' (e(N)) "}"
    }
}
file write `fh' _n "}" _n
file close `fh'
