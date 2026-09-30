* did_imputation, autosample: overall ATT and an event study with pre-trends
* on a panel with two always-treated units.
*   python tests/reference_parity/_fixtures/_generate_bjs_autosample_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_bjs_autosample_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/bjs_autosample.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/bjs_autosample_Stata.json", write replace
capture did_imputation y i t g
file write `fh' "{" _n `"  "rc_default": "' (_rc) "," _n
qui did_imputation y i t g, autosample
file write `fh' `"  "att": "' %24.16e (_b[tau]) `", "se": "' %24.16e (_se[tau]) `", "N": "' (e(N)) "," _n
qui did_imputation y i t g, autosample horizons(0/3) pretrends(2) minn(0)
file write `fh' `"  "es": {"'
local first 1
foreach c in tau0 tau1 tau2 tau3 pre1 pre2 {
    if !`first' file write `fh' ", "
    local first 0
    file write `fh' `""`c'": ["' %24.16e (_b[`c']) ", " %24.16e (_se[`c']) "]"
}
file write `fh' "}" _n "}" _n
file close `fh'
