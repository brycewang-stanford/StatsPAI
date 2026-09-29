* Stata ppmlhdfe reference for tests/reference_parity/test_ppmlhdfe_singletons_Stata_parity.py
* Singletons + separation dropped jointly, an absorbed regressor (ever) omitted,
* an interacted absorb (ind#year) and interacted cluster (city#year).
version 18
clear all
import delimited using "ppmlhdfe_singletons.csv", clear
egen cy = group(city year)
tempname fh
file open `fh' using "ppmlhdfe_singletons_Stata.json", write replace
file write `fh' "{" _n
local specs `" "A|y x1 d ever|id year|id" "B|y x1 d|id ind#year|cy" "'
local first 1
foreach s of local specs {
    tokenize "`s'", parse("|")
    local tag `1'
    local vars `3'
    local ab `5'
    local cl `7'
    ppmlhdfe `vars', absorb(`ab') vce(cluster `cl')
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `""`tag'": {"N": "' %21.0g (e(N)) `", "N_clust": "' %21.0g (e(N_clust)) `", "num_singletons": "' %21.0g (e(num_singletons)) `", "r2_p": "' %24.17e (e(r2_p))
    foreach v in x1 d {
        file write `fh' `", "b_`v'": "' %24.17e (_b[`v']) `", "se_`v'": "' %24.17e (_se[`v'])
    }
    file write `fh' "}"
}
file write `fh' _n "}" _n
file close `fh'
