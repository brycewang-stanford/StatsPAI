* Stacked PPML reference for tests/reference_parity/test_stacked_ppml_Stata_parity.py
* Stacking rule: cohort g's treated units, plus later-treated / never-treated
* units in the periods before their own treatment, window [g-3, g+2].
version 18
clear all
tempfile stack
import delimited using "stacked_ppml.csv", clear asdouble
gen double ft = cond(g == 0, 9999, g)
save `stack', replace emptyok
clear
tempfile out
save `out', emptyok
foreach c in 2013 2015 2016 {
    use `stack', clear
    keep if ft == `c' | (ft > `c' & year < ft)
    keep if year >= `c' - 3 & year <= `c' + 2
    gen cohort = `c'
    gen treated = ft == `c'
    gen rel = year - `c'
    append using `out'
    save `out', replace
}
use `out', clear
gen tp = treated * (rel >= 0)
egen uc = group(id cohort)
egen tc = group(year cohort)
forvalues k = 3(-1)2 {
    gen Dm`k' = treated * (rel == -`k')
}
forvalues k = 0/2 {
    gen D`k' = treated * (rel == `k')
}
tempname fh
file open `fh' using "stacked_ppml_Stata.json", write replace
ppmlhdfe y tp x, absorb(uc tc) vce(cluster city)
file write `fh' `"{"pooled": {"b": "' %24.17e (_b[tp]) `", "se": "' %24.17e (_se[tp]) `", "N": "' %12.0f (e(N)) `"},"' _n
ppmlhdfe y Dm3 Dm2 D0 D1 D2 x, absorb(uc tc) vce(cluster city)
file write `fh' `""event": {"b": ["' %24.17e (_b[Dm3]) `", "' %24.17e (_b[Dm2]) `", "' %24.17e (_b[D0]) `", "' %24.17e (_b[D1]) `", "' %24.17e (_b[D2]) `"], "se": ["' %24.17e (_se[Dm3]) `", "' %24.17e (_se[Dm2]) `", "' %24.17e (_se[D0]) `", "' %24.17e (_se[D1]) `", "' %24.17e (_se[D2]) `"], "N": "' %12.0f (e(N)) `"},"' _n
lincom (D0 + D1 + D2) / 3
file write `fh' `""event_att": {"b": "' %24.17e (r(estimate)) `", "se": "' %24.17e (r(se)) `"},"' _n
ppmlhdfe y tp x, absorb(uc tc id city#year) vce(cluster city)
file write `fh' `""pooled_absorb": {"b": "' %24.17e (_b[tp]) `", "se": "' %24.17e (_se[tp]) `", "N": "' %12.0f (e(N)) `"}}"' _n
file close `fh'
