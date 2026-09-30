* psmatch2 ties / ate reference for tests/reference_parity/test_psmatch2_ties_ate_Stata_parity.py
* The propensity score is given (pscore()), so the check isolates the matching rule.
version 18
clear all
import delimited using "psmatch2_ties.csv", clear asdouble
gen long row = _n
tempname fh
file open `fh' using "psmatch2_ties_Stata.json", write replace
file write `fh' "{" _n
psmatch2 treat, pscore(ps) outcome(y) neighbor(1) ties common
file write `fh' `""ties": {"att": "' %24.17e (r(att)) `", "seatt": "' %24.17e (r(seatt)) `"},"' _n
gen double w_ties = _weight
gen byte s_ties = _support
psmatch2 treat, pscore(ps) outcome(y) neighbor(1) ties ate common
file write `fh' `""ties_ate": {"att": "' %24.17e (r(att)) `", "seatt": "' %24.17e (r(seatt)) `", "atu": "' %24.17e (r(atu)) `", "ate": "' %24.17e (r(ate)) `"}"' _n
file write `fh' "}" _n
file close `fh'
gen double w_ate = _weight
gen byte s_ate = _support
sort row
keep row w_ties s_ties w_ate s_ate
format w_ties w_ate %24.17g
export delimited using "psmatch2_ties_Stata.csv", replace datafmt
