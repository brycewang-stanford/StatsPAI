* Reference for sp.positive_correlation_test(method='biprobit'): Stata 18
* biprobit on positive_correlation_test_data.csv (written by
* _generate_positive_correlation_test_data.py). Three cases:
*   same    : biprobit (cover = x1 x2) (claim = x1 x2), LR test of rho = 0
*   seemingly: biprobit (cover = x1 x2) (claim = x1 x3), LR test
*   cluster : same as "same" with vce(cluster g), Wald test of rho = 0
* For each: rho, athrho and its se, the chi2 of the rho = 0 test, the
* log likelihood. Run from the repository root:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_positive_correlation_test_Stata.do
version 18
set type double
clear all
import delimited "tests/reference_parity/_fixtures/positive_correlation_test_data.csv", clear asdouble

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/positive_correlation_test_Stata.json", write replace text
file write `fh' "{" _n

capture program drop _dump
program define _dump
    args fh key last
    local athrho = _b[/athrho]
    local athse = _se[/athrho]
    file write `fh' `"  "`key'": {"rho": "' %24.17e (e(rho)) `", "athrho": "' %24.17e (`athrho') `", "athrho_se": "' %24.17e (`athse') `", "chi2_c": "' %24.17e (e(chi2_c)) `", "ll": "' %24.17e (e(ll)) `", "N": "' (e(N)) `"}`last'"' _n
end

biprobit (cover = x1 x2) (claim = x1 x2)
_dump `fh' same ","
biprobit (cover = x1 x2) (claim = x1 x3)
_dump `fh' seemingly ","
biprobit (cover = x1 x2) (claim = x1 x2), vce(cluster g)
_dump `fh' cluster ""

file write `fh' "}" _n
file close `fh'
