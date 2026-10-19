* Frozen reference for sp.finegray(weights=) vs Stata 18 stcrreg after
* stset [pw=]. Run from the repository root:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_finegray_weighted_stata.do
* Data: _fixtures/survival_epi_cr.csv (the competing-risks fixture of
* test_survival_epi_R_parity.py) with deterministic weights
* w = 1 + 0.25 * mod(_n - 1, 7) and w2 = exp(0.3 * x1).
clear all
set type double
local fx "tests/reference_parity/_fixtures"
import delimited using "`fx'/survival_epi_cr.csv", clear
gen double w = 1 + 0.25*mod(_n-1, 7)
gen double w2 = exp(0.3*x1)
file open fh using "`fx'/finegray_weighted_stata.json", write replace
file write fh "{" _n
local first 1
foreach wv in w w2 {
    foreach c in 1 2 {
        local other = 3 - `c'
        stset time [pw=`wv'], failure(status==`c')
        stcrreg x1 x2, compete(status==`other')
        matrix b = e(b)
        matrix V = e(V)
        if !`first' file write fh "," _n
        local first 0
        file write fh `""`wv'_cause`c'": {"b": ["' %24.17e (b[1,1]) ", " %24.17e (b[1,2]) `"], "V": ["' %24.17e (V[1,1]) ", " %24.17e (V[1,2]) ", " %24.17e (V[2,1]) ", " %24.17e (V[2,2]) `"], "ll": "' %24.17e (e(ll)) `", "N": "' (e(N)) "}"
    }
}
file write fh "," _n `""provenance": {"stata": "`c(stata_version)'", "command": "stset time [pw=w], failure(status==c); stcrreg x1 x2, compete(status==other)"}"' _n "}" _n
file close fh
