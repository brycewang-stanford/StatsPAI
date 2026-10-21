* Reference for sp.bacon_decomposition(weights=): Stata bacondecomp
* (Goodman-Bacon, Goldring, Nichols) with analytic weights, on the mpdta
* replica of Track A module 20 (tests/r_parity/data/20_bacon.csv).
* The unit weight w = 1 + mod(countyreal, 7) is constant within county;
* the covariate x = mod(31 * countyreal + 17 * year, 101) / 10 varies over
* time. Both are rebuilt the same way in the test.
*
* Stata reports the decomposition by control type (e(sumdd): Timing_groups,
* Always_v_timing, Never_v_timing, plus Always_v_never and Within with
* controls), and the TWFE coefficient _b[post].
*
* Run from the repository root with Stata 18 and bacondecomp on the adopath:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_bacon_weights_Stata.do
version 18
set type double
clear all

import delimited "tests/r_parity/data/20_bacon.csv", clear asdouble
generate double w = 1 + mod(countyreal, 7)
generate double x = mod(31 * countyreal + 17 * year, 101) / 10
xtset countyreal year

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/bacon_weights_Stata.json", write replace text
file write `fh' "{" _n

capture program drop _dump
program define _dump
    args fh key last
    matrix S = e(sumdd)
    local names : rownames S
    file write `fh' `"  "`key'": {"b": "' %24.16e (_b[treat]) `", "rows": {"'
    local n = rowsof(S)
    forvalues i = 1/`n' {
        local nm : word `i' of `names'
        file write `fh' `""`nm'": ["' %24.16e (S[`i', 1]) ", " %24.16e (S[`i', 2]) "]"
        if `i' < `n' file write `fh' ", "
    }
    file write `fh' "}}"
    if "`last'" == "" file write `fh' ","
    file write `fh' _n
end

quietly bacondecomp lemp treat, nograph
_dump `fh' unweighted
quietly bacondecomp lemp treat [aw=w], nograph
_dump `fh' aweight
quietly bacondecomp lemp treat x [aw=w], nograph
_dump `fh' aweight_x last
file write `fh' "}" _n
file close `fh'
