*! 86_lprobust.do
*! Golden numbers for tests/reference_parity/test_lprobust_parity.py
*!
*! lprobust (nprobust) is the engine did_had is built on: did_had reads
*! tau_us / tau_bc / se_rb off e(Result) to get the counterfactual
*! outcome evolution of a quasi-untreated group at dose zero. Pinning it
*! on its own, at SUPPLIED bandwidths, is a precondition for reproducing
*! did_had -- bandwidth selection is a separate concern and conflating
*! the two makes a mismatch impossible to localize.
*!
*! Sweeps three kernels x three bandwidths, plus one case with b != h.
*! That last case is the one that catches whether the fit spans the
*! UNION of the h- and b-windows (N = 317) or only the h-window (241).
*!
*! Defaults exercised here: p(1), deriv(0), vce(nn) with 3 neighbours.
*!
*! Requires: nprobust. Install with
*!   net install nprobust, from(https://raw.githubusercontent.com/nppackages/nprobust/master/stata)
*! Stata 18 MP.

version 17
clear all
set more off

import delimited "data_86_lprobust.csv", clear
* The CSV round-trips through Stata's default float; recast so the
* comparison measures the estimator rather than single precision.
recast double d y, force
gen g0 = 0 if _n==1

capture program drop _row
program define _row
    args fh label first
    matrix R = e(Result)
    if !`first' file write `fh' "," _n
    file write `fh' `"  ""' "`label'" `"": {"N": "' %14.0f (R[1,4]) ///
        `", "tau_us": "' %22.17f (R[1,5]) `", "tau_bc": "' %22.17f (R[1,6]) ///
        `", "se_us": "' %22.17f (R[1,7]) `", "se_rb": "' %22.17f (R[1,8]) "}"
end

tempname fh
file open `fh' using "results/86_lprobust_Stata.json", write replace
file write `fh' "{" _n

local first 1
foreach k in epanechnikov triangular uniform {
    foreach hh in 0.5 0.8 1.5 {
        qui lprobust y d, eval(g0) h(`hh') b(`hh') kernel(`k')
        _row `fh' "`k'_h`hh'" `first'
        local first 0
    }
}

* b != h: exercises the union-of-windows behaviour.
qui lprobust y d, eval(g0) h(0.8) b(1.2) kernel(epanechnikov)
_row `fh' "epa_h0.8_b1.2" 0

file write `fh' _n `"  , "_meta": {"cmd": "lprobust", "vce": "nn(3)", "p": 1, "deriv": 0, "stata": "18 MP"}"' _n
file write `fh' "}" _n
file close `fh'

di as txt "wrote results/86_lprobust_Stata.json"
