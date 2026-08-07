*! 87_did_had.do
*! Golden numbers for tests/reference_parity/test_did_had_parity.py
*!
*! did_had (de Chaisemartin, Ciccia, D'Haultfoeuille & Knau,
*! arXiv:2405.04465) estimates a treatment effect in a heterogeneous
*! ADOPTION design: every group is untreated in period 1 and every
*! treated group adopts at the SAME period F, with heterogeneous dose.
*! There is no untreated group -- only "quasi-untreated" ones whose dose
*! is near zero -- so the counterfactual comes from the local polynomial
*! intercept at dose 0 rather than from a control group.
*!
*! The fixture panel has 300 groups, 6 periods, F = 4, doses from a
*! Gamma(1.3, 0.5) with 25 exact-zero-dose stayers, and an effect that
*! grows with exposure and is nonlinear in dose. That combination gives
*! quasi-untreated groups (so the QUG test does not reject), non-flat
*! effects, and flat placebos.
*!
*! e(estimates) is 5 x 10:
*!   Estimate, SE, LB CI, UB CI, N, BW, N in BW, QUG T, QUG p, rel. time
*!
*! NOTE the CI is NOT symmetric around Estimate. did_had reports the
*! conventional point estimate with a BIAS-CORRECTED interval, centred at
*! beta_qs - B_hat where B_hat = -(tau_us - tau_bc)/mean(dose).
*!
*! Requires: did_had, nprobust (for lprobust). Stata 18 MP.
*!   net install nprobust, from(https://raw.githubusercontent.com/nppackages/nprobust/master/stata)

version 17
clear all
set more off

import delimited "data_87_did_had.csv", clear
recast double d y, force

qui did_had y g t d, effects(3) placebo(2) graph_off
matrix R = e(estimates)

tempname fh
file open `fh' using "results/87_did_had_Stata.json", write replace
file write `fh' "{" _n

local rn : rownames R
local nr = rowsof(R)
forvalues i = 1/`nr' {
    local nm : word `i' of `rn'
    local comma ","
    if `i' == `nr' local comma ""
    * Stata missing (.) is not valid JSON: placebos carry no QUG test.
    local qt = cond(missing(R[`i',8]), "null", string(R[`i',8], "%22.17f"))
    local qp = cond(missing(R[`i',9]), "null", string(R[`i',9], "%22.17f"))
    file write `fh' `"  ""' "`nm'" `"": {"estimate": "' %22.17f (R[`i',1]) ///
        `", "se": "' %22.17f (R[`i',2]) `", "ci_lo": "' %22.17f (R[`i',3]) ///
        `", "ci_hi": "' %22.17f (R[`i',4]) `", "n_groups": "' %10.0f (R[`i',5]) ///
        `", "bw": "' %22.17f (R[`i',6]) `", "n_in_bw": "' %10.0f (R[`i',7]) ///
        `", "qug_t": "' "`qt'" `", "qug_p": "' "`qp'" ///
        `", "rel_time": "' %6.0f (R[`i',10]) "}`comma'" _n
}
file write `fh' `"  , "_meta": {"cmd": "did_had", "effects": 3, "placebo": 2, "kernel": "epanechnikov", "bw_method": "mse-dpi", "stata": "18 MP"}"' _n
file write `fh' "}" _n
file close `fh'

di as txt "wrote results/87_did_had_Stata.json"
