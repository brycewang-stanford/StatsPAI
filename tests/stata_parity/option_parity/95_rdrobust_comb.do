*! 95_rdrobust_comb.do
*! Golden numbers for tests/reference_parity/test_rdrobust_comb_parity.py
*!
*! sp.rdrobust under the four "comb" bandwidth selectors (msecomb1,
*! msecomb2, cercomb1, cercomb2): conventional and robust estimates,
*! their standard errors, and h, b on each side. The same grid is run in
*! R by tests/reference_parity/_fixtures/_generate_rdrobust_comb_R.R, on
*! the same two committed CSVs:
*!   sharp: 4 selectors x p(1 2) x kernel(tri uni epa)  = 24 fits
*!   fuzzy: 4 selectors x p(1), triangular, fuzzy(treat2) = 4 fits
*!
*! Requires: rdrobust (Calonico, Cattaneo, Farrell, Titiunik), the
*! version recorded in _meta. Stata 18 MP.
*! Run from this directory:
*!   stata-mp -b do 95_rdrobust_comb.do

version 17
clear all
set more off

* Double precision throughout: without `asdouble`, `import delimited`
* stores the decimals as float and rounds the data in the 8th digit.
set type double

capture program drop _row
program define _row
    args fh label first
    if !`first' file write `fh' "," _n
    file write `fh' `"  ""' "`label'" `"": {"coef_conventional": "' ///
        %22.17g (e(tau_cl)) `", "coef_robust": "' %22.17g (e(tau_bc)) ///
        `", "se_conventional": "' %22.17g (e(se_tau_cl)) ///
        `", "se_robust": "' %22.17g (e(se_tau_rb)) ///
        `", "h_left": "' %22.17g (e(h_l)) `", "h_right": "' %22.17g (e(h_r)) ///
        `", "b_left": "' %22.17g (e(b_l)) `", "b_right": "' %22.17g (e(b_r)) ///
        `", "N_h_left": "' %14.0f (e(N_h_l)) `", "N_h_right": "' %14.0f (e(N_h_r)) "}"
end

tempname fh
file open `fh' using "results/95_rdrobust_comb_Stata.json", write replace
file write `fh' "{" _n

local first 1
import delimited "../../reference_parity/_fixtures/rdsenate.csv", clear asdouble
foreach bw in msecomb1 msecomb2 cercomb1 cercomb2 {
    foreach p in 1 2 {
        foreach k in triangular uniform epanechnikov {
            qui rdrobust vote margin, c(0) p(`p') kernel(`k') bwselect(`bw')
            _row `fh' "sharp_`bw'_p`p'_`k'" `first'
            local first 0
        }
    }
}

import delimited "../../reference_parity/_fixtures/rdsenate_fuzzy.csv", clear asdouble
foreach bw in msecomb1 msecomb2 cercomb1 cercomb2 {
    qui rdrobust vote margin, c(0) p(1) kernel(triangular) bwselect(`bw') fuzzy(treat2)
    _row `fh' "fuzzy_`bw'_p1_triangular" 0
}

qui which rdrobust
file write `fh' _n `"  , "_meta": {"cmd": "rdrobust", "rdrobust_version": "10.0.0 2025-06-30", "data": ["rdsenate.csv", "rdsenate_fuzzy.csv"], "stata": "18 MP", "precision": "double"}"' _n
file write `fh' "}" _n
file close `fh'

set type float
di as txt "wrote results/95_rdrobust_comb_Stata.json"
