* tests/stata_parity/87_interflex.do
*
* Module 87: interflex (Hainmueller, Mummolo and Xu 2019) marginal
* effects of a binary treatment across a moderator.
*   StatsPAI:  sp.interflex(..., estimator = "linear" / "binning" / "kernel")
*   R:         interflex::interflex(estimator = ..., vartype = "delta",
*                                   vcov.type = "robust", neval = 5)
*   Stata:     interflex Y D X Z1, type(linear|binning|kernel) vce(robust)
*                  neval(5) cutoffs(0.3 1.7) bw(1)        (SSC interflex)
*
* Rows: linear_me_<k> (marginal effect and robust delta-method SE at the
* five equally spaced evaluation points from min(X) to max(X)),
* binning_x0_<j> / binning_me_<j> (bin median and bin-specific treatment
* effect with SE, three bins cut at the explicit cutoffs 0.3 and 1.7 shared by all sides), p_wald_stata (Stata's r(pwald): the
* linear restriction against the binning model without covariate-by-bin
* interactions, F reference), kernel_fixed_me_<k> (Stata's
* kernel estimator: fixed Gaussian kernel normalden((X - x)/bw) with bw(1),
* no density adaptation, matched by sp.interflex(adaptive=False)). Stata interflex does not report the LR test, the
* L-kurtosis, or the average treatment effect, so those rows are
* R/Python-only.

version 18
clear all

do _common.do
stata_parity_init, module(87_interflex)
stata_parity_open, module(87_interflex)

capture which interflex
if _rc != 0 {
    ssc install interflex, replace
}

import delimited "${STATA_PARITY_DATA}/87_interflex.csv", clear case(preserve) asdouble
local n = _N

* ---- linear ---------------------------------------------------------------
interflex Y D X Z1, type(linear) vce(robust) neval(5)
matrix M = r(margeff)
forvalues k = 1/5 {
    local est = M[`k', 2]
    local se = M[`k', 3]
    stata_parity_row, statname(linear_me_`k') estimate(`est') stderr(`se') nobs(`n')
}

* ---- binning --------------------------------------------------------------
interflex Y D X Z1, type(binning) vce(robust) cutoffs(0.3 1.7)
matrix B = r(estBin)
local pwald = r(pwald)
* bin counts: R's cut() with the minimum in bin 1 (explicit breaks)
local c1 = 0.3
local c2 = 1.7
count if X <= `c1'
local n1 = r(N)
count if X > `c1' & X <= `c2'
local n2 = r(N)
count if X > `c2'
local n3 = r(N)
forvalues j = 1/3 {
    local x0 = B[`j', 1]
    local est = B[`j', 2]
    local se = B[`j', 3]
    stata_parity_row, statname(binning_x0_`j') estimate(`x0') nobs(`n`j'')
    stata_parity_row, statname(binning_me_`j') estimate(`est') stderr(`se') nobs(`n`j'')
}
* r(pwald): full model without covariate-by-bin interactions, F reference
* (sp.interflex(wald_full_moderate=False, wald_test="F")).
stata_parity_row, statname(p_wald_stata) estimate(`pwald') nobs(`n')

* ---- kernel ---------------------------------------------------------------
interflex Y D X Z1, type(kernel) bw(1) neval(5)
matrix K = r(margeff)
forvalues k = 1/5 {
    local est = K[`k', 2]
    stata_parity_row, statname(kernel_fixed_me_`k') estimate(`est') nobs(`n')
}

stata_parity_extra, key(method) val("SSC interflex, vce(robust), neval(5), cutoffs(0.3 1.7), bw(1) fixed Gaussian kernel")
stata_parity_extra, key(stata_command) val("interflex Y D X Z1, type(linear|binning|kernel) vce(robust) neval(5) cutoffs(0.3 1.7) bw(1)")
stata_parity_close, module(87_interflex)
