* ---------------------------------------------------------------------------
* Generate the Stata pstest reference fixture for
* tests/reference_parity/test_pstest_parity.py
*
* Requires: Stata 18 + psmatch2 + pstest (ssc install psmatch2 installs both).
* Run:      stata -b do _generate_pstest.do   (from this directory)
*
* Conventions this fixture exists to pin
* --------------------------------------
* 1. The post-matching standardised bias keeps the UNMATCHED pooled SD:
*        bias_before = 100 (m1u - m0u) / sqrt((v1u + v0u)/2)
*        bias_after  = 100 (m1m - m0m) / sqrt((v1u + v0u)/2)
*    (pstest.ado lines 282 and 286 -- note both use v1u/v0u.)
*
* 2. Matched moments use importance weights, so `summarize x [iw=w]` divides
*    the variance by sum(w) - 1 rather than n - 1.
*
* 3. The Ps R2 / LR chi2 / Rubin B / Rubin R block comes from pstest's OWN
*    probit, refit on the matched sample with [iw=_weight] -- not from the
*    logit propensity score psmatch2 estimated:
*        qui probit `treated' `varlist' if `touse'
*        qui predict double `index0' if e(sample), xb
*        qui probit `treated' `varlist' [iw=`mweight'] if `support'==1 ...
*    Reusing _pscore reproduces the per-covariate rows exactly while getting
*    Rubin's B wrong by ~5%.
*
* Produces
* --------
*   pstest_data.csv   id x1 x2 d y _pscore _treated _support _weight
*   scalars recorded in pstest_stata.json
* ---------------------------------------------------------------------------
clear all
set seed 4242
set obs 200

gen id = _n
gen x1 = rnormal()
gen x2 = rnormal()
gen d  = rbinomial(1, invlogit(0.9*x1 - 0.5*x2))
gen y  = 1 + 0.8*x1 - 0.4*x2 + rnormal()

quietly psmatch2 d x1 x2, outcome(y) neighbor(1) logit

* --- the printed table, plus the returned summary scalars ----------------
pstest x1 x2, both
return list

* --- per-covariate rows at full precision --------------------------------
* pstest prints only 1 decimal, so recompute its own arithmetic here.
foreach v of varlist x1 x2 {
    qui sum `v' if _treated==1
    local m1u = r(mean)
    local v1u = r(Var)
    qui sum `v' if _treated==0
    local m0u = r(mean)
    local v0u = r(Var)
    qui sum `v' [iw=_weight] if _treated==1 & _support==1
    local m1m = r(mean)
    local v1m = r(Var)
    qui sum `v' [iw=_weight] if _treated==0 & _support==1
    local m0m = r(mean)
    local v0m = r(Var)

    local bias  = 100*(`m1u' - `m0u')/sqrt((`v1u' + `v0u')/2)
    local biasm = 100*(`m1m' - `m0m')/sqrt((`v1u' + `v0u')/2)
    local reduc = -100*(abs(`biasm') - abs(`bias'))/abs(`bias')

    di "`v'  bias_u=" %21.16e `bias' "  bias_m=" %21.16e `biasm'
    di "`v'  reduct=" %21.16e `reduc' ///
       "  vratio_u=" %21.16e `v1u'/`v0u' "  vratio_m=" %21.16e `v1m'/`v0m'
    di "`v'  m1u=" %21.16e `m1u' "  m0u=" %21.16e `m0u' ///
       "  m1m=" %21.16e `m1m' "  m0m=" %21.16e `m0m'
}

* --- export the frame so Python reads Stata's own weights ----------------
format x1 x2 y _pscore _weight %21.16e
export delimited id x1 x2 d y _pscore _treated _support _weight ///
    using "pstest_data.csv", replace datafmt
