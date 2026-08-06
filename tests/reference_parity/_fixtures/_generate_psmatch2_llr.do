* ---------------------------------------------------------------------------
* Generate the Stata reference fixture for local linear regression matching
* and the Mahalanobis metric, for
* tests/reference_parity/test_psmatch2_llr_parity.py
*
* Requires: Stata 18 + psmatch2 (ssc install psmatch2).
* Run:      stata -b do _generate_psmatch2_llr.do   (from this directory)
*
* Two behaviours of psmatch2 that this fixture exists to pin
* ---------------------------------------------------------
* 1. psmatch2 reports `seatt = .` for genuine LLR.  Local linear weights can
*    be negative, and the analytic formula sqrt(var1/N1 + var0*sum(w^2)/N1^2)
*    assumes they are not.  Only the point estimate is alignable.
*
* 2. `psmatch2 ..., llr` with the DEFAULT kernel (epan) does not perform local
*    linear regression matching at all.  psmatch2.ado rewrites the request as
*    nearest-neighbour matching on an lpoly-smoothed outcome:
*
*        // do nearest neighbor if llr with tricube
*        if ("`method'"=="llr" & "`kerneltype'"=="epan" & "`metric'"=="pscore") {
*            local method "neighbor"
*            ... lpoly `v' _pscore ..., deg(1) at(_pscore) gen(_s_`v')
*        }
*
*    so it reports a non-missing SE and a materially different ATT.  Only a
*    non-Epanechnikov kernel reaches psmatch2's own `_Match_llr` routine.
*
* Produces
* --------
*   psmatch2_llr_data.csv   id x1 x2 d y _pscore _treated _support _weight _y
*                           (the _weight/_y columns are from llr + tricube)
*   scalars recorded in psmatch2_llr_stata.json
* ---------------------------------------------------------------------------
clear all
set seed 4242
set obs 200

gen id = _n
gen x1 = rnormal()
gen x2 = rnormal()
gen d  = rbinomial(1, invlogit(0.9*x1 - 0.5*x2))
gen y  = 1 + 0.8*x1 - 0.4*x2 + rnormal()

* --- genuine LLR: every non-epan kernel goes through _Match_llr ------------
foreach kt in tricube biweight normal uniform {
    preserve
    quietly psmatch2 d x1 x2, outcome(y) llr kerneltype(`kt') bwidth(0.5) logit
    di "LLR_`kt' att = " %21.16e r(att) "   seatt = " r(seatt)
    restore
}

* --- llr + epan: the silent reroute to lpoly + nearest neighbour ----------
preserve
quietly psmatch2 d x1 x2, outcome(y) llr kerneltype(epan) bwidth(0.5) logit
di "LLR_EPAN att = " %21.16e r(att) "   seatt = " r(seatt)
restore

* --- plain kernel matching with the same kernel, for contrast ------------
preserve
quietly psmatch2 d x1 x2, outcome(y) kernel kerneltype(tricube) bwidth(0.5) logit
di "KERNEL_TRICUBE att = " %21.16e r(att) "  seatt = " %21.16e r(seatt)
restore

* --- Mahalanobis metric ---------------------------------------------------
preserve
quietly psmatch2 d, outcome(y) mahalanobis(x1 x2) neighbor(1)
di "MAHAL att = " %21.16e r(att) "  seatt = " %21.16e r(seatt)
restore

* --- export the data plus the tricube LLR frame for row-level comparison --
quietly psmatch2 d x1 x2, outcome(y) llr kerneltype(tricube) bwidth(0.5) logit
format x1 x2 y _pscore _weight _y %21.16e
export delimited id x1 x2 d y _pscore _treated _support _weight _y ///
    using "psmatch2_llr_data.csv", replace datafmt

* --- llr + epan: export the rerouted frame for the compat parity test ------
* _s_y is the lpoly-smoothed outcome; _y is the matched control's _s_y (NOT
* its raw y).  Verified in-session: max|_y - _s_y[match]| = 0 exactly while
* max|_y - y[match]| = 1.99.
preserve
quietly psmatch2 d x1 x2, outcome(y) llr kerneltype(epan) bwidth(0.5) logit
di "LLR_EPAN att = " %21.16e r(att) "  seatt = " %21.16e r(seatt)
format x1 x2 y _pscore _s_y _weight _y %21.16e
export delimited id x1 x2 d y _pscore _treated _support _s_y _weight _y ///
    using "psmatch2_llr_epan_data.csv", replace datafmt
restore
