* ---------------------------------------------------------------------------
* Generate the Stata psmatch2 reference fixture for
* tests/reference_parity/test_psmatch2_parity.py
*
* Requires: Stata 18 + psmatch2 (ssc install psmatch2).
* Run:      stata -b do _generate_psmatch2.do   (from this directory)
*
* Produces: psmatch2_data.csv  — base columns (row, x1, x2, d, y) plus the
*           psmatch2 variables (_id _pscore _treated _support _weight _n1 _nn
*           _pdif _y) for neighbor(1), logit, with replacement (the default).
*           The scalar ATT / SE are recorded in psmatch2_stata.json.
* ---------------------------------------------------------------------------
clear all
set seed 12345
set obs 40
gen x1 = rnormal()
gen x2 = rnormal()
gen ps_true = invlogit(0.8*x1 + 0.5*x2 - 0.3)
gen d = rbinomial(1, ps_true)
gen y = 1 + 2*d + 3*x1 + x2 + rnormal()
gen row = _n

* --- nearest-neighbour (digit-exact ATT + analytic SE) ---
preserve
quietly psmatch2 d x1 x2, outcome(y) neighbor(1) logit
di "NN  att = " %21.16e r(att) "  seatt = " %21.16e r(seatt)
format x1 x2 y _pscore _weight _pdif _y %21.16e
export delimited row _id x1 x2 d y _pscore _treated _support _weight ///
    _n1 _nn _pdif _y using "psmatch2_data.csv", replace datafmt
restore

* --- kernel (Epanechnikov, bwidth 0.5) ---
preserve
quietly psmatch2 d x1 x2, outcome(y) kernel kerneltype(epan) bwidth(0.5) logit
di "KERNEL att = " %21.16e r(att) "  seatt = " %21.16e r(seatt)
format x1 x2 y _pscore _weight _y %21.16e
export delimited row x1 x2 d y _pscore _treated _support _weight _y ///
    using "psmatch2_kernel_data.csv", replace datafmt
restore

* --- radius (= uniform kernel, bandwidth = caliper 0.1) ---
preserve
quietly psmatch2 d x1 x2, outcome(y) radius caliper(0.1) logit
di "RADIUS att = " %21.16e r(att) "  seatt = " %21.16e r(seatt)
format x1 x2 y _pscore _weight _y %21.16e
export delimited row x1 x2 d y _pscore _treated _support _weight _y ///
    using "psmatch2_radius_data.csv", replace datafmt
restore

* --- Abadie-Imbens (2006) robust SE: neighbor(1), ai(1) ---
* _self_y is the within-arm (T-T & C-C) matched outcome the AI variance uses.
preserve
quietly psmatch2 d x1 x2, outcome(y) neighbor(1) ai(1) logit
di "AI(1) att = " %21.16e r(att) "  seatt = " %21.16e r(seatt)
quietly psmatch2 d x1 x2, outcome(y) neighbor(1) ai(2) logit
di "AI(2) seatt = " %21.16e r(seatt)
quietly psmatch2 d x1 x2, outcome(y) neighbor(1) ai(1) logit
format y _pscore _weight _self_y %21.16e
export delimited row x1 x2 d y _pscore _treated _support _weight _self_y ///
    using "psmatch2_ai_data.csv", replace datafmt
restore

* The scalar ATT / SE values are recorded by method in psmatch2_stata.json.
