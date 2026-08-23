* tests/stata_parity/84_bjs_pretrends.do
*
* Module 84: BJS pre-treatment lead vector.
*   StatsPAI:  sp.did_imputation(..., pretrend_method="bjs")
*   R:         didimputation::did_imputation  (horizons only -- see 84_bjs_pretrends.R)
*   Stata:     did_imputation (Borusyak, SSC S458957)  <-- this file
*
* Module 16 pins the pooled ATT. This one pins the LEAD VECTOR, which is
* where the v1.23.0 correctness fix lived: StatsPAI previously reported
* the fect/did2s in-sample residual means while documenting this command,
* and the two differ by the untreated unit share.
*
* pretrends(3) estimates exactly three lead indicators and pools every
* earlier relative time into the omitted category. The fixture gives each
* treated cohort strictly more pre-periods than that, so `autosample` has
* nothing to drop and all three sides estimate on identical rows.
*
* Tolerance: rel_est < 1e-6. See compare.py::TOLERANCES for why the
* post-treatment standard errors are not inside the budget.

version 18
clear all

do _common.do
stata_parity_init, module(84_bjs_pretrends)
stata_parity_open, module(84_bjs_pretrends)

import delimited "${STATA_PARITY_DATA}/84_bjs_pretrends.csv", clear case(preserve)

* The shared CSV encodes never-treated as g == 0; did_imputation wants
* missing, otherwise those units read as treated from period 0.
replace g = . if g == 0

did_imputation y unit time g, pretrends(3) horizons(0/3) cluster(unit)

local n = e(N)
matrix B = e(b)
matrix V = e(V)

foreach k in 1 2 3 {
    local bv = B[1, "pre`k'"]
    local sv = sqrt(V["pre`k'", "pre`k'"])
    stata_parity_row, stat(pre`k'_att) est(`bv') stderr(`sv') nob(`n')
}

foreach h in 0 1 2 3 {
    local bv = B[1, "tau`h'"]
    local sv = sqrt(V["tau`h'", "tau`h'"])
    stata_parity_row, stat(tau`h'_att) est(`bv') stderr(`sv') nob(`n')
}

stata_parity_extra, key(method) val(did_imputation)
stata_parity_extra, key(never_treated_coding) val("replace g = . if g == 0")
stata_parity_extra, key(pretrend_convention) val("pretrends(k): k lead dummies, earlier relative times pooled into the omitted category")
stata_parity_extra, key(stata_command) val("did_imputation y unit time g, pretrends(3) horizons(0/3) cluster(unit)")

stata_parity_close, module(84_bjs_pretrends)
