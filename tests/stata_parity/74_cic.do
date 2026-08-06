* tests/stata_parity/74_cic.do
*
* Module 74: Athey--Imbens (2006) Changes-in-Changes.
*   StatsPAI:  sp.cic
*   R:         qte::CiC (2.0.0)
*   Stata:     cic (Kranker's port of the Athey--Imbens Matlab)  <-- this file
*
* Estimator choice. `cic` implements four estimators; the one that is
* like-for-like with qte::CiC is `discrete_ci` -- the discrete-support CIC
* estimator under conditional independence. `continuous` is the
* continuous-outcome formula and lands ~1% away on every row, so choosing it
* would grade a different estimand. This was measured, not assumed:
*
*     estimator      ATT        q10        q50
*     ------------------------------------------------
*     py / qte::CiC  1.9093840  2.0151497  1.6790716
*     discrete_ci    1.9116044  2.0151497  1.6892037
*     continuous     1.8891203  2.0044213  1.6856684
*
* Result: eight of the nine quantile treatment effects (q10--q40, q60--q90)
* agree with qte::CiC at machine precision under `discrete_ci`. Two rows do
* not, and they are registered in compare.py::STATA_HEADLINE_GAP_EXCEPTIONS
* rather than tolerated silently:
*
*   qte_50  -- rel 6.0e-3. The median is the one probability where the two
*              implementations' inverse-CDF tie-break differs; every other
*              decile is bit-identical, which localises the disagreement to
*              the tie rule rather than to the estimator.
*   cic_ATT -- rel 1.2e-3. The ATT integrates the estimated counterfactual
*              distribution, so it inherits the same tie rule at every
*              crossing rather than at one point.
*
* Standard errors are not compared: the R side runs se = FALSE, this runs
* vce(none), and StatsPAI's come from its own bootstrap.
*
* Tolerance: rel < 1e-6 on the eight joined quantile rows.

version 18
clear all

do _common.do
stata_parity_init, module(74_cic)
stata_parity_open, module(74_cic)

import delimited "${STATA_PARITY_DATA}/74_cic.csv", clear case(preserve)
count
local n = r(N)

* qte::CiC takes t / tmin1 = 2 / 1 on the raw coding; cic takes a 0/1 post
* indicator. Same four (group x time) cells either way.
gen byte post = (t == 2)

cic all y treat post, at(10(10)90) vce(none)

matrix B = e(b)

stata_parity_row, stat(cic_ATT) est(`=B[1, "discrete_ci:mean"]') nob(`n')
forvalues q = 10(10)90 {
    stata_parity_row, stat("qte_`q'") est(`=B[1, "discrete_ci:q`q'"]') nob(`n')
}

* Record the continuous-estimator column too, so the estimator choice above
* is auditable from the committed artifact rather than only from this
* comment. These names do not exist on the Python side, so compare.py never
* joins them.
stata_parity_extra_num, key(continuous_mean) val(`=B[1, "continuous:mean"]')
stata_parity_extra_num, key(continuous_q50)  val(`=B[1, "continuous:q50"]')

stata_parity_extra, key(stata_command) val("cic all y treat post, at(10(10)90) vce(none)")
stata_parity_extra, key(estimator) val("discrete_ci -- the discrete-support CIC under conditional independence, the like-for-like counterpart of qte::CiC")
stata_parity_extra, key(headline_gap) val("qte_50 rel 6.0e-3 and cic_ATT rel 1.2e-3 are inverse-CDF tie-break differences; the other eight deciles are bit-identical")
stata_parity_extra, key(se_convention) val("vce(none) here and se=FALSE in R; StatsPAI's SEs are bootstrap, so no SE row is joined")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(74_cic)
