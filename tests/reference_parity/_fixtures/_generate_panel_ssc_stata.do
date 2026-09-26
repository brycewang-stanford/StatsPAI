* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_panel_ssc_stata_parity.py
*
* Requires: Stata 18 (official xtreg / areg / regress only).
* Run:      stata -b do _generate_panel_ssc_stata.do   (from this directory)
*
* What this fixture pins
* ----------------------
* sp.panel(..., ssc="stata") standard errors, t / z convention and degrees
* of freedom against the Stata command each panel method corresponds to:
*
*   fe      xtreg y x1 x2, fe [vce()]       (areg, absorb(id) when the
*                                            cluster does not nest the unit)
*   twoway  xtreg y x1 x2 i.t, fe [vce()]
*   pooled  regress y x1 x2 [, vce()]
*   fd      regress D.y D.x1 D.x2, nocons [vce()]
*   re      xtreg y x1 x2, re [vce()]
*   be      xtreg y x1 x2, be
*
* The panel is unbalanced (units observed for 4 to 8 consecutive periods),
* units are nested in 15 states, and a state x period shock makes the
* state-clustered variance differ materially from the unit-clustered one.
*
* Data is generated and exported HERE so both sides read the same bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20260927
set obs 90

gen long   id  = _n
gen int    st  = mod(_n, 15) + 1
gen double a_i = rnormal()
gen int    Ti  = 4 + mod(_n, 5)
expand 8
bysort id: gen int t = _n
drop if t > Ti
gen double x1 = 0.6*a_i + rnormal()
gen double x2 = rnormal()
* state x period shock, shared by every unit of a state in a period
gen double u_st = .
set seed 424242
forvalues s = 1/15 {
    forvalues p = 1/8 {
        quietly replace u_st = rnormal() if st == `s' & t == `p'
    }
}
gen double y = 1 + 0.5*x1 - 0.3*x2 + a_i + 0.8*u_st ///
    + (0.5 + 0.5*abs(x2))*rnormal()
drop a_i u_st Ti
* analytic weight, constant within unit (xtreg requires that); exact in
* binary so both sides read the same value
gen double w = 0.5 + mod(id*7, 11)/4

format x1 x2 y %21.16e
export delimited id st t x1 x2 y w using "panel_ssc_data.csv", replace datafmt

xtset id t

capture program drop wr
program define wr
    * wr <name> <last:0|1> <terms...>
    args name last
    macro shift 2
    local terms `*'
    file write fh `"  "`name'": {"' _n
    foreach v of local terms {
        file write fh `"    "b_`v'": "' %21.16e (_b[`v']) `","' _n
        file write fh `"    "se_`v'": "' %21.16e (_se[`v']) `","' _n
    }
    local dfr = cond(missing(e(df_r)), -1, e(df_r))
    local ncl = cond(missing(e(N_clust)), -1, e(N_clust))
    file write fh `"    "df_r": "' %21.16e (`dfr') `","' _n
    file write fh `"    "N_clust": "' %21.16e (`ncl') `","' _n
    file write fh `"    "N": "' %21.16e (e(N)) _n
    file write fh `"  },"' _n
end

file open fh using "panel_ssc_stata.json", write replace text
file write fh "{" _n

* ---- fe ------------------------------------------------------------------
quietly xtreg y x1 x2, fe
wr fe_unadjusted 0 x1 x2
quietly xtreg y x1 x2, fe vce(robust)
wr fe_robust 0 x1 x2
quietly xtreg y x1 x2, fe vce(cluster id)
wr fe_cluster_id 0 x1 x2
quietly xtreg y x1 x2, fe vce(cluster st)
wr fe_cluster_st 0 x1 x2
* xtreg refuses clusters that do not nest the panels; areg counts the
* absorbed unit effects in the small-sample factor instead.
quietly areg y x1 x2, absorb(id) vce(cluster t)
wr fe_cluster_t 0 x1 x2

* ---- twoway --------------------------------------------------------------
quietly xtreg y x1 x2 i.t, fe
wr twoway_unadjusted 0 x1 x2
quietly xtreg y x1 x2 i.t, fe vce(robust)
wr twoway_robust 0 x1 x2
quietly xtreg y x1 x2 i.t, fe vce(cluster id)
wr twoway_cluster_id 0 x1 x2
quietly xtreg y x1 x2 i.t, fe vce(cluster st)
wr twoway_cluster_st 0 x1 x2

* ---- pooled --------------------------------------------------------------
quietly regress y x1 x2
wr pooled_unadjusted 0 x1 x2 _cons
quietly regress y x1 x2, vce(robust)
wr pooled_robust 0 x1 x2 _cons
quietly regress y x1 x2, vce(cluster id)
wr pooled_cluster_id 0 x1 x2 _cons
quietly regress y x1 x2, vce(cluster st)
wr pooled_cluster_st 0 x1 x2 _cons
quietly regress y x1 x2, vce(cluster t)
wr pooled_cluster_t 0 x1 x2 _cons

* ---- first differences ---------------------------------------------------
quietly regress D.y D.x1 D.x2, nocons
wr fd_unadjusted 0 D.x1 D.x2
quietly regress D.y D.x1 D.x2, nocons vce(robust)
wr fd_robust 0 D.x1 D.x2
quietly regress D.y D.x1 D.x2, nocons vce(cluster id)
wr fd_cluster_id 0 D.x1 D.x2
quietly regress D.y D.x1 D.x2, nocons vce(cluster st)
wr fd_cluster_st 0 D.x1 D.x2

* ---- random effects ------------------------------------------------------
quietly xtreg y x1 x2, re
wr re_unadjusted 0 x1 x2 _cons
quietly xtreg y x1 x2, re vce(robust)
wr re_robust 0 x1 x2 _cons
quietly xtreg y x1 x2, re vce(cluster id)
wr re_cluster_id 0 x1 x2 _cons
quietly xtreg y x1 x2, re vce(cluster st)
wr re_cluster_st 0 x1 x2 _cons

* ---- analytic weights ([aw=], constant within unit) -------------------
quietly xtreg y x1 x2 [aw=w], fe
wr w_fe_unadjusted 0 x1 x2
quietly xtreg y x1 x2 [aw=w], fe vce(cluster id)
wr w_fe_cluster_id 0 x1 x2
quietly xtreg y x1 x2 [aw=w], fe vce(cluster st)
wr w_fe_cluster_st 0 x1 x2
quietly xtreg y x1 x2 i.t [aw=w], fe vce(cluster st)
wr w_twoway_cluster_st 0 x1 x2
quietly xtreg y x1 x2 i.t [aw=w], fe
wr w_twoway_unadjusted 0 x1 x2
quietly regress y x1 x2 [aw=w]
wr w_pooled_unadjusted 0 x1 x2 _cons
quietly regress y x1 x2 [aw=w], vce(robust)
wr w_pooled_robust 0 x1 x2 _cons
quietly regress y x1 x2 [aw=w], vce(cluster st)
wr w_pooled_cluster_st 0 x1 x2 _cons

* ---- clusters that do not nest the unit ---------------------------------
* xtreg refuses these; areg counts every absorbed level and every dummy.
quietly areg y x1 x2 i.t, absorb(id) vce(cluster t)
wr twoway_cluster_t 0 x1 x2
quietly regress D.y D.x1 D.x2, nocons vce(cluster t)
wr fd_cluster_t 0 D.x1 D.x2

* ---- Mundlak correlated random effects ----------------------------------
* sp.panel(method='mundlak') is xtreg, re with the unit means of x added.
bysort id: egen double mean_x1 = mean(x1)
bysort id: egen double mean_x2 = mean(x2)
quietly xtreg y x1 x2 mean_x1 mean_x2, re vce(cluster st)
wr mundlak_cluster_st 0 x1 x2 mean_x1 mean_x2 _cons
quietly xtreg y x1 x2 mean_x1 mean_x2, re
wr mundlak_unadjusted 0 x1 x2 mean_x1 mean_x2 _cons
* Chamberlain: the means plus each period's deviation from them (t >= 2).
local cham
foreach v in x1 x2 {
    forvalues p = 2/8 {
        gen double cham_`v'_t`p' = cond(t == `p', `v' - mean_`v', 0)
        local cham `cham' cham_`v'_t`p'
    }
}
quietly xtreg y x1 x2 mean_x1 mean_x2 `cham', re vce(cluster st)
wr chamberlain_cluster_st 0 x1 x2 mean_x1 mean_x2 `cham' _cons
drop mean_x1 mean_x2 cham_*

* ---- between -------------------------------------------------------------
quietly xtreg y x1 x2, be
wr be_unadjusted 0 x1 x2 _cons

file write fh `"  "_meta": {"' _n
file write fh `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write fh `"    "flavor": "' `"""' "`c(flavor)' `c(edition_real)'" `"""' `","' _n
file write fh `"    "seed": 20260927,"' _n
file write fh `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write fh `"  }"' _n
file write fh "}" _n
file close fh
display "wrote panel_ssc_stata.json and panel_ssc_data.csv"
