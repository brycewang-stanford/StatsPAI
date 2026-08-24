* tests/stata_parity/85_twfe_event_study.do
*
* Module 85: dynamic TWFE event study -- the benchmark specification
* that every "TWFE-comparable" claim in the modern DiD literature is
* stated against.
*   StatsPAI:  sp.event_study(..., ref_period=-1)
*   R:         fixest::feols(y ~ i(rel_f, treat, ref=-1) | unit + time,
*                            cluster = ~unit, ssc = ssc(fixef.K = "none"))
*   Stata:     reghdfe y <relative-time dummies>, absorb(unit time)
*                                                 vce(cluster unit)
*
* Estimates: pinned at rel < 1e-9 on all three sides.
*
* Standard errors: py<->R agree to 9e-14 (fixest needs fixef.K="none",
* see the R script). Stata's SE differs from both by a fixed
* small-sample degrees-of-freedom factor, and the factor is exactly
* reconstructable rather than approximate:
*
*     var_Stata / var_py = (N - K_py) / (N - K_Stata)
*                        = (1620 - 8) / (1620 - 17)
*                        = 1.005614472863
*
* against a measured ratio of 1.005614472866. reghdfe's K counts the 8
* event-time coefficients plus the 8 non-redundant time effects plus the
* constant; sp.event_study counts only the 8 non-absorbed coefficients.
* fixest exposes this choice as ssc(fixef.K=), reghdfe does not: dof(none)
* moves in the other direction (it stops *removing* the unit effects that
* are redundant under clustering, giving df_a=188 and a larger SE still),
* and no dof() setting reaches K=8. The gap is therefore registered in
* compare.STATA_HEADLINE_GAP_EXCEPTIONS with this derivation rather than
* absorbed into a tolerance, and the .do file below runs reghdfe with its
* own default cluster d.f. handling -- no options chosen to manufacture
* agreement.
*
* Fixture: non-staggered (cohorts {5, never}), 180 units x 9 periods,
* window (-4, +4), reference period -1.

version 18
clear all

do _common.do
stata_parity_init, module(85_twfe_event_study)
stata_parity_open, module(85_twfe_event_study)

import delimited "${STATA_PARITY_DATA}/85_twfe_event_study.csv", clear case(preserve)

* Shared CSV codes never-treated as g == 0.
gen rel   = cond(g > 0, time - g, .)
gen treat = (g > 0)

* Explicit relative-time dummies rather than factor notation: with a
* never-treated group `rel` is missing for those units, and i.rel would
* drop them from the estimation sample entirely (the same trap the R
* side hit -- fixest reported "810 observations removed because of NA
* values"). Interacting an explicit indicator with `treat` keeps the
* never-treated units in as pure controls, which is what all three
* sides are estimating.
local esvars
forvalues k = -4/4 {
    if `k' != -1 {
        local nm = cond(`k' < 0, "m" + string(abs(`k')), "p" + string(`k'))
        gen d_`nm' = (treat == 1 & rel == `k')
        local esvars `esvars' d_`nm'
    }
}

reghdfe y `esvars', absorb(unit time) vce(cluster unit)

local n = e(N)
matrix B = e(b)
matrix V = e(V)

forvalues k = -4/4 {
    if `k' != -1 {
        local nm  = cond(`k' < 0, "m" + string(abs(`k')), "p" + string(`k'))
        local lbl = cond(`k' < 0, "es_-" + string(abs(`k')), "es_+" + string(`k'))
        local bv = B[1, "d_`nm'"]
        local sv = sqrt(V["d_`nm'", "d_`nm'"])
        stata_parity_row, statname(`lbl') estimate(`bv') stderr(`sv') nobs(`n')
    }
}

* Record the degrees-of-freedom accounting so a reviewer can verify the
* documented SE gap from the artifact alone, without rerunning Stata.
local K_stata = e(df_m) + e(df_a) + 1
local K_py    = e(df_m)
stata_parity_extra_num, key(stata_df_m) val(`=e(df_m)')
stata_parity_extra_num, key(stata_df_a) val(`=e(df_a)')
stata_parity_extra_num, key(stata_K) val(`K_stata')
stata_parity_extra_num, key(statspai_K) val(`K_py')
stata_parity_extra_num, key(implied_var_ratio) val(`=(`n' - `K_py') / (`n' - `K_stata')')
stata_parity_extra, key(method) val(reghdfe)
stata_parity_extra, key(se_convention) val("reghdfe counts absorbed time effects and the constant in the small-sample K; sp.event_study and fixest ssc(fixef.K=none) count only non-absorbed coefficients")
stata_parity_extra, key(stata_command) val("reghdfe y d_m4 d_m3 d_m2 d_p0 d_p1 d_p2 d_p3 d_p4, absorb(unit time) vce(cluster unit)")

stata_parity_close, module(85_twfe_event_study)
