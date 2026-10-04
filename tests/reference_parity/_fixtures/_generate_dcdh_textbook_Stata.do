* Stata references for test_dcdh_textbook_stata_parity.py.
* did_multiplegt_dyn (17jan2026), did_multiplegt_old, twowayfeweights,
* eventstudyinteract 0.1 (all SSC), Stata 18.
* Data: dcdh_textbook_data.csv (_generate_dcdh_textbook_data.py).
* Run from this folder: stata-mp -q -b do _generate_dcdh_textbook_Stata.do
version 18
clear all
set linesize 255
set graphics off
import delimited using "dcdh_textbook_data.csv", clear asdouble

tempname fh
file open `fh' using "dcdh_textbook_Stata.json", write replace
file write `fh' "{" _n
global first 1
global fh `fh'

cap program drop put
program put
    args key value
    if $first == 0 file write $fh "," _n
    global first 0
    if missing(`value') {
        file write $fh `""`key'": null"'
    }
    else {
        file write $fh `""`key'": "' %24.17e (`value')
    }
end

cap program drop put_dyn
program put_dyn
    args tag effects placebos
    forvalues l = 1/`effects' {
        put `tag'_effect_`l' e(Effect_`l')
        put `tag'_se_effect_`l' e(se_effect_`l')
        put `tag'_n_switchers_`l' e(N_switchers_effect_`l')
    }
    forvalues l = 1/`placebos' {
        put `tag'_placebo_`l' e(Placebo_`l')
        put `tag'_se_placebo_`l' e(se_placebo_`l')
        put `tag'_n_switchers_placebo_`l' e(N_switchers_placebo_`l')
    }
    put `tag'_av_tot e(Av_tot_effect)
    put `tag'_se_av_tot e(se_avg_total_effect)
    if `effects' > 1 put `tag'_p_joint_effects e(p_jointeffects)
    if `placebos' > 1 put `tag'_p_joint_placebo e(p_jointplacebo)
end

* ---- did_multiplegt_dyn on a count treatment, periods four years apart
did_multiplegt_dyn y g year d, effects(3) placebo(2) effects_equal(all) graph_off
put_dyn dyn 3 2
put dyn_p_equal e(p_equality_effects)

did_multiplegt_dyn y g year d, effects(3) placebo(2) normalized effects_equal(all) graph_off
put_dyn norm 3 2
put norm_p_equal e(p_equality_effects)

did_multiplegt_dyn y g year d, effects(3) same_switchers graph_off
put_dyn same 3 0

did_multiplegt_dyn y g year d, effects(2) placebo(1) weight(wt) cluster(state) graph_off
put_dyn wcl 2 1

did_multiplegt_dyn y g year d, effects(2) switchers(in) controls(x) graph_off
put_dyn inx 2 0

* by_path leaves the last path in e(); run one path at a time
did_multiplegt_dyn y g year d, effects(2) by_path(1) graph_off
put_dyn path1 2 0
did_multiplegt_dyn y g year d, effects(2) by_path(2) graph_off
put_dyn path2 2 0

* ---- DID_M of the 2020 paper
did_multiplegt_old y g year d, breps(0) placebo(2)
put didm_effect e(effect_0)
put didm_n_switchers e(N_switchers_effect_0)
put didm_placebo_1 e(placebo_1)
put didm_placebo_2 e(placebo_2)

* ---- twowayfeweights
cap program drop put_tw
program put_tw
    args tag
    put `tag'_beta e(beta)
    matrix M = e(M)
    put `tag'_n_plus M[1,1]
    put `tag'_sum_plus M[1,2]
    put `tag'_n_minus M[2,1]
    put `tag'_sum_minus M[2,2]
end
twowayfeweights y g year d, type(feTR) controls(x) test_random_weights(year) weight(wt)
put_tw fe
put fe_sigma e(lb_se_te)
put fe_sigma2 e(lb_se_te2)
matrix R = e(randomweightstest1)
put fe_rw_coef R[1,1]
put fe_rw_se R[1,2]
put fe_rw_corr R[1,4]

twowayfeweights dy g year dd d, type(fdTR) controls(x) test_random_weights(t)
put_tw fd
put fd_sigma e(lb_se_te)
put fd_sigma2 e(lb_se_te2)
matrix R = e(randomweightstest1)
put fd_rw_coef R[1,1]
put fd_rw_se R[1,2]
put fd_rw_corr R[1,4]

twowayfeweights y g year d, type(feTR) other_treatments(d2)
put ot_beta e(beta)
* with other treatments e(M1) is the treatment's own table, e(M2) the first other's
matrix M = e(M1)
put ot_n_plus M[1,1]
put ot_sum_plus M[1,2]
put ot_n_minus M[2,1]
put ot_sum_minus M[2,2]
matrix M2 = e(M2)
put ot_other_n_plus M2[1,1]
put ot_other_sum_plus M2[1,2]
put ot_other_n_minus M2[2,1]
put ot_other_sum_minus M2[2,2]

* ---- regress, vce(hc2, dfadjust)
regress y x d if t == 8, vce(hc2, dfadjust)
matrix T = r(table)
put hc2_b_x T[1,1]
put hc2_se_x T[2,1]
put hc2_p_x T[4,1]
put hc2_lo_x T[5,1]
put hc2_hi_x T[6,1]
put hc2_p_d T[4,2]
test x d
put hc2_F r(F)
put hc2_F_p r(p)
regress y x d, vce(hc2 state, dfadjust)
matrix T = r(table)
put cr2_se_d T[2,2]
put cr2_p_d T[4,2]
put cr2_lo_d T[5,2]
put cr2_hi_d T[6,2]
test x d
put cr2_F r(F)
put cr2_F_p r(p)

* ---- eventstudyinteract on the unbalanced binary panel, ends binned at -3 / +3
gen e = t - cohort if cohort > 0
gen never = cohort == 0
replace cohort = . if cohort == 0
forvalues k = 0/2 {
    gen L`k' = e == `k'
}
gen L3 = e >= 3 & e < .
gen F2 = e == -2
gen F3 = e <= -3
eventstudyinteract y2 L0 L1 L2 L3 F2 F3, absorb(i.g i.t) cohort(cohort) control_cohort(never) vce(cluster g)
matrix b = e(b_iw)
matrix V = e(V_iw)
local names L0 L1 L2 L3 F2 F3
forvalues j = 1/6 {
    local nm : word `j' of `names'
    put sa_`nm' b[1,`j']
    put sa_se_`nm' sqrt(V[`j',`j'])
}
eventstudyinteract y2 L0 L1 L2 L3 F2 F3 [aweight=wt], absorb(i.g i.t) cohort(cohort) control_cohort(never) vce(cluster g)
matrix b = e(b_iw)
matrix V = e(V_iw)
forvalues j = 1/6 {
    local nm : word `j' of `names'
    put saw_`nm' b[1,`j']
    put saw_se_`nm' sqrt(V[`j',`j'])
}

file write `fh' _n "}" _n
file close `fh'
