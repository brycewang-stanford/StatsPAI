*! 97_dcdh_options.do
*! Golden numbers for
*! tests/reference_parity/test_dcdh_option_values_stata_parity.py
*!
*! One did_multiplegt_dyn fit per option value on data_97_dcdh_options.csv
*! (written by _generate_data_97_dcdh_options.py), with the analytic
*! standard errors, the joint covariance e(V) and the Wald p-values, so
*! that every option value sp.did_multiplegt_dyn exposes has its estimate
*! AND its standard error pinned against the authors' command:
*!
*!   switchers(in) / switchers(out) / same_switchers / only_never_switchers
*!   trends_nonparam(reg) / normalized / weight(w) / cluster(cl) / controls(x)
*!   effects(1) / effects_equal(all) / an unbalanced panel / continuous(1|2)
*!   and one fit combining trends_nonparam, normalized, weight, cluster and
*!   switchers(in).
*!
*! It also records bootstrap(200, s) standard errors for five seeds: those
*! are Monte Carlo draws from Stata's stream and serve only as a stochastic
*! screen (S) of se_method='bootstrap', never as parity.
*!
*! Stata's effects(k) is StatsPAI's dynamic=k-1, and e(Av_tot_effect) is
*! StatsPAI's aggregation='switchers' headline.
*!
*! Requires: did_multiplegt_dyn (SSC). Stata 18 MP. Run from this folder:
*!   stata-mp -q -b do 97_dcdh_options.do

version 18
clear all
set more off
set linesize 255
set graphics off

* Double precision throughout (see README.md in this folder).
set type double
import delimited "data_97_dcdh_options.csv", clear asdouble

tempname fh
file open `fh' using "results/97_dcdh_options_Stata.json", write replace
file write `fh' "{" _n
global first 1
global fh `fh'

capture program drop put
program put
    args key value
    if $first == 0 file write $fh "," _n
    global first 0
    if missing(`value') {
        file write $fh `"  "`key'": null"'
    }
    else {
        file write $fh `"  "`key'": "' %24.17e (`value')
    }
end

capture program drop put_dyn
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
    * the joint covariance of the effects and placebos (Av_tot_eff, the
    * last row of e(V), is reported with zero covariances and left out)
    tempname V
    matrix `V' = e(V)
    local names : colnames `V'
    local k = colsof(`V') - 1
    forvalues i = 1/`k' {
        forvalues j = 1/`i' {
            local a : word `i' of `names'
            local b : word `j' of `names'
            put `tag'_V_`a'_`b' `V'[`i',`j']
        }
    }
end

* ---- one option at a time ------------------------------------------------
did_multiplegt_dyn y g t d, effects(3) placebo(2) effects_equal(all) graph_off
put_dyn base 3 2
put base_p_equal e(p_equality_effects)

did_multiplegt_dyn y g t d, effects(3) placebo(2) switchers(in) graph_off
put_dyn in 3 2

did_multiplegt_dyn y g t d, effects(3) placebo(2) switchers(out) graph_off
put_dyn out 3 2

did_multiplegt_dyn y g t d, effects(3) placebo(2) same_switchers graph_off
put_dyn same 3 2

did_multiplegt_dyn y g t d, effects(3) placebo(2) only_never_switchers graph_off
put_dyn never 3 2

did_multiplegt_dyn y g t d, effects(3) placebo(2) trends_nonparam(reg) graph_off
put_dyn tnp 3 2

did_multiplegt_dyn y g t d, effects(3) placebo(2) normalized graph_off
put_dyn norm 3 2

did_multiplegt_dyn y g t d, effects(3) placebo(2) weight(w) graph_off
put_dyn wt 3 2

did_multiplegt_dyn y g t d, effects(3) placebo(2) cluster(cl) graph_off
put_dyn cl 3 2

did_multiplegt_dyn y g t d, effects(3) placebo(2) controls(x) graph_off
put_dyn ctl 3 2

did_multiplegt_dyn y g t d, effects(1) placebo(1) graph_off
put_dyn e1 1 1

did_multiplegt_dyn y g t d, effects(3) placebo(2) trends_nonparam(reg) normalized weight(w) cluster(cl) switchers(in) graph_off
put_dyn combo 3 2

* ---- continuous period-one treatment --------------------------------------
did_multiplegt_dyn yc g t dc, effects(3) placebo(1) continuous(1) graph_off
put_dyn cont1 3 1

did_multiplegt_dyn yc g t dc, effects(3) placebo(1) continuous(2) graph_off
put_dyn cont2 3 1

* ---- bootstrap: a stochastic screen only -----------------------------------
forvalues s = 1/5 {
    did_multiplegt_dyn y g t d, effects(3) placebo(2) bootstrap(200, `s') graph_off
    forvalues l = 1/3 {
        put boot`s'_se_effect_`l' e(se_effect_`l')
    }
    forvalues l = 1/2 {
        put boot`s'_se_placebo_`l' e(se_placebo_`l')
    }
    put boot`s'_se_av_tot e(se_avg_total_effect)
}

* ---- unbalanced panel ------------------------------------------------------
preserve
drop if keep == 0
did_multiplegt_dyn y g t d, effects(3) placebo(2) graph_off
put_dyn unb 3 2
restore

file write `fh' "," _n `"  "_meta": {"cmd": "did_multiplegt_dyn", "stata": "18 MP", "precision": "double", "data": "data_97_dcdh_options.csv"}"' _n
file write `fh' "}" _n
file close `fh'

set type float
di as txt "wrote results/97_dcdh_options_Stata.json"
