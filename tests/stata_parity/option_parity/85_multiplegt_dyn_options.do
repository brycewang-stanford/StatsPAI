*! 85_multiplegt_dyn_options.do
*! Golden numbers for
*! tests/reference_parity/test_multiplegt_dyn_options_parity.py
*!
*! Pins the sample-restriction options of did_multiplegt_dyn against
*! StatsPAI's sp.did_multiplegt_dyn:
*!
*!   switchers(in)   -> switchers='in'
*!   switchers(out)  -> switchers='out'
*!   same_switchers  -> same_switchers=True
*!
*! Stata's effects(4) is StatsPAI's dynamic=3 (Stata indexes effects from
*! ℓ=1, StatsPAI from h=0), and e(Av_tot_effect) is StatsPAI's
*! aggregation='switchers' headline, not the 'simple' default.
*!
*! The fixture panel deliberately contains BOTH switch-on and switch-off
*! events plus never-switchers at both baseline levels — an absorbing
*! panel cannot tell these options apart, and cannot expose the bootstrap
*! switch-date defect this fixture caught (see the test module).
*!
*! Requires: did_multiplegt_dyn. Stata 18 MP.

version 17
clear all
set more off

import delimited "data_85_dcdh_switch.csv", clear

capture program drop _dumpdyn
program define _dumpdyn
    args fh label comma
    * NOTE: the e() scalars are Effect_1 / Placebo_1 with a CAPITAL letter.
    * Lower-case names silently return missing rather than erroring.
    file write `fh' `"  ""' "`label'" `"": {"' _n
    file write `fh' `"    "Av_tot_eff": "' %20.15f (e(Av_tot_effect)) "," _n
    forvalues k = 1/4 {
        file write `fh' `"    "Effect_`k'": "' %20.15f (e(Effect_`k')) "," _n
    }
    forvalues k = 1/2 {
        file write `fh' `"    "Placebo_`k'": "' %20.15f (e(Placebo_`k')) "," _n
    }
    file write `fh' `"    "N_switchers_effect_1": "' %14.0f (e(N_switchers_effect_1)) _n
    file write `fh' "  }`comma'" _n
end

tempname fh
file open `fh' using "results/85_multiplegt_dyn_options_Stata.json", write replace
file write `fh' "{" _n

qui did_multiplegt_dyn y i t d, effects(4) placebo(2) graph_off
_dumpdyn `fh' "pooled" ","

qui did_multiplegt_dyn y i t d, effects(4) placebo(2) switchers(in) graph_off
_dumpdyn `fh' "switchers_in" ","

qui did_multiplegt_dyn y i t d, effects(4) placebo(2) switchers(out) graph_off
_dumpdyn `fh' "switchers_out" ","

qui did_multiplegt_dyn y i t d, effects(4) placebo(2) same_switchers graph_off
_dumpdyn `fh' "same_switchers" ","

file write `fh' `"  "_meta": {"cmd": "did_multiplegt_dyn", "stata": "18 MP"}"' _n
file write `fh' "}" _n
file close `fh'

di as txt "wrote results/85_multiplegt_dyn_options_Stata.json"
