* tests/stata_parity/78_multiplegt_dyn.do
*
* Module 78: de Chaisemartin--D'Haultfoeuille intertemporal event study.
*   StatsPAI:  sp.did_multiplegt_dyn
*   R:         DIDmultiplegtDYN::did_multiplegt_dyn (2.3.4)
*   Stata:     did_multiplegt_dyn                    <-- this file
*
* The Stata command is the authors' own port and is therefore the natural
* third side rather than a re-implementation. Both designs the Python/R
* sides carry are reproduced here:
*
*   (a) the staggered absorbing panel (78_multiplegt_dyn.csv), reporting
*       Effect_1..4, Placebo_1..2 and the switcher-weighted aggregate
*       Av_tot_eff;
*   (b) the switch-off panel (78_multiplegt_dyn_off.csv), reporting
*       off_Effect_1..2 and off_Placebo_1 -- the branch that separates this
*       estimator from every cohort-based one.
*
* Index convention: the package labels effects from 1, so Effect_k is
* StatsPAI horizon k-1 and Placebo_k is horizon -k. The statistic names
* below use the package labelling so compare.py joins row-for-row.
*
* Standard errors are emitted for the record but are NOT the parity
* contract: did_multiplegt_dyn reports analytical influence-function SEs
* while sp.did_multiplegt_dyn has only a cluster bootstrap, so the two
* sides estimate different variance objects. The Python and R sides both
* emit se = null for this module; compare.py therefore never joins an SE
* row here.
*
* Tolerance: rel < 1e-6 on every effect, placebo and the aggregate.
* Dependency note: did_multiplegt_dyn requires gtools.

version 18
clear all

do _common.do
stata_parity_init, module(78_multiplegt_dyn)
stata_parity_open, module(78_multiplegt_dyn)

* ------------------------------------------------------------------
* (a) staggered absorbing design
* ------------------------------------------------------------------
import delimited "${STATA_PARITY_DATA}/78_multiplegt_dyn.csv", clear case(preserve)
count
local n_abs = r(N)

did_multiplegt_dyn y id t d, effects(4) placebo(2) cluster(id) graph_off

forvalues k = 1/4 {
    local bv = e(Effect_`k')
    local sv = e(se_effect_`k')
    local nk = e(N_switchers_effect_`k')
    stata_parity_row, stat("Effect_`k'") est(`bv') std(`sv') nob(`nk')
}
forvalues k = 1/2 {
    local bv = e(Placebo_`k')
    local sv = e(se_placebo_`k')
    local nk = e(N_switchers_placebo_`k')
    stata_parity_row, stat("Placebo_`k'") est(`bv') std(`sv') nob(`nk')
}

* The package's e(Av_tot_effect) is the switcher-weighted average, i.e.
* StatsPAI's aggregation="switchers". The Python side also emits an
* equal-weight variant (Av_tot_eff_simple_weights); the Stata command has
* no equal-weight aggregation option, so that statistic is deliberately
* left as a two-side (py/R) row.
local bv = e(Av_tot_effect)
local sv = e(se_avg_total_effect)
stata_parity_row, stat(Av_tot_eff) est(`bv') std(`sv') nob(`n_abs')

* ------------------------------------------------------------------
* (b) switch-off design
* ------------------------------------------------------------------
import delimited "${STATA_PARITY_DATA}/78_multiplegt_dyn_off.csv", clear case(preserve)
count
local n_off = r(N)

did_multiplegt_dyn y id t d, effects(2) placebo(1) cluster(id) graph_off

forvalues k = 1/2 {
    local bv = e(Effect_`k')
    local sv = e(se_effect_`k')
    local nk = e(N_switchers_effect_`k')
    stata_parity_row, stat("off_Effect_`k'") est(`bv') std(`sv') nob(`nk')
}
local bv = e(Placebo_1)
local sv = e(se_placebo_1)
local nk = e(N_switchers_placebo_1)
stata_parity_row, stat(off_Placebo_1) est(`bv') std(`sv') nob(`nk')

stata_parity_extra, key(stata_command) val("did_multiplegt_dyn y id t d, effects(4) placebo(2) cluster(id)")
stata_parity_extra, key(stata_command_off) val("did_multiplegt_dyn y id t d, effects(2) placebo(1) cluster(id)")
stata_parity_extra, key(aggregation) val("e(Av_tot_effect) = switcher-weighted; the equal-weight variant has no Stata counterpart")
stata_parity_extra, key(se_convention) val("analytical influence-function SEs; not joined -- py/R emit se=null because sp.did_multiplegt_dyn has only a cluster bootstrap")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(78_multiplegt_dyn)
