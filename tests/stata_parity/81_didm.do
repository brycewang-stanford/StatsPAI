* tests/stata_parity/81_didm.do
*
* Module 81: de Chaisemartin--D'Haultfoeuille (2020) DID_M.
*   StatsPAI:  sp.did_multiplegt
*   R:         DIDmultiplegt::did_multiplegt (archived 0.1.4)
*   Stata:     did_multiplegt_old                     <-- this file
*
* Version note, mirroring the R side. dCDH's current Stata package routes
* the classic estimator through `did_multiplegt (old) ...`, which is the
* separately installable `did_multiplegt_old` command; the modern
* `did_multiplegt_dyn` implements the intertemporal estimator instead
* (Track A module 78). The R side has the same split and pins the archived
* 0.1.4 for exactly this reason.
*
* Emitted rows:
*   effect      -- the static DID_M estimator
*   placebo_1   -- the lag-1 placebo
*
* `dynamic_1` is deliberately NOT emitted. did_multiplegt_old refuses to
* compute dynamic effects unless `robust_dynamic` is also requested:
*
*     "If you request the computation of some dynamic effects, you need to
*      request that your estimators be robust to dynamic effects."
*
* and `robust_dynamic` switches to the intertemporal estimator, whose
* horizon-1 number is a different estimand (it equals module 78's
* off_Effect_2 on this same panel, which is where it is already graded).
* The archived R 0.1.4 predates that guard and still returns the
* non-robust dynamic effect, so `dynamic_1` stays a two-side (py/R) row
* rather than being matched against a different Stata estimand.
*
* Placebo sign: on this fixture did_multiplegt_old and DIDmultiplegt 0.1.4
* agree on the sign (+0.0324), so the row joins directly and the
* placebo_sign switch documented on the Python side does not bite here.
*
* Standard errors are not emitted: this runs breps(0), as does the R side.
*
* Tolerance: rel < 1e-6.

version 18
clear all

do _common.do
stata_parity_init, module(81_didm)
stata_parity_open, module(81_didm)

import delimited "${STATA_PARITY_DATA}/81_didm.csv", clear case(preserve)
count
local n = r(N)

did_multiplegt_old y id t d, placebo(1) breps(0)

local eff = e(effect_0)
local pl1 = e(placebo_1)
local n_eff = e(N_effect_0)
local n_pl1 = e(N_placebo_1)

stata_parity_row, stat(effect)    est(`eff') nob(`n_eff')
stata_parity_row, stat(placebo_1) est(`pl1') nob(`n_pl1')

stata_parity_extra, key(stata_command) val("did_multiplegt_old y id t d, placebo(1) breps(0)")
stata_parity_extra, key(dynamic_1_omitted) val("did_multiplegt_old requires robust_dynamic for dynamic effects, which changes the estimand; graded instead as module 78 off_Effect_2")
stata_parity_extra, key(se_convention) val("breps(0) on both the Stata and R sides, so no SE row is produced")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(81_didm)
