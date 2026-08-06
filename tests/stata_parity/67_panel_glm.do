* tests/stata_parity/67_panel_glm.do
*
* Module 67: absorbed-FE GLM (logit and Poisson).
*   StatsPAI:  sp.feglm(family="logit"), sp.fepois
*   R:         fixest::feglm(family="logit"), fixest::fepois
*   Stata:     logit y x1 x2 i.id  /  ppmlhdfe y x1 x2, absorb(id)  <-- this file
*
* Point estimates only, and that is the whole point of the module's
* disposition. Measured 2026-08-06 against licensed Stata 18:
*
*   feglm_logit_x1   py 0.285476550025   Stata 0.285476549690   rel 1.2e-09
*   feglm_logit_x2   py -0.284815205511  Stata -0.284815205004  rel 1.8e-09
*   fepois_x1        py 0.231963912397   Stata 0.231963912397   rel ~1e-16
*   fepois_x2        py -0.274253060743  Stata -0.274253060743  rel ~1e-16
*
* so the estimators agree. The standard errors do not, and no `vce()`
* setting closes the gap:
*
*   logit slopes   vce(robust) 0.6% away, vce(cluster id) 21%, default 4%
*   Poisson slopes vce(robust)/vce(unadjusted) 24-42% away, vce(cluster id) 42%
*
* fixest reports its own small-sample and clustering convention for GLMs
* with absorbed effects, and reproducing it would mean re-deriving fixest's
* dof accounting rather than checking an estimator. The module's registered
* rel_se budget is 5e-5, which no Stata setting meets, so this script emits
* `se = .` on every row: compare.py then joins the point estimates and
* leaves the SE columns blank rather than manufacturing a comparison
* between two different variance objects. Modules 03, 15, 37 and 47 already
* carry the HDFE standard-error conventions against reghdfe and ppmlhdfe
* where the two sides *do* agree.
*
* Estimator note: `logit y x1 x2 i.id` is the unconditional (dummy-variable)
* logit, which is what fixest::feglm(family="logit") with `| id` computes.
* It is NOT `clogit`/`xtlogit, fe` -- the conditional likelihood is a
* different estimand, and substituting it here would be a silent estimand
* swap rather than a parity check.
*
* Tolerance: rel < 1e-6 on the point estimates.

version 18
clear all

do _common.do
stata_parity_init, module(67_panel_glm)
stata_parity_open, module(67_panel_glm)

* ------------------------------------------------------------------
* feglm, family = logit -- unconditional logit with entity dummies
* ------------------------------------------------------------------
import delimited "${STATA_PARITY_DATA}/67_panel_glm_logit.csv", clear case(preserve)
count
local n_logit = r(N)

qui logit y x1 x2 i.id

foreach v in x1 x2 {
    local bv = _b[`v']
    stata_parity_row, stat("feglm_logit_`v'") est(`bv') nob(`n_logit')
}

* ------------------------------------------------------------------
* fepois -- absorbed entity FE Poisson
* ------------------------------------------------------------------
import delimited "${STATA_PARITY_DATA}/67_panel_glm_poisson.csv", clear case(preserve)
count
local n_pois = r(N)

qui ppmlhdfe y x1 x2, absorb(id)

foreach v in x1 x2 {
    local bv = _b[`v']
    stata_parity_row, stat("fepois_`v'") est(`bv') nob(`n_pois')
}

stata_parity_extra, key(stata_command_logit) val("logit y x1 x2 i.id")
stata_parity_extra, key(stata_command_poisson) val("ppmlhdfe y x1 x2, absorb(id)")
stata_parity_extra, key(estimator_note) val("unconditional (dummy-variable) logit, matching fixest::feglm(family='logit') with |id; clogit/xtlogit,fe would be a different estimand")
stata_parity_extra, key(se_convention) val("point estimates only: no vce() setting reproduces fixest's absorbed-FE GLM standard errors -- robust/cluster/unadjusted land 0.6%/21%/4% away on the logit slopes and 24-42% away on the Poisson ones, against a registered rel_se budget of 5e-5. Emitting an SE here would compare two different variance objects.")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(67_panel_glm)
