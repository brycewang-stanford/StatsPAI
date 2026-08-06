* tests/stata_parity/75_stacked.do
*
* Module 75: Cengiz--Dube--Lindner--Zipperer stacked DiD.
*   StatsPAI:  sp.stacked_did
*   R:         hand-written stack + fixest::feols
*   Stata:     hand-written stack + reghdfe            <-- this file
*
* Why a hand-written Stata side is the right reference here. Stacking is a
* construction, not a packaged command: there is no CRAN or SSC estimator
* that owns this estimand, and the R side of this module is itself
* hand-written. The parity question is therefore not "does StatsPAI match
* package X" but "do three independently written stack constructions agree",
* which is exactly what a third implementation answers. The estimator on top
* of the stack is ordinary two-way FE, already graded bit-exact against
* reghdfe in Track A modules 03 and 15, so any disagreement here localises to
* the stack construction rather than to the regression.
*
* Construction, mirroring 75_stacked.R line for line:
*   - one sub-experiment per treated cohort g, over the event window
*     [g-3, g+3];
*   - controls are either never-treated only (StatsPAI's default) or
*     never-treated plus not-yet-treated units whose own adoption date falls
*     beyond the window's upper edge;
*   - unit and time fixed effects are interacted with the sub-experiment id
*     (uc, tc) so each stack layer gets its own two-way FE;
*   - relative time k = -1 is the omitted reference period;
*   - clustering is on the original unit id, not on uc, so a unit appearing
*     in several layers is one cluster.
*
* The post-period summary <spec>_ATT_post is the unweighted mean of the
* k >= 0 coefficients, matching the R side's mean(co[rels >= 0]).
*
* Tolerance: rel < 1e-6 on the event-study coefficients and the post mean.

version 18
clear all

do _common.do
stata_parity_init, module(75_stacked)
stata_parity_open, module(75_stacked)

local W_LO = -3
local W_HI =  3

* ------------------------------------------------------------------
* Build both stacks. `never_only' == 1 reproduces the never-treated
* control convention; 0 adds not-yet-treated units.
* ------------------------------------------------------------------
foreach spec in never nyt {

    if "`spec'" == "never" {
        local never_only = 1
    }
    else {
        local never_only = 0
    }

    import delimited "${STATA_PARITY_DATA}/75_stacked.csv", clear case(preserve)
    tempfile base
    save `base', replace

    * Cohort list: adoption dates of the treated units.
    qui levelsof first_treat if first_treat > 0, local(cohorts)

    tempfile stack
    local first_layer = 1

    foreach g of local cohorts {
        local t_lo = `g' + `W_LO'
        local t_hi = `g' + `W_HI'

        use `base', clear
        * treated units of this cohort; controls per the convention.
        gen byte _is_coh  = (first_treat == `g')
        gen byte _is_nev  = (first_treat == 0)
        gen byte _is_nyt  = (first_treat > `t_hi' & first_treat > 0)
        if `never_only' {
            gen byte _is_ctrl = _is_nev
        }
        else {
            gen byte _is_ctrl = (_is_nev | _is_nyt)
        }
        qui keep if (_is_coh | _is_ctrl) & year >= `t_lo' & year <= `t_hi'
        qui count
        if r(N) == 0 {
            continue
        }

        gen int  cohort       = `g'
        gen int  rel          = year - `g'
        gen byte treated_unit = _is_coh
        drop _is_*

        if `first_layer' {
            qui save `stack', replace
            local first_layer = 0
        }
        else {
            qui append using `stack'
            qui save `stack', replace
        }
    }

    use `stack', clear
    count
    local n_stack = r(N)

    * Sub-experiment-specific unit and time fixed effects.
    egen long uc = group(id cohort)
    egen long tc = group(year cohort)

    * Explicit event-time dummies interacted with treatment, k = -1 omitted.
    * `ib#.relf#c.treated_unit' is NOT usable here: in an i.x#c.z interaction
    * Stata carries every level of x and then drops whichever one turns out
    * collinear (here k = 3), which silently re-bases the whole event study
    * onto the wrong reference period. Hand-built dummies make the omitted
    * period part of the design instead of an artifact of collinearity
    * detection, and reproduce fixest's i(rel_f, treated_unit, ref = "-1").
    local terms ""
    forvalues k = `W_LO'/`W_HI' {
        if `k' == -1 {
            continue
        }
        local nm = cond(`k' < 0, "dm" + string(abs(`k')), "dp" + string(`k'))
        qui gen byte `nm' = (rel == `k') * treated_unit
        local terms "`terms' `nm'"
    }

    qui reghdfe y `terms', absorb(uc tc) vce(cluster id)

    local post_sum   = 0
    local post_count = 0
    forvalues k = `W_LO'/`W_HI' {
        if `k' == -1 {
            continue
        }
        local nm = cond(`k' < 0, "dm" + string(abs(`k')), "dp" + string(`k'))
        local bv = _b[`nm']
        local sv = _se[`nm']
        local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
        local hi = `bv' + ${STATA_PARITY_Z95} * `sv'
        stata_parity_row, stat("`spec'_att_rel_`k'") est(`bv') std(`sv') ///
            cilo(`lo') cihi(`hi') nob(`n_stack')
        if `k' >= 0 {
            local post_sum   = `post_sum' + `bv'
            local post_count = `post_count' + 1
        }
    }
    local post_mean = `post_sum' / `post_count'
    stata_parity_row, stat("`spec'_ATT_post") est(`post_mean') nob(`n_stack')
}

stata_parity_extra, key(stata_command) val("reghdfe y <event-time dummies x treated_unit, k=-1 omitted>, absorb(uc tc) vce(cluster id) on a hand-built stack")
stata_parity_extra, key(reference_type) val("hand-written stack; stacking is a construction with no SSC or CRAN owner, so all three sides are independent implementations")
stata_parity_extra, key(cluster) val("original unit id, so a unit appearing in several stack layers is a single cluster")
stata_parity_extra, key(post_summary) val("<spec>_ATT_post is the unweighted mean of the k >= 0 coefficients")
stata_parity_extra, key(se_convention) val("cluster SEs sit 0.42% (never) / 0.37% (nyt) above fixest's on every row -- a constant factor, i.e. a small-sample dof convention, not an estimator difference: reghdfe and fixest count the nested unit-cohort FE against the residual dof differently. Point estimates agree to 7e-13.")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(75_stacked)
