* tests/stata_parity/82_staggered.do
*
* Module 82: design-based staggered rollout (Roth & Sant'Anna 2023).
*   StatsPAI:  sp.staggered_rollout
*   R:         staggered::staggered / staggered_cs / staggered_sa (1.2.2)
*   Stata:     staggered (SSC, Caceres-Bravo's port of the same authors'
*              package) -- `ssc describe staggered` reports it as "module
*              implementing R staggered package based on Roth and Sant'Anna
*              (2023)".
*
* This is the only design-based module in the harness: identification comes
* from random adoption timing, not parallel trends. Both standard errors are
* emitted because reconciling only one would leave half the inference path
* unchecked -- the conservative Neyman bound and the adjusted one that
* subtracts the variance the randomisation itself identifies.
*
* efficient vs plugin mirrors the R side exactly: efficient leaves beta at
* betastar (the package default), plugin pins beta(1).

version 18
clear all

do _common.do
stata_parity_init, module(82_staggered)
stata_parity_open, module(82_staggered)

import delimited "${STATA_PARITY_DATA}/82_staggered.csv", clear case(preserve)

local stagopts i(unit) t(time) g(first_treat)

* ---- simple / cohort / calendar, each efficient and plugin --------------
foreach estimand in simple cohort calendar {
    foreach tag in efficient plugin {
        if "`tag'" == "efficient" {
            local betaopt ""
        }
        else {
            local betaopt "beta(1)"
        }
        quietly staggered y, `stagopts' estimand(`estimand') `betaopt'
        * e(results) is [estimate, se_adjusted, se_neyman] -- note the SE
        * order, which is not the one the column names suggest at a glance.
        * e(se_neyman)/e(se_adjusted) exist only for multi-estimate runs.
        matrix R = e(results)
        local nob = e(N)
        local est = R[1, 1]
        local va  = R[1, 2]
        local vn  = R[1, 3]
        stata_parity_row, stat(`estimand'_`tag') est(`est') nob(`nob')
        stata_parity_row, stat(`estimand'_`tag'_se_neyman) est(`vn') nob(`nob')
        stata_parity_row, stat(`estimand'_`tag'_se_adjusted) est(`va') nob(`nob')
    }
}

* ---- event study, horizons 0..2 ----------------------------------------
quietly staggered y, `stagopts' estimand(eventstudy) eventTime(0/2)
matrix R = e(results)
local nob = e(N)
forvalues k = 0/2 {
    local i = `k' + 1
    local est = R[`i', 1]
    local va  = R[`i', 2]
    local vn  = R[`i', 3]
    stata_parity_row, stat(eventstudy_e`k') est(`est') nob(`nob')
    stata_parity_row, stat(eventstudy_e`k'_se_neyman) est(`vn') nob(`nob')
    stata_parity_row, stat(eventstudy_e`k'_se_adjusted) est(`va') nob(`nob')
}

* ---- Callaway-Sant'Anna and Sun-Abraham comparisons ---------------------
foreach alias in cs sa {
    quietly staggered y, `stagopts' estimand(simple) `alias'
    matrix R = e(results)
    local nob = e(N)
    local est = R[1, 1]
    local va  = R[1, 2]
    local vn  = R[1, 3]
    stata_parity_row, stat(`alias'_simple) est(`est') nob(`nob')
    stata_parity_row, stat(`alias'_simple_se_neyman) est(`vn') nob(`nob')
    stata_parity_row, stat(`alias'_simple_se_adjusted) est(`va') nob(`nob')
}

stata_parity_extra, key(reference) val("staggered (SSC)")

stata_parity_close, module(82_staggered)
