* tests/stata_parity/73_did2s.do
*
* Module 73: Gardner (2022) two-stage difference-in-differences.
*   StatsPAI:  sp.gardner_did
*   R:         did2s::did2s (1.2.1)
*   Stata:     did2s                                  <-- this file
*
* Butts' `did2s` on SSC is the Stata port of the same two-stage estimator,
* so this is a like-for-like third side rather than a re-implementation.
*
* Data: the mpdta replica dumped by 73_did2s.py. `first_treat == 0` marks
* never-treated counties, so the post indicator is
* (first_treat > 0 & year >= first_treat) -- which is exactly the committed
* `treat` column; this script rebuilds it explicitly and asserts the two
* agree so an upstream fixture change cannot silently redefine treatment.
*
* SE convention. Like the R package, `did2s` propagates first-stage
* estimation error into the second-stage variance; sp.gardner_did's default
* vce="analytic" clusters the stage-2 residuals only and therefore lands
* low. This is the documented convention gap the module exists to record,
* and the Stata SE lands on R's number (rel 2.6e-10), which is the useful
* evidence: the gap is a StatsPAI default choice, not an R quirk. The
* Python side emits the matching bootstrap SE alongside.
*
* Tolerance: rel < 1e-6 on the point estimate.

version 18
clear all

do _common.do
stata_parity_init, module(73_did2s)
stata_parity_open, module(73_did2s)

import delimited "${STATA_PARITY_DATA}/73_did2s.csv", clear case(preserve)
count
local n = r(N)

gen byte treated = (first_treat > 0 & year >= first_treat)
qui count if treated != treat
if r(N) != 0 {
    display as error "fixture drift: rebuilt post indicator disagrees with committed treat column in `r(N)' rows"
    exit 459
}

did2s lemp, first_stage(i.countyreal i.year) second_stage(treated) ///
    treatment(treated) cluster(countyreal)

local bv = _b[treated]
local sv = _se[treated]
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'

stata_parity_row, stat(static_ATT) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')

stata_parity_extra, key(stata_command) val("did2s lemp, first_stage(i.countyreal i.year) second_stage(treated) treatment(treated) cluster(countyreal)")
stata_parity_extra, key(se_convention) val("did2s propagates stage-1 estimation error, matching did2s::did2s in R; sp.gardner_did vce='analytic' does not and is the documented gap")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(73_did2s)
