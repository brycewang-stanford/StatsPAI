* tests/stata_parity/04_csdid.do
*
* Module 04: CS-DiD simple ATT.
*   StatsPAI:  sp.callaway_santanna(...).simple_att
*   R:         did::att_gt + did::aggte(type="simple"), method=reg
*   Stata:     csdid + estat simple, method(reg)
*
* csdid uses Wald-style asymptotic SE; matches did::aggte with
* bstrap=FALSE.

version 18
clear all

do _common.do
stata_parity_init, module(04_csdid)
stata_parity_open, module(04_csdid)

import delimited "${STATA_PARITY_DATA}/04_csdid.csv", clear case(preserve)

* csdid expects the never-treated cohort coded as 0 (or as "Inf" string,
* not supported in Stata). Inspect: r_parity uses first_treat = 0 for
* never-treated. csdid syntax: outcome ivar(id) time(t) gvar(g).
* Default control group = never-treated (matches R's control_group="nevertreated").
* long2 requests the UNIVERSAL base period. StatsPAI defaults to it and R
* did defaults to 'varying'; the simple ATT averages post-treatment cells
* only and so cannot see the difference, which is why this module matched
* for years without the option being pinned. The event-study rows below
* can see it, and would disagree on every pre-treatment cell without it.
csdid lemp, ivar(countyreal) time(year) gvar(first_treat) method(reg) long2

* `estat simple` aggregates ATT(g,t) over post-treatment cells.
estat simple

* The post-`estat simple` ereturn matrix is r(b)/r(V) (not e()).
matrix B = r(b)
matrix V = r(V)
local bv = B[1, 1]
local sv = sqrt(V[1, 1])
local lo = `bv' - ${STATA_PARITY_Z95} * `sv'
local hi = `bv' + ${STATA_PARITY_Z95} * `sv'

count
local n = r(N)

stata_parity_row, stat(simple_ATT) est(`bv') std(`sv') cilo(`lo') cihi(`hi') nob(`n')

* ---- Aggregation vectors -------------------------------------------- *
* estat event / group / calendar post r(b), r(V) with csdid's own labels:
* Tm<k>/Tp<k> for event time, g<year> / t<year> for the others.
foreach spec in event group calendar {
    estat `spec'
    matrix B = r(b)
    matrix V = r(V)
    local nm : colnames B
    foreach v of local nm {
        local bv = B[1, "`v'"]
        local sv = sqrt(V["`v'", "`v'"])
        local key = "`v'"
        * Normalise csdid's labels onto the shared statistic names.
        if ("`spec'" == "event") {
            if (substr("`key'", 1, 2) == "Tm") {
                local key = "-" + substr("`key'", 3, .)
            }
            else if (substr("`key'", 1, 2) == "Tp") {
                local key = "+" + substr("`key'", 3, .)
            }
            else if ("`key'" == "Post_avg") local key = "overall"
            else continue
            stata_parity_row, stat(event_`key') est(`bv') std(`sv') nob(`n')
        }
        else {
            * csdid labels these G2004 / T2004 and GAverage / CAverage.
            * Strip only the single leading letter: a subinstr of "g"
            * also eats the one inside "Average".
            if (strpos("`key'", "Average") > 0) {
                local key = "overall"
            }
            else {
                local key = substr("`key'", 2, .)
            }
            stata_parity_row, stat(`spec'_`key') est(`bv') std(`sv') nob(`n')
        }
    }
}

stata_parity_extra, key(base_period) val("universal (csdid long2)")
stata_parity_extra, key(estimator) val(reg)
stata_parity_extra, key(control_group) val(nevertreated)
stata_parity_extra, key(stata_command) val("csdid ... method(reg) long2 | estat simple/event/group/calendar")

stata_parity_close, module(04_csdid)
