* tests/stata_parity/90_eivreg.do
*
* Module 90: errors-in-variables regression with known reliabilities.
*   StatsPAI:  sp.eivreg
*   R:         eivtools::eivreg 0.1-8 (bridge, written to reproduce Stata)
*   Stata:     eivreg (built in; the canonical implementation)
*
* Two designs: unweighted, and analytic weights [aw=w].
*
* Tolerance: rel < 1e-6.

version 18
clear all

do _common.do
stata_parity_init, module(90_eivreg)
stata_parity_open, module(90_eivreg)

import delimited "${STATA_PARITY_DATA}/90_eivreg.csv", clear case(preserve)

local n = _N

foreach design in unw aw {
    if "`design'" == "unw" {
        eivreg y x1 x2 z, r(x1 .8 x2 .7)
    }
    else {
        eivreg y x1 x2 z [aw=w], r(x1 .8 x2 .7)
    }
    foreach term in x1 x2 z _cons {
        local name = cond("`term'" == "_cons", "cons", "`term'")
        local b  = _b[`term']
        local se = _se[`term']
        stata_parity_row, stat(`design'_b_`name') est(`b') std(`se') nob(`n')
    }
}

stata_parity_extra, key(stata_command) ///
    val("eivreg y x1 x2 z [aw=w], r(x1 .8 x2 .7)")
stata_parity_extra, key(reference_role) val("canonical (built-in eivreg)")

stata_parity_close, module(90_eivreg)
