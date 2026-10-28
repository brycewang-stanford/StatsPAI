* tests/stata_parity/91_biprobit.do
*
* Module 91: seemingly unrelated bivariate probit.
*   StatsPAI:  sp.biprobit (model behind sp.positive_correlation_test)
*   R:         GJRM::gjrm(model = "B", copula = "N") (bridge)
*   Stata:     biprobit (built in; treated as canonical)
*
* Tight ML tolerances: at the defaults Stata stops about 3e-9 (relative)
* short of the maximum, which is inside the budget but would make the
* reference depend on the optimiser's stopping rule.
*
* Tolerance: rel < 1e-6.

version 18
clear all

do _common.do
stata_parity_init, module(91_biprobit)
stata_parity_open, module(91_biprobit)

import delimited "${STATA_PARITY_DATA}/91_biprobit.csv", clear case(preserve)

local n = _N

biprobit (y1 = x1 x3) (y2 = x2 x3), ///
    tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-14)

foreach t in x1 x3 _cons {
    local name = cond("`t'" == "_cons", "cons", "`t'")
    local b  = _b[y1:`t']
    local se = _se[y1:`t']
    stata_parity_row, stat(eq1_b_`name') est(`b') std(`se') nob(`n')
}
foreach t in x2 x3 _cons {
    local name = cond("`t'" == "_cons", "cons", "`t'")
    local b  = _b[y2:`t']
    local se = _se[y2:`t']
    stata_parity_row, stat(eq2_b_`name') est(`b') std(`se') nob(`n')
}
local b  = _b[/athrho]
local se = _se[/athrho]
stata_parity_row, stat(athrho) est(`b') std(`se') nob(`n')
local v = e(rho)
stata_parity_row, stat(rho) est(`v') nob(`n')
local v = e(ll)
stata_parity_row, stat(loglik) est(`v') nob(`n')
local v = e(chi2_c)
stata_parity_row, stat(lr_rho0) est(`v') nob(`n')

stata_parity_extra, key(stata_command) ///
    val("biprobit (y1 = x1 x3) (y2 = x2 x3), tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-14)")
stata_parity_extra, key(reference_role) val("canonical (built-in biprobit)")

stata_parity_close, module(91_biprobit)
