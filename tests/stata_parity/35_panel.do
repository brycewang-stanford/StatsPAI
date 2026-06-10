* tests/stata_parity/35_panel.do
*
* Module 35: Panel FE / RE / Hausman.
*   StatsPAI:  sp.panel
*   R:         plm::plm(... model="within"/"random") + plm::phtest
*   Stata:     xtreg, fe / xtreg, re / hausman
*
* Tolerance: rel < 1e-3 on FE / RE coefficients.

version 18
clear all

do _common.do
stata_parity_init, module(35_panel)
stata_parity_open, module(35_panel)

import delimited "${STATA_PARITY_DATA}/35_panel.csv", clear case(preserve)
xtset unit year

local n = _N

* Fixed effects (within)
xtreg y x1 x2, fe
local fe_b1 = _b[x1]
local fe_b2 = _b[x2]
local fe_se1 = _se[x1]
local fe_se2 = _se[x2]
estimates store fe

* Random effects (Swamy-Arora GLS, plm's default for type="random")
xtreg y x1 x2, re
local re_b1 = _b[x1]
local re_b2 = _b[x2]
local re_se1 = _se[x1]
local re_se2 = _se[x2]
estimates store re

* Hausman test
hausman fe re, sigmamore
local haus_chi2 = r(chi2)
local haus_p    = r(p)

stata_parity_row, stat(fe_beta_x1) est(`fe_b1') std(`fe_se1') nob(`n')
stata_parity_row, stat(fe_beta_x2) est(`fe_b2') std(`fe_se2') nob(`n')
stata_parity_row, stat(re_beta_x1) est(`re_b1') std(`re_se1') nob(`n')
stata_parity_row, stat(re_beta_x2) est(`re_b2') std(`re_se2') nob(`n')
stata_parity_row, stat(hausman_chi2_stata_sigmamore)   est(`haus_chi2') nob(`n')
stata_parity_row, stat(hausman_pvalue_stata_sigmamore) est(`haus_p')    nob(`n')

stata_parity_extra, key(method_fe) val(within)
stata_parity_extra, key(method_re) val(SwArora_GLS)
stata_parity_extra, key(stata_command) val("xtreg ..., fe/re + hausman")

stata_parity_close, module(35_panel)
