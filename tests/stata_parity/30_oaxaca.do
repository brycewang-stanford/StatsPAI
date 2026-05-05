* tests/stata_parity/30_oaxaca.do
*
* Module 30: Blinder-Oaxaca decomposition (threefold).
*   StatsPAI:  sp.oaxaca_blinder
*   R:         oaxaca::oaxaca (R = 100 bootstrap)
*   Stata:     oaxaca (Jann's oaxaca, threefold default)
*
* Tolerance: rel < 1e-3 on gap; threefold split.

version 18
clear all

do _common.do
stata_parity_init, module(30_oaxaca)
stata_parity_open, module(30_oaxaca)

import delimited "${STATA_PARITY_DATA}/30_oaxaca.csv", clear case(preserve)

* oaxaca syntax: oaxaca depvar varlist, by(group_indicator) [pooled|reference|...]
* Default = threefold decomposition.
oaxaca log_wage educ exper, by(female)

local n = e(N)
matrix B = e(b)

* oaxaca's e(b) layout (threefold default): equation:column.
*   overall:group_1, overall:group_2 (mean log_wage in each group)
*   overall:difference, overall:endowments, overall:coefficients, overall:interaction
local gap         = B[1, "overall:difference"]
local explained   = B[1, "overall:endowments"]
local unexplained = B[1, "overall:coefficients"]
local interaction = B[1, "overall:interaction"]
local mean_y_1    = B[1, "overall:group_1"]
local mean_y_2    = B[1, "overall:group_2"]

matrix V = e(V)
local se_explained   = sqrt(V["overall:endowments",   "overall:endowments"])
local se_unexplained = sqrt(V["overall:coefficients", "overall:coefficients"])

stata_parity_row, stat(gap)         est(`gap') nob(`n')
stata_parity_row, stat(explained)   est(`explained')   std(`se_explained')   nob(`n')
stata_parity_row, stat(unexplained) est(`unexplained') std(`se_unexplained') nob(`n')
* oaxaca::oaxaca labels group A as group 1 (female==0 in our DGP), group B as group 2 (female==1).
stata_parity_row, stat(mean_y_a) est(`mean_y_1') nob(`n')
stata_parity_row, stat(mean_y_b) est(`mean_y_2') nob(`n')

stata_parity_extra, key(decomposition) val(threefold)
stata_parity_extra, key(stata_command) val("oaxaca log_wage educ exper, by(female)")

stata_parity_close, module(30_oaxaca)
