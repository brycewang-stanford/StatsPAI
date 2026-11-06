* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_panel_xt_tools_stata_parity.py
*
* Reads panel_xt_tools.csv (written by _generate_panel_xt_tools_data.py) and
* writes panel_xt_tools_Stata.csv: one "key,value" row per number, %23.16e.
* Rows whose key starts with "meta." carry text (the Stata build and the
* community-command versions) in place of a number.
*
* Run (from this directory, Stata 18 MP):
*   python _generate_panel_xt_tools_data.py
*   stata-mp -b do _generate_panel_xt_tools_stata.do
* Community commands: xtserial (Stata Journal st0039) and xtoverid, ivreg2,
* ranktest (SSC), installed into the private ado directory
* _ado_panel_xt_tools/ next to this file (gitignored), not the user's PLUS.
* They are run as black boxes; StatsPAI is written from the method papers
* and the help files.
* ---------------------------------------------------------------------------
clear all
set more off
set type double
capture mkdir "_ado_panel_xt_tools"
adopath ++ "_ado_panel_xt_tools"
net set ado "_ado_panel_xt_tools"
capture which xtserial
if _rc net install st0039, from(https://www.stata-journal.com/software/sj3-2)
foreach pkg in ranktest ivreg2 xtoverid {
    capture which `pkg'
    if _rc ssc install `pkg'
}

capture program drop emit
program emit
    args key value
    file write fh "`key'," %23.16e (`value') _n
end
capture program drop banner
program banner
    * the "*!" version line -which- prints for a community command
    args cmd
    findfile `cmd'.ado
    tempname h
    file open `h' using "`r(fn)'", read text
    file read `h' line
    while substr(`"`line'"', 1, 2) != "*!" & r(eof) == 0 {
        file read `h' line
    }
    file close `h'
    local line = subinstr(`"`line'"', ",", ";", .)
    file write fh "meta.`cmd'," `"`line'"' _n
end
capture file close fh
file open fh using "panel_xt_tools_Stata.csv", write replace
file write fh "key,value" _n
file write fh "meta.stata_version,`c(stata_version)'" _n
file write fh "meta.born_date,`c(born_date)'" _n
file write fh "meta.edition,`c(edition_real)'" _n
foreach cmd in xtserial xtoverid ivreg2 {
    banner `cmd'
}

import delimited using "panel_xt_tools.csv", clear asdouble
xtset id t

* ===================================================================== xtsum
foreach v in y x1 ti xm {
    xtsum `v'
    foreach s in N n Tbar mean sd min max sd_b min_b max_b sd_w min_w max_w {
        emit xtsum.`v'.`s' r(`s')
    }
}

* ================================================================== xtserial
local s1 "x1 w x2"
local s2 "x1"
local s3 "x1 w x2 ti"
forvalues k = 1/3 {
    xtserial y `s`k''
    emit xtserial.s`k'.F r(F)
    emit xtserial.s`k'.p r(p)
    emit xtserial.s`k'.corr r(corr)
    emit xtserial.s`k'.df r(df)
    emit xtserial.s`k'.df_r r(df_r)
}
xtserial y x1 w x2 if bal == 1
emit xtserial.bal.F r(F)
emit xtserial.bal.p r(p)
emit xtserial.bal.corr r(corr)

* ================================================================== xtoverid
xtreg y x1 w x2 ti, re vce(robust)
xtoverid
emit xtoverid.rob.j r(j)
emit xtoverid.rob.df r(jdf)
emit xtoverid.rob.p r(jp)
xtreg y x1 w x2 ti, re vce(cluster g)
xtoverid
emit xtoverid.clu.j r(j)
emit xtoverid.clu.df r(jdf)
emit xtoverid.clu.p r(jp)
xtreg y x1 x2, re vce(robust)
xtoverid
emit xtoverid.two.j r(j)
emit xtoverid.two.df r(jdf)
emit xtoverid.two.p r(jp)
xtreg y x1 w x2 ti if bal == 1, re vce(robust)
xtoverid
emit xtoverid.bal.j r(j)
emit xtoverid.bal.df r(jdf)
emit xtoverid.bal.p r(jp)

* ================================================================= xthtaylor
capture program drop emitht
program emitht
    args tag
    foreach v in x1 w x2 ti zi _cons {
        capture emit ht.`tag'.b_`v' _b[`v']
        capture emit ht.`tag'.se_`v' _se[`v']
    }
    emit ht.`tag'.sigma_u e(sigma_u)
    emit ht.`tag'.sigma_e e(sigma_e)
    emit ht.`tag'.rho e(rho)
    emit ht.`tag'.N e(N)
    emit ht.`tag'.N_g e(N_g)
    capture emit ht.`tag'.chi2 e(chi2)
end
xthtaylor y x1 w x2 ti zi, endog(x2 zi)
emitht conv
xthtaylor y x1 w x2 ti zi, endog(x2 zi) vce(robust)
emitht rob
xthtaylor y x1 w x2 ti zi, endog(x2 zi) vce(cluster g)
emitht clu
* two endogenous time-varying regressors
xthtaylor y x1 w x2 ti zi, endog(w x2 zi)
emitht endog3
* Amemiya-MaCurdy on the balanced panels (common first period)
xthtaylor y x1 w x2 ti zi if bal == 1, endog(x2 zi) amacurdy
emitht am_conv
xthtaylor y x1 w x2 ti zi if bal == 1, endog(x2 zi) amacurdy vce(robust)
emitht am_rob
xthtaylor y x1 w x2 ti zi if bal == 1, endog(x2 zi) amacurdy vce(cluster g)
emitht am_clu

file close fh
