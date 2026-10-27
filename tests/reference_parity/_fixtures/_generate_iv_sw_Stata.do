* Sanderson-Windmeijer first-stage statistics from ivreg2 / ivreghdfe on
* iv_sw.csv: two and three endogenous regressors, iid / robust / cluster,
* without fixed effects (ivreg2) and with f + g absorbed (ivreghdfe),
* including a cluster variable fc in which f is nested.
*   python tests/reference_parity/_fixtures/_generate_iv_sw_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_iv_sw_Stata.do
* ivreg2 4.1.12, ranktest 2.0.04, ivreghdfe 1.1.4, reghdfe 6.13.1 (SSC);
* set PLUS to a private ado directory holding them before running.
version 18
clear all
set type double
import delimited using "tests/reference_parity/_fixtures/iv_sw.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/iv_sw_Stata.json", write replace
file write `fh' "{" _n
local first_case 1
foreach k in 2 3 {
    if `k' == 2 {
        local en d1 d2
        local iv z1 z2 z3
    }
    else {
        local en d1 d2 d3
        local iv z1 z2 z3 z4
    }
    foreach fe in nofe fe {
        foreach v in iid robust cluster nested {
            if "`fe'" == "nofe" & "`v'" == "nested" continue
            local opt
            if "`v'" == "robust" local opt robust
            if "`v'" == "cluster" local opt cluster(c)
            if "`v'" == "nested" local opt cluster(fc)
            if "`fe'" == "nofe" qui ivreg2 y w (`en' = `iv'), first `opt'
            else qui ivreghdfe y w (`en' = `iv'), absorb(f g) first `opt'
            matrix F = e(first)
            if !`first_case' file write `fh' "," _n
            local first_case 0
            file write `fh' `"  "k`k'_`fe'_`v'": {"N": "' (e(N))
            foreach r in SWF SWFdf1 SWFdf2 SWFp SWchi2 SWchi2p SWr2 F {
                file write `fh' `", "`r'": ["'
                local j 0
                foreach e of local en {
                    if `j' file write `fh' ", "
                    local ++j
                    file write `fh' %24.16e (F["`r'", "`e'"])
                }
                file write `fh' "]"
            }
            file write `fh' "}"
        }
    }
}
file write `fh' _n "}" _n
file close `fh'
