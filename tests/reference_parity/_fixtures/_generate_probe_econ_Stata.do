* Reference numbers for tests/reference_parity/test_probe_econ_parity.py
* (2026-11 blind applied-economics probe). Stata 18 MP on the CSVs bundled
* with StatsPAI; one "tag,kind,name,value" row per number.
* Run from the repository root:
*   stata-mp -q -b do tests/reference_parity/_fixtures/_generate_probe_econ_Stata.do
* Needs from SSC: oaxaca, boottest, ivreghdfe (+ ivreg2, reghdfe, ftools).
version 18
clear all
set more off
set type double

cap program drop dumpe
program define dumpe
    args tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    forvalues i = 1/`k' {
        local nm : word `i' of `names'
        file write fh "`tag',b,`nm'," %24.17e (`b'[1,`i']) _n
        file write fh "`tag',se,`nm'," %24.17e (sqrt(`V'[`i',`i'])) _n
    }
end
cap program drop dumps
program define dumps
    * dumps tag name value
    args tag name value
    file write fh "`tag',s,`name'," %24.17e (`value') _n
end

file open fh using "tests/reference_parity/_fixtures/probe_econ_Stata.csv", write replace
file write fh "tag,kind,name,value" _n

* ---- rank-sum p-values of a balance table (NSW / Lalonde) ---------------
import delimited using "src/statspai/datasets/data/lalonde_matchit.csv", clear asdouble
foreach v in age educ hispanic nodegree re74 {
    ranksum `v', by(treat)
    dumps ranksum `v' 2*normal(-abs(r(z)))
}

* ---- Oaxaca: delta-method SEs with the means held fixed -----------------
import delimited using "src/statspai/datasets/data/card_1995.csv", clear asdouble
local X educ exper expersq south smsa
oaxaca lwage `X', by(black) weight(0.5) vce(robust) fixed
dumpe oaxaca_reimers
local wc = 2307/3010
oaxaca lwage `X', by(black) weight(`wc') vce(robust) fixed
dumpe oaxaca_cotton
oaxaca lwage `X', by(black) omega vce(robust) fixed
dumpe oaxaca_pooled
oaxaca lwage `X', by(black) vce(robust) fixed
dumpe oaxaca_threefold

* ---- wild cluster bootstrap after regress: 8 clusters, enumerated -------
gen g = floor(exper/3)
regress lwage black south smsa educ, vce(cluster g)
foreach v in black south smsa educ {
    boottest `v', cluster(g) reps(999) nograph
    dumps boottest_p `v' r(p)
    matrix C = r(CI)
    dumps boottest_lo `v' C[1,1]
    dumps boottest_hi `v' C[1,2]
}

* ---- 2SLS with absorbed effects: fit statistics ------------------------
ivreghdfe lwage black south (educ = nearc4 nearc2), absorb(g smsa)
dumps ivreghdfe r2_within e(r2)
dumps ivreghdfe N e(N)

* ---- correlated random effects: xtreg, re on x and its unit means -------
import delimited using "src/statspai/datasets/data/sasp_panel.csv", clear asdouble
local X unsafe llength reg age_cl hot massage_cl
foreach v in id session lnw `X' {
    drop if missing(`v')
}
xtset id session
foreach v of local X {
    egen double m_`v' = mean(`v'), by(id)
}
xtreg lnw `X' m_*, re
dumps mundlak r2_w e(r2_w)
dumps mundlak r2_b e(r2_b)
dumps mundlak r2_o e(r2_o)
dumps mundlak N e(N)

* ---- xtreg, fe with i.year: what sp.stata sends to sp.feols as C(year) --
import delimited using "src/statspai/datasets/data/castle_2013.csv", clear asdouble
xtset sid year
xtreg l_homicide post unemployrt i.year, fe vce(cluster sid)
dumpe xtreg_fe_iyear

file close fh
