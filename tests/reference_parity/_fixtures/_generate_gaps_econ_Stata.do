* Stata 18 MP reference numbers for tests/reference_parity/test_gaps_econ_parity.py.
* One "tag,kind,name,value" row per number.
* Run from the repository root:
*   stata-mp -q -b do tests/reference_parity/_fixtures/_generate_gaps_econ_Stata.do
* Needs from SSC: oaxaca, ivreg2, ranktest, weakivtest (+ avar).
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
    args tag name value
    file write fh "`tag',s,`name'," %24.17e (`value') _n
end

file open fh using "tests/reference_parity/_fixtures/gaps_econ_Stata.csv", write replace
file write fh "tag,kind,name,value" _n

* ---- Oaxaca: covariate means treated as estimated (the oaxaca default) ---
import delimited using "src/statspai/datasets/data/card_1995.csv", clear asdouble
local X educ exper expersq south smsa
oaxaca lwage `X', by(black) weight(1) vce(robust) detail
dumpe oaxaca_ref0
oaxaca lwage `X', by(black) weight(0) vce(robust) detail
dumpe oaxaca_ref1
oaxaca lwage `X', by(black) weight(0.5) vce(robust) detail
dumpe oaxaca_reimers
local wc = 2307/3010
oaxaca lwage `X', by(black) weight(`wc') vce(robust) detail
dumpe oaxaca_cotton
oaxaca lwage `X', by(black) omega vce(robust)
dumpe oaxaca_pooled
oaxaca lwage `X', by(black) omega vce(robust) fixed
dumpe oaxaca_pooled_fixed
oaxaca lwage `X', by(black) vce(robust)
dumpe oaxaca_threefold
oaxaca lwage `X', by(black) weight(1) vce(robust) fixed
dumpe oaxaca_ref0_fixed

* ---- ivreg2 rank statistics and the weakivtest effective F ---------------
* Card (1995): educ instrumented by nearc4 and nearc2.
local W exper expersq black south smsa
ivreg2 lwage `W' (educ = nearc4 nearc2)
dumps iv_iid idstat e(idstat)
dumps iv_iid idp e(idp)
dumps iv_iid cdf e(cdf)
ivreg2 lwage `W' (educ = nearc4 nearc2), robust
dumps iv_robust idstat e(idstat)
dumps iv_robust idp e(idp)
dumps iv_robust rkf e(rkf)
weakivtest
dumps iv_robust F_eff r(F_eff)
ivreg2 lwage `W' (educ = nearc4 nearc2), cluster(exper)
dumps iv_cluster idstat e(idstat)
dumps iv_cluster idp e(idp)
dumps iv_cluster rkf e(rkf)
weakivtest
dumps iv_cluster F_eff r(F_eff)

file close fh
