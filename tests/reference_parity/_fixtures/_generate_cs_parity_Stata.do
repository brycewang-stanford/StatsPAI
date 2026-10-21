* Stata csdid reference for sp.callaway_santanna(parity="stata") / sp.aggte.
*
* csdid / csdid_estat are used as a black box: only their outputs are read.
* Same panels as _generate_cs_parity_R.R (cs_parity_mpdta.csv and
* cs_parity_always.csv). Each case runs csdid with its defaults (short
* pre-treatment gaps, method(dripw), analytic standard errors) and records
* every ATT(g, t) with its SE (e(b), e(V)), estat pretrend, and
* estat simple / group / calendar / event (r(table): b, se, ll, ul, df,
* crit). One more case refits the first one with wboot to check that the
* estat aggregations stay analytic, and that e(V) keeps the analytic
* variance.
*
* Run from tests/reference_parity/_fixtures (csdid 1.81, csdid_estat 1.6):
*   stata-mp -q -b do _generate_cs_parity_Stata.do
* Writes cs_parity_Stata.csv (one row per reported number).
version 18
clear all
set type double
set more off

tempname H
postfile `H' str24 panel byte(covariates notyet cluster wboot) str10 block ///
    str24 name double(b se ll ul df crit n) using cs_parity_Stata_tmp.dta, replace

capture program drop post_rtable
program define post_rtable
    args H panel cov ny cl wb block n
    tempname T
    matrix `T' = r(table)
    local names : colnames `T'
    local k = 1
    foreach nm of local names {
        post `H' ("`panel'") (`cov') (`ny') (`cl') (`wb') ("`block'") ("`nm'") ///
            (`T'[1,`k']) (`T'[2,`k']) (`T'[5,`k']) (`T'[6,`k']) (`T'[7,`k']) ///
            (`T'[8,`k']) (`n')
        local ++k
    }
end

foreach panel in cs_parity_mpdta cs_parity_always {
foreach cov in 0 1 {
foreach ny in 0 1 {
foreach cl in 0 1 {
    import delimited using "`panel'.csv", clear asdouble
    local xv = cond(`cov', "lpop", "")
    local nyo = cond(`ny', "notyet", "")
    local clo = cond(`cl', "cluster(state)", "")
    qui csdid lemp `xv', ivar(countyreal) time(year) gvar(first_treat) `nyo' `clo'
    local n = e(N)
    matrix b = e(b)
    matrix V = e(V)
    local eqs : coleq b
    local cols : colnames b
    local k = 0
    foreach c of local cols {
        local ++k
        local eq : word `k' of `eqs'
        if substr("`eq'", 1, 1) != "g" continue
        post `H' ("`panel'") (`cov') (`ny') (`cl') (0) ("attgt") ("`eq':`c'") ///
            (b[1,`k']) (sqrt(V[`k',`k'])) (.) (.) (.) (.) (`n')
    }
    qui estat pretrend
    post `H' ("`panel'") (`cov') (`ny') (`cl') (0) ("pretrend") ("chi2") ///
        (r(chi2)) (.) (.) (.) (r(df)) (r(pchi2)) (`n')
    foreach a in simple group calendar event {
        qui estat `a'
        post_rtable `H' `panel' `cov' `ny' `cl' 0 `a' `n'
    }
}
}
}
}

* wboot: e(V) and the estat aggregations
import delimited using "cs_parity_mpdta.csv", clear asdouble
qui csdid lemp, ivar(countyreal) time(year) gvar(first_treat) wboot rseed(20261021)
local n = e(N)
matrix b = e(b)
matrix V = e(V)
local eqs : coleq b
local cols : colnames b
local k = 0
foreach c of local cols {
    local ++k
    local eq : word `k' of `eqs'
    if substr("`eq'", 1, 1) != "g" continue
    post `H' ("cs_parity_mpdta") (0) (0) (0) (1) ("attgt") ("`eq':`c'") ///
        (b[1,`k']) (sqrt(V[`k',`k'])) (.) (.) (.) (.) (`n')
}
foreach a in simple group calendar event {
    qui estat `a'
    post_rtable `H' cs_parity_mpdta 0 0 0 1 `a' `n'
}
postclose `H'
use cs_parity_Stata_tmp.dta, clear
format b se ll ul crit %24.17e
export delimited using cs_parity_Stata.csv, replace datafmt
erase cs_parity_Stata_tmp.dta
