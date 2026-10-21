* Stata csdid reference for sp.callaway_santanna(parity="stata") / sp.aggte
* on a repeated cross-section and on an unbalanced panel.
*
* csdid / csdid_estat are used as a black box: only their outputs are read.
* Data (written by _generate_cs_parity_rc_unbal_data.py):
*   cs_parity_rcs.csv    mpdta county-years sampled per year, one id per row
*                        (rid); csdid runs without ivar(), i.e. as repeated
*                        cross-sections
*   cs_parity_unbal.csv  mpdta with 12% of county-years deleted; csdid runs
*                        with ivar(countyreal)
* Each case runs csdid with its defaults (short pre-treatment gaps,
* method(dripw), analytic standard errors), with or without lpop, never- or
* not-yet-treated comparisons, unit or state clusters, and records every
* ATT(g, t) with its SE (e(b), e(V)), estat pretrend, and estat simple /
* group / calendar / event (r(table): b, se, ll, ul, df, crit). Each case
* runs unweighted and with the sampling weight w as [iw=w] (csdid refuses
* pweights and aweights; its help file says iweights are treated as
* pweights internally). cs_parity_mpdta is the balanced panel with each
* county's w taken from cs_parity_unbal.csv, weighted cases only (the
* unweighted balanced panel is in test_cs_parity_switch.py).
*
* Run from tests/reference_parity/_fixtures (csdid 1.81, csdid_estat 1.6):
*   stata-mp -q -b do _generate_cs_parity_rc_unbal_Stata.do
* Writes cs_parity_rc_unbal_Stata.csv (one row per reported number).
version 18
clear all
set type double
set more off

import delimited using "cs_parity_unbal.csv", clear asdouble
keep countyreal w
duplicates drop
isid countyreal
tempfile county_w
save `county_w'

tempname H
postfile `H' str24 panel byte(covariates notyet cluster weighted) str10 block ///
    str24 name double(b se ll ul df crit n) using cs_parity_rcu_tmp.dta, replace

capture program drop post_rtable
program define post_rtable
    args H panel cov ny cl wt block n
    tempname T
    matrix `T' = r(table)
    local names : colnames `T'
    local k = 1
    foreach nm of local names {
        post `H' ("`panel'") (`cov') (`ny') (`cl') (`wt') ("`block'") ("`nm'") ///
            (`T'[1,`k']) (`T'[2,`k']) (`T'[5,`k']) (`T'[6,`k']) (`T'[7,`k']) ///
            (`T'[8,`k']) (`n')
        local ++k
    }
end

foreach panel in cs_parity_rcs cs_parity_unbal cs_parity_mpdta {
foreach cov in 0 1 {
foreach ny in 0 1 {
foreach cl in 0 1 {
foreach wt in 0 1 {
    if "`panel'" == "cs_parity_mpdta" & !`wt' continue
    import delimited using "`panel'.csv", clear asdouble
    if "`panel'" == "cs_parity_mpdta" {
        merge m:1 countyreal using `county_w', assert(match) nogenerate
    }
    local xv = cond(`cov', "lpop", "")
    local nyo = cond(`ny', "notyet", "")
    local clo = cond(`cl', "cluster(state)", "")
    local ivo = cond("`panel'" != "cs_parity_rcs", "ivar(countyreal)", "")
    local wo = cond(`wt', "[iw=w]", "")
    qui csdid lemp `xv' `wo', `ivo' time(year) gvar(first_treat) `nyo' `clo'
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
        post `H' ("`panel'") (`cov') (`ny') (`cl') (`wt') ("attgt") ("`eq':`c'") ///
            (b[1,`k']) (sqrt(V[`k',`k'])) (.) (.) (.) (.) (`n')
    }
    qui estat pretrend
    post `H' ("`panel'") (`cov') (`ny') (`cl') (`wt') ("pretrend") ("chi2") ///
        (r(chi2)) (.) (.) (.) (r(df)) (r(pchi2)) (`n')
    foreach a in simple group calendar event {
        qui estat `a'
        post_rtable `H' `panel' `cov' `ny' `cl' `wt' `a' `n'
    }
}
}
}
}
}
postclose `H'
use cs_parity_rcu_tmp.dta, clear
format b se ll ul crit %24.17e
export delimited using cs_parity_rc_unbal_Stata.csv, replace datafmt
erase cs_parity_rcu_tmp.dta
