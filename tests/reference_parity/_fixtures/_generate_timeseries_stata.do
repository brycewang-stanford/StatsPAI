* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_timeseries_R_parity.py
* (Stata-side block). Stata 18 built-ins (vecrank, var, vargranger, irf,
* estat sbcusum, estat sbsingle, newey, arch, xtunitroot) plus the SSC
* packages egranger (Schaffer) and itsa (Linden), installed into a private
* ado directory -- never PLUS:
*
*   _ado_timeseries/   (gitignored; created by this script)
*
* Run _generate_timeseries_data.py first, then from this directory:
*   stata-mp -b do _generate_timeseries_stata.do
* Writes timeseries_Stata.json (numbers as %23.16e).
* ---------------------------------------------------------------------------
version 18
set more off
set type double

local ADO "`c(pwd)'/_ado_timeseries"
capture mkdir "`ADO'"
sysdir set PLUS "`ADO'"
net set ado "`ADO'"
adopath ++ "`ADO'"
capture which egranger
if _rc ssc install egranger
capture which itsa
if _rc ssc install itsa

tempname fh
file open `fh' using "timeseries_Stata.json", write replace text
file write `fh' "{" _n

capture program drop jnum
program define jnum
    args fh key val last
    local sep = cond("`last'" == "last", "", ",")
    if missing(`val') file write `fh' `"  "`key'": null`sep'"' _n
    else file write `fh' `"  "`key'": "' %23.16e (`val') "`sep'" _n
end

capture program drop jmat
program define jmat
    * writes a matrix row-major as a flat JSON array
    args fh key M last
    local sep = cond("`last'" == "last", "", ",")
    local r = rowsof(`M')
    local c = colsof(`M')
    file write `fh' `"  "`key'": ["'
    forvalues i = 1/`r' {
        forvalues j = 1/`c' {
            local comma = cond(`i' == `r' & `j' == `c', "", ", ")
            local v = `M'[`i', `j']
            if missing(`v') file write `fh' "null`comma'"
            else file write `fh' %23.16e (`v') "`comma'"
        }
    }
    file write `fh' "]`sep'" _n
end

file write `fh' `"  "stata_version": "`c(stata_version)'","' _n
quietly which egranger
file write `fh' `"  "egranger_version": "1.0.6","' _n
file write `fh' `"  "itsa_version": "1.0.0","' _n

* ---- Johansen: vecrank ------------------------------------------------------
import delimited using "ts_coint.csv", clear asdouble
gen t = _n
tsset t
foreach tr in none rconstant constant rtrend trend {
    foreach p in 2 3 {
        quietly vecrank y1 y2 y3, lags(`p') trend(`tr') max
        matrix T = e(trace)
        matrix M = e(max)
        matrix L = e(lambda)
        jmat `fh' "vecrank_`tr'_p`p'_trace" T
        jmat `fh' "vecrank_`tr'_p`p'_max" M
        jmat `fh' "vecrank_`tr'_p`p'_lambda" L
        jnum `fh' "vecrank_`tr'_p`p'_N" e(N)
    }
}
_vecgetcv trace
matrix C = r(cv95)
jmat `fh' "vecgetcv_trace_95" C
matrix C = r(cv99)
jmat `fh' "vecgetcv_trace_99" C
_vecgetcv max
matrix C = r(cv95)
jmat `fh' "vecgetcv_max_95" C
matrix C = r(cv99)
jmat `fh' "vecgetcv_max_99" C

* ---- Engle-Granger: egranger ----------------------------------------------
foreach spec in "y1 y2|0|" "y1 y2|2|" "y1 y2 y3|1|" "w y1|1|" "y1 y2|1|trend" "y1 y2|1|qtrend" {
    tokenize "`spec'", parse("|")
    local vv `1'
    local L `3'
    local opt `5'
    if "`opt'" == "|" local opt
    quietly egranger `vv', lags(`L') `opt'
    local key = subinstr("`vv'", " ", "_", .) + "_L`L'" + cond("`opt'" != "", "_`opt'", "")
    jnum `fh' "eg_`key'_Zt" e(Zt)
    jnum `fh' "eg_`key'_cv1" e(cv1)
    jnum `fh' "eg_`key'_cv5" e(cv5)
    jnum `fh' "eg_`key'_cv10" e(cv10)
    jnum `fh' "eg_`key'_N1" e(N1)
    capture drop _egresid
}

* ---- VAR: var / vargranger / irf ------------------------------------------
import delimited using "ts_var.csv", clear asdouble
gen t = _n
tsset t
quietly var gdp infl rate, lags(1/2)
quietly vargranger
matrix G = r(gstats)
jmat `fh' "vargranger_chi2" G
quietly var gdp infl rate, lags(1/2) small
quietly vargranger
matrix G = r(gstats)
jmat `fh' "vargranger_small" G
quietly var gdp infl rate, lags(1/2) small dfk
quietly vargranger
matrix G = r(gstats)
jmat `fh' "vargranger_small_dfk" G

quietly var gdp infl rate, lags(1/2)
capture erase "_ts_irf.irf"
quietly irf create m1, step(8) set(_ts_irf, replace)
quietly var gdp infl rate, lags(1/2) dfk
quietly irf create m2, step(8)
preserve
use "_ts_irf.irf", clear
foreach m in m1 m2 {
    foreach r in gdp infl rate {
        foreach s in oirf irf coirf {
            quietly mkmat `s' if irfname == "`m'" & impulse == "infl" & response == "`r'", matrix(X)
            jmat `fh' "`s'_`m'_infl_`r'" X
        }
    }
}
restore
capture erase "_ts_irf.irf"

* ---- Bayesian VAR: original Minnesota prior with fixed covariance ----------
* Gibbs draws are exact for this model (efficiency ~ 1); the posterior means
* carry Monte-Carlo error MCSE, recorded alongside.
foreach spec in "arcov|" "varcov|" "arcov|selftight(0.2) crosstight(0.3) lagdecay(2) exogtight(10)" {
    tokenize "`spec'", parse("|")
    local cov `1'
    local extra `3'
    local tag = cond("`extra'" != "", "`cov'_custom", "`cov'")
    quietly bayes, minnfixedcovprior(`cov' `extra') rseed(20260918) ///
        mcmcsize(100000) burnin(1000): var gdp infl rate, lags(1/2)
    matrix C = e(`cov')
    jmat `fh' "bvar_`tag'_sigma0" C
    quietly bayesstats summary
    matrix M = r(summary)
    jmat `fh' "bvar_`tag'_summary" M
}

* ---- structural break: estat sbcusum / sbsingle ----------------------------
import delimited using "ts_break.csv", clear asdouble
tsset t
quietly regress y x
quietly estat sbcusum, generate(_cs)
jnum `fh' "sbcusum_y_x" r(cusum)
matrix C = r(cvalues)
jmat `fh' "sbcusum_cv" C
mkmat _cs if !missing(_cs), matrix(X)
jmat `fh' "sbcusum_path" X
drop _cs
quietly estat sbsingle, trim(15) swald
jnum `fh' "sbsingle_y_x_swald" r(chi2_swald)
local bd = real("`r(breakdate)'")
local lt = real("`r(ltrim)'")
local rt = real("`r(rtrim)'")
jnum `fh' "sbsingle_y_x_breakdate" `bd'
jnum `fh' "sbsingle_y_x_ltrim" `lt'
jnum `fh' "sbsingle_y_x_rtrim" `rt'
quietly regress y
quietly estat sbsingle, trim(15) swald
jnum `fh' "sbsingle_y_swald" r(chi2_swald)
local bd = real("`r(breakdate)'")
jnum `fh' "sbsingle_y_breakdate" `bd'

* ---- ITS: newey and itsa ----------------------------------------------------
import delimited using "ts_its.csv", clear asdouble
gen D = (_n - 1 >= 40)
gen tpost = cond(_n - 1 >= 40, _n - 1 - 40, 0)
tsset month
quietly newey y month D tpost, lag(4)
matrix b = e(b)
matrix V = e(V)
jmat `fh' "newey_b" b
jmat `fh' "newey_V" V
quietly itsa y, single trperiod(41) lag(4)
matrix b = e(b)
matrix V = e(V)
jmat `fh' "itsa_b" b
jmat `fh' "itsa_V" V

* ---- GARCH(1,1): arch -------------------------------------------------------
import delimited using "ts_garch.csv", clear asdouble
gen t = _n
tsset t
foreach v in oim opg robust {
    quietly arch r, arch(1) garch(1) vce(`v') ///
        tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-12)
    matrix b = e(b)
    matrix V = e(V)
    jmat `fh' "arch_`v'_b" b
    jmat `fh' "arch_`v'_V" V
    jnum `fh' "arch_`v'_ll" e(ll)
}

* ---- xtunitroot -------------------------------------------------------------
import delimited using "ts_panel.csv", clear asdouble
xtset id time
foreach tr in "" "trend" {
    local k = cond("`tr'" == "", "c", "ct")
    quietly xtunitroot llc y, lags(1) `tr'
    foreach s in tds td delta se_delta sbar Var_ep mu_adj sig_adj ttilde p_tds hac_lagm {
        jnum `fh' "xtur_llc_`k'_`s'" r(`s')
    }
    quietly xtunitroot ips y, lags(1) `tr'
    jnum `fh' "xtur_ips_`k'_wtbar" r(wtbar)
    jnum `fh' "xtur_ips_`k'_p" r(p_wtbar)
    quietly xtunitroot fisher y, dfuller lags(1) `tr'
    foreach s in P p_P Z p_Z L p_L Pm {
        jnum `fh' "xtur_fisher_`k'_`s'" r(`s')
    }
    quietly xtunitroot hadri y, `tr'
    jnum `fh' "xtur_hadri_`k'_z" r(z)
    quietly xtunitroot hadri y, robust `tr'
    jnum `fh' "xtur_hadri_robust_`k'_z" r(z)
}
quietly xtunitroot llc y, lags(1) noconstant
jnum `fh' "xtur_llc_n_tds" r(tds)
jnum `fh' "xtur_llc_n_td" r(td) last

file write `fh' "}" _n
file close `fh'
