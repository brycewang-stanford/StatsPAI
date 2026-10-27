* Reference for sp.fmm: Stata 18 fmm (finite mixture of regress / logit /
* poisson), with and without lcprob(), OIM / robust / cluster covariance and
* frequency / sampling weights, on the simulated CSVs written by
* _generate_fmm_data.py.
*
* Each fit starts from the simulated class (startvalues(classid cls)), whose
* numbering is StatsPAI's default class order, so Stata's labels are ours.
* The convergence criteria are tightened (nrtolerance 1e-13) so that the
* reported point is the maximum to rounding, not to Stata's default stop.
*
* Dumped per case: e(b), e(V), e(ll), e(N), e(rank), AIC / BIC (estat ic),
* estat lcprob and estat lcmean (margin, se, ci), and for the first ten
* observations the posterior class probabilities (classposteriorpr), the
* prior class probabilities (classpr) and the class-specific means (mu).
* For the logit cases also ll_manual: the mixture log-likelihood at e(b)
* computed by hand from predict's eta and classpr, sum ln(sum_k pr_k *
* invlogit(+-eta_k)). fmm's own e(ll) for logit differs from it on
* observations with an extreme linear predictor (see the test).
*
* Run from the repository root:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_fmm_Stata.do
version 18
set type double
clear all
set more off

local DIR "tests/reference_parity/_fixtures"
local TOL "nrtolerance(1e-13) tolerance(1e-10) ltolerance(1e-13)"

tempname fh
file open `fh' using "`DIR'/fmm_Stata.json", write replace text
file write `fh' "{" _n

capture program drop _wmat
program define _wmat
    * write matrix M as {"names": [...], "values": [[...], ...]}
    args fh M
    local cn : colfullnames `M'
    file write `fh' `"{"names": ["'
    local j = 0
    foreach c of local cn {
        local ++j
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`c'""'
    }
    file write `fh' "], " `""values": ["'
    forvalues i = 1/`=rowsof(`M')' {
        if `i' > 1 file write `fh' ", "
        file write `fh' "["
        forvalues j = 1/`=colsof(`M')' {
            if `j' > 1 file write `fh' ", "
            local v = `M'[`i', `j']
            if missing(`v') file write `fh' "null"
            else file write `fh' %24.17e (`v')
        }
        file write `fh' "]"
    }
    file write `fh' "]}"
end

capture program drop _case
program define _case
    * _case fh key k "lcprob/vce/weight options" "model spec" last
    args fh key k opts model last
    local DIR "tests/reference_parity/_fixtures"
    local TOL "nrtolerance(1e-13) tolerance(1e-10) ltolerance(1e-13)"
    local wgt ""
    local o `"`opts'"'
    if strpos(`"`o'"', "FW") {
        local wgt "[fw=fw]"
        local o = subinstr(`"`o'"', "FW", "", .)
    }
    if strpos(`"`o'"', "PW") {
        local wgt "[pw=fw]"
        local o = subinstr(`"`o'"', "PW", "", .)
    }
    fmm `k' `wgt', `o' startvalues(classid cls) `TOL': `model'
    assert e(converged) == 1
    tempname B V IC P M
    matrix `B' = e(b)
    matrix `V' = e(V)
    quietly estat ic
    matrix `IC' = r(S)
    quietly estat lcprob
    matrix `P' = r(table)
    quietly estat lcmean
    matrix `M' = r(table)
    file write `fh' `"  "`key'": {"ll": "' %24.17e (e(ll)) `", "N": "' (e(N)) `", "rank": "' (e(rank))
    file write `fh' `", "aic": "' %24.17e (`IC'[1, 5]) `", "bic": "' %24.17e (`IC'[1, 6])
    file write `fh' `", "b": "'
    _wmat `fh' `B'
    file write `fh' `", "V": "'
    _wmat `fh' `V'
    file write `fh' `", "lcprob": "'
    _wmat `fh' `P'
    file write `fh' `", "lcmean": "'
    _wmat `fh' `M'
    capture drop _cp* _pr* _mu*
    quietly predict double _cp*, classposteriorpr
    quietly predict double _pr*, classpr
    quietly predict double _mu*, mu
    tempname H
    mkmat _cp* _pr* _mu* in 1/10, matrix(`H')
    file write `fh' `", "head": "'
    _wmat `fh' `H'
    if word(`"`model'"', 1) == "logit" {
        capture drop _eta*
        capture drop _lm
        quietly predict double _eta*, eta
        quietly generate double _lm = 0
        forvalues j = 1/`k' {
            quietly replace _lm = _lm + _pr`j' * cond(y == 1, invlogit(_eta`j'), invlogit(-_eta`j'))
        }
        quietly replace _lm = ln(_lm)
        quietly summarize _lm
        file write `fh' `", "ll_manual": "' %24.17e (r(sum))
    }
    file write `fh' "}"
    if "`last'" == "" file write `fh' ","
    file write `fh' _n
end

import delimited "`DIR'/fmm_gauss2.csv", clear asdouble
_case `fh' gauss2_plain 2 "" "regress y x1 x2"
_case `fh' gauss2_lcprob 2 "lcprob(z)" "regress y x1 x2"
_case `fh' gauss2_lcprob_robust 2 "lcprob(z) vce(robust)" "regress y x1 x2"
_case `fh' gauss2_cluster 2 "vce(cluster g)" "regress y x1 x2"
_case `fh' gauss2_lcprob_fw 2 "lcprob(z) FW" "regress y x1 x2"
_case `fh' gauss2_lcprob_pw 2 "lcprob(z) PW" "regress y x1 x2"

import delimited "`DIR'/fmm_gauss3.csv", clear asdouble
_case `fh' gauss3_plain 3 "" "regress y x1"
_case `fh' gauss3_lcprob 3 "lcprob(z)" "regress y x1"

import delimited "`DIR'/fmm_logit2.csv", clear asdouble
_case `fh' logit2_plain 2 "" "logit y x1 x2"
_case `fh' logit2_lcprob 2 "lcprob(z)" "logit y x1 x2"

import delimited "`DIR'/fmm_pois2.csv", clear asdouble
_case `fh' pois2_plain 2 "" "poisson y x1 x2"
_case `fh' pois2_lcprob 2 "lcprob(z)" "poisson y x1 x2"
_case `fh' pois2_lcprob_cluster 2 "lcprob(z) vce(cluster g)" "poisson y x1 x2" last

file write `fh' "}" _n
file close `fh'
