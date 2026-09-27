* Stata reference for sp.sdid(treatment=...) -- tests/reference_parity/test_sdid_staggered_parity.py
*
* Reference: Stata `sdid` 2.0.2 (Pailanir & Clarke, SSC), the implementation
* accompanying Clarke, Pailanir, Athey & Imbens (2024), Stata Journal 24(4).
* Data (author-distributed with that paper, www.damianclarke.net/stata/):
*   prop99_example.dta -- Proposition 99 panel, block design, one treated unit
*   quota_example.dta  -- gender quotas, staggered adoption (7 cohorts)
* Run from this directory: stata -b do _generate_sdid_stata.do
* Numbers are written %24.17e; do not hand-edit sdid_stata.json.
*
* Precision: sdid.ado stores e(ATT) and e(se) through strofreal(), i.e. with
* about 7 significant digits. e(tau) (per-cohort tau and SE; adoption year in
* its last column) is a full double-precision matrix, so the tests rebuild the
* ATT from it with the eq. (7) weights -- treated units x post periods, written
* below as *_cohort_size -- and check the 7-digit e(ATT) / e(se) at rel 1e-6.
version 18
clear all
set more off
tempname fh
file open `fh' using "sdid_stata.json", write replace
file write `fh' "{" _n

program define _tau_json
    * write e(tau) as {"year": [tau, se?], ...}
    args fh key
    matrix T = e(tau)
    local c = colsof(T)
    file write `fh' `"  "`key'": {"'
    forvalues r = 1/`=rowsof(T)' {
        if `r' > 1 file write `fh' ", "
        local yr : display %9.0f T[`r', `c']
        local yr = strtrim("`yr'")
        file write `fh' `"""' "`yr'" `"": ["' %24.17e (T[`r', 1])
        if `c' == 3 file write `fh' ", " %24.17e (T[`r', 2])
        file write `fh' "]"
    }
    file write `fh' "}," _n
end

program define _cohort_json
    * treated units and post periods per adoption year (eq. 7 weights)
    args fh key unit time w
    tempvar first
    qui bys `unit': egen `first' = min(cond(`w' == 1, `time', .))
    qui sum `time'
    local tmax = r(max)
    qui levelsof `first', local(adopt)
    file write `fh' `"  "`key'": {"'
    local k 0
    foreach a of local adopt {
        qui count if `first' == `a' & `time' == `a'
        if `k' > 0 file write `fh' ", "
        file write `fh' `"""' "`a'" `"": ["' (r(N)) ", " (`tmax' - `a' + 1) "]"
        local ++k
    }
    file write `fh' "}," _n
end

* (A) Prop 99: block design, three estimators
use "prop99_example.dta", clear
foreach m in sdid did sc {
    qui sdid packspercapita state year treated, vce(noinference) method(`m')
    _tau_json `fh' prop99_`m'
    file write `fh' `"  "prop99_`m'_eATT": "' %24.17e (e(ATT)) "," _n
}
* which vce() Stata refuses with one treated unit and one cohort
foreach v in placebo bootstrap jackknife {
    capture qui sdid packspercapita state year treated, vce(`v') seed(1) reps(5)
    file write `fh' `"  "prop99_rc_`v'": "' (_rc) "," _n
}

* (B) quota: staggered adoption
use "quota_example.dta", clear
qui sdid womparl country year quota, vce(noinference)
_tau_json `fh' quota_tau
file write `fh' `"  "quota_eATT": "' %24.17e (e(ATT)) "," _n
_cohort_json `fh' quota_cohort_size country year quota
foreach v in placebo bootstrap jackknife {
    capture qui sdid womparl country year quota, vce(`v') seed(1) reps(5)
    file write `fh' `"  "quota_rc_`v'": "' (_rc) "," _n
}

* (C) quota, lngdp sample: projected covariates (beta from never-treated units)
use "quota_example.dta", clear
drop if lngdp == .
qui sdid womparl country year quota, vce(noinference) covariates(lngdp, projected)
_tau_json `fh' cov_tau
file write `fh' `"  "cov_eATT": "' %24.17e (e(ATT)) "," _n
matrix B = e(beta)
file write `fh' `"  "cov_beta": "' %24.17e (B[1, 1]) "," _n
_cohort_json `fh' cov_cohort_size country year quota

* (D) jackknife (deterministic): cohorts 2002 and 2003, two treated units each
use "quota_example.dta", clear
drop if inlist(country, "Algeria", "Kenya", "Samoa", "Swaziland", "Tanzania")
foreach m in sdid did sc {
    qui sdid womparl country year quota, vce(jackknife) method(`m')
    _tau_json `fh' jk_`m'_tau
    file write `fh' `"  "jk_`m'_eATT": "' %24.17e (e(ATT)) "," _n
    file write `fh' `"  "jk_`m'_ese": "' %24.17e (e(se)) "," _n
}
_cohort_json `fh' jk_cohort_size country year quota

* (E) jackknife with projected covariates: beta re-fitted per leave-one-out sample
drop if lngdp == .
qui sdid womparl country year quota, vce(jackknife) covariates(lngdp, projected)
_tau_json `fh' jk_cov_tau
file write `fh' `"  "jk_cov_eATT": "' %24.17e (e(ATT)) "," _n
file write `fh' `"  "jk_cov_ese": "' %24.17e (e(se)) "," _n
_cohort_json `fh' jk_cov_cohort_size country year quota

file write `fh' `"  "_reference": "Stata sdid 2.0.2 (SSC), Stata 18""' _n
file write `fh' "}" _n
file close `fh'
