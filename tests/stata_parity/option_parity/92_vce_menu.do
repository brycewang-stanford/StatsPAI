*! 92_vce_menu.do
*! Golden numbers for tests/reference_parity/test_vce_menu_parity.py
*!
*! The variance-estimator menu of sp.regress / sp.iv on one cross-section
*! (tests/reference_parity/_fixtures/vce_menu_data.csv), for the cells that
*! Stata defines and R does not, or defines differently:
*!   - ivregress 2sls / liml over-identified, with and without small, under
*!     unadjusted, robust and cluster variances;
*!   - ivreg2 fuller(1) (just- and over-identified; with and without small;
*!     classical, robust, cluster);
*!   - ivreg2 two-way clustering cluster(g yr) small, 2SLS / LIML / Fuller;
*!   - acreg Conley spatial HAC (uniform kernel, 100 km) for OLS and 2SLS;
*!   - boottest wild cluster bootstrap with 12 clusters and reps(4096), which
*!     enumerates every Rademacher sign vector: WCR after regress, WRE after
*!     ivreg2. The p-values are then deterministic;
*!   - newey (Newey-West, 4 lags), unweighted and [aw=w];
*!   - two designs whose acreg / ivreg2 covariance has negative variances
*!     (vce_menu_nonpsd_*.csv): those standard errors are missing.
*!
*! Requires: ivreg2 4.1.12, ranktest, acreg 1.1.0, boottest 4.5.3. Stata 18 MP.
*! Run from tests/stata_parity/option_parity/.

version 17
clear all
set more off

* Double precision throughout: without `asdouble`, `import delimited`
* stores the decimals as float and rounds the data in the 8th digit.
set type double
import delimited "../../reference_parity/_fixtures/vce_menu_data.csv", clear asdouble
tsset t

capture program drop _row
program define _row
    args fh label first
    if !`first' file write `fh' "," _n
    file write `fh' `"  ""' "`label'" `"": {"N": "' %14.0f (e(N))
    foreach v in d x1 x2 x3 _cons {
        file write `fh' `", "b_`v'": "' %24.16e (_b[`v']) `", "se_`v'": "' %24.16e (_se[`v'])
    }
    file write `fh' "}"
end

capture program drop _boot
program define _boot
    args fh label
    file write `fh' "," _n
    file write `fh' `"  ""' "`label'" `"": {"p": "' %24.16e (r(p)) `", "t": "' %24.16e (r(t)) `", "reps": "' %14.0f (r(reps)) "}"
end

tempname fh
file open `fh' using "results/92_vce_menu_Stata.json", write replace
file write `fh' "{" _n

* ---------------------------------------------------------------- OLS ----
qui acreg y d x1 x2 x3, spatial latitude(lat) longitude(lon) dist(100)
_row `fh' "ols_conley" 1
* acreg leaves the data sorted by its own keys
sort t
qui newey y d x1 x2 x3, lag(4)
_row `fh' "ols_newey" 0
qui newey y d x1 x2 x3 [aw=w], lag(4)
_row `fh' "ols_newey_aw" 0
qui regress y d x1 x2 x3, vce(cluster g)
_row `fh' "ols_cr1" 0
foreach p in d x1 x2 x3 {
    qui regress y d x1 x2 x3, vce(cluster g)
    qui boottest `p', reps(4096) weighttype(rademacher) nograph
    _boot `fh' "ols_wild_`p'"
}

* --------------------------------------------------------------- 2SLS ----
foreach id in over just {
    local inst = cond("`id'" == "over", "z1 z2", "z1")
    foreach est in 2sls liml {
        foreach sm in small nosmall {
            local s = cond("`sm'" == "small", "small", "")
            qui ivregress `est' y x1 x2 x3 (d = `inst'), `s'
            _row `fh' "ivregress_`id'_`est'_unadjusted_`sm'" 0
            qui ivregress `est' y x1 x2 x3 (d = `inst'), vce(robust) `s'
            _row `fh' "ivregress_`id'_`est'_robust_`sm'" 0
            qui ivregress `est' y x1 x2 x3 (d = `inst'), vce(cluster g) `s'
            _row `fh' "ivregress_`id'_`est'_cluster_`sm'" 0
        }
    }
    * ivreg2: Fuller(1), and LIML for the two-way rows
    foreach sm in small nosmall {
        local s = cond("`sm'" == "small", "small", "")
        qui ivreg2 y x1 x2 x3 (d = `inst'), fuller(1) `s'
        _row `fh' "ivreg2_`id'_fuller_classical_`sm'" 0
        qui ivreg2 y x1 x2 x3 (d = `inst'), fuller(1) robust `s'
        _row `fh' "ivreg2_`id'_fuller_robust_`sm'" 0
        qui ivreg2 y x1 x2 x3 (d = `inst'), fuller(1) cluster(g) `s'
        _row `fh' "ivreg2_`id'_fuller_cluster_`sm'" 0
    }
    qui ivreg2 y x1 x2 x3 (d = `inst'), cluster(g yr) small
    _row `fh' "ivreg2_`id'_2sls_twoway_small" 0
    qui ivreg2 y x1 x2 x3 (d = `inst'), liml cluster(g yr) small
    _row `fh' "ivreg2_`id'_liml_twoway_small" 0
    qui ivreg2 y x1 x2 x3 (d = `inst'), fuller(1) cluster(g yr) small
    _row `fh' "ivreg2_`id'_fuller_twoway_small" 0
    qui acreg y x1 x2 x3 (d = `inst'), spatial latitude(lat) longitude(lon) dist(100)
    _row `fh' "acreg_`id'_conley" 0
    sort t
    * WRE after ivreg2 with one-way clustering: the endogenous coefficient
    qui ivreg2 y x1 x2 x3 (d = `inst'), cluster(g) small
    _row `fh' "ivreg2_`id'_2sls_cluster_small" 0
    qui boottest d, reps(4096) weighttype(rademacher) nograph
    _boot `fh' "iv_wild_`id'_d"
}

* -------------------------------------------- negative variances ----
* Two designs on which the estimator's covariance matrix has negative
* diagonal entries. acreg and ivreg2 report those standard errors as
* missing; record the variances and which standard errors are missing.
import delimited "../../reference_parity/_fixtures/vce_menu_nonpsd_conley.csv", clear asdouble
qui acreg y x i.county (d = z), spatial latitude(lat) longitude(lon) dist(500)
matrix V = e(V)
file write `fh' "," _n `"  "nonpsd_conley": {"b_d": "' %24.16e (_b[d]) ///
    `", "V_d": "' %24.16e (V["d", "d"]) `", "V_x": "' %24.16e (V["x", "x"]) ///
    `", "se_d_missing": "' (missing(_se[d])) `", "se_x": "' %24.16e (_se[x]) "}"
import delimited "../../reference_parity/_fixtures/vce_menu_nonpsd_twoway.csv", clear asdouble
qui ivreg2 y x (d = z), cluster(a b) small
matrix V = e(V)
file write `fh' "," _n `"  "nonpsd_twoway": {"b_d": "' %24.16e (_b[d]) ///
    `", "V_d": "' %24.16e (V["d", "d"]) `", "V_x": "' %24.16e (V["x", "x"]) ///
    `", "V__cons": "' %24.16e (V["_cons", "_cons"]) ///
    `", "se_d_missing": "' (missing(_se[d])) ///
    `", "se__cons_missing": "' (missing(_se[_cons])) ///
    `", "se_x": "' %24.16e (_se[x]) "}"

file write `fh' "," _n
file write `fh' `"  "_meta": {"stata": "' (c(stata_version)) `", "precision": "double", "data": "tests/reference_parity/_fixtures/vce_menu_data.csv"}"' _n
file write `fh' "}" _n
file close `fh'
