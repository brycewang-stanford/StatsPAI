* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_ivprobit_ivtobit_stata_parity.py
*
* Requires: Stata 18 (ivprobit and ivtobit are official; no ado to install).
* Run:      stata -b do _generate_ivprobit_ivtobit_stata.do   (from this directory)
*
* What is recorded
* ----------------
* For a probit and a tobit outcome with endogenous continuous regressors:
*   - the maximum likelihood estimator with the default (oim), robust and
*     cluster covariance matrices, with one and with two endogenous regressors;
*   - Newey's (1987) minimum chi-squared two-step estimator;
*   - the Wald test of exogeneity each of them reports.
* Every block stores the full e(b) with Stata's own column names and the
* standard errors, so the test does not depend on which names Stata gives
* the ancillary parameters.
*
* Data is generated and exported HERE so both sides read the same bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20261004
set obs 1500

gen double x1 = rnormal()
gen byte   x2 = runiformint(0, 1)
gen double z1 = rnormal()
gen double z2 = rnormal()
gen double z3 = rnormal()
gen int    clust = mod(_n, 60) + 1

gen double v1 = rnormal()
gen double v2 = rnormal()
gen double u  = 0.5*v1 - 0.3*v2 + sqrt(1 - 0.25 - 0.09)*rnormal()
gen double w1 = 0.2 + 0.6*z1 + 0.4*z2 + 0.3*x1 + 0.9*v1
gen double w2 = -0.1 + 0.5*z3 - 0.3*z2 + 0.2*x2 + 0.4*v1 + 0.7*v2
gen double ystar = 0.3 + 0.8*w1 - 0.5*w2 + 0.4*x1 - 0.6*x2 + u
gen byte   d  = ystar > 0
gen double yc = max(0, 1.5*ystar + 0.5)
gen double yb = min(max(-1, ystar), 2.5)
drop v1 v2 u ystar

format x1 z1 z2 z3 w1 w2 yc yb %21.16e
export delimited d yc yb w1 w2 x1 x2 z1 z2 z3 clust ///
    using "ivprobit_ivtobit_data.csv", replace datafmt

capture program drop wr
program define wr
    args fh tag last
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    file write `fh' `"  "`tag'": {"' _n
    file write `fh' `"    "names": ["'
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        file write `fh' `""`nm'""'
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "]," _n `"    "b": ["'
    forvalues j = 1/`k' {
        file write `fh' %21.16e (`b'[1, `j'])
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "]," _n `"    "se": ["'
    forvalues j = 1/`k' {
        file write `fh' %21.16e (sqrt(`V'[`j', `j']))
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "]," _n
    if "`e(ll)'" != "" file write `fh' `"    "ll": "' %21.16e (e(ll)) "," _n
    if "`e(N_clust)'" != "" file write `fh' `"    "n_clusters": "' %21.16e (e(N_clust)) "," _n
    file write `fh' `"    "chi2_exog": "' %21.16e (e(chi2_exog)) "," _n
    file write `fh' `"    "p_exog": "' %21.16e (e(p_exog)) "," _n
    file write `fh' `"    "chi2": "' %21.16e (e(chi2)) "," _n
    file write `fh' `"    "N": "' %21.16e (e(N)) _n
    file write `fh' "  }," _n
end

* Stata's ml stops at nrtolerance(1e-5) by default, which leaves the
* coefficients a few 1e-6 short of the optimum. Tightening it here is what
* lets the comparison be held to 1e-6 instead of to the stopping rule.
local tight "tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-13)"

tempname fh
file open `fh' using "ivprobit_ivtobit_stata.json", write replace text
file write `fh' "{" _n

* ---- ivprobit ------------------------------------------------------------
quietly ivprobit d x1 x2 (w1 = z1 z2), `tight'
wr `fh' "ivprobit_mle"
quietly ivprobit d x1 x2 (w1 = z1 z2), vce(robust) `tight'
wr `fh' "ivprobit_mle_robust"
quietly ivprobit d x1 x2 (w1 = z1 z2), vce(cluster clust) `tight'
wr `fh' "ivprobit_mle_cluster"
quietly ivprobit d x1 x2 (w1 = z1 z2), twostep
wr `fh' "ivprobit_twostep"
quietly ivprobit d x1 x2 (w1 w2 = z1 z2 z3), `tight'
wr `fh' "ivprobit_mle_2endog"
quietly ivprobit d x1 x2 (w1 w2 = z1 z2 z3), twostep
wr `fh' "ivprobit_twostep_2endog"

* ---- ivtobit -------------------------------------------------------------
quietly ivtobit yc x1 x2 (w1 = z1 z2), ll(0) `tight'
wr `fh' "ivtobit_mle"
quietly ivtobit yc x1 x2 (w1 = z1 z2), ll(0) vce(robust) `tight'
wr `fh' "ivtobit_mle_robust"
quietly ivtobit yc x1 x2 (w1 = z1 z2), ll(0) vce(cluster clust) `tight'
wr `fh' "ivtobit_mle_cluster"
quietly ivtobit yc x1 x2 (w1 = z1 z2), ll(0) twostep
wr `fh' "ivtobit_twostep"
quietly ivtobit yc x1 x2 (w1 w2 = z1 z2 z3), ll(0) `tight'
wr `fh' "ivtobit_mle_2endog"
quietly ivtobit yc x1 x2 (w1 w2 = z1 z2 z3), ll(0) twostep
wr `fh' "ivtobit_twostep_2endog"
quietly ivtobit yb x1 x2 (w1 = z1 z2), ll(-1) ul(2.5) `tight'
wr `fh' "ivtobit_mle_twolimit"
quietly ivtobit yb x1 x2 (w1 = z1 z2), ll(-1) ul(2.5) twostep
wr `fh' "ivtobit_twostep_twolimit"

file write `fh' `"  "_meta": {"' _n
file write `fh' `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write `fh' `"    "flavor": "' `"""' "`c(flavor)' `c(edition_real)'" `"""' `","' _n
file write `fh' `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write `fh' `"  }"' _n
file write `fh' "}" _n
file close `fh'
display "wrote ivprobit_ivtobit_stata.json and ivprobit_ivtobit_data.csv"
