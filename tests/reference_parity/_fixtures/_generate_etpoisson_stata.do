* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_etpoisson_stata_parity.py
*
* Requires: Stata 18 (etpoisson is official).
* Run:      stata -b do _generate_etpoisson_stata.do   (from this directory)
*
* Poisson regression with a binary endogenous treatment: maximum likelihood
* with Gauss-Hermite quadrature over the outcome error. Recorded with the
* default number of quadrature points and with 64, under vce(oim),
* vce(robust) and vce(cluster), and the average treatment effect from
* margins. Data is generated and exported HERE so both sides read the same
* bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20261007
set obs 2500

gen double x1 = rnormal()
gen byte   x2 = runiformint(0, 1)
gen double z1 = rnormal()
gen int    clust = mod(_n, 100) + 1
gen double u   = rnormal()
gen double eps = 0.5*(0.6*u + sqrt(1 - 0.36)*rnormal())
gen byte   d   = (-0.2 + 0.8*z1 + 0.3*x1 + u) > 0
gen long   y   = rpoisson(exp(0.3 + 0.4*x1 - 0.3*x2 + 0.5*d + eps))
drop u eps

format x1 z1 %21.16e
export delimited y d x1 x2 z1 clust using "etpoisson_data.csv", replace datafmt

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
    if "`e(chi2_c)'" != "" file write `fh' `"    "chi2_c": "' %21.16e (e(chi2_c)) "," _n
    if "`e(n_quad)'" != "" file write `fh' `"    "n_quad": "' %21.16e (e(n_quad)) "," _n
    file write `fh' `"    "N": "' %21.16e (e(N)) _n
    file write `fh' "  }," _n
end


local tight "tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-13)"

tempname fh
file open `fh' using "etpoisson_stata.json", write replace text
file write `fh' "{" _n

quietly etpoisson y x1 x2, treat(d = z1 x1) `tight'
wr `fh' "default"
quietly etpoisson y x1 x2, treat(d = z1 x1) intpoints(64) `tight'
wr `fh' "q64"
quietly etpoisson y x1 x2, treat(d = z1 x1) intpoints(64) vce(robust) `tight'
wr `fh' "q64_robust"
quietly etpoisson y x1 x2, treat(d = z1 x1) intpoints(64) vce(cluster clust) `tight'
wr `fh' "q64_cluster"

quietly etpoisson y x1 x2, treat(d = z1 x1) intpoints(64) `tight'
quietly margins r.d
matrix T = r(table)
file write `fh' `"  "ate_q64": {"b": "' %21.16e (T[1,1]) `", "se": "' %21.16e (T[2,1]) "}," _n
quietly margins d
matrix T = r(table)
file write `fh' `"  "pomeans_q64": {"b0": "' %21.16e (T[1,1]) `", "b1": "' %21.16e (T[1,2]) "}," _n

file write `fh' `"  "_meta": {"' _n
file write `fh' `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write `fh' `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write `fh' `"  }"' _n
file write `fh' "}" _n
file close `fh'
