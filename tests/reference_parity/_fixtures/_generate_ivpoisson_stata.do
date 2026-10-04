* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_ivpoisson_stata_parity.py
*
* Requires: Stata 18 (ivpoisson is official; no ado to install).
* Run:      stata -b do _generate_ivpoisson_stata.do   (from this directory)
*
* Records `ivpoisson gmm` with additive (the default) and multiplicative
* errors, one-step, two-step and iterated, over-identified and exactly
* identified, robust and clustered. Data is generated and exported HERE so
* both sides read the same bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20261005
set obs 2000

gen double x1 = rnormal()
gen byte   x2 = runiformint(0, 1)
gen double z1 = rnormal()
gen double z2 = rnormal()
gen int    clust = mod(_n, 80) + 1
gen double v  = rnormal()
gen double w  = 0.5*z1 + 0.4*z2 + 0.3*x1 + 0.8*v
gen double eta = exp(0.5*v - 0.125 + 0.3*rnormal() - 0.045)
gen long   y  = rpoisson(exp(0.2 + 0.4*w + 0.3*x1 - 0.5*x2) * eta)
drop v eta

format x1 z1 z2 w %21.16e
export delimited y w x1 x2 z1 z2 clust using "ivpoisson_data.csv", replace datafmt

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
    if "`e(N_clust)'" != "" file write `fh' `"    "n_clusters": "' %21.16e (e(N_clust)) "," _n
    file write `fh' `"    "J": "' %21.16e (e(J)) "," _n
    file write `fh' `"    "J_df": "' %21.16e (e(J_df)) "," _n
    file write `fh' `"    "N": "' %21.16e (e(N)) _n
    file write `fh' "  }," _n
end


tempname fh
file open `fh' using "ivpoisson_stata.json", write replace text
file write `fh' "{" _n

quietly ivpoisson gmm y x1 x2 (w = z1 z2)
wr `fh' "additive_twostep"
quietly ivpoisson gmm y x1 x2 (w = z1 z2), onestep
wr `fh' "additive_onestep"
quietly ivpoisson gmm y x1 x2 (w = z1 z2), igmm
wr `fh' "additive_igmm"
quietly ivpoisson gmm y x1 x2 (w = z1 z2), multiplicative
wr `fh' "multiplicative_twostep"
quietly ivpoisson gmm y x1 x2 (w = z1 z2), multiplicative onestep
wr `fh' "multiplicative_onestep"
quietly ivpoisson gmm y x1 x2 (w = z1), multiplicative
wr `fh' "multiplicative_justid"
quietly ivpoisson gmm y x1 x2 (w = z1 z2), multiplicative vce(cluster clust) wmatrix(cluster clust)
wr `fh' "multiplicative_cluster"
quietly ivpoisson gmm y x1 x2 (w = z1 z2), vce(cluster clust) wmatrix(cluster clust)
wr `fh' "additive_cluster"
quietly ivpoisson gmm y x1 x2 (w = z1 z2), vce(unadjusted) wmatrix(unadjusted)
wr `fh' "additive_unadjusted"
* wmatrix() follows vce() when it is not given: this block must equal
* additive_cluster. The next one sets them apart on purpose.
quietly ivpoisson gmm y x1 x2 (w = z1 z2), vce(cluster clust)
wr `fh' "additive_vcecluster_only"
quietly ivpoisson gmm y x1 x2 (w = z1 z2), vce(cluster clust) wmatrix(robust)
wr `fh' "additive_wrobust_vcecluster"
quietly ivpoisson gmm y x1 x2 (w = z1 z2), multiplicative igmm
wr `fh' "multiplicative_igmm"

file write `fh' `"  "_meta": {"' _n
file write `fh' `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write `fh' `"    "flavor": "' `"""' "`c(flavor)' `c(edition_real)'" `"""' `","' _n
file write `fh' `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write `fh' `"  }"' _n
file write `fh' "}" _n
file close `fh'
display "wrote ivpoisson_stata.json and ivpoisson_data.csv"
