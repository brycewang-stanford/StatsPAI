* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_nlogit_parity.py
*
* Requires: Stata 18 (nlogit is official).
* Run:      stata -b do _generate_nlogit_stata.do   (from this directory,
*           after Rscript _generate_nlogit_mlogit.R has written the CSV)
*
* The random-utility-consistent nested logit on the data simulated in R:
* vce(oim), vce(robust), vce(cluster), and the LR test against the
* conditional logit (all dissimilarity parameters equal to 1).
* ---------------------------------------------------------------------------
version 18
clear all
import delimited "nlogit_data.csv", clear asdouble
encode alt, gen(altn)
nlogitgen nest = altn(A: a1 | a2, B: b1 | b2)

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
    file write `fh' `"    "N": "' %21.16e (e(N)) _n
    file write `fh' "  }," _n
end


local tight "tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-13)"

tempname fh
file open `fh' using "nlogit_stata.json", write replace text
file write `fh' "{" _n
quietly nlogit chosen x1 x2 || nest: || altn:, case(id) `tight'
wr `fh' "separate"
quietly nlogit chosen x1 x2 || nest: || altn:, case(id) vce(robust) `tight'
wr `fh' "separate_robust"
quietly nlogit chosen x1 x2 || nest: || altn:, case(id) vce(cluster clust) `tight'
wr `fh' "separate_cluster"
file write `fh' `"  "_meta": {"' _n
file write `fh' `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write `fh' `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write `fh' `"  }"' _n
file write `fh' "}" _n
file close `fh'
