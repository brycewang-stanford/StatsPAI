* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_proximal_r_parity.py and
* tests/reference_parity/test_interference_r_parity.py (second reference
* beside _generate_proximal_interference_R.R).
*
* Requires: Stata 18 (official commands only; nothing to install).
* Run:      stata-mp -b do _generate_proximal_interference_Stata.do
*           (from this directory)
* Writes:   proximal_interference_Stata.json, a flat map
*           "case": [estimate, se].
*
* ivregress 2sls with `small` uses RSS / (n - k) (sp.proximal); with
* vce(robust) small it is HC1 (sp.double_negative_control); without small
* vce(robust) is HC0 (sp.peer_effects). regress, vce(robust) is HC1.
* ---------------------------------------------------------------------------
version 18
clear all
set type double

capture program drop putbs
program define putbs
    args fh key term
    file write `fh' `"  "`key'": ["' %24.17e (_b[`term']) ", " %24.17e (_se[`term']) "]," _n
end

file open fh using "proximal_interference_Stata.json", write replace text
file write fh "{" _n

* ------------------------------------------------------------- proximal
import delimited using "proximal_ref_data.csv", clear asdouble
ivregress 2sls y d x1 x2 (w1 = z1), small
putbs fh proximal_just_identified d
ivregress 2sls y d x1 (w1 = z1 z2), small
putbs fh proximal_over_identified d
ivregress 2sls y d x1 x2 (w1 w2 = z1 z2), small
putbs fh proximal_two_w d
ivregress 2sls y d (w1 = z1), small
putbs fh proximal_no_covariates d

regress nco d x1 x2, vce(robust)
putbs fh nco_covariates d
regress nco d, vce(robust)
putbs fh nco_bare d
regress y nce x1, vce(robust)
putbs fh nce_covariates nce
regress y nce, vce(robust)
putbs fh nce_bare nce

ivregress 2sls y d x1 x2 (nco = nce), vce(robust) small
putbs fh dnc_covariates d
ivregress 2sls y d (nco = nce), vce(robust) small
putbs fh dnc_bare d

* --------------------------------------------------------- interference
import delimited using "interference_unit_data.csv", clear asdouble
regress y d e_in e_out x
putbs fh io_covariates_inward e_in
putbs fh io_covariates_outward e_out
regress y d e_in e_out
putbs fh io_bare_inward e_in
putbs fh io_bare_outward e_out

import delimited using "interference_cluster_data.csv", clear asdouble
collapse (mean) y treat nshare pair, by(cluster)
regress y treat nshare
putbs fh cluster_cross_direct treat
putbs fh cluster_cross_spillover nshare
* matched pairs: paired t-test on the cluster means
keep pair treat y
reshape wide y, i(pair) j(treat)
ttest y1 == y0
file write fh `"  "matched_pair": ["' %24.17e (r(mu_1) - r(mu_2)) ", " %24.17e (r(se)) "]," _n

* --------------------------------------------------------- peer effects
import delimited using "peer_network_edges.csv", clear asdouble
mata: E = st_data(., ("from", "to"))
import delimited using "peer_network_data.csv", clear asdouble
mata:
    n = st_nobs()
    A = J(n, n, 0)
    for (k = 1; k <= rows(E); k++) {
        A[E[k, 1], E[k, 2]] = 1
        A[E[k, 2], E[k, 1]] = 1
    }
    rs = rowsum(A)
    rs = rs + (rs :== 0)
    W = A :/ rs
    X = st_data(., ("xa", "xb"))
    WX = W * X
    W2X = W * WX
    W3X = W * W2X
    st_store(., st_addvar("double", "wy"), W * st_data(., "y"))
    st_store(., st_addvar("double", ("wxa", "wxb")), WX)
    st_store(., st_addvar("double", ("w2xa", "w2xb")), W2X)
    st_store(., st_addvar("double", ("w3xa", "w3xb")), W3X)
end
ivregress 2sls y xa xb wxa wxb (wy = w2xa w2xb w3xa w3xb), vce(robust)
foreach t in wy xa xb wxa wxb _cons {
    putbs fh peer_contextual_`t' `t'
}
ivregress 2sls y xa xb (wy = wxa wxb w2xa w2xb w3xa w3xb), vce(robust)
foreach t in wy xa xb _cons {
    putbs fh peer_no_contextual_`t' `t'
}

file write fh `"  "version": "Stata `c(stata_version)' `c(edition_real)' `c(born_date)'""' _n
file write fh "}" _n
file close fh
