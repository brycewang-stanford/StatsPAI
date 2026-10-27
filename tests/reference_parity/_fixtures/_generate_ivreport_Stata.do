* Stata 18 MP reference numbers for the IV reporting tools
* (tests/reference_parity/test_ivreport_stata_parity.py):
*   A. regress ..., beta (e(beta)) and estimators run on pre-standardised
*      variables (regress / ivreg2 / reghdfe / ivreghdfe), for
*      sp.standardize_coefs;
*   B. acreg spatial HAC at several distance cutoffs, OLS and IV, uniform
*      and Bartlett kernels, for sp.conley_sensitivity and the acreg
*      translation;
*   C. weakiv after ivreg2 (wrapper and post-estimation forms), for the
*      weakiv translation.
* Packages: ivreg2 4.1.12, ranktest, weakiv 2.4.07, acreg 1.1.0 (Dec 2020),
* reghdfe 6.13.1, ivreghdfe 1.1.4.
*   python tests/reference_parity/_fixtures/_generate_ivreport_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_ivreport_Stata.do
* (run from the repository root)
version 18
clear all
set type double
import delimited using "tests/reference_parity/_fixtures/ivreport_fx.csv", clear asdouble

capture program drop jvec
program define jvec
    * write `"key": {"names": [...], "b": [...], "se": [...], "N": #}'
    args fh key
    tempname B V
    matrix `B' = e(b)
    matrix `V' = e(V)
    local names : colnames `B'
    local k = colsof(`B')
    file write `fh' `"  "`key'": {"names": ["'
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'""'
    }
    file write `fh' "], " _n `"    "b": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (`B'[1, `j'])
    }
    file write `fh' "], " _n `"    "se": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ", "
        * acreg reports a negative variance (non-PSD uniform kernel) as .
        if missing(sqrt(`V'[`j', `j'])) file write `fh' "null"
        else file write `fh' %24.17e (sqrt(`V'[`j', `j']))
    }
    file write `fh' "], " `""N": "' (e(N)) "}"
end

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/ivreport_Stata.json", write replace
file write `fh' "{" _n

* ---------------------------------------------------------------- A. beta
foreach spec in "" "[aw=wt]" {
    qui regress y x w1 w2 d `spec', beta
    local key = cond("`spec'" == "", "regress_beta", "regress_aw_beta")
    tempname BE
    matrix `BE' = e(beta)
    local names : colnames `BE'
    file write `fh' `"  "`key'": {"names": ["'
    local k = colsof(`BE')
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'""'
    }
    file write `fh' `"], "beta": ["'
    forvalues j = 1/`k' {
        if `j' > 1 file write `fh' ", "
        file write `fh' %24.17e (`BE'[1, `j'])
    }
    file write `fh' "]}," _n
}

* z-scores on the estimation sample (no missing values in the fixture)
foreach v in y x z z2 w1 w2 d {
    qui summarize `v'
    gen double s_`v' = (`v' - r(mean)) / r(sd)
}
qui regress s_y s_x s_w1 s_w2 s_d, vce(robust)
jvec `fh' "std_regress_robust"
file write `fh' "," _n
qui regress s_y s_x s_w1 s_w2 s_d, vce(cluster region)
jvec `fh' "std_regress_cluster"
file write `fh' "," _n
qui ivreg2 s_y s_w1 s_w2 s_d (s_x = s_z), robust
jvec `fh' "std_ivreg2_robust"
file write `fh' "," _n
qui ivreg2 s_y s_w1 s_w2 s_d (s_x = s_z s_z2), cluster(region) small
jvec `fh' "std_ivreg2_cluster_small"
file write `fh' "," _n
qui reghdfe s_y s_x s_w1 s_w2 s_d, absorb(fe) vce(cluster region)
jvec `fh' "std_reghdfe_cluster"
file write `fh' "," _n
qui ivreghdfe s_y s_w1 s_w2 s_d (s_x = s_z), absorb(fe) robust small
jvec `fh' "std_ivreghdfe_robust_small"
file write `fh' "," _n

* ---------------------------------------------------------------- B. acreg
foreach c in 25 75 150 300 {
    foreach kern in "" "bartlett" {
        local tag = cond("`kern'" == "", "unif", "bart")
        qui acreg y x w1 w2 d, spatial latitude(lat) longitude(lon) dist(`c') `kern'
        jvec `fh' "acreg_ols_`tag'_`c'"
        file write `fh' "," _n
        qui acreg y w1 w2 d (x = z), spatial latitude(lat) longitude(lon) dist(`c') `kern'
        jvec `fh' "acreg_iv_`tag'_`c'"
        file write `fh' "," _n
    }
}
qui acreg y x w1 w2 d, spatial latitude(lat) longitude(lon) dist(100) pfe1(fe)
jvec `fh' "acreg_ols_pfe_unif_100"
file write `fh' "," _n
qui acreg y x w1 w2 d, spatial latitude(lat) longitude(lon) dist(100) pfe1(fe) bartlett
jvec `fh' "acreg_ols_pfe_bart_100"
file write `fh' "," _n
qui acreg y w1 w2 d (x = z), spatial latitude(lat) longitude(lon) dist(100) pfe1(fe)
jvec `fh' "acreg_iv_pfe_unif_100"
file write `fh' "," _n

* ---------------------------------------------------------------- C. weakiv
capture program drop wdump
program define wdump
    args fh key
    file write `fh' `"  "`key'": {"'
    local first 1
    foreach s in N ar_chi2 ar_p k_chi2 k_p j_chi2 j_p clr_stat clr_p ar_df {
        capture confirm scalar e(`s')
        if !_rc {
            if !`first' file write `fh' ", "
            local first 0
            file write `fh' `""`s'": "' %24.17e (e(`s'))
        }
    }
    foreach s in ar_cset k_cset clr_cset {
        if !`first' file write `fh' ", "
        local first 0
        file write `fh' `""`s'": "' `""`e(`s')'""'
    }
    file write `fh' "}"
end

qui weakiv ivreg2 y w1 w2 d (x = z z2), robust
wdump `fh' "weakiv_robust"
file write `fh' "," _n
qui weakiv ivreg2 y w1 w2 d (x = z z2), null(0.3) cluster(region)
wdump `fh' "weakiv_cluster_null"
file write `fh' "," _n
qui weakiv ivreg2 y w1 w2 d (x = z z2), null(0.2)
wdump `fh' "weakiv_iid_null"
file write `fh' "," _n
qui ivreg2 y w1 w2 d (x = z z2), robust
qui weakiv
wdump `fh' "weakiv_after_ivreg2_robust"
file write `fh' "," _n
qui weakiv ivreg2 y w1 w2 d (x = z), robust level(90)
wdump `fh' "weakiv_robust_justid_l90"
file write `fh' _n "}" _n
file close `fh'
