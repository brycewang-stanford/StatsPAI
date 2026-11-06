* Stata 18 MP reference for the fd / small / first / regress options of
* sp.xtivreg (tests/test_xtivreg_stata.py, added 2026-11-06). panel.csv,
* two_endog.csv and gaps.csv are written by _data.py.
* From this directory: python _data.py . && stata-mp -b do stata_xtivreg_opts.do
* then python ../_stata_tsv.py stata_xtivreg_opts.tsv stata_xtivreg_opts.json
*
* xtivreg's first option only displays the first stages, and it leaves
* e() of the main fit contaminated (sigma_e, df_r of the previous
* command), so the main fits are dumped from runs without it. The first
* stages are dumped from the commands whose displayed tables they equal
* (checked in the log of this do-file): xtreg, fe / xtreg, be for fe / be,
* regress on D. for fd, and for re regress of the quasi-demeaned
* endogenous regressor on the quasi-demeaned instruments (theta_i from
* the main fit's e(sigma_u), e(sigma_e)), nocons.
set more off
set type double
file open fh using "stata_xtivreg_opts.tsv", write replace
file write fh "about" _tab "_meta" _tab "stata_version" _tab "`c(stata_version)'" _n
file write fh "about" _tab "_meta" _tab "born_date" _tab "`c(born_date)'" _n
file write fh "about" _tab "_meta" _tab "edition" _tab "`c(edition_real)'" _n
capture program drop dumpe
program dumpe
    args tag
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colnames `b'
    local k = colsof(`b')
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        file write fh "`tag'" _tab "`nm'" _tab %24.17g (`b'[1,`j']) _tab %24.17g (sqrt(`V'[`j',`j'])) _n
    }
    foreach s in sigma_u sigma_e rho r2_w r2_b r2_o N N_g N_clust chi2 F df_m df_r df_rz corr rss {
        capture file write fh "`tag'" _tab "e(`s')" _tab %24.17g (e(`s')) _tab "0" _n
    }
end
capture program drop dump_re_first
program dump_re_first
    * after xtivreg ..., re: the G2SLS first stage of `1' on `2'
    args tag endog exog clus
    tempvar Ti th
    quietly bysort id: egen double `Ti' = count(y)
    quietly gen double `th' = 1 - sqrt(e(sigma_e)^2 / (`Ti' * e(sigma_u)^2 + e(sigma_e)^2))
    local svars
    foreach v in `endog' `exog' {
        tempvar mv
        quietly bysort id: egen double `mv' = mean(`v')
        capture drop s_`v'
        quietly gen double s_`v' = `v' - `th' * `mv'
        if "`v'" != "`endog'" local svars `svars' s_`v'
    }
    capture drop s_cons
    quietly gen double s_cons = 1 - `th'
    if "`clus'" == "" regress s_`endog' `svars' s_cons, nocons
    else regress s_`endog' `svars' s_cons, nocons vce(cluster `clus')
    dumpe `tag'
end

import delimited using "panel.csv", clear asdouble
xtset id t
* first differences
xtivreg y x1 w (d = z1 z2), fd
dumpe fd
xtivreg y x1 w (d = z1 z2), fd vce(robust)
dumpe fd_robust
xtivreg y x1 w (d = z1 z2), fd vce(cluster c)
dumpe fd_cluster
xtivreg y x1 w (d = z1 z2), fd small
dumpe fd_small
xtivreg y x1 w (d = z1 z2), fd vce(cluster c) small
dumpe fd_cluster_small
* small with the other estimators
xtivreg y x1 (d = z1 z2), fe small
dumpe fe_small
xtivreg y x1 (d = z1 z2), fe vce(cluster c) small
dumpe fe_cluster_small
xtivreg y x1 w (d = z1 z2), re small
dumpe re_small
xtivreg y x1 w (d = z1 z2), re vce(cluster c) small
dumpe re_cluster_small
xtivreg y x1 w (d = z1 z2), re ec2sls small
dumpe re_ec2sls_small
xtivreg y x1 w (d = z1 z2), be small
dumpe be_small
* regress: the endogenous regressor treated as exogenous
xtivreg y x1 w (d = z1 z2), fe regress
dumpe fe_regress
xtreg y d x1 w, fe
dumpe xtreg_fe
xtivreg y x1 w (d = z1 z2), re regress
dumpe re_regress
xtivreg y x1 w (d = z1 z2), re nosa regress
dumpe re_nosa_regress
xtivreg y x1 w (d = z1 z2), be regress
dumpe be_regress
xtivreg y x1 w (d = z1 z2), fd regress
dumpe fd_regress
* first stages, displayed
xtivreg y x1 w (d = z1 z2), fe first
xtivreg y x1 w (d = z1 z2), fe vce(cluster c) first
xtivreg y x1 w (d = z1 z2), be first
xtivreg y x1 w (d = z1 z2), fd first
xtivreg y x1 w (d = z1 z2), fd vce(cluster c) first
xtivreg y x1 w (d = z1 z2), re first
xtivreg y x1 w (d = z1 z2), re vce(cluster c) first
* first stages, at full precision
xtreg d x1 w z1 z2, fe
dumpe fs_fe
xtreg d x1 w z1 z2, fe vce(cluster c)
dumpe fs_fe_cluster
xtreg d x1 w z1 z2, be
dumpe fs_be
regress D.d D.x1 D.w D.z1 D.z2
dumpe fs_fd
regress D.d D.x1 D.w D.z1 D.z2, vce(cluster c)
dumpe fs_fd_cluster
quietly xtivreg y x1 w (d = z1 z2), re
dump_re_first fs_re d "x1 w z1 z2"
quietly xtivreg y x1 w (d = z1 z2), re vce(cluster c)
dump_re_first fs_re_cluster d "x1 w z1 z2" c

import delimited using "two_endog.csv", clear asdouble
xtset id t
xtivreg y x1 w (d1 d2 = z1 z2 z3), fd
dumpe two_fd
xtivreg y x1 w (d1 d2 = z1 z2 z3), fd vce(cluster c) small
dumpe two_fd_cluster_small
xtivreg y x1 w (d1 d2 = z1 z2 z3), fd first
regress D.d1 D.x1 D.w D.z1 D.z2 D.z3
dumpe two_fs_fd_d1
regress D.d2 D.x1 D.w D.z1 D.z2 D.z3
dumpe two_fs_fd_d2

import delimited using "gaps.csv", clear asdouble
xtset id t
xtivreg y x1 w (d = z1 z2), fd
dumpe gaps_fd
xtivreg y x1 w (d = z1 z2), fd vce(cluster c)
dumpe gaps_fd_cluster
file close fh
