* Stata 18 MP reference for the multi-endogenous and remaining vce cases of
* tests/test_xtivreg_stata.py (added 2026-11-05). panel.csv is the
* unbalanced panel of stata_xtivreg.do; two_endog.csv has two endogenous
* regressors and three excluded instruments (both written by _data.py).
* From this directory: python _data.py . && stata-mp -b do stata_xtivreg_more.do
* then python ../_stata_tsv.py stata_xtivreg_more.tsv stata_xtivreg_more.json
set more off
set type double
file open fh using "stata_xtivreg_more.tsv", write replace
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
    foreach s in sigma_u sigma_e rho r2_w r2_b r2_o N N_g chi2 F {
        capture file write fh "`tag'" _tab "e(`s')" _tab %24.17g (e(`s')) _tab "0" _n
    }
end
import delimited using "panel.csv", clear asdouble
xtset id t
xtivreg y x1 w (d = z1 z2), re vce(robust)
dumpe re_robust
xtivreg y x1 w (d = z1 z2), re ec2sls vce(robust)
dumpe re_ec2sls_robust
xtivreg y x1 w (d = z1 z2), re nosa vce(cluster c)
dumpe re_nosa_cluster
import delimited using "two_endog.csv", clear asdouble
xtset id t
xtivreg y x1 w (d1 d2 = z1 z2 z3), re
dumpe two_re_g2sls
xtivreg y x1 w (d1 d2 = z1 z2 z3), re ec2sls
dumpe two_re_ec2sls
xtivreg y x1 w (d1 d2 = z1 z2 z3), re nosa
dumpe two_re_nosa
xtivreg y x1 w (d1 d2 = z1 z2 z3), re vce(cluster c)
dumpe two_re_cluster
xtivreg y x1 (d1 d2 = z1 z2 z3), fe
dumpe two_fe
xtivreg y x1 (d1 d2 = z1 z2 z3), fe vce(robust)
dumpe two_fe_robust
xtivreg y x1 (d1 d2 = z1 z2 z3), fe vce(cluster c)
dumpe two_fe_cluster
xtivreg y x1 w (d1 d2 = z1 z2 z3), be
dumpe two_be
xtivreg y x1 w (d1 d2 = z1 z2 z3), re vce(robust)
dumpe two_re_robust
xtivreg y x1 w (d1 d2 = z1 z2 z3), re ec2sls vce(cluster c)
dumpe two_re_ec2sls_cluster
* exactly identified, two endogenous regressors
xtivreg y x1 w (d1 d2 = z1 z2), re
dumpe two_re_exact
xtivreg y x1 (d1 d2 = z1 z2), fe
dumpe two_fe_exact
* a balanced panel (the first five periods of the panels with at least five)
import delimited using "two_endog_bal.csv", clear asdouble
xtset id t
xtivreg y x1 w (d1 d2 = z1 z2 z3), re
dumpe twobal_re_g2sls
xtivreg y x1 w (d1 d2 = z1 z2 z3), re ec2sls
dumpe twobal_re_ec2sls
xtivreg y x1 w (d1 d2 = z1 z2 z3), re vce(robust)
dumpe twobal_re_robust
xtivreg y x1 (d1 d2 = z1 z2 z3), re
dumpe twobal_re_now
xtivreg y x1 (d1 d2 = z1 z2 z3), fe
dumpe twobal_fe
xtivreg y x1 w (d1 d2 = z1 z2 z3), be
dumpe twobal_be
file close fh
