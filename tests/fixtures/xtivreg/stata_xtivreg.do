* Stata 18 MP reference for tests/test_xtivreg_stata.py.
* From this directory: python _data.py . && stata-mp -b do stata_xtivreg.do
* (the tab-separated output is stored as stata_xtivreg.json)
set more off
set type double
import delimited using "panel.csv", clear asdouble
xtset id t
file open fh using "stata_xtivreg.tsv", write replace
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
    foreach s in sigma_u sigma_e rho r2_w r2_b r2_o N N_g {
        capture file write fh "`tag'" _tab "e(`s')" _tab %24.17g (e(`s')) _tab "0" _n
    }
end
xtivreg y x1 w (d = z1 z2), re
dumpe re_g2sls
xtivreg y x1 w (d = z1 z2), re ec2sls
dumpe re_ec2sls
xtivreg y x1 w (d = z1 z2), re nosa
dumpe re_nosa
xtivreg y x1 w (d = z1 z2), re vce(cluster c)
dumpe re_cluster
xtivreg y x1 (d = z1 z2), fe
dumpe fe
xtivreg y x1 (d = z1 z2), fe vce(robust)
dumpe fe_robust
xtivreg y x1 (d = z1 z2), fe vce(cluster c)
dumpe fe_cluster
xtivreg y x1 w (d = z1 z2), be
dumpe be
file close fh
