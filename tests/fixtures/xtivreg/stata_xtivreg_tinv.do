* Stata 18 MP reference for the time-invariant regressor / instrument cases
* of tests/test_xtivreg_stata.py (xtivreg, re). panel.csv is the unbalanced
* panel of stata_xtivreg.do, balanced.csv a balanced one; zb is the panel
* mean of z1, written by _data.py so both sides read the same bytes.
* From this directory: python _data.py . && stata-mp -b do stata_xtivreg_tinv.do
* (the tab-separated output is stored as stata_xtivreg_tinv.json)
set more off
set type double
file open fh using "stata_xtivreg_tinv.tsv", write replace
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
    foreach s in sigma_u sigma_e rho N N_g {
        capture file write fh "`tag'" _tab "e(`s')" _tab %24.17g (e(`s')) _tab "0" _n
    }
end
import delimited using "panel.csv", clear asdouble
xtset id t
xtivreg y x1 w (d = zb), re
dumpe unb_only_tinv
xtivreg y x1 w (d = zb), re vce(robust)
dumpe unb_only_tinv_robust
xtivreg y x1 w (d = zb), re ec2sls
dumpe unb_only_tinv_ec2sls
import delimited using "balanced.csv", clear asdouble
xtset id t
xtivreg y x1 wd (d = z1 zb), re
dumpe bal_g2sls
xtivreg y x1 wd (d = z1 zb), re ec2sls
dumpe bal_ec2sls
xtivreg y x1 wd (d = z1 zb), re nosa
dumpe bal_nosa
xtivreg y x1 wd (d = z1 zb), re vce(cluster c)
dumpe bal_cluster
xtivreg y x1 (d = z1 z2), re
dumpe bal_plain
xtivreg y x1 wd (d = zb), re
dumpe bal_only_tinv
xtivreg y x1 wd (d = zb), re vce(robust)
dumpe bal_only_tinv_robust
* continuous time-invariant regressor: with vce(robust) / vce(cluster) the
* K behind sigma_e counts it (rounding noise in its within transform)
xtivreg y x1 w (d = z1), re
dumpe bal_w_cont
xtivreg y x1 w (d = z1), re vce(robust)
dumpe bal_w_cont_robust
file close fh
