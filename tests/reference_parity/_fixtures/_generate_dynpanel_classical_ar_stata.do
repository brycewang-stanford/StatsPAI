version 18
clear all
set more off
mata: mata set matafavor speed, perm
import delimited using "dynpanel_abdata.csv", clear
xtset id year
file open F using "ar_out.csv", write replace text
file write F "spec,ar1,ar2" _n
capture program drop dump
program define dump
    syntax , spec(string)
    file write F "`spec'," %21.16e (e(ar1)) "," %21.16e (e(ar2)) _n
end
quietly xtabond2 n L.n, gmm(n, lag(2 .))
dump, spec(sys_nox_cls)
quietly xtabond2 n L.n, gmm(n, lag(2 .)) h(1)
dump, spec(sys_nox_cls_h1)
quietly xtabond2 n L.n, gmm(n, lag(2 .)) h(2)
dump, spec(sys_nox_cls_h2)
quietly xtabond2 n L.n w k, gmm(n, lag(2 .)) iv(w k)
dump, spec(sys_wk_cls)
quietly xtabond2 n L.n, gmm(n, lag(2 .)) noleveleq noconstant orthogonal
dump, spec(fod_diff_nox_cls)
quietly xtabond2 n L.n w k, gmm(n, lag(2 .)) iv(w k, equation(diff)) noleveleq noconstant orthogonal
dump, spec(fod_diff_wk_cls)
quietly xtabond2 n L.n, gmm(n, lag(2 .)) orthogonal
dump, spec(fod_sys_nox_cls)
quietly xtabond2 n L.n, gmm(n, lag(2 .)) noleveleq noconstant
dump, spec(diff_nox_cls)
quietly xtabond2 n L.n, gmm(n, lag(2 .)) robust
dump, spec(sys_nox_rob)
file close F
* Classical-variance AR statistics and the FOD classical SE (black box,
* Stata 18 MP, xtabond2 from SSC), run on dynpanel_abdata.csv. The values
* are frozen in tests/reference_parity/test_dynpanel_classical_ar_stata.py.
* FOD classical SE of L.n:
*   xtabond2 n L.n w k, gmm(n, lag(2 .)) iv(w k, equation(diff)) noleveleq noconstant orthogonal
*   -> b = 3.97106788174169e-01, se = 9.06500167086847e-02
