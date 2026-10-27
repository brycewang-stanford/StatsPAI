* Reference values for tests/reference_parity/test_eivreg_stata_parity.py
* Stata 18: eivreg with two error-prone regressors, unweighted, with aweights
* and with frequency weights. Run from this directory.
* Writes eivreg_stata.txt (one "key value" pair per line).
clear all
set type double
import delimited using "eivreg_data.csv", clear
generate long fw = ceil(w)
tempname fh
file open `fh' using "eivreg_stata.txt", write replace text
foreach cs in plain one aw fw {
    local wt ""
    local rel "x1 .8 x2 .75"
    if "`cs'" == "aw" local wt "[aw = w]"
    if "`cs'" == "fw" local wt "[fw = fw]"
    if "`cs'" == "one" local rel "x1 .9"
    eivreg y x1 x2 z `wt', reliab(`rel')
    matrix b = e(b)
    matrix V = e(V)
    local names : colnames b
    local j = 0
    foreach v of local names {
        local ++j
        file write `fh' "`cs'_b_`v' " %21.15e (b[1,`j']) _n
        file write `fh' "`cs'_se_`v' " %21.15e (sqrt(V[`j',`j'])) _n
    }
    file write `fh' "`cs'_cov_x1_x2 " %21.15e (V[1,2]) _n
    file write `fh' "`cs'_N " %21.15e (e(N)) _n
    file write `fh' "`cs'_F " %21.15e (e(F)) _n
    file write `fh' "`cs'_r2 " %21.15e (e(r2)) _n
    file write `fh' "`cs'_rmse " %21.15e (e(rmse)) _n
    file write `fh' "`cs'_df_r " %21.15e (e(df_r)) _n
}
* Rows with an analytic weight of zero leave the sample (e(N) = 77).
preserve
replace w = 0 in 1/3
eivreg y x1 x2 z [aw = w], reliab(x1 .8 x2 .75)
file write `fh' "aw0_b_x1 " %21.15e (_b[x1]) _n
file write `fh' "aw0_se_x1 " %21.15e (_se[x1]) _n
file write `fh' "aw0_N " %21.15e (e(N)) _n
restore
* Without reliab() eivreg is regress.
eivreg y x1 x2 z
matrix b = e(b)
file write `fh' "none_b_x1 " %21.15e (b[1,1]) _n
* The smallest admissible reliability is the R2 of x1 on the other regressors
* and y (0.435934 here): Stata refuses below it with r(399).
foreach r in 0.4360 0.4359 0.10 {
    capture eivreg y x1 x2 z, reliab(x1 `r')
    file write `fh' "rc_x1_`r' " %21.15e (_rc) _n
}
file close `fh'
type "eivreg_stata.txt"
